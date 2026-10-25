"""Ranking, frozen reward re-ranking, and paired uncertainty for the second study.

All reward estimates remain observational model predictions. The statistical
functions compare saved recommendations; they do not estimate a deployed policy's
causal value. No outcome is used by the re-ranker or the recommendation policy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import evaluate_policy


def _reward_function(reward):
    return reward.score_candidates if hasattr(reward, "score_candidates") else reward


def evaluate(
    logits_fn: Callable,
    examples: pd.DataFrame,
    vocab_size: int,
    reward,
    popularity: np.ndarray | None = None,
    genres: Mapping | None = None,
    batch_size: int = 128,
    exclude_seen: bool = True,
):
    """Use the common full-catalog protocol and add factual weighted relevance.

    ``factual_completion_weighted_hit@K`` is user-macro mean of the observed
    target completion multiplied by its hit indicator. Unknown targets remain
    misses. Missing factual completions are excluded only from these factual
    metrics, and their availability is reported separately.
    """
    metrics, per = evaluate_policy(
        logits_fn, examples, vocab_size, _reward_function(reward),
        popularity=popularity, item_genres=genres, batch_size=batch_size,
        exclude_seen=exclude_seen,
    )
    if "reward" in examples:
        factual = examples.reward.to_numpy(dtype=float, copy=True)
        if len(per) != len(factual):
            raise ValueError("Evaluation and factual outcome counts differ")
        for key in ("user_id", "target", "day"):
            if key in examples and key in per:
                if not np.array_equal(per[key].to_numpy(), examples[key].to_numpy()):
                    raise ValueError(f"Evaluation and factual rows disagree on {key}")
        per["actual_completion"] = factual
        added = []
        for k in (10, 20):
            for label, source in (("factual_completion_weighted_hit", f"recall@{k}"),
                                  ("factual_completion_weighted_dcg", f"ndcg@{k}")):
                name = f"{label}@{k}"
                per[name] = factual * per[source].to_numpy()
                added.append(name)
        metrics.update(per.groupby("user_id")[added].mean().mean().to_dict())
        metrics["factual_completion_available_fraction"] = float(np.isfinite(factual).mean())
        metrics["factual_completion_n"] = int(np.isfinite(factual).sum())
        engaged = per[per["actual_completion"] >= .5]
        metrics["engaged_target_n"] = len(engaged)
        metrics["engaged_target_users"] = int(engaged.user_id.nunique())
        if len(engaged):
            for metric in ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]:
                metrics["engaged_" + metric] = float(engaged.groupby("user_id")[metric].mean().mean())
    known = per[per["known_target"]]
    if len(known):
        for metric in ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]:
            metrics["known_catalog_" + metric] = float(known.groupby("user_id")[metric].mean().mean())
    metrics["reward_interpretation"] = "Predictions on recommended items, not observed policy outcomes"
    return metrics, per


class RewardReranker:
    """Frozen ``base_logit + alpha * reward`` on the base policy's top-M items.

    The base top-M is selected after excluding the complete observed history.
    Outside candidates receive -1e8 to preserve finite-logit callback contracts;
    their probability under softmax is numerically zero. Ranking K must be <= M.
    With alpha=0 the original top-K is unchanged. The same candidate cache may
    be shared by a validation alpha grid; use a distinct ``cache_namespace`` for
    every base checkpoint. Candidate restriction and alpha are experimental
    choices, not off-policy correction.
    """

    def __init__(
        self,
        base_logits_fn: Callable,
        reward,
        vocab_size: int,
        alpha: float,
        top_m: int = 200,
        reward_type: str = "mean",
        cache: dict | None = None,
        cache_namespace: str | None = None,
        uncertainty_weight: float = 1.,
        exclude_seen: bool = True,
    ):
        if vocab_size < 4 or top_m < 20:
            raise ValueError("Need a real-item vocabulary and top_m >= 20")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        aliases = {"completion": "mean", "lcb": "mean-std", "conservative": "mean-std"}
        self.reward_type = aliases.get(reward_type, reward_type)
        if self.reward_type not in ("mean", "mean-std", "return7"):
            raise ValueError("reward_type must be mean, mean-std, or return7")
        self.base_logits_fn = base_logits_fn
        self.reward_fn = _reward_function(reward)
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.top_m = min(int(top_m), vocab_size - 3)
        self.uncertainty_weight = float(uncertainty_weight)
        self.exclude_seen = bool(exclude_seen)
        self.cache = {} if cache is None else cache
        self.namespace = (cache_namespace if cache_namespace is not None else id(base_logits_fn),
                          self.top_m, self.vocab_size, self.exclude_seen)
        self.computed_contexts = 0
        self.reward_cache_hits = 0
        self.reward_fallback_items = 0

    def _key(self, history):
        return self.namespace, tuple(int(token) for token in history)

    def _populate(self, histories):
        missing, keys = [], []
        queued = set()
        for history in histories:
            key = self._key(history)
            if key not in self.cache and key not in queued:
                keys.append(key)
                missing.append(list(history))
                queued.add(key)
        if not missing:
            return
        scores = np.asarray(self.base_logits_fn(missing), dtype=np.float64)
        if scores.shape != (len(missing), self.vocab_size):
            raise ValueError("Base callback returned a different vocabulary or batch")
        if not np.isfinite(scores[:, 3:]).all():
            raise ValueError("Base real-item logits must be finite")
        eligible = scores.copy()
        eligible[:, :3] = -np.inf
        if self.exclude_seen:
            for row, history in enumerate(missing):
                seen = np.asarray(history, dtype=int)
                seen = seen[(seen >= 3) & (seen < self.vocab_size)]
                eligible[row, seen] = -np.inf
        # Stable sorting makes equal-score candidate choice explicit.
        candidates = np.argsort(-eligible, axis=1, kind="stable")[:, :self.top_m]
        usable = np.isfinite(np.take_along_axis(eligible, candidates, axis=1))
        safe = np.where(usable, candidates, 3)
        raw = self.reward_fn(missing, safe)
        if not isinstance(raw, dict):
            raw = {"mean": np.asarray(raw, dtype=float)}
        heads = {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}
        if "mean" not in heads:
            raise ValueError("Reward predictions must contain mean")
        for head, values in heads.items():
            if values.shape != candidates.shape or not np.isfinite(values[usable]).all():
                raise ValueError(f"Invalid reward head: {head}")
        base = np.take_along_axis(scores, candidates, axis=1)
        for row, key in enumerate(keys):
            mask = usable[row]
            self.cache[key] = {
                "items": candidates[row, mask].astype(np.int32),
                "base": base[row, mask].astype(np.float32),
                "heads": {name: values[row, mask].copy() for name, values in heads.items()},
            }
        self.computed_contexts += len(keys)

    def score_candidates(self, histories, candidates):
        """Reuse cached RM heads on top-M candidates; query the RM elsewhere.

        Passing this re-ranker itself as ``reward`` to ``evaluate`` avoids
        repeating CatBoost inference for every alpha in a validation sweep.
        The callback is still valid for arbitrary in-vocabulary candidates:
        off-support items are evaluated by the original reward model in one
        padded batch, and only its requested entries are copied to the result.
        """
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim != 2 or candidates.shape[0] != len(histories):
            raise ValueError("candidates must have shape [len(histories), C]")
        if np.any((candidates < 0) | (candidates >= self.vocab_size)):
            raise ValueError("Candidate token outside frozen vocabulary")
        self._populate(histories)
        if not len(histories):
            return {"mean": np.empty(candidates.shape, dtype=np.float32)}
        first = self.cache[self._key(histories[0])]
        head_names = tuple(first["heads"])
        out = {name: np.empty(candidates.shape, dtype=np.float32) for name in head_names}
        fallback_rows, fallback_positions = [], []
        for row, history in enumerate(histories):
            entry = self.cache[self._key(history)]
            if set(entry["heads"]) != set(head_names):
                raise ValueError("Cached reward heads disagree across contexts")
            # Search by token ID without retaining a Python dictionary per
            # context. The small int32 index adds substantially less memory.
            if "item_order" not in entry:
                entry["item_order"] = np.argsort(entry["items"], kind="stable").astype(np.int32)
            order = entry["item_order"]
            sorted_items = entry["items"][order]
            positions = np.searchsorted(sorted_items, candidates[row])
            hit = positions < len(sorted_items)
            if hit.any():
                active = np.flatnonzero(hit)
                hit[active] &= sorted_items[positions[active]] == candidates[row, active]
            if hit.any():
                columns = order[positions[hit]]
                for name in head_names:
                    out[name][row, hit] = entry["heads"][name][columns]
            missed = np.flatnonzero(~hit)
            self.reward_cache_hits += int(hit.sum())
            self.reward_fallback_items += len(missed)
            if len(missed):
                fallback_rows.append(row)
                fallback_positions.append(missed)
        if fallback_rows:
            width = max(map(len, fallback_positions))
            off_support = np.full((len(fallback_rows), width), 3, dtype=np.int64)
            for index, (row, positions) in enumerate(zip(fallback_rows, fallback_positions)):
                off_support[index, :len(positions)] = candidates[row, positions]
            raw = self.reward_fn([histories[row] for row in fallback_rows], off_support)
            if not isinstance(raw, dict):
                raw = {"mean": np.asarray(raw, dtype=float)}
            if set(raw) != set(head_names):
                raise ValueError("Fallback reward heads disagree with cached schema")
            for name in head_names:
                values = np.asarray(raw[name], dtype=np.float32)
                if values.shape != off_support.shape:
                    raise ValueError(f"Invalid fallback reward head: {name}")
                for index, (row, positions) in enumerate(zip(fallback_rows, fallback_positions)):
                    selected = values[index, :len(positions)]
                    if not np.isfinite(selected).all():
                        raise ValueError(f"Nonfinite fallback reward head: {name}")
                    out[name][row, positions] = selected
        return out

    def __call__(self, histories):
        self._populate(histories)
        out = np.full((len(histories), self.vocab_size), -1e8, dtype=np.float64)
        for row, history in enumerate(histories):
            entry = self.cache[self._key(history)]
            heads = entry["heads"]
            if self.reward_type == "mean":
                reward = heads["mean"]
            elif self.reward_type == "return7":
                if "return7" not in heads:
                    raise ValueError("return7 predictions unavailable")
                reward = heads["return7"]
            else:
                if "std" not in heads:
                    raise ValueError("Ensemble standard deviation unavailable")
                reward = heads["mean"] - self.uncertainty_weight * heads["std"]
            out[row, entry["items"]] = entry["base"] + self.alpha * reward
        return out


def _as_frames(value):
    return [value] if isinstance(value, pd.DataFrame) else list(value)


def _paired_user_values(baseline: pd.DataFrame, tuned: pd.DataFrame, metric: str):
    keys = ["row_id", "user_id"]
    for frame in (baseline, tuned):
        if frame.duplicated(keys).any():
            raise ValueError("Repeated row/user keys in evaluation")
        if metric not in frame:
            raise ValueError(f"Missing metric {metric}")
    identities = [key for key in ("target", "day", "context_key", "target_set_key") if key in baseline and key in tuned]
    left = baseline[keys + identities + [metric]]
    right = tuned[keys + identities + [metric]]
    pairs = left.merge(right, on=keys, suffixes=("_base", "_tuned"), validate="one_to_one")
    if len(pairs) != len(baseline) or len(pairs) != len(tuned) or pairs.empty:
        raise ValueError("Need exactly the same nonempty evaluation rows")
    for identity in identities:
        if not (pairs[f"{identity}_base"] == pairs[f"{identity}_tuned"]).all():
            raise ValueError(f"Paired examples disagree on {identity}")
    values = pairs.groupby("user_id")[[f"{metric}_base", f"{metric}_tuned"]].mean()
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Paired per-user values must be finite")
    return values


def _interval(values, confidence):
    alpha = (1 - confidence) / 2
    return {"low": float(np.quantile(values, alpha)), "high": float(np.quantile(values, 1-alpha))}


def bootstrap_family(
    baseline: pd.DataFrame | Sequence[pd.DataFrame],
    tuned: Sequence[pd.DataFrame],
    metric: str,
    margin: float = .05,
    n_bootstrap: int = 2000,
    confidence: float = .95,
    seed: int = 42,
):
    """Paired user and training-seed uncertainty, kept explicitly distinct.

    A single baseline is shared across all tuned seeds. Alternatively a baseline
    list of the same size gives matched independent SFT/RL runs. User draws are
    shared across seeds because every checkpoint is evaluated on the same users.
    The crossed bootstrap independently resamples users and matched training runs;
    it is exploratory with only a few seeds, not a large-sample guarantee.

    Relative non-inferiority tests ``tuned - (1-margin)*baseline > 0``. Its lower
    one-sided bound resamples BOTH terms, rather than treating the estimated
    baseline and therefore the absolute margin as constants.
    """
    bases, tunes = _as_frames(baseline), _as_frames(tuned)
    if not tunes or len(bases) not in (1, len(tunes)):
        raise ValueError("Need one shared baseline or one baseline per tuned seed")
    if not 0 < confidence < 1 or not 0 <= margin < 1 or n_bootstrap < 100:
        raise ValueError("Invalid confidence, relative margin, or bootstrap budget")
    shared = len(bases) == 1
    if shared:
        bases = bases * len(tunes)
    pairs = [_paired_user_values(base, tune, metric) for base, tune in zip(bases, tunes)]
    users = pairs[0].index
    if any(not pair.index.equals(users) for pair in pairs):
        raise ValueError("Every training seed must have the same users")
    base_values = np.stack([pair[f"{metric}_base"].to_numpy() for pair in pairs])
    tuned_values = np.stack([pair[f"{metric}_tuned"].to_numpy() for pair in pairs])
    delta = tuned_values - base_values
    ni = tuned_values - (1-margin) * base_values
    n_seeds, n_users = delta.shape
    rng = np.random.default_rng(seed)
    user_draws, seed_draws, joint_draws = (np.empty(n_bootstrap) for _ in range(3))
    ni_user, ni_joint = np.empty(n_bootstrap), np.empty(n_bootstrap)
    user_mean_delta, user_mean_ni = delta.mean(axis=0), ni.mean(axis=0)
    seed_mean_delta = delta.mean(axis=1)
    for index in range(n_bootstrap):
        sampled_users = rng.integers(0, n_users, size=n_users)
        sampled_seeds = rng.integers(0, n_seeds, size=n_seeds)
        user_draws[index] = user_mean_delta[sampled_users].mean()
        ni_user[index] = user_mean_ni[sampled_users].mean()
        seed_draws[index] = seed_mean_delta[sampled_seeds].mean()
        joint_draws[index] = delta[np.ix_(sampled_seeds, sampled_users)].mean()
        ni_joint[index] = ni[np.ix_(sampled_seeds, sampled_users)].mean()
    observed = float(delta.mean())
    # Sign-flip test: conditional on these checkpoints and exchangeable paired
    # user differences. It does not account for selection over configurations.
    extreme = 0
    for start in range(0, n_bootstrap, 128):
        count = min(128, n_bootstrap-start)
        signs = 2*rng.integers(0, 2, size=(count, n_users))-1
        statistic = (signs * user_mean_delta).mean(axis=1)
        extreme += int((np.abs(statistic) >= abs(observed)-1e-15).sum())
    alpha = 1-confidence
    user_lower = float(np.quantile(ni_user, alpha))
    joint_lower = float(np.quantile(ni_joint, alpha))
    baseline_mean = float(base_values.mean())
    return {
        "metric": metric, "baseline_mean": baseline_mean,
        "tuned_mean": float(tuned_values.mean()), "delta": observed,
        "relative_delta": observed/baseline_mean if baseline_mean != 0 else None,
        "n_users": n_users, "n_examples": len(tunes[0]), "n_seeds": n_seeds,
        "baseline_mode": "shared_fixed_checkpoint" if shared else "matched_training_seeds",
        "confidence": confidence, "n_bootstrap": n_bootstrap,
        "ci_user": _interval(user_draws, confidence),
        "ci_training_seed": _interval(seed_draws, confidence),
        "ci_joint": _interval(joint_draws, confidence),
        "per_seed_delta": delta.mean(axis=1).tolist(),
        "paired_randomization_p_two_sided": (extreme+1)/(n_bootstrap+1),
        "noninferiority": {
            "relative_margin": margin, "absolute_margin_at_point_estimate": margin*baseline_mean,
            "statistic": "tuned - (1-relative_margin)*baseline",
            "statistic_mean": float(ni.mean()), "one_sided_confidence": confidence,
            "lower_bound_user": user_lower, "lower_bound_joint": joint_lower,
            "passes_user": bool(user_lower > 0), "passes_joint": bool(joint_lower > 0),
            "primary": "joint" if n_seeds > 1 else "user",
        },
        "caveat": "Few training seeds; conditional randomization p-value; no correction for model selection or causal interpretation",
    }


def holm_adjust(p_values: Mapping[str, float]):
    """Holm step-down adjustment for a declared family of hypothesis tests."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted, previous = {}, 0.
    count = len(ordered)
    for rank, name in enumerate(ordered):
        p = float(p_values[name])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p-values must lie in [0,1]")
        previous = max(previous, min(1., (count-rank)*p))
        adjusted[name] = previous
    return {name: adjusted[name] for name in p_values}


def day_set_metrics(per_df: pd.DataFrame, examples: pd.DataFrame | None = None):
    """KION sensitivity: evaluate all positives of one observed user-day as a set.

    Contexts are averaged within user and then across users. Each unavailable
    raw item still contributes to the Recall denominator and ideal DCG. Raw
    ``item_id`` (or ``raw_item_id``) from ``examples`` distinguishes multiple OOV
    targets mapped to UNK. Without raw IDs, known tokens are deduplicated and
    every OOV row is conservatively treated as a distinct positive; this matches
    KION's unique user-item source rows but is explicitly reported as an
    assumption. This calendar-day evaluation is not intended for exact-event
    datasets such as Yambda.
    """
    required = {"row_id", "user_id", "day", "context_key", "target", "top_items"}
    if per_df.empty or not required.issubset(per_df):
        raise ValueError("Need nonempty user/day/context predictions with targets and top_items")
    if per_df.duplicated(["row_id", "user_id"]).any():
        raise ValueError("Evaluation rows must be unique")
    frame = per_df.copy()
    raw_available = False
    raw_name = None
    if examples is not None:
        if len(examples) != len(frame):
            raise ValueError("Prediction and raw-label row counts differ")
        raw_name = next((name for name in ("item_id", "raw_item_id") if name in examples), None)
        source = examples.reset_index(drop=True).copy()
        source["row_id"] = np.arange(len(source))
        identity = [name for name in ("user_id", "target", "day") if name in source]
        if raw_name:
            source["source_raw_item_id"] = source[raw_name]
        columns = ["row_id"]+identity+(["source_raw_item_id"] if raw_name else [])
        frame = frame.merge(source[columns], on="row_id", suffixes=("", "_raw"), validate="one_to_one")
        if len(frame) != len(per_df):
            raise ValueError("Prediction and raw-label row IDs differ")
        for name in identity:
            if not (frame[name] == frame[name+"_raw"]).all():
                raise ValueError(f"Prediction and raw-label identities disagree on {name}")
        raw_available = raw_name is not None
    def slate(value):
        values = json.loads(value) if isinstance(value, str) else value
        return tuple(int(item) for item in values)
    frame["_slate"] = frame.top_items.map(slate)
    keys = ["user_id", "day", "context_key"]
    records = []
    discounts = 1/np.log2(np.arange(2, 22, dtype=float))
    for index, (identity, group) in enumerate(frame.groupby(keys, sort=False, dropna=False)):
        if group._slate.nunique() != 1:
            raise ValueError("One user-day prefix has inconsistent recommendation slates")
        positives = set()
        known_tokens = set()
        for row in group.itertuples(index=False):
            token = int(row.target)
            raw = getattr(row, "source_raw_item_id", None) if raw_available else None
            if raw is not None and pd.notna(raw):
                positives.add(("raw", str(raw)))
            elif token >= 3:
                positives.add(("token", str(token)))
            else:
                positives.add(("unknown_row", str(row.row_id)))
            if token >= 3:
                known_tokens.add(token)
        n_positive = len(positives)
        selected = group._slate.iloc[0]
        record = {"row_id": index, "user_id": identity[0], "day": identity[1],
            "context_key": identity[2], "target_set_key": json.dumps(sorted(positives), ensure_ascii=False),
            "n_positives": n_positive, "n_known_positive_tokens": len(known_tokens),
            "n_source_rows": len(group)}
        for k in (10, 20):
            hits, dcg = set(), 0.
            for rank, item in enumerate(selected[:k]):
                if item in known_tokens and item not in hits:
                    hits.add(item)
                    dcg += discounts[rank]
            record[f"set_recall@{k}"] = len(hits)/n_positive
            record[f"set_ndcg@{k}"] = dcg/discounts[:min(k, n_positive)].sum()
        records.append(record)
    contexts = pd.DataFrame(records)
    metric_columns = [name for name in contexts if name.startswith(("set_recall@", "set_ndcg@"))]
    metrics = contexts.groupby("user_id")[metric_columns].mean().mean().to_dict()
    metrics.update({"set_n_contexts": len(contexts), "set_n_users": int(contexts.user_id.nunique()),
        "set_n_positive_items": int(contexts.n_positives.sum()),
        "set_aggregation": "user macro after averaging distinct user-day contexts",
        "set_raw_item_identity": raw_available,
        "set_oov_rule": "Distinct raw items" if raw_available else "Every OOV source row treated as a distinct positive"})
    return metrics, contexts
