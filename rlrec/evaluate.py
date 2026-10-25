"""Full-catalog evaluation and explicitly model-based rollout diagnostics.

Callbacks keep this module independent of a particular PyTorch model. Unknown
targets remain in metric denominators; observed history is excluded from slates.
No metric in this module estimates the causal value of a deployed policy.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

LogitsFn = Callable[[list[list[int]]], np.ndarray]
RewardFn = Callable[[list[list[int]], np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]]


def _logits(logits_fn: LogitsFn, histories: list[list[int]], vocab_size: int) -> np.ndarray:
    scores = np.asarray(logits_fn(histories), dtype=np.float64)
    if scores.shape != (len(histories), vocab_size):
        raise ValueError(f"Expected logits {(len(histories), vocab_size)}, got {scores.shape}")
    if not np.isfinite(scores[:, 3:]).all():
        raise ValueError("Real-item logits must be finite")
    return scores


def _mask(scores: np.ndarray, histories: Sequence[Sequence[int]], exclude_seen: bool = True) -> np.ndarray:
    scores = scores.copy()
    scores[:, :3] = -np.inf
    if exclude_seen:
        for row, history in zip(scores, histories):
            seen = np.asarray(history, dtype=int)
            seen = seen[(seen >= 3) & (seen < scores.shape[1])]
            row[seen] = -np.inf
    return scores


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    # Stable sort makes ties reproducible across platforms and never invents items
    # if fewer than k eligible items remain.
    order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, scores.shape[1])]
    order[~np.isfinite(np.take_along_axis(scores, order, axis=1))] = -1
    return order


def _rewards(reward_fn: RewardFn, histories: list[list[int]], candidates: np.ndarray) -> dict[str, np.ndarray]:
    valid = candidates >= 3
    safe_candidates = np.where(valid, candidates, 3)
    result = reward_fn(histories, safe_candidates)
    if isinstance(result, dict):
        heads = {key: np.asarray(value, dtype=float) for key, value in result.items()}
    elif isinstance(result, tuple):
        heads = {"mean": np.asarray(result[0], dtype=float), "std": np.asarray(result[1], dtype=float)}
    else:
        heads = {"mean": np.asarray(result, dtype=float)}
    if "mean" not in heads:
        raise ValueError("Reward dictionary must contain mean")
    for values in heads.values():
        if values.shape != candidates.shape:
            raise ValueError("Reward callback must return arrays shaped like candidates")
        if not np.isfinite(values[valid]).all():
            raise ValueError("Reward predictions must be finite")
    return {key: np.where(valid, values, np.nan) for key, values in heads.items()}


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _slate_diversity(items: Sequence[int], genres: Mapping[int, Sequence[str]]) -> float:
    sets = [set(genres.get(int(item), ())) for item in items if item >= 3]
    distances = [1 - len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :] if a | b]
    return _mean(distances)


def evaluate_policy(
    logits_fn: LogitsFn,
    examples: pd.DataFrame,
    vocab_size: int,
    reward_fn: RewardFn | None = None,
    popularity: np.ndarray | None = None,
    item_genres: Mapping[int, Sequence[str]] | None = None,
    batch_size: int = 128,
    ks: Sequence[int] = (10, 20),
    seed: int = 42,
    exclude_seen: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate every row on the same frozen, full real-item vocabulary.

    ``examples`` needs history and target; user_id/day are copied if supplied.
    A row has one next-item target, so Recall@K equals HitRate@K here. Targets
    equal to UNK (2) are failures rather than silently removed observations.
    ``popularity`` must contain training-only interaction counts by token.
    Reward values are predictions on recommended items, never factual outcomes.
    """
    if examples.empty or vocab_size < 4 or batch_size < 1:
        raise ValueError("Need nonempty examples, a real-item vocabulary, and positive batch_size")
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) < 1:
        raise ValueError("ks must contain positive integers")
    if popularity is not None:
        popularity = np.asarray(popularity, dtype=float)
        if popularity.shape != (vocab_size,) or (popularity < 0).any():
            raise ValueError("popularity must be a nonnegative vector of vocab_size")
    rows: list[dict] = []
    max_k = max(ks)
    for start in range(0, len(examples), batch_size):
        batch = examples.iloc[start : start + batch_size]
        histories = [[int(x) for x in history] for history in batch.history]
        scores = _mask(_logits(logits_fn, histories, vocab_size), histories, exclude_seen=exclude_seen)
        top = _topk(scores, max_k)
        rewards = None if reward_fn is None else _rewards(reward_fn, histories, top)
        for j, (_, example) in enumerate(batch.iterrows()):
            target = int(example.target)
            slate = top[j]
            match = np.flatnonzero(slate == target) if target >= 3 else np.array([], dtype=int)
            rank = int(match[0] + 1) if match.size else 0
            row = {
                "row_id": start + j,
                "user_id": int(example.user_id) if "user_id" in example else start + j,
                "target": target,
                "known_target": bool(3 <= target < vocab_size),
                "history_length": sum(x >= 3 for x in histories[j]),
                "context_key": ",".join(map(str, histories[j])),
                "rank": rank,
                "top_items": [int(x) for x in slate if x >= 3],
            }
            if "day" in example:
                row["day"] = int(example.day)
            for k in ks:
                hit = bool(0 < rank <= k)
                row[f"recall@{k}"] = float(hit)
                row[f"ndcg@{k}"] = float(1 / np.log2(rank + 1)) if hit else 0.0
                row[f"mrr@{k}"] = float(1 / rank) if hit else 0.0
                if rewards is not None:
                    names = {"mean": "predicted_reward", "std": "reward_uncertainty", "audit": "audit_reward", "return7": "predicted_return7"}
                    for head, values in rewards.items():
                        row[f"{names.get(head, 'predicted_' + head)}@{k}"] = _mean(values[j, :k])
                if popularity is not None:
                    valid = slate[:k][slate[:k] >= 3]
                    row[f"log_popularity@{k}"] = _mean(np.log1p(popularity[valid]))
                    probability = (popularity[valid] + 1) / (popularity[3:].sum() + vocab_size - 3)
                    row[f"novelty_bits@{k}"] = _mean(-np.log2(probability))
                if item_genres is not None:
                    row[f"genre_diversity@{k}"] = _slate_diversity(slate[:k], item_genres)
            if rewards is not None:
                row["predicted_reward@1"] = float(rewards["mean"][j, 0])
            rows.append(row)
    per_example = pd.DataFrame(rows)
    metric_columns = [col for col in per_example if "@" in col]
    user_metrics = per_example.groupby("user_id")[metric_columns].mean()
    metrics = {col: _mean(user_metrics[col]) for col in metric_columns}
    metrics.update({f"micro_{col}": _mean(per_example[col]) for col in metric_columns if col.startswith(("recall", "ndcg", "mrr"))})
    context_keys = ["user_id", "context_key"] + (["day"] if "day" in per_example else [])
    contexts = per_example.drop_duplicates(context_keys)
    metrics.update({
        "n_examples": len(rows),
        "n_users": int(per_example.user_id.nunique()),
        "n_contexts": len(contexts),
        "aggregation": "user_macro: average examples within user, then users; behavior on unique contexts",
        "known_target_rate": float(per_example.known_target.mean()),
        "history_length_mean": float(per_example.history_length.mean()),
        "empty_slate_rate": float(np.mean([not row["top_items"] for row in rows])),
        "exclude_seen": bool(exclude_seen),
    })
    for k in ks:
        slates = [slate[:k] for slate in contexts.top_items]
        counts = Counter(item for slate in slates for item in slate)
        frequencies = np.asarray(list(counts.values()), dtype=float)
        probabilities = frequencies / frequencies.sum() if frequencies.size else frequencies
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        metrics[f"catalog_coverage@{k}"] = len(counts) / (vocab_size - 3)
        metrics[f"recommendation_entropy@{k}"] = entropy
        metrics[f"normalized_entropy@{k}"] = entropy / np.log(vocab_size - 3) if vocab_size > 4 else 0.0
        metrics[f"top_item_share@{k}"] = float(probabilities.max()) if probabilities.size else 0.0
        metrics[f"unique_slate_ratio@{k}"] = len(set(map(tuple, slates))) / len(slates)
        rng = np.random.default_rng(seed)
        similarities = []
        if len(slates) > 1:
            for _ in range(min(1000, len(slates) * 2)):
                a, b = rng.choice(len(slates), 2, replace=False)
                left, right = set(slates[a]), set(slates[b])
                if left | right:
                    similarities.append(len(left & right) / len(left | right))
        metrics[f"personalization@{k}"] = 1 - _mean(similarities)
    first = Counter(slate[0] for slate in contexts.top_items if slate)
    metrics["top1_concentration"] = max(first.values(), default=0) / len(contexts)
    return metrics, per_example


def bootstrap_difference(
    baseline: pd.DataFrame,
    tuned: pd.DataFrame,
    column: str,
    seed: int = 42,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Paired user-cluster bootstrap of the user-macro difference.

    Entire users are resampled to preserve correlations between their days.
    Point estimate gives every user the same weight as evaluate_policy.
    """
    keys = ["row_id", "user_id"]
    for frame in (baseline, tuned):
        if frame.duplicated(keys).any():
            raise ValueError("Evaluation row/user keys must be unique")
    pairs = baseline[keys + [column]].merge(tuned[keys + [column]], on=keys, suffixes=("_base", "_tuned"), validate="one_to_one")
    if len(pairs) != len(baseline) or len(pairs) != len(tuned):
        raise ValueError("Paired bootstrap needs exactly the same evaluation examples")
    if pairs.empty or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Need paired rows, positive n_bootstrap and confidence in (0,1)")
    for identity in ("target", "day"):
        if identity in baseline and identity in tuned:
            identities = baseline[keys + [identity]].merge(tuned[keys + [identity]], on=keys, suffixes=("_base", "_tuned"))
            if not (identities[f"{identity}_base"] == identities[f"{identity}_tuned"]).all():
                raise ValueError(f"Paired examples disagree on {identity}")
    pairs["delta"] = pairs[f"{column}_tuned"] - pairs[f"{column}_base"]
    if not np.isfinite(pairs.delta).all():
        raise ValueError("Bootstrap values must be finite")
    groups = pairs.groupby("user_id").delta.mean()
    values = groups.to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, len(groups), size=len(groups))
        draws[i] = values[indices].mean()
    alpha = (1 - confidence) / 2
    return {
        "metric": column,
        "delta": float(values.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "confidence": confidence,
        "n_users": len(groups),
        "n_examples": len(pairs),
        "n_bootstrap": n_bootstrap,
        "unit": "user_cluster_macro",
    }


def _probabilities(scores: np.ndarray) -> np.ndarray:
    maxima = np.max(scores, axis=1, keepdims=True)
    finite = np.isfinite(maxima)
    shifted = scores - np.where(finite, maxima, 0)
    exponents = np.exp(shifted)
    total = exponents.sum(axis=1, keepdims=True)
    return np.divide(exponents, total, out=np.zeros_like(exponents), where=total > 0)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    middle = (p + q) / 2
    def term(x: np.ndarray) -> np.ndarray:
        ratio = np.divide(x, middle, out=np.ones_like(x), where=middle > 0)
        return np.where(x > 0, x * np.log(np.maximum(ratio, 1e-300)), 0).sum(axis=1)
    return (term(p) + term(q)) / 2


def evaluate_rollouts(
    logits_fn: LogitsFn,
    examples: pd.DataFrame,
    vocab_size: int,
    reward_fn: RewardFn | None = None,
    horizon: int = 5,
    max_examples: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    exclude_observed: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Compare greedy autoregressive, frozen-prefix and unmasked rollouts.

    Safe modes exclude already generated items. With exclude_observed=True they
    also exclude observed items; setting it False permits music re-listening.
    Unmasked mode is diagnostic and allows repetitions. Generated items are hypothetical actions,
    never real user responses; subsequent reward values are simulator estimates.
    JS drift compares distributions on the same original unseen-item support.
    """
    if examples.empty or horizon < 1 or max_examples < 1 or batch_size < 1 or vocab_size < 4:
        raise ValueError("Need examples, real items and positive rollout limits")
    rng = np.random.default_rng(seed)
    # Multiple positive targets may share one user-day prefix. Sample contexts,
    # so days with many positives do not multiply the same hypothetical rollout.
    context_frame = pd.DataFrame({"context": [tuple(map(int, history)) for history in examples.history]})
    for key in ("user_id", "day"):
        if key in examples:
            context_frame[key] = examples[key].to_numpy()
    positions = context_frame.drop_duplicates().index.to_numpy()
    indices = np.sort(rng.choice(positions, min(max_examples, len(positions)), replace=False))
    chosen = examples.iloc[indices]
    records = []
    for start in range(0, len(chosen), batch_size):
        batch = chosen.iloc[start : start + batch_size]
        originals = [[int(x) for x in history] for history in batch.history]
        frozen_logits = _logits(logits_fn, originals, vocab_size)
        frozen_probs = _probabilities(_mask(frozen_logits, originals, exclude_seen=exclude_observed))
        for mode in ("autoregressive_safe", "frozen_prefix_safe", "autoregressive_unmasked"):
            current = [history.copy() for history in originals]
            generated: list[list[int]] = [[] for _ in originals]
            for step in range(1, horizon + 1):
                scores = frozen_logits if mode == "frozen_prefix_safe" else _logits(logits_fn, current, vocab_size)
                js = _js_divergence(_probabilities(_mask(scores, originals, exclude_seen=exclude_observed)), frozen_probs)
                forbidden = current if exclude_observed else generated
                eligible = _mask(scores, forbidden, exclude_seen=mode != "autoregressive_unmasked")
                actions = _topk(eligible, 1)
                rm_contexts = originals if mode == "frozen_prefix_safe" else current
                if reward_fn is not None:
                    r_generated = _rewards(reward_fn, rm_contexts, actions)
                    r_original = _rewards(reward_fn, originals, actions)
                for j, (_, example) in enumerate(batch.iterrows()):
                    action = int(actions[j, 0])
                    row = {
                        "row_id": int(indices[start + j]),
                        "user_id": int(example.user_id) if "user_id" in example else int(indices[start + j]),
                        "mode": mode,
                        "step": step,
                        "item": action,
                        "seen_in_observed_history": float(action >= 3 and action in originals[j]),
                        "repeated_generated_item": float(action >= 3 and action in generated[j]),
                        "js_from_original_context": float(js[j]),
                    }
                    if reward_fn is not None:
                        row["predicted_reward_generated_context"] = float(r_generated["mean"][j, 0])
                        row["predicted_reward_original_context"] = float(r_original["mean"][j, 0])
                        row["reward_context_gap"] = float(r_generated["mean"][j, 0] - r_original["mean"][j, 0])
                        for head in ("audit", "return7", "std"):
                            if head in r_generated:
                                row[f"{head}_generated_context"] = float(r_generated[head][j, 0])
                                row[f"{head}_original_context"] = float(r_original[head][j, 0])
                    records.append(row)
                    if action >= 3:
                        generated[j].append(action)
                        current[j].append(action)
    frame = pd.DataFrame(records)
    columns = ["seen_in_observed_history", "repeated_generated_item", "js_from_original_context"]
    if reward_fn is not None:
        columns += ["predicted_reward_generated_context", "predicted_reward_original_context", "reward_context_gap"]
        columns += [col for col in frame if col.endswith(("_generated_context", "_original_context")) and col not in columns]
    summary = {
        "n_examples": len(chosen),
        "horizon": horizon,
        "exclude_observed": bool(exclude_observed),
        "interpretation": "Hypothetical model-generated contexts; no observed responses or causal policy value.",
        "by_mode_step": frame.groupby(["mode", "step"])[columns].mean().reset_index().to_dict("records"),
    }
    return summary, frame
