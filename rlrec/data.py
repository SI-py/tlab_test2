"""Day-aware KION preparation. No order is invented among same-day targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULTS = dict(n_users=15000, vocab_size=1500, seed=42, min_history=3,
                max_length=32, max_train_examples=120000,
                train_end="2021-07-19", rm_valid_end="2021-07-26",
                valid_end="2021-08-02", test_end="2021-08-09")


def day_number(value):
    return int(pd.Timestamp(value).toordinal())


@dataclass
class PreparedData:
    train: pd.DataFrame
    rm_valid: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame
    item_ids: list[int]
    manifest: dict
    config: dict

    @property
    def vocab_size(self):
        return len(self.item_ids)


def prepare_frames(events: pd.DataFrame, items: pd.DataFrame, config=None):
    cfg = DEFAULTS | (config or {})
    ends = [day_number(cfg[k]) for k in ("train_end", "rm_valid_end", "valid_end", "test_end")]
    if not all(a < b for a, b in zip(ends, ends[1:])):
        raise ValueError("Split boundaries must be strictly increasing")
    events = events.copy()
    events["day"] = pd.to_datetime(events.last_watch_dt).map(pd.Timestamp.toordinal).astype(int)
    if events.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Expected aggregated unique KION user-item pairs")
    prefix = events[events.day < ends[0]]
    counts = prefix.groupby("user_id").size()
    eligible = np.sort(counts[counts >= cfg["min_history"] + 1].index.to_numpy())
    if len(eligible) == 0:
        raise ValueError("No train-eligible users")
    rng = np.random.default_rng(cfg["seed"])
    cohort = np.sort(rng.choice(eligible, min(cfg["n_users"], len(eligible)), replace=False))
    events = events[events.user_id.isin(cohort)].sort_values(["user_id", "day", "item_id"])
    prefix = events[events.day < ends[0]]
    popularity = prefix.groupby("item_id").size().rename("count").reset_index()
    popularity = popularity.sort_values(["count", "item_id"], ascending=[False, True])
    item_ids = [-1, -2, -3] + popularity.item_id.head(cfg["vocab_size"]).astype(int).tolist()
    mapping = {raw: token for token, raw in enumerate(item_ids) if token >= 3}
    events["token"] = events.item_id.map(mapping).fillna(2).astype(int)
    max_day = int(events.day.max())
    rows = [[], [], [], []]
    for uid, user in events.groupby("user_id", sort=True):
        history = []
        all_days = np.sort(user.day.unique())
        for day, batch in user.groupby("day", sort=True):
            split = int(np.searchsorted(ends, day, side="right"))
            if split >= 4:
                break
            next_days = all_days[all_days > day]
            label_end = ends[0] if split == 0 else max_day + 1
            mature = day + 7 < label_end
            return7 = float(bool(len(next_days) and next_days[0] <= day + 7)) if mature else np.nan
            if len(history) >= cfg["min_history"]:
                for row in batch.itertuples():
                    reward = float(np.clip(row.watched_pct / 100.0, 0, 1)) if pd.notna(row.watched_pct) else np.nan
                    rows[split].append(dict(user_id=int(uid), day=int(day),
                        history=history.copy(), target=int(row.token),
                        item_id=int(row.item_id), reward=reward, return7=return7,
                        total_dur=float(row.total_dur)))
            # Deterministic *past-day* representation; no same-day target is in context.
            history.extend(batch.token.astype(int).tolist())
    columns = ["user_id", "day", "history", "target", "item_id", "reward", "return7", "total_dur"]
    frames = [pd.DataFrame(part, columns=columns) for part in rows]
    original_train_rows = len(frames[0])
    if len(frames[0]) > cfg["max_train_examples"]:
        chosen = np.sort(rng.choice(len(frames[0]), cfg["max_train_examples"], replace=False))
        frames[0] = frames[0].iloc[chosen].reset_index(drop=True)
    metadata = items.copy()
    metadata["token"] = metadata.item_id.map(mapping)
    metadata = metadata[metadata.token.notna()].copy()
    metadata["token"] = metadata.token.astype(int)
    manifest = dict(config=cfg, cohort=cohort.tolist(), item_ids=item_ids,
                    rows={name: len(frame) for name, frame in zip(("train", "rm_valid", "valid", "test"), frames)},
                    original_train_rows=original_train_rows, max_observed_day=max_day,
                    unknown_target_fraction={name: float((frame.target == 2).mean()) if len(frame) else None
                        for name, frame in zip(("train", "rm_valid", "valid", "test"), frames)},
                    return7_label_rule="Train: day+7 < train_end. Evaluation: day+7 <= max observed day; labels use future only for scoring, never inputs.",
                    ordering="Each target of a user-day has identical context containing earlier days only; earlier day ties sorted by item_id.")
    return PreparedData(*frames, metadata, item_ids, manifest, cfg)


def prepare_data(raw_dir, config=None):
    root = Path(raw_dir)
    events = pd.read_csv(root / "interactions.csv", usecols=["user_id", "item_id", "last_watch_dt", "watched_pct", "total_dur"])
    items = pd.read_csv(root / "items.csv", usecols=["item_id", "content_type", "release_year", "genres", "title"])
    return prepare_frames(events, items, config)
