"""Describe out-of-catalog targets using the cohort's observed training prefix."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ('train', 'rm_valid', 'valid', 'test', 'confirmation')
CATEGORIES = ('in_vocab', 'train_seen_excluded', 'unseen_in_cohort_train')



def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_kion(path, cohort, train_end):
    counts, source_rows, cohort_rows = Counter(), 0, 0
    cutoff = pd.Timestamp(train_end)
    for frame in pd.read_csv(path, usecols=["user_id", "item_id", "last_watch_dt"], chunksize=250000):
        source_rows += len(frame)
        selected = frame[frame.user_id.isin(cohort)]
        cohort_rows += len(selected)
        prefix = selected[pd.to_datetime(selected.last_watch_dt) < cutoff]
        counts.update({int(raw): int(count) for raw, count in prefix.item_id.value_counts().items()})
    return counts, {
        "source_rows_scanned": source_rows, "cohort_source_rows": cohort_rows,
        "train_cutoff": str(cutoff.date()),
        "rule": "All aggregated KION user-item source rows of the frozen cohort with last_watch_dt strictly before train_end, including the initial history before supervised targets.",
    }


def count_yambda(path, cohort, train_end, batch_users=128):
    counts, scanned_users, selected_users, selected_listens = Counter(), 0, set(), 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_users, columns=["uid", "timestamp", "played_ratio_pct", "item_id"]):
        columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
        uids = columns["uid"].to_numpy(zero_copy_only=False)
        scanned_users += len(uids)
        for row, uid in enumerate(uids):
            if int(uid) not in cohort:
                continue
            require(int(uid) not in selected_users, f"Duplicate Yambda user row: {uid}")
            selected_users.add(int(uid))
            # Listen identity exactly follows the official adapter. Ratio values
            # are never read: only validity distinguishes listens from feedback.
            listen = pc.is_valid(columns["played_ratio_pct"][row].values).to_numpy(zero_copy_only=False)
            times = columns["timestamp"][row].values.to_numpy(zero_copy_only=False)
            raw_ids = columns["item_id"][row].values.to_numpy(zero_copy_only=False)
            require(len(listen) == len(times) == len(raw_ids), f"Yambda event-list length mismatch: {uid}")
            selected_listens += int(listen.sum())
            prefix = listen & (times < int(train_end))
            counts.update(map(int, raw_ids[prefix]))
    require(selected_users == cohort, "Some frozen cohort users are missing from the Yambda source")
    return counts, {
        "source_user_rows_scanned": scanned_users, "cohort_source_users": len(selected_users),
        "cohort_all_period_listens": selected_listens, "train_cutoff_relative_seconds": int(train_end),
        "rule": "All listen events of the frozen cohort strictly before train_end_timestamp, both organic and recommendation-driven; explicit feedback is excluded by null playback-field validity, exactly as in the frozen adapter.",
    }


def classify_frame(frame, split, cohort, vocabulary, train_items):
    raw_column = next((column for column in ("item_id", "raw_item_id") if column in frame), None)
    require(raw_column is not None, f"{split}: prepared targets have no raw item-ID column")
    require({"user_id", "target"}.issubset(frame), f"{split}: missing target/user identity")
    require(frame[raw_column].notna().all(), f"{split}: missing raw target item IDs")
    users = frame.user_id.to_numpy(dtype=np.int64)
    raw_ids = frame[raw_column].to_numpy(dtype=np.int64)
    targets = frame.target.to_numpy(dtype=np.int64)
    require(set(map(int, users)).issubset(cohort), f"{split}: targets outside the frozen cohort")
    expected_tokens = np.fromiter((vocabulary.get(int(raw), 2) for raw in raw_ids), dtype=np.int64, count=len(raw_ids))
    require(np.array_equal(expected_tokens, targets), f"{split}: raw-ID/token mapping differs from the frozen catalogue")
    codes = np.fromiter((0 if int(raw) in vocabulary else 1 if int(raw) in train_items else 2 for raw in raw_ids), dtype=np.int8, count=len(raw_ids))
    require(np.array_equal(codes == 0, targets >= 3), f"{split}: in_vocab iff target>=3 failed")
    if split == "train":
        require(not (codes == 2).any(), "A prepared training target was not seen in the reconstructed cohort train prefix")
    work = pd.DataFrame({"user_id": users, "category": codes})
    user_counts = pd.crosstab(work.user_id, work.category).reindex(columns=range(3), fill_value=0)
    macro = user_counts.div(user_counts.sum(axis=1), axis=0).mean(axis=0)
    class_counts = {category: int((codes == index).sum()) for index, category in enumerate(CATEGORIES)}
    n_rows = len(frame)
    oov = class_counts["train_seen_excluded"]+class_counts["unseen_in_cohort_train"]
    return {
        "n_rows": n_rows, "n_users": len(user_counts), "raw_item_column": raw_column,
        "n_distinct_raw_target_items": len(np.unique(raw_ids)), "class_counts": class_counts,
        "class_user_counts": {category: int((user_counts[index] > 0).sum()) for index, category in enumerate(CATEGORIES)},
        "class_distinct_item_counts": {category: len(np.unique(raw_ids[codes == index])) for index, category in enumerate(CATEGORIES)},
        "fractions_micro": {category: count/n_rows if n_rows else None for category, count in class_counts.items()},
        "fractions_user_macro": {category: float(macro[index]) if len(user_counts) else None for index, category in enumerate(CATEGORIES)},
        "oov_composition_micro": {category: class_counts[category]/oov if oov else None for category in CATEGORIES[1:]},
    }


def audit_run(run, dataset, raw_directory):
    run, raw_directory = Path(run).resolve(), Path(raw_directory).resolve()
    config_path, manifest_path, prepared_path = (run/name for name in ("config.json", "data_manifest.json", "prepared.pkl"))
    config, manifest = json.loads(config_path.read_text()), json.loads(manifest_path.read_text())
    require(config.get("dataset") == dataset, f"Expected {dataset} data run: {run}")
    # Pickle is an existing trusted data artifact. Outcomes may be serialized in
    # it, but are never accessed or used by this audit.
    with prepared_path.open("rb") as handle:
        prepared = pickle.load(handle)
    require(prepared.manifest == manifest, f"{dataset}: prepared and saved data manifests differ")
    require(prepared.item_ids == manifest["item_ids"], f"{dataset}: prepared and frozen catalogue IDs differ")
    cohort = set(map(int, manifest["cohort"]))
    require(len(cohort) == len(manifest["cohort"]), f"{dataset}: duplicated cohort users")
    item_ids = manifest["item_ids"]
    require(item_ids[:3] == [-1, -2, -3] and len(set(item_ids)) == len(item_ids), f"{dataset}: invalid reserved/real catalogue IDs")
    vocabulary = {int(raw): token for token, raw in enumerate(item_ids) if token >= 3}
    source_name = "interactions.csv" if dataset == "kion" else "yambda50m_multi.parquet"
    source = raw_directory/source_name
    if dataset == "kion":
        counts, scan = count_kion(source, cohort, manifest["config"]["train_end"])
    else:
        cutoff = manifest["config"].get("train_end_timestamp")
        require(cutoff is not None, "Yambda manifest needs an exact-second train_end_timestamp")
        counts, scan = count_yambda(source, cohort, cutoff)
    require(bool(counts), f"{dataset}: empty reconstructed training catalogue")
    ranked = sorted(counts, key=lambda raw: (-counts[raw], raw))
    require(ranked[:len(vocabulary)] == item_ids[3:], f"{dataset}: reconstructed frequency/ID tie-break catalogue differs from the frozen adapter")
    splits = {}
    for split in SPLITS:
        frame = getattr(prepared, split)
        require(len(frame) == manifest["rows"][split], f"{dataset}/{split}: prepared and manifest row counts differ")
        splits[split] = classify_frame(frame, split, cohort, vocabulary, counts)
        recorded_oov = manifest.get("unknown_target_fraction", {}).get(split)
        if recorded_oov is not None:
            actual_oov = 1-splits[split]["fractions_micro"]["in_vocab"]
            require(np.isclose(actual_oov, recorded_oov, rtol=0, atol=1e-12), f"{dataset}/{split}: reconstructed OOV fraction differs from the manifest")
    del prepared
    evidence = {
        "n_cohort_users": len(cohort), "n_catalog_items": len(vocabulary),
        "train_prefix": {**scan, "n_raw_events": sum(counts.values()), "n_distinct_raw_items": len(counts),
            "n_train_items_excluded_from_catalog": len(counts)-len(vocabulary),
            "in_vocab_raw_event_fraction": sum(counts[raw] for raw in vocabulary)/sum(counts.values()),
            "catalog_rule": "Descending cohort-train raw item frequency, ties ascending raw item ID."},
        "splits": splits,
    }
    return evidence, counts, vocabulary


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def save(output, analyzed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'catalog.json'
    datasets = json.loads(path.read_text()).get('datasets', {}) if path.exists() else {}
    datasets.update({dataset: evidence for dataset, (evidence, _, _) in analyzed.items()})
    rows = []
    for dataset, evidence in sorted(datasets.items()):
        for split, summary in evidence['splits'].items():
            for category in CATEGORIES:
                rows.append({'dataset': dataset, 'split': split, 'category': category,
                             'n_rows': summary['n_rows'], 'n_users': summary['n_users'],
                             'class_count': summary['class_counts'][category],
                             'class_user_count': summary['class_user_counts'][category],
                             'class_distinct_item_count': summary['class_distinct_item_counts'][category],
                             'fraction_micro': summary['fractions_micro'][category],
                             'fraction_user_macro': summary['fractions_user_macro'][category]})
    pd.DataFrame(rows).to_csv(output / 'catalog.csv', index=False)
    result = {'definitions': {
        'in_vocab': 'Target belongs to the recommendation catalogue.',
        'train_seen_excluded': 'Target appeared in the cohort training prefix but did not enter the top-frequency catalogue.',
        'unseen_in_cohort_train': 'Target did not appear in this cohort before its training cutoff; it may exist elsewhere in the service.',
        'fractions_micro': 'Fraction of target rows, including OOV targets.',
        'fractions_user_macro': 'Within-user fraction, then equal average over users.',
        'training_counts': 'All raw cohort train events, including initial history and organic Yambda listens.'},
        'datasets': datasets}
    write_json(path, result)
    return path



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kion", type=Path); parser.add_argument("--yambda", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    runs = {dataset: run for dataset, run in (("kion", args.kion), ("yambda", args.yambda)) if run is not None}
    if not runs: parser.error("Pass --kion and/or --yambda")
    analyzed = {}
    for dataset, run in runs.items():
        print(f"CATALOG {dataset}: raw IDs only", flush=True)
        analyzed[dataset] = audit_run(run, dataset, args.data)
    path = save(args.output, analyzed)
    print(json.dumps({"catalog": str(path.resolve()), "test_fractions_micro": {
        dataset: evidence["splits"]["test"]["fractions_micro"] for dataset, (evidence, _, _) in analyzed.items()}}, ensure_ascii=False))

if __name__ == '__main__':
    main()
