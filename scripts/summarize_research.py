"""Export experiment tables and article inputs from completed runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SPLITS = ("validation", "test", "confirmation")
COUNT_FIELDS = {
    "n_examples", "n_users", "n_contexts", "factual_completion_n",
    "set_n_contexts", "set_n_users", "set_n_positive_items",
}
RM_MODELS = ("ensemble", "audit", "early_item_mean", "constant",
             "item_only_model", "history_only_model")


def read_json(path):
    return json.loads(Path(path).read_text())


def compact_metadata(value):
    """Keep scientific summaries, without runtime bookkeeping or ID lists."""
    if isinstance(value, dict):
        return {key: compact_metadata(item) for key, item in value.items()
                if key not in {"cohort", "item_ids", "environment"}}
    if isinstance(value, list):
        return [compact_metadata(item) for item in value]
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(values):
    values = [value for value in values if number(value)]
    return {
        "mean": statistics.fmean(values) if values else None,
        "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
        "n_runs": len(values),
    }


def summarize_run(directory, dataset):
    results = read_json(Path(directory) / "results.json")
    config, selection = results["config"], results["selection"]
    grouped = defaultdict(list)
    for name, entry in results["models"].items():
        grouped[entry["family"]].append(dict(name=name, **entry))
    families, per_run, rollouts = {}, [], []
    for family, entries in sorted(grouped.items()):
        summary = {"n_runs": len(entries), "seeds": [entry.get("seed") for entry in entries]}
        for key in ("training_seconds", "sampled_actions"):
            if any(number(entry.get(key)) for entry in entries):
                summary[key] = aggregate(entry.get(key) for entry in entries)
        for split in SPLITS:
            keys = {key for entry in entries for key, value in entry[split].items()
                    if number(value) and key not in COUNT_FIELDS}
            summary[split] = {key: aggregate(entry[split].get(key) for entry in entries) for key in sorted(keys)}
        families[family] = summary
        for entry in entries:
            item = {key: entry[key] for key in ("name", "family", "seed", *SPLITS)}
            item.update({key: entry[key] for key in ("variant", "rerank", "training_seconds", "sampled_actions") if key in entry})
            per_run.append(item)
            if "rollout" in entry:
                rollouts.append({"name": entry["name"], "family": family,
                                 "seed": entry["seed"], **entry["rollout"]})
    selected_stats = {}
    for split, comparisons in results["statistics"].items():
        selected_stats[split] = {}
        if selection.get("selected_rl"):
            selected_stats[split]["selected_vs_baseline"] = comparisons[f"{selection['selected_rl']}_vs_baseline"]
            for comparator in ("matched_control", "reranker"):
                selected_stats[split][f"selected_vs_{comparator}"] = comparisons[f"selected_rl_vs_{comparator}"]
    summary = {
        "dataset": dataset, "config": config, "selection": selection,
        "architecture": results["sft_selection"], "families": families,
        "per_run": per_run, "rollouts": rollouts, "rerank_grid": results["rerank_grid"],
        "reward": {"splits": results["reward_metrics"]},
        "statistics": selected_stats, "all_statistics": results["statistics"],
    }
    return compact_metadata(summary), compact_metadata(results)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            result[name] = item
    return result


def csv_tables(datasets):
    tables = defaultdict(list)
    for dataset, evidence in datasets.items():
        for family, entry in evidence["families"].items():
            for split in SPLITS:
                for metric, summary in entry[split].items():
                    tables["family_metrics"].append(dict(dataset=dataset, family=family, split=split, metric=metric, **summary))
        for item in evidence["per_run"]:
            for split in SPLITS:
                tables["per_run_metrics"].append({"dataset": dataset, "name": item["name"], "family": item["family"], "seed": item["seed"], "split": split,
                                                   **{key: value for key, value in item[split].items() if number(value) or value is None}})
        for table, source in (("selected_statistics", evidence["statistics"]), ("all_statistics", evidence["all_statistics"])):
            for split, comparisons in source.items():
                for comparison, metrics in comparisons.items():
                    for metric, stat in metrics.items():
                        tables[table].append({"dataset": dataset, "split": split, "comparison": comparison,
                                             "selected_rl": evidence["selection"]["selected_rl"], "primary_safety_metric": metric == "ndcg@10", **flatten(stat)})
        for split, entry in evidence["reward"]["splits"].items():
            for model in RM_MODELS:
                for metric, value in entry.get(model, {}).items():
                    if number(value) or value is None:
                        tables["reward_metrics"].append({"dataset": dataset, "split": split, "model": model, "metric": metric, "value": value, "n": entry.get("n"), "n_users": entry.get("n_users")})
            retention = entry.get("return7", {})
            for model in ("full", "history_only", "constant"):
                tables["retention_metrics"].append({"dataset": dataset, "split": split, "model": model,
                    **{key: retention.get(key) for key in ("n", "n_user_days", "weighting", "label_mean", "training_label_mean", "reason")}, **retention.get(model, {})})
            for table, key in (("reward_calibration", "completion_calibration"), ("reward_support", "support_error_bins")):
                for row in entry.get(key, []):
                    tables[table].append({"dataset": dataset, "split": split, **row})
            for head, row in entry.get("action_sensitivity", {}).items():
                if isinstance(row, dict):
                    tables["action_sensitivity"].append({"dataset": dataset, "split": split, "head": head, **row})
            tables["reward_diagnostics"].append({"dataset": dataset, "split": split, **flatten({key: entry[key] for key in (
                "ensemble_user_bootstrap", "ensemble_disagreement_diagnostics", "completion_calibration_gap", "duration_diagnostics", "feature_importance_groups") if key in entry})})
        for rollout in evidence["rollouts"]:
            for row in rollout["by_mode_step"]:
                tables["rollout_summary"].append({"dataset": dataset, **{key: rollout.get(key) for key in ("name", "family", "seed", "n_examples", "horizon", "exclude_observed")}, **row})
        for key, entry in evidence["rerank_grid"].items():
            for seed, metrics in entry["by_seed"].items():
                tables["rerank_grid"].append({"dataset": dataset, "recipe": key, "seed": seed, "alpha": entry["alpha"], "reward_type": entry["reward_type"],
                                             "selected": key == evidence["selection"]["reranker"]["key"], **{name: value for name, value in metrics.items() if number(value)}})
    return tables


def write_outputs(output, datasets, raw_results):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = csv_tables(datasets)
    for name, rows in sorted(tables.items()):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    for dataset, results in raw_results.items():
        (output / f"{dataset}_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    path = output / "summary.json"
    path.write_text(json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kion", type=Path)
    parser.add_argument("--yambda", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    paths = {dataset: path for dataset, path in (("kion", args.kion), ("yambda", args.yambda)) if path is not None}
    if not paths:
        parser.error("Supply --kion and/or --yambda")
    summarized = {dataset: summarize_run(path, dataset) for dataset, path in paths.items()}
    path = write_outputs(args.output, {dataset: pair[0] for dataset, pair in summarized.items()},
                         {dataset: pair[1] for dataset, pair in summarized.items()})
    print(f"Saved experiment tables to {path.parent}")


if __name__ == "__main__":
    main()
