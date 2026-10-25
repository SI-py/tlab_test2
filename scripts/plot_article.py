"""Publication figures from frozen research-v2 exports; no model scoring.

Default dimensions match the printed dimensions in article.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tbank-article-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


DATASETS = ("kion", "yambda")
WIDTH = 6.6
HEIGHT_POINTS = {"reward_calibration.png": 160, "frontier.png": 280,
                 "duration.png": 185, "rollout.png": 210,
                 "rerank_validation.png": 216}
SHORT_NAMES = {
    "popularity": "Популярность", "itemknn": "ItemKNN", "baseline": "GPT-2",
    "matched_control": "KL+CE (PG=0)", "raw_h5": "Raw RL",
    "anchored_h1": "RL H=1", "anchored_h5": "RL H=5", "appended_h5": "RM append",
    "conservative_h5": "Консерв.", "retention_h5": "Возврат",
    "duration_neutral_h5": "Без тренда", "seconds_h5": "Секунды", "reranker": "Rerank",
}
COLORS = {
    "baseline": "#252525", "itemknn": "#846b42", "matched_control": "#777777", "raw_h5": "#a45e50",
    "anchored_h1": "#7895a9", "anchored_h5": "#456c89", "appended_h5": "#79939c",
    "conservative_h5": "#7a8461", "retention_h5": "#8d7858",
    "duration_neutral_h5": "#aa8a50", "seconds_h5": "#8b7094", "reranker": "#4b7b6b",
}
SELECTED_COLOR = "#8b7094"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.15,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
    })


def groups(result):
    grouped = defaultdict(list)
    for model in result["models"].values():
        grouped[model["family"]].append(model)
    return grouped


def values(entries, metric):
    return np.asarray([entry["test"][metric] for entry in entries
                       if isinstance(entry.get("test", {}).get(metric), (int, float))
                       and np.isfinite(entry["test"][metric])], float)


def panel(ax, title=None):
    if title:
        ax.set_title(title, pad=5, fontweight="normal")
    ax.grid(alpha=.18, linewidth=.55)
    ax.set_axisbelow(True)
    for coordinate in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        coordinate.set_major_formatter(formatter)


def missing(ax, message="Нет сохранённого результата"):
    ax.set_axis_off()
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, fontsize=8)


def save(fig, output, name):
    fig.savefig(Path(output)/name, dpi=300)
    plt.close(fig)


def annotate_means(ax, labels, point_locations):
    """Greedy display-space layout for five named policies and leader lines."""
    fig = ax.figure; fig.canvas.draw()
    renderer = fig.canvas.get_renderer(); scale = fig.dpi/72
    bounds = ax.get_window_extent(renderer); occupied = []
    locations = ax.transData.transform(np.asarray(point_locations))
    offsets = [(dx, dy) for dy in (8, -10, 21, -23, 35, -37, 49, -51)
               for dx in (6, -6, 19, -19, 34, -34, 0)]
    for text, x, y, color in labels:
        anchor = ax.transData.transform((x, y))
        width, height, _ = renderer.get_text_width_height_descent(text, FontProperties(family="DejaVu Serif", size=8), False)
        width += 4*scale; height += 4*scale
        candidates = []
        for dx, dy in offsets:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            px, py = anchor+np.asarray((dx, dy))*scale
            left = px if ha == "left" else px-width if ha == "right" else px-width/2
            box = (left, py-height/2, left+width, py+height/2)
            outside = max(0, bounds.x0+2-box[0])+max(0, box[2]-bounds.x1+2)+max(0, bounds.y0+2-box[1])+max(0, box[3]-bounds.y1+2)
            overlap = sum(max(0, min(box[2], old[2])-max(box[0], old[0])) *
                          max(0, min(box[3], old[3])-max(box[1], old[1])) for old in occupied)
            obscured = sum(box[0]-3 < px0 < box[2]+3 and box[1]-3 < py0 < box[3]+3 for px0, py0 in locations)
            cost = 1000*outside + 20*overlap + 120*obscured + dx*dx*.03 + dy*dy*.035
            candidates.append((cost, dx, dy, ha, box))
        _, dx, dy, ha, box = min(candidates, key=lambda item: item[0]); occupied.append(box)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    color=color, fontsize=8, zorder=8, annotation_clip=False,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .25, "alpha": .9},
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": .45, "alpha": .65,
                                "shrinkA": 1, "shrinkB": 3})


def calibration(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT_POINTS["reward_calibration.png"]/72), sharex=True, sharey=True)
    counts = [row["n"] for result in results.values() for row in result["reward_metrics"]["test"]["completion_calibration"] if row["n"]]
    largest = max(counts, default=1)
    for ax, dataset in zip(axes, DATASETS):
        if dataset not in results:
            missing(ax); continue
        rows = [row for row in results[dataset]["reward_metrics"]["test"]["completion_calibration"]
                if row["n"] > 0 and row["predicted_mean"] is not None and row["observed_mean"] is not None]
        panel(ax, f"{dataset.upper()}  (n={sum(row['n'] for row in rows):,})".replace(",", " "))
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=.8)
        if rows:
            x = [row["predicted_mean"] for row in rows]; y = [row["observed_mean"] for row in rows]
            ax.plot(x, y, color="#456c89", alpha=.7, linewidth=.8, zorder=2)
            ax.scatter(x, y, s=[max(5, 85*row["n"]/largest) for row in rows], color="#456c89",
                       edgecolors="white", linewidths=.45, zorder=3)
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xticks=(0, .5, 1), yticks=(0, .5, 1), xlabel="Прогноз RM")
    axes[0].set_ylabel("Фактическая доля")
    count_labels = sorted(set((max(1, int(largest/4)), largest)))
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(max(5, 85*n/largest)),
                      markerfacecolor="#456c89", markeredgecolor="white", label=f"n={n:,}".replace(",", " ")) for n in count_labels]
    handles.append(Line2D([], [], color="#999999", linestyle="--", linewidth=.8, label="Идеальная калибровка"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .005), ncol=3, frameon=False,
               handletextpad=.5, columnspacing=1.5)
    fig.subplots_adjust(left=.09, right=.985, bottom=.31, top=.84, wspace=.2)
    save(fig, output, "reward_calibration.png")


def frontier(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT_POINTS["frontier.png"]/72))
    labels_pending = []
    for column, dataset in enumerate(DATASETS):
        if dataset not in results:
            for ax in axes[:, column]: missing(ax)
            continue
        result = results[dataset]; grouped = groups(result); selected = result["selection"]["selected_rl"]
        highlighted = {"baseline", "itemknn", "matched_control", "raw_h5", "reranker", selected}
        for row, metric in enumerate(("factual_completion_weighted_hit@10", "predicted_reward@10")):
            ax = axes[row, column]; panel(ax, dataset.upper() if row == 0 else None)
            labels, all_points = [], []
            for family, entries in grouped.items():
                pairs = [(entry["test"].get("ndcg@10"), entry["test"].get(metric)) for entry in entries]
                pairs = np.asarray([(x, y) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)])
                if not len(pairs): continue
                x, y = pairs.mean(axis=0); all_points.extend(pairs.tolist()); named = family in highlighted
                color = COLORS.get(family, SELECTED_COLOR) if named else "#b9b9b9"
                ax.scatter(pairs[:, 0], pairs[:, 1], color=color, s=10, alpha=.5, linewidths=0, zorder=2)
                marker = "*" if family == selected else {"baseline": "D", "itemknn": "X", "matched_control": "s", "raw_h5": "v", "reranker": "P"}.get(family, "o")
                ax.scatter([x], [y], color=color, marker=marker, s=69 if family == selected else 28 if named else 19,
                           edgecolors="white", linewidths=.5, zorder=4 if named else 3)
                if named:
                    label = SHORT_NAMES.get(family, "RL")+(" *" if family == selected else "")
                    labels.append((label, x, y, color))
            ax.margins(x=.19, y=.22)
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("NDCG@10")
            if column == 0: ax.set_ylabel("r × Hit@10" if row == 0 else "Награда RM")
            labels.sort(key=lambda item: (not item[0].endswith(" *"), item[0]))
            labels_pending.append((ax, labels, all_points))
    handles = [Line2D([], [], marker="o", linestyle="none", color="#777777", markersize=2.7, label="Отдельный seed"),
               Line2D([], [], marker="o", linestyle="none", color="#555555", markersize=5, label="Среднее семейства"),
               Line2D([], [], marker="*", linestyle="none", color=SELECTED_COLOR, markersize=8, label="RL, выбранный по valid")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .005), ncol=3, frameon=False,
               handletextpad=.5, columnspacing=1.3)
    fig.subplots_adjust(left=.105, right=.955, bottom=.15, top=.92, hspace=.46, wspace=.31)
    for ax, labels, points in labels_pending: annotate_means(ax, labels, points)
    save(fig, output, "frontier.png")


def duration(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT_POINTS["duration.png"]/72), sharey=True)
    if "yambda" not in results:
        for ax in axes: missing(ax, "Длительность доступна в Yambda")
    else:
        grouped = groups(results["yambda"])
        families = [family for family in ("baseline", "raw_h5", "anchored_h1", "anchored_h5", "conservative_h5",
                                          "duration_neutral_h5", "seconds_h5", "reranker") if family in grouped]
        for ax, metric, title in zip(axes, ("predicted_duration_seconds@10", "predicted_expected_seconds@10"),
                                    ("Длительность трека", "Ожидаемое прослушивание")):
            panel(ax, title)
            for index, family in enumerate(families):
                points = values(grouped[family], metric)
                if not len(points): continue
                color = COLORS[family]
                ax.hlines(index, 0, points.mean(), color=color, linewidth=1.4, alpha=.35)
                ax.scatter(points, np.full(len(points), index), s=15, color=color, alpha=.55, linewidths=0, zorder=3)
                ax.scatter([points.mean()], [index], s=30, marker="D", color=color, edgecolors="white", linewidths=.5, zorder=4)
            ax.set(xlabel="Секунды, top-10", yticks=range(len(families)), yticklabels=[SHORT_NAMES[family] for family in families])
            ax.set_xlim(left=0); ax.xaxis.set_major_locator(MaxNLocator(4)); ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
        axes[0].invert_yaxis()
    fig.text(.56, .015, "Малые точки - seed; ромб - среднее", ha="center", fontsize=8)
    fig.subplots_adjust(left=.16, right=.985, bottom=.22, top=.87, wspace=.21)
    save(fig, output, "duration.png")


def rollout_style(family):
    if family == "baseline": return COLORS[family], "-", "o"
    if family == "raw_h5": return COLORS[family], "--", "v"
    if family == "anchored_h5": return COLORS[family], "-.", "s"
    if family == "reranker": return COLORS[family], ":", "D"
    return SELECTED_COLOR, "-", "*"


def rollout(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT_POINTS["rollout.png"]/72))
    for column, dataset in enumerate(DATASETS):
        if dataset not in results:
            for ax in axes[:, column]: missing(ax)
            continue
        result = results[dataset]; grouped = groups(result); selected = result["selection"]["selected_rl"]
        families = list(dict.fromkeys(family for family in ("baseline", "raw_h5", "anchored_h5", selected, "reranker") if family in grouped))
        for row, metric in enumerate(("predicted_reward_generated_context", "reward_context_gap")):
            ax = axes[row, column]; panel(ax, dataset.upper() if row == 0 else None)
            if row == 1: ax.axhline(0, color="#888888", linewidth=.6, alpha=.6)
            for family in families:
                by_step = defaultdict(list)
                for entry in grouped[family]:
                    for item in entry.get("rollout", {}).get("by_mode_step", []):
                        if item["mode"] == "autoregressive_safe" and item.get(metric) is not None:
                            by_step[int(item["step"])].append(item[metric])
                if not by_step: continue
                steps = sorted(by_step); means = [np.mean(by_step[step]) for step in steps]
                color, linestyle, marker = rollout_style(family)
                ax.plot(steps, means, color=color, linestyle=linestyle, marker=marker, markersize=2.7, zorder=3)
                if family == selected:
                    ax.scatter([steps[-1]], [means[-1]], marker="*", s=48, color=color, edgecolors="white", linewidths=.4, zorder=5)
            ax.set_xticks(range(1, 6)); ax.yaxis.set_major_locator(MaxNLocator(3)); ax.margins(x=.05, y=.17)
            if row == 1: ax.set_xlabel("Шаг генерации")
            if column == 0: ax.set_ylabel("Награда RM" if row == 0 else "Сдвиг оценки")
    handles = []
    for family in ("baseline", "raw_h5", "anchored_h5", "reranker"):
        color, linestyle, marker = rollout_style(family)
        handles.append(Line2D([], [], color=color, linestyle=linestyle, marker=marker, markersize=3, label=SHORT_NAMES[family]))
    handles.append(Line2D([], [], marker="*", linestyle="none", color=SELECTED_COLOR, markersize=8, label="Выбранный RL"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .002), ncol=5, frameon=False,
               handlelength=1.8, handletextpad=.4, columnspacing=1.1)
    fig.subplots_adjust(left=.105, right=.985, bottom=.20, top=.90, hspace=.30, wspace=.27)
    save(fig, output, "rollout.png")


def rerank_validation(results, output):
    """Supplement: the saved development frontier, never a test alpha sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3))
    pending, csv_rows = [], []
    recipes = (("mean", "Среднее RM", "#456c89", "-", "o"),
               ("mean-std", "Среднее RM − σ", "#4b7b6b", "--", "s"))
    for ax, dataset in zip(axes, DATASETS):
        if dataset not in results:
            missing(ax); continue
        result = results[dataset]; selected = result["selection"]["reranker"]["key"]
        panel(ax, dataset.upper()); candidates, all_points = [], []
        ax.axvline(result["selection"]["ndcg_guardrail"], color="#999999", linewidth=.8,
                   linestyle=":", zorder=1)
        for reward_type, _, color, linestyle, marker in recipes:
            points = []
            for key, entry in result["rerank_grid"].items():
                if entry["reward_type"] != reward_type:
                    continue
                metrics = entry["mean_metrics"]
                x, y = metrics["ndcg@10"], metrics["factual_completion_weighted_hit@10"]
                if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                    raise ValueError(f"Nonfinite saved reranker validation point: {dataset}/{key}")
                alpha = float(entry["alpha"]); chosen = key == selected
                points.append((alpha, x, y, chosen))
                csv_rows.append({"dataset": dataset, "split": "validation", "recipe": key,
                    "reward_type": reward_type, "alpha": alpha, "n_seeds": len(entry["by_seed"]),
                    "ndcg@10": x, "factual_completion_weighted_hit@10": y, "selected": chosen})
            points.sort(key=lambda row: row[0])
            if not points:
                continue
            ax.plot([row[1] for row in points], [row[2] for row in points], color=color,
                    linestyle=linestyle, marker=marker, markersize=3, zorder=3)
            for alpha, x, y, chosen in points:
                all_points.append((x, y))
                if chosen:
                    ax.scatter([x], [y], marker="*", s=78, color=SELECTED_COLOR,
                               edgecolors="white", linewidths=.6, zorder=5)
                if alpha in (0., 1., 4., 8.) or chosen:
                    label = "α="+f"{alpha:g}".replace(".", ",")+(" *" if chosen else "")
                    candidates.append((alpha, chosen, label, x, y, color))
        ax.set_xlabel("Валидация: NDCG@10")
        ax.margins(x=.22, y=.25)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        pending.append((ax, candidates, all_points))
    axes[0].set_ylabel("Валидация: r × Hit@10")
    handles = [Line2D([], [], color=color, linestyle=linestyle, marker=marker, markersize=3, label=label)
               for _, label, color, linestyle, marker in recipes]
    handles.append(Line2D([], [], marker="*", linestyle="none", color=SELECTED_COLOR,
                          markersize=8, label="Выбрано по валидации"))
    handles.append(Line2D([], [], color="#999999", linewidth=.8, linestyle=":", label="95% NDCG GPT-2"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .005), ncol=2,
               frameon=False, handletextpad=.5, columnspacing=1.4)
    fig.subplots_adjust(left=.105, right=.955, bottom=.28, top=.88, wspace=.31)
    fig.canvas.draw()
    for ax, candidates, points in pending:
        labels, accepted = [], []
        # Curves coincide at alpha=0, and can nearly coincide elsewhere. Label
        # such an alpha once, preserving the selected point as the first choice.
        for alpha, chosen, label, x, y, color in sorted(candidates, key=lambda row: (not row[1], row[0])):
            location = ax.transData.transform((x, y))
            if any(old_alpha == alpha and np.linalg.norm(location-old_location) < 12*fig.dpi/72
                   for old_alpha, old_location in accepted):
                continue
            accepted.append((alpha, location)); labels.append((label, x, y, color))
        annotate_means(ax, labels, points)
    path = Path(output)/"rerank_validation.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "recipe", "reward_type", "alpha",
                                "n_seeds", "ndcg@10", "factual_completion_weighted_hit@10", "selected"])
        writer.writeheader(); writer.writerows(csv_rows)
    save(fig, output, "rerank_validation.png")


def build_figures(evidence, output=None):
    evidence = Path(evidence); output = Path(output) if output is not None else evidence/"figures"
    results = {}
    for dataset in DATASETS:
        path = evidence/f"{dataset}_results.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("version") != 2 or result.get("dataset") != dataset:
                raise ValueError(f"Expected research-v2 {dataset} export: {path}")
            results[dataset] = result
    if not results: raise ValueError(f"No {{kion,yambda}}_results.json exports in {evidence}")
    output.mkdir(parents=True, exist_ok=True); style()
    for make in (calibration, frontier, duration, rollout, rerank_validation): make(results, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Defaults to EVIDENCE/figures")
    args = parser.parse_args(); output = build_figures(args.evidence, args.output)
    print(json.dumps({"figures": str(output.resolve()), "files": list(HEIGHT_POINTS), "dpi": 300, "minimum_font_pt": 8}, ensure_ascii=False))


if __name__ == "__main__":
    main()
