"""V4/V5 artifacts for longitudinal subspace selection.

V4 builds a risk-utility frontier table/figure using a common comparator:
E4a's leave-one-subject-out outer-best single branch.

V5 interprets the selected longitudinal feature subspaces by mapping selected
tangent dimensions back to bands and channel pairs.  This is an explanatory
analysis, not a deployment component.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stieger_neuro_feature_baseline import (
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    SENSORIMOTOR21,
    feature_covariances,
    load_subject,
    lower_tail_cvar,
    weighted_quantile,
)
from stieger_longitudinal_metric_pilot import (
    aggregate_stats,
    metric_score,
    source_only_score,
    top_fraction_indices,
    unflatten_stats,
)
from stieger_longitudinal_selection_ablation import (
    drift_only_score,
    sep_no_drift_score,
    target_only_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_E4A_DIR = RESULTS_DIR / "260626_stieger_neuro_fixed_fusion"
DEFAULT_E5B_DIR = RESULTS_DIR / "260627_stieger_longitudinal_metric_pilot"
DEFAULT_ABLATION_DIR = RESULTS_DIR / "260627_stieger_longitudinal_selection_ablation"
DEFAULT_OUTPUT = RESULTS_DIR / "260627_stieger_longitudinal_frontier_interpretation"
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")
SCORE_FAMILIES = {
    "longitudinal": metric_score,
    "source_only": source_only_score,
    "sep_no_drift": sep_no_drift_score,
    "target_only": target_only_score,
    "drift_only": drift_only_score,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--e4a-dir", type=Path, default=DEFAULT_E4A_DIR)
    parser.add_argument("--e5b-dir", type=Path, default=DEFAULT_E5B_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.25, 0.1])
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_subjects(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", maxsplit=1))
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        return list(json.load(handle)["records"])


def subject_balanced_weights(subjects: Sequence[int]) -> np.ndarray:
    counts = Counter(int(subject) for subject in subjects)
    return np.asarray(
        [1.0 / (len(counts) * counts[int(subject)]) for subject in subjects],
        dtype=np.float64,
    )


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(bootstrap, len(values)))
    estimates = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def summarize_vs_outer(
    selected: Sequence[Mapping[str, object]],
    outer: Mapping[tuple[int, int, str], Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    by_subject: dict[int, list[tuple[Mapping[str, object], Mapping[str, object]]]] = defaultdict(list)
    session_diffs = []
    session_subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        if key not in outer:
            continue
        comparator = outer[key]
        by_subject[int(row["subject"])].append((row, comparator))
        session_diffs.append(float(row["test_acc"]) - float(comparator["test_acc"]))
        session_subjects.append(int(row["subject"]))
    subject_gains = []
    subject_accs = []
    for pairs in by_subject.values():
        selected_correct = sum(int(row["test_correct"]) for row, _ in pairs)
        outer_correct = sum(int(comp["test_correct"]) for _, comp in pairs)
        total = sum(int(row["n_eval"]) for row, _ in pairs)
        subject_accs.append(100.0 * selected_correct / total)
        subject_gains.append(100.0 * (selected_correct - outer_correct) / total)
    values = np.asarray(session_diffs, dtype=np.float64)
    subjects = np.asarray(session_subjects, dtype=np.int64)
    weights = subject_balanced_weights(subjects)
    weights /= weights.sum()
    gains = np.asarray(subject_gains, dtype=np.float64)
    return {
        "n_subjects": len(by_subject),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": float(np.mean(subject_accs)),
        "gain_vs_outer_best_single_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_outer_best_single_subject_bootstrap_95ci": bootstrap_ci(
            gains, bootstrap, seed
        ),
        "gain_vs_outer_best_single_session_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_outer_best_single_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_outer_best_single_q05_pp": weighted_quantile(values, weights, 0.05),
        "p_gain_vs_outer_best_single_lt_minus5": float(np.sum(weights[values < -5.0])),
    }


def scoped(rows: Sequence[Mapping[str, object]], method: str, condition: str) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and (
            row["condition"] == condition
            if condition != "primary_pooled"
            else row["condition"] in PRIMARY_CONDITIONS
        )
    ]


def method_rows(records: Sequence[Mapping[str, object]], method: str) -> list[Mapping[str, object]]:
    return [row for row in records if row["method"] == method]


def build_frontier(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    e4a = load_records(args.e4a_dir / "selection_records.json")
    e5b = load_records(args.e5b_dir / "selection_records.json")
    ablation = load_records(args.ablation_dir / "selection_records.json")
    outer = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in e4a
        if row["method"] == "outer_best_single"
    }
    sources = [
        ("E4a equal posterior", e4a, "posterior__equal__prefix_ea", "E4a"),
        ("E5b longitudinal q0.25", e5b, "select_q0p25", "E5b"),
        ("E5b longitudinal q0.10", e5b, "select_q0p10", "E5b"),
        ("E5b outer best fraction", e5b, "outer_best_fraction", "E5b"),
        ("source-only outer best", e5b, "outer_best_source_fraction", "ablation"),
        ("V2 longitudinal outer best", ablation, "outer_best_longitudinal", "V2"),
        ("V2 source-only outer best", ablation, "outer_best_source_only", "V2"),
        ("V2 sep-no-drift outer best", ablation, "outer_best_sep_no_drift", "V2"),
        ("V2 target-only outer best", ablation, "outer_best_target_only", "V2"),
    ]
    table: list[dict[str, object]] = []
    report: dict[str, object] = {}
    for condition in ("pure_lr", "pure_ud", "primary_pooled"):
        report[condition] = {}
        for label, records, method, family in sources:
            selected = scoped(records, method, condition)
            if not selected:
                continue
            summary = summarize_vs_outer(
                selected, outer, args.bootstrap, args.seed + len(label) + len(condition)
            )
            row = {
                "condition": condition,
                "family": family,
                "label": label,
                "method": method,
                **summary,
            }
            table.append(row)
            report[condition][label] = summary
    return table, report


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(table: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    primary = [row for row in table if row["condition"] == "primary_pooled"]
    if not primary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {
        "E4a": "#777777",
        "E5b": "#1f77b4",
        "ablation": "#ff7f0e",
        "V2": "#2ca02c",
    }
    for ax, y_key, ylabel in (
        (axes[0], "loss_r10_vs_outer_best_single_pp", "R10 loss vs outer-best-single (pp)"),
        (axes[1], "p_gain_vs_outer_best_single_lt_minus5", "P(loss < -5pp)"),
    ):
        for row in primary:
            x = float(row["gain_vs_outer_best_single_subject_pooled_pp"])
            y = float(row[y_key])
            if y_key.startswith("p_"):
                y *= 100.0
            label = str(row["label"])
            ax.scatter(
                x,
                y,
                s=80,
                color=colors.get(str(row["family"]), "#333333"),
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(label.replace(" outer best", ""), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Mean gain vs outer-best-single (pp)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Risk-utility frontier: primary LR+UD")
    fig.savefig(output_dir / "risk_utility_frontier_primary.png", dpi=180)
    plt.close(fig)


def load_all_stats(metric_dir: Path, subjects: Sequence[int]) -> dict[int, dict]:
    output = {}
    for subject in subjects:
        payload = np.load(metric_dir / "metric_stats" / f"S{subject}.npz")
        output[int(subject)] = unflatten_stats({key: payload[key] for key in payload.files})
    return output


def aggregate_all_subjects(
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    condition: str,
    feature: str,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for stats_by_condition in all_stats.values():
        stats = stats_by_condition.get(condition, {}).get(feature)
        if not stats:
            continue
        for key, value in stats.items():
            if key not in total:
                total[key] = np.asarray(value, dtype=np.float64).copy()
            else:
                total[key] += np.asarray(value, dtype=np.float64)
    count = float(total["count"][0])
    return {key: value / count for key, value in total.items() if key != "count"} | {
        "count": np.asarray([count], dtype=np.float64)
    }


def channel_indices(channels: Sequence[str], channel_set: str) -> list[int]:
    names = list(map(str, channels))
    if channel_set == "all":
        return list(range(len(names)))
    if channel_set != "sensorimotor21":
        raise ValueError(channel_set)
    return [names.index(channel) for channel in SENSORIMOTOR21]


def feature_dimension_map(data: Mapping[str, np.ndarray], feature: str) -> list[dict[str, object]]:
    config = FEATURE_CONFIGS[feature]
    channels = [str(channel) for channel in data["channels"].tolist()]
    selected_idx = channel_indices(channels, str(config["channels"]))
    selected_channels = [channels[index] for index in selected_idx]
    d = len(selected_channels)
    upper = list(zip(*np.triu_indices(d)))
    mapping = []
    offset = 0
    for band in config["bands"]:
        for local_index, (i, j) in enumerate(upper):
            mapping.append(
                {
                    "feature_index": offset + local_index,
                    "band": str(band),
                    "channel_i": selected_channels[i],
                    "channel_j": selected_channels[j],
                    "is_diagonal": bool(i == j),
                }
            )
        offset += len(upper)
    return mapping


def interpret_selection(args: argparse.Namespace) -> dict[str, object]:
    subjects = parse_subjects(args.subjects)
    all_stats = load_all_stats(args.e5b_dir, subjects)
    data = load_subject(args.cache_dir, subjects[0])
    rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    report: dict[str, object] = {}
    for condition in ("pure_lr", "pure_ud"):
        report[condition] = {}
        for feature in args.features:
            stats = aggregate_all_subjects(all_stats, condition, feature)
            mapping = feature_dimension_map(data, feature)
            scores = {family: fn(stats) for family, fn in SCORE_FAMILIES.items()}
            report[condition][feature] = {}
            for family, score in scores.items():
                for fraction in args.fractions:
                    selected = set(map(int, top_fraction_indices(score, float(fraction))))
                    band_counts = Counter()
                    channel_counts = Counter()
                    diag = 0
                    offdiag = 0
                    for item in mapping:
                        index = int(item["feature_index"])
                        if index not in selected:
                            continue
                        band_counts[str(item["band"])] += 1
                        channel_counts[str(item["channel_i"])] += 1
                        channel_counts[str(item["channel_j"])] += 1
                        if item["is_diagonal"]:
                            diag += 1
                        else:
                            offdiag += 1
                        rows.append(
                            {
                                "condition": condition,
                                "feature": feature,
                                "family": family,
                                "fraction": float(fraction),
                                "score": float(score[index]),
                                **item,
                            }
                        )
                    total = max(len(selected), 1)
                    report[condition][feature][f"{family}_q{fraction:.2f}"] = {
                        "n_selected": len(selected),
                        "diagonal_fraction": float(diag / total),
                        "offdiagonal_fraction": float(offdiag / total),
                        "band_counts": dict(band_counts),
                        "top_channels": channel_counts.most_common(12),
                    }
                    for channel, count in channel_counts.items():
                        channel_rows.append(
                            {
                                "condition": condition,
                                "feature": feature,
                                "family": family,
                                "fraction": float(fraction),
                                "channel": channel,
                                "participation_count": int(count),
                            }
                        )
    write_csv(args.output_dir / "selected_feature_rows.csv", rows)
    write_csv(args.output_dir / "selected_channel_participation.csv", channel_rows)
    plot_interpretation(channel_rows, report, args.output_dir)
    return report


def plot_interpretation(
    channel_rows: Sequence[Mapping[str, object]],
    report: Mapping[str, object],
    output_dir: Path,
) -> None:
    selected = [
        row
        for row in channel_rows
        if row["family"] == "longitudinal" and float(row["fraction"]) in (0.1, 0.25)
    ]
    if not selected:
        return
    for condition in ("pure_lr", "pure_ud"):
        for feature in DEFAULT_FEATURES:
            subset = [
                row
                for row in selected
                if row["condition"] == condition
                and row["feature"] == feature
                and float(row["fraction"]) == (0.1 if condition == "pure_lr" else 0.25)
            ]
            if not subset:
                continue
            top = sorted(subset, key=lambda row: int(row["participation_count"]), reverse=True)[:15]
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.bar(
                [str(row["channel"]) for row in top],
                [int(row["participation_count"]) for row in top],
                color="#1f77b4",
            )
            ax.set_title(f"Top selected channels: {condition}, {feature}")
            ax.set_ylabel("Selected pair participation")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.25)
            fig.savefig(output_dir / f"top_channels_{condition}_{feature}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontier_table, frontier_report = build_frontier(args)
    write_csv(args.output_dir / "risk_utility_frontier.csv", frontier_table)
    plot_frontier(frontier_table, args.output_dir)
    interpretation = interpret_selection(args)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "fractions": sorted(set(map(float, args.fractions)), reverse=True),
            "e4a_dir": str(args.e4a_dir),
            "e5b_dir": str(args.e5b_dir),
            "ablation_dir": str(args.ablation_dir),
        },
        "frontier": frontier_report,
        "interpretation": interpretation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    if args.quiet:
        print(
            json.dumps(
                {
                    "frontier_rows": len(frontier_table),
                    "summary_json": str(args.output_dir / "summary.json"),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
