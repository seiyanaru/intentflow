"""E8: selected-feature geometry for source-side tangent subspace selection.

This analysis is explanatory, not a deployment component.

It asks why the current evidence differs by dataset:

* Stieger LR: longitudinal q0.10 beats source-only q0.25.
* Lee2019 LR: source-only q0.25 beats longitudinal q0.10.

The script aligns the feature-geometry readout across datasets:

* selected feature overlap;
* diagonal vs off-diagonal covariance dimensions;
* top participating channels;
* sensorimotor concentration.

Stieger uses the already generated V4/V5 selected-feature rows.
Lee2019 reconstructs the selected tangent dimensions from the cached subject
features and maps them back to channel pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    add_stats,
    longitudinal_score,
    parse_subjects,
    session_metric_stats,
    source_only_score,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_STIEGER_DIR = RESULTS_DIR / "260627_stieger_longitudinal_frontier_interpretation"
DEFAULT_LEE_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_source_side_feature_geometry_e8"

# First 62 EEG channels from Lee2019 raw after excluding EMG/STI channels.
LEE2019_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "FC3",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CPz",
    "CP4",
    "P1",
    "P2",
    "POz",
    "FT9",
    "FTT9h",
    "TTP7h",
    "TP7",
    "TPP9h",
    "FT10",
    "FTT10h",
    "TPP8h",
    "TP8",
    "TPP10h",
    "F9",
    "F10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "PO3",
    "PO4",
]

SENSORIMOTOR = {
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
}

METHODS = (
    ("longitudinal", 0.10),
    ("longitudinal", 0.25),
    ("source_only", 0.10),
    ("source_only", 0.25),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--lee-cache-dir", type=Path, default=DEFAULT_LEE_CACHE)
    parser.add_argument("--stieger-dir", type=Path, default=DEFAULT_STIEGER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_label(family: str, fraction: float) -> str:
    return f"{family}_q{fraction:.2f}"


def tangent_rows(
    channels: Sequence[str],
    selected_indices: np.ndarray,
    score: np.ndarray,
    dataset: str,
    condition: str,
    feature: str,
    family: str,
    fraction: float,
) -> list[dict[str, object]]:
    upper_i, upper_j = np.triu_indices(len(channels))
    rows: list[dict[str, object]] = []
    for index in selected_indices:
        i = int(upper_i[int(index)])
        j = int(upper_j[int(index)])
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "feature": feature,
                "family": family,
                "fraction": float(fraction),
                "method": method_label(family, fraction),
                "feature_index": int(index),
                "score": float(score[int(index)]),
                "band": "broad_8_30",
                "channel_i": str(channels[i]),
                "channel_j": str(channels[j]),
                "is_diagonal": bool(i == j),
            }
        )
    return rows


def summarize_selected(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {}
    channel_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    any_sensor = 0
    both_sensor = 0
    for row in rows.itertuples(index=False):
        ci = str(row.channel_i)
        cj = str(row.channel_j)
        channel_counts[ci] += 1
        channel_counts[cj] += 1
        pair_counts[f"{ci}-{cj}"] += 1
        if ci in SENSORIMOTOR or cj in SENSORIMOTOR:
            any_sensor += 1
        if ci in SENSORIMOTOR and cj in SENSORIMOTOR:
            both_sensor += 1
    total_participation = sum(channel_counts.values())
    sensor_participation = sum(
        count for channel, count in channel_counts.items() if channel in SENSORIMOTOR
    )
    return {
        "n_selected": int(len(rows)),
        "diagonal_fraction": float(rows["is_diagonal"].mean()),
        "offdiagonal_fraction": float(1.0 - rows["is_diagonal"].mean()),
        "any_sensorimotor_pair_fraction": float(any_sensor / len(rows)),
        "both_sensorimotor_pair_fraction": float(both_sensor / len(rows)),
        "sensorimotor_participation_fraction": float(sensor_participation / total_participation),
        "top_channels": channel_counts.most_common(12),
        "top_pairs": pair_counts.most_common(12),
    }


def aggregate_lee_stats(subjects: Sequence[int], cache_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    total: dict[str, np.ndarray] = {}
    n_subjects = 0
    for subject in subjects:
        payload = np.load(cache_dir / f"S{subject}.npz")
        subject_payload = {key: payload[key] for key in payload.files}
        stats = session_metric_stats(
            subject_payload["source_features"],
            subject_payload["source_labels"],
            subject_payload["target_features"],
            subject_payload["target_labels"],
        )
        add_stats(total, stats)
        n_subjects += 1
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged, n_subjects


def lee_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    subjects = parse_subjects(args.subjects)
    stats, _ = aggregate_lee_stats(subjects, args.lee_cache_dir)
    scores = {
        "longitudinal": longitudinal_score(stats),
        "source_only": source_only_score(stats),
    }
    rows: list[dict[str, object]] = []
    for family, fraction in METHODS:
        score = scores[family]
        selected = top_fraction_indices(score, float(fraction))
        rows.extend(
            tangent_rows(
                LEE2019_CHANNELS,
                selected,
                score,
                dataset="Lee2019",
                condition="pure_lr",
                feature="broad_all62",
                family=family,
                fraction=float(fraction),
            )
        )
    return rows


def stieger_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    path = args.stieger_dir / "selected_feature_rows.csv"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["condition"].isin(["pure_lr", "pure_ud"]))
        & (frame["family"].isin(["longitudinal", "source_only"]))
        & (frame["fraction"].isin([0.10, 0.25]))
    ].copy()
    frame.insert(0, "dataset", "Stieger")
    frame["method"] = frame.apply(
        lambda row: method_label(str(row["family"]), float(row["fraction"])),
        axis=1,
    )
    return frame.to_dict("records")


def build_summaries(feature_rows: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    for keys, group in feature_rows.groupby(
        ["dataset", "condition", "feature", "family", "fraction", "method"],
        observed=True,
    ):
        dataset, condition, feature, family, fraction, method = keys
        summary_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "feature": feature,
                "family": family,
                "fraction": float(fraction),
                "method": method,
                **summarize_selected(group),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    pairs = [
        ("longitudinal", 0.10, "source_only", 0.25),
        ("longitudinal", 0.10, "source_only", 0.10),
        ("longitudinal", 0.25, "source_only", 0.25),
    ]
    for (dataset, condition, feature), group in feature_rows.groupby(
        ["dataset", "condition", "feature"], observed=True
    ):
        for family_a, fraction_a, family_b, fraction_b in pairs:
            a = group[(group["family"] == family_a) & (group["fraction"] == fraction_a)]
            b = group[(group["family"] == family_b) & (group["fraction"] == fraction_b)]
            if a.empty or b.empty:
                continue
            set_a = set(map(int, a["feature_index"].tolist()))
            set_b = set(map(int, b["feature_index"].tolist()))
            intersection = set_a & set_b
            union = set_a | set_b
            overlap_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "feature": feature,
                    "method_a": method_label(family_a, fraction_a),
                    "method_b": method_label(family_b, fraction_b),
                    "n_a": int(len(set_a)),
                    "n_b": int(len(set_b)),
                    "n_intersection": int(len(intersection)),
                    "jaccard": float(len(intersection) / len(union)) if union else float("nan"),
                    "a_covered_by_b": float(len(intersection) / len(set_a)) if set_a else float("nan"),
                    "b_covered_by_a": float(len(intersection) / len(set_b)) if set_b else float("nan"),
                }
            )
    return summary_rows, overlap_rows


def write_top_channel_rows(summary_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    rows: list[dict[str, object]] = []
    for item in summary_rows:
        for rank, (channel, count) in enumerate(item.get("top_channels", []), start=1):
            rows.append(
                {
                    "dataset": item["dataset"],
                    "condition": item["condition"],
                    "feature": item["feature"],
                    "method": item["method"],
                    "rank": int(rank),
                    "channel": channel,
                    "count": int(count),
                }
            )
    write_csv(output, rows)


def plot_lee_top_channels(summary_rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    selected = [
        row
        for row in summary_rows
        if row["dataset"] == "Lee2019"
        and row["condition"] == "pure_lr"
        and row["feature"] == "broad_all62"
        and row["method"] in {"source_only_q0.25", "longitudinal_q0.10"}
    ]
    for row in selected:
        top = list(row["top_channels"][:12])
        if not top:
            continue
        channels = [str(ch) for ch, _ in top][::-1]
        counts = [int(count) for _, count in top][::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#d55e00" if ch in SENSORIMOTOR else "#0072b2" for ch in channels]
        ax.barh(channels, counts, color=colors)
        ax.set_xlabel("Selected feature participation count")
        ax.set_title(f"Lee2019 {row['method']}: top channels")
        fig.tight_layout()
        fig.savefig(output_dir / f"lee2019_top_channels_{row['method']}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = stieger_rows(args) + lee_rows(args)
    feature_rows = pd.DataFrame(rows)
    feature_rows.to_csv(args.output_dir / "selected_feature_geometry_rows.csv", index=False)
    summary_rows, overlap_rows = build_summaries(feature_rows)
    write_csv(args.output_dir / "feature_geometry_summary.csv", summary_rows)
    write_csv(args.output_dir / "feature_overlap.csv", overlap_rows)
    write_top_channel_rows(summary_rows, args.output_dir / "top_channel_rows.csv")
    plot_lee_top_channels(summary_rows, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "overlap": overlap_rows,
                "notes": {
                    "purpose": "explanatory E8 selected-feature geometry",
                    "lee_channel_source": "first 62 EEG channels inspected from Lee2019 raw; EMG/STI excluded",
                    "stieger_source": str(args.stieger_dir / "selected_feature_rows.csv"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_feature_rows": int(len(feature_rows)),
                "n_summary_rows": int(len(summary_rows)),
                "n_overlap_rows": int(len(overlap_rows)),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
