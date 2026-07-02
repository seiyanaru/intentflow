"""Cluster-bootstrap task-context contrasts from the Stieger audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cross_adapter_core import risk_utility_summary, subject_balanced_weights


CONTRASTS = {
    "mixed_horizontal_minus_pure_lr": ("mixed_horizontal", "pure_lr"),
    "mixed_vertical_minus_pure_ud": ("mixed_vertical", "pure_ud"),
    "two_d_horizontal_minus_pure_lr": ("two_d_horizontal", "pure_lr"),
    "two_d_vertical_minus_pure_ud": ("two_d_vertical", "pure_ud"),
    "pure_ud_minus_pure_lr": ("pure_ud", "pure_lr"),
    "two_d_minus_pure_lr": ("two_d", "pure_lr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adapter", default="prefix_ea")
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metrics(rows: list[dict], adapter: str) -> dict[str, float]:
    selected = [row for row in rows if row["adapter"] == adapter]
    summary = risk_utility_summary(selected, adapter)
    subjects = [int(row["subject"]) for row in selected]
    weights = subject_balanced_weights(subjects)
    weights /= weights.sum()
    return {
        "utility": float(summary["utility_mean_delta_pp"]),
        "risk_r10": float(summary["risk_r10"]),
        "p_delta_lt_minus5": float(summary["p_delta_lt_minus5"]),
        "source_accuracy": float(
            np.sum(weights * np.asarray([row["source_acc"] for row in selected]))
        ),
        "mean_n_eval": float(
            np.sum(weights * np.asarray([row["n_eval"] for row in selected]))
        ),
    }


def relabel_bootstrap_rows(
    rows: list[dict],
    sampled_subjects: np.ndarray,
) -> list[dict]:
    by_subject: dict[int, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(int(row["subject"]), []).append(row)
    output: list[dict] = []
    for new_subject, original_subject in enumerate(sampled_subjects):
        for row in by_subject[int(original_subject)]:
            copied = dict(row)
            copied["subject"] = new_subject
            output.append(copied)
    return output


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary.read_text())
    rows = report["rows"]
    subjects = np.asarray(sorted({int(row["subject"]) for row in rows}))
    rng = np.random.default_rng(args.seed)
    output = {
        "summary": str(args.summary),
        "adapter": args.adapter,
        "n_subjects": len(subjects),
        "contrasts": {},
        "prefix_minus_full_by_condition": {},
    }
    for name, (condition_a, condition_b) in CONTRASTS.items():
        rows_a = [row for row in rows if row["condition"] == condition_a]
        rows_b = [row for row in rows if row["condition"] == condition_b]
        point_a = metrics(rows_a, args.adapter)
        point_b = metrics(rows_b, args.adapter)
        point = {
            key: point_a[key] - point_b[key]
            for key in point_a
        }
        bootstrap_values = {key: [] for key in point}
        for _ in range(args.bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            boot_a = metrics(
                relabel_bootstrap_rows(rows_a, sampled), args.adapter
            )
            boot_b = metrics(
                relabel_bootstrap_rows(rows_b, sampled), args.adapter
            )
            for key in point:
                bootstrap_values[key].append(boot_a[key] - boot_b[key])
        output["contrasts"][name] = {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "difference_a_minus_b": point,
            "subject_bootstrap_95ci": {
                key: [
                    float(value)
                    for value in np.quantile(values, [0.025, 0.975])
                ]
                for key, values in bootstrap_values.items()
            },
        }
    for condition in ("pure_lr", "pure_ud", "two_d"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        prefix_point = metrics(condition_rows, "prefix_ea")
        full_point = metrics(condition_rows, "full_ea")
        point = {
            key: prefix_point[key] - full_point[key]
            for key in prefix_point
        }
        bootstrap_values = {key: [] for key in point}
        for _ in range(args.bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            boot_rows = relabel_bootstrap_rows(condition_rows, sampled)
            boot_prefix = metrics(boot_rows, "prefix_ea")
            boot_full = metrics(boot_rows, "full_ea")
            for key in point:
                bootstrap_values[key].append(
                    boot_prefix[key] - boot_full[key]
                )
        output["prefix_minus_full_by_condition"][condition] = {
            "difference_prefix_minus_full": point,
            "subject_bootstrap_95ci": {
                key: [
                    float(value)
                    for value in np.quantile(values, [0.025, 0.975])
                ]
                for key, values in bootstrap_values.items()
            },
        }
    output_path = args.output or args.summary.with_name("contrasts.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
