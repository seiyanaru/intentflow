"""E16: fixed-policy robustness vs small-pool source-side validation.

E15 showed that source-pool validation can be unstable, especially on Lee2019.
This script answers the practical follow-up:

    If the source pool is small, should we validate a candidate from that small
    pool, or simply use a fixed robust policy learned from prior experiments?

It compares E15 validation policies against fixed policies computed from E9
records under the same full-feature baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_STIEGER = RESULTS_DIR / "260628_stieger_fraction_sweep_e9_full"
DEFAULT_LEE = RESULTS_DIR / "260628_lee2019_fraction_sweep_e9_full"
DEFAULT_E15 = RESULTS_DIR / "260628_e15_source_pool_size_ablation_full" / "summary.json"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e16_fixed_policy_vs_validation"

FIXED_METHODS = (
    "full_q1p00",
    "source_only_q0p25",
    "source_only_q0p50",
    "longitudinal_q0p05",
    "longitudinal_q0p10",
    "longitudinal_q0p20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stieger-dir", type=Path, default=DEFAULT_STIEGER)
    parser.add_argument("--lee-dir", type=Path, default=DEFAULT_LEE)
    parser.add_argument("--e15-summary", type=Path, default=DEFAULT_E15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
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


def load_stieger(stieger_dir: Path) -> pd.DataFrame:
    payload = json.loads((stieger_dir / "selection_records.json").read_text())
    frame = pd.DataFrame(payload["records"])
    frame = frame[frame["condition"].isin(["pure_lr", "pure_ud"])].copy()
    frame = frame.rename(columns={"test_correct": "correct", "test_acc": "accuracy"})
    frame["dataset"] = "Stieger"
    frame["unit_id"] = (
        frame["subject"].astype(str)
        + "__S"
        + frame["session"].astype(str)
        + "__"
        + frame["condition"].astype(str)
    )
    frame = frame[
        ["dataset", "subject", "session", "condition", "unit_id", "method", "correct", "n_eval", "accuracy"]
    ]
    if "full_q1p00" not in set(frame["method"]):
        full = frame[frame["method"] == "longitudinal_q1p00"].copy()
        full["method"] = "full_q1p00"
        frame = pd.concat([frame, full], ignore_index=True)
    return frame


def load_lee(lee_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(lee_dir / "selection_records.csv")
    frame["dataset"] = "Lee2019"
    frame["session"] = 1
    frame["condition"] = "lee_lr"
    frame["unit_id"] = frame["subject"].astype(str)
    frame["correct"] = np.rint(frame["accuracy"] * frame["n_eval"] / 100.0).astype(int)
    return frame[
        ["dataset", "subject", "session", "condition", "unit_id", "method", "correct", "n_eval", "accuracy"]
    ]


def bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def weighted_session_risk(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "gain_vs_full_q05_unit_pp": float("nan"),
            "loss_r10_vs_full_unit_pp": float("nan"),
            "p_gain_vs_full_lt_minus5_unit": float("nan"),
        }
    frame = pd.DataFrame(rows)
    counts = frame.groupby("subject")["unit_id"].transform("count")
    n_subjects = frame["subject"].nunique()
    weights = (1.0 / (n_subjects * counts)).to_numpy(dtype=float)
    gain = frame["gain_vs_full"].to_numpy(dtype=float)
    order = np.argsort(gain)
    gain_sorted = gain[order]
    weight_sorted = weights[order]
    cumulative = np.cumsum(weight_sorted)
    q05 = float(gain_sorted[np.searchsorted(cumulative, 0.05, side="left")])
    boundary = int(np.searchsorted(cumulative, 0.10, side="left"))
    tail_mask = np.zeros_like(gain_sorted, dtype=bool)
    tail_mask[: min(boundary + 1, len(tail_mask))] = True
    r10 = float(-np.sum(weight_sorted[tail_mask] * gain_sorted[tail_mask]) / np.sum(weight_sorted[tail_mask]))
    return {
        "gain_vs_full_q05_unit_pp": q05,
        "loss_r10_vs_full_unit_pp": r10,
        "p_gain_vs_full_lt_minus5_unit": float(np.sum(weights[gain < -5.0])),
    }


def summarize_selected(
    records: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_subject[int(row["subject"])].append(row)
    subject_acc: list[float] = []
    subject_gain: list[float] = []
    for rows in by_subject.values():
        correct = sum(int(row["correct"]) for row in rows)
        full_correct = sum(int(row["full_correct"]) for row in rows)
        total = sum(int(row["n_eval"]) for row in rows)
        subject_acc.append(100.0 * correct / total)
        subject_gain.append(100.0 * (correct - full_correct) / total)
    gains = np.asarray(subject_gain, dtype=np.float64)
    acc = np.asarray(subject_acc, dtype=np.float64)
    choices = Counter(str(row["chosen_candidate"]) for row in records)
    return {
        "n_subjects": int(len(by_subject)),
        "n_units": int(len(records)),
        "accuracy_mean": float(acc.mean()),
        "gain_vs_full_mean_pp": float(gains.mean()),
        "gain_subject_bootstrap_95ci": bootstrap_ci(gains, bootstrap, seed),
        **weighted_session_risk(records),
        "chosen_candidate_counts": dict(choices),
    }


def fixed_records(
    frame: pd.DataFrame,
    dataset: str,
    policy_name: str,
    selector: Mapping[str, str] | str,
) -> list[dict[str, object]]:
    data = frame[frame["dataset"] == dataset].copy()
    full = data[data["method"] == "full_q1p00"][
        ["unit_id", "correct", "accuracy"]
    ].rename(columns={"correct": "full_correct", "accuracy": "full_accuracy"})
    data = data.merge(full, on="unit_id", how="left")
    rows: list[dict[str, object]] = []
    for _, full_row in data[data["method"] == "full_q1p00"].iterrows():
        condition = str(full_row["condition"])
        method = selector[condition] if isinstance(selector, dict) else str(selector)
        selected = data[(data["unit_id"] == full_row["unit_id"]) & (data["method"] == method)]
        if selected.empty:
            continue
        item = selected.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "policy": policy_name,
                "subject": int(full_row["subject"]),
                "session": int(full_row["session"]),
                "condition": condition,
                "unit_id": str(full_row["unit_id"]),
                "chosen_candidate": method,
                "correct": int(item["correct"]),
                "full_correct": int(full_row["correct"]),
                "n_eval": int(full_row["n_eval"]),
                "accuracy": float(item["accuracy"]),
                "full_accuracy": float(full_row["accuracy"]),
                "gain_vs_full": float(item["accuracy"] - full_row["accuracy"]),
            }
        )
    return rows


def build_fixed_summary(frame: pd.DataFrame, bootstrap: int, seed: int) -> dict[str, dict[str, object]]:
    policies: list[tuple[str, str, Mapping[str, str] | str]] = []
    for dataset in ("Stieger", "Lee2019"):
        for method in FIXED_METHODS:
            policies.append((dataset, f"fixed__{method}", method))
    policies.append(
        (
            "Stieger",
            "fixed__condition_long_lr05_ud20",
            {"pure_lr": "longitudinal_q0p05", "pure_ud": "longitudinal_q0p20"},
        )
    )
    summary: dict[str, dict[str, object]] = {}
    for dataset, policy, selector in policies:
        rows = fixed_records(frame, dataset, policy, selector)
        if not rows:
            continue
        summary[f"{dataset}__{policy}"] = summarize_selected(
            rows,
            bootstrap=bootstrap,
            seed=seed + len(dataset) + len(policy),
        )
    return summary


def e15_rows(e15_summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary = e15_summary["summary"]
    assert isinstance(summary, dict)
    for key, item in summary.items():
        dataset, policy, pool = key.split("__", maxsplit=2)
        rows.append(
            {
                "dataset": dataset,
                "validation_policy": policy,
                "pool_size": pool.removeprefix("m"),
                "validation_gain_pp": float(item["gain_vs_full_mean_pp"]),
                "validation_gain_ci_low": float(item["gain_repeat_95ci"][0]),
                "validation_gain_ci_high": float(item["gain_repeat_95ci"][1]),
                "validation_r10": float(item["loss_r10_vs_full_unit_pp"]),
                "validation_p_loss_gt5": float(item["p_gain_vs_full_lt_minus5_unit"]),
                "validation_retention_vs_all": float(item.get("gain_retention_vs_all", float("nan"))),
                "validation_regret_vs_all_pp": float(item.get("regret_vs_all_pp", float("nan"))),
                "validation_top_choices": json.dumps(item["chosen_candidate_rates"], sort_keys=True),
            }
        )
    return rows


def compare_validation_to_fixed(
    fixed_summary: Mapping[str, Mapping[str, object]],
    e15_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    validation_rows = e15_rows(e15_summary)
    for val in validation_rows:
        dataset = str(val["dataset"])
        fixed_candidates = [
            (key, item)
            for key, item in fixed_summary.items()
            if key.startswith(dataset + "__fixed__") and key != f"{dataset}__fixed__full_q1p00"
        ]
        # Stieger condition_risk is allowed to use the condition-known fixed policy.
        if dataset == "Stieger" and str(val["validation_policy"]) != "condition_risk":
            fixed_candidates = [
                (key, item)
                for key, item in fixed_candidates
                if "condition_long_lr05_ud20" not in key
            ]
        best_gain_key, best_gain = max(
            fixed_candidates,
            key=lambda pair: float(pair[1]["gain_vs_full_mean_pp"]),
        )
        best_safe_key, best_safe = max(
            fixed_candidates,
            key=lambda pair: (
                float(pair[1]["gain_vs_full_mean_pp"])
                if float(pair[1]["p_gain_vs_full_lt_minus5_unit"]) <= float(val["validation_p_loss_gt5"]) + 0.01
                else -1e9
            ),
        )
        rows.append(
            {
                **val,
                "best_fixed_by_gain": best_gain_key.split("__fixed__", maxsplit=1)[1],
                "best_fixed_gain_pp": float(best_gain["gain_vs_full_mean_pp"]),
                "best_fixed_gain_ci_low": float(best_gain["gain_subject_bootstrap_95ci"][0]),
                "best_fixed_gain_ci_high": float(best_gain["gain_subject_bootstrap_95ci"][1]),
                "best_fixed_r10": float(best_gain["loss_r10_vs_full_unit_pp"]),
                "best_fixed_p_loss_gt5": float(best_gain["p_gain_vs_full_lt_minus5_unit"]),
                "validation_minus_best_fixed_pp": float(val["validation_gain_pp"]) - float(best_gain["gain_vs_full_mean_pp"]),
                "best_fixed_with_similar_risk": best_safe_key.split("__fixed__", maxsplit=1)[1],
                "best_similar_risk_fixed_gain_pp": float(best_safe["gain_vs_full_mean_pp"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([load_stieger(args.stieger_dir), load_lee(args.lee_dir)], ignore_index=True)
    frame = frame[frame["method"].isin(set(FIXED_METHODS) | {"longitudinal_q1p00"})].copy()
    fixed = build_fixed_summary(frame, bootstrap=int(args.bootstrap), seed=int(args.seed))
    e15 = json.loads(args.e15_summary.read_text())
    comparisons = compare_validation_to_fixed(fixed, e15)

    write_csv(args.output_dir / "fixed_policy_summary.csv", [{"dataset_policy": k, **v} for k, v in sorted(fixed.items())])
    write_csv(args.output_dir / "validation_vs_fixed.csv", comparisons)
    report = {
        "config": {
            "fixed_methods": list(FIXED_METHODS),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "e15_summary": str(args.e15_summary),
        },
        "fixed_summary": fixed,
        "validation_vs_fixed": comparisons,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_fixed_policies": len(fixed),
                "n_comparisons": len(comparisons),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
