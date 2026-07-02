"""E6-fast: source-side nested candidate selection from existing result records.

This script answers the immediate post-hoc criticism:

    "You report longitudinal on Stieger and source-only on Lee2019.  Did you
     choose those after looking at the target dataset?"

For each held-out target subject, we select a candidate using only the other
subjects' already-computed validation records, then evaluate the selected
candidate on the held-out subject.

Important limitation:

* For Stieger, this is a fast nested-selection proxy built from existing LOSO
  records.  The source-validation rows for inner subjects were originally
  computed with metric statistics that excluded the inner subject, not the
  outer held-out subject.  Therefore it is not a final double-LOSO proof.
* For Lee2019, because the feature statistics are subject-level and the records
  are simpler, this is closer to the desired source-side selection audit.

Use this as a go/no-go check before spending time on exact double-LOSO Stieger.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_STIEGER = RESULTS_DIR / "260627_stieger_longitudinal_selection_ablation"
DEFAULT_LEE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot54"
DEFAULT_OUTPUT = RESULTS_DIR / "260627_source_side_nested_selection_from_records"

CANDIDATES = (
    "full_q1p00",
    "source_only_q0p25",
    "source_only_q0p10",
    "longitudinal_q0p25",
    "longitudinal_q0p10",
    "sep_no_drift_q0p25",
    "sep_no_drift_q0p10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stieger-dir", type=Path, default=DEFAULT_STIEGER)
    parser.add_argument("--lee-dir", type=Path, default=DEFAULT_LEE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
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
    frame = frame.rename(
        columns={
            "test_correct": "correct",
            "test_acc": "accuracy",
        }
    )
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
    full = frame[frame["method"] == "longitudinal_q1p00"].copy()
    full["method"] = "full_q1p00"
    return pd.concat([frame, full], ignore_index=True)


def load_lee(lee_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(lee_dir / "selection_records.csv")
    frame = frame.rename(columns={"accuracy": "accuracy"})
    frame["dataset"] = "Lee2019"
    frame["session"] = 1
    frame["condition"] = "lee_lr"
    frame["unit_id"] = frame["subject"].astype(str)
    frame["correct"] = np.rint(frame["accuracy"] * frame["n_eval"] / 100.0).astype(int)
    return frame[
        ["dataset", "subject", "session", "condition", "unit_id", "method", "correct", "n_eval", "accuracy"]
    ]


def add_gain_vs_full(frame: pd.DataFrame) -> pd.DataFrame:
    full = (
        frame[frame["method"] == "full_q1p00"][
            ["dataset", "unit_id", "accuracy", "correct", "n_eval"]
        ]
        .rename(
            columns={
                "accuracy": "full_accuracy",
                "correct": "full_correct",
                "n_eval": "full_n_eval",
            }
        )
        .drop_duplicates(["dataset", "unit_id"])
    )
    out = frame.merge(full, on=["dataset", "unit_id"], how="left")
    out["gain_vs_full"] = out["accuracy"] - out["full_accuracy"]
    return out


def subject_balanced_metrics(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "mean_gain": float("nan"),
            "p_gain_lt_minus5": float("nan"),
            "q05_gain": float("nan"),
            "r10_loss": float("nan"),
        }
    rows = rows.copy()
    counts = rows.groupby("subject")["unit_id"].transform("count")
    n_subjects = rows["subject"].nunique()
    rows["weight"] = 1.0 / (n_subjects * counts)
    gain = rows["gain_vs_full"].to_numpy(dtype=float)
    weight = rows["weight"].to_numpy(dtype=float)
    order = np.argsort(gain)
    gain_sorted = gain[order]
    weight_sorted = weight[order]
    cumulative = np.cumsum(weight_sorted)
    q05 = float(gain_sorted[np.searchsorted(cumulative, 0.05, side="left")])
    tail_mask = cumulative <= 0.10
    if not np.any(tail_mask):
        tail_mask[0] = True
    # Include the boundary point for a conservative lower-tail estimate.
    boundary = int(np.searchsorted(cumulative, 0.10, side="left"))
    tail_mask[: min(boundary + 1, len(tail_mask))] = True
    tail_weight = weight_sorted[tail_mask]
    tail_gain = gain_sorted[tail_mask]
    return {
        "mean_gain": float(np.sum(weight * gain)),
        "p_gain_lt_minus5": float(np.sum(weight[gain < -5.0])),
        "q05_gain": q05,
        "r10_loss": float(-np.sum(tail_weight * tail_gain) / np.sum(tail_weight)),
    }


def choose_candidate(
    source_rows: pd.DataFrame,
    candidates: Sequence[str],
    risk_threshold: float,
    mean_only: bool = False,
) -> tuple[str, list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    for candidate in candidates:
        rows = source_rows[source_rows["method"] == candidate]
        item = {
            "candidate": candidate,
            **subject_balanced_metrics(rows),
            "n_units": int(rows["unit_id"].nunique()),
            "n_subjects": int(rows["subject"].nunique()),
        }
        metrics.append(item)
    eligible = [
        item
        for item in metrics
        if np.isfinite(float(item["mean_gain"]))
        and (mean_only or float(item["p_gain_lt_minus5"]) <= risk_threshold)
    ]
    if not eligible:
        return "full_q1p00", metrics
    eligible = sorted(
        eligible,
        key=lambda item: (
            float(item["mean_gain"]),
            -float(item["r10_loss"]),
            -CANDIDATES.index(str(item["candidate"]))
            if str(item["candidate"]) in CANDIDATES
            else 0,
        ),
        reverse=True,
    )
    return str(eligible[0]["candidate"]), metrics


def selected_rows_for_policy(
    frame: pd.DataFrame,
    dataset: str,
    policy: str,
    candidates: Sequence[str],
    risk_threshold: float,
    mean_only: bool,
    condition_specific: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = frame[frame["dataset"] == dataset].copy()
    selected_records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for held_subject in sorted(data["subject"].unique()):
        source = data[data["subject"] != held_subject]
        target = data[data["subject"] == held_subject]
        if condition_specific:
            conditions = sorted(target["condition"].unique())
        else:
            conditions = ["__all__"]
        chosen_by_condition: dict[str, str] = {}
        for condition in conditions:
            if condition_specific:
                source_scope = source[source["condition"] == condition]
            else:
                source_scope = source
            chosen, metrics = choose_candidate(
                source_scope,
                candidates=candidates,
                risk_threshold=risk_threshold,
                mean_only=mean_only,
            )
            chosen_by_condition[condition] = chosen
            for item in metrics:
                diagnostics.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "held_subject": int(held_subject),
                        "condition_scope": condition,
                        "chosen_candidate": chosen,
                        **item,
                    }
                )
        for _, full_row in target[target["method"] == "full_q1p00"].iterrows():
            condition = str(full_row["condition"])
            chosen = chosen_by_condition[condition if condition_specific else "__all__"]
            row = target[
                (target["unit_id"] == full_row["unit_id"]) & (target["method"] == chosen)
            ]
            if row.empty:
                raise RuntimeError(f"Missing chosen row for {dataset} S{held_subject} {chosen}")
            selected = row.iloc[0]
            selected_records.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "subject": int(held_subject),
                    "session": int(full_row["session"]),
                    "condition": condition,
                    "unit_id": str(full_row["unit_id"]),
                    "chosen_candidate": chosen,
                    "correct": int(selected["correct"]),
                    "full_correct": int(full_row["correct"]),
                    "n_eval": int(full_row["n_eval"]),
                    "accuracy": float(selected["accuracy"]),
                    "full_accuracy": float(full_row["accuracy"]),
                    "gain_vs_full": float(selected["accuracy"] - full_row["accuracy"]),
                }
            )
    return selected_records, diagnostics


def fixed_candidate_records(frame: pd.DataFrame, dataset: str, candidate: str) -> list[dict[str, object]]:
    data = frame[frame["dataset"] == dataset].copy()
    records: list[dict[str, object]] = []
    for _, full_row in data[data["method"] == "full_q1p00"].iterrows():
        row = data[(data["unit_id"] == full_row["unit_id"]) & (data["method"] == candidate)]
        if row.empty:
            continue
        selected = row.iloc[0]
        records.append(
            {
                "dataset": dataset,
                "policy": f"fixed__{candidate}",
                "subject": int(full_row["subject"]),
                "session": int(full_row["session"]),
                "condition": str(full_row["condition"]),
                "unit_id": str(full_row["unit_id"]),
                "chosen_candidate": candidate,
                "correct": int(selected["correct"]),
                "full_correct": int(full_row["correct"]),
                "n_eval": int(full_row["n_eval"]),
                "accuracy": float(selected["accuracy"]),
                "full_accuracy": float(full_row["accuracy"]),
                "gain_vs_full": float(selected["accuracy"] - full_row["accuracy"]),
            }
        )
    return records


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_records(records: Sequence[Mapping[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_subject[int(row["subject"])].append(row)
    subject_acc = []
    subject_gain = []
    for rows in by_subject.values():
        correct = sum(int(row["correct"]) for row in rows)
        full_correct = sum(int(row["full_correct"]) for row in rows)
        total = sum(int(row["n_eval"]) for row in rows)
        subject_acc.append(100.0 * correct / total)
        subject_gain.append(100.0 * (correct - full_correct) / total)
    subject_acc = np.asarray(subject_acc, dtype=float)
    subject_gain = np.asarray(subject_gain, dtype=float)
    unit_gains = np.asarray([float(row["gain_vs_full"]) for row in records], dtype=float)
    tail_count = max(1, int(np.ceil(0.1 * len(unit_gains)))) if len(unit_gains) else 1
    rng = np.random.default_rng(seed)
    choices = Counter(str(row["chosen_candidate"]) for row in records)
    return {
        "n_subjects": int(len(by_subject)),
        "n_units": int(len(records)),
        "accuracy_subject_pooled_mean": float(subject_acc.mean()) if len(subject_acc) else float("nan"),
        "gain_vs_full_subject_pooled_pp": float(subject_gain.mean()) if len(subject_gain) else float("nan"),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(subject_gain, rng, bootstrap),
        "gain_vs_full_q05_unit_pp": float(np.quantile(unit_gains, 0.05)) if len(unit_gains) else float("nan"),
        "loss_r10_vs_full_unit_pp": float(-np.sort(unit_gains)[:tail_count].mean()) if len(unit_gains) else float("nan"),
        "p_gain_vs_full_lt_minus5_unit": float(np.mean(unit_gains < -5.0)) if len(unit_gains) else float("nan"),
        "chosen_candidate_counts": dict(choices),
        "chosen_candidate_rates": {key: float(value / len(records)) for key, value in sorted(choices.items())},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([load_stieger(args.stieger_dir), load_lee(args.lee_dir)], ignore_index=True)
    frame = add_gain_vs_full(frame)
    frame = frame[frame["method"].isin(CANDIDATES)].copy()

    all_selected: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    policies = [
        ("global_risk", False, False),
        ("condition_risk", False, True),
        ("global_mean", True, False),
        ("condition_mean", True, True),
    ]
    datasets = sorted(frame["dataset"].unique())
    for dataset in datasets:
        for policy, mean_only, condition_specific in policies:
            selected, diagnostics = selected_rows_for_policy(
                frame,
                dataset=dataset,
                policy=policy,
                candidates=CANDIDATES,
                risk_threshold=float(args.risk_threshold),
                mean_only=mean_only,
                condition_specific=condition_specific,
            )
            all_selected.extend(selected)
            all_diagnostics.extend(diagnostics)
        for candidate in CANDIDATES:
            all_selected.extend(fixed_candidate_records(frame, dataset, candidate))

    write_csv(args.output_dir / "nested_selection_records.csv", all_selected)
    write_csv(args.output_dir / "source_validation_diagnostics.csv", all_diagnostics)

    summary: dict[str, dict[str, object]] = {}
    for (dataset, policy), group in pd.DataFrame(all_selected).groupby(["dataset", "policy"]):
        summary[f"{dataset}__{policy}"] = summarize_records(
            group.to_dict("records"),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
    summary_rows = [
        {"dataset_policy": key, **value}
        for key, value in sorted(summary.items())
    ]
    write_csv(args.output_dir / "nested_summary.csv", summary_rows)
    (args.output_dir / "nested_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "candidates": list(CANDIDATES),
                    "risk_threshold": float(args.risk_threshold),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
                    "stieger_limitation": (
                        "fast proxy from existing LOSO records; not exact double-LOSO"
                    ),
                },
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_selected_records": len(all_selected),
                "n_diagnostics": len(all_diagnostics),
                "summary_json": str(args.output_dir / "nested_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
