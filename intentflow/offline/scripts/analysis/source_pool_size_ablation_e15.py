"""E15: source-pool size ablation for source-side candidate selection.

This script stress-tests E10:

    Does source-side validation still select a useful subspace candidate when
    only a small historical source pool is available?

It uses existing E9 selection records, so it is a fast/proxy audit rather than
an exact recomputation of subspace scores.  For each held-out target subject,
we sample m source subjects from the remaining subjects, choose the candidate
using only those sampled source subjects, then evaluate on the held-out subject.

The candidate set is deliberately small and matches the E10 primary set.  This
is a stability audit of the current rule, not another candidate-search method.
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
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e15_source_pool_size_ablation"

PRIMARY_CANDIDATES = (
    "full_q1p00",
    "source_only_q0p15",
    "source_only_q0p25",
    "source_only_q0p50",
    "longitudinal_q0p05",
    "longitudinal_q0p10",
    "longitudinal_q0p20",
    "longitudinal_q0p25",
    "longitudinal_q0p50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stieger-dir", type=Path, default=DEFAULT_STIEGER)
    parser.add_argument("--lee-dir", type=Path, default=DEFAULT_LEE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-sizes", nargs="+", default=["3", "5", "10", "20", "40", "all"])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
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


def parse_pool_sizes(values: Sequence[str]) -> list[int | str]:
    output: list[int | str] = []
    for value in values:
        token = str(value).strip().lower()
        if token == "all":
            output.append("all")
        else:
            output.append(int(token))
    return output


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


def add_gain_vs_full(frame: pd.DataFrame) -> pd.DataFrame:
    full = (
        frame[frame["method"] == "full_q1p00"][
            ["dataset", "unit_id", "accuracy", "correct", "n_eval"]
        ]
        .rename(columns={"accuracy": "full_accuracy", "correct": "full_correct", "n_eval": "full_n_eval"})
        .drop_duplicates(["dataset", "unit_id"])
    )
    output = frame.merge(full, on=["dataset", "unit_id"], how="left")
    output["gain_vs_full"] = output["accuracy"] - output["full_accuracy"]
    return output


def source_validation_metrics(rows: pd.DataFrame) -> dict[str, float]:
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
    boundary = int(np.searchsorted(cumulative, 0.10, side="left"))
    tail_mask = np.zeros_like(gain_sorted, dtype=bool)
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
) -> tuple[str, list[dict[str, object]]]:
    diagnostics: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_rows = source_rows[source_rows["method"] == candidate]
        diagnostics.append(
            {
                "candidate": candidate,
                **source_validation_metrics(candidate_rows),
                "n_units": int(candidate_rows["unit_id"].nunique()),
                "n_subjects": int(candidate_rows["subject"].nunique()),
            }
        )
    eligible = [
        item
        for item in diagnostics
        if np.isfinite(float(item["mean_gain"]))
        and float(item["p_gain_lt_minus5"]) <= risk_threshold
    ]
    if not eligible:
        return "full_q1p00", diagnostics
    rank = {candidate: index for index, candidate in enumerate(candidates)}
    chosen = sorted(
        eligible,
        key=lambda item: (
            float(item["mean_gain"]),
            -float(item["r10_loss"]),
            -rank[str(item["candidate"])],
        ),
        reverse=True,
    )[0]
    return str(chosen["candidate"]), diagnostics


def sampled_subjects(
    rng: np.random.Generator,
    available: Sequence[int],
    pool_size: int | str,
) -> list[int]:
    available = list(map(int, available))
    if pool_size == "all":
        return available
    m = min(int(pool_size), len(available))
    return sorted(map(int, rng.choice(available, size=m, replace=False)))


def selected_records_for_run(
    data: pd.DataFrame,
    dataset: str,
    policy: str,
    held_subject: int,
    pool_size_label: str,
    repeat: int,
    sampled: Sequence[int],
    candidates: Sequence[str],
    risk_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = data[data["subject"].isin(sampled)]
    target = data[data["subject"] == held_subject]
    condition_specific = policy == "condition_risk"
    conditions = sorted(target["condition"].unique()) if condition_specific else ["__all__"]
    chosen_by_condition: dict[str, str] = {}
    diagnostics: list[dict[str, object]] = []
    for condition in conditions:
        source_scope = source[source["condition"] == condition] if condition_specific else source
        chosen, metrics = choose_candidate(source_scope, candidates, risk_threshold)
        chosen_by_condition[condition] = chosen
        for item in metrics:
            diagnostics.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "pool_size": pool_size_label,
                    "repeat": int(repeat),
                    "held_subject": int(held_subject),
                    "condition_scope": condition,
                    "chosen_candidate": chosen,
                    "sampled_subjects": " ".join(map(str, sampled)),
                    **item,
                }
            )
    selected: list[dict[str, object]] = []
    for _, full_row in target[target["method"] == "full_q1p00"].iterrows():
        condition = str(full_row["condition"])
        chosen = chosen_by_condition[condition if condition_specific else "__all__"]
        row = target[(target["unit_id"] == full_row["unit_id"]) & (target["method"] == chosen)]
        if row.empty:
            raise RuntimeError(f"Missing selected row: {dataset} S{held_subject} {chosen}")
        item = row.iloc[0]
        selected.append(
            {
                "dataset": dataset,
                "policy": policy,
                "pool_size": pool_size_label,
                "repeat": int(repeat),
                "subject": int(held_subject),
                "session": int(full_row["session"]),
                "condition": condition,
                "unit_id": str(full_row["unit_id"]),
                "sampled_n": int(len(sampled)),
                "chosen_candidate": chosen,
                "correct": int(item["correct"]),
                "full_correct": int(full_row["correct"]),
                "n_eval": int(full_row["n_eval"]),
                "accuracy": float(item["accuracy"]),
                "full_accuracy": float(full_row["accuracy"]),
                "gain_vs_full": float(item["accuracy"] - full_row["accuracy"]),
            }
        )
    return selected, diagnostics


def run_ablation(
    frame: pd.DataFrame,
    dataset: str,
    policies: Sequence[str],
    pool_sizes: Sequence[int | str],
    repeats: int,
    candidates: Sequence[str],
    risk_threshold: float,
    seed: int,
    quiet: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = frame[frame["dataset"] == dataset].copy()
    subjects = sorted(map(int, data["subject"].unique()))
    selected: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for policy in policies:
        for pool_size in pool_sizes:
            pool_label = str(pool_size)
            n_repeats = 1 if pool_size == "all" else int(repeats)
            for repeat in range(n_repeats):
                for held in subjects:
                    available = [subject for subject in subjects if int(subject) != int(held)]
                    sampled = sampled_subjects(rng, available, pool_size)
                    rows, diag = selected_records_for_run(
                        data,
                        dataset,
                        policy,
                        int(held),
                        pool_label,
                        int(repeat),
                        sampled,
                        candidates,
                        risk_threshold,
                    )
                    selected.extend(rows)
                    diagnostics.extend(diag)
                if not quiet and (repeat + 1) % 25 == 0:
                    print(f"{dataset} {policy} m={pool_label}: repeat {repeat + 1}/{n_repeats}", flush=True)
    return selected, diagnostics


def summarize_group(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    repeat_subject = (
        frame.groupby(["repeat", "subject"], sort=False)
        .agg(correct=("correct", "sum"), full_correct=("full_correct", "sum"), n_eval=("n_eval", "sum"))
        .reset_index()
    )
    repeat_subject["accuracy"] = 100.0 * repeat_subject["correct"] / repeat_subject["n_eval"]
    repeat_subject["gain"] = 100.0 * (
        repeat_subject["correct"] - repeat_subject["full_correct"]
    ) / repeat_subject["n_eval"]
    repeat_mean = repeat_subject.groupby("repeat", sort=False).agg(
        accuracy=("accuracy", "mean"),
        gain=("gain", "mean"),
    )

    unit = frame.copy()
    counts = unit.groupby(["repeat", "subject"])["unit_id"].transform("count")
    n_repeats = unit["repeat"].nunique()
    n_subjects = unit["subject"].nunique()
    unit["weight"] = 1.0 / (n_repeats * n_subjects * counts)
    gain = unit["gain_vs_full"].to_numpy(dtype=float)
    weight = unit["weight"].to_numpy(dtype=float)
    order = np.argsort(gain)
    gain_sorted = gain[order]
    weight_sorted = weight[order]
    cumulative = np.cumsum(weight_sorted)
    q05 = float(gain_sorted[np.searchsorted(cumulative, 0.05, side="left")])
    boundary = int(np.searchsorted(cumulative, 0.10, side="left"))
    tail_mask = np.zeros_like(gain_sorted, dtype=bool)
    tail_mask[: min(boundary + 1, len(tail_mask))] = True
    r10 = float(-np.sum(weight_sorted[tail_mask] * gain_sorted[tail_mask]) / np.sum(weight_sorted[tail_mask]))
    choices = Counter(str(row["chosen_candidate"]) for row in rows)
    family_counts = Counter(str(row["chosen_candidate"]).split("_q", maxsplit=1)[0] for row in rows)
    return {
        "n_repeats": int(n_repeats),
        "n_subjects": int(n_subjects),
        "n_units": int(len(rows)),
        "accuracy_mean": float(repeat_mean["accuracy"].mean()),
        "accuracy_repeat_95ci": [
            float(np.quantile(repeat_mean["accuracy"], 0.025)),
            float(np.quantile(repeat_mean["accuracy"], 0.975)),
        ],
        "gain_vs_full_mean_pp": float(repeat_mean["gain"].mean()),
        "gain_repeat_95ci": [
            float(np.quantile(repeat_mean["gain"], 0.025)),
            float(np.quantile(repeat_mean["gain"], 0.975)),
        ],
        "gain_vs_full_q05_unit_pp": q05,
        "loss_r10_vs_full_unit_pp": r10,
        "p_gain_vs_full_lt_minus5_unit": float(np.sum(weight[gain < -5.0])),
        "chosen_candidate_counts": dict(choices),
        "chosen_candidate_rates": {key: float(value / len(rows)) for key, value in sorted(choices.items())},
        "chosen_family_rates": {key: float(value / len(rows)) for key, value in sorted(family_counts.items())},
    }


def summarize(selected: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    frame = pd.DataFrame(selected)
    summary: dict[str, dict[str, object]] = {}
    for (dataset, policy, pool_size), group in frame.groupby(["dataset", "policy", "pool_size"], sort=False):
        key = f"{dataset}__{policy}__m{pool_size}"
        summary[key] = summarize_group(group.to_dict("records"))

    # Add regret/retention against the all-source point for each dataset/policy.
    for key, item in list(summary.items()):
        dataset, policy, _ = key.split("__", maxsplit=2)
        all_key = f"{dataset}__{policy}__mall"
        if all_key not in summary:
            continue
        all_gain = float(summary[all_key]["gain_vs_full_mean_pp"])
        gain = float(item["gain_vs_full_mean_pp"])
        item["regret_vs_all_pp"] = float(all_gain - gain)
        item["gain_retention_vs_all"] = float(gain / all_gain) if all_gain > 0 else float("nan")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_sizes = parse_pool_sizes(args.pool_sizes)
    frame = pd.concat([load_stieger(args.stieger_dir), load_lee(args.lee_dir)], ignore_index=True)
    frame = add_gain_vs_full(frame)
    frame = frame[frame["method"].isin(PRIMARY_CANDIDATES)].copy()

    selected_all: list[dict[str, object]] = []
    diagnostics_all: list[dict[str, object]] = []
    runs = [
        ("Stieger", ["global_risk", "condition_risk"]),
        ("Lee2019", ["global_risk"]),
    ]
    for dataset, policies in runs:
        selected, diagnostics = run_ablation(
            frame,
            dataset=dataset,
            policies=policies,
            pool_sizes=pool_sizes,
            repeats=int(args.repeats),
            candidates=PRIMARY_CANDIDATES,
            risk_threshold=float(args.risk_threshold),
            seed=int(args.seed) + len(dataset),
            quiet=bool(args.quiet),
        )
        selected_all.extend(selected)
        diagnostics_all.extend(diagnostics)

    write_csv(args.output_dir / "selected_records.csv", selected_all)
    write_csv(args.output_dir / "source_validation_diagnostics.csv", diagnostics_all)
    summary = summarize(selected_all)
    write_csv(
        args.output_dir / "summary.csv",
        [{"dataset_policy_pool": key, **value} for key, value in sorted(summary.items())],
    )
    report = {
        "config": {
            "pool_sizes": [str(value) for value in pool_sizes],
            "repeats": int(args.repeats),
            "risk_threshold": float(args.risk_threshold),
            "seed": int(args.seed),
            "candidates": list(PRIMARY_CANDIDATES),
            "stieger_limitation": "fast proxy from existing LOSO records; not exact double-LOSO",
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_selected_records": len(selected_all),
                "n_diagnostics": len(diagnostics_all),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
