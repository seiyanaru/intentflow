"""E6 exact nested source-side subspace selection for Lee2019_MI.

For each held-out target subject H:

1. Build source-validation candidates using only subjects != H.
2. For each inner validation subject S != H, compute feature scores from
   subjects excluding both H and S.
3. Evaluate each candidate on S session0 -> session1.
4. Select the candidate with highest source-validation mean gain subject to a
   risk constraint P(gain < -5pp) <= threshold.
5. Recompute scores using all subjects excluding H and evaluate once on H.

This is the exact version of the E6 source-side selection audit for Lee2019.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LEE_OUTPUT,
    SCORE_FUNCTIONS,
    fit_predict_accuracy,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260627_lee2019_exact_nested_subspace_selection"
DEFAULT_CACHE = DEFAULT_LEE_OUTPUT / "subject_cache"
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
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument(
        "--target-subjects",
        default=None,
        help=(
            "Optional held-out subjects to evaluate. Source-validation statistics "
            "are still built from --subjects, so this is only a runtime sharding aid."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
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


def add_stats(target: dict[str, np.ndarray], stats: Mapping[str, np.ndarray], sign: float = 1.0) -> None:
    for key, value in stats.items():
        if key not in target:
            target[key] = sign * np.asarray(value, dtype=np.float64).copy()
        else:
            target[key] += sign * np.asarray(value, dtype=np.float64)


def average_stats(total: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = float(np.asarray(total["count"])[0])
    if count <= 0:
        raise ValueError("No source subjects left for aggregate stats")
    averaged = {
        key: np.asarray(value, dtype=np.float64) / count
        for key, value in total.items()
        if key != "count"
    }
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def aggregate_excluding(
    total_stats: Mapping[str, np.ndarray],
    stats_by_subject: Mapping[int, Mapping[str, np.ndarray]],
    excluded: set[int],
) -> dict[str, np.ndarray]:
    aggregate: dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in total_stats.items()
    }
    for subject in excluded:
        add_stats(aggregate, stats_by_subject[int(subject)], sign=-1.0)
    return average_stats(aggregate)


def candidate_indices(candidate: str, stats: Mapping[str, np.ndarray], dim: int) -> np.ndarray:
    if candidate == "full_q1p00":
        return np.arange(dim, dtype=np.int64)
    family, qtoken = candidate.rsplit("_", maxsplit=1)
    fraction = float(qtoken.replace("q", "").replace("p", "."))
    if family not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown candidate family: {candidate}")
    return top_fraction_indices(SCORE_FUNCTIONS[family](stats), fraction)


def evaluate_candidate(
    payload: Mapping[str, np.ndarray],
    candidate: str,
    stats: Mapping[str, np.ndarray],
) -> float:
    indices = candidate_indices(candidate, stats, payload["source_features"].shape[1])
    return fit_predict_accuracy(
        payload["source_features"],
        payload["source_labels"],
        payload["target_features"],
        payload["target_labels"],
        indices,
    )


def source_validation_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    by_candidate: dict[str, list[float]] = {candidate: [] for candidate in CANDIDATES}
    for row in rows:
        for candidate in CANDIDATES:
            by_candidate[candidate].append(float(row[f"{candidate}__gain_vs_full"]))
    metrics: dict[str, dict[str, float]] = {}
    for candidate, gains in by_candidate.items():
        values = np.asarray(gains, dtype=np.float64)
        tail = np.sort(values)[: max(1, int(np.ceil(0.1 * len(values))))]
        metrics[candidate] = {
            "mean_gain": float(values.mean()),
            "p_gain_lt_minus5": float(np.mean(values < -5.0)),
            "q05_gain": float(np.quantile(values, 0.05)),
            "r10_loss": float(-tail.mean()),
        }
    return metrics


def choose_candidate(metrics: Mapping[str, Mapping[str, float]], risk_threshold: float) -> str:
    eligible = [
        candidate
        for candidate, item in metrics.items()
        if float(item["p_gain_lt_minus5"]) <= risk_threshold
    ]
    if not eligible:
        return "full_q1p00"
    return sorted(
        eligible,
        key=lambda candidate: (
            float(metrics[candidate]["mean_gain"]),
            -float(metrics[candidate]["r10_loss"]),
            -CANDIDATES.index(candidate),
        ),
        reverse=True,
    )[0]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(records: Sequence[Mapping[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    gains = np.asarray([float(row["gain_vs_full"]) for row in records], dtype=np.float64)
    acc = np.asarray([float(row["accuracy"]) for row in records], dtype=np.float64)
    tail = np.sort(gains)[: max(1, int(np.ceil(0.1 * len(gains))))]
    rng = np.random.default_rng(seed)
    choices = Counter(str(row["chosen_candidate"]) for row in records)
    return {
        "n_subjects": int(len(records)),
        "accuracy_mean": float(acc.mean()),
        "gain_vs_full_mean_pp": float(gains.mean()),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, bootstrap),
        "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)),
        "loss_r10_vs_full_pp": float(-tail.mean()),
        "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)),
        "chosen_candidate_counts": dict(choices),
        "chosen_candidate_rates": {key: float(value / len(records)) for key, value in sorted(choices.items())},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    target_subjects = parse_subjects(args.target_subjects) if args.target_subjects else subjects
    unknown_targets = sorted(set(target_subjects) - set(subjects))
    if unknown_targets:
        raise ValueError(f"--target-subjects must be a subset of --subjects: {unknown_targets}")

    payloads: dict[int, dict[str, np.ndarray]] = {}
    for subject in subjects:
        payloads[subject] = load_subject_features(
            subject,
            cache_dir=args.cache_dir,
            prefix=int(args.prefix),
            eval_start=int(args.eval_start),
            force_cache=bool(args.force_cache),
        )
        if not args.quiet:
            print(f"S{subject}: payload loaded", flush=True)

    stats_by_subject: dict[int, dict[str, np.ndarray]] = {}
    total_stats: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        stats_by_subject[subject] = stats
        add_stats(total_stats, stats)

    validation_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []

    for held_subject in target_subjects:
        source_validation_rows: list[dict[str, object]] = []
        for inner_subject in subjects:
            if inner_subject == held_subject:
                continue
            stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject, inner_subject})
            payload = payloads[inner_subject]
            full_acc = evaluate_candidate(payload, "full_q1p00", stats)
            row: dict[str, object] = {
                "held_subject": int(held_subject),
                "inner_subject": int(inner_subject),
                "full_accuracy": full_acc,
            }
            for candidate in CANDIDATES:
                acc = evaluate_candidate(payload, candidate, stats)
                row[f"{candidate}__accuracy"] = acc
                row[f"{candidate}__gain_vs_full"] = acc - full_acc
            source_validation_rows.append(row)
            validation_rows.append(row)

        metrics = source_validation_metrics(source_validation_rows)
        chosen = choose_candidate(metrics, risk_threshold=float(args.risk_threshold))
        outer_stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject})
        outer_payload = payloads[held_subject]
        full_acc = evaluate_candidate(outer_payload, "full_q1p00", outer_stats)
        chosen_acc = evaluate_candidate(outer_payload, chosen, outer_stats)
        selected_rows.append(
            {
                "subject": int(held_subject),
                "chosen_candidate": chosen,
                "accuracy": chosen_acc,
                "full_accuracy": full_acc,
                "gain_vs_full": chosen_acc - full_acc,
                "source_validation_metrics": json.dumps(metrics, sort_keys=True),
            }
        )
        for candidate in CANDIDATES:
            acc = evaluate_candidate(outer_payload, candidate, outer_stats)
            fixed_rows.append(
                {
                    "subject": int(held_subject),
                    "chosen_candidate": candidate,
                    "accuracy": acc,
                    "full_accuracy": full_acc,
                    "gain_vs_full": acc - full_acc,
                }
            )
        if not args.quiet:
            print(
                f"S{held_subject}: chosen={chosen} full={full_acc:.1f} chosen={chosen_acc:.1f}",
                flush=True,
            )

    write_csv(args.output_dir / "exact_nested_selection_records.csv", selected_rows)
    write_csv(args.output_dir / "exact_nested_source_validation.csv", validation_rows)
    write_csv(args.output_dir / "fixed_candidate_records.csv", fixed_rows)

    summary = {
        "nested_risk": summarize(selected_rows, int(args.bootstrap), int(args.seed)),
    }
    for candidate in CANDIDATES:
        rows = [row for row in fixed_rows if row["chosen_candidate"] == candidate]
        summary[f"fixed__{candidate}"] = summarize(rows, int(args.bootstrap), int(args.seed))

    write_csv(
        args.output_dir / "exact_nested_summary.csv",
        [{"method": key, **value} for key, value in summary.items()],
    )
    (args.output_dir / "exact_nested_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "subjects": subjects,
                    "target_subjects": target_subjects,
                    "candidates": list(CANDIDATES),
                    "risk_threshold": float(args.risk_threshold),
                    "prefix": int(args.prefix),
                    "eval_start": int(args.eval_start),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
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
                "n_selected": len(selected_rows),
                "summary_json": str(args.output_dir / "exact_nested_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
