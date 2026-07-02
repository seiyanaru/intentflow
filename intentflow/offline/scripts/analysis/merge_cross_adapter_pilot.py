"""Merge sharded Stieger cross-adapter pilot outputs.

This is a pilot diagnostic, not the preregistered confirmatory analysis.  It
combines subject JSON files, recomputes subject-balanced risk/utility metrics,
and reports residualized subject rankings and odd/even-session reliability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cross_adapter_core import risk_utility_summary


ADAPTERS = ("ea", "adabn", "tent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing shard_*/subjects/S*_seed*.json files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-subjects", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def safe_spearman(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n_subjects": len(x), "rho": None, "pvalue": None}
    result = spearmanr(x, y)
    return {
        "n_subjects": len(x),
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def residualize_rows(rows: list[dict]) -> list[dict]:
    """Remove the pilot approximation of preregistered fixed effects."""
    residualized: list[dict] = []
    for adapter in ADAPTERS:
        selected = [row for row in rows if row["adapter"] == adapter]
        if not selected:
            continue
        source = np.asarray([row["source_acc"] for row in selected], dtype=float)
        session = np.asarray([row["session"] for row in selected], dtype=float)
        n_eval = np.asarray([row["n_eval"] for row in selected], dtype=float)
        design = np.column_stack(
            [
                np.ones(len(selected)),
                source,
                source**2,
                session,
                n_eval,
            ]
        )
        outcome = np.asarray([row["delta_pp"] for row in selected], dtype=float)
        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
        residuals = outcome - design @ coefficients
        for row, residual in zip(selected, residuals):
            copied = dict(row)
            copied["residual_delta_pp"] = float(residual)
            residualized.append(copied)
    return residualized


def subject_means(rows: list[dict], value_key: str) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        values[row["adapter"]][int(row["subject"])].append(float(row[value_key]))
    return {
        adapter: {
            subject: float(np.mean(subject_values))
            for subject, subject_values in by_subject.items()
        }
        for adapter, by_subject in values.items()
    }


def pairwise_correlations(
    means: dict[str, dict[int, float]]
) -> dict[str, dict[str, float | int | None]]:
    correlations: dict[str, dict[str, float | int | None]] = {}
    for index, adapter_a in enumerate(ADAPTERS):
        for adapter_b in ADAPTERS[index + 1 :]:
            common = sorted(set(means.get(adapter_a, {})) & set(means.get(adapter_b, {})))
            correlations[f"{adapter_a}__{adapter_b}"] = safe_spearman(
                [means[adapter_a][subject] for subject in common],
                [means[adapter_b][subject] for subject in common],
            )
    return correlations


def split_half_reliability(rows: list[dict]) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for adapter in ADAPTERS:
        selected = [row for row in rows if row["adapter"] == adapter]
        odd: dict[int, list[float]] = defaultdict(list)
        even: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            target = odd if int(row["session"]) % 2 else even
            target[int(row["subject"])].append(float(row["residual_delta_pp"]))
        common = sorted(
            subject
            for subject in set(odd) & set(even)
            if odd[subject] and even[subject]
        )
        correlation = safe_spearman(
            [float(np.mean(odd[subject])) for subject in common],
            [float(np.mean(even[subject])) for subject in common],
        )
        rho = correlation["rho"]
        correlation["spearman_brown_rho"] = (
            float(2 * rho / (1 + rho)) if rho is not None and rho > -1 else None
        )
        output[adapter] = correlation
    return output


def main() -> None:
    args = parse_args()
    subject_paths = sorted(
        args.input_root.glob(f"**/subjects/S*_seed{args.seed}.json")
    )
    payloads = [json.loads(path.read_text()) for path in subject_paths]
    subjects = sorted({int(payload["subject"]) for payload in payloads})
    rows = [row for payload in payloads for row in payload["rows"]]
    residualized = residualize_rows(rows)

    source_by_subject = {
        int(payload["subject"]): float(payload["mean_target_source_accuracy"])
        for payload in payloads
    }
    raw_means = subject_means(rows, "delta_pp")
    residual_means = subject_means(residualized, "residual_delta_pp")
    present_adapters = tuple(
        adapter for adapter in ADAPTERS if any(row["adapter"] == adapter for row in rows)
    )
    summaries = {
        adapter: risk_utility_summary(rows, adapter)
        for adapter in present_adapters
    }
    adapter_activity = {}
    for adapter in present_adapters:
        selected = [row for row in rows if row["adapter"] == adapter]
        subject_values = list(raw_means.get(adapter, {}).values())
        adapter_activity[adapter] = {
            "mean_absolute_session_delta_pp": float(
                np.mean([abs(float(row["delta_pp"])) for row in selected])
            ),
            "fraction_sessions_nonzero_delta": float(
                np.mean([not np.isclose(float(row["delta_pp"]), 0) for row in selected])
            ),
            "sd_subject_mean_delta_pp": float(np.std(subject_values, ddof=1)),
        }

    completed_enough = len(subjects) >= int(np.ceil(0.875 * args.expected_subjects))
    median_source = float(np.median(list(source_by_subject.values())))
    report = {
        "analysis_scope": (
            "Pilot diagnostic only. Correlations have no bootstrap CI and do not "
            "replace the preregistered 62-subject analysis."
        ),
        "seed": args.seed,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_unique_target_sessions": len(
            {(int(row["subject"]), int(row["session"])) for row in rows}
        ),
        "pilot_acceptance": {
            "expected_subjects": args.expected_subjects,
            "completed_enough": completed_enough,
            "median_target_source_accuracy": median_source,
            "median_target_source_accuracy_ge_60": median_source >= 60.0,
            "all_resets_match": all(
                bool(payload["all_resets_match"]) for payload in payloads
            ),
            "phase1_full_run_authorized": (
                completed_enough
                and median_source >= 60.0
                and all(bool(payload["all_resets_match"]) for payload in payloads)
            ),
        },
        "source_accuracy_by_subject": source_by_subject,
        "risk_utility": summaries,
        "adapter_activity": adapter_activity,
        "subject_mean_delta_pp": raw_means,
        "raw_subject_mean_spearman": pairwise_correlations(raw_means),
        "residualized_subject_mean_spearman": pairwise_correlations(residual_means),
        "residualized_odd_even_split_half": split_half_reliability(residualized),
    }

    output_json = args.output_json or args.input_root / f"summary_16_seed{args.seed}.json"
    output_csv = args.output_csv or args.input_root / f"subject_means_seed{args.seed}.csv"
    output_json.write_text(json.dumps(report, indent=2))
    with output_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["subject", "source_acc", "ea_delta", "adabn_delta", "tent_delta"]
        )
        for subject in subjects:
            writer.writerow(
                [
                    subject,
                    source_by_subject[subject],
                    raw_means.get("ea", {}).get(subject),
                    raw_means.get("adabn", {}).get(subject),
                    raw_means.get("tent", {}).get(subject),
                ]
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
