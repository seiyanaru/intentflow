"""E9: Lee2019 fraction/compactness sweep for source-side subspace selection.

This is a lightweight version of ``lee2019_longitudinal_selection_pilot.py``:
it evaluates only the two families needed for the compactness frontier:

* source_only
* longitudinal

across a finer set of top-fractions.  It reuses the existing subject feature
cache, so no MOABB loading is needed when the cache exists.
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    fit_predict_accuracy,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
    top_fraction_indices,
    longitudinal_score,
    source_only_score,
    add_stats,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_lee2019_fraction_sweep_e9"
DEFAULT_FRACTIONS = (1.0, 0.50, 0.35, 0.25, 0.20, 0.15, 0.10, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def method_name(family: str, fraction: float) -> str:
    return f"{family}_{fraction_token(float(fraction))}"


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


def aggregate_stats(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        if int(subject) == int(held_subject):
            continue
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        add_stats(total, stats)
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(records: Sequence[Mapping[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    full = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if row["method"] == "full_q1p00"
    }
    output: dict[str, object] = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        by_subject = {int(row["subject"]): float(row["accuracy"]) for row in rows}
        acc = np.asarray([by_subject[s] for s in sorted(by_subject)], dtype=np.float64)
        gains = np.asarray(
            [by_subject[s] - full[s] for s in sorted(by_subject) if s in full],
            dtype=np.float64,
        )
        tail = np.sort(gains)[: max(1, int(np.ceil(0.1 * len(gains))))]
        output[method] = {
            "n_subjects": int(len(acc)),
            "accuracy_mean": float(acc.mean()),
            "gain_vs_full_mean_pp": float(gains.mean()),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(
                gains, rng, int(bootstrap)
            ),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)),
            "loss_r10_vs_full_pp": float(-tail.mean()),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)),
        }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    fractions = sorted(set(float(fraction) for fraction in args.fractions), reverse=True)
    if 1.0 not in fractions:
        fractions = [1.0, *fractions]

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
            print(f"S{subject}: loaded", flush=True)

    records: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "longitudinal": longitudinal_score(stats),
            "source_only": source_only_score(stats),
        }
        dim = payload["source_features"].shape[1]
        full_indices = np.arange(dim, dtype=np.int64)
        full_acc = fit_predict_accuracy(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
            full_indices,
        )
        records.append(
            {
                "subject": int(held_subject),
                "method": "full_q1p00",
                "family": "full",
                "fraction": 1.0,
                "n_selected": int(dim),
                "accuracy": full_acc,
            }
        )
        for family, score in scores.items():
            for fraction in fractions:
                indices = top_fraction_indices(score, float(fraction))
                acc = fit_predict_accuracy(
                    payload["source_features"],
                    payload["source_labels"],
                    payload["target_features"],
                    payload["target_labels"],
                    indices,
                )
                records.append(
                    {
                        "subject": int(held_subject),
                        "method": method_name(family, float(fraction)),
                        "family": family,
                        "fraction": float(fraction),
                        "n_selected": int(len(indices)),
                        "accuracy": acc,
                    }
                )
        if not args.quiet:
            best = max(
                row for row in records if int(row["subject"]) == int(held_subject)
            )
            print(f"S{held_subject}: full={full_acc:.1f}", flush=True)

    summary = summarize(records, int(args.bootstrap), int(args.seed))
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(
        args.output_dir / "summary.csv",
        [{"method": method, **values} for method, values in sorted(summary.items())],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "subjects": subjects,
                    "fractions": fractions,
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
                "n_records": len(records),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
