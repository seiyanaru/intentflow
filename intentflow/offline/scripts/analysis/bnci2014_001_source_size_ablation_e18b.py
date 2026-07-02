"""E18b: BNCI2014_001 source-size ablation.

E18 found that compact tangent subspace selection does not clearly beat the
full tangent baseline on BNCI2014_001.  A plausible mechanism is that BNCI is a
low-dimensional, high-source-trial regime: 22 channels -> 253 tangent features
and 144 source trials.  Lee2019_MI is much more underdetermined: 1953 tangent
features and only 100 source trials.

This script tests that mechanism inside the third dataset by reducing the
number of labeled source trials available to the held subject while keeping the
zero-target-label protocol and source-side subspace scores fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bnci2014_001_fraction_sweep_e18 import (  # noqa: E402
    aggregate_stats,
    bootstrap_ci,
    fraction_token,
    load_subject_features,
)
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    fit_predict_accuracy,
    longitudinal_score,
    parse_subjects,
    source_only_score,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_E18 = RESULTS_DIR / "260628_bnci2014_001_fraction_sweep_e18"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_bnci2014_001_source_size_ablation_e18b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-9")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_E18 / "subject_cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--per-class", nargs="+", type=int, default=[8, 12, 16, 24, 36, 54, 72])
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1)
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


def method_indices(scores: Mapping[str, np.ndarray], dim: int) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {"full_q1p00": np.arange(dim, dtype=np.int64)}
    for family, fractions in {
        "source_only": [0.25, 0.50, 0.80],
        "longitudinal": [0.10, 0.35, 0.70],
    }.items():
        for fraction in fractions:
            selected[f"{family}_{fraction_token(fraction)}"] = top_fraction_indices(
                scores[family], fraction
            )
    return selected


def balanced_source_subset(
    labels: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for label in sorted(map(int, np.unique(labels))):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < int(per_class):
            raise ValueError(
                f"class {label} has only {len(candidates)} trials, need {per_class}"
            )
        chosen.append(rng.choice(candidates, size=int(per_class), replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def summarize(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    per_classes = sorted({int(row["source_per_class"]) for row in records})
    full_lookup = {
        (int(row["subject"]), int(row["repeat"]), int(row["source_per_class"])): float(
            row["accuracy"]
        )
        for row in records
        if str(row["method"]) == "full_q1p00"
    }

    output: dict[str, dict[str, object]] = {}
    for per_class in per_classes:
        for method in methods:
            rows = [
                row
                for row in records
                if int(row["source_per_class"]) == int(per_class)
                and str(row["method"]) == method
            ]
            if not rows:
                continue
            by_subject_acc: dict[int, list[float]] = {}
            by_subject_gain: dict[int, list[float]] = {}
            for row in rows:
                key = (int(row["subject"]), int(row["repeat"]), int(row["source_per_class"]))
                full = full_lookup[key]
                subject = int(row["subject"])
                by_subject_acc.setdefault(subject, []).append(float(row["accuracy"]))
                by_subject_gain.setdefault(subject, []).append(float(row["accuracy"]) - full)
            subject_acc = np.asarray(
                [np.mean(by_subject_acc[s]) for s in subjects if s in by_subject_acc],
                dtype=np.float64,
            )
            subject_gain = np.asarray(
                [np.mean(by_subject_gain[s]) for s in subjects if s in by_subject_gain],
                dtype=np.float64,
            )
            key = f"source_per_class={per_class}|{method}"
            output[key] = {
                "source_per_class": int(per_class),
                "source_total": int(2 * per_class),
                "method": method,
                "n_subjects": int(len(subject_acc)),
                "n_pairs": int(len(rows)),
                "accuracy_mean": float(subject_acc.mean()) if len(subject_acc) else float("nan"),
                "accuracy_subject_bootstrap_95ci": bootstrap_ci(
                    subject_acc, rng, int(bootstrap)
                ),
                "gain_vs_full_mean_pp": float(subject_gain.mean())
                if len(subject_gain)
                else float("nan"),
                "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(
                    subject_gain, rng, int(bootstrap)
                ),
                "gain_vs_full_q05_pp": float(np.quantile(subject_gain, 0.05))
                if len(subject_gain)
                else float("nan"),
                "p_subject_gain_lt_minus5": float(np.mean(subject_gain < -5.0))
                if len(subject_gain)
                else float("nan"),
            }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    per_classes = sorted(set(int(value) for value in args.per_class))

    payloads: dict[int, dict[str, np.ndarray]] = {}
    for subject in subjects:
        payloads[subject] = load_subject_features(
            subject,
            cache_dir=args.cache_dir,
            prefix=int(args.prefix),
            eval_start=int(args.eval_start),
            force_cache=False,
        )
        if not args.quiet:
            print(f"S{subject}: loaded", flush=True)

    records: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "source_only": source_only_score(stats),
            "longitudinal": longitudinal_score(stats),
        }
        indices_by_method = method_indices(scores, payload["source_features"].shape[1])
        labels = payload["source_labels"]
        for per_class in per_classes:
            for repeat in range(int(args.repeats)):
                rng = np.random.default_rng(
                    int(args.seed)
                    + 1000003 * int(held_subject)
                    + 1009 * int(per_class)
                    + int(repeat)
                )
                source_subset = balanced_source_subset(labels, int(per_class), rng)
                for method, indices in indices_by_method.items():
                    accuracy = fit_predict_accuracy(
                        payload["source_features"][source_subset],
                        payload["source_labels"][source_subset],
                        payload["target_features"],
                        payload["target_labels"],
                        indices,
                    )
                    records.append(
                        {
                            "subject": int(held_subject),
                            "repeat": int(repeat),
                            "source_per_class": int(per_class),
                            "source_total": int(2 * per_class),
                            "method": method,
                            "n_selected": int(len(indices)),
                            "accuracy": float(accuracy),
                        }
                    )
            if not args.quiet:
                print(f"S{held_subject}: source_per_class={per_class} done", flush=True)

    summary = summarize(records, subjects, int(args.bootstrap), int(args.seed))
    write_csv(args.output_dir / "source_size_records.csv", records)
    write_csv(
        args.output_dir / "summary.csv",
        [metrics for _, metrics in sorted(summary.items())],
    )
    report = {
        "config": {
            "dataset": "BNCI2014_001",
            "paradigm": "LeftRightImagery",
            "subjects": subjects,
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "per_class": per_classes,
            "repeats": int(args.repeats),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "methods": sorted({str(row["method"]) for row in records}),
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
