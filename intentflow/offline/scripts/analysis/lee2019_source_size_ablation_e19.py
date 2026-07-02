"""E19: Lee2019 source-size ablation.

This tests the mechanism suggested by the BNCI2014_001 third-dataset check:
compact source-side tangent subspace selection may help mainly in
high-dimensional / source-scarce regimes.

Protocol:

* dataset/features: reuse Lee2019_MI LR subject cache from E9;
* source session: session 0;
* target session: session 1, prefix-EA reference already encoded in cache;
* held subject target labels are never used for feature ranking or fitting;
* source-side scores are learned from non-held subjects;
* only the held subject's labeled source trials are sub-sampled.

If the regime hypothesis is correct, compact policies should become more
useful as source_per_class decreases.
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

from lee2019_fraction_sweep_e9 import aggregate_stats  # noqa: E402
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    bootstrap_ci,
    fit_predict_accuracy,
    load_subject_features,
    longitudinal_score,
    parse_subjects,
    source_only_score,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260629_lee2019_source_size_ablation_e19"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument(
        "--eval-subjects",
        default=None,
        help=(
            "Subjects to evaluate. Defaults to --subjects. "
            "All --subjects are still loaded and used for non-held source-side scores."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--per-class", nargs="+", type=int, default=[8, 12, 16, 24, 36, 50])
    parser.add_argument("--source-only-fractions", nargs="+", type=float, default=[0.25, 0.50, 0.80])
    parser.add_argument("--longitudinal-fractions", nargs="+", type=float, default=[0.10, 0.20, 0.35, 0.70])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


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


def method_indices(
    scores: Mapping[str, np.ndarray],
    dim: int,
    source_only_fractions: Sequence[float],
    longitudinal_fractions: Sequence[float],
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {"full_q1p00": np.arange(dim, dtype=np.int64)}
    for family, fractions in {
        "source_only": source_only_fractions,
        "longitudinal": longitudinal_fractions,
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
                "p": int(next(row["p"] for row in rows)),
                "p_over_source_total": float(next(row["p"] for row in rows) / (2 * per_class)),
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
                "gain_vs_full_median_pp": float(np.median(subject_gain))
                if len(subject_gain)
                else float("nan"),
                "p_subject_gain_lt_minus5": float(np.mean(subject_gain < -5.0))
                if len(subject_gain)
                else float("nan"),
                "n_subject_gain_pos": int(np.sum(subject_gain > 0.0)),
                "n_subject_gain_neg": int(np.sum(subject_gain < 0.0)),
            }
    return output


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subjects = parse_subjects(args.subjects)
    eval_subjects = parse_subjects(args.eval_subjects) if args.eval_subjects else subjects
    per_classes = sorted(set(int(value) for value in args.per_class))

    payloads: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            payloads[subject] = load_subject_features(
                subject,
                cache_dir=args.cache_dir,
                prefix=int(args.prefix),
                eval_start=int(args.eval_start),
                force_cache=bool(args.force_cache),
            )
            if not args.quiet:
                payload = payloads[subject]
                print(
                    f"S{subject}: loaded dim={payload['source_features'].shape[1]} "
                    f"n_source={len(payload['source_labels'])} n_eval={len(payload['target_labels'])}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "load", "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    available_subjects = sorted(payloads)
    missing_eval = [subject for subject in eval_subjects if subject not in payloads]
    if missing_eval:
        raise ValueError(f"--eval-subjects must be a subset of --subjects, missing {missing_eval}")
    records: list[dict[str, object]] = []
    for held_subject in eval_subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "source_only": source_only_score(stats),
            "longitudinal": longitudinal_score(stats),
        }
        dim = int(payload["source_features"].shape[1])
        indices_by_method = method_indices(
            scores,
            dim,
            source_only_fractions=args.source_only_fractions,
            longitudinal_fractions=args.longitudinal_fractions,
        )
        labels = payload["source_labels"]
        max_per_class = min(int(np.sum(labels == label)) for label in np.unique(labels))
        for per_class in per_classes:
            effective_repeats = 1 if int(per_class) >= int(max_per_class) else int(args.repeats)
            for repeat in range(effective_repeats):
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
                            "p": int(dim),
                            "method": method,
                            "n_selected": int(len(indices)),
                            "accuracy": float(accuracy),
                        }
                    )
            if not args.quiet:
                print(f"S{held_subject}: source_per_class={per_class} done", flush=True)
        write_csv(args.output_dir / "source_size_records.partial.csv", records)

    return records, failures


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    eval_subjects = parse_subjects(args.eval_subjects) if args.eval_subjects else subjects
    records, failures = evaluate(args)
    summary = summarize(records, eval_subjects, int(args.bootstrap), int(args.seed))
    write_csv(args.output_dir / "source_size_records.csv", records)
    write_csv(args.output_dir / "failures.csv", failures)
    write_csv(
        args.output_dir / "summary.csv",
        [metrics for _, metrics in sorted(summary.items())],
    )
    report = {
        "config": {
            "dataset": "Lee2019_MI",
            "paradigm": "LeftRightImagery",
            "subjects": subjects,
            "eval_subjects": eval_subjects,
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "per_class": sorted(set(int(value) for value in args.per_class)),
            "source_only_fractions": [float(value) for value in args.source_only_fractions],
            "longitudinal_fractions": [float(value) for value in args.longitudinal_fractions],
            "repeats": int(args.repeats),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "methods": sorted({str(row["method"]) for row in records}),
        },
        "summary": summary,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
