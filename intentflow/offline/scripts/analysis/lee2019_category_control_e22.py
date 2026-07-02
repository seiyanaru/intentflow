"""E22: Lee2019 category-constrained q0.25 controls.

E21 suggested that, inside the Lee2019 all-channel source-side q0.25
selection, removing selected channel pairs that are neither sensorimotor nor
posterior can improve performance.  That observation is useful only if it
beats two strong non-neuroscience controls:

1. same-size score control: keep the top-k features by the original
   source-side score, where k equals the category-constrained subset size;
2. same-size random controls: keep k random features from the q0.25 selected
   set.

If the category rule does not beat these controls, it is not yet a mechanism;
it is mostly a pruning-size artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_channel_pair_category_e21 import (  # noqa: E402
    LEE2019_CHANNELS,
    channel_masks,
)
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    bootstrap_ci,
    fit_predict_accuracy,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
    source_only_score,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260630_lee2019_category_control_e22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument(
        "--random-controls",
        nargs="+",
        choices=("keep", "drop"),
        default=["keep"],
        help=(
            "Random same-size controls to evaluate.  'keep' and 'drop' have the "
            "same marginal distribution when the retained size is fixed; keep "
            "is the default to avoid redundant LDA fits."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=22)
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


def precompute_subject_stats(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    for subject, payload in payloads.items():
        output[int(subject)] = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
    return output


def aggregate_from_precomputed(
    subject_stats: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, stats in subject_stats.items():
        if int(subject) == int(held_subject):
            continue
        for key, value in stats.items():
            if key not in total:
                total[key] = np.asarray(value, dtype=np.float64).copy()
            else:
                total[key] += np.asarray(value, dtype=np.float64)
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def subject_accuracy_map(
    records: Sequence[Mapping[str, object]],
    method: str,
    subjects: Sequence[int],
) -> dict[int, float]:
    values = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == method
    }
    return {int(subject): values[int(subject)] for subject in subjects if int(subject) in values}


def summarize_values(
    values: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "bootstrap_95ci": [float("nan"), float("nan")],
            "q05": float("nan"),
            "median": float("nan"),
            "q95": float("nan"),
        }
    return {
        "mean": float(values.mean()),
        "bootstrap_95ci": bootstrap_ci(values, rng, int(bootstrap)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def summarize_methods(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    methods = sorted({str(row["method"]) for row in records})
    full = subject_accuracy_map(records, "full_q1p00", subjects)
    output: dict[str, dict[str, object]] = {}
    for i, method in enumerate(methods):
        by_subject = subject_accuracy_map(records, method, subjects)
        common_subjects = [s for s in subjects if s in by_subject and s in full]
        acc = np.asarray([by_subject[s] for s in common_subjects], dtype=np.float64)
        gains = np.asarray([by_subject[s] - full[s] for s in common_subjects], dtype=np.float64)
        n_selected_values = [
            int(row["n_selected"]) for row in records if str(row["method"]) == method
        ]
        tail = np.sort(gains)[: max(1, int(np.ceil(0.10 * len(gains))))] if len(gains) else np.asarray([])
        gain_stats = summarize_values(gains, bootstrap, seed + i)
        acc_stats = summarize_values(acc, bootstrap, seed + 10_000 + i)
        output[method] = {
            "method": method,
            "n_subjects": int(len(common_subjects)),
            "n_selected_mean": float(np.mean(n_selected_values)) if n_selected_values else float("nan"),
            "n_selected_min": int(np.min(n_selected_values)) if n_selected_values else 0,
            "n_selected_max": int(np.max(n_selected_values)) if n_selected_values else 0,
            "accuracy_mean": acc_stats["mean"],
            "accuracy_subject_bootstrap_95ci": acc_stats["bootstrap_95ci"],
            "gain_vs_full_mean_pp": gain_stats["mean"],
            "gain_vs_full_subject_bootstrap_95ci": gain_stats["bootstrap_95ci"],
            "gain_vs_full_q05_pp": gain_stats["q05"],
            "gain_vs_full_median_pp": gain_stats["median"],
            "loss_r10_vs_full_pp": float(-tail.mean()) if len(tail) else float("nan"),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
            "n_subject_gain_pos": int(np.sum(gains > 0.0)),
            "n_subject_gain_neg": int(np.sum(gains < 0.0)),
        }
    return output


def paired_delta_rows(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    left: str,
    right: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    left_map = subject_accuracy_map(records, left, subjects)
    right_map = subject_accuracy_map(records, right, subjects)
    common_subjects = [s for s in subjects if s in left_map and s in right_map]
    deltas = np.asarray([left_map[s] - right_map[s] for s in common_subjects], dtype=np.float64)
    stats = summarize_values(deltas, bootstrap, seed)
    return {
        "contrast": f"{left}_minus_{right}",
        "left": left,
        "right": right,
        "n_subjects": int(len(common_subjects)),
        "delta_accuracy_mean_pp": stats["mean"],
        "delta_accuracy_subject_bootstrap_95ci": stats["bootstrap_95ci"],
        "delta_accuracy_q05_pp": stats["q05"],
        "delta_accuracy_median_pp": stats["median"],
        "delta_accuracy_q95_pp": stats["q95"],
        "n_delta_pos": int(np.sum(deltas > 0.0)),
        "n_delta_neg": int(np.sum(deltas < 0.0)),
    }


def summarize_random_repeats(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    full = subject_accuracy_map(records, "full_q1p00", subjects)
    by_family_repeat: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        family = str(row.get("control_family", ""))
        if family.startswith("random"):
            by_family_repeat[(family, int(row["repeat"]))].append(row)

    repeat_rows: list[dict[str, object]] = []
    collapsed_records: list[dict[str, object]] = []
    for (family, repeat), rows in sorted(by_family_repeat.items()):
        by_subject = {int(row["subject"]): float(row["accuracy"]) for row in rows}
        common_subjects = [s for s in subjects if s in by_subject and s in full]
        gains = np.asarray([by_subject[s] - full[s] for s in common_subjects], dtype=np.float64)
        acc = np.asarray([by_subject[s] for s in common_subjects], dtype=np.float64)
        tail = np.sort(gains)[: max(1, int(np.ceil(0.10 * len(gains))))] if len(gains) else np.asarray([])
        repeat_rows.append(
            {
                "control_family": family,
                "repeat": int(repeat),
                "n_subjects": int(len(common_subjects)),
                "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
                "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
                "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
                "loss_r10_vs_full_pp": float(-tail.mean()) if len(tail) else float("nan"),
                "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
            }
        )

    families = sorted({row["control_family"] for row in repeat_rows})
    family_summary_rows: list[dict[str, object]] = []
    for family in families:
        family_repeat_rows = [row for row in repeat_rows if row["control_family"] == family]
        repeat_gain_means = np.asarray(
            [float(row["gain_vs_full_mean_pp"]) for row in family_repeat_rows],
            dtype=np.float64,
        )
        repeat_acc_means = np.asarray(
            [float(row["accuracy_mean"]) for row in family_repeat_rows],
            dtype=np.float64,
        )
        repeat_loss_r10 = np.asarray(
            [float(row["loss_r10_vs_full_pp"]) for row in family_repeat_rows],
            dtype=np.float64,
        )
        family_summary_rows.append(
            {
                "control_family": family,
                "n_repeats": int(len(family_repeat_rows)),
                "repeat_accuracy_mean_mean": float(repeat_acc_means.mean()),
                "repeat_accuracy_mean_q05": float(np.quantile(repeat_acc_means, 0.05)),
                "repeat_accuracy_mean_q50": float(np.quantile(repeat_acc_means, 0.50)),
                "repeat_accuracy_mean_q95": float(np.quantile(repeat_acc_means, 0.95)),
                "repeat_gain_vs_full_mean_mean_pp": float(repeat_gain_means.mean()),
                "repeat_gain_vs_full_mean_q05_pp": float(np.quantile(repeat_gain_means, 0.05)),
                "repeat_gain_vs_full_mean_q50_pp": float(np.quantile(repeat_gain_means, 0.50)),
                "repeat_gain_vs_full_mean_q95_pp": float(np.quantile(repeat_gain_means, 0.95)),
                "repeat_loss_r10_vs_full_mean": float(repeat_loss_r10.mean()),
                "repeat_loss_r10_vs_full_q50": float(np.quantile(repeat_loss_r10, 0.50)),
            }
        )

        by_subject_values: dict[int, list[float]] = defaultdict(list)
        n_selected_by_subject: dict[int, list[int]] = defaultdict(list)
        for row in records:
            if str(row.get("control_family", "")) != family:
                continue
            by_subject_values[int(row["subject"])].append(float(row["accuracy"]))
            n_selected_by_subject[int(row["subject"])].append(int(row["n_selected"]))
        for subject in subjects:
            if subject not in by_subject_values:
                continue
            collapsed_records.append(
                {
                    "subject": int(subject),
                    "method": f"{family}_mean",
                    "control_family": family,
                    "n_selected": float(np.mean(n_selected_by_subject[subject])),
                    "accuracy": float(np.mean(by_subject_values[subject])),
                }
            )

    return repeat_rows, family_summary_rows + collapsed_records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    masks = channel_masks(LEE2019_CHANNELS)
    rng = np.random.default_rng(int(args.seed))

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

    subject_stats = precompute_subject_stats(payloads)
    if not args.quiet:
        print("source-side stats: precomputed", flush=True)

    records: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_from_precomputed(subject_stats, held_subject)
        score = source_only_score(stats)
        dim = int(payload["source_features"].shape[1])
        full_indices = np.arange(dim, dtype=np.int64)
        selected = top_fraction_indices(score, float(args.fraction))
        category_keep = selected[masks["motor_or_posterior"][selected]]
        k = int(len(category_keep))
        drop_count = int(len(selected) - k)
        if k < 2:
            raise RuntimeError(f"S{held_subject}: category_keep has too few features: {k}")

        score_top_k = top_fraction_indices(score, float(k / dim))
        if len(score_top_k) > k:
            score_top_k = score_top_k[:k]
        elif len(score_top_k) < k:
            order = np.argsort(-score)
            score_top_k = order[:k].astype(np.int64)

        deterministic = {
            "full_q1p00": full_indices,
            "source_only_q0p25_all_selected": selected,
            "category_keep_motor_or_posterior": category_keep,
            "score_top_same_k": np.asarray(score_top_k, dtype=np.int64),
        }
        for method, indices in deterministic.items():
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
                    "method": method,
                    "control_family": "deterministic",
                    "repeat": -1,
                    "n_selected": int(len(indices)),
                    "accuracy": float(acc),
                }
            )

        for repeat in range(int(args.random_repeats)):
            perm = rng.permutation(selected)
            random_specs = {}
            if "keep" in args.random_controls:
                random_keep = np.asarray(perm[:k], dtype=np.int64)
                random_specs[f"random_keep_k_r{repeat:03d}"] = ("random_keep_k", random_keep)
            if "drop" in args.random_controls:
                random_drop = np.asarray(perm[drop_count:], dtype=np.int64)
                random_specs[f"random_drop_same_count_r{repeat:03d}"] = (
                    "random_drop_same_count",
                    random_drop,
                )
            for method, (family, indices) in random_specs.items():
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
                        "method": method,
                        "control_family": family,
                        "repeat": int(repeat),
                        "n_selected": int(len(indices)),
                        "accuracy": float(acc),
                    }
                )

        selection_rows.append(
            {
                "subject": int(held_subject),
                "dim": int(dim),
                "selected_q0p25": int(len(selected)),
                "category_keep_motor_or_posterior": int(k),
                "drop_neither_motor_nor_posterior": int(drop_count),
                "category_keep_fraction_of_selected": float(k / len(selected)),
                "score_top_same_k_fraction_of_full": float(k / dim),
            }
        )
        if not args.quiet:
            print(
                f"S{held_subject}: selected={len(selected)} keep={k} drop={drop_count}",
                flush=True,
            )

    repeat_rows, random_extra_rows = summarize_random_repeats(
        records, subjects, int(args.bootstrap), int(args.seed) + 30_000
    )
    collapsed_random_records = [
        row for row in random_extra_rows if "method" in row and str(row["method"]).endswith("_mean")
    ]
    random_family_rows = [row for row in random_extra_rows if "control_family" in row and "n_repeats" in row]
    all_summary_records = [
        row
        for row in records
        if str(row.get("control_family")) == "deterministic"
    ] + collapsed_random_records
    summary = summarize_methods(all_summary_records, subjects, int(args.bootstrap), int(args.seed))

    category_gain = float(summary["category_keep_motor_or_posterior"]["gain_vs_full_mean_pp"])
    for row in random_family_rows:
        family = str(row["control_family"])
        family_gains = np.asarray(
            [
                float(repeat_row["gain_vs_full_mean_pp"])
                for repeat_row in repeat_rows
                if str(repeat_row["control_family"]) == family
            ],
            dtype=np.float64,
        )
        row["category_gain_percentile_among_repeats"] = float(np.mean(family_gains <= category_gain))
        row["category_minus_repeat_mean_gain_mean_pp"] = float(category_gain - family_gains.mean())

    requested_contrasts = [
        (
            "category_keep_motor_or_posterior",
            "source_only_q0p25_all_selected",
            int(args.seed) + 40_001,
        ),
        (
            "category_keep_motor_or_posterior",
            "score_top_same_k",
            int(args.seed) + 40_002,
        ),
        (
            "category_keep_motor_or_posterior",
            "random_keep_k_mean",
            int(args.seed) + 40_003,
        ),
        (
            "category_keep_motor_or_posterior",
            "random_drop_same_count_mean",
            int(args.seed) + 40_004,
        ),
    ]
    available_methods = {str(row["method"]) for row in all_summary_records}
    contrasts = [
        paired_delta_rows(
            all_summary_records,
            subjects,
            left,
            right,
            int(args.bootstrap),
            contrast_seed,
        )
        for left, right, contrast_seed in requested_contrasts
        if left in available_methods and right in available_methods
    ]

    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "selection_size_by_subject.csv", selection_rows)
    write_csv(
        args.output_dir / "summary.csv",
        [metrics for _, metrics in sorted(summary.items())],
    )
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "random_repeat_summary.csv", repeat_rows)
    write_csv(args.output_dir / "random_family_summary.csv", random_family_rows)
    report = {
        "config": {
            "subjects": subjects,
            "fraction": float(args.fraction),
            "random_repeats": int(args.random_repeats),
            "random_controls": list(args.random_controls),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "paired_contrasts": contrasts,
        "random_family_summary": random_family_rows,
        "selection_size_by_subject": selection_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_repeat_rows": len(repeat_rows),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
