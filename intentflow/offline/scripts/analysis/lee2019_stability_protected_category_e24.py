"""E24-Lee: stability-protected neuro-category subspace selection.

E22 showed that hard motor/posterior category pruning is not clearly better
than source-score compactness on Lee2019.  E23 then showed that the same hard
rule fails on Stieger because it deletes stable frontal/frontopolar predictors.

This script tests the corrected hypothesis:

    anatomy should be a weak prior, not a hard deletion rule.

For the Lee2019 source_only_q0.25 selected set, keep all motor/posterior
dimensions, and keep selected other_other dimensions only when they are
near-unanimously reselected under source-subject leave-one-out stability.

Target labels of the held subject are not used for selection.
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

from lee2019_channel_pair_category_e21 import (  # noqa: E402
    LEE2019_CHANNELS,
    category_label,
    channel_masks,
)
from lee2019_category_control_e22 import (  # noqa: E402
    aggregate_from_precomputed,
    precompute_subject_stats,
)
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    bootstrap_ci,
    fit_predict_accuracy,
    load_subject_features,
    parse_subjects,
    source_only_score,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260630_lee2019_stability_protected_category_e24"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--phis", nargs="+", type=float, default=[0.95, 0.98, 1.0])
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=24)
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
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phi_token(phi: float) -> str:
    return f"phi{phi:.2f}".replace(".", "p")


def aggregate_excluding(
    subject_stats: Mapping[int, Mapping[str, np.ndarray]],
    excluded: set[int],
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, stats in subject_stats.items():
        if int(subject) in excluded:
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


def selection_frequency(
    subject_stats: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
    fraction: float,
    dim: int,
) -> np.ndarray:
    inner_subjects = [subject for subject in sorted(subject_stats) if int(subject) != int(held_subject)]
    counts = np.zeros(int(dim), dtype=np.float64)
    for dropped_subject in inner_subjects:
        stats = aggregate_excluding(subject_stats, {int(held_subject), int(dropped_subject)})
        selected = top_fraction_indices(source_only_score(stats), float(fraction))
        counts[selected] += 1.0
    return counts / max(1, len(inner_subjects))


def top_k_indices(score: np.ndarray, k: int) -> np.ndarray:
    score = np.where(np.isfinite(score), score, -np.inf)
    k = max(2, min(int(k), len(score)))
    selected = np.argpartition(score, -k)[-k:]
    return np.sort(selected.astype(np.int64))


def summarize_values(values: np.ndarray, bootstrap: int, seed: int) -> dict[str, object]:
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


def subject_accuracy(records: Sequence[Mapping[str, object]], method: str) -> dict[int, float]:
    return {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == method
    }


def summarize_methods(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    methods = sorted({str(row["method"]) for row in records})
    full = subject_accuracy(records, "full_q1p00")
    output: dict[str, dict[str, object]] = {}
    for i, method in enumerate(methods):
        by_subject = subject_accuracy(records, method)
        common = [subject for subject in subjects if subject in by_subject and subject in full]
        acc = np.asarray([by_subject[subject] for subject in common], dtype=np.float64)
        gains = np.asarray([by_subject[subject] - full[subject] for subject in common], dtype=np.float64)
        n_selected = [int(row["n_selected"]) for row in records if str(row["method"]) == method]
        tail = np.sort(gains)[: max(1, int(np.ceil(0.10 * len(gains))))] if len(gains) else np.asarray([])
        acc_stats = summarize_values(acc, bootstrap, seed + 10_000 + i)
        gain_stats = summarize_values(gains, bootstrap, seed + i)
        output[method] = {
            "method": method,
            "n_subjects": int(len(common)),
            "n_selected_mean": float(np.mean(n_selected)) if n_selected else float("nan"),
            "n_selected_min": int(np.min(n_selected)) if n_selected else 0,
            "n_selected_max": int(np.max(n_selected)) if n_selected else 0,
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


def paired_contrast(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    left: str,
    right: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    left_acc = subject_accuracy(records, left)
    right_acc = subject_accuracy(records, right)
    common = [subject for subject in subjects if subject in left_acc and subject in right_acc]
    deltas = np.asarray([left_acc[subject] - right_acc[subject] for subject in common], dtype=np.float64)
    stats = summarize_values(deltas, bootstrap, seed)
    return {
        "contrast": f"{left}_minus_{right}",
        "left": left,
        "right": right,
        "n_subjects": int(len(common)),
        "delta_accuracy_mean_pp": stats["mean"],
        "delta_accuracy_subject_bootstrap_95ci": stats["bootstrap_95ci"],
        "delta_accuracy_q05_pp": stats["q05"],
        "delta_accuracy_median_pp": stats["median"],
        "delta_accuracy_q95_pp": stats["q95"],
        "n_delta_pos": int(np.sum(deltas > 0.0)),
        "n_delta_neg": int(np.sum(deltas < 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    masks = channel_masks(LEE2019_CHANNELS)

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

    records: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    selection_size_rows: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_from_precomputed(subject_stats, held_subject)
        score = source_only_score(stats)
        dim = int(payload["source_features"].shape[1])
        selected = top_fraction_indices(score, float(args.fraction))
        hard_category = selected[masks["motor_or_posterior"][selected]]
        freq = selection_frequency(subject_stats, held_subject, float(args.fraction), dim)

        method_indices: dict[str, np.ndarray] = {
            "full_q1p00": np.arange(dim, dtype=np.int64),
            "source_only_q0p25__all_selected": selected,
            "source_only_q0p25__hard_category": hard_category,
            "source_only_q0p25__score_top_k_hard": top_k_indices(score, len(hard_category)),
        }
        for phi in sorted(set(float(value) for value in args.phis)):
            token = phi_token(phi)
            keep_mask = masks["motor_or_posterior"][selected] | (freq[selected] >= phi)
            protected = selected[keep_mask]
            method_indices[f"source_only_q0p25__protected_{token}"] = protected
            method_indices[f"source_only_q0p25__score_top_k_{token}"] = top_k_indices(
                score,
                len(protected),
            )
        for index in selected:
            stability_rows.append(
                {
                    "subject": int(held_subject),
                    "feature_index": int(index),
                    "category": category_label(int(index), masks),
                    "selection_frequency": float(freq[int(index)]),
                    "is_motor_or_posterior": bool(masks["motor_or_posterior"][int(index)]),
                }
            )
        for method, indices in method_indices.items():
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
                    "n_selected": int(len(indices)),
                    "accuracy": float(acc),
                }
            )
        selection_size_rows.append(
            {
                "subject": int(held_subject),
                "all_selected": int(len(selected)),
                "hard_category": int(len(hard_category)),
                **{
                    f"protected_{phi_token(float(phi))}": int(
                        len(
                            selected[
                                masks["motor_or_posterior"][selected]
                                | (freq[selected] >= float(phi))
                            ]
                        )
                    )
                    for phi in sorted(set(float(value) for value in args.phis))
                },
            }
        )
        if not args.quiet:
            print(
                f"S{held_subject}: all={len(selected)} hard={len(hard_category)} "
                + " ".join(
                    f"{phi_token(float(phi))}={selection_size_rows[-1][f'protected_{phi_token(float(phi))}']}"
                    for phi in sorted(set(float(value) for value in args.phis))
                ),
                flush=True,
            )

    summary = summarize_methods(records, subjects, int(args.bootstrap), int(args.seed))
    methods = {str(row["method"]) for row in records}
    contrasts: list[dict[str, object]] = []
    for left in sorted(method for method in methods if "__protected_" in method):
        token = left.rsplit("__protected_", maxsplit=1)[1]
        for right in (
            "source_only_q0p25__all_selected",
            "source_only_q0p25__hard_category",
            f"source_only_q0p25__score_top_k_{token}",
        ):
            if right in methods:
                contrasts.append(
                    paired_contrast(
                        records,
                        subjects,
                        left,
                        right,
                        int(args.bootstrap),
                        int(args.seed) + 100 * len(contrasts),
                    )
                )

    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "selection_stability.csv", stability_rows)
    write_csv(args.output_dir / "selection_size_by_subject.csv", selection_size_rows)
    write_csv(args.output_dir / "summary.csv", [metrics for _, metrics in sorted(summary.items())])
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    report = {
        "config": {
            "subjects": subjects,
            "fraction": float(args.fraction),
            "phis": sorted(set(float(value) for value in args.phis)),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "paired_contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_stability_rows": len(stability_rows),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
