"""E24-Stieger: stability-protected category selection on broad_all60 LR.

This is the Stieger counterpart of
``lee2019_stability_protected_category_e24.py``.  It tests whether the corrected
rule

    keep motor/posterior dimensions, and protect noncanonical dimensions when
    source-subject leave-one-out stability is near-unanimous

fixes the E23 failure of hard motor/posterior pruning.

The default policy is the Stieger main policy: longitudinal_q0.10 on pure LR
broad_all60.
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

from stieger_category_control_e23 import (  # noqa: E402
    DEFAULT_STATS_DIR,
    RESULTS_DIR,
    category_label,
    channel_masks,
    fit_predict_counts,
    load_metric_stats,
    paired_contrast,
    policy_label,
    score_for_policy,
    summarize_subject_methods,
    target_session_features,
    top_k_indices,
    write_csv,
)
from stieger_longitudinal_metric_pilot import top_fraction_indices  # noqa: E402
from stieger_neuro_feature_baseline import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    load_subject,
    parse_subjects,
)
from stieger_task_cov_cache import EEG60  # noqa: E402


DEFAULT_OUTPUT = RESULTS_DIR / "260630_stieger_stability_protected_category_e24"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--condition", default="pure_lr", choices=tuple(CONDITIONS))
    parser.add_argument("--feature", default="broad_all60", choices=tuple(FEATURE_CONFIGS))
    parser.add_argument("--family", default="longitudinal", choices=("source_only", "longitudinal"))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--phis", nargs="+", type=float, default=[0.95, 0.98, 1.0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def phi_token(phi: float) -> str:
    return f"phi{phi:.2f}".replace(".", "p")


def aggregate_excluding(
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    excluded: set[int],
    condition: str,
    feature: str,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, subject_stats in all_stats.items():
        if int(subject) in excluded:
            continue
        stats = subject_stats.get(condition, {}).get(feature)
        if not stats:
            continue
        for key, value in stats.items():
            if key not in total:
                total[key] = np.asarray(value, dtype=np.float64).copy()
            else:
                total[key] += np.asarray(value, dtype=np.float64)
    if not total:
        raise RuntimeError(f"No aggregate stats for condition={condition}, feature={feature}")
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def selection_frequency(
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    held_subject: int,
    condition: str,
    feature: str,
    family: str,
    fraction: float,
    dim: int,
) -> np.ndarray:
    inner_subjects = [subject for subject in sorted(all_stats) if int(subject) != int(held_subject)]
    counts = np.zeros(int(dim), dtype=np.float64)
    for dropped_subject in inner_subjects:
        stats = aggregate_excluding(
            all_stats,
            {int(held_subject), int(dropped_subject)},
            condition,
            feature,
        )
        score = score_for_policy(family, stats)
        selected = top_fraction_indices(score, float(fraction))
        counts[selected] += 1.0
    return counts / max(1, len(inner_subjects))


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    args: argparse.Namespace,
    masks: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source_features, source_labels, target_by_session, labels_by_session, sessions = target_session_features(
        data,
        str(args.condition),
        str(args.feature),
        int(args.prefix),
        int(args.eval_start),
        int(args.min_eval_trials),
    )
    dim = int(source_features.shape[1])
    stats = aggregate_excluding(
        all_stats,
        {int(subject)},
        str(args.condition),
        str(args.feature),
    )
    score = score_for_policy(str(args.family), stats)
    label = policy_label(str(args.family), float(args.fraction))
    selected = top_fraction_indices(score, float(args.fraction))
    hard = selected[masks["motor_or_posterior"][selected]]
    freq = selection_frequency(
        all_stats,
        int(subject),
        str(args.condition),
        str(args.feature),
        str(args.family),
        float(args.fraction),
        dim,
    )

    method_indices: dict[str, np.ndarray] = {
        "full_q1p00": np.arange(dim, dtype=np.int64),
        f"{label}__all_selected": selected,
        f"{label}__hard_category": hard,
        f"{label}__score_top_k_hard": top_k_indices(score, len(hard)),
    }
    for phi in sorted(set(float(value) for value in args.phis)):
        token = phi_token(phi)
        keep = masks["motor_or_posterior"][selected] | (freq[selected] >= phi)
        protected = selected[keep]
        method_indices[f"{label}__protected_{token}"] = protected
        method_indices[f"{label}__score_top_k_{token}"] = top_k_indices(score, len(protected))

    records: list[dict[str, object]] = []
    for method, indices in method_indices.items():
        counts_by_session = fit_predict_counts(
            source_features,
            source_labels,
            target_by_session,
            labels_by_session,
            indices,
        )
        for session in sessions:
            correct, total, acc = counts_by_session[int(session)]
            records.append(
                {
                    "subject": int(subject),
                    "session": int(session),
                    "condition": str(args.condition),
                    "feature": str(args.feature),
                    "method": method,
                    "n_selected": int(len(indices)),
                    "correct": int(correct),
                    "n_eval": int(total),
                    "accuracy": float(acc),
                }
            )

    stability_rows = [
        {
            "subject": int(subject),
            "feature_index": int(index),
            "category": category_label(int(index), masks),
            "selection_frequency": float(freq[int(index)]),
            "is_motor_or_posterior": bool(masks["motor_or_posterior"][int(index)]),
        }
        for index in selected
    ]
    size_row = {
        "subject": int(subject),
        "all_selected": int(len(selected)),
        "hard_category": int(len(hard)),
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
    return records, [size_row], stability_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_subjects = parse_subjects(args.subjects)
    all_stats = load_metric_stats(args.stats_dir, requested_subjects)
    subjects = [subject for subject in requested_subjects if subject in all_stats]
    masks = channel_masks(list(EEG60))

    records: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            data = load_subject(args.cache_dir, subject)
            subject_records, subject_size_rows, subject_stability_rows = evaluate_subject(
                subject,
                data,
                all_stats,
                args,
                masks,
            )
            records.extend(subject_records)
            size_rows.extend(subject_size_rows)
            stability_rows.extend(subject_stability_rows)
            if not args.quiet:
                latest = subject_size_rows[-1]
                print(
                    f"S{subject}: all={latest['all_selected']} hard={latest['hard_category']} "
                    + " ".join(
                        f"{phi_token(float(phi))}={latest[f'protected_{phi_token(float(phi))}']}"
                        for phi in sorted(set(float(value) for value in args.phis))
                    ),
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    summary = summarize_subject_methods(records, subjects, int(args.bootstrap), int(args.seed))
    label = policy_label(str(args.family), float(args.fraction))
    methods = {str(row["method"]) for row in records}
    contrasts: list[dict[str, object]] = []
    for left in sorted(method for method in methods if "__protected_" in method):
        token = left.rsplit("__protected_", maxsplit=1)[1]
        for right in (
            f"{label}__all_selected",
            f"{label}__hard_category",
            f"{label}__score_top_k_{token}",
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
    write_csv(args.output_dir / "selection_size_by_subject.csv", size_rows)
    write_csv(args.output_dir / "selection_stability.csv", stability_rows)
    write_csv(args.output_dir / "failures.csv", failures)
    write_csv(args.output_dir / "summary.csv", [metrics for _, metrics in sorted(summary.items())])
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    report = {
        "config": {
            "subjects": requested_subjects,
            "available_subjects": subjects,
            "condition": str(args.condition),
            "feature": str(args.feature),
            "family": str(args.family),
            "fraction": float(args.fraction),
            "phis": sorted(set(float(value) for value in args.phis)),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "paired_contrasts": contrasts,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_stability_rows": len(stability_rows),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
