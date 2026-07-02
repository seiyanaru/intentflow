"""E21: Lee2019 selected channel-pair category ablation.

E20 showed that restricting Lee2019 to sensorimotor20 removes most of the
all-channel source-side selection gain.  E21 asks what channel-pair categories
inside the all62 `source_only_q0p25` selection carry that advantage.

For each held subject, source-side scores are computed from the non-held
subjects exactly as in E9.  We then evaluate:

* the full tangent baseline;
* the complete selected subspace;
* category-only subsets of the selected subspace;
* leave-category-out subsets of the selected subspace.

The analysis is mechanistic; it is not a new method proposal.
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
DEFAULT_OUTPUT = RESULTS_DIR / "260630_lee2019_channel_pair_category_e21"


LEE2019_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "FC3",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CPz",
    "CP4",
    "P1",
    "P2",
    "POz",
    "FT9",
    "FTT9h",
    "TTP7h",
    "TP7",
    "TPP9h",
    "FT10",
    "FTT10h",
    "TPP8h",
    "TP8",
    "TPP10h",
    "F9",
    "F10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "PO3",
    "PO4",
]

SENSORIMOTOR = {
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
}

POSTERIOR = {
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "P1",
    "P2",
    "PO9",
    "PO10",
    "POz",
    "PO3",
    "PO4",
    "O1",
    "Oz",
    "O2",
}

METHOD_SPECS = (
    ("source_only", 0.25),
    ("longitudinal", 0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def method_label(family: str, fraction: float) -> str:
    return f"{family}_{fraction_token(fraction)}"


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


def channel_masks(channels: Sequence[str]) -> dict[str, np.ndarray]:
    upper_i, upper_j = np.triu_indices(len(channels))
    ci = np.asarray([str(channels[i]) for i in upper_i])
    cj = np.asarray([str(channels[j]) for j in upper_j])
    i_motor = np.asarray([c in SENSORIMOTOR for c in ci])
    j_motor = np.asarray([c in SENSORIMOTOR for c in cj])
    i_post = np.asarray([c in POSTERIOR for c in ci])
    j_post = np.asarray([c in POSTERIOR for c in cj])

    any_motor = i_motor | j_motor
    both_motor = i_motor & j_motor
    any_post = i_post | j_post
    both_post = i_post & j_post
    motor_post = (i_motor & j_post) | (i_post & j_motor)
    neither_motor_nor_post = (~any_motor) & (~any_post)

    return {
        "all": np.ones(len(upper_i), dtype=bool),
        "both_sensorimotor": both_motor,
        "sensorimotor_posterior": motor_post,
        "both_posterior": both_post,
        "any_sensorimotor": any_motor,
        "any_posterior": any_post,
        "no_sensorimotor": ~any_motor,
        "no_posterior": ~any_post,
        "neither_sensorimotor_nor_posterior": neither_motor_nor_post,
        "motor_or_posterior": any_motor | any_post,
        "diagonal": upper_i == upper_j,
        "offdiagonal": upper_i != upper_j,
    }


def category_label(index: int, masks: Mapping[str, np.ndarray]) -> str:
    if masks["both_sensorimotor"][index]:
        return "both_sensorimotor"
    if masks["sensorimotor_posterior"][index]:
        return "sensorimotor_posterior"
    if masks["both_posterior"][index]:
        return "both_posterior"
    if masks["any_sensorimotor"][index]:
        return "sensorimotor_other"
    if masks["any_posterior"][index]:
        return "posterior_other"
    return "other_other"


def subset_indices(
    selected: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    selected = np.asarray(selected, dtype=np.int64)
    subsets: dict[str, np.ndarray] = {
        "all_selected": selected,
        "only_both_sensorimotor": selected[masks["both_sensorimotor"][selected]],
        "only_sensorimotor_posterior": selected[masks["sensorimotor_posterior"][selected]],
        "only_both_posterior": selected[masks["both_posterior"][selected]],
        "only_any_sensorimotor": selected[masks["any_sensorimotor"][selected]],
        "only_any_posterior": selected[masks["any_posterior"][selected]],
        "only_no_sensorimotor": selected[masks["no_sensorimotor"][selected]],
        "only_motor_or_posterior": selected[masks["motor_or_posterior"][selected]],
        "drop_both_sensorimotor": selected[~masks["both_sensorimotor"][selected]],
        "drop_sensorimotor_posterior": selected[
            ~masks["sensorimotor_posterior"][selected]
        ],
        "drop_any_posterior": selected[~masks["any_posterior"][selected]],
        "drop_no_sensorimotor": selected[~masks["no_sensorimotor"][selected]],
    }
    return {key: value for key, value in subsets.items() if len(value) >= 2}


def summarize_records(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    full = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == "full_q1p00"
    }
    output: dict[str, dict[str, object]] = {}
    for method in methods:
        by_subject = {
            int(row["subject"]): float(row["accuracy"])
            for row in records
            if str(row["method"]) == method
        }
        acc = np.asarray(
            [by_subject[s] for s in subjects if s in by_subject],
            dtype=np.float64,
        )
        gains = np.asarray(
            [
                by_subject[s] - full[s]
                for s in subjects
                if s in by_subject and s in full
            ],
            dtype=np.float64,
        )
        n_selected_values = [
            int(row["n_selected"]) for row in records if str(row["method"]) == method
        ]
        output[method] = {
            "method": method,
            "n_subjects": int(len(acc)),
            "n_selected_mean": float(np.mean(n_selected_values))
            if n_selected_values
            else float("nan"),
            "n_selected_min": int(np.min(n_selected_values)) if n_selected_values else 0,
            "n_selected_max": int(np.max(n_selected_values)) if n_selected_values else 0,
            "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, int(bootstrap)),
            "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, int(bootstrap)),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "gain_vs_full_median_pp": float(np.median(gains)) if len(gains) else float("nan"),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
            "n_subject_gain_pos": int(np.sum(gains > 0.0)),
            "n_subject_gain_neg": int(np.sum(gains < 0.0)),
        }
    return output


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

    records: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "source_only": source_only_score(stats),
            "longitudinal": longitudinal_score(stats),
        }
        dim = int(payload["source_features"].shape[1])
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
                "subset": "full",
                "n_selected": int(dim),
                "accuracy": float(full_acc),
            }
        )
        for family, fraction in METHOD_SPECS:
            label = method_label(family, fraction)
            selected = top_fraction_indices(scores[family], float(fraction))
            counter = Counter(category_label(int(index), masks) for index in selected)
            for category, count in sorted(counter.items()):
                category_rows.append(
                    {
                        "subject": int(held_subject),
                        "base_method": label,
                        "category": category,
                        "count": int(count),
                        "fraction": float(count / len(selected)),
                    }
                )
            for index in selected:
                selected_rows.append(
                    {
                        "subject": int(held_subject),
                        "base_method": label,
                        "feature_index": int(index),
                        "score": float(scores[family][int(index)]),
                        "category": category_label(int(index), masks),
                    }
                )

            for subset_name, indices in subset_indices(selected, masks).items():
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
                        "method": f"{label}__{subset_name}",
                        "base_method": label,
                        "family": family,
                        "fraction": float(fraction),
                        "subset": subset_name,
                        "n_selected": int(len(indices)),
                        "accuracy": float(acc),
                    }
                )
        if not args.quiet:
            print(f"S{held_subject}: done", flush=True)

    summary = summarize_records(records, subjects, int(args.bootstrap), int(args.seed))
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "category_counts.csv", category_rows)
    write_csv(args.output_dir / "selected_feature_categories.csv", selected_rows)
    write_csv(
        args.output_dir / "summary.csv",
        [metrics for _, metrics in sorted(summary.items())],
    )
    report = {
        "config": {
            "subjects": subjects,
            "methods": [(family, fraction) for family, fraction in METHOD_SPECS],
            "sensorimotor": sorted(SENSORIMOTOR),
            "posterior": sorted(POSTERIOR),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_category_rows": len(category_rows),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
