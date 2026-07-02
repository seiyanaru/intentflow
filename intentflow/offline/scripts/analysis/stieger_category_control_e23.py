"""E23: Stieger broad-all60 category-constrained subspace controls.

E22 found on Lee2019 that the category-constrained q0.25 rule

    source-side selection -> keep selected pairs involving sensorimotor or posterior channels

beats same-size random pruning, but does not clearly beat score-top-k.

BNCI2014_001 is a weak external validation for this category rule because its
22-channel montage is already almost entirely sensorimotor/posterior.  This
script therefore tests the fixed category hypothesis on Stieger2021
`broad_all60`, where there are many frontal/temporal/non-motor channel pairs.

The experiment is deliberately not another selector search.  Fractions are
fixed from prior evidence:

* source_only_q0p25: Lee2019 best compact source-side policy;
* longitudinal_q0p10: Stieger best compact longitudinal policy.

For each held subject, source-side statistics from other subjects define the
score, and the held subject is evaluated prospectively from session-1 source to
later-session prefix-EA target features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from stieger_longitudinal_metric_pilot import (  # noqa: E402
    aggregate_stats,
    metric_score,
    source_only_score,
    top_fraction_indices,
    unflatten_stats,
)
from stieger_neuro_feature_baseline import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    SENSORIMOTOR21,
    concatenate_tangent_features,
    feature_covariances,
    load_subject,
    parse_subjects,
    references_for,
)
from stieger_task_cov_cache import EEG60  # noqa: E402


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_STATS_DIR = (
    RESULTS_DIR / "260627_stieger_longitudinal_metric_pilot" / "metric_stats"
)
DEFAULT_OUTPUT = RESULTS_DIR / "260630_stieger_category_control_e23"

POSTERIOR_STIEGER = {
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "Pz",
    "PO3",
    "PO4",
    "PO5",
    "PO6",
    "PO7",
    "PO8",
    "POz",
    "O1",
    "O2",
    "Oz",
}

POLICIES = (
    ("source_only", 0.25),
    ("longitudinal", 0.10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--condition", default="pure_lr", choices=tuple(CONDITIONS))
    parser.add_argument("--feature", default="broad_all60", choices=tuple(FEATURE_CONFIGS))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=23)
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


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def policy_label(family: str, fraction: float) -> str:
    return f"{family}_{fraction_token(fraction)}"


def channel_masks(channels: Sequence[str]) -> dict[str, np.ndarray]:
    upper_i, upper_j = np.triu_indices(len(channels))
    ci = np.asarray([str(channels[i]) for i in upper_i])
    cj = np.asarray([str(channels[j]) for j in upper_j])
    sensorimotor = set(SENSORIMOTOR21)
    posterior = set(POSTERIOR_STIEGER)
    i_motor = np.asarray([channel in sensorimotor for channel in ci])
    j_motor = np.asarray([channel in sensorimotor for channel in cj])
    i_post = np.asarray([channel in posterior for channel in ci])
    j_post = np.asarray([channel in posterior for channel in cj])
    any_motor = i_motor | j_motor
    any_post = i_post | j_post
    return {
        "motor_or_posterior": any_motor | any_post,
        "both_sensorimotor": i_motor & j_motor,
        "any_sensorimotor": any_motor,
        "any_posterior": any_post,
        "other_other": (~any_motor) & (~any_post),
    }


def category_label(index: int, masks: Mapping[str, np.ndarray]) -> str:
    if masks["both_sensorimotor"][index]:
        return "both_sensorimotor"
    if masks["any_sensorimotor"][index] and masks["any_posterior"][index]:
        return "sensorimotor_posterior"
    if masks["any_sensorimotor"][index]:
        return "sensorimotor_other"
    if masks["any_posterior"][index]:
        return "posterior_other"
    return "other_other"


def load_metric_stats(stats_dir: Path, subjects: Sequence[int]) -> dict[int, dict[str, dict[str, dict[str, np.ndarray]]]]:
    output: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for subject in subjects:
        path = stats_dir / f"S{subject}.npz"
        if not path.exists():
            continue
        payload = np.load(path)
        output[int(subject)] = unflatten_stats({key: payload[key] for key in payload.files})
    return output


def lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def fit_predict_counts(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features_by_session: Mapping[int, np.ndarray],
    target_labels_by_session: Mapping[int, np.ndarray],
    indices: np.ndarray,
) -> dict[int, tuple[int, int, float]]:
    model = lda().fit(source_features[:, indices], source_labels)
    output: dict[int, tuple[int, int, float]] = {}
    for session, target_features in target_features_by_session.items():
        labels = target_labels_by_session[session]
        predictions = model.predict(target_features[:, indices])
        correct = int(np.sum(predictions == labels))
        total = int(len(labels))
        output[int(session)] = (correct, total, float(100.0 * correct / total))
    return output


def target_session_features(
    data: Mapping[str, np.ndarray],
    condition: str,
    feature: str,
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], list[int]]:
    specification = CONDITIONS[condition]
    config = FEATURE_CONFIGS[feature]
    bands = tuple(str(band) for band in config["bands"])
    covariances = feature_covariances(data, config)
    mask = np.isin(data["task"], specification["tasks"]) & np.isin(
        data["target"], specification["targets"]
    )
    sessions = sorted(np.unique(data["session"][mask]).tolist())
    if len(sessions) < 2:
        raise RuntimeError(f"{condition}: need at least two sessions")
    source_session = int(sessions[0])
    source_indices = np.flatnonzero(mask & (data["session"] == source_session))
    source_labels = data["target"][source_indices]
    if len(np.unique(source_labels)) < 2:
        raise RuntimeError(f"{condition}: source session has <2 classes")
    source_references = references_for(covariances, bands, source_indices)
    source_features = concatenate_tangent_features(
        covariances,
        bands,
        source_indices,
        source_references,
    )
    target_features_by_session: dict[int, np.ndarray] = {}
    target_labels_by_session: dict[int, np.ndarray] = {}
    kept_sessions: list[int] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        if len(target_indices) <= eval_start:
            continue
        eval_indices = target_indices[eval_start:]
        if len(eval_indices) < min_eval_trials:
            continue
        prefix_indices = target_indices[:prefix]
        if len(prefix_indices) == 0:
            continue
        prefix_references = references_for(covariances, bands, prefix_indices)
        target_features_by_session[int(session)] = concatenate_tangent_features(
            covariances,
            bands,
            eval_indices,
            prefix_references,
        )
        target_labels_by_session[int(session)] = data["target"][eval_indices]
        kept_sessions.append(int(session))
    if not kept_sessions:
        raise RuntimeError(f"{condition}: no evaluable target sessions")
    return (
        source_features,
        source_labels,
        target_features_by_session,
        target_labels_by_session,
        kept_sessions,
    )


def score_for_policy(family: str, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    if family == "source_only":
        return source_only_score(stats)
    if family == "longitudinal":
        return metric_score(stats)
    raise ValueError(f"Unknown family={family}")


def top_k_indices(score: np.ndarray, k: int) -> np.ndarray:
    score = np.where(np.isfinite(score), score, -np.inf)
    k = max(2, min(int(k), len(score)))
    selected = np.argpartition(score, -k)[-k:]
    return np.sort(selected.astype(np.int64))


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    args: argparse.Namespace,
    masks: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source_features, source_labels, target_by_session, labels_by_session, sessions = target_session_features(
        data,
        str(args.condition),
        str(args.feature),
        int(args.prefix),
        int(args.eval_start),
        int(args.min_eval_trials),
    )
    stats = aggregate_stats(all_stats, int(subject), str(args.condition), str(args.feature))
    if stats is None:
        raise RuntimeError(f"S{subject}: no aggregate stats")

    dim = int(source_features.shape[1])
    records: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []

    index_by_method: dict[str, np.ndarray] = {
        "full_q1p00": np.arange(dim, dtype=np.int64),
    }
    method_meta: dict[str, dict[str, object]] = {
        "full_q1p00": {
            "family": "full",
            "base_policy": "full",
            "control_family": "deterministic",
            "repeat": -1,
        }
    }

    for family, fraction in POLICIES:
        label = policy_label(family, fraction)
        score = score_for_policy(family, stats)
        selected = top_fraction_indices(score, fraction)
        category_keep = selected[masks["motor_or_posterior"][selected]]
        k = int(len(category_keep))
        if k < 2:
            raise RuntimeError(f"S{subject} {label}: category_keep too small: {k}")
        score_top = top_k_indices(score, k)
        specs: dict[str, tuple[np.ndarray, str, int]] = {
            f"{label}__all_selected": (selected, "deterministic", -1),
            f"{label}__category_keep_motor_or_posterior": (
                category_keep,
                "deterministic",
                -1,
            ),
            f"{label}__score_top_same_k": (score_top, "deterministic", -1),
        }
        for repeat in range(int(args.random_repeats)):
            random_indices = np.sort(rng.choice(selected, size=k, replace=False).astype(np.int64))
            specs[f"{label}__random_keep_k_r{repeat:03d}"] = (
                random_indices,
                f"{label}__random_keep_k",
                int(repeat),
            )
        for method, (indices, control_family, repeat) in specs.items():
            index_by_method[method] = indices
            method_meta[method] = {
                "family": family,
                "base_policy": label,
                "control_family": control_family,
                "repeat": int(repeat),
            }
        counter = Counter(category_label(int(index), masks) for index in selected)
        for category, count in sorted(counter.items()):
            category_rows.append(
                {
                    "subject": int(subject),
                    "base_policy": label,
                    "category": category,
                    "count": int(count),
                    "fraction": float(count / len(selected)),
                }
            )
        size_rows.append(
            {
                "subject": int(subject),
                "base_policy": label,
                "dim": int(dim),
                "all_selected": int(len(selected)),
                "category_keep_motor_or_posterior": int(k),
                "drop_other_other": int(len(selected) - k),
                "category_keep_fraction_of_selected": float(k / len(selected)),
                "score_top_same_k_fraction_of_full": float(k / dim),
            }
        )

    for method, indices in index_by_method.items():
        counts_by_session = fit_predict_counts(
            source_features,
            source_labels,
            target_by_session,
            labels_by_session,
            indices,
        )
        meta = method_meta[method]
        for session in sessions:
            correct, total, acc = counts_by_session[int(session)]
            records.append(
                {
                    "subject": int(subject),
                    "session": int(session),
                    "condition": str(args.condition),
                    "feature": str(args.feature),
                    "method": method,
                    "family": meta["family"],
                    "base_policy": meta["base_policy"],
                    "control_family": meta["control_family"],
                    "repeat": meta["repeat"],
                    "n_selected": int(len(indices)),
                    "correct": int(correct),
                    "n_eval": int(total),
                    "accuracy": float(acc),
                }
            )
    return records, size_rows, category_rows


def subject_pooled_accuracy(
    records: Sequence[Mapping[str, object]],
    method: str,
) -> dict[int, float]:
    selected = [row for row in records if str(row["method"]) == method]
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        by_subject[int(row["subject"])].append(row)
    return {
        subject: float(
            100.0
            * sum(float(row["correct"]) for row in rows)
            / sum(int(row["n_eval"]) for row in rows)
        )
        for subject, rows in by_subject.items()
    }


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(int(bootstrap), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_subject_methods(
    subject_records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    methods = sorted({str(row["method"]) for row in subject_records})
    full = subject_pooled_accuracy(subject_records, "full_q1p00")
    output: dict[str, dict[str, object]] = {}
    for i, method in enumerate(methods):
        acc_map = subject_pooled_accuracy(subject_records, method)
        common = [subject for subject in subjects if subject in acc_map and subject in full]
        acc = np.asarray([acc_map[subject] for subject in common], dtype=np.float64)
        gains = np.asarray([acc_map[subject] - full[subject] for subject in common], dtype=np.float64)
        n_selected = [
            int(row["n_selected"])
            for row in subject_records
            if str(row["method"]) == method
        ]
        tail = np.sort(gains)[: max(1, int(np.ceil(0.10 * len(gains))))] if len(gains) else np.asarray([])
        output[method] = {
            "method": method,
            "n_subjects": int(len(common)),
            "n_sessions": int(
                len([row for row in subject_records if str(row["method"]) == method])
            ),
            "n_selected_mean": float(np.mean(n_selected)) if n_selected else float("nan"),
            "n_selected_min": int(np.min(n_selected)) if n_selected else 0,
            "n_selected_max": int(np.max(n_selected)) if n_selected else 0,
            "accuracy_subject_pooled_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, bootstrap, seed + i + 1000),
            "gain_vs_full_subject_pooled_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, bootstrap, seed + i),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "gain_vs_full_median_pp": float(np.median(gains)) if len(gains) else float("nan"),
            "loss_r10_vs_full_pp": float(-tail.mean()) if len(tail) else float("nan"),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
            "n_subject_gain_pos": int(np.sum(gains > 0.0)),
            "n_subject_gain_neg": int(np.sum(gains < 0.0)),
        }
    return output


def collapse_random_records(records: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    deterministic = [
        dict(row)
        for row in records
        if str(row.get("control_family")) == "deterministic"
    ]
    groups: dict[tuple[str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        control_family = str(row.get("control_family", ""))
        if not control_family.endswith("__random_keep_k"):
            continue
        groups[(control_family, int(row["subject"]), int(row["session"]))].append(row)
    collapsed: list[dict[str, object]] = []
    for (control_family, subject, session), rows in sorted(groups.items()):
        correct_mean = float(np.mean([int(row["correct"]) for row in rows]))
        total = int(rows[0]["n_eval"])
        template = dict(rows[0])
        template["method"] = f"{control_family}_mean"
        template["repeat"] = -1
        template["correct"] = correct_mean
        template["accuracy"] = float(100.0 * correct_mean / total)
        collapsed.append(template)
    return deterministic + collapsed, collapsed


def paired_contrast(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    left: str,
    right: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    left_acc = subject_pooled_accuracy(records, left)
    right_acc = subject_pooled_accuracy(records, right)
    common = [subject for subject in subjects if subject in left_acc and subject in right_acc]
    deltas = np.asarray([left_acc[subject] - right_acc[subject] for subject in common], dtype=np.float64)
    return {
        "contrast": f"{left}_minus_{right}",
        "left": left,
        "right": right,
        "n_subjects": int(len(common)),
        "delta_accuracy_subject_pooled_mean_pp": float(deltas.mean()) if len(deltas) else float("nan"),
        "delta_accuracy_subject_bootstrap_95ci": bootstrap_ci(deltas, bootstrap, seed),
        "delta_accuracy_q05_pp": float(np.quantile(deltas, 0.05)) if len(deltas) else float("nan"),
        "delta_accuracy_median_pp": float(np.median(deltas)) if len(deltas) else float("nan"),
        "delta_accuracy_q95_pp": float(np.quantile(deltas, 0.95)) if len(deltas) else float("nan"),
        "n_delta_pos": int(np.sum(deltas > 0.0)),
        "n_delta_neg": int(np.sum(deltas < 0.0)),
    }


def random_repeat_summary(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    full = subject_pooled_accuracy(records, "full_q1p00")
    families = sorted(
        {
            str(row["control_family"])
            for row in records
            if str(row.get("control_family", "")).endswith("__random_keep_k")
        }
    )
    for family in families:
        repeats = sorted(
            {
                int(row["repeat"])
                for row in records
                if str(row.get("control_family")) == family
            }
        )
        for repeat in repeats:
            repeat_records = [
                row
                for row in records
                if str(row.get("control_family")) == family
                and int(row["repeat"]) == int(repeat)
            ]
            method = str(repeat_records[0]["method"])
            acc = subject_pooled_accuracy(repeat_records, method)
            common = [subject for subject in subjects if subject in acc and subject in full]
            gains = np.asarray([acc[subject] - full[subject] for subject in common], dtype=np.float64)
            output.append(
                {
                    "control_family": family,
                    "repeat": int(repeat),
                    "n_subjects": int(len(common)),
                    "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
                    "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
                    "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    all_stats = load_metric_stats(args.stats_dir, subjects)
    available_subjects = [subject for subject in subjects if subject in all_stats]
    if not available_subjects:
        raise RuntimeError(f"No stats loaded from {args.stats_dir}")

    channels = list(EEG60)
    masks = channel_masks(channels)
    rng = np.random.default_rng(int(args.seed))

    records: list[dict[str, object]] = []
    selection_size_rows: list[dict[str, object]] = []
    category_count_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in available_subjects:
        try:
            data = load_subject(args.cache_dir, subject)
            subject_records, size_rows, category_rows = evaluate_subject(
                subject,
                data,
                all_stats,
                args,
                masks,
                rng,
            )
            records.extend(subject_records)
            selection_size_rows.extend(size_rows)
            category_count_rows.extend(category_rows)
            if not args.quiet:
                n_sessions = len({int(row["session"]) for row in subject_records})
                print(f"S{subject}: evaluated sessions={n_sessions}", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    summary_records, collapsed_random = collapse_random_records(records)
    summary = summarize_subject_methods(
        summary_records,
        available_subjects,
        int(args.bootstrap),
        int(args.seed),
    )
    contrasts: list[dict[str, object]] = []
    methods = {str(row["method"]) for row in summary_records}
    for family, fraction in POLICIES:
        label = policy_label(family, fraction)
        category_method = f"{label}__category_keep_motor_or_posterior"
        contrast_specs = (
            f"{label}__all_selected",
            f"{label}__score_top_same_k",
            f"{label}__random_keep_k_mean",
        )
        for j, right in enumerate(contrast_specs):
            if category_method in methods and right in methods:
                contrasts.append(
                    paired_contrast(
                        summary_records,
                        available_subjects,
                        category_method,
                        right,
                        int(args.bootstrap),
                        int(args.seed) + 10_000 + 100 * len(contrasts) + j,
                    )
                )

    repeat_rows = random_repeat_summary(records, available_subjects)

    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "summary_records_with_random_mean.csv", summary_records)
    write_csv(args.output_dir / "selection_size_by_subject.csv", selection_size_rows)
    write_csv(args.output_dir / "category_counts.csv", category_count_rows)
    write_csv(args.output_dir / "failures.csv", failures)
    write_csv(args.output_dir / "summary.csv", [metrics for _, metrics in sorted(summary.items())])
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "random_repeat_summary.csv", repeat_rows)
    report = {
        "config": {
            "subjects": subjects,
            "available_subjects": available_subjects,
            "condition": str(args.condition),
            "feature": str(args.feature),
            "policies": [(family, fraction) for family, fraction in POLICIES],
            "random_repeats": int(args.random_repeats),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "sensorimotor": list(SENSORIMOTOR21),
            "posterior": sorted(POSTERIOR_STIEGER),
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
                "n_summary_records": len(summary_records),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
