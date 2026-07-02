"""V1/V2 ablations for source-side longitudinal feature selection.

This script stress-tests the E5b result:

* V1: random top-k features.  If random top-k matches the proposed score, the
  result is just dimensionality reduction.
* V2: score ablations.  If source-only Fisher or separation-without-drift
  matches the proposed score, the longitudinal stability term is not doing
  meaningful work.

The protocol matches E5b:

* all sessions of the held-out subject are excluded from feature-score learning;
* the held-out subject trains branch-wise LDA using session 1 labels only;
* target sessions use prefix EA from the first unlabeled target trials only;
* target labels are used only for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stieger_neuro_feature_baseline import (
    CONDITIONS,
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    concatenate_tangent_features,
    feature_covariances,
    load_subject,
    lower_tail_cvar,
    parse_subjects,
    references_for,
    weighted_quantile,
)
from stieger_neuro_fixed_fusion import weighted_log_probability_prediction
from stieger_longitudinal_metric_pilot import (
    aggregate_stats,
    metric_score,
    source_only_score,
    top_fraction_indices,
    unflatten_stats,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_stieger_longitudinal_metric_pilot"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_stieger_longitudinal_selection_ablation"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_FRACTIONS = (1.0, 0.25, 0.1)
DEFAULT_RANDOM_SEEDS = (0, 1, 2, 3, 4)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANDOM_SEEDS),
    )
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown = [feature for feature in args.features if feature not in FEATURE_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if args.features[0] == args.features[1]:
        raise ValueError("--features must name two different branches.")
    if not args.fractions or any(fraction <= 0 or fraction > 1 for fraction in args.fractions):
        raise ValueError("--fractions must be non-empty values in (0, 1].")
    if 1.0 not in set(float(fraction) for fraction in args.fractions):
        raise ValueError("--fractions must include 1.0.")
    if not args.random_seeds:
        raise ValueError("--random-seeds must be non-empty.")
    if not (0 < args.prefix <= args.eval_start):
        raise ValueError("Require 0 < prefix <= eval-start.")


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def candidate_name(family: str, fraction: float) -> str:
    return f"{family}__{fraction_token(fraction)}__posterior_equal__prefix_ea"


def single_name(family: str, fraction: float, feature: str) -> str:
    return f"{family}__{fraction_token(fraction)}__single__{feature}__prefix_ea"


def random_family(seed: int) -> str:
    return f"random_r{seed}"


def candidate_acc_key(candidate: str) -> str:
    return f"candidate__{candidate}__acc"


def candidate_correct_key(candidate: str) -> str:
    return f"candidate__{candidate}__correct"


def lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_metric_stats(metric_dir: Path, subjects: Sequence[int]) -> dict[int, dict]:
    all_stats: dict[int, dict] = {}
    for subject in subjects:
        path = metric_dir / "metric_stats" / f"S{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = np.load(path)
        all_stats[int(subject)] = unflatten_stats(
            {key: payload[key] for key in payload.files}
        )
    return all_stats


def sep_no_drift_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    signal = np.maximum(stats["source_between"], 0.0) + np.maximum(
        stats["target_between"], 0.0
    )
    nuisance = (
        0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    score = signal / nuisance
    return np.where(np.isfinite(score), score, 0.0)


def drift_only_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    nuisance = (
        np.maximum(stats["same_class_drift"], 0.0)
        + 0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    score = 1.0 / nuisance
    return np.where(np.isfinite(score), score, 0.0)


def target_only_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    score = np.maximum(stats["target_between"], 0.0) / (
        0.25 * np.maximum(stats["target_within"], 0.0) + 1e-8
    )
    return np.where(np.isfinite(score), score, 0.0)


SCORE_FUNCTIONS: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "longitudinal": metric_score,
    "source_only": source_only_score,
    "sep_no_drift": sep_no_drift_score,
    "drift_only": drift_only_score,
    "target_only": target_only_score,
}


def stable_seed(*values: object) -> int:
    total = 17
    for value in values:
        text = str(value)
        for char in text:
            total = (total * 131 + ord(char)) % (2**32 - 1)
    return int(total)


def random_indices(dim: int, fraction: float, seed: int, *keys: object) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(dim, dtype=np.int64)
    k = max(2, int(np.ceil(float(fraction) * dim)))
    k = min(k, dim)
    rng = np.random.default_rng(stable_seed(seed, *keys))
    return np.sort(rng.choice(dim, size=k, replace=False).astype(np.int64))


def indices_for_family(
    family: str,
    stats: Mapping[str, np.ndarray] | None,
    dim: int,
    fraction: float,
    random_seed: int | None = None,
    random_keys: Sequence[object] = (),
) -> np.ndarray:
    if family.startswith("random_r"):
        if random_seed is None:
            raise ValueError("random_seed is required for random family")
        return random_indices(dim, fraction, random_seed, *random_keys)
    if stats is None:
        return np.arange(dim, dtype=np.int64)
    score = SCORE_FUNCTIONS[family](stats)
    return top_fraction_indices(score, fraction)


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    features: Sequence[str],
    fractions: Sequence[float],
    random_seeds: Sequence[int],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
    broad, neuro = features
    rows: list[dict[str, object]] = []
    families = list(SCORE_FUNCTIONS)
    random_families = [random_family(seed) for seed in random_seeds]
    for condition, specification in CONDITIONS.items():
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_session = int(sessions[0])
        source_indices = np.flatnonzero(mask & (data["session"] == source_session))
        source_labels = data["target"][source_indices]
        if len(np.unique(source_labels)) < 2:
            continue

        branch_data: dict[str, dict[str, object]] = {}
        selectors: dict[tuple[str, float, str], np.ndarray] = {}
        for feature in features:
            config = FEATURE_CONFIGS[feature]
            bands = tuple(str(band) for band in config["bands"])
            covariances = feature_covariances(data, config)
            source_references = references_for(covariances, bands, source_indices)
            source_features = concatenate_tangent_features(
                covariances, bands, source_indices, source_references
            )
            branch_data[feature] = {
                "bands": bands,
                "covariances": covariances,
                "source_features": source_features,
            }
            stats = aggregate_stats(all_stats, subject, condition, feature)
            for fraction in fractions:
                for family in families:
                    selectors[(family, float(fraction), feature)] = indices_for_family(
                        family,
                        stats,
                        source_features.shape[1],
                        float(fraction),
                    )
                for seed, family in zip(random_seeds, random_families):
                    selectors[(family, float(fraction), feature)] = indices_for_family(
                        family,
                        stats,
                        source_features.shape[1],
                        float(fraction),
                        random_seed=int(seed),
                        random_keys=(subject, condition, feature, fraction),
                    )

        model_families = families + random_families
        models: dict[tuple[str, float, str], LinearDiscriminantAnalysis] = {}
        for feature in features:
            source_features = branch_data[feature]["source_features"]  # type: ignore[assignment]
            for fraction in fractions:
                for family in model_families:
                    index = selectors[(family, float(fraction), feature)]
                    models[(family, float(fraction), feature)] = lda().fit(
                        source_features[:, index],  # type: ignore[index]
                        source_labels,
                    )

        for session in sessions[1:]:
            target_indices = np.flatnonzero(mask & (data["session"] == session))
            if len(target_indices) <= eval_start:
                continue
            eval_indices = target_indices[eval_start:]
            if len(eval_indices) < min_eval_trials:
                continue
            prefix_indices = target_indices[:prefix]
            eval_labels = data["target"][eval_indices]
            row: dict[str, object] = {
                "subject": int(subject),
                "session": int(session),
                "condition": condition,
                "source_session": source_session,
                "n_source": int(len(source_indices)),
                "n_prefix": int(len(prefix_indices)),
                "eval_start": int(eval_start),
                "n_eval": int(len(eval_indices)),
            }
            target_features: dict[str, np.ndarray] = {}
            for feature in features:
                branch = branch_data[feature]
                prefix_references = references_for(
                    branch["covariances"],  # type: ignore[arg-type]
                    branch["bands"],  # type: ignore[arg-type]
                    prefix_indices,
                )
                target_features[feature] = concatenate_tangent_features(
                    branch["covariances"],  # type: ignore[arg-type]
                    branch["bands"],  # type: ignore[arg-type]
                    eval_indices,
                    prefix_references,
                )

            for fraction in fractions:
                for family in model_families:
                    logp: dict[str, np.ndarray] = {}
                    for feature in features:
                        index = selectors[(family, float(fraction), feature)]
                        model = models[(family, float(fraction), feature)]
                        transformed = target_features[feature][:, index]
                        predictions = model.predict(transformed)
                        candidate = single_name(family, float(fraction), feature)
                        correct = int(np.sum(predictions == eval_labels))
                        row[candidate_correct_key(candidate)] = correct
                        row[candidate_acc_key(candidate)] = float(
                            100.0 * correct / len(eval_labels)
                        )
                        logp[feature] = model.predict_log_proba(transformed)
                    broad_model = models[(family, float(fraction), broad)]
                    neuro_model = models[(family, float(fraction), neuro)]
                    if not np.array_equal(broad_model.classes_, neuro_model.classes_):
                        raise RuntimeError("Broad/neuro class order mismatch")
                    predictions = weighted_log_probability_prediction(
                        logp[broad],
                        logp[neuro],
                        broad_model.classes_,
                        broad_weight=0.5,
                    )
                    candidate = candidate_name(family, float(fraction))
                    correct = int(np.sum(predictions == eval_labels))
                    row[candidate_correct_key(candidate)] = correct
                    row[candidate_acc_key(candidate)] = float(
                        100.0 * correct / len(eval_labels)
                    )
            rows.append(row)
    return rows


def build_rows(
    args: argparse.Namespace,
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "ablation_table.json"
    if table_path.exists() and not args.force:
        payload = json.loads(table_path.read_text())
        return list(payload["rows"]), list(payload.get("failures", []))
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    fractions = sorted(set(float(fraction) for fraction in args.fractions), reverse=True)
    subjects = parse_subjects(args.subjects)
    for subject in subjects:
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            rows.extend(payload["rows"])
            if not args.quiet:
                print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows = evaluate_subject(
                subject,
                data,
                all_stats,
                args.features,
                fractions,
                sorted(set(int(seed) for seed in args.random_seeds)),
                args.prefix,
                args.eval_start,
                args.min_eval_trials,
            )
            output.write_text(json.dumps({"subject": subject, "rows": subject_rows}, indent=2))
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    table_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    write_csv(args.output_dir / "ablation_table.csv", rows)
    return rows, failures


def candidate_accuracy(rows: Sequence[Mapping[str, object]], candidate: str) -> float:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)
    values = []
    for subject_rows in by_subject.values():
        correct = sum(int(row[candidate_correct_key(candidate)]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * correct / total)
    return float(np.mean(values))


def choose_by_accuracy(scored: Mapping[str, float], preferred: str) -> str:
    maximum = max(scored.values())
    tied = [key for key, value in scored.items() if np.isclose(value, maximum)]
    if preferred in tied:
        return preferred
    return sorted(tied)[0]


def build_selection_records(
    rows: Sequence[Mapping[str, object]],
    fractions: Sequence[float],
    random_seeds: Sequence[int],
) -> list[dict[str, object]]:
    fractions = sorted(set(float(fraction) for fraction in fractions), reverse=True)
    families = list(SCORE_FUNCTIONS) + [random_family(seed) for seed in random_seeds]
    records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        subjects = sorted({int(row["subject"]) for row in condition_rows})
        for held_subject in subjects:
            train_rows = [row for row in condition_rows if int(row["subject"]) != held_subject]
            test_rows = [row for row in condition_rows if int(row["subject"]) == held_subject]
            if not train_rows or not test_rows:
                continue
            family_candidates: dict[str, list[str]] = {
                family: [candidate_name(family, fraction) for fraction in fractions]
                for family in families
            }
            outer_candidates = {
                family: choose_by_accuracy(
                    {
                        candidate: candidate_accuracy(train_rows, candidate)
                        for candidate in candidates
                    },
                    preferred=candidate_name(family, 1.0),
                )
                for family, candidates in family_candidates.items()
            }
            for row in test_rows:
                for family in families:
                    for fraction in fractions:
                        candidate = candidate_name(family, fraction)
                        records.append(
                            {
                                "subject": int(row["subject"]),
                                "session": int(row["session"]),
                                "condition": str(row["condition"]),
                                "method": f"{family}_{fraction_token(fraction)}",
                                "candidate": candidate,
                                "test_correct": int(row[candidate_correct_key(candidate)]),
                                "n_eval": int(row["n_eval"]),
                                "test_acc": float(row[candidate_acc_key(candidate)]),
                            }
                        )
                    candidate = outer_candidates[family]
                    records.append(
                        {
                            "subject": int(row["subject"]),
                            "session": int(row["session"]),
                            "condition": str(row["condition"]),
                            "method": f"outer_best_{family}",
                            "candidate": candidate,
                            "test_correct": int(row[candidate_correct_key(candidate)]),
                            "n_eval": int(row["n_eval"]),
                            "test_acc": float(row[candidate_acc_key(candidate)]),
                        }
                    )
    return records


def _group_by_subject(rows: Sequence[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject"])].append(row)
    return grouped


def paired_gains(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> np.ndarray:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    values = []
    for subject, subject_rows in _group_by_subject(selected).items():
        selected_correct = sum(int(row["test_correct"]) for row in subject_rows)
        fixed_correct = sum(
            int(fixed_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
        )
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * (selected_correct - fixed_correct) / total)
    return np.asarray(values, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(bootstrap, len(values)))
    estimates = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def session_risk(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    differences = []
    subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        differences.append(float(row["test_acc"]) - float(fixed_by_key[key]["test_acc"]))
        subjects.append(int(row["subject"]))
    values = np.asarray(differences, dtype=np.float64)
    counts = Counter(subjects)
    weights = np.asarray(
        [1.0 / (len(counts) * counts[int(subject)]) for subject in subjects],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return {
        "gain_vs_full_session_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_full_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_full_q05_pp": weighted_quantile(values, weights, 0.05),
        "p_gain_vs_full_lt_minus5": float(np.sum(weights[values < -5.0])),
    }


def summarize_method(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if not selected:
        return {}
    gains = paired_gains(selected, fixed)
    candidates = Counter(str(row["candidate"]) for row in selected)
    return {
        "n_subjects": len({int(row["subject"]) for row in selected}),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": float(
            np.mean(
                [
                    100.0
                    * sum(int(row["test_correct"]) for row in subject_rows)
                    / sum(int(row["n_eval"]) for row in subject_rows)
                    for subject_rows in _group_by_subject(selected).values()
                ]
            )
        ),
        "gain_vs_full_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, bootstrap, seed),
        "selected_candidate_counts": dict(candidates),
        "selected_candidate_rates": {
            candidate: float(count / len(selected))
            for candidate, count in sorted(candidates.items())
        },
        **session_risk(selected, fixed),
    }


def summarize_records(
    records: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    report: dict[str, object] = {}
    methods = sorted({str(row["method"]) for row in records})
    for condition in [*CONDITIONS, "primary_pooled"]:
        scoped = [
            row
            for row in records
            if (
                row["condition"] == condition
                if condition != "primary_pooled"
                else row["condition"] in PRIMARY_CONDITIONS
            )
        ]
        fixed = [row for row in scoped if row["method"] == "longitudinal_q1p00"]
        if not fixed:
            continue
        condition_report: dict[str, object] = {}
        for method in methods:
            selected = [row for row in scoped if row["method"] == method]
            condition_report[method] = summarize_method(
                selected,
                fixed,
                bootstrap,
                seed + len(method) + len(condition),
            )
        report[condition] = condition_report
    return report


def summarize_random_across_seeds(summary: Mapping[str, object], random_seeds: Sequence[int]) -> dict[str, object]:
    output: dict[str, object] = {}
    for condition, condition_summary in summary.items():
        if not isinstance(condition_summary, dict):
            continue
        condition_output: dict[str, object] = {}
        for fraction in DEFAULT_FRACTIONS:
            method_values = []
            for seed in random_seeds:
                method = f"{random_family(seed)}_{fraction_token(fraction)}"
                if method in condition_summary:
                    method_values.append(condition_summary[method])
            if not method_values:
                continue
            gains = [float(value["gain_vs_full_subject_pooled_pp"]) for value in method_values]
            accs = [float(value["test_accuracy_subject_pooled_mean"]) for value in method_values]
            condition_output[f"random_{fraction_token(fraction)}"] = {
                "n_random_seeds": len(method_values),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_min": float(np.min(accs)),
                "accuracy_max": float(np.max(accs)),
                "gain_vs_full_mean": float(np.mean(gains)),
                "gain_vs_full_min": float(np.min(gains)),
                "gain_vs_full_max": float(np.max(gains)),
            }
        output[condition] = condition_output
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    all_stats = load_metric_stats(args.metric_dir, subjects)
    rows, failures = build_rows(args, all_stats)
    fractions = sorted(set(float(fraction) for fraction in args.fractions), reverse=True)
    random_seeds = sorted(set(int(seed) for seed in args.random_seeds))
    records = build_selection_records(rows, fractions, random_seeds)
    summary = summarize_records(records, args.bootstrap, args.seed)
    random_summary = summarize_random_across_seeds(summary, random_seeds)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "fractions": fractions,
            "random_seeds": random_seeds,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "cache_dir": str(args.cache_dir),
            "metric_dir": str(args.metric_dir),
            "primary_conditions": list(PRIMARY_CONDITIONS),
            "comparator": "longitudinal_q1p00, equivalent to E4a equal posterior + prefix-EA",
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "summaries": summary,
        "random_summary": random_summary,
    }
    (args.output_dir / "selection_records.json").write_text(
        json.dumps({"records": records}, indent=2)
    )
    write_csv(args.output_dir / "selection_records.csv", records)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    if args.quiet:
        print(
            json.dumps(
                {
                    "n_subjects": report["n_subjects"],
                    "n_rows": report["n_rows"],
                    "n_failures": len(failures),
                    "summary_json": str(args.output_dir / "summary.json"),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
