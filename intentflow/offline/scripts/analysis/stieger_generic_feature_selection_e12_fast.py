"""E12-fast: generic feature-selection baselines on Stieger2021.

This audit tests whether the Stieger compact-subspace gains from E9/E10 can be
explained by simple within-subject source-session feature selection.

It reuses E9 records for the existing source-side candidates and computes only
generic held-subject source-session selectors:

* self_source_fisher: univariate Fisher score from the held subject's source session;
* self_source_variance: feature variance from the held subject's source session.

The evaluation protocol matches the Stieger longitudinal-selection ablation:

* source decoder: first available session labels for the held subject;
* target decoder reference: prefix EA from the first unlabeled target trials;
* two branches: broad_all60 and fb_sensorimotor21_mu_beta;
* branch fusion: equal posterior/log-probability fusion;
* target labels are used only for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

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


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_E9_RECORDS = RESULTS_DIR / "260628_stieger_fraction_sweep_e9_full" / "selection_records.json"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e12_stieger_generic_feature_selection_fast"
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")
REFERENCE_METHODS = (
    "longitudinal_q1p00",
    "longitudinal_q0p05",
    "longitudinal_q0p10",
    "longitudinal_q0p20",
    "source_only_q0p25",
    "source_only_q0p50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--e9-records", type=Path, default=DEFAULT_E9_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.25, 0.50])
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_fraction_indices(score: np.ndarray, fraction: float) -> np.ndarray:
    dim = len(score)
    if fraction >= 1.0:
        return np.arange(dim, dtype=np.int64)
    score = np.where(np.isfinite(score), score, -np.inf)
    k = max(2, int(np.ceil(float(fraction) * dim)))
    k = min(k, dim)
    return np.sort(np.argpartition(score, -k)[-k:].astype(np.int64))


def self_source_fisher_score(source_features: np.ndarray, source_labels: np.ndarray) -> np.ndarray:
    classes = sorted(map(int, np.unique(source_labels)))
    if len(classes) < 2:
        return np.zeros(source_features.shape[1], dtype=np.float64)
    centroids = []
    variances = []
    for label in classes:
        selected = source_features[source_labels == label]
        centroids.append(selected.mean(axis=0))
        variances.append(selected.var(axis=0))
    centroids = np.asarray(centroids, dtype=np.float64)
    within = np.mean(np.asarray(variances, dtype=np.float64), axis=0)
    score = np.maximum(centroids.var(axis=0), 0.0) / (0.25 * np.maximum(within, 0.0) + 1e-8)
    return np.where(np.isfinite(score), score, 0.0)


def self_source_variance_score(source_features: np.ndarray) -> np.ndarray:
    score = np.var(source_features, axis=0)
    return np.where(np.isfinite(score), score, 0.0)


def reference_records(path: Path, subjects: Sequence[int]) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    subject_set = set(int(subject) for subject in subjects)
    records: list[dict[str, object]] = []
    for row in payload["records"]:
        if int(row["subject"]) not in subject_set:
            continue
        if str(row["condition"]) not in PRIMARY_CONDITIONS:
            continue
        method = str(row["method"])
        if method not in REFERENCE_METHODS:
            continue
        if method == "longitudinal_q1p00":
            method = "full_q1p00"
        records.append(
            {
                "subject": int(row["subject"]),
                "session": int(row["session"]),
                "condition": str(row["condition"]),
                "method": method,
                "test_correct": int(row["test_correct"]),
                "n_eval": int(row["n_eval"]),
                "test_acc": float(row["test_acc"]),
                "source": "E9_reference",
            }
        )
    return records


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    features: Sequence[str],
    fractions: Sequence[float],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
    broad, neuro = features
    rows: list[dict[str, object]] = []
    for condition, specification in CONDITIONS.items():
        if condition not in PRIMARY_CONDITIONS:
            continue
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(data["target"], specification["targets"])
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
        models: dict[tuple[str, float, str], LinearDiscriminantAnalysis] = {}
        for feature in features:
            config = FEATURE_CONFIGS[feature]
            bands = tuple(str(band) for band in config["bands"])
            covariances = feature_covariances(dict(data), config)
            source_references = references_for(covariances, bands, source_indices)
            source_features = concatenate_tangent_features(covariances, bands, source_indices, source_references)
            branch_data[feature] = {
                "bands": bands,
                "covariances": covariances,
                "source_features": source_features,
            }
            scores = {
                "self_source_fisher": self_source_fisher_score(source_features, source_labels),
                "self_source_variance": self_source_variance_score(source_features),
            }
            for family, score in scores.items():
                for fraction in fractions:
                    indices = top_fraction_indices(score, float(fraction))
                    selectors[(family, float(fraction), feature)] = indices
                    models[(family, float(fraction), feature)] = lda().fit(
                        source_features[:, indices],
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
            for family in ("self_source_fisher", "self_source_variance"):
                for fraction in fractions:
                    logp: dict[str, np.ndarray] = {}
                    for feature in features:
                        indices = selectors[(family, float(fraction), feature)]
                        model = models[(family, float(fraction), feature)]
                        logp[feature] = model.predict_log_proba(target_features[feature][:, indices])
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
                    correct = int(np.sum(predictions == eval_labels))
                    rows.append(
                        {
                            "subject": int(subject),
                            "session": int(session),
                            "condition": str(condition),
                            "method": f"{family}_{fraction_token(float(fraction))}",
                            "test_correct": correct,
                            "n_eval": int(len(eval_labels)),
                            "test_acc": float(100.0 * correct / len(eval_labels)),
                            "source": "E12_fast",
                        }
                    )
    return rows


def build_generic_records(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    subjects = parse_subjects(args.subjects)
    fractions = sorted(set(float(fraction) for fraction in args.fractions))
    for subject in subjects:
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            rows.extend(payload["records"])
            if not args.quiet:
                print(f"S{subject}: resume ({len(payload['records'])} records)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows = evaluate_subject(
                subject,
                data,
                args.features,
                fractions,
                int(args.prefix),
                int(args.eval_start),
                int(args.min_eval_trials),
            )
            output.write_text(json.dumps({"subject": int(subject), "records": subject_rows}, indent=2))
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} records", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    return rows, failures


def _group_by_subject(rows: Sequence[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject"])].append(row)
    return grouped


def in_scope(row: Mapping[str, object], scope: str) -> bool:
    if scope == "primary_pooled":
        return str(row["condition"]) in PRIMARY_CONDITIONS
    return str(row["condition"]) == scope


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(int(bootstrap), len(values)))
    estimates = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def session_risk(selected: Sequence[Mapping[str, object]], full: Sequence[Mapping[str, object]]) -> dict[str, float]:
    full_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in full
    }
    values = []
    subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        if key not in full_by_key:
            continue
        values.append(float(row["test_acc"]) - float(full_by_key[key]["test_acc"]))
        subjects.append(int(row["subject"]))
    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) == 0:
        return {
            "gain_vs_full_session_mean_pp": float("nan"),
            "loss_r10_vs_full_pp": float("nan"),
            "gain_vs_full_q05_pp": float("nan"),
            "p_gain_vs_full_lt_minus5": float("nan"),
        }
    counts = Counter(subjects)
    weights = np.asarray([1.0 / (len(counts) * counts[int(subject)]) for subject in subjects], dtype=np.float64)
    weights /= weights.sum()
    return {
        "gain_vs_full_session_mean_pp": float(np.sum(weights * values_arr)),
        "loss_r10_vs_full_pp": float(-lower_tail_cvar(values_arr, weights, 0.1)),
        "gain_vs_full_q05_pp": weighted_quantile(values_arr, weights, 0.05),
        "p_gain_vs_full_lt_minus5": float(np.sum(weights[values_arr < -5.0])),
    }


def summarize_method(
    selected: Sequence[Mapping[str, object]],
    full: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if not selected:
        return {}
    full_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in full
    }
    subject_gains = []
    subject_acc = []
    for subject, subject_rows in _group_by_subject(selected).items():
        correct = sum(int(row["test_correct"]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        full_correct = sum(
            int(full_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
            if (subject, int(row["session"]), str(row["condition"])) in full_by_key
        )
        subject_acc.append(100.0 * correct / total)
        subject_gains.append(100.0 * (correct - full_correct) / total)
    gains = np.asarray(subject_gains, dtype=np.float64)
    acc = np.asarray(subject_acc, dtype=np.float64)
    return {
        "n_subjects": len({int(row["subject"]) for row in selected}),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": float(np.mean(acc)),
        "gain_vs_full_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, bootstrap, seed),
        **session_risk(selected, full),
    }


def summarize_records(records: Sequence[Mapping[str, object]], bootstrap: int, seed: int) -> dict[str, object]:
    report: dict[str, object] = {}
    methods = sorted({str(row["method"]) for row in records})
    scopes = [*PRIMARY_CONDITIONS, "primary_pooled"]
    for scope in scopes:
        scoped = [row for row in records if in_scope(row, scope)]
        full = [row for row in scoped if row["method"] == "full_q1p00"]
        if not full:
            continue
        scope_report: dict[str, object] = {}
        for method in methods:
            selected = [row for row in scoped if row["method"] == method]
            if not selected:
                continue
            scope_report[method] = summarize_method(
                selected,
                full,
                int(bootstrap),
                int(seed) + len(method) + len(scope),
            )
        report[scope] = scope_report
    return report


def paired_difference(
    records: Sequence[Mapping[str, object]],
    scope: str,
    method_a: str,
    method_b: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    scoped = [row for row in records if in_scope(row, scope)]
    a = [row for row in scoped if row["method"] == method_a]
    b = [row for row in scoped if row["method"] == method_b]
    b_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in b
    }
    values = []
    for subject, subject_rows in _group_by_subject(a).items():
        correct_a = sum(int(row["test_correct"]) for row in subject_rows)
        correct_b = sum(
            int(b_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
            if (subject, int(row["session"]), str(row["condition"])) in b_by_key
        )
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * (correct_a - correct_b) / total)
    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) == 0:
        return {"scope": scope, "method_a": method_a, "method_b": method_b}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values_arr, size=(int(bootstrap), len(values_arr)), replace=True).mean(axis=1)
    return {
        "scope": scope,
        "method_a": method_a,
        "method_b": method_b,
        "mean_diff_pp": float(values_arr.mean()),
        "bootstrap_95ci": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "p_diff_lt_0": float(np.mean(sampled < 0.0)),
        "n_subjects": int(len(values_arr)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    reference = reference_records(args.e9_records, subjects)
    generic, failures = build_generic_records(args)
    records = reference + generic
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "failures.csv", failures)
    summary = summarize_records(records, int(args.bootstrap), int(args.seed))
    summary_rows = []
    for scope, scoped in summary.items():
        if not isinstance(scoped, dict):
            continue
        for method, metrics in scoped.items():
            summary_rows.append({"scope": scope, "method": method, **metrics})
    write_csv(args.output_dir / "summary.csv", summary_rows)
    comparisons = []
    for scope in [*PRIMARY_CONDITIONS, "primary_pooled"]:
        for ref, generic_method in (
            ("longitudinal_q0p05", "self_source_variance_q0p05"),
            ("longitudinal_q0p10", "self_source_variance_q0p10"),
            ("longitudinal_q0p20", "self_source_variance_q0p20"),
            ("source_only_q0p25", "self_source_variance_q0p25"),
            ("source_only_q0p25", "self_source_fisher_q0p25"),
        ):
            comparisons.append(
                paired_difference(records, scope, ref, generic_method, int(args.bootstrap), int(args.seed))
            )
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "fractions": [float(value) for value in args.fractions],
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "min_eval_trials": int(args.min_eval_trials),
            "cache_dir": str(args.cache_dir),
            "e9_records": str(args.e9_records),
            "reference_methods": list(REFERENCE_METHODS),
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_reference_records": len(reference),
                "n_generic_records": len(generic),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
