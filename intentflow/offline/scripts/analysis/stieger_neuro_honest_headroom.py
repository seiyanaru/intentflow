"""E3: independent-block test of broad/neuro branch headroom.

E1b chose the best branch on the same suffix on which it reported accuracy.
That is useful as a descriptive oracle, but it can substantially overestimate
deployable headroom.  This script instead fixes a causal three-block protocol:

* trials 1--32: unlabeled target prefix used only to estimate EA references;
* trials 33--64: optional *labelled* calibration block for choosing a branch;
* trials 65 onward: independent final test block.

The script is deliberately a falsification experiment.  It does not claim a
calibration-free selector: any method using the middle block is reported with
its exact label budget.  Fixed-branch and empirical-Bayes selection decisions
are fitted in leave-one-subject-out fashion, so the held-out subject is never
used to set the default branch or the shrinkage strength.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

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
)
from stieger_neuro_feature_complementarity import (
    bootstrap_ci_by_subject,
    subject_balanced_weights,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260626_stieger_neuro_honest_headroom"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_BUDGETS = (4, 8, 16, 32)
DEFAULT_PRIOR_STRENGTHS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Exactly two prefix-EA feature branches to compare.",
    )
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--calibration-start",
        type=int,
        default=32,
        help="Zero-based start of labelled calibration trials (default: trial 33).",
    )
    parser.add_argument(
        "--calibration-end",
        type=int,
        default=64,
        help="Exclusive zero-based end of calibration trials (default: trial 64).",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=64,
        help="Zero-based start of the independent test block (default: trial 65).",
    )
    parser.add_argument("--min-test-trials", type=int, default=40)
    parser.add_argument("--label-budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--prior-strengths",
        nargs="+",
        type=float,
        default=list(DEFAULT_PRIOR_STRENGTHS),
        help="Pseudo-count strengths searched only inside outer-train subjects.",
    )
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.features) != 2:
        raise ValueError("E3 is a two-branch test; pass exactly two --features.")
    unknown = [feature for feature in args.features if feature not in FEATURE_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if not (0 < args.prefix <= args.calibration_start):
        raise ValueError("Require 0 < prefix <= calibration-start.")
    if not (args.calibration_start < args.calibration_end <= args.test_start):
        raise ValueError("Require calibration-start < calibration-end <= test-start.")
    calibration_size = args.calibration_end - args.calibration_start
    if any(budget <= 0 or budget > calibration_size for budget in args.label_budgets):
        raise ValueError(f"label budgets must be in [1, {calibration_size}].")
    if args.min_test_trials <= 0:
        raise ValueError("min-test-trials must be positive.")


def arm_name(feature: str) -> str:
    return f"{feature}__prefix_ea"


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_subject_condition(
    subject: int,
    condition: str,
    specification: Mapping[str, tuple[int, ...]],
    data: Mapping[str, np.ndarray],
    features: Sequence[str],
    prefix: int,
    calibration_start: int,
    calibration_end: int,
    test_start: int,
    min_test_trials: int,
) -> list[dict[str, object]]:
    """Build independent calibration/test outcomes for one subject and condition."""
    mask = np.isin(data["task"], specification["tasks"]) & np.isin(
        data["target"], specification["targets"]
    )
    sessions = sorted(np.unique(data["session"][mask]).tolist())
    if len(sessions) < 2:
        return []
    source_session = int(sessions[0])
    source_indices = np.flatnonzero(mask & (data["session"] == source_session))
    source_labels = data["target"][source_indices]
    if len(np.unique(source_labels)) < 2:
        return []

    models: dict[str, dict[str, object]] = {}
    for feature in features:
        config = FEATURE_CONFIGS[feature]
        bands = tuple(str(band) for band in config["bands"])
        covariances = feature_covariances(data, config)
        source_references = references_for(covariances, bands, source_indices)
        source_features = concatenate_tangent_features(
            covariances, bands, source_indices, source_references
        )
        classifier = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        ).fit(source_features, source_labels)
        models[feature] = {
            "bands": bands,
            "covariances": covariances,
            "classifier": classifier,
        }

    rows: list[dict[str, object]] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        if len(target_indices) < calibration_end:
            continue
        calibration_indices = target_indices[calibration_start:calibration_end]
        test_indices = target_indices[test_start:]
        if len(test_indices) < min_test_trials:
            continue
        calibration_labels = data["target"][calibration_indices]
        test_labels = data["target"][test_indices]
        row: dict[str, object] = {
            "subject": int(subject),
            "session": int(session),
            "condition": condition,
            "source_session": source_session,
            "n_prefix": int(prefix),
            "n_calibration": int(len(calibration_indices)),
            "n_test": int(len(test_indices)),
        }
        for feature in features:
            model = models[feature]
            target_references = references_for(
                model["covariances"], model["bands"], target_indices[:prefix]
            )
            calibration_predictions = model["classifier"].predict(
                concatenate_tangent_features(
                    model["covariances"],
                    model["bands"],
                    calibration_indices,
                    target_references,
                )
            )
            test_predictions = model["classifier"].predict(
                concatenate_tangent_features(
                    model["covariances"],
                    model["bands"],
                    test_indices,
                    target_references,
                )
            )
            arm = arm_name(feature)
            calibration_correct = (calibration_predictions == calibration_labels).astype(int)
            test_correct = (test_predictions == test_labels).astype(int)
            row[f"arm__{arm}__calibration_correct"] = calibration_correct.tolist()
            row[f"arm__{arm}__test_correct"] = int(test_correct.sum())
            row[f"arm__{arm}__test_acc"] = float(test_correct.mean() * 100.0)
        rows.append(row)
    return rows


def load_or_build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "headroom_table.json"
    if table_path.exists() and not args.force:
        payload = json.loads(table_path.read_text())
        return list(payload["rows"]), list(payload.get("failures", []))

    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            rows.extend(payload["rows"])
            if not args.quiet:
                print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows: list[dict[str, object]] = []
            for condition, specification in CONDITIONS.items():
                subject_rows.extend(
                    evaluate_subject_condition(
                        subject,
                        condition,
                        specification,
                        data,
                        args.features,
                        args.prefix,
                        args.calibration_start,
                        args.calibration_end,
                        args.test_start,
                        args.min_test_trials,
                    )
                )
            output.write_text(json.dumps({"subject": subject, "rows": subject_rows}, indent=2))
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} rows", flush=True)
        except Exception as error:  # keep long cache builds resumable
            failure = {"subject": int(subject), "error": repr(error)}
            failures.append(failure)
            print(f"S{subject}: FAIL {error!r}", flush=True)
    payload = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "prefix": args.prefix,
            "calibration_start": args.calibration_start,
            "calibration_end": args.calibration_end,
            "test_start": args.test_start,
            "min_test_trials": args.min_test_trials,
            "cache_dir": str(args.cache_dir),
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "rows": rows,
    }
    table_path.write_text(json.dumps(payload, indent=2))
    write_csv(args.output_dir / "headroom_table.csv", rows)
    return rows, failures


def subject_pooled_accuracy(rows: Sequence[Mapping[str, object]], arm: str) -> float:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)
    values = []
    for subject_rows in by_subject.values():
        if arm == "_selected_arm":
            correct = sum(
                int(row["test_correct"])
                if "test_correct" in row
                else int(
                    row[
                        f"arm__{row.get('_selected_arm', row.get('selected_arm'))}__test_correct"
                    ]
                )
                for row in subject_rows
            )
        else:
            correct = sum(int(row[f"arm__{arm}__test_correct"]) for row in subject_rows)
        total = sum(int(row["n_test"]) for row in subject_rows)
        values.append(100.0 * correct / total)
    return float(np.mean(values))


def prior_means(rows: Sequence[Mapping[str, object]], arms: Sequence[str]) -> np.ndarray:
    return np.asarray([subject_pooled_accuracy(rows, arm) / 100.0 for arm in arms])


def best_fixed_index(rows: Sequence[Mapping[str, object]], arms: Sequence[str]) -> int:
    return int(np.argmax(prior_means(rows, arms)))


def choose_index(scores: np.ndarray, fallback_index: int) -> int:
    maximum = float(np.max(scores))
    candidates = np.flatnonzero(np.isclose(scores, maximum, atol=1e-12))
    if int(fallback_index) in candidates.tolist():
        return int(fallback_index)
    return int(candidates[0])


def calibration_successes(row: Mapping[str, object], arms: Sequence[str], budget: int) -> np.ndarray:
    return np.asarray(
        [sum(row[f"arm__{arm}__calibration_correct"][:budget]) for arm in arms],
        dtype=np.float64,
    )


def choose_calibrated_arm(
    row: Mapping[str, object],
    arms: Sequence[str],
    budget: int,
    prior: np.ndarray,
    strength: float,
    fallback_index: int,
) -> int:
    successes = calibration_successes(row, arms, budget)
    if strength <= 0:
        scores = successes / float(budget)
    else:
        scores = (strength * prior + successes) / (strength + budget)
    return choose_index(scores, fallback_index)


def grouped_inner_splits(subjects: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_subjects = sorted(set(int(subject) for subject in subjects))
    if len(unique_subjects) < 2:
        return []
    splitter = GroupKFold(n_splits=min(n_splits, len(unique_subjects)))
    dummy = np.zeros(len(subjects))
    return [(train, valid) for train, valid in splitter.split(dummy, groups=subjects)]


def score_strength(
    rows: Sequence[Mapping[str, object]],
    arms: Sequence[str],
    budget: int,
    strength: float,
    inner_splits: int,
) -> float:
    subjects = np.asarray([int(row["subject"]) for row in rows])
    scores: list[float] = []
    for train_indices, valid_indices in grouped_inner_splits(subjects, inner_splits):
        train_rows = [rows[int(index)] for index in train_indices]
        valid_rows = [rows[int(index)] for index in valid_indices]
        prior = prior_means(train_rows, arms)
        fallback = int(np.argmax(prior))
        selected: list[dict[str, object]] = []
        for row in valid_rows:
            chosen = choose_calibrated_arm(
                row, arms, budget, prior, strength, fallback
            )
            selected.append(dict(row, _selected_arm=arms[chosen]))
        scores.append(
            subject_pooled_accuracy(
                selected,
                "_selected_arm",
            )
        )
    return float(np.mean(scores)) if scores else float("-inf")


def select_strength(
    rows: Sequence[Mapping[str, object]],
    arms: Sequence[str],
    budget: int,
    strengths: Sequence[float],
    inner_splits: int,
) -> float:
    scores = {
        float(strength): score_strength(rows, arms, budget, float(strength), inner_splits)
        for strength in strengths
    }
    # Prefer more shrinkage on an exact tie: it is the safer calibration rule.
    return max(scores, key=lambda strength: (scores[strength], strength))


def method_record(
    row: Mapping[str, object],
    method: str,
    budget: int,
    arm: str,
    default_arm: str,
    selected_strength: float | None,
) -> dict[str, object]:
    calibration = row[f"arm__{arm}__calibration_correct"]
    return {
        "subject": int(row["subject"]),
        "session": int(row["session"]),
        "condition": str(row["condition"]),
        "method": method,
        "label_budget": int(budget),
        "selected_arm": arm,
        "outer_default_arm": default_arm,
        "selected_prior_strength": selected_strength,
        "selected_calibration_correct": int(sum(calibration[:budget])) if budget else None,
        "selected_calibration_acc": float(np.mean(calibration[:budget]) * 100.0) if budget else None,
        "test_correct": int(row[f"arm__{arm}__test_correct"]),
        "n_test": int(row["n_test"]),
        "test_acc": float(row[f"arm__{arm}__test_acc"]),
    }


def build_selection_records(
    rows: Sequence[Mapping[str, object]],
    arms: Sequence[str],
    budgets: Sequence[int],
    prior_strengths: Sequence[float],
    inner_splits: int,
) -> list[dict[str, object]]:
    """LOSO evaluation of fixed, raw-label, and nested-EB decisions."""
    records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            continue
        subjects = np.asarray([int(row["subject"]) for row in condition_rows])
        for held_subject in sorted(set(subjects.tolist())):
            train_rows = [row for row in condition_rows if int(row["subject"]) != held_subject]
            test_rows = [row for row in condition_rows if int(row["subject"]) == held_subject]
            prior = prior_means(train_rows, arms)
            fallback = int(np.argmax(prior))
            default_arm = arms[fallback]
            selected_strengths = {
                budget: select_strength(
                    train_rows,
                    arms,
                    int(budget),
                    prior_strengths,
                    inner_splits,
                )
                for budget in budgets
            }
            for row in test_rows:
                records.append(
                    method_record(
                        row,
                        "outer_best_fixed",
                        0,
                        default_arm,
                        default_arm,
                        None,
                    )
                )
                for budget in budgets:
                    raw_index = choose_calibrated_arm(
                        row,
                        arms,
                        int(budget),
                        prior,
                        0.0,
                        fallback,
                    )
                    records.append(
                        method_record(
                            row,
                            "raw_label_selector",
                            int(budget),
                            arms[raw_index],
                            default_arm,
                            0.0,
                        )
                    )
                    strength = selected_strengths[int(budget)]
                    eb_index = choose_calibrated_arm(
                        row,
                        arms,
                        int(budget),
                        prior,
                        strength,
                        fallback,
                    )
                    records.append(
                        method_record(
                            row,
                            "nested_eb_label_selector",
                            int(budget),
                            arms[eb_index],
                            default_arm,
                            float(strength),
                        )
                    )
    return records


def per_subject_paired_gains(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> np.ndarray:
    selected_by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        if key not in fixed_by_key:
            raise RuntimeError(f"Missing fixed comparator for {key}")
        selected_by_subject[int(row["subject"])].append(row)
    gains: list[float] = []
    for subject, subject_rows in selected_by_subject.items():
        selected_correct = sum(int(row["test_correct"]) for row in subject_rows)
        fixed_correct = sum(
            int(fixed_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
        )
        total = sum(int(row["n_test"]) for row in subject_rows)
        gains.append(100.0 * (selected_correct - fixed_correct) / total)
    return np.asarray(gains, dtype=np.float64)


def paired_gain(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> float:
    return float(np.mean(per_subject_paired_gains(selected, fixed)))


def bootstrap_paired_gain(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
    n_bootstrap: int,
    seed: int,
) -> list[float] | None:
    subject_gains = per_subject_paired_gains(selected, fixed)
    if len(subject_gains) < 3:
        return None
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(subject_gains),
        size=(n_bootstrap, len(subject_gains)),
    )
    estimates = subject_gains[sampled_indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def session_risk(selected: Sequence[Mapping[str, object]], fixed: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    differences = []
    subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        baseline = fixed_by_key[key]
        differences.append(float(row["test_acc"]) - float(baseline["test_acc"]))
        subjects.append(int(row["subject"]))
    values = np.asarray(differences, dtype=np.float64)
    weights = subject_balanced_weights(subjects)
    weights = weights / weights.sum()
    return {
        "gain_vs_outer_best_fixed_session_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_outer_best_fixed_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_outer_best_fixed_q05_pp": float(
            np.sort(values)[np.searchsorted(np.cumsum(weights[np.argsort(values)]), 0.05, side="left")]
        ),
        "p_gain_vs_outer_best_fixed_lt_minus5": float(np.sum(weights[values < -5.0])),
    }


def summarize_method(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if not selected:
        return {}
    arm_counts = Counter(str(row["selected_arm"]) for row in selected)
    strengths = [row["selected_prior_strength"] for row in selected if row["selected_prior_strength"] is not None]
    return {
        "n_subjects": len({int(row["subject"]) for row in selected}),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": subject_pooled_accuracy(selected, "_selected_arm"),
        "gain_vs_outer_best_fixed_subject_pooled_pp": paired_gain(selected, fixed),
        "gain_vs_outer_best_fixed_subject_bootstrap_95ci": bootstrap_paired_gain(
            selected, fixed, bootstrap, seed
        ),
        "selected_arm_counts": dict(arm_counts),
        "selected_arm_rates": {
            arm: float(count / len(selected)) for arm, count in sorted(arm_counts.items())
        },
        "selected_prior_strength_counts": dict(Counter(str(value) for value in strengths)),
        **session_risk(selected, fixed),
    }


def summarize_records(
    records: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    report: dict[str, object] = {}
    conditions = list(CONDITIONS) + ["primary_pooled"]
    methods = sorted({(str(row["method"]), int(row["label_budget"])) for row in records})
    for condition in conditions:
        scoped = [
            row
            for row in records
            if (
                row["condition"] == condition
                if condition != "primary_pooled"
                else row["condition"] in PRIMARY_CONDITIONS
            )
        ]
        if not scoped:
            continue
        fixed = [row for row in scoped if row["method"] == "outer_best_fixed"]
        condition_report: dict[str, object] = {}
        for method, budget in methods:
            selected = [
                row
                for row in scoped
                if row["method"] == method and int(row["label_budget"]) == budget
            ]
            key = method if method == "outer_best_fixed" else f"{method}__k{budget}"
            condition_report[key] = summarize_method(
                selected, fixed, bootstrap, seed + budget + len(key)
            )
        report[condition] = condition_report
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = load_or_build_rows(args)
    arms = [arm_name(feature) for feature in args.features]
    selection_records = build_selection_records(
        rows,
        arms,
        sorted(set(args.label_budgets)),
        sorted(set(args.prior_strengths)),
        args.inner_splits,
    )
    summary = summarize_records(selection_records, args.bootstrap, args.seed)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "arms": arms,
            "prefix": args.prefix,
            "calibration_block": [args.calibration_start + 1, args.calibration_end],
            "test_block_start": args.test_start + 1,
            "min_test_trials": args.min_test_trials,
            "label_budgets": sorted(set(args.label_budgets)),
            "prior_strengths": sorted(set(args.prior_strengths)),
            "outer": "leave-one-subject-out",
            "inner": "GroupKFold on outer-train subjects",
            "primary_conditions": list(PRIMARY_CONDITIONS),
            "cache_dir": str(args.cache_dir),
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "summaries": summary,
    }
    (args.output_dir / "selection_records.json").write_text(
        json.dumps({"records": selection_records}, indent=2)
    )
    write_csv(args.output_dir / "selection_records.csv", selection_records)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
