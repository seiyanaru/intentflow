"""G1: Does unlabeled channel quality predict useful channel suppression?

This is a falsification audit, not a proposed adaptation method.  For each
target session it computes channel-patch quality only from the first 32
unlabeled trials, then evaluates an *offline counterfactual*: train and test a
separate broad-band Riemannian LDA after dropping that spatial patch.  The
counterfactual benefit is available only for analysis labels.

The central question is whether quality features measured without target labels
predict that benefit under leave-one-subject-out (LOSO) evaluation.  If not,
dynamic channel reliability is not a credible next method for this substrate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stieger_neuro_feature_baseline import (
    CONDITIONS,
    DEFAULT_CACHE,
    concatenate_tangent_features,
    feature_covariances,
    load_subject,
    parse_subjects,
    references_for,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260626_stieger_channel_cause_audit"
)

# Patches partition the 60 cached channels exactly once.  They are intentionally
# coarse: a cause audit needs stable spatial units, not a post-hoc search over
# 60 individual electrodes.
PATCHES = {
    "frontal": (
        "AF3", "AF4", "Fp1", "Fpz", "Fp2", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
    ),
    "left_sensorimotor": ("FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3", "CP5"),
    "right_sensorimotor": ("FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6"),
    "midline_sensorimotor": ("FCz", "Cz", "CPz"),
    "left_posterior": ("FT7", "T7", "TP7", "P1", "P3", "P5", "P7", "PO3", "PO5", "PO7"),
    "right_posterior": ("FT8", "T8", "TP8", "P2", "P4", "P6", "P8", "PO4", "PO6", "PO8"),
    "midline_posterior": ("Pz", "POz", "Oz"),
    "occipital": ("O1", "O2"),
}
QUALITY_COLUMNS = (
    "q_log_variance_shift",
    "q_trial_log_variance_instability",
    "q_spatial_correlation_shift",
    "q_band_ratio_inconsistency",
)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patch_indices(channels: np.ndarray) -> dict[str, np.ndarray]:
    names = [str(value) for value in channels.tolist()]
    expected = set(names)
    declared = [channel for patch in PATCHES.values() for channel in patch]
    if set(declared) != expected or len(declared) != len(set(declared)):
        missing = sorted(expected - set(declared))
        extra = sorted(set(declared) - expected)
        raise RuntimeError(f"PATCHES must partition channels; missing={missing}, extra={extra}")
    return {
        patch: np.asarray([names.index(channel) for channel in patch_channels], dtype=np.int64)
        for patch, patch_channels in PATCHES.items()
    }


def subselect(covariances: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return covariances[:, indices][:, :, indices]


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.clip(np.diag(covariance), 1e-12, None)
    scale = np.sqrt(np.outer(diagonal, diagonal))
    return covariance / scale


def patch_quality(
    source_covariances: Mapping[str, np.ndarray],
    target_covariances: Mapping[str, np.ndarray],
    source_indices: np.ndarray,
    prefix_indices: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    source_broad = source_covariances["broad_8_30"][source_indices].mean(axis=0)
    target_broad_trials = target_covariances["broad_8_30"][prefix_indices]
    target_broad = target_broad_trials.mean(axis=0)
    source_variance = np.clip(np.diag(source_broad), 1e-12, None)
    target_variance = np.clip(np.diag(target_broad), 1e-12, None)
    log_ratio = np.log(target_variance / source_variance)
    trial_log_variance = np.log(
        np.clip(np.diagonal(target_broad_trials, axis1=1, axis2=2), 1e-12, None)
    )
    correlation_shift = np.abs(
        correlation_matrix(target_broad) - correlation_matrix(source_broad)
    )
    band_log_ratios = []
    for band in ("mu_8_13", "low_beta_13_20", "high_beta_20_30"):
        source_mean = source_covariances[band][source_indices].mean(axis=0)
        target_mean = target_covariances[band][prefix_indices].mean(axis=0)
        band_log_ratios.append(
            np.log(
                np.clip(np.diag(target_mean), 1e-12, None)
                / np.clip(np.diag(source_mean), 1e-12, None)
            )
        )
    return {
        "q_log_variance_shift": float(np.mean(np.abs(log_ratio[indices]))),
        "q_trial_log_variance_instability": float(
            np.mean(np.std(trial_log_variance[:, indices], axis=0))
        ),
        "q_spatial_correlation_shift": float(np.mean(correlation_shift[indices, :])),
        "q_band_ratio_inconsistency": float(
            np.mean(np.std(np.asarray(band_log_ratios)[:, indices], axis=0))
        ),
    }


def fit_source_classifier(
    covariances: Mapping[str, np.ndarray],
    source_indices: np.ndarray,
    source_labels: np.ndarray,
):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    references = references_for(covariances, ("broad_8_30",), source_indices)
    features = concatenate_tangent_features(
        covariances, ("broad_8_30",), source_indices, references
    )
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        features, source_labels
    )


def target_accuracy(
    classifier,
    covariances: Mapping[str, np.ndarray],
    prefix_indices: np.ndarray,
    eval_indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, int]:
    references = references_for(covariances, ("broad_8_30",), prefix_indices)
    features = concatenate_tangent_features(
        covariances, ("broad_8_30",), eval_indices, references
    )
    predictions = classifier.predict(features)
    correct = int(np.sum(predictions == labels))
    return float(100.0 * correct / len(labels)), correct


def evaluate_subject_condition(
    subject: int,
    condition: str,
    specification: Mapping[str, tuple[int, ...]],
    data: Mapping[str, np.ndarray],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
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

    all_covariances = {
        band: data[f"cov_{band}"].astype(np.float64)
        for band in ("broad_8_30", "mu_8_13", "low_beta_13_20", "high_beta_20_30")
    }
    full_broad = {"broad_8_30": all_covariances["broad_8_30"]}
    full_classifier = fit_source_classifier(full_broad, source_indices, source_labels)
    patches = patch_indices(data["channels"])
    patch_models: dict[str, tuple[np.ndarray, Mapping[str, np.ndarray], object]] = {}
    all_indices = np.arange(len(data["channels"]), dtype=np.int64)
    for patch, dropped_indices in patches.items():
        keep = np.setdiff1d(all_indices, dropped_indices, assume_unique=True)
        patch_covariances = {"broad_8_30": subselect(full_broad["broad_8_30"], keep)}
        patch_models[patch] = (
            keep,
            patch_covariances,
            fit_source_classifier(patch_covariances, source_indices, source_labels),
        )

    rows: list[dict[str, object]] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        if len(target_indices) <= eval_start:
            continue
        prefix_indices = target_indices[:prefix]
        eval_indices = target_indices[eval_start:]
        eval_labels = data["target"][eval_indices]
        if len(eval_indices) < min_eval_trials:
            continue
        full_accuracy, full_correct = target_accuracy(
            full_classifier,
            full_broad,
            prefix_indices,
            eval_indices,
            eval_labels,
        )
        for patch, dropped_indices in patches.items():
            keep, patch_covariances, patch_classifier = patch_models[patch]
            patch_accuracy, patch_correct = target_accuracy(
                patch_classifier,
                patch_covariances,
                prefix_indices,
                eval_indices,
                eval_labels,
            )
            rows.append(
                {
                    "subject": int(subject),
                    "session": int(session),
                    "condition": condition,
                    "patch": patch,
                    "patch_size": int(len(dropped_indices)),
                    "source_session": source_session,
                    "n_prefix": int(prefix),
                    "n_eval": int(len(eval_indices)),
                    "full_acc": full_accuracy,
                    "full_correct": full_correct,
                    "patch_ablation_acc": patch_accuracy,
                    "patch_ablation_correct": patch_correct,
                    "ablation_benefit_pp": patch_accuracy - full_accuracy,
                    **patch_quality(
                        all_covariances,
                        all_covariances,
                        source_indices,
                        prefix_indices,
                        dropped_indices,
                    ),
                }
            )
    return rows


def load_or_build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "channel_cause_table.json"
    if table_path.exists() and not args.force:
        payload = json.loads(table_path.read_text())
        return list(payload["rows"]), list(payload.get("failures", []))
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            all_rows.extend(json.loads(output.read_text())["rows"])
            if not args.quiet:
                print(f"S{subject}: resume", flush=True)
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
                        args.prefix,
                        args.eval_start,
                        args.min_eval_trials,
                    )
                )
            output.write_text(json.dumps({"subject": subject, "rows": subject_rows}, indent=2))
            all_rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    payload = {
        "config": {
            "subjects": args.subjects,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "patches": PATCHES,
            "cache_dir": str(args.cache_dir),
        },
        "n_subjects": len({int(row["subject"]) for row in all_rows}),
        "n_rows": len(all_rows),
        "failures": failures,
        "rows": all_rows,
    }
    table_path.write_text(json.dumps(payload, indent=2))
    write_csv(args.output_dir / "channel_cause_table.csv", all_rows)
    return all_rows, failures


def one_hot(rows: Sequence[Mapping[str, object]], key: str) -> tuple[np.ndarray, list[str]]:
    values = sorted({str(row[key]) for row in rows})
    value_to_index = {value: index for index, value in enumerate(values)}
    output = np.zeros((len(rows), len(values)), dtype=np.float64)
    for index, row in enumerate(rows):
        output[index, value_to_index[str(row[key])]] = 1.0
    return output, values


def design_matrix(rows: Sequence[Mapping[str, object]], include_patch_identity: bool) -> np.ndarray:
    numeric = np.asarray(
        [[float(row[column]) for column in QUALITY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    if not include_patch_identity:
        return numeric
    patch, _ = one_hot(rows, "patch")
    return np.concatenate([numeric, patch], axis=1)


def subject_pooled_accuracy(records: Sequence[Mapping[str, object]], correct_key: str) -> float:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        grouped[int(row["subject"])].append(row)
    values = []
    for subject_rows in grouped.values():
        correct = sum(int(row[correct_key]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * correct / total)
    return float(np.mean(values))


def bootstrap_gain(records: Sequence[Mapping[str, object]], n_bootstrap: int, seed: int) -> list[float] | None:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        by_subject[int(row["subject"])].append(row)
    subjects = sorted(by_subject)
    if len(subjects) < 3:
        return None
    gains = []
    for subject in subjects:
        selected_correct = sum(int(row["selected_correct"]) for row in by_subject[subject])
        baseline_correct = sum(int(row["full_correct"]) for row in by_subject[subject])
        total = sum(int(row["n_eval"]) for row in by_subject[subject])
        gains.append(100.0 * (selected_correct - baseline_correct) / total)
    gains = np.asarray(gains, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(gains), size=(n_bootstrap, len(gains)))
    return [float(value) for value in np.quantile(gains[sampled].mean(axis=1), [0.025, 0.975])]


def fit_predict_loso(
    rows: Sequence[Mapping[str, object]],
    include_patch_identity: bool,
    alpha: float,
) -> list[dict[str, object]]:
    subjects = np.asarray([int(row["subject"]) for row in rows])
    predictions: list[dict[str, object]] = []
    for held_subject in sorted(set(subjects.tolist())):
        train_mask = subjects != held_subject
        test_mask = subjects == held_subject
        train_rows = [row for row, keep in zip(rows, train_mask) if keep]
        test_rows = [row for row, keep in zip(rows, test_mask) if keep]
        x_train = design_matrix(train_rows, include_patch_identity)
        x_test = design_matrix(test_rows, include_patch_identity)
        y_train = np.asarray([float(row["ablation_benefit_pp"]) for row in train_rows])
        scaler = StandardScaler()
        model = Ridge(alpha=alpha).fit(scaler.fit_transform(x_train), y_train)
        predicted = model.predict(scaler.transform(x_test))
        for row, value in zip(test_rows, predicted):
            predictions.append(
                {
                    **row,
                    "model": "quality_plus_patch" if include_patch_identity else "quality_only",
                    "predicted_ablation_benefit_pp": float(value),
                }
            )
    return predictions


def selector_records(predictions: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[(int(row["subject"]), int(row["session"]), str(row["condition"]))].append(row)
    selected: list[dict[str, object]] = []
    for group_rows in grouped.values():
        baseline = group_rows[0]
        chosen = max(group_rows, key=lambda row: float(row["predicted_ablation_benefit_pp"]))
        use_patch = float(chosen["predicted_ablation_benefit_pp"]) > 0.0
        selected.append(
            {
                "subject": int(baseline["subject"]),
                "session": int(baseline["session"]),
                "condition": str(baseline["condition"]),
                "n_eval": int(baseline["n_eval"]),
                "full_correct": int(baseline["full_correct"]),
                "full_acc": float(baseline["full_acc"]),
                "selected_patch": str(chosen["patch"]) if use_patch else "none",
                "selected_correct": int(chosen["patch_ablation_correct"]) if use_patch else int(baseline["full_correct"]),
                "selected_acc": float(chosen["patch_ablation_acc"]) if use_patch else float(baseline["full_acc"]),
                "oracle_acc": max([float(baseline["full_acc"])] + [float(row["patch_ablation_acc"]) for row in group_rows]),
            }
        )
    return selected


def evaluate_predictions(
    predictions: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    actual = np.asarray([float(row["ablation_benefit_pp"]) for row in predictions])
    predicted = np.asarray([float(row["predicted_ablation_benefit_pp"]) for row in predictions])
    positive = actual > 1.0
    auc = None
    if len(np.unique(positive)) == 2:
        auc = float(roc_auc_score(positive.astype(int), predicted))
    selected = selector_records(predictions)
    full_accuracy = subject_pooled_accuracy(selected, "full_correct")
    selected_accuracy = subject_pooled_accuracy(selected, "selected_correct")
    oracle_by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        oracle_by_subject[int(row["subject"])].append(row)
    oracle_accuracy = float(
        np.mean(
            [
                np.average(
                    [float(row["oracle_acc"]) for row in subject_rows],
                    weights=[int(row["n_eval"]) for row in subject_rows],
                )
                for subject_rows in oracle_by_subject.values()
            ]
        )
    )
    return {
        "n_rows": len(predictions),
        "n_subjects": len({int(row["subject"]) for row in predictions}),
        "benefit_prediction_spearman": float(spearmanr(predicted, actual).statistic),
        "benefit_gt_1pp_auroc": auc,
        "full_accuracy_subject_pooled_mean": full_accuracy,
        "selector_accuracy_subject_pooled_mean": selected_accuracy,
        "selector_gain_vs_full_pp": selected_accuracy - full_accuracy,
        "selector_gain_subject_bootstrap_95ci": bootstrap_gain(selected, bootstrap, seed),
        "oracle_accuracy_subject_pooled_mean": oracle_accuracy,
        "oracle_gain_vs_full_pp": oracle_accuracy - full_accuracy,
        "selected_patch_counts": dict(Counter(row["selected_patch"] for row in selected)),
    }


def summarize(rows: Sequence[Mapping[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    summaries: dict[str, object] = {}
    all_predictions: list[dict[str, object]] = []
    for condition in list(CONDITIONS) + ["primary_pooled"]:
        scoped = [
            row
            for row in rows
            if row["condition"] in PRIMARY_CONDITIONS
            if condition == "primary_pooled"
        ] if condition == "primary_pooled" else [row for row in rows if row["condition"] == condition]
        if not scoped:
            continue
        condition_summary: dict[str, object] = {}
        for include_identity in (False, True):
            predictions = fit_predict_loso(scoped, include_identity, args.ridge_alpha)
            condition_summary[
                "quality_plus_patch" if include_identity else "quality_only"
            ] = evaluate_predictions(predictions, args.bootstrap, args.seed + int(include_identity))
            all_predictions.extend(predictions)
        summaries[condition] = condition_summary
    return summaries, all_predictions


def main() -> None:
    args = parse_args()
    if args.prefix > args.eval_start:
        raise ValueError("prefix must not exceed eval-start")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = load_or_build_rows(args)
    summaries, predictions = summarize(rows, args)
    write_csv(args.output_dir / "meta_predictions.csv", predictions)
    report = {
        "config": {
            "subjects": args.subjects,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "ridge_alpha": args.ridge_alpha,
            "patches": PATCHES,
            "quality_columns": QUALITY_COLUMNS,
            "outer": "leave-one-subject-out",
            "counterfactual": "retrain source LDA and target prefix EA after dropping one patch",
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
