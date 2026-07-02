"""E7: Stieger source-session depth ablation.

Question:

    Does the longitudinal same-class drift term help only when source-side
    longitudinal statistics are estimated from enough sessions?

For each held-out subject, the evaluation protocol matches E5b/V1/V2:

* held-out subject is excluded from feature-score learning;
* held-out subject trains the decoder on its first session labels only;
* each later session uses an unlabeled prefix only for EA reference;
* labels of later held-out sessions are used only for final evaluation.

The intervention is on the *source-side metric learning* only.  For each source
subject we restrict the number of sessions used to estimate the longitudinal
score:

    K=2,3,5,all

where K includes the baseline/source session, so K=2 means one source->target
transition per source subject.
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
from stieger_longitudinal_metric_pilot import (
    add_stats,
    flatten_stats,
    metric_score,
    session_metric_stats,
    source_only_score,
    top_fraction_indices,
    unflatten_stats,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_stieger_session_depth_ablation"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_DEPTHS = ("2", "3", "5", "all")
DEFAULT_FRACTIONS = (0.25, 0.10)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--depths", nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stats", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown = [feature for feature in args.features if feature not in FEATURE_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if args.features[0] == args.features[1]:
        raise ValueError("--features must name two different branches.")
    if not args.fractions or any(fraction <= 0 or fraction > 1 for fraction in args.fractions):
        raise ValueError("--fractions must be in (0, 1].")
    if not (0 < args.prefix <= args.eval_start):
        raise ValueError("Require 0 < prefix <= eval-start.")
    for depth in args.depths:
        if depth != "all" and int(depth) < 2:
            raise ValueError("Depth must be >=2 or 'all'.")


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def full_candidate() -> str:
    return "full_q1p00__posterior_equal__prefix_ea"


def candidate_name(depth: str, family: str, fraction: float) -> str:
    return f"depth_{depth}__{family}__{fraction_token(fraction)}__posterior_equal__prefix_ea"


def single_name(depth: str, family: str, fraction: float, feature: str) -> str:
    return f"depth_{depth}__{family}__{fraction_token(fraction)}__single__{feature}__prefix_ea"


def method_name(depth: str, family: str, fraction: float) -> str:
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


def lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def depth_sessions(sessions: Sequence[int], depth: str) -> list[int]:
    """Return target sessions used for source-side metric estimation."""
    later = list(sessions[1:])
    if depth == "all":
        return later
    # depth includes the source session, so use depth-1 transitions.
    return later[: max(0, int(depth) - 1)]


def compute_subject_depth_stats(
    data: Mapping[str, np.ndarray],
    features: Sequence[str],
    depths: Sequence[str],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    output: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for condition, specification in CONDITIONS.items():
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(int(value) for value in np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_session = int(sessions[0])
        source_indices = np.flatnonzero(mask & (data["session"] == source_session))
        source_labels = data["target"][source_indices]
        if len(np.unique(source_labels)) < 2:
            continue
        for feature in features:
            config = FEATURE_CONFIGS[feature]
            bands = tuple(str(band) for band in config["bands"])
            covariances = feature_covariances(data, config)
            source_references = references_for(covariances, bands, source_indices)
            source_features = concatenate_tangent_features(
                covariances, bands, source_indices, source_references
            )
            for depth in depths:
                total: dict[str, np.ndarray] = {}
                for session in depth_sessions(sessions, str(depth)):
                    target_indices = np.flatnonzero(mask & (data["session"] == session))
                    if len(target_indices) <= eval_start:
                        continue
                    eval_indices = target_indices[eval_start:]
                    if len(eval_indices) < min_eval_trials:
                        continue
                    prefix_indices = target_indices[:prefix]
                    prefix_references = references_for(covariances, bands, prefix_indices)
                    target_features = concatenate_tangent_features(
                        covariances, bands, eval_indices, prefix_references
                    )
                    stats = session_metric_stats(
                        source_features,
                        source_labels,
                        target_features,
                        data["target"][eval_indices],
                    )
                    if stats is not None:
                        add_stats(total, stats)
                if total:
                    output[str(depth)][condition][feature] = total
    return {
        depth: {condition: dict(by_feature) for condition, by_feature in by_condition.items()}
        for depth, by_condition in output.items()
    }


def stats_npz_path(output_dir: Path, subject: int) -> Path:
    return output_dir / "depth_stats" / f"S{subject}.npz"


def flatten_depth_stats(
    stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for depth, by_condition in stats.items():
        for condition, by_feature in by_condition.items():
            for feature, by_metric in by_feature.items():
                for metric, value in by_metric.items():
                    arrays[f"{depth}__{condition}__{feature}__{metric}"] = np.asarray(value)
    return arrays


def unflatten_depth_stats(
    payload: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    output: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for key, value in payload.items():
        depth, condition, feature, metric = key.split("__", maxsplit=3)
        output[depth][condition][feature][metric] = np.asarray(value, dtype=np.float64)
    return {
        depth: {
            condition: {feature: dict(metrics) for feature, metrics in by_feature.items()}
            for condition, by_feature in by_condition.items()
        }
        for depth, by_condition in output.items()
    }


def build_depth_stats(
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]], list[dict[str, object]]]:
    stats_dir = args.output_dir / "depth_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[int, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]] = {}
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        path = stats_npz_path(args.output_dir, subject)
        if path.exists() and not args.force and not args.force_stats:
            payload = np.load(path)
            all_stats[int(subject)] = unflatten_depth_stats(
                {key: payload[key] for key in payload.files}
            )
            if not args.quiet:
                print(f"S{subject}: depth stats resume", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            stats = compute_subject_depth_stats(
                data,
                args.features,
                [str(depth) for depth in args.depths],
                int(args.prefix),
                int(args.eval_start),
                int(args.min_eval_trials),
            )
            np.savez_compressed(path, **flatten_depth_stats(stats))
            all_stats[int(subject)] = stats
            if not args.quiet:
                n_items = sum(
                    len(by_feature)
                    for by_condition in stats.values()
                    for by_feature in by_condition.values()
                )
                print(f"S{subject}: depth stats {n_items} items", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "depth_stats", "error": repr(error)})
            print(f"S{subject}: depth stats FAIL {error!r}", flush=True)
    return all_stats, failures


def aggregate_stats(
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]]],
    held_subject: int,
    depth: str,
    condition: str,
    feature: str,
) -> dict[str, np.ndarray] | None:
    total: dict[str, np.ndarray] = {}
    for subject, subject_stats in all_stats.items():
        if int(subject) == int(held_subject):
            continue
        stats = subject_stats.get(str(depth), {}).get(condition, {}).get(feature)
        if not stats:
            continue
        add_stats(total, stats)
    if not total or float(total.get("count", np.asarray([0.0]))[0]) <= 0:
        return None
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def indices_from_stats(
    family: str,
    stats: Mapping[str, np.ndarray] | None,
    dim: int,
    fraction: float,
) -> np.ndarray:
    if stats is None:
        return np.arange(dim, dtype=np.int64)
    if family == "longitudinal":
        score = metric_score(stats)
    elif family == "source_only":
        score = source_only_score(stats)
    else:
        raise ValueError(f"Unknown family: {family}")
    return top_fraction_indices(score, fraction)


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]]],
    features: Sequence[str],
    depths: Sequence[str],
    fractions: Sequence[float],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
    broad, neuro = features
    rows: list[dict[str, object]] = []
    families = ("longitudinal", "source_only")
    depths = [str(depth) for depth in depths]
    fractions = [float(fraction) for fraction in fractions]
    for condition, specification in CONDITIONS.items():
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(int(value) for value in np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_session = int(sessions[0])
        source_indices = np.flatnonzero(mask & (data["session"] == source_session))
        source_labels = data["target"][source_indices]
        if len(np.unique(source_labels)) < 2:
            continue

        branch_data: dict[str, dict[str, object]] = {}
        selectors: dict[tuple[str, str, float, str], np.ndarray] = {}
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
            for depth in depths:
                for family in families:
                    stats = aggregate_stats(all_stats, subject, depth, condition, feature)
                    for fraction in fractions:
                        selectors[(depth, family, float(fraction), feature)] = indices_from_stats(
                            family,
                            stats,
                            source_features.shape[1],
                            float(fraction),
                        )

        full_models: dict[str, LinearDiscriminantAnalysis] = {}
        models: dict[tuple[str, str, float, str], LinearDiscriminantAnalysis] = {}
        for feature in features:
            source_features = branch_data[feature]["source_features"]  # type: ignore[assignment]
            full_models[feature] = lda().fit(source_features, source_labels)  # type: ignore[arg-type]
            for depth in depths:
                for family in families:
                    for fraction in fractions:
                        index = selectors[(depth, family, float(fraction), feature)]
                        models[(depth, family, float(fraction), feature)] = lda().fit(
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

            # Full baseline.
            full_logp: dict[str, np.ndarray] = {}
            for feature in features:
                full_logp[feature] = full_models[feature].predict_log_proba(target_features[feature])
            if not np.array_equal(full_models[broad].classes_, full_models[neuro].classes_):
                raise RuntimeError("Full model class order mismatch")
            predictions = weighted_log_probability_prediction(
                full_logp[broad],
                full_logp[neuro],
                full_models[broad].classes_,
                broad_weight=0.5,
            )
            correct = int(np.sum(predictions == eval_labels))
            base_row = {
                "subject": int(subject),
                "session": int(session),
                "condition": condition,
                "depth": "full",
                "method": "full_q1p00",
                "candidate": full_candidate(),
                "test_correct": correct,
                "n_eval": int(len(eval_labels)),
                "test_acc": float(100.0 * correct / len(eval_labels)),
            }
            rows.append(base_row)

            for depth in depths:
                for family in families:
                    for fraction in fractions:
                        logp: dict[str, np.ndarray] = {}
                        for feature in features:
                            index = selectors[(depth, family, float(fraction), feature)]
                            model = models[(depth, family, float(fraction), feature)]
                            transformed = target_features[feature][:, index]
                            logp[feature] = model.predict_log_proba(transformed)
                        broad_model = models[(depth, family, float(fraction), broad)]
                        neuro_model = models[(depth, family, float(fraction), neuro)]
                        if not np.array_equal(broad_model.classes_, neuro_model.classes_):
                            raise RuntimeError("Selected model class order mismatch")
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
                                "condition": condition,
                                "depth": str(depth),
                                "method": method_name(depth, family, float(fraction)),
                                "candidate": candidate_name(depth, family, float(fraction)),
                                "test_correct": correct,
                                "n_eval": int(len(eval_labels)),
                                "test_acc": float(100.0 * correct / len(eval_labels)),
                            }
                        )
    return rows


def build_eval_rows(
    args: argparse.Namespace,
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows_path = args.output_dir / "selection_records.json"
    if rows_path.exists() and not args.force and not args.force_eval:
        payload = json.loads(rows_path.read_text())
        return list(payload["records"]), list(payload.get("failures", []))
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        path = subject_dir / f"S{subject}.json"
        if path.exists() and not args.force and not args.force_eval:
            payload = json.loads(path.read_text())
            rows.extend(payload["rows"])
            if not args.quiet:
                print(f"S{subject}: eval resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows = evaluate_subject(
                subject,
                data,
                all_stats,
                args.features,
                [str(depth) for depth in args.depths],
                [float(fraction) for fraction in args.fractions],
                int(args.prefix),
                int(args.eval_start),
                int(args.min_eval_trials),
            )
            path.write_text(json.dumps({"subject": int(subject), "rows": subject_rows}, indent=2))
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: eval {len(subject_rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "eval", "error": repr(error)})
            print(f"S{subject}: eval FAIL {error!r}", flush=True)
    rows_path.write_text(json.dumps({"records": rows, "failures": failures}, indent=2))
    write_csv(args.output_dir / "selection_records.csv", rows)
    return rows, failures


def group_by_subject(rows: Sequence[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject"])].append(row)
    return grouped


def paired_subject_gains(
    selected: Sequence[Mapping[str, object]],
    full: Sequence[Mapping[str, object]],
) -> np.ndarray:
    full_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row for row in full
    }
    values = []
    for subject, subject_rows in group_by_subject(selected).items():
        correct = sum(int(row["test_correct"]) for row in subject_rows)
        full_correct = sum(
            int(full_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
        )
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * (correct - full_correct) / total)
    return np.asarray(values, dtype=np.float64)


def paired_subject_diff(
    first: Sequence[Mapping[str, object]],
    second: Sequence[Mapping[str, object]],
) -> np.ndarray:
    second_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row for row in second
    }
    values = []
    for subject, subject_rows in group_by_subject(first).items():
        correct_a = sum(int(row["test_correct"]) for row in subject_rows)
        correct_b = sum(
            int(second_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
        )
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * (correct_a - correct_b) / total)
    return np.asarray(values, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(int(bootstrap), len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def unit_risk(
    selected: Sequence[Mapping[str, object]],
    full: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    full_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row for row in full
    }
    values = []
    subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        values.append(float(row["test_acc"]) - float(full_by_key[key]["test_acc"]))
        subjects.append(int(row["subject"]))
    values = np.asarray(values, dtype=np.float64)
    counts = Counter(subjects)
    weights = np.asarray(
        [1.0 / (len(counts) * counts[int(subject)]) for subject in subjects],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return {
        "gain_vs_full_unit_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_full_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_full_q05_pp": weighted_quantile(values, weights, 0.05),
        "p_gain_vs_full_lt_minus5": float(np.sum(weights[values < -5.0])),
    }


def summarize_method(
    selected: Sequence[Mapping[str, object]],
    full: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    gains = paired_subject_gains(selected, full)
    subject_acc = []
    for subject_rows in group_by_subject(selected).values():
        correct = sum(int(row["test_correct"]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        subject_acc.append(100.0 * correct / total)
    return {
        "n_subjects": int(len(subject_acc)),
        "n_units": int(len(selected)),
        "accuracy_subject_pooled_mean": float(np.mean(subject_acc)),
        "gain_vs_full_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, bootstrap, seed),
        **unit_risk(selected, full),
    }


def summarize_records(
    records: Sequence[Mapping[str, object]],
    depths: Sequence[str],
    fractions: Sequence[float],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    comparison: dict[str, object] = {}
    groups = {
        "primary_pooled": [row for row in records if row["condition"] in PRIMARY_CONDITIONS],
        "pure_lr": [row for row in records if row["condition"] == "pure_lr"],
        "pure_ud": [row for row in records if row["condition"] == "pure_ud"],
    }
    for group_name, group_rows in groups.items():
        full = [row for row in group_rows if row["method"] == "full_q1p00"]
        by_method: dict[str, object] = {
            "full_q1p00": summarize_method(full, full, bootstrap, seed)
        }
        by_comparison: dict[str, object] = {}
        for depth in [str(depth) for depth in depths]:
            for family in ("longitudinal", "source_only"):
                for fraction in [float(fraction) for fraction in fractions]:
                    method = method_name(depth, family, fraction)
                    selected = [
                        row
                        for row in group_rows
                        if row["depth"] == depth and row["method"] == method
                    ]
                    by_method[f"{depth}__{method}"] = summarize_method(
                        selected, full, bootstrap, seed
                    )
            for fraction in [float(fraction) for fraction in fractions]:
                long_rows = [
                    row
                    for row in group_rows
                    if row["depth"] == depth
                    and row["method"] == method_name(depth, "longitudinal", fraction)
                ]
                source_rows = [
                    row
                    for row in group_rows
                    if row["depth"] == depth
                    and row["method"] == method_name(depth, "source_only", fraction)
                ]
                diff = paired_subject_diff(long_rows, source_rows)
                by_comparison[f"{depth}__longitudinal_minus_source_only__{fraction_token(fraction)}"] = {
                    "n_subjects": int(len(diff)),
                    "mean_pp": float(np.mean(diff)),
                    "bootstrap_95ci": bootstrap_ci(diff, bootstrap, seed),
                    "q05_pp": float(np.quantile(diff, 0.05)) if len(diff) else float("nan"),
                    "p_lt_0": float(np.mean(diff < 0)) if len(diff) else float("nan"),
                }
            # Best fixed fraction within family, chosen post-hoc for mechanism readout only.
            for long_fraction in [float(fraction) for fraction in fractions]:
                for source_fraction in [float(fraction) for fraction in fractions]:
                    long_rows = [
                        row
                        for row in group_rows
                        if row["depth"] == depth
                        and row["method"] == method_name(depth, "longitudinal", long_fraction)
                    ]
                    source_rows = [
                        row
                        for row in group_rows
                        if row["depth"] == depth
                        and row["method"] == method_name(depth, "source_only", source_fraction)
                    ]
                    diff = paired_subject_diff(long_rows, source_rows)
                    by_comparison[
                        f"{depth}__long_q{long_fraction:.2f}_minus_source_q{source_fraction:.2f}"
                    ] = {
                        "n_subjects": int(len(diff)),
                        "mean_pp": float(np.mean(diff)),
                        "bootstrap_95ci": bootstrap_ci(diff, bootstrap, seed),
                        "q05_pp": float(np.quantile(diff, 0.05)) if len(diff) else float("nan"),
                        "p_lt_0": float(np.mean(diff < 0)) if len(diff) else float("nan"),
                    }
        summary[group_name] = by_method
        comparison[group_name] = by_comparison
    return {"summary": summary, "comparisons": comparison}


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats, failures = build_depth_stats(args)
    rows, eval_failures = build_eval_rows(args, all_stats)
    failures.extend(eval_failures)
    report = {
        "config": {
            "subjects": parse_subjects(args.subjects),
            "depths": [str(depth) for depth in args.depths],
            "fractions": [float(fraction) for fraction in args.fractions],
            "features": list(args.features),
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "n_rows": len(rows),
        "failures": failures,
        **summarize_records(
            rows,
            [str(depth) for depth in args.depths],
            [float(fraction) for fraction in args.fractions],
            int(args.bootstrap),
            int(args.seed),
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Flat summary for quick reading.
    flat_rows: list[dict[str, object]] = []
    for group, methods in report["summary"].items():
        for method, values in methods.items():
            flat_rows.append({"group": group, "method": method, **values})
    write_csv(args.output_dir / "summary.csv", flat_rows)
    flat_comparisons: list[dict[str, object]] = []
    for group, comparisons in report["comparisons"].items():
        for comparison, values in comparisons.items():
            flat_comparisons.append({"group": group, "comparison": comparison, **values})
    write_csv(args.output_dir / "comparisons.csv", flat_comparisons)
    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
