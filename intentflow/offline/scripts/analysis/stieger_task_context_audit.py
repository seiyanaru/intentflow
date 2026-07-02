"""Prospective Riemann-EA audit of hidden task context in Stieger2021.

The exact existing horizontal epoch cache is reused.  Three conditions are
trained and evaluated independently:

* mixed_horizontal: task 1 LR plus task 3 horizontal trials (legacy setting)
* pure_lr: task 1 only
* two_d_horizontal: task 3 target 1/2 only

For every target session, the first 32 condition-specific trials estimate the
prospective EA reference and trials 33+ are evaluated.  Full-session EA is
reported on the same evaluation trials as a transductive upper-reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cross_adapter_core import risk_utility_summary
from stieger_cross_adapter_pilot import parse_subjects


DEFAULT_EPOCH_CACHE = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
)
DEFAULT_METADATA = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_task_metadata"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260622_stieger_task_context_audit"
)
CONDITIONS = {
    "mixed_horizontal": (1, 3),
    "pure_lr": (1,),
    "two_d_horizontal": (3,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--epoch-cache", type=Path, default=DEFAULT_EPOCH_CACHE)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def trial_covariances(x: np.ndarray) -> np.ndarray:
    return (
        np.einsum("nct,ndt->ncd", x.astype(np.float64), x, optimize=True)
        / x.shape[-1]
    )


def invsqrt_spd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values, vectors = eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors * values**-0.5) @ vectors.T


def tangent_features(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    transformed = np.einsum(
        "ij,njk,lk->nil", reference, covariances, reference, optimize=True
    )
    transformed = 0.5 * (transformed + transformed.transpose(0, 2, 1))
    values, vectors = np.linalg.eigh(transformed)
    values = np.clip(values, 1e-10, None)
    logs = np.einsum(
        "nij,nj,nkj->nik", vectors, np.log(values), vectors, optimize=True
    )
    upper = np.triu_indices(covariances.shape[1])
    scale = np.sqrt(2.0) * np.ones(
        (covariances.shape[1], covariances.shape[1])
    )
    np.fill_diagonal(scale, 1.0)
    return logs[:, upper[0], upper[1]] * scale[upper]


def condition_rows(
    subject: int,
    condition: str,
    tasks: tuple[int, ...],
    covariances: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    task_numbers: np.ndarray,
    prefix: int,
) -> list[dict]:
    condition_mask = np.isin(task_numbers, tasks)
    session_ids = sorted(np.unique(sessions[condition_mask]).tolist())
    if len(session_ids) < 2:
        return []
    source_session = session_ids[0]
    source_mask = condition_mask & (sessions == source_session)
    source_covariances = covariances[source_mask]
    source_labels = labels[source_mask]
    if len(np.unique(source_labels)) < 2:
        return []
    source_reference = invsqrt_spd(source_covariances.mean(axis=0))
    classifier = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto"
    ).fit(
        tangent_features(source_covariances, source_reference),
        source_labels,
    )

    rows: list[dict] = []
    for session in session_ids[1:]:
        target_mask = condition_mask & (sessions == session)
        target_covariances = covariances[target_mask]
        target_labels = labels[target_mask]
        if len(target_labels) <= prefix or len(np.unique(target_labels)) < 2:
            continue
        evaluation_covariances = target_covariances[prefix:]
        evaluation_labels = target_labels[prefix:]
        source_prediction = classifier.predict(
            tangent_features(evaluation_covariances, source_reference)
        )
        source_accuracy = float(
            np.mean(source_prediction == evaluation_labels) * 100
        )
        for adapter, reference_covariances in (
            ("prefix_ea", target_covariances[:prefix]),
            ("full_ea", target_covariances),
        ):
            target_reference = invsqrt_spd(reference_covariances.mean(axis=0))
            prediction = classifier.predict(
                tangent_features(evaluation_covariances, target_reference)
            )
            adapted_accuracy = float(
                np.mean(prediction == evaluation_labels) * 100
            )
            rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "condition": condition,
                    "adapter": adapter,
                    "source_acc": source_accuracy,
                    "adapted_acc": adapted_accuracy,
                    "delta_pp": adapted_accuracy - source_accuracy,
                    "n_prefix": prefix,
                    "n_eval": int(len(evaluation_labels)),
                }
            )
    return rows


def one_way_icc(rows: list[dict], adapter: str) -> float | None:
    selected = [row for row in rows if row["adapter"] == adapter]
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        grouped[int(row["subject"])].append(float(row["delta_pp"]))
    grouped = {key: value for key, value in grouped.items() if len(value) >= 2}
    if len(grouped) < 3:
        return None
    values = np.asarray([value for group in grouped.values() for value in group])
    counts = np.asarray([len(group) for group in grouped.values()], dtype=float)
    means = np.asarray([np.mean(group) for group in grouped.values()])
    grand = float(np.mean(values))
    n_observations = len(values)
    n_subjects = len(grouped)
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(
        sum(
            np.sum((np.asarray(group) - np.mean(group)) ** 2)
            for group in grouped.values()
        )
    )
    ms_between = ss_between / (n_subjects - 1)
    ms_within = ss_within / (n_observations - n_subjects)
    n0 = (
        n_observations - float(np.sum(counts**2)) / n_observations
    ) / (n_subjects - 1)
    subject_variance = max((ms_between - ms_within) / n0, 0.0)
    return float(subject_variance / (subject_variance + ms_within))


def bootstrap_icc(
    rows: list[dict],
    adapter: str,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> list[float] | None:
    selected = [row for row in rows if row["adapter"] == adapter]
    subjects = sorted({int(row["subject"]) for row in selected})
    if len(subjects) < 3:
        return None
    by_subject = {
        subject: [row for row in selected if int(row["subject"]) == subject]
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        bootstrap_rows: list[dict] = []
        for new_subject, original_subject in enumerate(sampled):
            for row in by_subject[int(original_subject)]:
                copied = dict(row)
                copied["subject"] = new_subject
                bootstrap_rows.append(copied)
        value = one_way_icc(bootstrap_rows, adapter)
        if value is not None:
            values.append(value)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def summarize(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for condition in CONDITIONS:
        condition_rows_ = [row for row in rows if row["condition"] == condition]
        if not condition_rows_:
            continue
        output[condition] = {}
        for adapter in ("prefix_ea", "full_ea"):
            if not any(row["adapter"] == adapter for row in condition_rows_):
                continue
            summary = risk_utility_summary(condition_rows_, adapter)
            summary["icc_delta"] = one_way_icc(condition_rows_, adapter)
            summary["icc_subject_bootstrap_95ci"] = bootstrap_icc(
                condition_rows_, adapter
            )
            output[condition][adapter] = summary
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    failures: list[dict] = []
    for subject in parse_subjects(args.subjects):
        output_path = subject_dir / f"S{subject}.json"
        if output_path.exists() and not args.force:
            payload = json.loads(output_path.read_text())
            print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            all_rows.extend(payload["rows"])
            continue
        try:
            cache = np.load(args.epoch_cache / f"S{subject}_epochs.npz")
            metadata = np.load(
                args.metadata_dir / f"S{subject}_taskmeta.npz"
            )
            x = cache["X"].astype(np.float32)
            labels = cache["y"].astype(np.int64)
            sessions = cache["sess"].astype(np.int64)
            task_numbers = metadata["task"].astype(np.int64)
            if not np.array_equal(sessions, metadata["session"]):
                raise RuntimeError("cache/metadata session mismatch")
            covariances = trial_covariances(x)
            rows: list[dict] = []
            for condition, tasks in CONDITIONS.items():
                rows.extend(
                    condition_rows(
                        subject,
                        condition,
                        tasks,
                        covariances,
                        labels,
                        sessions,
                        task_numbers,
                        args.prefix,
                    )
                )
            payload = {"subject": subject, "rows": rows}
            output_path.write_text(json.dumps(payload, indent=2))
            all_rows.extend(rows)
            print(f"S{subject}: {len(rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": subject, "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    report = {
        "config": {
            "subjects": args.subjects,
            "prefix": args.prefix,
            "conditions": {key: list(value) for key, value in CONDITIONS.items()},
            "epoch_cache": str(args.epoch_cache),
            "metadata_dir": str(args.metadata_dir),
        },
        "n_rows": len(all_rows),
        "n_subjects": len({int(row["subject"]) for row in all_rows}),
        "failures": failures,
        "summaries": summarize(all_rows),
        "rows": all_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
