"""Task-aware prospective Riemann-EA audit on all Stieger paradigms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cross_adapter_core import risk_utility_summary
from stieger_cross_adapter_pilot import parse_subjects
from stieger_task_context_audit import (
    bootstrap_icc,
    invsqrt_spd,
    one_way_icc,
    tangent_features,
)


DEFAULT_CACHE = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_task_cov_cache"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260622_stieger_task_aware_riemann"
)
CONDITIONS = {
    "pure_lr": {"tasks": (1,), "targets": (1, 2)},
    "pure_ud": {"tasks": (2,), "targets": (3, 4)},
    "two_d": {"tasks": (3,), "targets": (1, 2, 3, 4)},
    "two_d_horizontal": {"tasks": (3,), "targets": (1, 2)},
    "two_d_vertical": {"tasks": (3,), "targets": (3, 4)},
    "mixed_horizontal": {"tasks": (1, 3), "targets": (1, 2)},
    "mixed_vertical": {"tasks": (2, 3), "targets": (3, 4)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--eval-size",
        type=int,
        help="If set, evaluate only this many trials after the prefix.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_subject(cache_dir: Path, subject: int) -> dict[str, np.ndarray]:
    paths = sorted(
        (cache_dir / f"S{subject}").glob("session_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(cache_dir / f"S{subject}")
    arrays: dict[str, list[np.ndarray]] = {
        "covariances": [],
        "task": [],
        "target": [],
        "session": [],
        "run": [],
        "trial": [],
    }
    for path in paths:
        payload = np.load(path)
        n_trials = len(payload["task"])
        arrays["covariances"].append(payload["covariances"].astype(np.float64))
        arrays["task"].append(payload["task"].astype(np.int64))
        arrays["target"].append(payload["target"].astype(np.int64))
        arrays["session"].append(
            np.full(n_trials, int(payload["session"]), dtype=np.int64)
        )
        arrays["run"].append(payload["run"].astype(np.int64))
        arrays["trial"].append(payload["trial"].astype(np.int64))
    return {key: np.concatenate(value) for key, value in arrays.items()}


def evaluate_condition(
    subject: int,
    condition: str,
    specification: dict,
    data: dict[str, np.ndarray],
    prefix: int,
    eval_size: int | None,
) -> list[dict]:
    mask = np.isin(data["task"], specification["tasks"]) & np.isin(
        data["target"], specification["targets"]
    )
    sessions = sorted(np.unique(data["session"][mask]).tolist())
    if len(sessions) < 2:
        return []
    source_mask = mask & (data["session"] == sessions[0])
    source_covariances = data["covariances"][source_mask]
    source_labels = data["target"][source_mask]
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
    for session in sessions[1:]:
        target_mask = mask & (data["session"] == session)
        covariances = data["covariances"][target_mask]
        labels = data["target"][target_mask]
        tasks = data["task"][target_mask]
        if len(labels) <= prefix or len(np.unique(labels)) < 2:
            continue
        evaluation_stop = (
            min(len(labels), prefix + eval_size)
            if eval_size is not None
            else len(labels)
        )
        evaluation_covariances = covariances[prefix:evaluation_stop]
        evaluation_labels = labels[prefix:evaluation_stop]
        evaluation_tasks = tasks[prefix:evaluation_stop]
        if len(evaluation_labels) == 0:
            continue
        source_predictions = classifier.predict(
            tangent_features(evaluation_covariances, source_reference)
        )
        source_accuracy = float(
            np.mean(source_predictions == evaluation_labels) * 100
        )
        for adapter, reference_trials in (
            ("prefix_ea", covariances[:prefix]),
            ("full_ea", covariances),
        ):
            target_reference = invsqrt_spd(reference_trials.mean(axis=0))
            predictions = classifier.predict(
                tangent_features(evaluation_covariances, target_reference)
            )
            adapted_accuracy = float(
                np.mean(predictions == evaluation_labels) * 100
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
                    "eval_task1_fraction": float(
                        np.mean(evaluation_tasks == 1)
                    ),
                    "eval_task2_fraction": float(
                        np.mean(evaluation_tasks == 2)
                    ),
                    "eval_task3_fraction": float(
                        np.mean(evaluation_tasks == 3)
                    ),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    summaries: dict[str, dict] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        summaries[condition] = {}
        for adapter in ("prefix_ea", "full_ea"):
            summary = risk_utility_summary(selected, adapter)
            summary["icc_delta"] = one_way_icc(selected, adapter)
            summary["icc_subject_bootstrap_95ci"] = bootstrap_icc(
                selected, adapter
            )
            summaries[condition][adapter] = summary
    return summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    failures: list[dict] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            all_rows.extend(payload["rows"])
            print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            rows: list[dict] = []
            for condition, specification in CONDITIONS.items():
                rows.extend(
                    evaluate_condition(
                        subject,
                        condition,
                        specification,
                        data,
                        args.prefix,
                        args.eval_size,
                    )
                )
            output.write_text(json.dumps({"subject": subject, "rows": rows}, indent=2))
            all_rows.extend(rows)
            print(f"S{subject}: {len(rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": subject, "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    report = {
        "config": {
            "subjects": args.subjects,
            "prefix": args.prefix,
            "eval_size": args.eval_size,
            "cache_dir": str(args.cache_dir),
            "conditions": CONDITIONS,
        },
        "n_subjects": len({int(row["subject"]) for row in all_rows}),
        "n_rows": len(all_rows),
        "failures": failures,
        "summaries": summarize(all_rows),
        "rows": all_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
