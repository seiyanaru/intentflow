"""G1a: Fast prerequisite for channel-reliability adaptation.

For each target session, derive spatial-patch quality features solely from its
unlabeled prefix and ask whether they predict *existing* broad prefix-EA harm.
This is deliberately cheaper than patch-ablation counterfactuals.  If it has no
LOSO signal, channel-reliability adaptation has no empirical premise and the
costly G1b counterfactual audit should not be run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stieger_channel_cause_audit import PATCHES, patch_indices, patch_quality
from stieger_neuro_feature_baseline import CONDITIONS, DEFAULT_CACHE, load_subject, parse_subjects


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_E1_SUMMARY = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_feature_baseline"
    / "summary.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260626_stieger_channel_quality_harm_audit"
)
DEFAULT_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--e1-summary-json", type=Path, default=DEFAULT_E1_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
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


def e1_lookup(path: Path) -> dict[tuple[int, int, str], Mapping[str, object]]:
    payload = json.loads(path.read_text())
    selected = {}
    for row in payload["rows"]:
        if row["feature_config"] == "broad_all60" and row["adapter"] == "prefix_ea":
            selected[(int(row["subject"]), int(row["session"]), str(row["condition"]))] = row
    return selected


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    lookup: Mapping[tuple[int, int, str], Mapping[str, object]],
    conditions: Sequence[str],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
    bands = ("broad_8_30", "mu_8_13", "low_beta_13_20", "high_beta_20_30")
    covariances = {band: data[f"cov_{band}"].astype(np.float64) for band in bands}
    patches = patch_indices(data["channels"])
    rows: list[dict[str, object]] = []
    for condition in conditions:
        specification = CONDITIONS[condition]
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_indices = np.flatnonzero(mask & (data["session"] == sessions[0]))
        for session in sessions[1:]:
            target_indices = np.flatnonzero(mask & (data["session"] == session))
            if len(target_indices) <= eval_start:
                continue
            eval_indices = target_indices[eval_start:]
            if len(eval_indices) < min_eval_trials:
                continue
            outcome = lookup.get((subject, int(session), condition))
            if outcome is None:
                continue
            row: dict[str, object] = {
                "subject": int(subject),
                "session": int(session),
                "condition": condition,
                "n_eval": int(len(eval_indices)),
                "source_acc": float(outcome["source_acc"]),
                "adapted_acc": float(outcome["adapted_acc"]),
                "ea_delta_pp": float(outcome["delta_pp"]),
                "ea_harm5": int(float(outcome["delta_pp"]) < -5.0),
            }
            prefix_indices = target_indices[:prefix]
            for patch, indices in patches.items():
                for key, value in patch_quality(
                    covariances, covariances, source_indices, prefix_indices, indices
                ).items():
                    row[f"{patch}__{key}"] = float(value)
            rows.append(row)
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    table_path = args.output_dir / "quality_harm_table.json"
    if table_path.exists() and not args.force:
        return list(json.loads(table_path.read_text())["rows"])
    lookup = e1_lookup(args.e1_summary_json)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            rows.extend(json.loads(output.read_text())["rows"])
            continue
        try:
            subject_rows = evaluate_subject(
                subject,
                load_subject(args.cache_dir, subject),
                lookup,
                args.conditions,
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
    (args.output_dir / "quality_harm_table.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2)
    )
    write_csv(args.output_dir / "quality_harm_table.csv", rows)
    return rows


def feature_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(column for column in rows[0] if "__q_" in column)


def evaluate_loso(rows: Sequence[Mapping[str, object]], alpha: float) -> list[dict[str, object]]:
    columns = feature_columns(rows)
    subjects = np.asarray([int(row["subject"]) for row in rows])
    predictions: list[dict[str, object]] = []
    for held_subject in sorted(set(subjects.tolist())):
        train = subjects != held_subject
        test = subjects == held_subject
        x_train = np.asarray([[float(row[column]) for column in columns] for row, keep in zip(rows, train) if keep])
        y_train = np.asarray([float(row["ea_delta_pp"]) for row, keep in zip(rows, train) if keep])
        x_test = np.asarray([[float(row[column]) for column in columns] for row, keep in zip(rows, test) if keep])
        scaler = StandardScaler()
        model = Ridge(alpha=alpha).fit(scaler.fit_transform(x_train), y_train)
        predicted = model.predict(scaler.transform(x_test))
        for row, value in zip([row for row, keep in zip(rows, test) if keep], predicted):
            predictions.append({**row, "predicted_ea_delta_pp": float(value)})
    return predictions


def summarize(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    actual = np.asarray([float(row["ea_delta_pp"]) for row in predictions])
    predicted = np.asarray([float(row["predicted_ea_delta_pp"]) for row in predictions])
    harm = np.asarray([int(row["ea_harm5"]) for row in predictions])
    return {
        "n_subjects": len({int(row["subject"]) for row in predictions}),
        "n_sessions": len(predictions),
        "delta_prediction_spearman": float(spearmanr(predicted, actual).statistic),
        "harm5_auroc_using_negative_predicted_delta": float(roc_auc_score(harm, -predicted)) if len(np.unique(harm)) == 2 else None,
        "observed_harm5_rate": float(np.mean(harm)),
    }


def main() -> None:
    args = parse_args()
    if any(condition not in CONDITIONS for condition in args.conditions):
        raise ValueError("Unknown condition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args)
    reports: dict[str, object] = {}
    all_predictions: list[dict[str, object]] = []
    for condition in list(args.conditions) + ["pooled"]:
        scoped = rows if condition == "pooled" else [row for row in rows if row["condition"] == condition]
        if not scoped:
            continue
        predictions = evaluate_loso(scoped, args.ridge_alpha)
        reports[condition] = summarize(predictions)
        all_predictions.extend(predictions)
    write_csv(args.output_dir / "quality_harm_predictions.csv", all_predictions)
    report = {
        "config": {
            "subjects": args.subjects,
            "conditions": args.conditions,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "ridge_alpha": args.ridge_alpha,
            "patches": PATCHES,
            "outer": "leave-one-subject-out",
            "outcome": "broad_all60 prefix-EA delta versus source",
        },
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
