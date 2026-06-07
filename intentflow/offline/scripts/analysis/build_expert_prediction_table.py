"""Build trial-level expert prediction tables for adaptation portfolio studies.

The first portfolio prototype uses already-saved logits from source / EA variants
instead of re-running checkpoints. This script collects those logits into:

- expert_trial_table.csv: long-format per-trial predictions and probabilities
- expert_summary.csv: per-subject, per-expert metrics
- expert_portfolio_arrays.npz: dense arrays for fast online portfolio experiments
- expert_portfolio_summary.json: metadata and oracle winners

Labels are used only for reporting accuracy / oracle gaps, not for any selector
or online score.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


EPS = 1e-12


@dataclass
class ExpertSpec:
    name: str
    kind: str
    path_pattern: Optional[str] = None
    first: Optional[str] = None
    second: Optional[str] = None
    weight_first: float = 0.5
    components: Optional[dict[str, float]] = None


def default_experts(args: argparse.Namespace) -> list[ExpertSpec]:
    return [
        ExpertSpec(name="source", kind="latest", path_pattern=args.source_logits_pattern),
        ExpertSpec(name="full_ea", kind="latest", path_pattern=args.full_ea_logits_pattern),
        ExpertSpec(name="shrink_0.1", kind="latest", path_pattern=args.shrink01_logits_pattern),
        ExpertSpec(name="shrink_0.03", kind="latest", path_pattern=args.shrink003_logits_pattern),
        ExpertSpec(
            name="partial_0.5_shrink_0.1",
            kind="latest",
            path_pattern=args.partial_logits_pattern,
        ),
        ExpertSpec(
            name="full_shrink_mean",
            kind="fusion",
            first="full_ea",
            second="shrink_0.1",
            weight_first=0.5,
        ),
        ExpertSpec(
            name="source_full_mean",
            kind="fusion",
            first="source",
            second="full_ea",
            weight_first=0.5,
        ),
        ExpertSpec(
            name="source_full_shrink_304030",
            kind="weighted_fusion",
            components={"source": 0.3, "full_ea": 0.4, "shrink_0.1": 0.3},
        ),
        ExpertSpec(
            name="source_full_shrink_454510",
            kind="weighted_fusion",
            components={"source": 0.45, "full_ea": 0.45, "shrink_0.1": 0.10},
        ),
    ]


def subjects_from_arg(arg: str) -> list[int]:
    if arg == "all":
        return list(range(1, 10))
    return [int(x) for x in arg.split(",") if x.strip()]


def latest_match(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def load_labels(results_root: Path, sid: int, pattern: str) -> Optional[np.ndarray]:
    match = latest_match(results_root, pattern.format(sid=sid))
    if match is None or not match.exists():
        return None
    return np.load(match)["labels"].astype(np.int64)


def load_logits_for_spec(
    spec: ExpertSpec,
    sid: int,
    results_root: Path,
    loaded_logits: dict[str, np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if spec.kind in ("path", "latest"):
        path = latest_match(results_root, spec.path_pattern.format(sid=sid))
        if path is None or not path.exists():
            return None, None
        return np.load(path), str(path)

    if spec.kind == "fusion":
        if spec.first not in loaded_logits or spec.second not in loaded_logits:
            return None, None
        first = softmax(loaded_logits[spec.first])
        second = softmax(loaded_logits[spec.second])
        probs = spec.weight_first * first + (1.0 - spec.weight_first) * second
        return np.log(np.clip(probs, EPS, 1.0)), f"fusion({spec.first},{spec.second})"

    if spec.kind == "weighted_fusion":
        if not spec.components:
            raise ValueError(f"{spec.name}: weighted_fusion requires components")
        missing = [name for name in spec.components if name not in loaded_logits]
        if missing:
            return None, None
        total = float(sum(spec.components.values()))
        probs = None
        for name, weight in spec.components.items():
            part = softmax(loaded_logits[name]) * (float(weight) / total)
            probs = part if probs is None else probs + part
        source = ",".join(f"{name}:{weight:g}" for name, weight in spec.components.items())
        return np.log(np.clip(probs, EPS, 1.0)), f"weighted_fusion({source})"

    raise ValueError(f"Unknown expert kind: {spec.kind}")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), EPS, None)


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, EPS, 1.0))).sum(axis=-1)


def margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(probs, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def prior_kl(preds: np.ndarray, n_classes: int) -> tuple[float, float, list[int]]:
    counts = np.bincount(preds, minlength=n_classes)
    prior = counts / np.clip(counts.sum(), 1, None)
    kl = float((prior * np.log(np.clip(prior, EPS, 1.0) / (1.0 / n_classes))).sum())
    return kl, float(prior.max()), counts.astype(int).tolist()


def maybe_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def load_subject_diagnostics(path: Optional[str]) -> dict[int, dict]:
    if not path:
        return {}
    obj = json.load(open(path))
    return {int(row["subject"]): row for row in obj.get("rows", [])}


def flatten_cov(prefix: str, row: Optional[dict]) -> dict[str, float]:
    if not row:
        return {}
    out = {
        f"{prefix}_cov_distance": row.get("cov_distance"),
    }
    for domain in ("train", "test"):
        values = row.get(domain, {})
        for key in ("condition", "diag_cv", "diag_ratio"):
            out[f"{prefix}_{domain}_{key}"] = values.get(key)
    return out


def flatten_reliability(prefix: str, row: Optional[dict]) -> dict[str, float]:
    if not row:
        return {}
    out = {}
    summaries = row.get("weight_summary", {})
    for mode, values in summaries.items():
        for key in ("min", "mean", "max"):
            out[f"{prefix}_{mode}_{key}"] = values.get(key)
        out[f"{prefix}_{mode}_low_count"] = len(values.get("low_channels", []))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: maybe_float(row.get(key, "")) for key in fieldnames})


def build_tables(args: argparse.Namespace) -> dict:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    subjects = subjects_from_arg(args.subjects)
    specs = default_experts(args)
    expert_names = [spec.name for spec in specs]
    cov_by_subject = load_subject_diagnostics(args.cov_diagnostics)
    rel_by_subject = load_subject_diagnostics(args.reliability_diagnostics)

    n_subjects = len(subjects)
    n_experts = len(specs)
    n_trials = args.n_trials
    n_classes = args.n_classes
    logits_array = np.full(
        (n_subjects, n_experts, n_trials, n_classes), np.nan, dtype=np.float32
    )
    probs_array = np.full_like(logits_array, np.nan)
    preds_array = np.full((n_subjects, n_experts, n_trials), -1, dtype=np.int16)
    labels_array = np.full((n_subjects, n_trials), -1, dtype=np.int16)
    available = np.zeros((n_subjects, n_experts), dtype=bool)

    trial_rows: list[dict] = []
    summary_rows: list[dict] = []
    subject_rows: list[dict] = []
    paths: dict[str, dict[str, str]] = {}

    for sidx, sid in enumerate(subjects):
        labels = load_labels(results_root, sid, args.labels_pattern)
        if labels is None:
            print(f"S{sid}: labels not found, skip")
            continue
        if labels.shape[0] != n_trials:
            raise ValueError(f"S{sid}: expected {n_trials} labels, got {labels.shape}")
        labels_array[sidx] = labels

        loaded_logits: dict[str, np.ndarray] = {}
        paths[str(sid)] = {}
        for eidx, spec in enumerate(specs):
            logits, source_path = load_logits_for_spec(spec, sid, results_root, loaded_logits)
            if logits is None:
                print(f"S{sid}: missing {spec.name}")
                continue
            if logits.shape != (n_trials, n_classes):
                raise ValueError(
                    f"S{sid} {spec.name}: expected {(n_trials, n_classes)}, got {logits.shape}"
                )

            loaded_logits[spec.name] = logits
            paths[str(sid)][spec.name] = source_path or ""
            probs = softmax(logits)
            preds = probs.argmax(axis=-1).astype(np.int16)
            ent = entropy(probs)
            conf = probs.max(axis=-1)
            marg = margin(probs)
            correct = preds == labels
            kl, dominance, counts = prior_kl(preds, n_classes)

            logits_array[sidx, eidx] = logits.astype(np.float32)
            probs_array[sidx, eidx] = probs.astype(np.float32)
            preds_array[sidx, eidx] = preds
            available[sidx, eidx] = True

            summary = {
                "subject": sid,
                "expert": spec.name,
                "available": 1,
                "path": source_path,
                "acc": float(correct.mean() * 100.0),
                "error": float((1.0 - correct.mean()) * 100.0),
                "entropy_mean": float(ent.mean()),
                "entropy_std": float(ent.std()),
                "confidence_mean": float(conf.mean()),
                "confidence_std": float(conf.std()),
                "margin_mean": float(marg.mean()),
                "margin_std": float(marg.std()),
                "prior_kl": kl,
                "dominance": dominance,
                "counts": json.dumps(counts),
            }
            summary.update(flatten_cov("cov", cov_by_subject.get(sid)))
            summary.update(flatten_reliability("rel", rel_by_subject.get(sid)))
            summary_rows.append(summary)

            if args.write_trial_csv:
                for trial in range(n_trials):
                    row = {
                        "subject": sid,
                        "trial": trial,
                        "expert": spec.name,
                        "label": int(labels[trial]),
                        "pred": int(preds[trial]),
                        "correct": int(correct[trial]),
                        "confidence": float(conf[trial]),
                        "entropy": float(ent[trial]),
                        "margin": float(marg[trial]),
                    }
                    for cls in range(n_classes):
                        row[f"logit_{cls}"] = float(logits[trial, cls])
                        row[f"prob_{cls}"] = float(probs[trial, cls])
                    trial_rows.append(row)

        subject_summaries = [row for row in summary_rows if row["subject"] == sid]
        if subject_summaries:
            oracle = max(subject_summaries, key=lambda row: row["acc"])
            source = next((row for row in subject_summaries if row["expert"] == "source"), None)
            full = next((row for row in subject_summaries if row["expert"] == "full_ea"), None)
            row = {
                "subject": sid,
                "oracle_expert": oracle["expert"],
                "oracle_acc": oracle["acc"],
                "source_acc": None if source is None else source["acc"],
                "full_ea_acc": None if full is None else full["acc"],
                "oracle_gap_vs_source": None if source is None else oracle["acc"] - source["acc"],
                "oracle_gap_vs_full_ea": None if full is None else oracle["acc"] - full["acc"],
                "available_experts": json.dumps([r["expert"] for r in subject_summaries]),
            }
            row.update(flatten_cov("cov", cov_by_subject.get(sid)))
            row.update(flatten_reliability("rel", rel_by_subject.get(sid)))
            subject_rows.append(row)
            print(
                f"S{sid}: oracle={oracle['expert']} {oracle['acc']:.2f}%, "
                f"source={row['source_acc']:.2f}%, full={row['full_ea_acc']:.2f}%"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "expert_summary.csv", summary_rows)
    write_csv(output_dir / "expert_subject_oracle.csv", subject_rows)
    if args.write_trial_csv:
        write_csv(output_dir / "expert_trial_table.csv", trial_rows)

    np.savez_compressed(
        output_dir / "expert_portfolio_arrays.npz",
        subjects=np.asarray(subjects, dtype=np.int16),
        experts=np.asarray(expert_names),
        logits=logits_array,
        probs=probs_array,
        preds=preds_array,
        labels=labels_array,
        available=available,
    )

    accs = [row["oracle_acc"] for row in subject_rows]
    src = [row["source_acc"] for row in subject_rows if row["source_acc"] is not None]
    full = [row["full_ea_acc"] for row in subject_rows if row["full_ea_acc"] is not None]
    metadata = {
        "results_root": str(results_root),
        "subjects": subjects,
        "experts": [asdict(spec) for spec in specs],
        "paths": paths,
        "summary": {
            "n_subjects": len(subject_rows),
            "source_mean": float(np.mean(src)) if src else None,
            "full_ea_mean": float(np.mean(full)) if full else None,
            "oracle_mean": float(np.mean(accs)) if accs else None,
            "oracle_gap_vs_source": float(np.mean(accs) - np.mean(src)) if accs and src else None,
            "oracle_gap_vs_full_ea": float(np.mean(accs) - np.mean(full)) if accs and full else None,
        },
        "subject_rows": subject_rows,
    }
    with open(output_dir / "expert_portfolio_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="intentflow/offline/results")
    parser.add_argument(
        "--output_dir",
        default="intentflow/offline/results/research_outputs/260602_expert_portfolio_table",
    )
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--n_trials", type=int, default=288)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--cov_diagnostics", default=None)
    parser.add_argument("--reliability_diagnostics", default=None)
    parser.add_argument("--write_trial_csv", action="store_true")
    parser.add_argument(
        "--labels_pattern",
        default="ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz",
    )
    parser.add_argument(
        "--source_logits_pattern",
        default=(
            "c_aug_true_9subj_20260506_004923/sources/s{sid}/"
            "logits_s{sid}_tcformer_policy_safe_otta.npy"
        ),
    )
    parser.add_argument(
        "--full_ea_logits_pattern",
        default="ea_aware_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy",
    )
    parser.add_argument(
        "--shrink01_logits_pattern",
        default="ea_sh01_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy",
    )
    parser.add_argument(
        "--shrink003_logits_pattern",
        default="ea_sh003_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy",
    )
    parser.add_argument(
        "--partial_logits_pattern",
        default="partial_ea_p05_sh01_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy",
    )
    args = parser.parse_args()

    metadata = build_tables(args)
    print("-" * 80)
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
