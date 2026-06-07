"""Evaluate a label-free adaptive selector over source/full-EA/shrink-EA logits.

This is an analysis script, not a final online implementation. It uses labels
only to report accuracy after the selector has made its choice from unlabeled
statistics:

1. Prefer full EA when its predicted distribution looks sane relative to source.
2. Reject full EA when it creates class-prior collapse or confidence/entropy
   degradation.
3. If full EA is rejected, try shrink-EA candidates; otherwise fall back source.

The intent is to quantify whether Reliability-Aware Alignment has enough
headroom before moving the rule into the online inference path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class Candidate:
    name: str
    path: str
    acc: float
    prior_kl: float
    dominance: float
    entropy: float
    confidence: float
    margin: float
    counts: list[int]
    rejected: bool = False
    reject_reason: str = ""


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def candidate_from_probs(name: str, path: str, probs: np.ndarray, labels: np.ndarray) -> Candidate:
    pred = probs.argmax(axis=1)
    counts = np.bincount(pred, minlength=4)
    prior = counts / counts.sum()
    top2 = np.sort(probs, axis=1)[:, -2:]
    prior_kl = float((prior * np.log(np.clip(prior, 1e-12, 1.0) / 0.25)).sum())
    return Candidate(
        name=name,
        path=path,
        acc=float((pred == labels).mean() * 100.0),
        prior_kl=prior_kl,
        dominance=float(prior.max()),
        entropy=float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean()),
        confidence=float(probs.max(axis=1).mean()),
        margin=float((top2[:, 1] - top2[:, 0]).mean()),
        counts=counts.astype(int).tolist(),
    )


def prediction_metrics(logits_path: Path, labels: np.ndarray) -> Candidate:
    probs = softmax(np.load(logits_path))
    return candidate_from_probs("", str(logits_path), probs, labels)


def fused_candidate(
    name: str,
    first: Candidate,
    second: Candidate,
    labels: np.ndarray,
) -> Candidate:
    first_probs = softmax(np.load(first.path))
    second_probs = softmax(np.load(second.path))
    probs = 0.5 * (first_probs + second_probs)
    return candidate_from_probs(name, f"mean({first.name},{second.name})", probs, labels)


def latest_dir(pattern: str) -> Optional[Path]:
    matches = sorted(Path().glob(pattern))
    return matches[-1] if matches else None


def source_candidate(results_root: Path, sid: int, labels: np.ndarray) -> Candidate:
    path = (
        results_root
        / "c_aug_true_9subj_20260506_004923"
        / "sources"
        / f"s{sid}"
        / f"logits_s{sid}_tcformer_policy_safe_otta.npy"
    )
    cand = prediction_metrics(path, labels)
    cand.name = "source"
    return cand


def ea_candidate(
    results_root: Path,
    sid: int,
    name: str,
    dir_pattern: str,
    labels: np.ndarray,
) -> Optional[Candidate]:
    rel = str(results_root / dir_pattern.format(sid=sid))
    run_dir = latest_dir(rel)
    if run_dir is None:
        return None
    path = run_dir / f"logits_s{sid}_TCFormer.npy"
    if not path.exists():
        return None
    cand = prediction_metrics(path, labels)
    cand.name = name
    return cand


def load_labels(results_root: Path, sid: int) -> Optional[np.ndarray]:
    run_dir = latest_dir(str(results_root / f"ea_aware_tcformer_s{sid}_seed0_*"))
    if run_dir is None:
        return None
    path = run_dir / f"features_s{sid}_TCFormer.npz"
    if not path.exists():
        return None
    return np.load(path)["labels"]


def reject_candidate(cand: Candidate, source: Candidate, args: argparse.Namespace) -> None:
    dkl = cand.prior_kl - source.prior_kl
    dent = cand.entropy - source.entropy
    dconf = cand.confidence - source.confidence
    reasons = []
    if cand.dominance > args.max_dominance:
        reasons.append(f"dominance>{args.max_dominance:g}")
    if dkl > args.max_delta_kl:
        reasons.append(f"delta_kl>{args.max_delta_kl:g}")
    if dent > args.max_entropy_increase and dconf < -args.min_conf_drop:
        reasons.append("entropy_up_conf_down")
    cand.rejected = bool(reasons)
    cand.reject_reason = ",".join(reasons)


def reject_by_covariance(
    candidates: Dict[str, Candidate],
    cov_row: Optional[dict],
    args: argparse.Namespace,
) -> None:
    if cov_row is None:
        return
    test_cond = cov_row["test"]["condition"]
    for name in ("shrink_0.1", "partial_0.5_shrink_0.1"):
        cand = candidates.get(name)
        if cand is None:
            continue
        if test_cond > args.max_shrink_condition:
            cand.rejected = True
            reason = f"test_condition>{args.max_shrink_condition:g}"
            cand.reject_reason = (
                f"{cand.reject_reason},{reason}" if cand.reject_reason else reason
            )


def select_candidate(
    candidates: Dict[str, Candidate],
    cov_row: Optional[dict],
    args: argparse.Namespace,
    labels: np.ndarray,
) -> Candidate:
    source = candidates["source"]
    for name, cand in candidates.items():
        if name != "source":
            reject_candidate(cand, source, args)
    reject_by_covariance(candidates, cov_row, args)

    shrink = candidates.get("shrink_0.1")
    if (
        cov_row is not None
        and cov_row["test"]["diag_cv"] > args.prefer_shrink_diag_cv
        and shrink is not None
        and not shrink.rejected
    ):
        return shrink

    full = candidates.get("full_ea")
    if (
        args.fuse_full_shrink
        and full is not None
        and shrink is not None
        and not full.rejected
        and not shrink.rejected
    ):
        fused = fused_candidate("full_shrink_mean", full, shrink, labels)
        reject_candidate(fused, source, args)
        if not fused.rejected:
            candidates[fused.name] = fused
            return fused

    if full is not None and not full.rejected:
        return full

    # If full EA fails, prefer the shrinkage-only model. Partial EA is kept as a
    # secondary fallback because it helped S7 but hurt S2 in the current data.
    for fallback_name in ("shrink_0.1", "partial_0.5_shrink_0.1"):
        cand = candidates.get(fallback_name)
        if cand is not None and not cand.rejected:
            return cand

    return source


def subjects_from_arg(arg: str) -> Iterable[int]:
    if arg == "all":
        return range(1, 10)
    return [int(x) for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="intentflow/offline/results")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_delta_kl", type=float, default=0.05)
    parser.add_argument("--max_dominance", type=float, default=0.42)
    parser.add_argument("--max_entropy_increase", type=float, default=0.11)
    parser.add_argument("--min_conf_drop", type=float, default=0.05)
    parser.add_argument("--cov_diagnostics", default=None)
    parser.add_argument("--max_shrink_condition", type=float, default=30000.0)
    parser.add_argument("--prefer_shrink_diag_cv", type=float, default=0.15)
    parser.add_argument("--fuse_full_shrink", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    cov_by_subject = {}
    if args.cov_diagnostics:
        cov_obj = json.load(open(args.cov_diagnostics))
        cov_by_subject = {int(row["subject"]): row for row in cov_obj["rows"]}

    rows = []
    for sid in subjects_from_arg(args.subjects):
        labels = load_labels(results_root, sid)
        if labels is None:
            print(f"S{sid}: labels not found, skip")
            continue

        candidates: Dict[str, Candidate] = {
            "source": source_candidate(results_root, sid, labels)
        }
        for name, pattern in [
            ("full_ea", "ea_aware_tcformer_s{sid}_seed0_*"),
            ("shrink_0.1", "ea_sh01_tcformer_s{sid}_seed0_*"),
            ("partial_0.5_shrink_0.1", "partial_ea_p05_sh01_tcformer_s{sid}_seed0_*"),
        ]:
            cand = ea_candidate(results_root, sid, name, pattern, labels)
            if cand is not None:
                candidates[name] = cand

        cov_row = cov_by_subject.get(sid)
        selected = select_candidate(candidates, cov_row, args, labels)
        source = candidates["source"]
        full = candidates.get("full_ea")
        row = {
            "subject": sid,
            "selected": selected.name,
            "selected_acc": selected.acc,
            "source_acc": source.acc,
            "full_ea_acc": None if full is None else full.acc,
            "delta_vs_source": selected.acc - source.acc,
            "delta_vs_full_ea": None if full is None else selected.acc - full.acc,
            "covariance": cov_row,
            "candidates": {name: asdict(cand) for name, cand in candidates.items()},
        }
        rows.append(row)
        full_acc = "NA" if full is None else f"{full.acc:5.2f}"
        print(
            f"S{sid}: select={selected.name:22} acc={selected.acc:5.2f} "
            f"source={source.acc:5.2f} full={full_acc} "
            f"d_src={selected.acc - source.acc:+5.2f}"
        )
        for cand in candidates.values():
            if cand.name == "source":
                continue
            mark = "REJECT" if cand.rejected else "pass"
            print(
                f"  {cand.name:22} {mark:6} acc={cand.acc:5.2f} "
                f"kl={cand.prior_kl:.3f} dom={cand.dominance:.3f} "
                f"ent={cand.entropy:.3f} conf={cand.confidence:.3f} "
                f"{cand.reject_reason}"
            )
        if cov_row is not None:
            print(
                f"  cov: cond={cov_row['test']['condition']:.1f} "
                f"diag_cv={cov_row['test']['diag_cv']:.3f} "
                f"diag_ratio={cov_row['test']['diag_ratio']:.2f} "
                f"cov_dist={cov_row['cov_distance']:.3f}"
            )

    if not rows:
        raise SystemExit("No rows evaluated.")

    src_mean = float(np.mean([r["source_acc"] for r in rows]))
    full_rows = [r for r in rows if r["full_ea_acc"] is not None]
    full_mean = float(np.mean([r["full_ea_acc"] for r in full_rows]))
    selected_mean = float(np.mean([r["selected_acc"] for r in rows]))
    print("-" * 80)
    print(f"source mean:   {src_mean:5.2f}")
    print(f"full EA mean:  {full_mean:5.2f}")
    print(f"selected mean: {selected_mean:5.2f}")
    print(f"delta selected-source: {selected_mean - src_mean:+5.2f} pp")
    print(f"delta selected-fullEA: {selected_mean - full_mean:+5.2f} pp")

    if args.output:
        out = {
            "thresholds": {
                "max_delta_kl": args.max_delta_kl,
                "max_dominance": args.max_dominance,
                "max_entropy_increase": args.max_entropy_increase,
                "min_conf_drop": args.min_conf_drop,
                "max_shrink_condition": args.max_shrink_condition,
                "prefer_shrink_diag_cv": args.prefer_shrink_diag_cv,
            },
            "summary": {
                "source_mean": src_mean,
                "full_ea_mean": full_mean,
                "selected_mean": selected_mean,
                "delta_selected_source": selected_mean - src_mean,
                "delta_selected_full_ea": selected_mean - full_mean,
            },
            "rows": rows,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
