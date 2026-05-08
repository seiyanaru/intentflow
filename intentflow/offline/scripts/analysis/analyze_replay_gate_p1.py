"""P1 analysis for replay-based SafeCommit.

This script answers a narrower question than the aggregate accuracy tables:

1. Did the replay gate actually reject candidate operators?
2. Which reject reasons dominated?
3. Relative to policy_safe_no_shallow, where did replay gain or lose trials?

The trial-level "good_reject_proxy" and "over_reject_proxy" columns are
proxies, not causal counterfactuals: policy and replay trajectories can diverge
after earlier commits. They are still useful for P1 because they identify
concrete trials where replay rejected a candidate and the final outcome differed
from the policy baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_REPLAY_VARIANTS = ("replay_safe_uniform", "replay_h6_weighted")
DEFAULT_POLICY_VARIANT = "policy_safe_no_shallow"
DEFAULT_SOURCE_VARIANT = "source_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="intentflow/offline/results/c_aug_true_9subj_20260506_004923",
        help="Result directory containing eval/<unit>/<variant> outputs.",
    )
    parser.add_argument("--policy-variant", default=DEFAULT_POLICY_VARIANT)
    parser.add_argument("--source-variant", default=DEFAULT_SOURCE_VARIANT)
    parser.add_argument("--replay-variants", nargs="+", default=list(DEFAULT_REPLAY_VARIANTS))
    parser.add_argument(
        "--out-prefix",
        default="p1_replay_gate_analysis",
        help="Output prefix under --result-dir.",
    )
    parser.add_argument(
        "--focus-units",
        nargs="*",
        default=("s2", "s7", "s2_seed1", "s2_seed2", "s2_seed3", "s7_seed1", "s7_seed2", "s7_seed3"),
        help="Units for which representative gain/loss examples are printed.",
    )
    return parser.parse_args()


def parse_acc(result_dir: Path) -> Optional[float]:
    """Parse Test Acc from a train_pipeline result directory as percent."""
    path = result_dir / "results.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        match = re.search(r"Test Acc:\s+([\d.]+)", line)
        if match:
            return float(match.group(1)) * 100.0
    return None


def find_npz(result_dir: Path, unit: str, variant: str) -> Optional[Path]:
    variant_dir = result_dir / "eval" / unit / variant
    if not variant_dir.exists():
        return None
    matches = sorted(variant_dir.glob("*.npz"))
    return matches[0] if matches else None


def finite_bool_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        f = arr.astype(float)
        return np.isfinite(f) & (f > 0.5)
    return arr.astype(bool)


def string_array(values: np.ndarray, n: int, default: str = "") -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=object)
    arr = np.asarray(values).astype(str)
    if arr.shape[0] != n:
        raise ValueError(f"Expected array length {n}, got {arr.shape[0]}")
    return arr


def split_pipe(value: str) -> List[str]:
    if value is None:
        return []
    value = str(value)
    if value == "" or value.lower() == "nan":
        return []
    return value.split("|")


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_trial_data(npz_path: Path) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    correct = finite_bool_array(d["correct"])
    n = int(correct.shape[0])
    labels = np.asarray(d["label"]).astype(int) if "label" in d.files else np.full(n, -1, dtype=int)
    preds = np.asarray(d["pred"]).astype(int) if "pred" in d.files else np.full(n, -1, dtype=int)
    committed = finite_bool_array(d["committed"]) if "committed" in d.files else np.zeros(n, dtype=bool)
    abstained = finite_bool_array(d["abstained"]) if "abstained" in d.files else np.zeros(n, dtype=bool)
    operator = string_array(d["operator"] if "operator" in d.files else None, n, default="")
    reason = string_array(d["safecommit_reason"] if "safecommit_reason" in d.files else None, n, default="")

    data = {
        "path": str(npz_path),
        "n": n,
        "correct": correct,
        "labels": labels,
        "preds": preds,
        "committed": committed,
        "abstained": abstained,
        "operator": operator,
        "reason": reason,
        "candidate_ops": string_array(d["candidate_ops"] if "candidate_ops" in d.files else None, n, default=""),
        "candidate_reasons": string_array(
            d["candidate_reasons"] if "candidate_reasons" in d.files else None, n, default=""
        ),
        "candidate_tiers": string_array(d["candidate_tiers"] if "candidate_tiers" in d.files else None, n, default=""),
        "candidate_passes": string_array(
            d["candidate_passes"] if "candidate_passes" in d.files else None, n, default=""
        ),
        "candidate_sim_scores": string_array(
            d["candidate_sim_scores"] if "candidate_sim_scores" in d.files else None, n, default=""
        ),
    }

    for key in ("sim_score", "replay_admitted", "replay_safe_pass", "replay_buffer_size"):
        if key in d.files:
            data[key] = np.asarray(d[key])
    return data


def iter_candidates(trial_data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    n = trial_data["n"]
    for i in range(n):
        ops = split_pipe(trial_data["candidate_ops"][i])
        reasons = split_pipe(trial_data["candidate_reasons"][i])
        tiers = split_pipe(trial_data["candidate_tiers"][i])
        passes = split_pipe(trial_data["candidate_passes"][i])
        sims = split_pipe(trial_data["candidate_sim_scores"][i])
        m = max(len(ops), len(reasons), len(tiers), len(passes), len(sims))
        for j in range(m):
            pass_token = passes[j] if j < len(passes) else ""
            if pass_token not in ("0", "1"):
                passed = None
            else:
                passed = pass_token == "1"
            yield {
                "trial": i,
                "op": ops[j] if j < len(ops) else "",
                "reason": reasons[j] if j < len(reasons) else "",
                "tier": tiers[j] if j < len(tiers) else "",
                "passed": passed,
                "sim_score": parse_float(sims[j] if j < len(sims) else ""),
            }


def has_replay_sim_reject(trial_data: Dict[str, Any], idx: int) -> bool:
    reasons = split_pipe(trial_data["candidate_reasons"][idx])
    passes = split_pipe(trial_data["candidate_passes"][idx])
    for reason, passed in zip(reasons, passes):
        if passed == "0" and reason == "replay_sim_score_drop":
            return True
    return False


def trial_candidate_string(trial_data: Dict[str, Any], idx: int) -> str:
    ops = split_pipe(trial_data["candidate_ops"][idx])
    reasons = split_pipe(trial_data["candidate_reasons"][idx])
    passes = split_pipe(trial_data["candidate_passes"][idx])
    sims = split_pipe(trial_data["candidate_sim_scores"][idx])
    parts = []
    for j, op in enumerate(ops):
        reason = reasons[j] if j < len(reasons) else ""
        passed = passes[j] if j < len(passes) else ""
        sim = sims[j] if j < len(sims) else ""
        parts.append(f"{op}:{passed}:{reason}:{sim}")
    return "; ".join(parts)


def summarize_variant(data: Dict[str, Any]) -> Dict[str, Any]:
    candidates = list(iter_candidates(data))
    passed = [c for c in candidates if c["passed"] is True]
    rejected = [c for c in candidates if c["passed"] is False]
    replay_rejected = [c for c in rejected if c["reason"] == "replay_sim_score_drop"]
    replay_rejected_sims = [c["sim_score"] for c in replay_rejected if math.isfinite(c["sim_score"])]
    passed_sims = [c["sim_score"] for c in passed if math.isfinite(c["sim_score"])]
    sim_commit = []
    if "sim_score" in data:
        sim = np.asarray(data["sim_score"]).astype(float)
        mask = np.isfinite(sim) & data["committed"]
        sim_commit = sim[mask].tolist()
    admitted = None
    if "replay_admitted" in data:
        admitted = int(finite_bool_array(data["replay_admitted"]).sum())
    return {
        "n": data["n"],
        "acc": float(data["correct"].mean() * 100.0),
        "commits": int(data["committed"].sum()),
        "abstains": int(data["abstained"].sum()),
        "admitted": admitted,
        "candidates": len(candidates),
        "candidate_pass": len(passed),
        "candidate_reject": len(rejected),
        "candidate_reject_rate": float(len(rejected) / len(candidates)) if candidates else 0.0,
        "replay_sim_reject": len(replay_rejected),
        "reject_reasons": dict(Counter(c["reason"] for c in rejected)),
        "reject_ops": dict(Counter(c["op"] for c in rejected)),
        "replay_sim_reject_ops": dict(Counter(c["op"] for c in replay_rejected)),
        "pass_ops": dict(Counter(c["op"] for c in passed)),
        "sim_commit_mean": float(np.mean(sim_commit)) if sim_commit else None,
        "sim_commit_min": float(np.min(sim_commit)) if sim_commit else None,
        "sim_commit_max": float(np.max(sim_commit)) if sim_commit else None,
        "replay_sim_reject_mean": float(np.mean(replay_rejected_sims)) if replay_rejected_sims else None,
        "replay_sim_reject_min": float(np.min(replay_rejected_sims)) if replay_rejected_sims else None,
        "replay_sim_reject_max": float(np.max(replay_rejected_sims)) if replay_rejected_sims else None,
        "candidate_pass_sim_mean": float(np.mean(passed_sims)) if passed_sims else None,
        "candidate_pass_sim_min": float(np.min(passed_sims)) if passed_sims else None,
        "candidate_pass_sim_max": float(np.max(passed_sims)) if passed_sims else None,
    }


def compare_policy_replay(policy: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    if policy["n"] != replay["n"]:
        raise ValueError(f"Trial length mismatch: policy={policy['n']} replay={replay['n']}")
    if not np.array_equal(policy["labels"], replay["labels"]):
        raise ValueError("Label arrays do not match; cannot compare trial-by-trial.")

    p_ok = policy["correct"]
    r_ok = replay["correct"]
    gain_idx = np.flatnonzero((~p_ok) & r_ok)
    loss_idx = np.flatnonzero(p_ok & (~r_ok))
    same_correct = int((p_ok & r_ok).sum())
    same_wrong = int(((~p_ok) & (~r_ok)).sum())

    def count_with(mask_idx: Sequence[int], predicate) -> int:
        return int(sum(1 for i in mask_idx if predicate(int(i))))

    good_reject_proxy = count_with(
        gain_idx,
        lambda i: bool(policy["committed"][i]) and has_replay_sim_reject(replay, i),
    )
    over_reject_proxy = count_with(
        loss_idx,
        lambda i: bool(policy["committed"][i]) and has_replay_sim_reject(replay, i),
    )
    gain_replay_commit = count_with(gain_idx, lambda i: bool(replay["committed"][i]))
    loss_replay_commit = count_with(loss_idx, lambda i: bool(replay["committed"][i]))
    gain_policy_commit = count_with(gain_idx, lambda i: bool(policy["committed"][i]))
    loss_policy_commit = count_with(loss_idx, lambda i: bool(policy["committed"][i]))

    return {
        "policy_acc": float(p_ok.mean() * 100.0),
        "replay_acc": float(r_ok.mean() * 100.0),
        "delta_replay_minus_policy": float((r_ok.mean() - p_ok.mean()) * 100.0),
        "gain_trials": int(len(gain_idx)),
        "loss_trials": int(len(loss_idx)),
        "net_trials": int(len(gain_idx) - len(loss_idx)),
        "same_correct": same_correct,
        "same_wrong": same_wrong,
        "good_reject_proxy": good_reject_proxy,
        "over_reject_proxy": over_reject_proxy,
        "gain_replay_commit": gain_replay_commit,
        "loss_replay_commit": loss_replay_commit,
        "gain_policy_commit": gain_policy_commit,
        "loss_policy_commit": loss_policy_commit,
        "gain_indices": gain_idx[:12].astype(int).tolist(),
        "loss_indices": loss_idx[:12].astype(int).tolist(),
    }


def example_rows(policy: Dict[str, Any], replay: Dict[str, Any], indices: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for idx in indices:
        i = int(idx)
        rows.append(
            {
                "trial": i,
                "label": int(policy["labels"][i]),
                "policy_pred": int(policy["preds"][i]),
                "policy_op": str(policy["operator"][i]),
                "policy_committed": bool(policy["committed"][i]),
                "replay_pred": int(replay["preds"][i]),
                "replay_op": str(replay["operator"][i]),
                "replay_committed": bool(replay["committed"][i]),
                "replay_candidates": trial_candidate_string(replay, i),
            }
        )
    return rows


def discover_units(result_dir: Path, variants: Sequence[str]) -> List[str]:
    eval_dir = result_dir / "eval"
    units = []
    for unit_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        if all((unit_dir / v).exists() for v in variants):
            units.append(unit_dir.name)
    return units


def mean(values: Sequence[float]) -> Optional[float]:
    nums = [v for v in values if v is not None and not math.isnan(v)]
    return float(sum(nums) / len(nums)) if nums else None


def fmt(value: Optional[float], nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{nd}f}"


def write_markdown(
    path: Path,
    result_dir: Path,
    policy_variant: str,
    source_variant: str,
    replay_variants: Sequence[str],
    payload: Dict[str, Any],
    focus_units: Sequence[str],
) -> None:
    lines = [
        "# P1 Replay Gate Analysis",
        "",
        f"Result dir: `{result_dir}`",
        f"Policy baseline: `{policy_variant}`",
        f"Source baseline: `{source_variant}`",
        "",
        "## Aggregate Accuracy",
        "",
        "| variant | mean_acc | mean_delta_vs_source | worst_delta_vs_source | units |",
        "|---|---:|---:|---:|---:|",
    ]

    for variant, row in payload["aggregate_accuracy"].items():
        lines.append(
            f"| {variant} | {fmt(row['mean_acc'])} | {fmt(row['mean_delta_vs_source'])} | "
            f"{fmt(row['worst_delta_vs_source'])} | {row['n_units']} |"
        )

    lines += [
        "",
        "## Replay Candidate Gate",
        "",
        "| variant | candidates | pass | reject | reject_rate | replay_sim_reject | commits | admitted | sim_commit_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in replay_variants:
        row = payload["gate_summary"][variant]
        lines.append(
            f"| {variant} | {row['candidates']} | {row['candidate_pass']} | {row['candidate_reject']} | "
            f"{row['candidate_reject_rate'] * 100.0:.1f}% | {row['replay_sim_reject']} | "
            f"{row['commits']} | {row['admitted']} | {fmt(row['sim_commit_mean'], 5)} |"
        )

    lines += [
        "",
        "Reject reason counts:",
    ]
    for variant in replay_variants:
        lines.append(f"- `{variant}`: `{payload['gate_summary'][variant]['reject_reasons']}`")
    lines += [
        "",
        "Replay-sim reject operator counts:",
    ]
    for variant in replay_variants:
        row = payload["gate_summary"][variant]
        lines.append(
            f"- `{variant}`: ops=`{row['replay_sim_reject_ops']}`, "
            f"mean_rejected_sim={fmt(row['replay_sim_reject_mean'], 6)}, "
            f"mean_passed_sim={fmt(row['candidate_pass_sim_mean'], 6)}"
        )

    lines += [
        "",
        "## Policy vs Replay Trial Outcomes",
        "",
        "Proxy columns require both: policy committed on the trial and replay had a `replay_sim_score_drop` candidate.",
        "If both proxy columns are zero, the accuracy difference is trajectory-level rather than a same-trial direct rescue.",
        "",
    ]
    for variant in replay_variants:
        lines += [
            f"### {variant}",
            "",
            "| unit | source | policy | replay | replay-policy | gains | losses | net | good_reject_proxy | over_reject_proxy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for unit, row in payload["comparisons"][variant].items():
            lines.append(
                f"| {unit} | {fmt(row['source_acc'])} | {fmt(row['policy_acc'])} | {fmt(row['replay_acc'])} | "
                f"{fmt(row['delta_replay_minus_policy'])} | {row['gain_trials']} | {row['loss_trials']} | "
                f"{row['net_trials']} | {row['good_reject_proxy']} | {row['over_reject_proxy']} |"
            )
        agg = payload["comparison_aggregate"][variant]
        lines.append(
            f"| **mean/sum** | {fmt(agg['source_mean'])} | {fmt(agg['policy_mean'])} | {fmt(agg['replay_mean'])} | "
            f"{fmt(agg['delta_mean'])} | {agg['gain_trials']} | {agg['loss_trials']} | {agg['net_trials']} | "
            f"{agg['good_reject_proxy']} | {agg['over_reject_proxy']} |"
        )
        lines.append("")

    lines += [
        "## Focus Examples",
        "",
    ]
    for variant in replay_variants:
        examples = payload["examples"][variant]
        if not examples:
            continue
        lines += [f"### {variant}", ""]
        for unit in focus_units:
            if unit not in examples:
                continue
            unit_ex = examples[unit]
            lines += [f"#### {unit}", "", "Gains: policy wrong, replay correct."]
            if unit_ex["gains"]:
                for row in unit_ex["gains"]:
                    lines.append(
                        f"- trial {row['trial']}: y={row['label']}, policy={row['policy_pred']} "
                        f"({row['policy_op']}, commit={row['policy_committed']}), "
                        f"replay={row['replay_pred']} ({row['replay_op']}, commit={row['replay_committed']}), "
                        f"cand=`{row['replay_candidates']}`"
                    )
            else:
                lines.append("- none in first sampled examples")
            lines += ["", "Losses: policy correct, replay wrong."]
            if unit_ex["losses"]:
                for row in unit_ex["losses"]:
                    lines.append(
                        f"- trial {row['trial']}: y={row['label']}, policy={row['policy_pred']} "
                        f"({row['policy_op']}, commit={row['policy_committed']}), "
                        f"replay={row['replay_pred']} ({row['replay_op']}, commit={row['replay_committed']}), "
                        f"cand=`{row['replay_candidates']}`"
                    )
            else:
                lines.append("- none in first sampled examples")
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    variants = [args.source_variant, args.policy_variant, *args.replay_variants]
    units = discover_units(result_dir, variants)
    if not units:
        raise SystemExit(f"No eval units found under {result_dir}/eval with variants {variants}")

    unit_acc: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    unit_data: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for unit in units:
        for variant in variants:
            unit_acc[unit][variant] = parse_acc(result_dir / "eval" / unit / variant)
            npz_path = find_npz(result_dir, unit, variant)
            if npz_path is not None:
                unit_data[unit][variant] = load_trial_data(npz_path)

    aggregate_accuracy = {}
    for variant in variants:
        accs = [unit_acc[u][variant] for u in units if unit_acc[u][variant] is not None]
        deltas = [
            unit_acc[u][variant] - unit_acc[u][args.source_variant]
            for u in units
            if unit_acc[u][variant] is not None and unit_acc[u][args.source_variant] is not None
        ]
        aggregate_accuracy[variant] = {
            "n_units": len(accs),
            "mean_acc": mean(accs),
            "mean_delta_vs_source": mean(deltas),
            "worst_delta_vs_source": min(deltas) if deltas else None,
            "ntr_s_at_0p5": int(sum(1 for d in deltas if d < -0.5)),
        }

    gate_summary = {}
    for variant in args.replay_variants:
        rows = []
        reason_counts = Counter()
        reject_ops = Counter()
        pass_ops = Counter()
        replay_sim_reject_ops = Counter()
        replay_sim_reject_sims = []
        candidate_pass_sims = []
        for unit in units:
            if variant not in unit_data[unit]:
                continue
            row = summarize_variant(unit_data[unit][variant])
            rows.append(row)
            reason_counts.update(row["reject_reasons"])
            reject_ops.update(row["reject_ops"])
            replay_sim_reject_ops.update(row["replay_sim_reject_ops"])
            pass_ops.update(row["pass_ops"])
            for key, sink in (
                ("replay_sim_reject_mean", replay_sim_reject_sims),
                ("candidate_pass_sim_mean", candidate_pass_sims),
            ):
                if row[key] is not None:
                    # Use unit means only for compact reporting.
                    sink.append(row[key])
        candidates = sum(r["candidates"] for r in rows)
        rejects = sum(r["candidate_reject"] for r in rows)
        sim_means = [r["sim_commit_mean"] for r in rows if r["sim_commit_mean"] is not None]
        gate_summary[variant] = {
            "candidates": candidates,
            "candidate_pass": sum(r["candidate_pass"] for r in rows),
            "candidate_reject": rejects,
            "candidate_reject_rate": float(rejects / candidates) if candidates else 0.0,
            "replay_sim_reject": sum(r["replay_sim_reject"] for r in rows),
            "commits": sum(r["commits"] for r in rows),
            "abstains": sum(r["abstains"] for r in rows),
            "admitted": sum(r["admitted"] or 0 for r in rows),
            "reject_reasons": dict(reason_counts),
            "reject_ops": dict(reject_ops),
            "replay_sim_reject_ops": dict(replay_sim_reject_ops),
            "pass_ops": dict(pass_ops),
            "sim_commit_mean": mean(sim_means),
            "replay_sim_reject_mean": mean(replay_sim_reject_sims),
            "candidate_pass_sim_mean": mean(candidate_pass_sims),
        }

    comparisons = {variant: {} for variant in args.replay_variants}
    comparison_aggregate = {variant: {} for variant in args.replay_variants}
    examples = {variant: {} for variant in args.replay_variants}
    for variant in args.replay_variants:
        for unit in units:
            if args.policy_variant not in unit_data[unit] or variant not in unit_data[unit]:
                continue
            comp = compare_policy_replay(unit_data[unit][args.policy_variant], unit_data[unit][variant])
            comp["source_acc"] = unit_acc[unit][args.source_variant]
            # Prefer results.txt accuracy for public numbers; npz is used for trial decomposition.
            comp["policy_acc"] = unit_acc[unit][args.policy_variant] or comp["policy_acc"]
            comp["replay_acc"] = unit_acc[unit][variant] or comp["replay_acc"]
            comp["delta_replay_minus_policy"] = comp["replay_acc"] - comp["policy_acc"]
            comparisons[variant][unit] = comp
            if unit in args.focus_units:
                pdat = unit_data[unit][args.policy_variant]
                rdat = unit_data[unit][variant]
                examples[variant][unit] = {
                    "gains": example_rows(pdat, rdat, comp["gain_indices"][:4]),
                    "losses": example_rows(pdat, rdat, comp["loss_indices"][:4]),
                }

        comp_rows = list(comparisons[variant].values())
        comparison_aggregate[variant] = {
            "source_mean": mean([r["source_acc"] for r in comp_rows]),
            "policy_mean": mean([r["policy_acc"] for r in comp_rows]),
            "replay_mean": mean([r["replay_acc"] for r in comp_rows]),
            "delta_mean": mean([r["delta_replay_minus_policy"] for r in comp_rows]),
            "gain_trials": sum(r["gain_trials"] for r in comp_rows),
            "loss_trials": sum(r["loss_trials"] for r in comp_rows),
            "net_trials": sum(r["net_trials"] for r in comp_rows),
            "good_reject_proxy": sum(r["good_reject_proxy"] for r in comp_rows),
            "over_reject_proxy": sum(r["over_reject_proxy"] for r in comp_rows),
        }

    payload = {
        "result_dir": str(result_dir),
        "units": units,
        "aggregate_accuracy": aggregate_accuracy,
        "gate_summary": gate_summary,
        "comparisons": comparisons,
        "comparison_aggregate": comparison_aggregate,
        "examples": examples,
    }

    json_path = result_dir / f"{args.out_prefix}.json"
    md_path = result_dir / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_markdown(
        md_path,
        result_dir,
        args.policy_variant,
        args.source_variant,
        args.replay_variants,
        payload,
        args.focus_units,
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
