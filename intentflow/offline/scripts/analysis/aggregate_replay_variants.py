"""Aggregate replay OTTA variant sweeps.

Supports:
- Plan C: 9 subjects, seed0, aug-True result tree.
- Plan B: S2/S4/S6/S7, seeds 1-3 result tree.

The output includes mean/worst/NTR, per-subject tables, and a subject/session
oracle over the supplied variants. The oracle is analysis-only; it quantifies
how much room remains for a future unsupervised profile selector.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_VARIANTS = [
    "source_only",
    "policy_safe_no_shallow",
    "replay_safe_uniform",
    "replay_h6_weighted",
    "replay_uniform_no_tier2",
    "replay_uniform_no_shallow",
    "replay_h6_no_shallow",
    "replay_best_uniform",
    "replay_best_h6_weighted",
    "replay_best_uniform_no_shallow",
    "replay_best_h6_no_shallow",
    "replay_uniform_tol_5e5",
    "replay_uniform_tol_1e4",
    "replay_uniform_tol_2e4",
    "replay_h6_tol_5e5",
    "replay_h6_tol_1e4",
    "replay_h6_tol_2e4",
    "replay_proto_bias_only",
    "replay_h6_proto_bias_only",
    "replay_deep_proto_bias_only",
    "replay_h6_deep_proto_bias_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--plan", choices=["c", "b"], required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--out-prefix", default="overnight_replay_summary")
    return parser.parse_args()


def parse_acc(path: Path) -> Optional[float]:
    f = path / "results.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m:
            return float(m.group(1)) * 100.0
    return None


def units_for_plan(plan: str) -> List[Tuple[str, int, Optional[int]]]:
    if plan == "c":
        return [(f"s{s}", s, None) for s in range(1, 10)]
    return [
        (f"s{s}_seed{seed}", s, seed)
        for s in (2, 4, 6, 7)
        for seed in (1, 2, 3)
    ]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def pstdev(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    base = Path(args.result_dir)
    units = units_for_plan(args.plan)

    table: Dict[str, Dict[str, Optional[float]]] = {}
    for variant in args.variants:
        table[variant] = {}
        for unit, _subject, _seed in units:
            table[variant][unit] = parse_acc(base / "eval" / unit / variant)

    source = table.get("source_only", {})
    rows = []
    for variant in args.variants:
        accs = [table[variant][unit] for unit, _s, _seed in units if table[variant][unit] is not None]
        deltas = [
            table[variant][unit] - source[unit]
            for unit, _s, _seed in units
            if table[variant][unit] is not None and source.get(unit) is not None
        ]
        rows.append(
            {
                "variant": variant,
                "n": len(accs),
                "mean_acc": mean(accs),
                "mean_delta_vs_source": mean(deltas),
                "worst_delta_vs_source": min(deltas) if deltas else None,
                "ntr_s_at_0p5": sum(1 for d in deltas if d < -0.5),
                "per_unit": table[variant],
            }
        )

    evaluated_variants = [r["variant"] for r in rows if r["n"] > 0]
    oracle_units = []
    for unit, subject, seed in units:
        vals = [
            (table[v][unit], v)
            for v in evaluated_variants
            if table[v].get(unit) is not None
        ]
        if not vals:
            continue
        best_acc, best_variant = max(vals)
        oracle_units.append(
            {
                "unit": unit,
                "subject": subject,
                "seed": seed,
                "best_acc": best_acc,
                "best_variant": best_variant,
            }
        )

    payload = {
        "plan": args.plan,
        "result_dir": str(base),
        "variants": args.variants,
        "rows": rows,
        "oracle_units": oracle_units,
        "oracle_mean": mean([r["best_acc"] for r in oracle_units]),
    }

    json_path = base / f"{args.out_prefix}.json"
    md_path = base / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    md = [f"# Replay Variant Sweep Summary ({args.plan.upper()})", ""]
    md.append("| variant | n | mean | Δmean vs source | worst vs source | NTR-S |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["n"] == 0:
            continue
        md.append(
            f"| {row['variant']} | {row['n']} | {fmt(row['mean_acc'])} | "
            f"{fmt(row['mean_delta_vs_source'], signed=True)} | "
            f"{fmt(row['worst_delta_vs_source'], signed=True)} | "
            f"{row['ntr_s_at_0p5']}/{row['n']} |"
        )

    if args.plan == "c":
        subjects = list(range(1, 10))
        md.extend(["", "## Per-Subject Accuracy", ""])
        md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
        md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
        for row in rows:
            if row["n"] == 0:
                continue
            cells = [fmt(row["per_unit"].get(f"s{s}")) for s in subjects]
            md.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
    else:
        subjects = [2, 4, 6, 7]
        md.extend(["", "## Per-Subject Mean ± Std", ""])
        md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
        md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
        for row in rows:
            if row["n"] == 0:
                continue
            cells = []
            for s in subjects:
                vals = [
                    row["per_unit"].get(f"s{s}_seed{seed}")
                    for seed in (1, 2, 3)
                    if row["per_unit"].get(f"s{s}_seed{seed}") is not None
                ]
                cells.append("-" if not vals else f"{mean(vals):.2f} ± {pstdev(vals):.2f}")
            md.append(f"| {row['variant']} | " + " | ".join(cells) + " |")

    md.extend(["", "## Variant Oracle", ""])
    md.append(f"Oracle mean over evaluated variants: `{fmt(payload['oracle_mean'])}`")
    md.append("")
    md.append("| unit | best_variant | best_acc |")
    md.append("|---|---|---:|")
    for item in oracle_units:
        md.append(f"| {item['unit']} | {item['best_variant']} | {item['best_acc']:.2f} |")

    md_path.write_text("\n".join(md) + "\n")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
