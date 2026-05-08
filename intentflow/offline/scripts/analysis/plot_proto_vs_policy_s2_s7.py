"""Plot S2/S7 comparison across the four variants from 260505 sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir",
        default="intentflow/offline/results/proto_vs_policy_safe_s2_s7_20260505_113330",
    )
    parser.add_argument(
        "--out_dir",
        default="docs/research_progress/figures/260505",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads((sweep_dir / "summary.json").read_text())
    variants = ["source_only", "proto_otta_default", "policy_safe_default", "policy_safe_no_shallow"]
    s2 = {v: next((r["acc"] for r in rows if r["variant"] == v and r["subject"] == 2), None) for v in variants}
    s7 = {v: next((r["acc"] for r in rows if r["variant"] == v and r["subject"] == 7), None) for v in variants}

    # Plot 1: grouped bars (S2/S7 per variant) with delta annotations.
    src_s2 = s2["source_only"]
    src_s7 = s7["source_only"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(variants))
    w = 0.36
    s2_bars = ax.bar(x - w / 2, [s2[v] for v in variants], width=w, label="S2 (harm subject)", color="#d96459")
    s7_bars = ax.bar(x + w / 2, [s7[v] for v in variants], width=w, label="S7 (gain subject)", color="#588c7e")

    for v_idx, v in enumerate(variants):
        if v == "source_only":
            continue
        d_s2 = s2[v] - src_s2
        d_s7 = s7[v] - src_s7
        ax.annotate(
            f"{d_s2:+.2f}",
            xy=(v_idx - w / 2, s2[v]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#7a1a14" if d_s2 < 0 else "#1c4f1c",
        )
        ax.annotate(
            f"{d_s7:+.2f}",
            xy=(v_idx + w / 2, s7[v]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#7a1a14" if d_s7 < 0 else "#1c4f1c",
        )

    ax.axhline(src_s2, color="#d96459", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(src_s7, color="#588c7e", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9)
    ax.set_ylabel("test accuracy [%]")
    ax.set_title("260505 S2/S7 sweep: variant × subject (Δ vs source_only annotated)")
    ax.set_ylim(55, 92)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "260505_s2s7_grouped_bars.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")

    # Plot 2: delta-vs-source scatter (S2 Δ on x, S7 Δ on y) with quadrants.
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    colors = {
        "proto_otta_default": "#3b6cb7",
        "policy_safe_default": "#a05ab8",
        "policy_safe_no_shallow": "#e08e1a",
    }
    for v in variants:
        if v == "source_only":
            continue
        dx = s2[v] - src_s2
        dy = s7[v] - src_s7
        ax.scatter(dx, dy, s=110, color=colors[v], edgecolor="black", linewidth=0.7, zorder=3, label=v)
        ax.annotate(v, xy=(dx, dy), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    lim = 4.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.fill_between([0, lim], 0, lim, color="#588c7e", alpha=0.08)
    ax.text(lim - 0.2, lim - 0.2, "rescue both", ha="right", va="top", fontsize=10, color="#1c4f1c")
    ax.text(-lim + 0.2, -lim + 0.2, "harm both", ha="left", va="bottom", fontsize=10, color="#7a1a14")
    ax.set_xlabel("ΔS2 vs source_only [pp]")
    ax.set_ylabel("ΔS7 vs source_only [pp]")
    ax.set_title("Subject-conditional polarity (S2 vs S7)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path2 = out_dir / "260505_polarity_scatter.png"
    fig.savefig(out_path2, dpi=180)
    plt.close(fig)
    print(f"saved {out_path2}")


if __name__ == "__main__":
    main()
