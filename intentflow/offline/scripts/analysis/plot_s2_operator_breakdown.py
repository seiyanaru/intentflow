"""Plot per-trial diagnostics from the S2 policy_safe_otta run (260429)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="intentflow/offline/results/tcformer_policy_safe_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2252/policy_safe_otta_stats_s2_tcformer_policy_safe_otta.npz",
    )
    parser.add_argument(
        "--out_dir",
        default="docs/research_progress/figures/260505",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)

    op = d["operator"].astype(str)
    counts = Counter(op)
    order = ["abstain", "no_update", "prototype_update", "logit_bias_update", "deep_BN_update", "hybrid_BN_update", "shallow_var_update"]
    labels = [k for k in order if counts[k] > 0]
    values = [counts[k] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0]})

    bars = axes[0].barh(labels[::-1], values[::-1], color="#3b6cb7")
    for b, v in zip(bars, values[::-1]):
        axes[0].text(v + 1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    axes[0].set_xlabel("count over 288 trials")
    axes[0].set_title("S2 / policy_safe_otta operator breakdown\n(committed and abstained both shown)")
    axes[0].grid(axis="x", alpha=0.3)

    energy_z = d.get("energy_z", None)
    abstained = d["abstained"].astype(bool)
    committed = d["committed"].astype(bool)
    correct = d["correct"].astype(bool)
    orig_correct = d["original_correct"].astype(bool)

    cats = [
        ("abstain", abstained),
        ("commit", committed),
        ("no_update", ~abstained & ~committed),
    ]
    bar_labels = []
    correct_pct = []
    orig_pct = []
    n_each = []
    for name, mask in cats:
        n = int(mask.sum())
        n_each.append(n)
        bar_labels.append(f"{name}\n(n={n})")
        if n == 0:
            correct_pct.append(0)
            orig_pct.append(0)
        else:
            correct_pct.append(correct[mask].mean() * 100)
            orig_pct.append(orig_correct[mask].mean() * 100)

    x = np.arange(len(cats))
    w = 0.36
    axes[1].bar(x - w / 2, orig_pct, width=w, label="acc(original_pred)", color="#a8a8a8")
    axes[1].bar(x + w / 2, correct_pct, width=w, label="acc(final_pred)", color="#d96459")
    for i in range(len(cats)):
        axes[1].text(x[i] - w / 2, orig_pct[i] + 1, f"{orig_pct[i]:.0f}", ha="center", fontsize=9)
        axes[1].text(x[i] + w / 2, correct_pct[i] + 1, f"{correct_pct[i]:.0f}", ha="center", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bar_labels)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("acc [%]")
    axes[1].set_title("S2 outcome by gate decision\nfinal_pred ≡ original_pred for every trial")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "260505_s2_operator_breakdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
