#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    return float(v)


def to_int(v):
    return int(v)


def build_subject_tables(summary):
    rows = summary["subject_results"]
    gains = sorted({float(r["gain_std"]) for r in rows})
    subjects = sorted({int(r["subject_id"]) for r in rows})

    both = np.zeros((len(subjects), len(gains)), dtype=float)
    a1 = np.zeros_like(both)
    a2 = np.zeros_like(both)
    annot = [["" for _ in gains] for _ in subjects]

    for r in rows:
        gi = gains.index(float(r["gain_std"]))
        si = subjects.index(int(r["subject_id"]))
        both[si, gi] = 1.0 if r["both_pass"] else 0.0
        a1[si, gi] = int(r["a1_positive_shallow"])
        a2[si, gi] = int(r["a2_positive_shallow"])
        annot[si][gi] = f"{int(r['a1_positive_shallow'])}/{int(r['a2_positive_shallow'])}"
    return subjects, gains, both, a1, a2, annot


def build_shallow_subject_improvement(summary):
    rows = summary["subject_results"]
    gains = sorted({float(r["gain_std"]) for r in rows})
    subjects = sorted({int(r["subject_id"]) for r in rows})
    mat = np.zeros((len(subjects), len(gains)), dtype=float)

    for r in rows:
        si = subjects.index(int(r["subject_id"]))
        gi = gains.index(float(r["gain_std"]))
        shallow = [x["improvement"] for x in r["layers"] if x["bucket"] == "shallow"]
        mat[si, gi] = float(np.mean(shallow))
    return subjects, gains, mat


def build_layer_stats(csv_rows):
    shallow_rows = [r for r in csv_rows if r["bucket"] == "shallow"]
    gains = sorted({to_float(r["gain_std"]) for r in shallow_rows})
    layer_order = sorted(
        {(to_int(r["layer_index"]), r["layer_name"]) for r in shallow_rows},
        key=lambda x: x[0],
    )
    layer_names = [name for _, name in layer_order]

    impr = {name: [] for name in layer_names}
    cos = {name: [] for name in layer_names}

    for gain in gains:
        gain_rows = [r for r in shallow_rows if abs(to_float(r["gain_std"]) - gain) < 1e-12]
        for name in layer_names:
            layer_rows = [r for r in gain_rows if r["layer_name"] == name]
            impr[name].append(np.mean([to_float(r["improvement"]) for r in layer_rows]))
            cos[name].append(np.mean([to_float(r["cos_sim"]) for r in layer_rows]))

    return gains, layer_names, impr, cos


def plot_summary(summary, out_path: Path):
    subjects, gains, both, a1, a2, annot = build_subject_tables(summary)
    _, _, subj_impr = build_shallow_subject_improvement(summary)
    gain_keys = [f"{g:.2f}" for g in gains]

    summary_gain = summary["summary_by_gain"]
    a1_pass = [summary_gain[f"{g:.6f}"]["a1_pass"] for g in gains]
    a2_pass = [summary_gain[f"{g:.6f}"]["a2_pass"] for g in gains]
    both_pass = [summary_gain[f"{g:.6f}"]["both_pass"] for g in gains]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(gains))
    w = 0.24
    ax.bar(x - w, a1_pass, width=w, label="A1 pass", color="#4C78A8")
    ax.bar(x, a2_pass, width=w, label="A2 pass", color="#F58518")
    ax.bar(x + w, both_pass, width=w, label="both pass", color="#54A24B")
    ax.set_xticks(x, gain_keys)
    ax.set_ylim(0, 9)
    ax.set_ylabel("subjects passed")
    ax.set_xlabel("gain std")
    ax.set_title("Pass Count by Gain")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    im = ax.imshow(both, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(gains)), gain_keys)
    ax.set_yticks(np.arange(len(subjects)), [f"S{s}" for s in subjects])
    ax.set_xlabel("gain std")
    ax.set_title("Subject-wise Pass Map\n(annotation = A1/A2 positive shallow layers)")
    for i in range(len(subjects)):
        for j in range(len(gains)):
            color = "black" if both[i, j] < 0.5 else "white"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    vmax = max(abs(subj_impr.min()), abs(subj_impr.max())) or 1e-6
    im2 = ax.imshow(subj_impr, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(gains)), gain_keys)
    ax.set_yticks(np.arange(len(subjects)), [f"S{s}" for s in subjects])
    ax.set_xlabel("gain std")
    ax.set_title("Mean Shallow Improvement\n(KL(clean,eval) - KL(aug,eval))")
    for i in range(len(subjects)):
        for j in range(len(gains)):
            ax.text(j, i, f"{subj_impr[i, j]:+.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Phase A Proxy Validation Summary", fontsize=14)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layers(csv_rows, out_path: Path):
    gains, layer_names, impr, cos = build_layer_stats(csv_rows)
    gain_labels = [f"{g:.2f}" for g in gains]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    ax = axes[0]
    for name in layer_names:
        ax.plot(gain_labels, impr[name], marker="o", linewidth=2, label=name)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Shallow Layer Mean Improvement")
    ax.set_xlabel("gain std")
    ax.set_ylabel("mean improvement")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    for name in layer_names:
        ax.plot(gain_labels, cos[name], marker="o", linewidth=2, label=name)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Shallow Layer Mean Drift Cosine")
    ax.set_xlabel("gain std")
    ax.set_ylabel("mean cos_sim")
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Phase A Shallow-Layer Trend", fontsize=14)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--layer_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="phaseA_proxy_validation_20260408")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(Path(args.summary_json))
    csv_rows = load_rows(Path(args.layer_csv))

    summary_path = output_dir / f"{args.prefix}_summary.png"
    layer_path = output_dir / f"{args.prefix}_layers.png"

    plot_summary(summary, summary_path)
    plot_layers(csv_rows, layer_path)

    print(summary_path)
    print(layer_path)


if __name__ == "__main__":
    main()
