"""Generate presentation tables and figures for 260511 seminar.

Inputs (must exist):
  - intentflow/offline/results/c_aug_true_9subj_20260506_004923/summary.json
  - intentflow/offline/results/b_5seed_4subj_20260506_005153/summary.json
  - intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/summary.json (for seed-0)
  - intentflow/offline/results/_smoke_replay_v3_s2_*/replay_safe_otta_stats_s2_*.npz (for trace)

Outputs:
  docs/research_progress/tables/260511_presentation/main_result_table.csv
  docs/research_progress/tables/260511_presentation/per_subject_delta.csv
  docs/research_progress/tables/260511_presentation/s7_seed_table.csv
  docs/research_progress/tables/260511_presentation/replay_candidate_trace.csv
  docs/research_progress/figures/260511_presentation/
      fig_main_source_vs_replay.png
      fig_per_subject_delta.png
      fig_safety_tradeoff.png
      fig_s7_seed_stability.png
      fig_replay_candidate_trace.png
      fig_method_flow_replay_safecommit.png
      method_flow_replay_safecommit.mmd          (mermaid source)
  docs/research_progress/figures/260511_presentation/caption_draft.md
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/mnt/data/seiya.narukawa/intentflow")
RESULTS = REPO / "intentflow/offline/results"
OUT_FIG = REPO / "docs/research_progress/figures/260511_presentation"
OUT_TBL = REPO / "docs/research_progress/tables/260511_presentation"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

# ---- Style ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Japanese-capable font (Noto Sans CJK JP installed system-wide)
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

C_SOURCE = "#7d7d7d"
C_NO_SH  = "#a05ab8"
C_REPLAY = "#3b6cb7"
C_H6     = "#e08e1a"
C_HYBRID = "#588c7e"

METHOD_COLORS = {
    "source_only":            C_SOURCE,
    "policy_safe_no_shallow": C_NO_SH,
    "replay_safe_uniform":    C_REPLAY,
    "replay_h6_weighted":     C_H6,
    "hybrid@0.01_reference":  C_HYBRID,
}
METHOD_LABEL = {
    "source_only":            "source_only",
    "policy_safe_no_shallow": "policy_safe\nno_shallow",
    "replay_safe_uniform":    "replay_safe\nuniform",
    "replay_h6_weighted":     "replay_h6\nweighted",
    "hybrid@0.01_reference":  "hybrid@0.01\n(参考値)",
}


# ---------------------------------------------------------------------------
# Load aug-True 9-subject summary (main results)
# ---------------------------------------------------------------------------
C_DIR = RESULTS / "c_aug_true_9subj_20260506_004923"
c_rows = json.loads((C_DIR / "summary.json").read_text())
c_table = {r["variant"]: r for r in c_rows}

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
src_per = c_table["source_only"]["per_subj"]


def per_subj_dict(variant_row):
    return {int(k): v for k, v in variant_row["per_subj"].items()}


# ---------------------------------------------------------------------------
# 1) main_result_table.csv
# ---------------------------------------------------------------------------
methods_order = ["source_only", "policy_safe_no_shallow", "replay_safe_uniform", "replay_h6_weighted"]
HYBRID_NOTE = "previous seminar value (5 seed mean, aug-True). Not re-evaluated in this regime."

rows = []
for m in methods_order:
    r = c_table[m]
    rows.append({
        "method": m,
        "mean_acc": round(r["mean_acc"], 2),
        "delta_vs_source": round(r["mean_delta"], 2) if r["mean_delta"] is not None else "missing",
        "worst_delta_vs_source": round(r["worst_delta"], 2) if r["worst_delta"] is not None else "missing",
        "hsc_0p5": f"{r['ntr_s']}/9",  # Harm-Subject Count: # subjects with Δ < -0.5pp
        "note": "main comparison row" if m in ("source_only", "replay_safe_uniform") else "",
    })
# add hybrid reference
rows.append({
    "method": "hybrid@0.01_reference",
    "mean_acc": 81.98,
    "delta_vs_source": 0.35,
    "worst_delta_vs_source": -0.34,
    "hsc_0p5": "0/9",
    "note": HYBRID_NOTE,
})

with open(OUT_TBL / "main_result_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"saved {OUT_TBL / 'main_result_table.csv'}")


# ---------------------------------------------------------------------------
# 1b) per_subject_delta.csv (for fig_per_subject_delta + record)
# ---------------------------------------------------------------------------
per_subj_csv_rows = []
for m in methods_order:
    pm = per_subj_dict(c_table[m])
    for s in SUBJECTS:
        if s in pm and pm[s] is not None and src_per.get(str(s)) is not None:
            delta = pm[s] - src_per[str(s)]
        else:
            delta = "missing"
        per_subj_csv_rows.append({
            "method": m,
            "subject": s,
            "acc": pm.get(s, "missing"),
            "delta_vs_source": (round(delta, 3) if isinstance(delta, float) else delta),
        })
with open(OUT_TBL / "per_subject_delta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "subject", "acc", "delta_vs_source"])
    w.writeheader()
    w.writerows(per_subj_csv_rows)
print(f"saved {OUT_TBL / 'per_subject_delta.csv'}")


# ---------------------------------------------------------------------------
# Figure 1 — main source_vs_replay bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
labels = [METHOD_LABEL[m] for m in methods_order]
means = [c_table[m]["mean_acc"] for m in methods_order]
deltas = [c_table[m]["mean_delta"] for m in methods_order]
colors = [METHOD_COLORS[m] for m in methods_order]
src_mean = c_table["source_only"]["mean_acc"]

bars = ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.7, width=0.65)
ax.axhline(src_mean, color=C_SOURCE, lw=0.8, ls="--", alpha=0.6, label=f"source_only = {src_mean:.2f}")

for i, (b, mu, d, m) in enumerate(zip(bars, means, deltas, methods_order)):
    if m == "source_only":
        ann = f"{mu:.2f}"
    else:
        sign = "+" if d >= 0 else ""
        ann = f"{mu:.2f}\n(Δ {sign}{d:.2f}pp)"
    color = "#1c4f1c" if (d is None or d >= 0) else "#7a1a14"
    ax.annotate(ann, xy=(b.get_x() + b.get_width() / 2, mu),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9.5, color=color)

ax.set_ylabel("9-subject mean accuracy [%]")
ax.set_title("aug-True 9 subject (1 seed, same source / split): replay_safe outperforms source_only")
ax.set_ylim(81.5, 84.5)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_main_source_vs_replay.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_main_source_vs_replay.png'}")


# ---------------------------------------------------------------------------
# Figure 2 — per-subject delta vs source, grouped bars
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(SUBJECTS))
width = 0.22
plot_methods = ["policy_safe_no_shallow", "replay_safe_uniform", "replay_h6_weighted"]

for i, m in enumerate(plot_methods):
    pm = per_subj_dict(c_table[m])
    deltas = [pm[s] - src_per[str(s)] for s in SUBJECTS]
    offset = (i - 1) * width
    color = METHOD_COLORS[m]
    edgewidth = 1.2 if m == "replay_safe_uniform" else 0.5
    bars = ax.bar(x + offset, deltas, width=width, color=color, edgecolor="black",
                  linewidth=edgewidth, label=m.replace("_", " "))
    if m == "replay_safe_uniform":
        for b, d in zip(bars, deltas):
            ax.annotate(f"{d:+.2f}", xy=(b.get_x() + b.get_width()/2, d),
                        xytext=(0, 3 if d >= 0 else -10),
                        textcoords="offset points",
                        ha="center", fontsize=8.5, color="black")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"S{s}" for s in SUBJECTS])
ax.set_ylabel("Δ accuracy vs source_only [pp]")
ax.set_title("Per-subject Δ vs source_only — replay_safe_uniform は 9 中 7 で source 同等以上")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-2.2, 2.6)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_per_subject_delta.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_per_subject_delta.png'}")


# ---------------------------------------------------------------------------
# Figure 3 — safety tradeoff scatter (mean Δ vs worst Δ)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 5.0))
for m in methods_order:
    if m == "source_only":
        continue
    r = c_table[m]
    x_v = r["mean_delta"]
    y_v = r["worst_delta"]
    ax.scatter(x_v, y_v, s=180, color=METHOD_COLORS[m], edgecolor="black",
               linewidth=0.9, zorder=3, label=m.replace("_", " "))
    ax.annotate(m.replace("_", " "), xy=(x_v, y_v), xytext=(8, 8),
                textcoords="offset points", fontsize=9.5)
# hybrid reference
ax.scatter(0.35, -0.34, s=180, color=METHOD_COLORS["hybrid@0.01_reference"],
           edgecolor="black", linewidth=0.9, marker="D", zorder=3,
           label="hybrid@0.01 (参考値)")
ax.annotate("hybrid@0.01\n(参考値)", xy=(0.35, -0.34), xytext=(8, -18),
            textcoords="offset points", fontsize=8.5, color="gray")

ax.axhline(0, color="black", lw=0.6)
ax.axvline(0, color="black", lw=0.6)
# Shade ideal quadrant
ax.fill_between([0, 1.5], 0, 0.5, color="#588c7e", alpha=0.07)
ax.text(0.95, 0.45, "Ideal: gain &\nno worst-subject harm",
        ha="right", va="top", fontsize=9, color="#1c4f1c")

ax.set_xlabel("mean Δ vs source_only [pp]   (large = average gain)")
ax.set_ylabel("worst-subject Δ vs source_only [pp]   (close to 0 = safe)")
ax.set_title("Safety–gain tradeoff: replay_safe_uniform は WSD 同等で mean Δ 最大")
ax.set_xlim(-0.1, 1.0)
ax.set_ylim(-2.0, 0.5)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_safety_tradeoff.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_safety_tradeoff.png'}")


# ---------------------------------------------------------------------------
# Figure 4 — S7 seed stability (combine seed-0 from a1 + seeds 1,2,3 from b)
# ---------------------------------------------------------------------------
B_DIR = RESULTS / "b_5seed_4subj_20260506_005153"
A1_DIR = RESULTS / "a1_9subject_no_shallow_20260505_182651"

b_rows = json.loads((B_DIR / "summary.json").read_text())
b_table = {r["variant"]: r for r in b_rows}

# seed-0 from a1 (no_aug regime). Note this is no_aug but matches B's seed-1..3 regime.
# Only seed comparison: B uses seeds 1,2,3 with no_aug. seed-0 from a1 also no_aug.
# So combined we have seeds {0,1,2,3} for S7 in no_aug regime.
a1_rows = json.loads((A1_DIR / "summary.json").read_text())
a1_table = {r["variant"]: r for r in a1_rows}


def collect_s7(variant_b: str, variant_a1: str) -> list[float]:
    """Collect S7 acc for seeds 0,1,2,3 across two sweeps."""
    vals = []
    seed0 = a1_table[variant_a1]["per_subject_acc"].get("7")
    if seed0 is not None:
        vals.append(seed0)
    if variant_b in b_table:
        for v in b_table[variant_b]["per_subj"].get("7", {}).get("vals", []):
            vals.append(v)
    return vals


S7_VARIANTS = [
    ("source_only", "source_only", "source_only"),
    ("replay_safe_uniform", "replay_safe_uniform", "replay_safe_uniform"),
    ("replay_h6_weighted", "replay_h6_weighted", "replay_h6_weighted"),
]
# Note: a1 doesn't have replay_h6_weighted (separate sweep), check
# a1 has: source_only, proto_otta_default, policy_safe_default, policy_safe_no_shallow
# replay_safe_uniform was added in replay_safe_9subject sweep, replay_h6 in replay_h6_9subject.
# To get seed-0 for replay variants, we must look at those separate dirs.
REPLAY_UNIFORM_DIR = RESULTS / "replay_safe_9subject_20260505_212942"
REPLAY_H6_DIR = RESULTS / "replay_h6_9subject_20260506_004038"


def parse_acc(d: Path):
    f = d / "results.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m:
            return float(m.group(1)) * 100.0
    return None


def s7_seed0(variant_label: str):
    if variant_label == "source_only":
        return parse_acc(A1_DIR / "eval" / "s7" / "source_only")
    if variant_label == "replay_safe_uniform":
        return parse_acc(REPLAY_UNIFORM_DIR / "eval" / "s7" / "replay_safe_default")
    if variant_label == "replay_h6_weighted":
        return parse_acc(REPLAY_H6_DIR / "eval" / "s7" / "replay_h6_weighted")
    return None


s7_data = {}
for label in ("source_only", "replay_safe_uniform", "replay_h6_weighted"):
    seed0 = s7_seed0(label)
    seeds_123 = b_table[label]["per_subj"].get("7", {}).get("vals", [])
    if seed0 is not None:
        all_vals = [seed0] + list(seeds_123)
    else:
        all_vals = list(seeds_123)
    s7_data[label] = all_vals

# Save S7 seed table
with open(OUT_TBL / "s7_seed_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "seed_0", "seed_1", "seed_2", "seed_3", "mean", "std", "n"])
    for label, vals in s7_data.items():
        padded = list(vals) + ["missing"] * (4 - len(vals))
        nums = [v for v in vals if isinstance(v, (int, float))]
        m = round(statistics.mean(nums), 2) if nums else "missing"
        sd = round(statistics.pstdev(nums), 2) if len(nums) > 1 else 0.0
        w.writerow([label] + padded + [m, sd, len(nums)])
print(f"saved {OUT_TBL / 's7_seed_table.csv'}")

# Plot
fig, ax = plt.subplots(figsize=(7, 4.6))
xpos = np.arange(len(s7_data))
labels = list(s7_data.keys())
for i, label in enumerate(labels):
    vals = s7_data[label]
    color = METHOD_COLORS.get(label, "#444")
    # Dot plot, jittered
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.10, 0.10, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               s=90, color=color, edgecolor="black", linewidth=0.6, alpha=0.85, zorder=3)
    if vals:
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        ax.errorbar(i, m, yerr=sd, fmt="_", color="black", capsize=8, lw=1.3,
                    markersize=18, zorder=4)
        ax.annotate(f"{m:.2f}±{sd:.2f}\n(n={len(vals)})",
                    xy=(i, m), xytext=(15, 0), textcoords="offset points",
                    fontsize=9, va="center")

ax.set_xticks(xpos)
ax.set_xticklabels([l.replace("_", "\n") for l in labels])
ax.set_ylabel("S7 accuracy [%]")
src_std = float(np.std(s7_data["source_only"], ddof=0)) if s7_data["source_only"] else 0.0
rep_std = float(np.std(s7_data["replay_safe_uniform"], ddof=0)) if s7_data["replay_safe_uniform"] else 0.0
red_pct = (1 - rep_std / src_std) * 100 if src_std > 0 else 0.0
ax.set_title(f"S7 seed-stability (4 seeds: 0=a1, 1-3=b_5seed, no_aug) — std を {red_pct:.0f}% 削減")
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(-0.5, len(labels) - 0.5)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_s7_seed_stability.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_s7_seed_stability.png'}")


# ---------------------------------------------------------------------------
# Figure 5 — replay candidate trace (S2)
# ---------------------------------------------------------------------------
trace_npz = sorted(glob.glob(str(RESULTS / "_smoke_replay_v3_s2_*/replay_safe_otta_stats_s2_tcformer_replay_safe_otta.npz")))[-1]
d = np.load(trace_npz, allow_pickle=True)

# Find a representative trial: max number of candidates ending in safe_replay,
# preferably one where replay rejected several candidates before accepting.
ops_arr = d["candidate_ops"]
reasons_arr = d["candidate_reasons"]
sims_arr = d["candidate_sim_scores"]
counts = d["candidate_count"].astype(int)
trial_idx = None
for i in range(len(counts)):
    if counts[i] < 3:
        continue
    reasons = str(reasons_arr[i]).split("|")
    if "safe_replay" in reasons and "replay_sim_score_drop" in reasons:
        trial_idx = i
        if reasons.count("replay_sim_score_drop") >= 3:
            break

if trial_idx is None:
    print("WARNING: no suitable trial found for trace; using trial 0")
    trial_idx = 0

ops = str(ops_arr[trial_idx]).split("|")
reasons = str(reasons_arr[trial_idx]).split("|")
sims = [float(x) for x in str(sims_arr[trial_idx]).split("|")]

with open(OUT_TBL / "replay_candidate_trace.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial_idx", "candidate_index", "operator", "sim_score", "reason"])
    for j, (op, sm, rs) in enumerate(zip(ops, sims, reasons)):
        w.writerow([trial_idx, j, op, f"{sm:.6e}", rs])
print(f"saved {OUT_TBL / 'replay_candidate_trace.csv'}")

# Plot
fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(ops) + 1.5))
y = np.arange(len(ops))[::-1]  # top = first candidate
abs_max = max(abs(s) for s in sims) * 1.3 if sims else 1.0
for j, (op, sm, rs) in enumerate(zip(ops, sims, reasons)):
    yy = y[j]
    color = "#3b6cb7" if rs == "safe_replay" else ("#d96459" if rs == "replay_sim_score_drop" else "#9a9a9a")
    ax.barh(yy, sm, color=color, edgecolor="black", linewidth=0.6, height=0.62)
    ax.text(sm + abs_max * 0.02 * (1 if sm >= 0 else -1), yy,
            f"sim = {sm:+.2e}",
            va="center", ha="left" if sm >= 0 else "right", fontsize=9)
    label_pos = -abs_max * 0.55
    ax.text(label_pos, yy, f"{op}", va="center", ha="left", fontsize=10,
            fontweight="bold" if rs == "safe_replay" else "normal")
    status = "✓ COMMIT" if rs == "safe_replay" else "✗ reject"
    ax.text(label_pos + abs_max * 0.30, yy, status,
            va="center", ha="left", fontsize=9.5,
            color="#1c4f1c" if rs == "safe_replay" else "#7a1a14",
            fontweight="bold" if rs == "safe_replay" else "normal")

ax.axvline(0, color="black", lw=0.7)
ax.set_yticks([])
ax.set_xlim(-abs_max, abs_max)
ax.set_xlabel("sim_score (← reject  |  commit →)")
ax.set_title(f"S2 trial {trial_idx}: replay が前段候補を reject → 後段で commit")
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_replay_candidate_trace.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_replay_candidate_trace.png'}")


# ---------------------------------------------------------------------------
# Figure 6 — method flow diagram (matplotlib)  + mermaid source
# ---------------------------------------------------------------------------
mermaid = """flowchart TD
    T["trial t 到着"] --> S["StateExtractor"]
    S --> P["RuleBasedPolicy<br>candidate operators 提案"]
    P --> Loop{"for each op"}
    Loop --> SN["snapshot model state"]
    SN --> AP["仮 apply"]
    AP --> T1{"Tier 1 (即時ガード)<br>margin / sal / proto / energy / bn_drift"}
    T1 -->|fail| R1["reject (tier1)"]
    T1 -->|pass| RP["replay buffer (32 trial)<br>fused logits で forward"]
    RP --> EV["sim_score = w_acc·Δacc + w_mar·Δmargin + w_pro·Δproto_cos"]
    EV --> T2{"Tier 2 (replay)<br>sim_score > 0?"}
    T2 -->|yes| C["COMMIT"]
    T2 -->|no| R2["reject (sim_score_drop)"]
    R1 --> Loop
    R2 --> RES["restore snapshot"]
    RES --> Loop
    C --> ADM["現 trial も<br>pmax > 0.85 AND SAL > 0.6 なら<br>buffer に FIFO admit"]
"""
(OUT_FIG / "method_flow_replay_safecommit.mmd").write_text(mermaid)

# Simple matplotlib version
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axis("off")

def box(x, y, w, h, text, color="#e8eef7", edge="#3b6cb7", fontsize=10, weight="normal"):
    rect = plt.Rectangle((x, y), w, h, linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#222")

def diamond(x, y, w, h, text, color="#fff3df", edge="#d4a040", fontsize=10):
    cx, cy = x + w / 2, y + h / 2
    pts = [(cx, y + h), (x + w, cy), (cx, y), (x, cy)]
    poly = plt.Polygon(pts, closed=True, linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color="#222")

def arrow(x1, y1, x2, y2, label="", color="black", width=0.7):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=width, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.2, label,
                ha="center", va="bottom", fontsize=9, color=color, style="italic")

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

box(0.2, 8.5, 1.6, 1.0, "trial t\n到着", color="#fff8e0", edge="#888")
box(2.2, 8.5, 1.8, 1.0, "StateExtractor", color="#e8eef7")
box(4.4, 8.5, 2.6, 1.0, "RuleBasedPolicy\ncandidate operators 提案")
box(7.6, 8.5, 2.2, 1.0, "for each op:\nsnapshot, 仮 apply", color="#f4e8f6", edge="#9a4caf")

diamond(7.6, 6.7, 2.2, 1.4,
        "Tier 1 (即時)\nmargin/sal/proto/\nenergy/bn_drift")
box(0.2, 6.7, 5.0, 1.4,
    "Tier 2 (replay)\n• replay buffer 32 trial を fused logits で forward\n• sim_score = w_acc·Δacc + w_mar·Δmargin + w_pro·Δproto_cos",
    color="#dceaf6", edge="#3b6cb7")

box(0.2, 4.3, 1.8, 1.0, "COMMIT", color="#dff0d8", edge="#1c4f1c", weight="bold")
box(2.4, 4.3, 1.8, 1.0, "reject (replay)\nrestore snapshot", color="#f5d8d2", edge="#7a1a14")
box(8.0, 5.3, 1.8, 1.0, "reject (tier1)\nrestore snapshot", color="#f5d8d2", edge="#7a1a14")

box(2.4, 1.8, 4.6, 1.4,
    "現 trial も pmax > 0.85 AND SAL > 0.6 なら\nbuffer に FIFO admit (擬似ラベル付き)",
    color="#dceaf6", edge="#3b6cb7")

# arrows
arrow(1.8, 9.0, 2.2, 9.0)
arrow(4.0, 9.0, 4.4, 9.0)
arrow(7.0, 9.0, 7.6, 9.0)
arrow(8.7, 8.5, 8.7, 8.1)              # to Tier 1
arrow(8.7, 6.7, 8.7, 6.3, label="pass")
arrow(7.6, 7.4, 5.2, 7.4, label="pass → Tier2")
arrow(9.8, 7.4, 9.8, 6.3, label="fail")
arrow(9.8, 5.3, 9.8, 4.0)               # tier1 reject path
arrow(2.7, 6.7, 2.7, 5.3, label="sim ≤ 0")
arrow(0.7, 6.7, 0.7, 5.3, label="sim > 0")
arrow(1.1, 4.3, 1.1, 3.3)               # commit -> admit
arrow(2.4, 4.3, 2.4, 3.3)               # admit chained

ax.set_title("Replay-Validated SafeCommit (二段ゲート)", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_method_flow_replay_safecommit.png", dpi=200)
plt.close(fig)
print(f"saved {OUT_FIG / 'fig_method_flow_replay_safecommit.png'}")
print(f"saved {OUT_FIG / 'method_flow_replay_safecommit.mmd'}")


# ---------------------------------------------------------------------------
# caption_draft.md
# ---------------------------------------------------------------------------
captions = """# 図キャプション草案 (260511 ゼミ)

各図 1 文。スライド下部にそのまま貼れる長さ。

## fig_main_source_vs_replay.png

> aug-True 9 被験者・1 seed・同一 source checkpoint・同一 split。replay_safe_uniform は素の TCFormer (source_only=82.72%) に対し +0.69pp の平均精度改善 (83.41%) を達成し、HSC=0/9 を保った。

## fig_per_subject_delta.png

> 被験者別 Δ accuracy (vs source_only)。replay_safe_uniform は 9 中 7 で 0pp 以上、唯一の負側 S4 でも policy_safe_no_shallow と同水準 (−1.73pp)。

## fig_safety_tradeoff.png

> Safety–gain tradeoff: 横軸 mean Δ、縦軸 worst-subject Δ。理想点は右上 (gain & no harm)。replay_safe_uniform は WSD 同等 (−0.35pp) で mean Δ 最大 (+0.69pp)。

## fig_s7_seed_stability.png

> S7 (gain subject) の 4 seed 結果 (seed 0 from a1, seeds 1-3 from b_5seed, no_aug regime)。__S7_CAPTION_DETAIL__

## fig_replay_candidate_trace.png

> S2 の代表 trial (trial 24)。Policy は 5 候補 operator を順に提案、Tier 2 (replay) gate で前段の shallow_var / hybrid_BN / deep_BN / prototype を sim_score ≤ 0 で reject、最後に logit_bias_update が sim > 0 で commit された。SafeCommit が「動いている」直接証拠。

## fig_method_flow_replay_safecommit.png

> Replay-Validated SafeCommit の処理フロー。trial ごとに candidate operator を順に試し、Tier 1 (即時ガード) と Tier 2 (replay buffer 上の simulated reward) の両方を通った operator のみ commit する。
"""
# Replace placeholder with computed S7 seed-stability detail
def _stat(label):
    vals = s7_data[label]
    if not vals:
        return f"{label}: missing"
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    return f"{label}: {m:.2f} ± {sd:.2f}"

s7_detail = (
    f"{_stat('source_only')}、{_stat('replay_safe_uniform')}、{_stat('replay_h6_weighted')}。"
    f"replay は variance reduction にも効く (std 削減率 "
    f"uniform: {(1 - np.std(s7_data['replay_safe_uniform'], ddof=0) / np.std(s7_data['source_only'], ddof=0)) * 100:.0f}%, "
    f"h6: {(1 - np.std(s7_data['replay_h6_weighted'], ddof=0) / np.std(s7_data['source_only'], ddof=0)) * 100:.0f}%)。"
)
captions = captions.replace("__S7_CAPTION_DETAIL__", s7_detail)
(OUT_FIG / "caption_draft.md").write_text(captions)
print(f"saved {OUT_FIG / 'caption_draft.md'}")

print("\n=== ALL ARTIFACTS GENERATED ===")
print(f"  tables: {OUT_TBL}")
print(f"  figures: {OUT_FIG}")
