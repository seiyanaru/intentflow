from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "research_progress" / "regular_seminar_2605"
FIG = OUT / "figures"
TAB = OUT / "tables"

C_SUMMARY = (
    ROOT
    / "intentflow"
    / "offline"
    / "results"
    / "c_aug_true_9subj_20260506_004923"
    / "dc_replay_72h_summary.json"
)
B_SUMMARY = (
    ROOT
    / "intentflow"
    / "offline"
    / "results"
    / "b_5seed_4subj_20260506_005153"
    / "dc_replay_72h_summary.json"
)
C_EVAL = C_SUMMARY.parent / "eval"
B_EVAL = B_SUMMARY.parent / "eval"


SELECTED = [
    ("source_only", "Source"),
    ("replay_safe_uniform", "Replay-Safe"),
    ("dc_corr_alpha_01", "DC safe a=.01"),
    ("dc_grid_a01_m65_n8_tol0", "DC safe grid"),
    ("dc_correction_memory_no_commit", "DC L1+L2"),
    ("dc_grid_a04_m55_n8_tol5e5", "DC high mean"),
    ("dc_grid_a08_m45_n8_tol0", "DC B best"),
]

DIAG_VARIANTS = [
    ("dc_correction_memory_no_commit", "L1+L2 no commit"),
    ("dc_replay_gated", "Replay-gated"),
    ("dc_commit_no_replay_gate", "No replay gate"),
    ("dc_random_sparse_p10", "Random sparse"),
    ("dc_grid_a04_m55_n8_tol5e5", "High mean tol"),
]


def load_rows(path: Path) -> dict[str, dict]:
    with path.open() as f:
        data = json.load(f)
    return {row["variant"]: row for row in data["rows"]}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bar_summary(rows: dict[str, dict], path: Path, title: str) -> None:
    variants = [(v, label) for v, label in SELECTED if v in rows]
    labels = [label for _, label in variants]
    means = [rows[v]["mean_acc"] for v, _ in variants]
    deltas = [rows[v]["mean_delta_vs_source"] for v, _ in variants]
    hsc = [rows[v]["hsc_at_0p5"] for v, _ in variants]

    colors = []
    for v, _ in variants:
        if v == "source_only":
            colors.append("#8a8f98")
        elif v == "replay_safe_uniform":
            colors.append("#2f6fbb")
        elif rows[v]["hsc_at_0p5"] == 0:
            colors.append("#3f9b6d")
        elif rows[v]["hsc_at_0p5"] <= 1:
            colors.append("#e2a72e")
        else:
            colors.append("#c94c4c")

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axhline(rows["source_only"]["mean_acc"], color="#555555", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ymin = min(means) - 0.7
    ymax = max(means) + 0.5
    ax.set_ylim(ymin, ymax)
    for i, (m, d, h) in enumerate(zip(means, deltas, hsc)):
        ax.text(i, m + 0.06, f"{m:.2f}\n{d:+.2f}pp\nHSC {h}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def tradeoff(c_rows: dict[str, dict], b_rows: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    for ax, rows, title in [
        (axes[0], c_rows, "Plan C: 9 subjects"),
        (axes[1], b_rows, "Plan B: 4 subjects x 3 seeds"),
    ]:
        xs = [r["mean_delta_vs_source"] for r in rows.values()]
        ys = [r["worst_delta_vs_source"] for r in rows.values()]
        cs = [r["hsc_at_0p5"] for r in rows.values()]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis_r", s=28, alpha=0.75, edgecolor="none")
        ax.axhline(-0.5, color="#c94c4c", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Mean delta vs source (pp)")
        ax.set_ylabel("Worst delta vs source (pp)")
        ax.grid(alpha=0.22)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        ax.set_xlim(x_min - 0.08, x_max + 0.20)
        ax.set_ylim(y_min - 0.15, y_max + 0.18)
        for v, label in [
            ("replay_safe_uniform", "Replay"),
            ("dc_grid_a04_m55_n8_tol5e5", "DC high"),
            ("dc_grid_a08_m45_n8_tol0", "DC B best"),
            ("dc_corr_alpha_01", "DC safe"),
        ]:
            if v in rows:
                r = rows[v]
                ax.scatter(
                    [r["mean_delta_vs_source"]],
                    [r["worst_delta_vs_source"]],
                    s=90,
                    facecolor="none",
                    edgecolor="#111111",
                    linewidth=1.2,
                )
                ax.annotate(
                    label,
                    (r["mean_delta_vs_source"], r["worst_delta_vs_source"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("HSC@0.5pp")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def per_subject_delta(rows: dict[str, dict], path: Path) -> None:
    src = rows["source_only"]["per_unit"]
    variants = [
        ("replay_safe_uniform", "Replay-Safe"),
        ("dc_corr_alpha_01", "DC safe a=.01"),
        ("dc_correction_memory_no_commit", "DC L1+L2"),
        ("dc_grid_a04_m55_n8_tol5e5", "DC high mean"),
    ]
    subjects = list(src.keys())
    x = np.arange(len(subjects))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    colors = ["#2f6fbb", "#3f9b6d", "#e2a72e", "#c94c4c"]
    for i, ((v, label), color) in enumerate(zip(variants, colors)):
        deltas = [rows[v]["per_unit"][s] - src[s] for s in subjects]
        ax.bar(x + (i - 1.5) * width, deltas, width=width, label=label, color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(-0.5, color="#c94c4c", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subjects])
    ax.set_ylabel("Accuracy delta vs source (pp)")
    ax.set_title("Plan C per-subject delta")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def seed_subject_delta(rows: dict[str, dict], path: Path) -> None:
    src = rows["source_only"]["per_unit"]
    variants = [
        ("replay_safe_uniform", "Replay-Safe"),
        ("dc_correction_memory_no_commit", "DC L1+L2"),
        ("dc_grid_a04_m55_n8_tol5e5", "DC high mean"),
        ("dc_grid_a08_m45_n8_tol0", "DC B best"),
    ]
    subjects = ["s2", "s4", "s6", "s7"]
    x = np.arange(len(subjects))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.8, 4.1))
    colors = ["#2f6fbb", "#e2a72e", "#c94c4c", "#9467bd"]
    for i, ((v, label), color) in enumerate(zip(variants, colors)):
        means = []
        stds = []
        for s in subjects:
            deltas = [rows[v]["per_unit"][k] - src[k] for k in src if k.startswith(s + "_")]
            means.append(float(np.mean(deltas)))
            stds.append(float(np.std(deltas, ddof=0)))
        ax.bar(
            x + (i - 1.5) * width,
            means,
            yerr=stds,
            capsize=2,
            width=width,
            label=label,
            color=color,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(-0.5, color="#c94c4c", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subjects])
    ax.set_ylabel("Seed-mean delta vs source (pp)")
    ax.set_title("Plan B subject-wise seed stability")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def finite_mean(arr: np.ndarray) -> float:
    x = arr[np.isfinite(arr)]
    return float(x.mean()) if x.size else float("nan")


def diagnostics(root: Path, rows: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for variant, label in DIAG_VARIANTS:
        paths = list(root.glob(f"*/{variant}/dc_replay_otta_stats_*_tcformer_deferred_commit_replay_otta.npz"))
        if not paths:
            continue
        totals = {
            "trials": 0,
            "commit": 0.0,
            "model_commit": 0.0,
            "memory_admitted": 0.0,
            "candidate_trials": 0,
            "sim_finite": 0,
            "sim_pos": 0,
        }
        strengths = []
        drifts = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            cand = np.asarray(d["candidate_count"], dtype=float) if "candidate_count" in d.files else np.zeros(0)
            sim = np.asarray(d["sim_score"], dtype=float) if "sim_score" in d.files else np.full(len(cand), np.nan)
            totals["trials"] += len(cand)
            totals["commit"] += float(np.nansum(d["committed"])) if "committed" in d.files else 0.0
            totals["model_commit"] += (
                float(np.nansum(d["dc_model_state_committed"])) if "dc_model_state_committed" in d.files else 0.0
            )
            totals["memory_admitted"] += float(np.nansum(d["memory_admitted"])) if "memory_admitted" in d.files else 0.0
            totals["candidate_trials"] += int((cand > 0).sum())
            totals["sim_finite"] += int(np.isfinite(sim).sum())
            totals["sim_pos"] += int((sim[np.isfinite(sim)] > 0).sum())
            if "correction_strength" in d.files:
                strengths.append(np.asarray(d["correction_strength"], dtype=float))
            if "dc_persistent_drift_score" in d.files:
                drifts.append(np.asarray(d["dc_persistent_drift_score"], dtype=float))
        row = {
            "variant": variant,
            "label": label,
            "mean_acc": rows[variant]["mean_acc"] if variant in rows else float("nan"),
            "delta": rows[variant]["mean_delta_vs_source"] if variant in rows else float("nan"),
            "hsc": rows[variant]["hsc_at_0p5"] if variant in rows else 0,
            "trials": totals["trials"],
            "commit": int(totals["commit"]),
            "model_commit": int(totals["model_commit"]),
            "memory_admitted": int(totals["memory_admitted"]),
            "candidate_trials": totals["candidate_trials"],
            "sim_finite": totals["sim_finite"],
            "sim_pos": totals["sim_pos"],
            "correction_strength_mean": finite_mean(np.concatenate(strengths)) if strengths else float("nan"),
            "drift_max": float(np.nanmax(np.concatenate(drifts))) if drifts else float("nan"),
        }
        out.append(row)
    return out


def diagnostic_plot(c_diag: list[dict], b_diag: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, diag, title in [(axes[0], c_diag, "Plan C"), (axes[1], b_diag, "Plan B")]:
        labels = [d["label"] for d in diag]
        x = np.arange(len(labels))
        commits = [d["model_commit"] for d in diag]
        delta = [d["delta"] for d in diag]
        hsc = [d["hsc"] for d in diag]
        bars = ax.bar(x, commits, color="#6b8ec1", edgecolor="#333333", linewidth=0.5)
        ax.set_title(title, pad=6, fontsize=13)
        ax.set_ylim(0, max(180, max(commits) + 35))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Model-state commits")
        ax.grid(axis="y", alpha=0.22)
        for i, (bar, d, h) in enumerate(zip(bars, delta, hsc)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{d:+.2f}pp\nHSC {h}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("L3 commit diagnostics: commits do not explain the best gain", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def best_under_hsc(rows: dict[str, dict], limit: int) -> dict:
    candidates = [r for r in rows.values() if r["hsc_at_0p5"] <= limit]
    return max(candidates, key=lambda r: r["mean_acc"])


def write_analysis(c_rows: dict[str, dict], b_rows: dict[str, dict], c_diag: list[dict], b_diag: list[dict]) -> None:
    c_best = max(c_rows.values(), key=lambda r: r["mean_acc"])
    b_best = max(b_rows.values(), key=lambda r: r["mean_acc"])
    c_safe = best_under_hsc(c_rows, 0)
    b_safe1 = best_under_hsc(b_rows, 1)
    lines = [
        "# 2605 regular seminar result analysis",
        "",
        "## Headline",
        "",
        "- Plan C: the highest-mean DC variant improves over source, but introduces one harmful subject.",
        "- Plan B: seed stability favors replay_safe_uniform under HSC<=1; DC variants trade small mean gains for larger HSC.",
        "- The strongest DC gains are mainly from L1/L2 prediction correction and external memory, not from L3 model-state commits.",
        "",
        "## Key numbers",
        "",
        f"- Plan C best mean: `{c_best['variant']}` mean={c_best['mean_acc']:.2f}, delta={c_best['mean_delta_vs_source']:+.2f}pp, worst={c_best['worst_delta_vs_source']:+.2f}pp, HSC={c_best['hsc_at_0p5']}/{c_best['n']}.",
        f"- Plan C best HSC=0: `{c_safe['variant']}` mean={c_safe['mean_acc']:.2f}, delta={c_safe['mean_delta_vs_source']:+.2f}pp, worst={c_safe['worst_delta_vs_source']:+.2f}pp.",
        f"- Plan B best mean: `{b_best['variant']}` mean={b_best['mean_acc']:.2f}, delta={b_best['mean_delta_vs_source']:+.2f}pp, worst={b_best['worst_delta_vs_source']:+.2f}pp, HSC={b_best['hsc_at_0p5']}/{b_best['n']}.",
        f"- Plan B best HSC<=1: `{b_safe1['variant']}` mean={b_safe1['mean_acc']:.2f}, delta={b_safe1['mean_delta_vs_source']:+.2f}pp, worst={b_safe1['worst_delta_vs_source']:+.2f}pp, HSC={b_safe1['hsc_at_0p5']}/{b_safe1['n']}.",
        "",
        "## L3 diagnostics",
        "",
        "| plan | variant | delta | HSC | model_commit | memory_admitted | candidate_trials | sim_finite | sim_pos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for plan, diag in [("C", c_diag), ("B", b_diag)]:
        for d in diag:
            lines.append(
                f"| {plan} | {d['variant']} | {d['delta']:+.2f} | {d['hsc']} | {d['model_commit']} | {d['memory_admitted']} | {d['candidate_trials']} | {d['sim_finite']} | {d['sim_pos']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Replay-SafeCommit remains the most defensible current main result because it improves mean accuracy while preserving a low HSC in the seed-stability setting.",
            "2. DC-style correction has a real accuracy signal: Plan C reaches +1.23pp and Plan B reaches +0.49pp.",
            "3. The price is safety: the best DC variants increase HSC, especially for S4/S6 seeds.",
            "4. The fact that no-commit or zero-commit variants can match the best DC performance means the current novelty should be framed as commitless memory-corrected OTTA, with L3 deferred commit left as an open extension.",
        ]
    )
    (OUT / "analysis_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    c_rows = load_rows(C_SUMMARY)
    b_rows = load_rows(B_SUMMARY)

    table_rows = []
    for plan, rows in [("C", c_rows), ("B", b_rows)]:
        for variant, label in SELECTED:
            if variant in rows:
                r = rows[variant]
                table_rows.append(
                    {
                        "plan": plan,
                        "variant": variant,
                        "label": label,
                        "mean_acc": f"{r['mean_acc']:.2f}",
                        "delta": f"{r['mean_delta_vs_source']:+.2f}",
                        "worst": f"{r['worst_delta_vs_source']:+.2f}",
                        "hsc": f"{r['hsc_at_0p5']}/{r['n']}",
                    }
                )
    write_csv(
        TAB / "key_result_table.csv",
        table_rows,
        ["plan", "variant", "label", "mean_acc", "delta", "worst", "hsc"],
    )

    c_diag = diagnostics(C_EVAL, c_rows)
    b_diag = diagnostics(B_EVAL, b_rows)
    write_csv(
        TAB / "l3_diagnostics.csv",
        c_diag + b_diag,
        [
            "variant",
            "label",
            "mean_acc",
            "delta",
            "hsc",
            "trials",
            "commit",
            "model_commit",
            "memory_admitted",
            "candidate_trials",
            "sim_finite",
            "sim_pos",
            "correction_strength_mean",
            "drift_max",
        ],
    )

    bar_summary(c_rows, FIG / "fig_plan_c_key_results.png", "Plan C key results")
    bar_summary(b_rows, FIG / "fig_plan_b_key_results.png", "Plan B seed-stability results")
    tradeoff(c_rows, b_rows, FIG / "fig_safety_tradeoff.png")
    per_subject_delta(c_rows, FIG / "fig_plan_c_per_subject_delta.png")
    seed_subject_delta(b_rows, FIG / "fig_plan_b_subject_seed_delta.png")
    diagnostic_plot(c_diag, b_diag, FIG / "fig_l3_commit_diagnostics.png")
    write_analysis(c_rows, b_rows, c_diag, b_diag)


if __name__ == "__main__":
    main()
