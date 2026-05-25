from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "docs/research_progress/regular_seminar_2605/tables"
OUT_DIR = ROOT / "docs/research_progress/260525_seminar/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

font_manager.fontManager.addfont(FONT_REG)
font_manager.fontManager.addfont(FONT_MED)
font_manager.fontManager.addfont(FONT_BOLD)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2F6FBB"
GREEN = "#3F9B6D"
ORANGE = "#E2A72E"
RED = "#C94C4C"
GRAY = "#9AA4B2"
LIGHT_GRAY = "#EEF2F6"
TEXT = "#18212F"


def parse_delta(x):
    return float(str(x).replace("+", ""))


def parse_hsc(x):
    return int(str(x).split("/")[0])


def annotate_bar(ax, bars, values, hsc_values):
    for bar, val, hsc in zip(bars, values, hsc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.06 if val >= 0 else -0.08),
            f"{val:+.2f}\nHSC={hsc}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
            color=TEXT,
            fontweight="bold",
        )


def plot_plan(ax, df, plan, title):
    order = [
        "Source",
        "Replay-Safe",
        "DC L1+L2",
        "DC high mean",
        "DC B best",
    ]
    sub = df[df["plan"] == plan].copy()
    sub["delta_num"] = sub["delta"].map(parse_delta)
    sub["hsc_num"] = sub["hsc"].map(parse_hsc)
    sub = sub[sub["label"].isin(order)]
    sub["rank"] = sub["label"].map({k: i for i, k in enumerate(order)})
    sub = sub.sort_values("rank")

    labels = sub["label"].tolist()
    values = sub["delta_num"].tolist()
    hsc = sub["hsc_num"].tolist()
    colors = []
    for label, h in zip(labels, hsc):
        if label == "Source":
            colors.append(GRAY)
        elif label == "Replay-Safe":
            colors.append(GREEN)
        elif h == 0:
            colors.append(BLUE)
        elif h <= 1:
            colors.append(ORANGE)
        else:
            colors.append(RED)

    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="#22324A", linewidth=0.8)
    annotate_bar(ax, bars, values, hsc)

    ax.axhline(0, color="#334155", linewidth=1.0)
    ax.set_title(title, fontsize=14, fontweight="bold", color=BLUE, pad=10)
    ax.set_ylabel("Δ accuracy vs source (pp)", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(-0.25, max(values) + 0.45)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_l3(ax, diag):
    # First five rows: Plan C, next five rows: Plan B in the source table.
    diag = diag.copy()
    diag["eval"] = ["9-subject"] * 5 + ["seed-stability"] * 5
    diag["delta"] = diag["delta"].astype(float)
    diag["model_commit"] = diag["model_commit"].astype(int)
    diag["hsc"] = diag["hsc"].astype(int)

    labels = [
        "L1+L2\nno commit",
        "Replay\n-gated",
        "No replay\ngate",
        "Random\nsparse",
        "High mean\nzero commit",
    ]
    x_base = list(range(5))
    width = 0.34
    c = diag[diag["eval"] == "9-subject"].reset_index(drop=True)
    b = diag[diag["eval"] == "seed-stability"].reset_index(drop=True)

    ax2 = ax.twinx()
    bars_c = ax.bar(
        [x - width / 2 for x in x_base],
        c["model_commit"],
        width=width,
        color="#CFE3FA",
        edgecolor=BLUE,
        linewidth=1.0,
        label="model commits (9-subject)",
    )
    bars_b = ax.bar(
        [x + width / 2 for x in x_base],
        b["model_commit"],
        width=width,
        color="#F8D7D7",
        edgecolor=RED,
        linewidth=1.0,
        label="model commits (seed)",
    )

    ax2.plot(x_base, c["delta"], color=BLUE, marker="o", linewidth=2.0, label="Δ acc (9-subject)")
    ax2.plot(x_base, b["delta"], color=RED, marker="o", linewidth=2.0, label="Δ acc (seed)")

    for i, (bar, val) in enumerate(zip(bars_c, c["model_commit"])):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, str(val), ha="center", fontsize=8, color=BLUE)
    for i, (bar, val) in enumerate(zip(bars_b, b["model_commit"])):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, str(val), ha="center", fontsize=8, color=RED)
    for i, row in c.iterrows():
        ax2.text(i - 0.05, row["delta"] + 0.035, f"{row['delta']:+.2f}", fontsize=8, color=BLUE, ha="right")
    for i, row in b.iterrows():
        ax2.text(i + 0.05, row["delta"] - 0.05, f"{row['delta']:+.2f}", fontsize=8, color=RED, ha="left")

    ax.set_title("L3診断: commit回数は改善を説明しない", fontsize=14, fontweight="bold", color=BLUE, pad=10)
    ax.set_ylabel("model commit count", fontsize=10)
    ax2.set_ylabel("Δ accuracy vs source (pp)", fontsize=10)
    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(diag["model_commit"]) + 45)
    ax2.set_ylim(0.0, max(diag["delta"]) + 0.35)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2, frameon=False)


def main():
    key = pd.read_csv(IN_DIR / "key_result_table.csv")
    diag = pd.read_csv(IN_DIR / "l3_diagnostics.csv")

    fig = plt.figure(figsize=(16, 9), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.22)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_l3 = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "DC-Replay 追加検証: 補正力は出るが、安全性とL3 commitに課題",
        fontsize=18,
        fontweight="bold",
        color=BLUE,
        y=0.985,
    )

    plot_plan(ax_c, key, "C", "9被験者評価: 平均精度シグナル")
    plot_plan(ax_b, key, "B", "seed安定性評価: HSCが増える")
    plot_l3(ax_l3, diag)

    legend_items = [
        Patch(facecolor=GREEN, edgecolor="#22324A", label="Replay-Safe"),
        Patch(facecolor=ORANGE, edgecolor="#22324A", label="DC: HSC=1"),
        Patch(facecolor=RED, edgecolor="#22324A", label="DC: HSC≥2"),
        Patch(facecolor=GRAY, edgecolor="#22324A", label="Source"),
    ]
    fig.legend(legend_items, [p.get_label() for p in legend_items], loc="lower center", ncol=4, frameon=False, fontsize=10)

    fig.text(
        0.5,
        0.035,
        "読み取り: DC系は9被験者評価で最大 +1.23pp まで伸びるが、seed安定性ではHSCが増える。さらに、同じ精度でもmodel commit回数が0/111/150と変わるため、L3 commit単独では改善を説明できない。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=TEXT,
    )

    png = OUT_DIR / "fig_dc_replay_results_summary.png"
    svg = OUT_DIR / "fig_dc_replay_results_summary.svg"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
