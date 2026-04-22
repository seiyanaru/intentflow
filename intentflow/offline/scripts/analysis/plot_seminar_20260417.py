"""Plots for 2026-04-17 lab seminar.

生成する図:
  1. fig1_vanilla_vs_hybrid_per_subject.png   被験者別 source_only / vanilla OTTA / hybrid@0.01
  2. fig2_2x2_direction_ablation.png          2x2 ablation (shallow/deep × mean/var) heatmap, S2/S7
  3. fig3_ntr_wsd_summary.png                 設計比較 summary bar (NTR-S@0.5pp, WSD, meanΔ)
  4. fig4_hybrid_5seed_stability.png          hybrid@0.01 multi-seed (mom sweep)
  5. fig5_phaseB_suite.png                    plain / aug_only / aug+shinv suite
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_jp_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_jp_font_path).exists():
    font_manager.fontManager.addfont(_jp_font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[4] / "docs/research_progress/figures/seminar_20260417"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fig 1: 被験者別 accuracy (source_only / vanilla OTTA / hybrid@0.01)
# ---------------------------------------------------------------------------
def plot_fig1():
    subjects = [f"S{i}" for i in range(1, 10)]
    source = [84.03, 67.71, 90.97, 80.56, 76.74, 69.44, 92.36, 84.72, 88.19]
    # vanilla OTTA (bn_stat_clean, mom=0.1, bs=48) — 過去の既存設計
    vanilla = [85.42, 59.38, 90.97, 80.21, 76.04, 70.83, 95.14, 84.38, 88.89]
    # hybrid@0.01 (shallow_mean_deep_both) — 今回の最良設計
    hybrid = [85.07, 67.71, 91.67, 80.56, 76.74, 70.14, 93.40, 84.72, 87.85]

    src = np.array(source)
    van = np.array(vanilla)
    hyb = np.array(hybrid)

    fig, (ax_acc, ax_delta) = plt.subplots(
        2, 1, figsize=(11, 6.8), gridspec_kw={"height_ratios": [2.0, 1.0]}, sharex=True
    )

    x = np.arange(len(subjects))
    w = 0.27
    ax_acc.bar(x - w, src, w, label="source_only", color="#7f7f7f")
    ax_acc.bar(x, van, w, label="vanilla OTTA (bs=48, mom=0.1)", color="#d62728")
    ax_acc.bar(x + w, hyb, w, label="hybrid (shallow var 凍結, mom=0.01)", color="#2ca02c")
    ax_acc.set_ylabel("accuracy [%]")
    ax_acc.set_ylim(55, 100)
    ax_acc.set_title(
        "被験者別精度：vanilla OTTA は S2 を大きく壊す → hybrid はほぼ完全回復",
        fontsize=11,
    )
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.grid(axis="y", alpha=0.3)

    d_van = van - src
    d_hyb = hyb - src
    ax_delta.bar(x - w / 2, d_van, w, label="vanilla − source", color="#d62728")
    ax_delta.bar(x + w / 2, d_hyb, w, label="hybrid − source", color="#2ca02c")
    ax_delta.axhline(0, color="k", lw=0.7)
    ax_delta.axhline(-0.5, color="k", lw=0.5, ls=":", alpha=0.6)
    ax_delta.set_ylabel("Δ vs source [pp]")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(subjects)
    ax_delta.set_xlabel("subject (BCIC-IV 2a, session_T → session_E)")
    ax_delta.legend(loc="lower right", fontsize=9)
    ax_delta.grid(axis="y", alpha=0.3)

    for xi, (dv, dh) in enumerate(zip(d_van, d_hyb)):
        if dv <= -1.5:
            ax_delta.text(xi - w / 2, dv - 0.5, f"{dv:+.1f}", ha="center", fontsize=8, color="#d62728")
        if dh <= -0.3 or dh >= 0.3:
            ax_delta.text(xi + w / 2, dh + (0.2 if dh >= 0 else -0.6), f"{dh:+.1f}", ha="center", fontsize=8, color="#2ca02c")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_vanilla_vs_hybrid_per_subject.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: 2x2 direction ablation heatmap (S2 / S7)
# ---------------------------------------------------------------------------
def plot_fig2():
    rows = ["shallow (conv_block)", "deep (mix/reduce/TCN)"]
    cols = ["mean only", "var only"]
    s2 = np.array([[-0.35, -5.56], [-0.35, 0.00]])
    s7 = np.array([[0.35, 1.74], [0.70, 0.35]])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, mat, title in zip(axes, [s2, s7], ["S2 (harm 中心被験者)", "S7 (gain 中心被験者)"]):
        vmax = max(abs(mat).max(), 1.0)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_title(title)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat[i, j]
                color = "white" if abs(v) > 3 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Δ vs source [pp]")

    fig.suptitle(
        "2×2 direction ablation: S2 の harm は shallow×var にほぼ局在\n"
        "同じ操作(shallow var)が S7 では gain、S2 では harm → 被験者依存の方向",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_2x2_direction_ablation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: 設計比較サマリー (NTR-S@0.5pp, WSD, meanΔ)
# ---------------------------------------------------------------------------
def plot_fig3():
    conditions = [
        "vanilla\n(bs=48, mom=0.1)",
        "bs=1\nmom=0.1",
        "bs=1\nmom=0.01",
        "mean_only\nmom=0.01",
        "hybrid\nmom=0.01",
    ]
    ntrs = [4, 4, 2, 1, 0]         # NTR-S@0.5pp (out of 9)
    wsd = [-8.33, -7.99, -5.56, -0.69, -0.34]
    mean_delta = [-1.52, -1.04, -0.08, 0.20, 0.35]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = ["#d62728", "#e07b3e", "#ef9b00", "#59a14f", "#1f77b4"]

    axes[0].bar(conditions, ntrs, color=colors)
    axes[0].set_title("NTR-S@0.5pp (Δ<-0.5%) ↓ 少ないほど良い")
    axes[0].set_ylabel("# of subjects (out of 9)")
    axes[0].set_ylim(0, 9)
    for i, v in enumerate(ntrs):
        axes[0].text(i, v + 0.15, f"{v}/9", ha="center", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(conditions, wsd, color=colors)
    axes[1].set_title("WSD (Worst Subject Δ) ↑ 0 に近いほど良い")
    axes[1].set_ylabel("Δ [pp]")
    axes[1].axhline(0, color="k", lw=0.7)
    for i, v in enumerate(wsd):
        axes[1].text(i, v - 0.3, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(conditions, mean_delta, color=colors)
    axes[2].set_title("Mean Δ (all 9 subjects) ↑ 大きいほど良い")
    axes[2].set_ylabel("Δ [pp]")
    axes[2].axhline(0, color="k", lw=0.7)
    for i, v in enumerate(mean_delta):
        axes[2].text(i, v + (0.05 if v >= 0 else -0.15), f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(15)
            label.set_fontsize(9)

    fig.suptitle("OTTA 設計の進化：vanilla → hybrid で material harm を全被験者で排除 (0/9) かつ平均改善", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ntr_wsd_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: hybrid@0.01 の multi-seed stability
# ---------------------------------------------------------------------------
def plot_fig4():
    momenta = [0.01, 0.0125, 0.015, 0.0175, 0.02]
    both_mean = [0.37, 0.30, 0.29, 0.30, 0.29]
    both_std = [0.02, 0.08, 0.07, 0.09, 0.06]
    both_safe = [5, 4, 2, 2, 1]

    meanonly_mean = [0.22, 0.31, 0.30, 0.30, 0.30]
    meanonly_std = [0.02, 0.02, 0.02, 0.02, 0.02]
    meanonly_safe = [4, 5, 3, 5, 3]

    fig, (ax_mean, ax_safe) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax_mean.errorbar(momenta, both_mean, yerr=both_std, fmt="o-", capsize=4, label="shallow_mean + deep_both", color="#1f77b4")
    ax_mean.errorbar(momenta, meanonly_mean, yerr=meanonly_std, fmt="s-", capsize=4, label="shallow_mean + deep_mean_only", color="#ff7f0e")
    ax_mean.set_xlabel("momentum")
    ax_mean.set_ylabel("mean Δ [pp] (5 seeds)")
    ax_mean.set_title("hybrid@0.01 の平均改善：5 seed でも再現")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=9)

    width = 0.0018
    ax_safe.bar([m - width / 2 for m in momenta], both_safe, width, label="shallow_mean + deep_both", color="#1f77b4")
    ax_safe.bar([m + width / 2 for m in momenta], meanonly_safe, width, label="shallow_mean + deep_mean_only", color="#ff7f0e")
    ax_safe.set_xlabel("momentum")
    ax_safe.set_ylabel("safe@0.5 seeds / 5")
    ax_safe.set_ylim(0, 5.5)
    ax_safe.set_title("material harm なし (NTR-S@0.5pp=0/9) seed の数")
    ax_safe.grid(axis="y", alpha=0.3)
    ax_safe.legend(fontsize=9)
    ax_safe.set_xticks(momenta)

    fig.suptitle("hybrid の multi-seed × momentum sweep：mom=0.01 × deep_both が best", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_hybrid_5seed_stability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Phase B suite (plain / aug_only / aug+shinv)
# ---------------------------------------------------------------------------
def plot_fig5():
    conds = ["plain", "aug_only\n0.025", "aug_only\n0.05", "aug+shinv\n0.025", "aug+shinv\n0.05"]
    src = [79.67, 80.21, 79.51, 80.17, 79.78]
    hyb = [79.86, 80.63, 79.98, 80.21, 79.94]

    # S2 だけ抜き出し
    s2_src = [62.85, 60.76, 56.94, 60.07, 62.15]
    s2_hyb = [63.89, 62.50, 60.07, 62.15, 62.50]

    fig, (ax_all, ax_s2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(conds))
    w = 0.35
    ax_all.bar(x - w / 2, src, w, label="source_only", color="#7f7f7f")
    ax_all.bar(x + w / 2, hyb, w, label="+ hybrid@0.01", color="#1f77b4")
    ax_all.set_xticks(x)
    ax_all.set_xticklabels(conds, fontsize=9)
    ax_all.set_ylabel("mean accuracy [%]")
    ax_all.set_ylim(78, 82)
    ax_all.set_title("Phase B: all-9 mean accuracy\naug_only(0.025) が最良だが既存 hybrid frontier(81.98%) 未到達")
    ax_all.axhline(81.98, color="#d62728", ls="--", lw=1.2, label="既存 frontier (81.98%)")
    ax_all.grid(axis="y", alpha=0.3)
    ax_all.legend(fontsize=9)
    for xi, v in enumerate(hyb):
        ax_all.text(xi + w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

    ax_s2.bar(x - w / 2, s2_src, w, label="source_only", color="#7f7f7f")
    ax_s2.bar(x + w / 2, s2_hyb, w, label="+ hybrid@0.01", color="#1f77b4")
    ax_s2.set_xticks(x)
    ax_s2.set_xticklabels(conds, fontsize=9)
    ax_s2.set_ylabel("S2 accuracy [%]")
    ax_s2.set_title("Phase B: S2 canary\ntrain-time aug では S2 をまだ救えていない")
    ax_s2.axhline(67.71, color="#d62728", ls="--", lw=1.2, label="S2 source (67.71%)")
    ax_s2.grid(axis="y", alpha=0.3)
    ax_s2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_phaseB_suite.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6: online pipeline architecture (block diagram made with text boxes)
# ---------------------------------------------------------------------------
def plot_fig6():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#cfe2f3"):
        r = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=1.0)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
        )

    # Row 1: Windows ノート (Unicorn 取得 + bridge)
    box(0.2, 3.4, 2.6, 1.1, "Windows ノート\nUnicorn SDK 取得\nUDP bridge 送出", color="#fce5cd")
    # UDP line
    arrow(2.8, 4.0, 3.8, 4.0)
    ax.text(3.3, 4.15, "UDP (csv,8ch,250Hz)", ha="center", fontsize=8, color="#555")

    # Row 1: 研究室PC (Linux, GPU)
    box(3.8, 3.4, 4.0, 1.1, "研究室 PC (Linux/GPU)\nttt_broadcaster.py\n--source unicorn_udp", color="#d9ead3")
    arrow(7.8, 4.0, 8.8, 4.0)
    ax.text(8.3, 4.15, "WebSocket 8765", ha="center", fontsize=8, color="#555")

    # Row 1: Unity
    box(8.8, 3.4, 3.0, 1.1, "Unity HUD\nintent / confidence\n受信・可視化", color="#cfe2f3")

    # Row 2: 内部の broadcaster 処理の展開
    ax.text(1.5, 2.3, "①", fontsize=11, fontweight="bold")
    ax.text(5.8, 2.3, "②", fontsize=11, fontweight="bold")
    ax.text(10.3, 2.3, "③", fontsize=11, fontweight="bold")

    # Row 2 内訳
    box(0.2, 0.6, 2.6, 1.4, "① Unicorn UDP receiver\n 8ch / 250Hz サンプル受信\n リングバッファ", color="#fff2cc")
    box(3.2, 0.6, 5.2, 1.4,
        "② 推論ループ\n 4.0s 窓 / 0.25s hop\n StreamNormalizer\n TCFormer Hybrid (mom=0.01)\n pmax/neuro/energy gating",
        color="#fff2cc")
    box(8.8, 0.6, 3.0, 1.4, "③ WS clients / protocol\n intent, confidence,\n timestamps, protocol_version", color="#fff2cc")

    ax.set_title("オンライン構成：Unicorn → UDP bridge → ttt_broadcaster → WebSocket → Unity", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_online_pipeline.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7: offline OTTA 処理フロー (hybrid + Tri-Lock gating, 今回うまく行った構成)
# ---------------------------------------------------------------------------
def plot_fig7():
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, w, h, text, color="#cfe2f3", fontsize=9, fontweight="normal", edge="black", lw=1.0):
        r = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, lw=lw)
        ax.add_patch(r)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
        )

    def arrow(x1, y1, x2, y2, color="black", lw=1.3, style="->"):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        )

    ax.text(
        6.0, 10.6,
        "Offline OTTA 処理フロー (今回の最良構成: hybrid = shallow_mean_deep_both @ mom=0.01)",
        ha="center", fontsize=12, fontweight="bold",
    )

    # --- 1. 入力 EEG sample ---------------------------------------------------
    box(3.8, 9.3, 4.4, 0.9,
        "入力 EEG 窓  (1, 22ch, 1000samp)  ※ BCIC-IV 2a, 4.0s, 250Hz",
        color="#eeeeee", fontsize=9)
    arrow(6.0, 9.3, 6.0, 8.95)

    # --- 2. TCFormer forward (shallow BN / deep BN を分けて示す) -------------
    box(1.2, 7.1, 9.6, 1.85, "", color="#ffffff", edge="#444444", lw=1.0)
    ax.text(6.0, 8.72, "TCFormer forward  (BN 12 層)", ha="center", fontsize=10, fontweight="bold")

    box(1.7, 7.35, 4.0, 1.15,
        "shallow 6 BN\n(conv_block 前段)\n\n● running_mean: 更新\n● running_var: 凍結",
        color="#d4edda", fontsize=9, lw=1.2)
    box(6.3, 7.35, 4.0, 1.15,
        "deep 6 BN\n(mix / reduce / TCN)\n\n● running_mean: 更新\n● running_var: 更新",
        color="#d1ecf1", fontsize=9, lw=1.2)
    arrow(5.7, 7.925, 6.3, 7.925)

    arrow(6.0, 7.1, 6.0, 6.75)

    # --- 3. logits / softmax --------------------------------------------------
    box(4.4, 6.2, 3.2, 0.55, "logits  →  softmax", color="#eeeeee", fontsize=9)
    arrow(6.0, 6.2, 6.0, 5.85)

    # --- 4. Tri-Lock gating (3 条件 AND) --------------------------------------
    box(0.7, 4.7, 10.6, 1.15, "", color="#fff8e1", edge="#b38600", lw=1.2)
    ax.text(6.0, 5.62, "Tri-Lock gating   (3 条件 AND / すべて通過した sample のみ BN 更新に寄与)",
            ha="center", fontsize=10, fontweight="bold", color="#8a6d00")

    box(1.1, 4.8, 3.0, 0.75,
        "pmax ≥ τ_pmax\n(softmax 最大確率)",
        color="#fff3b0", fontsize=9)
    box(4.5, 4.8, 3.0, 0.75,
        "SAL ≥ τ_SAL\n(spectral align)",
        color="#fff3b0", fontsize=9)
    box(7.9, 4.8, 3.0, 0.75,
        "energy ∈ [τ_lo, τ_hi]\n(prototype 整合)",
        color="#fff3b0", fontsize=9)

    arrow(6.0, 4.7, 6.0, 4.35)

    # --- 5. pass / fail 分岐 --------------------------------------------------
    box(5.2, 3.85, 1.6, 0.5, "すべて pass?", color="#ffffff", fontsize=9, fontweight="bold")
    # pass (下左)
    arrow(5.35, 3.85, 3.0, 3.35, color="#1a7f37")
    ax.text(3.9, 3.7, "pass", color="#1a7f37", fontsize=9, fontweight="bold")
    # fail (下右)
    arrow(6.65, 3.85, 9.0, 3.35, color="#a40e26")
    ax.text(8.1, 3.7, "fail", color="#a40e26", fontsize=9, fontweight="bold")

    # --- 6. pass → hybrid BN 更新 (今回の要) --------------------------------
    box(0.7, 2.1, 4.6, 1.25,
        "BN 更新  (hybrid 方針)\n"
        "shallow: mean のみ  /  deep: mean + var\n"
        "momentum = 0.01\n"
        "→ source mean/var を緩やかに eval 側へ",
        color="#d4edda", fontsize=9, fontweight="bold", edge="#1a7f37", lw=1.5)
    arrow(3.0, 2.1, 3.0, 1.55)

    # --- 7. fail → no update --------------------------------------------------
    box(6.7, 2.1, 4.6, 1.25,
        "更新しない  (fail-closed)\n"
        "● BN 統計そのまま\n"
        "● abstain-safe (低信頼出力は保留)\n"
        "● 害を連鎖させない",
        color="#fde1e1", fontsize=9, edge="#a40e26", lw=1.2)
    arrow(9.0, 2.1, 9.0, 1.55)

    # --- 8. predict emit ------------------------------------------------------
    box(3.8, 0.6, 4.4, 0.95,
        "予測 emit  (argmax logits, confidence)\n次 sample へ  →  Online では Unity HUD / abstain",
        color="#cfe2f3", fontsize=9)
    # 横に短い注記
    ax.text(
        11.85, 5.28,
        "※ vanilla OTTA との差は\n"
        "「shallow running_var を凍結」\n"
        "「mom=0.1 → 0.01 に緩和」\n"
        "の 2 点のみ（最小介入）",
        ha="right", va="center", fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#888", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig7_offline_otta_flow.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    print("Saved figures to:", OUT)
