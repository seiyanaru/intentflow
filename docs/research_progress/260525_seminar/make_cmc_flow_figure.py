from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BLACK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

JP = font_manager.FontProperties(fname=FONT_REG)
JP_MED = font_manager.FontProperties(fname=FONT_MED)
JP_BOLD = font_manager.FontProperties(fname=FONT_BLACK)

BLUE = "#0B3A86"
LIGHT_BLUE = "#EEF6FF"
PALE_BLUE = "#F7FBFF"
GREEN = "#2F8A3A"
LIGHT_GREEN = "#EAF7EA"
RED = "#C73333"
LIGHT_RED = "#FFF0F0"
ORANGE = "#C68000"
LIGHT_ORANGE = "#FFF7E7"
GRAY = "#F5F7FA"
TEXT = "#111827"


def text(ax, x, y, s, size=12, color=TEXT, ha="center", va="center", weight="regular", **kwargs):
    fp = JP_BOLD if weight == "bold" else JP_MED if weight == "medium" else JP
    ax.text(x, y, s, fontproperties=fp, fontsize=size, color=color, ha=ha, va=va, **kwargs)


def rounded(ax, x, y, w, h, fc="white", ec=BLUE, lw=1.6, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#1F2937", lw=2.0, dashed=False, mutation=14):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    ax.add_patch(arr)
    return arr


def step_box(ax, i, x, y, w, h, title, lines, fc=PALE_BLUE):
    rounded(ax, x, y, w, h, fc=fc, ec=BLUE, lw=1.7, radius=0.012)
    circ = plt.Circle((x + 0.018, y + h - 0.026), 0.014, color=BLUE, zorder=5)
    ax.add_patch(circ)
    text(ax, x + 0.018, y + h - 0.026, str(i), size=10, color="white", weight="bold")
    text(ax, x + w / 2 + 0.006, y + h - 0.030, title, size=12.5, color=BLUE, weight="bold")
    yy = y + h - 0.086
    for line in lines:
        if isinstance(line, tuple):
            s, c, sz, wt = line
        else:
            s, c, sz, wt = line, TEXT, 10.4, "regular"
        text(ax, x + w / 2, yy, s, size=sz, color=c, weight=wt)
        yy -= 0.047


def mini_box(ax, x, y, w, h, label, fc=LIGHT_BLUE, ec="#3273C5", size=9.6):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.1, radius=0.008)
    text(ax, x + w / 2, y + h / 2, label, size=size, color=TEXT, weight="medium")


def panel(ax, x, y, w, h, label, title, ec="#3273C5", fc="white"):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.3, radius=0.010)
    ax.add_patch(Rectangle((x, y + h - 0.042), w, 0.042, facecolor=LIGHT_BLUE, edgecolor=ec, linewidth=1.0))
    text(ax, x + 0.030, y + h - 0.021, label, size=13, color=BLUE, weight="bold")
    text(ax, x + w / 2, y + h - 0.021, title, size=11.5, color=BLUE, weight="bold")


def draw_signal(ax, x, y, w, h):
    xs = [x + w * i / 50 for i in range(51)]
    vals = [
        0.10, -0.07, 0.02, -0.15, 0.20, -0.04, 0.03, -0.11, 0.08, 0.14,
        -0.18, -0.05, 0.10, 0.02, -0.09, 0.19, -0.16, 0.04, 0.12, -0.03,
        -0.08, 0.05, 0.17, -0.20, 0.06, 0.02, -0.04, 0.11, -0.12, 0.16,
        -0.03, -0.09, 0.04, 0.13, -0.18, 0.05, 0.00, -0.06, 0.09, 0.18,
        -0.10, -0.02, 0.15, -0.13, 0.04, 0.08, -0.05, 0.12, -0.15, 0.06,
        0.02,
    ]
    ys = [y + h / 2 + v * h for v in vals]
    ax.plot(xs, ys, color="#111827", linewidth=1.0, zorder=5)


def draw_bars(ax, x, y, w, h):
    vals = [0.36, 0.22, 0.82, 0.30, 0.44]
    bw = w / 8
    for i, v in enumerate(vals):
        bx = x + w * 0.12 + i * bw * 1.25
        ax.add_patch(Rectangle((bx, y), bw * 0.7, h * v, facecolor="#9BD7DD", edgecolor="#0F5260", linewidth=0.8))
    ax.plot([x, x + w * 0.86], [y, y], color="#333", linewidth=1.0)


def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Top row: trial-level flow.
    y_top, h_top = 0.54, 0.42
    gap = 0.010
    x0 = 0.014
    w = (0.972 - gap * 6) / 7
    xs = [x0 + i * (w + gap) for i in range(7)]

    steps = [
        ("入力の到着", ["t 番目の入力 x_t", "が到来する"]),
        ("現在のモデルで予測", ["TCFormer を固定したまま", "raw posterior を出す"]),
        ("memory support を推定", ["external memory から", "p_mem を計算する"]),
        ("予測だけを補正", ["model state は更新せず", "p_raw と p_mem を混合"]),
        ("高信頼か判定", ["p_final の信頼度と", "memory support を確認"]),
        ("memory に保存", ["信頼できる trial だけ", "external memory に追加"]),
        ("次の trial へ", ["model state は維持", "補正器だけを逐次更新"]),
    ]
    for i, (title, lines) in enumerate(steps, 1):
        step_box(ax, i, xs[i - 1], y_top, w, h_top, title, lines)
        if i < 7:
            arrow(ax, xs[i - 1] + w + 0.002, y_top + h_top / 2, xs[i] - 0.004, y_top + h_top / 2)

    # Step-specific illustrations.
    draw_signal(ax, xs[0] + 0.020, y_top + 0.095, w - 0.040, 0.13)

    mini_box(ax, xs[1] + 0.023, y_top + 0.080, w - 0.046, 0.135, "p_raw, feature", fc=LIGHT_BLUE)
    draw_bars(ax, xs[1] + 0.040, y_top + 0.105, w - 0.080, 0.070)

    mini_box(ax, xs[2] + 0.025, y_top + 0.205, w - 0.050, 0.052, "kNN / prototype", fc=LIGHT_BLUE)
    mini_box(ax, xs[2] + 0.025, y_top + 0.120, w - 0.050, 0.052, "density support", fc=LIGHT_BLUE)
    text(ax, xs[2] + w / 2, y_top + 0.080, "→ p_mem", size=11.5, color=BLUE, weight="bold")

    text(ax, xs[3] + w / 2, y_top + 0.225, "p_final =", size=11.5, color=TEXT, weight="bold")
    text(ax, xs[3] + w / 2, y_top + 0.178, "(1−λ_t) p_raw", size=10.2, color=TEXT)
    text(ax, xs[3] + w / 2, y_top + 0.136, "+ λ_t p_mem", size=10.2, color=TEXT)
    mini_box(ax, xs[3] + 0.028, y_top + 0.060, w - 0.056, 0.050, "λ_t は信頼度で制御", fc=LIGHT_GREEN, ec=GREEN)

    mini_box(ax, xs[4] + 0.026, y_top + 0.210, w - 0.052, 0.052, "Yes: memory に保存", fc=LIGHT_GREEN, ec=GREEN)
    mini_box(ax, xs[4] + 0.026, y_top + 0.125, w - 0.052, 0.052, "No: 保存しない", fc=LIGHT_RED, ec=RED)
    text(ax, xs[4] + w / 2, y_top + 0.075, "τ_conf, τ_ent, class balance", size=8.8, color=TEXT)

    mini_box(ax, xs[5] + 0.020, y_top + 0.222, w - 0.040, 0.048, "feature", fc="#F5FBFF")
    mini_box(ax, xs[5] + 0.020, y_top + 0.162, w - 0.040, 0.048, "posterior", fc="#F5FBFF")
    mini_box(ax, xs[5] + 0.020, y_top + 0.102, w - 0.040, 0.048, "pseudo-label", fc="#F5FBFF")
    mini_box(ax, xs[5] + 0.020, y_top + 0.042, w - 0.040, 0.048, "FIFO / class balance", fc="#F5FBFF")

    mini_box(ax, xs[6] + 0.025, y_top + 0.195, w - 0.050, 0.056, "更新しない", fc=LIGHT_RED, ec=RED)
    mini_box(ax, xs[6] + 0.025, y_top + 0.115, w - 0.050, 0.056, "予測だけ補正", fc=LIGHT_GREEN, ec=GREEN)

    # Trial loop dashed arrow.
    arrow(ax, xs[6] + w * 0.50, y_top - 0.018, xs[0] + w * 0.50, y_top - 0.018, color=BLUE, lw=1.8, dashed=True, mutation=12)
    text(ax, 0.50, y_top - 0.010, "次の trial へ", size=12.5, color=BLUE, weight="bold")

    # Down arrow from memory step.
    arrow(ax, xs[5] + w / 2, y_top - 0.002, xs[5] + w / 2, 0.505, color=GREEN, lw=2.4, mutation=16)

    # Panels.
    panel(ax, 0.030, 0.260, 0.290, 0.220, "A", "Replay-SafeCommit からの変更点")
    rows = [
        ("Replay-SafeCommit", "候補更新を replay で検証"),
        ("CMC-OTTA", "model state を動かさず予測補正"),
        ("主なリスク", "誤 commit → memory 汚染へ移る"),
        ("安全制御", "admission と λ_t 上限"),
    ]
    table_x, table_y = 0.040, 0.285
    row_h = 0.041
    col_w = [0.108, 0.155]
    for r, (a, b) in enumerate(rows):
        yy = table_y + (len(rows) - 1 - r) * row_h
        ax.add_patch(Rectangle((table_x, yy), col_w[0], row_h, facecolor="white", edgecolor="#8CB5DF", linewidth=0.8))
        ax.add_patch(Rectangle((table_x + col_w[0], yy), col_w[1], row_h, facecolor="white", edgecolor="#8CB5DF", linewidth=0.8))
        text(ax, table_x + col_w[0] / 2, yy + row_h / 2, a, size=8.9, ha="center", weight="medium")
        text(ax, table_x + col_w[0] + col_w[1] / 2, yy + row_h / 2, b, size=8.7, ha="center")

    panel(ax, 0.335, 0.260, 0.270, 0.220, "B", "posterior 補正のイメージ")
    mini_box(ax, 0.360, 0.365, 0.072, 0.050, "p_raw", fc=LIGHT_BLUE)
    mini_box(ax, 0.360, 0.305, 0.072, 0.050, "p_mem", fc=LIGHT_GREEN, ec=GREEN)
    arrow(ax, 0.435, 0.390, 0.480, 0.363, color="#111827", lw=1.6, mutation=12)
    arrow(ax, 0.435, 0.330, 0.480, 0.345, color="#111827", lw=1.6, mutation=12)
    rounded(ax, 0.482, 0.318, 0.085, 0.070, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.2, radius=0.008)
    text(ax, 0.5245, 0.353, "λ_t で混合", size=9.0, weight="bold")
    arrow(ax, 0.570, 0.353, 0.595, 0.353, color="#111827", lw=1.6, mutation=12)
    mini_box(ax, 0.610, 0.326, 0.075, 0.055, "p_final", fc="#F2F8FF", ec=BLUE)

    panel(ax, 0.620, 0.260, 0.350, 0.220, "C", "external memory から p_mem を推定")
    rounded(ax, 0.640, 0.292, 0.092, 0.145, fc=LIGHT_GREEN, ec=GREEN, lw=1.2, radius=0.030)
    text(ax, 0.686, 0.413, "external", size=9.5, weight="bold")
    text(ax, 0.686, 0.389, "memory", size=9.5, weight="bold")
    text(ax, 0.686, 0.348, "z_i, y_i", size=9.3)
    text(ax, 0.686, 0.320, "p_i, t_i", size=9.3)
    arrow(ax, 0.738, 0.365, 0.775, 0.365, color="#111827", lw=1.5, mutation=12)
    mini_box(ax, 0.780, 0.394, 0.088, 0.050, "kNN", fc="#F6FBFF")
    mini_box(ax, 0.780, 0.330, 0.088, 0.050, "prototype", fc="#F6FBFF")
    arrow(ax, 0.872, 0.365, 0.910, 0.365, color="#111827", lw=1.5, mutation=12)
    mini_box(ax, 0.916, 0.337, 0.043, 0.056, "p_mem", fc=LIGHT_BLUE, size=8.8)

    rounded(ax, 0.030, 0.070, 0.420, 0.155, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.2, radius=0.010)
    text(ax, 0.055, 0.194, "D", size=13, color=TEXT, weight="bold")
    text(ax, 0.105, 0.194, "external memory について", size=11.3, weight="bold", ha="left")
    bullets = [
        "信頼できる target evidence だけを保存する",
        "保存対象: feature, posterior, pseudo-label, confidence, trial index",
        "最大件数を超えたら FIFO。class balance も監視する",
    ]
    yy = 0.160
    for b in bullets:
        text(ax, 0.062, yy, "•", size=12, ha="left", weight="bold")
        text(ax, 0.080, yy, b, size=9.7, ha="left")
        yy -= 0.038

    rounded(ax, 0.525, 0.070, 0.270, 0.155, fc="white", ec="#1F2937", lw=1.1, radius=0.010)
    text(ax, 0.550, 0.194, "E", size=13, color=TEXT, weight="bold")
    text(ax, 0.592, 0.194, "凡例", size=11.3, weight="bold", ha="left")
    arrow(ax, 0.558, 0.150, 0.620, 0.150, color="#111827", lw=2.0, mutation=12)
    text(ax, 0.648, 0.150, "trial 内の処理の流れ", size=10.0, ha="left")
    arrow(ax, 0.558, 0.105, 0.620, 0.105, color=BLUE, lw=1.8, dashed=True, mutation=12)
    text(ax, 0.648, 0.105, "次の trial へ繰り返し", size=10.0, ha="left")

    rounded(ax, 0.815, 0.070, 0.155, 0.155, fc=LIGHT_RED, ec=RED, lw=1.1, radius=0.010)
    text(ax, 0.892, 0.190, "重要", size=11.5, color=RED, weight="bold")
    text(ax, 0.892, 0.150, "BN / prototype /", size=9.5)
    text(ax, 0.892, 0.122, "logit bias は", size=9.5)
    text(ax, 0.892, 0.094, "原則更新しない", size=9.5, color=RED, weight="bold")

    # Title band.
    text(ax, 0.500, 0.990, "Commitless Memory-Corrected OTTA の処理フロー", size=16, color=BLUE, weight="bold")

    png = OUT_DIR / "fig_cmc_otta_flow.png"
    svg = OUT_DIR / "fig_cmc_otta_flow.svg"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
