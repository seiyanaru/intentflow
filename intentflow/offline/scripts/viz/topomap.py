"""再利用可能なトポマップ生成器（人への説明用・JPEG出力）。

「どの電極が信号を持つか」を頭部図で可視化する。2モード：
  discriminative : 電極ごとの L手 vs R手 判別度（point-biserial r, 符号付き）＝decodability/worthiness
  power          : 電極ごとの band-power（log）＝signal strength/quality
  both           : 上記2枚を並べる（quality≠worthiness の対比に）

使い方の例:
  python intentflow/offline/scripts/viz/topomap.py --dataset bcic2a --mode discriminative
  python intentflow/offline/scripts/viz/topomap.py --dataset bcic2a --subjects 2,5,8 --mode both --out /tmp/x.jpg
引数:
  --dataset {bcic2a}     データセット（>=4ch のもの。bcic2bは3chでトポマップ非対応）
  --subjects             "all" / "1-9" / "1,3,5"（既定 all）
  --session {train,test} 既定 test
  --mode {discriminative,power,both}  既定 discriminative
  --fmin --fmax          帯域（既定 8-30Hz）
  --out                  出力JPEGパス（既定 results/figures/ に自動命名）
  --dpi                  既定 150
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
import mne; mne.set_log_level("ERROR")
from scipy.signal import butter, filtfilt
from scipy.stats import pointbiserialr
from moabb.paradigms import LeftRightImagery

DATASETS = {"bcic2a": ("BNCI2014_001", 9)}  # name, n_subjects ; 拡張時はここに追加

def parse_subjects(s, n):
    if s == "all": return list(range(1, n + 1))
    if "-" in s: a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]

def load(ds_obj, prm, subj):
    ep, lab, meta = prm.get_data(dataset=ds_obj, subjects=[subj], return_epochs=True)
    return ep, lab, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bcic2a", choices=list(DATASETS))
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--session", default="test", choices=["train", "test"])
    ap.add_argument("--mode", default="discriminative", choices=["discriminative", "power", "both"])
    ap.add_argument("--fmin", type=float, default=8.0); ap.add_argument("--fmax", type=float, default=30.0)
    ap.add_argument("--out", default=None); ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    name, n = DATASETS[a.dataset]
    from moabb.datasets import BNCI2014_001
    ds_obj = {"BNCI2014_001": BNCI2014_001}[name]()
    subs = parse_subjects(a.subjects, n)
    prm = LeftRightImagery(resample=250, fmin=a.fmin, fmax=a.fmax)
    b, ba = butter(4, [a.fmin / 125., a.fmax / 125.], btype="band")
    logbp = lambda X: np.log(np.var(filtfilt(b, ba, X.astype(np.float64), axis=-1), axis=-1) + 1e-8)

    # 収集
    info = None; data = []
    for s in subs:
        ep, lab, meta = load(ds_obj, prm, s)
        if info is None:
            info = ep.info.copy(); info.set_montage("standard_1020", on_missing="warn")
        sess = meta["session"].values; uniq = np.unique(sess)
        sel = sess == (uniq[1] if a.session == "test" else uniq[0])
        X = ep.get_data()[sel]; y = (lab[sel] == "right_hand").astype(int)
        F = logbp(X)  # (trial, ch)
        disc = np.array([pointbiserialr(y, F[:, c]).correlation for c in range(F.shape[1])])
        data.append(dict(s=s, disc=disc, power=F.mean(0)))
    if info["nchan"] < 4:
        sys.exit(f"[error] {a.dataset} は {info['nchan']}ch：トポマップは>=4ch必要")

    # mne/moabb の import で font がリセットされることがあるため描画直前に再設定
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    panels = [a.mode] if a.mode != "both" else ["discriminative", "power"]
    CFG = {"discriminative": dict(key="disc", cmap="RdBu_r", sym=True,
              title="L手 vs R手 判別度（どの電極にクラス信号があるか）",
              cbar="L手 ←  判別度 r  → R手"),
           "power": dict(key="power", cmap="YlOrRd", sym=False,
              title="各電極の信号の強さ（band-power）", cbar="弱い ← log power → 強い")}
    ncols = int(np.ceil(np.sqrt(len(subs)))); nrows = int(np.ceil(len(subs) / ncols))

    outs = []
    for panel in panels:
        cfg = CFG[panel]; vals = [d[cfg["key"]] for d in data]
        if cfg["sym"]:
            vmax = max(np.abs(v).max() for v in vals); vlim = (-vmax, vmax)
        else:
            vlim = (min(v.min() for v in vals), max(v.max() for v in vals))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.3, nrows * 3.5))
        axes = np.atleast_1d(axes).ravel()
        for ax, d in zip(axes, data):
            im, _ = mne.viz.plot_topomap(d[cfg["key"]], info, axes=ax, show=False,
                                         cmap=cfg["cmap"], vlim=vlim, contours=4, sensors=True)
            ax.set_title(f"S{d['s']}", fontsize=12)
        for ax in axes[len(data):]: ax.axis("off")
        fig.suptitle(f"{cfg['title']}（{a.dataset}, {a.session} session, {a.fmin:.0f}-{a.fmax:.0f}Hz）",
                     fontsize=13, weight="bold", y=0.99)
        cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.5, location="right"); cb.set_label(cfg["cbar"], fontsize=10)
        out = a.out
        if out is None:
            d = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/figures"
            os.makedirs(d, exist_ok=True)
            out = f"{d}/topomap_{a.dataset}_{a.session}_{panel}.jpg"
        elif a.mode == "both":
            base, ext = os.path.splitext(a.out); out = f"{base}_{panel}{ext or '.jpg'}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
        outs.append(out)
    for o in outs: print("saved", o)

if __name__ == "__main__":
    main()
