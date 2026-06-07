"""Decisive test of the PHYSIOLOGICAL-PLAUSIBILITY VETO direction (2026-06-05).

Question (NOT accuracy; accuracy ceiling is closed): is a model-EXTERNAL,
non-learned ERD lateralization signal COMPLEMENTARY to the trained head's
errors? Specifically: on trials where the TRUE source head is CONFIDENTLY
WRONG, does the ERD signal know the correct class? If yes => a label-free,
model-external confident-wrong VETO is possible (internal UQ is structurally
0% in the high-confidence region; cross-family witnesses are trained models
that share the boundary-drift failure mode).

True 2b source logits are cached (sanity: mean acc must ~= 87.7). ERD is pure
physiology (band-power lateralization), labels used only to (a) score and (b)
fix ONE global orientation bit per subject. No EA (would whiten away laterality).
Run with intentflow conda env.
"""
import os, sys, glob, warnings, json
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # intentflow/offline
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from datamodules.bcic4_2b import BCICIV2b

RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results"
SRC = sorted(glob.glob(f"{RES}/TCFormer_bcic2b_seed-0_aug-True_GPU0_*"),
             key=lambda d: len(glob.glob(f"{d}/logits_s*_TCFormer.npy")))[-1]
print(f"SRC_DIR={os.path.basename(SRC)}")

# Exact training preprocessing (matches cached logits; loads all 9 subjects).
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=-0.5,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=None)

def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

# band-power lateralization: C3=idx0, C4=idx2. ERD = contralateral power DECREASE.
bb, ba = butter(4, [8/125., 30/125.], btype="band")
def band(X):  # (n,3,T) -> filtered
    return filtfilt(bb, ba, X, axis=-1)
def li(X, t0, t1):  # log-power(C4) - log-power(C3) on samples [t0:t1]
    Xf = band(X[:, :, t0:t1])
    lp = np.log(np.var(Xf, axis=-1) + 1e-8)  # (n,3)
    return lp[:, 2] - lp[:, 0]
def getxy(ds):
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    y = np.array([int(ds[i][1]) for i in range(len(ds))]); return X, y

# windows (in samples @250Hz over the 1000-sample/4.0s epoch)
WINDOWS = {"full_0_4s": (0, 1000), "erd_0.5_2.5s": (125, 625)}
CONF_Q = 0.5   # "confident" = top 50% by softmax margin
BUDGET = 0.15  # abstain budget for veto comparison

rows = []
for sid in range(1, 10):
    z = np.load(f"{SRC}/logits_s{sid}_TCFormer.npy")
    lab = np.load(f"{SRC}/features_s{sid}_TCFormer.npz", allow_pickle=True)["labels"].astype(int)
    p = sm(z); pred = p.argmax(1); margin = np.sort(p, 1)[:, -1] - np.sort(p, 1)[:, -2]
    src_acc = (pred == lab).mean() * 100

    dm = BCICIV2b(prep, sid); dm.setup()
    X, y = getxy(dm.test_dataset)
    if len(y) != len(lab) or (y != lab).mean() > 0.01:
        print(f"S{sid}: ALIGN WARN n_dm={len(y)} n_cached={len(lab)} "
              f"match={(y==lab).mean() if len(y)==len(lab) else 'NA'}")
        n = min(len(y), len(lab))
        X, y, pred, margin, lab = X[:n], y[:n], pred[:n], margin[:n], lab[:n]
    yy = lab  # ground truth (cached, trial-aligned)

    conf = margin >= np.quantile(margin, CONF_Q)
    e_head = pred != yy
    cw = conf & e_head  # confident-wrong

    rec = {"sid": sid, "src_acc": round(src_acc, 2), "n": int(len(yy)),
           "n_conf_wrong": int(cw.sum())}
    for wname, (t0, t1) in WINDOWS.items():
        LI = li(X, t0, t1)
        LIc = LI - np.median(LI)             # label-free centering (2b balanced)
        raw_pred = (LIc > 0).astype(int)
        a = (raw_pred == yy).mean()
        flip = a < 0.5                        # ONE global orientation bit
        erd_pred = (1 - raw_pred) if flip else raw_pred
        erd_acc = max(a, 1 - a) * 100
        # complementarity: ERD accuracy ON the head's confident-wrong trials
        erd_on_cw = (erd_pred[cw] == yy[cw]).mean() * 100 if cw.sum() else float("nan")
        # error decorrelation (Jaccard error overlap, all trials)
        e_erd = erd_pred != yy
        jacc = (e_head & e_erd).sum() / max(1, (e_head | e_erd).sum()) * 100
        # veto @ budget: score = physiological disagreement strength
        dis = (pred != erd_pred).astype(float) * np.abs(LIc)
        k = max(1, int(BUDGET * len(yy)))
        vetoed = np.zeros(len(yy), bool); vetoed[np.argsort(dis)[::-1][:k]] = True
        vetoed &= (dis > 0)                   # only true disagreements
        veto_prec = e_head[vetoed].mean() * 100 if vetoed.sum() else float("nan")
        cw_recall = vetoed[cw].mean() * 100 if cw.sum() else float("nan")
        # internal-UQ baseline @ same budget: abstain lowest margin
        uq = np.zeros(len(yy), bool); uq[np.argsort(margin)[:k]] = True
        uq_cw_recall = uq[cw].mean() * 100 if cw.sum() else float("nan")
        rec[wname] = {
            "erd_acc": round(erd_acc, 1), "erd_on_confwrong": round(erd_on_cw, 1),
            "err_jaccard": round(jacc, 1),
            "veto_precision": round(veto_prec, 1), "veto_cw_recall": round(cw_recall, 1),
            "uq_cw_recall": round(uq_cw_recall, 1),
        }
    rows.append(rec)
    w = rec["erd_0.5_2.5s"]
    print(f"S{sid}: src={src_acc:5.2f} n_cw={cw.sum():3d} | "
          f"ERD_acc={w['erd_acc']:4.1f} ERD_on_CW={w['erd_on_confwrong']!s:>5} "
          f"jacc={w['err_jaccard']:4.1f} | veto_prec={w['veto_precision']!s:>5} "
          f"cw_rec={w['veto_cw_recall']!s:>5} vs uq_cw_rec={w['uq_cw_recall']!s:>5}", flush=True)

# pooled (weight by n_conf_wrong for ERD_on_CW, which is the decisive number)
def pooled_on_cw(win):
    num = den = 0
    for r in rows:
        if r["n_conf_wrong"] > 0 and not np.isnan(r[win]["erd_on_confwrong"]):
            num += r[win]["erd_on_confwrong"] / 100 * r["n_conf_wrong"]; den += r["n_conf_wrong"]
    return num / den * 100 if den else float("nan")

print("\n=== POOLED (2b, true source, seed0) ===")
for win in WINDOWS:
    macc = np.mean([r[win]["erd_acc"] for r in rows])
    on_cw = pooled_on_cw(win)
    vr = np.nanmean([r[win]["veto_cw_recall"] for r in rows])
    ur = np.nanmean([r[win]["uq_cw_recall"] for r in rows])
    print(f"{win:14s}: ERD_acc(mean)={macc:.1f} | ERD_on_CONF-WRONG(pooled)={on_cw:.1f} "
          f"(chance=50) | veto cw-recall={vr:.1f}% vs internal-UQ cw-recall={ur:.1f}% @budget15%")
print(f"\nmean src = {np.mean([r['src_acc'] for r in rows]):.2f} (sanity: ~87.7)")
print("VERDICT: ERD_on_CONF-WRONG >> 50 => model-external veto ALIVE (knows answer where head confidently fails).")
print("         ~= 50 => ERD uninformative exactly where needed => veto DEAD (confident-wrong stays label-locked).")

out = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260605_erd_veto_2b.json"
json.dump({"src_dir": os.path.basename(SRC), "rows": rows}, open(out, "w"), indent=2)
print(f"saved -> {out}")
