"""
M3 — CROSS-SESSION (chronological) multi-representation feature-union ceiling +
drift-augmentation upside bound.

Data: per-trial probs/logits[9 subj, E experts, 288 trials, 4 classes] for
source/full_ea/shrink_0.1/... in intentflow/offline/results/research_outputs/260602_expert_portfolio_table{,_seed1}.
bcic2a eval session = 288 trials, assumed CHRONOLOGICAL.

(a) Confirm F4/F6 cleanly under CHRONOLOGICAL split (calibrate first part, test last part):
    - single-alignment head (source logits)
    - multi-alignment fused head (concat source/full/shrink logits = 12-d)
    - label-free static blend 0.3/0.4/0.3 (NO labels, NO fit)
    vs the SAME heads under RANDOM CV (drift-free ceiling).
    Within-session-drift transfer gap = random-CV ceiling - chronological ceiling.
    Sweep calibration size.

(b) Drift-augmentation upside bound: how much does the OPTIMAL FIXED blend weight
    vary per-subject and per-window? If one universal weighting matches per-subject/
    per-window oracle, a single drift-robust trained representation could suffice;
    if the optimal weight is highly conditioned, you NEED conditioning (drift-tracking).
"""
import json
import numpy as np
from itertools import product

RNG = np.random.default_rng(0)
CORE = ["source", "full_ea", "shrink_0.1"]


def load(tag):
    d = np.load(f"intentflow/offline/results/research_outputs/{tag}/expert_portfolio_arrays.npz", allow_pickle=True)
    experts = list(d["experts"])
    idx = {e: experts.index(e) for e in CORE}
    logits = d["logits"]  # [S,E,T,C]
    probs = d["probs"]
    labels = d["labels"]  # [S,T]
    subjects = d["subjects"]
    S, _, T, C = logits.shape
    # core stacks: [S,T, 3, C]
    core_logits = np.stack([logits[:, idx[e]] for e in CORE], axis=2)
    core_probs = np.stack([probs[:, idx[e]] for e in CORE], axis=2)
    return dict(experts=experts, logits=core_logits, probs=core_probs,
                labels=labels, subjects=subjects, S=S, T=T, C=C)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---- multinomial logistic regression (closed enough; simple GD, L2) ----
def fit_logreg(X, y, C_classes, l2=1.0, iters=300, lr=0.5):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    W = np.zeros((d + 1, C_classes))
    Y = np.eye(C_classes)[y]
    for _ in range(iters):
        P = softmax(Xb @ W, axis=1)
        grad = Xb.T @ (P - Y) / n
        grad[:-1] += l2 * W[:-1] / n
        W -= lr * grad
    return W


def acc_logreg(Xtr, ytr, Xte, yte, C, l2=1.0):
    W = fit_logreg(Xtr, ytr, C, l2=l2)
    Xb = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    p = (Xb @ W).argmax(1)
    return (p == yte).mean()


def blend_acc(probs_subj, labels_subj, w=(0.3, 0.4, 0.3)):
    # probs_subj [T,3,C]
    w = np.array(w)
    b = (probs_subj * w[None, :, None]).sum(1)
    return (b.argmax(1) == labels_subj).mean()


# ============ PART (a) ============
def part_a(data, calib_fracs=(0.25, 0.33, 0.5, 0.67)):
    L = data["logits"]      # [S,T,3,C]
    P = data["probs"]
    Y = data["labels"]
    S, T, C = data["S"], data["T"], data["C"]
    out = {}
    for frac in calib_fracs:
        ncal = int(round(T * frac))
        res = {"single_chrono": [], "multi_chrono": [],
               "single_cv": [], "multi_cv": [],
               "blend_late": [], "blend_full": [],
               "source_late": []}
        for s in range(S):
            y = Y[s]
            # CHRONOLOGICAL: calibrate first ncal, test last (T-ncal)
            tr = np.arange(ncal)
            te = np.arange(ncal, T)
            ytr, yte = y[tr], y[te]
            # feature = logits flattened
            Xsrc = L[s, :, 0, :]               # source logits [T,C]
            Xmulti = L[s].reshape(T, 3 * C)    # concat 3 alignments [T,12]
            res["single_chrono"].append(acc_logreg(Xsrc[tr], ytr, Xsrc[te], yte, C))
            res["multi_chrono"].append(acc_logreg(Xmulti[tr], ytr, Xmulti[te], yte, C))
            # label-free blend on the SAME late (test) trials (deployable, no labels)
            res["blend_late"].append(blend_acc(P[s, te], yte))
            res["source_late"].append((Xsrc[te].argmax(1) == yte).mean())
            res["blend_full"].append(blend_acc(P[s], y))
            # RANDOM CV ceiling: same calib size but random split, avg over folds
            cv_single, cv_multi = [], []
            for _f in range(10):
                perm = RNG.permutation(T)
                tri, tei = perm[:ncal], perm[ncal:]
                cv_single.append(acc_logreg(Xsrc[tri], y[tri], Xsrc[tei], y[tei], C))
                cv_multi.append(acc_logreg(Xmulti[tri], y[tri], Xmulti[tei], y[tei], C))
            res["single_cv"].append(np.mean(cv_single))
            res["multi_cv"].append(np.mean(cv_multi))
        out[frac] = {k: np.array(v) for k, v in res.items()}
    return out


# ============ PART (b) ============
# grid over simplex weights for (source, full, shrink); find optimal FIXED blend
def simplex_grid(step=0.05):
    pts = []
    n = int(round(1 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return np.array(pts)  # [G,3]


def blend_acc_grid(probs_block, labels_block, grid):
    # probs_block [T,3,C], grid [G,3] -> acc per grid point
    # weighted sum over alignment axis
    # result [G,T,C]
    w = grid[:, None, :, None]                       # [G,1,3,1]
    blended = (probs_block[None] * w).sum(2)         # [G,T,C]
    preds = blended.argmax(-1)                        # [G,T]
    return (preds == labels_block[None]).mean(1)      # [G]


def part_b(data, n_windows=4):
    P = data["probs"]   # [S,T,3,C]
    Y = data["labels"]
    S, T = data["S"], data["T"]
    grid = simplex_grid(0.05)
    fixed = np.array([0.3, 0.4, 0.3])
    # universal weight = argmax of POOLED accuracy across all subjects/trials
    # build pooled acc per grid
    pooled = np.zeros(len(grid))
    per_subj_opt = []        # [S,3] optimal per subject
    per_subj_opt_acc = []
    per_subj_fixed_acc = []
    per_window_opt = []      # [S, n_windows, 3]
    per_window_opt_acc = []  # [S, n_windows]
    per_window_universal_acc = []
    per_window_subjopt_acc = []
    win_edges = np.linspace(0, T, n_windows + 1).astype(int)
    for s in range(S):
        accg = blend_acc_grid(P[s], Y[s], grid)  # [G]
        pooled += accg * T   # weight by trials (equal here)
        bi = accg.argmax()
        per_subj_opt.append(grid[bi])
        per_subj_opt_acc.append(accg[bi])
        # fixed acc for this subject
        per_subj_fixed_acc.append(blend_acc_grid(P[s], Y[s], fixed[None])[0])
        wopt, wacc = [], []
        for wi in range(n_windows):
            a, b = win_edges[wi], win_edges[wi + 1]
            wa = blend_acc_grid(P[s, a:b], Y[s, a:b], grid)
            wbi = wa.argmax()
            wopt.append(grid[wbi])
            wacc.append(wa[wbi])
        per_window_opt.append(np.array(wopt))
        per_window_opt_acc.append(np.array(wacc))
    pooled /= (S * T)
    universal = grid[pooled.argmax()]
    # now eval universal & subj-opt & window-opt at the window level
    for s in range(S):
        wu, wso, wo = [], [], []
        for wi in range(n_windows):
            a, b = win_edges[wi], win_edges[wi + 1]
            wu.append(blend_acc_grid(P[s, a:b], Y[s, a:b], universal[None])[0])
            wso.append(blend_acc_grid(P[s, a:b], Y[s, a:b], per_subj_opt[s][None])[0])
            wo.append(per_window_opt_acc[s][wi])
        per_window_universal_acc.append(np.array(wu))
        per_window_subjopt_acc.append(np.array(wso))
        per_window_opt_acc[s] = np.array(wo)
    return dict(
        grid=grid, fixed=fixed, universal=universal,
        per_subj_opt=np.array(per_subj_opt),
        per_subj_opt_acc=np.array(per_subj_opt_acc),
        per_subj_fixed_acc=np.array(per_subj_fixed_acc),
        per_window_opt=np.array(per_window_opt),                 # [S,W,3]
        per_window_opt_acc=np.array(per_window_opt_acc),         # [S,W]
        per_window_universal_acc=np.array(per_window_universal_acc),
        per_window_subjopt_acc=np.array(per_window_subjopt_acc),
        win_edges=win_edges,
    )


def fmt(a):
    return np.round(np.asarray(a) * 100, 2)


def main():
    report = {}
    for tag, label in [("260602_expert_portfolio_table", "seed0"),
                       ("260602_expert_portfolio_table_seed1", "seed1")]:
        data = load(tag)
        print("=" * 70)
        print(f"{label}  T={data['T']} C={data['C']} S={data['S']}")
        A = part_a(data)
        report[label] = {"part_a": {}, "part_b": {}}
        for frac, r in A.items():
            row = {k: float(np.mean(v)) for k, v in r.items()}
            gap_single = float(np.mean(r["single_cv"] - r["single_chrono"]))
            gap_multi = float(np.mean(r["multi_cv"] - r["multi_chrono"]))
            row["drift_gap_single_pp"] = gap_single * 100
            row["drift_gap_multi_pp"] = gap_multi * 100
            report[label]["part_a"][f"calib_{frac}"] = row
            print(f"-- calib_frac={frac} (ncal={int(round(data['T']*frac))}, test on rest) --")
            print(f"   blend_late (label-free, deploy) : {fmt(r['blend_late']).mean():.2f}  per-subj={fmt(r['blend_late'])}")
            print(f"   source_late (no adapt)          : {fmt(r['source_late']).mean():.2f}")
            print(f"   single-head CHRONO              : {fmt(r['single_chrono']).mean():.2f}")
            print(f"   multi-head  CHRONO              : {fmt(r['multi_chrono']).mean():.2f}")
            print(f"   single-head RANDOM-CV ceiling   : {fmt(r['single_cv']).mean():.2f}")
            print(f"   multi-head  RANDOM-CV ceiling   : {fmt(r['multi_cv']).mean():.2f}")
            print(f"   DRIFT GAP single (cv-chrono)    : {gap_single*100:+.2f} pp")
            print(f"   DRIFT GAP multi  (cv-chrono)    : {gap_multi*100:+.2f} pp")
            print(f"   multi CHRONO - blend_late       : {(np.mean(r['multi_chrono'])-np.mean(r['blend_late']))*100:+.2f} pp")
        B = part_b(data)
        print("-- PART B: optimal blend-weight spread --")
        print(f"   universal (pooled-opt) weight   : {np.round(B['universal'],3)}")
        print(f"   fixed weight                    : {B['fixed']}")
        pso = B["per_subj_opt"]
        print(f"   per-subject optimal weights (src,full,shrink):")
        for s in range(data["S"]):
            print(f"      S{data['subjects'][s]}: {np.round(pso[s],2)}  opt_acc={pso is not None and B['per_subj_opt_acc'][s]*100:.1f}  fixed_acc={B['per_subj_fixed_acc'][s]*100:.1f}")
        std_subj = pso.std(0)
        rng_subj = pso.max(0) - pso.min(0)
        print(f"   per-subject weight STD          : {np.round(std_subj,3)}  (src,full,shrink)")
        print(f"   per-subject weight RANGE        : {np.round(rng_subj,3)}")
        # per-window spread
        pwo = B["per_window_opt"]   # [S,W,3]
        std_win_within = pwo.std(1).mean(0)  # avg over subj of within-subject window std
        print(f"   per-WINDOW (within-subj) weight STD: {np.round(std_win_within,3)}")
        # accuracy: universal vs subj-opt vs window-opt at WINDOW granularity
        u = B["per_window_universal_acc"].mean()
        so = B["per_window_subjopt_acc"].mean()
        wo = B["per_window_opt_acc"].mean()
        print(f"   WINDOW-level acc: universal={u*100:.2f}  subj-opt={so*100:.2f}  window-opt(oracle)={wo*100:.2f}")
        print(f"   gap universal->subj-opt = {(so-u)*100:+.2f} pp ; subj-opt->window-oracle = {(wo-so)*100:+.2f} pp")
        report[label]["part_b"] = dict(
            universal=B["universal"].tolist(),
            per_subj_opt=pso.tolist(),
            per_subj_opt_acc=B["per_subj_opt_acc"].tolist(),
            per_subj_fixed_acc=B["per_subj_fixed_acc"].tolist(),
            per_subj_weight_std=std_subj.tolist(),
            per_subj_weight_range=rng_subj.tolist(),
            per_window_within_subj_weight_std=std_win_within.tolist(),
            window_acc_universal=float(u),
            window_acc_subjopt=float(so),
            window_acc_windoworacle=float(wo),
        )
    with open("intentflow/offline/results/research_outputs/260602_m3_cross_session_drift.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote intentflow/offline/results/research_outputs/260602_m3_cross_session_drift.json")


if __name__ == "__main__":
    main()
