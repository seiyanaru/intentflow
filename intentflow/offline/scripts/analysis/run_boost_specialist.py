"""Boosting-style cheap specialist vs uniform-D1 DA-L1 on bcic2a session drift.

HYPOTHESIS: train the EA-Riemannian tangent+LDA specialist with SAMPLE WEIGHTS that
up-weight train trials lying in TCFormer's CONFIDENT-ERROR subspace, so the specialist
specializes in T's failure mode. Ensemble specialist + blend. Does targeting T's failure
subspace beat the uniform-D1 DA-L1 (87.15) on held-out?

Two leak-free protocols:
  (P_LOSO)  failure prior derived from the OTHER 8 subjects' confident-error class/region;
            specialist trained on held-out subject's session_T; evaluate on its session_E.
  (P_SEED)  failure region defined on seed0 TCFormer eval; ensemble evaluated on seed1.
            (D1 tangent features are seed-independent: same raw EEG.)

Failure-region weighting (label-free at train time):
  - characterize each train trial by similarity (in tangent space) to the centroid of
    TCFormer confident-error eval trials, using class-conditional structure.
  - weight_i = 1 + beta * membership_i, membership in [0,1].
This needs no train-session TCFormer preds (we don't have them); the failure signal is
imported from the development source (other subjects / seed0) -> held-out subject/seed
sees NO information from its own test labels.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
EP0 = ROOT + "260602_expert_portfolio_table/expert_portfolio_arrays.npz"
EP1 = ROOT + "260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz"
CACHE = ROOT + "260603_boost_tangent_cache.npz"

def load_blend(path):
    d = np.load(path, allow_pickle=True)
    ex = list(d["experts"]); si = {e: i for i, e in enumerate(ex)}
    p = d["probs"]
    blend = 0.3*p[:, si["source"]] + 0.4*p[:, si["full_ea"]] + 0.3*p[:, si["shrink_0.1"]]
    src = p[:, si["source"]]
    return src, blend, d["labels"].astype(int)

def acc(P, y):
    return (P.argmax(-1) == y).mean()

c = np.load(CACHE, allow_pickle=True)
Ttr = [np.asarray(x, float) for x in c["Ttr"]]
ytr = [np.asarray(x, int) for x in c["ytr"]]
Tte = [np.asarray(x, float) for x in c["Tte"]]
yte = [np.asarray(x, int) for x in c["yte"]]

src0, blend0, lab0 = load_blend(EP0)
src1, blend1, lab1 = load_blend(EP1)
assert all(np.array_equal(lab0[s], yte[s]) for s in range(9)), "label order mismatch"

# ---- weighted shrinkage-LDA (matches sklearn lsqr+auto when weights uniform) ----
class WeightedShrinkLDA:
    """Sample-weighted LDA with Ledoit-Wolf-style auto shrinkage on the (weighted)
    within-class covariance. Reduces to sklearn solver='lsqr', shrinkage='auto'
    behaviour when all weights are equal (verified numerically downstream)."""
    def fit(self, X, y, w=None):
        n, d = X.shape
        if w is None:
            w = np.ones(n)
        w = w * (n / w.sum())              # normalize so sum(w)=n
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        priors, means = [], []
        Sw = np.zeros((d, d))
        for c in self.classes_:
            m = y == c
            wc = w[m]; Xc = X[m]
            sw = wc.sum()
            mu = (wc[:, None] * Xc).sum(0) / sw
            means.append(mu); priors.append(sw / w.sum())
            D = Xc - mu
            Sw += (wc[:, None] * D).T @ D
        Sw /= w.sum()
        self.means_ = np.array(means); self.priors_ = np.array(priors)
        # Ledoit-Wolf auto shrinkage target = scaled identity
        mu_tr = np.trace(Sw) / d
        target = mu_tr * np.eye(d)
        # shrinkage intensity (sklearn 'auto' uses Ledoit-Wolf); approximate with LW on
        # the pooled centered data (weighted)
        Xc_all = np.vstack([X[y == c] - means[i] for i, c in enumerate(self.classes_)])
        wc_all = np.concatenate([w[y == c] for c in self.classes_])
        emp = (wc_all[:, None] * Xc_all).T @ Xc_all / wc_all.sum()
        var = ((wc_all[:, None] * (Xc_all**2))[:, :, None] *
               Xc_all[:, None, :]) if False else None
        # LW intensity
        n_eff = wc_all.sum()
        X2 = Xc_all
        phi = 0.0
        # E[(x_i x_j - s_ij)^2] summed
        for chunk in range(0, X2.shape[0], 256):
            B = X2[chunk:chunk+256]; wb = wc_all[chunk:chunk+256]
            outer = np.einsum('ni,nj->nij', B, B)
            phi += (wb[:, None, None] * (outer - emp[None])**2).sum()
        phi /= n_eff
        gamma = ((emp - target)**2).sum()
        shrink = max(0.0, min(1.0, (phi / n_eff) / gamma)) if gamma > 0 else 0.0
        self.cov_ = (1 - shrink) * emp + shrink * target
        self.cov_ += 1e-9 * np.eye(d)
        self.coef_full = np.linalg.solve(self.cov_, self.means_.T).T  # K x d
        self.intercept_ = (-0.5 * np.einsum('kd,kd->k', self.means_, self.coef_full)
                           + np.log(self.priors_))

    def predict_proba_full(self, X, n=4):
        scores = X @ self.coef_full.T + self.intercept_  # N x K
        scores -= scores.max(1, keepdims=True)
        e = np.exp(scores); p = e / e.sum(1, keepdims=True)
        P = np.zeros((X.shape[0], n)); P[:, self.classes_] = p
        return P

def fit_specialist(Xtr, ytr, w=None):
    m = WeightedShrinkLDA(); m.fit(Xtr, ytr, w); return m

def predict4(lda, Xte, n=4):
    return lda.predict_proba_full(Xte, n)

# ---- per-subject specialist probs (uniform) ----
def uniform_specialist():
    out = []
    for s in range(9):
        lda = fit_specialist(Ttr[s], ytr[s])
        out.append(predict4(lda, Tte[s]))
    return out

# membership of train trials in TCFormer confident-error subspace.
# fail_feats: tangent features of confident-error trials (from a development source).
# returns weight per train trial = 1 + beta * normalized similarity.
def boost_weights(Xtr_s, fail_feats, beta, k=15):
    if len(fail_feats) == 0:
        return np.ones(len(Xtr_s))
    # standardize jointly for distance
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0) + 1e-8
    A = (Xtr_s - mu)/sd; F = (fail_feats - mu)/sd
    # for each train trial, distance to k nearest failure prototypes (mean)
    # (memory-light: loop in chunks)
    d2 = ((A[:, None, :] - F[None, :, :])**2).sum(-1)  # (ntr, nfail)
    kk = min(k, F.shape[0])
    nn = np.sort(d2, axis=1)[:, :kk].mean(1)
    # convert distance -> membership via rank (robust to scale)
    r = nn.argsort().argsort() / (len(nn)-1)   # 0=closest
    membership = 1.0 - r                        # 1=closest to failure region
    return 1.0 + beta * membership

# get TCFormer confident-error tangent feats on EVAL for subject s given that seed's blend
def conf_error_feats(s, src_probs, conf_q=0.6):
    p = src_probs[s]
    pred = p.argmax(1); y = yte[s]
    conf = p.max(1)
    wrong = pred != y
    thr = np.quantile(conf, conf_q)  # "confident" = upper part of confidence dist
    sel = wrong & (conf >= thr)
    return Tte[s][sel]

# ============ PROTOCOL P_LOSO (seed0) ============
# failure prior for held-out subject = pooled confident-error tangent feats of OTHER 8 subjects
def run_loso(beta, conf_q=0.6, w_spec=0.3):
    spec = []
    for s in range(9):
        others = [conf_error_feats(o, src0, conf_q) for o in range(9) if o != s]
        fail = np.concatenate([f for f in others if len(f) > 0], 0)
        w = boost_weights(Ttr[s], fail, beta)
        lda = fit_specialist(Ttr[s], ytr[s], w)
        spec.append(predict4(lda, Tte[s]))
    spec = np.array(spec)
    ens = blend0 + w_spec * spec
    return np.mean([acc(ens[s], yte[s]) for s in range(9)]), spec

# ============ PROTOCOL P_SEED (build seed0 -> eval seed1) ============
# failure region from seed0 src eval (own subject ok: eval on seed1 which is held out)
def run_seed(beta, conf_q=0.6, w_spec=0.3):
    spec = []
    for s in range(9):
        fail = conf_error_feats(s, src0, conf_q)  # seed0 failure region
        w = boost_weights(Ttr[s], fail, beta)
        lda = fit_specialist(Ttr[s], ytr[s], w)
        spec.append(predict4(lda, Tte[s]))
    spec = np.array(spec)
    ens = blend1 + w_spec * spec   # evaluate on SEED1 blend
    return np.mean([acc(ens[s], yte[s]) for s in range(9)]), spec

if __name__ == "__main__":
    # references
    uspec = np.array(uniform_specialist())
    print("=== references (seed0 eval) ===")
    print("source     ", round(np.mean([acc(src0[s], yte[s]) for s in range(9)]), 4))
    print("blend      ", round(np.mean([acc(blend0[s], yte[s]) for s in range(9)]), 4))
    dal1_0 = blend0 + 0.3*uspec
    print("DA-L1 unif ", round(np.mean([acc(dal1_0[s], yte[s]) for s in range(9)]), 4))
    print("=== references (seed1 eval) ===")
    print("source     ", round(np.mean([acc(src1[s], yte[s]) for s in range(9)]), 4))
    print("blend      ", round(np.mean([acc(blend1[s], yte[s]) for s in range(9)]), 4))
    dal1_1 = blend1 + 0.3*uspec
    print("DA-L1 unif ", round(np.mean([acc(dal1_1[s], yte[s]) for s in range(9)]), 4))

    print("\n=== P_LOSO (seed0, failure prior from other 8 subjects) ===")
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0]:
        a, _ = run_loso(beta)
        print(f"  beta={beta:>4}  ensemble acc={a:.4f}  (vs DA-L1 {np.mean([acc(dal1_0[s],yte[s]) for s in range(9)]):.4f})")

    print("\n=== P_SEED (build seed0 failure region -> eval seed1) ===")
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0]:
        a, _ = run_seed(beta)
        print(f"  beta={beta:>4}  ensemble acc={a:.4f}  (vs DA-L1seed1 {np.mean([acc(dal1_1[s],yte[s]) for s in range(9)]):.4f})")
