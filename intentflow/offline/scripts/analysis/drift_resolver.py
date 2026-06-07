"""Drift-aware cross-model per-trial RESOLVER (T=source deep vs D1=EA-Riemannian).

On T-vs-D1 disagreement trials, a small logistic resolver decides which model to trust,
trained CROSS-SUBJECT (LOSO) or CROSS-SEED. Label-free per-trial features include a
DRIFT signal = distance of the trial's source 64-d feature from the source-TRAIN centroid.

Final prediction:
  - agreement trials: that shared class
  - disagreement trials: resolver picks T or D1, take its argmax class

Compared on the SAME held-out against:
  - source T (82.72), blend (85.76), DA-L1 = blend + 0.3*D1 (87.15)
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OFFLINE = os.path.abspath(os.path.join(HERE, "..", ".."))
DOCS = "/mnt/data/seiya.narukawa/intentflow/docs/research_progress"

PORT0 = f"{DOCS}/260602_expert_portfolio_table/expert_portfolio_arrays.npz"
PORT1 = f"{DOCS}/260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz"
D1_0 = f"{DOCS}/260603_diverse_riemann_preds.npz"
FEAT0 = f"{OFFLINE}/results/source_train_features_s0.npz"

EPS = 1e-8


def kl(p, q):
    p = np.clip(p, EPS, 1); q = np.clip(q, EPS, 1)
    return (p * np.log(p / q)).sum(-1)


def entropy(p):
    p = np.clip(p, EPS, 1)
    return -(p * np.log(p)).sum(-1)


def topk_margin(p):
    s = np.sort(p, axis=-1)
    return s[..., -1] - s[..., -2]


def load_seed0():
    port = np.load(PORT0, allow_pickle=True)
    labels = port["labels"].astype(int)            # (9,288)
    T = port["probs"][:, 0].astype(np.float64)     # source (9,288,4)
    blend = (0.3 * port["probs"][:, 0] + 0.4 * port["probs"][:, 1]
             + 0.3 * port["probs"][:, 2]).astype(np.float64)
    D1 = np.load(D1_0)["probs"].astype(np.float64)  # (9,288,4)
    feat = np.load(FEAT0)
    train_feat = {s: feat[f"train_feat_{s+1}"] for s in range(9)}
    eval_feat = {s: feat[f"eval_feat_{s+1}"] for s in range(9)}
    return labels, T, D1, blend, train_feat, eval_feat


def drift_signal(eval_f, train_f):
    """Per-trial label-free drift: distance of eval source feature from source-TRAIN dist.
    Returns dict of features: euclid to train mean, mahalanobis (diag), z-cosine.
    Train stats only -> leak-free for the held-out eval trial itself (no eval labels)."""
    mu = train_f.mean(0)
    sd = train_f.std(0) + EPS
    # euclidean to centroid (standardized by train scale)
    z = (eval_f - mu) / sd
    euclid = np.linalg.norm(eval_f - mu, axis=1)
    mahal_diag = np.linalg.norm(z, axis=1)
    # cosine distance to mean direction
    mu_n = mu / (np.linalg.norm(mu) + EPS)
    ef_n = eval_f / (np.linalg.norm(eval_f, axis=1, keepdims=True) + EPS)
    cos = 1.0 - ef_n @ mu_n
    # distance to nearest train sample (knn-1) in standardized space
    return np.stack([euclid, mahal_diag, cos], axis=1)  # (288,3)


def build_trial_features(T, D1, eval_f, train_f):
    """All per-trial label-free features for the resolver, for one subject."""
    Tmax = T.max(-1); Tmarg = topk_margin(T); Tent = entropy(T)
    Dmax = D1.max(-1); Dmarg = topk_margin(D1)
    skl = 0.5 * (kl(T, D1) + kl(D1, T))
    conf_gap = Tmax - Dmax
    drift = drift_signal(eval_f, train_f)  # (N,3)
    feats = np.column_stack([Tmax, Tmarg, Tent, Dmax, Dmarg, skl, conf_gap, drift])
    return feats  # (N, 10)


def resolver_target(T, D1, y):
    """Label for disagreement trials: 1 => trust D1 (D1 correct & T wrong), else 0 => trust T."""
    tp = T.argmax(-1); dp = D1.argmax(-1)
    # target=1 when D1 strictly better (D1 right, T wrong); else 0 (prefer T)
    tgt = ((dp == y) & (tp != y)).astype(int)
    return tgt


def acc_per_subject(pred, labels):
    return np.array([(pred[s] == labels[s]).mean() for s in range(9)])


def final_pred_from_resolver(T, D1, choose_d1):
    """choose_d1: bool array (N,) True => use D1 argmax, else T argmax. Agreement handled outside."""
    tp = T.argmax(-1); dp = D1.argmax(-1)
    out = tp.copy()
    out[choose_d1] = dp[choose_d1]
    return out


def run_loso(labels, T, D1, blend, train_feat, eval_feat, C=1.0, thr=0.5):
    # precompute per-subject trial features + targets + disagreement masks
    feats = {}; tgts = {}; dis = {}
    for s in range(9):
        feats[s] = build_trial_features(T[s], D1[s], eval_feat[s], train_feat[s])
        tgts[s] = resolver_target(T[s], D1[s], labels[s])
        dis[s] = T[s].argmax(-1) != D1[s].argmax(-1)

    res_pred = {}
    for s in range(9):
        # train on disagreement trials of the other 8 subjects
        Xtr = np.concatenate([feats[o][dis[o]] for o in range(9) if o != s])
        ytr = np.concatenate([tgts[o][dis[o]] for o in range(9) if o != s])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(Xtr), ytr)

        pred = T[s].argmax(-1).copy()  # default: agreement => shared, disagreement default T
        ds = dis[s]
        if ds.sum() > 0:
            Xte = scaler.transform(feats[s][ds])
            p_d1 = clf.predict_proba(Xte)[:, 1]
            choose_d1 = p_d1 >= thr
            tp = T[s].argmax(-1); dp = D1[s].argmax(-1)
            idx = np.where(ds)[0]
            pred[idx[choose_d1]] = dp[idx[choose_d1]]
        res_pred[s] = pred
    return res_pred


def main():
    labels, T, D1, blend, train_feat, eval_feat = load_seed0()

    src_pred = {s: T[s].argmax(-1) for s in range(9)}
    blend_pred = {s: blend[s].argmax(-1) for s in range(9)}
    da_l1 = blend + 0.3 * D1
    da_pred = {s: da_l1[s].argmax(-1) for s in range(9)}

    acc_src = acc_per_subject(src_pred, labels)
    acc_blend = acc_per_subject(blend_pred, labels)
    acc_da = acc_per_subject(da_pred, labels)

    print("=== Baselines (seed0, all subjects) ===")
    print(f"source  mean {acc_src.mean()*100:.2f}")
    print(f"blend   mean {acc_blend.mean()*100:.2f}")
    print(f"DA-L1   mean {acc_da.mean()*100:.2f}")
    print()

    for C in [0.05, 0.1, 0.3, 1.0]:
        for thr in [0.5]:
            res = run_loso(labels, T, D1, blend, train_feat, eval_feat, C=C, thr=thr)
            acc_res = acc_per_subject(res, labels)
            hsc = (acc_res > acc_src).sum() - (acc_res < acc_src).sum()
            print(f"RESOLVER (LOSO logistic) C={C} thr={thr}: mean {acc_res.mean()*100:.2f} "
                  f"| vs blend {(acc_res.mean()-acc_blend.mean())*100:+.2f} "
                  f"vs DA-L1 {(acc_res.mean()-acc_da.mean())*100:+.2f} | HSC(vs src) {hsc:+d}")

    # detailed at best C
    res = run_loso(labels, T, D1, blend, train_feat, eval_feat, C=0.1, thr=0.5)
    acc_res = acc_per_subject(res, labels)
    print()
    print("per-subject (C=0.1):")
    for s in range(9):
        print(f"  s{s+1}: src {acc_src[s]*100:.1f} blend {acc_blend[s]*100:.1f} "
              f"DA-L1 {acc_da[s]*100:.1f} RESOLVER {acc_res[s]*100:.1f}")


if __name__ == "__main__":
    main()
