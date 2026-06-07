"""Soft drift-aware D1 gating on top of the blend (LOSO + cross-seed).

DA-L1 baseline: final = blend + 0.3 * D1   (fixed weight) -> 87.15
Candidate:      final = blend + w_trial * D1
  where w_trial in [0, w_max] is a per-trial, label-free, drift-aware weight produced by a
  small cross-subject resolver. The resolver predicts P(D1 helps) from label-free features
  (T/D1 confidence, sym-KL, agreement, DRIFT from source-train centroid) and maps it to a
  D1 weight. Trained LOSO and evaluated cross-seed (seed0 build -> seed1 eval).

Key question: does drift-adaptive D1 weighting beat the fixed-0.3 DA-L1 on held-out?
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
    p = np.clip(p, EPS, 1); return -(p * np.log(p)).sum(-1)

def margin(p):
    s = np.sort(p, -1); return s[..., -1] - s[..., -2]

def drift_feats(eval_f, train_f):
    mu = train_f.mean(0); sd = train_f.std(0) + EPS
    z = (eval_f - mu) / sd
    euclid = np.linalg.norm(eval_f - mu, axis=1)
    mahal = np.linalg.norm(z, axis=1)
    mu_n = mu / (np.linalg.norm(mu) + EPS)
    ef_n = eval_f / (np.linalg.norm(eval_f, 1, keepdims=True) + EPS) if False else \
        eval_f / (np.linalg.norm(eval_f, axis=1, keepdims=True) + EPS)
    cos = 1.0 - ef_n @ mu_n
    return np.stack([euclid, mahal, cos], 1)

def trial_feats(T, D1, eval_f, train_f, use_drift=True):
    base = np.column_stack([
        T.max(-1), margin(T), entropy(T),
        D1.max(-1), margin(D1),
        0.5 * (kl(T, D1) + kl(D1, T)),
        T.max(-1) - D1.max(-1),
        (T.argmax(-1) == D1.argmax(-1)).astype(float),
    ])
    if use_drift:
        base = np.column_stack([base, drift_feats(eval_f, train_f)])
    return base

def d1_helps_target(blend, D1, y, w=0.3):
    """1 if adding D1 (at the eval weight) flips blend from wrong->right or keeps right;
    train target = does soft D1 addition help this trial."""
    base_pred = blend.argmax(-1)
    add_pred = (blend + w * D1).argmax(-1)
    # helps: add correct and base wrong; hurts: add wrong and base correct
    helps = ((add_pred == y) & (base_pred != y)).astype(int)
    hurts = ((add_pred != y) & (base_pred == y)).astype(int)
    # 1 => want D1 weight; 0 => suppress. neutral trials labeled by whether D1 argmax==y
    tgt = np.where(helps == 1, 1, np.where(hurts == 1, 0, (D1.argmax(-1) == y).astype(int)))
    return tgt

def acc_ps(pred, labels):
    return np.array([(pred[s] == labels[s]).mean() for s in range(9)])


def load(seed):
    port = np.load(PORT1 if seed == 1 else PORT0, allow_pickle=True)
    labels = port["labels"].astype(int)
    T = port["probs"][:, 0].astype(np.float64)
    blend = (0.3*port["probs"][:,0] + 0.4*port["probs"][:,1] + 0.3*port["probs"][:,2]).astype(np.float64)
    return labels, T, blend, port


def load_d1(seed, ref_labels):
    # D1 only saved for seed0. For cross-seed eval we need D1 on seed1 too;
    # if unavailable, reuse seed0 D1 (classical Riemann is seed-agnostic given same data).
    d1 = np.load(D1_0)["probs"].astype(np.float64)
    return d1


def get_feats(seed):
    feat = np.load(FEAT0)  # seed0 source features (train+eval). Used for drift.
    train_feat = {s: feat[f"train_feat_{s+1}"] for s in range(9)}
    eval_feat = {s: feat[f"eval_feat_{s+1}"] for s in range(9)}
    return train_feat, eval_feat


def soft_gate_loso(labels, T, D1, blend, train_feat, eval_feat,
                   w_max=0.6, C=0.1, use_drift=True, fixed_w=0.3):
    feats = {s: trial_feats(T[s], D1[s], eval_feat[s], train_feat[s], use_drift) for s in range(9)}
    tgts = {s: d1_helps_target(blend[s], D1[s], labels[s], fixed_w) for s in range(9)}
    preds = {}
    for s in range(9):
        Xtr = np.concatenate([feats[o] for o in range(9) if o != s])
        ytr = np.concatenate([tgts[o] for o in range(9) if o != s])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(feats[s]))[:, 1]
        w = w_max * p[:, None]
        preds[s] = (blend[s] + w * D1[s]).argmax(-1)
    return preds


def main():
    labels, T, blend, port = load(0)
    D1 = load_d1(0, labels)
    train_feat, eval_feat = get_feats(0)

    acc_src = acc_ps({s: T[s].argmax(-1) for s in range(9)}, labels)
    acc_blend = acc_ps({s: blend[s].argmax(-1) for s in range(9)}, labels)
    da = blend + 0.3 * D1
    acc_da = acc_ps({s: da[s].argmax(-1) for s in range(9)}, labels)
    print("=== seed0 in-sample baselines ===")
    print(f"source {acc_src.mean()*100:.2f}  blend {acc_blend.mean()*100:.2f}  DA-L1 {acc_da.mean()*100:.2f}")
    print()
    print("=== Soft drift-aware D1 gate (LOSO, seed0) ===")
    for use_drift in [False, True]:
        for w_max in [0.3, 0.45, 0.6, 0.9]:
            for C in [0.1]:
                pr = soft_gate_loso(labels, T, D1, blend, train_feat, eval_feat,
                                    w_max=w_max, C=C, use_drift=use_drift)
                a = acc_ps(pr, labels)
                tag = "drift " if use_drift else "nodrift"
                print(f"  {tag} w_max={w_max} C={C}: {a.mean()*100:.2f} "
                      f"vs DA-L1 {(a.mean()-acc_da.mean())*100:+.2f}")


if __name__ == "__main__":
    main()
