"""Cross-seed (build seed0 -> eval seed1) honest test of the drift-aware D1 gate / resolver.

D1 (classical EA-Riemann) is computed on the raw EVAL session, identical across seeds,
so the same D1 array applies to both seed0 and seed1 (verified labels identical).
The deep T / blend differ by training seed.

Two candidates evaluated on held-out seed1:
 (A) HARD drift resolver: on T-vs-D1 disagreement, logistic picks T or D1.
 (B) SOFT drift gate: final = blend + w_trial*D1, w_trial from logistic.
Resolver/gate parameters and the logistic are FIT on seed0 only (cross-subject pooled),
then frozen and applied to seed1. Compared to blend(85.76) and DA-L1(87.15) on seed1.
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
D1F = f"{DOCS}/260603_diverse_riemann_preds.npz"
FEAT0 = f"{OFFLINE}/results/source_train_features_s0.npz"
EPS = 1e-8

kl = lambda p, q: (np.clip(p, EPS, 1) * np.log(np.clip(p, EPS, 1) / np.clip(q, EPS, 1))).sum(-1)
ent = lambda p: -(np.clip(p, EPS, 1) * np.log(np.clip(p, EPS, 1))).sum(-1)
def margin(p):
    s = np.sort(p, -1); return s[..., -1] - s[..., -2]

def drift_feats(eval_f, train_f):
    mu = train_f.mean(0); sd = train_f.std(0) + EPS
    z = (eval_f - mu) / sd
    euclid = np.linalg.norm(eval_f - mu, axis=1)
    mahal = np.linalg.norm(z, axis=1)
    mu_n = mu / (np.linalg.norm(mu) + EPS)
    ef_n = eval_f / (np.linalg.norm(eval_f, axis=1, keepdims=True) + EPS)
    return np.stack([euclid, mahal, 1.0 - ef_n @ mu_n], 1)

def trial_feats(T, D1, eval_f, train_f, use_drift=True):
    base = np.column_stack([
        T.max(-1), margin(T), ent(T), D1.max(-1), margin(D1),
        0.5*(kl(T, D1)+kl(D1, T)), T.max(-1)-D1.max(-1),
        (T.argmax(-1) == D1.argmax(-1)).astype(float)])
    if use_drift:
        base = np.column_stack([base, drift_feats(eval_f, train_f)])
    return base

def acc_ps(pred, labels):
    return np.array([(pred[s] == labels[s]).mean() for s in range(9)])

def load_port(seed):
    port = np.load(PORT1 if seed == 1 else PORT0, allow_pickle=True)
    labels = port["labels"].astype(int)
    T = port["probs"][:, 0].astype(np.float64)
    blend = (0.3*port["probs"][:,0]+0.4*port["probs"][:,1]+0.3*port["probs"][:,2]).astype(np.float64)
    return labels, T, blend

def gate_target(blend, D1, y, w=0.4):
    base = blend.argmax(-1); add = (blend + w*D1).argmax(-1)
    helps = (add == y) & (base != y); hurts = (add != y) & (base == y)
    return np.where(helps, 1, np.where(hurts, 0, (D1.argmax(-1) == y).astype(int)))

def resolver_target(T, D1, y):
    return ((D1.argmax(-1) == y) & (T.argmax(-1) != y)).astype(int)


def main():
    lab0, T0, bl0 = load_port(0)
    lab1, T1, bl1 = load_port(1)
    D1 = np.load(D1F)["probs"].astype(np.float64)  # shared across seeds
    feat = np.load(FEAT0)
    train_feat = {s: feat[f"train_feat_{s+1}"] for s in range(9)}
    eval_feat = {s: feat[f"eval_feat_{s+1}"] for s in range(9)}
    assert np.array_equal(lab0, lab1)

    da1 = bl1 + 0.3 * D1
    acc_src1 = acc_ps({s: T1[s].argmax(-1) for s in range(9)}, lab1)
    acc_bl1 = acc_ps({s: bl1[s].argmax(-1) for s in range(9)}, lab1)
    acc_da1 = acc_ps({s: da1[s].argmax(-1) for s in range(9)}, lab1)
    print("=== HELD-OUT seed1 baselines ===")
    print(f"source {acc_src1.mean()*100:.2f}  blend {acc_bl1.mean()*100:.2f}  DA-L1 {acc_da1.mean()*100:.2f}")
    print()

    # --- Candidate B: SOFT gate, fit on seed0 pooled (all 9 subj), apply to seed1 ---
    print("=== (B) SOFT drift gate: FIT seed0 (pooled) -> EVAL seed1 ===")
    for use_drift in [False, True]:
        feats0 = np.concatenate([trial_feats(T0[s], D1[s], eval_feat[s], train_feat[s], use_drift) for s in range(9)])
        tg0 = np.concatenate([gate_target(bl0[s], D1[s], lab0[s], 0.4) for s in range(9)])
        sc = StandardScaler().fit(feats0)
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(feats0), tg0)
        for w_max in [0.4, 0.6, 0.8]:
            preds = {}
            for s in range(9):
                f1 = trial_feats(T1[s], D1[s], eval_feat[s], train_feat[s], use_drift)
                p = clf.predict_proba(sc.transform(f1))[:, 1]
                preds[s] = (bl1[s] + (w_max * p[:, None]) * D1[s]).argmax(-1)
            a = acc_ps(preds, lab1)
            tag = "drift " if use_drift else "nodrift"
            print(f"  {tag} w_max={w_max}: seed1 {a.mean()*100:.2f}  vs DA-L1 {(a.mean()-acc_da1.mean())*100:+.2f}")
    print()

    # --- Candidate A: HARD resolver, fit on seed0 disagreement, apply seed1 ---
    print("=== (A) HARD drift resolver: FIT seed0 -> EVAL seed1 ===")
    for use_drift in [True]:
        dis0 = np.concatenate([(T0[s].argmax(-1) != D1[s].argmax(-1)) for s in range(9)])
        feats0 = np.concatenate([trial_feats(T0[s], D1[s], eval_feat[s], train_feat[s], use_drift) for s in range(9)])
        tg0 = np.concatenate([resolver_target(T0[s], D1[s], lab0[s]) for s in range(9)])
        sc = StandardScaler().fit(feats0[dis0])
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(feats0[dis0]), tg0[dis0])
        for thr in [0.5, 0.6, 0.7]:
            preds = {}
            for s in range(9):
                ds = T1[s].argmax(-1) != D1[s].argmax(-1)
                pred = T1[s].argmax(-1).copy()
                if ds.sum() > 0:
                    f1 = trial_feats(T1[s], D1[s], eval_feat[s], train_feat[s], use_drift)[ds]
                    pr = clf.predict_proba(sc.transform(f1))[:, 1]
                    idx = np.where(ds)[0]
                    pred[idx[pr >= thr]] = D1[s].argmax(-1)[idx[pr >= thr]]
                preds[s] = pred
            a = acc_ps(preds, lab1)
            print(f"  drift thr={thr}: seed1 {a.mean()*100:.2f}  vs DA-L1 {(a.mean()-acc_da1.mean())*100:+.2f}")

    # --- ablation: drift-feature-only resolver to isolate drift contribution ---
    print()
    print("=== drift-feature ablation: does drift add anything on seed1? (soft, w_max=0.6) ===")
    for keep in ["all", "no_drift", "only_drift_conf"]:
        def tf(T, D1, ef, trf):
            if keep == "all":
                return trial_feats(T, D1, ef, trf, True)
            if keep == "no_drift":
                return trial_feats(T, D1, ef, trf, False)
            # only drift + minimal conf
            return np.column_stack([T.max(-1), D1.max(-1), drift_feats(ef, trf)])
        feats0 = np.concatenate([tf(T0[s], D1[s], eval_feat[s], train_feat[s]) for s in range(9)])
        tg0 = np.concatenate([gate_target(bl0[s], D1[s], lab0[s], 0.4) for s in range(9)])
        sc = StandardScaler().fit(feats0)
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(sc.transform(feats0), tg0)
        preds = {}
        for s in range(9):
            f1 = tf(T1[s], D1[s], eval_feat[s], train_feat[s])
            p = clf.predict_proba(sc.transform(f1))[:, 1]
            preds[s] = (bl1[s] + (0.6 * p[:, None]) * D1[s]).argmax(-1)
        a = acc_ps(preds, lab1)
        print(f"  feats={keep}: seed1 {a.mean()*100:.2f}  vs DA-L1 {(a.mean()-acc_da1.mean())*100:+.2f}")


if __name__ == "__main__":
    main()
