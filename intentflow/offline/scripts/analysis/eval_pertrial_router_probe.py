"""Realizability probe: can a LABEL-FREE per-trial router capture the per-trial
whitener-oracle headroom (+6.5pp over per-subject oracle, seed-stable)?

Per trial we have 3 whiteners {source, full_ea, shrink_0.1} (frozen backbone,
EA-aware checkpoints). For each whitener we train a classifier P(prediction
correct | label-free per-trial features) on seed0 trials, then on held-out
seed1 route each trial to argmax predicted-correctness. Labels are used only as
TRAIN targets; all features are label-free and available at test time.

Decision gate: does routed accuracy beat (a) source, (b) per-subject oracle,
(c) the fixed 0.65/0.25/0.10 blend, with low per-subject harm, on HELD-OUT
seed1? If it captures ~0 of the gap, the per-trial direction is dead too.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NPZ = {
    "seed0": "intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",
    "seed1": "intentflow/offline/results/research_outputs/260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz",
}
WANT = ["source", "full_ea", "shrink_0.1"]
EPS = 1e-12


def load(seed):
    d = np.load(NPZ[seed], allow_pickle=True)
    experts = [str(x) for x in d["experts"].tolist()]
    eidx = [experts.index(n) for n in WANT]
    probs = d["probs"][:, eidx]          # [S, 3, T, C]
    labels = d["labels"]                 # [S, T]
    subjects = d["subjects"].astype(int)
    return probs, labels, subjects


def per_trial_features(probs_s):
    """probs_s: [3, T, C] -> features [T, F], all label-free."""
    feats = []
    preds = probs_s.argmax(-1)           # [3, T]
    for w in range(3):
        p = probs_s[w]
        conf = p.max(-1)
        ent = -(p * np.log(np.clip(p, EPS, 1.0))).sum(-1)
        top2 = np.sort(p, -1)[:, -2:]
        marg = top2[:, 1] - top2[:, 0]
        feats += [conf, ent, marg]
    # cross-whitener agreement / divergence (label-free)
    feats.append((preds[0] == preds[1]).astype(float))
    feats.append((preds[0] == preds[2]).astype(float))
    feats.append((preds[1] == preds[2]).astype(float))
    consensus = probs_s.mean(0)
    feats.append(-(consensus * np.log(np.clip(consensus, EPS, 1.0))).sum(-1))
    # pairwise symmetric KL source vs each
    for w in (1, 2):
        a, b = probs_s[0], probs_s[w]
        kl = (a * (np.log(np.clip(a, EPS, 1)) - np.log(np.clip(b, EPS, 1)))).sum(-1) + (
            b * (np.log(np.clip(b, EPS, 1)) - np.log(np.clip(a, EPS, 1)))
        ).sum(-1)
        feats.append(kl)
    return np.stack(feats, 1)            # [T, F]


def build_dataset(seed):
    probs, labels, subjects = load(seed)
    X, Y_correct, sid_of, trial_probs, trial_labels = [], [], [], [], []
    for si, sid in enumerate(subjects):
        f = per_trial_features(probs[si])            # [T, F]
        preds = probs[si].argmax(-1)                 # [3, T]
        correct = (preds == labels[si][None, :]).T   # [T, 3]
        X.append(f); Y_correct.append(correct)
        sid_of.append(np.full(f.shape[0], sid))
        trial_probs.append(probs[si]); trial_labels.append(labels[si])
    return (np.concatenate(X), np.concatenate(Y_correct), np.concatenate(sid_of),
            probs, labels, subjects)


def fit_predict(Xtr, ytr, Xte):
    """Train 3 correctness classifiers, return P(correct) [Nte,3]. sklearn if
    available, else standardized numpy logistic regression."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        P = np.zeros((Xte.shape[0], 3))
        for w in range(3):
            if ytr[:, w].min() == ytr[:, w].max():
                P[:, w] = float(ytr[:, w].mean())
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(Xtr, ytr[:, w])
            P[:, w] = clf.predict_proba(Xte)[:, 1]
        return P
    except Exception:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xa = (Xtr - mu) / sd; Xb = (Xte - mu) / sd
        Xa = np.c_[Xa, np.ones(len(Xa))]; Xb = np.c_[Xb, np.ones(len(Xb))]
        P = np.zeros((Xb.shape[0], 3))
        for w in range(3):
            y = ytr[:, w].astype(float)
            wts = np.zeros(Xa.shape[1])
            for _ in range(300):
                z = Xa @ wts; p = 1 / (1 + np.exp(-z))
                wts -= 0.1 * (Xa.T @ (p - y) / len(y) + 1e-3 * wts)
            P[:, w] = 1 / (1 + np.exp(-(Xb @ wts)))
        return P


def route_eval(Pcorrect, probs, labels, subjects, tag):
    """Pcorrect [N,3] over concatenated test trials in subject order."""
    off = 0
    rows = []
    src_all = full_all = routed_all = oracle_sub_all = 0.0
    n = 0
    for si, sid in enumerate(subjects):
        T = labels[si].shape[0]
        Pc = Pcorrect[off:off + T]; off += T
        preds = probs[si].argmax(-1)              # [3,T]
        route = Pc.argmax(1)                      # [T]
        routed_pred = preds[route, np.arange(T)]
        routed_acc = (routed_pred == labels[si]).mean() * 100
        accs = [(preds[w] == labels[si]).mean() * 100 for w in range(3)]
        src = accs[0]; subj_oracle = max(accs)
        rows.append(dict(subject=int(sid), routed=routed_acc, source=src,
                         full=accs[1], shrink=accs[2], subj_oracle=subj_oracle,
                         d_src=routed_acc - src))
        n += 1
    m = lambda k: float(np.mean([r[k] for r in rows]))
    harmed = [(r["subject"], round(r["d_src"], 2)) for r in rows if r["d_src"] < -1e-9]
    print(f"\n===== {tag} =====")
    print(f"{'S':>2} {'routed':>7} {'src':>6} {'subjOr':>7} {'dSrc':>6}")
    for r in rows:
        print(f"{r['subject']:>2} {r['routed']:7.2f} {r['source']:6.2f} {r['subj_oracle']:7.2f} {r['d_src']:+6.2f}")
    print(f"MEAN routed={m('routed'):.2f} source={m('source'):.2f} per-subj-oracle={m('subj_oracle'):.2f}")
    print(f"  routed-source: {m('routed')-m('source'):+.2f}pp | routed-subjOracle: {m('routed')-m('subj_oracle'):+.2f}pp")
    print(f"  HARMED: {len(harmed)}/{len(rows)} {harmed}")
    return dict(tag=tag, routed=m("routed"), source=m("source"),
                subj_oracle=m("subj_oracle"), harmed=len(harmed), rows=rows)


def main():
    out = {}
    # --- primary: train seed0 -> test seed1 (held-out seed) ---
    Xtr, ytr, _, _, _, _ = build_dataset("seed0")
    Xte, _, _, probs1, labels1, subj1 = build_dataset("seed1")
    P1 = fit_predict(Xtr, ytr, Xte)
    out["train_seed0_test_seed1"] = route_eval(P1, probs1, labels1, subj1,
                                               "TRAIN seed0 -> TEST seed1 (held-out seed)")

    # --- naive max-confidence routing (no training), seed1 ---
    off = 0; Pconf = np.zeros((Xte.shape[0], 3))
    probsX, labelsX, subjX = load("seed1")
    for si in range(len(subjX)):
        T = labelsX[si].shape[0]
        Pconf[off:off + T] = probsX[si].max(-1).T  # confidence per whitener
        off += T
    out["naive_maxconf_seed1"] = route_eval(Pconf, probs1, labels1, subj1,
                                            "NAIVE max-confidence routing (no training), seed1")

    # --- LOSO across subjects within seed0 (second generalization estimate) ---
    Xs0, ys0, sid0, probs0, labels0, subj0 = build_dataset("seed0")
    Ploso = np.zeros((Xs0.shape[0], 3))
    for sid in subj0:
        tr = sid0 != sid; te = sid0 == sid
        Ploso[te] = fit_predict(Xs0[tr], ys0[tr], Xs0[te])
    out["loso_seed0"] = route_eval(Ploso, probs0, labels0, subj0,
                                   "LOSO across subjects (seed0)")

    Path("intentflow/offline/results/research_outputs/260602_pertrial_router_probe.json").write_text(json.dumps(out, indent=2))
    print("\nwrote intentflow/offline/results/research_outputs/260602_pertrial_router_probe.json")


if __name__ == "__main__":
    main()
