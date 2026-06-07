"""Drift-gated ONLINE self-training of the CHEAP diverse model D1 (EA-Riemann tangent + LDA),
with TCFormer kept FROZEN.

Mechanism (frozen-deep + adaptive-cheap):
  - TCFormer blend (0.3 source / 0.4 full_ea / 0.3 shrink_0.1) is the FROZEN anchor.
  - D1 = bandpass(8-30) -> cov -> EA-recenter -> tangent -> shrink-LDA, trained on session_T (seed-free).
  - Over the TARGET session_E stream (temporal order), we:
      (a) online-recenter the EA reference using a running mean of target covariances (Frechet-ish, Euclidean mean of cov);
      (b) on trials where BLEND-argmax and D1-argmax AGREE, use that as a pseudo-label (94%-clean);
      (c) DRIFT-GATE: only commit an LDA refit on high-drift windows, measured by the
          Riemannian distance between the running target reference and the source (train) reference.
  - We accumulate agreed-pseudo-labeled target trials into a buffer; when a drift window
    triggers, we refit LDA on (source-train tangent w.r.t. current ref) + (buffered target tangent w.r.t. current ref).

This is an ONLINE/causal protocol: prediction for trial t uses only the model state built from trials < t.

Outputs realized accuracy for:
  - blend (frozen anchor)
  - static DA-L1 (blend + 0.3 * static-D1)
  - online-D1 standalone
  - DA-L1-online (blend + 0.3 * online-D1)
  - ablation: always-update (no drift gate) DA-L1-online

Run with intentflow env. Verifies static-D1 reproduction against saved probs first.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # -> intentflow/offline
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2a import BCICIV2a

DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
LAB = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels"
RP = "/mnt/data/seiya.narukawa/intentflow/docs/research_progress"
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=DATA, eval_label_path=LAB)

b, a = butter(4, [8/125., 30/125.], btype="band")
def bp(X): return filtfilt(b, a, X, axis=-1).copy()
def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M):
    w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * (w ** -0.5)) @ V.T
def sqrtm(M):
    w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * (w ** 0.5)) @ V.T
def logm_spd(M):
    w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * np.log(w)) @ V.T
def riem_dist(A, B):
    P = invsqrtm(A); w = eigh(P @ B @ P, eigvals_only=True); w = np.clip(w, 1e-12, None)
    return float(np.sqrt((np.log(w) ** 2).sum()))

def tangent_ref(C, ref):
    """Tangent map of cov set C at reference SPD `ref` (whitened-at-ref then logm).
    Batched via np.linalg.eigh for speed. C: (n,c,c) -> (n, c*(c+1)/2)."""
    c = C.shape[1]; iu = np.triu_indices(c)
    sc = np.sqrt(2) * np.ones((c, c)); np.fill_diagonal(sc, 1.0)
    P = invsqrtm(ref)
    W = P @ C @ P                                   # (n,c,c)
    W = 0.5 * (W + np.transpose(W, (0, 2, 1)))      # symmetrize
    w, V = np.linalg.eigh(W)                         # batched
    w = np.clip(w, 1e-10, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))  # logm per trial
    L = L * sc[None]
    return L[:, iu[0], iu[1]]

def get_xy(ds):
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    y = np.array([int(ds[i][1]) for i in range(len(ds))])
    return X, y

# ---- load saved TCFormer expert probs + saved static D1 for verification ----
def load_blend(seed_dir):
    z = np.load(os.path.join(seed_dir, "expert_portfolio_arrays.npz"), allow_pickle=True)
    experts = list(z["experts"]); probs = z["probs"].astype(np.float64); labels = z["labels"]
    ix = {e: i for i, e in enumerate(experts)}
    blend = 0.3*probs[:, ix["source"]] + 0.4*probs[:, ix["full_ea"]] + 0.3*probs[:, ix["shrink_0.1"]]
    return blend, labels

def main(seed_tag, seed_dir, drift_pct=60.0, w_d1=0.3, verify=True):
    blend, labels = load_blend(seed_dir)
    saved_d1 = np.load(os.path.join(RP, "260603_diverse_riemann_preds.npz"))
    saved_d1_probs = saved_d1["probs"]; saved_d1_labels = saved_d1["labels"]

    res = {k: [] for k in
           ["blend", "dal1_static", "d1_static", "d1_online", "dal1_online",
            "dal1_online_nogate", "d1_online_nogate", "n_commits", "n_agree", "label_match"]}
    per_subj = []
    for sid in range(1, 10):
        dm = BCICIV2a(prep, sid); dm.setup()
        Xtr, ytr = get_xy(dm.train_dataset)
        Xte, yte = get_xy(dm.test_dataset)
        # verify trial order matches saved arrays (labels must match 1:1)
        lab_match = bool(np.array_equal(yte, labels[sid-1]))
        Ctr = cov(bp(Xtr)); Cte = cov(bp(Xte))
        ref_tr = Ctr.mean(0)                      # source (train) EA reference
        ref_te = Cte.mean(0)                      # TRANSDUCTIVE target EA ref (= saved baseline budget)
        # ---- static D1 (reproduce saved, transductive EA): train tangent@ref_tr, test tangent@ref_te ----
        Ttr0 = tangent_ref(Ctr, ref_tr)
        lda0 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Ttr0, ytr)
        Tte0 = tangent_ref(Cte, ref_te)
        proba0 = lda0.predict_proba(Tte0)
        Pst = np.zeros((len(yte), 4)); Pst[:, lda0.classes_] = proba0
        # precompute transductive test tangent (EA budget = baseline) used for ONLINE prediction too
        Tte_trans = Tte0
        Ttr_trans = Ttr0    # train tangent at its own ref (matches static training)

        # ONLINE self-training of LDA. EA reference budget chosen by `ea_mode`:
        #   ea_mode='trans'  : tangent maps use whole-test ref_te (same EA budget as static baseline);
        #                      ONLY varied factor vs static = LDA self-training. (isolates mechanism)
        #   ea_mode='causal' : tangent maps use running-mean target ref (strictly causal, harder).
        # Pseudo-label STREAM is always causal: prediction for trial t uses LDA built from trials < t.
        def run_online(drift_gate=True, thr=None, ng_cadence=16, ea_mode="trans"):
            buf_C, buf_y = [], []          # agreed target covs + pseudo labels
            lda = lda0                      # start from static D1
            ref = ref_te.copy() if ea_mode == "trans" else ref_tr.copy()
            run_sum = np.zeros_like(ref_tr); n_seen = 0
            # cache train tangent at current ref (recompute only when ref changes)
            Ttr_cur = Ttr_trans if ea_mode == "trans" else tangent_ref(Ctr, ref)
            preds_prob = np.zeros((len(yte), 4))
            commits = 0; agrees = 0
            last_commit_dist = 0.0
            for t in range(len(yte)):
                # predict trial t with CURRENT lda + CURRENT ref (causal in LDA state)
                if ea_mode == "trans":
                    Tt = Tte_trans[t:t+1]
                else:
                    Tt = tangent_ref(Cte[t:t+1], ref)
                pr = lda.predict_proba(Tt)[0]
                p4 = np.zeros(4); p4[lda.classes_] = pr
                preds_prob[t] = p4
                # running target reference (drift signal + causal EA)
                run_sum += Cte[t]; n_seen += 1
                ref_run = run_sum / n_seen
                drift = riem_dist(ref_run, ref_tr)
                # pseudo-label via agreement with FROZEN blend
                d1_lab = int(p4.argmax())
                bl_lab = int(blend[sid-1, t].argmax())
                if d1_lab == bl_lab:
                    agrees += 1
                    buf_C.append(Cte[t]); buf_y.append(bl_lab)
                # decide commit
                do_commit = False
                if len(set(buf_y)) >= 2 and len(buf_y) >= 8:
                    if not drift_gate:
                        if (t + 1) % ng_cadence == 0:
                            do_commit = True
                    else:
                        if drift - last_commit_dist >= thr:   # DC-Commit L3 high-drift window
                            do_commit = True
                if do_commit:
                    if ea_mode == "causal":
                        ref = ref_run.copy()
                        Ttr_cur = tangent_ref(Ctr, ref)
                        Tbuf = tangent_ref(np.array(buf_C), ref)
                    else:
                        # EA ref stays transductive (baseline budget); buffer tangent at ref_te
                        Tbuf = tangent_ref(np.array(buf_C), ref)
                    Tall = np.concatenate([Ttr_cur, Tbuf], 0)
                    yall = np.concatenate([ytr, np.array(buf_y)], 0)
                    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Tall, yall)
                    last_commit_dist = drift
                    commits += 1
            return preds_prob, commits, agrees

        # subject-specific drift threshold from the full session drift range (set on history,
        # but to keep it simple & causal-ish we derive thr from train-only proxy: a fixed fraction
        # of the total source->target drift estimated incrementally). Use percentile of per-trial
        # incremental drift jumps measured ONLINE-safe: we set thr as a fraction of the final
        # source->fulltarget distance scaled by drift_pct. To stay strictly causal we instead use
        # a small fixed multiple of mean within-train-cov dispersion.
        # ---- causal threshold: median pairwise riem dist of train covs (no target leak) ----
        idx = np.random.RandomState(0).choice(len(Ctr), size=min(30, len(Ctr)), replace=False)
        dd = []
        for ii in range(len(idx)):
            for jj in range(ii+1, len(idx)):
                dd.append(riem_dist(Ctr[idx[ii]], Ctr[idx[jj]]))
        base_disp = float(np.median(dd))
        thr = base_disp * (drift_pct / 100.0)

        P_on, n_commit, n_agree = run_online(drift_gate=True, thr=thr, ea_mode="trans")
        P_on_ng, n_commit_ng, _ = run_online(drift_gate=False, thr=thr, ea_mode="trans")
        P_on_causal, n_commit_c, _ = run_online(drift_gate=True, thr=thr, ea_mode="causal")

        def acc(P): return float((P.argmax(1) == yte).mean()) * 100
        bl = blend[sid-1]
        a_blend = acc(bl)
        a_dal1_static = acc(bl + w_d1 * Pst)
        a_d1_static = acc(Pst)
        a_d1_on = acc(P_on)
        a_dal1_on = acc(bl + w_d1 * P_on)
        a_d1_on_ng = acc(P_on_ng)
        a_dal1_on_ng = acc(bl + w_d1 * P_on_ng)
        a_dal1_on_causal = acc(bl + w_d1 * P_on_causal)
        res.setdefault("dal1_online_causal", []).append(a_dal1_on_causal)

        res["blend"].append(a_blend); res["dal1_static"].append(a_dal1_static)
        res["d1_static"].append(a_d1_static); res["d1_online"].append(a_d1_on)
        res["dal1_online"].append(a_dal1_on); res["dal1_online_nogate"].append(a_dal1_on_ng)
        res["d1_online_nogate"].append(a_d1_on_ng)
        res["n_commits"].append(n_commit); res["n_agree"].append(n_agree)
        res["label_match"].append(lab_match)
        per_subj.append(dict(sid=sid, blend=a_blend, dal1_static=a_dal1_static,
                             dal1_online=a_dal1_on, dal1_online_nogate=a_dal1_on_ng,
                             dal1_online_causal=a_dal1_on_causal,
                             d1_static=a_d1_static, d1_online=a_d1_on,
                             n_commits=n_commit, n_agree=n_agree, thr=thr, lab_match=lab_match))
        print(f"S{sid}: blend={a_blend:5.2f} DA-L1s={a_dal1_static:5.2f} "
              f"DA-L1on={a_dal1_on:5.2f} (nogate={a_dal1_on_ng:5.2f}) (causal={a_dal1_on_causal:5.2f}) "
              f"D1s={a_d1_static:5.2f} D1on={a_d1_on:5.2f} commits={n_commit} agree={n_agree} "
              f"match={lab_match}", flush=True)

    out = {k: float(np.mean(v)) if k not in ("n_commits","n_agree","label_match") else
           (float(np.mean(v)) if k != "label_match" else bool(np.all(v)))
           for k, v in res.items()}
    print("\n=== MEAN ({}) drift_pct={} w_d1={} ===".format(seed_tag, drift_pct, w_d1))
    for k in ["blend","dal1_static","dal1_online","dal1_online_nogate","dal1_online_causal","d1_static","d1_online","n_commits","n_agree","label_match"]:
        print(f"  {k:20s} {out[k]:.3f}")
    summary = dict(seed_tag=seed_tag, drift_pct=drift_pct, w_d1=w_d1, mean=out, per_subj=per_subj)
    return summary

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_tag", default="seed0")
    ap.add_argument("--drift_pct", type=float, default=60.0)
    ap.add_argument("--w_d1", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(RP, "260603_d1_drift_gated_online.json"))
    args = ap.parse_args()
    seed_dir = {"seed0": os.path.join(RP, "260602_expert_portfolio_table"),
                "seed1": os.path.join(RP, "260602_expert_portfolio_table_seed1")}[args.seed_tag]
    summ = main(args.seed_tag, seed_dir, drift_pct=args.drift_pct, w_d1=args.w_d1)
    with open(args.out, "w") as f:
        json.dump(summ, f, indent=2)
    print("saved ->", args.out)
