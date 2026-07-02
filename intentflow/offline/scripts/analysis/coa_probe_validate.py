"""DECISIVE test of the CONVERGED design (my workflow + GPT deep-research both point here):
  (Q1) Does a label-free HEAD-TRANSPORT adapter (self-refit head) even help on Lee2019 n=54?
  (Q2) COUNTERFACTUAL-OVERRULE AUDIT (label-free): are BAD overrules concentrated on HIGH source
       margin, and does confident-overrule-mass H separate harmful sessions? (the adapter-specific gate)
  (Q3) k-SHOT PROBE-VETO: with k labeled trials/session, can we CERTIFY good vs bad confident flips,
       bound the negative-flip-rate, and recover gain while killing harm? (the real unlock)
Runs on the cached strong dump (no training). Deployed source decoder = LDA(F0,y0) (the E4 lda-mode,
reproduces +0.638). Adapter = self-refit shrink-LDA on (F1, source-pseudo-labels) = head transport.
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import spearmanr
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
NPZ = f"{RES}/260608_lee2019_pertrial_tcformer.npz"
rng = np.random.RandomState(0)

def fit_src(F0, y0):
    return LDA(solver="lsqr", shrinkage="auto").fit(F0, y0)

def margins(proba):  # top1-top2 margin per trial
    s = np.sort(proba, 1); return s[:, -1] - s[:, -2]

d = np.load(NPZ)
subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("F1_")})
rowsΔ = []; harm_feats = []; sess = []
DELTA = 1.0  # harm threshold (pp)
for s in subs:
    F0 = d[f"F0_{s}"].astype(np.float64); y0 = d[f"y0_{s}"]
    F1 = d[f"F1_{s}"].astype(np.float64); y1 = d[f"y1_{s}"]
    lda_s = fit_src(F0, y0)
    pred_s = lda_s.predict(F1); proba_s = lda_s.predict_proba(F1); m_s = margins(proba_s)
    yhat = pred_s
    # head-transport adapter = self-refit head on target pseudo-labels (needs >=2 classes present)
    if len(np.unique(yhat)) < 2:
        pred_a = pred_s
    else:
        pred_a = LDA(solver="lsqr", shrinkage="auto").fit(F1, yhat).predict(F1)
    acc_s = (pred_s == y1).mean() * 100; acc_a = (pred_a == y1).mean() * 100
    delta = acc_a - acc_s
    ov = pred_a != pred_s
    good = ov & (pred_a == y1) & (pred_s != y1)
    bad = ov & (pred_s == y1) & (pred_a != y1)
    # label-free adapter-specific gate features
    conf_s = proba_s.max(1)
    H = float((conf_s * ov).mean())                      # confident-overrule mass (high if flipping confident source)
    Hi = float((m_s[ov] >= np.median(m_s)).mean()) if ov.sum() else 0.0  # frac of flips on high-margin trials
    flip = float(ov.mean())
    disp = float(np.linalg.svd(proba_s, compute_uv=False).sum() / np.sqrt(len(F1) * proba_s.shape[1]))
    rowsΔ.append(delta)
    harm_feats.append(dict(H=H, Hi_frac=Hi, flip=flip, disp=disp,
                           mean_margin_good=float(m_s[good].mean()) if good.sum() else np.nan,
                           mean_margin_bad=float(m_s[bad].mean()) if bad.sum() else np.nan))
    sess.append(dict(s=s, acc_s=acc_s, acc_a=acc_a, delta=delta,
                     n_ov=int(ov.sum()), n_good=int(good.sum()), n_bad=int(bad.sum())))

Δ = np.array(rowsΔ)
print("="*78)
print(f"Q1: does label-free head-transport adapter help on Lee2019 n={len(subs)}?")
print(f"   mean Δacc = {Δ.mean():+.2f}pp  | helped {int((Δ>DELTA).sum())} | harmed {int((Δ<-DELTA).sum())} | "
      f"worst {Δ.min():+.1f} | best {Δ.max():+.1f}")
ng = sum(x['n_good'] for x in sess); nb = sum(x['n_bad'] for x in sess)
print(f"   total overrules: good(src wrong->fixed)={ng}  bad(src right->broken)={nb}")

# Q2: do BAD overrules sit on HIGHER source margin than GOOD? (the audit hypothesis)
mg = np.array([f["mean_margin_good"] for f in harm_feats]); mb = np.array([f["mean_margin_bad"] for f in harm_feats])
both = ~np.isnan(mg) & ~np.isnan(mb)
print("="*78)
print("Q2: counterfactual-overrule audit (label-free harm gate)")
print(f"   source-margin of BAD vs GOOD overrules (subj with both, n={both.sum()}): "
      f"bad={np.nanmean(mb[both]):.3f} vs good={np.nanmean(mg[both]):.3f}  "
      f"(audit hypothesis wants bad>good)")
# AUROC: does a label-free feature flag harmful sessions (Δ<-δ)?
harm = (Δ < -DELTA).astype(int)
def auroc(score, lab):
    if lab.sum()==0 or lab.sum()==len(lab): return float('nan')
    order=np.argsort(score); r=np.empty(len(score)); r[order]=np.arange(1,len(score)+1)
    pos=r[lab==1].sum(); n1=lab.sum(); n0=len(lab)-n1
    return (pos - n1*(n1+1)/2)/(n1*n0)
for key in ["H","Hi_frac","flip","disp"]:
    v=np.array([f[key] for f in harm_feats])
    print(f"   AUROC( {key:>8} -> harmful-session ) = {auroc(v,harm):.3f}   (Δ vs feature rho={spearmanr(v,Δ).correlation:+.2f})")
print(f"   [harmful sessions Δ<-{DELTA}: {int(harm.sum())}/{len(subs)}]")

# Q3: k-shot probe-veto — certify confident flips with k labels, bound NFR
print("="*78)
print("Q3: k-shot PROBE-VETO (certify good vs bad confident flips with k labels)")
print(f"   {'k':>4} {'certNFR':>8} {'rawNFR':>7} {'recovΔ':>8} {'vetoΔ':>7}  (certNFR<<rawNFR & vetoΔ>=0 => unlock works)")
for k in [4, 8, 16, 32]:
    cert_correct=[]; cert_wrong=[]; veto_deltas=[]; recov=[]
    for s in subs:
        F0=d[f"F0_{s}"].astype(np.float64); y0=d[f"y0_{s}"]; F1=d[f"F1_{s}"].astype(np.float64); y1=d[f"y1_{s}"]
        lda_s=fit_src(F0,y0); pred_s=lda_s.predict(F1)
        yhat=pred_s
        if len(np.unique(yhat))<2: continue
        pred_a=LDA(solver="lsqr",shrinkage="auto").fit(F1,yhat).predict(F1)
        ov=np.where(pred_a!=pred_s)[0]
        if len(ov)==0: continue
        # draw k labeled probe trials (stratified-ish), fit a probe head, certify a flip if probe agrees with adapter
        idx=rng.permutation(len(y1))[:k]
        if len(np.unique(y1[idx]))<2:  # need both classes in probe
            cont=True
            for _ in range(10):
                idx=rng.permutation(len(y1))[:k]
                if len(np.unique(y1[idx]))>=2: cont=False; break
            if cont: continue
        probe=LDA(solver="lsqr",shrinkage="auto").fit(F1[idx],y1[idx])
        probe_pred=probe.predict(F1)
        # certify flip: keep adapter's flip only if the labeled-probe head AGREES with the adapter (not source)
        certified = ov[probe_pred[ov]==pred_a[ov]]
        # NFR among certified flips (using held-out truth = all of y1, probe is small)
        for t in certified:
            (cert_correct if pred_a[t]==y1[t] else cert_wrong).append(1)
        # final preds: source everywhere, adopt adapter only on certified flips
        final=pred_s.copy(); final[certified]=pred_a[certified]
        veto_deltas.append(((final==y1).mean()-(pred_s==y1).mean())*100)
        # raw (no veto) recovered delta
        recov.append(((pred_a==y1).mean()-(pred_s==y1).mean())*100)
    cn=len(cert_correct)+len(cert_wrong)
    certNFR=len(cert_wrong)/cn if cn else float('nan')
    # raw NFR across all flips
    rb=sum(x['n_bad'] for x in sess); rg=sum(x['n_good'] for x in sess)
    rawNFR=rb/(rb+rg) if (rb+rg) else float('nan')
    print(f"   {k:>4} {certNFR:>8.3f} {rawNFR:>7.3f} {np.mean(recov):>+8.2f} {np.mean(veto_deltas):>+7.2f}")
print("="*78)
print("READ: Q1 tells if adaptation is even worth gating. Q2 tells if a label-free adapter-specific")
print("gate flags harm (AUROC>>0.5). Q3 tells if a k-shot veto certifies good flips (certNFR low) and")
print("keeps a non-negative safe delta. If Q2 fails but Q3 works => the small-probe unlock is the path.")
