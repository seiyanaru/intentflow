"""DECISION EXPERIMENT: Anisotropic trust-shrinkage EA vs scalar-alpha on Stieger2021.

Hypothesis (from anisotropic-trust-shrinkage-idea.md): EA's harm is DIRECTIONAL.
EA recenters the decoder along ~60 covariance-shift eigen-directions; harm comes from a few
mis-estimated directions. Scalar-alpha shrinks ALL directions uniformly, so cutting worst-case
also halves the gain. If instead we shrink ONLY the untrustworthy directions toward source and
keep the trustworthy gain directions at full EA, we may break the scalar trade-off.

Method (leak-free, label-FREE adapter; identical to probs_dump's frozen Riemann-tangent-LDA):
  - source: frozen session-1 Riemann-tangent-LDA. P1 = R1^{-1/2}, shrinkage='auto'.
  - EA recenter: whiten session-j covariances by Pj = Rj^{-1/2}  (full EA = a_i==1 for all i).
  - generalized eigendecomp of the WHITENING TRANSPORT:
        M = P1 @ Rj @ P1 = U diag(mu) U^T     (mu_i==1 => no shift along direction i)
    anisotropic reference  R(a) = R1^{1/2} U diag(mu^{a}) U^T R1^{1/2},  per-direction a_i in [0,1]
        a_i==1 -> R(a)=Rj (full EA);  a_i==0 -> R(a)=R1 (source).   whiten by R(a)^{-1/2}.
  - LABEL-FREE trust:
      trust-M (magnitude):  a_i = exp(-beta * |log mu_i|)   (large-shift dirs -> source; no bootstrap)
      trust-S (stability):  bootstrap trials -> CV of mu_i; unstable dirs -> source via a_i=exp(-gamma*CV)
  - EVALUATION: every condition (source / EA / scalar-alpha{.3,.5,.7} / aniso-M{beta} / aniso-S)
    is scored on the SAME held-out trials [K:] of each session, so the only thing that varies is
    the whitening reference. true Delta = (acc_cond[K:] - acc_source[K:]) * 100.
  - The adapter uses NO labels (recenter + trust from covariances only), so using all trials to
    build references is not a label leak; held-out [K:] keeps the acc comparison apples-to-apples
    with our prior SPREAD/probs results (K=32).

WIN: aniso harmed << EA(87) AND meanDelta >> scalar-alpha(+4.09). IDEAL Pareto: meanDelta > EA(+7.85)
AND worst > -4. REJECT: aniso ~ scalar-alpha at matched harmed -> gain dirs == harm dirs ->
the directional-granularity hypothesis is falsified.

Per-subject checkpoint + resume (rows appended to a .jsonl; rerun skips done subjects). CPU, minutes.
"""
import os, warnings, json, glob, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
CACHE = "/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
CKPT = f"{RES}/260612_stieger_aniso_rows.jsonl"   # per-(subj,session) rows, resume-safe
OUT = f"{RES}/260612_stieger_aniso.json"
K = 32                       # held-out eval = trials [K:], matching probs_dump/SPREAD
BETAS = [0.5, 1.0, 2.0]      # aniso-M sharpness: a_i = exp(-beta*|log mu_i|)
GAMMAS = [1.0, 2.0]          # aniso-S sharpness: a_i = exp(-gamma*CV_i)
ALPHAS = [0.3, 0.5, 0.7]     # scalar reference: R(alpha)=R1^{1/2} M^{alpha} R1^{1/2}, all dirs equal
N_BOOT = 20                  # bootstrap resamples for trust-S CV (cheap)
BOOT_SEED = 0

def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M): w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def sqrtm(M):    w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w** 0.5))@V.T
def logm_spd(M): w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T

def tangent_set(C, P):
    d=C.shape[-1]; iu=np.triu_indices(d)
    sc=np.sqrt(2)*np.ones((d,d)); sc[np.diag_indices(d)]=1; sc=sc[iu]
    return np.array([(logm_spd(P@C[i]@P))[iu]*sc for i in range(len(C))])

def aniso_whitener(R1, R1h, R1ih, U, mu, a):
    """P(a) = R(a)^{-1/2} where R(a) = R1^{1/2} U diag(mu^a) U^T R1^{1/2}, a is per-direction in [0,1]."""
    Ra = R1h @ (U * (mu**a)) @ U.T @ R1h
    Ra = 0.5*(Ra+Ra.T)
    return invsqrtm(Ra)

def acc(lda, C, P, y):
    return float((lda.predict_proba(tangent_set(C, P)).argmax(1) == y).mean())

def done_subjects():
    s=set()
    if os.path.exists(CKPT):
        for ln in open(CKPT):
            try: s.add(json.loads(ln)["sub"])
            except Exception: pass
    return s

files = sorted(glob.glob(f"{CACHE}/S*_epochs.npz"),
               key=lambda f:int(os.path.basename(f).split("_")[0][1:]))
DONE = done_subjects()
print(f"{len(files)} subjects; {len(DONE)} already in checkpoint -> resume", flush=True)
rng = np.random.RandomState(BOOT_SEED)

# sanity-check holders (verify a==1 reproduces EA and a==0 reproduces source, numerically)
sanity = {"max_abs_dev_a1_vs_EA": 0.0, "max_abs_dev_a0_vs_source": 0.0}

fout = open(CKPT, "a")
for f in files:
    sub = int(os.path.basename(f).split("_")[0][1:])
    if sub in DONE:
        continue
    try:
        z=np.load(f); X=z["X"].astype(np.float64); y=z["y"].astype(int); sess=z["sess"]
        order=sorted(set(sess.tolist())); s1=order[0]; m1=sess==s1
        if len(np.unique(y[m1]))<2:
            print(f"S{sub}: s1 single-class skip", flush=True); fout.flush(); continue
        C1=cov(X[m1]); R1=C1.mean(0); P1=invsqrtm(R1); R1h=sqrtm(R1); R1ih=P1
        lda=LDA(solver="lsqr", shrinkage="auto").fit(tangent_set(C1, P1), y[m1])
        nc=0
        for sj in order[1:]:
            mj=sess==sj; yj=y[mj]
            if len(yj)<K+10 or len(np.unique(yj))<2: continue
            Xj=X[mj]; Cj=cov(Xj); Rj=Cj.mean(0); Pj=invsqrtm(Rj)
            # transport eigendecomp: M = P1 Rj P1
            M = P1 @ Rj @ P1; M = 0.5*(M+M.T)
            mu, U = eigh(M); mu=np.clip(mu, 1e-10, None)
            logmu = np.log(mu)

            ye = yj[K:]                                       # held-out labels (eval only)
            CjE = Cj[K:]                                      # held-out covariances
            base = acc(lda, CjE, P1, ye)                      # SOURCE on held-out
            ea   = acc(lda, CjE, Pj, ye)                      # FULL EA on held-out

            # --- sanity: a==1 must equal EA, a==0 must equal source ---
            P_a1 = aniso_whitener(R1, R1h, R1ih, U, mu, np.ones_like(mu))
            P_a0 = aniso_whitener(R1, R1h, R1ih, U, mu, np.zeros_like(mu))
            sanity["max_abs_dev_a1_vs_EA"]     = max(sanity["max_abs_dev_a1_vs_EA"],
                                                     float(np.abs(acc(lda,CjE,P_a1,ye)-ea)))
            sanity["max_abs_dev_a0_vs_source"] = max(sanity["max_abs_dev_a0_vs_source"],
                                                     float(np.abs(acc(lda,CjE,P_a0,ye)-base)))

            row=dict(sub=int(sub), sj=int(sj), n_eval=int(len(ye)),
                     base=base, ea=ea,
                     shift_l2=float(np.sqrt((logmu**2).sum())),       # total transport magnitude
                     mu_min=float(mu.min()), mu_max=float(mu.max()))

            # scalar-alpha (all directions equal): a_i = alpha
            for al in ALPHAS:
                P=aniso_whitener(R1,R1h,R1ih,U,mu, np.full_like(mu,al))
                row[f"scalar_{al}"]=acc(lda,CjE,P,ye)

            # aniso-M (magnitude trust): a_i = exp(-beta*|log mu_i|)
            for b in BETAS:
                a=np.exp(-b*np.abs(logmu))
                P=aniso_whitener(R1,R1h,R1ih,U,mu,a)
                row[f"anisoM_{b}"]=acc(lda,CjE,P,ye)
                row[f"amean_M_{b}"]=float(a.mean())

            # aniso-S (stability trust): bootstrap trials -> CV of mu_i -> a_i=exp(-gamma*CV_i)
            nj=len(Xj); mus_boot=np.empty((N_BOOT, len(mu)))
            for bi in range(N_BOOT):
                idx=rng.randint(0, nj, nj)
                Rjb=cov(Xj[idx]).mean(0)
                Mb=P1@Rjb@P1; Mb=0.5*(Mb+Mb.T)
                wb=np.clip(eigh(Mb, eigvals_only=True),1e-10,None)
                mus_boot[bi]=np.sort(wb)
            mu_sorted_order=np.argsort(mu)                    # align eigvals by sorted magnitude
            cv=mus_boot.std(0)/(np.abs(mus_boot.mean(0))+1e-9)
            cv_aligned=np.empty_like(cv); cv_aligned[mu_sorted_order]=cv
            for g in GAMMAS:
                a=np.exp(-g*cv_aligned)
                P=aniso_whitener(R1,R1h,R1ih,U,mu,a)
                row[f"anisoS_{g}"]=acc(lda,CjE,P,ye)
                row[f"amean_S_{g}"]=float(a.mean())

            fout.write(json.dumps(row)+"\n"); nc+=1
        fout.flush()
        print(f"S{sub}: {nc} sessions  (sanity dev a1/EA={sanity['max_abs_dev_a1_vs_EA']:.2e} "
              f"a0/src={sanity['max_abs_dev_a0_vs_source']:.2e})", flush=True)
        del X,y,z
    except Exception as e:
        print(f"S{sub}: FAIL {str(e)[:110]}", flush=True); fout.flush()
fout.close()

# ---------- aggregate ----------
rows=[json.loads(ln) for ln in open(CKPT)]
n=len(rows)
base=np.array([r["base"] for r in rows])
def summ(key):
    eff=np.array([r[key] for r in rows])
    d=(eff-base)*100
    harmed=int((d<-1).sum())
    return dict(meanD=float(d.mean()), worst=float(d.min()), harmed=harmed,
                helped=int((d>1).sum()),
                p10=float(np.percentile(d,10)), median=float(np.median(d)))

conds=["ea"]+[f"scalar_{a}" for a in ALPHAS]+[f"anisoM_{b}" for b in BETAS]+[f"anisoS_{g}" for g in GAMMAS]
oracle=((np.maximum(np.array([r["ea"] for r in rows]), base)-base)*100).mean()
report={"n":n, "K":K, "oracle_EAvsSrc":float(oracle), "sanity":sanity, "conditions":{}}
print(f"\n=== Anisotropic trust-shrinkage vs scalar-alpha  (n={n} sessions, held-out [{K}:]) ===")
print(f"{'condition':>14} {'meanD':>7} {'worst':>7} {'p10':>6} {'median':>7} {'harmed':>7} {'helped':>7} {'a_mean':>7}")
for c in conds:
    s=summ(c)
    am=""
    if c.startswith("anisoM"): am=f"{np.mean([r['amean_M_'+c.split('_')[1]] for r in rows]):.3f}"
    if c.startswith("anisoS"): am=f"{np.mean([r['amean_S_'+c.split('_')[1]] for r in rows]):.3f}"
    report["conditions"][c]=s
    print(f"{c:>14} {s['meanD']:>+7.2f} {s['worst']:>+7.2f} {s['p10']:>+6.2f} "
          f"{s['median']:>+7.2f} {s['harmed']:>7d} {s['helped']:>7d} {am:>7}")

json.dump(report, open(OUT,"w"), indent=2)
print(f"\nsanity: a==1 vs EA max|dev|={sanity['max_abs_dev_a1_vs_EA']:.2e}, "
      f"a==0 vs source max|dev|={sanity['max_abs_dev_a0_vs_source']:.2e}  (both should be ~0)")
print(f"saved {OUT}")
print("WIN: an aniso row has harmed << EA and meanD >> scalar-0.5(+4.09). "
      "IDEAL: meanD > EA(+7.85) AND worst > -4. REJECT: aniso ~ scalar at matched harmed.")
