"""DA-DC + label-free DRIFT-vs-JUNK diagnosis. Two verified axes:
 signal-quality = eval-feature clusterability (junk detector); drift = train->eval centroid shift.
Part1 (subject-level, robust): do the axes separate regimes? does clusterability predict DA-DC
 reliability (acc)? does drift predict DA-DC gain over source?
Part2 (trial-level, self-consistent on clean features): is abstain (label-free trial-quality)
 MORE valuable on junk subjects than good ones? (safety where the signal is junk)
Baseline discipline: source~82.7, DA-DC~87.2.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
RES="intentflow/offline/results/research_outputs/"
exp=np.load(RES+"260602_expert_portfolio_table/expert_portfolio_arrays.npz",allow_pickle=True)
IX={n:[str(x) for x in exp["experts"].tolist()].index(n) for n in ["source","full_ea","shrink_0.1"]}
Lab=exp["labels"];src=exp["probs"][:,IX["source"]];fe=exp["probs"][:,IX["full_ea"]];sh=exp["probs"][:,IX["shrink_0.1"]]
R=np.load(RES+"260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
acc=lambda P,i:(P[i].argmax(1)==Lab[i]).mean()*100
d=np.load("intentflow/offline/results/source_train_features_s0.npz",allow_pickle=True)

clus=[];drift=[];sacc=[];dacc=[];gain=[]
for s in range(1,10):
    Ftr=d[f"train_feat_{s}"];Fte=d[f"eval_feat_{s}"]
    Z=(Fte-Fte.mean(0))/(Fte.std(0)+1e-8)
    clus.append(silhouette_score(Z,KMeans(4,n_init=5,random_state=0).fit_predict(Z)))
    drift.append(np.linalg.norm(Ftr.mean(0)-Fte.mean(0)))
    so=acc(src,s-1); da=((0.3*src[s-1]+0.4*fe[s-1]+0.3*sh[s-1]+0.3*R[s-1]).argmax(1)==Lab[s-1]).mean()*100
    sacc.append(so);dacc.append(da);gain.append(da-so)
print("=== Part1: subject-level diagnosis (2a) ===")
print(f"{'S':>3} {'clus(junk↓)':>11} {'drift':>7} {'src':>6} {'DA-DC':>6} {'gain':>6}")
for i in range(9): print(f"S{i+1:>2} {clus[i]:>11.3f} {drift[i]:>7.2f} {sacc[i]:>6.1f} {dacc[i]:>6.1f} {gain[i]:>+6.1f}")
print(f"MEAN src={np.mean(sacc):.1f}(~82.7) DA-DC={np.mean(dacc):.1f}(~87.2)")
print(f"\ncorr(clusterability, DA-DC acc) = {spearmanr(clus,dacc).correlation:+.2f}   [junk detector predicts reliability]")
print(f"corr(drift, DA-DC gain)        = {spearmanr(drift,gain).correlation:+.2f}   [drift detector predicts adaptation benefit]")
# regime table (median split)
cm,dm=np.median(clus),np.median(drift)
import itertools
print("\nregime (clusterability x drift, median split): mean DA-DC acc / mean gain / n")
for hi_c in [False,True]:
    for hi_d in [False,True]:
        idx=[i for i in range(9) if (clus[i]>=cm)==hi_c and (drift[i]>=dm)==hi_d]
        if idx:
            lab=("good-sig" if hi_c else "JUNK")+("/drift" if hi_d else "/stable")
            print(f"  {lab:>14}: DA-DC={np.mean([dacc[i] for i in idx]):.1f}  gain={np.mean([gain[i] for i in idx]):+.1f}  n={len(idx)}")

print("\n=== Part2: is abstain MORE valuable on JUNK subjects? (self-consistent clean-feature head) ===")
def maha(Xtr,ytr,Xte):
    cls=np.unique(ytr);M=np.stack([Xtr[ytr==c].mean(0) for c in cls])
    res=np.concatenate([Xtr[ytr==c]-Xtr[ytr==c].mean(0) for c in cls],0)
    P=np.linalg.pinv(LedoitWolf().fit(res).covariance_)
    return np.min(np.stack([np.einsum('ij,jk,ik->i',Xte-M[c],P,Xte-M[c]) for c in range(len(cls))],1),1)
junk_gain=[];good_gain=[]
med_c=np.median(clus)
for s in range(1,10):
    Ftr=d[f"train_feat_{s}"];ytr=d[f"train_label_{s}"];Fte=d[f"eval_feat_{s}"];yte=d[f"eval_label_{s}"]
    lda=LDA(solver="lsqr",shrinkage="auto").fit(Ftr,ytr);wrong=(lda.predict(Fte)!=yte).astype(int)
    dmn=maha(Ftr,ytr,Fte);keep=dmn<=np.quantile(dmn,0.80)
    benefit=(wrong.mean()-wrong[keep].mean())*100  # error reduction from abstaining 20%
    (junk_gain if clus[s-1]<med_c else good_gain).append(benefit)
print(f"abstain@80% error-reduction: JUNK subjects {np.mean(junk_gain):+.1f}pp  vs  GOOD subjects {np.mean(good_gain):+.1f}pp")
print("\nVERDICT: idea holds if (a) clusterability predicts DA-DC reliability, (b) drift predicts gain,")
print("         (c) abstain helps MORE on junk subjects => DA-DC + drift/junk diagnosis = accuracy(DA-DC)+safety(junk-abstain).")
