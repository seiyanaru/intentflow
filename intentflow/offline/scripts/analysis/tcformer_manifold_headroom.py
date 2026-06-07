"""Deep-manifold headroom test on the REAL backbone (TCFormer), using cached
EA-aware TCFormer features (288,64) + logits on session_E. No retraining.
Same 4 numbers as the EEGNet proxy:
  (1) softmax acc ; (2) LDA-probe CV (TRUE y, linear ceiling) ;
  (3) kNN CV (TRUE y, MANIFOLD ceiling) ; (4) transductive kNN (label-free).
Verdict: (3)>>(1) manifold headroom exists; (4)>(1) label-free engine works;
non-regression = no subject's transduct < softmax.
"""
import glob, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import StratifiedKFold
RR="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/"
def softmax(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
print(f"{'subj':>4} {'softmax':>7} {'LDAprobe':>8} {'kNN_ceil':>8} {'transduct':>9} {'manifold_hr':>11}")
rows=[]
for sid in range(1,10):
    fd=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz"))
    ld=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy"))
    if not fd or not ld: print(f"S{sid}: missing"); continue
    d=np.load(fd[-1]);F=d["features"].reshape(d["features"].shape[0],-1).astype(np.float64);y=d["labels"].astype(int)
    logit=np.load(ld[-1]);p=softmax(logit);soft=(p.argmax(1)==y).mean()*100
    Fz=(F-F.mean(0))/(F.std(0)+1e-8)
    skf=StratifiedKFold(5,shuffle=True,random_state=0);lda=[];knn=[]
    for tr,va in skf.split(Fz,y):
        lda.append((LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr").fit(Fz[tr],y[tr]).predict(Fz[va])==y[va]).mean()*100)
        knn.append((KNeighborsClassifier(10).fit(Fz[tr],y[tr]).predict(Fz[va])==y[va]).mean()*100)
    ldaA=np.mean(lda);knnA=np.mean(knn)
    conf=p.max(1);seed=np.where(conf>=np.median(conf),p.argmax(1),-1)
    trans=(LabelSpreading(kernel="knn",n_neighbors=10,max_iter=50).fit(Fz,seed).transduction_==y).mean()*100
    rows.append((sid,soft,ldaA,knnA,trans))
    print(f"S{sid:>3} {soft:>7.1f} {ldaA:>8.1f} {knnA:>8.1f} {trans:>9.1f} {knnA-soft:>+11.1f}")
m=lambda j:np.mean([r[j] for r in rows])
print(f"\nMEAN softmax {m(1):.1f} | LDAprobe {m(2):.1f} | kNN_ceil {m(3):.1f} | transduct {m(4):.1f}")
ill=[r for r in rows if r[1]<m(1)]
if ill:
    g=lambda j:np.mean([r[j] for r in ill])
    print(f"BELOW-MEAN(weak) subset: softmax {g(1):.1f} | kNN_ceil {g(3):.1f} ({g(3)-g(1):+.1f}) | transduct {g(4):.1f} ({g(4)-g(1):+.1f})")
reg=[(r[0],r[4]-r[1]) for r in rows if r[4]<r[1]]
print(f"non-regression: {'ALL OK' if not reg else 'HARMED '+str(reg)}")
print("\nNOTE: features = EA-aware TCFormer penultimate (64-d), the real deployed backbone.")
