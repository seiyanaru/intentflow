"""Decisive FIRST test for the illiteracy-rescue engine: does a deep model's feature
MANIFOLD have recoverable structure on illiterate subjects, and is it label-free reachable?
One EEGNet run/subject gives 4 numbers on session_E:
  (1) softmax acc            -- current deep model
  (2) LDA-probe CV (TRUE y)  -- linear ceiling on deep features
  (3) kNN CV   (TRUE y)      -- MANIFOLD ceiling (upper bound for transductive smoothing)
  (4) transductive kNN (label-free, seeded by deep-confident pseudo-labels) -- realizable
Verdict: (3)>>(1) => headroom exists; (4)->(3) & >(1) => label-free engine works.
Proxy backbone (EEGNet) for phenomenon existence; port to TCFormer if positive.
Run with intentflow conda env, GPU.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import StratifiedKFold
from datamodules.bcic4_2a import BCICIV2a
from braindecode.models import EEGNetv4
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    return X,np.array([int(ds[i][1]) for i in range(len(ds))])
def train(Xtr,ytr,C,T,epochs=300):
    net=EEGNetv4(C,4,n_times=T,final_conv_length='auto').to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev);n=len(yt)
    for ep in range(epochs):
        perm=torch.randperm(n,device=dev)
        for i in range(0,n,48):
            idx=perm[i:i+48];out=net(Xt[idx]);loss=nn.functional.cross_entropy(out,yt[idx])
            opt.zero_grad();loss.backward();opt.step()
    net.eval();return net
def softmax_np(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

print(f"device={dev}")
print(f"{'subj':>4} {'softmax':>7} {'LDAprobe':>8} {'kNN_ceil':>8} {'transduct':>9} {'manifold_headroom':>17}")
rows=[]
for sid in range(1,10):
    dm=BCICIV2a(prep,sid);dm.setup()
    Xtr,ytr=getxy(dm.train_dataset);Xte,yte=getxy(dm.test_dataset)
    C,Tn=Xtr.shape[1],Xtr.shape[2]
    net=train(Xtr,ytr,C,Tn)
    feats={}
    # hook input to last Conv2d (the classifier) -> penultimate feature map
    convs=[m for m in net.modules() if isinstance(m,nn.Conv2d)]
    h=convs[-1].register_forward_pre_hook(lambda mod,inp:feats.__setitem__('f',inp[0].detach()))
    with torch.no_grad():
        logit=net(torch.tensor(Xte).to(dev)); F=feats['f'].reshape(len(yte),-1).cpu().numpy()
    h.remove()
    p=softmax_np(logit.cpu().numpy()); soft=(p.argmax(1)==yte).mean()*100
    Fz=(F-F.mean(0))/(F.std(0)+1e-8)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    lda=[];knn=[]
    for tr,va in skf.split(Fz,yte):
        lda.append((LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr").fit(Fz[tr],yte[tr]).predict(Fz[va])==yte[va]).mean()*100)
        knn.append((KNeighborsClassifier(10).fit(Fz[tr],yte[tr]).predict(Fz[va])==yte[va]).mean()*100)
    ldaA=np.mean(lda);knnA=np.mean(knn)
    # label-free transductive: seed with deep-confident pseudo-labels (top 50% confidence)
    conf=p.max(1);thr=np.median(conf);seed=np.where(conf>=thr,p.argmax(1),-1)
    ls=LabelSpreading(kernel="knn",n_neighbors=10,max_iter=50).fit(Fz,seed)
    trans=(ls.transduction_==yte).mean()*100
    rows.append((sid,soft,ldaA,knnA,trans))
    print(f"S{sid:>3} {soft:>7.1f} {ldaA:>8.1f} {knnA:>8.1f} {trans:>9.1f} {knnA-soft:>+17.1f}",flush=True)
m=lambda j:np.mean([r[j] for r in rows])
print(f"\nMEAN softmax {m(1):.1f} | LDAprobe {m(2):.1f} | kNN_ceil {m(3):.1f} | transduct {m(4):.1f}")
ill=[r for r in rows if r[1]<82]  # illiterate-ish (low softmax)
if ill:
    g=lambda j:np.mean([r[j] for r in ill])
    print(f"ILLITERATE subset (softmax<82): softmax {g(1):.1f} | kNN_ceil {g(3):.1f} (headroom {g(3)-g(1):+.1f}) | transduct {g(4):.1f} ({g(4)-g(1):+.1f})")
print("\nVERDICT: kNN_ceil>>softmax => manifold headroom EXISTS. transduct>softmax (label-free) => engine WORKS. non-regression = no subject's transduct < softmax.")
