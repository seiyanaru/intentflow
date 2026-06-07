"""CFCS make-or-break early signal. Question: can a deep model be TRAINED (drift-aug
+ selective-confidence / deep-gambler) so its OWN confidence flags confident-errors
under REAL drift, BEATING the train-free C2 cross-family disagreement gate (which
needs a 2nd model at test)? Fast testbed = EEGNet (braindecode). 3 subjects.
Compare selective-risk (AURC, lower=better) on session_E:
  (single-UQ) vanilla EEGNet entropy ; (C2) EEGNet vs D1-Riemann disagreement ;
  (CFCS) drift+deep-gambler EEGNet's own abstain signal (SINGLE model at test).
Run with intentflow conda env, GPU.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn as nn
from datamodules.bcic4_2a import BCICIV2a
from braindecode.models import EEGNetv4
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
D1all=np.load("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]

def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    y=np.array([int(ds[i][1]) for i in range(len(ds))]); return X,y
def drift_aug(x):  # x:(B,C,T) tensor; simulate session drift: per-trial channel gain + small spatial mix
    B,C,T=x.shape
    g=torch.exp(0.25*torch.randn(B,C,1,device=x.device))            # log-normal channel gains
    M=torch.eye(C,device=x.device)[None]+0.10*torch.randn(B,C,C,device=x.device)  # small spatial mixing
    return torch.einsum('bij,bjt->bit', M, x*g)
def softmax_np(z): z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def aurc(conf, correct):  # higher conf kept first
    o=np.argsort(-conf);c=correct[o];return float(np.trapz(1-np.cumsum(c)/np.arange(1,len(c)+1), np.arange(1,len(c)+1)/len(c)))

def train_eegnet(Xtr,ytr,C,T,gambler=False,drift=False,o_reward=2.6,epochs=250):
    ncls=4+(1 if gambler else 0)
    net=EEGNetv4(C,4 if not gambler else ncls,n_times=T,final_conv_length='auto').to(dev) if False else None
    # build EEGNet with ncls outputs
    net=EEGNetv4(C,ncls,n_times=T,final_conv_length='auto').to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev)
    n=len(yt)
    for ep in range(epochs):
        perm=torch.randperm(n,device=dev)
        for i in range(0,n,48):
            idx=perm[i:i+48];xb=Xt[idx];yb=yt[idx]
            if drift: xb=drift_aug(xb)
            out=net(xb)
            if gambler:
                p=torch.softmax(out,1)
                py=p[torch.arange(len(yb)),yb]; pab=p[:,-1]
                loss=(-torch.log(py + pab/o_reward + 1e-8)).mean()
            else:
                loss=nn.functional.cross_entropy(out[:, :4], yb)
            opt.zero_grad();loss.backward();opt.step()
    net.eval();return net
def predict(net,X,gambler=False):
    with torch.no_grad():
        out=net(torch.tensor(X).to(dev)).cpu().numpy()
    return out

SUBS=[2,5,6]
print(f"device={dev}; subjects={SUBS}")
rows=[]
for sid in SUBS:
    dm=BCICIV2a(prep,sid);dm.setup()
    Xtr,ytr=getxy(dm.train_dataset);Xte,yte=getxy(dm.test_dataset)
    C,T=Xtr.shape[1],Xtr.shape[2]
    D1=D1all[sid-1]
    # vanilla EEGNet (for single-UQ + C2)
    netv=train_eegnet(Xtr,ytr,C,T,gambler=False,drift=False)
    zv=predict(netv,Xte)[:, :4]; pv=softmax_np(zv); ev=pv.argmax(1); accv=(ev==yte).mean()*100
    # CFCS EEGNet (drift + deep-gambler)
    netc=train_eegnet(Xtr,ytr,C,T,gambler=True,drift=True)
    zc=predict(netc,Xte,gambler=True); pc=softmax_np(zc)
    # base predictor for selective-risk = vanilla EEGNet (fair: same decision, different abstain signals)
    correct=(ev==yte).astype(float)
    m=lambda p:np.sort(p,1)[:,-1]-np.sort(p,1)[:,-2]
    # signals (higher conf=keep)
    s_single=m(pv)                                  # vanilla margin
    def symkl(p,q):return ((p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(q,1e-12,1)))).sum(1)+(q*(np.log(np.clip(q,1e-12,1))-np.log(np.clip(p,1e-12,1)))).sum(1))
    s_c2=-symkl(pv,D1)                              # cross-family disagreement (2 models at test)
    s_cfcs=-pc[:,-1]                                # CFCS abstain logit (single model at test): high abstain=low conf
    print(f"S{sid}: EEGNet acc {accv:.1f} | AURC  single {aurc(s_single,correct):.4f}  C2 {aurc(s_c2,correct):.4f}  CFCS {aurc(s_cfcs,correct):.4f}",flush=True)
    rows.append((aurc(s_single,correct),aurc(s_c2,correct),aurc(s_cfcs,correct)))
A=lambda i:float(np.mean([r[i] for r in rows]))
print(f"\nMEAN AURC (lower=better): single-UQ {A(0):.4f} | C2-gate(2 models) {A(1):.4f} | CFCS(1 model) {A(2):.4f}")
print("MAKE-OR-BREAK: CFCS must be <= C2 to justify single-model deployment. If CFCS >> C2, the 2nd model is needed (C2 is the deployable thing, CFCS-single fails).")
