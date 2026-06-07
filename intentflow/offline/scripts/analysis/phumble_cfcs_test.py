"""P-HUMBLE: the final CFCS make-or-break (complementary, safety-primary, EEGNet).
CFCS-humble = train EEGNet with drift-aug + an entropy penalty on the region where,
on the DRIFTED sample, the cross-family D (EA-Riemann tangent+LDA) DISAGREES with T
AND T is confident (the confident-WRONG-prone region). D is a fixed teacher computed
on each drifted batch. Test-time keeps D (DA-DC). Question: does humble-T + DA-DC catch
MORE confident-wrong / lower AURC than vanilla-T + DA-DC, with accuracy non-regression?
Run with intentflow conda env, GPU.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn as nn
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2a import BCICIV2a
from braindecode.models import EEGNetv4
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
D1all=np.load("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
b,a=butter(4,[8/125.,30/125.],btype="band")
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32); y=np.array([int(ds[i][1]) for i in range(len(ds))]); return X,y
def invsqrtm(M):w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def logm_spd(M):w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T
def tangent_vec(C,ref_P):
    Ca=ref_P@C@ref_P; c=C.shape[0]; iu=np.triu_indices(c); s=np.sqrt(2)*np.ones((c,c));np.fill_diagonal(s,1.);return (logm_spd(Ca)*s)[iu]
class Dteacher:
    """EA-Riemann tangent + LDA fit on clean session_T; inference on (drifted) raw batches."""
    def __init__(self,Xtr,ytr):
        Xf=filtfilt(b,a,Xtr,axis=-1); Cs=np.einsum('nct,ndt->ncd',Xf,Xf)/Xf.shape[-1]
        self.P=invsqrtm(Cs.mean(0)); V=np.array([tangent_vec(C,self.P) for C in Cs])
        self.lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(V,ytr)
    def pred(self,Xraw):  # Xraw (B,C,T) numpy -> argmax class
        Xf=filtfilt(b,a,Xraw,axis=-1); Cs=np.einsum('nct,ndt->ncd',Xf,Xf)/Xf.shape[-1]
        V=np.array([tangent_vec(C,self.P) for C in Cs]); return self.lda.predict(V)
def drift_aug(x):
    B,C,T=x.shape; g=torch.exp(0.25*torch.randn(B,C,1,device=x.device)); M=torch.eye(C,device=x.device)[None]+0.10*torch.randn(B,C,C,device=x.device)
    return torch.einsum('bij,bjt->bit',M,x*g)
def softmax_np(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def aurc(conf,correct):o=np.argsort(-conf);c=correct[o];return float(np.trapz(1-np.cumsum(c)/np.arange(1,len(c)+1),np.arange(1,len(c)+1)/len(c)))

def train(Xtr,ytr,C,T,humble=False,Dt=None,epochs=300,lam=0.5):
    net=EEGNetv4(C,4,n_times=T,final_conv_length='auto').to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev);n=len(yt)
    for ep in range(epochs):
        perm=torch.randperm(n,device=dev)
        for i in range(0,n,48):
            idx=perm[i:i+48];xb=Xt[idx];yb=yt[idx]
            if humble: xb=drift_aug(xb)
            out=net(xb);ce=nn.functional.cross_entropy(out,yb)
            loss=ce
            if humble:
                p=torch.softmax(out,1);conf=p.max(1).values;tp=p.argmax(1)
                xb_np=xb.detach().cpu().numpy(); dp=torch.tensor(Dt.pred(xb_np),device=dev)
                flag=(conf>0.5)&(tp!=dp)                       # confident & cross-family disagree
                if flag.any():
                    ent=-(p*torch.log(p+1e-8)).sum(1)
                    loss=ce - lam*ent[flag].mean()             # raise entropy (humble) on flagged
            opt.zero_grad();loss.backward();opt.step()
    net.eval();return net
def predict(net,X):
    with torch.no_grad(): return net(torch.tensor(X).to(dev)).cpu().numpy()

SUBS=list(range(1,10))
print(f"device={dev}")
def symkl(p,q):return ((p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(q,1e-12,1)))).sum(1)+(q*(np.log(np.clip(q,1e-12,1))-np.log(np.clip(p,1e-12,1)))).sum(1))
res={k:[] for k in ["acc_v","acc_h","aurc_v","aurc_h","cw_catch_v","cw_catch_h"]}
for sid in SUBS:
    dm=BCICIV2a(prep,sid);dm.setup();Xtr,ytr=getxy(dm.train_dataset);Xte,yte=getxy(dm.test_dataset)
    C,T=Xtr.shape[1],Xtr.shape[2];D1=D1all[sid-1]
    Dt=Dteacher(Xtr.astype(np.float64),ytr)
    netv=train(Xtr,ytr,C,T,humble=False)
    neth=train(Xtr,ytr,C,T,humble=True,Dt=Dt)
    pv=softmax_np(predict(netv,Xte));ph=softmax_np(predict(neth,Xte))
    # DA-DC: T + 0.3 D1 ; safety signal = high-conf region, abstain by (T margin low) OR (T vs D1 disagree)
    def evalsys(p):
        ens=(p+0.3*D1).argmax(1);acc=(ens==yte).mean()*100
        tp=p.argmax(1);Tc=tp==yte;marg=np.sort(p,1)[:,-1]-np.sort(p,1)[:,-2]
        # combined abstain signal: T's own (margin) + cross-family disagreement (the DA-DC gate)
        sig=-marg + 1.0*(symkl(p,D1)/ (symkl(p,D1).std()+1e-8))   # higher=suspicious
        a=aurc(-sig, Tc.astype(float))
        # confident-wrong catch at 15% budget
        cw=(marg>=np.median(marg))&(~Tc); n=int(len(yte)*0.15);o=np.argsort(-sig)[:n];fl=np.zeros(len(yte),bool);fl[o]=True
        catch=(fl&cw).sum()/max(1,cw.sum())*100
        return acc,a,catch
    av,auv,cv=evalsys(pv);ah,auh,ch=evalsys(ph)
    res["acc_v"].append(av);res["acc_h"].append(ah);res["aurc_v"].append(auv);res["aurc_h"].append(auh);res["cw_catch_v"].append(cv);res["cw_catch_h"].append(ch)
    print(f"S{sid}: DA-DC acc vanilla {av:.1f}->humble {ah:.1f} | AURC {auv:.3f}->{auh:.3f} | cw-catch@15% {cv:.0f}->{ch:.0f}",flush=True)
m=lambda k:float(np.mean(res[k]))
print(f"\n=== P-HUMBLE (EEGNet, 9 subj): vanilla vs CFCS-humble, both in DA-DC ===")
print(f"accuracy: {m('acc_v'):.2f} -> {m('acc_h'):.2f} ({m('acc_h')-m('acc_v'):+.2f})  [non-regression?]")
print(f"AURC(lower=better): {m('aurc_v'):.4f} -> {m('aurc_h'):.4f} ({m('aurc_h')-m('aurc_v'):+.4f})  [safety]")
print(f"confident-wrong catch@15%: {m('cw_catch_v'):.1f}% -> {m('cw_catch_h'):.1f}% ({m('cw_catch_h')-m('cw_catch_v'):+.1f})  [safety, KEY]")
print("\nSUCCESS = humble improves AURC / cw-catch with accuracy non-regression. Else CFCS adds nothing over plain DA-DC.")
