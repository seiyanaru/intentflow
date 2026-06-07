import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EP = np.load('/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz', allow_pickle=True)
probs=EP['probs']; labels=EP['labels']; experts=list(EP['experts'])
i_src=experts.index('source'); i_full=experts.index('full_ea'); i_shr=experts.index('shrink_0.1')
blend=0.3*probs[:,i_src]+0.4*probs[:,i_full]+0.3*probs[:,i_shr]
D1=np.load('/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz')['probs']
dal1=blend+0.3*D1
bank=np.load('/tmp/bank_probs_loso.npz')['probs']; bank_mean=bank.mean(1)
def acc(p):return (p.argmax(-1)==labels).mean()
def acc_s(p,s):return (p[s].argmax(-1)==labels[s]).mean()

Cgrid=[0.02,0.05,0.1,0.2,0.5,1.0]

def run(featfn, label):
    X={s:featfn(s) for s in range(9)}
    out=np.zeros((9,288,4)); chosenC=[]
    for h in range(9):
        tr=[s for s in range(9) if s!=h]
        # nested: pick C by inner-LOSO over the 8 training subjects
        bestC=None; bestA=-1
        for C in Cgrid:
            innacc=[]
            for v in tr:
                inn=[s for s in tr if s!=v]
                Xi=np.concatenate([X[s] for s in inn]); yi=np.concatenate([labels[s] for s in inn])
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))
                clf.fit(Xi,yi)
                pv=clf.predict_proba(X[v]); innacc.append((pv.argmax(-1)==labels[v]).mean())
            m=np.mean(innacc)
            if m>bestA: bestA=m; bestC=C
        chosenC.append(bestC)
        Xtr=np.concatenate([X[s] for s in tr]); ytr=np.concatenate([labels[s] for s in tr])
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=bestC,max_iter=3000))
        clf.fit(Xtr,ytr); out[h]=clf.predict_proba(X[h])
    print(f"{label}: nested-LOSO acc {round(acc(out)*100,2)}  chosenC={chosenC}")
    return out

print("DA-L1 fixed:", round(acc(dal1)*100,2))
o1=run(lambda s: np.concatenate([blend[s],D1[s],bank_mean[s]],axis=1), "[blend,D1,bank_mean]")
o2=run(lambda s: np.concatenate([blend[s],D1[s]],axis=1), "[blend,D1] (no bank)")
o3=run(lambda s: np.concatenate([blend[s],D1[s]]+[bank[s,bi] for bi in range(6)],axis=1), "[blend,D1,all6bands]")

# per-subject for best (o1)
src=[acc_s(probs[:,i_src],s) for s in range(9)]
da=[acc_s(dal1,s) for s in range(9)]
fz=[acc_s(o1,s) for s in range(9)]
print("\nsubj  src    DA-L1  STACK[blend,D1,bank]")
for s in range(9):
    print(f"S{s+1}  {src[s]*100:5.1f}  {da[s]*100:5.1f}  {fz[s]*100:5.1f}")
print("mean ", round(np.mean(src)*100,2), round(np.mean(da)*100,2), round(np.mean(fz)*100,2))
print("HSC stack>source:", sum(1 for s in range(9) if fz[s]>src[s]),"/9")
print("stack>DA-L1:", sum(1 for s in range(9) if fz[s]>da[s]),"/9 ; >=:", sum(1 for s in range(9) if fz[s]>=da[s]-1e-9),"/9")
print("bank marginal over [blend,D1] stack:", round((acc(o1)-acc(o2))*100,2),"pp")
