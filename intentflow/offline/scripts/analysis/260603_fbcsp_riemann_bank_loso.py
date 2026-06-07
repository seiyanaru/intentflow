import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NB=6
R=np.load('/tmp/bank_covs.npz')
covs_tr={(s,bi):R[f'tr_{s}_{bi}'] for s in range(1,10) for bi in range(NB)}
covs_te={(s,bi):R[f'te_{s}_{bi}'] for s in range(1,10) for bi in range(NB)}
RAW=np.load('/tmp/raw_bcic2a_cropped.npz')
ytr={s:RAW[f'ytr_{s}'] for s in range(1,10)}
yte={s:RAW[f'yte_{s}'] for s in range(1,10)}

def softmax(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)

# LOSO: for held-out subject h, train each band tangent+LDA on pooled 8 subjects' BOTH sessions (with labels), predict h's test
bank_probs_loso = np.zeros((9, NB, 288, 4))  # held-out subject probs per band
for hi_idx, h in enumerate(range(1,10)):
    train_subj=[s for s in range(1,10) if s!=h]
    for bi in range(NB):
        # pool train: use other subjects' session_T covs+labels AND session_E covs+labels
        Xc=[]; yc=[]
        for s in train_subj:
            Xc.append(covs_tr[(s,bi)]); yc.append(ytr[s])
            Xc.append(covs_te[(s,bi)]); yc.append(yte[s])
        Xc=np.concatenate(Xc); yc=np.concatenate(yc)
        ts=TangentSpace(metric='riemann')
        Ftr=ts.fit_transform(Xc)
        clf=make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        clf.fit(Ftr, yc)
        # held-out test
        Fte=ts.transform(covs_te[(h,bi)])
        if hasattr(clf,'predict_proba'):
            p=clf.predict_proba(Fte)
        else:
            p=softmax(clf.decision_function(Fte))
        bank_probs_loso[h-1,bi]=p
    acc=[(bank_probs_loso[h-1,bi].argmax(-1)==yte[h]).mean() for bi in range(NB)]
    print('held-out',h,'per-band acc',[round(a*100,1) for a in acc])

np.savez_compressed('/tmp/bank_probs_loso.npz', probs=bank_probs_loso)
print('saved')
