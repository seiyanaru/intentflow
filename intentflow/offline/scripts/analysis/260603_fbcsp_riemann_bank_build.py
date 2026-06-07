import numpy as np, time, warnings
warnings.filterwarnings("ignore")
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

SFREQ=250
BANDS=[(4,8),(8,12),(12,16),(16,20),(20,26),(26,32)]
R=np.load('/tmp/raw_bcic2a_cropped.npz')
Xtr={s:R[f'Xtr_{s}'] for s in range(1,10)}
ytr={s:R[f'ytr_{s}'] for s in range(1,10)}
Xte={s:R[f'Xte_{s}'] for s in range(1,10)}
yte={s:R[f'yte_{s}'] for s in range(1,10)}

def bandpass(X, lo, hi):
    b,a=butter(4,[lo/(SFREQ/2),hi/(SFREQ/2)],btype='band')
    return filtfilt(b,a,X,axis=-1)

def cov_set(X):
    # X (N,C,T) -> covariances (N,C,C) with shrinkage via oas
    return Covariances(estimator='oas').transform(X)

def ea_whiten(covs_ref, covs_apply):
    # EA: reference mean = arithmetic mean of covs (Euclidean Alignment, He&Wu) over the set being aligned
    Rmean = covs_ref.mean(axis=0)
    # R^-1/2
    w,v=eigh(Rmean)
    w=np.clip(w,1e-12,None)
    Rinv2 = v @ np.diag(w**-0.5) @ v.T
    out = np.array([Rinv2 @ c @ Rinv2 for c in covs_apply])
    return out

# Precompute per-subject per-band EA-aligned covs for train(session_T) and test(session_E)
# EA reference is computed WITHIN each subject's own session set (label-free): train ref from train cov, test ref from test cov
t0=time.time()
covs_tr={}  # (sid,band)-> aligned covs train
covs_te={}
for bi,(lo,hi) in enumerate(BANDS):
    for s in range(1,10):
        ctr=cov_set(bandpass(Xtr[s],lo,hi))
        cte=cov_set(bandpass(Xte[s],lo,hi))
        covs_tr[(s,bi)]=ea_whiten(ctr,ctr)
        covs_te[(s,bi)]=ea_whiten(cte,cte)
    print('band',bi,(lo,hi),'done elapsed',round(time.time()-t0,1))

np.savez_compressed('/tmp/bank_covs.npz',
    **{f'tr_{s}_{bi}':covs_tr[(s,bi)] for s in range(1,10) for bi in range(len(BANDS))},
    **{f'te_{s}_{bi}':covs_te[(s,bi)] for s in range(1,10) for bi in range(len(BANDS))})
print('saved covs elapsed', round(time.time()-t0,1))
