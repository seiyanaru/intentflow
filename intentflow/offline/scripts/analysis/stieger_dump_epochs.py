"""ONE-TIME dump of preprocessed Stieger2021 epochs so all downstream analysis avoids the 233s/subject
MOABB get_data re-filtering. Saves per subject: X (n,60,T) float32, y (0/1), sess (int). Checkpointed +
resumable (skips subjects already dumped). After this, signals/sequential/gate run in seconds from cache.
"""
import os, warnings, glob, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
from moabb.datasets import Stieger2021
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
OUT = "/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
os.makedirs(OUT, exist_ok=True)
EEG60 = ['AF3','AF4','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','FCz','FT7','FT8','Fp1','Fp2','Fpz','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO5','PO6','PO7','PO8','POz','Pz','T7','T8','TP7','TP8']
prm = LeftRightImagery(channels=EEG60, resample=250, fmin=8, fmax=30)
ds = Stieger2021()
done = {int(os.path.basename(f).split("_")[0][1:]) for f in glob.glob(f"{OUT}/S*_epochs.npz")}
print(f"resume: {len(done)} subjects already dumped", flush=True)
for sub in ds.subject_list:
    if sub in done: continue
    try:
        X, y, meta = prm.get_data(dataset=ds, subjects=[sub])
        yb = (y == 'right_hand').astype(np.int8)
        sess = np.array([int(v) for v in meta['session'].values], dtype=np.int16)
        np.savez_compressed(f"{OUT}/S{sub}_epochs.npz", X=X.astype(np.float32), y=yb, sess=sess)
        print(f"S{sub}: dumped X{X.shape} y{yb.shape}", flush=True)
        del X, y, meta
    except Exception as e:
        print(f"S{sub}: FAIL {str(e)[:90]}", flush=True)
print(f"\ndone. cache dir: {OUT}")
