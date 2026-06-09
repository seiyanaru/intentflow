"""Build 2a per-trial E4 npz (F0/F1/y0/y1) from the EXISTING verified source features
(source_train_features_s0.npz: within-subject session_T(train)->session_E(eval) penultimate
features from the source TCFormer). CPU only, no GPU, no re-run. k=4 (4-class). The harness
runs lda mode (LDA(F0,y0)->F1), consistent with Lee2019's lda mode and the +0.78 lineage.
"""
import numpy as np
SRC = "intentflow/offline/results/source_train_features_s0.npz"
OUT = "intentflow/offline/results/research_outputs/260608_2a_pertrial.npz"
d = np.load(SRC, allow_pickle=True)
npz = {}
for s in range(1, 10):
    npz[f"F0_{s}"] = d[f"train_feat_{s}"].astype(np.float32)
    npz[f"F1_{s}"] = d[f"eval_feat_{s}"].astype(np.float32)
    npz[f"y0_{s}"] = np.asarray(d[f"train_label_{s}"]).astype(np.int64)
    npz[f"y1_{s}"] = np.asarray(d[f"eval_label_{s}"]).astype(np.int64)
    print(f"S{s}: F0{npz[f'F0_{s}'].shape} F1{npz[f'F1_{s}'].shape} classes={len(np.unique(npz[f'y0_{s}']))}")
np.savez_compressed(OUT, **npz)
print("saved", OUT, "(2a, subjects 1-9, lda mode)")
