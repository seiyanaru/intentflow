"""Dump 2b per-trial E4 arrays (F0/F1/y0/y1) from the VERIFIED 2b baseline checkpoints.
Mirrors test_drift_vs_junk_2b.py feature extraction EXACTLY (config has NO EA -> source features).
GPU forward only, NO training (~minutes). k=2. Output: 260608_2b_pertrial.npz.
SANITY: harness lda mode acc should ~87.7 (2b source).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, yaml
from models.tcformer.tcformer import TCFormer
from datamodules.bcic4_2b import BCICIV2b
dev = "cuda" if torch.cuda.is_available() else "cpu"; assert dev == "cuda"
RUN = "intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347"
OUT = "intentflow/offline/results/research_outputs/260608_2b_pertrial.npz"
cfg = yaml.safe_load(open(RUN + "/config.yaml"))
prep = dict(cfg["preprocessing"]); prep.update(test_batch_size=48, num_workers=0, eval_label_path=None, data_path=None)
assert "ea" not in prep or not prep.get("ea"), "config unexpectedly has EA"
def build(ck):
    hp = ck["hyper_parameters"]
    m = TCFormer(n_channels=hp["n_channels"], n_classes=hp["n_classes"], F1=hp["F1"], temp_kernel_lengths=tuple(hp["temp_kernel_lengths"]),
        pool_length_1=hp["pool_length_1"], pool_length_2=hp["pool_length_2"], D=hp["D"], dropout_conv=hp["dropout_conv"], d_group=hp["d_group"],
        tcn_depth=hp["tcn_depth"], kernel_length_tcn=hp["kernel_length_tcn"], dropout_tcn=hp["dropout_tcn"], use_group_attn=hp["use_group_attn"],
        q_heads=hp["q_heads"], kv_heads=hp["kv_heads"], trans_depth=hp["trans_depth"], trans_dropout=hp["trans_dropout"])
    m.load_state_dict(ck["state_dict"], strict=False); return m.eval().to(dev)
cap = {}
def feats(net, X):
    h = net.model.tcn_head.classifier.register_forward_hook(lambda mod, i, o: cap.__setitem__('f', (i[0] if isinstance(i, (tuple, list)) else i).detach().cpu().float()))
    F = []
    with torch.no_grad():
        for i in range(0, len(X), 48):
            net(torch.from_numpy(X[i:i + 48]).float().to(dev)); Ff = cap['f']; F.append((Ff[:, :, 0] if Ff.ndim == 3 else Ff).numpy())
    h.remove(); return np.concatenate(F)
def arr(ds):
    X = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
    if X.shape[-1] > 1000: X = X[..., :1000]
    return X, np.array([int(ds[i][1]) for i in range(len(ds))])
npz = {}
for s in range(1, 10):
    ck = torch.load(os.path.join(RUN, "checkpoints", f"subject_{s}_model.ckpt"), map_location="cpu", weights_only=False)
    net = build(ck); dm = BCICIV2b(prep, s); dm.setup()
    Xtr, ytr = arr(dm.train_dataset); Xte, yte = arr(dm.test_dataset)
    Ftr = feats(net, Xtr); Fte = feats(net, Xte); del net; torch.cuda.empty_cache()
    npz[f"F0_{s}"] = Ftr.astype(np.float32); npz[f"F1_{s}"] = Fte.astype(np.float32)
    npz[f"y0_{s}"] = ytr.astype(np.int64); npz[f"y1_{s}"] = yte.astype(np.int64)
    print(f"S{s}: F0{Ftr.shape} F1{Fte.shape} classes={len(np.unique(ytr))}", flush=True)
np.savez_compressed(OUT, **npz); print("saved", OUT, "(2b, subjects 1-9, lda mode)")
