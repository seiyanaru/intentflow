"""Extract 64-d source penultimate features for session_T (train) and session_E (eval).

Source expert = no-EA TCFormer baseline (matches features_s*.npz saved during baseline run).
We re-run the saved checkpoints with the same feature hook (input to tcn_head.classifier)
to obtain TRAIN features, which are needed to compute a per-trial DRIFT signal
(distance of an eval trial's source feature from the source-TRAIN centroid).

Validation: re-extracted eval features must match the saved features_s*.npz closely.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OFFLINE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, OFFLINE)

from datamodules.bcic4_2a import BCICIV2a  # noqa: E402
from models.tcformer.tcformer import TCFormer  # noqa: E402

CKPT_DIR = os.path.join(OFFLINE, "results/baseline_5seed_s0_20260309_122106/checkpoints")
FEAT_DIR = os.path.join(OFFLINE, "results/baseline_5seed_s0_20260309_122106")
OUT = os.path.join(OFFLINE, "results/source_train_features_s0.npz")

PREP = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/",
            eval_label_path="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels")

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "GPU required per policy"


def build_model(ckpt):
    hp = ckpt["hyper_parameters"]
    model = TCFormer(
        n_channels=hp["n_channels"], n_classes=hp["n_classes"], F1=hp["F1"],
        temp_kernel_lengths=tuple(hp["temp_kernel_lengths"]),
        pool_length_1=hp["pool_length_1"], pool_length_2=hp["pool_length_2"],
        D=hp["D"], dropout_conv=hp["dropout_conv"], d_group=hp["d_group"],
        tcn_depth=hp["tcn_depth"], kernel_length_tcn=hp["kernel_length_tcn"],
        dropout_tcn=hp["dropout_tcn"], use_group_attn=hp["use_group_attn"],
        q_heads=hp["q_heads"], kv_heads=hp["kv_heads"], trans_depth=hp["trans_depth"],
        trans_dropout=hp["trans_dropout"],
    )
    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "model." in m], f"missing model params: {missing[:5]}"
    model.eval().to(device)
    return model


captured = {}


def hook(module, inp, out):
    x = inp[0] if isinstance(inp, (tuple, list)) else inp
    captured["f"] = x.detach().cpu().float()


def run_split(model, X, y):
    # X: (N, C, T) float32; returns features (N,64), logits (N,4)
    feats, logits = [], []
    model.model.tcn_head.classifier.register_forward_hook(hook)
    with torch.no_grad():
        for i in range(0, len(X), 48):
            xb = torch.from_numpy(X[i:i+48]).float().to(device)
            out = model(xb)
            logits.append(out.detach().cpu().float())
            feats.append(captured["f"].clone())
    F = torch.cat(feats).numpy()
    L = torch.cat(logits).numpy()
    if F.ndim == 3:
        F = F[:, :, 0]
    return F, L


def get_arrays(dm, which):
    ds = dm.train_dataset if which == "train" else dm.test_dataset
    X = np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
    y = np.array([int(ds[i][1]) for i in range(len(ds))])
    if X.shape[-1] > 1000:
        X = X[..., :1000]
    return X, y


train_feats, train_labels, eval_feats, eval_labels = {}, {}, {}, {}
for sid in range(1, 10):
    ckpt = torch.load(os.path.join(CKPT_DIR, f"subject_{sid}_model.ckpt"),
                      map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    dm = BCICIV2a(PREP, sid)
    dm.setup()
    Xtr, ytr = get_arrays(dm, "train")
    Xte, yte = get_arrays(dm, "test")
    Ftr, _ = run_split(model, Xtr, ytr)
    Fte, Lte = run_split(model, Xte, yte)
    # validate eval features vs saved
    saved = np.load(os.path.join(FEAT_DIR, f"features_s{sid}_TCFormer.npz"))
    sf = saved["features"]
    if sf.ndim == 3:
        sf = sf[:, :, 0]
    pred_acc = (Lte.argmax(1) == yte).mean()
    if sf.shape == Fte.shape:
        corr = np.corrcoef(sf.ravel(), Fte.ravel())[0, 1]
        mad = np.abs(sf - Fte).mean()
    else:
        corr, mad = float("nan"), float("nan")
    print(f"s{sid}: train {Ftr.shape} eval {Fte.shape} | eval_acc={pred_acc:.3f} "
          f"feat_corr_vs_saved={corr:.4f} mad={mad:.4f}")
    train_feats[sid] = Ftr; train_labels[sid] = ytr
    eval_feats[sid] = Fte; eval_labels[sid] = yte

np.savez(OUT,
         **{f"train_feat_{s}": train_feats[s] for s in range(1, 10)},
         **{f"train_label_{s}": train_labels[s] for s in range(1, 10)},
         **{f"eval_feat_{s}": eval_feats[s] for s in range(1, 10)},
         **{f"eval_label_{s}": eval_labels[s] for s in range(1, 10)})
print("saved", OUT)
