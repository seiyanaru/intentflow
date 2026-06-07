"""Sagittal-symmetry TTA (label-free, novel): MI has a physiological L/R symmetry --
reflecting EEG across the midline (swap left<->right sensorimotor channels) should swap
the left-hand/right-hand class. A drift-biased model violates this. Symmetrize the
prediction:  p_sym = 0.5*( p(x) + classswap( p(reflect_LR(x)) ) ).
Uses ONLY the frozen deep model (no weaker auxiliary). Compare acc(p_sym) vs acc(p(x)).
Non-collapsing (a constant predictor violates equivariance). Run: intentflow env, GPU.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, yaml
from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
dev="cuda" if torch.cuda.is_available() else "cpu"
if dev!="cuda": raise RuntimeError("need GPU")

# BCIC2a 22ch montage L<->R mirror permutation (Fz..POz standard order)
PERM2A=[0,5,4,3,2,1,12,11,10,9,8,7,6,17,16,15,14,13,20,19,18,21]
SWAP2A=[1,0,2,3]   # class0 left-hand <-> class1 right-hand; feet/tongue fixed
PERM2B=[2,1,0]; SWAP2B=[1,0]   # C3,Cz,C4 -> swap C3<->C4 ; left<->right

def sm(z): return torch.softmax(z,1)
@torch.no_grad()
def run(model, X, perm, swap):
    Xt=torch.tensor(X).to(dev)
    p0=sm(model(Xt)).cpu().numpy()
    pr=sm(model(Xt[:,perm,:])).cpu().numpy()[:,swap]   # reflected + class-swapped
    return p0, pr

def getX(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    y=np.array([int(ds[i][1]) for i in range(len(ds))]); return X,y

def load_model(ckpt, cfg, ds_cls, sid):
    mk=dict(cfg["model_kwargs"]); mk["n_channels"]=ds_cls.channels; mk["n_classes"]=ds_cls.classes
    m=get_model_cls("tcformer")(**mk,max_epochs=cfg.get("max_epochs",1000),subject_id=sid,model_name="tcformer",results_dir=".")
    c=torch.load(ckpt,map_location="cpu",weights_only=False); m.load_state_dict(c["state_dict"]); m.to(dev).eval(); return m

def eval_ds(name, ckpt_glob, cfg_path, ds_name, perm, swap):
    cfg=yaml.safe_load(open(cfg_path)); ds_cls=get_datamodule_cls(ds_name)
    cfg["preprocessing"]=dict(cfg["preprocessing"]); cfg["preprocessing"]["ea"]={"enabled":False}  # SOURCE baseline (no EA)
    print(f"\n=== {name}: do-no-harm GATED sagittal-symmetry [base = SOURCE, EA-off] ===")
    print(f"{'subj':>4} {'base':>7} {'sym_LR':>8} {'GATED':>7} {'gΔ':>6} {'fire%':>6}")
    base=[];symlr=[];gat=[]
    for sid in range(1,10):
        cks=sorted(glob.glob(ckpt_glob.format(sid=sid)))
        if not cks: print(f"S{sid} no ckpt"); continue
        dm=ds_cls(cfg["preprocessing"],subject_id=sid); dm.setup("fit")
        X,y=getX(dm.test_dataset)
        m=load_model(cks[-1],cfg,ds_cls,sid)
        p0,pr=run(m,X,perm,swap)                          # pr = reflected + class-swapped
        a0=(p0.argmax(1)==y).mean()*100
        plr=p0.copy(); plr[:,0:2]=0.5*(p0[:,0:2]+pr[:,0:2]); plr=plr/plr.sum(1,keepdims=True)
        # DO-NO-HARM GATE: correct a trial only when the model VIOLATES symmetry (argmax disagree)
        # AND the reflected view is at least as confident (the violation is a real, trustworthy error signal).
        disagree = p0.argmax(1)!=pr.argmax(1)
        trust = pr.max(1) >= p0.max(1)
        fire = disagree & trust
        pg=p0.copy(); pg[fire]=plr[fire]
        symlr.append((plr.argmax(1)==y).mean()*100); gat.append((pg.argmax(1)==y).mean()*100)
        base.append(a0)
        print(f"S{sid:>3} {a0:>7.1f} {symlr[-1]:>8.1f} {gat[-1]:>7.1f} {gat[-1]-a0:>+6.1f} {fire.mean()*100:>6.1f}")
        del m; torch.cuda.empty_cache()
    dlt=np.array(gat)-np.array(base)
    print(f"{'MEAN':>4} {np.mean(base):>7.1f} {np.mean(symlr):>8.1f} {np.mean(gat):>7.1f} {np.mean(dlt):>+6.1f}   worst-subj Δ={dlt.min():+.1f} (>=−1 = do-no-harm OK)")
    return np.mean(base),np.mean(gat)

RR="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/"
cfg2a=sorted(glob.glob(RR+"ea_aware_tcformer_s1_seed0_*/config.yaml"))[-1]
eval_ds("2a (22ch)", RR+"ea_aware_tcformer_s{sid}_seed0_*/checkpoints/subject_{sid}_model.ckpt", cfg2a, "bcic2a", PERM2A, SWAP2A)
cfg2b=sorted(glob.glob(RR+"ea_aware_tcformer_bcic2b_s12_seed0_*/config.yaml"))
if cfg2b:
    eval_ds("2b (3ch)", RR+"ea_aware_tcformer_bcic2b_s*_seed0_*/checkpoints/subject_{sid}_model.ckpt", cfg2b[-1], "bcic2b", PERM2B, SWAP2B)
print("\nVERDICT: sym-TTA Δ>0 on BOTH datasets (label-free, frozen, any-ch) => novel general accuracy gain.")
print("(base no-EA should be ~82.7 on 2a = loader sanity. asym% = how badly drift broke L/R equivariance.)")
