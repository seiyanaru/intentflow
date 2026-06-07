"""v0: does ACCUMULATED noisy-ErrP feedback, driving a source-anchored linear
HEAD adapter on frozen features, break the +2-3pp label-free cap and reach the
confident-error headroom?

Frozen backbone = the EA-aware TCFormer (penultimate features (288,64) + logits
(288,4) saved per subject). We recover the deployed head theta0 by least squares
(logits = feat@W0.T + b0), then run an ONLINE PREQUENTIAL loop on the eval
session: predict with the current head (causal), get a NOISY ErrP correctness
signal, buffer it, and every B trials update ONLY the head via
  L = w_c * CE(p,k)            on ErrP-correct trials   (reinforce shown class k)
    + w_e * (-log(1-p[k]))     on ErrP-error trials     (negative learning: push k down)
    + lambda * ||theta-theta0||^2                         (source anchor)
  with a hard deviation clamp ||theta-theta0||_F <= rho   (harm-first).

Key reference = CLEAN supervised head ceiling (perfect feedback). If even that
cannot beat the cap, head adaptation is dead regardless of ErrP. If it can, the
question is how much ErrP noise erodes it.
Caveat: trials processed in saved order (assumed ~chronological); full-EA backbone.
"""

from __future__ import annotations
import glob
import numpy as np

EPS = 1e-9


def load_subject(sid):
    fd = sorted(glob.glob(f"intentflow/offline/results/ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz"))[0]
    ld = sorted(glob.glob(f"intentflow/offline/results/ea_aware_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy"))[0]
    f = np.load(fd, allow_pickle=True)
    h = f["features"].reshape(f["features"].shape[0], -1).astype(np.float64)   # (T,64)
    y = f["labels"].astype(int)
    z0 = np.load(ld).astype(np.float64)                                        # (T,4)
    return h, y, z0


def recover_head(h, z0):
    A = np.concatenate([h, np.ones((len(h), 1))], 1)        # (T,65)
    Theta, *_ = np.linalg.lstsq(A, z0, rcond=None)          # (65,4)
    W = Theta[:-1].T                                        # (4,64)
    b = Theta[-1]                                           # (4,)
    return W, b


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def grad_step(W, b, W0, b0, buf, lr, lam, rho, w_c, w_e, ksteps):
    """buf: list of (h[64], k_int, is_error_bool). In-place-ish SGD on (W,b)."""
    H = np.stack([x[0] for x in buf])           # (n,64)
    K = np.array([x[1] for x in buf])           # (n,)
    ERR = np.array([x[2] for x in buf])         # (n,) bool
    n = len(buf)
    for _ in range(ksteps):
        p = softmax(H @ W.T + b)                 # (n,4)
        gz = np.zeros_like(p)                    # grad wrt logits, (n,4)
        # ErrP-correct -> CE toward k:  g = p - onehot(k)
        cidx = ~ERR
        if cidx.any():
            gc = p[cidx].copy()
            gc[np.arange(cidx.sum()), K[cidx]] -= 1.0
            gz[cidx] += w_c * gc
        # ErrP-error -> negative learning -log(1-p_k): g_k=p_k, g_{j!=k}=-p_k p_j/(1-p_k)
        eidx = ERR
        if eidx.any():
            pe = p[eidx]; ke = K[eidx]
            pk = pe[np.arange(len(ke)), ke]                  # (m,)
            ge = -(pk[:, None] / np.clip(1 - pk[:, None], EPS, None)) * pe
            ge[np.arange(len(ke)), ke] = pk
            gz[eidx] += w_e * ge
        gW = gz.T @ H / n + 2 * lam * (W - W0)
        gb = gz.mean(0) + 2 * lam * (b - b0)
        W = W - lr * gW
        b = b - lr * gb
        # harm-first deviation clamp
        dev = np.sqrt(((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
        cap = rho
        if dev > cap:
            s = cap / dev
            W = W0 + (W - W0) * s
            b = b0 + (b - b0) * s
    return W, b


def online_run(h, y, W0, b0, mode, eps, rng, B=16, lr=0.05, lam=0.01,
               rho=3.0, w_c=1.0, w_e=3.0, ksteps=5):
    """mode='errp' (noisy binary) or 'clean' (true-label CE ceiling)."""
    W, b = W0.copy(), b0.copy()
    buf = []
    correct = np.zeros(len(y), bool)
    for t in range(len(y)):
        z = h[t] @ W.T + b
        k = int(np.argmax(z))
        correct[t] = (k == y[t])                 # causal prequential
        if mode == "clean":
            # perfect feedback: treat as 'correct' with true label as target k=y
            buf.append((h[t], int(y[t]), False))
        else:
            true_err = (k != y[t])
            reported_err = true_err if rng.random() > eps else (not true_err)  # flip w.p. eps
            buf.append((h[t], k, bool(reported_err)))
        if len(buf) >= B:
            W, b = grad_step(W, b, W0, b0, buf, lr, lam, rho, w_c, w_e, ksteps)
    return correct.mean() * 100


def offline_ceiling(h, y, W0, b0, lam=0.01, rho=3.0, lr=0.1, iters=300):
    """Upper bound: supervised head refit on ALL eval labels, anchored+clamped."""
    W, b = W0.copy(), b0.copy()
    onehot = np.eye(4)[y]
    for _ in range(iters):
        p = softmax(h @ W.T + b)
        gz = p - onehot
        gW = gz.T @ h / len(y) + 2 * lam * (W - W0)
        gb = gz.mean(0) + 2 * lam * (b - b0)
        W -= lr * gW; b -= lr * gb
        dev = np.sqrt(((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
        if dev > rho:
            s = rho / dev; W = W0 + (W - W0) * s; b = b0 + (b - b0) * s
    return (np.argmax(h @ W.T + b, 1) == y).mean() * 100


def main():
    subs = range(1, 10)
    rng = np.random.RandomState(0)
    rows = {}
    EPS_GRID = [0.0, 0.10, 0.15, 0.20, 0.25]
    REPS = 12
    base_accs, head_match, top2 = [], [], []
    ceil, clean_online = [], []
    errp = {e: [] for e in EPS_GRID}
    for sid in subs:
        h, y, z0 = load_subject(sid)
        W0, b0 = recover_head(h, z0)
        base = (np.argmax(h @ W0.T + b0, 1) == y).mean() * 100
        match = (np.argmax(h @ W0.T + b0, 1) == np.argmax(z0, 1)).mean() * 100
        t2 = (np.sort(np.argsort(-z0, 1)[:, :2], 1) == y[:, None]).any(1).mean() * 100
        base_accs.append(base); head_match.append(match); top2.append(t2)
        ceil.append(offline_ceiling(h, y, W0, b0))
        clean_online.append(online_run(h, y, W0, b0, "clean", 0.0, rng))
        for e in EPS_GRID:
            accs = [online_run(h, y, W0, b0, "errp", e, np.random.RandomState(100 + r)) for r in range(REPS)]
            errp[e].append(np.mean(accs))
    m = lambda a: float(np.mean(a))
    print(f"head-recovery argmax match vs saved logits: {m(head_match):.1f}% (should be ~100)")
    print(f"\nbackbone = EA-aware (full-EA) TCFormer, online prequential, n=9, REPS={REPS}\n")
    print(f"no-adapt (deployed)          : {m(base_accs):6.2f}")
    print(f"top2 oracle (reference)      : {m(top2):6.2f}  (+{m(top2)-m(base_accs):.2f})")
    print(f"CLEAN supervised head ceiling: {m(ceil):6.2f}  (+{m(ceil)-m(base_accs):.2f})   <- can head-adapt even break the cap?")
    print(f"CLEAN online (perfect ErrP)  : {m(clean_online):6.2f}  (+{m(clean_online)-m(base_accs):.2f})")
    print(f"\nErrP-adapt (noisy binary), accuracy & delta vs no-adapt:")
    print(f"{'eps(ErrP err)':>14} {'acc':>7} {'delta':>7}")
    for e in EPS_GRID:
        a = m(errp[e])
        print(f"{e:>14.2f} {a:7.2f} {a-m(base_accs):>+7.2f}")
    # harm at eps=0.20
    e = 0.20
    deltas = [errp[e][i] - base_accs[i] for i in range(9)]
    harmed = [(i + 1, round(d, 2)) for i, d in enumerate(deltas) if d < -1e-9]
    print(f"\nharm @ eps=0.20: {len(harmed)}/9 harmed {harmed}; worst {min(deltas):+.2f}")
    print("\nVERDICT GUIDE: if CLEAN ceiling <= +3pp -> head-adapt capped (double-lock holds).")
    print("If CLEAN >> +3pp but ErrP@0.20 collapses to <=+2-3pp -> headroom exists but noise locks it.")
    print("If ErrP@0.20 clearly > +3pp -> the mechanism breaks the cap: pursue.")


if __name__ == "__main__":
    main()
