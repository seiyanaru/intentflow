"""SEQUENTIAL harm-persistence veto on Stieger2021 (cases.npz only, no GPU, no re-decode).

Motivation (verified facts, our work):
  - per-session EA-adapt Δacc is net-beneficial (+7.85pp) but has a harmful tail (83/524=16%).
  - harm PERSISTS in time: P(harm_j | harm_{j-1})=0.39 vs base 0.16 (2.4x); 84% of harmed sessions
    concentrate in the top-33% subjects; 29/60 subjects are NEVER harmed.
  - => the harmful tail is NOT random; it clusters by subject & in consecutive sessions.

Idea: don't try to judge each session in isolation (the mirage: minimal-label per-session pooling
degenerates to all-adopt). Instead use the LONGITUDINAL signal: a few probe labels on the PREVIOUS
session tell you whether THIS subject is currently in a harmful regime; veto (keep-source) while in it.

LEAK-FREE & label-accounted:
  - For session j (j>=2 in temporal order), the veto decision uses ONLY the first m probe trials of
    the PREVIOUS session j-1 (already-spent calibration), never any trial of j. Evaluation of j is on
    its own held-out trials [k_eval:] (we reserve the same first-k as not-evaluated to be conservative).
  - Label budget counted explicitly: m probe trials per session that we choose to probe.

Deciders compared (all leak-free):
  - never            : keep-source always (Δ=0, harmed=0; trivial-safe floor)
  - always           : adapt always (the unconditional baseline ~ Wimpff; harmed=full tail)
  - cohort-EB k=8    : the mirage baseline (all-adopt at small k)
  - per-session LCB k : isolated per-session gate at budget k (no time structure)
  - SEQ-prevΔ        : adapt j unless previous session's probe Δ̂ (m labels) was clearly negative
  - SEQ-runlen       : CUSUM-like; veto after a run of negative-probe sessions, resume after a positive
  - SEQ + cohort-prior: combine sequential veto with cohort-EB adopt-prior (adopt by default, veto on persistence)
Metric: effΔ vs source, harmed (adopt & true Δ<-1), %oracle, adopt-rate, and TOTAL LABELS spent.
"""
import numpy as np, json
from collections import defaultdict
np.random.seed(0)
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
SNAP = "/tmp/stieger_snap.npz"
d = np.load(SNAP)
keys = sorted({k[3:] for k in d.files if k.startswith("cs_")}, key=lambda s:(int(s.split("_")[0]),int(s.split("_")[1])))

# build per-subject temporally-ordered session list: each = dict(sj, cs, ca)
bysub = defaultdict(list)
for k in keys:
    sub, sj = map(int, k.split("_"))
    bysub[sub].append((sj, d[f"cs_{k}"].astype(float), d[f"ca_{k}"].astype(float)))
for sub in bysub: bysub[sub].sort(key=lambda t: t[0])

K_EVAL = 8   # reserve first K_EVAL of every session as non-eval (held-out = [K_EVAL:]); keeps eval disjoint from any probe

def held(cs, ca):
    """true held-out source/adapt acc + true Δpp on eval trials only."""
    return cs[K_EVAL:].mean(), ca[K_EVAL:].mean(), (ca[K_EVAL:].mean()-cs[K_EVAL:].mean())*100

def probe_dhat(cs, ca, m):
    """leak-free probe Δ̂ (pp) and its SE from FIRST m trials (used only as PRIOR-session evidence)."""
    diff = (ca - cs)[:m]
    return diff.mean()*100, max(diff.std(ddof=1)/np.sqrt(m)*100 if m>1 else 50.0, 0.1)

# ---- gather flat arrays for the non-sequential baselines (isolated per-session at budget k) ----
def eval_policy(decide, m_probe):
    """decide: fn(subject_state, prev_probe, cur_index) -> (adopt: bool, labels_used:int).
    Returns aggregate metrics over all evaluable sessions (j>=2 so a 'previous' exists; j=1 always keep-source)."""
    effs=[]; bases=[]; oras=[]; adopts=[]; harms=0; labels=0; tds=[]
    for sub, seq in bysub.items():
        prev_probe = None
        for idx,(sj,cs,ca) in enumerate(seq):
            bs, ad_acc, td = held(cs, ca)
            if idx == 0:
                # first session: no prior evidence -> keep-source (safe default), spend nothing
                adopt=False; lu=0
            else:
                adopt, lu = decide(sub, prev_probe, idx, seq)
            eff = ad_acc if adopt else bs
            effs.append(eff); bases.append(bs); oras.append(max(bs,ad_acc)); adopts.append(adopt); tds.append(td)
            if adopt and td < -1: harms += 1
            labels += lu
            # spend m_probe labels on THIS session to inform the NEXT session's decision
            prev_probe = probe_dhat(cs, ca, m_probe); labels += m_probe
    effs=np.array(effs);bases=np.array(bases);oras=np.array(oras);adopts=np.array(adopts);tds=np.array(tds)
    base=bases.mean()
    effΔ=(effs.mean()-base)*100; ora=(oras.mean()-base)*100; alw=(np.where(True,0,0),) # placeholder
    return dict(effΔ=float(effΔ), pct_oracle=float(effΔ/ora*100 if ora>1e-6 else np.nan),
                harmed=int(harms), adopt=float(adopts.mean()), labels=int(labels), n=int(len(effs)),
                oracle=float(ora))

# ---- reference policies (no sequential structure) ----
def isolated(name, rule, m):
    # rule decides using CURRENT session's own first-m probe (the classic per-session gate) -- leak-free vs eval [K_EVAL:] only if m<=K_EVAL... to be safe use m and eval[max(m,K_EVAL):]
    effs=[];bases=[];oras=[];adopts=[];harms=0;labels=0
    for sub,seq in bysub.items():
        for sj,cs,ca in seq:
            mm=max(m,K_EVAL)
            bs=cs[mm:].mean();ad=ca[mm:].mean();td=(ad-bs)*100
            dh,se=probe_dhat(cs,ca,m)
            adopt=rule(dh,se); labels+=m
            eff=ad if adopt else bs
            effs.append(eff);bases.append(bs);oras.append(max(bs,ad));adopts.append(adopt)
            if adopt and td<-1: harms+=1
    effs=np.array(effs);bases=np.array(bases);oras=np.array(oras);base=bases.mean()
    effΔ=(effs.mean()-base)*100;ora=(oras.mean()-base)*100
    return name,dict(effΔ=float(effΔ),pct_oracle=float(effΔ/ora*100 if ora>1e-6 else np.nan),
                     harmed=int(harms),adopt=float(np.mean(adopts)),labels=int(labels),n=int(len(effs)),oracle=float(ora))

results={}
# trivial floors / baselines
results["never (keep-source)"]   = isolated("never",lambda dh,se:False,0)[1]
results["always (~Wimpff)"]      = isolated("always",lambda dh,se:True,0)[1]
# isolated per-session LCB at k=8 and k=32 (the mirage regime vs label-hungry regime)
results["per-sess LCB k=8"]      = isolated("lcb8", lambda dh,se: dh-1.645*se>0, 8)[1]
results["per-sess LCB k=32"]     = isolated("lcb32",lambda dh,se: dh-1.645*se>0, 32)[1]

# ---- SEQUENTIAL deciders (use PREVIOUS session's probe, m labels) ----
def seq_prev(thresh):
    # adopt unless previous session's probe Δ̂ was clearly negative (< thresh)
    def decide(sub, prev, idx, seq):
        if prev is None: return True, 0
        dh, se = prev
        return (dh > thresh), 0   # labels for prev probe already counted as m_probe in eval_policy
    return decide

def seq_runlen(thresh, need):
    # veto while in a run of >=need consecutive negative-probe sessions; resume on a positive probe
    state={}
    def decide(sub, prev, idx, seq):
        if prev is None: return True, 0
        dh,se=prev
        run=state.get(sub,0)
        run = run+1 if dh < thresh else 0
        state[sub]=run
        return (run < need), 0
    return decide

def seq_subjmean(thresh):
    # adopt unless this subject's CUMULATIVE past probe Δ̂ mean (all prior sessions, m labels each) is < thresh
    hist=defaultdict(list)
    def decide(sub, prev, idx, seq):
        # prev is previous session's (dh,se); accumulate per subject
        if prev is not None: hist[sub].append(prev[0])
        if not hist[sub]: return True, 0
        return (np.mean(hist[sub]) > thresh), 0
    return decide

def seq_subjmean_lcb(thresh, zz=1.0):
    # adopt unless subject cumulative past Δ̂ LCB (mean - z*se/sqrt(#sess)) < thresh  -> "confidently bad subject"
    hist=defaultdict(list)
    def decide(sub, prev, idx, seq):
        if prev is not None: hist[sub].append(prev[0])
        h=hist[sub]
        if not h: return True, 0
        m_=np.mean(h); s_=(np.std(h,ddof=1)/np.sqrt(len(h))) if len(h)>1 else 8.0
        return (m_ - zz*s_ > thresh), 0
    return decide

for m in [4, 8, 16]:
    results[f"SEQ-prevΔ (m={m}, θ=0)"]   = eval_policy(seq_prev(0.0), m)
    results[f"SEQ-runlen (m={m}, θ=0,need=1)"] = eval_policy(seq_runlen(0.0,1), m)
    results[f"SEQ-subjMean (m={m}, θ=0)"] = eval_policy(seq_subjmean(0.0), m)
    results[f"SEQ-subjMean (m={m}, θ=+2)"] = eval_policy(seq_subjmean(2.0), m)
    results[f"SEQ-subjLCB (m={m}, θ=0)"] = eval_policy(seq_subjmean_lcb(0.0), m)

# ---- report ----
print(f"{'policy':>30} {'effΔ':>7} {'%ora':>6} {'harmed':>7} {'adopt':>6} {'labels':>7}  (n={results['never (keep-source)']['n']}, oracle={results['always (~Wimpff)']['oracle']:+.2f})")
for name,r in results.items():
    print(f"{name:>30} {r['effΔ']:>+7.2f} {r['pct_oracle']:>5.0f}% {r['harmed']:>7d} {r['adopt']:>6.2f} {r['labels']:>7d}")
json.dump(results, open(f"{RES}/260610_stieger_sequential_veto.json","w"), indent=2)
print(f"\nsaved {RES}/260610_stieger_sequential_veto.json")
print("WIN if a SEQ policy gets harmed << 81 at FEWER total labels than per-sess LCB k=32, while keeping effΔ positive.")
