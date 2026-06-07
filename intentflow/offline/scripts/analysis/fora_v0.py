"""FORA v0 (deploy-time logic, E0a-grounded regime-dependent design; CFCS-train
not included yet). Validate on saved preds, held-out cross-seed, FROZEN rule.
- LOW-confidence region (deep margin < per-subject median): accuracy regime ->
  DA-L1 ensemble (blend + w*D) tips close calls.
- HIGH-confidence region (margin >= median): safety regime -> if deep & cross-family
  AGREE trust; if DISAGREE -> ABSTAIN (suspected confident-wrong).
Compare: (acc) FORA vs DA-L1 vs blend; (safety) confident-wrong caught + risk-coverage
vs single-margin abstain; (cost) cross-family invocation is a cheap classical model.
"""
import numpy as np
D1 = np.load("intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
def load(seed):
    p=f"intentflow/offline/results/research_outputs/260602_expert_portfolio_table{'' if seed==0 else '_seed1'}/expert_portfolio_arrays.npz"
    e=np.load(p,allow_pickle=True);ex=[str(x) for x in e["experts"].tolist()]
    return e["probs"],e["labels"],{n:ex.index(n) for n in ["source","full_ea","shrink_0.1"]}
m=lambda L:float(np.nanmean(L))
W=0.3
for seed in [0,1]:
    Tp,Lab,EI=load(seed)
    acc_blend=[];acc_dal1=[];acc_fora=[]
    ab_rate=[];cw_total=[];cw_fora=[];cw_margin=[]
    cov=[];risk_fora=[];risk_margin=[]
    for i in range(9):
        y=Lab[i];T=Tp[i,EI["source"]];F=Tp[i,EI["full_ea"]];S=Tp[i,EI["shrink_0.1"]];Dd=D1[i]
        blend=0.3*T+0.4*F+0.3*S
        tp=T.argmax(1);dp=Dd.argmax(1);bp=blend.argmax(1)
        marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2]
        hi=marg>=np.median(marg)            # high-confidence region (label-free)
        dal1=(blend+W*Dd).argmax(1)
        # FORA accuracy: low-conf -> DA-L1; high-conf -> blend (agree or not, keep best static)
        fora=np.where(hi, bp, dal1)
        acc_blend.append((bp==y).mean()*100);acc_dal1.append((dal1==y).mean()*100);acc_fora.append((fora==y).mean()*100)
        # FORA safety: abstain set = high-conf AND deep-crossfamily disagree
        abstain = hi & (tp!=dp)
        ab_rate.append(abstain.mean()*100)
        hi_err = hi & (bp!=y)                # confident-wrong (high-conf, deployed-wrong)
        cw_total.append(hi_err.sum())
        cw_fora.append((abstain & hi_err).sum())
        # margin abstain at SAME budget
        b=abstain.sum()
        mo=np.argsort(marg)[:b]              # lowest-margin trials (margin's abstain choice)
        mmask=np.zeros(len(y),bool);mmask[mo]=True
        cw_margin.append((mmask & hi_err).sum())
        # risk-coverage: keep non-abstained, accuracy
        kept=~abstain
        cov.append(kept.mean()*100);risk_fora.append((bp[kept]!=y[kept]).mean()*100)
        keptm=~mmask;risk_margin.append((bp[keptm]!=y[keptm]).mean()*100)
    tag="seed0(in-sample threshold)" if seed==0 else "seed1(HELD-OUT, frozen rule)"
    print(f"\n===== {tag} =====")
    print(f"ACCURACY: blend {m(acc_blend):.2f} | DA-L1 {m(acc_dal1):.2f} | FORA {m(acc_fora):.2f}")
    cwt=sum(cw_total);print(f"SAFETY: abstain-rate {m(ab_rate):.1f}% | confident-wrong total {cwt} | caught by FORA {sum(cw_fora)} ({100*sum(cw_fora)/max(1,cwt):.0f}%) vs margin-at-same-budget {sum(cw_margin)} ({100*sum(cw_margin)/max(1,cwt):.0f}%)")
    print(f"        kept-coverage {m(cov):.1f}% | risk(kept) FORA {m(risk_fora):.2f}% vs margin {m(risk_margin):.2f}%")
print("\nNote: FORA accuracy ~= DA-L1 by design (the new value is the SAFETY layer: catching confident-wrong margin keeps). D = cheap classical (sub-ms). CFCS-train would shrink the confident-wrong mass further.")
