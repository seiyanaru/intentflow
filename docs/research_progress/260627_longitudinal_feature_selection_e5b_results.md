# E5b0/E5b1: source-side longitudinal feature selection — 2026-06-27

## 結論

**Accuracy breakthrough. Safety is improved vs E4a relative to outer-best-single, but not solved.**

E5aで class-conditional geometry が branch gap を説明することが見えたため、
held-out subject を完全に除外し、他被験者の multi-session labels だけから
class-stableな特徴次元を選ぶ pilot を実行した。

最初に試した diagonal metric scaling は、LDAでは全strengthで予測が完全に同じになった。
これは実質 no-op である。理由は、LDA が可逆な対角スケーリングにほぼ不変だからである。
したがって、metric scaling ではなく **feature selection**、つまり不安定な次元を実際に落とす方向に切り替えた。

結果、E4a equal posterior + prefix-EA に対して primary LR+UD で
**+2.409pp [95% CI +1.532, +3.366]**。

さらに E4a の outer-best-single 比較に直すと、
**+3.919pp [95% CI +3.061, +4.818]**。
これはこれまでの試行の中で最も強い accuracy signal である。

ただし、安全手法ではない。E4a equal に対する per-session lower-tail loss は大きい。
一方で、outer-best-single を基準にした risk は E4a equal より改善した。

```text
E4a equal posterior vs outer best single:
  gain +1.509pp
  R10 loss 11.241
  P(loss<-5pp) 16.15%

E5b longitudinal feature selection vs outer best single:
  gain +3.919pp
  R10 loss 10.589
  P(loss<-5pp) 12.55%
```

したがって主張は、

```text
safe adaptation solved
```

ではなく、

```text
source-side longitudinal labels can learn a class-stable feature subspace
that substantially improves the risk-utility frontier of zero-target-label
cross-session EEG-MI adaptation.
```

である。

## 実装

Script:

```text
intentflow/offline/scripts/analysis/stieger_longitudinal_metric_pilot.py
```

Output:

```text
intentflow/offline/results/research_outputs/260627_stieger_longitudinal_metric_pilot/
```

Command:

```bash
/home/islabshi/anaconda3/envs/intentflow/bin/python \
  intentflow/offline/scripts/analysis/stieger_longitudinal_metric_pilot.py \
  --subjects 1-62 \
  --output-dir intentflow/offline/results/research_outputs/260627_stieger_longitudinal_metric_pilot \
  --force-eval \
  --quiet \
  --strengths 0 \
  --fractions 1.0 0.75 0.5 0.25 0.1
```

Protocol:

```text
Stieger2021
subjects = 62
rows = 1558 session-condition rows
failures = 0
source = session 1
target prefix = first 32 trials, labels unused for EA
evaluation = trial 65 onward
branches = broad_all60 + fb_sensorimotor21_mu_beta
classifier = branch-wise LDA + equal posterior fusion
held-out subject = all sessions excluded from feature-score learning
```

Feature score:

```text
longitudinal score =
  (source class separation + target class separation)
  /
  (same-class source-target drift + source within-class variance + target within-class variance)
```

Retained fractions:

```text
q = 1.0, 0.75, 0.5, 0.25, 0.1
```

`q=1.0` is the E4a equal posterior + prefix-EA equivalent.

## Main results: vs E4a equal posterior

### LR

| method | acc | gain vs E4a equal | 95% CI | R10 loss vs E4a | P(loss<-5pp) |
|---|---:|---:|---:|---:|---:|
| q1.00 | 66.873 | 0.000 | [0, 0] | 0.000 | 0.00% |
| q0.75 | 67.502 | +0.629 | [-0.085, +1.343] | 9.612 | 14.91% |
| q0.50 | 68.279 | +1.406 | [+0.575, +2.265] | 9.902 | 14.24% |
| q0.25 | 69.159 | +2.286 | [+1.201, +3.437] | 10.318 | 14.13% |
| q0.10 | **70.431** | **+3.558** | **[+2.351, +4.879]** | 9.688 | 13.38% |

### UD

| method | acc | gain vs E4a equal | 95% CI | R10 loss vs E4a | P(loss<-5pp) |
|---|---:|---:|---:|---:|---:|
| q1.00 | 63.657 | 0.000 | [0, 0] | 0.000 | 0.00% |
| q0.75 | 64.390 | +0.733 | [+0.093, +1.450] | 9.080 | 12.59% |
| q0.50 | 64.674 | +1.017 | [+0.309, +1.813] | 10.060 | 14.29% |
| q0.25 | **64.848** | **+1.192** | **[+0.285, +2.112]** | 12.519 | 19.15% |
| q0.10 | 64.740 | +1.084 | [+0.090, +2.094] | 14.027 | 20.62% |

### Primary LR+UD

`outer_best_fraction` chooses q0.10 for LR and q0.25 for UD by other subjects only.

| method | acc | gain vs E4a equal | 95% CI | R10 loss vs E4a | P(loss<-5pp) |
|---|---:|---:|---:|---:|---:|
| q1.00 | 65.325 | 0.000 | [0, 0] | 0.000 | 0.00% |
| q0.75 | 66.010 | +0.685 | [+0.144, +1.283] | 9.368 | 13.72% |
| q0.50 | 66.551 | +1.226 | [+0.628, +1.914] | 9.981 | 14.31% |
| q0.25 | 67.082 | +1.758 | [+0.957, +2.610] | 11.529 | 16.65% |
| q0.10 | 67.681 | +2.357 | [+1.476, +3.298] | 12.152 | 17.06% |
| outer_best_fraction | **67.734** | **+2.409** | **[+1.532, +3.366]** | 11.266 | 16.25% |

Interpretation:

- Mean accuracy gain is large and statistically clear.
- Per-session dominance over E4a equal is false; lower-tail loss remains large.
- The right framing is risk-utility frontier, not safe selector.

## Main results: vs E4a outer best single

| method | condition | acc | gain vs outer best single | 95% CI | R10 loss | P(loss<-5pp) |
|---|---|---:|---:|---:|---:|---:|
| E4a equal | primary | 65.325 | +1.509 | [+0.808, +2.216] | 11.241 | 16.15% |
| E5b q0.50 | primary | 66.551 | +2.736 | [+2.065, +3.469] | 10.644 | 13.10% |
| E5b q0.25 | primary | 67.082 | +3.267 | [+2.452, +4.114] | 10.668 | 14.10% |
| E5b q0.10 | primary | 67.681 | +3.866 | [+2.987, +4.797] | 11.580 | 13.50% |
| E5b outer_best_fraction | primary | **67.734** | **+3.919** | **[+3.061, +4.818]** | **10.589** | **12.55%** |

This is a Pareto improvement over E4a equal when both are measured relative to the same
outer-best-single comparator:

```text
mean gain:   +1.509 -> +3.919
R10 loss:    11.241 -> 10.589
P<-5pp:      16.15% -> 12.55%
```

It still does not satisfy a strict safety gate, but it clearly moves the frontier.

## Source-only top-k ablation

To check whether this is merely dimensionality reduction, a source-only Fisher-style
feature selection baseline was added:

```text
source-only score =
  source class separation / source within-class variance
```

Primary result:

| method | acc | gain vs E4a equal | 95% CI | R10 loss vs E4a | P(loss<-5pp) |
|---|---:|---:|---:|---:|---:|
| source-only outer_best_fraction | 67.002 | +1.677 | [+0.959, +2.368] | 10.314 | 13.62% |
| longitudinal outer_best_fraction | **67.734** | **+2.409** | **[+1.532, +3.366]** | 11.266 | 16.25% |

Direct paired comparison:

| comparison | condition | mean diff | 95% CI |
|---|---|---:|---:|
| longitudinal outer_best_fraction - source-only outer_best_fraction | LR | +1.129 | [+0.389, +1.903] |
| longitudinal outer_best_fraction - source-only outer_best_fraction | UD | +0.351 | [-0.399, +1.080] |
| longitudinal outer_best_fraction - source-only outer_best_fraction | primary | **+0.732** | **[+0.199, +1.264]** |

Interpretation:

- A large part of the gain comes from dimensionality reduction / regularization.
- But the longitudinal drift term adds a statistically positive primary gain over source-only selection.
- The longitudinal contribution is strongest in LR and weaker in UD.

## Research judgment

### 守る

```text
source-side feature subspace selection is a real accuracy lever.
```

Evidence:

- primary +2.409pp over E4a equal
- primary +3.919pp over E4a outer-best-single
- q0.10 LR reaches 70.431%, close to the previous 4-feature oracle scale

### 守る, but carefully

```text
longitudinal class-stability contributes beyond source-only discriminative top-k.
```

Evidence:

- primary longitudinal - source-only direct paired diff = +0.732pp
- CI [+0.199, +1.264]

This is not huge, but it is nonzero and conceptually important.

### 撤回 / 禁止

```text
diagonal metric scaling improves LDA.
```

It was a no-op. Do not use this as a method.

```text
E5b solved safety.
```

False. Per-session losses remain nontrivial. The claim is frontier improvement, not no-harm adaptation.

## Why this is a better research direction

Previous failures said:

```text
target prefix label-free signals cannot reliably choose branch / gate / channel.
```

E5b says:

```text
do not choose sessions online.
Instead, use training subjects' longitudinal labels to remove unstable feature dimensions
before held-out subject deployment.
```

This is a different control structure:

```text
old:
  target session signal -> choose/gate adaptation

new:
  source-side longitudinal data -> learn stable subspace
  held-out target session -> no label, no router, only prefix EA
```

This is the first direction in the Stieger line that gives a large, honest accuracy gain
without target-session labels.

## Next required validation

Do not overclaim yet. The next experiments are mandatory:

```text
V1 random top-k baseline
  Is any 10-25% feature subset enough, or is the score meaningful?

V2 score ablation
  source-only separation
  source+target separation without drift penalty
  drift penalty only
  full longitudinal score

V3 second dataset
  Lee2019_MI or another multi-session MI dataset.

V4 risk-utility frontier plot
  x = mean gain vs outer best single
  y = R10 loss or P(loss<-5pp)
  points = E4a equal, source-only top-k, longitudinal top-k.
```

If V1/V2 confirm the score is meaningful and V3 preserves direction, this becomes the
main thesis story.
