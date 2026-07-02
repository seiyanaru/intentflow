# E42: final frozen rule comparison

Date: 2026-07-01

## One-line verdict

E42は、現時点の最終候補を支持した。

```text
regime-level family
+ source-validation modal beta=0.7
+ source-margin-calibrated interpolation
```

が、E36/E38/E41 local selectorより良い。

ただし、厳密には:

```text
beta=1 weighted Ridgeに対する改善は平均では一貫しているが、
paired CI はまだ跨ぐ。
```

なので、主張は:

```text
beta=0.7 interpolation improves the risk-utility frontier
```

であり、

```text
beta=0.7 is statistically decisively better than beta=1 everywhere
```

ではない。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/e42_final_rule_comparison.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e42_final_rule_comparison/`

Main files:

- `e42_final_records.csv`
- `e42_final_summary.csv`
- `e42_key_contrasts.csv`
- `e42_source_validation_support.csv`
- `e42_pass_fail.csv`

## Frozen final rule

E42 freezes:

| Regime | Family | beta |
|---|---|---:|
| Lee2019 sensorimotor20 | source-weighted | 0.7 |
| BNCI2014_001 full-source | LDA | 0.0 |
| BNCI2014_001 m8/class | longitudinal-weighted | 0.7 |

This is based on E41 source-validation support:

| Regime | Mode family | Mode beta | Counts |
|---|---|---:|---|
| Lee sensorimotor20 | source | 0.7 | source 49/54, beta=0.7 in 39/54 |
| BNCI full-source | LDA | 0.0 | LDA 9/9 |
| BNCI m8/class | longitudinal | 0.7 | longitudinal 8/9, beta=0.7 in 5/9 |

## Main final table: unit-level

| Regime | Method | Gain vs LDA | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---|---:|---:|---:|---:|---:|
| Lee | final beta=0.7 | **+2.500** | -6.250 | -6.875 | 7.4% | 24.1% |
| Lee | beta=1 weighted | +2.384 | -6.250 | -7.500 | 9.3% | 33.3% |
| Lee | beta=0.6 risk-first | +2.222 | **-4.625** | **-6.250** | **5.6%** | 25.9% |
| Lee | E38 binary fallback | +2.384 | -6.250 | -7.500 | 9.3% | 33.3% |
| Lee | E41 local selected | +2.153 | -5.438 | -6.667 | 7.4% | 27.8% |
| Lee | best hard top-k | +0.972 | -5.438 | -8.125 | 7.4% | 37.0% |
| BNCI full | final beta=0.7 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| BNCI full | best hard top-k | +0.448 | -1.774 | -2.419 | 0.0% | 33.3% |
| BNCI m8 | final beta=0.7 | **+2.470** | -5.645 | -6.201 | 6.3% | 26.0% |
| BNCI m8 | beta=1 weighted | +2.327 | -7.258 | -7.898 | 8.3% | 28.8% |
| BNCI m8 | beta=0.6 risk-first | +2.361 | **-4.839** | **-5.506** | **3.8%** | 26.4% |
| BNCI m8 | E38 binary fallback | +1.910 | -6.169 | -7.036 | 6.3% | 24.3% |
| BNCI m8 | E41 local selected | +2.184 | -6.452 | -6.646 | 6.3% | 27.4% |
| BNCI m8 | best hard top-k | +1.619 | -9.677 | -10.567 | 14.6% | 41.3% |

## Pass/fail

All predefined E42 checks passed:

| Criterion | Value | Pass |
|---|---:|---|
| Lee final beta=0.7 gain >= +2.4pp | +2.500 | yes |
| Lee final harm no worse than beta=1 | 0.0185 paired P(diff<-5) | yes |
| BNCI full final stays LDA | 0.000 | yes |
| BNCI m8 unit final gain >= +2.3pp | +2.470 | yes |
| BNCI m8 unit final P<-5 <= E38 binary fallback | 0.000 delta | yes |
| Lee source-validation modal beta is 0.7 | 0.7 | yes |
| BNCI m8 source-validation modal beta is 0.7 | 0.7 | yes |

## Key paired contrasts

### Lee

| Contrast | Mean diff | Interpretation |
|---|---:|---|
| final beta=0.7 vs LDA | +2.500 | strong gain |
| final beta=0.7 vs beta=1 | +0.116 | small, CI crosses zero |
| final beta=0.7 vs E38 binary | +0.116 | small, CI crosses zero |
| final beta=0.7 vs E41 local | +0.347 | positive, CI barely above zero |
| final beta=0.7 vs best hard top-k | +1.528 | clear mean gain |

### BNCI m8/class, unit-level

| Contrast | Mean diff | Interpretation |
|---|---:|---|
| final beta=0.7 vs LDA | +2.470 | strong gain |
| final beta=0.7 vs beta=1 | +0.143 | small, CI crosses zero |
| final beta=0.7 vs E38 binary | +0.560 | clear mean gain |
| final beta=0.7 vs E41 local | +0.286 | clear mean gain |
| final beta=0.7 vs best hard top-k | +0.851 | positive, but hard top-k comparison is descriptive |

## What E42 establishes

### Strongly supported

```text
source-anchor interpolation is better than binary fallback/local selection.
```

The important comparison is not only beta=1:

- E38 binary fallback is worse on BNCI m8 mean;
- E41 local selected is worse on both Lee and BNCI m8;
- hard top-k is weaker and riskier in the positive regimes.

### Moderately supported

```text
beta=0.7 is a good accuracy-first operating point.
```

Evidence:

- selected by source-validation mode in Lee and BNCI m8;
- best or near-best mean in both positive regimes;
- no worse than E38 binary fallback in P(gain<-5);
- better than beta=1 in mean and lower-tail metrics.

But the beta=0.7 vs beta=1 paired mean difference is small:

```text
Lee:     +0.116pp
BNCI m8: +0.143pp
```

So do not oversell the beta-specific superiority.

### Important risk-first alternative

```text
beta=0.6 is a useful risk-first operating point.
```

It gives up mean but improves lower-tail risk:

| Regime | beta=0.7 gain/P<-5 | beta=0.6 gain/P<-5 |
|---|---:|---:|
| Lee | +2.500 / 7.4% | +2.222 / 5.6% |
| BNCI m8 unit | +2.470 / 6.3% | +2.361 / 3.8% |

This supports a risk-utility frontier story rather than a single magic beta.

## What not to claim

Do not claim:

```text
E42 proves beta=0.7 is universally optimal.
```

Do not claim:

```text
the method is no-harm.
```

Even final beta=0.7 has:

```text
Lee P(gain<-5) = 7.4%
BNCI m8 unit P(gain<-5) = 6.3%
```

The honest claim is:

```text
The method improves the mean/risk tradeoff relative to binary fallback,
local source-validation selection, hard top-k, and pure weighted Ridge.
```

## Current paper-level framing

The strongest current contribution is:

```text
Source-validated reliability-weighted interpolation for zero-target-label
cross-session EEG-MI.
```

Core claims:

1. soft reliability-weighted Ridge improves mean accuracy in positive regimes;
2. source-validation is poor as an individual benefit predictor;
3. source-validation is useful for regime-level family and beta selection;
4. source-anchored interpolation gives a controllable risk-utility frontier;
5. local switching is repeatedly worse than regime-level fixation.

This is substantially stronger than the earlier selective-adaptation story.

## Next action

Do not build another selector.

Next, do one of:

1. E43: external/regime stress test:
   - add Stieger or another multi-session MI setting if feasible;
   - or sweep BNCI source size to show when LDA vs interpolation is selected.

2. E44: final figures/tables:
   - risk-utility frontier plot beta=0..1;
   - final method comparison table;
   - source-validation support table;
   - local-selection failure table.

If time is limited, prioritize E44 for a workshop-ready story.
