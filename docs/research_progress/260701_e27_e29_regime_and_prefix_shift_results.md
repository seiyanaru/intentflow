# E27/E29: unified regime table and Lee2019 target-prefix shift diagnostic

Date: 2026-07-01

## 結論

E27で、ここまでの結果はかなり整理された。

> source-side tangent subspace selection は本物だが、効く条件は dataset name や task family ではなく、feature/statistical regime に依存する。

E29で、次の手法候補だった target-prefix shift は一部signalを持つが、Lee2019 nested failure の主因である `longitudinal_q0p10` vs `source_only_q0p25` の取り違えは説明できなかった。

したがって、現時点の判断は以下。

```text
守る:
  regime-aware fixed compact subspace policy

弱く守る:
  target-prefix shift as a harm/risk diagnostic

まだ守れない:
  target-prefix shiftでsubject-wise candidate selectionを改善する
```

## E27 unified regime table

Script:

`intentflow/offline/scripts/analysis/unified_regime_table_e27.py`

Output:

`intentflow/offline/results/research_outputs/260701_e27_unified_regime_table/`

Main files:

- `unified_regime_table.csv`
- `lee_source_size_trend_summary.csv`
- `verdict.json`

### Main table

| Dataset/regime | Policy | p/n | Gain | q05 | P(gain < -5pp) | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Stieger primary | condition-known LR q0.05 / UD q0.20 | n/a ensemble | +2.719 | -10.606 | 14.1% | longitudinal stability helps in multi-session/source-scarce ensemble |
| Stieger primary | global longitudinal q0.10 | n/a ensemble | +2.357 | -11.111 | 17.1% | robust global Stieger policy |
| Lee2019 all62 | fixed source_only q0.25 | 19.53 | +2.176 | -6.250 | 7.4% | broad all-channel denoising |
| Lee2019 all62 | nested source-validation | 19.53 | +1.366 | -11.688 | 16.7% | fails vs fixed q0.25 |
| Lee2019 sensorimotor20 | best mean source_only q0.10 | 2.10 | +1.389 | -7.938 | 20.4% | compact selection loses robust signal after channel restriction |
| BNCI full-source | best mean longitudinal q0.70 | 1.76 | +0.448 | -1.774 | 0.0% | low-dimensional/source-richer regime, full tangent already strong |
| BNCI m8/class | best mean longitudinal q0.10 | 15.81 | +1.277 | -2.177 | 0.0% | compact selection becomes mildly useful when p/n is high |
| BNCI m8/class | stable longitudinal q0.70 | 15.81 | +0.468 | -0.570 | 0.0% | smaller but stable positive effect |

### Updated interpretation

The old story:

```text
Lee-like LR -> source_only
Stieger-like multi-session -> longitudinal
```

is too simple.

The better story:

```text
high-dimensional/source-scarce regimes benefit from source-side tangent subspace selection.
The best score family/compactness depends on whether the useful signal is broad source-discriminative denoising or longitudinally stable compact cores.
```

E19 matters here:

- Lee `source_only_q0p25` is broadly strong across source sizes.
- Lee `longitudinal_q0p10` strengthens when source is scarce, but has worse tail risk.
- BNCI full-source is low p/n and aggressive compactness hurts.
- BNCI source-scarce simulation makes compactness mildly useful again.

## E29 target-prefix feature-shift diagnostic

Script:

`intentflow/offline/scripts/analysis/lee2019_target_prefix_shift_e29.py`

Output:

`intentflow/offline/results/research_outputs/260701_lee2019_target_prefix_shift_e29/`

Main files:

- `candidate_prefix_shift_metrics.csv`
- `prefix_metric_predictability.csv`
- `long_vs_source_prefix_shift.csv`
- `long_vs_source_shift_predictability.csv`
- `summary.json`

### Question

Can unlabeled target prefix explain which candidate will fail?

Used only:

- held subject source session features and labels
- held subject target prefix features
- no target prefix labels
- no target eval labels for computing metrics

Outer labels are used only after the fact for diagnostic correlations.

### Candidate-level result

Across non-full candidate pairs:

| Prefix metric | Spearman metric vs outer gain | AUROC for outer harm |
|---|---:|---:|
| prefix_abs_z_shift_mean | -0.125 | 0.604 |
| prefix_abs_z_shift_q90 | -0.099 | 0.595 |
| prefix_log_var_ratio_abs_mean | +0.042 | 0.507 |
| prefix_log_var_ratio_abs_q90 | +0.025 | 0.526 |
| prefix_energy_z_mean | -0.206 | **0.646** |
| prefix_std_z_mean | -0.230 | 0.587 |

Interpretation:

```text
There is a weak-to-moderate harm diagnostic signal.
High prefix energy/shift tends to indicate risk.
But the ranking signal is not strong enough to solve candidate choice.
```

### Longitudinal vs source_only failure result

The critical failure was:

```text
nested sometimes chooses longitudinal_q0p10,
but source_only_q0p25 would have been better.
```

For all subjects:

| Delta metric | Spearman with outer long-source gain | AUROC for long harm |
|---|---:|---:|
| delta prefix_abs_z_shift mean | -0.107 | 0.526 |
| delta log-var-ratio mean | -0.073 | 0.433 |
| delta prefix_energy_z mean | +0.355 | 0.330 |
| delta prefix_std_z mean | +0.320 | 0.410 |

This does **not** support the hoped-for mechanism.

For the 15 subjects where nested chose `longitudinal_q0p10`:

| Metric | Mean | 95% CI |
|---|---:|---:|
| outer long - source gain | -2.917 | [-6.000, -0.250] |
| delta prefix_abs_z_shift mean | -0.003 | [-0.013, +0.007] |
| delta log-var-ratio mean | -0.007 | [-0.023, +0.009] |

So, the target prefix metrics do not say:

```text
longitudinal_q0p10 looks more shifted than source_only_q0p25
```

when nested makes the bad choice.

## Judgment

### 守る

Target-prefix feature metrics may be useful as a weak harm diagnostic.

Evidence:

- candidate-level AUROC up to 0.646 for outer harm
- negative Spearman between prefix energy/std and outer gain

### 撤回 / 弱める

Target-prefix feature shift, at least in this simple form, does not explain the Lee nested failure.

Evidence:

- long-vs-source delta metrics do not predict long-vs-source outer gain
- nested-chose-long failures show almost zero prefix-shift difference between long and source subsets

### 次にやるべきこと

Do **not** immediately claim target-prefix shift solves selection.

Next experiment should be a cheap model-level pilot:

```text
E30:
  weighted logistic / ridge classifier
  feature weights = source discriminability with optional target-prefix risk penalty
```

Why still worth trying?

E29 may fail at candidate-level hard subset selection, but soft feature regularization could still use weak per-feature risk signal without needing to choose between hard candidates.

Decision criterion for E30:

- Beat LDA source_only_q0p25 by >= +0.5pp, or
- Match its mean while reducing P(gain < -5pp), or
- Clearly improve the 15 nested-long failure subjects.

If E30 fails, stop target-prefix method development and return to the regime-aware fixed-policy story.

