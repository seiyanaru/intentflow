# 2026-06-01 Adaptive EA selector v0

## Goal

Full EA固定では被験者ごとに効き方が割れたため、ラベルなし統計で source / full EA / shrink EA を選ぶ selector v0 を実装した。

実装:

- `intentflow/offline/scripts/analysis/eval_adaptive_ea_selector.py`
- 出力JSON: `docs/research_progress/260601_adaptive_ea_selector_v0.json`

## Rule

Source logitsを基準に、full EA候補を先に評価する。

Full EAを拒否する条件:

- predicted class dominance > 0.42
- source比で prior KL 増加 > 0.05
- entropy増加 > 0.11 かつ confidence低下 > 0.05

Full EAが通ればfull EAを採用。拒否された場合のみ `shrinkage=0.1` を試し、これも拒否ならsourceへ戻す。`partial power=0.5, shrink=0.1` は二次fallback。

重要: accuracyは選択後の評価だけに使い、選択には使っていない。

## Result

| subj | selected | selected | source | full EA | delta vs source |
|---:|---|---:|---:|---:|---:|
| S1 | full EA | 87.85 | 85.76 | 87.85 | +2.08 |
| S2 | full EA | 75.69 | 72.57 | 75.69 | +3.12 |
| S3 | full EA | 92.36 | 93.06 | 92.36 | -0.69 |
| S4 | full EA | 83.33 | 81.94 | 83.33 | +1.39 |
| S5 | shrink 0.1 | 77.78 | 77.43 | 72.57 | +0.35 |
| S6 | full EA | 73.61 | 70.49 | 73.61 | +3.12 |
| S7 | shrink 0.1 | 93.06 | 90.97 | 83.68 | +2.08 |
| S8 | full EA | 90.28 | 84.38 | 90.28 | +5.90 |
| S9 | source | 87.85 | 87.85 | 88.19 | +0.00 |

Mean:

- source: 82.72
- full EA: 83.06
- adaptive selector v0: 84.65
- delta vs source: +1.93pp
- delta vs full EA: +1.58pp

## Interpretation

This is the first positive evidence for Reliability-Aware Alignment.

The selector recovered the two major full-EA failures:

- S5: full EA 72.57 -> shrink 77.78
- S7: full EA 83.68 -> shrink 93.06

It preserved major full-EA successes:

- S2: full EA 75.69 was kept; shrink candidates were rejected because prediction prior collapsed.
- S8: full EA 90.28 was kept.

The remaining miss is S3: full EA was selected but source was 0.69pp better. This is small, but it shows v0 cannot detect all harmless/negative EA cases.

S9 was conservatively sent to source because full EA increased entropy and reduced confidence. Full EA was actually +0.34pp, so this is an acceptable but slightly conservative rejection.

## Warning

This is not yet a publishable adaptive method.

Reasons:

- Thresholds were chosen after observing S2/S5/S7/S8 behavior, so overfitting risk is real.
- Current evaluation selects among already trained candidate models. The online implementation must define when candidate logits are available and how often the decision is updated.
- Covariance diagnostics motivated the rule, but v0 selection itself is mostly prediction-statistic based. v1 should include explicit covariance reliability.

## Next v1

Add explicit covariance reliability:

```text
R = mean(x x^T / T)
R_lambda = (1-lambda) R + lambda * tr(R)/C * I
```

Candidate set:

- source/no EA
- full EA: lambda=0
- shrink EA: lambda=0.1

Reliability features:

- predicted prior KL
- prediction dominance
- entropy/confidence shift from source
- test covariance condition number
- channel variance CV
- train-test covariance distance when train reference is available

The valid claim should be: a label-free reliability selector improves mean accuracy while reducing worst-case harm versus fixed full EA.
