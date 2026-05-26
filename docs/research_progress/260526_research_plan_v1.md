# 研究計画 v1 — Accuracy-first: Alignment-first Selective Minimal Feature Adaptation (EEG-MI)

> v0([260526_research_plan_v0.md](260526_research_plan_v0.md))は safety/理解主導だった。**v1で精度主軸に転換**。根拠: 内部分析([260526_oracle_ceiling](260526_oracle_ceiling/260526_oracle_ceiling_analysis.md))+ Deep Research 3本([deep-research-report (1)〜(3)](deep-research-report%20(3).md))+ TCFormer_Hybrid失敗分析。

---

## 0. 主張(候補)

> **EEG-MIのtop2 slack(2a +11.84pp)を、EA条件付き・難例限定のminimal feature adaptationで回収し、Base TCFormer凍結+平均軽量を保ちつつ精度を大きく上げる。**

## 1. 主軸と制約

- **主軸: 精度向上**(「精度が上がらなきゃ意味がない」)。
- 制約: ①軽量(平均計算コストを低く、難例だけ更新)②大きく壊さない(worst-drop制御)。

## 2. 確定した地図(出発点)

| 事実 | 根拠 |
|---|---|
| 出力補正(prior/温度/confidence)では精度上がらない。EA/特徴適応が唯一の道 | oracle天井: prior +3.36, top2 +11.84(2a)。calibration: 確信誤りは本物 |
| L3 model-state commit は no-op | l3_diagnostics.csv |
| **triggered feature adaptation は repo で失敗済み**(Hybrid: 2a -1.15, HGD -13.66) | README §4。差分は **EA前段 + Base凍結** が無かったこと |
| 大gainの外部相場は feature update 側 | T-TIME +2.9〜6.1pp、dual-stage +3.6pp |

## 3. リサーチクエスチョン

- **RQ1**: EA/RA + AdaBN だけで、出力補正天井(+3.36)を超えられるか。
- **RQ2**: 難例限定の minimal feature adaptation で top2 slack(+11.84)をどこまで回収できるか。
- **RQ3**: 「EA + Base凍結 + 難例更新」は、TCFormer_Hybrid(EA無し・別アーキTTT)より良いか(=失敗の原因がEA/凍結欠如だったか)。
- **RQ4**: 平均計算コスト(latency)を Active BCI目標(300–500ms)内に保てるか。

## 4. 仮説(1文、棄却可能)

> EA(整列)+ Base凍結 + 難例限定 tiny更新 で、2aで出力補正天井(+3.36)を超え、top2 slack(+11.84)の一部を回収する。HGDは余地小で小gain。

棄却条件: EA+selective が multi-seed で source を有意に超えない、または Hybrid と同様に強baselineで壊す。

## 5. 必須設計(Hybrid失敗の回避)

1. **EA(特徴整列)を前段に**(Hybridに無かった、T3A崩壊の原因)。
2. **Base TCFormer(84.67%)を凍結**(Hybridは別アーキ訓練で強い特徴を捨てた)。
3. **強baseline(HGD)では適応を最小/オフ**(余地小、Hybrid -13.66)。
4. triggered更新は補助。**単独に頼らない**(Hybridが既に失敗)。

## 6. 最初の実験(3段、同一backbone上でceiling切り分け)

| Step | 内容 | 目的 | コスト |
|---|---|---|---|
| 1 | **EA/RA + AdaBN only**(Base凍結、再学習不要・既存checkpoint) | EA単体で出力補正天井を超えるか | 軽(test時変換+BN) |
| 2 | **uncertainty-triggered tiny update**(adapter/head/BN affine)vs always-update | feature adaptationの回収分とtriggerの効果 | 中(難例のみBP) |
| 3 | **top2-aware local reranker**(frozen特徴) | no-BPでの回収分 | 軽(近傍探索) |

## 7. 評価

- **データセット**: 2a(主)、2b、HGD。OpenBMI/Lee2019は余力で。
- **seed**: multi-seed必須(single-seed gainはノイズと判別不能)。検定は subject×seed単位。
- **metrics**: mean acc, kappa, per-subject delta, **worst-subject drop**, HSC@0.5/1.0/2.0, **latency/trial(post-update含む)**, 平均適応コスト(更新発火率)。
- **baseline**: source_only / EA-only / Wimpff-OTTA / T-TIME / BFT-style no-BP / **TCFormer_Hybrid(失敗例として明示)**。
- **達成基準**: 2aで EA+selective が source を multi-seed有意に上回り、worst-drop が既存baseline以下、平均latency が目標内。

## 8. 競合と差別化

| 競合 | 軸 | 差別化 |
|---|---|---|
| T-TIME | 全パラ更新+batch8、+2.9〜6.1pp | 重い。こちらは Base凍結+難例のみで平均軽量 |
| BFT 2026 | no-BP transformation aggregation | こちらは EA+selective feature adaptation(別機構) |
| TCFormer_Hybrid | EA無し別アーキTTT、Baseに負け | EA+Base凍結が差分。失敗の原因を実証的に示す |

## 9. 罠(明記)

- triggered更新"だけ"で Hybrid を再現する(EA/凍結を欠く)。
- 2a single-seed だけで勝ったと言う。
- latency を ensemble/post-update 抜きで主張。
- 強baseline(HGD)で適応して壊す。

## 10. マイルストーン

1. EA/RA + AdaBN 実装(Step1、既存checkpoint使用)
2. Step1測定(2a/2b/HGD、multi-seed)→ EA単体ceiling確定
3. Step2/3(triggered tiny update / top2 reranker)
4. 評価系整備(worst-drop, latency, 適応発火率)
5. 比較対象(Wimpff/T-TIME/BFT-style/Hybrid)を同一プロトコルで
6. 論文化判断(主貢献の確定)

---

## 次の一歩(最小)

**Step1: EA/RA + AdaBN only を Base凍結TCFormerに test-time適用**(再学習不要、`sources/sN/checkpoints/` 使用)。EA変種(EA / incremental EA / Riemannian)を選び、2a で source比を測る。これが「Hybridに無かったEA単体」の効果測定で、以降の設計の土台。
