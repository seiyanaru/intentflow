# 次にやること(ゼミ向け): Base凍結を諦めて EA-aware 再学習へ

> **TL;DR**
> 1. 出力補正型OTTA(DC-Replay / CMC)は理論天井 +3.36pp で頭打ち、実測 +1.23pp。
> 2. 入力にEAを後付けすると壊滅(-32pp)。「Base凍結+入力を保つ」操作の限界が出揃った。
> 3. **次は train/test 両方にEAを入れて TCFormer を再学習**(アーキ不変、前処理だけ変更)。1被験者スモーク → 2a 9被験者 → 2b/HGD。

---

## 1. 研究の問い(変わっていない)

EEG-MI の cross-session / cross-subject 分布ずれに対し、**軽量制約(Active BCI、平均計算コスト低)下で精度を大きく上げる**手法を作る。

## 2. これまでの経路と発見

| Phase | 何をやったか | 結果 |
|---|---|---|
| 1 | Policy-SafeCommit / Replay-SafeCommit | +0.69pp, HSC=0/9。**安全だが補正力弱い** |
| 2 | DC-Replay(L1出力補正 / L2外部メモリ / L3 model-state commit) | 9subj single-seed best **+1.23pp**、seed込み **+0.26〜0.49pp**。**L3 commitは no-op**(commit 0/111/150で精度同一)|
| 解析 | oracle天井分析(2a/2b/HGD) | 出力補正(prior/logit_bias)天井 **+3.36pp**(2a, 後知恵上界)。top2天井 +11.84pp(2a)。Hybrid(別アーキTTT)は2a -1.15、HGD -13.66 |
| Phase3 (今回) | 入力整列(EA)を Base凍結に test-time後付け | s2で **-32pp 壊滅** |

## 3. 今回判明したこと(決定的・一本の原理)

**「これまで壊れなかった」と「天井が低い」は同じ根。**

| アプローチ | 入力(EEG)に触る? | Baseから見た入力分布 | Base凍結で動く? | 精度天井 |
|---|:---:|---|:---:|---|
| posterior / prior 補正 | NO | 学習時と同じ | ✓ | 低(+3.36) |
| prototype 補正 | NO | 学習時と同じ | ✓ | 低 |
| BN統計の test更新(AdaBN系) | NO | 学習時と同じ | ✓ | 限定的 |
| replay-gated commit | NO(内部統計のみ) | 学習時と同じ | ✓ | L3 = no-op |
| **EA(入力白色化)** | **YES** | **学習時と別物** | **✗(-32pp)** | 高(再学習なら) |

> **原理**: Base TCFormer は学習時の入力分布で動くことを前提に固まっている。
> - 入力を保てば安全だが、できる介入が小さい → 天井 +3.36pp。
> - 入力を変えるなら、学習も変える必要がある(EAは前処理、train/test両方に適用が前提)。

これで oracle・実測・EAスモークが**同じ原理から一致して**「Base凍結のままでは頭打ち」を示した。

## 4. 次にやること:Step A — EA-aware 再学習

### 設計

- **TCFormer 本体アーキは不変**(Hybridのように別アーキを訓練しない)。
- **train/test 両方に session-EA 前処理を入れる**(session毎に共分散 R を計算し、`R^{-1/2}` で白色化)。
- 既存pipelineでそのまま訓練・評価(**新規モデル不要、最小変更**)。

### 実装の最小変更

| 何を | どこ |
|---|---|
| EA前処理を追加(session毎にR、白色化) | `intentflow/offline/datamodules/` の BCICIV2a に flag追加 |
| 新flag/configで派生(既存sourceは非破壊) | `preprocessing.ea: true` |
| 学習・評価 | 既存 `train_pipeline.py` のまま |

## 5. なぜこれをやるか(Why、根拠つき)

### (a) データが指している

- **top2 oracle天井 +11.84pp** (2a):「正解は2位以内に入っているのに1位にできない」trialが 11.84pp ある。
- **L1(prior/logit-bias)では +3.36pp で頭打ち**。`p(誤1位)−p(正解2位)` 中央値0.334、確信誤りが本物(ECE 5.6%・T*=0.89 で過剰自信ではない)。
- → 確信誤りを覆すには**入力側の介入(EA)or 特徴空間の介入**が必要。前者がEA-aware 再学習。

### (b) 外部相場の整合

- **T-TIME**(EAベース online TL): binary MI で **+2.9〜6.1pp**(Siyang Li et al., 2024)。
- **Wimpff 2024**(alignment + BN + EM): 2a continual cross-subject で約 +2pp。
- **dual-stage alignment + self-sup** (2025): MI平均 **+3.6pp**。
- → 大gainは feature update 側、特に **alignment** が共通の土台。

### (c) EAの正しい使い方を踏襲

EAは「train も test も同じ整列で共通空間に揃える」前処理。Base凍結に後付けすると壊れる(今回 -32pp で実証)。**train/test 両方に適用するのが本来の使い方**。

### (d) Hybrid失敗の再現を避ける設計

Hybrid失敗の3要素は (1) 別アーキを訓練、(2) alignment 無し、(3) 凍結放棄。今回踏むのは (3) だけ:

- (1) **アーキ不変**(TCFormerのまま)
- (2) **EAを必ず入れる**
- (3) 凍結を諦めるが、再学習は通常の学習(test時に重みを動かす TTT は入れない)

## 6. 想定される結果と棄却条件

| | 内容 |
|---|---|
| 成功条件 | 2a multi-seed で source(82.72)に対し**有意な mean gain**(期待 +2〜5pp)、worst-drop が既存baseline 以下 |
| 棄却条件 | source 以下、または 1人以上 大きく壊す(worst-drop > 既存baseline) |
| 罠 | z-scale と EA の順序、TCFormer の Multi-Kernel Conv との整合、session毎 R の計算粒度。**実装中に出てくる具体問題は、その場で集中砲火の Deep Research に投げる** |

## 7. 何を見るか(評価)

- mean acc, kappa, **per-subject delta**, **worst-subject drop**, HSC@0.5/1.0/2.0
- **multi-seed 必須**(single-seed は ノイズと判別不能、計画v1 通り)
- latency / 平均適応コスト(EAは train-time のみ、test時は素のforward → 軽量)
- (Step Aが効けば次に) AdaBN / triggered tiny update を追加検討

## 8. マイルストーン

1. **BCICIV2a datamodule に session-EA前処理を追加**(最小変更、既存非破壊)
2. **s2 スモーク学習**(EA-Base を訓練、shape確認、EA効果の初期値)
3. **2a 9被験者 × multi-seed**で再学習・評価
4. **2b / HGD** で同様(汎用性確認、HGDは天井小なので過剰期待しない)
5. 結果次第で Step B(test時の minimal適応、AdaBN/triggered)へ

## 9. ゼミQAに備える補足

- **なぜBase凍結に固執していたか**: 壊れない+軽量だから。oracle と EAスモークで「凍結のままでは精度が上がらない」と二重に確定。
- **なぜ EA か**: cross-session EEG の定番、T-TIME の土台、+2〜6pp の実証あり、実装変更が最小(前処理だけ)。
- **なぜ別モデルを使わないか**: Base TCFormer は 2a で 84.67%と強い backbone。アーキを変えると Hybrid失敗(2a -1.15、HGD -13.66)を再現する。
- **軽量制約は満たすか**: EAは train時のみ R を計算、test時は `R^{-1/2}·x` の行列乗算1回。**平均計算コスト低を維持**。

## 10. 参照(自リポジトリ内、ゼミでURL/パス出せる)

- 計画 v1: [260526_research_plan_v1.md](260526_research_plan_v1.md)
- oracle天井 + top2 + calibration 統合分析: [260526_oracle_ceiling_analysis.md](260526_oracle_ceiling_analysis.md)
- Deep Research 3本: deep-research-report (1)〜(3).md
- 実装スクリプト(EA scaffold): [eval_ea.py](../../intentflow/offline/scripts/analysis/eval_ea.py)
- Hybrid失敗(README §4 / README_Hybrid.md)
