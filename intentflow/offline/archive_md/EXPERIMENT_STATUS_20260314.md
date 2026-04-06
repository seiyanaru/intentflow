# 実験ステータス — 2026-03-14

## 現在実行中の実験

### OTTA SAL Threshold Sweep
- **開始**: 2026-03-14 14:32 JST
- **推定完了**: 2026-03-15 朝（約14-20時間）
- **PID**: 937494 (timeout wrapper) → 937496 (train_pipeline)
- **GPU**: NVIDIA GeForce RTX 2080 Ti (11GB)

| SAL値 | 状態 | 結果 |
|-------|------|------|
| 0.35 | **実行中** (S1 Epoch 125/500) | 待ち |
| 0.40 | 待機中 | — |
| 0.45 | 待機中 | — |
| 0.50 | 待機中 | — |

**進捗確認コマンド**:
```bash
tail -3 results/otta_sal_sweep_sal035_s0_20260314_143230.log
ps aux | grep train_pipeline | grep -v grep
```

**結果確認コマンド（完了後）**:
```bash
cat results/launch_otta_sal_sweep_20260314_143229.log | grep "Results Summary" -A 10
```

---

## 完了済み実験の全体像

### 1. Clean Baseline (TCFormer, BCIC-IV 2a, 4クラスMI)

| 指標 | Seed 0 | 5-Seed平均 | 論文目標 |
|------|--------|-----------|---------|
| Accuracy | 82.79% | 82.91% | 84.79% |
| Std | ±8.05% | ±8.34% | — |
| Gap | -2.00% | -1.88% | — |

**Subject別詳細（Seed 0）**:

| Subject | Acc | 評価 |
|---------|-----|------|
| S7 | 92.71% | Top |
| S3 | 92.01% | Top |
| S9 | 88.89% | 良好 |
| S1 | 87.50% | 良好 |
| S8 | 84.72% | 平均 |
| S4 | 81.25% | やや低 |
| S5 | 76.74% | 低 |
| S2 | 71.88% | ボトルネック |
| S6 | 69.44% | 最低 |

### 2. OTTA（現行閾値: SAL=0.5）

- **結果**: 82.99%（Baseline +0.20%）
- **問題**: 適応率が低すぎる（16.7-44.8%）
- **原因**: SAL=0.5閾値が保守的すぎ、大半のテストサンプルが適応拒否される

### 3. Window Timing実験

- **0.0-4.0s（現行）**: 82.79% → **正解**
- **0.5-4.5s（神経科学的標準）**: 72.07% → **大幅悪化**
- **考察**: load_bcic4.pyのエポック切り出しタイミングが0.0sで既にcue onset後を捉えている可能性

### 4. 3/13 SAL Sweep（失敗・再実行中）

| 試行 | エラー | 原因 |
|------|--------|------|
| 114030 | PermissionError | MNE lock file権限 |
| 114120 | CPUAccelerator | conda未活性/GPU未検出 |
| 114543 | ハング（S2 Epoch 3で26h停止） | num_workers過多によるデッドロック推定 |

**3/14修正内容**:
1. MNE lock file削除
2. OTTA config に `num_workers: 4` 追加
3. sweep script に timeout/エラーハンドリング追加

---

## 修正履歴

| 日付 | 修正 | 影響 |
|------|------|------|
| 3/6 | EarlyStopping除去 | +1.81% (80.98→82.79%) |
| 3/6 | Validation leakage修正 | SD: 12.05→8.05% |
| 3/14 | num_workers: 8→4 | デッドロック防止 |
| 3/14 | sweep timeout追加 | ハング時5h自動kill |

---

## 次のアクション（SAL sweep結果待ち後）

### SAL sweep結果に応じた分岐

**Case A: SAL緩和で有意改善あり（+1%以上）**
1. 最良SAL値で5-seed検証
2. pmax閾値の同時最適化（0.7 → {0.5, 0.6, 0.7}）
3. 論文実験セクション執筆

**Case B: SAL緩和でも改善微小（<1%）**
1. strict_tri_lock=false の効果確認（tri-lock → pmax+SALのdual-lock）
2. Soft gating（sigmoid重み付け）への切り替え検討
3. OTTA戦略の根本見直し（Tent/EATA/TTT）

**Case C: SAL緩和で悪化**
1. OTTAアプローチ自体の再考
2. Subject-adaptive threshold（被験者ごとのキャリブレーション）
3. BN統計量のみの軽量適応に切り替え

---

## 研究の新規性と論文戦略に関する所見

### 現状の新規性の評価

**主張できる新規性（現時点）**:
1. **Pmax-SAL Gated OTTA**: 確信度（pmax）とソース整合性（SAL）の二重ゲーティングによる選択的テスト時適応
2. **Energy-based OOD detection との統合**: tri-lock条件（pmax + SAL + energy）による安全な適応
3. **Neuro-beta gating**: 脳波のチャネル注意重みからモーター領域活性度を推定し、閾値を動的調整

**新規性の弱点（率直な評価）**:
- Pmax閾値ベースのゲーティングは TTA/OTTA文献で一般的（EATA, SAR等で類似手法あり）
- SALは本質的にprototype距離の亜種であり、既存のTent+filtering手法と差別化が弱い
- 現時点の実験結果（+0.20%）では新規性を支える実験的エビデンスが不足
- BCIC-IV 2aのみでの評価は査読で「限定的」と指摘されるリスク高

### 論文を通すための戦略提案

#### 提案1: ストーリーを「安全なBCI適応」に振る（最推奨）

**理由**: 精度改善幅（+0.20%）で勝負するのは厳しい。代わりに**安全性**をメインストーリーにする。

- BCIは医療・支援技術であり、誤分類のリスクが高い
- 既存OTTA手法（Tent, CoTTA等）は無条件適応 → 脳波のノイズ/アーティファクトで崩壊リスク
- 提案手法は「適応しない方が安全」な場面を検出できる
- **評価軸を変える**: 精度だけでなく「negative transfer率」「worst-case被験者の安全性」で評価
- Subject 2での-7.99%悪化（無条件OTTA）vs 提案手法の制御を示す

**追加実験**:
- Tent/CoTTA/EATAとの比較で「negative transferの抑制率」を示す
- ノイズ注入実験（筋電アーティファクト混入時のrobustness）
- Abstain率 vs 精度のトレードオフカーブ（ROC的な分析）

#### 提案2: Online/Streaming設定での実証を加える

**理由**: オフライン結果だけでは「なぜOTTAが必要か」の説得力が弱い。

- `intentflow/online/` にストリーミング推論基盤がある → 活用
- セッション間ドリフト（session_T → session_E）をリアルタイムで適応する実験
- レイテンシ制約下での適応（5.67ms response time は十分高速）
- **デモ動画**: Unicornデバイスでのリアルタイムデモは査読者へのインパクト大

#### 提案3: データセットを追加する（必須に近い）

**単一データセットでの評価はNeurIPS/AAAI/ICML系では厳しい。以下を追加推奨**:
- **BCIC-IV 2b** (2クラスMI): 既にパイプライン対応済み
- **HGD (High Gamma Dataset)**: configに存在、比較的容易に追加
- **OpenBMI** or **Lee2019**: 大規模（54被験者）で統計的信頼性が高い
- **physionet MI**: 公開データで再現性を保証

#### 提案4: Ablation Studyを体系化する

**査読で必ず聞かれる質問への事前回答**:
- pmax単体 vs SAL単体 vs pmax+SAL vs tri-lock → 各ゲートの貢献度
- energy gatingの有無 → OOD検出の効果
- neuro-betaの有無 → 神経科学的知見の価値
- 適応層の選択（BNのみ vs 全層 vs TCN headのみ）

#### 提案5: 理論的根拠を補強する

- SAL閾値がなぜ機能するかの理論的分析（feature space上のprototypeからの距離と適応リスクの関係）
- Energy scoreのcalibration特性の分析（source分布とtarget分布の乖離度推定）
- Neuro-betaの神経科学的妥当性（mu/betaリズムとモーター領域活性の対応）

### 投稿先の検討

| 会議/ジャーナル | 適合度 | 理由 |
|----------------|--------|------|
| **IEEE TNSRE** | ★★★★★ | BCI特化、安全性ストーリーとの親和性高 |
| **Journal of Neural Engineering** | ★★★★★ | BCI+適応の実用研究向き |
| **NeurIPS (Workshop)** | ★★★★☆ | Brain-Computer Interface workshopに投稿 |
| **AAAI/IJCAI** | ★★★☆☆ | データセット追加+ablation必須 |
| **ICML/NeurIPS (main)** | ★★☆☆☆ | 理論的貢献または大規模実証が追加で必要 |
| **EMBC/BCI Conference** | ★★★★★ | 実用BCI研究として最も適切 |

### 最も効果的な次の一手（研究インパクト最大化の観点）

1. **SAL sweep結果を待つ**（今ここ）
2. **Tent/EATA/CoTTAとの比較実験を追加**（安全性での差別化の実証）
3. **BCIC-IV 2bでの追加評価**（既存パイプラインで即実行可能）
4. **Negative transfer分析**（Subject 2, 6での詳細分析）
5. **論文ドラフト着手**（安全性ストーリーで構成）
