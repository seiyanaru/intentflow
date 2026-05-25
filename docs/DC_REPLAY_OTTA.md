# DC-Replay OTTA 完全ガイド

**これを読めば DC-Replay モデルが何をしているか全部わかる、を目的にしたまとめ。**

対象コード:
- [models/tcformer_deferred_commit_replay_otta.py](../intentflow/offline/models/tcformer_deferred_commit_replay_otta.py) — Lightning ラッパー
- [models/deferred_commit_replay_otta.py](../intentflow/offline/models/deferred_commit_replay_otta.py) — DC 本体(L1/L2/L3 ポリシー)
- [models/dc_replay_memory.py](../intentflow/offline/models/dc_replay_memory.py) — 外部メモリ(L2 の貯蔵庫)
- [models/replay_safe_commit_otta.py](../intentflow/offline/models/replay_safe_commit_otta.py) — リプレイ検証 SafeCommit(L3 の tier-2 ゲート)
- [models/policy_safe_commit_otta.py](../intentflow/offline/models/policy_safe_commit_otta.py) — 基盤(状態抽出・オペレータ・tier-1 ガード)
- 設定例: [configs/tcformer_deferred_commit_replay_otta/](../intentflow/offline/configs/tcformer_deferred_commit_replay_otta/)

---

## 0. 一言で

**DC = Deferred Commit(遅延コミット)。** テスト時にオンライン適応するが、「モデルの重みを書き換える適応」を最後まで遅延させる。適応を **可逆性の高い順に3段階** に分け、不可逆な操作ほど稀に・強い証拠を要求して実行する。

> OTTA は擬似ラベルが間違いやすく、重みを直接動かすと collapse / drift でベースラインを壊す。DC はそのリスクを「階層化 + ゲート」で抑える設計。

---

## 1. アーキテクチャ(継承の積み上げ)

```
nn.Module
└─ PolicySafeCommitOTTA           基盤:状態抽出 / オペレータ群 / tier-1 即時ガード / RuleBasedPolicy
   └─ ReplayPolicySafeCommitOTTA  +リプレイバッファ +tier-2 リプレイ検証(ReplaySafeCommit)
      └─ DeferredCommitReplayOTTA +L1予測補正 +L2外部メモリ採用 +L3遅延コミット
                                   (= DC-Replay の本体)
```

- **TCFormer 本体は常に `eval()` + `no_grad`**([forward_once](../intentflow/offline/models/policy_safe_commit_otta.py#L921-L936))。バックプロップは一切しない。適応は「本体の外側にある制御層(sidecar)」が担う。
- だから DC が触れるのは次の3つの軽量な状態だけ:
  - `logit_bias`(クラス別 logit バイアス、`register_buffer`)
  - `target_prototypes`(クラスごとの特徴プロトタイプ)
  - BN の running stats(※ DC の既定 config では L3 で BN は使わない。後述)

`TCFormerDeferredCommitReplayOTTA` は Lightning モジュールで、`on_test_start` で DC 本体を構築し source 統計を計算、`test_step` で1試行ずつ `dc_replay_otta(x)` を呼ぶ([test_step](../intentflow/offline/models/tcformer_deferred_commit_replay_otta.py#L231-L262))。**テストは `test_batch_size: 1`**(1試行=オンライン1ステップ)。

---

## 2. 1試行の処理フロー(これが全体像)

[DeferredCommitReplayOTTA.forward](../intentflow/offline/models/deferred_commit_replay_otta.py#L508-L610) が毎試行やること:

```
入力 x (1, C, T)
  │
  ├─[forward_once] TCFormer本体で classifier_logits を計算
  │   └─ compose_logits: classifier_logits + logit_bias、必要なら proto fusion(α)
  │        ↓
  ├─【L1】_apply_prediction_correction:
  │     source事前分布 と メモリ事前分布 の対数比で logits をシフト(重みは触らない)
  │        ↓ corrected_logits を「その試行の出力 logits」に採用
  │
  ├─ state_extractor.extract: pmax / margin / entropy / sal / energy / neuro_score 等を計算
  │        ↓
  ├─【L3 判定】_compute_drift_diagnostics → _propose_dc_commit_candidates
  │     メモリから prior drift / proto drift を測り、コミット候補を立てるか決める
  │        ↓
  ├─【L3 実行】_run_commit_policy → SafeCommit.run
  │     候補を仮適用し tier-1(即時)+ tier-2(リプレイ)ガードを通過したものだけ commit
  │        ↓ final_state(コミット後)で pred / pmax を確定
  │
  ├─【L2】_maybe_admit_to_replay:
  │     この試行が信頼できるなら外部メモリに採用(重みは触らない)
  │        ↓
  └─ record にすべて記録(dc_level_reached, dc_commit_reason, …)→ trial_logs に追加
```

> 注:**L1 → L3 → L2 の順で実行される**(L2 採用は「コミット後の最終状態」を使って最後に判断するため)。番号は「やる順番」ではなく「重み変更に対する深さ・不可逆性」を表す。L1=触らない / L2=貯めるだけ / L3=重みを変える。

各試行が **どこまで到達したか** は `dc_level_reached` に残る([L544](../intentflow/offline/models/deferred_commit_replay_otta.py#L544)):

| 値 | 意味 |
|---|---|
| 1.0 | L1 のみ(予測補正だけ。重みもメモリも変えず) |
| 2.0 | L2 まで(メモリに採用した。重みは不変) |
| 3.0 | L3 まで(重みを実際に commit した) |

---

## 3. L1 — 予測補正(毎試行 / 完全に可逆)

実装: [_apply_prediction_correction](../intentflow/offline/models/deferred_commit_replay_otta.py#L199-L256)

- **やること:** その試行の logits を、クラス事前分布のズレで補正するだけ。**モデル状態は一切変えない。**
- `memory_prior` モード(既定): `correction_vec = log(source_prior) - log(memory_prior)`
  - source_prior = 学習データのクラス分布([compute_source_statistics](../intentflow/offline/models/deferred_commit_replay_otta.py#L172-L184))
  - memory_prior = L2 外部メモリ内のターゲット試行のクラス分布([_memory_prior](../intentflow/offline/models/deferred_commit_replay_otta.py#L192-L197))
  - メモリのターゲット数が `dc_min_memory_for_correction`(既定8)未満なら補正は発火しない(`correction_active=False`)
- 補正は平均0化し `±dc_max_prior_correction` でクリップ、`dc_prior_correction_strength`(既定0.2)倍して加算。
- **意味:** 「このテスト被験者ではクラス分布が学習時とズレている」という事前分布バイアスを、重みを動かさず予測側で吸収する。最も軽量・安全な適応。

記録される診断: `correction_active`, `correction_strength`(補正前後の KL), `raw_corrected_disagreement`(補正で予測が変わったか)など。

---

## 4. L2 — 外部メモリ採用(信頼できる試行だけ貯める / 可逆)

貯蔵庫: [ExternalMemoryBuffer](../intentflow/offline/models/dc_replay_memory.py#L18)(FIFO, 既定 capacity=32)
採用判定: [_maybe_admit_to_replay](../intentflow/offline/models/deferred_commit_replay_otta.py#L338-L397) / スコア計算 [_memory_admission_features](../intentflow/offline/models/deferred_commit_replay_otta.py#L293-L336)

- **やること:** 「この試行は信頼できるか」をスコアで判定し、閾値超えのものだけメモリに追加。**重みは変えない。** 後の L1 事前分布と L3 drift 判定の材料になる。
- 採用スコア(加点 − 減点):

| 寄与 | 重み(既定) | 符号 |
|---|---|---|
| 信頼度 pmax | 0.35 | + |
| 密度支持 density | 0.15 | + |
| 時間的一貫性 temporal | 0.15 | + |
| プロトタイプ支持 proto | 0.15 | + |
| クラスバランス改善 balance | 0.10 | + |
| 不確実性 entropy | 0.15 | − |
| OOD リスク energy | 0.20 | − |
| raw/補正の不一致 | 0.10 | − |

- スコア ≥ `dc_memory_admission_threshold`(既定0.55)なら採用。
- 採用時、`source_tag` を付ける: 補正で予測が変わった試行は `target_corrected_only`、それ以外は `target_pseudo`。reliability などの metadata も一緒に保存。
- メモリは「ソース warmup 試行」(`warmup_source`)でシードされる([seed_from_loader](../intentflow/offline/models/dc_replay_memory.py#L101-L137))。これにより最初からリプレイ検証(後述)が動ける。

---

## 5. L3 — 遅延コミット(重みを実際に変える / 不可逆 / 二段ゲート)

L3 だけが本体の状態(`logit_bias` / `target_prototypes`)を変える。発火には **「候補提案ゲート」→「SafeCommit 検証」** の二段をすべて通過する必要がある。

### 5-1. 候補提案ゲート([_propose_dc_commit_candidates](../intentflow/offline/models/deferred_commit_replay_otta.py#L436-L484))

上から順に、1つでも引っかかれば `no_update`(=L3しない)。止まった理由は `dc_commit_reason` に残る。

```
1. severe OOD か?                     → abstain(予測棄権)         reason=energy_severe_ood
2. commit無効 / 適応OFF?              → no_update                  reason=commit_disabled
3. メモリのターゲット数 < 16?         → no_update                  reason=memory_not_ready
4. クールダウン中?(前回commitから8試行未満) → no_update           reason=cooldown
5. drift で候補を立てる ↓
```

drift は2種類([_compute_drift_diagnostics](../intentflow/offline/models/deferred_commit_replay_otta.py#L399-L434)):

| drift | 計算 | 閾値(既定) | 対応オペレータ |
|---|---|---|---|
| **prior drift** | KL(メモリのクラス分布 ‖ source 分布) | `dc_prior_drift_threshold` 0.08 | `logit_bias_update` |
| **proto drift** | `dc_proto_drift_reference(0.75) − メモリの prototype 支持平均` | `dc_proto_drift_threshold` 0.08 | `prototype_update` |

- さらに総合 `drift_score = prior + proto` が `dc_drift_score_threshold`(0.08)未満なら候補を全消し → reason=`no_persistent_drift`。
- 候補が立てば `[候補, …, no_update]` を SafeCommit へ → reason=`persistent_drift`。
- **L3 で許可される操作は意図的に2つだけ**(`dc_allowed_commit_operators = [logit_bias_update, prototype_update]`)。BN 更新など他オペレータは「この階層が検証されるまで対象外」と docstring で明言。

### 5-2. SafeCommit 検証(候補が立っても、ここを通らないと commit しない)

`dc_commit_mode` で検証の厳しさが変わる([_run_commit_policy](../intentflow/offline/models/deferred_commit_replay_otta.py#L486-L506)):

| mode | 使うゲート | 説明 |
|---|---|---|
| `replay_gated`(既定) | tier-1 + tier-2 | 即時ガード + リプレイ検証の両方 |
| `no_replay_gate` | tier-1 のみ | リプレイ検証なし(アブレーション用) |
| `random_sparse` | tier-1 のみ + 確率発火 | drift 無関係に確率 `dc_random_commit_prob` で commit(対照群) |
| `none` | なし | L3 完全無効(L1+L2 のみ) |

**tier-1(即時ガード, [SafeCommit.is_safe](../intentflow/offline/models/policy_safe_commit_otta.py#L552-L589)):** 候補を仮適用→同じ x を再 forward し、悪化していないか確認。
- margin 低下 / SAL 低下 / prototype margin 低下 / energy が OOD 化 / BN drift 過大 / shallow var drift 過大 → reject。

**tier-2(リプレイ検証, [ReplaySafeCommit.is_safe](../intentflow/offline/models/replay_safe_commit_otta.py#L203-L304)):** 候補を仮適用した状態でメモリ全体を再評価し、**未来への効果** を測る。
- `sim_score = w_acc·Δacc + w_margin·Δmargin + w_proto·Δproto`(既定 1.0 / 0.3 / 0.3)
- **sim_score が `sim_score_tolerance`(既定0.0)を厳密に超えたときだけ合格。** 改善ゼロは不合格。
- なぜ必要か: tier-1 は「同じ試行」を見るので forward-only OTTA では効果がほぼ測れない(pred_changed≈0)。リプレイは「直近の高信頼試行の集合」で更新の効果を擬似的に先取りする([docstring](../intentflow/offline/models/replay_safe_commit_otta.py#L10-L16))。

`no_update` も常に候補に混ざっているので、「何もしない方が安全」なら重みは動かない。最終的に `committed=True` かつ operator が `no_update`/`abstain` でないときだけ `adapted=True`(= `dc_level_reached=3.0`)。

### 5-3. L3 オペレータの中身

- **`logit_bias_update`** ([apply_logit_bias_update](../intentflow/offline/models/policy_safe_commit_otta.py#L158-L170)): `delta = one_hot(pred) − softmax(logits)` の平均を momentum(0.02)で `logit_bias` に EMA、`±max_logit_bias`(0.25)でクリップ。→ クラス事前分布の恒久的補正。
- **`prototype_update`** ([apply_prototype_update](../intentflow/offline/models/policy_safe_commit_otta.py#L139-L156)): 予測クラスの `target_prototype` を、その試行の特徴へ momentum(0.05)で EMA 更新。→ ターゲット領域の特徴中心を追従。

> L1 とL3-logit_bias は似て見えるが別物。L1 は「その試行限りの logits シフト(揮発)」、L3-logit_bias は「buffer に貯まる恒久パラメータの更新(不揮発)」。

---

## 6. 状態量(state)早見表

毎試行 [StateExtractor.extract](../intentflow/offline/models/policy_safe_commit_otta.py#L303-L370) が作る、判定の材料。

| キー | 意味 | 使われ方 |
|---|---|---|
| `pmax` / `entropy` / `margin` | 予測の確信度・曖昧さ | L2採用スコア, policy, tier-1 |
| `sal` / `target_sal` | source/target プロトタイプとの整合(cos) | tier-1, L2 |
| `prototype_margin` | プロトタイプ top1−top2 | tier-1 |
| `energy` / `energy_z` / `energy_ood` / `energy_severe_ood` | OOD 度合い(自由エネルギー) | abstain 判定, L2 減点 |
| `neuro_score` / `eca_motor_noise_ratio` | 運動野 vs ノイズ chの ECA 注意比 | neuro ガード(既定 off) |
| `shallow_var_drift_risk` | 浅層 BN 分散の source からの乖離 | BN系 policy(DC既定では未使用) |

---

## 7. ログ(`.npz`)の主要カラム

保存先: `results/.../dc_replay_otta_stats_s{subject}_{model}.npz`([on_test_epoch_end](../intentflow/offline/models/tcformer_deferred_commit_replay_otta.py#L264-L295))。1行=1試行。

分析でまず見るべきもの:

| カラム | 意味 |
|---|---|
| `dc_level_reached` | 各試行が L1/L2/L3 のどこまで行ったか(1/2/3) |
| `dc_commit_reason/*` | L3 がどのゲートで止まったか(`memory_not_ready`/`cooldown`/`no_persistent_drift`/`persistent_drift`…) |
| `dc_model_state_committed` | 実際に重みを commit したか(0/1) |
| `dc_prior_drift_score` / `dc_proto_drift_score` / `dc_persistent_drift_score` | drift の値 |
| `correction_active` / `correction_strength` / `raw_corrected_disagreement` | L1 補正の効き具合 |
| `memory_add_score` / `memory_admitted` / `memory_buffer_size` | L2 採用の判定とメモリ規模 |
| `replay_d_acc` / `sim_score` / `replay_safe_pass` | L3 tier-2 リプレイ検証の結果 |
| `pred` / `original_pred` / `label` / `correct` / `original_correct` | 適応後 vs 適応前(L1のみ補正前)の正誤 |

> `original_pred` は L1 補正 *後* の before_state 予測である点に注意([forward L521](../intentflow/offline/models/deferred_commit_replay_otta.py#L521))。「完全に素の予測」ではなく「L1適用済み・L3適用前」の予測。

---

## 8. 主要ハイパラ早見表(config の `model_kwargs`)

| 層 | パラメータ | 既定 | 効果 |
|---|---|---|---|
| L1 | `dc_correction_mode` | `memory_prior` | none/static_prior/memory_prior |
| L1 | `dc_prior_correction_strength` | 0.2 | 補正の強さ |
| L1 | `dc_min_memory_for_correction` | 8 | 補正発火に必要なメモリ数 |
| L2 | `dc_memory_admission_threshold` | 0.55 | 採用スコア閾値 |
| L2 | `replay_capacity` | 32 | メモリ容量 |
| L3 | `dc_commit_mode` | `replay_gated` | 検証の厳しさ(none で L3 無効) |
| L3 | `dc_min_memory_for_commit` | 16 | commit に必要なメモリ数 |
| L3 | `dc_commit_cooldown` | 8 | commit 間の最小間隔 |
| L3 | `dc_*_drift_threshold` | 0.08 | drift 発火閾値 |
| L3 | `dc_allowed_commit_operators` | logit_bias, prototype | 許可オペレータ |
| 検証 | `sim_score_tolerance` | 0.0 | tier-2 合格ライン |
| 安全 | `abstain_on_ood` | true | severe OOD で予測棄権 |

---

## 9. 検証・分析の入口

### 実行(GPU 必須)
```bash
conda activate intentflow
python -c "import torch; assert torch.cuda.is_available()"   # CUDA preflight
python intentflow/offline/train_pipeline.py \
  --model tcformer_deferred_commit_replay_otta --dataset bcic2a --gpu_id 0
```
smoke config: [tcformer_deferred_commit_replay_otta_s2_smoke.yaml](../intentflow/offline/configs/tcformer_deferred_commit_replay_otta/tcformer_deferred_commit_replay_otta_s2_smoke.yaml)

### まず確認すべきこと(主張の妥当性に直結)
1. **`dc_level_reached` の分布** — L3 がほぼ 0 回なら、複雑な L3 は実質効いておらず「L1+L2 だけで性能が出ている」可能性。
2. **`dc_commit_reason/*` の内訳** — どのゲート(memory_not_ready / cooldown / no_persistent_drift)で止まっているか。
3. **段階的アブレーション** — 各層の寄与を分離するには以下を揃える:
   - L1 のみ: `dc_commit_mode=none`, `dc_enable_memory_update=false`
   - L1+L2: `dc_commit_mode=none`
   - L1+L2+L3(replay_gated): 既定
   - 対照: `dc_commit_mode=no_replay_gate`(tier-2 の寄与), `random_sparse`(drift 判定の寄与)
4. `correct` vs `original_correct` の差 — 適応が正味で効いているか、回帰していないか。

---

## 10. 落とし穴・注意

- **本体は学習しない。** 勾配適応ではなく、buffer 3点(logit_bias / prototype / BN stats)の制御だけ。「TTT のような重み更新」を期待すると誤解する。
- **L1 と L3-logit_bias の二重補正** — どちらもクラス事前分布をいじる。効果が交絡しうるので、寄与分離には L1/L3 個別アブレーションが要る。
- **ゲートが厳しい** — memory≥16, cooldown=8, drift≥0.08, sim_score>0 を全部満たさないと L3 は出ない。データによっては L3 が稀。発火回数を必ず実測する。
- **`test_batch_size: 1` 前提** — メモリ採用も drift もシングル試行前提。バッチ>1だと採用・補正パスが想定外になる([_maybe_admit_to_replay の shape ガード](../intentflow/offline/models/deferred_commit_replay_otta.py#L343))。
- **OOD 安全** — severe OOD は abstain(fail-closed)。この契約を消さない。

---

## 11. 30秒サマリ

1. TCFormer 本体は凍結。適応は外側の制御層が担う。
2. **L1**: 毎試行、事前分布ズレで logits を補正(揮発・可逆)。
3. **L2**: 信頼できる試行だけ外部メモリに貯める(可逆)。重み変更の証拠源。
4. **L3**: メモリが「恒久的 drift」を示し、二段ガード(即時 + リプレイ)を通った時だけ重みを更新(不可逆)。許可は logit_bias と prototype の2操作のみ。
5. 思想: **安全な操作ほど頻繁に、危険な操作ほど稀に・強い証拠を要求して。**
6. 主張前に必ず `dc_level_reached` と `dc_commit_reason` を実測し、L3 が本当に効いているか確認する。
