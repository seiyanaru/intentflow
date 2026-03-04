# 現在のOTTA実装（Neuro-Gated OTTA）のコード構造まとめ

現在リポジトリ（`intentflow`）に実装されているオンラインテスト時適応（OTTA）関連のロジックについて、該当ファイルとその役割を調査した結果をまとめます。

## 1. コア実装ファイル：`intentflow/offline/models/pmax_sal_otta.py`

このファイルに、研究計画書に記載されている「Pmax-SAL Gated OTTA」および直近の「Z-score正規化によるConservative Gating（旧称: Phase 7）」の核となるロジックが実装されています。

### コアクラス①：`OnlineNormalizer` (L20-L86)
- **役割:** Neuro-Score（SAL）のオンラインZ-score正規化を行うクラスです。
- **特徴:**
  - `running_mean`, `running_var` を指数移動平均（EMA, `momentum=0.1`）で更新する。
  - 初期値として事前知識（Prior Init: 平均0.0, 分散0.0001）を持たせている。
  - 最終的に `z = (x - mean) / std` を計算し、`[-3.0, 3.0]` の範囲にクリッピングしたZスコアを返す。

### コアクラス②：`PmaxSAL_OTTA` (L89-L604)
- **役割:** TCFormerモデルをラップし、推論時に動的にBatch Normalization (BN) 層の統計量を更新するメインクラスです。
- **主なプロセス:**
  1. **特徴抽出**: TCFormerの分類器直前の出力をフック（`_register_feature_hook`）して取得。
  2. **アテンション重みの取得**: `TCFormer`内部のECAブロックから、チャネルごとのアテンション重みを取得。
  3. **Neuro-Score計算 (`compute_neuro_score`)**: 
     - 運動野（Motor）チャンネルとノイズ（Noise）チャンネルに対するアテンションの総和を比較し、`[-1, 1]` のスコアを算出。
  4. **Conservative Gating (`compute_gate`)**:
     - Neuro-Scoreを `OnlineNormalizer` でZスコア化。
     - **Zスコアが負（アーティファクト疑い）の場合のみ**、適応の閾値（`pmax_th`, `sal_th`）を `beta * ReLU(-z)` の分だけ動的に**引き上げ**、適応を抑制する（`modifier`ロジック）。
  5. **適応の実行 (`_update_bn_stats`)**:
     - 閾値をクリアした信頼度の高いサンプルのみを用いて、モデルを一時的に `train()` モードにし、BNパラメータを更新する。

---

## 2. 実験シミュレーションスクリプト：`simulate_neuro_conservative_gating.py`

- **役割:** 完全なモデル推論を回す代わりに、過去の実験（`results/...`）で保存された各種スコア（SAL, Neuro-Score等）の `.npz` ファイルをロードし、**「Conservative Neuro-Gating（旧称: Phase 7）を適用した場合、どのサンプルが適応（Flip）対象になるか」** を高速にシミュレーションするためのスクリプトです。
- **特徴:**
  - 内部に簡易版の `VirtualOnlineNormalizer` を持ち、擬似的にバッチサイズ（144）区切りでZスコア計算と適応判定（`should_adapt`）を行っている。
  - パラメータ（`beta`, `base_sal_th` など）のキャリブレーション（調整と結果確認）に現在使われているファイルです。

---

## 3. モデル評価への組み込み：`debug_otta.py` など

- `debug_otta.py` を見ると、ベースモデル（TCFormer）の評価時に `model.enable_otta = True` および `model.otta.enable_adaptation = True` とフラグを立てることで、推論ループ内で透過的に `PmaxSAL_OTTA.forward()` が呼ばれ、オンライン適応が行われるアーキテクチャになっています。
- また、推論開始前に `model.train_dataloader_ref` を用いて、ソースデータからプロトタイプベクトル（特徴量のクラスター中心）を事前計算（`compute_source_prototypes`）する仕組みになっています。

---

### 次のステップへの示唆
コードは「チャネルアテンションを利用したNeuro-Gating」と「Zスコアによる保守的閾値調整（Conservative Gating）」が見事にPyTorchのモジュールとして統合されています。

現在開いている `simulate_neuro_conservative_gating.py` で `beta` や `base_sal_th` の最適なバランスを見つけることは、実際に `debug_otta.py` や本番の評価スクリプトを回して精度向上（Acc改善）を証明するための最重要ステップとなります。
