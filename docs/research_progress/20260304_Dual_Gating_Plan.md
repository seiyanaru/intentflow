# Dual Gating (Neuro + Statistical) 実装プラン詳細（旧称: Phase 7）

先ほど承認いただいた方針に基づき、具体的にコードをどのように変更するか（特に数式とロジック）を整理しました。変更の核心は、現在の「Neuro-Score」による生理学的ゲートに加えて、**「Energy Score」を用いた統計的ゲート**を導入し、適応の条件をより厳しく（確実なものに）することです。

## 1. 解決したい課題（現行実装の弱点）
現在の `PmaxSAL_OTTA` では、適応（BN更新）の条件が以下のようになっています。
- **条件:** `pmax > pmax_th` かつ `SAL > sal_th`
- **問題点:** `SAL`（Source Alignment Level）は、単に予測クラスのプロトタイプとのコサイン類似度です。アーティファクトによって特徴量が大きく歪んだ場合でも、たまたまあるクラスの方向に少し寄っていれば `SAL` が高くなってしまい、適応がオンになってしまうリスクがありました。（Codexも指摘している「プロトタイプ未計算時にSAL=1になるバグ」も潜在的なリスクです）

## 2. 新しいゲート：Energy Score (統計的分布ゲート) に追加
最新のTTAトレンド（BTTA-DGやEBM: Energy-Based Models）に倣い、**Energy Score**を導入します。Energy Scoreは、出力ロジット（Softmax前）の分布から、そのサンプルが「学習データ（Source）の分布にどれくらい近いか（In-Distributionか）」を測る統計指標です。

$$ E(x) = -T \cdot \log \sum_{i} \exp(f_i(x) / T) $$
- $f_i(x)$: クラス $i$ の出力ロジット
- $T$: Temperatureパラメータ (通常 $T=1.0$)

Energy Score $E(x)$ が**低い**ほど、モデルはそのサンプルに馴染みがあり（In-Distribution）、**高い**ほど未知の分布（Out-of-Distribution, アーティファクトなど）であることを示します。

## 3. Dual Gating の最終的な判定ロジック

適応を許可する条件（`should_adapt` が `True` になる条件）を、以下のように**三重のロック（Tri-Lock）**にします。

1. **Confidence Gate（既存）:** `pmax > pmax_th`
   - モデルがはっきり推論できているか？
2. **Physiological Gate（既存・改善）:** `Z(Neuro-Score) による pmax_th の引き上げ`
   - 運動野を見ていなければ、閾値（`pmax_th`）を引き上げて適応をブロックする（Conservative Gating）。
3. **Statistical Gate（新規！！）:** `Energy Score < energy_th`
   - そのサンプルは統計的に学習データの分布から外れすぎていないか？（Out-of-Distributionなら適応しない）

## 4. コードの具体的な変更予定箇所 (`pmax_sal_otta.py`)

1. **`__init__` メソッド:**
   - 新たな閾値パラメータ `energy_threshold`（仮に -10.0付近で調整）を設定可能にする。
   - `neuro_beta` のデフォルト値を `5.0` から `0.1` に変更（シミュレーション側と整合させる）。

2. **`compute_energy` メソッドの追加:**
   ```python
   def compute_energy(self, logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
       # logits: [B, C]
       energy = -T * torch.logsumexp(logits / T, dim=1)
       return energy # [B]
   ```

3. **`compute_sal` メソッドの修正:**
   - プロトタイプが準備されていない場合、`sal = 1` ではなく `sal = 0`（警告付き）を返すように安全側に倒す。

4. **`compute_gate` メソッドの修正:**
   - `Energy Score` を受け取り、上記の「Tri-Lock条件」を組み込む。

---

このロジックにより、**「脳波としてまともであり（Neuro）、かつ統計的にもおかしなデータではなく（Energy）、かつはっきり予測できている（Pmax）」**サンプルのときのみ、安全に適応（BN更新）が行われるようになります。

この具体的なコード変更方針で実装を進めてもよろしいでしょうか？
