---
name: ablation-design
description: EEG や Active BCI 実験で、公平なアブレーションを設計するときに使う。特に cross-subject MI classification、OTTA、TTT、gating、calibration、latency-sensitive な比較設計で使う。
---

# アブレーション設計

config や学習コードを変更する前の実験計画に使う。

## ワークフロー
1. 仮説を1文で書く。
2. 明示的に factorial design が求められていない限り、操作因子は1つに限定する。
3. 統制変数を固定する: dataset、split、seed 方針、前処理、metrics、hardware assumption、baseline。
4. 比較対象と成功条件を定義する。
5. 想定される confounder と、それを検出するための logging / diagnostics を列挙する。

## 出力フォーマット
- Hypothesis
- Manipulated variable
- Controlled variables
- Comparison target
- Expected metric movement
- Confounders
- Minimal run plan
