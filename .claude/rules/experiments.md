---
paths:
  - "configs/**/*.yaml"
  - "intentflow/offline/**/*.py"
  - "scripts/experiments/**"
  - "intentflow/offline/scripts/**"
---

# 実験ルール

- task が明示的に grid や factorial ablation でない限り、複数の実験因子を同時に変えない。
- baseline の configs や scripts は保持し、比較対象を直接変更せず、新しい config 名で派生を追加する。
- dataset、split、preprocessing、seed、model、loss、adaptation rule、evaluation metric の変更はすべて明示的に記録する。
- metric の回帰も、改善と同じくらい明確に報告する。
- baseline として使っている過去の result directory や summary を上書きしない。
- 変更が再現性に効くなら、その exact comparison の再実行方法を説明する。
