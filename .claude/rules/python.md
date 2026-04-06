---
paths:
  - "**/*.py"
  - "tests/**/*.py"
---

# Python ルール

- 小さく、戻しやすい edit を優先する。
- 変更に必須でない限り、既存の type hints と function signatures を保つ。
- hidden global state、silent device move、silent dtype change を入れない。
- 数値処理でロジックが非自明なら、tensor や array の shape 前提を明示する。
- コメントは最小限にし、コードから自明でない reasoning だけを書く。
- 振る舞いが変わるなら、最小限で relevant な test を追加または更新する。
