---
paths:
  - "intentflow/online/**/*.py"
  - "configs/online.yaml"
  - "configs/adapt.yaml"
  - "deployment/**/*"
  - "apps/unity_client/Assets/Scripts/**/*.cs"
---

# Online Safety Rules

- Preserve abstention behavior for uncertain predictions unless the task explicitly redesigns it.
- Preserve or strengthen ERRP and safety-fallback handling; do not remove safeguards silently.
- Prefer fail-closed defaults for streaming, networking, and adaptation thresholds.
- Call out any latency, buffering, or concurrency risk introduced by the change.
- Keep external protocol compatibility in mind for Unity and server-side message contracts.
