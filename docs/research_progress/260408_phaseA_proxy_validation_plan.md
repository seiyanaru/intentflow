# 260408 Phase-A Proxy Validation Plan

## Goal

Implement and run **Phase A** before train-time changes (Phase B/C):

- `A-1`: marginal proximity (`KL(clean_T || eval_E)` vs `KL(aug_T || eval_E)`)
- `A-2`: drift direction consistency (`cos(drift_aug, drift_eval)`)

This is a go/no-go gate for whether gain-jitter augmentation is a valid session-shift proxy.

## Why this order

Current evidence says harmful transfer is localized to shallow BN variance drift.  
If augmentation does not reproduce the same shallow drift direction, train-time simulation can optimize the wrong target.

Therefore, Phase A is required before implementing Phase B/C losses.

## Scope

- Dataset: `bcic2a`
- Subjects: `1..9`
- Gain jitter std sweep: `0.05, 0.10, 0.20`
- Splits:
  - `clean_T`: training session data
  - `aug_T`: channel gain jitter applied to `clean_T`
  - `eval_E`: evaluation session data

## Metrics

For each BN layer:

1. `KL(clean || eval)`
2. `KL(aug || eval)`
3. `improvement = KL(clean || eval) - KL(aug || eval)`  
   positive is better
4. `cos(drift_aug, drift_eval)` where  
   `drift_aug = [mu_aug - mu_clean, logvar_aug - logvar_clean]`  
   `drift_eval = [mu_eval - mu_clean, logvar_eval - logvar_clean]`

Shallow/deep grouping currently follows existing OTTA convention (first half BN layers = shallow, second half = deep).

## Pass criteria

Per-subject, per-gain-std:

- `A-1 pass`: shallow layers with `improvement > 0` are majority
- `A-2 pass`: shallow layers with `cos_sim > 0` are majority

Global go condition:

- prioritize gain settings with high `(A-1 && A-2)` subject pass rate
- reject gain settings that frequently show negative shallow drift cosine

## Implementation

- Added: `intentflow/offline/utils/bn_stat_collector.py`
  - Loads source checkpoint per subject
  - Uses `tcformer` backbone for Phase A (OTTA wrapper is intentionally bypassed)
  - Hooks BN input activations
  - Computes layer-wise channel Gaussian stats
  - Outputs JSON/CSV summaries, pass rates, and KL channel diagnostics (median/max)
- Added: `intentflow/offline/scripts/run_proxy_validation.sh`
  - Wrapper to run all subjects and gain sweep reproducibly

## Planned next step after Phase A

Only if pass rates are acceptable:

1. Phase B (`aug-only` training baseline)
2. Phase C (`virtual BN` train-time simulation)

If Phase A fails, redesign augmentation first (do not proceed to B/C).
