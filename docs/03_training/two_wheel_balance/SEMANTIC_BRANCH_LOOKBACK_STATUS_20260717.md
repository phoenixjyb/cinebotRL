# Semantic Branch Lookback Status

Date: 2026-07-17

## Purpose

Ep4's derived diagnostic rejects at semantic interval 50 after the accepted
prefix reaches `0.0473395 m` position error and `29.503574 Nm` peak gravity.
The next target moves `0.0174841 m`; the paired holonomic seed is disconnected
and exceeds the gravity bound. The evidence points to accumulated greedy
base/arm branch selection rather than gimbal, attitude, target speed, or path
roughness.

## Implemented foundation

Commit `e319bd54d4eab4a31e64b94c3331cc3a350331d8` adds the runtime-inert core for
an opt-in bounded lookback:

- `SemanticBranchAlternative` records a hard-gate decision, local/future score,
  deterministic lineage, endpoint state, and full previous control.
- `select_semantic_branch_beam` rejects malformed/non-finite branches, drops
  every hard-infeasible branch, deterministically sorts future score before
  local score, deduplicates state/control-equivalent branches, and applies a
  fixed beam width.
- `truncate_exact_source_prefix_for_semantic_lookback` rewinds only complete
  source intervals from a checkpoint that has already passed the fail-closed
  loader. It preserves all retained arrays and ordered anchor-map entries,
  recomputes the retimed count, and reconstructs the prior 8D control from the
  stored 5D base/arm control plus the adjacent gimbal-state delta.
- Rewind cannot cross source anchor zero or accept an incomplete/invalid map.

The synthetic boundary test proves why beam retention is needed: the locally
better branch dead-ends at the next transition, while a retained hard-feasible
alternative survives.

## Verification

- Focused retarget/checkpoint suite: `49 passed`, two pre-existing pytest
  configuration warnings.
- Repository-wide suite: `253 passed`, the same two warnings.
- Python compilation and `git diff --check` pass.
- Branch and GitHub remote both point to `e319bd5` after push.

## Deliberate runtime boundary

The current commit adds no CLI option and does not call either helper from the
production semantic solver. Existing greedy behavior, checkpoint identity,
source geometry/timestamps, rate limits, plant, physical gates, output schema,
and candidate admission are unchanged. No ep4/ep7 solver run, Isaac playback,
capture, BC, PPO, or residual learning was launched.

This foundation is not evidence that ep4 passes. It is not valid for dynamic
evaluation or training.

## Required integration before another canary

The next code-only patch must:

1. Add default-disabled, checkpoint-identity-bound settings for lookback source
   intervals and beam width.
2. On an explicitly requested resume, load and validate the original
   checkpoint first, then rewind through the reviewed helper.
3. Generate more than one complete hard-feasible interval alternative from
   each retained branch; do not merely rename a one-step greedy rank.
4. Retain deterministic branch history across the bounded window and score
   future target reachability plus gravity slack without changing any hard
   gate.
5. Collapse to one branch only after crossing the previous rejection interval,
   then continue the unchanged greedy path.
6. Emit branch lineage/count/rejection diagnostics and bind every setting into
   the checkpoint code/CLI identity.
7. Prove disabled-default equivalence, exact prefix/map/clock preservation,
   deterministic beam behavior, and fail-closed resume/config mismatch in
   focused and full CPU tests.

Only after that patch is independently reviewed should one fresh, CPU-only,
training-disabled ep4 canary be considered. Gate relaxation, additional
waypoint reduction, larger geometric relief, Isaac, and learning remain out of
scope.
