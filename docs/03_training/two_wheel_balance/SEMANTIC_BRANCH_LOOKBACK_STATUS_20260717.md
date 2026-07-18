# Semantic Branch Lookback Status

Date: 2026-07-17
Updated: 2026-07-18

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

- Foundation commit verification: focused `49 passed`; repository-wide `253
  passed`.
- Production integration commit
  `3c7f83f929e01159926f8e1930271b3db3d4bee9`: focused retarget/checkpoint
  `53 passed`; all `test_two_wheel*.py` files `181 passed`; repository-wide
  `257 passed`. All runs report only the same two pre-existing pytest
  configuration warnings.
- Python compilation and `git diff --check` pass.
- Windows-Python checkpoint code-contract SHA-256 at `3c7f83f` is
  `25d30b95ad0c0ce8a4848c1f3c4d8e627ab208295f5938de6c4ed3928eff6f5a`.
- Local branch and GitHub ref `codex/ep4-time-reparameterization` both point to
  `3c7f83f` after push.

## Production integration

Commit `3c7f83f` adds three default-disabled, checkpoint-identity-bound options:

- `--enable-semantic-branch-lookback`
- `--semantic-branch-lookback-source-intervals` (default `6`)
- `--semantic-branch-beam-width` (default `4`, bounded to `2..4` when enabled)

When explicitly enabled for a checkpoint resume, the production solver loads
the checkpoint through the existing fail-closed identity/source validator,
rewinds only complete source intervals, and runs deterministic complete-history
ranking variants through the checkpoint's prior rejection interval. Speculative
histories never write the checkpoint. Only distinct histories that pass every
unchanged hard gate are retained. The selected crossing prefix is atomically
saved before the normal greedy continuation resumes.

The four ranking variants preserve the same feasibility predicate and differ
only in deterministic ordering: existing score, gravity-first,
position-first, and source-arm-continuity-first. Branch count, lineage,
rejections, selection, and local/one-step future scores are emitted in the
result diagnostics. The default-disabled path calls the original single-branch
solver, and its resumed-vs-uninterrupted regression remains exact.

## Deliberate runtime boundary

No source position, attitude, timestamp, map, plant, rate limit, gravity/pitch,
tracking, gimbal, or admission gate was changed. The old derived ep4 checkpoint
cannot be consumed directly because the new commit, code-contract, and CLI
identity are intentionally different; any future use requires a separately
reviewed lineage-safe checkpoint migration.

No ep4/ep7 solver run, Isaac playback, capture, BC, PPO, or residual learning
was launched. The integration is not evidence that ep4 passes and remains
invalid for dynamic evaluation or training.

## Next review gate

Before any new ep4 canary, independently review the production diff and approve
one lineage-safe migration of the existing verified prefix to commit `3c7f83f`
with the opt-in lookback settings sealed in its identity. A future canary must
remain CPU-only, bounded, single-owner, and training-disabled. Gate relaxation,
additional waypoint reduction, larger geometric relief, Isaac, and learning
remain out of scope.
