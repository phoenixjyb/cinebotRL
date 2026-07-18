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
- Persistence hardening commit
  `8a3632f49a2ac004c91aff3f52a04301333bb1ee`: focused checkpoint/journal
  `32 passed`; all `test_two_wheel*.py` files `186 passed`; repository-wide
  `262 passed`. All runs report only the same two pre-existing pytest
  configuration warnings.
- Python compilation and `git diff --check` pass.
- Windows-Python checkpoint code-contract SHA-256 at `3c7f83f` is
  `25d30b95ad0c0ce8a4848c1f3c4d8e627ab208295f5938de6c4ed3928eff6f5a`.
- Implementation commit `8a3632f` is pushed on
  `codex/ep4-time-reparameterization`; later documentation-only commits do not
  alter its code-contract hash.
- Windows-Python checkpoint code-contract SHA-256 at `8a3632f` is
  `b90809aa0d3d50e8e4caf6c17257fd55462ae927894086a833cd0fa70e86b365`.

## Production integration

Commit `3c7f83f` adds three default-disabled, checkpoint-identity-bound options:

- `--enable-semantic-branch-lookback`
- `--semantic-branch-lookback-source-intervals` (default `6`)
- `--semantic-branch-beam-width` (default `4`, bounded to `2..4` when enabled)

When explicitly enabled for a checkpoint resume, the production solver loads
the checkpoint through the existing fail-closed identity/source validator,
rewinds only complete source intervals, and runs deterministic complete-history
ranking variants through the checkpoint's prior rejection interval. A
speculative history never writes the main checkpoint. Each completed rejected
history is atomically recorded in an identity-bound JSON journal; each completed
successful history is durably stored in a separate standard exact-source
checkpoint. Restarts skip completed ranks and continue from the first unfinished
rank. Only distinct histories that pass every unchanged hard gate are retained.
Selection uses a journal-first pending transaction, atomically installs the
selected history as the main checkpoint, and then marks the transaction selected
before normal greedy continuation resumes.

The journal binds the complete 11-field checkpoint identity, original main
checkpoint SHA-256, rejection/replay intervals, lookback, beam width, per-rank
status, successful-prefix SHA-256 and scores, seed-family diagnostics, and
`valid_for_training=false`. It fails closed on malformed data, identity or hash
mismatch, orphan checkpoints, invalid rank ordering, corrupted successful
prefixes, and a current main checkpoint that does not extend the selected
prefix. A restart after an interrupted pending selection either installs the
exact recorded prefix or rejects; it cannot silently choose another history.

The four ranking variants preserve the same feasibility predicate and differ
only in deterministic ordering: existing score, gravity-first,
position-first, and source-arm-continuity-first. Branch count, lineage,
rejections, selection, and local/one-step future scores are emitted in the
result diagnostics. The default-disabled path calls the original single-branch
solver, and its resumed-vs-uninterrupted regression remains exact.

## Bounded canary evidence

The lineage-safe migration to commit `23bca6b` preserved every accepted prefix
array byte-for-byte and produced the training-disabled checkpoint:

`gate1_ep4_relief20mm_reservecap8_branchlookback_23bca6b_checkpoint_20260718.npz`

Its SHA-256 remains
`7ad0703528fb096ff44a07fbbbbee900bd23821047ae347e1ef81dc67ce9d2b7`.

- V1 was stopped after a shell output-path expansion defect. It did not enter a
  valid evidence namespace and did not modify the checkpoint.
- V2 was stopped after a later riser Isaac process violated exclusivity. It
  completed replay intervals 44 through 49 only and did not modify the
  checkpoint.
- V3 ran exclusively for the exact 1800-second bound. Rank history 0 replayed
  intervals 44 through 49 and rejected at the prior interval 50; history 1
  replayed intervals 44 through 49 and timed out while evaluating interval 50.
  Histories 2 and 3 did not start. The output namespace is empty and the main
  checkpoint SHA-256 is unchanged.

V3 is classified
`exclusive_timeout_after_history0_rejection_during_history1_interval50`, not a
physical pass/fail. Its timeout audit is:

`evaluation_results/two_wheel_exact_source_v1/gate1_ep4_relief20mm_reservecap8_branchlookback_23bca6b_canary_v3_20260718.timeout_audit.json`

Audit SHA-256:
`d2740ac9812a1633d229791dc923c5701f9c10a786ecd791fa3ef63cda49a5a8`.

This proves the original lookback solver was computationally non-resumable
under the bound: every process discarded completed rank outcomes and replayed
the expensive interval-50 search. Commit `8a3632f` fixes that persistence defect
without changing solver behavior.

## Persisted four-rank outcome

The accepted prefix was migrated to clean pushed tip `d3125d8` with code
contract `b90809aa...`. The migration changed exactly `git_commit` and
`code_contract_sha256`; all nine checkpoint arrays remained byte-identical.
The migrated main checkpoint SHA-256 is
`dcd4e6e320ed74f83e41be888b5124a10a4bb350b5b84a78bb1393f10e94f569`.

Bounded, single-owner CPU tranches then completed all four deterministic rank
variants. Ranks 0, 1, 2, and 3 all rejected at semantic interval 50. The final
rank reported `0.057255 m` position error, `0.033310 deg` attitude error,
`29.503980 Nm` gravity, `3.928948083 deg` equilibrium pitch, and
`0.016186103` minimum gimbal margin against the unchanged `0.005` hard
requirement. Across the beam, `hard_feasible_distinct_history_count=0`.

The identity-bound journal now has selection status `exhausted`, SHA-256
`36e2d522b8d977bd63942bb6b330101a274dafd370667a6fa6999396878d08e3`,
and four durable rejected histories. The main checkpoint is unchanged, no
successful rank checkpoint or candidate NPZ exists, and the final fail-closed
result remains `valid_for_dynamic_evaluation=false` and
`valid_for_training=false`.

Final namespace:

`gate1_ep4_relief20mm_reservecap8_branchjournal_d3125d8_canary_v7_20260718`

Final result SHA-256:
`40cc2c81b404df41df67436c355d401c29a773512c382c3b940b393e51cb78d9`.

Final summary SHA-256:
`6596b59e8342c7a4ecea501a3999c4b3c1f48678bad73ce846e9f73dfed517ca`.

## Deliberate runtime boundary

No source position, attitude, timestamp, map, plant, rate limit, gravity/pitch,
tracking, gimbal, or admission gate was changed. The old derived ep4 checkpoint
cannot be consumed directly by the post-`8a3632f` branch because the commit and
code-contract identity are intentionally different; any future use requires a
separately reviewed lineage-safe checkpoint migration. Its CLI config and all
solver settings remain unchanged.

CPU-only canaries under the post-`8a3632f` code contract were launched solely
to complete the persisted four-rank diagnostic. No ep7 run, Isaac playback,
capture, BC, PPO, or residual learning was launched. These are computational
diagnostics only; ep4 remains invalid for dynamic evaluation and training.

## Next review gate

Do not rerun this four-rank ep4 search: its deterministic beam is exhausted at
interval 50 under the current source, derived timing, plant, and unchanged
gates. Any next ep4 action requires a separately reviewed upstream/solver
proposal with new evidence and provenance; it must not silently relax a gate or
mutate this derived package. Ep7, Isaac, playback, capture, BC, PPO, and residual
learning remain out of scope.
