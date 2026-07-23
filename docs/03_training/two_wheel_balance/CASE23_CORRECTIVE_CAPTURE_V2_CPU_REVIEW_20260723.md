# Case-23 Corrective Capture V2 CPU Review

Date: 2026-07-23

## Decision

**GO for a later, separately authorized, exactly-one case-23 v2
corrective-label capture.**

This review does not issue an authorization token and does not authorize
Isaac, GPU use, dataset conversion, corpus merge, BC, PPO, holdout access, or
training.

## Fresh Runtime Boundary

- Namespace:
  `20260723_model_based_corrective_teacher_case23_capture_v2_exclusive`
- Reviewed parent:
  `94ebd9b9ba72f74be09934eed495a110d90af419`
- Implementation and authoritative preflight commit:
  `4a6e45643a2efd6b713054fa5862b0ae4a506e8d`
- Contract SHA-256:
  `9ef7a279de8266a5b3f6cdb73e6395a2c69985ead1563145e8eb26900a035aeb`
- Contract Git blob:
  `2a1481fc5838b00735c03f8d85662c06b1f978b5`

The rejected v1 namespace remains preserved. V2 uses a separate namespace and
does not overwrite, delete, or reinterpret the rejected attempt.

## Path Repair

The failed v1 attempt proved that a backslash adjacent to `$NAMESPACE` could
escape expansion. V2 assembles the Windows output path as three shell segments:
the expanded root, a single-quoted literal path and separator, then the
expanded namespace. The regression test executes this expression through the
Windows authoritative Python environment and confirms the final path contains
the concrete v2 namespace and no literal shell variable.

The same safe expression is present in the permanently no-token v1 wrapper so
the repository no longer preserves a misleading path example.

## Contract Invariants

The `.98` preflight proves:

- clean tracked worktree and `HEAD == upstream`;
- fresh v2 namespace and preserved rejected v1 namespace;
- exact case 23, train split, plan, passed pair, corrective and wrench profiles;
- frozen LQR gains and unchanged `[0.05, 0.05, 0.02]` residual scales;
- active `leadshine_400w_engineering_sample_v1` simulation plant;
- 750 W candidate remains disabled for simulation, runtime, and training;
- effective post-supervisor labels remain the only future training targets;
- conversion, BC, PPO, holdouts, and training remain closed.

The focused cross-platform suite passed `14 tests` with two configuration
warnings. The authoritative `.98` CPU suite at sanitized commit `4a6e456`
passed `941 tests`, skipped `12`, and emitted two configuration warnings in
`85.23 s`.

## Exact Next Gate

A later explicit authorization must create exactly one mode-`0600` token
outside the repository and provide its lowercase SHA-256 out of band through
the runtime environment. Neither the token nor its hash may be committed.
Before launch, the validator must again prove a clean pushed commit, fresh
namespace, exact identities, matching out-of-band hash, and exclusive GPU
ownership.

The attempt must stop after one execution regardless of pass or reject. A pass
opens only a separate conversion review. It does not authorize conversion,
merge, BC, PPO, or training.
