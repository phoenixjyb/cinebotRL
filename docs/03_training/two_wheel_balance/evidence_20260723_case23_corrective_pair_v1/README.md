# Case-23 model-based corrective pair

This directory is the tracked evidence subset from the single authorized
baseline-first case-23 paired canary executed on `.98` at runtime commit
`d77a1d494be79e442798e34368d865de1cf7ce25`.

- Namespace: `20260723_model_based_corrective_teacher_case23_pair_v1_exclusive`
- Contract SHA-256: `e5b5b360efdb0334412fb156d77dba7e0a6eb605651c16bffc280a8076caa043`
- Baseline position p95/max: `0.0593848022 / 0.0740444738 m`
- Candidate position p95/max: `0.0534133640 / 0.0679118294 m`
- Absolute/relative p95 improvement: `0.0059714383 m / 10.055499%`
- Final status SHA-256: `67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb`

Both rollouts passed the unchanged dynamic gate and deterministic perturbation
contract. The finalizer passed every paired improvement/no-regression check and
verified GPU release. The one-use token was consumed before Isaac. No labels,
capture dataset, normalized corpus, BC, PPO, policy rollout, or training were
authorized or created. This evidence opens only a separate case-23 corrective
capture review.

The complete runtime logs remain in the authoritative `.98` namespace above.

