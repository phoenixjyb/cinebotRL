# Case 23 corrective pair proposal v1

This CPU-only proposal pins case 23 as the first new diverse paired canary after
the converted case-30 pilot. It uses a distinct case-23 profile identity and
does not authorize reuse of the case-30 profile.

The proposed experiment is baseline first, then candidate, with identical
plan, seeds, physics, clocks, and a proposed 20-step 20 N pulse at execution
phase time 4.964847 s. The candidate must improve position p95 by at least
0.003 m and 2% without the existing bounded regressions.

No runtime wrapper or authorization token exists. GPU launch, label capture,
dataset creation/merge, BC, PPO, and training remain closed. Proposal SHA-256
is `ef520558f4240add67667e1cbd3146c987d7b67aafaf6ad781b2dc9c576a2387`.
