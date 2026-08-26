# Film Brain Isaac self-collision fixtures

These fixtures exercise the experimental, opt-in Isaac Lab worker. Parse a
fixture and re-encode it as sorted compact JSON without a trailing newline
before sending it to `isaac_self_collision_worker.py`.

The evaluation fixture contains one expected clear configuration and one known
self-colliding configuration. It is a discrete self-collision check only. A
successful result does not establish continuous collision freedom, environment
collision freedom, minimum clearance, PnC model equivalence, robot transport,
motion authority, or physical feasibility.

Run `IDENTITY` first and compare its USD and policy hashes with the evaluation
fixture. Update a pinned hash only after reviewing the changed model or policy.
