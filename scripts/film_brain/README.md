# Film Brain Isaac collision tools

This directory contains an opt-in Isaac Lab experiment for sampled Proto2
self-collision checks. Normal CineBotRL training and evaluation remain unchanged:
the environment enables the extra PhysX policy and pair sensors only when the
Film Brain worker or probe sets its private process flags.

`isaac_self_collision_worker.py` accepts one canonical JSON object on stdin and
emits one canonical JSON object on stdout without a trailing newline. Run the
`IDENTITY` operation first, review its USD and collision-policy hashes, then pin
those hashes in an evaluation request. The fixtures under
`tests/fixtures/film_brain_isaac_self_collision` contain one clear pose and one
known colliding pose. Fixture files must be parsed and re-encoded as sorted,
compact JSON before invocation.

`isaac_self_collision_probe.py` is a broader diagnostic used to inventory the
USD, tune invariant-pair filters, and scan deterministic joint configurations.
It is not the Cloud worker protocol.

Both tools are simulator-only. They do not evaluate swept/continuous collision,
environment collision, minimum clearance, PnC model equivalence, robot
transport, motion authority, or physical feasibility. The worker also requires
the derived `ee1_level_pitch` coordinate to be zero until model equivalence is
proved.

On the validated RTX PRO 4000 Blackwell host, a cold worker call took roughly
6–9 seconds. A caller must use a bounded timeout above that measured cold-start
latency; the existing 5-second Cloud worker envelope is not yet suitable.
