# Model-Based Corrective Temporal Projection Audit

This CPU-only audit evaluates the admitted case-30 corrective dataset without
opening BC, PPO, holdouts, learned rollout, or any Isaac runtime.

The requested corrective-teacher intent obeys the physical slew limits
`[0.10, 0.10, 0.04]` with zero violations. Effective post-supervisor labels
contain `30/49/8` per-channel slew violations across `87` transitions, but
every violation touches a channel clipped by the deterministic command
supervisor. They are projection discontinuities, not teacher-command chatter.

`model_based_residual_safety_projection_v1` reconstructs the recorded final
commands within `1.2e-7` and effective normalized actions within `3.9e-6`,
including the exact clipping mask. A future model-based BC contract must:

- keep effective post-supervisor residuals as pointwise targets;
- treat network outputs as requested bounded residuals;
- project outputs through the deterministic supervisor before pointwise loss;
- regularize or gate requested-output slew independently;
- never naively reject clipped effective-label transitions as teacher chatter.

The summary is valid only for BC contract review. It is not valid for training
and does not authorize dataset merge, BC, PPO, or learned runtime.
