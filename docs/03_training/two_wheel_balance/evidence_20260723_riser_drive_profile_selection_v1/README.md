# Riser drive profile selection v1

This CPU-only audit makes the current plant identity explicit:

- Active simulation profile: Leadshine 400 W engineering sample, `300 N`
  transient simulation cap, `1.0 m/s`, and the existing 400 W thermal monitor.
- Production-design candidate: Leadshine 750 W motor and drive, calculated
  `550.259 N` rated force and `1650.777 N` peak force, but not enabled in the
  URDF, Isaac actuator, thermal model, runtime, or training pipeline.

The profile cannot be switched with an environment variable or command-line
flag. Activating the 750 W candidate requires supplier and bench evidence,
coordinated source and asset changes, and complete dynamic requalification.
Existing dynamic evidence, corrective captures, and BC checkpoints are not
reusable after a plant-profile switch.

Summary SHA-256:
`39a700de3985175e4e8415f1f23beef4264b103daa7ce8847f4ac0fe69f879f7`.

Audit-script SHA-256:
`f8bcea857b84104fb5cdbf79aab7b3681fd569aac2ff4e0bf6b6a01e75443eff`.
