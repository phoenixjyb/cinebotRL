# Case-7 corrective capture v2

This evidence preserves the one authorized case-7 corrective-label capture at
runtime commit `d0365653571d50523584e80e2ec1943febdfe6d4`.

SolidWorks and Siemens NX remained active. The launch admission passed with
`5.385 GiB` free Windows RAM and `10,491 MiB` free GPU memory. The five-second
runtime monitor collected `46` samples; minimum observed headroom was
`2.836 GiB` RAM and `8,072 MiB` GPU memory. No pressure termination was
requested, and the monitored process exited cleanly.

The rollout completed all `6,597` steps and the full `18.117 s` execution
clock. Position p95/max error was `0.1249/0.1411 m`; attitude p95/max error
was `0.150/0.224 deg`; maximum pitch was `6.138 deg`. Action, riser, and
proxy saturation ratios were zero, and no attitude IK failure or runtime
termination occurred.

The finalizer reopened the archive and admitted the capture for a separate CPU
conversion. This evidence does not authorize conversion, corpus merge, BC,
PPO, holdout execution, or training, and the capture remains
`valid_for_training=false`.
