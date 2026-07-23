# Case 8 validation-pair readiness

This CPU-only audit binds the first selected validation case to the validation
tranche rather than the train tranche.

The selected case-8 plan preserves `663` source anchors and `662` transitions.
Its source duration is `12.940941 s`; its separately retimed execution duration
is `18.1173174 s`. Camera height remains within
`0.600000-1.605452 m`.

The existing zero-residual dynamic gate passes with position p95/max
`0.131254/0.143331 m`, peak pitch `6.147057 deg`, no termination, zero applied
residual, and no training. The residual-label envelope remains below the
frozen `0.95` screen.

Four low-motion windows are available. The longest is `3.431994 s`. However,
camera lever-arm correction saturation is `0.920061`, so case 8 requires its
own conservative validation profile. Reuse of case-30, case-23, case-6,
case-2, or case-7 profiles is forbidden.

The local readiness regression suite passes `23 passed, 2 warnings in
0.45 s`. The same focused suite on `.98` passes `23 passed, 2 warnings in
1.45 s`, and regenerates byte-identical evidence at SHA-256
`8ba7d6613b53cca7f266cb0052680b2d0cc4a71e363e10abbadcb8c789526983`.
The authoritative `.98` CPU suite passes
`1232 passed, 12 skipped, 2 warnings in 151.02 s` at implementation commit
`155affbd7348d6eecae541a08a6908cc31e397ed`.

No validation runtime, holdout, authorization token, GPU launch, label
capture, conversion, merge, BC, PPO, or training is opened.
