# Case 16 validation-pair readiness

This CPU-only audit binds the second selected validation case to its exact
v12 plan and exclusive zero-residual Gate C result.

The plan preserves `896` source anchors and `895` transitions, with separate
`17.548706 s` source and `26.028630 s` execution clocks. Camera height remains
within `1.339969-1.392495 m`.

The zero-residual gate passes at `0.080600/0.081492 m` position p95/max and
`6.030922 deg` peak pitch. The playback applied no residual and produced no
training artifact.

Case 16 is not suitable for the case-8 pulse profile. Base linear velocity,
yaw rate, and proxy rate all touch their frozen limits, camera lever-arm
correction saturation reaches `0.958736`, and no low-motion window lasts
`0.10 s`. Therefore
`safe_window_absent_requires_structural_profile=true`.

The next bounded task is a CPU-only structural natural-error profile that
uses the existing trajectory error without adding an external wrench. No
validation runtime, authorization token, capture, conversion, merge, BC, PPO,
or training is opened.
