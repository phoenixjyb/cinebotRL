# Case-7 Corrective Capture Route CPU Evidence

This checkpoint prepares, but does not authorize, one case-7 corrective-label
capture.

The route is pinned to the successful case-7 paired canary, its projection
audit, the exact plan, drive profile, controller assets, robot assets, and
capture/finalizer code. The `.98` tokenless preflight verified all 19
identities and left runtime, GPU, label capture, dataset creation, BC, PPO, and
training closed.

Verification:

- Local focused tests: `13 passed, 2 warnings in 0.19s`.
- `.98` focused tests: `13 passed, 2 warnings in 0.65s`.
- `.98` full CPU suite: `1382 passed, 12 skipped, 2 warnings in 221.90s`.
- `preflight_windows.json` SHA-256:
  `7ead7a8277c853960ae89bb78a5217df26af4d145c1f0e57f061f8a19eb4490c`.

No runtime namespace or authorization token was created. The next
data-producing operation requires the separate instruction:
`Authorize exactly one case-7 corrective-label capture.`
