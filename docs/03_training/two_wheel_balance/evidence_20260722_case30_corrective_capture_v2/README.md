# Case 30 corrective capture v2

This directory preserves the sole authorized v2 case-30 corrective capture.
The one-use runtime commit was
`ca755bdcf4498fe39f19735ed666e96ca11bed96`. The external mode-`0600`
authorization token was consumed before Isaac started and was absent after the
run. No second capture, conversion, BC, PPO, or training run was launched.

The finalizer passed every admission, dynamic, thermal, controller,
perturbation, heartbeat, GPU-release, and archive check. The archive is
admitted only for the separately reviewed CPU conversion step; it is not a
normalized training dataset and is not valid for training.

Key evidence:

- 11,411 aligned samples with 65 observation features and three residual
  channels.
- Source and execution clocks end at 18.144412 s and 29.22248819392579 s.
- Requested normalized action maxima are `[0.394906, 0.657500, 0.900000]`.
- Effective post-supervisor maxima are `[0.394906, 0.657500, 0.267339]`.
- Per-channel command clipping affects `[200, 308, 333]` rows and remains
  explicit in the archive.
- Exactly 20 perturbation-active rows are present; initialization rows are
  excluded.
- Peak position error is 0.159790 m and peak pitch is 7.045321 degrees.
- Capture SHA-256 is
  `ec0f13030ce755c38e31c138507537f461126312b0c268832bc6bf9a40e4e8cb`.

Use `SHA256SUMS` to verify the preserved files.

The complete runtime logs remain in the sealed `.98` namespace. Their
SHA-256 values are `3238e6d347a55d1d38ca9fb08c165711ea8373ee960eb2a783e2286c8016e55f`
for playback and
`e0b9ec3186e677c34289a85e72e4bc91e3cd3d8ce5cfdea16d74e1c0be0554b2`
for finalization.
