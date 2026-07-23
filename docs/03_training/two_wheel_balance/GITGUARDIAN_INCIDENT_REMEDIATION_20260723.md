# GitGuardian Incident Remediation

Date: 2026-07-23

## Scope and Finding

The exact GitGuardian detector and incident URL were not available in the
connected mailbox or GitHub status APIs. The branch audit isolated the most
likely trigger to a recently committed, high-entropy SHA-256 value named as a
one-use runtime authorization token hash.

The value was not a GitHub, cloud, SSH, or application credential, and the
underlying runtime token was never committed. It was nevertheless unsuitable
for Git because its name and entropy resemble a credential and because runtime
authorization material should not be durable repository state.

## Remediation

- Rewrote the nine affected branch commits from the first introduction of the
  value through the previous branch tip.
- Replaced the consumed hash in rejected historical evidence with an explicit
  redaction marker while retaining the rejection outcome.
- Kept the rejected capture namespace, logs, and no-training evidence.
- Changed the case-23 v2 authorization contract so the token and its lowercase
  SHA-256 are supplied out of band only at execution time.
- Required the token to be a non-symlink mode-`0600` file outside the
  repository and used constant-time hash comparison.
- Added ignore rules for local runtime-secret paths and authorization-token
  files.
- Added a Gitleaks GitHub Actions gate for pushes, pull requests, manual runs,
  and a daily full-history scan.

No case-23 runtime authorization token is issued by this remediation. Isaac,
GPU launch, label capture, conversion, BC, PPO, and training remain closed.

## Verification

Before the rewritten branch is published:

- `gitleaks git . --log-opts=codex/two-wheel-riser-rl` scanned 822 commits and
  approximately 48.37 MB with no leaks.
- `gitleaks dir .` scanned the current worktree and found no leaks.
- An exact-history search for the isolated incident value returned no result on
  the rewritten local branch.
- Focused case-23 authorization and recursive-goal tests passed.

The prior remote branch tip must be replaced only with an exact
`--force-with-lease`. A fresh single-branch clone must then pass Gitleaks before
the incident is considered remediated at the branch level.

## Operational Follow-up

Git history removal does not revoke a real credential. If the GitGuardian
incident identifies any provider-issued credential rather than the isolated
authorization digest, that credential must be revoked and rotated immediately.
The incident can be resolved in GitGuardian only after checking its detector,
file, commit, and validity details against this audit.
