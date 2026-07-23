# Case 23 corrective conversion execution CPU preflight v2

This directory preserves the corrected no-token, no-write CPU preflight for
the one-shot case-23 v4 corrective conversion route. The preflight ran on
`.98` at clean synchronized commit
`298805562202320c72319f7adb0f955fd9568116`.

The original execution route correctly required a mode-`0600` token, but the
`.98` `/mnt/c` and `/mnt/g` DrvFS mounts report files as mode `0777` and cannot
host such a token. Commit `2988055` adds the only supported secure path:

- the token is created outside the repository on WSL ext4, under
  `/home/yanbo`;
- WSL verifies mode `0600`, non-symlink status, and the out-of-band hash;
- Windows Python reads the same file through the pinned Ubuntu
  `\\wsl.localhost` path;
- alternate WSL distributions and ordinary DrvFS tokens fail closed.

The real `.98` focused suite passes with `9 passed, 2 warnings`. Every
repository, contract, and identity check in this preflight is true, and
`cpu_contract_ready` is true.

No authorization token was issued. `conversion_authorized`, output creation,
corpus merge, BC, PPO, and training all remain false. The production
conversion namespace is absent.

The preflight summary SHA-256 is
`b1d76609bac8982d3bd4af818c15e18ad58dc88b24093617a8fe90018069f739`.
