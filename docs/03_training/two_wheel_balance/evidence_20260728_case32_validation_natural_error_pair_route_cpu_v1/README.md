# Case 32 Validation Natural-Error Pair Route CPU Evidence

This directory seals the CPU-only preflight for the disabled-by-default case-32
held-out validation pair route.

- Implementation commit: `904cbf087cd8be0f89e4da53fb042ad5ad58bb3a`
- Reviewed parent: `da20ebb9158a9251e3919f3e6b04032c17827afe`
- Case/split: `32` / `validation`
- Identity count: `27`
- Runtime authorization token issued: `false`
- Isaac/GPU runtime started: `false`
- Teacher, label, or dataset created: `false`
- BC, PPO, or training started: `false`

The canonical preflight passed every repository, identity, document, route, and
closed-learning check with `HEAD == upstream`. A deliberate tokenless
`--execute` attempt failed closed before Python or Isaac with exit code `4`,
and the runtime namespace was not created.

This evidence makes the route separately reviewable. It does not authorize a
runtime canary, label capture, dataset conversion, merge, BC, PPO, or training.
