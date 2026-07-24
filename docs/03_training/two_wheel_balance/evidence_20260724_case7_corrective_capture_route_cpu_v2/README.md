# Case-7 corrective capture route v2

This CPU-only evidence repairs the pending case-7 capture route after the
shared archive finalizer gained explicit split support.

Commit `2743c64f8c118dd3a549968fb603ae64ec9c59d1` changes only the
`capture_finalizer_runtime` SHA-256 and Git blob identity in the existing
case-7 capture contract. No controller argument, plan, corrective profile,
wrench profile, dynamic gate, namespace, authorization rule, or runtime
command changed.

The canonical `.98` preflight passes every identity and contract check. A
tokenless `--execute` still exits `4` with
`runtime_authorization_not_issued`, and the capture namespace remains absent.

This route is ready for a separately authorized single case-7 corrective-label
capture. It does not itself authorize runtime, capture, conversion, corpus
merge, BC, PPO, or training.
