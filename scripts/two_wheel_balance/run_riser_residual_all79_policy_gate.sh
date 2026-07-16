#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
DATASET_STAMP="${RISER_DATASET_STAMP:-20260716_residual_all79_phase_v3_clean}"
POLICY_STAMP="${RISER_POLICY_STAMP:-20260716_residual_bc_phase_v3_clean}"
HOLDOUT_STAMP="${RISER_ROLLOUT_STAMP:-20260716_residual_holdout_phase_v3_clean}"
ALL79_STAMP="${RISER_POLICY_ALL79_STAMP:-20260716_residual_policy_all79_phase_v3_clean}"
PLAN_STAMP="${RISER_ALL79_PLAN_STAMP:-20260716_all79_playback_inputs_v3}"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
HOLDOUT_ROOT="$ROOT/artifacts/two_wheel_riser/$HOLDOUT_STAMP"
ALL79_ROOT="$ROOT/artifacts/two_wheel_riser/$ALL79_STAMP"
ALL79_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$ALL79_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
SCRIPT_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
GATE_SCRIPT="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
POLICY_FILE="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_REPORT="$POLICY_ROOT/report.json"
HOLDOUT_SUMMARY="$HOLDOUT_ROOT/summary.json"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -s "$POLICY_FILE" && -s "$POLICY_REPORT" && -s "$HOLDOUT_SUMMARY" ]] || {
  printf 'offline or holdout policy gate is incomplete\n' >&2
  exit 2
}

POLICY_HASH="$({ python3 - "$POLICY_FILE" "$POLICY_REPORT" "$HOLDOUT_SUMMARY" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

policy = Path(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
holdout = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
digest = hashlib.sha256(policy.read_bytes()).hexdigest()
checks = {
    "offline_gate": report.get("offline_gate_passed") is True,
    "holdout_gate": holdout.get("passed") is True,
    "report_hash": report.get("torchscript_sha256") == digest,
    "holdout_hash": holdout.get("policy_sha256") == digest,
    "holdout_count": holdout.get("case_count") == 8,
    "ppo_not_authorized": holdout.get("ppo_authorized") is False,
}
if not all(checks.values()):
    raise SystemExit(f"all-79 policy admission rejected: {checks}")
print(digest)
PY
})"
POLICY_COMMIT="$(python3 - "$POLICY_REPORT" <<'PY'
import json
from pathlib import Path
import re
import sys

commit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get(
    "source_commit"
)
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit("policy report has no valid source commit")
print(commit)
PY
)"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$POLICY_COMMIT" ]] || {
  printf 'all-79 evaluator must use the policy source commit: %s\n' "$POLICY_COMMIT" >&2
  exit 2
}
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make all-79 provenance ambiguous\n' >&2
  exit 2
}

mkdir -p "$ALL79_ROOT/gates" "$ALL79_ROOT/logs"
python3 - "$ALL79_ROOT/admission.json" "$POLICY_HASH" "$POLICY_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = {
    "schema": "cinebotrl_two_wheel_riser_policy_all79_admission_v1",
    "cases": list(range(1, 80)),
    "policy_sha256": sys.argv[2],
    "git_commit": sys.argv[3],
}
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise SystemExit("all-79 admission differs from existing partial run")
else:
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
PY

gate_is_resumable() {
  local case_number="$1"
  local padded="$2"
  local gate="$ALL79_ROOT/gates/case_$padded.json"
  [[ -s "$gate" ]] || return 1
  python3 - "$gate" "$case_number" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("cases") == [case]
    and payload.get("trajectory_command_source") == "torchscript_residual_policy"
    and payload.get("tracking_profile") == "riser_phase_consistent_v2"
    and payload.get("phase_feedforward_contract")
    == "derivatives_scaled_by_progress_v1"
    and payload.get("passed") is True
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("case") == case
    and payload["results"][0].get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
}

for case_number in $(seq 1 79); do
  padded="$(printf '%04d' "$case_number")"
  if gate_is_resumable "$case_number" "$padded"; then
    printf 'learned case %s already passed\n' "$padded"
    continue
  fi
  printf 'running learned case %s\n' "$padded"
  "$PY" -u -X utf8 "$SCRIPT_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" --cases "$case_number" \
    --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
    --output "$ALL79_WIN\\gates\\case_$padded.json" --headless \
    >"$ALL79_ROOT/logs/case_$padded.log" 2>&1 || {
      tail -n 80 "$ALL79_ROOT/logs/case_$padded.log" >&2
      exit 4
    }
done

python3 "$GATE_SCRIPT" \
  --mode all79 \
  --teacher-dir "$DATASET_ROOT/gates" \
  --learned-dir "$ALL79_ROOT/gates" \
  --cases "$(seq -s, 1 79)" \
  --policy "$POLICY_FILE" \
  --output "$ALL79_ROOT/summary.json"

printf 'learned all-79 policy gate passed: %s\n' "$ALL79_ROOT"
