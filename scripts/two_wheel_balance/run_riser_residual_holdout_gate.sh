#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
DATASET_STAMP="${RISER_DATASET_STAMP:-20260717_residual_all79_exact_source_v1}"
POLICY_STAMP="${RISER_POLICY_STAMP:-20260717_residual_bc_exact_source_v1}"
ROLLOUT_STAMP="${RISER_ROLLOUT_STAMP:-20260717_residual_holdout_exact_source_v1}"
PLAN_STAMP="${RISER_ALL79_PLAN_STAMP:-20260717_all79_playback_exact_source_v1}"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
ROLLOUT_ROOT="$ROOT/artifacts/two_wheel_riser/$ROLLOUT_STAMP"
ROLLOUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$ROLLOUT_STAMP"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
SCRIPT_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
GATE_SCRIPT="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
DATASET_SUMMARY="$DATASET_ROOT/summary.json"
POLICY_REPORT="$POLICY_ROOT/report.json"
POLICY_FILE="$POLICY_ROOT/residual_policy.torchscript.pt"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -s "$DATASET_SUMMARY" && -s "$POLICY_REPORT" && -s "$POLICY_FILE" ]] || {
  printf 'dataset or offline policy admission is incomplete\n' >&2
  exit 2
}
EVAL_CASES="$({ python3 - "$DATASET_SUMMARY" "$POLICY_REPORT" "$POLICY_FILE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

dataset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
policy = Path(sys.argv[3])
split_cases = dataset.get("dataset", {}).get("split_cases", {})
cases = sorted(set(split_cases.get("holdout", [])))
checks = {
    "dataset_passed": dataset.get("passed") is True,
    "dataset_79": dataset.get("passed_case_count") == 79,
    "offline_gate": report.get("offline_gate_passed") is True,
    "rollout_authorized": report.get("learned_rollout_authorized") is True,
    "policy_hash": report.get("torchscript_sha256")
    == hashlib.sha256(policy.read_bytes()).hexdigest(),
    "eval_case_count": len(cases) == 8,
}
if not all(checks.values()):
    raise SystemExit(f"holdout admission rejected: {checks}")
print(",".join(str(case) for case in cases))
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
  printf 'holdout evaluator must use the policy source commit: %s\n' "$POLICY_COMMIT" >&2
  exit 2
}
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make holdout provenance ambiguous\n' >&2
  exit 2
}

mkdir -p "$ROLLOUT_ROOT/zero" "$ROLLOUT_ROOT/learned" "$ROLLOUT_ROOT/logs"
python3 - "$ROLLOUT_ROOT/admission.json" "$EVAL_CASES" "$POLICY_FILE" "$POLICY_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = {
    "schema": "cinebotrl_two_wheel_riser_holdout_admission_v1",
    "cases": [int(item) for item in sys.argv[2].split(",")],
    "policy_sha256": hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
    "git_commit": sys.argv[4],
}
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise SystemExit("rollout admission differs from existing partial run")
else:
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
PY

rollout_is_resumable() {
  local mode="$1"
  local case_number="$2"
  local padded="$3"
  local expected_source="$4"
  local gate="$ROLLOUT_ROOT/$mode/case_$padded.json"
  [[ -s "$gate" ]] || return 1
  python3 - "$gate" "$case_number" "$expected_source" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("cases") == [case]
    and payload.get("trajectory_command_source") == sys.argv[3]
    and payload.get("tracking_profile") == "riser_phase_consistent_v2"
    and payload.get("phase_feedforward_contract")
    == "derivatives_scaled_by_progress_v1"
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("case") == case
)
raise SystemExit(0 if valid else 1)
PY
}

for case_number in ${EVAL_CASES//,/ }; do
  padded="$(printf '%04d' "$case_number")"
  if rollout_is_resumable zero "$case_number" "$padded" zero_policy_action_baseline; then
    printf 'zero-policy-action case %s already captured\n' "$padded"
  else
    printf 'running zero-policy-action case %s\n' "$padded"
    "$PY" -u -X utf8 "$SCRIPT_WIN" \
      --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" --cases "$case_number" \
      --zero-policy-action \
      --output "$ROLLOUT_WIN\\zero\\case_$padded.json" --headless \
      >"$ROLLOUT_ROOT/logs/zero_case_$padded.log" 2>&1 || true
  fi
  [[ -s "$ROLLOUT_ROOT/zero/case_$padded.json" ]] || {
    printf 'zero-policy-action case did not produce a gate: %s\n' "$padded" >&2
    exit 4
  }

  if rollout_is_resumable learned "$case_number" "$padded" torchscript_residual_policy; then
    printf 'learned case %s already captured\n' "$padded"
  else
    printf 'running learned case %s\n' "$padded"
    "$PY" -u -X utf8 "$SCRIPT_WIN" \
      --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" --cases "$case_number" \
      --residual-policy "$POLICY_WIN" --residual-policy-device cuda \
      --output "$ROLLOUT_WIN\\learned\\case_$padded.json" --headless \
      >"$ROLLOUT_ROOT/logs/learned_case_$padded.log" 2>&1 || {
        tail -n 80 "$ROLLOUT_ROOT/logs/learned_case_$padded.log" >&2
        exit 5
      }
  fi
done

python3 "$GATE_SCRIPT" \
  --mode holdout \
  --teacher-dir "$DATASET_ROOT/gates" \
  --zero-dir "$ROLLOUT_ROOT/zero" \
  --learned-dir "$ROLLOUT_ROOT/learned" \
  --cases "$EVAL_CASES" \
  --policy "$POLICY_FILE" \
  --output "$ROLLOUT_ROOT/summary.json"

printf 'case-disjoint residual rollout gate passed: %s\n' "$ROLLOUT_ROOT"
