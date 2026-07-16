#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
STAMP="${RISER_ALL79_STAMP:-20260716_residual_all79_v1}"
ARTIFACTS_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
ARTIFACTS_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$STAMP"
PLAN_DIR_WIN="${RISER_ALL79_PLAN_DIR_WIN:-$WIN_ROOT\\artifacts\\two_wheel_riser\\20260716_all79_playback_inputs}"
GAINS_WIN="${RISER_GAINS_WIN:-$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json}"
SCRIPT_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
MERGER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\build_riser_residual_dataset.py"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -d "$ROOT/artifacts/two_wheel_riser/20260716_all79_playback_inputs" ]] || {
  printf 'missing all-79 playback inputs\n' >&2
  exit 2
}
mkdir -p "$ARTIFACTS_WSL/cases" "$ARTIFACTS_WSL/gates" "$ARTIFACTS_WSL/logs"

gate_is_resumable() {
  local case_number="$1"
  local padded="$2"
  local gate="$ARTIFACTS_WSL/gates/case_$padded.json"
  local dataset="$ARTIFACTS_WSL/cases/case_${padded}_executed_residual_v1.npz"
  [[ -s "$gate" && -s "$dataset" ]] || return 1
  python3 - "$gate" "$case_number" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("passed") is True
    and payload.get("passed_case_count") == 1
    and payload.get("cases") == [case]
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
}

for case_number in $(seq 1 79); do
  padded="$(printf '%04d' "$case_number")"
  if gate_is_resumable "$case_number" "$padded"; then
    printf 'case %s already passed; resuming\n' "$padded"
    continue
  fi
  printf 'running case %s\n' "$padded"
  "$PY" -u -X utf8 "$SCRIPT_WIN" \
    --gains "$GAINS_WIN" \
    --plan-dir "$PLAN_DIR_WIN" \
    --cases "$case_number" \
    --dataset-dir "$ARTIFACTS_WIN\\cases" \
    --output "$ARTIFACTS_WIN\\gates\\case_$padded.json" \
    --headless >"$ARTIFACTS_WSL/logs/case_$padded.log" 2>&1 || {
      tail -n 80 "$ARTIFACTS_WSL/logs/case_$padded.log" >&2
      printf 'case %s process failed\n' "$padded" >&2
      exit 3
    }
  if ! gate_is_resumable "$case_number" "$padded"; then
    tail -n 80 "$ARTIFACTS_WSL/logs/case_$padded.log" >&2
    printf 'case %s dynamic or dataset gate failed\n' "$padded" >&2
    exit 4
  fi
  printf 'case %s passed\n' "$padded"
done

"$PY" -u -X utf8 "$MERGER_WIN" \
  --case-dir "$ARTIFACTS_WIN\\cases" \
  --output "$ARTIFACTS_WIN\\all79_residual_dataset_v1.npz" \
  --expected-count 79 >"$ARTIFACTS_WSL/merge.log" 2>&1

python3 - "$ARTIFACTS_WSL" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
commit = sys.argv[2]
gates = []
for path in sorted((root / "gates").glob("case_*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    gates.append(
        {
            "case": result["case"],
            "completed_steps": result["completed_steps"],
            "position_error_p95_m": result["position_error_p95_m"],
            "position_error_max_m": result["position_error_max_m"],
            "attitude_error_p95_deg": result["attitude_error_p95_deg"],
            "attitude_error_max_deg": result["attitude_error_max_deg"],
            "pitch_max_deg": result["pitch_max_deg"],
            "passed": result["passed"],
        }
    )
dataset_summary = json.loads(
    (root / "all79_residual_dataset_v1.summary.json").read_text(encoding="utf-8")
)
summary = {
    "schema": "cinebotrl_two_wheel_riser_all79_dynamic_dataset_gate_v1",
    "git_commit": commit,
    "training_started": False,
    "ppo_authorized": False,
    "case_count": len(gates),
    "passed_case_count": sum(item["passed"] for item in gates),
    "total_steps": sum(item["completed_steps"] for item in gates),
    "worst_metrics": {
        key: max(item[key] for item in gates)
        for key in (
            "position_error_p95_m",
            "position_error_max_m",
            "attitude_error_p95_deg",
            "attitude_error_max_deg",
            "pitch_max_deg",
        )
    },
    "dataset": dataset_summary,
    "cases": gates,
    "passed": len(gates) == 79 and all(item["passed"] for item in gates),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["passed"] else 5)
PY

printf 'all-79 dynamic residual dataset gate passed: %s\n' "$ARTIFACTS_WSL"
