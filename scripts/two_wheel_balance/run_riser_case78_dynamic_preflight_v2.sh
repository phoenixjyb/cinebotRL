#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="20260721_case78_dynamic_qualification_v2_heartbeat_exclusive"
readonly CONTRACT_RELATIVE="scripts/two_wheel_balance/case78_dynamic_cpu_contract_v2.json"

reject() {
  local reason="$1"
  local code="${2:-7}"
  python3 - "$reason" <<'PY' >&2
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_case78_dynamic_rejection_v2",
    "reason": sys.argv[1],
    "case": 78,
    "runtime_authorized": False,
    "gpu_launch_authorized": False,
    "dynamic_qualification_started": False,
    "split_changed": False,
    "dataset_created": False,
    "dagger_authorized": False,
    "bc_authorized": False,
    "ppo_authorized": False,
    "passed": False,
}, indent=2))
PY
  exit "$code"
}

protected_variables=(
  RISER_ROOT
  RISER_WIN_ROOT
  ISAAC_PYTHON
  RISER_CASE78_AUTHORIZATION
  RISER_CASE78_NAMESPACE
  RISER_CASE78_CONTRACT
  RISER_CASE78_PLAN
  RISER_CASE78_OUTPUT
  RISER_CASE78_TIMEOUT
  RISER_CASE78_HEARTBEAT
)
for variable in "${protected_variables[@]}"; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" != --execute ]] || reject "runtime_authorization_not_issued"
[[ "$MODE" == --preflight ]] || reject "unsupported_mode:$MODE" 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTRACT="$ROOT/$CONTRACT_RELATIVE"
VALIDATOR="$SCRIPT_DIR/validate_riser_case78_dynamic_contract_v2.py"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader
  )"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "Stop"
      $queryProcessId = $PID
      Get-CimInstance Win32_Process |
        Where-Object {
          $_.ProcessId -ne $queryProcessId -and (
            $_.Name -eq "kit.exe" -or
            $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc"
          )
        } |
        ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$NAMESPACE" \
  --output "$ADMISSION" >/dev/null
assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5

python3 - "$ADMISSION" <<'PY'
import json
from pathlib import Path
import sys
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result["ownership_preflight_passed"] = True
result["runtime_started"] = False
print(json.dumps(result, indent=2))
PY

