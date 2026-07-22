#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly NAMESPACE="20260722_model_based_corrective_teacher_case30_pair_v1_exclusive"
readonly CONTRACT_RELATIVE="model_based_corrective_teacher_case30_pair_contract_v1.json"
readonly VALIDATOR_RELATIVE="validate_model_based_corrective_teacher_case30_pair.py"
readonly CONTRACT="$ROOT/scripts/two_wheel_balance/$CONTRACT_RELATIVE"
readonly VALIDATOR="$ROOT/scripts/two_wheel_balance/$VALIDATOR_RELATIVE"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

reject() {
  printf '{"reason":"%s","runtime_started":false,"passed":false}\n' "$1" >&2
  exit "${2:-7}"
}

for variable in RISER_ROOT RISER_WIN_ROOT ISAAC_PYTHON \
  RISER_CORRECTIVE_CASE30_NAMESPACE RISER_CORRECTIVE_CASE30_CONTRACT \
  RISER_CORRECTIVE_CASE30_PROFILE RISER_CORRECTIVE_CASE30_PLAN \
  RISER_CORRECTIVE_CASE30_PERTURBATION RISER_CORRECTIVE_CASE30_OUTPUT \
  RISER_CORRECTIVE_CASE30_AUTHORIZATION_FILE; do
  [[ -z "${!variable+x}" ]] || reject "conflicting_environment_override:$variable"
done

MODE="${1:---preflight}"
[[ "$MODE" == --preflight || "$MODE" == --execute ]] || reject "unsupported_mode" 2
[[ "$MODE" != --execute ]] || reject "runtime_authorization_not_issued" 4

assert_gpu_free() {
  local wsl_owners compute_owners windows_owners
  wsl_owners="$(
    ps -ef | grep -E '[p]ython(\.exe)? .*(smoke_.*playback|train_riser_residual_bc)\.py' || true
  )"
  compute_owners="$(
    "$NVIDIA_SMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true
  )"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command '
      $queryProcessId = $PID
      Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $queryProcessId -and (
          $_.Name -eq "kit.exe" -or
          $_.CommandLine -match "smoke_.*playback|train_riser_residual_bc"
        )
      } | ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  [[ -z "$wsl_owners" && -z "$compute_owners" && -z "$windows_owners" ]]
}

ADMISSION="$(mktemp)"
trap 'rm -f "$ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --contract "$CONTRACT" \
  --repo-root "$ROOT" \
  --namespace "$NAMESPACE" \
  --output "$ADMISSION" >/dev/null
assert_gpu_free || reject "exclusive_gpu_ownership_failed" 5
cat "$ADMISSION"
