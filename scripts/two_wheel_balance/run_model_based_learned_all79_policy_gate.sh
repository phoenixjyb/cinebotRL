#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
PREFLIGHT="$ROOT/scripts/two_wheel_balance/validate_model_based_learned_all79_admission.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
MODE="${1:---preflight}"

reject() {
  printf '{"passed":false,"reason":"%s","runtime_started":false}\n' "$1" >&2
  exit "${2:-6}"
}

[[ "$MODE" == --preflight || "$MODE" == --execute || "$MODE" == --resume ]] \
  || reject unsupported_mode 2

required_environment=(
  RISER_MODEL_BASED_LEARNED_ALL79_ADMISSION
  RISER_MODEL_BASED_LEARNED_ALL79_BC_REPORT
  RISER_MODEL_BASED_LEARNED_ALL79_POLICY
  RISER_MODEL_BASED_LEARNED_ALL79_PLAN_MANIFEST
  RISER_MODEL_BASED_LEARNED_ALL79_SOURCE_MANIFEST
  RISER_MODEL_BASED_LEARNED_ALL79_LQR_GAINS
  RISER_MODEL_BASED_LEARNED_ALL79_ROBOT_BUILD_AUDIT
  RISER_MODEL_BASED_LEARNED_ALL79_ROBOT_USD
  RISER_MODEL_BASED_LEARNED_ALL79_DRIVE_PROFILE_SELECTION
  RISER_MODEL_BASED_LEARNED_ALL79_VALIDATION_REPORT
  RISER_MODEL_BASED_LEARNED_ALL79_HOLDOUT_REPORT
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || reject "missing_environment:$name" 2
done

receipt="$(mktemp -p "$ROOT" .learned_all79_preflight.XXXXXX.json)"
trap 'rm -f "$receipt"' EXIT
to_windows_path() {
  if [[ "$1" =~ ^[A-Za-z]:\\ ]]; then
    printf '%s\n' "$1"
  else
    wslpath -w "$1"
  fi
}
RISER_GIT_ROOT_WSL="$ROOT" \
  WSLENV="${WSLENV:+${WSLENV}:}RISER_GIT_ROOT_WSL" \
  "$ISAAC_PYTHON" "$(to_windows_path "$PREFLIGHT")" \
  --admission "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_ADMISSION")" \
  --bc-report "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_BC_REPORT")" \
  --policy "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_POLICY")" \
  --plan-manifest \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_PLAN_MANIFEST")" \
  --source-manifest \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_SOURCE_MANIFEST")" \
  --lqr-gains "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_LQR_GAINS")" \
  --robot-build-audit \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_ROBOT_BUILD_AUDIT")" \
  --robot-usd "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_ROBOT_USD")" \
  --drive-profile-selection \
  "$(to_windows_path \
    "$RISER_MODEL_BASED_LEARNED_ALL79_DRIVE_PROFILE_SELECTION")" \
  --validation-gate-report \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_VALIDATION_REPORT")" \
  --holdout-gate-report \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_ALL79_HOLDOUT_REPORT")" \
  --require-authorized \
  --output "$(to_windows_path "$receipt")"

if [[ "$MODE" == --preflight ]]; then
  cat "$receipt"
  exit 0
fi

namespace="${RISER_MODEL_BASED_LEARNED_ALL79_NAMESPACE:-}"
[[ "$namespace" =~ ^[A-Za-z0-9_.-]+$ ]] || reject invalid_or_missing_namespace 2
output="$ROOT/artifacts/two_wheel_riser/$namespace"
if [[ "$MODE" == --execute ]]; then
  [[ ! -e "$output" ]] || reject runtime_namespace_already_exists 3
else
  [[ -d "$output" ]] || reject runtime_namespace_missing_for_resume 3
  cmp -s "$RISER_MODEL_BASED_LEARNED_ALL79_ADMISSION" \
    "$output/admission.json" || reject resume_admission_mismatch 3
  cmp -s "$receipt" "$output/preflight.json" \
    || reject resume_preflight_mismatch 3
fi
[[ -x "$ISAAC_PYTHON" ]] || reject missing_isaac_python 2
command -v nvidia-smi >/dev/null 2>&1 || reject missing_nvidia_smi 2
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] \
  || reject gpu_compute_owner_present 4
[[ -z "$(pgrep -af '[s]moke_riser_reference_playback.py' || true)" ]] \
  || reject playback_owner_present 4

mkdir -p "$output/teacher" "$output/learned" "$output/logs"
if [[ "$MODE" == --execute ]]; then
  cp "$receipt" "$output/preflight.json"
  cp "$RISER_MODEL_BASED_LEARNED_ALL79_ADMISSION" "$output/admission.json"
fi

plan_dir="$(dirname "$RISER_MODEL_BASED_LEARNED_ALL79_PLAN_MANIFEST")"
plan_win="$(wslpath -w "$plan_dir")"
gains_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_ALL79_LQR_GAINS")"
policy_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_ALL79_POLICY")"
output_win="$(wslpath -w "$output")"
execution_commit="$(git -C "$ROOT" rev-parse HEAD)"

rollout_is_valid() {
  local path="$1"
  local case_number="$2"
  local source="$3"
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$case_number" "$source" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("cases") == [case]
    and payload.get("trajectory_command_source") == sys.argv[3]
    and payload.get("tracking_profile")
    == "riser_recovery_direction_v4_camera_lever_arm_v1"
    and payload.get("phase_feedforward_contract")
    == "derivatives_scaled_by_progress_v1"
    and payload.get("policy_command_base") == "model_based_planner"
    and payload.get("residual_action_scales") == [0.05, 0.05, 0.02]
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
  teacher="$output/teacher/case_$padded.json"
  learned="$output/learned/case_$padded.json"
  if [[ -e "$teacher" ]] && ! rollout_is_valid \
    "$teacher" "$case_number" model_based_planner_plus_zero_policy_residual; then
    reject "invalid_existing_teacher_case_$padded" 5
  fi
  if [[ ! -e "$teacher" ]]; then
    timeout --signal=TERM --kill-after=30s 1800 \
      "$ISAAC_PYTHON" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$gains_win" \
    --plan-dir "$plan_win" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$case_number" \
    --controller-wz-kp 1.05 \
    --maximum-duration-scale 3.0 \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain 1.0 \
    --maximum-camera-lever-arm-correction-m 0.05 \
    --residual-action-scales 0.05,0.05,0.02 \
    --policy-command-base model_based_planner \
    --zero-policy-action \
    --output "$output_win\\teacher\\case_$padded.json" \
    --headless >"$output/logs/teacher_case_$padded.log" 2>&1
    rollout_is_valid \
      "$teacher" "$case_number" model_based_planner_plus_zero_policy_residual \
      || reject "teacher_case_${padded}_failed_gate" 5
  fi
  if [[ -e "$learned" ]] && ! rollout_is_valid \
    "$learned" "$case_number" model_based_planner_plus_torchscript_residual; then
    reject "invalid_existing_learned_case_$padded" 5
  fi
  if [[ ! -e "$learned" ]]; then
    timeout --signal=TERM --kill-after=30s 1800 \
      "$ISAAC_PYTHON" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$gains_win" \
    --plan-dir "$plan_win" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$case_number" \
    --controller-wz-kp 1.05 \
    --maximum-duration-scale 3.0 \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain 1.0 \
    --maximum-camera-lever-arm-correction-m 0.05 \
    --residual-action-scales 0.05,0.05,0.02 \
    --policy-command-base model_based_planner \
    --residual-policy "$policy_win" \
    --residual-policy-device cuda \
    --output "$output_win\\learned\\case_$padded.json" \
    --headless >"$output/logs/learned_case_$padded.log" 2>&1
    rollout_is_valid \
      "$learned" "$case_number" model_based_planner_plus_torchscript_residual \
      || reject "learned_case_${padded}_failed_gate" 5
  fi
done

python3 "$GATE" \
  --mode all79 \
  --teacher-dir "$output/teacher" \
  --learned-dir "$output/learned" \
  --cases "$(seq -s, 1 79)" \
  --policy "$RISER_MODEL_BASED_LEARNED_ALL79_POLICY" \
  --expected-tracking-profile \
  riser_recovery_direction_v4_camera_lever_arm_v1 \
  --policy-command-contract \
  model_based_planner_plus_bounded_policy_residual_v1 \
  --rollout-admission "$output/admission.json" \
  --preflight-receipt "$output/preflight.json" \
  --plan-manifest "$RISER_MODEL_BASED_LEARNED_ALL79_PLAN_MANIFEST" \
  --execution-commit "$execution_commit" \
  --output "$output/summary.json"

printf 'model-based learned all-79 gate passed: %s\n' "$output"
