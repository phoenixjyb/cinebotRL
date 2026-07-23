#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
PREFLIGHT="$ROOT/scripts/two_wheel_balance/validate_model_based_learned_split_admission.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
SPLIT_MODE="${1:-}"
RUN_MODE="${2:---preflight}"

reject() {
  printf '{"passed":false,"reason":"%s","runtime_started":false}\n' "$1" >&2
  exit "${2:-6}"
}

[[ "$SPLIT_MODE" == validation_canary || "$SPLIT_MODE" == holdout ]] \
  || reject invalid_split_mode 2
[[ "$RUN_MODE" == --preflight || "$RUN_MODE" == --execute \
  || "$RUN_MODE" == --resume ]] || reject invalid_run_mode 2

required_environment=(
  RISER_MODEL_BASED_LEARNED_SPLIT_ADMISSION
  RISER_MODEL_BASED_LEARNED_SPLIT_BC_REPORT
  RISER_MODEL_BASED_LEARNED_SPLIT_POLICY
  RISER_MODEL_BASED_LEARNED_SPLIT_PLAN_MANIFEST
  RISER_MODEL_BASED_LEARNED_SPLIT_SOURCE_MANIFEST
  RISER_MODEL_BASED_LEARNED_SPLIT_LQR_GAINS
  RISER_MODEL_BASED_LEARNED_SPLIT_ROBOT_BUILD_AUDIT
  RISER_MODEL_BASED_LEARNED_SPLIT_ROBOT_USD
  RISER_MODEL_BASED_LEARNED_SPLIT_DRIVE_PROFILE_SELECTION
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || reject "missing_environment:$name" 2
done
if [[ "$SPLIT_MODE" == holdout ]]; then
  [[ -n "${RISER_MODEL_BASED_LEARNED_SPLIT_PRIOR_VALIDATION_REPORT:-}" ]] \
    || reject missing_prior_validation_report 2
fi

receipt="$(mktemp -p "$ROOT" .learned_split_preflight.XXXXXX.json)"
trap 'rm -f "$receipt"' EXIT
to_windows_path() {
  if [[ "$1" =~ ^[A-Za-z]:\\ ]]; then
    printf '%s\n' "$1"
  else
    wslpath -w "$1"
  fi
}
preflight_win="$(to_windows_path "$PREFLIGHT")"
preflight_args=(
  --mode "$SPLIT_MODE"
  --admission "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_ADMISSION")"
  --bc-report "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_BC_REPORT")"
  --policy "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_POLICY")"
  --plan-manifest \
    "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_PLAN_MANIFEST")"
  --source-manifest \
    "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_SOURCE_MANIFEST")"
  --lqr-gains "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_LQR_GAINS")"
  --robot-build-audit \
    "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_ROBOT_BUILD_AUDIT")"
  --robot-usd "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_SPLIT_ROBOT_USD")"
  --drive-profile-selection \
    "$(to_windows_path \
      "$RISER_MODEL_BASED_LEARNED_SPLIT_DRIVE_PROFILE_SELECTION")"
  --require-authorized
  --output "$(to_windows_path "$receipt")"
)
if [[ "$SPLIT_MODE" == holdout ]]; then
  preflight_args+=(
    --prior-validation-gate-report
    "$(to_windows_path \
      "$RISER_MODEL_BASED_LEARNED_SPLIT_PRIOR_VALIDATION_REPORT")"
  )
fi
"$ISAAC_PYTHON" "$preflight_win" "${preflight_args[@]}"

if [[ "$RUN_MODE" == --preflight ]]; then
  cat "$receipt"
  exit 0
fi

namespace="${RISER_MODEL_BASED_LEARNED_SPLIT_NAMESPACE:-}"
[[ "$namespace" =~ ^[A-Za-z0-9_.-]+$ ]] || reject invalid_or_missing_namespace 2
output="$ROOT/artifacts/two_wheel_riser/$namespace"
if [[ "$RUN_MODE" == --execute ]]; then
  [[ ! -e "$output" ]] || reject runtime_namespace_already_exists 3
else
  [[ -d "$output" ]] || reject runtime_namespace_missing_for_resume 3
  cmp -s "$RISER_MODEL_BASED_LEARNED_SPLIT_ADMISSION" \
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

mkdir -p "$output/baseline" "$output/learned" "$output/logs"
if [[ "$RUN_MODE" == --execute ]]; then
  cp "$receipt" "$output/preflight.json"
  cp "$RISER_MODEL_BASED_LEARNED_SPLIT_ADMISSION" "$output/admission.json"
fi

plan_dir="$(dirname "$RISER_MODEL_BASED_LEARNED_SPLIT_PLAN_MANIFEST")"
plan_win="$(wslpath -w "$plan_dir")"
gains_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_SPLIT_LQR_GAINS")"
policy_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_SPLIT_POLICY")"
playback_win="$(wslpath -w "$PLAYBACK")"
output_win="$(wslpath -w "$output")"
execution_commit="$(git -C "$ROOT" rev-parse HEAD)"
cases="$(python3 - "$receipt" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(str(case) for case in payload["cases"]))
PY
)"

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

for case_number in ${cases//,/ }; do
  padded="$(printf '%04d' "$case_number")"
  baseline="$output/baseline/case_$padded.json"
  learned="$output/learned/case_$padded.json"
  if [[ -e "$baseline" ]] && ! rollout_is_valid \
    "$baseline" "$case_number" model_based_planner_plus_zero_policy_residual; then
    reject "invalid_existing_baseline_case_$padded" 5
  fi
  if [[ ! -e "$baseline" ]]; then
    timeout --signal=TERM --kill-after=30s 1800 \
      "$ISAAC_PYTHON" -u -X utf8 "$playback_win" \
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
      --output "$output_win\\baseline\\case_$padded.json" \
      --headless >"$output/logs/baseline_case_$padded.log" 2>&1
    rollout_is_valid \
      "$baseline" "$case_number" model_based_planner_plus_zero_policy_residual \
      || reject "baseline_case_${padded}_failed_gate" 5
  fi
  if [[ -e "$learned" ]] && ! rollout_is_valid \
    "$learned" "$case_number" model_based_planner_plus_torchscript_residual; then
    reject "invalid_existing_learned_case_$padded" 5
  fi
  if [[ ! -e "$learned" ]]; then
    timeout --signal=TERM --kill-after=30s 1800 \
      "$ISAAC_PYTHON" -u -X utf8 "$playback_win" \
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
  --mode "$SPLIT_MODE" \
  --teacher-dir "$output/baseline" \
  --zero-dir "$output/baseline" \
  --learned-dir "$output/learned" \
  --cases "$cases" \
  --policy "$RISER_MODEL_BASED_LEARNED_SPLIT_POLICY" \
  --expected-tracking-profile \
  riser_recovery_direction_v4_camera_lever_arm_v1 \
  --policy-command-contract \
  model_based_planner_plus_bounded_policy_residual_v1 \
  --rollout-admission "$output/admission.json" \
  --preflight-receipt "$output/preflight.json" \
  --plan-manifest "$RISER_MODEL_BASED_LEARNED_SPLIT_PLAN_MANIFEST" \
  --execution-commit "$execution_commit" \
  --output "$output/summary.json"

printf 'model-based learned %s gate passed: %s\n' "$SPLIT_MODE" "$output"
