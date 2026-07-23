#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
D3D12_DEFAULT_WIN='G:\isaaclab\apps\isaaclab.python.headless.rendering.d3d12.kit'
D3D12_DEFAULT_WSL='/mnt/g/isaaclab/apps/isaaclab.python.headless.rendering.d3d12.kit'
D3D12_EXPERIENCE="${ISAAC_D3D12_EXPERIENCE:-$D3D12_DEFAULT_WIN}"
D3D12_EXPERIENCE_WSL="${ISAAC_D3D12_EXPERIENCE_WSL:-$D3D12_DEFAULT_WSL}"
PREFLIGHT="$ROOT/scripts/two_wheel_balance/"
PREFLIGHT+="validate_model_based_learned_render_admission.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
MEDIA_AUDITOR="$ROOT/scripts/two_wheel_balance/"
MEDIA_AUDITOR+="audit_model_based_learned_render_media.py"
MODE="${1:---preflight}"

reject() {
  printf '{"passed":false,"reason":"%s","runtime_started":false}\n' "$1" >&2
  exit "${2:-6}"
}

[[ "$MODE" == --preflight || "$MODE" == --execute || "$MODE" == --resume ]] \
  || reject invalid_mode 2
required_environment=(
  RISER_MODEL_BASED_LEARNED_RENDER_ADMISSION
  RISER_MODEL_BASED_LEARNED_RENDER_ALL79_REPORT
  RISER_MODEL_BASED_LEARNED_RENDER_ALL79_ADMISSION
  RISER_MODEL_BASED_LEARNED_RENDER_ALL79_PREFLIGHT
  RISER_MODEL_BASED_LEARNED_RENDER_POLICY
  RISER_MODEL_BASED_LEARNED_RENDER_PLAN_MANIFEST
  RISER_MODEL_BASED_LEARNED_RENDER_SOURCE_MANIFEST
  RISER_MODEL_BASED_LEARNED_RENDER_LQR_GAINS
  RISER_MODEL_BASED_LEARNED_RENDER_ROBOT_BUILD_AUDIT
  RISER_MODEL_BASED_LEARNED_RENDER_ROBOT_USD
  RISER_MODEL_BASED_LEARNED_RENDER_DRIVE_PROFILE_SELECTION
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || reject "missing_environment:$name" 2
done

receipt="$(mktemp -p "$ROOT" .learned_render_preflight.XXXXXX.json)"
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
  --admission "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ADMISSION")" \
  --all79-report \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ALL79_REPORT")" \
  --all79-admission \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ALL79_ADMISSION")" \
  --all79-preflight \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ALL79_PREFLIGHT")" \
  --policy "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_POLICY")" \
  --plan-manifest \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_PLAN_MANIFEST")" \
  --source-manifest \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_SOURCE_MANIFEST")" \
  --lqr-gains "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_LQR_GAINS")" \
  --robot-build-audit \
  "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ROBOT_BUILD_AUDIT")" \
  --robot-usd "$(to_windows_path "$RISER_MODEL_BASED_LEARNED_RENDER_ROBOT_USD")" \
  --drive-profile-selection \
  "$(to_windows_path \
    "$RISER_MODEL_BASED_LEARNED_RENDER_DRIVE_PROFILE_SELECTION")" \
  --require-authorized \
  --output "$(to_windows_path "$receipt")"
if [[ "$MODE" == --preflight ]]; then
  cat "$receipt"
  exit 0
fi

namespace="${RISER_MODEL_BASED_LEARNED_RENDER_NAMESPACE:-}"
[[ "$namespace" =~ ^[A-Za-z0-9_.-]+$ ]] || reject invalid_or_missing_namespace 2
output="$ROOT/artifacts/two_wheel_riser/$namespace"
if [[ "$MODE" == --execute ]]; then
  [[ ! -e "$output" ]] || reject runtime_namespace_already_exists 3
else
  [[ -d "$output" ]] || reject runtime_namespace_missing_for_resume 3
  cmp -s "$RISER_MODEL_BASED_LEARNED_RENDER_ADMISSION" \
    "$output/admission.json" || reject resume_admission_mismatch 3
  cmp -s "$receipt" "$output/preflight.json" \
    || reject resume_preflight_mismatch 3
fi
[[ -x "$ISAAC_PYTHON" ]] || reject missing_isaac_python 2
[[ -f "$D3D12_EXPERIENCE_WSL" ]] || reject missing_d3d12_experience 2
command -v ffprobe >/dev/null 2>&1 || reject missing_ffprobe 2
command -v nvidia-smi >/dev/null 2>&1 || reject missing_nvidia_smi 2
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] \
  || reject gpu_compute_owner_present 4
[[ -z "$(pgrep -af '[s]moke_riser_reference_playback.py' || true)" ]] \
  || reject playback_owner_present 4

mkdir -p "$output/rollouts" "$output/videos" "$output/logs"
if [[ "$MODE" == --execute ]]; then
  cp "$receipt" "$output/preflight.json"
  cp "$RISER_MODEL_BASED_LEARNED_RENDER_ADMISSION" "$output/admission.json"
fi
plan_dir="$(dirname "$RISER_MODEL_BASED_LEARNED_RENDER_PLAN_MANIFEST")"
plan_win="$(wslpath -w "$plan_dir")"
gains_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_RENDER_LQR_GAINS")"
policy_win="$(wslpath -w "$RISER_MODEL_BASED_LEARNED_RENDER_POLICY")"
playback_win="$(wslpath -w "$PLAYBACK")"
output_win="$(wslpath -w "$output")"
cases=(1 15 31 50 73 79)

rollout_is_valid() {
  local path="$1" case_number="$2"
  [[ -s "$path" ]] || return 1
  python3 - "$path" "$case_number" <<'PY'
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("cases") == [case]
    and payload.get("passed") is True
    and payload.get("trajectory_command_source")
    == "model_based_planner_plus_torchscript_residual"
    and payload.get("tracking_profile")
    == "riser_recovery_direction_v4_camera_lever_arm_v1"
    and payload.get("policy_command_base") == "model_based_planner"
    and payload.get("residual_action_scales") == [0.05, 0.05, 0.02]
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("case") == case
    and payload["results"][0].get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
}

case_video() {
  find "$output/videos/case_$(printf '%04d' "$1")" -maxdepth 1 \
    -type f -name '*.mp4' | sort | tail -1
}

for case_number in "${cases[@]}"; do
  padded="$(printf '%04d' "$case_number")"
  rollout="$output/rollouts/case_$padded.json"
  video="$(case_video "$case_number" || true)"
  if [[ -e "$rollout" || -n "$video" ]]; then
    rollout_is_valid "$rollout" "$case_number" \
      && [[ -n "$video" && -s "$video" ]] \
      || reject "invalid_existing_render_case_$padded" 5
    continue
  fi
  mkdir -p "$output/videos/case_$padded"
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
    --video-dir "$output_win\\videos\\case_$padded" \
    --video-frame-stride 8 \
    --video-fps 25 \
    --output "$output_win\\rollouts\\case_$padded.json" \
    --headless --enable_cameras --experience "$D3D12_EXPERIENCE" \
    >"$output/logs/case_$padded.log" 2>&1
  video="$(case_video "$case_number" || true)"
  rollout_is_valid "$rollout" "$case_number" \
    && [[ -n "$video" && -s "$video" ]] \
    || reject "render_case_${padded}_failed" 5
done

media_args=(
  --admission "$output/admission.json"
  --preflight "$output/preflight.json"
  --policy "$RISER_MODEL_BASED_LEARNED_RENDER_POLICY"
  --all79-report "$RISER_MODEL_BASED_LEARNED_RENDER_ALL79_REPORT"
  --output "$output/media_manifest.json"
)
for case_number in "${cases[@]}"; do
  padded="$(printf '%04d' "$case_number")"
  media_args+=(
    --case-rollout "$case_number=$output/rollouts/case_$padded.json"
    --case-video "$case_number=$(case_video "$case_number")"
  )
done
python3 "$MEDIA_AUDITOR" "${media_args[@]}"
printf 'media passed; explicit visual review remains required: %s\n' "$output"
