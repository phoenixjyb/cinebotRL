#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
D3D12_EXPERIENCE="${ISAAC_D3D12_EXPERIENCE:-G:\\isaaclab\\apps\\isaaclab.python.headless.rendering.d3d12.kit}"
D3D12_EXPERIENCE_WSL="${ISAAC_D3D12_EXPERIENCE_WSL:-/mnt/g/isaaclab/apps/isaaclab.python.headless.rendering.d3d12.kit}"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MODE="${1:---preflight}"
CASE=4
PADDED=0004
PROFILE="riser_recovery_direction_v4_camera_lever_arm_v1"
PLAN_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
CAPTURE_STAMP="20260721_initial_teacher42_raw_capture_v1_exclusive"
DATASET_STAMP="20260721_initial_teacher41_subset_30_5_5_v1"
POLICY_STAMP="20260721_initial_teacher40_bc_v1"
OUTPUT_STAMP="20260721_initial_teacher40_bc_case4_rendered_canary_v3"
PLAN_ROOT="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
CAPTURE_ROOT="$ROOT/artifacts/two_wheel_riser/$CAPTURE_STAMP"
DATASET_ROOT="$ROOT/artifacts/two_wheel_riser/$DATASET_STAMP"
POLICY_ROOT="$ROOT/artifacts/two_wheel_riser/$POLICY_STAMP"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$OUTPUT_STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$OUTPUT_STAMP"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
ROLLOUT_GATE="$ROOT/scripts/two_wheel_balance/gate_riser_residual_rollouts.py"
PLAN="$PLAN_ROOT/case_${PADDED}_smoothed_riser_plan_v1.npz"
PLAN_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP"
TEACHER_GATE="$CAPTURE_ROOT/gates/case_${PADDED}.json"
DATASET_SUMMARY="$DATASET_ROOT/initial_teacher40_30_5_5_v1.summary.json"
POLICY_FINAL="$POLICY_ROOT/final_status.json"
POLICY_REPORT="$POLICY_ROOT/report.json"
POLICY="$POLICY_ROOT/residual_policy.torchscript.pt"
POLICY_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$POLICY_STAMP\\residual_policy.torchscript.pt"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
PLAN_SHA256="16e962e57b906d18561cc8640c4788c719bf95817492896c252affe6920e3ddb"
MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
TEACHER_GATE_SHA256="245f592ec81eee562a1e0dc4afb278c77c3b5d04046b1362701d75d4c995d79c"
DATASET_SUMMARY_SHA256="815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542"
POLICY_FINAL_SHA256="05771143d93cbce0abd20124a787c1d7348cc482238b3fdb52dd6ad2ea038cdf"
POLICY_REPORT_SHA256="44437ed005aa69718c244fda8c8fddb58ebf95c308c9387d11d85e4ff62ce104"
POLICY_SHA256="6d86812d3ef63093e00d938e8aa4120146dd30f52dce007af569c3ece989d1dd"
PLAYBACK_SHA256="ff078ee6e6ed6cbb23d814547f4c4cb275c238a995c4cb1110796c49dc4e4904"
ROLLOUT_GATE_SHA256="e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
POLICY_COMMIT="d093b10bda36ef8dfbb588d07ee0359a2680401b"
SCALE_BINDING_COMMIT="0ca3e2df89ab9efd421896750683d78ec9b0f3fb"
AUTHORIZATION_SHA256="9e7975341a5396b69c9f08ca38a3401d85b264d5338438863dc8a6158142f0be"

if [[ "$MODE" != --preflight && "$MODE" != --execute ]]; then
  printf 'usage: %s [--preflight|--execute]\n' "$0" >&2
  exit 2
fi
[[ -x "$PY" && -x /usr/bin/ffmpeg ]] || {
  printf 'missing Isaac Python or ffmpeg\n' >&2
  exit 2
}
[[ -f "$D3D12_EXPERIENCE_WSL" ]] || {
  printf 'missing D3D12 render experience: %s\n' "$D3D12_EXPERIENCE_WSL" >&2
  exit 2
}

sha256() { sha256sum "$1" | awk '{print $1}'; }
identity_matches() { [[ -s "$1" && "$(sha256 "$1")" == "$2" ]]; }

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

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 180); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  printf 'GPU did not release within 180 seconds\n' >&2
  return 1
}

HEAD="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{upstream}')"
[[ "$HEAD" == "$UPSTREAM" ]] || { printf 'HEAD is not pushed\n' >&2; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is not clean\n' >&2
  exit 3
}
git -C "$ROOT" merge-base --is-ancestor "$POLICY_COMMIT" "$HEAD"
git -C "$ROOT" merge-base --is-ancestor "$SCALE_BINDING_COMMIT" "$HEAD"
identity_matches "$PLAN" "$PLAN_SHA256"
identity_matches "$PLAN_ROOT/manifest.json" "$MANIFEST_SHA256"
identity_matches "$TEACHER_GATE" "$TEACHER_GATE_SHA256"
identity_matches "$DATASET_SUMMARY" "$DATASET_SUMMARY_SHA256"
identity_matches "$POLICY_FINAL" "$POLICY_FINAL_SHA256"
identity_matches "$POLICY_REPORT" "$POLICY_REPORT_SHA256"
identity_matches "$POLICY" "$POLICY_SHA256"
identity_matches "$PLAYBACK" "$PLAYBACK_SHA256"
identity_matches "$ROLLOUT_GATE" "$ROLLOUT_GATE_SHA256"
identity_matches "$GAINS" "$GAINS_SHA256"
identity_matches "$ROBOT_USD" "$ROBOT_USD_SHA256"

python3 - "$DATASET_SUMMARY" "$POLICY_FINAL" "$POLICY_REPORT" \
  "$TEACHER_GATE" "$POLICY_SHA256" "$PROFILE" <<'PY'
import json
from pathlib import Path
import sys

dataset, final, report, teacher = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:5]
]
policy_hash, profile = sys.argv[5:]
checks = {
    "case_is_validation": 4 in dataset.get("split_cases", {}).get("validation", []),
    "holdout_untouched": 4 not in dataset.get("split_cases", {}).get("holdout", []),
    "frozen_scales": dataset.get("action_scales") == [0.35000000000000003, 0.4, 0.1],
    "policy_gate": final.get("passed") is True
    and final.get("learned_rollout_authorized") is True
    and final.get("learned_rollout_started") is False,
    "ppo_closed": final.get("ppo_authorized") is False
    and final.get("ppo_started") is False,
    "policy_hash": report.get("torchscript_sha256") == policy_hash,
    "teacher_case": teacher.get("cases") == [4]
    and teacher.get("passed") is True
    and teacher.get("tracking_profile") == profile,
    "teacher_source": teacher.get("trajectory_command_source")
    == "deterministic_teacher",
}
if not all(checks.values()):
    raise SystemExit(f"rendered canary admission rejected: {checks}")
PY

if [[ "$MODE" == --preflight ]]; then
  python3 - "$HEAD" "$OUTPUT" <<'PY'
import json
import sys
print(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_bc_case4_render_preflight_v1",
    "runtime_commit": sys.argv[1],
    "case": 4,
    "split": "validation",
    "modes": ["zero_policy_action_baseline", "torchscript_residual_policy"],
    "residual_action_scales": [0.35, 0.4, 0.1],
    "video_frame_stride": 8,
    "video_fps_raw": 25,
    "video_fps_delivery": 50,
    "output": sys.argv[2],
    "holdout_opened": False,
    "ppo_authorized": False,
    "runtime_started": False,
    "passed": True,
}, indent=2))
PY
  exit 0
fi

AUTHORIZATION_FILE="${RISER_CASE4_RENDER_AUTHORIZATION_FILE:-}"
[[ -n "$AUTHORIZATION_FILE" && -f "$AUTHORIZATION_FILE" ]] || {
  printf 'missing one-use rendered-canary authorization file\n' >&2
  exit 4
}
[[ "$(stat -c '%a' "$AUTHORIZATION_FILE")" == 600 ]] || exit 4
[[ "$(sha256 "$AUTHORIZATION_FILE")" == "$AUTHORIZATION_SHA256" ]] || exit 4
[[ ! -e "$OUTPUT" ]] || {
  printf 'refusing to overwrite rendered canary: %s\n' "$OUTPUT" >&2
  exit 5
}
assert_gpu_free || { printf 'GPU is not exclusive\n' >&2; exit 5; }

mkdir -p "$OUTPUT/zero" "$OUTPUT/learned" "$OUTPUT/logs" \
  "$OUTPUT/videos/zero_raw" "$OUTPUT/videos/learned_raw" "$OUTPUT/videos/delivery"
python3 - "$OUTPUT/admission.json" "$HEAD" "$POLICY" "$PLAN" \
  "$TEACHER_GATE" "$PLAYBACK" "$AUTHORIZATION_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_bc_case4_render_admission_v1",
    "runtime_commit": sys.argv[2],
    "case": 4,
    "split": "validation",
    "policy": identity(sys.argv[3]),
    "plan": identity(sys.argv[4]),
    "teacher_gate": identity(sys.argv[5]),
    "playback": identity(sys.argv[6]),
    "authorization_sha256": sys.argv[7],
    "residual_action_scales": [0.35, 0.4, 0.1],
    "zero_rollout_authorized": True,
    "learned_rollout_authorized": True,
    "holdout_opened": False,
    "ppo_authorized": False,
    "passed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
rm -f "$AUTHORIZATION_FILE"

run_rollout() {
  local name="$1" extra="$2" status=0
  assert_gpu_free || return 5
  timeout --signal=TERM --kill-after=30s 1500 \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PLAN_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" --controller-wz-kp 1.05 --maximum-duration-scale 3.0 \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain 1.0 \
    --maximum-camera-lever-arm-correction-m 0.05 \
    --residual-action-scales 0.35,0.40,0.10 \
    $extra --video-dir "$OUTPUT_WIN\\videos\\${name}_raw" \
    --video-frame-stride 8 --video-fps 25 \
    --output "$OUTPUT_WIN\\$name\\case_${PADDED}.json" \
    --headless --enable_cameras --experience "$D3D12_EXPERIENCE" \
    >"$OUTPUT/logs/${name}.log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$OUTPUT/logs/${name}.exit_code"
  wait_for_gpu_release || return 5
  [[ -s "$OUTPUT/$name/case_${PADDED}.json" ]] || return 6
  find "$OUTPUT/videos/${name}_raw" -maxdepth 1 -type f -name '*.mp4' | grep -q . || return 6
  return 0
}

run_rollout zero "--zero-policy-action"
run_rollout learned "--residual-policy $POLICY_WIN --residual-policy-device cuda" || true

ZERO_RAW="$(find "$OUTPUT/videos/zero_raw" -maxdepth 1 -type f -name '*.mp4' | sort | tail -1)"
LEARNED_RAW="$(find "$OUTPUT/videos/learned_raw" -maxdepth 1 -type f -name '*.mp4' | sort | tail -1)"
[[ -s "$ZERO_RAW" && -s "$LEARNED_RAW" ]]
ZERO_50="$OUTPUT/videos/delivery/case_0004_zero_baseline_50fps.mp4"
LEARNED_50="$OUTPUT/videos/delivery/case_0004_bc_residual_50fps.mp4"
COMPARE="$OUTPUT/videos/delivery/case_0004_baseline_vs_bc_residual_50fps.mp4"
/usr/bin/ffmpeg -hide_banner -loglevel error -y -i "$ZERO_RAW" \
  -vf fps=50 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$ZERO_50"
/usr/bin/ffmpeg -hide_banner -loglevel error -y -i "$LEARNED_RAW" \
  -vf fps=50 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$LEARNED_50"
/usr/bin/ffmpeg -hide_banner -loglevel error -y -i "$ZERO_50" -i "$LEARNED_50" \
  -filter_complex \
  "[0:v]scale=960:-2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Baseline - zero residual':x=24:y=24:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[z];[1:v]scale=960:-2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='BC residual policy':x=24:y=24:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[l];[z][l]hstack=inputs=2:shortest=1[v]" \
  -map '[v]' -r 50 -c:v libx264 -crf 20 -pix_fmt yuv420p \
  -movflags +faststart "$COMPARE"

GATE_STATUS=0
python3 "$ROLLOUT_GATE" --mode validation_canary \
  --teacher-dir "$CAPTURE_ROOT/gates" --zero-dir "$OUTPUT/zero" \
  --learned-dir "$OUTPUT/learned" --cases "$CASE" --policy "$POLICY" \
  --expected-tracking-profile "$PROFILE" --maximum-regression-fraction 0.05 \
  --minimum-zero-improvement-fraction 0.05 --output "$OUTPUT/summary.json" \
  >"$OUTPUT/logs/rollout_gate.log" 2>&1 || GATE_STATUS=$?
printf '%s\n' "$GATE_STATUS" >"$OUTPUT/logs/rollout_gate.exit_code"

python3 - "$OUTPUT" "$HEAD" "$GATE_STATUS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
status = int(sys.argv[3])
def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
zero = root / "videos/delivery/case_0004_zero_baseline_50fps.mp4"
learned = root / "videos/delivery/case_0004_bc_residual_50fps.mp4"
comparison = root / "videos/delivery/case_0004_baseline_vs_bc_residual_50fps.mp4"
checks = {
    "rollout_gate_exit_zero": status == 0,
    "rollout_gate_passed": summary.get("passed") is True,
    "validation_case_only": summary.get("cases") == [4],
    "profile_bound": summary.get("expected_tracking_profile")
    == "riser_recovery_direction_v4_camera_lever_arm_v1",
    "all_videos_present": zero.is_file() and learned.is_file() and comparison.is_file(),
}
passed = all(checks.values())
(root / "final_status.json").write_text(json.dumps({
    "schema": "cinebotrl_two_wheel_riser_initial_bc_case4_render_final_v1",
    "runtime_commit": sys.argv[2],
    "checks": checks,
    "admission": identity(root / "admission.json"),
    "zero_gate": identity(root / "zero/case_0004.json"),
    "learned_gate": identity(root / "learned/case_0004.json"),
    "summary": identity(summary_path) if summary_path.is_file() else None,
    "zero_video": identity(zero),
    "learned_video": identity(learned),
    "comparison_video": identity(comparison),
    "holdout_opened": False,
    "ppo_authorized": False,
    "passed": passed,
}, indent=2) + "\n", encoding="utf-8")
PY

exit "$GATE_STATUS"
