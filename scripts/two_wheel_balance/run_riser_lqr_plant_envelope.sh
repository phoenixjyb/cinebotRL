#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
readonly WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
readonly PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
readonly NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
readonly POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
readonly REVIEWED_EVALUATOR_PARENT="0dc1aad417aecdc0ed1b29110d08e7abf5db0622"
readonly EVALUATOR_SHA256="f5aa042326c85f0b46bb5de872664cd7cd42b4d7aa2ecaedc6961e55aa84a570"
readonly METRICS_SHA256="5041b711eb9d8026f2086898443d8ff4c4fba64662b9b051d27ced5181a51c2e"
readonly TRACKING_SHA256="f2cfe1abcaf1f225461b0ccfa9d26ab205e2dc9624047be77edeab9f67f754e7"
readonly ROBOT_CONFIG_SHA256="31b84c21baf8fb043c8653bfb592217b97d9edbd7a1a0e17633b86f4f36f05e2"
readonly GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
readonly ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
readonly TIMEOUT_SECONDS=420

if [[ $# -ne 1 ]]; then
  printf 'usage: %s low|mid|high\n' "$0" >&2
  exit 7
fi

readonly SHARD="$1"
case "$SHARD" in
  low)
    readonly RISER_POSITION_M="0.0"
    readonly AUTHORIZATION="AUTHORIZED_RISER_LQR_PLANT_ENVELOPE_VXKP072_LOW_V1"
    readonly NAMESPACE="20260720_riser_lqr_plant_envelope_vxkp072_low_v1_exclusive"
    ;;
  mid)
    readonly RISER_POSITION_M="0.6"
    readonly AUTHORIZATION="AUTHORIZED_RISER_LQR_PLANT_ENVELOPE_VXKP072_MID_V1"
    readonly NAMESPACE="20260720_riser_lqr_plant_envelope_vxkp072_mid_v1_exclusive"
    ;;
  high)
    readonly RISER_POSITION_M="1.2"
    readonly AUTHORIZATION="AUTHORIZED_RISER_LQR_PLANT_ENVELOPE_VXKP072_HIGH_V1"
    readonly NAMESPACE="20260720_riser_lqr_plant_envelope_vxkp072_high_v1_exclusive"
    ;;
  *)
    printf 'unknown riser plant-envelope shard: %s\n' "$SHARD" >&2
    exit 7
    ;;
esac

protected_variables=(
  RISER_ROOT
  RISER_WIN_ROOT
  ISAAC_PYTHON
  RISER_PLANT_ENVELOPE_NAMESPACE
  RISER_PLANT_ENVELOPE_POSITION_M
  RISER_PLANT_ENVELOPE_NUM_ENVS
  RISER_PLANT_ENVELOPE_TIMEOUT_SECONDS
  RISER_PLANT_ENVELOPE_OUTPUT
)
for variable in "${protected_variables[@]}"; do
  if [[ -n "${!variable+x}" ]]; then
    printf 'riser plant-envelope contract rejects environment override: %s\n' "$variable" >&2
    exit 7
  fi
done

if [[ "${RISER_PLANT_ENVELOPE_AUTHORIZATION:-}" != "$AUTHORIZATION" ]]; then
  printf 'riser plant-envelope %s authorization is absent or unknown\n' "$SHARD" >&2
  exit 7
fi

readonly EVALUATOR="$ROOT/scripts/two_wheel_balance/evaluate_lqr_tracking_push.py"
readonly EVALUATOR_WIN="$WIN_ROOT\scripts\two_wheel_balance\evaluate_lqr_tracking_push.py"
readonly METRICS="$ROOT/src/rl_platform/tasks/two_wheel_balance/metrics.py"
readonly TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
readonly ROBOT_CONFIG="$ROOT/src/rl_platform/robots/two_wheel_balance/config.py"
readonly GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
readonly GAINS_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json"
readonly ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
readonly RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_lqr_plant_envelope.sh"
readonly OUTPUT="$ROOT/artifacts/two_wheel_riser/$NAMESPACE"
readonly OUTPUT_WIN="$WIN_ROOT\artifacts\two_wheel_riser\$NAMESPACE"

assert_resources_free() {
  local wsl_owners windows_owners compute_owners
  wsl_owners="$(ps -ef | grep -E '[e]valuate_lqr_tracking_push\.py|[s]moke_.*playback\.py|[t]rain_.*\.py' || true)"
  windows_owners="$(
    "$POWERSHELL" -NoProfile -NonInteractive -Command \
      '$p="evaluate_lqr_tracking_push.py|smoke_.*playback.py|train_.*\.py"; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match $p } | ForEach-Object { "$($_.ProcessId) $($_.CommandLine)" }' \
      2>/dev/null | tr -d '\r' || true
  )"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$wsl_owners" || -n "$windows_owners" || -n "$compute_owners" ]]; then
    printf 'riser plant-envelope resources are not exclusive\n' >&2
    [[ -z "$wsl_owners" ]] || printf 'WSL:\n%s\n' "$wsl_owners" >&2
    [[ -z "$windows_owners" ]] || printf 'Windows:\n%s\n' "$windows_owners" >&2
    [[ -z "$compute_owners" ]] || printf 'NVIDIA:\n%s\n' "$compute_owners" >&2
    return 1
  fi
}

wait_for_release() {
  local attempt
  for attempt in $(seq 1 90); do
    if assert_resources_free 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  printf 'riser plant-envelope owner did not release within 90 seconds\n' >&2
  assert_resources_free
}

[[ -x "$PY" ]] || { printf 'missing Isaac Python\n' >&2; exit 2; }
[[ -x "$NVIDIA_SMI" ]] || { printf 'missing WSL NVIDIA ownership probe\n' >&2; exit 2; }
[[ -x "$POWERSHELL" ]] || { printf 'missing Windows ownership probe\n' >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 2; }
[[ "$(sha256sum "$EVALUATOR" | awk '{print $1}')" == "$EVALUATOR_SHA256" ]] || { printf 'evaluator hash mismatch\n' >&2; exit 2; }
[[ "$(sha256sum "$METRICS" | awk '{print $1}')" == "$METRICS_SHA256" ]] || { printf 'controller hash mismatch\n' >&2; exit 2; }
[[ "$(sha256sum "$TRACKING" | awk '{print $1}')" == "$TRACKING_SHA256" ]] || { printf 'tracking hash mismatch\n' >&2; exit 2; }
[[ "$(sha256sum "$ROBOT_CONFIG" | awk '{print $1}')" == "$ROBOT_CONFIG_SHA256" ]] || { printf 'robot config hash mismatch\n' >&2; exit 2; }
[[ "$(sha256sum "$GAINS" | awk '{print $1}')" == "$GAINS_SHA256" ]] || { printf 'gains hash mismatch\n' >&2; exit 2; }
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || { printf 'robot USD hash mismatch\n' >&2; exit 2; }

git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make runtime provenance ambiguous\n' >&2
  exit 2
}
readonly COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
readonly UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || { printf 'runtime commit is not pushed\n' >&2; exit 2; }
git -C "$ROOT" merge-base --is-ancestor "$REVIEWED_EVALUATOR_PARENT" "$COMMIT" || {
  printf 'reviewed evaluator parent is not in runtime lineage\n' >&2
  exit 2
}
assert_resources_free || exit 5

mkdir -p "$OUTPUT/logs" "$OUTPUT/gates"
python3 - "$OUTPUT/admission.json" "$COMMIT" "$UPSTREAM" "$SHARD" \
  "$RISER_POSITION_M" "$NAMESPACE" "$AUTHORIZATION" \
  evaluator "$EVALUATOR" controller "$METRICS" tracking "$TRACKING" \
  robot_config "$ROBOT_CONFIG" gains "$GAINS" robot_usd "$ROBOT_USD" \
  runner "$RUNNER" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
commit, upstream, shard, position, namespace, authorization = sys.argv[2:8]
identity_args = sys.argv[8:]
if len(identity_args) % 2:
    raise SystemExit("runtime identity arguments must be label/path pairs")
payload = {
    "schema": "recomo_two_wheel_riser_lqr_plant_envelope_admission_v1",
    "runtime_commit": commit,
    "upstream_commit": upstream,
    "reviewed_evaluator_parent": "0dc1aad417aecdc0ed1b29110d08e7abf5db0622",
    "shard": shard,
    "riser_position_m": float(position),
    "namespace": namespace,
    "authorization_sha256": hashlib.sha256(authorization.encode()).hexdigest(),
    "command_contract": {
        "robot_form": "riser",
        "vx_commands_mps": [-0.2, 0.2],
        "wz_commands_rad_s": [0.0],
        "push_forces_n": [-20.0, 20.0],
        "plant_uncertainty_profile": "provisional_prior_v1",
        "scenario_count": 56,
        "controller_profile": "structural_robust_v1",
        "vx_kp": 0.72,
        "total_pitch_limit_deg": 6.0,
        "action_limit": 0.8,
        "minimum_success_rate": 1.0,
        "maximum_direction_speed_asymmetry_mps": 0.05,
    },
    "runtime_identities": {
        identity_args[index]: {
            "path": str(Path(identity_args[index + 1]).resolve()),
            "sha256": hashlib.sha256(
                Path(identity_args[index + 1]).read_bytes()
            ).hexdigest(),
        }
        for index in range(0, len(identity_args), 2)
    },
    "checks": {
        "head_equals_upstream": commit == upstream,
        "fresh_namespace": True,
        "exclusive_resources": True,
        "single_height_shard": True,
        "no_learned_action": True,
        "no_dataset_or_training": True,
    },
    "runtime_authorized": True,
    "capture_started": False,
    "bc_started": False,
    "ppo_started": False,
    "training_started": False,
}
payload["passed"] = all(payload["checks"].values())
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

STATUS=0
timeout --signal=TERM --kill-after=30s "$TIMEOUT_SECONDS" \
  "$PY" -u -X utf8 "$EVALUATOR_WIN" \
  --gains "$GAINS_WIN" \
  --robot-form riser \
  --riser-position-m "$RISER_POSITION_M" \
  --num-envs 56 \
  --horizon-steps 2000 \
  --vx-commands=-0.2,0.2 \
  --wz-commands=0 \
  --push-forces-n=-20,20 \
  --controller-profile structural_robust_v1 \
  --vx-kp 0.72 \
  --pitch-reference-limit-deg 6 \
  --limit-total-pitch-reference \
  --reset-opposing-vx-integral-on-directional-deficit \
  --vx-integral-reset-reference-deadband-mps 0.05 \
  --use-root-velocity-outer-feedback \
  --plant-uncertainty-profile provisional_prior_v1 \
  --minimum-success-rate 1.0 \
  --maximum-direction-speed-asymmetry-mps 0.05 \
  --output "$OUTPUT_WIN\gates\result.json" \
  --headless >"$OUTPUT/logs/runtime.log" 2>&1 || STATUS=$?
printf '%s\n' "$STATUS" >"$OUTPUT/logs/exit_code"
wait_for_release || exit 5

python3 - "$OUTPUT" "$COMMIT" "$SHARD" "$RISER_POSITION_M" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
commit, shard, position_text = sys.argv[2:5]
position = float(position_text)
admission_path = root / "admission.json"
result_path = root / "gates/result.json"
log_path = root / "logs/runtime.log"
exit_path = root / "logs/exit_code"
status = int(exit_path.read_text().strip())
result = json.loads(result_path.read_text()) if result_path.is_file() else None
checks = {
    "runtime_exit_zero": status == 0,
    "result_written": result is not None,
}
if result is not None:
    controller = result.get("controller", {})
    command = result.get("command", {})
    push = result.get("push", {})
    plant = result.get("plant_uncertainty", {})
    summary = result.get("summary", {})
    riser = summary.get("riser_plant") or {}
    runtime = plant.get("runtime", {})
    scenarios = result.get("scenarios", [])
    checks.update(
        {
            "schema": result.get("schema")
            == "recomo_two_wheel_riser_cascaded_lqr_tracking_push_gate_v1",
            "full_riser_asset": result.get("robot_form") == "riser"
            and str(result.get("robot_asset_usd", "")).replace("\\", "/").endswith(
                "/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
            ),
            "nominal_mass_28kg": abs(runtime.get("nominal_total_mass_kg", 0.0) - 28.0)
            <= 0.1,
            "riser_height": abs(riser.get("riser_position_target_m", -1.0) - position)
            <= 1e-12,
            "finite_com_bias": all(
                math.isfinite(riser.get(name, math.nan))
                for name in (
                    "equilibrium_pitch_bias_min_deg",
                    "equilibrium_pitch_bias_max_deg",
                )
            ),
            "riser_hold": riser.get("riser_hold_error_max_m", math.inf)
            <= result.get("thresholds", {}).get("maximum_riser_hold_error_m", -1.0),
            "gimbal_hold": riser.get("gimbal_hold_error_max_deg", math.inf)
            <= result.get("thresholds", {}).get("maximum_gimbal_hold_error_deg", -1.0),
            "controller_contract": controller.get("vx_kp") == 0.72
            and controller.get("vx_ki") == 0.075
            and controller.get("vx_integral_limit") == 0.7
            and controller.get("limit_total_pitch_reference") is True
            and controller.get("reset_opposing_vx_integral_on_directional_deficit")
            is True
            and controller.get("vx_integral_reset_reference_deadband_mps") == 0.05
            and controller.get("use_root_velocity_outer_feedback") is True
            and controller.get("pitch_reference_limit_deg") == 6.0
            and controller.get("action_limit") == 0.8,
            "command_contract": command.get("vx_m_s") == [-0.2, 0.2]
            and command.get("wz_rad_s") == [0.0]
            and push.get("forces_x_n") == [-20.0, 20.0],
            "provisional_variations": plant.get("profile") == "provisional_prior_v1"
            and plant.get("variation_count") == 14,
            "complete_scenarios": summary.get("scenarios") == 56
            and len(scenarios) == 56
            and all(item.get("passed") is True for item in scenarios),
            "direction_symmetry": summary.get("direction_contract_complete") is True
            and summary.get("direction_speed_asymmetry_mps", math.inf) <= 0.05,
            "dynamic_pass": result.get("passed") is True
            and summary.get("success_rate") == 1.0,
            "no_learning": result.get("learned_action_applied") is False
            and result.get("residual_dataset") is None
            and result.get("capture_started") is False
            and result.get("bc_started") is False
            and result.get("ppo_started") is False
            and result.get("training_started") is False,
        }
    )
hashes = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in {
        "admission": admission_path,
        "runtime_log": log_path,
        "exit_code": exit_path,
        **({"result": result_path} if result_path.is_file() else {}),
    }.items()
}
payload = {
    "schema": "recomo_two_wheel_riser_lqr_plant_envelope_final_status_v1",
    "runtime_commit": commit,
    "shard": shard,
    "riser_position_m": position,
    "checks": checks,
    "hashes": hashes,
    "passed": all(checks.values()),
    "first_dynamic_reject": result is not None and result.get("passed") is not True,
    "capture_started": False,
    "bc_started": False,
    "ppo_started": False,
    "training_started": False,
}
(root / "final_status.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
raise SystemExit(0 if payload["passed"] else 4)
PY

printf 'riser plant-envelope shard passed: %s\n' "$OUTPUT"
