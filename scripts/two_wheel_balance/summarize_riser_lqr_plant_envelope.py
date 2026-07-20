#!/usr/bin/env python3
"""Seal one completed full-riser LQR plant-envelope shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "recomo_two_wheel_riser_lqr_plant_envelope_final_status_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: object, expected: float, *, tolerance: float = 1e-9) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=tolerance
    )


def summarize(
    root: Path,
    *,
    runtime_commit: str,
    shard: str,
    riser_position_m: float,
) -> dict[str, object]:
    admission_path = root / "admission.json"
    result_path = root / "gates/result.json"
    log_path = root / "logs/runtime.log"
    exit_path = root / "logs/exit_code"
    required = (admission_path, result_path, log_path, exit_path)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"missing plant-envelope evidence: {missing}")

    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = int(exit_path.read_text(encoding="utf-8").strip())
    controller = result.get("controller", {})
    command = result.get("command", {})
    push = result.get("push", {})
    plant = result.get("plant_uncertainty", {})
    summary = result.get("summary", {})
    riser = summary.get("riser_plant") or {}
    runtime = plant.get("runtime", {})
    scenarios = result.get("scenarios", [])
    thresholds = result.get("thresholds", {})

    checks = {
        "runtime_exit_zero": status == 0,
        "admission_contract": admission.get("passed") is True
        and admission.get("runtime_commit") == runtime_commit
        and admission.get("shard") == shard
        and close(admission.get("riser_position_m"), riser_position_m),
        "schema": result.get("schema")
        == "recomo_two_wheel_riser_cascaded_lqr_tracking_push_gate_v1",
        "full_riser_asset": result.get("robot_form") == "riser"
        and str(result.get("robot_asset_usd", "")).replace("\\", "/").endswith(
            "/assets_own/recomoProto2_two_wheel_riser/"
            "recomoProto2_two_wheel_riser.usd"
        ),
        "nominal_mass_28kg": close(
            runtime.get("nominal_total_mass_kg"), 28.0, tolerance=0.1
        ),
        "riser_height": close(
            riser.get("riser_position_target_m"), riser_position_m
        ),
        "finite_com_bias": all(
            isinstance(riser.get(name), (int, float))
            and math.isfinite(float(riser[name]))
            for name in (
                "equilibrium_pitch_bias_min_deg",
                "equilibrium_pitch_bias_max_deg",
            )
        ),
        "hold_threshold_contract": close(
            thresholds.get("maximum_riser_hold_error_m"), 0.03
        )
        and close(thresholds.get("maximum_gimbal_hold_error_deg"), 1.0),
        "riser_hold": riser.get("riser_hold_error_max_m", math.inf)
        <= thresholds.get("maximum_riser_hold_error_m", -1.0),
        "gimbal_hold": riser.get("gimbal_hold_error_max_deg", math.inf)
        <= thresholds.get("maximum_gimbal_hold_error_deg", -1.0),
        "controller_contract": close(controller.get("vx_kp"), 0.72)
        and close(controller.get("vx_ki"), 0.075)
        and close(controller.get("vx_integral_limit"), 0.7)
        and controller.get("limit_total_pitch_reference") is True
        and controller.get("reset_opposing_vx_integral_on_directional_deficit")
        is True
        and close(
            controller.get("vx_integral_reset_reference_deadband_mps"), 0.05
        )
        and controller.get("use_root_velocity_outer_feedback") is True
        and controller.get("semantic_proxy_state_adapter") is True
        and close(controller.get("pitch_reference_limit_deg"), 6.0)
        and close(controller.get("action_limit"), 0.8),
        "command_contract": command.get("vx_m_s") == [-0.2, 0.2]
        and command.get("wz_rad_s") == [0.0]
        and push.get("forces_x_n") == [-20.0, 20.0],
        "provisional_variations": plant.get("profile")
        == "provisional_prior_v1"
        and plant.get("variation_count") == 14,
        "complete_scenarios": summary.get("scenarios") == 56
        and len(scenarios) == 56
        and all(item.get("passed") is True for item in scenarios),
        "direction_symmetry": summary.get("direction_contract_complete") is True
        and summary.get("direction_speed_asymmetry_mps", math.inf) <= 0.05,
        "dynamic_pass": result.get("passed") is True
        and close(summary.get("success_rate"), 1.0),
        "no_learning": result.get("learned_action_applied") is False
        and result.get("residual_dataset") is None
        and result.get("capture_started") is False
        and result.get("bc_started") is False
        and result.get("ppo_started") is False
        and result.get("training_started") is False,
    }
    hashes = {
        "admission": sha256(admission_path),
        "runtime_log": sha256(log_path),
        "exit_code": sha256(exit_path),
        "result": sha256(result_path),
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "runtime_commit": runtime_commit,
        "shard": shard,
        "riser_position_m": riser_position_m,
        "checks": checks,
        "hashes": hashes,
        "summarizer_sha256": sha256(Path(__file__)),
        "passed": all(checks.values()),
        "first_dynamic_reject": result.get("passed") is not True,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "training_started": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--shard", choices=("low", "mid", "high"), required=True)
    parser.add_argument("--riser-position-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        args.root,
        runtime_commit=args.runtime_commit,
        shard=args.shard,
        riser_position_m=args.riser_position_m,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
