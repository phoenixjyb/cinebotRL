from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/summarize_riser_lqr_plant_envelope.py"
)
SPEC = importlib.util.spec_from_file_location("riser_plant_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_fixture(root: Path) -> None:
    (root / "gates").mkdir(parents=True)
    (root / "logs").mkdir()
    admission = {
        "passed": True,
        "runtime_commit": "abc123",
        "shard": "low",
        "riser_position_m": 0.0,
    }
    result = {
        "schema": "recomo_two_wheel_riser_cascaded_lqr_tracking_push_gate_v1",
        "robot_form": "riser",
        "robot_asset_usd": (
            "G:\\repo\\assets_own\\recomoProto2_two_wheel_riser\\"
            "recomoProto2_two_wheel_riser.usd"
        ),
        "controller": {
            "vx_kp": 0.72,
            "vx_ki": 0.075,
            "vx_integral_limit": 0.7,
            "limit_total_pitch_reference": True,
            "reset_opposing_vx_integral_on_directional_deficit": True,
            "vx_integral_reset_reference_deadband_mps": 0.05,
            "use_root_velocity_outer_feedback": True,
            "semantic_proxy_state_adapter": True,
            "pitch_reference_limit_deg": 6.000000000000001,
            "action_limit": 0.8,
        },
        "command": {"vx_m_s": [-0.2, 0.2], "wz_rad_s": [0.0]},
        "push": {"forces_x_n": [-20.0, 20.0]},
        "plant_uncertainty": {
            "profile": "provisional_prior_v1",
            "variation_count": 14,
            "runtime": {"nominal_total_mass_kg": 28.0},
        },
        "thresholds": {
            "maximum_riser_hold_error_m": 0.03,
            "maximum_gimbal_hold_error_deg": 1.0,
        },
        "summary": {
            "scenarios": 56,
            "success_rate": 1.0,
            "direction_contract_complete": True,
            "direction_speed_asymmetry_mps": 0.02,
            "riser_plant": {
                "riser_position_target_m": 0.0,
                "riser_hold_error_max_m": 0.001,
                "gimbal_hold_error_max_deg": 0.1,
                "equilibrium_pitch_bias_min_deg": -1.0,
                "equilibrium_pitch_bias_max_deg": 3.0,
            },
        },
        "scenarios": [{"passed": True} for _ in range(56)],
        "learned_action_applied": False,
        "residual_dataset": None,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "training_started": False,
        "passed": True,
    }
    (root / "admission.json").write_text(json.dumps(admission))
    (root / "gates/result.json").write_text(json.dumps(result))
    (root / "logs/runtime.log").write_text("completed\n")
    (root / "logs/exit_code").write_text("0\n")


def test_summary_accepts_float_roundoff_without_relaxing_contract(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    payload = MODULE.summarize(
        tmp_path,
        runtime_commit="abc123",
        shard="low",
        riser_position_m=0.0,
    )
    assert payload["passed"] is True
    assert all(payload["checks"].values())
    assert payload["hashes"]["result"]
    assert payload["summarizer_sha256"]


def test_summary_keeps_no_learning_and_dynamic_outcomes_hard(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    path = tmp_path / "gates/result.json"
    result = json.loads(path.read_text())
    result["training_started"] = True
    path.write_text(json.dumps(result))
    payload = MODULE.summarize(
        tmp_path,
        runtime_commit="abc123",
        shard="low",
        riser_position_m=0.0,
    )
    assert payload["passed"] is False
    assert payload["checks"]["no_learning"] is False


def test_summary_rejects_missing_evidence(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE.summarize(
            tmp_path,
            runtime_commit="abc123",
            shard="low",
            riser_position_m=0.0,
        )
