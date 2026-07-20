from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/evaluate_lqr_tracking_push.py"
RUNNER = ROOT / "scripts/two_wheel_balance/run_riser_lqr_plant_envelope.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_tracking_push_script_remains_syntactically_valid() -> None:
    ast.parse(source())


def test_balance_chassis_remains_the_default_robot_form() -> None:
    tree = ast.parse(source())
    robot_form_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--robot-form"
    ]
    assert len(robot_form_calls) == 1
    defaults = {
        keyword.arg: keyword.value for keyword in robot_form_calls[0].keywords
    }
    assert isinstance(defaults["default"], ast.Constant)
    assert defaults["default"].value == "balance"


def test_riser_mode_uses_full_asset_and_physical_com_bias() -> None:
    text = source()
    assert 'cfg.robot_cfg = copy.deepcopy(TWO_WHEEL_RISER_CFG)' in text
    assert 'cfg.robot_cfg.init_state.joint_pos["riser_joint"]' in text
    assert "equilibrium_pitch_from_world_com(" in text
    assert "pitch_bias_override_rad=pitch_bias_override_rad" in text
    assert "body_com_pos_w" in text


def test_riser_mode_carries_the_case42_controller_contract() -> None:
    text = source()
    for option in (
        "--limit-total-pitch-reference",
        "--reset-opposing-vx-integral-on-directional-deficit",
        "--vx-integral-reset-reference-deadband-mps",
        "--use-root-velocity-outer-feedback",
    ):
        assert option in text
    assert "outer_vx_feedback_m_s=outer_vx_feedback_m_s" in text
    assert '"limit_total_pitch_reference": config.limit_total_pitch_reference' in text
    assert (
        '"reset_opposing_vx_integral_on_directional_deficit": (' in text
    )


def test_riser_hold_and_no_learning_evidence_fail_closed() -> None:
    text = source()
    assert "riser_hold_error_max_m[index]" in text
    assert "gimbal_hold_error_max_deg[index]" in text
    assert '"learned_action_applied": False' in text
    assert '"residual_dataset": None' in text
    assert '"capture_started": False' in text
    assert '"bc_started": False' in text
    assert '"ppo_started": False' in text
    assert '"training_started": False' in text


def test_riser_gate_requires_bidirectional_speed_symmetry() -> None:
    text = source()
    assert "--maximum-direction-speed-asymmetry-mps" in text
    assert '"direction_contract_complete": direction_contract_complete' in text
    assert '"direction_speed_asymmetry_mps": direction_speed_asymmetry_mps' in text
    assert 'summary["direction_contract_complete"] is True' in text
    assert (
        'summary["direction_speed_asymmetry_mps"]\n'
        "                <= args.maximum_direction_speed_asymmetry_mps"
    ) in text


def test_plant_envelope_runner_is_single_height_hash_bound_and_no_learning() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'case "$SHARD" in' in text
    for position in (
        'RISER_POSITION_M="0.0"',
        'RISER_POSITION_M="0.6"',
        'RISER_POSITION_M="1.2"',
    ):
        assert position in text
    assert "REVIEWED_EVALUATOR_PARENT" in text
    assert "EVALUATOR_SHA256" in text
    assert "ROBOT_USD_SHA256" in text
    assert 'git -C "$ROOT" diff --quiet' in text
    assert '[[ "$COMMIT" == "$UPSTREAM" ]]' in text
    assert "assert_resources_free" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "--robot-form riser" in text
    assert "--vx-kp 0.72" in text
    assert "--limit-total-pitch-reference" in text
    assert "--reset-opposing-vx-integral-on-directional-deficit" in text
    assert "--use-root-velocity-outer-feedback" in text
    assert "--plant-uncertainty-profile provisional_prior_v1" in text
    assert "--minimum-success-rate 1.0" in text
    assert '"capture_started": False' in text
    assert '"bc_started": False' in text
    assert '"ppo_started": False' in text
    assert '"training_started": False' in text


def test_plant_envelope_runner_pins_current_evaluator_blob() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r'readonly EVALUATOR_SHA256="([0-9a-f]{64})"', text)
    assert match is not None
    assert match.group(1) == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def test_plant_envelope_runner_rejects_unknown_auth_and_overrides_pre_runtime() -> None:
    base_env = os.environ.copy()
    for shard, env in (
        ("unknown", base_env),
        ("low", base_env),
        (
            "low",
            base_env
            | {
                "RISER_PLANT_ENVELOPE_AUTHORIZATION": (
                    "AUTHORIZED_RISER_LQR_PLANT_ENVELOPE_VXKP072_LOW_V1"
                ),
                "RISER_PLANT_ENVELOPE_NUM_ENVS": "1",
            },
        ),
    ):
        result = subprocess.run(
            ["bash", str(RUNNER), shard],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 7
