from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/evaluate_lqr_tracking_push.py"


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
