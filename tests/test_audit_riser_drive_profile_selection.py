import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT / "scripts/two_wheel_balance/audit_riser_drive_profile_selection.py"
)
PRODUCTION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_hardware_production_candidate_v1/summary.json"
)
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_drive_profile_selection_v1/summary.json"
)
EVIDENCE_SHA256 = (
    "39a700de3985175e4e8415f1f23beef4264b103daa7ce8847f4ac0fe69f879f7"
)
SPEC = importlib.util.spec_from_file_location("riser_drive_profile_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _production():
    return json.loads(PRODUCTION.read_text(encoding="utf-8"))


def _plant_files(tmp_path: Path) -> dict[str, Path]:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
<robot name="fixture">
  <joint name="riser_joint" type="prismatic">
    <limit lower="0.0" upper="1.2" effort="300.0" velocity="1.0"/>
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )
    config = tmp_path / "config.py"
    config.write_text(
        """
TWO_WHEEL_RISER_CFG.actuators["riser_position"] = ImplicitActuatorCfg(
    effort_limit_sim=300.0,
    velocity_limit_sim=1.0,
    stiffness=1200.0,
    damping=120.0,
    armature=0.1,
)
""".strip(),
        encoding="utf-8",
    )
    thermal = tmp_path / "riser_control.py"
    thermal.write_text(
        """
RISER_THERMAL_FORCE_CONTRACT = "leadshine_400w_first_order_monitor_v1"

class RiserMotorThermalMonitor:
    continuous_force_n: float = 292.3970042486123
    peak_force_n: float = 877.1910127458367
    thermal_time_constant_s: float = 30.0
""".strip(),
        encoding="utf-8",
    )
    return {
        "urdf_path": urdf,
        "config_path": config,
        "thermal_control_path": thermal,
    }


def test_active_profile_remains_400w_and_750w_is_review_only(
    tmp_path: Path,
) -> None:
    report = MODULE.build_report(_production(), **_plant_files(tmp_path))
    assert report["passed"]
    active = report["active_simulation_profile"]
    assert active["name"] == "leadshine_400w_engineering_sample_v1"
    assert active["simulation_effort_limit_n"] == pytest.approx(300.0)
    assert active["simulation_velocity_limit_mps"] == pytest.approx(1.0)
    assert active["continuous_force_reference_n"] == pytest.approx(
        292.3970042486123
    )
    candidate = report["production_design_candidate"]
    assert candidate["name"] == "leadshine_750w_production_candidate_v1"
    assert candidate["rated_linear_force_n"] == pytest.approx(550.2589292552624)
    assert candidate["simulation_enabled"] is False
    assert candidate["runtime_authorized"] is False
    assert candidate["valid_for_training"] is False
    switch = report["profile_switch_contract"]
    assert switch["environment_or_cli_switch_supported"] is False
    assert switch["existing_dynamic_evidence_reusable_after_switch"] is False
    assert switch["existing_corrective_captures_reusable_after_switch"] is False
    assert switch["existing_bc_checkpoint_reusable_after_switch"] is False


def test_committed_profile_selection_is_exact_and_closed() -> None:
    raw = EVIDENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EVIDENCE_SHA256
    report = json.loads(raw)
    assert report["passed"] is True
    assert report["active_simulation_profile"]["name"] == (
        "leadshine_400w_engineering_sample_v1"
    )
    candidate = report["production_design_candidate"]
    assert candidate["name"] == "leadshine_750w_production_candidate_v1"
    assert candidate["simulation_enabled"] is False
    assert report["classification"]["silent_750w_simulation_upgrade_rejected"]
    for field in (
        "gpu_work_started",
        "isaac_started",
        "capture_started",
        "bc_started",
        "ppo_started",
    ):
        assert report[field] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulation_motor_model_updated", True),
        ("runtime_authorized", True),
        ("valid_for_training", True),
        ("valid_for_hardware_transfer", True),
    ],
)
def test_unreviewed_750w_activation_fails_closed(
    tmp_path: Path, field, value
) -> None:
    production = copy.deepcopy(_production())
    production[field] = value
    report = MODULE.build_report(production, **_plant_files(tmp_path))
    assert not report["passed"]
    assert not all(report["checks"].values())
