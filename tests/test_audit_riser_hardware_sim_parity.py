import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_hardware_sim_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("riser_hardware_sim_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_300n_plant_is_transient_not_continuous_motor_force() -> None:
    report = _load_module().build_report()
    drive = report["drive"]
    classification = report["classification"]

    assert report["passed"]
    assert drive["rated_linear_force_n"] == pytest.approx(292.397004249)
    assert drive["peak_linear_force_n"] == pytest.approx(877.191012746)
    assert drive["simulated_to_rated_force_ratio"] == pytest.approx(
        1.026002304
    )
    assert classification["concept_screening_passed"]
    assert not classification["continuous_rated_force_parity_passed"]
    assert classification["training_admission_thermal_monitor_present"]
    assert not classification["active_thermal_force_derater_present"]
    assert not classification["valid_for_hardware_transfer"]
    assert not classification["valid_for_training"]


def test_rated_drive_covers_8kg_emergency_but_with_narrow_margin() -> None:
    report = _load_module().build_report()
    drive = report["drive"]

    assert report["checks"]["rated_force_covers_8kg_emergency_design_force"]
    assert drive["rated_force_margin_over_8kg_emergency"] == pytest.approx(
        1.0557373059
    )
    assert 1.05 < drive["rated_force_margin_over_8kg_emergency"] < 1.06
    assert drive["motor_speed_at_sim_velocity_limit_rpm"] == pytest.approx(
        2571.4285714
    )
    assert drive["rated_linear_speed_mps"] == pytest.approx(1.1666666667)


def test_cli_writes_a_non_training_parity_report(tmp_path: Path) -> None:
    output = tmp_path / "parity.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert report["schema"] == (
        "cinebotrl_two_wheel_riser_hardware_sim_parity_v2"
    )
    assert report["plant"]["camera_height_range_m"] == [0.6, 1.8]
    assert report["classification"]["simulated_300n_interpretation"] == (
        "transient_actuator_cap_not_continuous_rated_force"
    )
    assert not report["gpu_work_started"]
    assert not report["residual_capture_started"]
    assert not report["bc_started"]
    assert not report["ppo_started"]


def test_thermal_admission_matches_drive_but_is_not_an_active_derater() -> None:
    report = _load_module().build_report()
    thermal = report["thermal_admission"]

    assert thermal["contract"] == "leadshine_400w_first_order_monitor_v1"
    assert thermal["continuous_force_n"] == pytest.approx(
        report["drive"]["rated_linear_force_n"]
    )
    assert thermal["peak_force_n"] == pytest.approx(
        report["drive"]["peak_linear_force_n"]
    )
    assert thermal["thermal_time_constant_s"] == 30.0
    assert all(thermal["runner_checks"].values())
    assert not thermal["active_force_derating"]
