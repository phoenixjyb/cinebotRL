import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/calculate_riser_hardware_envelope.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("riser_hardware_envelope", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_three_to_one_reduction_closes_heavy_case_torque_gap() -> None:
    report = _load_module().build_report()
    cases = {
        (item["reduction_ratio"], item["moving_mass_kg"]): item
        for item in report["sizing_cases"]
    }

    assert not cases[(2.0, 8.0)]["checks"]["rated_torque_passed"]
    assert cases[(2.0, 8.0)]["motor_torque_nm"] == pytest.approx(
        1.4917006455686699
    )
    assert cases[(3.0, 8.0)]["passed"]
    assert cases[(3.0, 8.0)]["motor_torque_nm"] == pytest.approx(
        0.99446709704578
    )
    assert cases[(3.0, 8.0)]["motor_speed_rpm"] == pytest.approx(
        2571.428571428571
    )


def test_report_preserves_optical_range_and_mechanical_buffer() -> None:
    report = _load_module().build_report()
    stopping = report["stopping_envelope"]

    assert report["passed"]
    assert report["recommendation"]["software_camera_height_range_m"] == [0.6, 1.8]
    assert stopping["usable_software_stroke_m"] == pytest.approx(1.2)
    assert stopping["full_speed_stopping_distance_m"] == pytest.approx(0.12)
    assert stopping["recommended_mechanical_stroke_m"] == pytest.approx(1.5)
    assert not report["valid_for_procurement"]
    assert not report["gpu_work_started"]
    assert not report["training_started"]


def test_three_to_one_barely_passes_worst_direction_emergency_braking() -> None:
    report = _load_module().build_report()
    cases = {
        (item["reduction_ratio"], item["moving_mass_kg"]): item
        for item in report["emergency_braking_cases"]
    }

    assert not cases[(2.0, 8.0)]["checks"]["rated_torque_passed"]
    assert cases[(3.0, 8.0)]["passed"]
    assert cases[(3.0, 8.0)]["motor_torque_nm"] == pytest.approx(
        1.2029507651895495
    )
    assert 1.0 < cases[(3.0, 8.0)]["rated_torque_margin_ratio"] < 1.1
    assert report["checks"]["three_to_one_passes_all_emergency_braking_cases"]


def test_cli_writes_deterministic_machine_readable_report(tmp_path: Path) -> None:
    output = tmp_path / "hardware_envelope.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert report["schema"] == "cinebotrl_two_wheel_riser_hardware_envelope_v1"
    assert report["recommendation"]["preferred_reduction_ratio"] == 3.0
    assert report["checks"]["buffered_mechanical_stroke_is_1_50_m"]
