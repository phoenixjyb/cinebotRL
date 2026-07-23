import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/reduce_riser_bench_log.py"
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_BENCH_RAW_LOG_TEMPLATE_20260723.csv"
)
SPEC = importlib.util.spec_from_file_location("riser_bench_reducer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _base_row(time_s: float) -> dict[str, object]:
    return {
        "time_s": time_s,
        "phase": "continuous",
        "trial_id": 0,
        "active_command": 0,
        "steady_state": 0,
        "stop_trigger_event": 0,
        "commanded_velocity_mps": 0.0,
        "measured_velocity_mps": 0.0,
        "position_m": 0.5,
        "phase_current_a": 1.0,
        "dc_input_current_a": 1.0,
        "dc_bus_voltage_v": 48.0,
        "ambient_temperature_c": 25.0,
        "motor_housing_temperature_c": 30.0,
        "drive_temperature_c": 29.0,
        "fault_active": 0,
        "tooth_jump_detected": 0,
        "position_loss_detected": 0,
    }


def _healthy_rows() -> list[dict[str, object]]:
    rows = []
    for second in range(1801):
        row = _base_row(float(second))
        active = second % 10 < 6
        row.update(
            {
                "active_command": int(active),
                "steady_state": int(active),
                "commanded_velocity_mps": 1.0 if active else 0.0,
                "measured_velocity_mps": 0.97 if active else 0.0,
                "position_m": 0.5 + 0.1 * (second % 5),
                "phase_current_a": 8.0 if active else 1.0,
                "dc_input_current_a": 12.0 if active else 1.0,
                "dc_bus_voltage_v": 58.0,
                "motor_housing_temperature_c": 30.0 + second / 180.0,
                "drive_temperature_c": 29.0 + second / 225.0,
            }
        )
        rows.append(row)
    time_s = 1801.0
    for trial_id in range(1, 11):
        start_position = 0.4 + 0.01 * trial_id
        for step, (velocity, distance) in enumerate(
            ((0.96, 0.0), (0.70, 0.03), (0.40, 0.07), (0.0, 0.10))
        ):
            row = _base_row(time_s)
            row.update(
                {
                    "phase": "emergency_stop",
                    "trial_id": trial_id,
                    "stop_trigger_event": int(step == 0),
                    "commanded_velocity_mps": 0.0,
                    "measured_velocity_mps": velocity,
                    "position_m": start_position + distance,
                    "phase_current_a": 25.0 if step == 1 else 8.0,
                    "dc_input_current_a": 18.0,
                    "dc_bus_voltage_v": 62.0,
                }
            )
            rows.append(row)
            time_s += 0.02
        time_s += 0.10
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MODULE.EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_healthy_log_reduces_numeric_bench_fields(tmp_path: Path) -> None:
    path = tmp_path / "bench.csv"
    _write(path, _healthy_rows())
    result = MODULE.reduce_log(path)
    assert result["passed"]
    assert result["valid_for_bench_measurement_numeric_merge"]
    assert result["continuous_duty"]["duration_s"] == 1800.0
    assert result["continuous_duty"]["duty_cycle_fraction"] == pytest.approx(0.6)
    assert result["continuous_duty"]["minimum_achieved_speed_mps"] == 0.97
    assert result["continuous_duty"]["phase_current_rms_a"] == 8.0
    assert result["continuous_duty"]["dc_input_current_rms_a"] == 12.0
    assert result["continuous_duty"]["final_thermal_slope_c_per_min"] == pytest.approx(
        1.0 / 3.0
    )
    assert result["emergency_stop"]["repetitions"] == 10
    assert result["emergency_stop"]["initial_speed_abs_min_mps"] == 0.96
    assert result["emergency_stop"]["worst_stopping_distance_m"] == pytest.approx(
        0.10
    )
    assert not result["ready_for_production_design_review"]
    assert not result["valid_for_production_procurement"]
    assert not result["valid_for_hardware_transfer"]
    assert not result["valid_for_training"]
    assert result["schema"] == MODULE.SCHEMA
    assert result["candidate_profile"] is None
    assert not result["valid_for_candidate_bound_bench_merge"]


def test_candidate_bound_reduction_records_750w_identity(tmp_path: Path) -> None:
    path = tmp_path / "bench.csv"
    _write(path, _healthy_rows())
    result = MODULE.reduce_log(
        path,
        candidate_profile="leadshine_750w_production_candidate_v1",
    )
    assert result["schema"] == MODULE.CANDIDATE_BOUND_SCHEMA
    assert result["candidate_profile"] == (
        "leadshine_750w_production_candidate_v1"
    )
    assert result["valid_for_bench_measurement_numeric_merge"]
    assert result["valid_for_candidate_bound_bench_merge"]
    assert not result["valid_for_hardware_transfer"]
    assert not result["valid_for_training"]


def test_unknown_candidate_profile_is_rejected_before_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported riser candidate profile"):
        MODULE.reduce_log(
            tmp_path / "does-not-need-to-exist.csv",
            candidate_profile="unknown",
        )


def test_empty_template_has_exact_columns_and_fails_closed() -> None:
    with TEMPLATE.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == MODULE.EXPECTED_COLUMNS
        assert list(reader) == []
    with pytest.raises(ValueError, match="no samples"):
        MODULE.reduce_log(TEMPLATE)


def test_reducer_rejects_duplicate_time(tmp_path: Path) -> None:
    rows = _healthy_rows()
    rows[10]["time_s"] = rows[9]["time_s"]
    path = tmp_path / "duplicate.csv"
    _write(path, rows)
    with pytest.raises(ValueError, match="increase strictly"):
        MODULE.reduce_log(path)


def test_reducer_rejects_missing_stop_trigger(tmp_path: Path) -> None:
    rows = _healthy_rows()
    for row in rows:
        if row["phase"] == "emergency_stop" and row["trial_id"] == 1:
            row["stop_trigger_event"] = 0
    path = tmp_path / "missing_trigger.csv"
    _write(path, rows)
    with pytest.raises(ValueError, match="exactly one trigger"):
        MODULE.reduce_log(path)


def test_reducer_rejects_non_boolean_flags(tmp_path: Path) -> None:
    rows = _healthy_rows()
    rows[0]["fault_active"] = 2
    path = tmp_path / "bad_bool.csv"
    _write(path, rows)
    with pytest.raises(ValueError, match="non-boolean"):
        MODULE.reduce_log(path)


def test_cli_writes_lf_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "bench.csv"
    output = tmp_path / "reduction.json"
    _write(source, _healthy_rows())
    command = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_cli_candidate_profile_emits_v2_reduction(tmp_path: Path) -> None:
    source = tmp_path / "bench.csv"
    output = tmp_path / "reduction-v2.json"
    _write(source, _healthy_rows())
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--candidate-profile",
            "leadshine_750w_production_candidate_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == MODULE.CANDIDATE_BOUND_SCHEMA
    assert payload["candidate_profile"] == (
        "leadshine_750w_production_candidate_v1"
    )
