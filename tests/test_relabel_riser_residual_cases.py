import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_NAMES,
    save_case_dataset,
)
from scripts.two_wheel_balance.audit_riser_residual_capture import (
    recommended_scales,
)


def test_recommended_scales_apply_margin_quantum_and_floor() -> None:
    scales = recommended_scales(
        np.array([0.2266, 0.1622, 0.0119]),
        np.array([0.2, 0.4, 0.1]),
        np.array([0.05, 0.05, 0.05]),
        1.10,
    )
    np.testing.assert_allclose(scales, [0.25, 0.4, 0.1])


def test_relabel_recovers_unclipped_action_from_raw_commands(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output" / "cases"
    source.mkdir()
    count = 4
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, 18] = 0.05
    teacher_commands = np.zeros((count, 3), dtype=np.float32)
    teacher_commands[:, 0] = np.array([-0.18, -0.16, 0.05, 0.10])
    payload = {
        "observations": observations,
        "actions": np.array(
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "case_ids": np.full(count, 20, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": teacher_commands,
    }
    source_file = source / "case_0020_executed_residual_v2.npz"
    save_case_dataset(source_file, 20, payload)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/two_wheel_balance/relabel_riser_residual_cases.py",
            "--source-case-dir",
            str(source),
            "--output-case-dir",
            str(output),
            "--action-scales",
            "0.3,0.4,0.1",
            "--expected-count",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with np.load(output / source_file.name, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        actions = np.asarray(data["actions"])
    assert metadata["action_scales"] == [0.3, 0.4, 0.1]
    assert metadata["source_action_scales"] == [0.3, 0.4, 0.1]
    np.testing.assert_allclose(actions[:, 0], [-0.23 / 0.3, -0.21 / 0.3, 0.0, 0.05 / 0.3])
    assert np.max(np.abs(actions)) < 1.0
    summary = json.loads(
        (output.parent / "relabel_summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert summary["case_count"] == 1


def test_capture_audit_separates_physics_from_clipped_labels(tmp_path: Path) -> None:
    gates = tmp_path / "gates"
    cases = tmp_path / "cases"
    gates.mkdir()
    cases.mkdir()
    count = 3
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, 18] = 0.05
    teacher_commands = np.zeros((count, 3), dtype=np.float32)
    teacher_commands[:, 0] = [-0.18, -0.16, 0.05]
    save_case_dataset(
        cases / "case_0001_executed_residual_v2.npz",
        1,
        {
            "observations": observations,
            "actions": np.array(
                [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "case_ids": np.ones(count, dtype=np.int16),
            "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
            "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
            "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
            "teacher_commands": teacher_commands,
        },
    )
    result_metrics = {
        "case": 1,
        "passed": True,
        "termination": None,
        "completed_steps": count,
        "position_error_p95_m": 0.1,
        "position_error_max_m": 0.11,
        "attitude_error_p95_deg": 0.1,
        "attitude_error_max_deg": 0.11,
        "pitch_max_deg": 4.0,
        "riser_servo_error_p95_m": 0.01,
        "proxy_servo_error_p95_deg": 0.1,
    }
    (gates / "case_0001.json").write_text(
        json.dumps(
            {
                "passed": True,
                "tracking_profile": "riser_phase_consistent_v2",
                "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
                "results": [result_metrics],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "capture_audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/two_wheel_balance/audit_riser_residual_capture.py",
            "--gate-dir",
            str(gates),
            "--case-dir",
            str(cases),
            "--plan-manifest",
            str(manifest),
            "--source-commit",
            "a" * 40,
            "--expected-count",
            "1",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["physical_capture_passed"]
    assert not summary["source_labels_admissible"]
    assert summary["relabel_required"]
    assert set(summary["raw_residual_abs_percentiles"]) == {"50", "90", "95", "99"}
    np.testing.assert_allclose(summary["raw_residual_signed_min"], [-0.23, 0.0, 0.0])
    np.testing.assert_allclose(summary["raw_residual_signed_max"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(summary["recommended_action_scales"], [0.3, 0.4, 0.1])
