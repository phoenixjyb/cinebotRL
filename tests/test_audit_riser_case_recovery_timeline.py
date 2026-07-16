import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_case_recovery_timeline.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    step: int,
    *,
    feedforward: float,
    command: float,
    yaw_error: float,
    base_error: float,
) -> dict[str, object]:
    return {
        "step": step,
        "elapsed_s": float(step),
        "phase_time_s": float(step) * 0.5,
        "phase_feedforward_v_mps": feedforward,
        "vx_reference_mps": command,
        "position_error_m": base_error * 0.8,
        "base_xy_error_m": base_error,
        "pitch_deg": 1.0,
        "proxy_signed_error_deg": [0.0, 0.0, yaw_error],
    }


def test_audit_separates_healthy_reverse_motion_from_post_fault_recovery(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "case.npz"
    np.savez_compressed(
        plan_path,
        execution_time_s=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        feedforward_v_wz=np.array(
            [[0.1, 0.0], [-0.1, 0.0], [-0.1, 0.0], [0.1, 0.0]]
        ),
    )
    trace = [
        _row(1, feedforward=0.1, command=0.12, yaw_error=0.1, base_error=0.02),
        _row(2, feedforward=-0.1, command=-0.15, yaw_error=0.2, base_error=0.04),
        _row(3, feedforward=-0.1, command=0.2, yaw_error=719.0, base_error=0.13),
        _row(4, feedforward=-0.01, command=0.4, yaw_error=719.0, base_error=0.30),
        _row(5, feedforward=0.01, command=-0.4, yaw_error=719.0, base_error=0.55),
    ]
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case": 74,
                        "dynamic_quality_passed": False,
                        "executed_residual_dataset": None,
                        "trace": trace,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case-json",
            str(case_path),
            "--expected-case-sha256",
            _sha256(case_path),
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            _sha256(plan_path),
            "--output",
            str(output),
        ],
        check=False,
    )
    audit = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert audit["passed"]
    assert audit["plan"]["direction_reversal_count"] == 2
    assert audit["pre_fault"]["reverse_sample_count"] == 1
    assert audit["pre_fault"]["velocity_saturation_sample_count"] == 0
    assert audit["fault"]["step"] == 3
    assert audit["first_base_gate_failure"]["step"] == 4
    assert audit["first_post_fault_velocity_saturation"]["step"] == 4
    assert audit["post_fault"]["vx_reference_sign_change_count"] == 1
    assert not audit["reverse_controller_change_authorized"]
    assert not audit["runtime_controller_changed"]
    assert not audit["gpu_work_started"]
    assert not audit["valid_for_training"]
