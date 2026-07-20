import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    save_raw_teacher_case,
)
from scripts.two_wheel_balance.audit_riser_raw_teacher_capture import (
    canonical_cross_platform_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/audit_riser_raw_teacher_capture.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, applied_residual: bool = False) -> list[str]:
    case = 2
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
                "passed": True,
                "selection_count_met": True,
                "selected_cases": [case],
                "source_manifest_sha256": "a" * 64,
                "portfolio_manifest_sha256": "b" * 64,
                "valid_for_training": False,
                "rows": [{"case": case, "plan_sha256": "c" * 64}],
            }
        ),
        encoding="utf-8",
    )
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "passed": True,
                "requested_cases": [case],
                "selection_sha256": _sha256(selection),
                "source_manifest_sha256": "a" * 64,
                "portfolio_manifest_sha256": "b" * 64,
                "plan_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    count = 4
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.2
    observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]] = -0.1
    observations[:, OBSERVATION_INDEX["riser_position_m"]] = 1.0
    teacher_commands = np.tile([0.52, 0.05, 1.02], (count, 1)).astype(np.float32)
    raw_commands = np.tile([0.32, 0.15, 0.02], (count, 1)).astype(np.float32)
    raw_case = tmp_path / "case_0002_executed_raw_teacher_v1.npz"
    save_raw_teacher_case(
        raw_case,
        case,
        {
            "observations": observations,
            "raw_residual_commands": raw_commands,
            "case_ids": np.full(count, case, dtype=np.int16),
            "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
            "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
            "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
            "teacher_commands": teacher_commands,
        },
    )
    gate = tmp_path / "case_0002.json"
    result = {
        "case": case,
        "passed": True,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "completed_steps": count,
        "source_duration_s": 1.0,
        "execution_duration_s": 1.5,
        "termination": None,
        "raw_residual_command_abs_max": [0.32, 0.15, 0.02],
        "raw_residual_label_applied_to_commands": False,
        "residual_action_abs_max": [0.1, 0.0, 0.0] if applied_residual else [0, 0, 0],
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": str(raw_case.resolve()),
        "checks": {"completed_reference": True},
    }
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "dynamic_quality_passed": True,
                "thermal_admission_passed": True,
                "controller_evidence_passed": True,
                "training_started": False,
                "ppo_authorized": False,
                "results": [result],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    return [
        sys.executable,
        str(SCRIPT),
        "--gate",
        str(gate),
        "--admission",
        str(admission),
        "--raw-case",
        str(raw_case),
        "--selection",
        str(selection),
        "--case",
        str(case),
        "--output",
        str(output),
    ]


def test_audits_scale_independent_capture(tmp_path: Path) -> None:
    command = _fixture(tmp_path)
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["capture_admission_passed"]
    np.testing.assert_allclose(
        audit["raw_residual_abs_max"], [0.32, 0.15, 0.02], atol=1e-7
    )
    assert audit["provisional_recommended_action_scales"] == [0.4, 0.4, 0.1]
    assert not audit["action_scale_frozen"]
    assert not audit["valid_for_training"]
    assert not audit["bc_authorized"]
    assert not audit["ppo_authorized"]


def test_rejects_capture_if_residual_was_applied(tmp_path: Path) -> None:
    command = _fixture(tmp_path, applied_residual=True)
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert not audit["checks"]["zero_applied_residual"]
    assert not audit["capture_admission_passed"]


def test_canonicalizes_windows_paths_for_wsl_audit() -> None:
    assert canonical_cross_platform_path(r"G:\wSpace\capture.npz") == (
        "/mnt/g/wSpace/capture.npz"
    )
