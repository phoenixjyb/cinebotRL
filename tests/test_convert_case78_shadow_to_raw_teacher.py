import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_NAMES,
    load_raw_teacher_case,
    save_shadow_teacher_trace,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/convert_case78_shadow_to_raw_teacher.py"


def test_converts_admitted_shadow_without_applying_labels(tmp_path: Path) -> None:
    count = 3
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    raw = np.asarray([[0.1, 0.0, 0.01], [0.2, -0.1, 0.02], [0.0, 0.1, 0.0]], dtype=np.float32)
    trace = tmp_path / "trace.npz"
    payload = {
        "observations": observations,
        "applied_residual_actions": np.zeros((count, 3), dtype=np.float32),
        "final_high_level_commands": raw.copy(),
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "case_ids": np.full(count, 78, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "post_step_position_error_m": np.zeros(count),
        "post_step_attitude_error_deg": np.zeros(count),
        "post_step_base_xy_yaw": np.zeros((count, 3)),
        "post_step_camera_position_world_m": np.zeros((count, 3)),
        "post_step_pitch_deg": np.zeros(count),
        "post_step_riser_position_m": np.zeros(count),
        "post_step_proxy_position_rad": np.zeros((count, 3)),
        "shadow_teacher_raw_residual_commands": raw,
        "shadow_teacher_normalized_residual_actions": raw / np.asarray([0.35, 0.4, 0.1]),
        "shadow_teacher_high_level_commands": raw.copy(),
    }
    save_shadow_teacher_trace(
        trace,
        78,
        payload,
        action_scales=np.asarray([0.35, 0.4, 0.1]),
        visited_state_source="deterministic_controller",
    )
    import hashlib
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "label_admission_passed": True,
        "raw_teacher_conversion_authorized": True,
        "offline_dataset_rebuild_authorized": True,
        "case": 78,
        "split": "validation",
        "row_count": count,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "inputs": {"trace": {"sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}},
    }))
    output = tmp_path / "case_0078_executed_raw_teacher_v1.npz"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--label-admission", str(admission), "--trace", str(trace), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metadata, restored = load_raw_teacher_case(output)
    assert metadata["valid_for_training"] is False
    np.testing.assert_array_equal(restored["raw_residual_commands"], raw)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["raw_teacher_conversion_passed"]
    assert not summary["bc_authorized"]
