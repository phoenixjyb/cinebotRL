from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/two_wheel_balance"))

from audit_riser_case42_plan_lineage import audit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_plan(path: Path, *, allocation_offset: float, time_scale: float) -> None:
    source_time = np.array([0.0, 1.0, 2.0])
    source_position = np.column_stack((source_time, source_time * 0.0, source_time + 1.0))
    source_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    physical_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    execution_time = source_time * time_scale
    np.savez(
        path,
        source_time_s=source_time,
        source_target_position_world_m=source_position,
        source_target_semantic_dfr_quat_xyzw=source_quat,
        source_anchor_execution_index=np.arange(3),
        target_position_world_m=source_position,
        target_semantic_dfr_quat_wxyz=physical_quat,
        smoothed_target_position_source_frame_m=source_position,
        base_xy_yaw=np.column_stack((source_time + allocation_offset, source_time * 0.0, source_time * 0.0)),
        riser_q=np.array([0.2, 0.3, 0.4]) + allocation_offset,
        proxy_gimbal_q=np.zeros((3, 3)) + allocation_offset,
        initialization_time_s=np.array([], dtype=float),
        initialization_state=np.empty((0, 1), dtype=float),
        time_s=execution_time,
        execution_time_s=execution_time,
        feedforward_v_wz=np.ones((2, 2)) / time_scale,
        feedforward_riser_velocity=np.ones(2) / time_scale,
        feedforward_proxy_velocity=np.ones((2, 3)) / time_scale,
    )


def write_gate(path: Path, *, completed: bool, p95: float, maximum: float) -> None:
    payload = {
        "training_started": False,
        "ppo_authorized": False,
        "results": [
            {
                "case": 42,
                "checks": {"completed_reference": completed},
                "dynamic_quality_passed": False,
                "executed_residual_dataset": None,
                "raw_residual_label_applied_to_commands": False,
                "completed_phase_time_s": 2.0 if completed else 1.0,
                "execution_duration_s": 2.0,
                "maximum_runtime_s": 6.0,
                "position_error_p95_m": p95,
                "position_error_max_m": maximum,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_args(tmp_path: Path) -> argparse.Namespace:
    baseline = tmp_path / "baseline.npz"
    preview = tmp_path / "preview.npz"
    retimed = tmp_path / "retimed.npz"
    write_plan(baseline, allocation_offset=0.0, time_scale=1.0)
    write_plan(preview, allocation_offset=0.1, time_scale=1.0)
    write_plan(retimed, allocation_offset=0.1, time_scale=1.5)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "case": 42,
                        "plan_sha256": digest(baseline),
                        "passed": True,
                        "timing_transition_kinematic_gate_passed": True,
                        "checks": {"source_time_verbatim": True},
                        "kinematic_checks": {"position_p95_bounded": True},
                        "kinematic_metrics": {
                            "position_error_p95_m": 0.14,
                            "position_error_max_m": 0.20,
                        },
                        "valid_for_training": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    preview_gate = tmp_path / "preview.json"
    retimed_gate = tmp_path / "retimed.json"
    write_gate(preview_gate, completed=True, p95=0.38, maximum=0.49)
    write_gate(retimed_gate, completed=False, p95=5.0, maximum=5.3)
    return argparse.Namespace(
        baseline_plan=baseline,
        expected_baseline_plan_sha256=digest(baseline),
        baseline_manifest=manifest,
        expected_baseline_manifest_sha256=digest(manifest),
        preview_plan=preview,
        expected_preview_plan_sha256=digest(preview),
        preview_gate=preview_gate,
        expected_preview_gate_sha256=digest(preview_gate),
        retimed_plan=retimed,
        expected_retimed_plan_sha256=digest(retimed),
        retimed_gate=retimed_gate,
        expected_retimed_gate_sha256=digest(retimed_gate),
        expected_case=42,
    )


def test_audit_accepts_closed_static_baseline_after_retime_reject(tmp_path: Path) -> None:
    payload = audit(make_args(tmp_path))
    assert payload["passed"] is True
    assert payload["baseline_candidate"]["dynamically_validated"] is False
    assert payload["runtime_authorized"] is False
    assert payload["valid_for_training"] is False


def test_audit_rejects_mutated_source_array(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    with np.load(args.retimed_plan, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["source_time_s"] = arrays["source_time_s"].copy()
    arrays["source_time_s"][1] += 0.01
    np.savez(args.retimed_plan, **arrays)
    args.expected_retimed_plan_sha256 = digest(args.retimed_plan)
    with pytest.raises(ValueError, match="lineage audit failed"):
        audit(args)


def test_audit_rejects_forged_gate_hash(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    args.expected_retimed_gate_sha256 = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(args)
