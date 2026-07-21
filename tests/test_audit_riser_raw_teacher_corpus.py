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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/audit_riser_raw_teacher_corpus.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, applied_case: int | None = None) -> list[str]:
    gate_dir = tmp_path / "gates"
    raw_dir = tmp_path / "raw"
    gate_dir.mkdir()
    raw_dir.mkdir()
    cases = list(range(1, 41))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
                "passed": True,
                "selected_cases": cases,
                "rows": [
                    {"case": case, "plan_sha256": f"{case:064x}"} for case in cases
                ],
            }
        ),
        encoding="utf-8",
    )
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_smoothed_representative_admission_v1",
                "passed": True,
                "requested_cases": cases,
                "selection_sha256": _sha(selection),
                "selected_plans": [
                    {"case": case, "plan_sha256": f"{case:064x}"} for case in cases
                ],
                "raw_teacher_capture_authorized": True,
                "normalized_dataset_capture_authorized": False,
                "residual_action_application_authorized": False,
                "training_started": False,
                "bc_authorized": False,
                "ppo_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    for case in cases:
        count = 2
        observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
        observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
        raw_value = 0.32 if case == 40 else 0.1
        raw_commands = np.tile([raw_value, 0.1, 0.01], (count, 1)).astype(
            np.float32
        )
        raw_path = raw_dir / f"case_{case:04d}_executed_raw_teacher_v1.npz"
        save_raw_teacher_case(
            raw_path,
            case,
            {
                "observations": observations,
                "raw_residual_commands": raw_commands,
                "case_ids": np.full(count, case, dtype=np.int16),
                "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
                "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
                "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
                "teacher_commands": np.tile(
                    [0.1 + raw_value, 0.1, 0.01], (count, 1)
                ).astype(np.float32),
            },
        )
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
            "executed_residual_dataset": None,
            "raw_residual_label_applied_to_commands": False,
            "residual_action_abs_max": (
                [0.1, 0.0, 0.0] if case == applied_case else [0.0, 0.0, 0.0]
            ),
            "raw_residual_command_abs_max": [raw_value, 0.1, 0.01],
            "checks": {"completed_reference": True},
        }
        (gate_dir / f"case_{case:04d}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "dynamic_quality_passed": True,
                    "thermal_admission_passed": True,
                    "controller_evidence_passed": True,
                    "results": [result],
                }
            ),
            encoding="utf-8",
        )
    return [
        sys.executable,
        str(SCRIPT),
        "--selection",
        str(selection),
        "--admission",
        str(admission),
        "--gate-dir",
        str(gate_dir),
        "--raw-dir",
        str(raw_dir),
        "--expected-count",
        "40",
        "--output",
        str(tmp_path / "audit.json"),
    ]


def test_audits_corpus_and_freezes_margin_scales(tmp_path: Path) -> None:
    result = subprocess.run(
        _fixture(tmp_path), cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["capture_admission_passed"]
    assert audit["case_count"] == 40
    assert audit["frozen_action_scales"] == [0.4, 0.4, 0.1]
    assert audit["action_scale_frozen"]
    assert audit["valid_for_bc_initialization"]
    assert not audit["bc_authorized"]
    assert not audit["ppo_authorized"]


def test_rejects_corpus_with_applied_residual(tmp_path: Path) -> None:
    result = subprocess.run(
        _fixture(tmp_path, applied_case=7),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "raw corpus case 7 failed" in result.stderr


def _make_subset_admission(tmp_path: Path, *, forge_excluded: bool = False) -> list[str]:
    command = _fixture(tmp_path)
    admission_path = tmp_path / "admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"passed": True}), encoding="utf-8")
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"stopped_case": 41}), encoding="utf-8")
    excluded_gate = tmp_path / "gates/case_0041.json"
    excluded_gate.write_text(
        json.dumps(
            {
                "passed": False,
                "dynamic_quality_passed": False,
                "results": [
                    {
                        "case": 41,
                        "passed": False,
                        "dynamic_quality_passed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    admission.update(
        {
            "schema": "cinebotrl_two_wheel_riser_raw_teacher_subset_admission_v1",
            "parent_admission": {"path": str(parent), "sha256": _sha(parent)},
            "progress_status": {
                "path": str(progress),
                "sha256": _sha(progress),
            },
            "retained_case_evidence": [
                {
                    "case": case,
                    "gate": {
                        "sha256": _sha(tmp_path / f"gates/case_{case:04d}.json")
                    },
                    "raw_case": {
                        "sha256": _sha(
                            tmp_path
                            / f"raw/case_{case:04d}_executed_raw_teacher_v1.npz"
                        )
                    },
                }
                for case in range(1, 41)
            ],
            "excluded_case_evidence": [
                {
                    "case": 41,
                    "gate": {
                        "sha256": "0" * 64
                        if forge_excluded
                        else _sha(excluded_gate)
                    },
                }
            ],
            "corpus_audit_authorized": True,
            "runtime_authorized": False,
            "new_raw_teacher_capture_authorized": False,
        }
    )
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    return command


def test_accepts_provenance_bound_subset_with_extra_rejected_gate(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        _make_subset_admission(tmp_path),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["top_checks"]["subset_provenance"]
    assert audit["subset_checks"]["excluded_are_dynamic_rejects"]


def test_rejects_subset_with_forged_excluded_gate_hash(tmp_path: Path) -> None:
    result = subprocess.run(
        _make_subset_admission(tmp_path, forge_excluded=True),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert not audit["subset_checks"]["excluded_gate_hashes"]
