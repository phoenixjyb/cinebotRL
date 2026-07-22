import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_initial_teacher41_validation_tranche import (
    EXPECTED_CASES,
    LEARNED_SOURCE,
    ZERO_SOURCE,
    finalize,
)


PROFILE = "riser_recovery_direction_v4_camera_lever_arm_v1"
POLICY_SHA = "a" * 64
CONTRACT_SHA = "c" * 64


def _contract():
    return {
        "cases": EXPECTED_CASES,
        "case_contracts": {
            str(case): {
                "source_duration_s": float(case),
                "execution_duration_s": float(case + 1),
                "camera_lever_arm_cap_m": 0.05,
            }
            for case in EXPECTED_CASES
        },
        "controller_contract": {"tracking_profile": PROFILE},
        "inputs": {"policy_torchscript": {"sha256": POLICY_SHA}},
        "cpu_contract_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "runtime_authorization_token_issued": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
    }


def _write_rollout(path: Path, case: int, source: str, residual: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cases": [case],
                "trajectory_command_source": source,
                "tracking_profile": PROFILE,
                "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
                "raw_teacher_capture_started": False,
                "normalized_dataset_capture_started": False,
                "passed": True,
                "dynamic_quality_passed": True,
                "results": [
                    {
                        "case": case,
                        "source_duration_s": float(case),
                        "execution_duration_s": float(case + 1),
                        "executed_residual_dataset": None,
                        "residual_action_abs_max": residual,
                        "passed": True,
                        "dynamic_quality_passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "tranche"
    root.mkdir()
    (root / "admission.json").write_text(
        json.dumps(
            {
                "passed": True,
                "runtime_commit": "b" * 40,
                "cases": EXPECTED_CASES,
                "split": "validation",
                "cpu_contract": {"sha256": CONTRACT_SHA},
                "policy": {"sha256": POLICY_SHA},
            }
        ),
        encoding="utf-8",
    )
    for case in EXPECTED_CASES:
        _write_rollout(
            root / f"learned/case_{case:04d}.json",
            case,
            LEARNED_SOURCE,
            [0.2, 0.1, 0.05],
        )
        _write_rollout(
            root / f"zero/case_{case:04d}.json",
            case,
            ZERO_SOURCE,
            [0.0, 0.0, 0.0],
        )
        heartbeat = {
            "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
            "case": case,
            "dataset_created": False,
            "valid_for_training": False,
        }
        for mode in ("learned", "zero"):
            (root / f"{mode}/case_{case:04d}_runtime_heartbeat.json").write_text(
                json.dumps(heartbeat), encoding="utf-8"
            )
    rows = [
        {"case": case, "checks": {"hard": True, "comparison": True}}
        for case in EXPECTED_CASES
    ]
    (root / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "policy_sha256": POLICY_SHA,
                "cases": EXPECTED_CASES,
                "case_count": len(EXPECTED_CASES),
                "expected_tracking_profile": PROFILE,
                "maximum_regression_fraction": 0.05,
                "minimum_zero_improvement_fraction": 0.05,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return root


def _status():
    return {
        "schema": (
            "cinebotrl_two_wheel_riser_initial_teacher41_validation_tranche_process_status_v1"
        ),
        "cases": EXPECTED_CASES,
        "learned": {str(case): 0 for case in EXPECTED_CASES},
        "zero": {str(case): 0 for case in EXPECTED_CASES},
    }


def test_healthy_tranche_passes_without_opening_training(tmp_path: Path) -> None:
    result = finalize(
        _root(tmp_path),
        contract=_contract(),
        contract_sha256=CONTRACT_SHA,
        runtime_commit="b" * 40,
        process_status=_status(),
        gate_exit_code=0,
    )
    assert result["passed"]
    assert result["validation_tranche_passed"]
    assert not result["broad_rollout_authorized"]
    assert not result["holdout_opened"]
    assert not result["ppo_authorized"]


def test_rejects_missing_case_and_failed_process(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "learned/case_0022.json").unlink()
    status = _status()
    status["learned"]["22"] = 6
    result = finalize(
        root,
        contract=_contract(),
        contract_sha256=CONTRACT_SHA,
        runtime_commit="b" * 40,
        process_status=status,
        gate_exit_code=6,
    )
    assert not result["passed"]
    assert not result["checks"]["case22_learned_process_exit_zero"]
    assert not result["checks"]["case22_learned_rollout_contract"]


def test_rejects_dataset_or_summary_case_drift(tmp_path: Path) -> None:
    root = _root(tmp_path)
    heartbeat_path = root / "zero/case_0032_runtime_heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text())
    heartbeat["dataset_created"] = True
    heartbeat_path.write_text(json.dumps(heartbeat), encoding="utf-8")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cases"] = [16, 22]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = finalize(
        root,
        contract=_contract(),
        contract_sha256=CONTRACT_SHA,
        runtime_commit="b" * 40,
        process_status=_status(),
        gate_exit_code=0,
    )
    assert not result["checks"]["case32_no_dataset_created"]
    assert not result["checks"]["comparison_gate_passed"]
    assert not result["passed"]


def test_rejects_contract_that_opens_runtime_or_learning(tmp_path: Path) -> None:
    contract = _contract()
    contract["runtime_authorized"] = True
    contract["ppo_authorized"] = True
    result = finalize(
        _root(tmp_path),
        contract=contract,
        contract_sha256=CONTRACT_SHA,
        runtime_commit="b" * 40,
        process_status=_status(),
        gate_exit_code=0,
    )
    assert not result["checks"]["contract_runtime_was_closed"]
    assert not result["checks"]["contract_learning_closed"]
    assert not result["passed"]


def test_rejects_forged_process_status_or_contract_hash(tmp_path: Path) -> None:
    status = _status()
    status["cases"] = [16, 22]
    result = finalize(
        _root(tmp_path),
        contract=_contract(),
        contract_sha256="d" * 64,
        runtime_commit="b" * 40,
        process_status=status,
        gate_exit_code=0,
    )
    assert not result["checks"]["admission_present_and_passed"]
    assert not result["checks"]["process_status_contract_exact"]
    assert not result["passed"]
