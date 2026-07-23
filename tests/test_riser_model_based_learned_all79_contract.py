import copy
import hashlib
import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance.riser_model_based_learned_all79_contract import (
    ALL79_CASES,
    BC_REPORT_SCHEMA,
    CODE_IDENTITY_KEYS,
    DEFAULT_EVALUATION_CONFIG,
    HOLDOUT_GATE_SCHEMA,
    MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
    PLAN_SOURCE_CHECKS,
    VALIDATION_GATE_SCHEMA,
    validate_learned_all79_admission,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_corpus import (
    DEFAULT_RESERVED_HOLDOUT_CASES,
)
from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_all79_contract as contract,
)


ROOT = Path(__file__).parents[1]
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_LEARNED_ALL79_ADMISSION_TEMPLATE_20260723.json"
)
EXECUTION_COMMIT = "a" * 40
VALIDATION_CASES = [8, 16]


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _gate_report(
    root: Path,
    schema: str,
    cases: list[int],
    policy_sha: str,
) -> dict:
    rows = []
    for case in cases:
        identities = {}
        for role in ("teacher", "learned", "zero"):
            path = root / f"{schema}_{role}_{case}.json"
            path.write_text(f"{role}-{case}", encoding="utf-8")
            identities[f"{role}_rollout"] = _identity(path)
        rows.append(
            {
            "case": case,
            "checks": {
                "learned_hard_gate": True,
                "teacher_hard_gate": True,
                "bounded_residual": True,
            },
            "teacher": {},
            "learned": {},
            "zero": {},
            "learned_residual_action_abs_max": [0.5, 0.4, 0.3],
            "learned_beats_zero_position_p95": True,
            **identities,
            }
        )
    return {
        "schema": schema,
        "policy_sha256": policy_sha,
        "cases": cases,
        "case_count": len(cases),
        "maximum_regression_fraction": 0.05,
        "minimum_zero_improvement_fraction": 0.05,
        "expected_tracking_profile": (
            "riser_recovery_direction_v4_camera_lever_arm_v1"
        ),
        "policy_command_contract": (
            "model_based_planner_plus_bounded_policy_residual_v1"
        ),
        "residual_action_scales": [0.05, 0.05, 0.02],
        "rollout_admission": None,
        "preflight_receipt": None,
        "plan_manifest": None,
        "execution_commit": None,
        "means": {
            "teacher_position_p95_m": 0.10,
            "learned_position_p95_m": 0.08,
            "zero_position_p95_m": 0.10,
        },
        "aggregate_checks": {
            "all_case_checks": True,
            "learned_position_mean_within_teacher_budget": True,
            "learned_beats_zero_by_required_mean": True,
            "learned_beats_zero_on_majority_of_cases": True,
        },
        "rows": rows,
        "passed": True,
        "ppo_authorized": False,
    }


def _fixture(tmp_path: Path):
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    policy_identity = _identity(policy)
    bc_report_path = tmp_path / "bc_report.json"
    bc_report = {
        "schema": BC_REPORT_SCHEMA,
        "execution_commit": EXECUTION_COMMIT,
        "split_cases": {"train": [2, 6, 7, 23], "validation": VALIDATION_CASES},
        "torchscript": policy_identity,
        "offline_gate_passed": True,
        "passed": True,
        "valid_for_dynamic_canary": True,
        "training_started": True,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
    }
    bc_report_path.write_text(json.dumps(bc_report), encoding="utf-8")

    validation_path = tmp_path / "validation.json"
    validation = _gate_report(
        tmp_path,
        VALIDATION_GATE_SCHEMA,
        VALIDATION_CASES,
        policy_identity["sha256"],
    )
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    holdout_path = tmp_path / "holdout.json"
    holdout = _gate_report(
        tmp_path,
        HOLDOUT_GATE_SCHEMA,
        DEFAULT_RESERVED_HOLDOUT_CASES,
        policy_identity["sha256"],
    )
    holdout_path.write_text(json.dumps(holdout), encoding="utf-8")

    source_manifest_path = tmp_path / "source_manifest.json"
    source_items = [
        {
            "episode_index": case,
            "source_json_sha256": f"{case:064x}",
            "source_pose_count": 10 + case,
            "source_time_strictly_increasing": True,
            "trajectory_integrity_contract": "exact_source_v1",
            "integrity_passed": True,
        }
        for case in ALL79_CASES
    ]
    source_manifest = {
        "schema": "gik_exact_source_reference_package_v1",
        "trajectory_integrity_contract": "exact_source_v1",
        "episode_count": len(ALL79_CASES),
        "integrity_passed": True,
        "items": source_items,
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    plan_items = []
    for case, source_item in zip(ALL79_CASES, source_items, strict=True):
        plan = plan_dir / f"case_{case:04d}_smoothed_riser_plan_v1.npz"
        plan.write_bytes(f"plan-{case}".encode())
        plan_items.append(
            {
                "case": case,
                "file": plan.name,
                "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "source_json_sha256": source_item["source_json_sha256"],
                "source_pose_count": source_item["source_pose_count"],
                "checks": {name: True for name in PLAN_SOURCE_CHECKS},
            }
        )
    plan_manifest_path = plan_dir / "manifest.json"
    plan_manifest = {
        "schema": "cinebotrl_two_wheel_riser_smoothed_plan_export_v1",
        "plan_schema": "cinebotrl_two_wheel_riser_smoothed_plan_v1",
        "source_manifest_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "source_package_case_count": len(ALL79_CASES),
        "requested_cases": ALL79_CASES,
        "attempted_cases": ALL79_CASES,
        "portfolio_gate_passed": True,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "items": plan_items,
    }
    plan_manifest_path.write_text(json.dumps(plan_manifest), encoding="utf-8")
    plan_identity = {
        "path": plan_manifest_path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(plan_manifest_path.read_bytes()).hexdigest(),
    }

    for mode, cases, report, report_path in (
        (
            "validation_canary",
            VALIDATION_CASES,
            validation,
            validation_path,
        ),
        (
            "holdout",
            DEFAULT_RESERVED_HOLDOUT_CASES,
            holdout,
            holdout_path,
        ),
    ):
        split_admission_path = tmp_path / f"{mode}_admission.json"
        split_admission = {
            "schema": (
                "cinebotrl_two_wheel_riser_"
                "model_based_learned_split_admission_v1"
            ),
            "mode": mode,
            "policy": policy_identity,
            "plan_manifest": plan_identity,
            "execution_commit": EXECUTION_COMMIT,
            "evaluation_config": DEFAULT_EVALUATION_CONFIG,
            "cases": cases,
            "prior_validation_gate_report": (
                None if mode == "validation_canary" else _identity(validation_path)
            ),
            "model_selection_complete": mode == "holdout",
            "prior_validation_gate_passed": mode == "holdout",
            "split_evaluation_approved": True,
            "learned_rollout_authorized": True,
            "residual_capture_authorized": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
        }
        split_admission_path.write_text(
            json.dumps(split_admission),
            encoding="utf-8",
        )
        split_preflight_path = tmp_path / f"{mode}_preflight.json"
        split_preflight = {
            "schema": (
                "cinebotrl_two_wheel_riser_"
                "model_based_learned_split_preflight_v1"
            ),
            "mode": mode,
            "cases": cases,
            "execution_commit": EXECUTION_COMMIT,
            "admission": _identity(split_admission_path),
            "policy": policy_identity,
            "plan_manifest": plan_identity,
            "checks": {"contract": True, "clean": True},
            "runtime_started": False,
            "dataset_written": False,
            "capture_started": False,
            "bc_started": False,
            "ppo_started": False,
            "passed": True,
        }
        split_preflight_path.write_text(
            json.dumps(split_preflight),
            encoding="utf-8",
        )
        report["rollout_admission"] = _identity(split_admission_path)
        report["preflight_receipt"] = _identity(split_preflight_path)
        report["plan_manifest"] = plan_identity
        report["execution_commit"] = EXECUTION_COMMIT
        report_path.write_text(json.dumps(report), encoding="utf-8")

    runtime_assets = {}
    for name in (
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        runtime_assets[name] = path

    code_paths = {}
    code_identities = {}
    for name in CODE_IDENTITY_KEYS:
        path = tmp_path / f"{name}.py"
        path.write_text(name, encoding="utf-8")
        code_paths[name] = path
        code_identities[name] = _identity(path)
    admission = {
        "schema": MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
        "bc_report": _identity(bc_report_path),
        "policy": policy_identity,
        "plan_manifest": plan_identity,
        "source_manifest": _identity(source_manifest_path),
        **{
            name: _identity(path)
            for name, path in runtime_assets.items()
        },
        "validation_gate_report": _identity(validation_path),
        "holdout_gate_report": _identity(holdout_path),
        "execution_commit": EXECUTION_COMMIT,
        "code": code_identities,
        "evaluation_config": DEFAULT_EVALUATION_CONFIG,
        "validation_cases": VALIDATION_CASES,
        "holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        "all79_cases": ALL79_CASES,
        "model_selection_complete": True,
        "validation_gate_passed": True,
        "holdout_gate_passed": True,
        "holdout_opened_only_after_model_selection": True,
        "all79_evaluation_approved": True,
        "learned_rollout_authorized": True,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    return {
        "admission": admission,
        "bc_report_path": bc_report_path,
        "bc_report": bc_report,
        "policy_path": policy,
        "plan_manifest_path": plan_manifest_path,
        "source_manifest_path": source_manifest_path,
        **{f"{name}_path": path for name, path in runtime_assets.items()},
        "validation_report_path": validation_path,
        "validation_report": validation,
        "holdout_report_path": holdout_path,
        "holdout_report": holdout,
        "code_paths": code_paths,
    }


def _validate(fixture, *, require_authorized: bool = True) -> None:
    validate_learned_all79_admission(
        fixture["admission"],
        identity_root=fixture["bc_report_path"].parent,
        bc_report_path=fixture["bc_report_path"],
        bc_report=fixture["bc_report"],
        policy_path=fixture["policy_path"],
        plan_manifest_path=fixture["plan_manifest_path"],
        source_manifest_path=fixture["source_manifest_path"],
        lqr_gains_path=fixture["lqr_gains_path"],
        robot_build_audit_path=fixture["robot_build_audit_path"],
        robot_usd_path=fixture["robot_usd_path"],
        drive_profile_selection_path=fixture["drive_profile_selection_path"],
        validation_report_path=fixture["validation_report_path"],
        validation_report=fixture["validation_report"],
        holdout_report_path=fixture["holdout_report_path"],
        holdout_report=fixture["holdout_report"],
        code_paths=fixture["code_paths"],
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=require_authorized,
    )


@pytest.fixture(autouse=True)
def _accept_bc_and_policy_artifact(monkeypatch) -> None:
    monkeypatch.setattr(contract, "_bc_report_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        contract,
        "model_based_residual_torchscript_valid",
        lambda *args, **kwargs: True,
    )


def test_authorized_all79_admission_binds_all_prerequisites(tmp_path: Path) -> None:
    _validate(_fixture(tmp_path))


def test_all79_admission_rejects_invalid_policy_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "model_based_residual_torchscript_valid",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(ValueError, match="admission failed"):
        _validate(_fixture(tmp_path))


def test_all79_admission_rejects_unvalidated_bc_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(contract, "_bc_report_valid", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="admission failed"):
        _validate(_fixture(tmp_path))


def test_admission_preserves_majority_zero_baseline_gate_semantics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    holdout = fixture["holdout_report"]
    holdout["rows"][0]["learned_beats_zero_position_p95"] = False
    fixture["holdout_report_path"].write_text(
        json.dumps(holdout),
        encoding="utf-8",
    )
    fixture["admission"]["holdout_gate_report"] = _identity(
        fixture["holdout_report_path"]
    )
    _validate(fixture)


@pytest.mark.parametrize(
    "mutation",
    (
        "bc_hash",
        "policy_hash",
        "plan_hash",
        "plan_file",
        "source_hash",
        "robot_usd_hash",
        "code_hash",
        "validation_failed",
        "holdout_failed",
        "validation_provenance",
        "validation_admission_unsafe_rehashed",
        "holdout_preflight_unsafe_rehashed",
        "all79_cases",
        "config",
        "ppo",
        "authorization",
    ),
)
def test_admission_rejects_forged_or_open_downstream_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    admission = fixture["admission"]
    if mutation == "bc_hash":
        admission["bc_report"]["sha256"] = "0" * 64
    elif mutation == "policy_hash":
        admission["policy"]["sha256"] = "0" * 64
    elif mutation == "plan_hash":
        admission["plan_manifest"]["sha256"] = "0" * 64
    elif mutation == "plan_file":
        plan = fixture["plan_manifest_path"].parent / (
            "case_0023_smoothed_riser_plan_v1.npz"
        )
        plan.write_bytes(b"tampered-plan")
    elif mutation == "source_hash":
        admission["source_manifest"]["sha256"] = "0" * 64
    elif mutation == "robot_usd_hash":
        admission["robot_usd"]["sha256"] = "0" * 64
    elif mutation == "code_hash":
        admission["code"]["playback"]["sha256"] = "0" * 64
    elif mutation == "validation_failed":
        fixture["validation_report"]["passed"] = False
    elif mutation == "holdout_failed":
        fixture["holdout_report"]["aggregate_checks"][
            "learned_beats_zero_on_majority_of_cases"
        ] = False
    elif mutation == "validation_provenance":
        fixture["validation_report"]["rollout_admission"]["sha256"] = "0" * 64
    elif mutation == "validation_admission_unsafe_rehashed":
        split_admission = Path(
            fixture["validation_report"]["rollout_admission"]["path"]
        )
        split_admission = fixture["validation_report_path"].parent / split_admission
        payload = json.loads(split_admission.read_text(encoding="utf-8"))
        payload["ppo_authorized"] = True
        split_admission.write_text(json.dumps(payload), encoding="utf-8")
        fixture["validation_report"]["rollout_admission"] = _identity(
            split_admission
        )
        fixture["validation_report_path"].write_text(
            json.dumps(fixture["validation_report"]),
            encoding="utf-8",
        )
        admission["validation_gate_report"] = _identity(
            fixture["validation_report_path"]
        )
    elif mutation == "holdout_preflight_unsafe_rehashed":
        preflight = Path(
            fixture["holdout_report"]["preflight_receipt"]["path"]
        )
        preflight = fixture["holdout_report_path"].parent / preflight
        payload = json.loads(preflight.read_text(encoding="utf-8"))
        payload["checks"]["clean"] = False
        preflight.write_text(json.dumps(payload), encoding="utf-8")
        fixture["holdout_report"]["preflight_receipt"] = _identity(preflight)
        fixture["holdout_report_path"].write_text(
            json.dumps(fixture["holdout_report"]),
            encoding="utf-8",
        )
        admission["holdout_gate_report"] = _identity(
            fixture["holdout_report_path"]
        )
    elif mutation == "all79_cases":
        admission["all79_cases"] = ALL79_CASES[:-1]
    elif mutation == "config":
        admission["evaluation_config"] = copy.deepcopy(DEFAULT_EVALUATION_CONFIG)
        admission["evaluation_config"]["residual_action_scales"] = [0.1, 0.1, 0.1]
    elif mutation == "ppo":
        admission["ppo_authorized"] = True
    else:
        admission["all79_evaluation_approved"] = False
    with pytest.raises(ValueError, match="admission failed"):
        _validate(fixture)


def test_checked_in_template_is_structural_but_unusable() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["schema"] == MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA
    assert template["evaluation_config"] == DEFAULT_EVALUATION_CONFIG
    assert template["plan_manifest"] == {"path": None, "sha256": None}
    assert template["source_manifest"] == {"path": None, "sha256": None}
    assert template["robot_usd"]["sha256"] is None
    assert template["validation_cases"] == []
    assert template["holdout_cases"] == DEFAULT_RESERVED_HOLDOUT_CASES
    assert template["all79_cases"] == ALL79_CASES
    assert template["all79_evaluation_approved"] is False
    assert template["learned_rollout_authorized"] is False
    assert template["ppo_authorized"] is False
    assert template["training_started"] is False
