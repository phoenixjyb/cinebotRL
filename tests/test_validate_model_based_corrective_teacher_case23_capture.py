import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = ROOT / "scripts/two_wheel_balance/validate_model_based_corrective_teacher_case23_capture.py"
WRAPPER = ROOT / "scripts/two_wheel_balance/run_model_based_corrective_teacher_case23_capture.sh"
CONTRACT = ROOT / "scripts/two_wheel_balance/model_based_corrective_teacher_case23_capture_contract_v1.json"
DRIVE_PROFILE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_drive_profile_selection_v1/summary.json"
)
SPEC = importlib.util.spec_from_file_location("case23_capture_validator", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_committed_contract_pins_passed_pair_and_closes_consumed_authorization() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["case"] == 23
    assert contract["split"] == "train"
    assert contract["namespace"] == MODULE.NAMESPACE
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["identities"]["paired_final_status"]["sha256"] == (
        "67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb"
    )
    assert contract["identities"]["case23_plan"]["sha256"] == (
        "ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6"
    )
    assert contract["identities"]["drive_profile_selection"]["sha256"] == (
        "39a700de3985175e4e8415f1f23beef4264b103daa7ce8847f4ac0fe69f879f7"
    )
    assert "drive_profile_selection" in MODULE.REQUIRED_IDENTITIES
    assert "drive_profile_selection" in MODULE.TRACKED_IDENTITIES
    assert contract["execution_contract"] == MODULE.EXPECTED_EXECUTION
    assert contract["runtime_authorized"] is False
    assert contract["gpu_launch_authorized"] is False
    assert contract["authorization_token_issued"] is False
    assert contract["runtime_authorization_token_sha256"] == ""
    assert contract["label_capture_authorized"] is False
    assert contract["dataset_creation_authorized"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["training_started"] is False


def test_case23_adapter_passes_exact_configuration_to_generic_validator(monkeypatch) -> None:
    observed = {}

    def fake_validate(contract_path, repo, **kwargs):
        observed.update(kwargs)
        return {"schema": "base", "passed": True}

    monkeypatch.setattr(MODULE, "validate_capture", fake_validate)
    result = MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
    )
    assert result["schema"] == MODULE.ADMISSION_SCHEMA
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["reviewed_parent"] == MODULE.REVIEWED_PARENT
    assert observed["plan_identity_name"] == "case23_plan"
    assert observed["pair_schema"] == MODULE.PAIR_SCHEMA
    assert observed["required_identities"] == MODULE.REQUIRED_IDENTITIES
    assert observed["tracked_identities"] == MODULE.TRACKED_IDENTITIES
    assert observed["expected_execution"] == MODULE.EXPECTED_EXECUTION


def test_wrapper_is_capture_only_and_closes_consumed_authorization() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'readonly AUTHORIZATION_SHA256=""' in source
    assert 'readonly OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\\\$NAMESPACE"' in source
    assert "--cases 23" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--corrective-teacher-capture-dir" in source
    assert "--corrective-teacher-capture-admission" in source
    assert "--raw-teacher-capture" not in source
    assert '"$PY" -u -X utf8 train_riser_residual_bc' not in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    assert "runtime_authorization_not_issued" in result.stderr


def test_case23_adapter_rejects_wrong_namespace_before_runtime(monkeypatch) -> None:
    def fake_validate(contract_path, repo, **kwargs):
        return {
            "schema": "base",
            "passed": kwargs["namespace"] == kwargs["expected_namespace"],
        }

    monkeypatch.setattr(MODULE, "validate_capture", fake_validate)
    result = MODULE.validate(CONTRACT, ROOT, namespace="alternate")
    assert result["passed"] is False


def test_drive_profile_semantics_bind_capture_to_400w_plant() -> None:
    checks = MODULE.drive_profile_checks(
        {
            "identities": {
                "drive_profile_selection": {
                    "passed": True,
                    "path": str(DRIVE_PROFILE),
                }
            }
        }
    )
    assert all(checks.values())


def test_drive_profile_mismatch_revokes_runtime_authorization(
    monkeypatch, tmp_path: Path
) -> None:
    profile = json.loads(DRIVE_PROFILE.read_text(encoding="utf-8"))
    profile["production_design_candidate"]["simulation_enabled"] = True
    changed = tmp_path / "changed_profile.json"
    changed.write_text(json.dumps(profile), encoding="utf-8")

    def fake_validate(contract_path, repo, **kwargs):
        return {
            "schema": "base",
            "passed": True,
            "checks": {},
            "cpu_contract_ready": True,
            "authorization_consumed_before_isaac": True,
            "runtime_authorized": True,
            "gpu_launch_authorized": True,
            "label_capture_authorized": True,
            "identities": {
                "drive_profile_selection": {
                    "passed": True,
                    "path": str(changed),
                }
            },
        }

    monkeypatch.setattr(MODULE, "validate_capture", fake_validate)
    result = MODULE.validate(CONTRACT, ROOT, namespace=MODULE.NAMESPACE)
    assert result["drive_profile_checks"][
        "750w_candidate_not_simulation_enabled"
    ] is False
    assert result["cpu_contract_ready"] is False
    assert result["authorization_consumed_before_isaac"] is False
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["passed"] is False
