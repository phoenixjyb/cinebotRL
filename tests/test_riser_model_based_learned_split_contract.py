import copy
import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_split_contract as contract,
)
from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_all79_contract as all79_contract,
)


ROOT = Path(__file__).parents[1]
EXECUTION_COMMIT = "a" * 40
VALIDATION_CASES = [8, 16]
HOLDOUT_CASES = [3, 5, 13, 19, 24]
DEFAULT_EVALUATION_CONFIG = all79_contract.DEFAULT_EVALUATION_CONFIG
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_LEARNED_SPLIT_ADMISSION_TEMPLATE_20260723.json"
)


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": contract.sha256_file(path)}


def _fixture(
    tmp_path: Path,
    *,
    mode: str,
    authorized: bool,
) -> tuple[dict, dict, dict[str, Path], Path | None, dict | None]:
    paths = {}
    for name in (
        "policy",
        "source_manifest",
        "plan_manifest",
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    bc_admission = tmp_path / "bc_admission.json"
    bc_admission.write_text("{}\n", encoding="utf-8")
    bc_report = {
        "schema": (
            "cinebotrl_two_wheel_riser_"
            "model_based_corrective_bc_execution_report_v1"
        ),
        "admission": _identity(bc_admission),
        "execution_commit": EXECUTION_COMMIT,
        "split_cases": {
            "train": [1, 2, 6, 7],
            "validation": VALIDATION_CASES,
        },
        "reserved_holdout_cases": HOLDOUT_CASES,
        "torchscript": _identity(paths["policy"]),
        "passed": True,
        "offline_gate_passed": True,
        "valid_for_dynamic_canary": True,
        "training_started": True,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
    }
    bc_report_path = tmp_path / "bc_report.json"
    bc_report_path.write_text(json.dumps(bc_report), encoding="utf-8")
    paths["bc_report"] = bc_report_path
    code_paths = {}
    for name in contract.CODE_IDENTITY_KEYS:
        path = tmp_path / f"{name}.py"
        path.write_text(name, encoding="utf-8")
        code_paths[name] = path
    prior_path = None
    prior = None
    if mode == "holdout":
        prior_path = tmp_path / "validation_gate.json"
        prior = {"passed": True}
        prior_path.write_text(json.dumps(prior), encoding="utf-8")
    admission = {
        "schema": contract.MODEL_BASED_LEARNED_SPLIT_ADMISSION_SCHEMA,
        "mode": mode,
        "bc_report": _identity(bc_report_path),
        "policy": _identity(paths["policy"]),
        "plan_manifest": _identity(paths["plan_manifest"]),
        "source_manifest": _identity(paths["source_manifest"]),
        "lqr_gains": _identity(paths["lqr_gains"]),
        "robot_build_audit": _identity(paths["robot_build_audit"]),
        "robot_usd": _identity(paths["robot_usd"]),
        "drive_profile_selection": _identity(paths["drive_profile_selection"]),
        "prior_validation_gate_report": (
            None if prior_path is None else _identity(prior_path)
        ),
        "execution_commit": EXECUTION_COMMIT,
        "code": {name: _identity(path) for name, path in code_paths.items()},
        "evaluation_config": copy.deepcopy(DEFAULT_EVALUATION_CONFIG),
        "cases": (
            VALIDATION_CASES if mode == "validation_canary" else HOLDOUT_CASES
        ),
        "model_selection_complete": mode == "holdout",
        "prior_validation_gate_passed": mode == "holdout",
        "split_evaluation_approved": authorized,
        "learned_rollout_authorized": authorized,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    paths.update(code_paths)
    paths["code_paths"] = code_paths
    return admission, bc_report, paths, prior_path, prior


def _validate(
    tmp_path: Path,
    admission: dict,
    bc_report: dict,
    paths: dict,
    prior_path: Path | None,
    prior: dict | None,
    *,
    require_authorized: bool,
) -> None:
    contract.validate_learned_split_admission(
        admission,
        identity_root=tmp_path,
        mode=admission["mode"],
        bc_report_path=paths["bc_report"],
        bc_report=bc_report,
        policy_path=paths["policy"],
        plan_manifest_path=paths["plan_manifest"],
        source_manifest_path=paths["source_manifest"],
        lqr_gains_path=paths["lqr_gains"],
        robot_build_audit_path=paths["robot_build_audit"],
        robot_usd_path=paths["robot_usd"],
        drive_profile_selection_path=paths["drive_profile_selection"],
        prior_validation_report_path=prior_path,
        prior_validation_report=prior,
        code_paths=paths["code_paths"],
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=require_authorized,
    )


@pytest.fixture(autouse=True)
def _accept_structural_fixture_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(contract, "_bc_report_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        contract,
        "_exact_source_manifest_valid",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(contract, "_plan_manifest_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(contract, "_gate_report_valid", lambda *args, **kwargs: True)


def test_checked_in_template_is_closed_and_unusable() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["mode"] is None
    assert template["cases"] == []
    assert template["execution_commit"] is None
    assert template["prior_validation_gate_report"] is None
    assert template["split_evaluation_approved"] is False
    assert template["learned_rollout_authorized"] is False
    assert template["residual_capture_authorized"] is False
    assert template["bc_authorized"] is False
    assert template["ppo_authorized"] is False
    assert template["training_started"] is False


@pytest.mark.parametrize("mode", ["validation_canary", "holdout"])
def test_cpu_review_passes_but_runtime_requires_separate_authorization(
    tmp_path: Path,
    mode: str,
) -> None:
    fixture = _fixture(tmp_path, mode=mode, authorized=False)
    _validate(tmp_path, *fixture, require_authorized=False)
    with pytest.raises(ValueError, match="authorized"):
        _validate(tmp_path, *fixture, require_authorized=True)


@pytest.mark.parametrize("mode", ["validation_canary", "holdout"])
def test_exact_authorized_split_passes(tmp_path: Path, mode: str) -> None:
    fixture = _fixture(tmp_path, mode=mode, authorized=True)
    _validate(tmp_path, *fixture, require_authorized=True)


@pytest.mark.parametrize(
    ("mode", "mutation"),
    [
        ("validation_canary", "holdout_case"),
        ("validation_canary", "model_selection"),
        ("validation_canary", "prior_report"),
        ("holdout", "missing_prior"),
        ("holdout", "model_selection"),
        ("holdout", "wrong_cases"),
        ("holdout", "prior_not_passed"),
        ("holdout", "ppo"),
        ("holdout", "code_hash"),
        ("holdout", "commit"),
    ],
)
def test_rejects_stage_leakage_and_forged_identity(
    tmp_path: Path,
    mode: str,
    mutation: str,
) -> None:
    admission, bc_report, paths, prior_path, prior = _fixture(
        tmp_path,
        mode=mode,
        authorized=True,
    )
    if mutation == "holdout_case":
        admission["cases"] = [HOLDOUT_CASES[0], VALIDATION_CASES[0]]
    elif mutation == "model_selection":
        admission["model_selection_complete"] = mode == "validation_canary"
    elif mutation == "prior_report":
        prior_path = tmp_path / "unexpected.json"
        prior_path.write_text("{}", encoding="utf-8")
        prior = {}
        admission["prior_validation_gate_report"] = _identity(prior_path)
    elif mutation == "missing_prior":
        prior_path = None
        prior = None
        admission["prior_validation_gate_report"] = None
    elif mutation == "wrong_cases":
        admission["cases"] = [3, 5]
    elif mutation == "prior_not_passed":
        admission["prior_validation_gate_passed"] = False
    elif mutation == "ppo":
        admission["ppo_authorized"] = True
    elif mutation == "code_hash":
        admission["code"]["playback"]["sha256"] = "0" * 64
    else:
        admission["execution_commit"] = "b" * 40
    with pytest.raises(ValueError, match="admission failed"):
        _validate(
            tmp_path,
            admission,
            bc_report,
            paths,
            prior_path,
            prior,
            require_authorized=True,
        )
