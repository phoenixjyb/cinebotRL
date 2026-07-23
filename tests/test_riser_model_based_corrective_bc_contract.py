import copy
import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_bc_contract import (
    CODE_IDENTITY_KEYS,
    DEFAULT_BC_TRAINING_CONFIG,
    MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_SCHEMA,
    MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA,
    sha256_file,
    validate_bc_execution_admission,
    validate_bc_execution_report,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_training_dataset import (
    MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
)


ROOT = Path(__file__).parents[1]
CODE_PATHS = {
    "trainer": ROOT / "scripts/two_wheel_balance/train_riser_residual_bc.py",
    "adapter": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_corrective_bc_adapter.py"
    ),
    "loss_module": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_bc_loss.py"
    ),
    "policy_module": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_residual_policy.py"
    ),
    "training_dataset_module": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_corrective_training_dataset.py"
    ),
    "admission_module": (
        ROOT
        / "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_corrective_bc_contract.py"
    ),
}
EXECUTION_COMMIT = "a" * 40
PROMOTION_COMMIT = "b" * 40
SPLIT_CASES = {"train": [1, 2, 6, 7], "validation": [8, 16]}
HOLDOUT_CASES = [3, 5, 13, 19, 24]
ADMISSION_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_TEMPLATE_20260723.json"
)


def _identity(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _fixture(tmp_path: Path, *, authorized: bool) -> tuple[
    Path,
    dict[str, object],
    dict[str, object],
]:
    dataset = tmp_path / "training.npz"
    dataset.write_bytes(b"projection-aware-training-dataset")
    metadata = {
        "schema": MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
        "promotion_commit": PROMOTION_COMMIT,
        "split_cases": SPLIT_CASES,
        "valid_for_projection_aware_bc_input": True,
        "valid_for_training": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "training_started": False,
    }
    admission = {
        "schema": MODEL_BASED_CORRECTIVE_BC_EXECUTION_ADMISSION_SCHEMA,
        "dataset": _identity(dataset),
        "dataset_schema": MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
        "dataset_promotion_commit": PROMOTION_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "code": {name: _identity(path) for name, path in CODE_PATHS.items()},
        "optimizer_contract": (
            "exact_case_balanced_projection_aware_gradient_accumulation_v1"
        ),
        "validation_contract": (
            "projected_effective_action_case_balanced_validation_v1"
        ),
        "loss_contract": "model_based_projected_effective_action_bc_loss_v1",
        "projection_contract": "model_based_residual_safety_projection_v1",
        "training_config": copy.deepcopy(DEFAULT_BC_TRAINING_CONFIG),
        "split_cases": copy.deepcopy(SPLIT_CASES),
        "reserved_holdout_cases": HOLDOUT_CASES,
        "holdout_opened": False,
        "bc_execution_approved": authorized,
        "bc_authorized": authorized,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "training_started": False,
    }
    return dataset, metadata, admission


def test_checked_in_execution_admission_template_is_unusable() -> None:
    template = json.loads(ADMISSION_TEMPLATE.read_text(encoding="utf-8"))
    assert template["training_config"] == DEFAULT_BC_TRAINING_CONFIG
    assert template["training_config"]["policy_architecture"] == (
        MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
    )
    assert template["training_config"]["observation_dimension"] == 65
    assert template["training_config"]["base_observation_dimension"] == 26
    assert template["training_config"]["action_dimension"] == 3
    assert template["training_config"]["zero_initialize_action_head"] is True
    assert template["dataset"] == {"path": None, "sha256": None}
    assert template["dataset_promotion_commit"] is None
    assert template["execution_commit"] is None
    assert template["split_cases"] == {"train": [], "validation": []}
    assert template["bc_execution_approved"] is False
    assert template["bc_authorized"] is False
    assert template["training_started"] is False
    for name, path in CODE_PATHS.items():
        expected = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        assert template["code"][name] == expected


def test_cpu_review_admission_stays_closed_until_separately_authorized(
    tmp_path: Path,
) -> None:
    dataset, metadata, admission = _fixture(tmp_path, authorized=False)
    validate_bc_execution_admission(
        admission,
        dataset_path=dataset,
        dataset_metadata=metadata,
        code_paths=CODE_PATHS,
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=False,
    )
    with pytest.raises(ValueError, match="authorized"):
        validate_bc_execution_admission(
            admission,
            dataset_path=dataset,
            dataset_metadata=metadata,
            code_paths=CODE_PATHS,
            expected_execution_commit=EXECUTION_COMMIT,
            require_authorized=True,
        )


def test_exact_authorized_admission_passes_all_hash_and_split_checks(
    tmp_path: Path,
) -> None:
    dataset, metadata, admission = _fixture(tmp_path, authorized=True)
    assert set(admission["code"]) == CODE_IDENTITY_KEYS
    validate_bc_execution_admission(
        admission,
        dataset_path=dataset,
        dataset_metadata=metadata,
        code_paths=CODE_PATHS,
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=True,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset_hash",
        "adapter_hash",
        "config",
        "holdout",
        "commit",
        "ppo",
        "approval_mismatch",
    ],
)
def test_admission_rejects_forged_or_unsafe_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset, metadata, admission = _fixture(tmp_path, authorized=True)
    if mutation == "dataset_hash":
        admission["dataset"]["sha256"] = "0" * 64
    elif mutation == "adapter_hash":
        admission["code"]["adapter"]["sha256"] = "0" * 64
    elif mutation == "config":
        admission["training_config"]["learning_rate"] = 0.01
    elif mutation == "holdout":
        admission["split_cases"]["validation"].append(HOLDOUT_CASES[0])
    elif mutation == "commit":
        admission["execution_commit"] = "c" * 40
    elif mutation == "ppo":
        admission["ppo_authorized"] = True
    else:
        admission["bc_execution_approved"] = False
    with pytest.raises(ValueError, match="admission failed"):
        validate_bc_execution_admission(
            admission,
            dataset_path=dataset,
            dataset_metadata=metadata,
            code_paths=CODE_PATHS,
            expected_execution_commit=EXECUTION_COMMIT,
            require_authorized=True,
        )


def _metrics(improves: list[bool]) -> dict[str, object]:
    return {
        "loss_total": 0.01,
        "case_balanced_mse_per_action": [0.001, 0.002, 0.0015],
        "zero_requested_case_balanced_mse_per_action": [0.01, 0.02, 0.015],
        "improves_over_zero_requested": improves,
        "requested_action_abs_max": [0.4, 0.5, 0.3],
        "requested_slew_violation_count": [0, 0, 0],
    }


def _report_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    dataset, metadata, admission = _fixture(tmp_path, authorized=True)
    validate_bc_execution_admission(
        admission,
        dataset_path=dataset,
        dataset_metadata=metadata,
        code_paths=CODE_PATHS,
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=True,
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    checkpoint = tmp_path / "residual_policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    torchscript = tmp_path / "residual_policy.torchscript.pt"
    torchscript.write_bytes(b"torchscript")
    report = {
        "schema": MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA,
        "admission": _identity(admission_path),
        "dataset": admission["dataset"],
        "execution_commit": EXECUTION_COMMIT,
        "optimizer_contract": admission["optimizer_contract"],
        "validation_contract": admission["validation_contract"],
        "loss_contract": admission["loss_contract"],
        "training_config": admission["training_config"],
        "split_cases": admission["split_cases"],
        "reserved_holdout_cases": HOLDOUT_CASES,
        "epochs_run": 2,
        "optimizer_steps": 2,
        "best_epoch": 2,
        "history": [
            {
                "epoch": 1,
                "train_loss_total": 0.04,
                "train_loss_pointwise": 0.03,
                "train_loss_requested_slew": 0.1,
                "validation_loss_total": 0.03,
            },
            {
                "epoch": 2,
                "train_loss_total": 0.02,
                "train_loss_pointwise": 0.015,
                "train_loss_requested_slew": 0.05,
                "validation_loss_total": 0.01,
            },
        ],
        "split_metrics": {
            "train": _metrics([True, True, True]),
            "validation": _metrics([True, True, True]),
        },
        "offline_gate_passed": True,
        "holdout_used_for_model_selection": False,
        "holdout_metrics_computed": False,
        "checkpoint": _identity(checkpoint),
        "torchscript": _identity(torchscript),
        "bc_authorized": True,
        "training_started": True,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "passed": True,
        "valid_for_dynamic_canary": True,
    }
    return admission_path, admission, report


def test_success_report_requires_metrics_hashes_and_closed_holdout(
    tmp_path: Path,
) -> None:
    admission_path, admission, report = _report_fixture(tmp_path)
    validate_bc_execution_report(
        report,
        admission_path=admission_path,
        admission=admission,
        report_directory=tmp_path,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_artifact",
        "false_success",
        "holdout",
        "prediction_margin",
        "slew_violation",
    ],
)
def test_report_rejects_false_success_claims(
    tmp_path: Path,
    mutation: str,
) -> None:
    admission_path, admission, report = _report_fixture(tmp_path)
    if mutation == "missing_artifact":
        report["checkpoint"] = None
    elif mutation == "false_success":
        report["offline_gate_passed"] = False
    elif mutation == "holdout":
        report["holdout_metrics_computed"] = True
    elif mutation == "prediction_margin":
        report["split_metrics"]["validation"]["requested_action_abs_max"][0] = 0.95
    else:
        report["split_metrics"]["validation"][
            "requested_slew_violation_count"
        ][0] = 1
    with pytest.raises(ValueError, match="report failed"):
        validate_bc_execution_report(
            report,
            admission_path=admission_path,
            admission=admission,
            report_directory=tmp_path,
        )
