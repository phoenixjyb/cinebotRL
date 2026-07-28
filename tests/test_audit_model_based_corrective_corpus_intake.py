import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_corpus_intake.py"
)
SPEC = importlib.util.spec_from_file_location("corrective_corpus_intake", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    save_case_dataset,
)


def _audit(**overrides):
    inputs = {
        "train_selection_path": MODULE.DEFAULT_TRAIN_SELECTION,
        "validation_selection_path": MODULE.DEFAULT_VALIDATION_SELECTION,
        "case30_dataset_path": MODULE.DEFAULT_CASE30_DATASET,
        "case30_audit_path": MODULE.DEFAULT_CASE30_AUDIT,
        "case23_final_path": MODULE.DEFAULT_CASE23_FINAL,
        "case23_preflight_path": MODULE.DEFAULT_CASE23_PREFLIGHT,
        "case23_dataset_path": MODULE.DEFAULT_CASE23_DATASET,
        "case23_conversion_final_path": MODULE.DEFAULT_CASE23_CONVERSION_FINAL,
        "case23_recovery_audit_path": MODULE.DEFAULT_CASE23_RECOVERY_AUDIT,
        "case6_dataset_path": MODULE.DEFAULT_CASE6_DATASET,
        "case6_conversion_final_path": MODULE.DEFAULT_CASE6_CONVERSION_FINAL,
        "case6_admission_path": MODULE.DEFAULT_CASE6_ADMISSION,
        "case6_contract_path": MODULE.DEFAULT_CASE6_CONTRACT,
        "case6_conversion_result_path": MODULE.DEFAULT_CASE6_CONVERSION_RESULT,
        "case7_dataset_path": MODULE.DEFAULT_CASE7_DATASET,
        "case7_conversion_final_path": MODULE.DEFAULT_CASE7_CONVERSION_FINAL,
        "case7_admission_path": MODULE.DEFAULT_CASE7_ADMISSION,
        "case7_contract_path": MODULE.DEFAULT_CASE7_CONTRACT,
        "case7_conversion_result_path": MODULE.DEFAULT_CASE7_CONVERSION_RESULT,
        "case7_proposal_path": MODULE.DEFAULT_CASE7_PROPOSAL,
        "case8_dataset_path": MODULE.DEFAULT_CASE8_DATASET,
        "case8_conversion_final_path": MODULE.DEFAULT_CASE8_CONVERSION_FINAL,
        "case8_admission_path": MODULE.DEFAULT_CASE8_ADMISSION,
        "case8_contract_path": MODULE.DEFAULT_CASE8_CONTRACT,
        "case8_conversion_result_path": MODULE.DEFAULT_CASE8_CONVERSION_RESULT,
        "case8_proposal_path": MODULE.DEFAULT_CASE8_PROPOSAL,
    }
    inputs.update(overrides)
    return MODULE.audit_intake(**inputs)


def test_current_intake_reports_four_train_cases_and_one_validation_case() -> None:
    result = _audit()
    assert result["passed"] is True
    assert result["converted_train_cases"] == [6, 7, 23, 30]
    assert result["converted_validation_cases"] == [8]
    assert result["converted_train_case_count"] == 4
    assert result["converted_validation_case_count"] == 1
    assert result["missing_train_case_count"] == 0
    assert result["missing_validation_case_count"] == 1
    assert result["pending_minimum_train_cases"] == []
    assert result["pending_validation_cases"] == [16]
    assert result["next_bounded_action"] == (
        "authorize_exactly_one_case16_validation_paired_canary"
    )
    assert result["case23_conversion_authorized"] is False
    assert result["case23_conversion_output_created"] is True
    assert result["case6_conversion_authorized"] is False
    assert result["case6_conversion_output_created"] is True
    assert result["case7_conversion_authorized"] is False
    assert result["case7_conversion_output_created"] is True
    assert result["case8_conversion_authorized"] is False
    assert result["case8_conversion_output_created"] is True
    assert result["corpus_manifest_ready"] is False
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["dataset_conversion_authorized"] is False
    assert result["dataset_merge_authorized"] is False
    assert result["valid_for_bc_admission_review"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False
    assert result["valid_for_training"] is False


def test_real_train_and_case8_validation_datasets_are_reopened() -> None:
    result = _audit()
    rows = {row["case"]: row for row in result["cases"]}
    assert rows[30]["state"] == "converted_admitted_for_case_merge"
    assert rows[30]["sample_count"] == 11411
    assert rows[30]["valid_for_case_merge"] is True
    assert rows[23]["state"] == "converted_admitted_for_case_merge"
    assert rows[23]["sample_count"] == 3273
    assert rows[23]["valid_for_case_merge"] is True
    assert rows[6]["state"] == "converted_admitted_for_case_merge"
    assert rows[6]["sample_count"] == 7933
    assert rows[6]["valid_for_case_merge"] is True
    assert rows[7]["state"] == "converted_admitted_for_case_merge"
    assert rows[7]["sample_count"] == 6597
    assert rows[7]["valid_for_case_merge"] is True
    assert rows[8]["state"] == "converted_admitted_for_case_merge"
    assert rows[8]["split"] == "validation"
    assert rows[8]["sample_count"] == 6607
    assert rows[8]["valid_for_case_merge"] is True
    for case in (2, 16):
        assert rows[case]["state"] == "paired_canary_and_capture_required"
        assert rows[case]["valid_for_case_merge"] is False


def _synthetic_case23_conversion(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    metadata, payload = MODULE.load_case_dataset(
        MODULE.DEFAULT_CASE30_DATASET,
        expected_case=30,
        expected_split="train",
    )
    metadata = dict(metadata)
    metadata["case"] = 23
    payload = {name: np.asarray(value).copy() for name, value in payload.items()}
    payload["case_ids"][:] = 23
    dataset = tmp_path / "case_0023.npz"
    save_case_dataset(
        dataset,
        metadata,
        payload,
        expected_case=23,
        expected_split="train",
    )
    final = tmp_path / "final_status.json"
    final.write_text(
        json.dumps(
            {
                "schema": MODULE.CASE23_CONVERSION_FINAL_SCHEMA,
                "case": 23,
                "split": "train",
                "checks": {
                    "authorization_consumed": True,
                    "dataset_loaded": True,
                    "training_closed": True,
                },
                "dataset": {"sha256": MODULE._sha256(dataset)},
                "valid_for_case_merge": True,
                "merged_dataset_created": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
                "valid_for_training": False,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    recovery = tmp_path / "recovery_audit.json"
    recovery.write_text(
        json.dumps(
            {
                "schema": MODULE.CASE23_RECOVERY_SCHEMA,
                "case": 23,
                "split": "train",
                "execution": {
                    "converter_invocations": 1,
                    "converter_retry_performed": False,
                },
                "recovery": {
                    "canonical_sha256": MODULE._sha256(dataset),
                    "final_status_sha256": MODULE._sha256(final),
                },
                "result": {
                    "valid_for_case_merge": True,
                    "merged_dataset_created": False,
                    "bc_authorized": False,
                    "ppo_authorized": False,
                    "training_started": False,
                    "valid_for_training": False,
                },
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    return dataset, final, recovery


def test_valid_case23_dataset_advances_only_partial_intake(tmp_path: Path) -> None:
    dataset, final, recovery = _synthetic_case23_conversion(tmp_path)
    result = _audit(
        case23_dataset_path=dataset,
        case23_conversion_final_path=final,
        case23_recovery_audit_path=recovery,
    )
    assert result["converted_train_cases"] == [6, 7, 23, 30]
    assert result["missing_train_case_count"] == 0
    assert result["pending_minimum_train_cases"] == []
    assert result["next_bounded_action"] == (
        "authorize_exactly_one_case16_validation_paired_canary"
    )
    assert result["case23_conversion_output_created"] is True
    assert result["corpus_manifest_ready"] is False
    assert result["dataset_merge_authorized"] is False
    assert result["valid_for_training"] is False


def test_intake_rejects_case_identity_substitution() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        _audit(
            case23_dataset_path=MODULE.DEFAULT_CASE30_DATASET,
            case23_conversion_final_path=None,
            case23_recovery_audit_path=None,
        )


def test_intake_rejects_conversion_final_hash_mismatch(tmp_path: Path) -> None:
    dataset, final, recovery = _synthetic_case23_conversion(tmp_path)
    payload = json.loads(final.read_text())
    payload["dataset"]["sha256"] = "0" * 64
    final.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conversion final checks failed"):
        _audit(
            case23_dataset_path=dataset,
            case23_conversion_final_path=final,
            case23_recovery_audit_path=recovery,
        )


def test_intake_rejects_recovery_audit_hash_mismatch(tmp_path: Path) -> None:
    dataset, final, recovery = _synthetic_case23_conversion(tmp_path)
    payload = json.loads(recovery.read_text())
    payload["recovery"]["canonical_sha256"] = "0" * 64
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conversion final checks failed"):
        _audit(
            case23_dataset_path=dataset,
            case23_conversion_final_path=final,
            case23_recovery_audit_path=recovery,
        )


@pytest.mark.parametrize(
    ("argument", "field_path", "value"),
    [
        (
            "case6_conversion_final_path",
            ("dataset", "sha256"),
            "0" * 64,
        ),
        (
            "case6_admission_path",
            ("authorization_consumed_before_conversion",),
            False,
        ),
        (
            "case6_contract_path",
            ("conversion_authorized",),
            True,
        ),
        (
            "case6_conversion_result_path",
            ("effective_actions_used_as_training_targets",),
            False,
        ),
    ],
)
def test_intake_rejects_case6_conversion_contract_drift(
    tmp_path: Path,
    argument: str,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    source = {
        "case6_conversion_final_path": MODULE.DEFAULT_CASE6_CONVERSION_FINAL,
        "case6_admission_path": MODULE.DEFAULT_CASE6_ADMISSION,
        "case6_contract_path": MODULE.DEFAULT_CASE6_CONTRACT,
        "case6_conversion_result_path": MODULE.DEFAULT_CASE6_CONVERSION_RESULT,
    }[argument]
    payload = json.loads(source.read_text())
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    changed = tmp_path / source.name
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case6 conversion final checks failed"):
        _audit(**{argument: changed})


@pytest.mark.parametrize(
    ("argument", "field_path", "value"),
    [
        (
            "case7_conversion_final_path",
            ("dataset", "sha256"),
            "0" * 64,
        ),
        (
            "case7_admission_path",
            ("authorization_consumed_before_conversion",),
            False,
        ),
        (
            "case7_contract_path",
            ("conversion_authorized",),
            True,
        ),
        (
            "case7_conversion_result_path",
            ("effective_actions_used_as_training_targets",),
            False,
        ),
        (
            "case7_proposal_path",
            ("case",),
            6,
        ),
    ],
)
def test_intake_rejects_case7_conversion_contract_drift(
    tmp_path: Path,
    argument: str,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    source = {
        "case7_conversion_final_path": MODULE.DEFAULT_CASE7_CONVERSION_FINAL,
        "case7_admission_path": MODULE.DEFAULT_CASE7_ADMISSION,
        "case7_contract_path": MODULE.DEFAULT_CASE7_CONTRACT,
        "case7_conversion_result_path": MODULE.DEFAULT_CASE7_CONVERSION_RESULT,
        "case7_proposal_path": MODULE.DEFAULT_CASE7_PROPOSAL,
    }[argument]
    payload = json.loads(source.read_text())
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    changed = tmp_path / source.name
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case7 conversion final checks failed"):
        _audit(**{argument: changed})


@pytest.mark.parametrize(
    ("argument", "field_path", "value"),
    [
        (
            "case8_conversion_final_path",
            ("dataset", "sha256"),
            "0" * 64,
        ),
        (
            "case8_admission_path",
            ("authorization_consumed_before_conversion",),
            False,
        ),
        (
            "case8_contract_path",
            ("conversion_authorized",),
            True,
        ),
        (
            "case8_conversion_result_path",
            ("effective_actions_used_as_training_targets",),
            False,
        ),
        (
            "case8_proposal_path",
            ("split",),
            "train",
        ),
    ],
)
def test_intake_rejects_case8_validation_conversion_contract_drift(
    tmp_path: Path,
    argument: str,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    source = {
        "case8_conversion_final_path": MODULE.DEFAULT_CASE8_CONVERSION_FINAL,
        "case8_admission_path": MODULE.DEFAULT_CASE8_ADMISSION,
        "case8_contract_path": MODULE.DEFAULT_CASE8_CONTRACT,
        "case8_conversion_result_path": MODULE.DEFAULT_CASE8_CONVERSION_RESULT,
        "case8_proposal_path": MODULE.DEFAULT_CASE8_PROPOSAL,
    }[argument]
    payload = json.loads(source.read_text())
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    changed = tmp_path / source.name
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case8 conversion final checks failed"):
        _audit(**{argument: changed})


@pytest.mark.parametrize(
    ("source", "field", "value"),
    [
        ("train", "selected_cases", [30, 23, 6, 2]),
        ("validation", "selected_cases", [8, 22]),
        ("case30", "valid_for_case_merge", False),
        ("case23_final", "capture_admitted_for_dataset_conversion", False),
        ("case23_preflight", "conversion_authorized", True),
    ],
)
def test_intake_rejects_source_contract_drift(
    tmp_path: Path,
    source: str,
    field: str,
    value: object,
) -> None:
    paths = {
        "train": MODULE.DEFAULT_TRAIN_SELECTION,
        "validation": MODULE.DEFAULT_VALIDATION_SELECTION,
        "case30": MODULE.DEFAULT_CASE30_AUDIT,
        "case23_final": MODULE.DEFAULT_CASE23_FINAL,
        "case23_preflight": MODULE.DEFAULT_CASE23_PREFLIGHT,
    }
    payload = json.loads(paths[source].read_text())
    payload[field] = value
    changed = tmp_path / f"{source}.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    argument = {
        "train": "train_selection_path",
        "validation": "validation_selection_path",
        "case30": "case30_audit_path",
        "case23_final": "case23_final_path",
        "case23_preflight": "case23_preflight_path",
    }[source]
    with pytest.raises(ValueError, match="source checks failed"):
        _audit(**{argument: changed})


def test_cli_writes_closed_readiness_evidence(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    assert MODULE.main is not None
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    payload = json.loads(raw)
    assert payload["passed"] is True
    assert payload["corpus_manifest_ready"] is False
    assert payload["training_started"] is False
