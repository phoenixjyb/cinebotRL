import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (
    convert_admitted_capture,
    save_case_dataset,
)


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "finalize_model_based_corrective_conversion.py"
)
PROPOSAL_ROOT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_generic_corrective_conversion_proposals_v1"
)
CASE8_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case8_validation_conversion_proposal_v1/proposal.json"
)
CAPTURES = {
    8: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case8_validation_capture_v1"
    ),
    6: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260724_case6_corrective_capture_v1"
    ),
    23: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4"
    ),
}
SPEC = importlib.util.spec_from_file_location(
    "generic_conversion_finalizer",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, case: int, split: str = "train"):
    evidence = CAPTURES[case]
    capture = (
        evidence
        / f"capture/case_{case:04d}_corrective_teacher_capture_v2.npz"
    )
    source_final = evidence / "final_status.json"
    proposal = (
        CASE8_PROPOSAL
        if case == 8
        else PROPOSAL_ROOT / f"case_{case:04d}.json"
    )
    proposal_payload = json.loads(proposal.read_text())
    root = tmp_path / MODULE.namespace_for(case)
    root.mkdir(parents=True)
    metadata, payload = convert_admitted_capture(
        capture,
        source_final,
        expected_case=case,
        expected_split=split,
    )
    dataset = root / MODULE.dataset_name_for(case)
    save_case_dataset(
        dataset,
        metadata,
        payload,
        expected_case=case,
        expected_split=split,
    )
    execution_commit = "a" * 40
    admission = {
        "schema": MODULE.ADMISSION_SCHEMA,
        "case": case,
        "split": split,
        "passed": True,
        "git": {
            "head": execution_commit,
            "upstream": execution_commit,
        },
        "authorization_consumed_before_conversion": True,
        "conversion_authorized": True,
        "proposal": {
            "path": f"proposal_case_{case}.json",
            "sha256": _sha256(proposal),
        },
        "source_metrics": proposal_payload["metrics"],
        "source_capture_relative_path": proposal_payload["identities"][
            "source_capture"
        ]["path"],
        "source_final_status_relative_path": proposal_payload["identities"][
            "source_final_status"
        ]["path"],
        "source_capture_sha256": _sha256(capture),
        "source_final_status_sha256": _sha256(source_final),
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission))
    result = {
        "schema": MODULE.CONVERSION_RESULT_SCHEMA,
        "passed": True,
        "execute_requested": True,
        "output_created": True,
        "case": case,
        "split": split,
        "source_capture_sha256": _sha256(capture),
        "source_final_status_sha256": _sha256(source_final),
    }
    result_path = root / "conversion_result.json"
    result_path.write_text(json.dumps(result))
    return {
        "root": root,
        "admission": admission_path,
        "proposal": proposal,
        "capture": capture,
        "source_final": source_final,
        "conversion_result": result_path,
        "execution_commit": execution_commit,
    }


@pytest.mark.parametrize(
    ("case", "split", "sample_count", "clipped_rows"),
    [
        (6, "train", 7933, [0, 146, 0]),
        (23, "train", 3273, [0, 0, 0]),
        (8, "validation", 6607, [0, 0, 9]),
    ],
)
def test_generic_finalizer_reopens_and_seals_dataset(
    tmp_path,
    case,
    split,
    sample_count,
    clipped_rows,
) -> None:
    fixture = _fixture(tmp_path, case, split)
    result = MODULE.finalize(
        fixture["root"],
        fixture["admission"],
        fixture["proposal"],
        fixture["capture"],
        fixture["source_final"],
        fixture["conversion_result"],
        execution_commit=fixture["execution_commit"],
        converter_exit_code=0,
    )
    assert result["passed"] is True
    assert result["case"] == case
    assert result["split"] == split
    assert result["metrics"]["sample_count"] == sample_count
    assert result["metrics"]["clipped_rows"] == clipped_rows
    assert all(result["checks"].values())
    assert result["dataset"]["sha256"]
    assert result["valid_for_case_merge"] is True
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False
    assert result["valid_for_training"] is False


def test_generic_finalizer_rejects_failed_or_unconsumed_execution(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, 6)
    failed = MODULE.finalize(
        fixture["root"],
        fixture["admission"],
        fixture["proposal"],
        fixture["capture"],
        fixture["source_final"],
        fixture["conversion_result"],
        execution_commit=fixture["execution_commit"],
        converter_exit_code=1,
    )
    assert failed["passed"] is False
    assert failed["checks"]["converter_exit_zero"] is False

    admission = json.loads(fixture["admission"].read_text())
    admission["authorization_consumed_before_conversion"] = False
    admission["conversion_authorized"] = False
    fixture["admission"].write_text(json.dumps(admission))
    unconsumed = MODULE.finalize(
        fixture["root"],
        fixture["admission"],
        fixture["proposal"],
        fixture["capture"],
        fixture["source_final"],
        fixture["conversion_result"],
        execution_commit=fixture["execution_commit"],
        converter_exit_code=0,
    )
    assert unconsumed["passed"] is False
    assert unconsumed["checks"]["authorization_consumed"] is False


def test_generic_finalizer_rejects_proposal_or_commit_mismatch(
    tmp_path,
) -> None:
    fixture = _fixture(tmp_path, 23)
    admission = json.loads(fixture["admission"].read_text())
    admission["proposal"]["sha256"] = "0" * 64
    fixture["admission"].write_text(json.dumps(admission))
    bad_proposal = MODULE.finalize(
        fixture["root"],
        fixture["admission"],
        fixture["proposal"],
        fixture["capture"],
        fixture["source_final"],
        fixture["conversion_result"],
        execution_commit=fixture["execution_commit"],
        converter_exit_code=0,
    )
    assert bad_proposal["passed"] is False
    assert bad_proposal["checks"]["proposal"] is False

    fixture = _fixture(tmp_path / "wrong_commit", 23)
    wrong_commit = MODULE.finalize(
        fixture["root"],
        fixture["admission"],
        fixture["proposal"],
        fixture["capture"],
        fixture["source_final"],
        fixture["conversion_result"],
        execution_commit="b" * 40,
        converter_exit_code=0,
    )
    assert wrong_commit["passed"] is False
    assert wrong_commit["checks"]["admission_execution_commit"] is False
