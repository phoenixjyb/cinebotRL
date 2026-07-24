import hashlib
import importlib.util
import json
from pathlib import Path

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (
    convert_admitted_capture,
    save_case_dataset,
)


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "finalize_model_based_corrective_case6_conversion.py"
)
SOURCE_CAPTURE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_corrective_capture_v1/"
    "capture/case_0006_corrective_teacher_capture_v2.npz"
)
SOURCE_FINAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_corrective_capture_v1/final_status.json"
)
SPEC = importlib.util.spec_from_file_location("case6_conversion_finalizer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    root = tmp_path / MODULE.NAMESPACE
    root.mkdir()
    metadata, payload = convert_admitted_capture(
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        expected_case=6,
        expected_split="train",
    )
    dataset = root / MODULE.DATASET_NAME
    save_case_dataset(
        dataset,
        metadata,
        payload,
        expected_case=6,
        expected_split="train",
    )
    runtime_commit = "a" * 40
    admission = {
        "schema": MODULE.ADMISSION_SCHEMA,
        "case": 6,
        "split": "train",
        "passed": True,
        "git": {"head": runtime_commit, "upstream": runtime_commit},
        "authorization_consumed_before_conversion": True,
        "conversion_authorized": True,
        "source_capture_sha256": _sha256(SOURCE_CAPTURE),
        "source_final_status_sha256": _sha256(SOURCE_FINAL),
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission))
    result = {
        "schema": MODULE.CONVERSION_RESULT_SCHEMA,
        "passed": True,
        "execute_requested": True,
        "output_created": True,
        "case": 6,
        "split": "train",
        "source_capture_sha256": _sha256(SOURCE_CAPTURE),
        "source_final_status_sha256": _sha256(SOURCE_FINAL),
    }
    result_path = root / "conversion_result.json"
    result_path.write_text(json.dumps(result))
    return root, admission_path, result_path, runtime_commit


def test_finalizer_reopens_and_seals_case6_dataset(tmp_path) -> None:
    root, admission, conversion_result, commit = _fixture(tmp_path)
    result = MODULE.finalize(
        root,
        admission,
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        conversion_result,
        runtime_commit=commit,
        converter_exit_code=0,
    )
    assert result["passed"] is True
    assert result["case"] == 6
    assert result["split"] == "train"
    assert result["metrics"]["sample_count"] == 7933
    assert result["metrics"]["clipped_rows"] == [0, 146, 0]
    assert all(result["checks"].values())
    assert result["dataset"]["sha256"]
    assert result["valid_for_case_merge"] is True
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False
    assert result["valid_for_training"] is False


def test_finalizer_rejects_failed_converter_or_wrong_commit(tmp_path) -> None:
    root, admission, conversion_result, commit = _fixture(tmp_path)
    failed = MODULE.finalize(
        root,
        admission,
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        conversion_result,
        runtime_commit=commit,
        converter_exit_code=1,
    )
    assert failed["passed"] is False
    assert failed["checks"]["converter_exit_zero"] is False
    wrong_commit = MODULE.finalize(
        root,
        admission,
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        conversion_result,
        runtime_commit="b" * 40,
        converter_exit_code=0,
    )
    assert wrong_commit["passed"] is False
    assert wrong_commit["checks"]["admission_runtime_commit"] is False


def test_finalizer_rejects_unconsumed_authorization(tmp_path) -> None:
    root, admission, conversion_result, commit = _fixture(tmp_path)
    payload = json.loads(admission.read_text())
    payload["authorization_consumed_before_conversion"] = False
    payload["conversion_authorized"] = False
    admission.write_text(json.dumps(payload))
    result = MODULE.finalize(
        root,
        admission,
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        conversion_result,
        runtime_commit=commit,
        converter_exit_code=0,
    )
    assert result["passed"] is False
    assert result["checks"]["authorization_consumed"] is False


def test_finalizer_rejects_malformed_converter_result(tmp_path) -> None:
    root, admission, conversion_result, commit = _fixture(tmp_path)
    conversion_result.write_text("not-json")
    result = MODULE.finalize(
        root,
        admission,
        SOURCE_CAPTURE,
        SOURCE_FINAL,
        conversion_result,
        runtime_commit=commit,
        converter_exit_code=0,
    )
    assert result["passed"] is False
    assert result["checks"]["conversion_result"] is False
    assert result["conversion_result_error"]
