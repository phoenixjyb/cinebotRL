#!/usr/bin/env python3
"""Audit real corrective case datasets before final corpus assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_corpus import (  # noqa: E402
    DEFAULT_RESERVED_HOLDOUT_CASES,
    MINIMUM_TRAIN_CASES,
    MINIMUM_VALIDATION_CASES,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    load_case_dataset,
)


SCHEMA = "cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v3"
TRAIN_SELECTION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1"
)
VALIDATION_SELECTION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_validation_selection_v1"
)
CASE30_AUDIT_SCHEMA = (
    "cinebotrl_two_wheel_riser_case30_effective_label_conversion_audit_v1"
)
CASE23_FINAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_final_v2"
)
CASE23_PREFLIGHT_SCHEMA = (
    "cinebotrl_two_wheel_riser_case23_conversion_execution_admission_v1"
)
CASE23_CONVERSION_FINAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_case23_conversion_final_v1"
)
CASE23_RECOVERY_SCHEMA = (
    "cinebotrl_two_wheel_riser_case23_conversion_path_recovery_audit_v1"
)
CASE6_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_case6_conversion_execution_admission_v1"
)
CASE6_CONTRACT_SCHEMA = (
    "cinebotrl_two_wheel_riser_case6_conversion_execution_contract_v1"
)
CASE6_RESULT_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_conversion_result_v1"
)
CASE6_CONVERSION_FINAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_case6_conversion_final_v1"
)
TRAIN_TRANCHE = [30, 23, 6, 2, 7]
MINIMUM_TRAIN_TRANCHE = TRAIN_TRANCHE[:MINIMUM_TRAIN_CASES]
VALIDATION_TRANCHE = [8, 16]

DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
DEFAULT_TRAIN_SELECTION = (
    DOC_ROOT
    / "evidence_20260723_model_based_pair_tranche_v1/selection.json"
)
DEFAULT_VALIDATION_SELECTION = (
    DOC_ROOT
    / "evidence_20260723_model_based_corrective_validation_tranche_v1/"
    "selection.json"
)
DEFAULT_CASE30_DATASET = (
    DOC_ROOT
    / "evidence_20260723_case30_effective_label_conversion_v1/"
    "case_0030_model_based_corrective_case_dataset_v1.npz"
)
DEFAULT_CASE30_AUDIT = (
    DOC_ROOT
    / "evidence_20260723_case30_effective_label_conversion_v1/"
    "independent_audit.json"
)
DEFAULT_CASE23_FINAL = (
    DOC_ROOT
    / "evidence_20260723_case23_corrective_capture_v4/final_status.json"
)
DEFAULT_CASE23_PREFLIGHT = (
    DOC_ROOT
    / "evidence_20260724_case23_corrective_conversion_execution_cpu_v2/"
    "summary.json"
)
DEFAULT_CASE23_CONVERSION_ROOT = (
    DOC_ROOT
    / "evidence_20260724_case23_corrective_conversion_execution_cpu_v5"
)
DEFAULT_CASE23_DATASET = (
    DEFAULT_CASE23_CONVERSION_ROOT
    / "case_0023_model_based_corrective_case_dataset_v1.npz"
)
DEFAULT_CASE23_CONVERSION_FINAL = (
    DEFAULT_CASE23_CONVERSION_ROOT / "final_status.json"
)
DEFAULT_CASE23_RECOVERY_AUDIT = (
    DEFAULT_CASE23_CONVERSION_ROOT / "recovery_audit.json"
)
DEFAULT_CASE6_CONVERSION_ROOT = (
    DOC_ROOT
    / "evidence_20260724_case6_corrective_conversion_execution_cpu_v1"
)
DEFAULT_CASE6_DATASET = (
    DEFAULT_CASE6_CONVERSION_ROOT
    / "case_0006_model_based_corrective_case_dataset_v1.npz"
)
DEFAULT_CASE6_CONVERSION_FINAL = (
    DEFAULT_CASE6_CONVERSION_ROOT / "final_status.json"
)
DEFAULT_CASE6_ADMISSION = (
    DEFAULT_CASE6_CONVERSION_ROOT / "admission.json"
)
DEFAULT_CASE6_CONTRACT = DEFAULT_CASE6_CONVERSION_ROOT / "contract.json"
DEFAULT_CASE6_CONVERSION_RESULT = (
    DEFAULT_CASE6_CONVERSION_ROOT / "conversion_result.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _closed(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("bc_authorized") is False
        and payload.get("ppo_authorized") is False
        and payload.get("training_started") is False
        and payload.get("valid_for_training") is False
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _dataset_row(path: Path, *, case: int, split: str) -> dict[str, object]:
    metadata, _ = load_case_dataset(
        path,
        expected_case=case,
        expected_split=split,
    )
    return {
        "case": case,
        "split": split,
        "state": "converted_admitted_for_case_merge",
        "path": _display_path(path),
        "sha256": _sha256(path),
        "sample_count": int(metadata["sample_count"]),
        "valid_for_case_merge": True,
        "valid_for_training": False,
    }


def audit_intake(
    *,
    train_selection_path: Path,
    validation_selection_path: Path,
    case30_dataset_path: Path,
    case30_audit_path: Path,
    case23_final_path: Path,
    case23_preflight_path: Path,
    case23_dataset_path: Path | None = None,
    case23_conversion_final_path: Path | None = None,
    case23_recovery_audit_path: Path | None = None,
    case6_dataset_path: Path = DEFAULT_CASE6_DATASET,
    case6_conversion_final_path: Path = DEFAULT_CASE6_CONVERSION_FINAL,
    case6_admission_path: Path = DEFAULT_CASE6_ADMISSION,
    case6_contract_path: Path = DEFAULT_CASE6_CONTRACT,
    case6_conversion_result_path: Path = DEFAULT_CASE6_CONVERSION_RESULT,
) -> dict[str, object]:
    train_selection = _load_object(train_selection_path)
    validation_selection = _load_object(validation_selection_path)
    case30_audit = _load_object(case30_audit_path)
    case23_final = _load_object(case23_final_path)
    case23_preflight = _load_object(case23_preflight_path)
    checks = {
        "train_selection": (
            train_selection.get("schema") == TRAIN_SELECTION_SCHEMA
            and train_selection.get("selected_cases") == TRAIN_TRANCHE
            and train_selection.get("passed") is True
            and train_selection.get("runtime_authorized") is False
            and train_selection.get("label_capture_authorized") is False
            and train_selection.get("dataset_merge_authorized") is False
            and _closed(train_selection)
        ),
        "validation_selection": (
            validation_selection.get("schema") == VALIDATION_SELECTION_SCHEMA
            and validation_selection.get("selected_cases") == VALIDATION_TRANCHE
            and validation_selection.get("passed") is True
            and validation_selection.get("runtime_authorized") is False
            and validation_selection.get("label_capture_authorized") is False
            and validation_selection.get("dataset_conversion_authorized") is False
            and validation_selection.get("dataset_merge_authorized") is False
            and _closed(validation_selection)
        ),
        "case30_conversion": (
            case30_audit.get("schema") == CASE30_AUDIT_SCHEMA
            and case30_audit.get("case") == 30
            and case30_audit.get("passed") is True
            and case30_audit.get("valid_for_case_merge") is True
            and case30_audit.get("merged_dataset_created") is False
            and _closed(case30_audit)
        ),
        "case23_capture": (
            case23_final.get("schema") == CASE23_FINAL_SCHEMA
            and case23_final.get("case") == 23
            and case23_final.get("split") == "train"
            and case23_final.get("passed") is True
            and case23_final.get("capture_admitted_for_dataset_conversion") is True
            and case23_final.get("normalized_training_dataset_created") is False
            and _closed(case23_final)
        ),
        "case23_conversion_preflight": (
            case23_preflight.get("schema") == CASE23_PREFLIGHT_SCHEMA
            and case23_preflight.get("case") == 23
            and case23_preflight.get("split") == "train"
            and case23_preflight.get("passed") is True
            and case23_preflight.get("cpu_contract_ready") is True
            and case23_preflight.get("conversion_authorized") is False
            and case23_preflight.get("output_created") is False
            and case23_preflight.get("merged_dataset_created") is False
            and _closed(case23_preflight)
        ),
        "tranches_disjoint": (
            not set(TRAIN_TRANCHE) & set(VALIDATION_TRANCHE)
            and not set(TRAIN_TRANCHE) & set(DEFAULT_RESERVED_HOLDOUT_CASES)
            and not set(VALIDATION_TRANCHE) & set(DEFAULT_RESERVED_HOLDOUT_CASES)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"corrective corpus intake source checks failed: {checks}")

    case23_conversion_inputs = (
        case23_dataset_path,
        case23_conversion_final_path,
        case23_recovery_audit_path,
    )
    if any(path is None for path in case23_conversion_inputs) != all(
        path is None for path in case23_conversion_inputs
    ):
        raise ValueError(
            "case23 dataset, conversion final status, and recovery audit "
            "must be supplied together"
        )
    rows = [_dataset_row(case30_dataset_path, case=30, split="train")]
    if (
        case23_dataset_path is not None
        and case23_conversion_final_path is not None
        and case23_recovery_audit_path is not None
    ):
        conversion_final = _load_object(case23_conversion_final_path)
        recovery_audit = _load_object(case23_recovery_audit_path)
        dataset_sha = _sha256(case23_dataset_path)
        final_sha = _sha256(case23_conversion_final_path)
        conversion_checks = conversion_final.get("checks")
        conversion_dataset = conversion_final.get("dataset")
        recovery_execution = recovery_audit.get("execution")
        recovery_result = recovery_audit.get("result")
        recovery = recovery_audit.get("recovery")
        final_checks = {
            "schema": conversion_final.get("schema")
            == CASE23_CONVERSION_FINAL_SCHEMA,
            "case_split": conversion_final.get("case") == 23
            and conversion_final.get("split") == "train",
            "passed": conversion_final.get("passed") is True
            and conversion_final.get("valid_for_case_merge") is True,
            "dataset_hash": isinstance(conversion_dataset, Mapping)
            and conversion_dataset.get("sha256") == dataset_sha,
            "checks": isinstance(conversion_checks, Mapping)
            and bool(conversion_checks)
            and all(value is True for value in conversion_checks.values()),
            "training_closed": conversion_final.get("merged_dataset_created")
            is False
            and _closed(conversion_final),
            "recovery_audit": (
                recovery_audit.get("schema") == CASE23_RECOVERY_SCHEMA
                and recovery_audit.get("case") == 23
                and recovery_audit.get("split") == "train"
                and recovery_audit.get("passed") is True
                and isinstance(recovery_execution, Mapping)
                and recovery_execution.get("converter_invocations") == 1
                and recovery_execution.get("converter_retry_performed") is False
                and isinstance(recovery, Mapping)
                and recovery.get("canonical_sha256") == dataset_sha
                and recovery.get("final_status_sha256") == final_sha
                and isinstance(recovery_result, Mapping)
                and recovery_result.get("valid_for_case_merge") is True
                and recovery_result.get("merged_dataset_created") is False
                and _closed(recovery_result)
            ),
        }
        if not all(final_checks.values()):
            raise ValueError(
                f"case23 conversion final checks failed: {final_checks}"
            )
        rows.append(_dataset_row(case23_dataset_path, case=23, split="train"))
    else:
        rows.append(
            {
                "case": 23,
                "split": "train",
                "state": "capture_passed_conversion_authorization_required",
                "source_capture_sha256": case23_preflight[
                    "source_capture_sha256"
                ],
                "source_final_status_sha256": case23_preflight[
                    "source_final_status_sha256"
                ],
                "expected_output_relative_path": case23_preflight[
                    "output_relative_path"
                ],
                "valid_for_case_merge": False,
                "valid_for_training": False,
            }
        )
    case6_conversion_final = _load_object(case6_conversion_final_path)
    case6_admission = _load_object(case6_admission_path)
    case6_contract = _load_object(case6_contract_path)
    case6_conversion_result = _load_object(case6_conversion_result_path)
    case6_dataset_sha = _sha256(case6_dataset_path)
    case6_final_checks = case6_conversion_final.get("checks")
    case6_dataset_identity = case6_conversion_final.get("dataset")
    case6_git = case6_admission.get("git")
    case6_source_capture_sha = case6_admission.get("source_capture_sha256")
    case6_source_final_sha = case6_admission.get("source_final_status_sha256")
    case6_checks = {
        "schema": case6_conversion_final.get("schema")
        == CASE6_CONVERSION_FINAL_SCHEMA,
        "case_split": case6_conversion_final.get("case") == 6
        and case6_conversion_final.get("split") == "train",
        "passed": case6_conversion_final.get("passed") is True
        and case6_conversion_final.get("valid_for_case_merge") is True,
        "dataset_hash": isinstance(case6_dataset_identity, Mapping)
        and case6_dataset_identity.get("sha256") == case6_dataset_sha,
        "checks": isinstance(case6_final_checks, Mapping)
        and bool(case6_final_checks)
        and all(value is True for value in case6_final_checks.values()),
        "admission": (
            case6_admission.get("schema") == CASE6_ADMISSION_SCHEMA
            and case6_admission.get("case") == 6
            and case6_admission.get("split") == "train"
            and case6_admission.get("passed") is True
            and case6_admission.get("conversion_authorized") is True
            and case6_admission.get("authorization_consumed_before_conversion")
            is True
            and isinstance(case6_git, Mapping)
            and case6_git.get("head") == case6_git.get("upstream")
            == case6_conversion_final.get("runtime_commit")
            and case6_admission.get("output_created") is False
            and case6_admission.get("merged_dataset_created") is False
            and _closed(case6_admission)
        ),
        "contract_closed": (
            case6_contract.get("schema") == CASE6_CONTRACT_SCHEMA
            and case6_contract.get("case") == 6
            and case6_contract.get("split") == "train"
            and case6_contract.get("conversion_authorized") is False
            and case6_contract.get("authorization_token_issued") is False
            and case6_contract.get("authorization_token_sha256") == ""
            and case6_contract.get("output_created") is False
            and case6_contract.get("merged_dataset_created") is False
            and _closed(case6_contract)
        ),
        "conversion_result": (
            case6_conversion_result.get("schema") == CASE6_RESULT_SCHEMA
            and case6_conversion_result.get("case") == 6
            and case6_conversion_result.get("split") == "train"
            and case6_conversion_result.get("passed") is True
            and case6_conversion_result.get("execute_requested") is True
            and case6_conversion_result.get("output_created") is True
            and case6_conversion_result.get("sample_count") == 7933
            and case6_conversion_result.get("source_capture_sha256")
            == case6_source_capture_sha
            and case6_conversion_result.get("source_final_status_sha256")
            == case6_source_final_sha
            and case6_conversion_result.get(
                "requested_actions_used_as_training_targets"
            )
            is False
            and case6_conversion_result.get(
                "effective_actions_used_as_training_targets"
            )
            is True
            and case6_conversion_result.get("valid_for_case_merge") is True
            and case6_conversion_result.get("merged_dataset_created") is False
            and _closed(case6_conversion_result)
        ),
        "training_closed": case6_conversion_final.get(
            "merged_dataset_created"
        )
        is False
        and _closed(case6_conversion_final),
    }
    if not all(case6_checks.values()):
        raise ValueError(
            f"case6 conversion final checks failed: {case6_checks}"
        )
    rows.append(_dataset_row(case6_dataset_path, case=6, split="train"))

    converted_train = sorted(
        int(row["case"])
        for row in rows
        if row["split"] == "train" and row["valid_for_case_merge"] is True
    )
    converted_validation: list[int] = []
    pending_train = [
        case for case in MINIMUM_TRAIN_TRANCHE if case not in converted_train
    ]
    pending_validation = [
        case for case in VALIDATION_TRANCHE if case not in converted_validation
    ]
    state_by_case = {int(row["case"]): row for row in rows}
    for case in TRAIN_TRANCHE:
        state_by_case.setdefault(
            case,
            {
                "case": case,
                "split": "train",
                "state": "paired_canary_and_capture_required",
                "valid_for_case_merge": False,
                "valid_for_training": False,
            },
        )
    for case in VALIDATION_TRANCHE:
        state_by_case[case] = {
            "case": case,
            "split": "validation",
            "state": "paired_canary_and_capture_required",
            "valid_for_case_merge": False,
            "valid_for_training": False,
        }
    corpus_ready = (
        len(converted_train) >= MINIMUM_TRAIN_CASES
        and len(converted_validation) >= MINIMUM_VALIDATION_CASES
    )
    if 23 not in converted_train:
        next_action = "authorize_exactly_one_case23_v4_cpu_conversion"
    elif 6 not in converted_train:
        next_action = "authorize_exactly_one_case6_cpu_conversion"
    else:
        next_action = "authorize_exactly_one_case2_paired_canary"
    return {
        "schema": SCHEMA,
        "checks": checks,
        "inputs": {
            "train_selection": _identity(train_selection_path),
            "validation_selection": _identity(validation_selection_path),
            "case30_dataset": _identity(case30_dataset_path),
            "case30_audit": _identity(case30_audit_path),
            "case23_final_status": _identity(case23_final_path),
            "case23_conversion_preflight": _identity(case23_preflight_path),
            **(
                {
                    "case23_dataset": _identity(case23_dataset_path),
                    "case23_conversion_final": _identity(
                        case23_conversion_final_path
                    ),
                    "case23_recovery_audit": _identity(
                        case23_recovery_audit_path
                    ),
                }
                if case23_dataset_path is not None
                and case23_conversion_final_path is not None
                and case23_recovery_audit_path is not None
                else {}
            ),
            "case6_dataset": _identity(case6_dataset_path),
            "case6_conversion_final": _identity(
                case6_conversion_final_path
            ),
            "case6_admission": _identity(case6_admission_path),
            "case6_contract": _identity(case6_contract_path),
            "case6_conversion_result": _identity(
                case6_conversion_result_path
            ),
        },
        "required": {
            "minimum_train_cases": MINIMUM_TRAIN_CASES,
            "minimum_validation_cases": MINIMUM_VALIDATION_CASES,
            "minimum_train_tranche": MINIMUM_TRAIN_TRANCHE,
            "additional_train_candidate": [7],
            "validation_tranche": VALIDATION_TRANCHE,
            "reserved_holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        },
        "cases": [state_by_case[case] for case in TRAIN_TRANCHE + VALIDATION_TRANCHE],
        "converted_train_cases": converted_train,
        "converted_validation_cases": converted_validation,
        "converted_train_case_count": len(converted_train),
        "converted_validation_case_count": len(converted_validation),
        "missing_train_case_count": max(
            0, MINIMUM_TRAIN_CASES - len(converted_train)
        ),
        "missing_validation_case_count": max(
            0, MINIMUM_VALIDATION_CASES - len(converted_validation)
        ),
        "pending_minimum_train_cases": pending_train,
        "pending_validation_cases": pending_validation,
        "next_bounded_action": next_action,
        "case23_conversion_authorized": False,
        "case23_conversion_output_created": case23_dataset_path is not None,
        "case6_conversion_authorized": False,
        "case6_conversion_output_created": True,
        "corpus_manifest_ready": corpus_ready,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "valid_for_bc_admission_review": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-selection", type=Path, default=DEFAULT_TRAIN_SELECTION
    )
    parser.add_argument(
        "--validation-selection",
        type=Path,
        default=DEFAULT_VALIDATION_SELECTION,
    )
    parser.add_argument("--case30-dataset", type=Path, default=DEFAULT_CASE30_DATASET)
    parser.add_argument("--case30-audit", type=Path, default=DEFAULT_CASE30_AUDIT)
    parser.add_argument("--case23-final", type=Path, default=DEFAULT_CASE23_FINAL)
    parser.add_argument(
        "--case23-preflight", type=Path, default=DEFAULT_CASE23_PREFLIGHT
    )
    parser.add_argument(
        "--case23-dataset", type=Path, default=DEFAULT_CASE23_DATASET
    )
    parser.add_argument(
        "--case23-conversion-final",
        type=Path,
        default=DEFAULT_CASE23_CONVERSION_FINAL,
    )
    parser.add_argument(
        "--case23-recovery-audit",
        type=Path,
        default=DEFAULT_CASE23_RECOVERY_AUDIT,
    )
    parser.add_argument(
        "--case6-dataset", type=Path, default=DEFAULT_CASE6_DATASET
    )
    parser.add_argument(
        "--case6-conversion-final",
        type=Path,
        default=DEFAULT_CASE6_CONVERSION_FINAL,
    )
    parser.add_argument(
        "--case6-admission", type=Path, default=DEFAULT_CASE6_ADMISSION
    )
    parser.add_argument(
        "--case6-contract", type=Path, default=DEFAULT_CASE6_CONTRACT
    )
    parser.add_argument(
        "--case6-conversion-result",
        type=Path,
        default=DEFAULT_CASE6_CONVERSION_RESULT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_intake(
        train_selection_path=args.train_selection,
        validation_selection_path=args.validation_selection,
        case30_dataset_path=args.case30_dataset,
        case30_audit_path=args.case30_audit,
        case23_final_path=args.case23_final,
        case23_preflight_path=args.case23_preflight,
        case23_dataset_path=args.case23_dataset,
        case23_conversion_final_path=args.case23_conversion_final,
        case23_recovery_audit_path=args.case23_recovery_audit,
        case6_dataset_path=args.case6_dataset,
        case6_conversion_final_path=args.case6_conversion_final,
        case6_admission_path=args.case6_admission,
        case6_contract_path=args.case6_contract,
        case6_conversion_result_path=args.case6_conversion_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
