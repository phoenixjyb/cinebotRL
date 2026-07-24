#!/usr/bin/env python3
"""Reopen and seal one generic authorized corrective case conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (  # noqa: E402
    load_corrective_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    load_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    PREVIOUS_ACTION_INDICES,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_generic_corrective_conversion_final_v2"
)
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_generic_corrective_conversion_execution_"
    "admission_v2"
)
PROPOSAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_conversion_proposal_v1"
)
CONVERSION_RESULT_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_conversion_result_v1"
)


def namespace_for(case: int) -> str:
    return (
        f"model_based_corrective_case{case:04d}_"
        "conversion_execution_v2_cpu"
    )


def dataset_name_for(case: int) -> str:
    return f"case_{case:04d}_model_based_corrective_case_dataset_v1.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def finalize(
    root: Path,
    admission_path: Path,
    proposal_path: Path,
    source_capture_path: Path,
    source_final_status_path: Path,
    conversion_result_path: Path,
    *,
    execution_commit: str,
    converter_exit_code: int,
) -> dict[str, Any]:
    root = root.resolve()
    admission = _load_object(admission_path)
    proposal = _load_object(proposal_path)
    case_value = admission.get("case")
    split_value = admission.get("split")
    case = case_value if type(case_value) is int else -1
    split = split_value if isinstance(split_value, str) else ""
    namespace = namespace_for(case) if case > 0 else ""
    dataset_name = dataset_name_for(case) if case > 0 else ""

    conversion_result: dict[str, Any] = {}
    conversion_result_error: str | None = None
    try:
        conversion_result = _load_object(conversion_result_path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        conversion_result_error = str(exc)

    dataset_path = root / dataset_name
    dataset_error: str | None = None
    metadata: dict[str, Any] = {}
    dataset: dict[str, np.ndarray] = {}
    source_metadata: dict[str, Any] = {}
    source: dict[str, np.ndarray] = {}
    try:
        metadata, dataset = load_case_dataset(
            dataset_path,
            expected_case=case,
            expected_split=split,
        )
        source_metadata, source = load_corrective_capture(
            source_capture_path,
            expected_case=case,
            expected_split=split,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        dataset_error = str(exc)

    proposal_identity = admission.get("proposal")
    source_metrics = admission.get("source_metrics")
    checks: dict[str, bool] = {
        "namespace": bool(namespace) and root.name == namespace,
        "execution_commit": (
            re.fullmatch(r"[0-9a-f]{40}", execution_commit) is not None
        ),
        "converter_exit_zero": converter_exit_code == 0,
        "admission_schema": admission.get("schema") == ADMISSION_SCHEMA,
        "admission_passed": admission.get("passed") is True,
        "admission_case_split": case > 0
        and proposal.get("case") == case
        and proposal.get("split") == split,
        "admission_execution_commit": admission.get("git", {}).get("head")
        == execution_commit,
        "authorization_consumed": admission.get(
            "authorization_consumed_before_conversion"
        )
        is True
        and admission.get("conversion_authorized") is True,
        "proposal": proposal.get("schema") == PROPOSAL_SCHEMA
        and proposal.get("passed") is True
        and isinstance(proposal_identity, dict)
        and proposal_identity.get("sha256") == _sha256(proposal_path),
        "source_paths": admission.get("source_capture_relative_path")
        == proposal.get("identities", {})
        .get("source_capture", {})
        .get("path")
        and admission.get("source_final_status_relative_path")
        == proposal.get("identities", {})
        .get("source_final_status", {})
        .get("path"),
        "conversion_result": conversion_result_error is None
        and conversion_result.get("schema") == CONVERSION_RESULT_SCHEMA
        and conversion_result.get("passed") is True
        and conversion_result.get("execute_requested") is True
        and conversion_result.get("output_created") is True
        and conversion_result.get("case") == case
        and conversion_result.get("split") == split,
        "dataset_loaded": dataset_error is None and bool(dataset),
    }
    metrics: dict[str, Any] = {}
    if checks["dataset_loaded"]:
        actions = np.asarray(dataset["actions"])
        source_actions = np.asarray(
            source["effective_corrective_normalized_actions"]
        )
        requested = np.asarray(dataset["requested_actions_audit"])
        source_requested = np.asarray(
            source["requested_corrective_normalized_actions"]
        )
        observations = np.asarray(dataset["observations"])
        source_observations = np.asarray(source["observations"])
        non_previous = np.ones(observations.shape[1], dtype=bool)
        non_previous[list(PREVIOUS_ACTION_INDICES)] = False
        clipped_rows = np.count_nonzero(
            np.asarray(dataset["command_clipped"], dtype=bool),
            axis=0,
        ).tolist()
        checks.update(
            {
                "source_capture_hash": metadata.get(
                    "source_capture_sha256"
                )
                == _sha256(source_capture_path)
                == admission.get("source_capture_sha256")
                == conversion_result.get("source_capture_sha256"),
                "source_final_status_hash": metadata.get(
                    "source_final_status_sha256"
                )
                == _sha256(source_final_status_path)
                == admission.get("source_final_status_sha256")
                == conversion_result.get("source_final_status_sha256"),
                "runtime_identity": metadata.get("source_runtime_commit")
                == source_metadata.get("runtime_commit"),
                "case_split": metadata.get("case") == case
                and metadata.get("split") == split,
                "sample_count": isinstance(source_metrics, dict)
                and metadata.get("sample_count") == len(actions)
                == source_metrics.get("sample_count"),
                "effective_actions_exact": np.array_equal(
                    actions, source_actions
                ),
                "requested_actions_audit_exact": np.array_equal(
                    requested, source_requested
                ),
                "previous_action_recurrence": np.allclose(
                    observations[0, PREVIOUS_ACTION_INDICES],
                    0.0,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.array_equal(
                    observations[1:, PREVIOUS_ACTION_INDICES],
                    actions[:-1],
                ),
                "non_previous_observations_exact": np.array_equal(
                    observations[:, non_previous],
                    source_observations[:, non_previous],
                ),
                "elapsed_clock_exact": np.array_equal(
                    dataset["elapsed_time_s"], source["elapsed_time_s"]
                ),
                "execution_clock_exact": np.array_equal(
                    dataset["execution_time_s"], source["execution_time_s"]
                ),
                "source_clock_exact": np.array_equal(
                    dataset["source_time_s"], source["source_time_s"]
                ),
                "case_ids_exact": np.array_equal(
                    dataset["case_ids"], source["case_ids"]
                ),
                "clipped_rows": isinstance(source_metrics, dict)
                and clipped_rows == source_metrics.get("clipped_rows"),
                "training_closed": metadata.get(
                    "merged_dataset_created"
                )
                is False
                and metadata.get("bc_authorized") is False
                and metadata.get("ppo_authorized") is False
                and metadata.get("training_started") is False
                and metadata.get("valid_for_training") is False,
            }
        )
        metrics = {
            "sample_count": int(len(actions)),
            "observation_shape": list(observations.shape),
            "action_shape": list(actions.shape),
            "action_abs_max": np.max(np.abs(actions), axis=0).tolist(),
            "requested_action_abs_max": np.max(
                np.abs(requested), axis=0
            ).tolist(),
            "clipped_rows": clipped_rows,
            "elapsed_clock_end_s": float(dataset["elapsed_time_s"][-1]),
            "execution_clock_end_s": float(
                dataset["execution_time_s"][-1]
            ),
            "source_clock_end_s": float(dataset["source_time_s"][-1]),
        }
        checks["metrics_exact"] = isinstance(source_metrics, dict) and (
            metrics == source_metrics
        )
    passed = bool(checks) and all(checks.values())
    return {
        "schema": SCHEMA,
        "namespace": namespace,
        "case": case,
        "split": split,
        "execution_commit": execution_commit,
        "converter_exit_code": converter_exit_code,
        "checks": checks,
        "conversion_result_error": conversion_result_error,
        "dataset_error": dataset_error,
        "metrics": metrics,
        "proposal": {
            "path": str(proposal_path),
            "sha256": (
                _sha256(proposal_path) if proposal_path.is_file() else None
            ),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": (
                _sha256(dataset_path) if dataset_path.is_file() else None
            ),
            "size_bytes": (
                dataset_path.stat().st_size if dataset_path.is_file() else 0
            ),
        },
        "valid_for_case_merge": passed,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--source-final-status", type=Path, required=True)
    parser.add_argument("--conversion-result", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--converter-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        args.admission,
        args.proposal,
        args.source_capture,
        args.source_final_status,
        args.conversion_result,
        execution_commit=args.execution_commit,
        converter_exit_code=args.converter_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
