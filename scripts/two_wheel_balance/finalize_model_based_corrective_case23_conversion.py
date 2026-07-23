#!/usr/bin/env python3
"""Reopen and seal one authorized case-23 corrective case conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

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


SCHEMA = "cinebotrl_two_wheel_riser_case23_conversion_final_v1"
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_case23_conversion_execution_admission_v1"
)
CONVERSION_RESULT_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_conversion_result_v1"
)
CASE = 23
SPLIT = "train"
NAMESPACE = "20260723_model_based_corrective_case23_conversion_v1_cpu"
DATASET_NAME = "case_0023_model_based_corrective_case_dataset_v1.npz"


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


def finalize(
    root: Path,
    admission_path: Path,
    source_capture_path: Path,
    conversion_result_path: Path,
    *,
    runtime_commit: str,
    converter_exit_code: int,
) -> dict[str, object]:
    root = root.resolve()
    admission = _load_object(admission_path)
    conversion_result = (
        _load_object(conversion_result_path)
        if conversion_result_path.is_file()
        else {}
    )
    dataset_path = root / DATASET_NAME
    dataset_error: str | None = None
    metadata: dict[str, object] = {}
    dataset: dict[str, np.ndarray] = {}
    source_metadata: dict[str, object] = {}
    source: dict[str, np.ndarray] = {}
    try:
        metadata, dataset = load_case_dataset(
            dataset_path,
            expected_case=CASE,
            expected_split=SPLIT,
        )
        source_metadata, source = load_corrective_capture(
            source_capture_path,
            expected_case=CASE,
            expected_split=SPLIT,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        dataset_error = str(exc)

    checks: dict[str, bool] = {
        "namespace": root.name == NAMESPACE,
        "converter_exit_zero": converter_exit_code == 0,
        "admission_schema": admission.get("schema") == ADMISSION_SCHEMA,
        "admission_passed": admission.get("passed") is True,
        "admission_case_split": admission.get("case") == CASE
        and admission.get("split") == SPLIT,
        "admission_runtime_commit": admission.get("git", {}).get("head")
        == runtime_commit,
        "authorization_consumed": admission.get(
            "authorization_consumed_before_conversion"
        )
        is True
        and admission.get("conversion_authorized") is True,
        "conversion_result": conversion_result.get("schema")
        == CONVERSION_RESULT_SCHEMA
        and conversion_result.get("passed") is True
        and conversion_result.get("execute_requested") is True
        and conversion_result.get("output_created") is True
        and conversion_result.get("case") == CASE
        and conversion_result.get("split") == SPLIT,
        "dataset_loaded": dataset_error is None and bool(dataset),
    }
    metrics: dict[str, object] = {}
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
        checks.update(
            {
                "source_hash": metadata.get("source_capture_sha256")
                == _sha256(source_capture_path)
                == admission.get("source_capture_sha256"),
                "runtime_identity": metadata.get("source_runtime_commit")
                == source_metadata.get("runtime_commit"),
                "case_split": metadata.get("case") == CASE
                and metadata.get("split") == SPLIT,
                "sample_count": metadata.get("sample_count") == len(actions),
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
                "training_closed": metadata.get("merged_dataset_created")
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
            "action_abs_max": np.max(np.abs(actions), axis=0).tolist(),
            "requested_action_abs_max": np.max(
                np.abs(requested), axis=0
            ).tolist(),
            "clipped_rows": np.count_nonzero(
                np.asarray(dataset["command_clipped"], dtype=bool),
                axis=0,
            ).tolist(),
            "elapsed_clock_end_s": float(dataset["elapsed_time_s"][-1]),
            "execution_clock_end_s": float(dataset["execution_time_s"][-1]),
            "source_clock_end_s": float(dataset["source_time_s"][-1]),
        }
    passed = bool(checks) and all(checks.values())
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "case": CASE,
        "split": SPLIT,
        "runtime_commit": runtime_commit,
        "converter_exit_code": converter_exit_code,
        "checks": checks,
        "dataset_error": dataset_error,
        "metrics": metrics,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path) if dataset_path.is_file() else None,
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
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--conversion-result", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--converter-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        args.admission,
        args.source_capture,
        args.conversion_result,
        runtime_commit=args.runtime_commit,
        converter_exit_code=args.converter_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
