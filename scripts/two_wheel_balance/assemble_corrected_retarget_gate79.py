#!/usr/bin/env python3
"""Assemble an immutable 79-case schema-v3 offline-gate payload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


CANDIDATE_SCHEMA = "cinebotrl_two_wheel_corrected_semantic_retarget_v3"
QUARANTINED_SOURCE_PACKAGE_SHA256 = (
    "af035fb50f17322add90bf008427c9247dbbf08ee0bc38dd6d24172d9e3e14e4"
)
FORBIDDEN_KEYS = {
    "physical_gimbal_q",
    "physical_gimbal_joint_labels",
    "physical_gimbal_diagnostic",
    "target_cam_link_quat_wxyz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gate-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovered-cases", default="73,78")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    return value.item()


def validate_candidate(path: Path, case: int) -> None:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {
            "schema",
            "trajectory_integrity_contract",
            "source_trajectory_integrity_passed",
            "source_teacher_quality_passed",
            "valid_for_candidate_training",
            "case",
            "runtime_approved",
            "training_started",
            "position_target_link",
            "attitude_target_contract",
            "physical_gimbal_joint_labels_included",
            "time_s",
            "semantic_start_index",
            "target_position_world_m",
            "target_attitude_world_dfr_quat_wxyz",
            "base_arm_q",
            "control_v_wz_darm",
        }
        missing = required - keys
        forbidden = FORBIDDEN_KEYS & keys
        if missing or forbidden:
            raise ValueError(
                f"invalid keys in {path}: missing={sorted(missing)}, "
                f"forbidden={sorted(forbidden)}"
            )
        if scalar(data, "schema") != CANDIDATE_SCHEMA:
            raise ValueError(f"wrong schema in {path}")
        if scalar(data, "trajectory_integrity_contract") != "exact_source_v1":
            raise ValueError(f"candidate lacks exact_source_v1 in {path}")
        if not bool(scalar(data, "source_trajectory_integrity_passed")):
            raise ValueError(f"source trajectory integrity failed in {path}")
        if not bool(scalar(data, "source_teacher_quality_passed")):
            raise ValueError(f"source teacher quality failed in {path}")
        if not bool(scalar(data, "valid_for_candidate_training")):
            raise ValueError(f"candidate is not training-valid in {path}")
        if int(scalar(data, "case")) != case:
            raise ValueError(f"wrong case in {path}")
        if bool(scalar(data, "runtime_approved")):
            raise ValueError(f"runtime approval leaked into {path}")
        if bool(scalar(data, "training_started")):
            raise ValueError(f"training marker leaked into {path}")
        if bool(scalar(data, "physical_gimbal_joint_labels_included")):
            raise ValueError(f"physical gimbal labels leaked into {path}")
        if scalar(data, "position_target_link") != "ee1_tool":
            raise ValueError(f"wrong position target in {path}")
        if (
            scalar(data, "attitude_target_contract")
            != "world_semantic_DFR_quaternion_wxyz_option_B"
        ):
            raise ValueError(f"wrong attitude contract in {path}")
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        samples = len(time_s)
        semantic_start = int(scalar(data, "semantic_start_index"))
        if (
            time_s.ndim != 1
            or samples < 3
            or time_s[0] != 0.0
            or np.any(np.diff(time_s) <= 0.0)
            or not 1 <= semantic_start < samples - 1
        ):
            raise ValueError(f"invalid timing in {path}")
        expected_shapes = {
            "target_position_world_m": (samples, 3),
            "target_attitude_world_dfr_quat_wxyz": (samples, 4),
            "base_arm_q": (samples, 6),
            "control_v_wz_darm": (samples - 1, 5),
        }
        for key, shape in expected_shapes.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid {key} in {path}: {data[key].shape}")


def load_passed_result(path: Path, case: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if int(result.get("case", -1)) != case or result.get("passed") is not True:
        raise ValueError(f"case {case} result is not an offline pass: {path}")
    checks = result.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError(f"case {case} has incomplete checks: {path}")
    return result


def maximum(results: list[dict[str, object]], key: str) -> float:
    values = [float(result[key]) for result in results if key in result]
    if len(values) != len(results):
        raise ValueError(f"metric {key} is missing from one or more results")
    return max(values)


def main() -> int:
    args = parse_args()
    recovered = {
        int(value) for value in args.recovered_cases.split(",") if value.strip()
    }
    expected_cases = set(range(1, 80))
    if not recovered or not recovered < expected_cases:
        raise ValueError(f"invalid recovered cases: {sorted(recovered)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = json.loads(
        (args.base_gate_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        base_summary.get("candidate_schema") != CANDIDATE_SCHEMA
        or base_summary.get("trajectory_integrity_contract") != "exact_source_v1"
        or base_summary.get("source_trajectory_integrity_passed") is not True
        or base_summary.get("source_teacher_quality_passed") is not True
        or base_summary.get("source_package_sha256")
        == QUARANTINED_SOURCE_PACKAGE_SHA256
        or base_summary.get("source_package_case_count") != 79
        or base_summary.get("runtime_approved") is not False
        or base_summary.get("training_started") is not False
    ):
        raise ValueError("base gate summary does not satisfy the schema-v3 boundary")

    results = []
    for case in sorted(expected_cases):
        source = args.recovery_dir if case in recovered else args.base_gate_dir
        candidate_path = source / f"case_{case:04d}.npz"
        result_path = source / f"case_{case:04d}.result.json"
        validate_candidate(candidate_path, case)
        result = load_passed_result(result_path, case)
        results.append(result)
        shutil.copy2(candidate_path, args.output_dir / candidate_path.name)
        shutil.copy2(result_path, args.output_dir / result_path.name)

    summary = {
        "schema": "cinebotrl_two_wheel_corrected_semantic_retarget_offline_gate_v3",
        "candidate_schema": CANDIDATE_SCHEMA,
        "training_started": False,
        "runtime_approved": False,
        "trajectory_integrity_contract": "exact_source_v1",
        "source_trajectory_integrity_passed": True,
        "source_teacher_quality_passed": True,
        "source_teacher_package": base_summary["source_teacher_package"],
        "source_package_sha256": base_summary["source_package_sha256"],
        "source_package_case_count": 79,
        "evaluated_cases": sorted(expected_cases),
        "accepted_cases": sorted(expected_cases),
        "accepted_case_count": 79,
        "rejected_cases": [],
        "rejected_case_count": 0,
        "recovered_cases": sorted(recovered),
        "recovery_method": "backward_terminal_seeded_v1",
        "physical_gimbal_joint_labels_exported": False,
        "physical_gimbal_path_use": "internal_feasibility_and_retiming_only",
        "orientation_target_contract": "semantic_DFR_world_quaternion_wxyz_option_B",
        "accepted_maximum_position_error_m": maximum(
            results, "position_error_max_m"
        ),
        "accepted_maximum_physical_gimbal_ik_error_deg": maximum(
            results, "physical_gimbal_ik_max_error_deg"
        ),
        "accepted_maximum_physical_gimbal_rate_radps": maximum(
            results, "physical_gimbal_rate_max_radps"
        ),
        "accepted_maximum_arm_gravity_effort_nm": maximum(
            results, "maximum_arm_gravity_effort_nm"
        ),
        "offline_gate_passed": True,
        "runtime_gate_started": False,
        "rejection_policy": "no NPZ is emitted for failed or timed-out cases",
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    flat_results = [
        {key: value for key, value in result.items() if key != "checks"}
        for result in results
    ]
    fieldnames = list(
        dict.fromkeys(key for result in flat_results for key in result)
    )
    with (args.output_dir / "cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_results)

    files = sorted(
        path for path in args.output_dir.iterdir() if path.name != "SHA256SUMS"
    )
    checksum_lines = [f"{sha256(path)}  ./{path.name}" for path in files]
    (args.output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii", newline="\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
