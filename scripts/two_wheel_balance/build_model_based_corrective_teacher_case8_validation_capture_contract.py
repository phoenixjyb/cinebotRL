#!/usr/bin/env python3
"""Build the tokenless case-8 validation corrective capture contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from validate_model_based_corrective_teacher_case30_capture import (
    EXPECTED_CAPTURE,
    EXPECTED_HOLDOUT,
    EXPECTED_SCALES,
)
from validate_model_based_corrective_teacher_case8_validation_capture import (
    EXPECTED_EXECUTION,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATHS = {
    "paired_final_status": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case8_validation_pair_v2/final_status.json"
    ),
    "case8_plan": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260724_case8_validation_pair_readiness_cpu_v1/source/"
        "case_0008_smoothed_riser_plan_v1.npz"
    ),
    "perturbation_profile": (
        "scripts/two_wheel_balance/"
        "model_based_corrective_teacher_case8_validation_wrench_profile_v1.json"
    ),
    "perturbation_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_perturbation.py"
    ),
    "corrective_profile": (
        "scripts/two_wheel_balance/"
        "model_based_corrective_teacher_case8_validation_profile_v1.json"
    ),
    "drive_profile_selection": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_riser_drive_profile_selection_v1/summary.json"
    ),
    "lqr_gains": (
        "docs/03_training/two_wheel_balance/evidence_20260714_28kg/"
        "lqr_gains.json"
    ),
    "robot_build_audit": (
        "assets_own/recomoProto2_two_wheel_riser/build_audit.json"
    ),
    "robot_urdf": (
        "assets_own/recomoProto2_two_wheel_riser/"
        "recomoProto2_two_wheel_riser.urdf"
    ),
    "robot_usd": (
        "assets_own/recomoProto2_two_wheel_riser/"
        "recomoProto2_two_wheel_riser.usd"
    ),
    "playback": "scripts/two_wheel_balance/smoke_riser_reference_playback.py",
    "corrective_teacher_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_teacher.py"
    ),
    "corrective_capture_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_capture.py"
    ),
    "capture_validator_runtime": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_teacher_case30_capture.py"
    ),
    "capture_finalizer_runtime": (
        "scripts/two_wheel_balance/"
        "summarize_model_based_corrective_teacher_case30_capture.py"
    ),
    "resource_finalizer_runtime": (
        "scripts/two_wheel_balance/"
        "summarize_model_based_corrective_teacher_case7_capture.py"
    ),
    "contract_validator": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_teacher_case8_validation_capture.py"
    ),
    "preflight_wrapper": (
        "scripts/two_wheel_balance/"
        "run_model_based_corrective_teacher_case8_validation_capture.sh"
    ),
    "shared_windows_resource_guard": (
        "scripts/two_wheel_balance/check_windows_shared_resource_admission.py"
    ),
    "shared_windows_resource_monitor": (
        "scripts/two_wheel_balance/monitor_windows_shared_resource_pressure.py"
    ),
    "capture_finalizer": (
        "scripts/two_wheel_balance/"
        "summarize_model_based_corrective_teacher_case8_validation_capture.py"
    ),
}


def _git_blob(path: Path) -> str:
    if os.name == "nt":
        drive = PROJECT_ROOT.drive[0].lower()
        root_relative = PROJECT_ROOT.as_posix().split(":/", 1)[1]
        path_relative = path.as_posix().split(":/", 1)[1]
        command = [
            "wsl.exe",
            "--exec",
            "git",
            "-C",
            f"/mnt/{drive}/{root_relative}",
            "hash-object",
            f"/mnt/{drive}/{path_relative}",
        ]
    else:
        command = [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "hash-object",
            str(path),
        ]
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity(relative: str) -> dict[str, str]:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob_sha1": _git_blob(path),
    }


def build_contract() -> dict[str, object]:
    paired_path = PROJECT_ROOT / IDENTITY_PATHS["paired_final_status"]
    paired_sha = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "route_revision": "case8_validation_capture_v1_monitored_coexistence",
        "reviewed_parent_commit": REVIEWED_PARENT,
        "case": 8,
        "split": "validation",
        "namespace": NAMESPACE,
        "paired_canary": {
            "namespace": (
                "20260728_model_based_corrective_teacher_"
                "case8_validation_pair_v2_coexistence"
            ),
            "final_status_sha256": paired_sha,
            "dynamic_pair_completed": True,
            "validation_pair_passed": True,
            "label_capture_authorized_by_pair": False,
        },
        "residual_action_scales": EXPECTED_SCALES,
        "capture_schema_contract": EXPECTED_CAPTURE,
        "execution_contract": EXPECTED_EXECUTION,
        "identities": {
            name: _identity(path) for name, path in IDENTITY_PATHS.items()
        },
        "holdout_cases": EXPECTED_HOLDOUT,
        "holdout_opened": False,
        "validation_cases_opened": [8],
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "authorization_token_issued": False,
        "runtime_authorization_mode": "out_of_band_sha256_environment_v1",
        "runtime_authorization_token_sha256": "",
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "conversion_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite contract: {args.output}")
    payload = build_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
