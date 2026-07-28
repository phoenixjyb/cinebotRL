#!/usr/bin/env python3
"""Build the tokenless case-8 validation pair route contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from validate_model_based_corrective_teacher_case8_validation_pair import (
    EXPECTED_CONTROLLER_ARGUMENTS,
    EXPECTED_DYNAMIC_THRESHOLDS,
    EXPECTED_PAIR_CONTRACT,
    EXPECTED_RESOURCE_CONTRACT,
    EXPECTED_RESIDUAL_SCALES,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATHS = {
    "selection": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_model_based_corrective_validation_tranche_v1/"
        "selection.json"
    ),
    "readiness_audit": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260724_case8_validation_pair_readiness_cpu_v1/summary.json"
    ),
    "readiness_auditor": (
        "scripts/two_wheel_balance/"
        "audit_model_based_corrective_case8_validation_pair_readiness.py"
    ),
    "profile_proposal": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260724_case8_validation_pair_profile_cpu_v1/proposal.json"
    ),
    "profile_builder": (
        "scripts/two_wheel_balance/"
        "build_model_based_corrective_case8_validation_pair_profiles.py"
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
    "corrective_profile": (
        "scripts/two_wheel_balance/"
        "model_based_corrective_teacher_case8_validation_profile_v1.json"
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
    "playback": (
        "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
    ),
    "corrective_teacher_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_teacher.py"
    ),
    "perturbation_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_perturbation.py"
    ),
    "validation_assessment": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_validation.py"
    ),
    "preflight_wrapper": (
        "scripts/two_wheel_balance/"
        "run_model_based_corrective_teacher_case8_validation_pair.sh"
    ),
    "contract_validator": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_teacher_case8_validation_pair.py"
    ),
    "paired_finalizer": (
        "scripts/two_wheel_balance/"
        "summarize_model_based_corrective_teacher_case8_validation_pair.py"
    ),
    "shared_windows_resource_guard": (
        "scripts/two_wheel_balance/check_windows_shared_resource_admission.py"
    ),
    "shared_windows_resource_monitor": (
        "scripts/two_wheel_balance/"
        "monitor_windows_shared_resource_pressure.py"
    ),
}


def _identity(relative: str) -> dict[str, str]:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
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
    blob = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob_sha1": blob,
    }


def build_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "case": 8,
        "split": "validation",
        "namespace": NAMESPACE,
        "residual_action_scales": EXPECTED_RESIDUAL_SCALES,
        "controller_arguments": EXPECTED_CONTROLLER_ARGUMENTS,
        "unchanged_dynamic_gate_thresholds": EXPECTED_DYNAMIC_THRESHOLDS,
        "paired_experiment_contract": EXPECTED_PAIR_CONTRACT,
        "shared_resource_contract": EXPECTED_RESOURCE_CONTRACT,
        "identities": {
            name: _identity(path) for name, path in IDENTITY_PATHS.items()
        },
        "source_validation_cases": [8, 16, 22, 32, 78],
        "selected_validation_cases": [8, 16],
        "validation_runtime_opened": False,
        "holdout_opened": False,
        "cpu_preflight_ready": True,
        "runtime_route_contract_ready": True,
        "execution_route_complete": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "authorization_token_issued": False,
        "runtime_authorization_token_sha256": "",
        "teacher_admission_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
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
    args.output.write_bytes((json.dumps(payload, indent=2) + "\n").encode())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
