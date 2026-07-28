#!/usr/bin/env python3
"""Build the tokenless case-32 natural-error validation pair contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_model_based_corrective_teacher_case32_validation_natural_error_pair import (
    EXPECTED_CONTROLLER_ARGUMENTS,
    EXPECTED_DYNAMIC_THRESHOLDS,
    EXPECTED_PAIR_CONTRACT,
    EXPECTED_RESIDUAL_SCALES,
    EXPECTED_RESOURCE_CONTRACT,
    NAMESPACE,
    REVIEWED_PARENT,
    SCHEMA,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATHS = {
    "selection": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case32_validation_selection_cpu_v1/"
        "selection.json"
    ),
    "readiness_audit": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case32_validation_pair_readiness_cpu_v1/summary.json"
    ),
    "readiness_auditor": (
        "scripts/two_wheel_balance/"
        "audit_model_based_corrective_case32_validation_pair_readiness.py"
    ),
    "profile_proposal": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case32_validation_natural_error_profile_cpu_v1/"
        "proposal.json"
    ),
    "profile_builder": (
        "scripts/two_wheel_balance/"
        "build_model_based_corrective_case32_validation_natural_error_profile.py"
    ),
    "case32_plan": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
        "case_0032_smoothed_riser_plan_v1.npz"
    ),
    "baseline_dynamic_gate": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
        "case_0032_historical_dynamic_gate.json"
    ),
    "safety_projection_source": (
        "src/rl_platform/tasks/two_wheel_balance/riser_residual_policy.py"
    ),
    "corrective_profile": (
        "scripts/two_wheel_balance/model_based_corrective_teacher_"
        "case32_validation_natural_error_profile_v1.json"
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
    "projection_telemetry_engine": (
        "scripts/two_wheel_balance/smoke_riser_case2_natural_error_pair.py"
    ),
    "projection_evidence_engine": (
        "src/rl_platform/tasks/two_wheel_balance/"
        "riser_projection_evidence.py"
    ),
    "case32_playback_adapter": (
        "scripts/two_wheel_balance/"
        "smoke_riser_case32_validation_natural_error_pair.py"
    ),
    "corrective_teacher_runtime": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_teacher.py"
    ),
    "validation_assessment": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_validation.py"
    ),
    "natural_error_finalizer_engine": (
        "scripts/two_wheel_balance/"
        "summarize_model_based_corrective_teacher_case2_natural_error_pair.py"
    ),
    "preflight_wrapper": (
        "scripts/two_wheel_balance/run_model_based_corrective_teacher_"
        "case32_validation_natural_error_pair.sh"
    ),
    "route_validator_engine": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_teacher_case8_validation_pair.py"
    ),
    "contract_builder": (
        "scripts/two_wheel_balance/build_model_based_corrective_teacher_"
        "case32_validation_natural_error_pair_contract.py"
    ),
    "contract_validator": (
        "scripts/two_wheel_balance/validate_model_based_corrective_teacher_"
        "case32_validation_natural_error_pair.py"
    ),
    "paired_finalizer": (
        "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_"
        "case32_validation_natural_error_pair.py"
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
        "case": 32,
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
        "selected_validation_cases": [8, 32],
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
