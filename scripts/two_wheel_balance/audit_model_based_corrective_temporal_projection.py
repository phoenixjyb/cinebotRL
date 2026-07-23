#!/usr/bin/env python3
"""Audit temporal label integrity and deterministic safety projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (  # noqa: E402
    load_corrective_teacher_profile,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    load_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
    ModelBasedResidualSafetyProjection,
)


SCHEMA = "cinebotrl_two_wheel_riser_corrective_temporal_projection_audit_v1"
SLEW_TOLERANCE = 1e-6
PROJECTION_COMMAND_TOLERANCE = 2e-6
PROJECTION_ACTION_TOLERANCE = 5e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _rate_summary(
    actions: np.ndarray,
    elapsed_time_s: np.ndarray,
    action_scales: np.ndarray,
    maximum_slew_rates: np.ndarray,
    transition_clipped: np.ndarray,
) -> dict[str, object]:
    physical_rates = (
        np.abs(np.diff(actions, axis=0))
        / np.diff(elapsed_time_s)[:, None]
        * action_scales
    )
    violations = physical_rates > maximum_slew_rates + SLEW_TOLERANCE
    clipped_violations = violations & transition_clipped
    unclipped_violations = violations & ~transition_clipped
    return {
        "physical_rate_abs_p95": np.percentile(
            physical_rates, 95, axis=0
        ).tolist(),
        "physical_rate_abs_p99": np.percentile(
            physical_rates, 99, axis=0
        ).tolist(),
        "physical_rate_abs_max": np.max(physical_rates, axis=0).tolist(),
        "violation_count_per_channel": np.count_nonzero(
            violations, axis=0
        ).tolist(),
        "violation_transition_count": int(
            np.count_nonzero(np.any(violations, axis=1))
        ),
        "clipped_violation_count_per_channel": np.count_nonzero(
            clipped_violations, axis=0
        ).tolist(),
        "unclipped_violation_count_per_channel": np.count_nonzero(
            unclipped_violations, axis=0
        ).tolist(),
        "all_violations_associated_with_command_clipping": bool(
            np.all(~violations | transition_clipped)
        ),
        "slew_limit_passed": bool(not np.any(violations)),
    }


def audit(dataset_path: Path, corrective_profile_path: Path) -> dict[str, object]:
    metadata, payload = load_case_dataset(
        dataset_path, expected_case=30, expected_split="train"
    )
    _, profile, profile_identity = load_corrective_teacher_profile(
        corrective_profile_path, expected_case=30
    )
    action_scales = np.asarray(metadata["action_scales"], dtype=np.float64)
    maximum_slew_rates = np.asarray(
        profile.maximum_slew_rates, dtype=np.float64
    )
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    requested_actions = np.asarray(
        payload["requested_actions_audit"], dtype=np.float64
    )
    effective_actions = np.asarray(payload["actions"], dtype=np.float64)
    command_clipped = np.asarray(payload["command_clipped"], dtype=bool)
    transition_clipped = command_clipped[:-1] | command_clipped[1:]

    requested_summary = _rate_summary(
        requested_actions,
        elapsed,
        action_scales,
        maximum_slew_rates,
        transition_clipped,
    )
    effective_summary = _rate_summary(
        effective_actions,
        elapsed,
        action_scales,
        maximum_slew_rates,
        transition_clipped,
    )

    projection = ModelBasedResidualSafetyProjection(
        action_scales=action_scales.tolist()
    ).eval()
    with torch.inference_mode():
        projected_commands, projected_actions, projected_clipped = projection(
            torch.as_tensor(
                payload["model_based_commands"], dtype=torch.float32
            ),
            torch.as_tensor(requested_actions, dtype=torch.float32),
        )
    final_commands = np.asarray(
        payload["final_high_level_commands"], dtype=np.float64
    )
    projected_commands_np = projected_commands.numpy().astype(np.float64)
    projected_actions_np = projected_actions.numpy().astype(np.float64)
    projected_clipped_np = projected_clipped.numpy()
    command_error = float(
        np.max(np.abs(projected_commands_np - final_commands))
    )
    action_error = float(
        np.max(np.abs(projected_actions_np - effective_actions))
    )
    clipping_exact = bool(np.array_equal(projected_clipped_np, command_clipped))

    checks = {
        "case_dataset_admitted_for_merge_only": metadata.get(
            "valid_for_case_merge"
        )
        is True
        and metadata.get("valid_for_training") is False,
        "training_remains_closed": metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("training_started") is False,
        "requested_teacher_intent_obeys_slew": requested_summary[
            "slew_limit_passed"
        ]
        is True,
        "effective_slew_violations_are_observed": effective_summary[
            "slew_limit_passed"
        ]
        is False
        and effective_summary["violation_transition_count"] > 0,
        "effective_violations_are_supervisor_projection_only": effective_summary[
            "all_violations_associated_with_command_clipping"
        ]
        is True,
        "projection_reconstructs_final_commands": (
            command_error <= PROJECTION_COMMAND_TOLERANCE
        ),
        "projection_reconstructs_effective_actions": (
            action_error <= PROJECTION_ACTION_TOLERANCE
        ),
        "projection_reconstructs_clipping_mask": clipping_exact,
    }
    return {
        "schema": SCHEMA,
        "case": 30,
        "audit_script": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "dataset": {
            "path": portable_path(dataset_path),
            "sha256": sha256(dataset_path),
            "sample_count": int(len(effective_actions)),
        },
        "corrective_profile": {
            **profile_identity,
            "path": portable_path(corrective_profile_path),
            "maximum_slew_rates": maximum_slew_rates.tolist(),
        },
        "action_scales": action_scales.tolist(),
        "transition_count": int(len(effective_actions) - 1),
        "requested_teacher_intent": requested_summary,
        "effective_post_supervisor_labels": effective_summary,
        "safety_projection": {
            "contract": MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
            "final_command_reconstruction_max_error": command_error,
            "effective_action_reconstruction_max_error": action_error,
            "clipping_mask_exact": clipping_exact,
        },
        "recommended_training_contract": {
            "policy_output_semantics": "requested_bounded_residual_v1",
            "pointwise_training_target": "effective_post_supervisor_residual_v1",
            "loss_projection": MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
            "requested_actions_used_as_pointwise_targets": False,
            "effective_actions_remain_pointwise_targets": True,
            "requested_output_slew_regularization_required": True,
            "effective_label_slew_must_not_be_naively_gated_at_clipped_transitions": True,
            "runtime_safety_supervisor_remains_final_authority": True,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "valid_for_bc_contract_review": all(checks.values()),
        "valid_for_training": False,
        "runtime_authorized": False,
        "learned_rollout_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corrective-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.dataset, args.corrective_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
