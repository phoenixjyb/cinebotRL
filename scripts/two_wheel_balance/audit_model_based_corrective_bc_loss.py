#!/usr/bin/env python3
"""Audit projected BC-loss semantics against the admitted case-30 evidence."""

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
from rl_platform.tasks.two_wheel_balance.riser_model_based_bc_loss import (  # noqa: E402
    MODEL_BASED_PROJECTED_BC_LOSS,
    REQUESTED_OUTPUT_SLEW_REGULARIZATION,
    ModelBasedProjectedBCLoss,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    load_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
)

SCHEMA = "cinebotrl_two_wheel_riser_model_based_bc_loss_audit_v1"
DEFAULT_DATASET = (
    PROJECT_ROOT / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case30_effective_label_conversion_v1/"
    "case_0030_model_based_corrective_case_dataset_v1.npz"
)
DEFAULT_PROFILE = (
    PROJECT_ROOT / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case30_profile_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_bc_loss_v1/summary.json"
)
LOSS_MODULE_PATH = (
    PROJECT_ROOT / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_bc_loss.py"
)
PROJECTION_TOLERANCE = 5e-6
LOSS_TOLERANCE = 1e-9
SLEW_TOLERANCE = 1e-6


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _rate_violations(
    actions: np.ndarray,
    elapsed_time_s: np.ndarray,
    action_scales: np.ndarray,
    maximum_slew_rates: np.ndarray,
) -> np.ndarray:
    rates = (
        np.abs(np.diff(actions, axis=0))
        * action_scales
        / np.diff(elapsed_time_s)[:, None]
    )
    return rates > maximum_slew_rates + SLEW_TOLERANCE


def audit(dataset_path: Path, profile_path: Path) -> dict[str, object]:
    metadata, payload = load_case_dataset(
        dataset_path, expected_case=30, expected_split="train"
    )
    _, profile, profile_identity = load_corrective_teacher_profile(
        profile_path, expected_case=30
    )
    scales = np.asarray(metadata["action_scales"], dtype=np.float64)
    slew_rates = np.asarray(profile.maximum_slew_rates, dtype=np.float64)
    model_commands = np.asarray(payload["model_based_commands"], dtype=np.float32)
    requested = np.asarray(payload["requested_actions_audit"], dtype=np.float32)
    effective = np.asarray(payload["actions"], dtype=np.float32)
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    recorded_clipped = np.asarray(payload["command_clipped"], dtype=bool)

    previous_requested = np.empty_like(requested)
    previous_requested[0] = requested[0]
    previous_requested[1:] = requested[:-1]
    delta_time = np.empty(len(elapsed), dtype=np.float32)
    delta_time[0] = 1.0
    delta_time[1:] = np.diff(elapsed).astype(np.float32)
    transition_valid = np.ones(len(elapsed), dtype=bool)
    transition_valid[0] = False

    requested_tensor = torch.tensor(requested, requires_grad=True)
    loss_module = ModelBasedProjectedBCLoss(
        action_scales=scales.tolist(),
        maximum_slew_rates=slew_rates.tolist(),
        slew_regularization_weight=1.0,
    )
    (
        total_loss,
        pointwise_loss,
        slew_loss,
        projected_actions,
        projected_clipped,
    ) = loss_module(
        torch.from_numpy(model_commands),
        requested_tensor,
        torch.from_numpy(effective),
        torch.from_numpy(previous_requested),
        torch.from_numpy(delta_time),
        torch.from_numpy(transition_valid),
        torch.ones(len(elapsed), dtype=torch.float32),
    )
    total_loss.backward()
    gradient = requested_tensor.grad
    if gradient is None:
        raise RuntimeError("projected BC loss produced no requested-action gradient")

    projected_np = projected_actions.detach().numpy().astype(np.float64)
    projected_error = float(np.max(np.abs(projected_np - effective.astype(np.float64))))
    clipping_exact = bool(
        np.array_equal(projected_clipped.detach().numpy(), recorded_clipped)
    )
    naive_requested_effective_mse = float(
        np.mean(np.square(requested.astype(np.float64) - effective))
    )
    requested_violations = _rate_violations(requested, elapsed, scales, slew_rates)
    effective_violations = _rate_violations(effective, elapsed, scales, slew_rates)
    transition_clipped = recorded_clipped[:-1] | recorded_clipped[1:]
    effective_violations_unclipped = effective_violations & ~transition_clipped

    checks = {
        "case_dataset_is_merge_only": metadata.get("valid_for_case_merge") is True
        and metadata.get("valid_for_training") is False,
        "training_remains_closed": metadata.get("bc_authorized") is False
        and metadata.get("ppo_authorized") is False
        and metadata.get("training_started") is False,
        "projection_reconstructs_effective_targets": (
            projected_error <= PROJECTION_TOLERANCE
        ),
        "projected_pointwise_loss_is_numerically_zero": (
            float(pointwise_loss.detach()) <= LOSS_TOLERANCE
        ),
        "requested_slew_regularization_is_numerically_zero": (
            float(slew_loss.detach()) <= LOSS_TOLERANCE
        ),
        "naive_requested_target_loss_is_nonzero": (
            naive_requested_effective_mse > LOSS_TOLERANCE
        ),
        "requested_teacher_has_no_slew_violations": not np.any(requested_violations),
        "effective_target_has_projection_induced_slew_violations": (
            int(np.count_nonzero(effective_violations)) == 87
            and not np.any(effective_violations_unclipped)
        ),
        "projection_clipping_mask_is_exact": clipping_exact,
        "requested_action_gradient_is_finite": bool(
            torch.isfinite(gradient).all().item()
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "case": 30,
        "loss_contract": MODEL_BASED_PROJECTED_BC_LOSS,
        "projection_contract": MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
        "requested_slew_regularization_contract": (
            REQUESTED_OUTPUT_SLEW_REGULARIZATION
        ),
        "implementation": {
            "loss_module": _identity(LOSS_MODULE_PATH),
            "audit_script": _identity(Path(__file__)),
            "torchscript_compatible": True,
            "differentiable": True,
        },
        "inputs": {
            "dataset": _identity(dataset_path),
            "corrective_profile": {
                **profile_identity,
                "path": _identity(profile_path)["path"],
            },
        },
        "sample_count": len(requested),
        "transition_count": len(requested) - 1,
        "action_scales": scales.tolist(),
        "maximum_slew_rates": slew_rates.tolist(),
        "metrics": {
            "total_loss": float(total_loss.detach()),
            "projected_pointwise_loss": float(pointwise_loss.detach()),
            "requested_slew_regularization_loss": float(slew_loss.detach()),
            "naive_requested_to_effective_mse": (naive_requested_effective_mse),
            "projected_effective_action_max_error": projected_error,
            "requested_slew_violation_count_per_channel": np.count_nonzero(
                requested_violations, axis=0
            ).tolist(),
            "effective_slew_violation_count_per_channel": np.count_nonzero(
                effective_violations, axis=0
            ).tolist(),
            "effective_unclipped_slew_violation_count_per_channel": (
                np.count_nonzero(effective_violations_unclipped, axis=0).tolist()
            ),
            "recorded_clipped_sample_count": int(np.count_nonzero(recorded_clipped)),
            "requested_gradient_abs_max": float(torch.max(torch.abs(gradient)).item()),
        },
        "checks": checks,
        "passed": passed,
        "valid_for_bc_loss_contract_review": passed,
        "corpus_created": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "capture_authorized": False,
        "dataset_creation_authorized": False,
        "learned_rollout_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corrective-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite projected BC-loss audit: {args.output}"
        )
    report = audit(args.dataset, args.corrective_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
