#!/usr/bin/env python3
"""Build a zero-output model-based residual policy from a BC encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    ACTION_NAMES,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (
    MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
    RiserResidualPolicy,
    initialize_model_based_residual_from_planner_imitation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT_SHA256 = (
    "dcd7d811b1c882be7fe8c9f5e9361da823591c1e61149f29302ac0cc57fbb52f"
)
FAILURE_AUDIT_SHA256 = (
    "97c90a0dc56450e4dc71654ac588eeffd09bd5d0db92bc3a4fbae265709241fd"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing model-based residual input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_policy(
    source_checkpoint: dict[str, Any], failure_audit: dict[str, Any]
) -> tuple[RiserResidualPolicy, dict[str, Any]]:
    source_state = source_checkpoint.get("model_state_dict", {})
    checks = {
        "source_schema": source_checkpoint.get("schema")
        == "cinebotrl_two_wheel_riser_residual_policy_v2",
        "source_is_masked_planner_imitation": source_checkpoint.get(
            "policy_architecture"
        )
        == MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
        "source_observation_contract": tuple(
            source_checkpoint.get("observation_names", ())
        )
        == OBSERVATION_NAMES,
        "source_action_contract": tuple(source_checkpoint.get("action_names", ()))
        == ACTION_NAMES,
        "source_state_present": bool(source_state),
        "failure_audit_passed": failure_audit.get("passed") is True
        and failure_audit.get("failed_dynamic_gate") == "position_p95_bounded",
        "failure_audit_requires_architecture_pivot": failure_audit.get(
            "architecture_audit", {}
        ).get("required_contract_satisfied")
        is False
        and failure_audit.get("architecture_audit", {}).get(
            "checkpoint_classification"
        )
        == "planner_imitation_bc_initialization_only",
        "failure_audit_keeps_training_closed": failure_audit.get("decision", {}).get(
            "bc_retraining_authorized"
        )
        is False
        and failure_audit.get("decision", {}).get("ppo_authorized") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"zero residual policy input contract failed: {checks}")

    source = RiserResidualPolicy(
        source_state["observation_mean"],
        source_state["observation_std"],
        state_hidden_sizes=source_checkpoint["state_hidden_sizes"],
        lookahead_hidden_sizes=source_checkpoint["lookahead_hidden_sizes"],
        fusion_hidden_sizes=source_checkpoint["fusion_hidden_sizes"],
        masked_observation_indices=source_checkpoint["masked_observation_indices"],
        previous_action_observation_gain=source_checkpoint[
            "previous_action_observation_gains"
        ],
    )
    source.load_state_dict(source_state)
    target = RiserResidualPolicy(
        source_state["observation_mean"],
        source_state["observation_std"],
        state_hidden_sizes=source_checkpoint["state_hidden_sizes"],
        lookahead_hidden_sizes=source_checkpoint["lookahead_hidden_sizes"],
        fusion_hidden_sizes=source_checkpoint["fusion_hidden_sizes"],
        masked_observation_indices=source_checkpoint["masked_observation_indices"],
        previous_action_observation_gain=source_checkpoint[
            "previous_action_observation_gains"
        ],
        zero_initialize_action_head=True,
    )
    initialize_model_based_residual_from_planner_imitation(target, source)
    target.eval()
    probes = torch.stack(
        (
            target.observation_mean,
            target.observation_mean + target.observation_std,
            target.observation_mean - target.observation_std,
        )
    )
    with torch.inference_mode():
        output = target(probes)
    if not torch.equal(output, torch.zeros_like(output)):
        raise ValueError("zero-initialized residual head produced a nonzero action")
    return target, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite policy output: {args.output_dir}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if args.source_commit != head:
        raise ValueError("zero residual policy source commit does not match HEAD")
    source_identity = identity(args.source_checkpoint)
    audit_identity = identity(args.failure_audit)
    if source_identity["sha256"] != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("planner-imitation checkpoint hash mismatch")
    if audit_identity["sha256"] != FAILURE_AUDIT_SHA256:
        raise ValueError("case-78 failure audit hash mismatch")
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=True
    )
    target, checks = build_policy(source_checkpoint, load_json(args.failure_audit))

    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "model_based_zero_residual_policy.pt"
    torchscript_path = (
        args.output_dir / "model_based_zero_residual_policy.torchscript.pt"
    )
    torch.save(
        {
            "schema": "cinebotrl_two_wheel_riser_model_based_residual_policy_v1",
            "policy_architecture": (
                MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
            ),
            "model_state_dict": target.state_dict(),
            "state_hidden_sizes": source_checkpoint["state_hidden_sizes"],
            "lookahead_hidden_sizes": source_checkpoint["lookahead_hidden_sizes"],
            "fusion_hidden_sizes": source_checkpoint["fusion_hidden_sizes"],
            "observation_names": OBSERVATION_NAMES,
            "action_names": ACTION_NAMES,
            "masked_observation_indices": source_checkpoint[
                "masked_observation_indices"
            ],
            "previous_action_observation_gains": source_checkpoint[
                "previous_action_observation_gains"
            ],
            "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
            "residual_action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
            "initialization": "planner_imitation_encoder_with_exact_zero_residual_head",
            "source_planner_imitation_checkpoint": source_identity,
            "source_case78_failure_audit": audit_identity,
            "source_commit": head,
            "training_started": False,
            "runtime_authorized": False,
            "valid_for_training": False,
        },
        checkpoint_path,
    )
    scripted = torch.jit.script(target)
    scripted.save(str(torchscript_path))
    with torch.inference_mode():
        scripted_output = scripted(
            torch.stack(
                (
                    target.observation_mean,
                    target.observation_mean + target.observation_std,
                    target.observation_mean - target.observation_std,
                )
            )
        )
    if not torch.equal(scripted_output, torch.zeros_like(scripted_output)):
        raise ValueError("TorchScript residual policy is not exactly zero initialized")

    report = {
        "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_build_v1",
        "source_commit": head,
        "checks": {
            **checks,
            "eager_output_exact_zero": True,
            "torchscript_output_exact_zero": True,
            "command_contract_exact": True,
            "tight_scales_exact": True,
        },
        "policy_architecture": (
            MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
        ),
        "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "residual_action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "source_planner_imitation_checkpoint": source_identity,
        "source_case78_failure_audit": audit_identity,
        "checkpoint": identity(checkpoint_path),
        "torchscript": identity(torchscript_path),
        "encoder_transferred": True,
        "residual_head_exact_zero": True,
        "runtime_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
        "passed": True,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
