#!/usr/bin/env python3
"""Build the CPU-only BC contract for the resealed 41-case riser dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASHES = {
    "dataset": "03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497",
    "dataset_summary": "2b7b177f481fdc632aca2134d9eea69cec66814581a5e39d9c6a099e3d8bcbfb",
    "loader_audit": "7a764d9cc41e9d43dd808251e2b8466e7ef0940bd356cbebeee38ffcd88e34cb",
    "label_admission": "cd752e402c912d7a83767544c8059d2068979d10868dff2408fd93836b71033d",
    "original_report": "44437ed005aa69718c244fda8c8fddb58ebf95c308c9387d11d85e4ff62ce104",
    "original_final": "05771143d93cbce0abd20124a787c1d7348cc482238b3fdb52dd6ad2ea038cdf",
    "masked_report": "3f0efb4a2707b343a775dd5dd8b0ad49d6506474da627d8449ca81556cbbcd3e",
    "masked_final": "ded00f25dde299207dc0e3af0b611418e09d5368d4fc9e7cab53b57df9a36bba",
    "masked_canary_summary": "ff483e5ee8b975419fc75efbdf7c22e013a5c5dccc56df17e538d9113d35abdc",
    "masked_canary_final": "1a4fbcd16fa3490d9b187b4af90298c8c04f7d674060189ae8116cd500257cdb",
    "scheduled_final": "abab8f8726dc0ac545ae3f45f147e3e10e590ef507682ebe5197d0d80b5f447e",
    "gain010_final": "628380b9f9ae8f7a837ccea30906bfb5f9f3518d01124ea6dd4eb92e40f685bf",
    "channel_gain_final": "4e1190078383c94c987540c70790c27ef953af3995360fe131e8874c18455aa6",
}
TRAIN_CASES = [2, 4, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 21, 23, 25, 26, 28, 30, 31, 33, 34, 36, 37, 41, 52, 53, 66, 67, 68, 70, 74]
VALIDATION_CASES = [8, 16, 22, 32, 78]
HOLDOUT_CASES = [3, 5, 13, 19, 24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing BC-contract input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_contract(payloads: dict[str, dict[str, Any]], source_commit: str) -> dict[str, Any]:
    summary = payloads["dataset_summary"]
    loader = payloads["loader_audit"]
    label = payloads["label_admission"]
    original_report = payloads["original_report"]
    original_final = payloads["original_final"]
    masked_report = payloads["masked_report"]
    masked_final = payloads["masked_final"]
    canary = payloads["masked_canary_summary"]
    canary_final = payloads["masked_canary_final"]
    canary_means = canary.get("means", {})
    checks = {
        "source_commit_is_dataset_implementation_or_descendant": len(source_commit) == 40,
        "dataset_is_resealed_initialization_input": summary.get("passed") is True
        and summary.get("dataset_admission_passed") is True
        and summary.get("valid_for_bc_initialization") is True
        and summary.get("dataset_version") == "initial_teacher41_case78_31_5_5_v2"
        and summary.get("case_count") == 41
        and summary.get("row_count") == 486619,
        "split_exact": summary.get("split_cases")
        == {
            "train": TRAIN_CASES,
            "validation": VALIDATION_CASES,
            "holdout": HOLDOUT_CASES,
        },
        "dataset_learning_closed": summary.get("bc_authorized") is False
        and summary.get("ppo_authorized") is False
        and summary.get("training_started") is False,
        "production_loader_passed": loader.get("passed") is True
        and loader.get("dataset_sha256") == EXPECTED_HASHES["dataset"]
        and loader.get("row_count") == 486619
        and loader.get("holdout_metrics_computed") is False,
        "case78_labels_admitted_without_application": label.get(
            "label_admission_passed"
        )
        is True
        and label.get("labels_applied_to_commands") is False
        and label.get("holdout_opened") is False,
        "original_bc_was_only_offline_admitted": original_report.get(
            "offline_gate_passed"
        )
        is True
        and original_final.get("passed") is True
        and original_final.get("learned_rollout_started") is False,
        "masked_bc_was_best_admitted_architecture": masked_report.get(
            "policy_architecture"
        )
        == "state_shared_lookahead_fusion_previous_action_masked_v1"
        and masked_report.get("offline_gate_passed") is True
        and masked_final.get("passed") is True,
        "masked_canary_passed_absolute_not_teacher_budget": canary_final.get(
            "passed"
        )
        is False
        and canary.get("passed") is False
        and canary_means.get("learned_position_p95_m", 1.0) <= 0.15
        and canary_means.get("learned_position_p95_m", 0.0)
        > canary_means.get("teacher_position_p95_m", 1.0)
        and canary_means.get("learned_position_p95_m", 1.0)
        < canary_means.get("zero_position_p95_m", 0.0),
        "scheduled_and_gain_alternatives_rejected": all(
            payloads[name].get("passed") is False
            for name in ("scheduled_final", "gain010_final", "channel_gain_final")
        ),
        "holdout_remains_closed": summary.get("holdout_policy_metrics_computed")
        is False
        and summary.get("holdout_used_for_model_selection") is False
        and loader.get("holdout_metrics_computed") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"initial teacher-41 BC contract failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_bc_cpu_contract_v1",
        "source_commit": source_commit,
        "reviewed_dataset_implementation_commit": "301194698b2d3a7b684afe9e1478763b4891571b",
        "dataset_case_count": 41,
        "dataset_row_count": 486619,
        "split_cases": {
            "train": TRAIN_CASES,
            "validation": VALIDATION_CASES,
            "holdout": HOLDOUT_CASES,
        },
        "architecture_decision": {
            "policy_architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
            "mask_previous_action_observations": True,
            "scheduled_previous_action_max_probability": 0.0,
            "previous_action_observation_gain": 1.0,
            "reason": "best historical dynamic near-pass; scheduled and attenuated alternatives rejected",
        },
        "training_contract": {
            "method": "offline_behavior_cloning",
            "epochs_max": 80,
            "patience": 10,
            "batch_size": 4096,
            "learning_rate": 0.0003,
            "weight_decay": 0.00001,
            "seed": 20260722,
            "device": "cuda",
            "minimum_improvement_fraction": 0.05,
            "model_selection_splits": ["validation"],
            "holdout_opened": False,
        },
        "post_training_route": {
            "offline_gate_first": True,
            "dynamic_canary_order": [8, 78],
            "automatic_case78_launch": False,
            "automatic_broad_rollout": False,
        },
        "input_contract_checks": checks,
        "cpu_contract_ready": True,
        "runtime_authorization_token_issued": False,
        "bc_training_authorized": False,
        "learned_rollout_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "training_started": False,
        "valid_for_runtime": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED_HASHES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite BC CPU contract: {args.output}")
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if args.source_commit != actual_head:
        raise ValueError("BC-contract source commit does not match HEAD")
    if subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "301194698b2d3a7b684afe9e1478763b4891571b",
            actual_head,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("BC-contract HEAD does not descend from dataset implementation")
    paths = {name: getattr(args, name) for name in EXPECTED_HASHES}
    identities = {name: identity(path) for name, path in paths.items()}
    if any(
        identities[name]["sha256"] != expected
        for name, expected in EXPECTED_HASHES.items()
    ):
        raise ValueError("BC-contract input hash mismatch")
    payloads = {
        name: load_json(path)
        for name, path in paths.items()
        if name != "dataset"
    }
    result = build_contract(payloads, args.source_commit)
    result["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
