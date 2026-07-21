#!/usr/bin/env python3
"""Build a non-runtime proposal for training-split shadow measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing proposal input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_proposal(
    ranking: dict[str, object],
    teacher_gates: dict[int, dict[str, object]],
    *,
    candidate_count: int,
) -> dict[str, object]:
    selected = ranking["selected_training_cases"][:candidate_count]
    train_cases = set(ranking["train_cases"])
    excluded = set(ranking["validation_cases"]) | set(ranking["holdout_cases"])
    if candidate_count != 1:
        raise ValueError("the first bounded proposal must contain exactly one case")
    if len(selected) != 1 or not set(selected).issubset(train_cases):
        raise ValueError("proposal selection is not a training case")
    if set(selected) & excluded:
        raise ValueError("validation or holdout case entered the proposal")
    case = int(selected[0])
    gate = teacher_gates[case]
    result = gate["results"][0]
    gate_checks = {
        "case": result.get("case") == case,
        "runtime_passed": gate.get("passed") is True,
        "dynamic_quality": result.get("dynamic_quality_passed") is True,
        "thermal_admission": result.get("thermal_admission_passed") is True,
        "controller_evidence": result.get("controller_evidence_passed") is True,
        "no_training": gate.get("training_started") is False,
        "no_ppo": gate.get("ppo_authorized") is False,
    }
    if not all(gate_checks.values()):
        raise ValueError(f"candidate teacher gate failed: {gate_checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_dagger_shadow_measurement_proposal_v1",
        "reference_case": ranking["reference_case"],
        "reference_split": ranking["reference_split"],
        "proposed_cases": selected,
        "proposed_case_count": 1,
        "proposed_case_split": "train",
        "candidate_teacher_gate_checks": {str(case): gate_checks},
        "selection_contract": (
            "rank1_training_case_only_case4_validation_and_holdout_excluded_v1"
        ),
        "shadow_teacher_measurement_proposed": True,
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "dataset_creation_authorized": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--case4-final", type=Path, required=True)
    parser.add_argument("--case4-diagnosis", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--teacher-gate-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    selected = ranking["selected_training_cases"][: args.candidate_count]
    gates = {
        int(case): json.loads(
            (args.teacher_gate_dir / f"case_{int(case):04d}.json").read_text(
                encoding="utf-8"
            )
        )
        for case in selected
    }
    proposal = build_proposal(
        ranking, gates, candidate_count=args.candidate_count
    )
    proposal["inputs"] = {
        "ranking": _identity(args.ranking),
        "case4_final": _identity(args.case4_final),
        "case4_diagnosis": _identity(args.case4_diagnosis),
        "dataset": _identity(args.dataset),
        "policy": _identity(args.policy),
    }
    proposal["candidate_artifacts"] = {
        str(case): {
            "plan": _identity(
                args.plan_dir / f"case_{int(case):04d}_smoothed_riser_plan_v1.npz"
            ),
            "teacher_gate": _identity(
                args.teacher_gate_dir / f"case_{int(case):04d}.json"
            ),
        }
        for case in selected
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proposal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
