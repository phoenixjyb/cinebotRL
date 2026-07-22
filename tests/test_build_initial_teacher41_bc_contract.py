import pytest

from scripts.two_wheel_balance.build_initial_teacher41_bc_contract import (
    HOLDOUT_CASES,
    TRAIN_CASES,
    VALIDATION_CASES,
    build_contract,
)


def _payloads():
    return {
        "dataset_summary": {
            "passed": True,
            "dataset_admission_passed": True,
            "valid_for_bc_initialization": True,
            "dataset_version": "initial_teacher41_case78_31_5_5_v2",
            "case_count": 41,
            "row_count": 486619,
            "split_cases": {
                "train": TRAIN_CASES,
                "validation": VALIDATION_CASES,
                "holdout": HOLDOUT_CASES,
            },
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
            "holdout_policy_metrics_computed": False,
            "holdout_used_for_model_selection": False,
        },
        "loader_audit": {
            "passed": True,
            "dataset_sha256": "03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497",
            "row_count": 486619,
            "holdout_metrics_computed": False,
        },
        "label_admission": {
            "label_admission_passed": True,
            "labels_applied_to_commands": False,
            "holdout_opened": False,
        },
        "original_report": {"offline_gate_passed": True},
        "original_final": {"passed": True, "learned_rollout_started": False},
        "masked_report": {
            "policy_architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
            "offline_gate_passed": True,
        },
        "masked_final": {"passed": True},
        "masked_canary_summary": {
            "passed": False,
            "means": {
                "teacher_position_p95_m": 0.1287,
                "learned_position_p95_m": 0.1374,
                "zero_position_p95_m": 1.0722,
            },
        },
        "masked_canary_final": {"passed": False},
        "scheduled_final": {"passed": False},
        "gain010_final": {"passed": False},
        "channel_gain_final": {"passed": False},
    }


def test_selects_masked_bc_without_authorizing_training() -> None:
    result = build_contract(_payloads(), "a" * 40)
    assert result["cpu_contract_ready"]
    assert result["architecture_decision"]["mask_previous_action_observations"]
    assert result["post_training_route"]["dynamic_canary_order"] == [8, 78]
    assert not result["bc_training_authorized"]
    assert not result["ppo_authorized"]
    assert not result["holdout_opened"]


def test_rejects_teacher_budget_claim_or_failed_loader() -> None:
    payloads = _payloads()
    payloads["loader_audit"]["passed"] = False
    payloads["masked_canary_summary"]["means"]["learned_position_p95_m"] = 0.12
    with pytest.raises(ValueError, match="BC contract failed"):
        build_contract(payloads, "a" * 40)
