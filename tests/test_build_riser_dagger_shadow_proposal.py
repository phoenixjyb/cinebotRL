from scripts.two_wheel_balance.build_riser_dagger_shadow_proposal import (
    build_proposal,
)


def _ranking() -> dict:
    return {
        "reference_case": 4,
        "reference_split": "validation",
        "selected_training_cases": [21, 30, 31],
        "train_cases": [21, 30, 31],
        "validation_cases": [4, 8],
        "holdout_cases": [3, 5],
    }


def _gate(case: int = 21) -> dict:
    return {
        "passed": True,
        "training_started": False,
        "ppo_authorized": False,
        "results": [
            {
                "case": case,
                "dynamic_quality_passed": True,
                "thermal_admission_passed": True,
                "controller_evidence_passed": True,
            }
        ],
    }


def test_proposal_selects_only_rank1_training_case_and_keeps_runtime_closed() -> None:
    proposal = build_proposal(_ranking(), {21: _gate()}, candidate_count=1)
    assert proposal["proposed_cases"] == [21]
    assert proposal["proposed_case_split"] == "train"
    assert proposal["runtime_authorized"] is False
    assert proposal["authorization_token_issued"] is False
    assert proposal["dataset_created"] is False
    assert proposal["dagger_authorized"] is False
    assert proposal["bc_authorized"] is False
    assert proposal["ppo_authorized"] is False
    assert proposal["valid_for_training"] is False


def test_proposal_rejects_more_than_one_initial_case() -> None:
    try:
        build_proposal(_ranking(), {21: _gate(), 30: _gate(30)}, candidate_count=2)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multi-case initial proposal was accepted")


def test_proposal_rejects_failed_teacher_gate() -> None:
    gate = _gate()
    gate["results"][0]["dynamic_quality_passed"] = False
    try:
        build_proposal(_ranking(), {21: gate}, candidate_count=1)
    except ValueError as error:
        assert "teacher gate failed" in str(error)
    else:
        raise AssertionError("failed teacher gate was accepted")
