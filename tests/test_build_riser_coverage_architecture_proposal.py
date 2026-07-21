from scripts.two_wheel_balance.build_riser_coverage_architecture_proposal import (
    EXPECTED_HOLDOUT_CASES,
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    build_proposal,
)


TEACHER_SHA = "a" * 64
CASE4_TRACE_SHA = "b" * 64
CASE21_DIAGNOSIS_SHA = "c" * 64
LOCALIZED_SHA = "d" * 64


def _identity(sha256: str) -> dict[str, str]:
    return {"path": "/test/input", "sha256": sha256}


def _inputs() -> dict[str, object]:
    dataset_metadata = {
        "split_cases": {
            "train": EXPECTED_TRAIN_CASES,
            "validation": EXPECTED_VALIDATION_CASES,
            "holdout": EXPECTED_HOLDOUT_CASES,
        }
    }
    case4_final = {
        "passed": True,
        "valid_for_training": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "shadow_trace": _identity(CASE4_TRACE_SHA),
    }
    case4_diagnosis = {
        "dagger_dataset_proposal_supported": True,
        "material_shadow_shift_by_channel": [True, True, False],
        "inputs": {
            "teacher_dataset": _identity(TEACHER_SHA),
            "shadow_trace": _identity(CASE4_TRACE_SHA),
        },
    }
    case21_final = {
        "case": 21,
        "split": "train",
        "passed": True,
        "diagnosis": _identity(CASE21_DIAGNOSIS_SHA),
    }
    case21_diagnosis = {
        "dagger_dataset_proposal_supported": False,
        "material_shadow_shift_by_channel": [False, False, False],
        "inputs": {"teacher_dataset": _identity(TEACHER_SHA)},
    }
    localized_audit = {
        "coverage_admission_passed": False,
        "proposed_runtime_cases": [],
        "ranked_training_cases": [
            {"case": 18, "reference_score_ratio": 3.642897236}
        ],
        "inputs": {
            "teacher_dataset": _identity(TEACHER_SHA),
            "shadow_trace": _identity(CASE4_TRACE_SHA),
        },
    }
    unused_audit = {
        "coverage_expansion_admission_passed": False,
        "proposed_shadow_measurement_cases": [],
        "best_existing_training": {"case": 30},
        "best_unused_admitted": {"case": 78},
        "unused_admitted_cases": [20, 78],
        "unused_to_existing_best_score_ratio": 1.229251733,
        "inputs": {
            "teacher_dataset": _identity(TEACHER_SHA),
            "localized_audit": _identity(LOCALIZED_SHA),
            "shadow_trace": _identity(CASE4_TRACE_SHA),
        },
    }
    input_identities = {
        "teacher_dataset": _identity(TEACHER_SHA),
        "case4_final": _identity("e" * 64),
        "case4_diagnosis": _identity("f" * 64),
        "case21_final": _identity("1" * 64),
        "case21_diagnosis": _identity(CASE21_DIAGNOSIS_SHA),
        "localized_audit": _identity(LOCALIZED_SHA),
        "unused_audit": _identity("2" * 64),
    }
    return {
        "dataset_metadata": dataset_metadata,
        "case4_final": case4_final,
        "case4_diagnosis": case4_diagnosis,
        "case21_final": case21_final,
        "case21_diagnosis": case21_diagnosis,
        "localized_audit": localized_audit,
        "unused_audit": unused_audit,
        "input_identities": input_identities,
    }


def test_proposal_prefers_one_train_case_perturbation() -> None:
    proposal = build_proposal(**_inputs())
    option = proposal["options"]["controlled_perturbation"]
    assert proposal["decision"] == "controlled_perturbation_contract_first"
    assert option["first_bounded_measurement_case"] == 30
    assert option["case_split"] == "train"
    assert option["perturbation_contract"]["randomized"] is False
    assert option["perturbation_contract"]["measurement_trace_only"] is True


def test_proposal_does_not_open_runtime_or_training() -> None:
    proposal = build_proposal(**_inputs())
    assert proposal["runtime_authorized"] is False
    assert proposal["authorization_token_issued"] is False
    assert proposal["runtime_namespace_created"] is False
    assert proposal["dataset_created"] is False
    assert proposal["dagger_authorized"] is False
    assert proposal["bc_authorized"] is False
    assert proposal["ppo_authorized"] is False
    assert proposal["valid_for_training"] is False


def test_split_reset_is_explicit_fallback_and_preserves_holdout() -> None:
    proposal = build_proposal(**_inputs())
    option = proposal["options"]["transparent_split_reset"]
    assert option["recommendation"] == "fallback_only"
    assert option["case4_permanently_retired_from_validation"] is True
    assert 4 in option["proposed_train_cases_not_applied"]
    assert 4 not in option["proposed_validation_cases_not_applied"]
    assert 78 in option["proposed_validation_cases_not_applied"]
    assert option["holdout_cases_unchanged"] == EXPECTED_HOLDOUT_CASES
    assert proposal["case4_split_changed"] is False
    assert proposal["case78_validation_admitted"] is False
    assert proposal["holdout_opened"] is False


def test_split_or_identity_drift_fails_closed() -> None:
    inputs = _inputs()
    inputs["dataset_metadata"]["split_cases"]["validation"] = [4, 8, 16, 22, 78]
    try:
        build_proposal(**inputs)
    except ValueError as error:
        assert "input contract failed" in str(error)
    else:
        raise AssertionError("validation split drift was accepted")

    inputs = _inputs()
    inputs["case4_final"]["shadow_trace"]["sha256"] = "0" * 64
    try:
        build_proposal(**inputs)
    except ValueError as error:
        assert "input contract failed" in str(error)
    else:
        raise AssertionError("case4 shadow identity drift was accepted")
