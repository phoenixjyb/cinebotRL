import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case23_conversion_readiness.py"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_review_contract_v1.json"
)
SPEC = importlib.util.spec_from_file_location("case23_conversion_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _repository_checks() -> dict[str, bool]:
    return {
        "canonical_contract": True,
        "contract_tracked": True,
        "contract_blob_matches_head": True,
        "head_matches_upstream": True,
        "tracked_worktree_clean": True,
        "reviewed_parent_is_ancestor": True,
    }


def _review(contract: Path = CONTRACT):
    return MODULE.audit_readiness(
        contract,
        ROOT,
        repository_checks=_repository_checks(),
        git_state={"head": "a" * 40, "upstream": "a" * 40},
    )


def test_windows_repository_path_maps_to_wsl_git_path() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"


def test_real_case23_v4_source_is_ready_for_separate_conversion() -> None:
    result = _review()
    assert result["passed"] is True
    assert result["case"] == 23
    assert result["split"] == "train"
    assert result["prospective_dataset_metrics"]["sample_count"] == 3273
    assert result["source_checks"]["effective_actions_exact"] is True
    assert result["source_checks"]["requested_actions_audit_exact"] is True
    assert result["source_checks"]["previous_action_rebuilt"] is True
    assert result["source_checks"]["non_previous_observations_exact"] is True
    assert result["source_checks"]["elapsed_clock_exact"] is True
    assert result["source_checks"]["execution_clock_exact"] is True
    assert result["source_checks"]["source_clock_exact"] is True
    assert result["output_created"] is False
    assert result["conversion_authorized"] is False
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False
    assert result["valid_for_training"] is False
    assert not Path(result["output"]).exists()


def test_review_rejects_any_repository_provenance_failure() -> None:
    for name in _repository_checks():
        checks = _repository_checks()
        checks[name] = False
        result = MODULE.audit_readiness(
            CONTRACT,
            ROOT,
            repository_checks=checks,
            git_state={"head": "a" * 40, "upstream": "a" * 40},
        )
        assert result["passed"] is False
        assert result["decision"] == "do_not_convert_case23_v4"
        assert result["output_created"] is False


def test_review_rejects_forged_identity_or_open_authorization(tmp_path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["identities"]["source_capture"]["sha256"] = "0" * 64
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")
    assert _review(forged)["passed"] is False

    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["conversion_contract"]["conversion_authorized"] = True
    forged.write_text(json.dumps(payload), encoding="utf-8")
    assert _review(forged)["passed"] is False


def test_review_rejects_case_split_or_output_substitution(tmp_path) -> None:
    for field, value in (
        ("case", 30),
        ("split", "validation"),
    ):
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        payload[field] = value
        forged = tmp_path / f"forged_{field}.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        assert _review(forged)["passed"] is False

    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["conversion_contract"]["output_relative_path"] = "alternate.npz"
    forged = tmp_path / "forged_output.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")
    assert _review(forged)["passed"] is False
