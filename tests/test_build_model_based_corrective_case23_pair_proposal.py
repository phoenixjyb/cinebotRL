import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/build_model_based_corrective_case23_pair_proposal.py"
)
SPEC = importlib.util.spec_from_file_location("case23_pair_proposal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path):
    portfolio_path = tmp_path / "portfolio/manifest.json"
    plan_path = portfolio_path.parent / "case_0023.npz"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"case23-plan")
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    portfolio = {
        "schema": MODULE.PORTFOLIO_SCHEMA,
        "items": [
            {
                "case": 23,
                "file": plan_path.name,
                "plan_sha256": plan_sha,
                "source_duration_s": 9.0,
                "execution_duration_s": 10.0,
                "passed": True,
                "timing_transition_kinematic_gate_passed": True,
            }
        ],
    }
    _write(portfolio_path, portfolio)
    selection_path = tmp_path / "selection.json"
    selection = {
        "schema": MODULE.SELECTION_SCHEMA,
        "identities": {
            "portfolio": {
                "sha256": hashlib.sha256(portfolio_path.read_bytes()).hexdigest()
            }
        },
        "selected_cases": [30, 23],
        "selected_rows": [
            {
                "case": 23,
                "selection_role": "same_seed_paired_canary_required",
                "plan_sha256": plan_sha,
            }
        ],
        "validation_cases": MODULE.VALIDATION_CASES,
        "holdout_cases": MODULE.HOLDOUT_CASES,
        "case30_profile_reuse_authorized": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }
    _write(selection_path, selection)
    profile_path = tmp_path / "profile.json"
    _write(
        profile_path,
        {
            "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
            "case": 23,
            "longitudinal_gain_s_inv": 0.2,
            "lateral_to_yaw_gain_rad_s_m": 0.3,
            "vertical_gain": 0.3,
            "deadbands_m": [0.01, 0.01, 0.005],
            "maximum_residuals": [0.045, 0.045, 0.018],
            "maximum_slew_rates": [0.1, 0.1, 0.04],
        },
    )
    return selection, portfolio, profile_path, selection_path, portfolio_path


def _build(tmp_path: Path):
    selection, portfolio, profile, selection_path, portfolio_path = _fixture(tmp_path)
    return MODULE.build_proposal(
        selection,
        portfolio,
        profile,
        selection_path=selection_path,
        portfolio_path=portfolio_path,
    )


def test_case23_proposal_is_specific_paired_and_runtime_closed(tmp_path) -> None:
    result = _build(tmp_path)
    assert result["case"] == 23
    assert result["split"] == "train"
    assert result["proposed_perturbation"]["start_phase_time_s"] == 5.0
    assert result["paired_experiment"]["rollout_order"] == ["baseline", "candidate"]
    assert result["runtime_route_implemented"] is False
    assert result["authorization_token_issued"] is False
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["dataset_merge_authorized"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["valid_for_training"] is False


@pytest.mark.parametrize("failure", ["role", "holdout", "plan", "profile", "runtime"])
def test_case23_proposal_rejects_unreviewed_or_open_inputs(tmp_path, failure) -> None:
    selection, portfolio, profile, selection_path, portfolio_path = _fixture(tmp_path)
    if failure == "role":
        selection["selected_rows"][0]["selection_role"] = "converted_pilot_anchor"
    elif failure == "holdout":
        selection["holdout_cases"] = [3, 5]
    elif failure == "plan":
        portfolio["items"][0]["plan_sha256"] = "0" * 64
    elif failure == "profile":
        payload = json.loads(profile.read_text(encoding="utf-8"))
        payload["case"] = 30
        _write(profile, payload)
    else:
        selection["gpu_launch_authorized"] = True
    with pytest.raises(ValueError):
        MODULE.build_proposal(
            selection,
            portfolio,
            profile,
            selection_path=selection_path,
            portfolio_path=portfolio_path,
        )
