import json
from pathlib import Path

import pytest

from rl_platform.tasks.two_wheel_balance.riser_perturbation import (
    PERTURBATION_SCHEMA,
    DeterministicWrenchPulse,
    DeterministicWrenchPulseRuntime,
    load_deterministic_wrench_profile,
)


def _payload() -> dict[str, object]:
    return {
        "schema": PERTURBATION_SCHEMA,
        "case": 30,
        "start_phase_time_s": 4.0,
        "duration_steps": 20,
        "force_body_x_n": -20.0,
        "application_height_m": 0.5,
    }


def test_profile_load_is_exact_and_hash_bound(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")
    profile, identity = load_deterministic_wrench_profile(path)
    assert profile.case == 30
    assert profile.duration_steps == 20
    assert identity["path"] == str(path.resolve())
    assert len(identity["sha256"]) == 64


def test_profile_requires_explicit_case_identity_for_case23(tmp_path) -> None:
    payload = _payload()
    payload["case"] = 23
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid deterministic wrench profile"):
        load_deterministic_wrench_profile(path)
    profile, identity = load_deterministic_wrench_profile(
        path, expected_case=23
    )
    assert profile.case == 23
    assert len(identity["sha256"]) == 64


@pytest.mark.parametrize("expected_case", [0, -1, True, 23.0])
def test_profile_rejects_invalid_expected_case(tmp_path, expected_case) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="expected perturbation case"):
        load_deterministic_wrench_profile(path, expected_case=expected_case)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case", 4),
        ("duration_steps", 51),
        ("duration_steps", 1.5),
        ("force_body_x_n", 0.0),
        ("force_body_x_n", 40.01),
        ("application_height_m", 1.01),
        ("start_phase_time_s", -0.1),
    ],
)
def test_profile_rejects_unreviewed_or_unbounded_values(
    tmp_path, field, value
) -> None:
    payload = _payload()
    payload[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_deterministic_wrench_profile(path)


def test_profile_rejects_missing_or_extra_fields(tmp_path) -> None:
    for payload in (
        {key: value for key, value in _payload().items() if key != "case"},
        _payload() | {"random_seed": 1},
    ):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected.*fields"):
            load_deterministic_wrench_profile(path)


def test_pulse_uses_phase_trigger_and_exact_execution_step_duration() -> None:
    profile = DeterministicWrenchPulse(
        case=30,
        start_phase_time_s=1.0,
        duration_steps=3,
        force_body_x_n=-20.0,
        application_height_m=0.5,
    )
    runtime = DeterministicWrenchPulseRuntime(profile)
    commands = [
        runtime.command(step=0, phase_time_s=0.0),
        runtime.command(step=1, phase_time_s=0.9),
        runtime.command(step=2, phase_time_s=1.0),
        runtime.command(step=3, phase_time_s=1.0),
        runtime.command(step=4, phase_time_s=1.0),
        runtime.command(step=5, phase_time_s=1.0),
    ]
    assert commands == [0.0, 0.0, -20.0, -20.0, -20.0, 0.0]
    assert all(runtime.contract_checks().values())
    summary = runtime.summary()
    assert summary["trigger_step"] == 2
    assert summary["active_step_count"] == 3
    assert summary["applied_to_planner_commands"] is False
    assert summary["applied_to_policy_actions"] is False


def test_disabled_contract_is_command_identical_and_admitted() -> None:
    runtime = DeterministicWrenchPulseRuntime(None)
    assert [runtime.command(step=step, phase_time_s=step / 200) for step in range(5)] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert all(runtime.contract_checks().values())
    assert runtime.summary()["enabled"] is False


def test_playback_wires_measurement_only_body_wrench_without_command_changes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
    ).read_text(encoding="utf-8")
    pre_app = source.split("app = AppLauncher(args).app", 1)[0]
    loop = source.split("for step in range(maximum_steps):", 1)[1]
    assert '"--deterministic-wrench-profile"' in pre_app
    assert "load_deterministic_wrench_profile" in pre_app
    assert "expected_case=requested_cases[0]" in pre_app
    assert "requires its one pinned case only" in pre_app
    assert "requires learned-policy shadow-teacher" in pre_app
    assert "is_global=False" in source
    assert "perturbation_runtime.command(" in loop
    assert loop.index("perturbation_runtime.command(") < loop.index("env.step(")
    assert '"perturbation_applied_to_planner_commands": False' in source
    assert '"perturbation_applied_to_policy_actions": False' in source
    assert '"perturbation_contract_passed"' in source
