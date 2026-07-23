import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_learned_split_admission.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_learned_split_policy_gate.sh"
)
SPEC = importlib.util.spec_from_file_location("learned_split_preflight", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path, *, mode: str = "validation_canary") -> argparse.Namespace:
    paths = {}
    for name in (
        "admission",
        "bc_report",
        "policy",
        "plan_manifest",
        "source_manifest",
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
    ):
        path = tmp_path / name
        path.write_bytes(b"policy" if name == "policy" else b"{}")
        paths[name] = path
    admission = {
        "cases": [8, 16] if mode == "validation_canary" else [3, 5, 13, 19, 24],
        "split_evaluation_approved": True,
        "learned_rollout_authorized": True,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    paths["admission"].write_text(json.dumps(admission), encoding="utf-8")
    paths["bc_report"].write_text(
        json.dumps({"execution_commit": "a" * 40}),
        encoding="utf-8",
    )
    prior = None
    if mode == "holdout":
        prior = tmp_path / "validation_report"
        prior.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        mode=mode,
        **paths,
        prior_validation_gate_report=prior,
        require_authorized=True,
        output=tmp_path / "result.json",
    )


def test_preflight_binds_clean_head_and_all_identities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = {}
    monkeypatch.setattr(
        MODULE,
        "validate_learned_split_admission",
        lambda admission, **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(
        MODULE,
        "_git",
        lambda *args: "" if args[0] == "status" else "a" * 40,
    )
    result = MODULE.validate(_args(tmp_path))
    assert result["passed"] is True
    assert result["runtime_started"] is False
    assert result["dataset_written"] is False
    assert result["capture_started"] is False
    assert result["bc_started"] is False
    assert result["ppo_started"] is False
    assert observed["mode"] == "validation_canary"
    assert observed["expected_execution_commit"] == "a" * 40
    assert observed["require_authorized"] is True
    assert set(observed["code_paths"]) == MODULE.CODE_IDENTITY_KEYS


@pytest.mark.parametrize("failure", ["upstream", "dirty", "bc_commit"])
def test_preflight_rejects_ambiguous_execution_state(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "validate_learned_split_admission",
        lambda *args, **kwargs: None,
    )

    def fake_git(*args: str) -> str:
        if args[0] == "status":
            return " M tracked.py" if failure == "dirty" else ""
        if args[0] == "rev-parse" and args[1] == "@{upstream}":
            return "b" * 40 if failure == "upstream" else "a" * 40
        return "a" * 40

    monkeypatch.setattr(MODULE, "_git", fake_git)
    args = _args(tmp_path)
    if failure == "bc_commit":
        args.bc_report.write_text(
            json.dumps({"execution_commit": "b" * 40}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="preflight failed"):
        MODULE.validate(args)


def test_holdout_preflight_forwards_prior_validation_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = {}
    monkeypatch.setattr(
        MODULE,
        "validate_learned_split_admission",
        lambda admission, **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(
        MODULE,
        "_git",
        lambda *args: "" if args[0] == "status" else "a" * 40,
    )
    args = _args(tmp_path, mode="holdout")
    result = MODULE.validate(args)
    assert result["mode"] == "holdout"
    assert result["prior_validation_gate_report"] is not None
    assert observed["prior_validation_report_path"] == (
        args.prior_validation_gate_report.resolve()
    )


def test_wrapper_rejects_before_namespace_and_uses_current_model_based_route() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    preflight = source.index('"$ISAAC_PYTHON" "$preflight_win"')
    namespace = source.index('mkdir -p "$output/baseline"')
    playback = source.index('"$ISAAC_PYTHON" -u -X utf8 "$playback_win"')
    assert preflight < namespace < playback
    assert "to_windows_path" in source
    assert 'mktemp -p "$ROOT"' in source
    assert 'python3 "$PREFLIGHT"' not in source
    assert "--require-authorized" in source
    assert "missing_prior_validation_report" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--residual-action-scales 0.05,0.05,0.02" in source
    assert "model_based_planner_plus_bounded_policy_residual_v1" in source
    assert '--rollout-admission "$output/admission.json"' in source
    assert '--preflight-receipt "$output/preflight.json"' in source
    assert "resume_admission_mismatch" in source
    assert "resume_preflight_mismatch" in source
    assert "timeout --signal=TERM --kill-after=30s 1800" in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "validation_canary", "--execute"],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 2
    assert "missing_environment" in result.stderr
