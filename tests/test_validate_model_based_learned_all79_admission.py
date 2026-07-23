import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_learned_all79_admission.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_learned_all79_policy_gate.sh"
)
SPEC = importlib.util.spec_from_file_location("learned_all79_preflight", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path) -> argparse.Namespace:
    names = (
        "admission",
        "bc_report",
        "policy",
        "plan_manifest",
        "source_manifest",
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
        "validation_gate_report",
        "holdout_gate_report",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        if name == "policy":
            path.write_bytes(b"policy")
        else:
            path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    paths["bc_report"].write_text(
        json.dumps({"execution_commit": "a" * 40}),
        encoding="utf-8",
    )
    return argparse.Namespace(
        **paths,
        require_authorized=True,
        output=tmp_path / "result.json",
    )


def test_preflight_binds_clean_head_and_forwards_every_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = {}

    def fake_validate(admission, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        MODULE.contract,
        "validate_learned_all79_admission",
        fake_validate,
    )
    monkeypatch.setattr(
        MODULE,
        "_git",
        lambda *args: "" if args[0] == "status" else "a" * 40,
    )
    result = MODULE.validate(_args(tmp_path))
    assert result["passed"] is True
    assert result["runtime_started"] is False
    assert result["dataset_created"] is False
    assert result["ppo_started"] is False
    assert observed["require_authorized"] is True
    assert observed["expected_execution_commit"] == "a" * 40
    assert set(observed["code_paths"]) == MODULE.contract.CODE_IDENTITY_KEYS


def test_preflight_preserves_contract_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MODULE.contract,
        "validate_learned_all79_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forged")),
    )
    monkeypatch.setattr(
        MODULE,
        "_git",
        lambda *args: "" if args[0] == "status" else "a" * 40,
    )
    result = MODULE.validate(_args(tmp_path))
    assert result["passed"] is False
    assert result["checks"]["contract"] is False
    assert result["error"] == "forged"
    assert result["runtime_started"] is False


def test_wrapper_rejects_before_namespace_or_isaac_without_admission() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    preflight = source.index('"$ISAAC_PYTHON" "$(to_windows_path "$PREFLIGHT")"')
    namespace = source.index('mkdir -p "$output/teacher"')
    playback = source.index('"$ISAAC_PYTHON" -u -X utf8 "$PLAYBACK_WIN"')
    assert preflight < namespace < playback
    assert "to_windows_path" in source
    assert 'RISER_GIT_ROOT_WSL="$ROOT"' in source
    assert 'mktemp -p "$ROOT"' in source
    assert 'python3 "$PREFLIGHT"' not in source
    assert "--require-authorized" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--residual-action-scales 0.05,0.05,0.02" in source
    assert "model_based_planner_plus_bounded_policy_residual_v1" in source
    assert '--rollout-admission "$output/admission.json"' in source
    assert '--preflight-receipt "$output/preflight.json"' in source
    assert '--execution-commit "$execution_commit"' in source
    assert "MODE\" == --resume" in source
    assert "resume_admission_mismatch" in source
    assert "resume_preflight_mismatch" in source
    assert "rollout_is_valid()" in source
    assert "timeout --signal=TERM --kill-after=30s 1800" in source
    assert "--residual-action-scales 0.30,0.40,0.10" not in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 2
    assert "missing_environment" in result.stderr
