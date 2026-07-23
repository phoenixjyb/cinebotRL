import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from rl_platform.tasks.two_wheel_balance import (
    riser_model_based_learned_render_contract as contract,
)


ROOT = Path(__file__).parents[1]
EXECUTION_COMMIT = "a" * 40
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_LEARNED_RENDER_ADMISSION_TEMPLATE_20260723.json"
)
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_learned_render_admission.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/run_model_based_learned_render_gate.sh"
)
MEDIA_AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/audit_model_based_learned_render_media.py"
)
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/finalize_model_based_learned_render.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREFLIGHT_MODULE = _module(VALIDATOR, "learned_render_preflight")
MEDIA_MODULE = _module(MEDIA_AUDITOR, "learned_render_media")
FINALIZER_MODULE = _module(FINALIZER, "learned_render_finalizer")


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": contract.sha256_file(path)}


def _fixture(tmp_path: Path, *, authorized: bool):
    paths = {}
    for name in (
        "policy",
        "all79_admission",
        "all79_preflight",
        "plan_manifest",
        "source_manifest",
        "lqr_gains",
        "robot_build_audit",
        "robot_usd",
        "drive_profile_selection",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    policy_sha = contract.sha256_file(paths["policy"])
    all79 = {
        "schema": "cinebotrl_two_wheel_riser_residual_all79_gate_v1",
        "policy_sha256": policy_sha,
        "execution_commit": EXECUTION_COMMIT,
        "cases": list(range(1, 80)),
        "passed": True,
        "rows": [
            {"case": case, "checks": {"passed": True}}
            for case in range(1, 80)
        ],
    }
    all79_path = tmp_path / "all79.json"
    all79_path.write_text(json.dumps(all79), encoding="utf-8")
    paths["all79_report"] = all79_path
    code_paths = {}
    for name in contract.CODE_KEYS:
        path = tmp_path / f"{name}.py"
        path.write_text(name, encoding="utf-8")
        code_paths[name] = path
    admission = {
        "schema": contract.SCHEMA,
        "all79_report": _identity(all79_path),
        "all79_admission": _identity(paths["all79_admission"]),
        "all79_preflight": _identity(paths["all79_preflight"]),
        "policy": _identity(paths["policy"]),
        "plan_manifest": _identity(paths["plan_manifest"]),
        "source_manifest": _identity(paths["source_manifest"]),
        "lqr_gains": _identity(paths["lqr_gains"]),
        "robot_build_audit": _identity(paths["robot_build_audit"]),
        "robot_usd": _identity(paths["robot_usd"]),
        "drive_profile_selection": _identity(paths["drive_profile_selection"]),
        "execution_commit": EXECUTION_COMMIT,
        "code": {name: _identity(path) for name, path in code_paths.items()},
        "cases": contract.REPRESENTATIVE_CASES,
        "case_roles": contract.CASE_ROLES,
        "render_config": copy.deepcopy(contract.RENDER_CONFIG),
        "all79_gate_passed": True,
        "render_evaluation_approved": authorized,
        "learned_render_authorized": authorized,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    paths["code_paths"] = code_paths
    return admission, all79, paths


def _validate(tmp_path: Path, admission: dict, all79: dict, paths: dict, auth=True):
    contract.validate_render_admission(
        admission,
        identity_root=tmp_path,
        all79_report_path=paths["all79_report"],
        all79_report=all79,
        all79_admission_path=paths["all79_admission"],
        all79_preflight_path=paths["all79_preflight"],
        policy_path=paths["policy"],
        plan_manifest_path=paths["plan_manifest"],
        source_manifest_path=paths["source_manifest"],
        lqr_gains_path=paths["lqr_gains"],
        robot_build_audit_path=paths["robot_build_audit"],
        robot_usd_path=paths["robot_usd"],
        drive_profile_selection_path=paths["drive_profile_selection"],
        code_paths=paths["code_paths"],
        expected_execution_commit=EXECUTION_COMMIT,
        require_authorized=auth,
    )


@pytest.fixture(autouse=True)
def _accept_source_and_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        contract,
        "_exact_source_manifest_valid",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(contract, "_plan_manifest_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        contract,
        "model_based_residual_torchscript_valid",
        lambda *args, **kwargs: True,
    )


def test_template_is_closed_and_selects_diverse_cases() -> None:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert value["cases"] == [1, 15, 31, 50, 73, 79]
    assert set(value["case_roles"]) == {"1", "15", "31", "50", "73", "79"}
    assert value["render_evaluation_approved"] is False
    assert value["learned_render_authorized"] is False
    assert value["ppo_authorized"] is False


def test_cpu_review_is_valid_but_runtime_requires_authorization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, authorized=False)
    _validate(tmp_path, *fixture, auth=False)
    with pytest.raises(ValueError, match="authorized"):
        _validate(tmp_path, *fixture, auth=True)


def test_exact_authorized_render_admission_passes(tmp_path: Path) -> None:
    _validate(tmp_path, *_fixture(tmp_path, authorized=True))


def test_render_admission_rejects_invalid_policy_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        contract,
        "model_based_residual_torchscript_valid",
        lambda *args, **kwargs: False,
    )
    with pytest.raises(ValueError, match="admission failed"):
        _validate(tmp_path, *_fixture(tmp_path, authorized=True))


@pytest.mark.parametrize(
    "mutation",
    ("policy", "all79", "case", "config", "code", "ppo", "row"),
)
def test_render_admission_rejects_forged_or_unsafe_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    admission, all79, paths = _fixture(tmp_path, authorized=True)
    if mutation == "policy":
        admission["policy"]["sha256"] = "0" * 64
    elif mutation == "all79":
        all79["passed"] = False
    elif mutation == "case":
        admission["cases"] = [1, 31, 73]
    elif mutation == "config":
        admission["render_config"]["residual_action_scales"] = [0.3, 0.4, 0.1]
    elif mutation == "code":
        admission["code"]["playback"]["sha256"] = "0" * 64
    elif mutation == "ppo":
        admission["ppo_authorized"] = True
    else:
        all79["rows"][72]["checks"]["passed"] = False
    with pytest.raises(ValueError, match="admission failed"):
        _validate(tmp_path, admission, all79, paths)


def _preflight_args(tmp_path: Path) -> argparse.Namespace:
    admission, all79, paths = _fixture(tmp_path, authorized=True)
    admission_path = tmp_path / "render_admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    paths["all79_report"].write_text(json.dumps(all79), encoding="utf-8")
    return argparse.Namespace(
        admission=admission_path,
        all79_report=paths["all79_report"],
        all79_admission=paths["all79_admission"],
        all79_preflight=paths["all79_preflight"],
        policy=paths["policy"],
        plan_manifest=paths["plan_manifest"],
        source_manifest=paths["source_manifest"],
        lqr_gains=paths["lqr_gains"],
        robot_build_audit=paths["robot_build_audit"],
        robot_usd=paths["robot_usd"],
        drive_profile_selection=paths["drive_profile_selection"],
        require_authorized=True,
        output=tmp_path / "result.json",
    )


def test_preflight_binds_clean_head_and_deep_all79_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = {"deep": False}
    monkeypatch.setattr(
        PREFLIGHT_MODULE.contract,
        "validate_render_admission",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        PREFLIGHT_MODULE,
        "_load_auditor",
        lambda: type(
            "Auditor",
            (),
            {
                "_validate_all79_report": staticmethod(
                    lambda *args, **kwargs: observed.update(deep=True)
                )
            },
        ),
    )
    monkeypatch.setattr(
        PREFLIGHT_MODULE,
        "_git",
        lambda *args: "" if args[0] == "status" else EXECUTION_COMMIT,
    )
    result = PREFLIGHT_MODULE.validate(_preflight_args(tmp_path))
    assert result["passed"] is True
    assert result["runtime_started"] is False
    assert result["recording_started"] is False
    assert observed["deep"] is True


def _media_args(tmp_path: Path) -> argparse.Namespace:
    paths = {}
    for name in ("admission", "preflight", "policy", "all79_report"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    videos = []
    rollouts = []
    for case in contract.REPRESENTATIVE_CASES:
        video = tmp_path / f"case_{case:04d}.mp4"
        video.write_bytes(f"video-{case}".encode())
        rollout = tmp_path / f"case_{case:04d}.json"
        rollout.write_text(
            json.dumps(
                {
                    "cases": [case],
                    "passed": True,
                    "trajectory_command_source": (
                        "model_based_planner_plus_torchscript_residual"
                    ),
                    "tracking_profile": contract.RENDER_CONFIG["tracking_profile"],
                    "policy_command_base": "model_based_planner",
                    "residual_action_scales": [0.05, 0.05, 0.02],
                    "results": [{"case": case, "passed": True}],
                }
            ),
            encoding="utf-8",
        )
        videos.append(f"{case}={video}")
        rollouts.append(f"{case}={rollout}")
    return argparse.Namespace(
        **paths,
        case_video=videos,
        case_rollout=rollouts,
        output=tmp_path / "media.json",
    )


def test_media_audit_and_explicit_visual_review_finalize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MEDIA_MODULE,
        "probe",
        lambda path: {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "duration_s": 10.0,
        },
    )
    media = MEDIA_MODULE.audit(_media_args(tmp_path))
    assert media["passed"] is True
    assert media["manual_visual_review_required"] is True
    media_path = tmp_path / "media.json"
    media_path.write_text(json.dumps(media), encoding="utf-8")
    review = {
        "schema": "cinebotrl_two_wheel_riser_learned_render_visual_review_v1",
        "cases": contract.REPRESENTATIVE_CASES,
        "videos": media["videos"],
        "reviewer": "test-reviewer",
        "reviewed_at_utc": "2026-07-23T12:00:00Z",
        "visual_checks": {
            name: True for name in FINALIZER_MODULE.VISUAL_CHECKS
        },
        "notes": "",
        "passed": True,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = FINALIZER_MODULE.finalize(media_path, review_path)
    assert report["passed"] is True
    assert report["schema"].endswith("learned_render_audit_v2")


def test_finalizer_rejects_blank_reviewer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        MEDIA_MODULE,
        "probe",
        lambda path: {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "duration_s": 10.0,
        },
    )
    media = MEDIA_MODULE.audit(_media_args(tmp_path))
    media_path = tmp_path / "media.json"
    media_path.write_text(json.dumps(media), encoding="utf-8")
    review = {
        "schema": "cinebotrl_two_wheel_riser_learned_render_visual_review_v1",
        "cases": contract.REPRESENTATIVE_CASES,
        "videos": media["videos"],
        "reviewer": "",
        "reviewed_at_utc": "2026-07-23T12:00:00Z",
        "visual_checks": {
            name: True for name in FINALIZER_MODULE.VISUAL_CHECKS
        },
        "notes": "",
        "passed": True,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        FINALIZER_MODULE.finalize(media_path, review_path)


def test_wrapper_preflights_before_namespace_and_uses_d3d12() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    preflight = source.index('"$ISAAC_PYTHON" "$(to_windows_path "$PREFLIGHT")"')
    namespace = source.index('mkdir -p "$output/rollouts"')
    playback = source.index('"$ISAAC_PYTHON" -u -X utf8 "$playback_win"')
    assert preflight < namespace < playback
    assert "to_windows_path" in source
    assert 'RISER_GIT_ROOT_WSL="$ROOT"' in source
    assert "WSLENV=" in source
    assert 'mktemp -p "$ROOT"' in source
    assert 'python3 "$PREFLIGHT"' not in source
    assert "--require-authorized" in source
    assert "--enable_cameras" in source
    assert "--experience \"$D3D12_EXPERIENCE\"" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--residual-action-scales 0.05,0.05,0.02" in source
    assert "cases=(1 15 31 50 73 79)" in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 2
    assert "missing_environment" in result.stderr
