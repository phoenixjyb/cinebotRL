import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_goal_completion.py"
SPEC = importlib.util.spec_from_file_location("riser_goal_completion_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _current_inputs():
    paths = {
        "goal": MODULE.DEFAULT_GOAL,
        "asset": MODULE.DEFAULT_ASSET_AUDIT,
        "lqr": MODULE.DEFAULT_LQR_GATE,
        "baseline": MODULE.DEFAULT_BASELINE,
        "exact_source": MODULE.DEFAULT_EXACT_SOURCE,
        "hardware": MODULE.DEFAULT_HARDWARE,
        "bench": MODULE.DEFAULT_BENCH,
    }
    payloads = {name: MODULE._load_json(path) for name, path in paths.items()}
    identities = {name: MODULE._identity(path) for name, path in paths.items()}
    identities["auditor"] = MODULE._identity(SCRIPT)
    return payloads, identities


def _build_current(**overrides):
    payloads, identities = _current_inputs()
    payloads.update(overrides)
    return MODULE.build_report(
        **payloads,
        git_state={
            "root": str(ROOT),
            "branch": MODULE.EXPECTED_BRANCH,
            "head": "a" * 40,
            "upstream": "a" * 40,
            "tracked_dirty": False,
        },
        learning={
            "training_metadata": None,
            "bc_report": None,
            "all79_report": None,
            "learned_render_report": None,
        },
        inputs=identities,
    )


def test_current_goal_audit_passes_foundations_but_not_learning() -> None:
    report = _build_current()
    assert report["goal_achieved"] is False
    assert report["required_gate_pass_count"] == 6
    assert report["required_gate_count"] == 10
    assert report["completion_blockers"] == [
        "model_based_corrective_training_corpus",
        "projection_aware_bc_policy",
        "learned_policy_all79_dynamic_gate",
        "learned_policy_render_audit",
    ]
    for gate in (
        "isolated_worktree_and_branch",
        "arm_free_robot_asset",
        "frozen_lqr_balance_baseline",
        "riser_height_and_speed_baseline",
        "exact_source_all79_reference",
        "riser_motor_and_mechanism_recommendation",
    ):
        assert report["gates"][gate]["passed"] is True
    assert report["gates"]["physical_riser_bench_qualification"][
        "required_for_goal"
    ] is False
    assert report["gates"]["physical_riser_bench_qualification"]["passed"] is False
    assert report["runtime_started"] is False
    assert report["bc_started_by_audit"] is False
    assert report["ppo_started_by_audit"] is False
    assert report["git"] == {
        "branch": MODULE.EXPECTED_BRANCH,
        "head": "a" * 40,
        "upstream": "a" * 40,
        "tracked_dirty": False,
    }
    assert report["gates"]["isolated_worktree_and_branch"]["evidence"] == [
        f"git:{MODULE.EXPECTED_BRANCH}@{'a' * 40}"
    ]


def test_goal_audit_rejects_arm_or_1p9m_contract_drift() -> None:
    payloads, _ = _current_inputs()
    asset = json.loads(json.dumps(payloads["asset"]))
    asset["movable_joint_names"].append("joint1_arm_yaw")
    asset["checks"]["arm_joints_absent"] = False
    goal = json.loads(json.dumps(payloads["goal"]))
    goal["robot_contract"]["camera_height_m"] = [0.6, 1.9]
    report = _build_current(asset=asset, goal=goal)
    assert report["gates"]["arm_free_robot_asset"]["passed"] is False
    assert report["gates"]["riser_height_and_speed_baseline"]["passed"] is False
    assert "arm_free_robot_asset" in report["completion_blockers"]
    assert "riser_height_and_speed_baseline" in report["completion_blockers"]


def test_goal_audit_rejects_dirty_or_diverged_worktree() -> None:
    payloads, identities = _current_inputs()
    report = MODULE.build_report(
        **payloads,
        git_state={
            "root": str(ROOT),
            "branch": MODULE.EXPECTED_BRANCH,
            "head": "a" * 40,
            "upstream": "b" * 40,
            "tracked_dirty": True,
        },
        learning={
            "training_metadata": None,
            "bc_report": None,
            "all79_report": None,
            "learned_render_report": None,
        },
        inputs=identities,
    )
    assert report["gates"]["isolated_worktree_and_branch"]["passed"] is False
    assert "isolated_worktree_and_branch" in report["completion_blockers"]


def test_cli_writes_incomplete_evidence_without_starting_work(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    report = json.loads(payload)
    assert report["goal_achieved"] is False
    assert report["runtime_started"] is False
    assert report["bc_started_by_audit"] is False
    assert report["ppo_started_by_audit"] is False


def test_cli_fails_closed_without_allow_incomplete(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 8
    assert json.loads(output.read_text(encoding="utf-8"))["goal_achieved"] is False


def test_partial_learning_arguments_are_rejected(tmp_path: Path) -> None:
    fake_dataset = tmp_path / "dataset.npz"
    fake_dataset.write_bytes(b"not-a-dataset")
    try:
        MODULE._optional_learning_evidence(
            training_dataset=fake_dataset,
            bc_admission=None,
            bc_report=None,
            all79_report=None,
            learned_render_report=None,
        )
    except ValueError as error:
        assert "must be supplied together" in str(error)
    else:
        raise AssertionError("partial learning evidence was accepted")


def test_translates_windows_worktree_path_for_wsl_git() -> None:
    assert (
        MODULE._windows_to_wsl_path(r"G:\wSpace\cinebotRL-two-wheel-riser")
        == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    )


def _rollout_metrics(value: float) -> dict[str, float]:
    return {name: value for name in MODULE.ROLLOUT_METRICS}


def _valid_all79_report(policy_sha256: str) -> dict:
    rows = [
        {
            "case": case,
            "checks": {
                "learned_hard_gate": True,
                "teacher_hard_gate": True,
                "bounded_residual": True,
                "regression_position_error_p95_m": True,
            },
            "teacher": _rollout_metrics(1.0),
            "learned": _rollout_metrics(0.9),
            "learned_residual_action_abs_max": [0.5, 0.4, 0.3],
        }
        for case in range(1, 80)
    ]
    return {
        "schema": "cinebotrl_two_wheel_riser_residual_all79_gate_v1",
        "policy_sha256": policy_sha256,
        "cases": list(range(1, 80)),
        "case_count": 79,
        "maximum_regression_fraction": 0.05,
        "minimum_zero_improvement_fraction": None,
        "expected_tracking_profile": "riser_phase_consistent_v2",
        "means": {
            "teacher_position_p95_m": 1.0,
            "learned_position_p95_m": 0.9,
        },
        "aggregate_checks": {
            "all_case_checks": True,
            "learned_position_mean_within_teacher_budget": True,
        },
        "rows": rows,
        "passed": True,
        "ppo_authorized": False,
    }


def test_all79_validator_recomputes_every_case_and_aggregate() -> None:
    policy_sha = "a" * 64
    MODULE._validate_all79_report(
        _valid_all79_report(policy_sha),
        policy_sha256=policy_sha,
    )


def test_all79_validator_rejects_forged_or_regressed_rows() -> None:
    policy_sha = "a" * 64
    for mutation in ("policy", "case", "check", "regression", "mean"):
        report = _valid_all79_report(policy_sha)
        if mutation == "policy":
            report["policy_sha256"] = "b" * 64
        elif mutation == "case":
            report["rows"][10]["case"] = 10
        elif mutation == "check":
            report["rows"][10]["checks"]["learned_hard_gate"] = False
        elif mutation == "regression":
            report["rows"][10]["learned"]["position_error_max_m"] = 1.2
        else:
            report["means"]["learned_position_p95_m"] = 0.8
        try:
            MODULE._validate_all79_report(report, policy_sha256=policy_sha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"forged all-79 report accepted: {mutation}")


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _valid_render_report(
    tmp_path: Path,
) -> tuple[dict, Path, Path]:
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    all79 = tmp_path / "all79.json"
    all79.write_bytes(b"all79\n")
    videos = []
    for case in (1, 31, 73):
        path = tmp_path / f"case_{case:04d}.mp4"
        path.write_bytes(f"video-{case}".encode())
        videos.append(
            {
                "case": case,
                **_identity(path),
                "codec": "h264",
                "width": 1280,
                "height": 720,
                "fps": 25.0,
                "duration_s": 10.0,
            }
        )
    return {
        "schema": "cinebotrl_two_wheel_riser_learned_render_audit_v1",
        "policy": _identity(policy),
        "source_all79_report": _identity(all79),
        "cases": [1, 31, 73],
        "videos": videos,
        "visual_checks": {
            "robot_asset_intact": True,
            "riser_motion_visible": True,
            "camera_and_gimbal_visible": True,
            "no_detached_links": True,
        },
        "passed": True,
        "training_started": False,
        "ppo_authorized": False,
    }, policy, all79


def test_render_validator_binds_policy_all79_videos_and_visual_review(
    tmp_path: Path,
) -> None:
    report, policy, all79 = _valid_render_report(tmp_path)
    MODULE._validate_learned_render_report(
        report,
        report_directory=tmp_path,
        policy_path=policy,
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        all79_report_path=all79,
    )


def test_render_validator_rejects_forged_video_and_detached_robot(
    tmp_path: Path,
) -> None:
    for mutation in ("video", "detached"):
        directory = tmp_path / mutation
        directory.mkdir()
        report, policy, all79 = _valid_render_report(directory)
        if mutation == "video":
            report["videos"][0]["sha256"] = "0" * 64
        else:
            report["visual_checks"]["no_detached_links"] = False
        try:
            MODULE._validate_learned_render_report(
                report,
                report_directory=directory,
                policy_path=policy,
                policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                all79_report_path=all79,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"forged render report accepted: {mutation}")
