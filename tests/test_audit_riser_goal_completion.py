import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


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
    assert report["schema"] == MODULE.GOAL_COMPLETION_AUDIT_SCHEMA
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
    readiness = report["pre_training_readiness"]
    assert readiness == {
        "architecture_contract_passed": True,
        "policy_architecture": (
            "model_based_shared_encoder_zero_initialized_residual_v1"
        ),
        "observation_dimension": 65,
        "base_observation_dimension": 26,
        "lookahead_horizon_count": 3,
        "lookahead_channel_count_per_horizon": 13,
        "action_dimension": 3,
        "zero_initialize_action_head": True,
        "corrective_case_datasets_available": 1,
        "corrective_training_corpus_cases_available": 0,
        "minimum_train_cases": 4,
        "minimum_validation_cases": 2,
        "next_case": 23,
        "next_operation": "case23_v4_cpu_conversion",
        "next_operation_authorized": False,
        "bc_authorized": False,
        "training_authorized": False,
        "ppo_authorized": False,
        "runtime_authorized": False,
        "ready_for_bc_execution": False,
    }
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


def test_goal_audit_rejects_residual_architecture_drift() -> None:
    payloads, _ = _current_inputs()
    goal = json.loads(json.dumps(payloads["goal"]))
    residual = goal["current_stage"]["status_refresh_20260723"][
        "residual_dnn_admission_contract"
    ]
    residual["observation_dimension"] = 26
    report = _build_current(goal=goal)
    readiness = report["pre_training_readiness"]
    assert readiness["architecture_contract_passed"] is False
    assert readiness["ready_for_bc_execution"] is False


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
            learned_all79_admission=None,
            learned_all79_preflight_receipt=None,
            learned_plan_manifest=None,
            learned_source_manifest=None,
            learned_lqr_gains=None,
            learned_robot_build_audit=None,
            learned_robot_usd=None,
            learned_drive_profile_selection=None,
            validation_gate_report=None,
            holdout_gate_report=None,
            all79_report=None,
            learned_render_report=None,
        )
    except ValueError as error:
        assert "must be supplied together" in str(error)
    else:
        raise AssertionError("partial learning evidence was accepted")


def test_all79_report_requires_separate_rollout_admission_chain(
    tmp_path: Path,
) -> None:
    all79 = tmp_path / "all79.json"
    all79.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="requires BC, admission, validation, and holdout evidence",
    ):
        MODULE._optional_learning_evidence(
            training_dataset=None,
            bc_admission=None,
            bc_report=None,
            learned_all79_admission=None,
            learned_all79_preflight_receipt=None,
            learned_plan_manifest=None,
            learned_source_manifest=None,
            learned_lqr_gains=None,
            learned_robot_build_audit=None,
            learned_robot_usd=None,
            learned_drive_profile_selection=None,
            validation_gate_report=None,
            holdout_gate_report=None,
            all79_report=all79,
            learned_render_report=None,
        )


def test_translates_windows_worktree_path_for_wsl_git() -> None:
    assert (
        MODULE._windows_to_wsl_path(r"G:\wSpace\cinebotRL-two-wheel-riser")
        == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    )


def _rollout_metrics(value: float) -> dict[str, float]:
    return {name: value for name in MODULE.ROLLOUT_METRICS}


def _valid_all79_report(
    tmp_path: Path,
    policy_sha256: str,
) -> tuple[dict, dict[str, Path | str]]:
    execution_commit = "a" * 40
    admission = tmp_path / "admission.json"
    admission.write_text("{}\n", encoding="utf-8")
    plan_manifest = tmp_path / "plan_manifest.json"
    plan_manifest.write_text("{}\n", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_model_based_learned_all79_"
                    "preflight_v1"
                ),
                "passed": True,
                "execution_commit": execution_commit,
                "head": execution_commit,
                "runtime_started": False,
                "dataset_created": False,
                "residual_capture_started": False,
                "bc_started": False,
                "ppo_started": False,
            }
        ),
        encoding="utf-8",
    )

    def artifact(path: Path) -> dict[str, str]:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

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
            "teacher_rollout": artifact(
                _write_rollout_identity(tmp_path, case, "teacher")
            ),
            "learned_rollout": artifact(
                _write_rollout_identity(tmp_path, case, "learned")
            ),
        }
        for case in range(1, 80)
    ]
    report = {
        "schema": "cinebotrl_two_wheel_riser_residual_all79_gate_v1",
        "policy_sha256": policy_sha256,
        "cases": list(range(1, 80)),
        "case_count": 79,
        "maximum_regression_fraction": 0.05,
        "minimum_zero_improvement_fraction": None,
        "expected_tracking_profile": (
            "riser_recovery_direction_v4_camera_lever_arm_v1"
        ),
        "policy_command_contract": (
            "model_based_planner_plus_bounded_policy_residual_v1"
        ),
        "residual_action_scales": [0.05, 0.05, 0.02],
        "rollout_admission": artifact(admission),
        "preflight_receipt": artifact(preflight),
        "plan_manifest": artifact(plan_manifest),
        "execution_commit": execution_commit,
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
    return report, {
        "admission": admission,
        "preflight": preflight,
        "plan_manifest": plan_manifest,
        "execution_commit": execution_commit,
    }


def _write_rollout_identity(root: Path, case: int, role: str) -> Path:
    path = root / f"{role}_{case:04d}.json"
    path.write_text(f"{role}-{case}", encoding="utf-8")
    return path


def test_all79_validator_recomputes_every_case_and_aggregate(
    tmp_path: Path,
) -> None:
    policy_sha = "a" * 64
    report, evidence = _valid_all79_report(tmp_path, policy_sha)
    MODULE._validate_all79_report(
        report,
        policy_sha256=policy_sha,
        report_directory=tmp_path,
        admission_path=evidence["admission"],
        preflight_path=evidence["preflight"],
        plan_manifest_path=evidence["plan_manifest"],
        execution_commit=evidence["execution_commit"],
    )


def test_all79_validator_rejects_forged_or_regressed_rows(
    tmp_path: Path,
) -> None:
    policy_sha = "a" * 64
    for mutation in (
        "policy",
        "case",
        "check",
        "regression",
        "mean",
        "rollout_hash",
        "preflight_hash",
    ):
        mutation_root = tmp_path / mutation
        mutation_root.mkdir()
        report, evidence = _valid_all79_report(mutation_root, policy_sha)
        if mutation == "policy":
            report["policy_sha256"] = "b" * 64
        elif mutation == "case":
            report["rows"][10]["case"] = 10
        elif mutation == "check":
            report["rows"][10]["checks"]["learned_hard_gate"] = False
        elif mutation == "regression":
            report["rows"][10]["learned"]["position_error_max_m"] = 1.2
        elif mutation == "rollout_hash":
            report["rows"][10]["teacher_rollout"]["sha256"] = "0" * 64
        elif mutation == "preflight_hash":
            report["preflight_receipt"]["sha256"] = "0" * 64
        else:
            report["means"]["learned_position_p95_m"] = 0.8
        try:
            MODULE._validate_all79_report(
                report,
                policy_sha256=policy_sha,
                report_directory=mutation_root,
                admission_path=evidence["admission"],
                preflight_path=evidence["preflight"],
                plan_manifest_path=evidence["plan_manifest"],
                execution_commit=evidence["execution_commit"],
            )
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
    rollouts = []
    for case in (1, 15, 31, 50, 73, 79):
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
        rollout = tmp_path / f"case_{case:04d}.json"
        rollout.write_text(
            json.dumps(
                {
                    "cases": [case],
                    "passed": True,
                    "trajectory_command_source": (
                        "model_based_planner_plus_torchscript_residual"
                    ),
                    "tracking_profile": (
                        "riser_recovery_direction_v4_camera_lever_arm_v1"
                    ),
                    "policy_command_base": "model_based_planner",
                    "residual_action_scales": [0.05, 0.05, 0.02],
                }
            ),
            encoding="utf-8",
        )
        rollouts.append({"case": case, **_identity(rollout)})
    admission = tmp_path / "render_admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_"
                    "model_based_learned_render_admission_v1"
                ),
                "cases": [1, 15, 31, 50, 73, 79],
                "all79_gate_passed": True,
                "render_evaluation_approved": True,
                "learned_render_authorized": True,
                "residual_capture_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "render_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_"
                    "model_based_learned_render_preflight_v1"
                ),
                "cases": [1, 15, 31, 50, 73, 79],
                "admission": _identity(admission),
                "passed": True,
                "runtime_started": False,
                "recording_started": False,
            }
        ),
        encoding="utf-8",
    )
    visual_checks = {
        "robot_asset_intact": True,
        "riser_motion_visible": True,
        "camera_and_gimbal_visible": True,
        "wheel_ground_contact_plausible": True,
        "no_detached_links": True,
        "no_abnormal_oscillation": True,
    }
    media = {
        "schema": "cinebotrl_two_wheel_riser_learned_render_media_manifest_v1",
        "policy": _identity(policy),
        "source_all79_report": _identity(all79),
        "admission": _identity(admission),
        "preflight": _identity(preflight),
        "cases": [1, 15, 31, 50, 73, 79],
        "rollout_gates": rollouts,
        "videos": videos,
        "media_checks": {
            str(case): {"passed": True}
            for case in (1, 15, 31, 50, 73, 79)
        },
        "manual_visual_review_required": True,
        "runtime_started": True,
        "recording_started": True,
        "training_started": False,
        "ppo_authorized": False,
        "passed": True,
    }
    media_path = tmp_path / "media.json"
    media_path.write_text(json.dumps(media), encoding="utf-8")
    review = {
        "schema": "cinebotrl_two_wheel_riser_learned_render_visual_review_v1",
        "cases": [1, 15, 31, 50, 73, 79],
        "videos": videos,
        "reviewer": "test-reviewer",
        "reviewed_at_utc": "2026-07-23T12:00:00Z",
        "visual_checks": visual_checks,
        "notes": "",
        "passed": True,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return {
        "schema": "cinebotrl_two_wheel_riser_learned_render_audit_v2",
        "policy": _identity(policy),
        "source_all79_report": _identity(all79),
        "render_admission": _identity(admission),
        "render_preflight": _identity(preflight),
        "media_manifest": _identity(media_path),
        "visual_review": _identity(review_path),
        "cases": [1, 15, 31, 50, 73, 79],
        "rollout_gates": rollouts,
        "videos": videos,
        "visual_checks": visual_checks,
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
    for mutation in ("video", "detached", "reviewer"):
        directory = tmp_path / mutation
        directory.mkdir()
        report, policy, all79 = _valid_render_report(directory)
        if mutation == "video":
            report["videos"][0]["sha256"] = "0" * 64
        elif mutation == "detached":
            report["visual_checks"]["no_detached_links"] = False
        else:
            review_path = directory / report["visual_review"]["path"]
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["reviewer"] = ""
            review_path.write_text(json.dumps(review), encoding="utf-8")
            report["visual_review"] = _identity(review_path)
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
