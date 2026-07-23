import importlib.util
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
    report = json.loads(output.read_text(encoding="utf-8"))
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
