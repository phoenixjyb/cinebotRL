import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/validate_model_based_corrective_teacher_case23_pair.py"
)
WRAPPER = (
    ROOT / "scripts/two_wheel_balance/run_model_based_corrective_teacher_case23_pair.sh"
)
SPEC = importlib.util.spec_from_file_location("case23_pair_validator", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT = ROOT / MODULE.CONTRACT_RELATIVE_PATH


def _run(*args, cwd: Path | None = None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _identity(repo: Path, relative: str, *, tracked: bool) -> dict[str, str]:
    path = repo / relative
    result = {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if tracked:
        result["git_blob_sha1"] = _run(
            "git", "hash-object", str(path), cwd=repo
        ).stdout.strip()
    return result


def _fixture_repo(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    paths = {
        "proposal": "docs/proposal.json",
        "selection": "docs/selection.json",
        "readiness_audit": "docs/readiness.json",
        "readiness_auditor": "scripts/readiness.py",
        "case23_plan": "artifacts/case23_plan.npz",
        "perturbation_profile": "scripts/wrench.json",
        "corrective_profile": "scripts/corrective_profile.json",
        "lqr_gains": "docs/gains.json",
        "robot_build_audit": "assets/build_audit.json",
        "robot_urdf": "assets/robot.urdf",
        "robot_usd": "assets/robot.usd",
        "playback": "scripts/playback.py",
        "corrective_teacher_runtime": "src/corrective.py",
        "perturbation_runtime": "src/perturbation.py",
        "preflight_wrapper": "scripts/wrapper.sh",
        "contract_validator": "scripts/validator.py",
        "paired_finalizer": "scripts/finalizer.py",
    }
    plan_sha = ""
    _write(repo / paths["case23_plan"], "fixture:case23-plan\n")
    plan_sha = hashlib.sha256((repo / paths["case23_plan"]).read_bytes()).hexdigest()
    _write(
        repo / paths["proposal"],
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_model_based_corrective_"
                    "case23_pair_proposal_v1"
                ),
                "case": 23,
                "split": "train",
                "passed": True,
                "paired_admission_contract": (
                    "same_seed_paired_dynamic_improvement_before_label_capture_v1"
                ),
                "identities": {
                    "plan": {"sha256": plan_sha},
                    "corrective_profile": {"sha256": "PROFILE_SHA"},
                },
                "runtime_route_implemented": False,
                "authorization_token_issued": False,
                "runtime_authorized": False,
                "gpu_launch_authorized": False,
                "label_capture_authorized": False,
                "dataset_created": False,
                "dataset_merge_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
            }
        ),
    )
    _write(
        repo / paths["selection"],
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_model_based_pair_"
                    "tranche_selection_v1"
                ),
                "passed": True,
                "selected_cases": MODULE.EXPECTED_SELECTED,
                "selected_rows": [
                    {
                        "case": 23,
                        "plan_sha256": plan_sha,
                        "selection_role": "same_seed_paired_canary_required",
                    }
                ],
                "validation_cases": MODULE.EXPECTED_VALIDATION,
                "holdout_cases": MODULE.EXPECTED_HOLDOUT,
                "runtime_authorized": False,
                "gpu_launch_authorized": False,
                "label_capture_authorized": False,
                "dataset_merge_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
            }
        ),
    )
    _write(
        repo / paths["readiness_audit"],
        json.dumps(
            {
                "schema": (
                    "cinebotrl_two_wheel_riser_case23_pair_readiness_audit_v1"
                ),
                "case": 23,
                "passed": True,
                "checks": {"healthy": True},
                "plan_sha256": plan_sha,
                "decision": "recommend_exactly_one_bounded_case23_pair_canary",
                "runtime_authorized": False,
                "gpu_launch_authorized": False,
                "label_capture_authorized": False,
                "dataset_created": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
                "valid_for_training": False,
            }
        ),
    )
    corrective = {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": 23,
        "maximum_residuals": [0.045, 0.045, 0.018],
    }
    _write(repo / paths["corrective_profile"], json.dumps(corrective))
    profile_sha = hashlib.sha256(
        (repo / paths["corrective_profile"]).read_bytes()
    ).hexdigest()
    proposal_path = repo / paths["proposal"]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["identities"]["corrective_profile"]["sha256"] = profile_sha
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    wrench = {
        "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
        "case": 23,
        "start_phase_time_s": 4.9648469999999945,
        "duration_steps": 20,
        "force_body_x_n": 20.0,
        "application_height_m": 0.5,
    }
    _write(repo / paths["perturbation_profile"], json.dumps(wrench))
    for name, relative in paths.items():
        if name not in {
            "proposal",
            "selection",
            "readiness_audit",
            "case23_plan",
            "corrective_profile",
            "perturbation_profile",
        }:
            _write(repo / relative, f"fixture:{name}\n")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "reviewed parent", cwd=repo)
    parent = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    monkeypatch.setattr(MODULE, "REVIEWED_PARENT", parent)
    identities = {
        name: _identity(repo, relative, tracked=name in MODULE.TRACKED_IDENTITIES)
        for name, relative in paths.items()
    }
    contract = {
        "schema": MODULE.SCHEMA,
        "reviewed_parent_commit": parent,
        "case": 23,
        "split": "train",
        "namespace": MODULE.NAMESPACE,
        "residual_action_scales": MODULE.EXPECTED_RESIDUAL_SCALES,
        "controller_arguments": MODULE.EXPECTED_CONTROLLER_ARGUMENTS,
        "unchanged_dynamic_gate_thresholds": MODULE.EXPECTED_DYNAMIC_THRESHOLDS,
        "paired_experiment_contract": MODULE.EXPECTED_PAIR_CONTRACT,
        "identities": identities,
        "validation_cases": MODULE.EXPECTED_VALIDATION,
        "validation_opened": False,
        "holdout_cases": MODULE.EXPECTED_HOLDOUT,
        "holdout_opened": False,
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "authorization_token_issued": False,
        "runtime_authorization_token_sha256": "",
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    contract_path = repo / MODULE.CONTRACT_RELATIVE_PATH
    _write(contract_path, json.dumps(contract))
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "contract", cwd=repo)
    _run("git", "init", "--bare", str(remote))
    _run("git", "remote", "add", "origin", str(remote), cwd=repo)
    _run("git", "push", "-u", "origin", "main", cwd=repo)
    return repo, contract_path


def test_validator_accepts_clean_pushed_closed_contract(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["passed"] is True
    assert result["cpu_contract_ready"] is True
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False


def test_validator_rejects_any_authorization_file(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    token = tmp_path / "forged-token"
    token.write_text("not authorized\n", encoding="utf-8")
    result = MODULE.validate(
        contract, repo, namespace=MODULE.NAMESPACE, authorization_file=token
    )
    assert result["checks"]["no_authorization_file"] is False
    assert result["passed"] is False
    assert result["runtime_authorized"] is False


def test_validator_rejects_alternate_contract_path(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(contract.read_bytes())
    result = MODULE.validate(alternate, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["canonical_contract_path"] is False
    assert result["passed"] is False


def test_validator_rejects_forged_identity(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["identities"]["case23_plan"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["all_identity_hashes_match"] is False
    assert result["passed"] is False


def test_wrapper_rejects_environment_override_before_python() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--preflight"],
        env={"RISER_ROOT": "/tmp/forged"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "conflicting_environment_override:RISER_ROOT" in result.stderr


def test_wrapper_execute_is_unconditionally_unauthorized() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"], capture_output=True, text=True
    )
    assert result.returncode == 4
    assert "runtime_authorization_not_issued" in result.stderr


def test_wrapper_keeps_runtime_behind_empty_authorization_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    gate_index = source.index('readonly AUTHORIZATION_SHA256=""')
    execute_check = source.index('[[ -n "$AUTHORIZATION_SHA256" ]]')
    playback_index = source.index('timeout --signal=TERM --kill-after=30s 600')
    assert gate_index < execute_check < playback_index
    assert "--corrective-teacher-profile" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source


def test_committed_contract_keeps_runtime_and_learning_closed() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert set(payload["identities"]) == MODULE.REQUIRED_IDENTITIES
    assert payload["runtime_authorization_token_sha256"] == ""
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "authorization_token_issued",
        "label_capture_authorized",
        "dataset_creation_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert payload[field] is False
