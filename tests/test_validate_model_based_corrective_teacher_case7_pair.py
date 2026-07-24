import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/validate_model_based_corrective_teacher_case7_pair.py"
)
WRAPPER = (
    ROOT / "scripts/two_wheel_balance/run_model_based_corrective_teacher_case7_pair.sh"
)
SPEC = importlib.util.spec_from_file_location("case7_pair_validator", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT = ROOT / MODULE.CONTRACT_RELATIVE_PATH


def _run(*args, cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=check
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _identity(repo: Path, relative: str) -> dict[str, str]:
    path = repo / relative
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob_sha1": _run("git", "hash-object", str(path), cwd=repo).stdout.strip(),
    }


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _closed_payload() -> dict[str, bool]:
    return {
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }


def _fixture_repo(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    paths = {
        "selection": "docs/selection.json",
        "readiness_audit": "docs/readiness.json",
        "readiness_auditor": "scripts/readiness.py",
        "profile_proposal": "docs/proposal.json",
        "profile_builder": "scripts/profile_builder.py",
        "case7_plan": "docs/case7_plan.npz",
        "perturbation_profile": "scripts/wrench.json",
        "corrective_profile": "scripts/corrective.json",
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
    _write(repo / paths["case7_plan"], "fixture-case7-plan\n")
    plan_sha = hashlib.sha256((repo / paths["case7_plan"]).read_bytes()).hexdigest()
    selection = {
        "schema": "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1",
        "passed": True,
        "selected_cases": MODULE.EXPECTED_SELECTED,
        "selected_rows": [
            {
                "case": 7,
                "plan_sha256": plan_sha,
                "selection_role": "same_seed_paired_canary_required",
            }
        ],
        "validation_cases": MODULE.EXPECTED_VALIDATION,
        "holdout_cases": MODULE.EXPECTED_HOLDOUT,
        "dataset_merge_authorized": False,
        **_closed_payload(),
    }
    readiness = {
        "schema": "cinebotrl_two_wheel_riser_case7_pair_readiness_cpu_v1",
        "case": 7,
        "passed": True,
        "selection_checks": {"healthy": True},
        "plan_checks": {"healthy": True},
        "gate_checks": {"healthy": True},
        "metric_checks": {"healthy": True},
        "inputs": {"plan": {"sha256": plan_sha}},
        "case_specific_profile_required": True,
        "case23_profile_reuse_authorized": False,
        "case6_profile_reuse_authorized": False,
        "case2_profile_reuse_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        **_closed_payload(),
    }
    corrective = {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": 7,
        "maximum_residuals": [
            0.019165321461451848,
            0.010077209250079967,
            0.0012628383956008627,
        ],
        "maximum_slew_rates": [
            0.05475806131843385,
            0.028792026428799907,
            0.0036081097017167506,
        ],
    }
    wrench = {
        "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
        "case": 7,
        "start_phase_time_s": 2.851306269086607,
        "duration_steps": 20,
        "force_body_x_n": 20.0,
        "application_height_m": 0.5,
    }
    _write(repo / paths["selection"], json.dumps(selection))
    _write(repo / paths["readiness_audit"], json.dumps(readiness))
    _write(repo / paths["corrective_profile"], json.dumps(corrective))
    _write(repo / paths["perturbation_profile"], json.dumps(wrench))
    corrective_sha = hashlib.sha256(
        (repo / paths["corrective_profile"]).read_bytes()
    ).hexdigest()
    wrench_sha = hashlib.sha256(
        (repo / paths["perturbation_profile"]).read_bytes()
    ).hexdigest()
    proposal = {
        "schema": "cinebotrl_two_wheel_riser_case7_pair_profile_proposal_cpu_v1",
        "case": 7,
        "split": "train",
        "passed": True,
        "input_checks": {"healthy": True},
        "shape_checks": {"healthy": True},
        "formula_checks": {"healthy": True},
        "identities": {
            "plan": {"sha256": plan_sha},
            "corrective_profile": {"sha256": corrective_sha},
            "wrench_profile": {"sha256": wrench_sha},
        },
        "pair_profile_cpu_ready": True,
        "runtime_route_implemented": False,
        "authorization_token_issued": False,
        "projection_envelope": {
            "label_contract": "effective_post_supervisor_residual_only",
            "effective_normalized_action_abs_max": 0.38330642922903724,
            "full_plan": {
                "directions": {
                    "negative": {
                        "command_clipped_transition_count": [0, 0, 4]
                    }
                }
            },
            "pulse_window": {"all_requested_commands_unclipped": True},
        },
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        **_closed_payload(),
    }
    _write(repo / paths["profile_proposal"], json.dumps(proposal))
    for name, relative in paths.items():
        if name not in {
            "selection",
            "readiness_audit",
            "profile_proposal",
            "case7_plan",
            "perturbation_profile",
            "corrective_profile",
        }:
            _write(repo / relative, f"fixture:{name}\n")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "reviewed parent", cwd=repo)
    parent = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    monkeypatch.setattr(MODULE, "REVIEWED_PARENT", parent)
    identities = {name: _identity(repo, relative) for name, relative in paths.items()}
    contract = {
        "schema": MODULE.SCHEMA,
        "reviewed_parent_commit": parent,
        "case": 7,
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
        "runtime_route_contract_ready": True,
        "execution_route_complete": True,
        "dataset_creation_authorized": False,
        **_closed_payload(),
        "runtime_authorized": True,
        "gpu_launch_authorized": True,
        "authorization_token_issued": True,
        "runtime_authorization_token_sha256": (
            MODULE.EXPECTED_AUTHORIZATION_SHA256
        ),
    }
    contract_path = repo / MODULE.CONTRACT_RELATIVE_PATH
    _write(contract_path, json.dumps(contract))
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "contract", cwd=repo)
    _run("git", "init", "--bare", str(remote))
    _run("git", "remote", "add", "origin", str(remote), cwd=repo)
    _run("git", "push", "-u", "origin", "main", cwd=repo)
    return repo, contract_path


def test_validator_accepts_clean_pushed_cpu_only_contract(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["passed"] is True
    assert result["cpu_contract_ready"] is True
    assert result["execution_route_complete"] is True
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False


def test_validator_accepts_only_matching_external_mode_0600_token(
    tmp_path, monkeypatch
) -> None:
    token = tmp_path / "authorization"
    token.write_text("fixture-token\n", encoding="utf-8")
    token.chmod(0o600)
    if os.name == "nt":
        original_token_checks = MODULE.token_checks

        def windows_fixture_token_checks(path, expected_sha256, repo):
            checks = original_token_checks(path, expected_sha256, repo)
            checks["authorization_mode_0600"] = True
            return checks

        monkeypatch.setattr(
            MODULE, "token_checks", windows_fixture_token_checks
        )
    token_sha = hashlib.sha256(token.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "EXPECTED_AUTHORIZATION_SHA256", token_sha)
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    result = MODULE.validate(
        contract,
        repo,
        namespace=MODULE.NAMESPACE,
        authorization_file=token,
    )
    assert result["passed"] is True
    assert all(result["authorization_checks"].values())
    assert result["authorization_consumed_before_isaac"] is True
    assert result["runtime_authorized"] is True
    assert result["gpu_launch_authorized"] is True
    assert result["label_capture_authorized"] is False
    assert result["dataset_creation_authorized"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False


def test_validator_rejects_bad_token_mode_or_hash(
    tmp_path, monkeypatch
) -> None:
    token = tmp_path / "authorization"
    token.write_text("fixture-token\n", encoding="utf-8")
    token_sha = hashlib.sha256(token.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "EXPECTED_AUTHORIZATION_SHA256", token_sha)
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    token.chmod(0o644)
    bad_mode = MODULE.validate(
        contract,
        repo,
        namespace=MODULE.NAMESPACE,
        authorization_file=token,
    )
    assert bad_mode["authorization_checks"]["authorization_mode_0600"] is False
    assert bad_mode["runtime_authorized"] is False
    token.chmod(0o600)
    token.write_text("wrong-token\n", encoding="utf-8")
    bad_hash = MODULE.validate(
        contract,
        repo,
        namespace=MODULE.NAMESPACE,
        authorization_file=token,
    )
    assert bad_hash["authorization_checks"]["authorization_hash_matches"] is False
    assert bad_hash["runtime_authorized"] is False


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
    payload["identities"]["case7_plan"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["all_identity_hashes_match"] is False
    assert result["passed"] is False


def test_validator_rejects_projection_contract_drift(
    tmp_path, monkeypatch
) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    proposal_path = repo / "docs/proposal.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["projection_envelope"]["pulse_window"][
        "all_requested_commands_unclipped"
    ] = False
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["all_identity_hashes_match"] is False
    assert result["checks"]["profile_proposal_contract"] is False
    assert result["passed"] is False


def test_validator_rejects_wrong_case7_pulse_phase(
    tmp_path, monkeypatch
) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    wrench_path = repo / "scripts/wrench.json"
    wrench = json.loads(wrench_path.read_text(encoding="utf-8"))
    wrench["start_phase_time_s"] = 17.185168504529333
    wrench_path.write_text(json.dumps(wrench), encoding="utf-8")
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["all_identity_hashes_match"] is False
    assert result["checks"]["perturbation_contract"] is False
    assert result["passed"] is False


def test_validator_rejects_dirty_or_unpushed_head(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture_repo(tmp_path, monkeypatch)
    _write(repo / "tracked.txt", "new\n")
    _run("git", "add", "tracked.txt", cwd=repo)
    _run("git", "commit", "-m", "not pushed", cwd=repo)
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["checks"]["head_matches_upstream"] is False
    assert result["passed"] is False


def test_wrapper_execute_fails_before_python_or_isaac() -> None:
    result = _run("bash", str(WRAPPER), "--execute", check=False)
    assert result.returncode == 4
    payload = json.loads(result.stderr)
    assert payload["reason"] == "authorization_file_missing"
    assert payload["python_started"] is False
    assert payload["isaac_started"] is False
    assert payload["runtime_started"] is False


def test_wrapper_rejects_wrong_supplied_token_before_isaac(tmp_path) -> None:
    token = tmp_path / "token"
    token.write_text("not authorized\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        env={
            **os.environ,
            "RISER_CORRECTIVE_CASE7_AUTHORIZATION_FILE": str(token),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert token.exists()


def test_wrapper_rejects_environment_override_before_python() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--preflight"],
        env={"RISER_ROOT": "/tmp/forged"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "conflicting_environment_override:RISER_ROOT" in result.stderr


def test_wrapper_has_one_pinned_authorized_runtime_route() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    pinned_token = source.index(
        f'readonly AUTHORIZATION_SHA256="{MODULE.EXPECTED_AUTHORIZATION_SHA256}"'
    )
    execute_reject = source.index('reject "runtime_authorization_not_issued" 4')
    python_start = source.index('python3 "$VALIDATOR"')
    playback_start = source.index("timeout --signal=TERM --kill-after=30s 600")
    assert pinned_token < execute_reject < python_start < playback_start
    assert "ISAAC_PYTHON" in source
    assert "smoke_riser_reference_playback.py" in source
    assert "summarize_model_based_corrective_teacher_case7_pair.py" in source
    assert "--corrective-teacher-profile" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert "--corrective-teacher-capture-dir" not in source


def test_committed_contract_authorizes_only_one_paired_canary() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert set(payload["identities"]) == MODULE.REQUIRED_IDENTITIES
    assert payload["cpu_preflight_ready"] is True
    assert payload["runtime_route_contract_ready"] is True
    assert payload["execution_route_complete"] is True
    assert payload["runtime_authorization_token_sha256"] == (
        MODULE.EXPECTED_AUTHORIZATION_SHA256
    )
    assert payload["runtime_authorized"] is True
    assert payload["gpu_launch_authorized"] is True
    assert payload["authorization_token_issued"] is True
    for field in (
        "label_capture_authorized",
        "dataset_creation_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert payload[field] is False


def test_committed_contract_pins_current_identity_bytes() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for identity in payload["identities"].values():
        path = ROOT / identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
        assert _git_blob_sha1(path) == identity["git_blob_sha1"]
