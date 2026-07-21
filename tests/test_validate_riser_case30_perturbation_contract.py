import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/validate_riser_case30_perturbation_contract.py"
WRAPPER = ROOT / "scripts/two_wheel_balance/run_riser_case30_perturbation_measurement.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("case30_contract_validator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    ).stdout.strip()


def _fixture(tmp_path: Path):
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(remote))

    files = {}
    for name in ("plan", "policy", "dataset", "gains", "robot", "runtime"):
        path = repo / f"{name}.bin"
        path.write_bytes(name.encode())
        files[name] = path
    profile = repo / "profile.json"
    profile.write_text(json.dumps({
        "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
        "case": 30,
        "start_phase_time_s": 15.666592937559889,
        "duration_steps": 20,
        "force_body_x_n": 20.0,
        "application_height_m": 0.5,
    }))
    proposal = repo / "proposal.json"
    proposal.write_text(json.dumps({
        "schema": "cinebotrl_two_wheel_riser_case30_perturbation_proposal_v1",
        "decision_status": "cpu_only_profile_not_runtime_authorization",
        "case": 30,
        "split": "train",
        "profile": {"sha256": _sha(profile)},
        "case30_plan": {"sha256": _sha(files["plan"])},
        "runtime_authorized": False,
        "dataset_created": False,
    }))
    files |= {"profile": profile, "proposal": proposal}
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "reviewed parent")
    parent = _git(repo, "rev-parse", "HEAD")

    identities = {
        name: {
            "path": str(path.relative_to(repo)),
            "sha256": _sha(path),
            "git_blob_sha1": _git(repo, "hash-object", str(path)),
        }
        for name, path in files.items()
    }
    identities["case30_plan"] = identities.pop("plan")
    contract = repo / "scripts/two_wheel_balance/case30_perturbation_measurement_contract_v1.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(json.dumps({
        "schema": "cinebotrl_two_wheel_riser_case30_perturbation_contract_v1",
        "case": 30,
        "split": "train",
        "reviewed_parent_commit": parent,
        "namespace": "fresh-case30",
        "profile_payload": json.loads(profile.read_text()),
        "controller_arguments": {
            "controller_wz_kp": 1.05,
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "residual_policy_device": "cuda",
        },
        "residual_action_scales": [0.35, 0.4, 0.1],
        "unchanged_dynamic_gate_thresholds": {
            "maximum_pitch_deg": 12.0,
            "maximum_position_p95_m": 0.15,
            "maximum_position_error_m": 0.25,
            "maximum_attitude_p95_deg": 5.0,
            "maximum_attitude_error_deg": 10.0,
            "maximum_riser_servo_error_m": 0.03,
            "maximum_proxy_servo_error_deg": 5.0,
            "maximum_internal_proxy_rate_deg_s": 360.0,
            "maximum_saturation_ratio": 0.2,
        },
        "measurement_contract": {
            "shadow_teacher_applied_to_commands": False,
            "perturbation_applied_to_planner_commands": False,
            "perturbation_applied_to_policy_actions": False,
            "dynamic_and_perturbation_outcomes_independent": True,
            "stop_after_case30": True,
            "maximum_runtime_seconds": 600,
        },
        "cpu_preflight_ready": True,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "identities": identities,
    }))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "contract")
    _git(repo, "push", "-u", "origin", "main")
    return repo, contract, parent


def _validate(module, repo: Path, contract: Path):
    return module.validate(contract, repo, namespace="fresh-case30")


def test_cpu_contract_passes_without_runtime_authorization(
    tmp_path, monkeypatch
) -> None:
    repo, contract, parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(module, "NAMESPACE", "fresh-case30")
    result = _validate(module, repo, contract)
    assert result["cpu_contract_ready"]
    assert result["passed"]
    assert not result["runtime_authorized"]
    assert not result["gpu_launch_authorized"]
    assert not result["measurement_authorized"]


def test_contract_rejects_unpushed_or_forged_parent(tmp_path, monkeypatch) -> None:
    repo, contract, parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(module, "NAMESPACE", "fresh-case30")
    (repo / "local.txt").write_text("local")
    _git(repo, "add", "local.txt")
    _git(repo, "commit", "-m", "local")
    assert not _validate(module, repo, contract)["checks"]["head_matches_upstream"]

    _git(repo, "push")
    payload = json.loads(contract.read_text())
    payload["reviewed_parent_commit"] = "0" * 40
    contract.write_text(json.dumps(payload))
    _git(repo, "add", str(contract.relative_to(repo)))
    _git(repo, "commit", "-m", "forged parent")
    _git(repo, "push")
    assert not _validate(module, repo, contract)["checks"]["reviewed_parent_matches"]


def test_contract_rejects_alternate_path_identity_or_namespace(
    tmp_path, monkeypatch
) -> None:
    repo, contract, parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(module, "NAMESPACE", "fresh-case30")
    alternate = repo / "alternate.json"
    alternate.write_bytes(contract.read_bytes())
    assert not _validate(module, repo, alternate)["checks"]["canonical_contract_path"]

    (repo / "policy.bin").write_bytes(b"drift")
    assert not _validate(module, repo, contract)["checks"]["all_identity_hashes_match"]
    _git(repo, "restore", "policy.bin")
    (repo / "artifacts/two_wheel_riser/fresh-case30").mkdir(parents=True)
    assert not _validate(module, repo, contract)["checks"]["namespace_is_fresh"]


def test_contract_rejects_runtime_or_training_authorization(tmp_path, monkeypatch) -> None:
    repo, contract, parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(module, "NAMESPACE", "fresh-case30")
    payload = json.loads(contract.read_text())
    payload["runtime_authorized"] = True
    payload["runtime_authorization_token_sha256"] = "0" * 64
    contract.write_text(json.dumps(payload))
    _git(repo, "add", str(contract.relative_to(repo)))
    _git(repo, "commit", "-m", "bad authorization")
    _git(repo, "push")
    result = _validate(module, repo, contract)
    assert not result["checks"]["cpu_preflight_only"]
    assert not result["checks"]["no_runtime_token"]
    assert not result["passed"]


def test_wrapper_source_has_no_executable_runtime_path_or_token() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "smoke_riser_reference_playback.py" not in source
    assert "AUTHORIZED_CASE30" not in source
    assert "mkdir" not in source


@pytest.mark.skipif(os.name == "nt", reason="wrapper executes under WSL bash")
def test_wrapper_rejects_execute_and_environment_override() -> None:
    execute = subprocess.run(
        ["bash", str(WRAPPER), "--execute"], capture_output=True, text=True
    )
    assert execute.returncode == 7
    rejection = json.loads(execute.stderr)
    assert rejection["reason"] == "runtime_authorization_not_issued"
    assert rejection["runtime_started"] is False

    env = os.environ.copy() | {"RISER_CASE30_GPU_AUTHORIZATION": "forged"}
    override = subprocess.run(
        ["bash", str(WRAPPER), "--preflight"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert override.returncode == 7
    assert "conflicting_environment_override" in override.stderr
