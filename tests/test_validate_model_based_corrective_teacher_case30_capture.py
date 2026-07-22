import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = ROOT / "scripts/two_wheel_balance/validate_model_based_corrective_teacher_case30_capture.py"
WRAPPER = ROOT / "scripts/two_wheel_balance/run_model_based_corrective_teacher_case30_capture.sh"
SPEC = importlib.util.spec_from_file_location("corrective_capture_validator", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _run(*args, cwd: Path):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    paths = {name: f"fixtures/{name}.json" for name in MODULE.REQUIRED_IDENTITIES}
    paired = {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_case30_pair_final_v1",
        "case": 30,
        "split": "train",
        "passed": True,
        "corrective_target_admission_passed": True,
        "label_capture_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "candidate_metrics": {"plan_sha256": "a" * 64},
        "paired_admission": {
            "position_p95_absolute_improvement_m": 0.006,
            "position_p95_relative_improvement": 0.04,
        },
    }
    profile = {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": 30,
        "maximum_residuals": [0.045, 0.045, 0.018],
    }
    for name, relative in paths.items():
        value = paired if name == "paired_final_status" else profile if name == "corrective_profile" else f"fixture:{name}\n"
        _write(repo / relative, value)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "reviewed parent", cwd=repo)
    parent = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    monkeypatch.setattr(MODULE, "REVIEWED_PARENT", parent)
    identities = {}
    for name, relative in paths.items():
        path = repo / relative
        identities[name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if name in MODULE.TRACKED_IDENTITIES:
            identities[name]["git_blob_sha1"] = _run(
                "git", "hash-object", str(path), cwd=repo
            ).stdout.strip()
    identities["case30_plan"]["sha256"] = "a" * 64
    # Keep the fixture bytes consistent with the deliberately simple plan identity.
    _write(repo / paths["case30_plan"], "plan")
    identities["case30_plan"]["sha256"] = hashlib.sha256(
        (repo / paths["case30_plan"]).read_bytes()
    ).hexdigest()
    paired["candidate_metrics"]["plan_sha256"] = identities["case30_plan"]["sha256"]
    _write(repo / paths["paired_final_status"], paired)
    identities["paired_final_status"]["sha256"] = hashlib.sha256(
        (repo / paths["paired_final_status"]).read_bytes()
    ).hexdigest()
    identities["paired_final_status"]["git_blob_sha1"] = _run(
        "git", "hash-object", str(repo / paths["paired_final_status"]), cwd=repo
    ).stdout.strip()
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "evidence", cwd=repo)
    contract = {
        "schema": MODULE.SCHEMA,
        "reviewed_parent_commit": parent,
        "case": 30,
        "split": "train",
        "namespace": MODULE.NAMESPACE,
        "residual_action_scales": MODULE.EXPECTED_SCALES,
        "capture_schema_contract": MODULE.EXPECTED_CAPTURE,
        "identities": identities,
        "holdout_cases": MODULE.EXPECTED_HOLDOUT,
        "holdout_opened": False,
        "validation_cases_opened": [],
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
    _write(contract_path, contract)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "contract", cwd=repo)
    _run("git", "init", "--bare", str(remote), cwd=repo)
    _run("git", "remote", "add", "origin", str(remote), cwd=repo)
    _run("git", "push", "-u", "origin", "main", cwd=repo)
    return repo, contract_path


def test_clean_pushed_contract_is_cpu_ready_but_runtime_closed(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture(tmp_path, monkeypatch)
    result = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert result["passed"] is True
    assert result["runtime_authorized"] is False
    assert result["label_capture_authorized"] is False


def test_validator_rejects_dirty_or_diverged_repo(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture(tmp_path, monkeypatch)
    (repo / "dirty.txt").write_text("untracked is intentionally ignored", encoding="utf-8")
    tracked = next(repo.glob("fixtures/*.json"))
    tracked.write_text("dirty", encoding="utf-8")
    dirty = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert dirty["checks"]["tracked_worktree_clean"] is False
    _run("git", "restore", str(tracked), cwd=repo)
    _write(repo / "new.txt", "new")
    _run("git", "add", "new.txt", cwd=repo)
    _run("git", "commit", "-m", "unpushed", cwd=repo)
    diverged = MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)
    assert diverged["checks"]["head_matches_upstream"] is False


def test_validator_rejects_alternate_contract_and_weak_pair(tmp_path, monkeypatch) -> None:
    repo, contract = _fixture(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(contract.read_bytes())
    assert MODULE.validate(alternate, repo, namespace=MODULE.NAMESPACE)["passed"] is False
    pair = repo / "fixtures/paired_final_status.json"
    payload = json.loads(pair.read_text(encoding="utf-8"))
    payload["paired_admission"]["position_p95_absolute_improvement_m"] = 0.001
    _write(pair, payload)
    assert MODULE.validate(contract, repo, namespace=MODULE.NAMESPACE)["passed"] is False


def test_wrapper_has_no_runtime_route_or_authorization() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "runtime_authorization_not_issued" in source
    assert "smoke_riser_reference_playback.py" not in source
    result = subprocess.run(["bash", str(WRAPPER), "--execute"], capture_output=True, text=True)
    assert result.returncode == 4
    assert "runtime_authorization_not_issued" in result.stderr
