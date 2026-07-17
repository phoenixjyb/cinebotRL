import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/validate_riser_case74_recovery_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("case74_contract_validator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
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

    tracked = repo / "runtime.py"
    tracked.write_text("VALUE = 1\n")
    source = repo / "source_manifest.json"
    source.write_text("source\n")
    plan = repo / "case_0074.npz"
    plan.write_bytes(b"plan")
    gains = repo / "gains.json"
    gains.write_text("gains\n")
    usd = repo / "robot.usd"
    usd.write_text("usd\n")
    portfolio = repo / "manifest.json"
    portfolio.write_text(
        json.dumps(
            {
                "source_manifest_sha256": _sha(source),
                "kinematic_accepted_cases": [74],
                "items": [{"case": 74, "plan_sha256": _sha(plan)}],
            }
        )
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "reviewed parent")
    reviewed_parent = _git(repo, "rev-parse", "HEAD")

    identities = {}
    for name, path in {
        "source_manifest": source,
        "portfolio_manifest": portfolio,
        "case74_plan": plan,
        "lqr_gains": gains,
        "robot_usd": usd,
        "runtime": tracked,
    }.items():
        identities[name] = {
            "path": str(path.relative_to(repo)),
            "sha256": _sha(path),
            "git_blob_sha1": _git(repo, "hash-object", str(path)),
        }
    contract = (
        repo
        / "scripts/two_wheel_balance/case74_recovery_v4_contract_v1.json"
    )
    contract.parent.mkdir(parents=True)
    contract.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_case74_recovery_v4_contract_v1",
                "case": 74,
                "reviewed_controller_parent_commit": reviewed_parent,
                "namespace": "fresh-case74",
                "tracking_profile": "riser_recovery_direction_v4",
                "recovery_error_range_m": [0.2, 0.4],
                "source_manifest_sha256": _sha(source),
                "runtime_authorization_token": None,
                "runtime_authorized": False,
                "gpu_launch_authorized": False,
                "residual_capture_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "identities": identities,
            }
        )
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "contract")
    _git(repo, "push", "-u", "origin", "main")
    return repo, contract, reviewed_parent


def _validate(module, repo: Path, contract: Path):
    return module.validate(contract, repo, namespace="fresh-case74")


def test_contract_resolves_upstream_and_keeps_runtime_closed(
    tmp_path: Path, monkeypatch
) -> None:
    repo, contract, reviewed_parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", reviewed_parent)
    admission = _validate(module, repo, contract)

    assert admission["identity_passed"]
    assert admission["runtime_commit"] == _git(repo, "rev-parse", "@{u}")
    assert admission["contract_git_blob_sha1"] == _git(
        repo, "rev-parse", f"HEAD:{module.CONTRACT_RELATIVE_PATH}"
    )
    assert not admission["runtime_authorized"]
    assert not admission["gate_c_execution_authorized"]
    assert not admission["valid_for_training"]


def test_contract_rejects_forged_upstream_or_parent(
    tmp_path: Path, monkeypatch
) -> None:
    repo, contract, reviewed_parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", reviewed_parent)

    (repo / "local_only.txt").write_text("not pushed\n")
    _git(repo, "add", "local_only.txt")
    _git(repo, "commit", "-m", "local only")
    admission = _validate(module, repo, contract)
    assert not admission["checks"]["head_matches_upstream"]

    _git(repo, "push")
    payload = json.loads(contract.read_text())
    payload["reviewed_controller_parent_commit"] = "0" * 40
    contract.write_text(json.dumps(payload))
    _git(repo, "add", str(contract.relative_to(repo)))
    _git(repo, "commit", "-m", "forged parent")
    _git(repo, "push")
    admission = _validate(module, repo, contract)
    assert not admission["checks"]["reviewed_parent_matches"]


def test_contract_rejects_alternate_path_or_hash_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    repo, contract, reviewed_parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", reviewed_parent)

    alternate = repo / "alternate-contract.json"
    alternate.write_bytes(contract.read_bytes())
    admission = _validate(module, repo, alternate)
    assert not admission["checks"]["canonical_contract_path"]
    assert not admission["checks"]["contract_blob_matches_head"]

    payload = json.loads(contract.read_text())
    payload["identities"]["lqr_gains"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(payload))
    _git(repo, "add", str(contract.relative_to(repo)))
    _git(repo, "commit", "-m", "bad hash")
    _git(repo, "push")
    admission = _validate(module, repo, contract)
    assert not admission["checks"]["all_identity_hashes_match"]


def test_contract_rejects_dirty_runtime_or_existing_namespace(
    tmp_path: Path, monkeypatch
) -> None:
    repo, contract, reviewed_parent = _fixture(tmp_path)
    module = _load_module()
    monkeypatch.setattr(module, "REVIEWED_PARENT", reviewed_parent)

    (repo / "runtime.py").write_text("VALUE = 2\n")
    admission = _validate(module, repo, contract)
    assert not admission["checks"]["tracked_worktree_clean"]
    assert not admission["checks"]["all_identity_hashes_match"]

    _git(repo, "restore", "runtime.py")
    (repo / "artifacts/two_wheel_riser/fresh-case74").mkdir(parents=True)
    admission = _validate(module, repo, contract)
    assert not admission["checks"]["namespace_is_fresh"]
