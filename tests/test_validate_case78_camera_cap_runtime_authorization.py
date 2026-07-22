import json
import hashlib
from pathlib import Path

from scripts.two_wheel_balance import (
    validate_case78_camera_cap_runtime_authorization as module,
)


def test_runtime_token_requires_exact_content_mode_and_hash(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text(module.AUTHORIZATION + "\n", encoding="utf-8")
    token.chmod(0o600)
    expected = hashlib.sha256(token.read_bytes()).hexdigest()

    assert all(module.token_checks(token, expected).values())

    token.chmod(0o644)
    assert not module.token_checks(token, expected)["authorization_mode_0600"]


def test_runtime_token_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text(module.AUTHORIZATION + "\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)

    checks = module.token_checks(link, hashlib.sha256(link.read_bytes()).hexdigest())
    assert checks["authorization_not_symlink"] is False


def test_runtime_implementation_pin_is_resolved() -> None:
    assert module.RUNTIME_IMPLEMENTATION_COMMIT == (
        "8c00406b0a726f1e785d303986a44a3323476478"
    )


def test_checked_in_runtime_contract_has_exact_scope_and_identities() -> None:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/two_wheel_balance/case78_camera_cap_runtime_authorization_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["schema"] == module.SCHEMA
    assert contract["case"] == 78
    assert contract["current_split"] == "unused"
    assert contract["reviewed_cpu_commit"] == module.REVIEWED_CPU_COMMIT
    assert (
        contract["runtime_implementation_commit"]
        == module.RUNTIME_IMPLEMENTATION_COMMIT
    )
    assert contract["namespace"] == module.NAMESPACE
    assert set(contract["identities"]) == module.REQUIRED_IDENTITIES
    assert contract["one_case_only"] is True
    assert contract["maximum_runtime_seconds"] == 5400
    assert contract["camera_recovery_governor_required"] is False
    assert contract["maximum_camera_lever_arm_correction_m"] == 0.10
    assert contract["split_change_authorized"] is False
    assert contract["dataset_creation_authorized"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["runtime_authorization_token_sha256"] == hashlib.sha256(
        (module.AUTHORIZATION + "\n").encode()
    ).hexdigest()
