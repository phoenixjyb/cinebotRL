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


def test_runtime_implementation_pin_must_be_resolved_before_contract() -> None:
    assert module.RUNTIME_IMPLEMENTATION_COMMIT.endswith("_PENDING")
