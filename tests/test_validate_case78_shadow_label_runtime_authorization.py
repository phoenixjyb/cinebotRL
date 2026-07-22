import hashlib
import os
from pathlib import Path

import pytest

from scripts.two_wheel_balance import (
    validate_case78_shadow_label_runtime_authorization as module,
)


@pytest.mark.skipif(os.name == "nt", reason="POSIX token mode is enforced in WSL")
def test_shadow_runtime_token_requires_exact_content_mode_and_hash(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text(module.AUTHORIZATION + "\n", encoding="utf-8")
    token.chmod(0o600)
    expected = hashlib.sha256(token.read_bytes()).hexdigest()
    assert all(module.token_checks(token, expected).values())
    token.chmod(0o644)
    assert not module.token_checks(token, expected)["authorization_mode_0600"]


@pytest.mark.skipif(os.name == "nt", reason="symlink contract is enforced in WSL")
def test_shadow_runtime_token_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text(module.AUTHORIZATION + "\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    checks = module.token_checks(link, hashlib.sha256(link.read_bytes()).hexdigest())
    assert checks["authorization_not_symlink"] is False


def test_shadow_runtime_identity_scope_is_exact() -> None:
    assert module.REQUIRED_IDENTITIES == {
        "cpu_contract",
        "runtime_summarizer",
        "runtime_validator",
        "runtime_wrapper",
    }
    assert module.NAMESPACE == (
        "20260722_case78_shadow_label_measurement_v1_exclusive"
    )
