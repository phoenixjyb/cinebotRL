import hashlib
import os
from pathlib import Path

import pytest

from scripts.two_wheel_balance.validate_riser_case78_runtime_authorization import (
    AUTHORIZATION,
    token_checks,
)


@pytest.mark.skipif(
    os.name == "nt",
    reason="runtime authorization token mode is enforced by the WSL wrapper",
)
def test_exact_mode_0600_token_passes(tmp_path: Path) -> None:
    token = tmp_path / "authorization"
    token.write_bytes((AUTHORIZATION + "\n").encode())
    os.chmod(token, 0o600)
    checks = token_checks(
        token,
        hashlib.sha256((AUTHORIZATION + "\n").encode()).hexdigest(),
    )
    assert all(checks.values())


def test_forged_token_fails_closed(tmp_path: Path) -> None:
    token = tmp_path / "authorization"
    token.write_bytes(b"FORGED\n")
    os.chmod(token, 0o600)
    checks = token_checks(
        token,
        hashlib.sha256((AUTHORIZATION + "\n").encode()).hexdigest(),
    )
    assert not checks["authorization_hash_matches"]
    assert not checks["authorization_content_matches"]
