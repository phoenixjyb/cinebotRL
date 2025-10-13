"""Project-specific extensions that sit on top of Isaac Lab."""

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]
