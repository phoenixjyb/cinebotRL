"""Robot asset helpers and configuration providers."""

from pathlib import Path


def assets_root() -> Path:
    """Return the absolute path to the `assets_own` directory."""
    return Path(__file__).resolve().parents[3] / "assets_own"
