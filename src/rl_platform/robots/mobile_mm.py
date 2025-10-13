"""Configuration helpers for the mobile manipulator RL task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import assets_root


@dataclass(frozen=True)
class MobileManipulatorAssets:
    """Container describing the USD bundle generated from the URDF importer."""

    usd_path: Path
    config_dir: Path

    def validate(self) -> None:
        """Raise FileNotFoundError if expected assets are missing."""
        if not self.usd_path.is_file():
            raise FileNotFoundError(f"Missing USD file: {self.usd_path}")
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Missing configuration directory: {self.config_dir}")


def get_mobile_mm_assets() -> MobileManipulatorAssets:
    """Return paths to the mobile manipulator USD and supporting configuration."""
    usd_dir = assets_root() / "usd"
    return MobileManipulatorAssets(
        usd_path=usd_dir / "mobile_manipulator_PPR_base_corrected.usd",
        config_dir=usd_dir / "configuration",
    )


def get_mobile_mm_usd_path() -> Path:
    """Shortcut returning just the USD stage path."""
    assets = get_mobile_mm_assets()
    assets.validate()
    return assets.usd_path


def ensure_env_ready() -> None:
    """Sanity-check that the USD bundle exists before Isaac Lab loads it."""
    assets = get_mobile_mm_assets()
    assets.validate()
