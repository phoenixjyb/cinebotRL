"""Configuration helpers for the RecomoProto1 robot RL task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import assets_root


@dataclass(frozen=True)
class RecomoProto1Assets:
    """Container describing the USD bundle for the RecomoProto1 robot."""

    usd_path: Path
    config_dir: Path

    def validate(self) -> None:
        """Raise FileNotFoundError if expected assets are missing."""
        if not self.usd_path.is_file():
            raise FileNotFoundError(f"Missing USD file: {self.usd_path}")
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Missing configuration directory: {self.config_dir}")


def get_recomoproto1_assets() -> RecomoProto1Assets:
    """Return paths to the current RecomoProto USD and supporting configuration."""
    # The task name is kept for compatibility, but the active robot asset is
    # Proto2 from the latest PNC/MoveIt URDF.
    usd_dir = assets_root() / "recomoProto2-1190_moveit"
    return RecomoProto1Assets(
        usd_path=usd_dir / "recomoProto2-1190_moveit.usd",
        config_dir=usd_dir / "configuration" if (usd_dir / "configuration").is_dir() else usd_dir,
    )


def get_recomoproto1_usd_path() -> Path:
    """Shortcut returning just the USD stage path."""
    assets = get_recomoproto1_assets()
    assets.validate()
    return assets.usd_path


def ensure_env_ready() -> None:
    """Sanity-check that the USD bundle exists before Isaac Lab loads it."""
    assets = get_recomoproto1_assets()
    assets.validate()
