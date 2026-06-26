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
    # Latest PNC URDF (recomoProto2-1190) converted to USD.
    usd_dir = assets_root() / "recomoProto2-1190_moveit"
    usd_path = usd_dir / "recomoProto2-1190_moveit.usd"

    # Fallbacks keep the previous Proto1/legacy setups runnable while Proto2 is
    # being generated or validated on a new machine.
    if not usd_path.is_file():
        proto1_dir = assets_root() / "recomoProto1-1190_moveit"
        proto1_path = proto1_dir / "recomoProto1-1190_moveit.usd"
        if proto1_path.is_file():
            usd_dir = proto1_dir
            usd_path = proto1_path
        else:
            legacy_dir = assets_root() / "usd"
            usd_path = legacy_dir / "mobile_manipulator_PPR_theta_x_y.usd"
            usd_dir = legacy_dir
        import warnings
        warnings.warn(
            f"Proto2 PNC USD not found at {assets_root() / 'recomoProto2-1190_moveit'}. "
            f"Falling back to {usd_path}. Run URDF-to-USD conversion to use Proto2.",
            stacklevel=2,
        )

    return MobileManipulatorAssets(
        usd_path=usd_path,
        config_dir=usd_dir / "configuration" if (usd_dir / "configuration").is_dir() else usd_dir,
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
