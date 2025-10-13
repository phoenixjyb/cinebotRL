"""Helpers for dealing with ROS-style package layout on disk."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class PackageContext:
    package_dir: Path
    package_name: str
    urdf_path: Path

    @property
    def package_xml_path(self) -> Path:
        return self.package_dir / "package.xml"

    @property
    def meshes_dir(self) -> Path:
        return self.package_dir / "meshes"

    @property
    def srdf_dir(self) -> Path:
        return self.package_dir / "srdf"

    @classmethod
    def from_directory(cls, package_dir: Path, *, urdf_path: Optional[Path] = None) -> "PackageContext":
        package_dir = package_dir.resolve()
        package_xml = package_dir / "package.xml"
        if not package_xml.exists():
            raise FileNotFoundError(f"Expected package.xml in {package_dir}")
        tree = ET.parse(package_xml)
        root = tree.getroot()
        if root.tag != "package":
            raise ValueError(f"package.xml root must be <package>, got <{root.tag}>")
        name_elem = root.find("name")
        if name_elem is None or not name_elem.text:
            raise ValueError("package.xml missing <name> entry")
        package_name = name_elem.text.strip()

        if urdf_path is None:
            urdf_candidates = list(_find_urdf_files(package_dir))
            if not urdf_candidates:
                raise FileNotFoundError(f"No URDF files found under {package_dir}")
            if len(urdf_candidates) > 1:
                raise ValueError(
                    "Multiple URDF files found. Please specify one explicitly: "
                    + ", ".join(str(path.relative_to(package_dir)) for path in urdf_candidates)
                )
            urdf_path = urdf_candidates[0]
        else:
            urdf_path = urdf_path.resolve()
            if not urdf_path.exists():
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        return cls(package_dir=package_dir, package_name=package_name, urdf_path=urdf_path)

    def resolve_uri(self, uri: str) -> Path:
        if uri.startswith("package://"):
            remainder = uri[len("package://"):]
            parts = remainder.split("/", 1)
            if len(parts) == 1:
                raise ValueError(f"Malformed package URI: {uri}")
            package_name, rel_path = parts
            if package_name and package_name != self.package_name:
                # Allow cross-package reference but warn by raising ValueError for now.
                raise ValueError(
                    f"URI targets package '{package_name}' but context is '{self.package_name}'."
                )
            return self.package_dir / rel_path
        return (self.urdf_path.parent / uri).resolve()


def _find_urdf_files(package_dir: Path) -> Iterable[Path]:
    for pattern in ("*.urdf", "*.xacro"):
        for path in package_dir.rglob(pattern):
            if path.is_file():
                yield path

