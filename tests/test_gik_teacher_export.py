"""Regression tests for MATLAB teacher obstacle export."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.imitation.export_gik_teacher_manifest import (  # noqa: E402
    ROBOT_FOOTPRINT_RADIUS,
    compute_obstacle_clearance,
    read_obstacle_geometry,
)


def test_matlab_cell_obstacles_are_dereferenced():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teacher.mat"
        with h5py.File(path, "w") as f:
            refs = f.create_group("#refs#")
            centers = [refs.create_dataset(f"center_{i}", data=np.asarray(value)[:, None]) for i, value in enumerate(((1.0, 2.0), (3.0, 4.0)))]
            radii = [refs.create_dataset(f"radius_{i}", data=np.asarray([[value]])) for i, value in enumerate((0.2, 0.3))]
            margins = [refs.create_dataset(f"margin_{i}", data=np.asarray([[value]])) for i, value in enumerate((0.1, 0.1))]
            group = f.create_group("log/floorDiscs")
            for name, values in (("Center", centers), ("Radius", radii), ("SafetyMargin", margins)):
                cells = group.create_dataset(name, shape=(1, 2), dtype=h5py.ref_dtype)
                for index, value in enumerate(values):
                    cells[0, index] = value.ref

        with h5py.File(path, "r") as f:
            center_xy, radius, margin = read_obstacle_geometry(f)
        np.testing.assert_allclose(center_xy, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(radius, np.asarray([0.2, 0.3]))
        np.testing.assert_allclose(margin, np.asarray([0.1, 0.1]))


def test_clearance_uses_robot_footprint_not_safety_margin():
    q = np.zeros((2, 9), dtype=np.float32)
    clearance = compute_obstacle_clearance(
        q,
        centers_xy=np.asarray([[1.0, 0.0]], dtype=np.float32),
        radii=np.asarray([0.2], dtype=np.float32),
        safety_margins=np.asarray([0.9], dtype=np.float32),
    )
    np.testing.assert_allclose(clearance, np.asarray([[1.0 - 0.2 - ROBOT_FOOTPRINT_RADIUS]]))


if __name__ == "__main__":
    test_matlab_cell_obstacles_are_dereferenced()
    test_clearance_uses_robot_footprint_not_safety_margin()
    print("GIK teacher export assertions passed")
