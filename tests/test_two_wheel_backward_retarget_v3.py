from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

from retarget_corrected_teacher_v3_backward import (  # noqa: E402
    inverse_integrate_unicycle,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    integrate_unicycle,
)


def test_inverse_unicycle_exactly_recovers_predecessor() -> None:
    states = (
        np.array([0.4, -0.8, 0.7]),
        np.array([-0.2, 0.3, -1.1]),
    )
    controls = ((0.3, 0.0), (-0.25, 0.35), (0.4, -0.4))
    for state in states:
        for velocity, yaw_rate in controls:
            successor = integrate_unicycle(state, velocity, yaw_rate, 0.1)
            recovered = inverse_integrate_unicycle(
                successor, velocity, yaw_rate, 0.1
            )
            np.testing.assert_allclose(recovered, state, atol=1e-12, rtol=0.0)


def test_inverse_unicycle_rejects_invalid_input() -> None:
    with np.testing.assert_raises(ValueError):
        inverse_integrate_unicycle(np.zeros(2), 0.0, 0.0, 0.1)
    with np.testing.assert_raises(ValueError):
        inverse_integrate_unicycle(np.zeros(3), 0.0, 0.0, 0.0)
