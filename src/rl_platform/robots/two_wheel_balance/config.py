"""Isaac Lab articulation configuration for the two-wheel balance prototype."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TWO_WHEEL_USD_PATH = (
    PROJECT_ROOT
    / "assets_own"
    / "recomoProto2_two_wheel_balance"
    / "recomoProto2_two_wheel_balance.usd"
)


TWO_WHEEL_BALANCE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TWO_WHEEL_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The URDF root is the ground-plane axle midpoint. Keep this offset tiny
        # so collision-origin errors are visible instead of hidden by a drop.
        pos=(0.0, 0.0, 0.003),
        joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
        joint_vel={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
    ),
    actuators={
        "wheel_effort": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        )
    },
)
