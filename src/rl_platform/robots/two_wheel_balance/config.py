"""Isaac Lab articulation configuration for the two-wheel balance prototype."""

import copy

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
TWO_WHEEL_WHOLE_BODY_USD_PATH = (
    PROJECT_ROOT
    / "assets_own"
    / "recomoProto2_two_wheel_whole_body"
    / "recomoProto2_two_wheel_whole_body.usd"
)
TWO_WHEEL_WHOLE_BODY_ATTITUDE_USD_PATH = (
    PROJECT_ROOT
    / "assets_own"
    / "recomoProto2_two_wheel_whole_body_attitude"
    / "recomoProto2_two_wheel_whole_body_attitude.usd"
)
TWO_WHEEL_RISER_USD_PATH = (
    PROJECT_ROOT
    / "assets_own"
    / "recomoProto2_two_wheel_riser"
    / "recomoProto2_two_wheel_riser.usd"
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


TWO_WHEEL_WHOLE_BODY_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TWO_WHEEL_WHOLE_BODY_USD_PATH),
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
        pos=(0.0, 0.0, 0.003),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            "joint6_arm_yaw": 0.0,
            "joint5_arm_pitch": 1.5707963268,
            "joint4_elbow_pitch": 2.3561944902,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheel_effort": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "arm_home_hold": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint6_arm_yaw",
                "joint5_arm_pitch",
                "joint4_elbow_pitch",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=0.5,
            stiffness=200.0,
            damping=20.0,
        ),
    },
)


TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG = copy.deepcopy(TWO_WHEEL_WHOLE_BODY_CFG)
TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG.spawn.usd_path = str(
    TWO_WHEEL_WHOLE_BODY_ATTITUDE_USD_PATH
)
TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG.init_state.joint_pos.update(
    {
        "joint3_gimbal_yaw": 0.0,
        "joint2_gimbal_roll": 0.0,
        "joint1_gimbal_pitch": 0.0,
    }
)
TWO_WHEEL_WHOLE_BODY_ATTITUDE_CFG.actuators["gimbal_attitude_adapter"] = (
    ImplicitActuatorCfg(
        joint_names_expr=[
            "joint3_gimbal_yaw",
            "joint2_gimbal_roll",
            "joint1_gimbal_pitch",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=0.5,
        stiffness=400.0,
        damping=40.0,
        # The source CAD inertias omit reflected motor/gear inertia.  A small
        # armature prevents the low-inertia gimbal axes from taking an
        # unrealistic velocity impulse when the two-wheel base contacts ground.
        armature=0.01,
    )
)


TWO_WHEEL_RISER_CFG = copy.deepcopy(TWO_WHEEL_BALANCE_CFG)
TWO_WHEEL_RISER_CFG.spawn.usd_path = str(TWO_WHEEL_RISER_USD_PATH)
TWO_WHEEL_RISER_CFG.init_state.joint_pos.update(
    {
        "riser_joint": 0.3,
        "joint3_gimbal_yaw": 0.0,
        "joint2_gimbal_roll": 0.0,
        "joint1_gimbal_pitch": 0.0,
    }
)
TWO_WHEEL_RISER_CFG.init_state.joint_vel = {".*": 0.0}
TWO_WHEEL_RISER_CFG.actuators["riser_position"] = ImplicitActuatorCfg(
    joint_names_expr=["riser_joint"],
    effort_limit_sim=300.0,
    velocity_limit_sim=1.0,
    stiffness=1200.0,
    damping=120.0,
    armature=0.05,
)
TWO_WHEEL_RISER_CFG.actuators["gimbal_attitude_adapter"] = ImplicitActuatorCfg(
    joint_names_expr=[
        "joint3_gimbal_yaw",
        "joint2_gimbal_roll",
        "joint1_gimbal_pitch",
    ],
    effort_limit_sim=10.0,
    velocity_limit_sim=0.5,
    stiffness=400.0,
    damping=40.0,
    armature=0.01,
)
