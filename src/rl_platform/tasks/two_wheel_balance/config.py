"""Configuration for the isolated two-wheel balance DirectRLEnv."""

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_BALANCE_CFG


@configclass
class RecomoTwoWheelBalanceEnvCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 10.0
    action_space = 2
    observation_space = 10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=2.0,
        replicate_physics=True,
        # The URDF importer emits a layered USD that does not clone reliably in
        # Fabric on Isaac Sim 5.1. USD cloning is slower but deterministic.
        clone_in_fabric=False,
    )
    # Keep this out of InteractiveSceneCfg's auto-discovery. The DirectRLEnv
    # owns spawning/cloning explicitly, matching Isaac Lab's cartpole task.
    robot_cfg = TWO_WHEEL_BALANCE_CFG
    base_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/base_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    torque_limit_nm = 20.0  # Provisional until Kt/gear/current limits are measured.
    wheel_speed_hard_limit = 20.0
    fall_pitch_rad = math.radians(35.0)
    fall_roll_rad = math.radians(35.0)
    forbidden_body_contact_force_n = 5.0
    reset_pitch_rad = 0.0
    command_vx = 0.0
    command_wz = 0.0

    upright_sigma = math.radians(10.0)
    pitch_rate_scale = -0.05
    vx_tracking_scale = 0.5
    wz_tracking_scale = 0.25
    wheel_speed_scale = -0.002
    action_magnitude_scale = -0.002
    action_rate_scale = -0.01
    alive_scale = 0.2
    termination_scale = -2.0
