#!/usr/bin/env python3
"""Evaluate frozen chassis tracking under deterministic upper-body pushes."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


POLICY_HZ = 200.0
WHEEL_RADIUS_M = 0.1016
RISER_LOWER_M = 0.0
RISER_UPPER_M = 1.2
RISER_PROXY_JOINTS = (
    "joint1_gimbal_pitch",
    "joint2_gimbal_roll",
    "joint3_gimbal_yaw",
)


parser = argparse.ArgumentParser()
parser.add_argument("--gains", type=Path, required=True)
parser.add_argument(
    "--robot-form",
    choices=("balance", "riser"),
    default="balance",
    help="Select the lightweight balance chassis or the complete riser plant.",
)
parser.add_argument(
    "--riser-position-m",
    type=float,
    default=0.3,
    help="Static riser joint position used by the full-riser plant gate.",
)
parser.add_argument("--maximum-riser-hold-error-m", type=float, default=0.01)
parser.add_argument("--maximum-gimbal-hold-error-deg", type=float, default=1.0)
parser.add_argument(
    "--maximum-direction-speed-asymmetry-mps", type=float, default=0.05
)
parser.add_argument("--num-envs", type=int, default=24)
parser.add_argument("--horizon-steps", type=int, default=2000)
parser.add_argument("--vx-commands", default="-0.2,0.2")
parser.add_argument("--wz-commands", default="-0.4,0,0.4")
parser.add_argument("--push-forces-n", default="-40,-20,20,40")
parser.add_argument("--command-start-step", type=int, default=200)
parser.add_argument("--push-start-step", type=int, default=800)
parser.add_argument("--push-duration-steps", type=int, default=20)
parser.add_argument("--push-height-m", type=float, default=0.5)
parser.add_argument("--tracking-settle-steps", type=int, default=100)
parser.add_argument("--recovery-hold-steps", type=int, default=50)
parser.add_argument("--maximum-recovery-s", type=float, default=2.0)
parser.add_argument("--recovery-tilt-deg", type=float, default=2.0)
parser.add_argument("--recovery-angular-rate", type=float, default=0.2)
parser.add_argument("--balance-tilt-margin-deg", type=float, default=0.5)
parser.add_argument("--balance-angular-rate-margin", type=float, default=0.05)
parser.add_argument("--recovery-vx-error", type=float, default=0.10)
parser.add_argument("--recovery-wz-error", type=float, default=0.15)
parser.add_argument("--maximum-tilt-deg", type=float, default=12.0)
parser.add_argument("--maximum-post-vx-rmse", type=float, default=0.10)
parser.add_argument("--maximum-post-wz-rmse", type=float, default=0.15)
parser.add_argument("--maximum-saturation-ratio", type=float, default=0.10)
parser.add_argument("--minimum-success-rate", type=float, default=0.95)
parser.add_argument(
    "--controller-profile",
    choices=("default", "structural_robust_v1"),
    default="default",
)
parser.add_argument("--vx-kp", type=float)
parser.add_argument("--vx-ki", type=float)
parser.add_argument("--wz-kp", type=float)
parser.add_argument("--wz-ki", type=float)
parser.add_argument("--wz-feedforward", type=float)
parser.add_argument("--vx-integral-limit", type=float)
parser.add_argument("--wz-integral-limit", type=float)
parser.add_argument(
    "--governor-include-opposing-bias", action="store_true", default=None
)
parser.add_argument("--pitch-reference-limit-deg", type=float)
parser.add_argument("--limit-total-pitch-reference", action="store_true")
parser.add_argument(
    "--reset-opposing-vx-integral-on-directional-deficit",
    action="store_true",
)
parser.add_argument(
    "--vx-integral-reset-reference-deadband-mps",
    type=float,
    default=0.05,
)
parser.add_argument("--use-root-velocity-outer-feedback", action="store_true")
parser.add_argument("--vx-reference-slew-rate", type=float)
parser.add_argument("--wz-reference-slew-rate", type=float)
parser.add_argument("--path-progress-governor", action="store_true", default=None)
parser.add_argument("--governor-bias-start-deg", type=float, default=0.5)
parser.add_argument("--governor-bias-full-deg", type=float, default=2.5)
parser.add_argument("--governor-minimum-progress-scale", type=float, default=0.75)
parser.add_argument(
    "--tracking-reference",
    choices=("requested", "admitted"),
    default="requested",
)
parser.add_argument("--minimum-path-progress-scale", type=float, default=0.0)
parser.add_argument(
    "--plant-uncertainty-profile",
    choices=("nominal", "provisional_prior_v1", "diagnostic_v1"),
    default="nominal",
)
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (
    np.isfinite(args.riser_position_m)
    and RISER_LOWER_M <= args.riser_position_m <= RISER_UPPER_M
):
    parser.error(
        f"--riser-position-m must be in [{RISER_LOWER_M}, {RISER_UPPER_M}]"
    )
if not (
    np.isfinite(args.maximum_riser_hold_error_m)
    and args.maximum_riser_hold_error_m > 0.0
):
    parser.error("--maximum-riser-hold-error-m must be positive")
if not (
    np.isfinite(args.maximum_gimbal_hold_error_deg)
    and args.maximum_gimbal_hold_error_deg > 0.0
):
    parser.error("--maximum-gimbal-hold-error-deg must be positive")
if not (
    np.isfinite(args.maximum_direction_speed_asymmetry_mps)
    and args.maximum_direction_speed_asymmetry_mps >= 0.0
):
    parser.error("--maximum-direction-speed-asymmetry-mps must be non-negative")
if not (
    np.isfinite(args.vx_integral_reset_reference_deadband_mps)
    and args.vx_integral_reset_reference_deadband_mps > 0.0
):
    parser.error("--vx-integral-reset-reference-deadband-mps must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_RISER_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    PlantVariation,
    allocate_common_yaw_action,
    cascaded_lqr_config,
    cascaded_lqr_action,
    diagnostic_plant_variations,
    provisional_plant_variations,
)
from rl_platform.tasks.two_wheel_balance.whole_body_tracking import (
    equilibrium_pitch_from_world_com,
)
from task_spec import register_isaac_lab_tasks


def parse_csv(value: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"expected finite comma-separated values, got {value!r}")
    return result


def single_joint_id(robot, joint_name: str) -> int:
    joint_ids = robot.find_joints(joint_name)[0]
    if len(joint_ids) != 1:
        raise RuntimeError(f"expected exactly one {joint_name}, got {joint_ids}")
    return int(joint_ids[0])


def equilibrium_pitch_biases(env, body_masses: torch.Tensor) -> np.ndarray:
    """Resolve the physical pitch equilibrium for each complete riser plant."""

    robot = env.robot
    if not hasattr(robot.data, "body_com_pos_w"):
        raise RuntimeError("Isaac articulation data does not expose body_com_pos_w")
    body_com_positions = robot.data.body_com_pos_w
    center_of_mass_world = (
        torch.sum(body_masses[:, :, None] * body_com_positions, dim=1)
        / torch.sum(body_masses, dim=1)[:, None]
    )
    root_positions = robot.data.root_pos_w.detach().cpu().numpy()
    root_quaternions = robot.data.root_quat_w.detach().cpu().numpy()
    center_of_mass_world_np = center_of_mass_world.detach().cpu().numpy()
    return np.asarray(
        [
            equilibrium_pitch_from_world_com(
                root_positions[index],
                root_quaternions[index],
                center_of_mass_world_np[index],
                WHEEL_RADIUS_M,
            )[0]
            for index in range(env.num_envs)
        ],
        dtype=np.float64,
    )


def build_scenarios(
    num_envs: int,
    vx_commands: np.ndarray,
    wz_commands: np.ndarray,
    push_forces_n: np.ndarray,
    plant_variations: tuple[PlantVariation, ...],
) -> list[dict[str, object]]:
    combinations = list(
        itertools.product(vx_commands, wz_commands, push_forces_n, plant_variations)
    )
    return [
        {
            "vx_ref_m_s": float(combinations[index % len(combinations)][0]),
            "wz_ref_rad_s": float(combinations[index % len(combinations)][1]),
            "push_force_x_n": float(combinations[index % len(combinations)][2]),
            "push_impulse_x_ns": float(
                combinations[index % len(combinations)][2]
                * args.push_duration_steps
                / POLICY_HZ
            ),
            "plant": asdict(combinations[index % len(combinations)][3]),
        }
        for index in range(num_envs)
    ]


def apply_plant_variations(env, variations: list[PlantVariation]) -> dict[str, object]:
    """Apply deterministic per-environment PhysX properties at initialization."""
    if len(variations) != env.num_envs:
        raise ValueError("plant variation count must match the environment count")
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")
    view = env.robot.root_physx_view
    default_masses = env.robot.data.default_mass.cpu()
    nominal_total_mass_kg = default_masses.sum(dim=1)
    resolved_mass_scale = torch.tensor(
        [
            item.target_total_mass_kg / float(nominal_total_mass_kg[index])
            if item.target_total_mass_kg is not None
            else item.mass_scale
            for index, item in enumerate(variations)
        ],
        device="cpu",
    )
    inertia_scale = torch.tensor([item.inertia_scale for item in variations], device="cpu")

    masses = view.get_masses().clone()
    masses[env_ids] = default_masses[env_ids] * resolved_mass_scale[:, None]
    view.set_masses(masses, env_ids)

    inertias = view.get_inertias().clone()
    inertias[env_ids] = (
        env.robot.data.default_inertia.cpu()[env_ids]
        * resolved_mass_scale[:, None, None]
        * inertia_scale[:, None, None]
    )
    view.set_inertias(inertias, env_ids)

    base_body_id = int(env._base_body_idx[0])
    coms = view.get_coms().clone()
    coms[env_ids, base_body_id, 0] += torch.tensor(
        [item.com_offset_x_m for item in variations], device="cpu"
    )
    coms[env_ids, base_body_id, 2] += torch.tensor(
        [item.com_offset_z_m for item in variations], device="cpu"
    )
    view.set_coms(coms, env_ids)

    materials = view.get_material_properties().clone()
    material_before = materials[:, 0, :].tolist()
    friction_env_ids = [
        index for index, item in enumerate(variations) if item.static_friction is not None
    ]
    if friction_env_ids:
        friction_indices = torch.tensor(friction_env_ids, dtype=torch.long, device="cpu")
        materials[friction_indices, :, 0] = torch.tensor(
            [variations[index].static_friction for index in friction_env_ids],
            device="cpu",
        )[:, None]
        materials[friction_indices, :, 1] = torch.tensor(
            [variations[index].dynamic_friction for index in friction_env_ids],
            device="cpu",
        )[:, None]
        materials[friction_indices, :, 2] = 0.0
        view.set_material_properties(materials, friction_indices)

    applied_masses = view.get_masses()
    return {
        "body_names": list(env.robot.body_names),
        "nominal_total_mass_kg": float(nominal_total_mass_kg[0]),
        "nominal_body_masses_kg": default_masses[0].tolist(),
        "resolved_mass_scale": resolved_mass_scale.tolist(),
        "applied_total_mass_kg": applied_masses.sum(dim=1).tolist(),
        "com_shift_body": env.robot.body_names[base_body_id],
        "first_shape_material_before_variation": material_before,
        "friction_application": (
            "explicit friction cases override all robot collision shapes against the nominal ground; "
            "other cases preserve the loaded material"
        ),
    }


def set_push_wrench(env, force_x_n: np.ndarray) -> None:
    force = torch.zeros((env.num_envs, 1, 3), device=env.device)
    torque = torch.zeros_like(force)
    force[:, 0, 0] = torch.as_tensor(force_x_n, dtype=torch.float32, device=env.device)
    torque[:, 0, 1] = force[:, 0, 0] * args.push_height_m
    env.robot.set_external_force_and_torque(
        forces=force,
        torques=torque,
        body_ids=env._base_body_idx,
        is_global=True,
    )


def safe_rmse(squared_error_sum: float, count: int) -> float:
    return float(np.sqrt(squared_error_sum / count)) if count else 0.0


def main() -> int:
    push_end_step = args.push_start_step + args.push_duration_steps
    latest_recovery_step = push_end_step + int(round(args.maximum_recovery_s * POLICY_HZ))
    if args.num_envs < 1 or args.horizon_steps < 1:
        raise ValueError("--num-envs and --horizon-steps must be positive")
    if not 0 <= args.command_start_step < args.push_start_step < push_end_step:
        raise ValueError("command and push windows must be ordered")
    if latest_recovery_step >= args.horizon_steps:
        raise ValueError("horizon must include a post-recovery measurement window")
    if args.tracking_settle_steps < 0 or args.recovery_hold_steps < 1:
        raise ValueError("invalid settle or recovery hold steps")
    if args.command_start_step + args.tracking_settle_steps >= args.push_start_step:
        raise ValueError("command must settle before the pre-push balance window")
    if args.push_height_m < 0.0:
        raise ValueError("--push-height-m must be non-negative")
    if not 0.0 <= args.minimum_path_progress_scale <= 1.0:
        raise ValueError("--minimum-path-progress-scale must be in [0, 1]")

    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])
    action_limit = float(gain_data["action_limit"])
    vx_commands = parse_csv(args.vx_commands)
    wz_commands = parse_csv(args.wz_commands)
    push_forces_n = parse_csv(args.push_forces_n)
    if args.plant_uncertainty_profile == "diagnostic_v1":
        plant_variations = diagnostic_plant_variations()
    elif args.plant_uncertainty_profile == "provisional_prior_v1":
        plant_variations = provisional_plant_variations()
    else:
        plant_variations = (PlantVariation("nominal"),)
    expected_scenarios = (
        len(vx_commands)
        * len(wz_commands)
        * len(push_forces_n)
        * len(plant_variations)
    )
    if (
        args.plant_uncertainty_profile != "nominal"
        and args.num_envs != expected_scenarios
    ):
        raise ValueError(
            "non-nominal plant profile requires complete Cartesian coverage: "
            f"set --num-envs {expected_scenarios}"
        )
    scenarios = build_scenarios(
        args.num_envs,
        vx_commands,
        wz_commands,
        push_forces_n,
        plant_variations,
    )
    vx_ref = np.asarray([item["vx_ref_m_s"] for item in scenarios], dtype=np.float64)
    wz_ref = np.asarray([item["wz_ref_rad_s"] for item in scenarios], dtype=np.float64)
    push_force = np.asarray([item["push_force_x_n"] for item in scenarios], dtype=np.float64)
    scenario_plants = [PlantVariation(**item["plant"]) for item in scenarios]
    torque_scale = np.asarray(
        [item.torque_scale for item in scenario_plants], dtype=np.float32
    )
    action_delay_steps = np.asarray(
        [item.action_delay_steps for item in scenario_plants], dtype=np.int64
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    if args.robot_form == "riser":
        cfg.robot_cfg = copy.deepcopy(TWO_WHEEL_RISER_CFG)
        cfg.robot_cfg.init_state.joint_pos["riser_joint"] = args.riser_position_m
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode=None,
        disable_env_checker=True,
    )
    obs, _ = env.reset(seed=args.seed)
    unwrapped = env.unwrapped
    riser_joint_id = None
    proxy_joint_ids: list[int] = []
    riser_position_target = None
    proxy_position_target = None
    if args.robot_form == "riser":
        riser_joint_id = single_joint_id(unwrapped.robot, "riser_joint")
        proxy_joint_ids = [
            single_joint_id(unwrapped.robot, name) for name in RISER_PROXY_JOINTS
        ]
        riser_position_target = torch.full(
            (args.num_envs, 1),
            args.riser_position_m,
            dtype=torch.float32,
            device=unwrapped.device,
        )
        proxy_position_target = torch.zeros(
            (args.num_envs, len(proxy_joint_ids)),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        unwrapped.robot.set_joint_position_target(
            riser_position_target, joint_ids=[riser_joint_id]
        )
        unwrapped.robot.set_joint_velocity_target(
            torch.zeros_like(riser_position_target), joint_ids=[riser_joint_id]
        )
        unwrapped.robot.set_joint_position_target(
            proxy_position_target, joint_ids=proxy_joint_ids
        )
    plant_runtime = {
        "body_names": list(unwrapped.robot.body_names),
        "nominal_total_mass_kg": float(
            unwrapped.robot.data.default_mass[0].sum().item()
        ),
        "nominal_body_masses_kg": unwrapped.robot.data.default_mass[0].tolist(),
    }
    if args.plant_uncertainty_profile != "nominal":
        plant_runtime = apply_plant_variations(unwrapped, scenario_plants)
    body_masses = unwrapped.robot.root_physx_view.get_masses().to(
        device=unwrapped.device
    )
    set_push_wrench(unwrapped, np.zeros(args.num_envs))
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    active = np.ones(args.num_envs, dtype=bool)
    survived = np.zeros(args.num_envs, dtype=bool)
    duration_steps = np.full(args.num_envs, args.horizon_steps, dtype=np.int64)
    integrals = np.zeros((args.num_envs, 6), dtype=np.float64)
    final_controller_state = np.zeros_like(integrals)
    requested_action_np = np.zeros(
        (args.num_envs, len(ACTION_NAMES)), dtype=np.float32
    )
    action_np = np.zeros_like(requested_action_np)
    action_history = np.zeros(
        (int(np.max(action_delay_steps)) + 1, args.num_envs, len(ACTION_NAMES)),
        dtype=np.float32,
    )
    controller_overrides = {
        name: value
        for name, value in {
            "vx_kp": args.vx_kp,
            "vx_ki": args.vx_ki,
            "wz_kp": args.wz_kp,
            "wz_ki": args.wz_ki,
            "wz_feedforward": args.wz_feedforward,
            "vx_integral_limit": args.vx_integral_limit,
            "wz_integral_limit": args.wz_integral_limit,
            "governor_include_opposing_bias": args.governor_include_opposing_bias,
            "limit_total_pitch_reference": args.limit_total_pitch_reference,
            "reset_opposing_vx_integral_on_directional_deficit": (
                args.reset_opposing_vx_integral_on_directional_deficit
            ),
            "vx_integral_reset_reference_deadband_mps": (
                args.vx_integral_reset_reference_deadband_mps
            ),
            "vx_reference_slew_rate_m_s2": args.vx_reference_slew_rate,
            "wz_reference_slew_rate_rad_s2": args.wz_reference_slew_rate,
            "path_progress_governor_enabled": args.path_progress_governor,
            "governor_bias_start_rad": np.radians(args.governor_bias_start_deg),
            "governor_bias_full_rad": np.radians(args.governor_bias_full_deg),
            "governor_minimum_progress_scale": args.governor_minimum_progress_scale,
            "pitch_reference_limit_rad": (
                np.radians(args.pitch_reference_limit_deg)
                if args.pitch_reference_limit_deg is not None
                else None
            ),
        }.items()
        if value is not None
    }
    config = cascaded_lqr_config(
        args.controller_profile, action_limit=action_limit, **controller_overrides
    )

    balance_hold = np.zeros(args.num_envs, dtype=np.int64)
    tracking_hold = np.zeros(args.num_envs, dtype=np.int64)
    balance_recovery_steps = np.full(args.num_envs, -1, dtype=np.int64)
    tracking_recovery_steps = np.full(args.num_envs, -1, dtype=np.int64)
    peak_pitch_deg = np.zeros(args.num_envs, dtype=np.float64)
    peak_roll_deg = np.zeros(args.num_envs, dtype=np.float64)
    peak_wheel_speed = np.zeros(args.num_envs, dtype=np.float64)
    saturated_actions = np.zeros(args.num_envs, dtype=np.int64)
    action_samples = np.zeros(args.num_envs, dtype=np.int64)
    post_vx_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_vx_com_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_vx_odometry_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_wz_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_admitted_vx_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_admitted_wz_squared_error = np.zeros(args.num_envs, dtype=np.float64)
    post_vx_sum = np.zeros(args.num_envs, dtype=np.float64)
    post_vx_odometry_sum = np.zeros(args.num_envs, dtype=np.float64)
    post_wz_sum = np.zeros(args.num_envs, dtype=np.float64)
    post_samples = np.zeros(args.num_envs, dtype=np.int64)
    admitted_vx_ref = np.zeros(args.num_envs, dtype=np.float64)
    admitted_wz_ref = np.zeros(args.num_envs, dtype=np.float64)
    path_progress_scale_min = np.ones(args.num_envs, dtype=np.float64)
    path_progress_scale_sum = np.zeros(args.num_envs, dtype=np.float64)
    path_progress_scale_samples = np.zeros(args.num_envs, dtype=np.int64)
    wheel_mix_saturated = np.zeros(args.num_envs, dtype=np.int64)
    wheel_mix_samples = np.zeros(args.num_envs, dtype=np.int64)
    post_wheel_mix_saturated = np.zeros(args.num_envs, dtype=np.int64)
    post_wheel_mix_samples = np.zeros(args.num_envs, dtype=np.int64)
    allocation_common_squared_loss = np.zeros(args.num_envs, dtype=np.float64)
    allocation_yaw_squared_loss = np.zeros(args.num_envs, dtype=np.float64)
    post_allocation_common_squared_loss = np.zeros(args.num_envs, dtype=np.float64)
    post_allocation_yaw_squared_loss = np.zeros(args.num_envs, dtype=np.float64)
    peak_wheel_command_preclip = np.zeros(args.num_envs, dtype=np.float64)
    current_pitch_reference = np.zeros(args.num_envs, dtype=np.float64)
    current_yaw_correction = np.zeros(args.num_envs, dtype=np.float64)
    post_pitch_reference_sum = np.zeros(args.num_envs, dtype=np.float64)
    post_yaw_correction_sum = np.zeros(args.num_envs, dtype=np.float64)
    pre_pitch_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_roll_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_pitch_rate_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_roll_rate_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_pitch_deg = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_pitch_rate = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_wheel_speed = np.zeros(args.num_envs, dtype=np.float64)
    equilibrium_pitch_bias_min_deg = np.full(
        args.num_envs, np.inf, dtype=np.float64
    )
    equilibrium_pitch_bias_max_deg = np.full(
        args.num_envs, -np.inf, dtype=np.float64
    )
    riser_hold_error_max_m = np.zeros(args.num_envs, dtype=np.float64)
    gimbal_hold_error_max_deg = np.zeros(args.num_envs, dtype=np.float64)

    for step in range(args.horizon_steps):
        command_active = step >= args.command_start_step
        step_vx_ref = vx_ref if command_active else np.zeros(args.num_envs)
        step_wz_ref = wz_ref if command_active else np.zeros(args.num_envs)
        unwrapped.vx_ref.copy_(
            torch.as_tensor(step_vx_ref, dtype=torch.float32, device=unwrapped.device)
        )
        unwrapped.wz_ref.copy_(
            torch.as_tensor(step_wz_ref, dtype=torch.float32, device=unwrapped.device)
        )
        if step == args.push_start_step:
            set_push_wrench(unwrapped, push_force)
        elif step == push_end_step:
            set_push_wrench(unwrapped, np.zeros(args.num_envs))

        if args.robot_form == "riser":
            unwrapped.robot.set_joint_position_target(
                riser_position_target, joint_ids=[riser_joint_id]
            )
            unwrapped.robot.set_joint_velocity_target(
                torch.zeros_like(riser_position_target), joint_ids=[riser_joint_id]
            )
            unwrapped.robot.set_joint_position_target(
                proxy_position_target, joint_ids=proxy_joint_ids
            )

        if step % control_interval == 0:
            pitch_bias_override_rad = None
            if args.robot_form == "riser":
                pitch_bias_override_rad = equilibrium_pitch_biases(
                    unwrapped, body_masses
                )
                pitch_bias_deg = np.degrees(pitch_bias_override_rad)
                equilibrium_pitch_bias_min_deg = np.minimum(
                    equilibrium_pitch_bias_min_deg, pitch_bias_deg
                )
                equilibrium_pitch_bias_max_deg = np.maximum(
                    equilibrium_pitch_bias_max_deg, pitch_bias_deg
                )
            outer_vx_feedback_m_s = None
            if args.use_root_velocity_outer_feedback:
                outer_vx_feedback_m_s = (
                    unwrapped._state_terms()["vx"].detach().cpu().numpy()
                )
            requested_action_np, integrals, diagnostics = cascaded_lqr_action(
                current_states,
                step_vx_ref,
                step_wz_ref,
                gain,
                integrals,
                control_dt=control_interval / POLICY_HZ,
                config=config,
                pitch_bias_override_rad=pitch_bias_override_rad,
                outer_vx_feedback_m_s=outer_vx_feedback_m_s,
            )
            admitted_vx_ref = diagnostics["governed_vx_ref"]
            admitted_wz_ref = diagnostics["governed_wz_ref"]
            current_pitch_reference = diagnostics["pitch_reference"]
            current_yaw_correction = diagnostics["yaw_correction"]
            if command_active:
                progress_scale = diagnostics["path_progress_scale"]
                path_progress_scale_min[active] = np.minimum(
                    path_progress_scale_min[active], progress_scale[active]
                )
                path_progress_scale_sum[active] += progress_scale[active]
                path_progress_scale_samples[active] += 1
            requested_action_np = requested_action_np.astype(np.float32)
        history_index = step % len(action_history)
        action_history[history_index] = requested_action_np
        for env_index, delay_steps in enumerate(action_delay_steps):
            if step >= delay_steps:
                delayed_index = (step - delay_steps) % len(action_history)
                action_np[env_index] = action_history[delayed_index, env_index]
            else:
                action_np[env_index] = 0.0
        action_np *= torque_scale[:, None]
        action_np[~active] = 0.0
        _, effective_action_np, wheel_saturated = allocate_common_yaw_action(action_np)
        allocation_loss = action_np - effective_action_np
        if command_active:
            wheel_mix_saturated[active] += np.count_nonzero(
                wheel_saturated[active], axis=1
            )
            wheel_mix_samples[active] += 2
            allocation_common_squared_loss[active] += np.square(
                allocation_loss[active, 0]
            )
            allocation_yaw_squared_loss[active] += np.square(
                allocation_loss[active, 1]
            )
            peak_wheel_command_preclip[active] = np.maximum(
                peak_wheel_command_preclip[active],
                np.abs(action_np[active]).sum(axis=1),
            )
        if step >= latest_recovery_step:
            post_wheel_mix_saturated[active] += np.count_nonzero(
                wheel_saturated[active], axis=1
            )
            post_wheel_mix_samples[active] += 2
            post_allocation_common_squared_loss[active] += np.square(
                allocation_loss[active, 0]
            )
            post_allocation_yaw_squared_loss[active] += np.square(
                allocation_loss[active, 1]
            )
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action_np, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        pitch_deg = np.degrees(np.abs(current_states[:, 0]))
        roll_deg = np.degrees(np.abs(state["roll"].detach().cpu().numpy()))
        pitch_rate = np.abs(current_states[:, 1])
        roll_rate = np.abs(unwrapped.robot.data.root_ang_vel_b[:, 0].detach().cpu().numpy())
        vx_truth = state["vx"].detach().cpu().numpy()
        vx_com_truth = (
            unwrapped.robot.data.root_com_lin_vel_b[:, 0].detach().cpu().numpy()
        )
        vx_odometry = config.wheel_radius_m * current_states[:, 3]
        wz_truth = state["yaw_rate"].detach().cpu().numpy()
        wheel_speed = state["max_abs_wheel_velocity"].detach().cpu().numpy()
        if args.robot_form == "riser":
            actual_riser = (
                unwrapped.robot.data.joint_pos[:, riser_joint_id]
                .detach()
                .cpu()
                .numpy()
            )
            actual_proxy = (
                unwrapped.robot.data.joint_pos[:, proxy_joint_ids]
                .detach()
                .cpu()
                .numpy()
            )
            riser_hold_error_max_m = np.maximum(
                riser_hold_error_max_m,
                np.abs(actual_riser - args.riser_position_m),
            )
            gimbal_hold_error_max_deg = np.maximum(
                gimbal_hold_error_max_deg,
                np.degrees(np.max(np.abs(actual_proxy), axis=1)),
            )
        requested_vx_error = np.abs(vx_truth - step_vx_ref)
        requested_wz_error = np.abs(wz_truth - step_wz_ref)
        admitted_vx_error = np.abs(vx_truth - admitted_vx_ref)
        admitted_wz_error = np.abs(wz_truth - admitted_wz_ref)
        if args.tracking_reference == "admitted":
            tracking_vx_error = admitted_vx_error
            tracking_wz_error = admitted_wz_error
        else:
            tracking_vx_error = requested_vx_error
            tracking_wz_error = requested_wz_error

        if step == args.command_start_step - 1:
            pre_command_pitch_deg[:] = np.degrees(current_states[:, 0])
            pre_command_pitch_rate[:] = current_states[:, 1]
            pre_command_wheel_speed[:] = current_states[:, 3]

        if args.push_start_step - args.tracking_settle_steps <= step < args.push_start_step:
            pre_pitch_max[active] = np.maximum(pre_pitch_max[active], pitch_deg[active])
            pre_roll_max[active] = np.maximum(pre_roll_max[active], roll_deg[active])
            pre_pitch_rate_max[active] = np.maximum(
                pre_pitch_rate_max[active], pitch_rate[active]
            )
            pre_roll_rate_max[active] = np.maximum(
                pre_roll_rate_max[active], roll_rate[active]
            )

        if step >= args.push_start_step:
            peak_pitch_deg[active] = np.maximum(peak_pitch_deg[active], pitch_deg[active])
            peak_roll_deg[active] = np.maximum(peak_roll_deg[active], roll_deg[active])
            peak_wheel_speed[active] = np.maximum(peak_wheel_speed[active], wheel_speed[active])
        if command_active:
            saturated_actions[active] += np.count_nonzero(
                np.abs(requested_action_np[active]) >= action_limit - 1e-6,
                axis=1,
            )
            action_samples[active] += len(ACTION_NAMES)

        if step >= push_end_step:
            pitch_recovery_limit = np.maximum(
                args.recovery_tilt_deg,
                pre_pitch_max + args.balance_tilt_margin_deg,
            )
            roll_recovery_limit = np.maximum(
                args.recovery_tilt_deg,
                pre_roll_max + args.balance_tilt_margin_deg,
            )
            pitch_rate_recovery_limit = np.maximum(
                args.recovery_angular_rate,
                pre_pitch_rate_max + args.balance_angular_rate_margin,
            )
            roll_rate_recovery_limit = np.maximum(
                args.recovery_angular_rate,
                pre_roll_rate_max + args.balance_angular_rate_margin,
            )
            inside_balance = (
                (pitch_deg <= pitch_recovery_limit)
                & (roll_deg <= roll_recovery_limit)
                & (pitch_rate <= pitch_rate_recovery_limit)
                & (roll_rate <= roll_rate_recovery_limit)
                & active
            )
            inside_tracking = (
                (tracking_vx_error <= args.recovery_vx_error)
                & (tracking_wz_error <= args.recovery_wz_error)
                & active
            )
            balance_hold[inside_balance] += 1
            balance_hold[~inside_balance] = 0
            tracking_hold[inside_tracking] += 1
            tracking_hold[~inside_tracking] = 0
            new_balance = active & (balance_recovery_steps < 0) & (
                balance_hold >= args.recovery_hold_steps
            )
            new_tracking = active & (tracking_recovery_steps < 0) & (
                tracking_hold >= args.recovery_hold_steps
            )
            balance_recovery_steps[new_balance] = (
                step - push_end_step - args.recovery_hold_steps + 1
            )
            tracking_recovery_steps[new_tracking] = (
                step - push_end_step - args.recovery_hold_steps + 1
            )

        if step >= latest_recovery_step:
            post_vx_squared_error[active] += np.square(vx_truth[active] - vx_ref[active])
            post_vx_com_squared_error[active] += np.square(
                vx_com_truth[active] - vx_ref[active]
            )
            post_vx_odometry_squared_error[active] += np.square(
                vx_odometry[active] - vx_ref[active]
            )
            post_wz_squared_error[active] += np.square(wz_truth[active] - wz_ref[active])
            post_admitted_vx_squared_error[active] += np.square(
                vx_truth[active] - admitted_vx_ref[active]
            )
            post_admitted_wz_squared_error[active] += np.square(
                wz_truth[active] - admitted_wz_ref[active]
            )
            post_vx_sum[active] += vx_truth[active]
            post_vx_odometry_sum[active] += vx_odometry[active]
            post_wz_sum[active] += wz_truth[active]
            post_pitch_reference_sum[active] += current_pitch_reference[active]
            post_yaw_correction_sum[active] += current_yaw_correction[active]
            post_samples[active] += 1

        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        finished = active & (terminated_np | truncated_np)
        duration_steps[finished] = step + 1
        survived[finished] = truncated_np[finished]
        final_controller_state[finished] = integrals[finished]
        active[finished] = False
        integrals[~active] = 0.0
        if not np.any(active):
            break

    set_push_wrench(unwrapped, np.zeros(args.num_envs))
    survived[active] = True
    final_controller_state[active] = integrals[active]
    env.close()

    scenario_results = []
    for index, scenario in enumerate(scenarios):
        balance_recovery_s = (
            float(balance_recovery_steps[index] / POLICY_HZ)
            if balance_recovery_steps[index] >= 0
            else None
        )
        tracking_recovery_s = (
            float(tracking_recovery_steps[index] / POLICY_HZ)
            if tracking_recovery_steps[index] >= 0
            else None
        )
        saturation_ratio = saturated_actions[index] / max(action_samples[index], 1)
        post_vx_rmse = safe_rmse(
            post_vx_squared_error[index], int(post_samples[index])
        )
        post_vx_com_rmse = safe_rmse(
            post_vx_com_squared_error[index], int(post_samples[index])
        )
        post_vx_odometry_rmse = safe_rmse(
            post_vx_odometry_squared_error[index], int(post_samples[index])
        )
        post_wz_rmse = safe_rmse(
            post_wz_squared_error[index], int(post_samples[index])
        )
        post_admitted_vx_rmse = safe_rmse(
            post_admitted_vx_squared_error[index], int(post_samples[index])
        )
        post_admitted_wz_rmse = safe_rmse(
            post_admitted_wz_squared_error[index], int(post_samples[index])
        )
        selected_post_vx_rmse = (
            post_admitted_vx_rmse
            if args.tracking_reference == "admitted"
            else post_vx_rmse
        )
        selected_post_wz_rmse = (
            post_admitted_wz_rmse
            if args.tracking_reference == "admitted"
            else post_wz_rmse
        )
        progress_scale_mean = (
            path_progress_scale_sum[index] / path_progress_scale_samples[index]
            if path_progress_scale_samples[index]
            else 1.0
        )
        allocation_samples = max(int(wheel_mix_samples[index] / 2), 1)
        post_allocation_samples = max(int(post_wheel_mix_samples[index] / 2), 1)
        post_sample_count = max(int(post_samples[index]), 1)
        passed = bool(
            survived[index]
            and balance_recovery_s is not None
            and balance_recovery_s <= args.maximum_recovery_s
            and tracking_recovery_s is not None
            and tracking_recovery_s <= args.maximum_recovery_s
            and max(peak_pitch_deg[index], peak_roll_deg[index]) <= args.maximum_tilt_deg
            and selected_post_vx_rmse <= args.maximum_post_vx_rmse
            and selected_post_wz_rmse <= args.maximum_post_wz_rmse
            and saturation_ratio <= args.maximum_saturation_ratio
            and path_progress_scale_min[index] >= args.minimum_path_progress_scale
            and (
                args.robot_form != "riser"
                or (
                    riser_hold_error_max_m[index]
                    <= args.maximum_riser_hold_error_m
                    and gimbal_hold_error_max_deg[index]
                    <= args.maximum_gimbal_hold_error_deg
                )
            )
        )
        scenario_results.append(
            {
                **scenario,
                "survived": bool(survived[index]),
                "duration_steps": int(duration_steps[index]),
                "balance_recovery_s": balance_recovery_s,
                "tracking_recovery_s": tracking_recovery_s,
                "peak_pitch_deg": float(peak_pitch_deg[index]),
                "peak_roll_deg": float(peak_roll_deg[index]),
                "peak_wheel_speed_rad_s": float(peak_wheel_speed[index]),
                "pre_command_state": {
                    "pitch_deg": float(pre_command_pitch_deg[index]),
                    "pitch_rate": float(pre_command_pitch_rate[index]),
                    "mean_wheel_velocity_rad_s": float(
                        pre_command_wheel_speed[index]
                    ),
                },
                "pre_push_balance_envelope": {
                    "pitch_deg": float(pre_pitch_max[index]),
                    "roll_deg": float(pre_roll_max[index]),
                    "pitch_rate": float(pre_pitch_rate_max[index]),
                    "roll_rate": float(pre_roll_rate_max[index]),
                },
                "balance_recovery_limits": {
                    "pitch_deg": float(
                        max(
                            args.recovery_tilt_deg,
                            pre_pitch_max[index] + args.balance_tilt_margin_deg,
                        )
                    ),
                    "roll_deg": float(
                        max(
                            args.recovery_tilt_deg,
                            pre_roll_max[index] + args.balance_tilt_margin_deg,
                        )
                    ),
                    "pitch_rate": float(
                        max(
                            args.recovery_angular_rate,
                            pre_pitch_rate_max[index]
                            + args.balance_angular_rate_margin,
                        )
                    ),
                    "roll_rate": float(
                        max(
                            args.recovery_angular_rate,
                            pre_roll_rate_max[index]
                            + args.balance_angular_rate_margin,
                        )
                    ),
                },
                "post_vx_rmse": post_vx_rmse,
                "post_vx_com_rmse": post_vx_com_rmse,
                "post_vx_odometry_rmse": post_vx_odometry_rmse,
                "post_wz_rmse": post_wz_rmse,
                "post_vx_mean": float(post_vx_sum[index] / post_sample_count),
                "post_vx_odometry_mean": float(
                    post_vx_odometry_sum[index] / post_sample_count
                ),
                "post_wz_mean": float(post_wz_sum[index] / post_sample_count),
                "post_admitted_vx_rmse": post_admitted_vx_rmse,
                "post_admitted_wz_rmse": post_admitted_wz_rmse,
                "selected_post_vx_rmse": selected_post_vx_rmse,
                "selected_post_wz_rmse": selected_post_wz_rmse,
                "path_progress_scale_min": float(path_progress_scale_min[index]),
                "path_progress_scale_mean": float(progress_scale_mean),
                "action_saturation_ratio": float(saturation_ratio),
                "riser_plant": (
                    {
                        "riser_position_target_m": args.riser_position_m,
                        "riser_hold_error_max_m": float(
                            riser_hold_error_max_m[index]
                        ),
                        "gimbal_hold_error_max_deg": float(
                            gimbal_hold_error_max_deg[index]
                        ),
                        "equilibrium_pitch_bias_min_deg": float(
                            equilibrium_pitch_bias_min_deg[index]
                        ),
                        "equilibrium_pitch_bias_max_deg": float(
                            equilibrium_pitch_bias_max_deg[index]
                        ),
                    }
                    if args.robot_form == "riser"
                    else None
                ),
                "control_allocation": {
                    "wheel_saturation_ratio": float(
                        wheel_mix_saturated[index] / max(wheel_mix_samples[index], 1)
                    ),
                    "post_wheel_saturation_ratio": float(
                        post_wheel_mix_saturated[index]
                        / max(post_wheel_mix_samples[index], 1)
                    ),
                    "peak_wheel_command_preclip": float(
                        peak_wheel_command_preclip[index]
                    ),
                    "common_authority_loss_rmse": safe_rmse(
                        allocation_common_squared_loss[index], allocation_samples
                    ),
                    "yaw_authority_loss_rmse": safe_rmse(
                        allocation_yaw_squared_loss[index], allocation_samples
                    ),
                    "post_common_authority_loss_rmse": safe_rmse(
                        post_allocation_common_squared_loss[index],
                        post_allocation_samples,
                    ),
                    "post_yaw_authority_loss_rmse": safe_rmse(
                        post_allocation_yaw_squared_loss[index], post_allocation_samples
                    ),
                    "post_pitch_reference_mean_deg": float(
                        np.degrees(post_pitch_reference_sum[index] / post_sample_count)
                    ),
                    "post_yaw_correction_mean": float(
                        post_yaw_correction_sum[index] / post_sample_count
                    ),
                    "final_vx_integral": float(final_controller_state[index, 0]),
                    "final_wz_integral": float(final_controller_state[index, 1]),
                    "latched_pitch_bias_deg": float(
                        np.degrees(final_controller_state[index, 2])
                    ),
                },
                "passed": passed,
            }
        )

    total_post_samples = int(np.sum(post_samples))
    success_rate = float(np.mean([item["passed"] for item in scenario_results]))
    aggregate_requested_vx_rmse = safe_rmse(
        float(np.sum(post_vx_squared_error)), total_post_samples
    )
    aggregate_requested_wz_rmse = safe_rmse(
        float(np.sum(post_wz_squared_error)), total_post_samples
    )
    aggregate_admitted_vx_rmse = safe_rmse(
        float(np.sum(post_admitted_vx_squared_error)), total_post_samples
    )
    aggregate_admitted_wz_rmse = safe_rmse(
        float(np.sum(post_admitted_wz_squared_error)), total_post_samples
    )
    aggregate_selected_vx_rmse = (
        aggregate_admitted_vx_rmse
        if args.tracking_reference == "admitted"
        else aggregate_requested_vx_rmse
    )
    aggregate_selected_wz_rmse = (
        aggregate_admitted_wz_rmse
        if args.tracking_reference == "admitted"
        else aggregate_requested_wz_rmse
    )
    riser_plant_summary = None
    if args.robot_form == "riser":
        riser_plant_summary = {
            "riser_position_target_m": args.riser_position_m,
            "riser_hold_error_max_m": float(np.max(riser_hold_error_max_m)),
            "gimbal_hold_error_max_deg": float(
                np.max(gimbal_hold_error_max_deg)
            ),
            "equilibrium_pitch_bias_min_deg": float(
                np.min(equilibrium_pitch_bias_min_deg)
            ),
            "equilibrium_pitch_bias_max_deg": float(
                np.max(equilibrium_pitch_bias_max_deg)
            ),
        }
    direction_tracking = {}
    for name, predicate in (
        ("reverse", lambda value: value < 0.0),
        ("forward", lambda value: value > 0.0),
    ):
        rows = [
            row for row in scenario_results if predicate(row["vx_ref_m_s"])
        ]
        direction_tracking[name] = {
            "scenario_count": len(rows),
            "command_mean_mps": (
                float(np.mean([row["vx_ref_m_s"] for row in rows]))
                if rows
                else None
            ),
            "achieved_mean_mps": (
                float(np.mean([row["post_vx_mean"] for row in rows]))
                if rows
                else None
            ),
            "selected_tracking_rmse_mean_mps": (
                float(np.mean([row["selected_post_vx_rmse"] for row in rows]))
                if rows
                else None
            ),
        }
    direction_contract_complete = all(
        direction_tracking[name]["scenario_count"] > 0
        for name in ("reverse", "forward")
    )
    direction_speed_asymmetry_mps = None
    if direction_contract_complete:
        direction_speed_asymmetry_mps = abs(
            abs(direction_tracking["forward"]["achieved_mean_mps"])
            - abs(direction_tracking["reverse"]["achieved_mean_mps"])
        )
    summary = {
        "scenarios": len(scenario_results),
        "success_rate": success_rate,
        "survival_rate": float(np.mean(survived)),
        "balance_recovery_rate": float(
            np.mean(balance_recovery_steps >= 0)
        ),
        "tracking_recovery_rate": float(
            np.mean(tracking_recovery_steps >= 0)
        ),
        "balance_recovery_s_max": max(
            (item["balance_recovery_s"] for item in scenario_results if item["balance_recovery_s"] is not None),
            default=None,
        ),
        "tracking_recovery_s_max": max(
            (item["tracking_recovery_s"] for item in scenario_results if item["tracking_recovery_s"] is not None),
            default=None,
        ),
        "peak_pitch_deg_max": float(np.max(peak_pitch_deg)),
        "peak_roll_deg_max": float(np.max(peak_roll_deg)),
        "peak_wheel_speed_rad_s_max": float(np.max(peak_wheel_speed)),
        "post_vx_rmse": aggregate_requested_vx_rmse,
        "post_vx_com_rmse": safe_rmse(
            float(np.sum(post_vx_com_squared_error)), total_post_samples
        ),
        "post_vx_odometry_rmse": safe_rmse(
            float(np.sum(post_vx_odometry_squared_error)), total_post_samples
        ),
        "post_wz_rmse": aggregate_requested_wz_rmse,
        "post_admitted_vx_rmse": aggregate_admitted_vx_rmse,
        "post_admitted_wz_rmse": aggregate_admitted_wz_rmse,
        "selected_post_vx_rmse": aggregate_selected_vx_rmse,
        "selected_post_wz_rmse": aggregate_selected_wz_rmse,
        "path_progress_scale_min": float(np.min(path_progress_scale_min)),
        "path_progress_scale_mean": float(
            np.sum(path_progress_scale_sum)
            / max(int(np.sum(path_progress_scale_samples)), 1)
        ),
        "action_saturation_ratio": int(np.sum(saturated_actions))
        / max(int(np.sum(action_samples)), 1),
        "wheel_mix_saturation_ratio": int(np.sum(wheel_mix_saturated))
        / max(int(np.sum(wheel_mix_samples)), 1),
        "post_wheel_mix_saturation_ratio": int(np.sum(post_wheel_mix_saturated))
        / max(int(np.sum(post_wheel_mix_samples)), 1),
        "peak_wheel_command_preclip_max": float(np.max(peak_wheel_command_preclip)),
        "post_vx_mean": float(np.sum(post_vx_sum) / max(total_post_samples, 1)),
        "post_vx_odometry_mean": float(
            np.sum(post_vx_odometry_sum) / max(total_post_samples, 1)
        ),
        "post_wz_mean": float(np.sum(post_wz_sum) / max(total_post_samples, 1)),
        "riser_plant": riser_plant_summary,
        "direction_tracking": direction_tracking,
        "direction_contract_complete": direction_contract_complete,
        "direction_speed_asymmetry_mps": direction_speed_asymmetry_mps,
    }
    result = {
        "schema": (
            "recomo_two_wheel_riser_cascaded_lqr_tracking_push_gate_v1"
            if args.robot_form == "riser"
            else "recomo_two_wheel_cascaded_lqr_tracking_push_gate_v4"
        ),
        "seed": args.seed,
        "robot_form": args.robot_form,
        "robot_asset_usd": str(cfg.robot_cfg.spawn.usd_path),
        "tracking_reference": args.tracking_reference,
        "controller_profile": args.controller_profile,
        "gains": str(args.gains.resolve()),
        "selected_gain_scale": gain_data["selected_gain_scale"],
        "controller": {
            "vx_kp": config.vx_kp,
            "vx_ki": config.vx_ki,
            "wz_kp": config.wz_kp,
            "wz_ki": config.wz_ki,
            "wz_feedforward": config.wz_feedforward,
            "vx_integral_limit": config.vx_integral_limit,
            "wz_integral_limit": config.wz_integral_limit,
            "governor_include_opposing_bias": config.governor_include_opposing_bias,
            "limit_total_pitch_reference": config.limit_total_pitch_reference,
            "reset_opposing_vx_integral_on_directional_deficit": (
                config.reset_opposing_vx_integral_on_directional_deficit
            ),
            "vx_integral_reset_reference_deadband_mps": (
                config.vx_integral_reset_reference_deadband_mps
            ),
            "use_root_velocity_outer_feedback": (
                args.use_root_velocity_outer_feedback
            ),
            "pitch_bias_adaptation_rate": config.pitch_bias_adaptation_rate,
            "pitch_bias_limit_deg": float(np.degrees(config.pitch_bias_limit_rad)),
            "vx_reference_slew_rate_m_s2": config.vx_reference_slew_rate_m_s2,
            "wz_reference_slew_rate_rad_s2": config.wz_reference_slew_rate_rad_s2,
            "path_progress_governor_enabled": config.path_progress_governor_enabled,
            "governor_bias_start_deg": float(np.degrees(config.governor_bias_start_rad)),
            "governor_bias_full_deg": float(np.degrees(config.governor_bias_full_rad)),
            "governor_minimum_progress_scale": config.governor_minimum_progress_scale,
            "pitch_reference_limit_deg": float(
                np.degrees(config.pitch_reference_limit_rad)
            ),
            "action_limit": config.action_limit,
            "controller_hz": POLICY_HZ / control_interval,
        },
        "command": {
            "start_step": args.command_start_step,
            "vx_m_s": vx_commands.tolist(),
            "wz_rad_s": wz_commands.tolist(),
        },
        "push": {
            "start_step": args.push_start_step,
            "duration_steps": args.push_duration_steps,
            "duration_s": args.push_duration_steps / POLICY_HZ,
            "forces_x_n": push_forces_n.tolist(),
            "application_height_above_base_com_m": args.push_height_m,
            "application": "global_x_force_plus_equivalent_global_y_pitch_torque",
        },
        "plant_uncertainty": {
            "profile": args.plant_uncertainty_profile,
            "variation_count": len(plant_variations),
            "runtime": plant_runtime,
            "mass_interpretation": (
                "uniform scale of all current rigid-body masses; stress cases are diagnostic, "
                "not alternate validated mass distributions"
            ),
            "inertia_interpretation": "default inertia scaled by mass scale and independent inertia scale",
            "com_interpretation": "local base_link COM offset",
            "torque_interpretation": "per-environment reduction of the requested normalized action",
            "delay_interpretation": "whole policy steps at 200 Hz",
        },
        "thresholds": {
            "tracking_reference": args.tracking_reference,
            "minimum_path_progress_scale": args.minimum_path_progress_scale,
            "minimum_success_rate": args.minimum_success_rate,
            "maximum_recovery_s": args.maximum_recovery_s,
            "recovery_tilt_deg": args.recovery_tilt_deg,
            "recovery_angular_rate": args.recovery_angular_rate,
            "balance_tilt_margin_deg": args.balance_tilt_margin_deg,
            "balance_angular_rate_margin": args.balance_angular_rate_margin,
            "recovery_vx_error": args.recovery_vx_error,
            "recovery_wz_error": args.recovery_wz_error,
            "maximum_tilt_deg": args.maximum_tilt_deg,
            "maximum_post_vx_rmse": args.maximum_post_vx_rmse,
            "maximum_post_wz_rmse": args.maximum_post_wz_rmse,
            "maximum_saturation_ratio": args.maximum_saturation_ratio,
            "maximum_riser_hold_error_m": args.maximum_riser_hold_error_m,
            "maximum_gimbal_hold_error_deg": (
                args.maximum_gimbal_hold_error_deg
            ),
            "maximum_direction_speed_asymmetry_mps": (
                args.maximum_direction_speed_asymmetry_mps
            ),
        },
        "summary": summary,
        "scenarios": scenario_results,
        "controller_observations": [
            "pitch",
            "pitch_rate",
            "mean_wheel_position",
            "mean_wheel_velocity",
            "wheel_velocity_difference",
            "yaw_rate",
            "vx_ref",
            "wz_ref",
        ],
        "evaluation_only_truth": ["base_link_vx", "base_roll"],
        "learned_action_applied": False,
        "residual_dataset": None,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "training_started": False,
    }
    result["passed"] = bool(
        success_rate >= args.minimum_success_rate
        and summary["selected_post_vx_rmse"] <= args.maximum_post_vx_rmse
        and summary["selected_post_wz_rmse"] <= args.maximum_post_wz_rmse
        and summary["action_saturation_ratio"] <= args.maximum_saturation_ratio
        and summary["path_progress_scale_min"] >= args.minimum_path_progress_scale
        and (
            args.robot_form != "riser"
            or (
                summary["riser_plant"]["riser_hold_error_max_m"]
                <= args.maximum_riser_hold_error_m
                and summary["riser_plant"]["gimbal_hold_error_max_deg"]
                <= args.maximum_gimbal_hold_error_deg
                and summary["direction_contract_complete"] is True
                and summary["direction_speed_asymmetry_mps"]
                <= args.maximum_direction_speed_asymmetry_mps
            )
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    import threading

    watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    watchdog.cancel()
raise SystemExit(exit_code)
