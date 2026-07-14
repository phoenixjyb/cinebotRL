#!/usr/bin/env python3
"""Evaluate frozen chassis tracking under deterministic upper-body pushes."""

from __future__ import annotations

import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("--gains", type=Path, required=True)
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
parser.add_argument("--vx-kp", type=float)
parser.add_argument("--vx-ki", type=float)
parser.add_argument("--wz-kp", type=float)
parser.add_argument("--wz-ki", type=float)
parser.add_argument("--wz-feedforward", type=float)
parser.add_argument("--vx-integral-limit", type=float)
parser.add_argument("--wz-integral-limit", type=float)
parser.add_argument("--pitch-reference-limit-deg", type=float)
parser.add_argument(
    "--plant-uncertainty-profile",
    choices=("nominal", "provisional_prior_v1", "diagnostic_v1"),
    default="nominal",
)
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    CascadedLQRConfig,
    PlantVariation,
    cascaded_lqr_action,
    diagnostic_plant_variations,
    provisional_plant_variations,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0


def parse_csv(value: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"expected finite comma-separated values, got {value!r}")
    return result


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
    plant_runtime = {
        "body_names": list(unwrapped.robot.body_names),
        "nominal_total_mass_kg": float(
            unwrapped.robot.data.default_mass[0].sum().item()
        ),
        "nominal_body_masses_kg": unwrapped.robot.data.default_mass[0].tolist(),
    }
    if args.plant_uncertainty_profile != "nominal":
        plant_runtime = apply_plant_variations(unwrapped, scenario_plants)
    set_push_wrench(unwrapped, np.zeros(args.num_envs))
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    active = np.ones(args.num_envs, dtype=bool)
    survived = np.zeros(args.num_envs, dtype=bool)
    duration_steps = np.full(args.num_envs, args.horizon_steps, dtype=np.int64)
    integrals = np.zeros((args.num_envs, 4), dtype=np.float64)
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
            "pitch_reference_limit_rad": (
                np.radians(args.pitch_reference_limit_deg)
                if args.pitch_reference_limit_deg is not None
                else None
            ),
        }.items()
        if value is not None
    }
    config = CascadedLQRConfig(action_limit=action_limit, **controller_overrides)

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
    post_samples = np.zeros(args.num_envs, dtype=np.int64)
    pre_pitch_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_roll_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_pitch_rate_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_roll_rate_max = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_pitch_deg = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_pitch_rate = np.zeros(args.num_envs, dtype=np.float64)
    pre_command_wheel_speed = np.zeros(args.num_envs, dtype=np.float64)

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

        if step % control_interval == 0:
            requested_action_np, integrals, _ = cascaded_lqr_action(
                current_states,
                step_vx_ref,
                step_wz_ref,
                gain,
                integrals,
                control_dt=control_interval / POLICY_HZ,
                config=config,
            )
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
        vx_error = np.abs(vx_truth - step_vx_ref)
        wz_error = np.abs(wz_truth - step_wz_ref)

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
                (vx_error <= args.recovery_vx_error)
                & (wz_error <= args.recovery_wz_error)
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
            post_samples[active] += 1

        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        finished = active & (terminated_np | truncated_np)
        duration_steps[finished] = step + 1
        survived[finished] = truncated_np[finished]
        active[finished] = False
        integrals[~active] = 0.0
        if not np.any(active):
            break

    set_push_wrench(unwrapped, np.zeros(args.num_envs))
    survived[active] = True
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
        passed = bool(
            survived[index]
            and balance_recovery_s is not None
            and balance_recovery_s <= args.maximum_recovery_s
            and tracking_recovery_s is not None
            and tracking_recovery_s <= args.maximum_recovery_s
            and max(peak_pitch_deg[index], peak_roll_deg[index]) <= args.maximum_tilt_deg
            and post_vx_rmse <= args.maximum_post_vx_rmse
            and post_wz_rmse <= args.maximum_post_wz_rmse
            and saturation_ratio <= args.maximum_saturation_ratio
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
                "action_saturation_ratio": float(saturation_ratio),
                "passed": passed,
            }
        )

    total_post_samples = int(np.sum(post_samples))
    success_rate = float(np.mean([item["passed"] for item in scenario_results]))
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
        "post_vx_rmse": safe_rmse(float(np.sum(post_vx_squared_error)), total_post_samples),
        "post_vx_com_rmse": safe_rmse(
            float(np.sum(post_vx_com_squared_error)), total_post_samples
        ),
        "post_vx_odometry_rmse": safe_rmse(
            float(np.sum(post_vx_odometry_squared_error)), total_post_samples
        ),
        "post_wz_rmse": safe_rmse(float(np.sum(post_wz_squared_error)), total_post_samples),
        "action_saturation_ratio": int(np.sum(saturated_actions))
        / max(int(np.sum(action_samples)), 1),
    }
    result = {
        "schema": "recomo_two_wheel_cascaded_lqr_tracking_push_gate_v2",
        "seed": args.seed,
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
            "pitch_bias_adaptation_rate": config.pitch_bias_adaptation_rate,
            "pitch_bias_limit_deg": float(np.degrees(config.pitch_bias_limit_rad)),
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
        "training_started": False,
    }
    result["passed"] = bool(
        success_rate >= args.minimum_success_rate
        and summary["post_vx_rmse"] <= args.maximum_post_vx_rmse
        and summary["post_wz_rmse"] <= args.maximum_post_wz_rmse
        and summary["action_saturation_ratio"] <= args.maximum_saturation_ratio
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
