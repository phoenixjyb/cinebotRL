#!/usr/bin/env python3
"""Evaluate the frozen nominal LQR under deterministic fore/aft pushes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--horizon-steps", type=int, default=2000)
parser.add_argument("--push-start-step", type=int, default=200)
parser.add_argument("--push-duration-steps", type=int, default=20)
parser.add_argument("--push-forces-n", default="-60,-40,-20,20,40,60")
parser.add_argument("--push-height-m", type=float, default=0.5)
parser.add_argument("--initial-pitch-deg", default="0")
parser.add_argument("--recovery-pitch-deg", type=float, default=2.0)
parser.add_argument("--recovery-pitch-rate", type=float, default=0.2)
parser.add_argument("--recovery-hold-steps", type=int, default=50)
parser.add_argument("--maximum-recovery-s", type=float, default=2.0)
parser.add_argument("--maximum-pitch-deg", type=float, default=15.0)
parser.add_argument("--maximum-saturation-ratio", type=float, default=0.10)
parser.add_argument("--minimum-success-rate", type=float, default=0.95)
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
    lqr_action,
    recovery_window_steps,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0


def parse_csv(value: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"expected finite comma-separated values, got {value!r}")
    return result


def write_initial_states(env, states: np.ndarray) -> None:
    unwrapped = env.unwrapped
    device = unwrapped.device
    env_ids = torch.arange(unwrapped.num_envs, device=device, dtype=torch.long)
    values = torch.as_tensor(states, dtype=torch.float32, device=device)
    root_state = unwrapped.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += unwrapped.scene.env_origins[env_ids]
    half_pitch = 0.5 * values[:, 0]
    root_state[:, 3] = torch.cos(half_pitch)
    root_state[:, 4] = 0.0
    root_state[:, 5] = torch.sin(half_pitch)
    root_state[:, 6] = 0.0
    root_state[:, 7:] = 0.0

    joint_pos = unwrapped.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = unwrapped.robot.data.default_joint_vel[env_ids].clone()
    unwrapped.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    unwrapped.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    unwrapped.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    unwrapped.actions.zero_()
    unwrapped.policy_actions.zero_()
    unwrapped.previous_actions.zero_()
    unwrapped.wheel_efforts.zero_()
    unwrapped.episode_length_buf.zero_()


def set_push_forces(env, force_x_n: np.ndarray, push_height_m: float) -> None:
    unwrapped = env.unwrapped
    forces = torch.zeros((unwrapped.num_envs, 1, 3), device=unwrapped.device)
    torques = torch.zeros_like(forces)
    forces[:, 0, 0] = torch.as_tensor(force_x_n, dtype=torch.float32, device=unwrapped.device)
    # Equivalent wrench for a horizontal force applied above the base COM.
    torques[:, 0, 1] = forces[:, 0, 0] * push_height_m
    unwrapped.robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=unwrapped._base_body_idx,
        is_global=True,
    )


def build_scenarios(
    num_envs: int,
    pitches_deg: np.ndarray,
    push_forces_n: np.ndarray,
    push_duration_steps: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    combinations = [(pitch, force) for pitch in pitches_deg for force in push_forces_n]
    states = np.zeros((num_envs, len(LQR_STATE_NAMES)), dtype=np.float64)
    scenarios: list[dict[str, float]] = []
    for index in range(num_envs):
        pitch_deg, force_n = combinations[index % len(combinations)]
        states[index, 0] = math.radians(float(pitch_deg))
        scenarios.append(
            {
                "initial_pitch_deg": float(pitch_deg),
                "push_force_x_n": float(force_n),
                "push_impulse_x_ns": float(force_n * push_duration_steps / POLICY_HZ),
            }
        )
    return states, scenarios


def main() -> int:
    if args.num_envs < 1 or args.horizon_steps < 1:
        raise ValueError("--num-envs and --horizon-steps must be positive")
    if not 0 <= args.push_start_step < args.horizon_steps:
        raise ValueError("--push-start-step must be inside the episode")
    if args.push_duration_steps < 1:
        raise ValueError("--push-duration-steps must be positive")
    if args.push_height_m < 0.0:
        raise ValueError("--push-height-m must be non-negative")
    push_end_step = args.push_start_step + args.push_duration_steps
    if push_end_step >= args.horizon_steps:
        raise ValueError("push window must end before the episode horizon")
    if args.recovery_hold_steps < 1:
        raise ValueError("--recovery-hold-steps must be positive")

    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])
    action_limit = float(gain_data["action_limit"])
    push_forces_n = parse_csv(args.push_forces_n)
    pitches_deg = parse_csv(args.initial_pitch_deg)
    initial_states, scenarios = build_scenarios(
        args.num_envs,
        pitches_deg,
        push_forces_n,
        args.push_duration_steps,
    )
    scenario_forces = np.asarray([item["push_force_x_n"] for item in scenarios])
    initial_condition_only, evaluation_start_step, recovery_start_step = (
        recovery_window_steps(scenario_forces, args.push_start_step, push_end_step)
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
    env.reset(seed=args.seed)
    write_initial_states(env, initial_states)
    set_push_forces(env, np.zeros(args.num_envs), args.push_height_m)

    current_states = initial_states.copy()
    action_np = np.zeros((args.num_envs, len(ACTION_NAMES)), dtype=np.float32)
    active = np.ones(args.num_envs, dtype=bool)
    survived = np.zeros(args.num_envs, dtype=bool)
    recovered = np.zeros(args.num_envs, dtype=bool)
    recovery_hold = np.zeros(args.num_envs, dtype=np.int64)
    recovery_steps = np.full(args.num_envs, -1, dtype=np.int64)
    duration_steps = np.full(args.num_envs, args.horizon_steps, dtype=np.int64)
    peak_pitch_deg = np.degrees(np.abs(initial_states[:, 0]))
    peak_pitch_rate = np.zeros(args.num_envs, dtype=np.float64)
    peak_wheel_speed = np.zeros(args.num_envs, dtype=np.float64)
    saturated_actions = np.zeros(args.num_envs, dtype=np.int64)
    action_samples = np.zeros(args.num_envs, dtype=np.int64)

    for step in range(args.horizon_steps):
        if step == args.push_start_step:
            set_push_forces(env, scenario_forces, args.push_height_m)
        elif step == push_end_step:
            set_push_forces(env, np.zeros(args.num_envs), args.push_height_m)

        if step % control_interval == 0:
            action_np = lqr_action(current_states, gain, action_limit=action_limit).astype(
                np.float32
            )
        action_np[~active] = 0.0
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action_np, device=env.unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = env.unwrapped._state_terms()
        pitch_deg = np.degrees(np.abs(current_states[:, 0]))
        pitch_rate = np.abs(current_states[:, 1])
        wheel_speed = state["max_abs_wheel_velocity"].detach().cpu().numpy()
        if step >= evaluation_start_step:
            peak_pitch_deg[active] = np.maximum(peak_pitch_deg[active], pitch_deg[active])
            peak_pitch_rate[active] = np.maximum(peak_pitch_rate[active], pitch_rate[active])
            peak_wheel_speed[active] = np.maximum(peak_wheel_speed[active], wheel_speed[active])
            saturated_actions[active] += np.count_nonzero(
                np.abs(action_np[active]) >= action_limit - 1e-6,
                axis=1,
            )
            action_samples[active] += len(ACTION_NAMES)

        if step >= recovery_start_step:
            inside_recovery = (
                (pitch_deg <= args.recovery_pitch_deg)
                & (pitch_rate <= args.recovery_pitch_rate)
                & active
            )
            recovery_hold[inside_recovery] += 1
            recovery_hold[~inside_recovery] = 0
            newly_recovered = active & (~recovered) & (
                recovery_hold >= args.recovery_hold_steps
            )
            recovery_steps[newly_recovered] = (
                step - recovery_start_step - args.recovery_hold_steps + 1
            )
            recovered[newly_recovered] = True

        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        finished = active & (terminated_np | truncated_np)
        duration_steps[finished] = step + 1
        survived[finished] = truncated_np[finished]
        active[finished] = False
        if not np.any(active):
            break

    set_push_forces(env, np.zeros(args.num_envs), args.push_height_m)
    survived[active] = True
    env.close()

    scenario_results = []
    for index, scenario in enumerate(scenarios):
        recovery_s = (
            float(recovery_steps[index] / POLICY_HZ) if recovery_steps[index] >= 0 else None
        )
        passed = bool(
            survived[index]
            and recovered[index]
            and recovery_s is not None
            and recovery_s <= args.maximum_recovery_s
            and peak_pitch_deg[index] <= args.maximum_pitch_deg
        )
        scenario_results.append(
            {
                **scenario,
                "survived": bool(survived[index]),
                "recovered": bool(recovered[index]),
                "recovery_s": recovery_s,
                "duration_steps": int(duration_steps[index]),
                "peak_pitch_deg": float(peak_pitch_deg[index]),
                "peak_pitch_rate": float(peak_pitch_rate[index]),
                "peak_wheel_speed_rad_s": float(peak_wheel_speed[index]),
                "action_saturation_ratio": float(
                    saturated_actions[index] / max(action_samples[index], 1)
                ),
                "passed": passed,
            }
        )

    success_rate = float(np.mean([item["passed"] for item in scenario_results]))
    recovered_times = [
        item["recovery_s"] for item in scenario_results if item["recovery_s"] is not None
    ]
    total_saturated = int(np.sum(saturated_actions))
    total_action_samples = int(np.sum(action_samples))
    result = {
        "schema": "recomo_two_wheel_lqr_push_gate_v2",
        "seed": args.seed,
        "gains": str(args.gains.resolve()),
        "selected_gain_scale": gain_data["selected_gain_scale"],
        "policy_hz": POLICY_HZ,
        "control_interval_steps": control_interval,
        "controller_hz": POLICY_HZ / control_interval,
        "push": {
            "disturbance_mode": (
                "initial_pitch_only" if initial_condition_only else "external_push"
            ),
            "start_step": args.push_start_step,
            "duration_steps": args.push_duration_steps,
            "duration_s": args.push_duration_steps / POLICY_HZ,
            "application_height_above_base_com_m": args.push_height_m,
            "application": "global_x_force_plus_equivalent_base_link_pitch_torque",
        },
        "measurement": {
            "evaluation_start_step": evaluation_start_step,
            "recovery_start_step": recovery_start_step,
            "initial_pitch_included_in_peak": True,
        },
        "thresholds": {
            "minimum_success_rate": args.minimum_success_rate,
            "maximum_recovery_s": args.maximum_recovery_s,
            "maximum_pitch_deg": args.maximum_pitch_deg,
            "maximum_saturation_ratio": args.maximum_saturation_ratio,
        },
        "summary": {
            "scenarios": len(scenario_results),
            "success_rate": success_rate,
            "survival_rate": float(np.mean([item["survived"] for item in scenario_results])),
            "recovery_rate": float(np.mean([item["recovered"] for item in scenario_results])),
            "recovery_s_max": float(max(recovered_times)) if recovered_times else None,
            "peak_pitch_deg_max": float(max(item["peak_pitch_deg"] for item in scenario_results)),
            "peak_wheel_speed_rad_s_max": float(
                max(item["peak_wheel_speed_rad_s"] for item in scenario_results)
            ),
            "action_saturation_ratio": total_saturated / max(total_action_samples, 1),
        },
        "scenarios": scenario_results,
        "training_started": False,
    }
    result["passed"] = bool(
        success_rate >= args.minimum_success_rate
        and result["summary"]["action_saturation_ratio"]
        <= args.maximum_saturation_ratio
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
