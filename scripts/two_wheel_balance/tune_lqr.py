#!/usr/bin/env python3
"""Identify and tune a scripted LQR on the corrected nominal Isaac plant."""

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
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--horizon-steps", type=int, default=2000)
parser.add_argument("--control-interval-steps", type=int, default=4)
parser.add_argument("--seed", type=int, default=20260713)
parser.add_argument("--pitch-deg", default="-8,-5,-2,2,5,8")
parser.add_argument("--yaw-rate", default="-0.3,0,0.3")
parser.add_argument("--gain-scales", default="0.3,0.4,0.5,0.6,0.8")
parser.add_argument("--q-diag", default="10000,0.1,0,0.1,0.5,0")
parser.add_argument("--r-diag", default="0.1,0.5")
parser.add_argument("--action-limit", type=float, default=0.8)
parser.add_argument("--minimum-success-rate", type=float, default=0.95)
parser.add_argument("--maximum-pitch-p95-deg", type=float, default=10.0)
parser.add_argument("--maximum-saturation-ratio", type=float, default=0.10)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    controllability_matrix,
    lqr_action,
    solve_discrete_lqr,
)
from task_spec import register_isaac_lab_tasks


STATE_EPSILON = np.array([0.005, 0.05, 0.01, 0.10, 0.10, 0.05], dtype=np.float64)
ACTION_EPSILON = np.array([0.02, 0.02], dtype=np.float64)
LONGITUDINAL_STATE_INDICES = (0, 1, 3)
YAW_STATE_INDICES = (4,)


def parse_csv_floats(value: str, expected: int | None = None) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if expected is not None and result.shape != (expected,):
        raise ValueError(f"expected {expected} comma-separated values, got {value!r}")
    if not np.isfinite(result).all():
        raise ValueError(f"values must be finite: {value!r}")
    return result


def jsonable_matrix(value: np.ndarray) -> list:
    return np.asarray(value).tolist()


def state_tensor(env) -> torch.Tensor:
    state = env.unwrapped._state_terms()
    return torch.stack(
        (
            state["pitch"],
            state["pitch_rate"],
            state["mean_wheel_position"],
            state["mean_wheel_velocity"],
            state["wheel_velocity_difference"],
            state["yaw_rate"],
        ),
        dim=-1,
    )


def write_states(env, states: np.ndarray) -> None:
    unwrapped = env.unwrapped
    device = unwrapped.device
    env_ids = torch.arange(unwrapped.num_envs, device=device, dtype=torch.long)
    state_values = torch.as_tensor(states, dtype=torch.float32, device=device)
    root_state = unwrapped.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += unwrapped.scene.env_origins[env_ids]
    half_pitch = 0.5 * state_values[:, 0]
    root_state[:, 3] = torch.cos(half_pitch)
    root_state[:, 4] = 0.0
    root_state[:, 5] = torch.sin(half_pitch)
    root_state[:, 6] = 0.0
    root_state[:, 7:] = 0.0
    root_state[:, 11] = state_values[:, 1]
    root_state[:, 12] = state_values[:, 5]

    joint_pos = unwrapped.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = unwrapped.robot.data.default_joint_vel[env_ids].clone()
    wheel_ids = unwrapped._wheel_joint_idx
    joint_pos[:, wheel_ids[0]] = state_values[:, 2]
    joint_pos[:, wheel_ids[1]] = state_values[:, 2]
    joint_vel[:, wheel_ids[0]] = state_values[:, 3] - 0.5 * state_values[:, 4]
    joint_vel[:, wheel_ids[1]] = state_values[:, 3] + 0.5 * state_values[:, 4]

    unwrapped.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    unwrapped.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    unwrapped.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    unwrapped.actions.zero_()
    unwrapped.policy_actions.zero_()
    unwrapped.previous_actions.zero_()
    unwrapped.wheel_efforts.zero_()
    unwrapped.episode_length_buf.zero_()


def identify_discrete_model(env) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    state_dim = len(LQR_STATE_NAMES)
    action_dim = len(ACTION_NAMES)
    required_envs = 2 * (state_dim + action_dim)
    if env.unwrapped.num_envs < required_envs:
        raise ValueError(f"identification needs at least {required_envs} environments")

    env.reset(seed=args.seed)
    initial_states = np.zeros((env.unwrapped.num_envs, state_dim), dtype=np.float64)
    actions = np.zeros((env.unwrapped.num_envs, action_dim), dtype=np.float64)
    pairs: list[tuple[str, int, int, int]] = []
    cursor = 0
    for state_index, epsilon in enumerate(STATE_EPSILON):
        initial_states[cursor, state_index] = epsilon
        initial_states[cursor + 1, state_index] = -epsilon
        pairs.append(("state", state_index, cursor, cursor + 1))
        cursor += 2
    for action_index, epsilon in enumerate(ACTION_EPSILON):
        actions[cursor, action_index] = epsilon
        actions[cursor + 1, action_index] = -epsilon
        pairs.append(("action", action_index, cursor, cursor + 1))
        cursor += 2

    write_states(env, initial_states)
    measured_initial = state_tensor(env).detach().cpu().numpy().astype(np.float64)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=env.unwrapped.device)
    for _ in range(args.control_interval_steps):
        obs, _, terminated, truncated, _ = env.step(action_tensor)
        if bool(torch.any(terminated[:required_envs])) or bool(torch.any(truncated[:required_envs])):
            raise RuntimeError("small-signal identification unexpectedly terminated an environment")
    next_states = obs["policy"][:, :state_dim].detach().cpu().numpy().astype(np.float64)

    a = np.zeros((state_dim, state_dim), dtype=np.float64)
    b = np.zeros((state_dim, action_dim), dtype=np.float64)
    for kind, index, positive, negative in pairs:
        delta = next_states[positive] - next_states[negative]
        if kind == "state":
            denominator = measured_initial[positive, index] - measured_initial[negative, index]
            if abs(denominator) < 1e-12:
                denominator = 2.0 * STATE_EPSILON[index]
            a[:, index] = delta / denominator
        else:
            b[:, index] = delta / (2.0 * ACTION_EPSILON[index])

    controllability = controllability_matrix(a, b)
    longitudinal_a = a[np.ix_(LONGITUDINAL_STATE_INDICES, LONGITUDINAL_STATE_INDICES)]
    longitudinal_b = b[np.ix_(LONGITUDINAL_STATE_INDICES, (0,))]
    yaw_a = a[np.ix_(YAW_STATE_INDICES, YAW_STATE_INDICES)]
    yaw_b = b[np.ix_(YAW_STATE_INDICES, (1,))]
    longitudinal_controllability = controllability_matrix(longitudinal_a, longitudinal_b)
    yaw_controllability = controllability_matrix(yaw_a, yaw_b)
    diagnostics = {
        "state_names": list(LQR_STATE_NAMES),
        "action_names": list(ACTION_NAMES),
        "state_epsilon": STATE_EPSILON.tolist(),
        "action_epsilon": ACTION_EPSILON.tolist(),
        "control_interval_steps": args.control_interval_steps,
        "controller_hz": 200.0 / args.control_interval_steps,
        "measured_initial_pairs": measured_initial[:required_envs].tolist(),
        "open_loop_eigenvalues": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(a)],
        "controllability_rank": int(np.linalg.matrix_rank(controllability, tol=1e-7)),
        "controllability_condition": float(np.linalg.cond(controllability)),
        "controllability_singular_values": np.linalg.svd(controllability, compute_uv=False).tolist(),
        "longitudinal_controllability_rank": int(
            np.linalg.matrix_rank(longitudinal_controllability)
        ),
        "longitudinal_controllability_singular_values": np.linalg.svd(
            longitudinal_controllability, compute_uv=False
        ).tolist(),
        "longitudinal_state_names": [LQR_STATE_NAMES[index] for index in LONGITUDINAL_STATE_INDICES],
        "yaw_controllability_rank": int(np.linalg.matrix_rank(yaw_controllability)),
        "yaw_controllability_singular_values": np.linalg.svd(
            yaw_controllability, compute_uv=False
        ).tolist(),
        "yaw_state_names": [LQR_STATE_NAMES[index] for index in YAW_STATE_INDICES],
        "yaw_input_norm": float(np.linalg.norm(yaw_b)),
    }
    return a, b, diagnostics


def scenario_states(num_envs: int, pitches_deg: np.ndarray, yaw_rates: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    combinations = [(pitch, yaw_rate) for pitch in pitches_deg for yaw_rate in yaw_rates]
    states = np.zeros((num_envs, len(LQR_STATE_NAMES)), dtype=np.float64)
    scenarios: list[dict[str, float]] = []
    for index in range(num_envs):
        pitch_deg, yaw_rate = combinations[index % len(combinations)]
        states[index, 0] = math.radians(float(pitch_deg))
        states[index, 5] = float(yaw_rate)
        scenarios.append({"pitch_deg": float(pitch_deg), "yaw_rate": float(yaw_rate)})
    return states, scenarios


def evaluate_gain(
    env,
    gain: np.ndarray,
    gain_scale: float,
    initial_states: np.ndarray,
    scenarios: list[dict[str, float]],
) -> dict[str, object]:
    env.reset(seed=args.seed + int(round(gain_scale * 1000)))
    write_states(env, initial_states)
    current_states = initial_states.copy()
    active = np.ones(env.unwrapped.num_envs, dtype=bool)
    lengths = np.full(env.unwrapped.num_envs, args.horizon_steps, dtype=np.int64)
    successes = np.zeros(env.unwrapped.num_envs, dtype=bool)
    pitch_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    action_np = np.zeros((env.unwrapped.num_envs, len(ACTION_NAMES)), dtype=np.float32)

    for step in range(args.horizon_steps):
        if step % args.control_interval_steps == 0:
            action_np = lqr_action(
                current_states,
                gain_scale * gain,
                action_limit=args.action_limit,
            ).astype(np.float32)
        action_np[~active] = 0.0
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action_np, device=env.unwrapped.device)
        )
        next_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        active_indices = np.flatnonzero(active)
        if active_indices.size:
            pitch_samples.append(np.abs(next_states[active_indices, 0]))
            action_samples.append(np.abs(action_np[active_indices]))
        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        finished = active & (terminated_np | truncated_np)
        lengths[finished] = step + 1
        successes[finished] = truncated_np[finished]
        active[finished] = False
        current_states = next_states
        if not np.any(active):
            break
    successes[active] = True

    pitch_values = np.concatenate(pitch_samples) if pitch_samples else np.array([np.inf])
    action_values = np.concatenate(action_samples, axis=0) if action_samples else np.ones((1, 2))
    scenario_results = []
    for initial, scenario, length, success in zip(
        initial_states, scenarios, lengths, successes
    ):
        scenario_results.append(
            {
                **scenario,
                "initial_state": initial.tolist(),
                "duration_steps": int(length),
                "duration_s": float(length / 200.0),
                "success": bool(success),
            }
        )
    result = {
        "gain_scale": gain_scale,
        "episodes": int(len(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_duration_steps": float(np.mean(lengths)),
        "minimum_duration_steps": int(np.min(lengths)),
        "abs_pitch_p95_deg": float(np.degrees(np.percentile(pitch_values, 95))),
        "abs_pitch_max_deg": float(np.degrees(np.max(pitch_values))),
        "action_abs_p95": float(np.percentile(action_values, 95)),
        "action_saturation_ratio": float(np.mean(action_values >= args.action_limit - 1e-6)),
        "scenarios": scenario_results,
    }
    result["passed"] = bool(
        result["success_rate"] >= args.minimum_success_rate
        and result["abs_pitch_p95_deg"] <= args.maximum_pitch_p95_deg
        and result["action_saturation_ratio"] <= args.maximum_saturation_ratio
    )
    return result


def main() -> int:
    if args.num_envs < 16:
        raise ValueError("--num-envs must be at least 16 for central-difference identification")
    if args.horizon_steps < 1:
        raise ValueError("--horizon-steps must be positive")
    if args.control_interval_steps < 1:
        raise ValueError("--control-interval-steps must be positive")
    q_diag = parse_csv_floats(args.q_diag, len(LQR_STATE_NAMES))
    r_diag = parse_csv_floats(args.r_diag, len(ACTION_NAMES))
    gain_scales = parse_csv_floats(args.gain_scales)
    pitches_deg = parse_csv_floats(args.pitch_deg)
    yaw_rates = parse_csv_floats(args.yaw_rate)
    if np.any(q_diag < 0.0) or np.any(r_diag <= 0.0) or np.any(gain_scales <= 0.0):
        raise ValueError("Q must be nonnegative; R and gain scales must be positive")

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

    a, b, identification = identify_discrete_model(env)
    print(
        "identified model: "
        f"full_rank={identification['controllability_rank']} "
        f"longitudinal_rank={identification['longitudinal_controllability_rank']} "
        f"yaw_rank={identification['yaw_controllability_rank']} "
        f"yaw_input_norm={identification['yaw_input_norm']:.6g}",
        flush=True,
    )
    if identification["longitudinal_controllability_rank"] != len(LONGITUDINAL_STATE_INDICES):
        raise RuntimeError(f"longitudinal model lacks sufficient authority: {identification}")
    if identification["yaw_controllability_rank"] < 1:
        raise RuntimeError(f"yaw input has no measurable authority: {identification}")
    q = np.diag(q_diag)
    r = np.diag(r_diag)
    longitudinal_solution = solve_discrete_lqr(
        a[np.ix_(LONGITUDINAL_STATE_INDICES, LONGITUDINAL_STATE_INDICES)],
        b[np.ix_(LONGITUDINAL_STATE_INDICES, (0,))],
        q[np.ix_(LONGITUDINAL_STATE_INDICES, LONGITUDINAL_STATE_INDICES)],
        r[:1, :1],
    )
    yaw_solution = solve_discrete_lqr(
        a[np.ix_(YAW_STATE_INDICES, YAW_STATE_INDICES)],
        b[np.ix_(YAW_STATE_INDICES, (1,))],
        q[np.ix_(YAW_STATE_INDICES, YAW_STATE_INDICES)],
        r[1:2, 1:2],
    )
    gain = np.zeros((len(ACTION_NAMES), len(LQR_STATE_NAMES)), dtype=np.float64)
    gain[0, list(LONGITUDINAL_STATE_INDICES)] = longitudinal_solution.gain[0]
    gain[1, list(YAW_STATE_INDICES)] = yaw_solution.gain[0]
    closed_loop_eigenvalues = np.linalg.eigvals(a - b @ gain)
    spectral_radius = float(np.max(np.abs(closed_loop_eigenvalues)))
    if spectral_radius > 1.00001:
        raise RuntimeError(f"identified LQR is not stable: spectral radius {spectral_radius}")
    print(f"closed-loop spectral radius: {spectral_radius:.9f}", flush=True)

    initial_states, scenarios = scenario_states(args.num_envs, pitches_deg, yaw_rates)
    evaluations = [
        evaluate_gain(env, gain, float(scale), initial_states, scenarios)
        for scale in gain_scales
    ]
    for evaluation in evaluations:
        print(
            f"scale={evaluation['gain_scale']:.3f} "
            f"success={evaluation['success_rate']:.3f} "
            f"duration={evaluation['mean_duration_steps']:.1f} "
            f"pitch_p95={evaluation['abs_pitch_p95_deg']:.3f} "
            f"sat={evaluation['action_saturation_ratio']:.4f}",
            flush=True,
        )
    selected = max(
        evaluations,
        key=lambda value: (
            value["passed"],
            value["success_rate"],
            value["mean_duration_steps"],
            -value["abs_pitch_p95_deg"],
            -value["action_saturation_ratio"],
        ),
    )
    selected_gain = selected["gain_scale"] * gain

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_model = {
        "schema": "recomo_two_wheel_identified_linear_model_v1",
        "seed": args.seed,
        "policy_hz": 200.0,
        "controller_hz": 200.0 / args.control_interval_steps,
        "a": jsonable_matrix(a),
        "b": jsonable_matrix(b),
        "q_diag": q_diag.tolist(),
        "r_diag": r_diag.tolist(),
        **identification,
    }
    gains = {
        "schema": "recomo_two_wheel_lqr_gain_v1",
        "state_names": list(LQR_STATE_NAMES),
        "action_names": list(ACTION_NAMES),
        "synthesis": "three_state_balance_plus_differential_wheel_velocity",
        "regulated_state_names": {
            "longitudinal": [LQR_STATE_NAMES[index] for index in LONGITUDINAL_STATE_INDICES],
            "yaw": [LQR_STATE_NAMES[index] for index in YAW_STATE_INDICES],
        },
        "nominal_gain": jsonable_matrix(gain),
        "selected_gain_scale": selected["gain_scale"],
        "selected_gain": jsonable_matrix(selected_gain),
        "action_limit": args.action_limit,
        "control_interval_steps": args.control_interval_steps,
        "riccati_iterations": {
            "longitudinal": longitudinal_solution.iterations,
            "yaw": yaw_solution.iterations,
        },
        "riccati_solver": {
            "longitudinal": longitudinal_solution.solver,
            "yaw": yaw_solution.solver,
        },
        "riccati_residual_max_abs": {
            "longitudinal": longitudinal_solution.residual_max_abs,
            "yaw": yaw_solution.residual_max_abs,
        },
        "closed_loop_eigenvalues": [
            [float(value.real), float(value.imag)] for value in closed_loop_eigenvalues
        ],
        "closed_loop_spectral_radius": spectral_radius,
    }
    gate = {
        "schema": "recomo_two_wheel_lqr_nominal_gate_v1",
        "nominal_assumptions": {
            "wheel_diameter_m": 0.2032,
            "wheel_track_m": 0.620,
            "model_mass_kg": 26.0,
            "torque_limit_nm": cfg.torque_limit_nm,
            "status": "provisional_simulation_only",
        },
        "thresholds": {
            "minimum_success_rate": args.minimum_success_rate,
            "maximum_pitch_p95_deg": args.maximum_pitch_p95_deg,
            "maximum_saturation_ratio": args.maximum_saturation_ratio,
            "horizon_steps": args.horizon_steps,
            "horizon_s": args.horizon_steps / 200.0,
        },
        "selected": selected,
        "passed": bool(selected["passed"]),
        "training_started": False,
    }
    (args.output_dir / "linear_model.json").write_text(
        json.dumps(linear_model, indent=2), encoding="utf-8"
    )
    (args.output_dir / "lqr_gains.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")
    (args.output_dir / "candidate_evaluations.json").write_text(
        json.dumps(evaluations, indent=2), encoding="utf-8"
    )
    (args.output_dir / "lqr_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"linear_model": linear_model, "gains": gains, "gate": gate}, indent=2))
    env.close()
    return 0 if gate["passed"] else 2


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
