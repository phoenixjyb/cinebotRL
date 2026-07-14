#!/usr/bin/env python3
"""Tune a bounded chassis-tracking outer loop around the frozen balance LQR."""

from __future__ import annotations

import argparse
import itertools
import json
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
parser.add_argument("--vx-commands", default="-0.2,0,0.2")
parser.add_argument("--wz-commands", default="-0.4,0,0.4")
parser.add_argument(
    "--candidates",
    default="0.6:0.25:0:0:0.6",
)
parser.add_argument("--first-command-start", type=int, default=200)
parser.add_argument("--first-command-end", type=int, default=800)
parser.add_argument("--reverse-command-start", type=int, default=1000)
parser.add_argument("--reverse-command-end", type=int, default=1600)
parser.add_argument("--tracking-settle-steps", type=int, default=100)
parser.add_argument("--maximum-vx-rmse", type=float, default=0.08)
parser.add_argument("--maximum-wz-rmse", type=float, default=0.12)
parser.add_argument("--maximum-pitch-deg", type=float, default=10.0)
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
    CascadedLQRConfig,
    cascaded_lqr_action,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0


def parse_csv(value: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"expected finite comma-separated values, got {value!r}")
    return result


def parse_candidates(value: str) -> list[tuple[float, float, float, float, float]]:
    result = []
    for item in value.split(","):
        values = [float(part.strip()) for part in item.split(":")]
        if len(values) != 5 or not np.isfinite(values).all() or min(values) < 0.0:
            raise ValueError(
                "invalid vx_kp:wz_kp:wz_ki:wheel_difference_kp:wz_feedforward "
                f"candidate {item!r}"
            )
        result.append((values[0], values[1], values[2], values[3], values[4]))
    if not result:
        raise ValueError("at least one candidate is required")
    return result


def build_scenarios(
    num_envs: int, vx_commands: np.ndarray, wz_commands: np.ndarray
) -> list[dict[str, float]]:
    combinations = list(itertools.product(vx_commands, wz_commands))
    return [
        {
            "vx_first": float(combinations[index % len(combinations)][0]),
            "wz_first": float(combinations[index % len(combinations)][1]),
        }
        for index in range(num_envs)
    ]


def command_at_step(
    step: int, scenarios: list[dict[str, float]]
) -> tuple[np.ndarray, np.ndarray, bool]:
    vx = np.zeros(len(scenarios), dtype=np.float64)
    wz = np.zeros(len(scenarios), dtype=np.float64)
    tracking_window = False
    if args.first_command_start <= step < args.first_command_end:
        vx = np.asarray([item["vx_first"] for item in scenarios])
        wz = np.asarray([item["wz_first"] for item in scenarios])
        tracking_window = step >= args.first_command_start + args.tracking_settle_steps
    elif args.reverse_command_start <= step < args.reverse_command_end:
        vx = -np.asarray([item["vx_first"] for item in scenarios])
        wz = -np.asarray([item["wz_first"] for item in scenarios])
        tracking_window = step >= args.reverse_command_start + args.tracking_settle_steps
    return vx, wz, tracking_window


def safe_rmse(squared_error_sum: float, count: int) -> float:
    return float(np.sqrt(squared_error_sum / count)) if count else 0.0


def evaluate_candidate(
    env,
    gain: np.ndarray,
    control_interval: int,
    action_limit: float,
    scenarios: list[dict[str, float]],
    vx_kp: float,
    wz_kp: float,
    wz_ki: float,
    wheel_difference_kp: float,
    wz_feedforward: float,
) -> dict[str, object]:
    obs, _ = env.reset(seed=args.seed)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    num_envs = env.unwrapped.num_envs
    active = np.ones(num_envs, dtype=bool)
    survived = np.zeros(num_envs, dtype=bool)
    duration_steps = np.full(num_envs, args.horizon_steps, dtype=np.int64)
    integrals = np.zeros((num_envs, 6), dtype=np.float64)
    action_np = np.zeros((num_envs, len(ACTION_NAMES)), dtype=np.float32)
    pitch_reference_max = 0.0
    pitch_max = 0.0
    wheel_speed_max = 0.0
    saturated = 0
    action_count = 0
    vx_squared_error = 0.0
    wz_squared_error = 0.0
    vx_count = 0
    wz_count = 0
    per_env_vx_squared_error = np.zeros(num_envs, dtype=np.float64)
    per_env_wz_squared_error = np.zeros(num_envs, dtype=np.float64)
    per_env_vx_response_dot = np.zeros(num_envs, dtype=np.float64)
    per_env_wz_response_dot = np.zeros(num_envs, dtype=np.float64)
    per_env_vx_reference_squared = np.zeros(num_envs, dtype=np.float64)
    per_env_wz_reference_squared = np.zeros(num_envs, dtype=np.float64)
    per_env_vx_count = np.zeros(num_envs, dtype=np.int64)
    per_env_wz_count = np.zeros(num_envs, dtype=np.int64)
    config = CascadedLQRConfig(
        vx_kp=vx_kp,
        wz_kp=wz_kp,
        vx_ki=0.0,
        wz_ki=wz_ki,
        wheel_difference_kp=wheel_difference_kp,
        wz_feedforward=wz_feedforward,
        wz_integral_limit=0.5,
        action_limit=action_limit,
    )

    for step in range(args.horizon_steps):
        vx_ref, wz_ref, tracking_window = command_at_step(step, scenarios)
        env.unwrapped.vx_ref.copy_(
            torch.as_tensor(vx_ref, dtype=torch.float32, device=env.unwrapped.device)
        )
        env.unwrapped.wz_ref.copy_(
            torch.as_tensor(wz_ref, dtype=torch.float32, device=env.unwrapped.device)
        )
        if step % control_interval == 0:
            action_np, integrals, diagnostics = cascaded_lqr_action(
                current_states,
                vx_ref,
                wz_ref,
                gain,
                integrals,
                control_dt=control_interval / POLICY_HZ,
                config=config,
            )
            pitch_reference_max = max(
                pitch_reference_max,
                float(np.degrees(np.max(np.abs(diagnostics["pitch_reference"])))),
            )
            action_np = action_np.astype(np.float32)
        action_np[~active] = 0.0
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action_np, device=env.unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = env.unwrapped._state_terms()
        pitch = np.degrees(np.abs(current_states[:, 0]))
        vx_truth = state["vx"].detach().cpu().numpy()
        wz_truth = state["yaw_rate"].detach().cpu().numpy()
        wheel_speed = state["max_abs_wheel_velocity"].detach().cpu().numpy()
        if np.any(active):
            pitch_max = max(pitch_max, float(np.max(pitch[active])))
            wheel_speed_max = max(wheel_speed_max, float(np.max(wheel_speed[active])))
            saturated += int(
                np.count_nonzero(np.abs(action_np[active]) >= action_limit - 1e-6)
            )
            action_count += int(np.count_nonzero(active) * len(ACTION_NAMES))
        if tracking_window:
            vx_mask = active & (np.abs(vx_ref) > 1e-9)
            wz_mask = active & (np.abs(wz_ref) > 1e-9)
            vx_error_squared = np.square(vx_truth - vx_ref)
            wz_error_squared = np.square(wz_truth - wz_ref)
            vx_squared_error += float(np.sum(vx_error_squared[vx_mask]))
            wz_squared_error += float(np.sum(wz_error_squared[wz_mask]))
            vx_count += int(np.count_nonzero(vx_mask))
            wz_count += int(np.count_nonzero(wz_mask))
            per_env_vx_squared_error[vx_mask] += vx_error_squared[vx_mask]
            per_env_wz_squared_error[wz_mask] += wz_error_squared[wz_mask]
            per_env_vx_response_dot[vx_mask] += vx_truth[vx_mask] * vx_ref[vx_mask]
            per_env_wz_response_dot[wz_mask] += wz_truth[wz_mask] * wz_ref[wz_mask]
            per_env_vx_reference_squared[vx_mask] += np.square(vx_ref[vx_mask])
            per_env_wz_reference_squared[wz_mask] += np.square(wz_ref[wz_mask])
            per_env_vx_count[vx_mask] += 1
            per_env_wz_count[wz_mask] += 1

        terminated_np = terminated.detach().cpu().numpy().astype(bool)
        truncated_np = truncated.detach().cpu().numpy().astype(bool)
        finished = active & (terminated_np | truncated_np)
        duration_steps[finished] = step + 1
        survived[finished] = truncated_np[finished]
        active[finished] = False
        integrals[~active] = 0.0
        if not np.any(active):
            break

    survived[active] = True
    scenario_results = []
    for index, scenario in enumerate(scenarios):
        scenario_results.append(
            {
                **scenario,
                "vx_reverse": -scenario["vx_first"],
                "wz_reverse": -scenario["wz_first"],
                "survived": bool(survived[index]),
                "duration_steps": int(duration_steps[index]),
                "vx_rmse": safe_rmse(
                    per_env_vx_squared_error[index], int(per_env_vx_count[index])
                ),
                "wz_rmse": safe_rmse(
                    per_env_wz_squared_error[index], int(per_env_wz_count[index])
                ),
                "vx_response_gain": float(
                    per_env_vx_response_dot[index]
                    / max(per_env_vx_reference_squared[index], 1e-12)
                ),
                "wz_response_gain": float(
                    per_env_wz_response_dot[index]
                    / max(per_env_wz_reference_squared[index], 1e-12)
                ),
            }
        )
    result = {
        "vx_kp": vx_kp,
        "wz_kp": wz_kp,
        "wheel_difference_kp": wheel_difference_kp,
        "wz_feedforward": wz_feedforward,
        "vx_ki": 0.0,
        "wz_ki": wz_ki,
        "success_rate": float(np.mean(survived)),
        "vx_rmse": safe_rmse(vx_squared_error, vx_count),
        "wz_rmse": safe_rmse(wz_squared_error, wz_count),
        "pitch_max_deg": pitch_max,
        "pitch_reference_max_deg": pitch_reference_max,
        "wheel_speed_max_rad_s": wheel_speed_max,
        "action_saturation_ratio": saturated / max(action_count, 1),
        "vx_response_gain": float(
            np.sum(per_env_vx_response_dot)
            / max(float(np.sum(per_env_vx_reference_squared)), 1e-12)
        ),
        "wz_response_gain": float(
            np.sum(per_env_wz_response_dot)
            / max(float(np.sum(per_env_wz_reference_squared)), 1e-12)
        ),
        "scenarios": scenario_results,
    }
    result["passed"] = bool(
        result["success_rate"] >= args.minimum_success_rate
        and result["vx_rmse"] <= args.maximum_vx_rmse
        and result["wz_rmse"] <= args.maximum_wz_rmse
        and result["pitch_max_deg"] <= args.maximum_pitch_deg
        and result["action_saturation_ratio"] <= args.maximum_saturation_ratio
    )
    result["score"] = float(
        result["vx_rmse"] / args.maximum_vx_rmse
        + result["wz_rmse"] / args.maximum_wz_rmse
        + result["pitch_max_deg"] / args.maximum_pitch_deg
        + result["action_saturation_ratio"] / args.maximum_saturation_ratio
    )
    return result


def main() -> int:
    windows = (
        0 <= args.first_command_start < args.first_command_end,
        args.first_command_end <= args.reverse_command_start < args.reverse_command_end,
        args.reverse_command_end < args.horizon_steps,
    )
    if args.num_envs < 1 or args.horizon_steps < 1 or not all(windows):
        raise ValueError("invalid environment count, horizon, or command windows")
    if args.tracking_settle_steps < 0:
        raise ValueError("--tracking-settle-steps must be non-negative")
    if args.first_command_start + args.tracking_settle_steps >= args.first_command_end:
        raise ValueError("first command has no tracking measurement window")
    if args.reverse_command_start + args.tracking_settle_steps >= args.reverse_command_end:
        raise ValueError("reverse command has no tracking measurement window")

    gain_data = json.loads(args.gains.resolve().read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")
    control_interval = int(gain_data["control_interval_steps"])
    action_limit = float(gain_data["action_limit"])
    vx_commands = parse_csv(args.vx_commands)
    wz_commands = parse_csv(args.wz_commands)
    candidates = parse_candidates(args.candidates)
    scenarios = build_scenarios(args.num_envs, vx_commands, wz_commands)

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
    evaluations = []
    for vx_kp, wz_kp, wz_ki, wheel_difference_kp, wz_feedforward in candidates:
        evaluation = evaluate_candidate(
            env,
            gain,
            control_interval,
            action_limit,
            scenarios,
            vx_kp,
            wz_kp,
            wz_ki,
            wheel_difference_kp,
            wz_feedforward,
        )
        evaluations.append(evaluation)
        print(
            f"candidate vx_kp={vx_kp:g} wz_kp={wz_kp:g} "
            f"wz_ki={wz_ki:g} wheel_diff_kp={wheel_difference_kp:g} "
            f"wz_ff={wz_feedforward:g}: "
            f"survival={evaluation['success_rate']:.3f} "
            f"vx_rmse={evaluation['vx_rmse']:.4f} "
            f"wz_rmse={evaluation['wz_rmse']:.4f} "
            f"pitch_max={evaluation['pitch_max_deg']:.3f} "
            f"sat={evaluation['action_saturation_ratio']:.4f} "
            f"passed={evaluation['passed']}",
            flush=True,
        )
    env.close()

    passing = [item for item in evaluations if item["passed"]]
    selected = min(passing, key=lambda item: item["score"]) if passing else None
    result = {
        "schema": "recomo_two_wheel_cascaded_lqr_tracking_gate_v2",
        "seed": args.seed,
        "gains": str(args.gains.resolve()),
        "selected_gain_scale": gain_data["selected_gain_scale"],
        "policy_hz": POLICY_HZ,
        "controller_hz": POLICY_HZ / control_interval,
        "commands": {
            "vx_m_s": vx_commands.tolist(),
            "wz_rad_s": wz_commands.tolist(),
            "first_window_steps": [args.first_command_start, args.first_command_end],
            "reverse_window_steps": [args.reverse_command_start, args.reverse_command_end],
            "tracking_settle_steps": args.tracking_settle_steps,
        },
        "thresholds": {
            "minimum_success_rate": args.minimum_success_rate,
            "maximum_vx_rmse": args.maximum_vx_rmse,
            "maximum_wz_rmse": args.maximum_wz_rmse,
            "maximum_pitch_deg": args.maximum_pitch_deg,
            "maximum_saturation_ratio": args.maximum_saturation_ratio,
        },
        "evaluations": evaluations,
        "selected": selected,
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
        "evaluation_only_truth": ["base_vx"],
        "training_started": False,
        "passed": selected is not None,
    }
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
