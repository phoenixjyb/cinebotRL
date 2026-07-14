#!/usr/bin/env python3
"""Replay retargeted all-79 candidates with the whole-body balance controller."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gains", type=Path, required=True)
parser.add_argument("--retarget-dir", type=Path, required=True)
parser.add_argument("--cases", default="1,20,28,50,79")
parser.add_argument("--maximum-pitch-deg", type=float, default=12.0)
parser.add_argument("--maximum-arm-error-deg", type=float, default=10.0)
parser.add_argument("--maximum-position-p95-m", type=float, default=0.15)
parser.add_argument("--maximum-position-error-m", type=float, default=0.25)
parser.add_argument("--maximum-action-saturation-ratio", type=float, default=0.20)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import torch

from rl_platform.robots.two_wheel_balance import TWO_WHEEL_WHOLE_BODY_CFG
from rl_platform.tasks.two_wheel_balance import RecomoTwoWheelBalanceEnvCfg
from rl_platform.tasks.two_wheel_balance.metrics import (
    ACTION_NAMES,
    LQR_STATE_NAMES,
    cascaded_lqr_action,
    cascaded_lqr_config,
)
from task_spec import register_isaac_lab_tasks


POLICY_HZ = 200.0
ARM_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
)


def load_candidate(case: int) -> dict[str, np.ndarray]:
    path = args.retarget_dir / f"case_{case:04d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        candidate = {name: np.asarray(data[name]) for name in data.files}
    expected = {
        "time_s",
        "target_position_world_m",
        "base_arm_q",
        "control_v_wz_darm",
    }
    if not expected <= set(candidate):
        raise ValueError(f"candidate {path} is missing {sorted(expected - set(candidate))}")
    time_s = candidate["time_s"]
    if (
        time_s.ndim != 1
        or len(time_s) < 2
        or time_s[0] != 0.0
        or np.any(np.diff(time_s) <= 0.0)
    ):
        raise ValueError(f"invalid candidate time in {path}")
    if candidate["base_arm_q"].shape != (len(time_s), 6):
        raise ValueError(f"invalid candidate state shape in {path}")
    if candidate["target_position_world_m"].shape != (len(time_s), 3):
        raise ValueError(f"invalid candidate target shape in {path}")
    if candidate["control_v_wz_darm"].shape != (len(time_s) - 1, 5):
        raise ValueError(f"invalid candidate control shape in {path}")
    return candidate


def interpolate(candidate: dict[str, np.ndarray], elapsed_s: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    time_s = candidate["time_s"]
    upper = int(np.searchsorted(time_s, elapsed_s, side="right"))
    upper = min(max(upper, 1), len(time_s) - 1)
    lower = upper - 1
    dt = float(time_s[upper] - time_s[lower])
    alpha = np.clip((elapsed_s - time_s[lower]) / dt, 0.0, 1.0)
    state = (1.0 - alpha) * candidate["base_arm_q"][lower] + alpha * candidate[
        "base_arm_q"
    ][upper]
    target = (1.0 - alpha) * candidate["target_position_world_m"][lower] + alpha * candidate[
        "target_position_world_m"
    ][upper]
    control = candidate["control_v_wz_darm"][lower]
    return state[3:], target, float(control[0]), float(control[1])


def evaluate_case(env, case: int, candidate: dict[str, np.ndarray], gain: np.ndarray, control_interval: int) -> dict[str, object]:
    obs, _ = env.reset(seed=20260714 + case)
    unwrapped = env.unwrapped
    arm_ids = []
    for name in ARM_JOINTS:
        ids = unwrapped.robot.find_joints(name)[0]
        if len(ids) != 1:
            raise RuntimeError(f"expected one joint named {name}, got {ids}")
        arm_ids.append(ids[0])
    tool_ids = unwrapped.robot.find_bodies("ee1_tool")[0]
    if len(tool_ids) != 1:
        raise RuntimeError(f"expected physical ee1_tool body, got {tool_ids}")

    controller_state = np.zeros((1, 6), dtype=np.float64)
    current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
    action = np.zeros((1, len(ACTION_NAMES)), dtype=np.float32)
    config = cascaded_lqr_config("structural_robust_v1")
    requested_steps = int(math.ceil(float(candidate["time_s"][-1]) * POLICY_HZ)) + 1
    position_errors = []
    vx_errors = []
    wz_errors = []
    peak_pitch_deg = 0.0
    peak_arm_error_deg = 0.0
    saturated_actions = 0
    action_count = 0
    termination = None
    completed_steps = 0

    for step in range(requested_steps):
        elapsed_s = min(step / POLICY_HZ, float(candidate["time_s"][-1]))
        arm_target, position_target, vx_ref, wz_ref = interpolate(candidate, elapsed_s)
        arm_target_tensor = torch.as_tensor(
            arm_target[None, :], dtype=torch.float32, device=unwrapped.device
        )
        unwrapped.robot.set_joint_position_target(arm_target_tensor, joint_ids=arm_ids)
        unwrapped.vx_ref.fill_(vx_ref)
        unwrapped.wz_ref.fill_(wz_ref)
        if step % control_interval == 0:
            action, controller_state, _ = cascaded_lqr_action(
                current_states,
                np.array([vx_ref]),
                np.array([wz_ref]),
                gain,
                controller_state,
                control_dt=control_interval / POLICY_HZ,
                config=config,
            )
            action = action.astype(np.float32)
        saturated_actions += int(np.count_nonzero(np.abs(action) >= config.action_limit - 1e-6))
        action_count += action.size
        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=unwrapped.device)
        )
        current_states = obs["policy"][:, : len(LQR_STATE_NAMES)].detach().cpu().numpy()
        state = unwrapped._state_terms()
        actual_position = (
            unwrapped.robot.data.body_pos_w[0, tool_ids[0]].detach().cpu().numpy()
        )
        position_errors.append(float(np.linalg.norm(actual_position - position_target)))
        actual_arm = unwrapped.robot.data.joint_pos[0, arm_ids]
        arm_error = torch.max(torch.abs(actual_arm - arm_target_tensor[0])).item()
        peak_arm_error_deg = max(peak_arm_error_deg, math.degrees(arm_error))
        peak_pitch_deg = max(
            peak_pitch_deg, math.degrees(abs(float(state["pitch"][0].item())))
        )
        vx_errors.append(float(state["vx"][0].item()) - vx_ref)
        wz_errors.append(float(state["yaw_rate"][0].item()) - wz_ref)
        completed_steps = step + 1
        if bool((terminated | truncated)[0].item()):
            termination = {
                "step": completed_steps,
                "elapsed_s": elapsed_s,
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "reset_reason_counts": dict(unwrapped.reset_reason_counts),
            }
            break

    errors = np.asarray(position_errors)
    saturation_ratio = saturated_actions / max(action_count, 1)
    checks = {
        "completed_horizon": completed_steps == requested_steps,
        "no_termination": termination is None,
        "peak_pitch_below_limit": peak_pitch_deg <= args.maximum_pitch_deg,
        "peak_arm_error_below_limit": peak_arm_error_deg <= args.maximum_arm_error_deg,
        "position_p95_below_limit": float(np.percentile(errors, 95))
        <= args.maximum_position_p95_m,
        "position_max_below_limit": float(np.max(errors))
        <= args.maximum_position_error_m,
        "action_saturation_below_limit": saturation_ratio
        <= args.maximum_action_saturation_ratio,
    }
    return {
        "case": case,
        "duration_s": float(candidate["time_s"][-1]),
        "requested_steps": requested_steps,
        "completed_steps": completed_steps,
        "peak_pitch_deg": peak_pitch_deg,
        "peak_arm_error_deg": peak_arm_error_deg,
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(np.max(errors)),
        "vx_rmse_mps": float(np.sqrt(np.mean(np.square(vx_errors)))),
        "wz_rmse_radps": float(np.sqrt(np.mean(np.square(wz_errors)))),
        "action_saturation_ratio": saturation_ratio,
        "termination": termination,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    cases = [int(item) for item in args.cases.split(",") if item.strip()]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError(f"invalid cases: {cases}")
    candidates = {case: load_candidate(case) for case in cases}
    gain_data = json.loads(args.gains.read_text(encoding="utf-8"))
    gain = np.asarray(gain_data["selected_gain"], dtype=np.float64)
    control_interval = int(gain_data["control_interval_steps"])
    if gain.shape != (len(ACTION_NAMES), len(LQR_STATE_NAMES)):
        raise ValueError(f"invalid gain shape: {gain.shape}")

    register_isaac_lab_tasks()
    cfg = RecomoTwoWheelBalanceEnvCfg()
    cfg.seed = 20260714
    cfg.scene.num_envs = 1
    cfg.robot_cfg = TWO_WHEEL_WHOLE_BODY_CFG
    cfg.episode_length_s = max(float(item["time_s"][-1]) for item in candidates.values()) + 2.0
    cfg.reset_pitch_rad = 0.0
    cfg.control_mode = "direct"
    env = gym.make(
        "RecomoTwoWheelBalance-v0",
        cfg=cfg,
        render_mode=None,
        disable_env_checker=True,
    )
    results = [
        evaluate_case(env, case, candidates[case], gain, control_interval)
        for case in cases
    ]
    env.close()
    result = {
        "schema": "recomo_two_wheel_all79_whole_body_playback_smoke_v1",
        "training_started": False,
        "controller_profile": "structural_robust_v1",
        "cases": cases,
        "passed_case_count": sum(item["passed"] for item in results),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close()
raise SystemExit(exit_code)
