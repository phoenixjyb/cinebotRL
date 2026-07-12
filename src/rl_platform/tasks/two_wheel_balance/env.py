"""Low-level two-wheel balance task with deployable observations only."""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .config import RecomoTwoWheelBalanceEnvCfg


def _roll_pitch_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(dim=-1)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))
    return roll, pitch


class RecomoTwoWheelBalanceEnv(DirectRLEnv):
    """Floating-base inverted-pendulum plant controlled by wheel effort."""

    cfg: RecomoTwoWheelBalanceEnvCfg

    def __init__(self, cfg: RecomoTwoWheelBalanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_joint_idx = self.robot.find_joints("left_wheel_joint")[0]
        self._right_joint_idx = self.robot.find_joints("right_wheel_joint")[0]
        if len(self._left_joint_idx) != 1 or len(self._right_joint_idx) != 1:
            raise RuntimeError(
                "two-wheel contract requires exactly left_wheel_joint and right_wheel_joint"
            )
        self._wheel_joint_idx = self._left_joint_idx + self._right_joint_idx
        self._base_body_idx = self.robot.find_bodies("base_link")[0]
        if len(self._base_body_idx) != 1:
            raise RuntimeError("two-wheel contract requires exactly one base_link body")

        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.policy_actions = torch.zeros_like(self.actions)
        self.previous_actions = torch.zeros_like(self.actions)
        self.wheel_efforts = torch.zeros((self.num_envs, 2), device=self.device)
        self.vx_ref = torch.full((self.num_envs,), self.cfg.command_vx, device=self.device)
        self.wz_ref = torch.full((self.num_envs,), self.cfg.command_wz, device=self.device)
        self.nonfinite_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.reset_reason_counts = {
            "fall": 0,
            "forbidden_body_contact": 0,
            "wheel_overspeed": 0,
            "nonfinite": 0,
            "timeout": 0,
        }
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (
                "upright",
                "pitch_rate",
                "vx_tracking",
                "wz_tracking",
                "wheel_speed",
                "action_magnitude",
                "action_rate",
                "alive",
                "termination",
            )
        }
        self.last_reward_terms: dict[str, torch.Tensor] = {}

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.base_contact_sensor = ContactSensor(self.cfg.base_contact_sensor)
        self.scene.sensors["base_contact_sensor"] = self.base_contact_sensor
        spawn_ground_plane(
            prim_path="/World/Ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                )
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.policy_actions.copy_(torch.clamp(actions, -1.0, 1.0))
        if self.cfg.control_mode == "pd_residual":
            state = self._state_terms()
            pd_common = torch.clamp(
                self.cfg.pd_common_kp * state["pitch"]
                + self.cfg.pd_common_kd * state["pitch_rate"],
                -self.cfg.pd_common_action_limit,
                self.cfg.pd_common_action_limit,
            )
            self.actions.copy_(self.cfg.policy_residual_scale * self.policy_actions)
            self.actions[:, 0].add_(pd_common)
            self.actions.clamp_(-1.0, 1.0)
        elif self.cfg.control_mode == "direct":
            self.actions.copy_(self.policy_actions)
        else:
            raise ValueError(f"unsupported control_mode: {self.cfg.control_mode}")
        common = self.actions[:, 0]
        yaw = self.actions[:, 1]
        self.wheel_efforts[:, 0] = common + yaw
        self.wheel_efforts[:, 1] = common - yaw
        self.wheel_efforts.clamp_(-1.0, 1.0).mul_(self.cfg.torque_limit_nm)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.wheel_efforts, joint_ids=self._wheel_joint_idx)

    def _state_terms(self) -> dict[str, torch.Tensor]:
        roll, pitch = _roll_pitch_from_quat_wxyz(self.robot.data.root_quat_w)
        wheel_pos = self.robot.data.joint_pos[:, self._wheel_joint_idx]
        wheel_vel = self.robot.data.joint_vel[:, self._wheel_joint_idx]
        base_contact_force = torch.linalg.norm(
            self.base_contact_sensor.data.net_forces_w, dim=-1
        ).max(dim=1).values
        return {
            "roll": roll,
            "pitch": pitch,
            "pitch_rate": self.robot.data.root_ang_vel_b[:, 1],
            "yaw_rate": self.robot.data.root_ang_vel_b[:, 2],
            "vx": self.robot.data.root_lin_vel_b[:, 0],
            "mean_wheel_position": wheel_pos.mean(dim=1),
            "mean_wheel_velocity": wheel_vel.mean(dim=1),
            "wheel_velocity_difference": wheel_vel[:, 1] - wheel_vel[:, 0],
            "max_abs_wheel_velocity": wheel_vel.abs().max(dim=1).values,
            "base_contact_force": base_contact_force,
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._state_terms()
        obs = torch.stack(
            (
                state["pitch"],
                state["pitch_rate"],
                state["mean_wheel_position"],
                state["mean_wheel_velocity"],
                state["wheel_velocity_difference"],
                state["yaw_rate"],
                self.vx_ref,
                self.wz_ref,
                self.previous_actions[:, 0],
                self.previous_actions[:, 1],
            ),
            dim=-1,
        )
        finite = torch.isfinite(obs).all(dim=1)
        self.nonfinite_count += torch.count_nonzero(~finite)
        return {"policy": torch.nan_to_num(obs)}

    def _get_rewards(self) -> torch.Tensor:
        state = self._state_terms()
        action_delta = self.actions - self.previous_actions
        rewards = {
            "upright": torch.exp(-torch.square(state["pitch"] / self.cfg.upright_sigma)),
            "pitch_rate": self.cfg.pitch_rate_scale * torch.square(state["pitch_rate"]),
            "vx_tracking": self.cfg.vx_tracking_scale
            * torch.exp(-torch.square(state["vx"] - self.vx_ref) / 0.04),
            "wz_tracking": self.cfg.wz_tracking_scale
            * torch.exp(-torch.square(state["yaw_rate"] - self.wz_ref) / 0.09),
            "wheel_speed": self.cfg.wheel_speed_scale
            * torch.square(state["mean_wheel_velocity"]),
            "action_magnitude": self.cfg.action_magnitude_scale
            * torch.sum(torch.square(self.actions), dim=1),
            "action_rate": self.cfg.action_rate_scale
            * torch.sum(torch.square(action_delta), dim=1),
            "alive": self.cfg.alive_scale * (~self.reset_terminated).float(),
            "termination": self.cfg.termination_scale * self.reset_terminated.float(),
        }
        if self.cfg.enable_reward_term_telemetry:
            self.last_reward_terms = {
                key: value.detach().clone() for key, value in rewards.items()
            }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._state_terms()
        finite = torch.stack(tuple(state.values()), dim=-1).isfinite().all(dim=1)
        fall = (state["pitch"].abs() > self.cfg.fall_pitch_rad) | (
            state["roll"].abs() > self.cfg.fall_roll_rad
        )
        overspeed = state["max_abs_wheel_velocity"] > self.cfg.wheel_speed_hard_limit
        forbidden_contact = state["base_contact_force"] > self.cfg.forbidden_body_contact_force_n
        terminated = fall | forbidden_contact | overspeed | (~finite)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if hasattr(self, "reset_terminated"):
            state = self._state_terms()
            fall = (state["pitch"].abs() > self.cfg.fall_pitch_rad) | (
                state["roll"].abs() > self.cfg.fall_roll_rad
            )
            overspeed = state["max_abs_wheel_velocity"] > self.cfg.wheel_speed_hard_limit
            forbidden_contact = (
                state["base_contact_force"] > self.cfg.forbidden_body_contact_force_n
            )
            finite = torch.stack(tuple(state.values()), dim=-1).isfinite().all(dim=1)
            nonfinite = ~finite
            overspeed = overspeed & finite
            fall = fall & finite & (~overspeed)
            forbidden_contact = forbidden_contact & finite & (~overspeed) & (~fall)
            self.reset_reason_counts["fall"] += int(torch.count_nonzero(fall[env_ids]).item())
            self.reset_reason_counts["forbidden_body_contact"] += int(
                torch.count_nonzero(forbidden_contact[env_ids]).item()
            )
            self.reset_reason_counts["wheel_overspeed"] += int(
                torch.count_nonzero(overspeed[env_ids]).item()
            )
            self.reset_reason_counts["nonfinite"] += int(torch.count_nonzero(nonfinite[env_ids]).item())
            self.reset_reason_counts["timeout"] += int(
                torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            )

        self.extras["log"] = {}
        if len(env_ids) > 0:
            for key, values in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{key}"] = values[env_ids].mean().item()
                values[env_ids] = 0.0

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.reset_pitch_rad:
            half_pitch = 0.5 * self.cfg.reset_pitch_rad
            root_state[:, 3] = math.cos(half_pitch)
            root_state[:, 4] = 0.0
            root_state[:, 5] = math.sin(half_pitch)
            root_state[:, 6] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.policy_actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.wheel_efforts[env_ids] = 0.0

    def diagnostic_snapshot(self) -> dict[str, float | int | dict[str, int]]:
        state = self._state_terms()
        return {
            "abs_pitch_mean_deg": math.degrees(state["pitch"].abs().mean().item()),
            "abs_pitch_p95_deg": math.degrees(torch.quantile(state["pitch"].abs(), 0.95).item()),
            "abs_pitch_max_deg": math.degrees(state["pitch"].abs().max().item()),
            "abs_pitch_rate_max": state["pitch_rate"].abs().max().item(),
            "wheel_speed_max": state["max_abs_wheel_velocity"].max().item(),
            "base_contact_force_max_n": state["base_contact_force"].max().item(),
            "vx_mean": state["vx"].mean().item(),
            "yaw_rate_mean": state["yaw_rate"].mean().item(),
            "left_wheel_velocity_mean": self.robot.data.joint_vel[:, self._left_joint_idx].mean().item(),
            "right_wheel_velocity_mean": self.robot.data.joint_vel[:, self._right_joint_idx].mean().item(),
            "effort_saturation_ratio": (self.wheel_efforts.abs() >= self.cfg.torque_limit_nm).float().mean().item(),
            "nonfinite_count": int(self.nonfinite_count.item()),
            "reset_reason_counts": dict(self.reset_reason_counts),
            "control_mode": self.cfg.control_mode,
            "policy_residual_scale": self.cfg.policy_residual_scale,
        }
