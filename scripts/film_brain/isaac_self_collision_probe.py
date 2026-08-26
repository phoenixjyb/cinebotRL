#!/usr/bin/env python3
"""Run a bounded, machine-readable Proto2 self-collision/contact probe.

This is a no-training, no-robot-transport diagnostic for the Film Brain
feasibility worker.  It loads the exact USD used by RecomoProto2TrackEE-v0,
inspects the composed PhysX collision configuration, and exercises the current
base-versus-arm contact sensor with deterministic actions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")
os.environ.setdefault("FILM_BRAIN_ENABLE_SELF_COLLISION", "1")
os.environ.setdefault("FILM_BRAIN_COLLISION_PAIR_PROBE", "1")

REPORT_PREFIX = "FILM_BRAIN_REPORT_JSON="


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_value(value: Any) -> Any:
    """Convert common USD/Python values into deterministic JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _stage_collision_inventory() -> dict[str, Any]:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    collision_prims: list[dict[str, Any]] = []
    articulation_settings: list[dict[str, Any]] = []
    filtered_pairs: list[dict[str, Any]] = []

    for prim in stage.Traverse():
        self_collision_attr = prim.GetAttribute(
            "physxArticulation:enabledSelfCollisions"
        )
        if self_collision_attr and self_collision_attr.HasAuthoredValueOpinion():
            articulation_settings.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "enabled_self_collisions": bool(self_collision_attr.Get()),
                }
            )

        filtered_pairs_rel = prim.GetRelationship("physics:filteredPairs")
        if filtered_pairs_rel and filtered_pairs_rel.HasAuthoredTargets():
            filtered_pairs.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "targets": [str(target) for target in filtered_pairs_rel.GetTargets()],
                }
            )

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        collision_prims.append(
            {
                "prim_path": str(prim.GetPath()),
                "type_name": prim.GetTypeName(),
                "mesh_approximation": _json_value(approximation),
            }
        )

    return {
        "articulation_settings": articulation_settings,
        "filtered_pairs": filtered_pairs,
        "collision_prim_count": len(collision_prims),
        "collision_prims": collision_prims,
    }


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _max_tensor_norm(value: Any) -> float | None:
    if value is None:
        return None
    import torch

    if value.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(value, dim=-1).amax().item())


def _run_phase(env: Any, actions: Any, steps: int) -> dict[str, Any]:
    unwrapped = env.unwrapped
    chassis_sensor = unwrapped.scene["contact_sensor"]
    arm_sensor = unwrapped.scene["arm_contact_sensor"]
    filtered_forces: list[float] = []
    chassis_net_forces: list[float] = []
    broad_arm_net_forces: list[float] = []
    done_steps: list[int] = []
    filtered_pair_maxima: list[float] | None = None
    broad_arm_body_maxima: list[float] | None = None
    diagnostic_pair_maxima: dict[tuple[str, str], float] = {}

    for step_index in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        filtered_forces.append(
            float(unwrapped._get_filtered_contact_forces().amax().item())
        )
        force_matrix = getattr(chassis_sensor.data, "force_matrix_w", None)
        if force_matrix is not None:
            pair_values = (
                force_matrix[0, 0]
                .square()
                .sum(dim=-1)
                .sqrt()
                .detach()
                .cpu()
                .tolist()
            )
            if filtered_pair_maxima is None:
                filtered_pair_maxima = [0.0] * len(pair_values)
            filtered_pair_maxima = [
                max(previous, float(current))
                for previous, current in zip(filtered_pair_maxima, pair_values)
            ]
        for sensor_name, diagnostic_sensor in unwrapped.scene.sensors.items():
            if not sensor_name.startswith("film_brain_pair_sensor_"):
                continue
            diagnostic_matrix = getattr(
                diagnostic_sensor.data, "force_matrix_w", None
            )
            if diagnostic_matrix is None:
                continue
            diagnostic_values = (
                diagnostic_matrix[0, 0]
                .square()
                .sum(dim=-1)
                .sqrt()
                .detach()
                .cpu()
                .tolist()
            )
            sensor_body_names = list(getattr(diagnostic_sensor, "body_names", []))
            sensor_body_name = sensor_body_names[0] if sensor_body_names else sensor_name
            diagnostic_filters = list(
                diagnostic_sensor.cfg.filter_prim_paths_expr or []
            )
            for filter_path, current in zip(diagnostic_filters, diagnostic_values):
                key = (sensor_body_name, filter_path)
                diagnostic_pair_maxima[key] = max(
                    diagnostic_pair_maxima.get(key, 0.0), float(current)
                )
        chassis_net_forces.append(
            _max_tensor_norm(getattr(chassis_sensor.data, "net_forces_w", None)) or 0.0
        )
        arm_net_forces = getattr(arm_sensor.data, "net_forces_w", None)
        broad_arm_net_forces.append(_max_tensor_norm(arm_net_forces) or 0.0)
        if arm_net_forces is not None:
            arm_body_values = (
                arm_net_forces[0]
                .square()
                .sum(dim=-1)
                .sqrt()
                .detach()
                .cpu()
                .tolist()
            )
            if broad_arm_body_maxima is None:
                broad_arm_body_maxima = [0.0] * len(arm_body_values)
            broad_arm_body_maxima = [
                max(previous, float(current))
                for previous, current in zip(broad_arm_body_maxima, arm_body_values)
            ]
        if bool((terminated | truncated).any().item()):
            done_steps.append(step_index)

    def summarize(samples: list[float]) -> dict[str, Any]:
        return {
            "max_n": max(samples, default=0.0),
            "mean_n": sum(samples) / len(samples) if samples else 0.0,
            "steps_gt_0_001_n": sum(value > 0.001 for value in samples),
            "steps_gt_1_n": sum(value > 1.0 for value in samples),
            "steps_gt_5_n": sum(value > 5.0 for value in samples),
        }

    filter_paths = list(chassis_sensor.cfg.filter_prim_paths_expr or [])
    pair_maxima = filtered_pair_maxima or [0.0] * len(filter_paths)
    arm_body_names = list(getattr(arm_sensor, "body_names", []))
    arm_body_maxima = broad_arm_body_maxima or [0.0] * len(arm_body_names)
    return {
        "steps": steps,
        "filtered_base_arm": summarize(filtered_forces),
        "chassis_net": summarize(chassis_net_forces),
        "broad_arm_net": summarize(broad_arm_net_forces),
        "broad_arm_body_max_n": [
            {"body_name": name, "max_n": maximum}
            for name, maximum in zip(arm_body_names, arm_body_maxima)
        ],
        "filtered_pair_max_n": [
            {"filter_prim_path_expr": path, "max_n": maximum}
            for path, maximum in zip(filter_paths, pair_maxima)
        ],
        "diagnostic_pair_max_n": [
            {
                "sensor_body_name": sensor_body_name,
                "filter_prim_path_expr": filter_path,
                "max_n": maximum,
            }
            for (sensor_body_name, filter_path), maximum in sorted(
                diagnostic_pair_maxima.items()
            )
            if maximum > 0.001
        ],
        "done_steps": done_steps,
    }


def _read_diagnostic_pairs(unwrapped: Any) -> list[dict[str, Any]]:
    pair_maxima: dict[tuple[str, str], float] = {}
    for sensor_name, sensor in unwrapped.scene.sensors.items():
        if not sensor_name.startswith("film_brain_pair_sensor_"):
            continue
        force_matrix = getattr(sensor.data, "force_matrix_w", None)
        if force_matrix is None:
            continue
        values = (
            force_matrix[0, 0]
            .square()
            .sum(dim=-1)
            .sqrt()
            .detach()
            .cpu()
            .tolist()
        )
        body_names = list(getattr(sensor, "body_names", []))
        sensor_body_name = body_names[0] if body_names else sensor_name
        filter_paths = list(sensor.cfg.filter_prim_paths_expr or [])
        for filter_path, value in zip(filter_paths, values):
            filter_body_name = filter_path.rsplit("/", 1)[-1]
            pair = tuple(sorted((sensor_body_name, filter_body_name)))
            pair_maxima[pair] = max(pair_maxima.get(pair, 0.0), float(value))
    return [
        {"bodies": list(pair), "max_n": maximum}
        for pair, maximum in sorted(pair_maxima.items())
        if maximum > 0.001
    ]


def _scan_joint_configurations(env: Any, sample_count: int) -> dict[str, Any]:
    import torch

    from rl_platform.tasks.mobile_mm.joint_names import ARM_JOINT_NAMES

    unwrapped = env.unwrapped
    unwrapped._initialize_joint_limits()
    unwrapped._verify_joint_mapping()
    arm_ids = unwrapped._get_joint_ids(ARM_JOINT_NAMES, "_arm_joint_ids")
    limits = unwrapped.robot.data.soft_joint_pos_limits[0, arm_ids]
    lower = limits[:, 0]
    upper = limits[:, 1]
    midpoint = 0.5 * (lower + upper)

    candidates = [
        midpoint,
        lower,
        upper,
        torch.where(
            torch.arange(len(arm_ids), device=unwrapped.device) % 2 == 0,
            lower,
            upper,
        ),
        torch.where(
            torch.arange(len(arm_ids), device=unwrapped.device) % 2 == 0,
            upper,
            lower,
        ),
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260826)
    while len(candidates) < sample_count:
        unit = torch.rand(len(arm_ids), generator=generator, dtype=torch.float32)
        candidates.append(lower + unit.to(unwrapped.device) * (upper - lower))

    sample_results: list[dict[str, Any]] = []
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    for sample_index, target in enumerate(candidates[:sample_count]):
        env.reset()
        targets = target.unsqueeze(0)
        zeros = torch.zeros_like(targets)
        unwrapped.robot.set_joint_position_target(
            targets, joint_ids=arm_ids, env_ids=env_ids
        )
        unwrapped.robot.write_joint_state_to_sim(
            targets, zeros, joint_ids=arm_ids, env_ids=env_ids
        )
        unwrapped._lock_base_ppr_joints(env_ids=env_ids)
        unwrapped._lock_passive_joints(env_ids=env_ids)
        unwrapped.scene.write_data_to_sim()
        for _ in range(2):
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(unwrapped.sim.get_physics_dt())

        pairs = _read_diagnostic_pairs(unwrapped)
        maximum = max((pair["max_n"] for pair in pairs), default=0.0)
        sample_results.append(
            {
                "sample_index": sample_index,
                "joint_positions_rad": target.detach().cpu().tolist(),
                "max_pair_force_n": maximum,
                "pairs_gt_0_001_n": pairs,
            }
        )

    ranked = sorted(
        sample_results,
        key=lambda sample: sample["max_pair_force_n"],
        reverse=True,
    )
    return {
        "seed": 20260826,
        "sample_count": sample_count,
        "joint_names": list(ARM_JOINT_NAMES),
        "joint_lower_rad": lower.detach().cpu().tolist(),
        "joint_upper_rad": upper.detach().cpu().tolist(),
        "samples_gt_1_n": sum(
            sample["max_pair_force_n"] > 1.0 for sample in sample_results
        ),
        "samples_gt_5_n": sum(
            sample["max_pair_force_n"] > 5.0 for sample in sample_results
        ),
        "max_pair_force_n": ranked[0]["max_pair_force_n"] if ranked else 0.0,
        "top_samples": ranked[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument("--forced-steps", type=int, default=80)
    parser.add_argument("--scan-samples", type=int, default=32)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": "recomo.film_brain.isaac_self_collision_probe.v1",
        "status": "error",
        "authority": {
            "robot_transport": False,
            "motion_authority": False,
            "continuous_collision_checked": False,
            "pnc_equivalence_claimed": False,
            "physical_feasibility_claimed": False,
        },
        "error": None,
    }
    simulation_app = None
    env = None

    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(
            headless=args.headless,
            enable_cameras=False,
            device=args.device,
        )
        simulation_app = launcher.app

        import gymnasium as gym
        import torch

        from rl_platform.robots.mobile_mm import get_mobile_mm_usd_path
        from task_spec import register_isaac_lab_tasks

        register_isaac_lab_tasks()
        env = gym.make(
            "RecomoProto2TrackEE-v0",
            num_envs=1,
            headless=args.headless,
            enable_obstacles=False,
        )
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.robot
        chassis_sensor = unwrapped.scene["contact_sensor"]
        force_matrix = getattr(chassis_sensor.data, "force_matrix_w", None)
        usd_path = Path(get_mobile_mm_usd_path()).resolve()

        # This probe records contacts rather than allowing the environment to
        # auto-reset immediately at its hard collision threshold.
        original_terminate = bool(unwrapped.task_cfg.terminate_on_self_collision)
        unwrapped.task_cfg.terminate_on_self_collision = False

        zero_actions = torch.zeros((1, 9), device=unwrapped.device)
        forced_actions = torch.tensor(
            [[1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0]],
            device=unwrapped.device,
        )
        baseline = _run_phase(env, zero_actions, args.baseline_steps)
        forced = _run_phase(env, forced_actions, args.forced_steps)
        joint_scan = _scan_joint_configurations(env, args.scan_samples)

        stage_inventory = _stage_collision_inventory()
        authored_values = {
            item["enabled_self_collisions"]
            for item in stage_inventory["articulation_settings"]
        }
        report.update(
            {
                "status": "ok",
                "runtime": {
                    "device": str(unwrapped.device),
                    "torch": _package_version("torch"),
                    "isaaclab": _package_version("isaaclab"),
                    "isaacsim": _package_version("isaacsim"),
                    "gymnasium": _package_version("gymnasium"),
                },
                "model": {
                    "task_id": "RecomoProto2TrackEE-v0",
                    "usd_path": str(usd_path),
                    "usd_sha256": _sha256(usd_path),
                    "body_names": list(robot.body_names),
                    "joint_names": list(robot.joint_names),
                },
                "stage": stage_inventory,
                "self_collision_authored_values": sorted(authored_values),
                "contact_sensor": {
                    "prim_path": str(chassis_sensor.cfg.prim_path),
                    "filter_prim_paths_expr": list(
                        chassis_sensor.cfg.filter_prim_paths_expr or []
                    ),
                    "force_matrix_available": force_matrix is not None,
                    "force_matrix_shape": _tensor_shape(force_matrix),
                    "net_forces_shape": _tensor_shape(
                        getattr(chassis_sensor.data, "net_forces_w", None)
                    ),
                },
                "environment_collision_termination_was_enabled": original_terminate,
                "probe_collision_termination_disabled": True,
                "baseline": baseline,
                "forced_extreme_action": {
                    "action": forced_actions[0].detach().cpu().tolist(),
                    **forced,
                },
                "joint_configuration_scan": joint_scan,
                "interpretation": {
                    "filtered_contact_observed": (
                        forced["filtered_base_arm"]["max_n"] > 0.001
                    ),
                    "broad_arm_contact_observed": (
                        forced["broad_arm_net"]["max_n"] > 0.001
                    ),
                    "all_authored_articulations_enable_self_collision": (
                        bool(authored_values) and authored_values == {True}
                    ),
                    "positive_collision_control_observed": (
                        joint_scan["samples_gt_1_n"] > 0
                    ),
                    "coverage": "base-versus-configured-arm-and-ee-bodies only",
                },
            }
        )
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Isaac Kit shutdown may replace or close buffered standard streams on
        # Windows.  Emit and flush the evidence while the app is still alive.
        print(
            REPORT_PREFIX
            + json.dumps(report, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
