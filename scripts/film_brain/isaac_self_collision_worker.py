#!/usr/bin/env python3
"""Isaac Lab discrete self-collision worker for the Film Brain prototype.

Protocol: one canonical JSON object on stdin and one canonical JSON object on
stdout, without trailing newlines.  The worker has no network, Robot transport,
motion-authority, continuous-collision, environment-collision, clearance, PnC
equivalence, or physical-feasibility claim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
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

REQUEST_SCHEMA = "film-brain-isaac-self-collision-request/0.1.0"
RESPONSE_SCHEMA = "film-brain-isaac-self-collision-response/0.1.0"
MAXIMUM_INPUT_BYTES = 4 * 1024 * 1024
MAXIMUM_SAMPLES = 64
CONTACT_THRESHOLD_N = 1.0
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{1,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COORDINATE_ORDER = (
    "base_joint_vx",
    "base_joint_vy",
    "base_joint_wz",
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "ee1_level_pitch",
    "ee1_gimbal_pitch",
    "ee1_gimbal_roll",
    "ee1_gimbal_yaw",
)
AUTHORITY = {
    "robot_transport": False,
    "motion_authority": False,
    "physical_feasibility_claimed": False,
    "pnc_execution_semantics_claimed": False,
}
CAPABILITIES = {
    "discrete_self_collision": True,
    "continuous_or_swept_collision": False,
    "environment_collision": False,
    "minimum_clearance": False,
    "attached_body_policy_equivalent_to_pnc": False,
    "robot_model_equivalent_to_pnc": False,
}


class RequestError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _configure_isaac_process_environment() -> None:
    os.environ.setdefault("ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS", "1")
    os.environ.setdefault("FILM_BRAIN_ENABLE_SELF_COLLISION", "1")
    os.environ.setdefault("FILM_BRAIN_COLLISION_PAIR_PROBE", "1")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
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


def _read_request() -> tuple[dict[str, Any], bytes]:
    raw = sys.stdin.buffer.read(MAXIMUM_INPUT_BYTES + 1)
    if not raw or len(raw) > MAXIMUM_INPUT_BYTES:
        raise RequestError("INPUT_SIZE_INVALID", "input must be 1..4194304 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError("INPUT_UTF8_INVALID", "input must be UTF-8") from exc
    try:
        request = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestError("INPUT_JSON_INVALID", "input must contain one JSON object") from exc
    if not isinstance(request, dict):
        raise RequestError("INPUT_OBJECT_REQUIRED", "request must be an object")
    if _canonical_bytes(request) != raw:
        raise RequestError(
            "INPUT_NOT_CANONICAL",
            "request must be sorted compact canonical JSON without a newline",
        )
    return request, raw


def _validate_common(request: dict[str, Any]) -> str:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise RequestError("SCHEMA_VERSION_INVALID", "request schema is unsupported")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not IDENTIFIER_RE.fullmatch(request_id):
        raise RequestError("REQUEST_ID_INVALID", "request_id is invalid")
    if request.get("authority") != AUTHORITY:
        raise RequestError("AUTHORITY_INVALID", "all authority flags must be false")
    return request_id


def _validate_evaluation_request(request: dict[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "request_id",
        "operation",
        "authority",
        "robot_model",
        "collision_policy",
        "coordinate_order",
        "samples",
        "continuous_collision_required",
        "environment_collision_required",
    }
    if set(request) != required_keys:
        raise RequestError("REQUEST_FIELDS_INVALID", "evaluation fields are not exact")
    if request.get("operation") != "EVALUATE_DISCRETE_SELF_COLLISION_ONLY":
        raise RequestError("OPERATION_INVALID", "evaluation operation is invalid")
    if request.get("continuous_collision_required") is not False:
        raise RequestError(
            "CONTINUOUS_COLLISION_UNSUPPORTED",
            "this worker evaluates discrete configurations only",
        )
    if request.get("environment_collision_required") is not False:
        raise RequestError(
            "ENVIRONMENT_COLLISION_UNSUPPORTED",
            "this worker evaluates self-collision only",
        )
    robot_model = request.get("robot_model")
    if not isinstance(robot_model, dict) or set(robot_model) != {
        "robot_model_id",
        "robot_model_version",
        "expected_usd_sha256",
    }:
        raise RequestError("ROBOT_MODEL_FIELDS_INVALID", "robot model fields are invalid")
    if (
        not isinstance(robot_model.get("robot_model_id"), str)
        or not IDENTIFIER_RE.fullmatch(robot_model["robot_model_id"])
        or not isinstance(robot_model.get("robot_model_version"), str)
        or not IDENTIFIER_RE.fullmatch(robot_model["robot_model_version"])
        or not isinstance(robot_model.get("expected_usd_sha256"), str)
        or not SHA256_RE.fullmatch(robot_model["expected_usd_sha256"])
    ):
        raise RequestError("ROBOT_MODEL_IDENTITY_INVALID", "robot model identity is invalid")
    policy = request.get("collision_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "policy_id",
        "expected_policy_sha256",
        "contact_threshold_n",
    }:
        raise RequestError("COLLISION_POLICY_FIELDS_INVALID", "collision policy fields are invalid")
    if (
        not isinstance(policy.get("policy_id"), str)
        or not IDENTIFIER_RE.fullmatch(policy["policy_id"])
        or not isinstance(policy.get("expected_policy_sha256"), str)
        or not SHA256_RE.fullmatch(policy["expected_policy_sha256"])
        or policy.get("contact_threshold_n") != CONTACT_THRESHOLD_N
    ):
        raise RequestError("COLLISION_POLICY_INVALID", "collision policy identity is invalid")
    if request.get("coordinate_order") != list(COORDINATE_ORDER):
        raise RequestError("COORDINATE_ORDER_INVALID", "coordinate order must match the Film Brain 10D order")
    samples = request.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAXIMUM_SAMPLES:
        raise RequestError("SAMPLE_COUNT_INVALID", "sample count must be 1..64")
    timestamps: list[int] = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"timestamp_ns", "positions"}:
            raise RequestError("SAMPLE_FIELDS_INVALID", "sample fields are invalid")
        timestamp = sample.get("timestamp_ns")
        positions = sample.get("positions")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise RequestError("SAMPLE_TIMESTAMP_INVALID", "sample timestamp is invalid")
        if (
            not isinstance(positions, list)
            or len(positions) != len(COORDINATE_ORDER)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in positions
            )
        ):
            raise RequestError("SAMPLE_POSITIONS_INVALID", "sample positions are invalid")
        if abs(float(positions[6])) > 1.0e-9:
            raise RequestError(
                "LEVEL_PITCH_MAPPING_UNPROVED",
                "ee1_level_pitch must remain zero until Isaac/PnC model equivalence is proved",
            )
        timestamps.append(timestamp)
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise RequestError("SAMPLE_TIME_ORDER_INVALID", "timestamps must strictly increase")


def _relative_robot_path(path: Any) -> str:
    text = str(path)
    marker = "/Robot/base_root/"
    return text.split(marker, 1)[1] if marker in text else text


def _stage_identity(unwrapped: Any, usd_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    articulation_settings: list[dict[str, Any]] = []
    collision_prims: list[dict[str, Any]] = []
    filtered_pairs: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if "/Robot/base_root/" not in str(prim.GetPath()):
            continue
        self_collision_attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if self_collision_attr and self_collision_attr.HasAuthoredValueOpinion():
            articulation_settings.append(
                {
                    "prim_path": _relative_robot_path(prim.GetPath()),
                    "enabled_self_collisions": bool(self_collision_attr.Get()),
                }
            )
        relation = prim.GetRelationship("physics:filteredPairs")
        if relation and relation.HasAuthoredTargets():
            filtered_pairs.append(
                {
                    "body": _relative_robot_path(prim.GetPath()),
                    "targets": sorted(
                        _relative_robot_path(target) for target in relation.GetTargets()
                    ),
                }
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            approximation = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            collision_prims.append(
                {
                    "prim_path": _relative_robot_path(prim.GetPath()),
                    "collision_enabled": True if enabled is None else bool(enabled),
                    "mesh_approximation": None if approximation is None else str(approximation),
                }
            )
    policy = {
        "policy_id": "recomo-proto2-isaac-discrete-self-collision-v1",
        "contact_threshold_n": CONTACT_THRESHOLD_N,
        "self_collision_articulation_settings": sorted(
            articulation_settings, key=lambda value: value["prim_path"]
        ),
        "collision_prims": sorted(collision_prims, key=lambda value: value["prim_path"]),
        "filtered_pairs": sorted(filtered_pairs, key=lambda value: value["body"]),
        "sensor_body_names": sorted(
            {
                body_name
                for name, sensor in unwrapped.scene.sensors.items()
                if name.startswith("film_brain_pair_sensor_")
                for body_name in getattr(sensor, "body_names", [])
            }
        ),
        "discrete_only": True,
        "environment_collision_evaluated": False,
        "pnc_srdf_equivalence_reviewed": False,
    }
    usd_sha256 = _file_sha256(usd_path)
    backend_identity = {
        "backend_kind": "ISAAC_LAB_PHYSX_DISCRETE_SELF_COLLISION",
        "robot_model_id": "recomo-proto2-isaac-usd",
        "robot_model_version": "recomoProto2-1190-wrapper-baked-scaled",
        "usd_path_basename": usd_path.name,
        "usd_sha256": usd_sha256,
        "policy_sha256": _canonical_sha256(policy),
        "worker_source_sha256": _file_sha256(SCRIPT_PATH),
        "environment_source_sha256": _file_sha256(
            PROJECT_ROOT / "src" / "rl_platform" / "tasks" / "mobile_mm" / "env.py"
        ),
        "runtime": {
            "isaaclab": _package_version("isaaclab"),
            "isaacsim": _package_version("isaacsim"),
            "torch": _package_version("torch"),
            "gymnasium": _package_version("gymnasium"),
            "device": str(unwrapped.device),
        },
    }
    return backend_identity, policy


def _read_contact_pairs(unwrapped: Any) -> list[dict[str, Any]]:
    maxima: dict[tuple[str, str], float] = {}
    for sensor_name, sensor in unwrapped.scene.sensors.items():
        if not sensor_name.startswith("film_brain_pair_sensor_"):
            continue
        matrix = getattr(sensor.data, "force_matrix_w", None)
        if matrix is None:
            continue
        values = matrix[0, 0].square().sum(dim=-1).sqrt().detach().cpu().tolist()
        body_names = list(getattr(sensor, "body_names", []))
        sensor_body = body_names[0] if body_names else sensor_name
        filters = list(sensor.cfg.filter_prim_paths_expr or [])
        for filter_path, force_n in zip(filters, values):
            filter_body = filter_path.rsplit("/", 1)[-1]
            pair = tuple(sorted((sensor_body, filter_body)))
            maxima[pair] = max(maxima.get(pair, 0.0), float(force_n))
    return [
        {
            "bodies": list(pair),
            "max_contact_force_n": maximum,
            "collision": maximum > CONTACT_THRESHOLD_N,
        }
        for pair, maximum in sorted(maxima.items())
        if maximum > 0.001
    ]


def _film_brain_positions_to_isaac_arm(positions: list[int | float]) -> tuple[float, ...]:
    """Map the Film Brain 10D FLU contract to Proto2's six physical arm joints."""
    return (
        float(positions[3]),
        float(positions[4]),
        float(positions[5]),
        float(positions[9]),
        float(positions[8]),
        float(positions[7]),
    )


def _evaluate(unwrapped: Any, env: Any, request: dict[str, Any]) -> dict[str, Any]:
    import torch

    from rl_platform.tasks.mobile_mm.joint_names import ARM_JOINT_NAMES

    unwrapped._initialize_joint_limits()
    unwrapped._verify_joint_mapping()
    arm_ids = unwrapped._get_joint_ids(ARM_JOINT_NAMES, "_arm_joint_ids")
    limits = unwrapped.robot.data.soft_joint_pos_limits[0, arm_ids]
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    results: list[dict[str, Any]] = []
    for sample in request["samples"]:
        positions = [float(value) for value in sample["positions"]]
        # Film Brain FLU order -> Isaac physical arm/gimbal order:
        # arm yaw/pitch/elbow + gimbal yaw/roll/pitch. The derived level-pitch
        # coordinate is required to be zero by request validation.
        target = torch.tensor(
            _film_brain_positions_to_isaac_arm(positions),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        if bool(((target < limits[:, 0]) | (target > limits[:, 1])).any().item()):
            raise RequestError(
                "JOINT_LIMIT_VIOLATION",
                f"sample {sample['timestamp_ns']} exceeds Isaac soft joint limits",
            )
        env.reset()
        targets = target.unsqueeze(0)
        zeros = torch.zeros_like(targets)
        unwrapped.robot.set_joint_position_target(targets, joint_ids=arm_ids, env_ids=env_ids)
        unwrapped.robot.write_joint_state_to_sim(
            targets, zeros, joint_ids=arm_ids, env_ids=env_ids
        )
        unwrapped._lock_base_ppr_joints(env_ids=env_ids)
        unwrapped._lock_passive_joints(env_ids=env_ids)
        unwrapped.scene.write_data_to_sim()
        for _ in range(2):
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(unwrapped.sim.get_physics_dt())
        pairs = _read_contact_pairs(unwrapped)
        maximum = max((pair["max_contact_force_n"] for pair in pairs), default=0.0)
        results.append(
            {
                "timestamp_ns": sample["timestamp_ns"],
                "collision": maximum > CONTACT_THRESHOLD_N,
                "max_contact_force_n": maximum,
                "contact_pairs": pairs,
            }
        )
    collision_results = [result for result in results if result["collision"]]
    return {
        "result_status": "SAMPLED_COLLISION" if collision_results else "SAMPLED_CLEAR",
        "evaluated_sample_count": len(results),
        "collision_sample_count": len(collision_results),
        "first_collision_timestamp_ns": (
            collision_results[0]["timestamp_ns"] if collision_results else None
        ),
        "maximum_contact_force_n": max(
            (result["max_contact_force_n"] for result in results), default=0.0
        ),
        "sample_results": results,
        "base_coordinates_elided_by_rigid_transform_invariance": True,
        "level_pitch_required_zero": True,
    }


def _response_base(request_id: str, request_sha256: str, operation: str) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_id": request_id,
        "request_sha256": request_sha256,
        "operation": operation,
        "capabilities": dict(CAPABILITIES),
        "authority": dict(AUTHORITY),
    }


def _error_response(
    request: dict[str, Any] | None,
    request_raw: bytes,
    status: str,
    code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_id": request.get("request_id") if isinstance(request, dict) else None,
        "request_sha256": hashlib.sha256(request_raw).hexdigest() if request_raw else None,
        "operation": request.get("operation") if isinstance(request, dict) else None,
        "response_status": status,
        "error": {"code": code, "detail": detail},
        "capabilities": dict(CAPABILITIES),
        "authority": dict(AUTHORITY),
    }


def _emit_response(response_fd: int, response: dict[str, Any]) -> None:
    response["response_sha256"] = _canonical_sha256(response)
    os.write(response_fd, _canonical_bytes(response))


def _run_with_isaac(
    request: dict[str, Any], request_raw: bytes, request_sha256: str, response_fd: int
) -> int:
    simulation_app = None
    env = None
    response: dict[str, Any]
    exit_code = 0
    try:
        _configure_isaac_process_environment()
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=False, device="cuda:0")
        simulation_app = launcher.app
        import gymnasium as gym

        from rl_platform.robots.mobile_mm import get_mobile_mm_usd_path
        from task_spec import register_isaac_lab_tasks

        register_isaac_lab_tasks()
        env = gym.make(
            "RecomoProto2TrackEE-v0",
            num_envs=1,
            headless=True,
            enable_obstacles=False,
        )
        env.reset()
        unwrapped = env.unwrapped
        usd_path = Path(get_mobile_mm_usd_path()).resolve()
        backend_identity, collision_policy = _stage_identity(unwrapped, usd_path)
        response = _response_base(request["request_id"], request_sha256, request["operation"])
        response["backend_identity"] = backend_identity
        response["collision_policy"] = collision_policy
        if request["operation"] == "IDENTITY":
            response["response_status"] = "IDENTITY_READY"
            response["evaluation"] = None
        else:
            expected_model = request["robot_model"]
            expected_policy = request["collision_policy"]
            if expected_model["robot_model_id"] != backend_identity["robot_model_id"]:
                raise RequestError("ROBOT_MODEL_ID_MISMATCH", "robot model id does not match")
            if expected_model["robot_model_version"] != backend_identity["robot_model_version"]:
                raise RequestError(
                    "ROBOT_MODEL_VERSION_MISMATCH", "robot model version does not match"
                )
            if expected_model["expected_usd_sha256"] != backend_identity["usd_sha256"]:
                raise RequestError("ROBOT_MODEL_SHA256_MISMATCH", "USD SHA-256 does not match")
            if expected_policy["policy_id"] != collision_policy["policy_id"]:
                raise RequestError(
                    "COLLISION_POLICY_ID_MISMATCH", "collision policy id does not match"
                )
            if expected_policy["expected_policy_sha256"] != backend_identity["policy_sha256"]:
                raise RequestError(
                    "COLLISION_POLICY_SHA256_MISMATCH",
                    "collision policy SHA-256 does not match",
                )

            response["response_status"] = "EVALUATION_COMPLETED"
            response["evaluation"] = _evaluate(unwrapped, env, request)
    except RequestError as exc:
        exit_code = 2
        response = _error_response(
            request, request_raw, "REJECTED", exc.code, exc.detail
        )
    except Exception as exc:
        exit_code = 1
        response = _error_response(
            request,
            request_raw,
            "ERROR",
            "WORKER_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Isaac Kit may close or replace standard streams during shutdown.
        # Emit the sole protocol response after the environment closes but
        # while the application and the preserved caller stdout are alive.
        _emit_response(response_fd, response)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
    return exit_code


def main() -> int:
    request: dict[str, Any] | None = None
    request_raw = b""
    try:
        request, request_raw = _read_request()
        request_id = _validate_common(request)
        operation = request.get("operation")
        if operation == "IDENTITY":
            if set(request) != {"schema_version", "request_id", "operation", "authority"}:
                raise RequestError("REQUEST_FIELDS_INVALID", "identity fields are not exact")
        else:
            _validate_evaluation_request(request)
        request_sha256 = hashlib.sha256(request_raw).hexdigest()

        # Native Kit logging is redirected to an anonymous temporary file so
        # stdout remains exactly one canonical JSON object.
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        with tempfile.TemporaryFile() as diagnostic_log:
            os.dup2(diagnostic_log.fileno(), 1)
            os.dup2(diagnostic_log.fileno(), 2)
            try:
                exit_code = _run_with_isaac(
                    request, request_raw, request_sha256, saved_stdout
                )
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
                os.close(saved_stdout)
                os.close(saved_stderr)
        return exit_code
    except RequestError as exc:
        exit_code = 2
        response = _error_response(
            request, request_raw, "REJECTED", exc.code, exc.detail
        )
    except Exception as exc:
        exit_code = 1
        response = _error_response(
            request,
            request_raw,
            "ERROR",
            "WORKER_ERROR",
            f"{type(exc).__name__}: {exc}",
        )

    _emit_response(1, response)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
