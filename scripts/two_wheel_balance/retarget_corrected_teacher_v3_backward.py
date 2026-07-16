#!/usr/bin/env python3
"""Recover hard schema-v3 cases by solving the semantic path backward.

This is deliberately separate from the verified forward retargeter.  A case is
exported only when the backward path, the gravity-safe acquisition, and all
existing schema-v3 gates pass without relaxation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from retarget_corrected_teacher_v3_nonholonomic import (  # noqa: E402
    CANDIDATE_SCHEMA,
    FORBIDDEN_EXPORT_KEYS,
    SOURCE_PASSIVE_DFR_JOINTS,
    SemanticReference,
    build_feasible_acquisition,
    load_semantic_reference,
    physical_camera_rotation,
    sha256,
)
from rl_platform.tasks.two_wheel_balance.all79_reference import (  # noqa: E402
    SparseTeacher,
    discover_v3_package,
    quaternion_slerp_wxyz,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    UrdfPhysicalCameraKinematics,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (  # noqa: E402
    UrdfPositionKinematics,
    integrate_unicycle,
)


RECOVERY_SCHEMA = f"{CANDIDATE_SCHEMA}_backward_recovery_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-package", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--target-urdf", type=Path, required=True)
    parser.add_argument("--terminal-states-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="73,78")
    parser.add_argument("--acquisition-dt-s", type=float, default=0.1)
    parser.add_argument("--minimum-acquisition-duration-s", type=float, default=3.0)
    parser.add_argument("--maximum-acquisition-linear-velocity", type=float, default=0.15)
    parser.add_argument("--maximum-acquisition-yaw-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-arm-rate", type=float, default=0.2)
    parser.add_argument("--maximum-acquisition-gimbal-rate", type=float, default=0.2)
    parser.add_argument("--maximum-linear-velocity", type=float, default=0.4)
    parser.add_argument("--maximum-yaw-rate", type=float, default=0.4)
    parser.add_argument("--maximum-arm-rate", type=float, default=0.5)
    parser.add_argument("--maximum-gimbal-rate", type=float, default=0.25)
    parser.add_argument("--maximum-arm-gravity-effort-nm", type=float, default=29.5)
    parser.add_argument("--gravity-effort-tolerance-nm", type=float, default=0.01)
    parser.add_argument("--maximum-position-p95-m", type=float, default=0.10)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.20)
    parser.add_argument("--maximum-ik-error-deg", type=float, default=0.1)
    parser.add_argument("--maximum-step-position-error-m", type=float, default=0.05)
    parser.add_argument("--maximum-time-scale", type=int, default=24)
    return parser.parse_args()


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def inverse_integrate_unicycle(
    next_base_q: np.ndarray,
    velocity: float,
    yaw_rate: float,
    dt: float,
) -> np.ndarray:
    """Return the unique predecessor for a constant unicycle control."""

    next_base_q = np.asarray(next_base_q, dtype=np.float64)
    if next_base_q.shape != (3,) or dt <= 0.0:
        raise ValueError("invalid base state or timestep")
    next_x, next_y, next_yaw = next_base_q
    previous_yaw = next_yaw - yaw_rate * dt
    if abs(yaw_rate) < 1e-9:
        previous_x = next_x - velocity * dt * math.cos(previous_yaw)
        previous_y = next_y - velocity * dt * math.sin(previous_yaw)
    else:
        previous_x = next_x - velocity / yaw_rate * (
            math.sin(next_yaw) - math.sin(previous_yaw)
        )
        previous_y = next_y + velocity / yaw_rate * (
            math.cos(next_yaw) - math.cos(previous_yaw)
        )
    return np.array([previous_x, previous_y, previous_yaw])


def _root_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def _state_metrics(
    state: np.ndarray,
    target_position: np.ndarray,
    target_attitude: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
) -> tuple[float, float, float]:
    position_error = float(
        np.linalg.norm(position_kinematics.position(state[:6]) - target_position)
    )
    target_rotation = quaternion_matrix_wxyz(
        semantic_dfr_to_physical_cam_quat_wxyz(target_attitude)
    )
    attitude_error_deg = math.degrees(
        float(
            np.linalg.norm(
                rotation_error_vector(
                    physical_camera_rotation(state, camera_kinematics),
                    target_rotation,
                )
            )
        )
    )
    gravity_nm = float(
        np.max(
            np.abs(position_kinematics.gravitational_effort_nm(state[:6]))
        )
    )
    return position_error, attitude_error_deg, gravity_nm


def _predecessor(
    next_state: np.ndarray,
    control: np.ndarray,
    dt: float,
) -> np.ndarray:
    return np.concatenate(
        (
            inverse_integrate_unicycle(
                next_state[:3], control[0], control[1], dt
            ),
            next_state[3:6] - control[2:5],
            next_state[6:9] - control[5:8],
        )
    )


def _control_bounds(
    next_state: np.ndarray,
    dt: float,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    arm_delta_lower = np.maximum(
        -args.maximum_arm_rate * dt,
        next_state[3:6] - position_kinematics.arm_upper,
    )
    arm_delta_upper = np.minimum(
        args.maximum_arm_rate * dt,
        next_state[3:6] - position_kinematics.arm_lower,
    )
    gimbal_delta_lower = np.maximum(
        -args.maximum_gimbal_rate * dt,
        next_state[6:9] - camera_kinematics.gimbal_upper,
    )
    gimbal_delta_upper = np.minimum(
        args.maximum_gimbal_rate * dt,
        next_state[6:9] - camera_kinematics.gimbal_lower,
    )
    lower = np.concatenate(
        (
            [-args.maximum_linear_velocity, -args.maximum_yaw_rate],
            arm_delta_lower,
            gimbal_delta_lower,
        )
    )
    upper = np.concatenate(
        (
            [args.maximum_linear_velocity, args.maximum_yaw_rate],
            arm_delta_upper,
            gimbal_delta_upper,
        )
    )
    return lower, upper


def _solve_predecessor(
    next_state: np.ndarray,
    target_position: np.ndarray,
    target_attitude: np.ndarray,
    source_q: np.ndarray,
    previous_control: np.ndarray,
    dt: float,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    lower, upper = _control_bounds(
        next_state, dt, position_kinematics, camera_kinematics, args
    )
    target_rotation = quaternion_matrix_wxyz(
        semantic_dfr_to_physical_cam_quat_wxyz(target_attitude)
    )
    gimbal_center = 0.5 * (
        camera_kinematics.gimbal_lower + camera_kinematics.gimbal_upper
    )
    gimbal_range = camera_kinematics.gimbal_upper - camera_kinematics.gimbal_lower

    def residual(control: np.ndarray) -> np.ndarray:
        state = _predecessor(next_state, control, dt)
        position_error = (
            position_kinematics.position(state[:6]) - target_position
        ) / 0.01
        attitude_error = rotation_error_vector(
            physical_camera_rotation(state, camera_kinematics), target_rotation
        ) / math.radians(args.maximum_ik_error_deg)
        source_delta = state[:6] - source_q
        source_regularization = np.concatenate(
            (0.1 * source_delta[:3], source_delta[3:6] / 0.35)
        )
        gravity_effort = position_kinematics.gravitational_effort_nm(state[:6])
        gravity_overrun = np.maximum(
            np.abs(gravity_effort) - args.maximum_arm_gravity_effort_nm, 0.0
        ) / 0.02
        return np.concatenate(
            (
                position_error,
                attitude_error,
                source_regularization,
                gravity_overrun,
                0.002 * control,
                0.01 * (state[6:9] - gimbal_center) / gimbal_range,
            )
        )

    def solve(seed: np.ndarray):
        return least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=120,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

    source_seed = np.zeros(8)
    source_seed[1] = wrap_angle(float(next_state[2] - source_q[2])) / dt
    displacement = next_state[:2] - source_q[:2]
    source_heading = np.array([math.cos(source_q[2]), math.sin(source_q[2])])
    source_seed[0] = float(np.dot(displacement, source_heading) / dt)
    source_seed[2:5] = next_state[3:6] - source_q[3:6]
    predicted = _predecessor(next_state, np.clip(source_seed, lower, upper), dt)
    gimbal_result = camera_kinematics.solve_semantic_attitude_continuous(
        _root_quaternion(predicted[2]),
        predicted[3:6],
        target_attitude,
        next_state[6:9],
    )
    source_seed[5:8] = next_state[6:9] - gimbal_result.gimbal_q

    seeds = [previous_control, np.zeros(8), source_seed]
    solutions = [solve(seed) for seed in seeds]

    def rank(solution) -> tuple[bool, float]:
        state = _predecessor(next_state, solution.x, dt)
        metrics = _state_metrics(
            state,
            target_position,
            target_attitude,
            position_kinematics,
            camera_kinematics,
        )
        feasible = (
            metrics[0] <= args.maximum_step_position_error_m
            and metrics[1] <= args.maximum_ik_error_deg
            and metrics[2]
            <= args.maximum_arm_gravity_effort_nm
            + args.gravity_effort_tolerance_nm
        )
        score = (
            metrics[0] / 0.02
            + metrics[1] / args.maximum_ik_error_deg
            + max(0.0, metrics[2] - args.maximum_arm_gravity_effort_nm)
        )
        return feasible, score

    selected = min(solutions, key=lambda item: (not rank(item)[0], rank(item)[1]))
    state = _predecessor(next_state, selected.x, dt)
    return state, selected.x, _state_metrics(
        state,
        target_position,
        target_attitude,
        position_kinematics,
        camera_kinematics,
    )


def solve_backward_reference(
    reference: SemanticReference,
    source_base_arm_q: np.ndarray,
    terminal_state: np.ndarray,
    position_kinematics: UrdfPositionKinematics,
    camera_kinematics: UrdfPhysicalCameraKinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    terminal_metrics = _state_metrics(
        terminal_state,
        reference.positions_m[-1],
        reference.attitudes_wxyz[-1],
        position_kinematics,
        camera_kinematics,
    )
    if (
        terminal_metrics[0] > 1e-4
        or terminal_metrics[1] > 0.01
        or terminal_metrics[2]
        > args.maximum_arm_gravity_effort_nm + args.gravity_effort_tolerance_nm
    ):
        raise ValueError(f"terminal state failed independent feasibility: {terminal_metrics}")

    reverse_states = [terminal_state.copy()]
    reverse_targets = [reference.positions_m[-1].copy()]
    reverse_attitudes = [reference.attitudes_wxyz[-1].copy()]
    reverse_controls: list[np.ndarray] = []
    reverse_dt_s: list[float] = []
    previous_control = np.zeros(8)
    interval_scales: list[int] = []

    scales = tuple(
        scale for scale in (1, 2, 4, 8, 12, 16, 24) if scale <= args.maximum_time_scale
    )
    if not scales:
        raise ValueError("maximum time scale must be at least one")

    for index in range(len(reference.time_s) - 2, -1, -1):
        source_dt = float(reference.time_s[index + 1] - reference.time_s[index])
        segment_end = reverse_states[-1].copy()
        attempts = []
        for time_scale in scales:
            state = segment_end.copy()
            control_seed = previous_control.copy()
            states = []
            controls = []
            targets = []
            attitudes = []
            metrics = []
            for reverse_substep in range(time_scale - 1, -1, -1):
                fraction = reverse_substep / time_scale
                target_position = (
                    (1.0 - fraction) * reference.positions_m[index]
                    + fraction * reference.positions_m[index + 1]
                )
                target_attitude = quaternion_slerp_wxyz(
                    reference.attitudes_wxyz[index],
                    reference.attitudes_wxyz[index + 1],
                    np.array([fraction]),
                )[0]
                source_q = (
                    (1.0 - fraction) * source_base_arm_q[index]
                    + fraction * source_base_arm_q[index + 1]
                )
                predecessor, control, step_metrics = _solve_predecessor(
                    state,
                    target_position,
                    target_attitude,
                    source_q,
                    control_seed,
                    source_dt,
                    position_kinematics,
                    camera_kinematics,
                    args,
                )
                state = predecessor
                control_seed = control
                states.append(state.copy())
                controls.append(control.copy())
                targets.append(target_position)
                attitudes.append(target_attitude)
                metrics.append(step_metrics)
            feasible = all(
                item[0] <= args.maximum_step_position_error_m
                and item[1] <= args.maximum_ik_error_deg
                and item[2]
                <= args.maximum_arm_gravity_effort_nm
                + args.gravity_effort_tolerance_nm
                for item in metrics
            )
            score = max(
                item[0] / 0.02
                + item[1] / args.maximum_ik_error_deg
                + max(0.0, item[2] - args.maximum_arm_gravity_effort_nm)
                for item in metrics
            )
            attempts.append(
                (feasible, score, time_scale, states, controls, targets, attitudes)
            )
            if feasible:
                break

        selected = min(
            (item for item in attempts if item[0]),
            key=lambda item: item[1],
            default=None,
        )
        if selected is None:
            best = min(attempts, key=lambda item: item[1])
            raise RuntimeError(
                f"backward interval {index}->{index + 1} has no feasible branch; "
                f"best scale={best[2]} score={best[1]:.6f}"
            )
        _, _, time_scale, states, controls, targets, attitudes = selected
        interval_scales.append(time_scale)
        reverse_states.extend(states)
        reverse_controls.extend(controls)
        reverse_dt_s.extend([source_dt] * len(controls))
        reverse_targets.extend(targets)
        reverse_attitudes.extend(attitudes)
        previous_control = controls[-1]

    states = np.asarray(reverse_states[::-1])
    controls = np.asarray(reverse_controls[::-1])
    dt_s = np.asarray(reverse_dt_s[::-1])
    targets = np.asarray(reverse_targets[::-1])
    attitudes = np.asarray(reverse_attitudes[::-1])
    time_s = np.concatenate((np.zeros(1), np.cumsum(dt_s)))
    diagnostics = {
        "backward_interval_time_scales_chronological": interval_scales[::-1],
        "backward_retimed_interval_count": sum(scale > 1 for scale in interval_scales),
        "backward_candidate_samples": len(states),
    }
    return time_s, states, controls, targets, attitudes, diagnostics


def _load_terminal_states(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = {
        int(item["case"]): np.asarray(item["base_arm_gimbal_q"], dtype=np.float64)
        for item in payload["cases"]
        if item.get("feasible")
    }
    if any(state.shape != (9,) for state in states.values()):
        raise ValueError("terminal states must have shape (9,)")
    return states


def process_case(
    case: int,
    teacher: SparseTeacher,
    terminal_state: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    result_path = args.output_dir / f"case_{case:04d}.result.json"
    try:
        source_kinematics = UrdfPositionKinematics(
            args.source_urdf.resolve(),
            passive_joint_positions=SOURCE_PASSIVE_DFR_JOINTS,
        )
        position_kinematics = UrdfPositionKinematics(args.target_urdf.resolve())
        camera_kinematics = UrdfPhysicalCameraKinematics(args.target_urdf.resolve())
        source_mat = args.source_batch / f"episode_{case:04d}" / "teacher_smoke.mat"
        reference = load_semantic_reference(
            teacher, source_mat.resolve(), source_kinematics
        )
        (
            semantic_time,
            semantic_states,
            semantic_controls,
            semantic_targets,
            semantic_attitudes,
            backward_diagnostics,
        ) = solve_backward_reference(
            reference,
            teacher.base_arm_q,
            terminal_state,
            position_kinematics,
            camera_kinematics,
            args,
        )
        (
            acquisition_time,
            acquisition_states,
            acquisition_controls,
            acquisition_targets,
            acquisition_attitudes,
            _,
            acquisition_diagnostics,
        ) = build_feasible_acquisition(
            semantic_states[0], position_kinematics, camera_kinematics, args
        )
        time_s = np.concatenate(
            (acquisition_time, acquisition_time[-1] + semantic_time[1:])
        )
        internal_states = np.vstack((acquisition_states, semantic_states[1:]))
        # Physical gimbal deltas are internal feasibility variables, never
        # learned controls or labels in the schema-v3 candidate.
        controls = np.vstack((acquisition_controls, semantic_controls[:, :5]))
        targets = np.vstack((acquisition_targets, semantic_targets[1:]))
        attitudes = np.vstack((acquisition_attitudes, semantic_attitudes[1:]))
        base_arm_q = internal_states[:, :6]
        achieved = np.asarray(
            [position_kinematics.position(state) for state in base_arm_q]
        )
        position_errors = np.linalg.norm(achieved - targets, axis=1)
        attitude_errors = np.asarray(
            [
                _state_metrics(
                    state,
                    target_position,
                    target_attitude,
                    position_kinematics,
                    camera_kinematics,
                )[1]
                for state, target_position, target_attitude in zip(
                    internal_states, targets, attitudes, strict=True
                )
            ]
        )
        gravity = np.asarray(
            [
                np.max(
                    np.abs(position_kinematics.gravitational_effort_nm(state))
                )
                for state in base_arm_q
            ]
        )
        gimbal_rates = np.abs(np.diff(internal_states[:, 6:9], axis=0)) / np.diff(
            time_s
        )[:, None]
        semantic_start_index = len(acquisition_time) - 1
        base_transition_errors = []
        arm_transition_errors = []
        for index in range(semantic_start_index, len(controls)):
            predicted_base = integrate_unicycle(
                base_arm_q[index, :3],
                controls[index, 0],
                controls[index, 1],
                float(time_s[index + 1] - time_s[index]),
            )
            base_transition_errors.append(
                float(np.max(np.abs(predicted_base - base_arm_q[index + 1, :3])))
            )
            arm_transition_errors.append(
                float(
                    np.max(
                        np.abs(
                            base_arm_q[index, 3:6]
                            + controls[index, 2:5]
                            - base_arm_q[index + 1, 3:6]
                        )
                    )
                )
            )
        base_transition_max = max(base_transition_errors)
        arm_transition_max = max(arm_transition_errors)
        checks = {
            "position_p95_bounded": float(np.percentile(position_errors, 95))
            <= args.maximum_position_p95_m,
            "position_maximum_bounded": float(np.max(position_errors))
            <= args.maximum_position_error_m,
            "all_physical_gimbal_ik_converged": float(np.max(attitude_errors))
            <= args.maximum_ik_error_deg,
            "physical_gimbal_ik_error_bounded": float(np.max(attitude_errors))
            <= args.maximum_ik_error_deg,
            "physical_gimbal_rate_bounded": float(np.max(gimbal_rates))
            <= args.maximum_gimbal_rate + 1e-9,
            "arm_gravity_effort_bounded": float(np.max(gravity))
            <= args.maximum_arm_gravity_effort_nm
            + args.gravity_effort_tolerance_nm,
            "nonholonomic_transition_exact": base_transition_max <= 1e-10,
            "arm_transition_exact": arm_transition_max <= 1e-10,
            "physical_gimbal_joint_labels_not_exported": True,
            "runtime_approval_remains_false": True,
            "training_not_started": True,
        }
        summary = {
            "schema": RECOVERY_SCHEMA,
            "case": case,
            "source_samples": len(reference.time_s),
            "candidate_samples": len(time_s),
            "semantic_start_index": semantic_start_index,
            "position_error_p95_m": float(np.percentile(position_errors, 95)),
            "position_error_max_m": float(np.max(position_errors)),
            "physical_gimbal_ik_max_error_deg": float(np.max(attitude_errors)),
            "physical_gimbal_rate_max_radps": float(np.max(gimbal_rates)),
            "maximum_arm_gravity_effort_nm": float(np.max(gravity)),
            "nonholonomic_transition_max_abs": base_transition_max,
            "arm_transition_max_abs": arm_transition_max,
            **backward_diagnostics,
            **acquisition_diagnostics,
            "checks": checks,
            "passed": all(checks.values()),
            "runtime_approved": False,
            "training_started": False,
        }
        arrays = {
            "schema": np.asarray(CANDIDATE_SCHEMA),
            "retarget_method": np.asarray("backward_terminal_seeded_v1"),
            "case": np.int32(case),
            "runtime_approved": np.bool_(False),
            "training_started": np.bool_(False),
            "position_target_link": np.asarray("ee1_tool"),
            "attitude_target_contract": np.asarray(
                "world_semantic_DFR_quaternion_wxyz_option_B"
            ),
            "physical_gimbal_joint_labels_included": np.bool_(False),
            "time_s": time_s,
            "source_time_s": reference.time_s,
            "semantic_start_index": np.int32(semantic_start_index),
            "target_position_world_m": targets,
            "target_attitude_world_dfr_quat_wxyz": attitudes,
            "achieved_position_world_m": achieved,
            "base_arm_q": base_arm_q,
            "control_v_wz_darm": controls,
            "position_error_m": position_errors,
            "source_teacher_sha256": np.asarray(sha256(teacher.path)),
            "source_mat_sha256": np.asarray(sha256(reference.source_mat)),
        }
        forbidden = FORBIDDEN_EXPORT_KEYS & set(arrays)
        if forbidden:
            raise AssertionError(f"physical gimbal labels leaked: {forbidden}")
        if summary["passed"]:
            np.savez_compressed(args.output_dir / f"case_{case:04d}.npz", **arrays)
    except Exception as error:
        summary = {
            "schema": RECOVERY_SCHEMA,
            "case": case,
            "passed": False,
            "runtime_approved": False,
            "training_started": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    result_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teachers = discover_v3_package(args.teacher_package)
    terminal_states = _load_terminal_states(args.terminal_states_json)
    cases = [int(value) for value in args.cases.split(",") if value.strip()]
    if not cases or any(case not in teachers or case not in terminal_states for case in cases):
        raise ValueError(f"missing teacher or terminal state for cases: {cases}")
    results = []
    for case in cases:
        result = process_case(case, teachers[case], terminal_states[case], args)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    summary = {
        "schema": "cinebotrl_two_wheel_backward_retarget_batch_recovery_v1",
        "cases": cases,
        "passed_case_count": sum(bool(item["passed"]) for item in results),
        "runtime_approved": False,
        "training_started": False,
        "passed": all(bool(item["passed"]) for item in results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
