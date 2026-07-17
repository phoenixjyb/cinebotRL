"""Duration-preserving, pose-invariant timing derivation for exact-source ep4."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .exact_source_reference import (
    EXPECTED_MANIFEST_SHA256,
    ExactSourceReference,
    discover_exact_source_references,
    sha256,
)


PACKAGE_SCHEMA = "cinebotrl_ep4_duration_preserving_time_warp_v1"
DERIVATION_CONTRACT = "duration_preserving_bounded_time_reparameterization_v2"
ARRAY_HASH_CONTRACT = "shape_ascii_nul_little_endian_float64_c_order_v1"
PAIRED_SEED_SCHEMA = "cinebotrl_ep4_time_warp_integrity_seed_v1"
PAIRED_SEED_PACKAGE_SCHEMA = "cinebotrl_ep4_time_warp_integrity_seed_package_v1"
EXPECTED_RAW_SEED_MANIFEST_SHA256 = (
    "36ce147d320f723f778ddd33a6f9465345ff8c859a358c61fec2284ce19cdc30"
)
EXPECTED_RAW_EP4_SEED_SHA256 = (
    "3720026d83a8ca64e6e7f8d6bb7938013c8d57faa357f37ac5890cda7d5b7d22"
)
RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS = 0.4
RETARGET_MAXIMUM_YAW_RATE_RADPS = 0.4
RETARGET_MAXIMUM_ARM_RATE_RADPS = 0.5


@dataclass(frozen=True)
class TimeReparameterizationConfig:
    episode_index: int = 4
    time_allocation_strategy: str = "minimum_l2"
    translation_speed_cap_mps: float = 0.40
    angular_speed_cap_radps: float = 0.35
    minimum_interval_dt_s: float = 1.0e-3
    localized_transition_start_1based: int | None = None
    localized_transition_end_1based: int | None = None
    localized_translation_speed_cap_mps: float | None = None
    diagnostic_transition_start_1based: int = 190
    diagnostic_transition_end_1based: int = 205


@dataclass(frozen=True)
class TimeReparameterizationResult:
    source_time_s: np.ndarray
    derived_time_s: np.ndarray
    source_dt_s: np.ndarray
    derived_dt_s: np.ndarray
    segment_distance_m: np.ndarray
    segment_angle_rad: np.ndarray
    translation_speed_cap_mps: np.ndarray
    lower_dt_s: np.ndarray
    source_translation_speed_mps: np.ndarray
    derived_translation_speed_mps: np.ndarray
    source_angular_speed_radps: np.ndarray
    derived_angular_speed_radps: np.ndarray
    recovery_regions: tuple[tuple[int, int], ...]
    slowdown_regions: tuple[tuple[int, int], ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quaternion_interval_angles(attitudes_xyzw: np.ndarray) -> np.ndarray:
    attitudes = np.asarray(attitudes_xyzw, dtype=np.float64)
    _require(attitudes.ndim == 2 and attitudes.shape[1] == 4, "bad attitude shape")
    _require(np.isfinite(attitudes).all(), "attitudes contain non-finite values")
    norms = np.linalg.norm(attitudes, axis=1)
    _require(np.allclose(norms, 1.0, atol=1.0e-9), "attitudes are not unit quaternions")
    dots = np.abs(np.sum(attitudes[:-1] * attitudes[1:], axis=1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def _project_with_lower_bounds(
    source_dt_s: np.ndarray,
    lower_dt_s: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Return the unique minimum-L2 fixed-sum vector above lower bounds."""

    source_dt_s = np.asarray(source_dt_s, dtype=np.float64)
    lower_dt_s = np.asarray(lower_dt_s, dtype=np.float64)
    _require(source_dt_s.shape == lower_dt_s.shape, "dt/lower-bound shape mismatch")
    residual_duration = float(duration_s - np.sum(lower_dt_s))
    tolerance = 1.0e-12 * max(1.0, abs(duration_s))
    _require(
        residual_duration >= -tolerance,
        "fixed duration is shorter than the sum of interval lower bounds",
    )
    residual_duration = max(0.0, residual_duration)

    unconstrained = source_dt_s - lower_dt_s
    order = np.sort(unconstrained)[::-1]
    cumulative = np.cumsum(order)
    active = np.flatnonzero(
        order - (cumulative - residual_duration) / np.arange(1, len(order) + 1) > 0.0
    )
    if active.size == 0:
        theta = float((cumulative[-1] - residual_duration) / len(order))
    else:
        rho = int(active[-1])
        theta = float((cumulative[rho] - residual_duration) / (rho + 1))
    projected = np.maximum(unconstrained - theta, 0.0)

    # Bind the floating-point residual to one deterministic active interval.
    correction_index = int(np.argmax(projected))
    projected[correction_index] += residual_duration - float(np.sum(projected))
    result = lower_dt_s + projected
    _require(np.all(result > 0.0), "projection produced a non-positive interval")
    _require(
        np.all(result >= lower_dt_s - tolerance),
        "projection violated an interval lower bound",
    )
    _require(
        abs(float(np.sum(result)) - duration_s) <= tolerance,
        "projection did not preserve total duration",
    )
    return result


def _contiguous_regions(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return ()
    regions: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            regions.append((start, previous))
            start = current
        previous = current
    regions.append((start, previous))
    return tuple(regions)


def derive_time_reparameterization(
    source_time_s: np.ndarray,
    positions_m: np.ndarray,
    attitudes_xyzw: np.ndarray,
    config: TimeReparameterizationConfig = TimeReparameterizationConfig(),
) -> TimeReparameterizationResult:
    source_time = np.asarray(source_time_s, dtype=np.float64)
    positions = np.asarray(positions_m, dtype=np.float64)
    attitudes = np.asarray(attitudes_xyzw, dtype=np.float64)
    count = len(source_time)
    _require(count >= 2, "trajectory must contain at least two poses")
    _require(source_time.shape == (count,) and np.isfinite(source_time).all(), "bad time")
    _require(positions.shape == (count, 3) and np.isfinite(positions).all(), "bad positions")
    _require(attitudes.shape == (count, 4), "bad attitudes")
    source_dt = np.diff(source_time)
    _require(np.all(source_dt > 0.0), "source timestamps are not strictly increasing")
    _require(config.translation_speed_cap_mps > 0.0, "translation speed cap must be positive")
    _require(config.angular_speed_cap_radps > 0.0, "angular speed cap must be positive")
    _require(config.minimum_interval_dt_s > 0.0, "minimum interval dt must be positive")
    _require(
        config.time_allocation_strategy
        in {"minimum_l2", "proportional_lower_bounds"},
        "unsupported time allocation strategy",
    )
    localized_values = (
        config.localized_transition_start_1based,
        config.localized_transition_end_1based,
        config.localized_translation_speed_cap_mps,
    )
    localized_enabled = any(value is not None for value in localized_values)
    _require(
        not localized_enabled or all(value is not None for value in localized_values),
        "localized translation cap requires start, end, and speed",
    )
    _require(
        1
        <= config.diagnostic_transition_start_1based
        <= config.diagnostic_transition_end_1based
        <= count - 1,
        "diagnostic transition window is outside the trajectory",
    )

    distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angle = _quaternion_interval_angles(attitudes)
    translation_speed_cap = np.full_like(source_dt, config.translation_speed_cap_mps)
    if localized_enabled:
        localized_start = int(config.localized_transition_start_1based)
        localized_end = int(config.localized_transition_end_1based)
        localized_cap = float(config.localized_translation_speed_cap_mps)
        _require(
            1 <= localized_start <= localized_end <= count - 1,
            "localized transition window is outside the trajectory",
        )
        _require(localized_cap > 0.0, "localized translation speed cap must be positive")
        _require(
            localized_cap <= config.translation_speed_cap_mps,
            "localized translation speed cap must not exceed the global cap",
        )
        translation_speed_cap[localized_start - 1 : localized_end] = localized_cap
    minimum_dt = np.full_like(source_dt, config.minimum_interval_dt_s)
    lower_dt = np.maximum.reduce(
        (
            minimum_dt,
            distance / translation_speed_cap,
            angle / config.angular_speed_cap_radps,
        )
    )
    duration = float(source_time[-1] - source_time[0])
    if config.time_allocation_strategy == "minimum_l2":
        derived_dt = _project_with_lower_bounds(source_dt, lower_dt, duration)
    else:
        lower_sum = float(np.sum(lower_dt))
        tolerance = 1.0e-12 * max(1.0, duration)
        _require(
            lower_sum <= duration + tolerance,
            "fixed duration is shorter than the sum of interval lower bounds",
        )
        derived_dt = lower_dt * (duration / lower_sum)
    derived_time = np.concatenate(
        ([source_time[0]], source_time[0] + np.cumsum(derived_dt))
    )
    derived_time[-1] = source_time[-1]
    derived_dt = np.diff(derived_time)

    tolerance = 1.0e-12 * max(1.0, duration)
    _require(np.all(derived_dt > 0.0), "derived timestamps are not strictly increasing")
    _require(
        np.all(derived_dt >= lower_dt - tolerance),
        "derived timestamps violate a speed lower bound",
    )
    _require(derived_time[0] == source_time[0], "derived start timestamp changed")
    _require(derived_time[-1] == source_time[-1], "derived end timestamp changed")

    source_translation_speed = distance / source_dt
    derived_translation_speed = distance / derived_dt
    source_angular_speed = angle / source_dt
    derived_angular_speed = angle / derived_dt
    changed_tolerance = 1.0e-12
    return TimeReparameterizationResult(
        source_time_s=source_time.copy(),
        derived_time_s=derived_time,
        source_dt_s=source_dt,
        derived_dt_s=derived_dt,
        segment_distance_m=distance,
        segment_angle_rad=angle,
        translation_speed_cap_mps=translation_speed_cap,
        lower_dt_s=lower_dt,
        source_translation_speed_mps=source_translation_speed,
        derived_translation_speed_mps=derived_translation_speed,
        source_angular_speed_radps=source_angular_speed,
        derived_angular_speed_radps=derived_angular_speed,
        recovery_regions=_contiguous_regions(derived_dt < source_dt - changed_tolerance),
        slowdown_regions=_contiguous_regions(derived_dt > source_dt + changed_tolerance),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _region_records(regions: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [
        {
            "first_interval_0based": start,
            "last_interval_0based": end,
            "first_transition_1based": start + 1,
            "last_transition_1based": end + 1,
            "source_anchor_from_0based": start,
            "source_anchor_to_0based": end + 1,
            "interval_count": end - start + 1,
        }
        for start, end in regions
    ]


def _result_metrics(
    result: TimeReparameterizationResult,
    config: TimeReparameterizationConfig,
) -> dict[str, object]:
    timestamp_shift = result.derived_time_s - result.source_time_s
    hard = slice(
        config.diagnostic_transition_start_1based - 1,
        config.diagnostic_transition_end_1based,
    )
    metrics: dict[str, object] = {
        "time_allocation_strategy": config.time_allocation_strategy,
        "source_duration_s": float(result.source_time_s[-1] - result.source_time_s[0]),
        "derived_duration_s": float(result.derived_time_s[-1] - result.derived_time_s[0]),
        "position_deviation_max_m": 0.0,
        "position_deviation_rms_m": 0.0,
        "orientation_deviation_max_rad": 0.0,
        "orientation_deviation_rms_rad": 0.0,
        "cartesian_arc_length_relative_error": 0.0,
        "max_abs_timestamp_shift_s": float(np.max(np.abs(timestamp_shift))),
        "rms_timestamp_shift_s": float(np.sqrt(np.mean(timestamp_shift**2))),
        "source_dt_s": _distribution(result.source_dt_s),
        "derived_dt_s": _distribution(result.derived_dt_s),
        "source_translation_speed_mps": _distribution(
            result.source_translation_speed_mps
        ),
        "derived_translation_speed_mps": _distribution(
            result.derived_translation_speed_mps
        ),
        "source_angular_speed_radps": _distribution(result.source_angular_speed_radps),
        "derived_angular_speed_radps": _distribution(
            result.derived_angular_speed_radps
        ),
        "translation_speed_cap_mps": _distribution(
            result.translation_speed_cap_mps
        ),
        "lower_dt_sum_s": float(np.sum(result.lower_dt_s)),
        "duration_to_lower_bound_scale": float(
            (result.derived_time_s[-1] - result.derived_time_s[0])
            / np.sum(result.lower_dt_s)
        ),
        "slowdown_duration_added_s": float(
            np.sum(np.maximum(result.derived_dt_s - result.source_dt_s, 0.0))
        ),
        "recovery_duration_removed_s": float(
            np.sum(np.maximum(result.source_dt_s - result.derived_dt_s, 0.0))
        ),
        "slowdown_regions": _region_records(result.slowdown_regions),
        "recovery_regions": _region_records(result.recovery_regions),
        "diagnostic_window": {
            "numbering_contract": "transition k is source anchor k-1 to k",
            "first_transition_1based": config.diagnostic_transition_start_1based,
            "last_transition_1based": config.diagnostic_transition_end_1based,
            "observed_hard_transition_1based": 198,
            "observed_hard_source_anchor_from_0based": 197,
            "observed_hard_source_anchor_to_0based": 198,
            "source_translation_speed_max_mps": float(
                np.max(result.source_translation_speed_mps[hard])
            ),
            "derived_translation_speed_max_mps": float(
                np.max(result.derived_translation_speed_mps[hard])
            ),
            "source_angular_speed_max_radps": float(
                np.max(result.source_angular_speed_radps[hard])
            ),
            "derived_angular_speed_max_radps": float(
                np.max(result.derived_angular_speed_radps[hard])
            ),
        },
    }
    if config.localized_transition_start_1based is not None:
        start = int(config.localized_transition_start_1based)
        end = int(config.localized_transition_end_1based)
        localized = slice(start - 1, end)
        metrics["localized_translation_cap_window"] = {
            "numbering_contract": "transition k is source anchor k-1 to k",
            "first_transition_1based": start,
            "last_transition_1based": end,
            "translation_speed_cap_mps": float(
                config.localized_translation_speed_cap_mps
            ),
            "cartesian_arc_length_m": float(
                np.sum(result.segment_distance_m[localized])
            ),
            "derived_duration_s": float(np.sum(result.derived_dt_s[localized])),
            "derived_translation_speed_max_mps": float(
                np.max(result.derived_translation_speed_mps[localized])
            ),
        }
    return metrics


def _interval_csv(result: TimeReparameterizationResult) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "interval_index",
            "transition_1based",
            "source_anchor_from_0based",
            "source_anchor_to_0based",
            "source_dt_s",
            "derived_dt_s",
            "lower_dt_s",
            "translation_speed_cap_mps",
            "distance_m",
            "angle_rad",
            "source_translation_speed_mps",
            "derived_translation_speed_mps",
            "source_angular_speed_radps",
            "derived_angular_speed_radps",
            "role",
        )
    )
    for index in range(len(result.source_dt_s)):
        if result.derived_dt_s[index] > result.source_dt_s[index] + 1.0e-12:
            role = "slowdown"
        elif result.derived_dt_s[index] < result.source_dt_s[index] - 1.0e-12:
            role = "recovery"
        else:
            role = "unchanged"
        writer.writerow(
            (
                index,
                index + 1,
                index,
                index + 1,
                repr(float(result.source_dt_s[index])),
                repr(float(result.derived_dt_s[index])),
                repr(float(result.lower_dt_s[index])),
                repr(float(result.translation_speed_cap_mps[index])),
                repr(float(result.segment_distance_m[index])),
                repr(float(result.segment_angle_rad[index])),
                repr(float(result.source_translation_speed_mps[index])),
                repr(float(result.derived_translation_speed_mps[index])),
                repr(float(result.source_angular_speed_radps[index])),
                repr(float(result.derived_angular_speed_radps[index])),
                role,
            )
        )
    return stream.getvalue().encode("utf-8")


def _load_raw_integrity_seed(
    seed_package_dir: Path,
    episode_index: int,
    *,
    expected_manifest_sha256: str,
    expected_seed_sha256: str,
) -> tuple[Path, dict[str, np.ndarray]]:
    seed_package_dir = seed_package_dir.resolve()
    manifest_path = seed_package_dir / "manifest.json"
    _require(manifest_path.is_file(), "raw integrity-seed manifest is missing")
    _require(
        sha256(manifest_path) == expected_manifest_sha256,
        "raw integrity-seed manifest hash mismatch",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == "gik_exact_source_teacher_integrity_canaries_v1",
        "wrong raw integrity-seed package schema",
    )
    _require(manifest.get("valid_for_training") is False, "raw seed package claims training")
    episode_dir = seed_package_dir / f"episode_{episode_index:04d}"
    paths = sorted(episode_dir.glob("*.npz"))
    _require(len(paths) == 1, f"expected one raw seed NPZ for episode {episode_index}")
    seed_path = paths[0]
    _require(sha256(seed_path) == expected_seed_sha256, "raw integrity-seed hash mismatch")
    try:
        with np.load(seed_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        raise ValueError("raw integrity seed is unreadable") from exc
    _require(
        bool(np.asarray(arrays.get("valid_for_training", True)).reshape(-1)[0]) is False,
        "raw integrity seed claims training validity",
    )
    _require(
        int(np.asarray(arrays.get("episode_index", -1)).reshape(-1)[0]) == episode_index,
        "raw integrity-seed episode mismatch",
    )
    return seed_path, arrays


def _finite_difference_base_actions(
    base_arm_q: np.ndarray,
    time_s: np.ndarray,
    maximum_linear_velocity: float,
    maximum_angular_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require(maximum_linear_velocity > 0.0, "seed maximum linear velocity must be positive")
    _require(maximum_angular_velocity > 0.0, "seed maximum angular velocity must be positive")
    dt = np.diff(time_s)
    _require(np.all(dt > 0.0), "derived seed time must be strictly increasing")
    base = np.asarray(base_arm_q[:, :3], dtype=np.float64)
    velocity_world = np.diff(base[:, :2], axis=0) / dt[:, None]
    yaw_delta = (np.diff(base[:, 2]) + np.pi) % (2.0 * np.pi) - np.pi
    yaw_rate = yaw_delta / dt
    cosine = np.cos(base[:-1, 2])
    sine = np.sin(base[:-1, 2])
    vx_body = cosine * velocity_world[:, 0] + sine * velocity_world[:, 1]
    vy_body = -sine * velocity_world[:, 0] + cosine * velocity_world[:, 1]
    raw = np.column_stack(
        (
            vx_body / maximum_linear_velocity,
            vy_body / maximum_linear_velocity,
            yaw_rate / maximum_angular_velocity,
        )
    )
    return np.clip(raw, -1.0, 1.0).astype(np.float32), raw


def _seed_state_rate_diagnostics(
    base_arm_q: np.ndarray,
    result: TimeReparameterizationResult,
) -> dict[str, object]:
    base_arm_q = np.asarray(base_arm_q, dtype=np.float64)
    _require(
        base_arm_q.shape == (len(result.derived_time_s), 6),
        "integrity-seed state shape differs from derived reference",
    )
    base_distance = np.linalg.norm(np.diff(base_arm_q[:, :2], axis=0), axis=1)
    yaw_delta = np.abs(
        (np.diff(base_arm_q[:, 2]) + np.pi) % (2.0 * np.pi) - np.pi
    )
    arm_delta = np.abs(np.diff(base_arm_q[:, 3:], axis=0))

    def rates(dt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            base_distance / dt_s,
            yaw_delta / dt_s,
            arm_delta / dt_s[:, None],
        )

    source_base, source_yaw, source_arm = rates(result.source_dt_s)
    derived_base, derived_yaw, derived_arm = rates(result.derived_dt_s)
    source_arm_any = np.max(source_arm, axis=1)
    derived_arm_any = np.max(derived_arm, axis=1)

    base_lower = base_distance / RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS
    yaw_lower = yaw_delta / RETARGET_MAXIMUM_YAW_RATE_RADPS
    arm_lower = np.max(arm_delta, axis=1) / RETARGET_MAXIMUM_ARM_RATE_RADPS
    combined_lower = np.maximum.reduce((result.lower_dt_s, base_lower, yaw_lower, arm_lower))
    return {
        "role": "integrity_seed_prior_only_non_authoritative_not_policy_labels",
        "seed_state_rate_bounds_used_in_projection": False,
        "reason_not_used_as_constraints": (
            "The free-GIK holonomic seed is a branch prior, not an executable teacher. "
            "Constraining the semantic clock to its branch switches would create "
            "artificial pauses; "
            "the downstream solver may select a different continuous branch."
        ),
        "unchanged_retarget_limits": {
            "maximum_linear_velocity_mps": RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS,
            "maximum_yaw_rate_radps": RETARGET_MAXIMUM_YAW_RATE_RADPS,
            "maximum_arm_rate_radps": RETARGET_MAXIMUM_ARM_RATE_RADPS,
        },
        "source_clock": {
            "base_translation_rate_max_mps": float(np.max(source_base)),
            "yaw_rate_max_radps": float(np.max(source_yaw)),
            "arm_rate_max_radps": float(np.max(source_arm)),
            "arm_rate_max_by_axis_radps": np.max(source_arm, axis=0).tolist(),
            "base_limit_violation_count": int(
                np.count_nonzero(source_base > RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS)
            ),
            "yaw_limit_violation_count": int(
                np.count_nonzero(source_yaw > RETARGET_MAXIMUM_YAW_RATE_RADPS)
            ),
            "arm_limit_violation_count": int(
                np.count_nonzero(source_arm_any > RETARGET_MAXIMUM_ARM_RATE_RADPS)
            ),
        },
        "derived_clock": {
            "base_translation_rate_max_mps": float(np.max(derived_base)),
            "yaw_rate_max_radps": float(np.max(derived_yaw)),
            "arm_rate_max_radps": float(np.max(derived_arm)),
            "arm_rate_max_by_axis_radps": np.max(derived_arm, axis=0).tolist(),
            "base_limit_violation_count": int(
                np.count_nonzero(derived_base > RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS)
            ),
            "yaw_limit_violation_count": int(
                np.count_nonzero(derived_yaw > RETARGET_MAXIMUM_YAW_RATE_RADPS)
            ),
            "arm_limit_violation_count": int(
                np.count_nonzero(derived_arm_any > RETARGET_MAXIMUM_ARM_RATE_RADPS)
            ),
            "new_base_limit_violation_count": int(
                np.count_nonzero(
                    (source_base <= RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS)
                    & (derived_base > RETARGET_MAXIMUM_LINEAR_VELOCITY_MPS)
                )
            ),
            "new_yaw_limit_violation_count": int(
                np.count_nonzero(
                    (source_yaw <= RETARGET_MAXIMUM_YAW_RATE_RADPS)
                    & (derived_yaw > RETARGET_MAXIMUM_YAW_RATE_RADPS)
                )
            ),
            "new_arm_limit_violation_count": int(
                np.count_nonzero(
                    (source_arm_any <= RETARGET_MAXIMUM_ARM_RATE_RADPS)
                    & (derived_arm_any > RETARGET_MAXIMUM_ARM_RATE_RADPS)
                )
            ),
        },
        "counterfactual_all_seed_rate_bounds": {
            "lower_dt_sum_s": float(np.sum(combined_lower)),
            "maximum_required_interval_dt_s": float(np.max(combined_lower)),
            "feasible_with_fixed_duration": bool(
                np.sum(combined_lower)
                <= result.derived_time_s[-1] - result.derived_time_s[0] + 1.0e-12
            ),
            "rejected_for_this_derivation": True,
        },
    }


def _paired_seed_payload(
    raw_arrays: dict[str, np.ndarray],
    result: TimeReparameterizationResult,
    *,
    raw_seed_sha256: str,
    raw_manifest_sha256: str,
    raw_source_json_sha256: str,
    derived_source_json_sha256: str,
    time_warp_sha256: str,
) -> dict[str, np.ndarray | str | np.generic]:
    required = {
        "base_arm_actions",
        "q_current_base_arm_6",
        "q_next_base_arm_6",
        "desired_position_full_m",
        "desired_attitude_full_world_dfr_quat_wxyz",
        "max_linear_velocity",
        "max_angular_velocity",
    }
    _require(required <= raw_arrays.keys(), "raw integrity seed is missing required arrays")
    current = np.asarray(raw_arrays["q_current_base_arm_6"], dtype=np.float32)
    next_q = np.asarray(raw_arrays["q_next_base_arm_6"], dtype=np.float32)
    _require(current.shape == next_q.shape, "raw integrity-seed transition shape mismatch")
    base_arm_q = np.vstack((current, next_q[-1]))
    _require(len(base_arm_q) == len(result.derived_time_s), "raw seed/reference count mismatch")
    maximum_linear_velocity = float(np.asarray(raw_arrays["max_linear_velocity"]).item())
    maximum_angular_velocity = float(np.asarray(raw_arrays["max_angular_velocity"]).item())
    base_actions, base_raw = _finite_difference_base_actions(
        base_arm_q,
        result.derived_time_s,
        maximum_linear_velocity,
        maximum_angular_velocity,
    )
    raw_actions = np.asarray(raw_arrays["base_arm_actions"], dtype=np.float32)
    _require(raw_actions.shape == (len(result.derived_dt_s), 6), "raw action shape mismatch")
    derived_actions = raw_actions.copy()
    derived_actions[:, 3:] = base_actions

    payload: dict[str, np.ndarray | str | np.generic] = {
        key: np.asarray(value).copy() for key, value in raw_arrays.items()
    }
    payload.update(
        {
            "schema": PAIRED_SEED_SCHEMA,
            "base_arm_actions": derived_actions,
            "time_s": result.derived_time_s[1:].astype(np.float64),
            "dt_s": result.derived_dt_s.astype(np.float32),
            "desired_time_full_s": result.derived_time_s.astype(np.float64),
            "max_timestamp_error_s": np.float64(0.0),
            "max_abs_base_action_unclipped": np.float32(np.max(np.abs(base_raw))),
            "source_json_sha256": derived_source_json_sha256,
            "raw_integrity_seed_sha256": raw_seed_sha256,
            "raw_integrity_seed_manifest_sha256": raw_manifest_sha256,
            "raw_source_json_sha256": raw_source_json_sha256,
            "derived_source_json_sha256": derived_source_json_sha256,
            "time_warp_sha256": time_warp_sha256,
            "time_reparameterization_contract": DERIVATION_CONTRACT,
            "quality_hold_reasons": np.asarray(
                [
                    "free_gik_seed_only",
                    "derived_timing_not_cpu_feasibility_qualified",
                    "tracking_error_not_qualified",
                ]
            ),
            "valid_for_training": np.bool_(False),
            "valid_for_candidate_training": np.bool_(False),
            "teacher_quality_passed": np.bool_(False),
            "runtime_approved": np.bool_(False),
        }
    )
    return payload


def _save_npz_durable(path: Path, payload: dict[str, object]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_durable(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def derive_ep4_time_warp_package(
    reference_package_dir: Path,
    raw_integrity_seed_package_dir: Path,
    output_dir: Path,
    *,
    config: TimeReparameterizationConfig = TimeReparameterizationConfig(),
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_episodes: int = 79,
    expected_raw_seed_manifest_sha256: str = EXPECTED_RAW_SEED_MANIFEST_SHA256,
    expected_raw_seed_sha256: str = EXPECTED_RAW_EP4_SEED_SHA256,
) -> dict[str, object]:
    reference_package_dir = reference_package_dir.resolve()
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"output path already exists: {output_dir}")
    references = discover_exact_source_references(
        reference_package_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_episodes=expected_episodes,
    )
    _require(config.episode_index in references, "requested episode is absent")
    reference = references[config.episode_index]
    raw_seed_path, raw_seed_arrays = _load_raw_integrity_seed(
        raw_integrity_seed_package_dir,
        config.episode_index,
        expected_manifest_sha256=expected_raw_seed_manifest_sha256,
        expected_seed_sha256=expected_raw_seed_sha256,
    )
    source_payload = json.loads(reference.source_json.read_text(encoding="utf-8"))
    source_poses = source_payload["poses"]
    source_positions = np.asarray([pose["position"] for pose in source_poses], dtype=np.float64)
    source_attitudes = np.asarray(
        [pose["orientation"] for pose in source_poses], dtype=np.float64
    )
    result = derive_time_reparameterization(
        reference.time_s,
        source_positions,
        source_attitudes,
        config,
    )
    raw_current = np.asarray(raw_seed_arrays["q_current_base_arm_6"], dtype=np.float32)
    raw_next = np.asarray(raw_seed_arrays["q_next_base_arm_6"], dtype=np.float32)
    raw_base_arm_q = np.vstack((raw_current, raw_next[-1]))

    derived_payload = json.loads(reference.source_json.read_text(encoding="utf-8"))
    for pose, timestamp in zip(derived_payload["poses"], result.derived_time_s, strict=True):
        pose["time"] = float(timestamp)
    derived_source_bytes = (
        json.dumps(derived_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    intervals_bytes = _interval_csv(result)

    position_hash = array_sha256(source_positions)
    attitude_hash = array_sha256(source_attitudes)
    derived_source_sha256 = hashlib.sha256(derived_source_bytes).hexdigest()
    time_warp_identity = {
        "derivation_contract": DERIVATION_CONTRACT,
        "implementation_file_sha256": sha256(Path(__file__).resolve()),
        "config": asdict(config),
        "source_reference_manifest_sha256": reference.manifest_sha256,
        "source_json_sha256": reference.source_json_sha256,
        "raw_integrity_seed_manifest_sha256": expected_raw_seed_manifest_sha256,
        "raw_integrity_seed_sha256": expected_raw_seed_sha256,
        "source_time_array_sha256": array_sha256(reference.time_s),
        "derived_time_array_sha256": array_sha256(result.derived_time_s),
        "position_array_sha256": position_hash,
        "orientation_array_sha256": attitude_hash,
    }
    time_warp_sha256 = _canonical_json_sha256(time_warp_identity)
    paired_seed_payload = _paired_seed_payload(
        raw_seed_arrays,
        result,
        raw_seed_sha256=expected_raw_seed_sha256,
        raw_manifest_sha256=expected_raw_seed_manifest_sha256,
        raw_source_json_sha256=reference.source_json_sha256,
        derived_source_json_sha256=derived_source_sha256,
        time_warp_sha256=time_warp_sha256,
    )
    metrics = _result_metrics(result, config)
    metrics["integrity_seed_rate_diagnostics"] = _seed_state_rate_diagnostics(
        raw_base_arm_q, result
    )
    path_length = float(np.sum(result.segment_distance_m))
    manifest: dict[str, object] = {
        "schema": PACKAGE_SCHEMA,
        "derivation_contract": DERIVATION_CONTRACT,
        "array_hash_contract": ARRAY_HASH_CONTRACT,
        "episode_index": config.episode_index,
        "source": {
            "reference_package_manifest_sha256": reference.manifest_sha256,
            "source_json_sha256": reference.source_json_sha256,
            "pose_count": len(reference.time_s),
            "time_array_sha256": array_sha256(reference.time_s),
            "position_array_sha256": position_hash,
            "orientation_array_sha256": attitude_hash,
            "duration_s": float(reference.time_s[-1] - reference.time_s[0]),
            "cartesian_arc_length_m": path_length,
        },
        "derived": {
            "source_json": "source.json",
            "source_json_sha256": derived_source_sha256,
            "interval_evidence_csv": "intervals.csv",
            "interval_evidence_csv_sha256": hashlib.sha256(intervals_bytes).hexdigest(),
            "time_array_sha256": array_sha256(result.derived_time_s),
            "position_array_sha256": position_hash,
            "orientation_array_sha256": attitude_hash,
            "duration_s": float(result.derived_time_s[-1] - result.derived_time_s[0]),
            "cartesian_arc_length_m": path_length,
        },
        "time_warp_identity": time_warp_identity,
        "time_warp_sha256": time_warp_sha256,
        "paired_integrity_seed": {},
        "producer": {
            "implementation_file": (
                "src/rl_platform/tasks/two_wheel_balance/ep4_time_reparameterization.py"
            ),
            "implementation_file_sha256": time_warp_identity["implementation_file_sha256"],
        },
        "config": asdict(config),
        "metrics": metrics,
        "constraints": {
            "pose_count_preserved": True,
            "positions_byte_identical": True,
            "orientations_byte_identical": True,
            "first_pose_preserved": True,
            "last_pose_preserved": True,
            "start_timestamp_preserved": True,
            "end_timestamp_preserved": True,
            "total_duration_preserved": True,
            "timestamps_strictly_increasing": True,
            "cartesian_arc_length_relative_error": 0.0,
            "derivation_translation_speed_cap_mps": config.translation_speed_cap_mps,
            "localized_transition_start_1based": (
                config.localized_transition_start_1based
            ),
            "localized_transition_end_1based": config.localized_transition_end_1based,
            "localized_translation_speed_cap_mps": (
                config.localized_translation_speed_cap_mps
            ),
            "solver_or_admission_gates_modified": False,
        },
        "artifact_classification": "derived_duration_preserving_time_reparameterization",
        "quality_qualified_teacher": False,
        "valid_for_dynamic_evaluation": False,
        "valid_for_training": False,
        "training_started": False,
        "permitted_use": "CPU-only ep4 retarget feasibility input after separate review.",
        "forbidden_use": (
            "No Isaac playback, BC, PPO, or training admission from this derivation alone."
        ),
    }
    stage_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    _require(not stage_dir.exists(), f"staging path already exists: {stage_dir}")
    stage_dir.mkdir(parents=True)
    try:
        _write_durable(stage_dir / "source.json", derived_source_bytes)
        _write_durable(stage_dir / "intervals.csv", intervals_bytes)
        seed_episode_dir = (
            stage_dir
            / "paired_integrity_seed"
            / f"episode_{config.episode_index:04d}"
        )
        seed_episode_dir.mkdir(parents=True)
        _write_durable(seed_episode_dir / "source.json", derived_source_bytes)
        seed_name = f"episode_{config.episode_index:04d}_time_warp_integrity_seed_v1.npz"
        paired_seed_path = seed_episode_dir / seed_name
        _save_npz_durable(paired_seed_path, paired_seed_payload)
        paired_seed_sha256 = sha256(paired_seed_path)
        raw_base_actions = np.asarray(raw_seed_arrays["base_arm_actions"], dtype=np.float32)
        paired_base_actions = np.asarray(paired_seed_payload["base_arm_actions"], dtype=np.float32)
        paired_seed_manifest = {
            "schema": PAIRED_SEED_PACKAGE_SCHEMA,
            "episode_index": config.episode_index,
            "trajectory_integrity_contract": "exact_source_v1",
            "time_reparameterization_contract": DERIVATION_CONTRACT,
            "time_warp_sha256": time_warp_sha256,
            "raw_integrity_seed_manifest_sha256": expected_raw_seed_manifest_sha256,
            "raw_integrity_seed_sha256": expected_raw_seed_sha256,
            "raw_integrity_seed_filename": raw_seed_path.name,
            "derived_source_json_sha256": derived_source_sha256,
            "bundled_source_json": f"episode_{config.episode_index:04d}/source.json",
            "bundled_source_json_sha256": derived_source_sha256,
            "output_npz": f"episode_{config.episode_index:04d}/{seed_name}",
            "output_npz_sha256": paired_seed_sha256,
            "state_arrays_byte_identical": True,
            "pose_arrays_byte_identical": True,
            "time_dependent_base_actions_recomputed": True,
            "base_action_transition_count": len(paired_base_actions),
            "base_action_changed_transition_count": int(
                np.count_nonzero(
                    np.any(raw_base_actions[:, 3:] != paired_base_actions[:, 3:], axis=1)
                )
            ),
            "raw_max_abs_base_action_unclipped": float(
                np.asarray(raw_seed_arrays["max_abs_base_action_unclipped"]).item()
            ),
            "derived_max_abs_base_action_unclipped": float(
                np.asarray(paired_seed_payload["max_abs_base_action_unclipped"]).item()
            ),
            "quality_qualified_teacher": False,
            "valid_for_dynamic_evaluation": False,
            "valid_for_training": False,
            "training_started": False,
        }
        paired_seed_manifest_bytes = (
            json.dumps(
                paired_seed_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_durable(
            stage_dir / "paired_integrity_seed" / "manifest.json",
            paired_seed_manifest_bytes,
        )
        manifest["paired_integrity_seed"] = {
            "package_dir": "paired_integrity_seed",
            "manifest_sha256": hashlib.sha256(paired_seed_manifest_bytes).hexdigest(),
            "output_npz_sha256": paired_seed_sha256,
            "raw_integrity_seed_manifest_sha256": expected_raw_seed_manifest_sha256,
            "raw_integrity_seed_sha256": expected_raw_seed_sha256,
            "time_warp_sha256": time_warp_sha256,
            "valid_for_training": False,
        }
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_durable(stage_dir / "manifest.json", manifest_bytes)
        os.replace(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return manifest


def verify_ep4_time_warp_package(
    reference_package_dir: Path,
    raw_integrity_seed_package_dir: Path,
    derived_package_dir: Path,
    *,
    expected_episodes: int = 79,
) -> dict[str, object]:
    derived_package_dir = derived_package_dir.resolve()
    manifest_path = derived_package_dir / "manifest.json"
    _require(manifest_path.is_file(), "derived manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == PACKAGE_SCHEMA, "wrong derived package schema")
    _require(
        manifest.get("derivation_contract") == DERIVATION_CONTRACT,
        "wrong time-reparameterization contract",
    )
    _require(manifest.get("valid_for_training") is False, "derived package claims training")
    source_record = manifest.get("source")
    derived_record = manifest.get("derived")
    _require(isinstance(source_record, dict), "missing source provenance")
    _require(isinstance(derived_record, dict), "missing derived provenance")
    expected_manifest = str(source_record.get("reference_package_manifest_sha256", ""))
    references = discover_exact_source_references(
        reference_package_dir,
        expected_manifest_sha256=expected_manifest,
        expected_episodes=expected_episodes,
    )
    episode_index = int(manifest.get("episode_index", -1))
    _require(episode_index in references, "derived episode is absent from source package")
    reference = references[episode_index]
    _require(
        reference.source_json_sha256 == source_record.get("source_json_sha256"),
        "source JSON provenance hash changed",
    )

    source_path = derived_package_dir / str(derived_record.get("source_json", ""))
    intervals_path = derived_package_dir / str(
        derived_record.get("interval_evidence_csv", "")
    )
    _require(source_path.is_file(), "derived source JSON is missing")
    _require(intervals_path.is_file(), "derived interval evidence is missing")
    _require(
        sha256(source_path) == derived_record.get("source_json_sha256"),
        "derived JSON hash mismatch",
    )
    _require(
        sha256(intervals_path) == derived_record.get("interval_evidence_csv_sha256"),
        "derived interval evidence hash mismatch",
    )

    source_payload = json.loads(reference.source_json.read_text(encoding="utf-8"))
    derived_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_poses = source_payload.get("poses")
    derived_poses = derived_payload.get("poses")
    _require(isinstance(source_poses, list) and isinstance(derived_poses, list), "poses missing")
    _require(len(source_poses) == len(derived_poses), "derived pose count changed")
    source_positions = np.asarray([pose["position"] for pose in source_poses], dtype=np.float64)
    derived_positions = np.asarray([pose["position"] for pose in derived_poses], dtype=np.float64)
    source_attitudes = np.asarray([pose["orientation"] for pose in source_poses], dtype=np.float64)
    derived_attitudes = np.asarray(
        [pose["orientation"] for pose in derived_poses], dtype=np.float64
    )
    derived_time = np.asarray([pose["time"] for pose in derived_poses], dtype=np.float64)
    _require(
        array_sha256(source_positions) == source_record.get("position_array_sha256"),
        "source position provenance hash changed",
    )
    _require(
        array_sha256(source_attitudes) == source_record.get("orientation_array_sha256"),
        "source orientation provenance hash changed",
    )
    _require(source_positions.tobytes() == derived_positions.tobytes(), "positions changed")
    _require(source_attitudes.tobytes() == derived_attitudes.tobytes(), "orientations changed")
    _require(
        array_sha256(derived_time) == derived_record.get("time_array_sha256"),
        "derived timestamp hash mismatch",
    )

    config_record = manifest.get("config")
    _require(isinstance(config_record, dict), "derived config is missing")
    try:
        config = TimeReparameterizationConfig(**config_record)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid derived config") from exc
    expected = derive_time_reparameterization(
        reference.time_s,
        source_positions,
        source_attitudes,
        config,
    )
    _require(
        array_sha256(expected.derived_time_s) == array_sha256(derived_time),
        "derived timestamp mapping is not deterministic",
    )
    _require(derived_time[0] == reference.time_s[0], "derived start changed")
    _require(derived_time[-1] == reference.time_s[-1], "derived duration changed")
    _require(np.all(np.diff(derived_time) > 0.0), "derived timestamps are not strict")

    paired_record = manifest.get("paired_integrity_seed")
    _require(isinstance(paired_record, dict), "paired integrity-seed provenance is missing")
    expected_identity = {
        "derivation_contract": DERIVATION_CONTRACT,
        "implementation_file_sha256": sha256(Path(__file__).resolve()),
        "config": config_record,
        "source_reference_manifest_sha256": reference.manifest_sha256,
        "source_json_sha256": reference.source_json_sha256,
        "raw_integrity_seed_manifest_sha256": paired_record.get(
            "raw_integrity_seed_manifest_sha256"
        ),
        "raw_integrity_seed_sha256": paired_record.get("raw_integrity_seed_sha256"),
        "source_time_array_sha256": array_sha256(reference.time_s),
        "derived_time_array_sha256": array_sha256(derived_time),
        "position_array_sha256": array_sha256(source_positions),
        "orientation_array_sha256": array_sha256(source_attitudes),
    }
    _require(
        manifest.get("time_warp_identity") == expected_identity,
        "time-warp identity fields changed",
    )
    _require(
        manifest.get("time_warp_sha256") == _canonical_json_sha256(expected_identity),
        "time-warp identity hash mismatch",
    )
    seed_package_dir = derived_package_dir / str(paired_record.get("package_dir", ""))
    seed_manifest_path = seed_package_dir / "manifest.json"
    _require(seed_manifest_path.is_file(), "paired integrity-seed manifest is missing")
    _require(
        sha256(seed_manifest_path) == paired_record.get("manifest_sha256"),
        "paired integrity-seed manifest hash mismatch",
    )
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    _require(seed_manifest.get("schema") == PAIRED_SEED_PACKAGE_SCHEMA, "wrong paired seed schema")
    _require(seed_manifest.get("valid_for_training") is False, "paired seed claims training")
    _require(
        seed_manifest.get("time_warp_sha256") == manifest.get("time_warp_sha256"),
        "paired seed time-warp provenance mismatch",
    )
    bundled_seed_source = seed_package_dir / str(seed_manifest.get("bundled_source_json", ""))
    _require(bundled_seed_source.is_file(), "paired seed bundled source is missing")
    _require(
        sha256(bundled_seed_source) == seed_manifest.get("bundled_source_json_sha256"),
        "paired seed bundled source hash mismatch",
    )
    raw_seed_path, raw_arrays = _load_raw_integrity_seed(
        raw_integrity_seed_package_dir,
        episode_index,
        expected_manifest_sha256=str(
            paired_record.get("raw_integrity_seed_manifest_sha256", "")
        ),
        expected_seed_sha256=str(paired_record.get("raw_integrity_seed_sha256", "")),
    )
    paired_seed_path = seed_package_dir / str(seed_manifest.get("output_npz", ""))
    _require(paired_seed_path.is_file(), "paired integrity-seed NPZ is missing")
    _require(
        sha256(paired_seed_path) == seed_manifest.get("output_npz_sha256"),
        "paired integrity-seed NPZ hash mismatch",
    )
    try:
        with np.load(paired_seed_path, allow_pickle=False) as data:
            paired_arrays = {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        raise ValueError("paired integrity seed is unreadable") from exc
    for key in (
        "q_current_base_arm_6",
        "q_next_base_arm_6",
        "desired_position_full_m",
        "desired_attitude_full_world_dfr_quat_wxyz",
    ):
        _require(
            np.asarray(raw_arrays[key]).tobytes() == np.asarray(paired_arrays[key]).tobytes(),
            f"paired integrity-seed array changed: {key}",
        )
    _require(
        np.asarray(raw_arrays["base_arm_actions"])[:, :3].tobytes()
        == np.asarray(paired_arrays["base_arm_actions"])[:, :3].tobytes(),
        "paired integrity-seed arm actions changed",
    )
    paired_current = np.asarray(paired_arrays["q_current_base_arm_6"], dtype=np.float32)
    paired_next = np.asarray(paired_arrays["q_next_base_arm_6"], dtype=np.float32)
    paired_base_arm_q = np.vstack((paired_current, paired_next[-1]))
    expected_base_actions, expected_base_raw = _finite_difference_base_actions(
        paired_base_arm_q,
        derived_time,
        float(np.asarray(paired_arrays["max_linear_velocity"]).item()),
        float(np.asarray(paired_arrays["max_angular_velocity"]).item()),
    )
    _require(
        np.asarray(paired_arrays["base_arm_actions"])[:, 3:].tobytes()
        == expected_base_actions.tobytes(),
        "paired integrity-seed base actions were not recomputed",
    )
    _require(
        np.isclose(
            float(np.asarray(paired_arrays["max_abs_base_action_unclipped"]).item()),
            float(np.max(np.abs(expected_base_raw))),
            atol=1.0e-6,
            rtol=1.0e-6,
        ),
        "paired integrity-seed base-action metadata changed",
    )
    _require(
        np.array_equal(np.asarray(paired_arrays["desired_time_full_s"]), derived_time),
        "paired integrity-seed full timestamps differ from derived reference",
    )
    _require(
        np.array_equal(np.asarray(paired_arrays["time_s"]), derived_time[1:]),
        "paired integrity-seed transition timestamps differ from derived reference",
    )
    _require(
        np.allclose(
            np.asarray(paired_arrays["dt_s"], dtype=np.float64),
            np.diff(derived_time),
            atol=1.0e-8,
            rtol=1.0e-6,
        ),
        "paired integrity-seed dt differs from derived reference",
    )
    _require(
        str(np.asarray(paired_arrays["raw_integrity_seed_sha256"]).item())
        == sha256(raw_seed_path),
        "paired integrity-seed raw provenance mismatch",
    )
    _require(
        str(np.asarray(paired_arrays["time_warp_sha256"]).item())
        == manifest.get("time_warp_sha256"),
        "paired integrity-seed time-warp hash mismatch",
    )
    return manifest


def load_ep4_time_warp_reference(
    reference_package_dir: Path,
    raw_integrity_seed_package_dir: Path,
    derived_package_dir: Path,
    *,
    expected_episodes: int = 79,
) -> ExactSourceReference:
    """Load the verified derived clock without weakening the raw reference loader."""

    manifest = verify_ep4_time_warp_package(
        reference_package_dir,
        raw_integrity_seed_package_dir,
        derived_package_dir,
        expected_episodes=expected_episodes,
    )
    source_path = derived_package_dir.resolve() / str(manifest["derived"]["source_json"])
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    poses = payload["poses"]
    return ExactSourceReference(
        episode_index=int(manifest["episode_index"]),
        source_json=source_path,
        source_json_sha256=str(manifest["derived"]["source_json_sha256"]),
        manifest_sha256=sha256(derived_package_dir.resolve() / "manifest.json"),
        time_s=np.asarray([pose["time"] for pose in poses], dtype=np.float64),
        positions_m=np.asarray([pose["position"] for pose in poses], dtype=np.float64),
        attitudes_xyzw=np.asarray([pose["orientation"] for pose in poses], dtype=np.float64),
    )
