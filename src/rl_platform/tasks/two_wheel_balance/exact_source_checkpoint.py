"""Fail-closed checkpoints for exact-source nonholonomic retargeting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import zipfile

import numpy as np


CHECKPOINT_SCHEMA = "cinebotrl_two_wheel_exact_source_retarget_checkpoint_v1"
CHECKPOINT_CONTRACT = "exact_source_interval_prefix_v1"
ARRAY_NAMES = (
    "states",
    "controls",
    "target_positions",
    "target_attitudes",
    "execution_time_s",
    "position_errors_m",
    "attitude_errors_deg",
    "previous_control",
    "source_anchor_execution_index_prefix",
)


@dataclass(frozen=True)
class ExactSourceRetargetPrefix:
    states: np.ndarray
    controls: np.ndarray
    target_positions: np.ndarray
    target_attitudes: np.ndarray
    execution_time_s: np.ndarray
    position_errors_m: np.ndarray
    attitude_errors_deg: np.ndarray
    previous_control: np.ndarray
    source_anchor_execution_index_prefix: np.ndarray
    retimed_interval_count: int
    next_source_interval: int

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "states": np.asarray(self.states, dtype=np.float64),
            "controls": np.asarray(self.controls, dtype=np.float64),
            "target_positions": np.asarray(self.target_positions, dtype=np.float64),
            "target_attitudes": np.asarray(self.target_attitudes, dtype=np.float64),
            "execution_time_s": np.asarray(self.execution_time_s, dtype=np.float64),
            "position_errors_m": np.asarray(self.position_errors_m, dtype=np.float64),
            "attitude_errors_deg": np.asarray(self.attitude_errors_deg, dtype=np.float64),
            "previous_control": np.asarray(self.previous_control, dtype=np.float64),
            "source_anchor_execution_index_prefix": np.asarray(
                self.source_anchor_execution_index_prefix, dtype=np.int64
            ),
        }


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def source_time_sha256(source_time_s: np.ndarray) -> str:
    return array_sha256(np.asarray(source_time_s, dtype=np.float64))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_identity(identity: Mapping[str, object]) -> dict[str, object]:
    required = {
        "git_commit",
        "code_contract_sha256",
        "case",
        "retarget_cli_config",
        "retarget_cli_config_sha256",
        "reference_manifest_sha256",
        "reference_episode_sha256",
        "integrity_seed_sha256",
        "target_urdf_sha256",
        "source_pose_count",
        "source_time_sha256",
    }
    _require(set(identity) == required, "checkpoint identity fields differ")
    normalized = json.loads(json.dumps(identity, sort_keys=True))
    _require(
        normalized["retarget_cli_config_sha256"]
        == canonical_json_sha256(normalized["retarget_cli_config"]),
        "checkpoint CLI configuration hash mismatch",
    )
    for key in (
        "git_commit",
        "code_contract_sha256",
        "reference_manifest_sha256",
        "reference_episode_sha256",
        "integrity_seed_sha256",
        "target_urdf_sha256",
        "source_time_sha256",
    ):
        value = normalized[key]
        expected_length = 40 if key == "git_commit" else 64
        _require(
            isinstance(value, str)
            and len(value) == expected_length
            and all(character in "0123456789abcdef" for character in value),
            f"invalid checkpoint identity {key}",
        )
    _require(int(normalized["case"]) > 0, "invalid checkpoint case")
    _require(int(normalized["source_pose_count"]) >= 2, "invalid source pose count")
    return normalized


def _validate_prefix(
    prefix: ExactSourceRetargetPrefix,
    *,
    source_time_s: np.ndarray,
    source_positions_m: np.ndarray,
    source_attitudes_wxyz: np.ndarray,
    expected_anchor: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = prefix.arrays()
    count = len(source_time_s)
    states = arrays["states"]
    controls = arrays["controls"]
    positions = arrays["target_positions"]
    attitudes = arrays["target_attitudes"]
    time_s = arrays["execution_time_s"]
    position_errors = arrays["position_errors_m"]
    attitude_errors = arrays["attitude_errors_deg"]
    previous_control = arrays["previous_control"]
    mapping = arrays["source_anchor_execution_index_prefix"]
    sample_count = len(states)

    _require(np.asarray(source_time_s).shape == (count,), "bad source time shape")
    _require(source_positions_m.shape == (count, 3), "bad source position shape")
    _require(source_attitudes_wxyz.shape == (count, 4), "bad source attitude shape")
    _require(np.array_equal(source_time_s, np.asarray(source_time_s, dtype=np.float64)), "source time dtype changed")
    _require(np.isfinite(source_time_s).all(), "source time is non-finite")
    _require(source_time_s[0] == 0.0 and np.all(np.diff(source_time_s) > 0.0), "source time is not strict")
    _require(1 <= prefix.next_source_interval <= count, "bad next source interval")
    _require(prefix.retimed_interval_count >= 0, "negative retimed interval count")
    _require(states.shape == (sample_count, 9) and sample_count >= 1, "bad checkpoint states")
    _require(controls.shape == (sample_count - 1, 5), "bad checkpoint controls")
    _require(positions.shape == (sample_count, 3), "bad checkpoint positions")
    _require(attitudes.shape == (sample_count, 4), "bad checkpoint attitudes")
    _require(time_s.shape == (sample_count,), "bad checkpoint execution time")
    _require(position_errors.shape == (sample_count,), "bad checkpoint position errors")
    _require(attitude_errors.shape == (sample_count,), "bad checkpoint attitude errors")
    _require(previous_control.shape == (8,), "bad checkpoint previous control")
    _require(mapping.shape == (prefix.next_source_interval,), "checkpoint prefix is incomplete")
    _require(np.isfinite(np.concatenate([array.reshape(-1) for array in arrays.values()])).all(), "checkpoint contains non-finite values")
    _require(time_s[0] == 0.0 and (sample_count == 1 or np.all(np.diff(time_s) > 0.0)), "checkpoint execution time is not strict")
    _require(mapping[0] == 0 and mapping[-1] == sample_count - 1, "checkpoint anchor map endpoints differ")
    _require(len(mapping) == 1 or np.all(np.diff(mapping) > 0), "checkpoint anchor map is not strict")
    _require(np.array_equal(states[0], np.asarray(expected_anchor, dtype=np.float64)), "checkpoint anchor state changed")
    execution_steps_per_source_interval = np.diff(mapping)
    expanded_source_dt = [
            np.full(step_count, source_time_s[index] - source_time_s[index - 1])
            for index, step_count in enumerate(
                execution_steps_per_source_interval, start=1
            )
        ]
    expected_transition_dt = (
        np.concatenate(expanded_source_dt)
        if expanded_source_dt
        else np.empty(0, dtype=np.float64)
    )
    _require(
        np.allclose(
            np.diff(time_s), expected_transition_dt, atol=1e-12, rtol=1e-12
        ),
        "checkpoint execution transition dt differs from source/map expansion",
    )
    _require(
        np.allclose(
            positions[mapping],
            source_positions_m[: prefix.next_source_interval],
            atol=1e-12,
            rtol=0.0,
        ),
        "checkpoint source position prefix changed",
    )
    dots = np.abs(
        np.sum(
            attitudes[mapping]
            * source_attitudes_wxyz[: prefix.next_source_interval],
            axis=1,
        )
    )
    _require(np.all(dots >= 1.0 - 1e-10), "checkpoint source attitude prefix changed")
    _require(
        prefix.retimed_interval_count
        == int(np.count_nonzero(execution_steps_per_source_interval > 1)),
        "checkpoint retimed interval count differs from anchor map",
    )
    return arrays


def save_exact_source_checkpoint(
    path: Path,
    identity: Mapping[str, object],
    prefix: ExactSourceRetargetPrefix,
    *,
    source_time_s: np.ndarray,
    source_positions_m: np.ndarray,
    source_attitudes_wxyz: np.ndarray,
    expected_anchor: np.ndarray,
) -> None:
    path = path.resolve()
    normalized_identity = _validate_identity(identity)
    _require(
        normalized_identity["source_pose_count"] == len(source_time_s),
        "checkpoint source count differs",
    )
    _require(
        normalized_identity["source_time_sha256"] == source_time_sha256(source_time_s),
        "checkpoint source time hash differs",
    )
    arrays = _validate_prefix(
        prefix,
        source_time_s=source_time_s,
        source_positions_m=source_positions_m,
        source_attitudes_wxyz=source_attitudes_wxyz,
        expected_anchor=expected_anchor,
    )
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_contract": CHECKPOINT_CONTRACT,
        "identity": normalized_identity,
        "next_source_interval": prefix.next_source_interval,
        "retimed_interval_count": prefix.retimed_interval_count,
        "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        "valid_for_training": False,
        "training_started": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                checkpoint_metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_exact_source_checkpoint(
    path: Path,
    expected_identity: Mapping[str, object],
    *,
    source_time_s: np.ndarray,
    source_positions_m: np.ndarray,
    source_attitudes_wxyz: np.ndarray,
    expected_anchor: np.ndarray,
) -> ExactSourceRetargetPrefix:
    path = path.resolve()
    _require(path.is_file(), f"missing exact-source checkpoint: {path}")
    expected = _validate_identity(expected_identity)
    _require(
        expected["source_pose_count"] == len(source_time_s),
        "checkpoint source count differs",
    )
    _require(
        expected["source_time_sha256"] == source_time_sha256(source_time_s),
        "checkpoint source time hash differs",
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            _require(set(data.files) == {"checkpoint_metadata_json", *ARRAY_NAMES}, "checkpoint fields differ")
            raw_metadata = np.asarray(data["checkpoint_metadata_json"])
            _require(raw_metadata.size == 1, "checkpoint metadata is not scalar")
            metadata = json.loads(str(raw_metadata.reshape(-1)[0].item()))
            arrays = {key: np.asarray(data[key]) for key in ARRAY_NAMES}
    except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid exact-source checkpoint: {error}") from error
    _require(isinstance(metadata, dict), "checkpoint metadata is not an object")
    _require(metadata.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema differs")
    _require(metadata.get("checkpoint_contract") == CHECKPOINT_CONTRACT, "checkpoint contract differs")
    _require(metadata.get("identity") == expected, "checkpoint identity mismatch")
    _require(metadata.get("valid_for_training") is False, "checkpoint claims training validity")
    _require(metadata.get("training_started") is False, "checkpoint claims training started")
    hashes = metadata.get("array_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == set(ARRAY_NAMES), "checkpoint array hash manifest differs")
    for key, value in arrays.items():
        _require(hashes[key] == array_sha256(value), f"checkpoint {key} checksum mismatch")
    prefix = ExactSourceRetargetPrefix(
        states=arrays["states"],
        controls=arrays["controls"],
        target_positions=arrays["target_positions"],
        target_attitudes=arrays["target_attitudes"],
        execution_time_s=arrays["execution_time_s"],
        position_errors_m=arrays["position_errors_m"],
        attitude_errors_deg=arrays["attitude_errors_deg"],
        previous_control=arrays["previous_control"],
        source_anchor_execution_index_prefix=arrays[
            "source_anchor_execution_index_prefix"
        ],
        retimed_interval_count=int(metadata.get("retimed_interval_count", -1)),
        next_source_interval=int(metadata.get("next_source_interval", -1)),
    )
    try:
        _validate_prefix(
            prefix,
            source_time_s=source_time_s,
            source_positions_m=source_positions_m,
            source_attitudes_wxyz=source_attitudes_wxyz,
            expected_anchor=expected_anchor,
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid exact-source checkpoint prefix: {error}") from error
    return prefix
