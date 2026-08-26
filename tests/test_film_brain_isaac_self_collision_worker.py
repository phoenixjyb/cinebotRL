"""Pure protocol tests for the opt-in Film Brain Isaac collision worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "film_brain"
    / "isaac_self_collision_worker.py"
)
SPEC = importlib.util.spec_from_file_location("isaac_self_collision_worker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _evaluation_request() -> dict:
    return {
        "schema_version": MODULE.REQUEST_SCHEMA,
        "request_id": "unit-evaluation-request",
        "operation": "EVALUATE_DISCRETE_SELF_COLLISION_ONLY",
        "authority": dict(MODULE.AUTHORITY),
        "robot_model": {
            "robot_model_id": "recomo-proto2-isaac-usd",
            "robot_model_version": "recomoProto2-1190-wrapper-baked-scaled",
            "expected_usd_sha256": "0" * 64,
        },
        "collision_policy": {
            "policy_id": "recomo-proto2-isaac-discrete-self-collision-v1",
            "expected_policy_sha256": "1" * 64,
            "contact_threshold_n": MODULE.CONTACT_THRESHOLD_N,
        },
        "coordinate_order": list(MODULE.COORDINATE_ORDER),
        "samples": [
            {
                "timestamp_ns": 0,
                "positions": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.6],
            }
        ],
        "continuous_collision_required": False,
        "environment_collision_required": False,
    }


def test_contract_is_non_authoritative_and_discrete_only():
    assert MODULE.AUTHORITY == {
        "robot_transport": False,
        "motion_authority": False,
        "physical_feasibility_claimed": False,
        "pnc_execution_semantics_claimed": False,
    }
    assert MODULE.CAPABILITIES["discrete_self_collision"] is True
    assert MODULE.CAPABILITIES["continuous_or_swept_collision"] is False
    assert MODULE.CAPABILITIES["environment_collision"] is False
    assert MODULE.CAPABILITIES["robot_model_equivalent_to_pnc"] is False


def test_evaluation_contract_and_10d_mapping_are_accepted():
    request = _evaluation_request()
    MODULE._validate_common(request)
    MODULE._validate_evaluation_request(request)
    assert MODULE._film_brain_positions_to_isaac_arm(request["samples"][0]["positions"]) == (
        0.1,
        0.2,
        0.3,
        0.6,
        0.5,
        0.4,
    )


def test_nonzero_derived_level_pitch_fails_closed():
    request = _evaluation_request()
    request["samples"][0]["positions"][6] = 0.01
    with pytest.raises(MODULE.RequestError, match="LEVEL_PITCH_MAPPING_UNPROVED"):
        MODULE._validate_evaluation_request(request)


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("continuous_collision_required", "CONTINUOUS_COLLISION_UNSUPPORTED"),
        ("environment_collision_required", "ENVIRONMENT_COLLISION_UNSUPPORTED"),
    ],
)
def test_unsupported_collision_scope_fails_closed(field: str, code: str):
    request = _evaluation_request()
    request[field] = True
    with pytest.raises(MODULE.RequestError, match=code):
        MODULE._validate_evaluation_request(request)


def test_canonical_encoding_is_sorted_compact_utf8_without_newline():
    encoded = MODULE._canonical_bytes({"z": 1, "a": "相机"})
    assert encoded == b'{"a":"\\u76f8\\u673a","z":1}'
    assert not encoded.endswith(b"\n")


def test_isaac_opt_in_is_applied_only_when_runtime_starts(monkeypatch):
    for name in (
        "FILM_BRAIN_ENABLE_SELF_COLLISION",
        "FILM_BRAIN_COLLISION_PAIR_PROBE",
    ):
        monkeypatch.delenv(name, raising=False)
    MODULE._configure_isaac_process_environment()
    assert MODULE.os.environ["FILM_BRAIN_ENABLE_SELF_COLLISION"] == "1"
    assert MODULE.os.environ["FILM_BRAIN_COLLISION_PAIR_PROBE"] == "1"
