#!/usr/bin/env python3
"""Admit the sealed case-78 shadow series as an offline raw-teacher input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    OBSERVATION_INDEX,
    PREVIOUS_ACTION_INDICES,
    load_shadow_teacher_trace,
)


EXPECTED_HASHES = {
    "final": "63004e41d1185a8589c8715e620a5c976db44bfb6a130786214109a1ab2d5bd7",
    "gate": "ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459",
    "trace": "dc04cbef0aab9960018579292b9ff9ee25e8bd427cd4be641b4a9d96e04525e3",
    "runtime_admission": "7acab3500762f55546d33335df6a60f172d8989b80f88aa6cd37d189aa7cc7b0",
    "scale_audit": "fd2b97d5e4a6cada368f4fb776086ebcab10403df5b350d7fa16a694163b535c",
    "split_admission": "eac2c8c5389b0a8e3590d5b6355eaa80b50019091d5eb906408a6599c19cb623",
}
ACTION_SCALES = np.asarray([0.35, 0.4, 0.1], dtype=np.float64)
SOURCE_DURATION_S = 135.487646
EXECUTION_DURATION_S = 192.29956737098348


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing case-78 label-admission input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_admission(
    final: dict[str, Any],
    gate: dict[str, Any],
    runtime_admission: dict[str, Any],
    scale_audit: dict[str, Any],
    split_admission: dict[str, Any],
    trace_metadata: dict[str, Any],
    trace: dict[str, np.ndarray],
    *,
    expected_row_count: int = 83050,
    source_duration_s: float = SOURCE_DURATION_S,
    execution_duration_s: float = EXECUTION_DURATION_S,
) -> dict[str, Any]:
    raw = np.asarray(trace["shadow_teacher_raw_residual_commands"], dtype=np.float64)
    normalized = np.asarray(
        trace["shadow_teacher_normalized_residual_actions"], dtype=np.float64
    )
    applied = np.asarray(trace["applied_residual_actions"], dtype=np.float64)
    observations = np.asarray(trace["observations"], dtype=np.float64)
    teacher_commands = np.asarray(
        trace["shadow_teacher_high_level_commands"], dtype=np.float64
    )
    elapsed = np.asarray(trace["elapsed_time_s"], dtype=np.float64)
    phase = np.asarray(trace["phase_time_s"], dtype=np.float64)
    cases = np.asarray(trace["case_ids"])
    results = gate.get("results", [])
    gate_case = results[0] if len(results) == 1 else {}
    reconstructed = np.column_stack(
        (
            observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] + raw[:, 0],
            observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]] + raw[:, 1],
            observations[:, OBSERVATION_INDEX["riser_position_m"]] + raw[:, 2],
        )
    )
    overflow = np.abs(raw) >= ACTION_SCALES
    checks = {
        "sealed_final_passed": final.get("passed") is True
        and final.get("physical_quality_passed") is True
        and final.get("shadow_trace_passed") is True
        and final.get("candidate_scale_overflow_passed") is True,
        "runtime_evidence_is_non_training": final.get("labels_applied_to_commands")
        is False
        and final.get("dataset_created") is False
        and final.get("valid_for_training") is False
        and final.get("bc_authorized") is False
        and final.get("ppo_authorized") is False,
        "gate_is_same_passed_case": gate.get("cases") == [78]
        and gate.get("passed") is True
        and gate_case.get("case") == 78
        and gate_case.get("dynamic_quality_passed") is True
        and gate_case.get("thermal_admission_passed") is True
        and gate_case.get("controller_evidence_passed") is True
        and gate_case.get("termination") is None,
        "separate_clocks_exact": final.get("source_duration_s") == source_duration_s
        and final.get("execution_duration_s") == execution_duration_s
        and phase[-1] == execution_duration_s,
        "runtime_admission_exact": runtime_admission.get("case") == 78
        and runtime_admission.get("shadow_measurement_authorized") is True
        and runtime_admission.get("dataset_creation_authorized") is False
        and runtime_admission.get("bc_authorized") is False
        and runtime_admission.get("ppo_authorized") is False,
        "split_role_exact": split_admission.get("split_admitted") is True
        and split_admission.get("admitted_split_cases", {}).get("validation")
        == [8, 16, 22, 32, 78]
        and split_admission.get("admitted_split_cases", {}).get("holdout")
        == [3, 5, 13, 19, 24]
        and split_admission.get("holdout_opened") is False,
        "scale_contract_exact": scale_audit.get("teacher40_action_contract_retained")
        is True
        and np.allclose(
            np.asarray(scale_audit.get("teacher40_candidate_scale")),
            ACTION_SCALES,
            atol=1e-15,
            rtol=0.0,
        )
        and scale_audit.get("action_clipping_permitted") is False,
        "trace_is_deterministic_shadow_only": trace_metadata.get(
            "visited_state_source"
        )
        == "deterministic_controller"
        and trace_metadata.get("shadow_teacher_applied_to_commands") is False
        and trace_metadata.get("trace_only") is True
        and trace_metadata.get("valid_for_training") is False,
        "semantic_camera_contract": gate.get("position_observation_link")
        == "physical_cam_link_fk"
        and gate.get("target_attitude_contract")
        == "semantic_dfr_to_physical_cam_v1",
        "single_case_policy_rate_series": len(raw) == expected_row_count
        and np.array_equal(np.unique(cases), [78])
        and abs(float(elapsed[0])) <= 1e-12
        and np.all(np.diff(elapsed) > 0.0)
        and np.all(np.diff(phase) >= 0.0),
        "zero_applied_actions": bool(np.all(applied == 0.0)),
        "zero_previous_action_placeholders": bool(
            np.all(observations[:, PREVIOUS_ACTION_INDICES] == 0.0)
        ),
        "raw_normalization_exact": bool(
            np.allclose(raw, normalized * ACTION_SCALES, atol=2e-7, rtol=0.0)
        ),
        "teacher_command_reconstruction_exact": bool(
            np.allclose(reconstructed, teacher_commands, atol=2e-6, rtol=0.0)
        ),
        "zero_candidate_scale_overflow": bool(np.all(~overflow)),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"case-78 shadow-label admission failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_label_admission_v1",
        "case": 78,
        "split": "validation",
        "source_duration_s": source_duration_s,
        "execution_duration_s": execution_duration_s,
        "row_count": len(raw),
        "action_scales": ACTION_SCALES.tolist(),
        "raw_residual_abs_max": np.max(np.abs(raw), axis=0).tolist(),
        "normalized_residual_abs_max": np.max(np.abs(normalized), axis=0).tolist(),
        "overflow_sample_count": np.sum(overflow, axis=0).tolist(),
        "input_contract_checks": checks,
        "label_admission_passed": True,
        "raw_teacher_conversion_authorized": True,
        "offline_dataset_rebuild_authorized": True,
        "labels_applied_to_commands": False,
        "historical_dataset_rewrite_authorized": False,
        "holdout_opened": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--runtime-admission", type=Path, required=True)
    parser.add_argument("--scale-audit", type=Path, required=True)
    parser.add_argument("--split-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite label admission: {args.output}")
    paths = {
        "final": args.final,
        "gate": args.gate,
        "trace": args.trace,
        "runtime_admission": args.runtime_admission,
        "scale_audit": args.scale_audit,
        "split_admission": args.split_admission,
    }
    identities = {name: identity(path) for name, path in paths.items()}
    if any(
        identities[name]["sha256"] != EXPECTED_HASHES[name] for name in paths
    ):
        raise ValueError("case-78 label-admission input hash mismatch")
    trace_metadata, trace = load_shadow_teacher_trace(args.trace)
    result = build_admission(
        _load_json(args.final),
        _load_json(args.gate),
        _load_json(args.runtime_admission),
        _load_json(args.scale_audit),
        _load_json(args.split_admission),
        trace_metadata,
        trace,
    )
    if result["row_count"] != _load_json(args.final).get(
        "shadow_label_statistics", {}
    ).get("row_count"):
        raise ValueError("case-78 final/trace row count mismatch")
    result["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
