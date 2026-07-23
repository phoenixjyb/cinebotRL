#!/usr/bin/env python3
"""Audit case 7 before designing a case-specific paired canary."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
BASE_AUDITOR = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case6_pair_readiness.py"
)
SCHEMA = "cinebotrl_two_wheel_riser_case7_pair_readiness_cpu_v1"
CASE = 7
DEFAULT_SELECTION = (
    DOC_ROOT / "evidence_20260723_model_based_pair_tranche_v1/selection.json"
)
DEFAULT_PLAN = (
    DOC_ROOT
    / "evidence_20260724_case7_pair_readiness_cpu_v1/source/"
    "case_0007_smoothed_riser_plan_v1.npz"
)
DEFAULT_GATE = (
    DOC_ROOT
    / "evidence_20260724_case7_pair_readiness_cpu_v1/source/"
    "case_0007_dynamic_gate.json"
)


def _load_base_auditor():
    spec = importlib.util.spec_from_file_location(
        "case7_pair_readiness_shared_checks", BASE_AUDITOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared readiness checks: {BASE_AUDITOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CASE = CASE
    module.SCHEMA = SCHEMA
    return module


_BASE = _load_base_auditor()


def audit_readiness(
    selection_path: Path,
    plan_path: Path,
    gate_path: Path,
) -> dict[str, object]:
    """Apply the frozen paired-canary checks with case-7 identities."""

    result = _BASE.audit_readiness(selection_path, plan_path, gate_path)
    result["selection_checks"]["case7_role"] = result["selection_checks"].pop(
        "case6_role"
    )
    result["selection_checks"]["case7_checks"] = result[
        "selection_checks"
    ].pop("case6_checks")
    with np.load(plan_path, allow_pickle=False) as data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        execution_time_s = np.asarray(
            data["execution_time_s"], dtype=np.float64
        )
        source_time_s = np.asarray(data["source_time_s"], dtype=np.float64)
        source_anchor_execution_index = np.asarray(
            data["source_anchor_execution_index"], dtype=np.int64
        )
        initialization_time_s = np.asarray(
            data["initialization_time_s"], dtype=np.float64
        )
        initialization_state = np.asarray(
            data["initialization_state"], dtype=np.float64
        )
    sample_count = int(result["plan"]["sample_count"])
    source_execution_clock_checks = {
        "execution_clock_alias_exact": bool(
            np.array_equal(time_s, execution_time_s)
        ),
        "source_clock_shape": source_time_s.shape == (sample_count,),
        "source_clock_strictly_increasing": bool(
            source_time_s.shape == (sample_count,)
            and abs(float(source_time_s[0])) <= 1e-12
            and np.all(np.diff(source_time_s) > 0.0)
        ),
        "source_duration_metadata": bool(
            source_time_s.shape == (sample_count,)
            and np.isclose(
                float(source_time_s[-1]),
                float(result["plan"]["source_duration_s"]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "execution_duration_metadata": bool(
            time_s.shape == (sample_count,)
            and np.isclose(
                float(time_s[-1]),
                float(result["plan"]["execution_duration_s"]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "source_anchor_map_exact": bool(
            source_anchor_execution_index.shape == (sample_count,)
            and np.array_equal(
                source_anchor_execution_index,
                np.arange(sample_count, dtype=np.int64),
            )
        ),
        "initialization_separate_and_empty": bool(
            initialization_time_s.shape == (0,)
            and initialization_state.shape == (0, 7)
        ),
    }
    if not all(source_execution_clock_checks.values()):
        raise ValueError(
            "case-7 source/execution clock checks failed: "
            f"{source_execution_clock_checks}"
        )
    bounded_window_found = bool(
        result["profile_window_contract"]["bounded_window_found"]
    )
    result.update(
        {
            "schema": SCHEMA,
            "case": CASE,
            "source_execution_clock_checks": source_execution_clock_checks,
            "case23_profile_reuse_authorized": False,
            "case6_profile_reuse_authorized": False,
            "case2_profile_reuse_authorized": False,
            "pair_profile_cpu_ready": False,
            "safe_window_absent_requires_structural_profile": (
                not bounded_window_found
            ),
            "next_bounded_action": (
                "design_case7_specific_corrective_profile_cpu_only"
            ),
            "runtime_authorized": False,
            "gpu_launch_authorized": False,
            "label_capture_authorized": False,
            "dataset_conversion_authorized": False,
            "dataset_merge_authorized": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
            "valid_for_training": False,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dynamic-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_readiness(args.selection, args.plan, args.dynamic_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
