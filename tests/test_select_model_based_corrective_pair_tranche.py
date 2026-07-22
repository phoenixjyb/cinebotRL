import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/select_model_based_corrective_pair_tranche.py"
)
SPEC = importlib.util.spec_from_file_location("pair_tranche_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    portfolio_path = tmp_path / "portfolio/manifest.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [2, 4, 6, 30, 31, 52]
    items = []
    dynamic_rows = []
    for index, case in enumerate(cases):
        plan = portfolio_path.parent / f"case_{case:04d}.npz"
        plan.write_bytes(f"plan:{case}".encode())
        plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
        items.append(
            {
                "case": case,
                "file": plan.name,
                "plan_sha256": plan_sha,
                "source_duration_s": 5.0 + index,
                "execution_source_duration_ratio": 1.2 + 0.1 * index,
                "path_metrics": {"source_path_length_m": 0.5 + index},
                "kinematic_metrics": {
                    "position_error_p95_m": 0.02 * index,
                    "maximum_abs_base_linear_velocity_mps": 0.1 + 0.05 * index,
                    "maximum_abs_base_yaw_rate_radps": 0.05 + 0.07 * index,
                    "maximum_abs_riser_rate_mps": 0.02 + 0.03 * index,
                    "maximum_abs_raw_proxy_target_rate_radps": 0.1 + 0.04 * index,
                    "minimum_target_camera_height_m": 0.7,
                    "maximum_target_camera_height_m": 0.8 + 0.1 * index,
                },
                "timing_transition_kinematic_gate_passed": True,
                "passed": True,
            }
        )
        dynamic_rows.append(
            {
                "case": case,
                "plan_sha256": plan_sha,
                "gate_sha256": f"{index + 1:064x}",
                "raw_residual_command_abs_max": [
                    0.01 + 0.02 * index,
                    0.02 + 0.03 * index,
                    0.003 + 0.004 * index,
                ],
                "batch_summary_dynamic_quality_passed": True,
                "checks": {"dynamic": True, "thermal": True},
            }
        )
    portfolio = {
        "schema": MODULE.PORTFOLIO_SCHEMA,
        "items": items,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
    }
    portfolio_sha = _write(portfolio_path, portfolio)
    dynamic_path = tmp_path / "dynamic.json"
    dynamic = {
        "schema": MODULE.DYNAMIC_SCHEMA,
        "portfolio_manifest_sha256": portfolio_sha,
        "rows": dynamic_rows,
        "passed": True,
        "capture_gate_passed": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    _write(dynamic_path, dynamic)
    split_path = tmp_path / "split.json"
    split = {
        "schema": MODULE.SPLIT_SCHEMA,
        "admitted_split_cases": {
            "train": cases,
            "validation": [8, 16, 22, 32, 78],
            "holdout": MODULE.HOLDOUT_CASES,
        },
        "split_admitted": True,
        "holdout_opened": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    _write(split_path, split)
    audit_path = tmp_path / "audit.json"
    audit = {
        "schema": MODULE.CONVERSION_AUDIT_SCHEMA,
        "case": 30,
        "passed": True,
        "valid_for_case_merge": True,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    _write(audit_path, audit)
    return portfolio, dynamic, split, audit, portfolio_path, dynamic_path, split_path, audit_path


def _select(tmp_path: Path, *, size: int = 5):
    values = _fixture(tmp_path)
    return MODULE.build_selection(
        *values[:4],
        portfolio_path=values[4],
        dynamic_path=values[5],
        split_path=values[6],
        conversion_audit_path=values[7],
        tranche_size=size,
    )


def test_selector_is_deterministic_diverse_and_runtime_closed(tmp_path) -> None:
    result = _select(tmp_path)
    assert result["selected_cases"][0] == 30
    assert len(result["selected_cases"]) == 5
    assert len(set(result["selected_cases"])) == 5
    assert result["selected_rows"][0]["selection_role"] == "converted_pilot_anchor"
    assert all(
        row["selection_role"] == "same_seed_paired_canary_required"
        for row in result["selected_rows"][1:]
    )
    assert result["same_seed_pair_required_before_capture"] is True
    assert result["case30_profile_reuse_authorized"] is False
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["dataset_merge_authorized"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["valid_for_training"] is False


@pytest.mark.parametrize("failure", ["portfolio", "split", "audit", "plan", "dynamic"])
def test_selector_rejects_invalid_or_open_sources(tmp_path, failure) -> None:
    values = list(_fixture(tmp_path))
    if failure == "portfolio":
        values[0]["isaac_started"] = True
    elif failure == "split":
        values[2]["admitted_split_cases"]["holdout"] = [3, 5]
    elif failure == "audit":
        values[3]["valid_for_case_merge"] = False
    elif failure == "plan":
        values[0]["items"][0]["plan_sha256"] = "0" * 64
    else:
        values[1]["rows"][0]["checks"]["dynamic"] = False
    with pytest.raises(ValueError):
        MODULE.build_selection(
            *values[:4],
            portfolio_path=values[4],
            dynamic_path=values[5],
            split_path=values[6],
            conversion_audit_path=values[7],
            tranche_size=5,
        )


def test_selector_rejects_too_small_or_large_tranche(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least three"):
        _select(tmp_path, size=2)
    with pytest.raises(ValueError, match="not enough"):
        _select(tmp_path, size=7)
