import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/select_model_based_corrective_pair_tranche.py"
)
SPEC = importlib.util.spec_from_file_location("pair_tranche_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VALIDATION_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/"
    "select_model_based_corrective_validation_tranche.py"
)
VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "corrective_validation_selector", VALIDATION_SCRIPT
)
VALIDATION_MODULE = importlib.util.module_from_spec(VALIDATION_SPEC)
assert VALIDATION_SPEC.loader is not None
VALIDATION_SPEC.loader.exec_module(VALIDATION_MODULE)
VALIDATION_EVIDENCE = (
    Path(__file__).parents[1]
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_validation_tranche_v1/"
    "selection.json"
)
VALIDATION_EVIDENCE_SHA256 = (
    "5576c696e304eb9b9a173970e5fed06e887eccefe2d65a20678415148e22fa0b"
)


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    portfolio_path = tmp_path / "portfolio/manifest.json"
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    train_cases = [2, 4, 6, 30, 31, 52]
    validation_cases = [8, 16, 22, 32]
    cases = train_cases + validation_cases
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
                "checks": {
                    "row_dynamic_quality": True,
                    "gate_dynamic_quality": True,
                    "result_dynamic_quality": True,
                    "thermal": True,
                },
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
    # A later reject may make the enclosing batch summary false without
    # invalidating this already sealed per-case dynamic result.
    dynamic["rows"][0]["batch_summary_dynamic_quality_passed"] = False
    _write(dynamic_path, dynamic)
    split_path = tmp_path / "split.json"
    split = {
        "schema": MODULE.SPLIT_SCHEMA,
        "admitted_split_cases": {
            "train": train_cases,
            "validation": validation_cases + [78],
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


def test_validation_selector_chooses_two_diverse_closed_cases(tmp_path) -> None:
    values = _fixture(tmp_path)
    result = VALIDATION_MODULE.build_validation_selection(
        *values[:4],
        portfolio_path=values[4],
        dynamic_path=values[5],
        split_path=values[6],
        conversion_audit_path=values[7],
        tranche_size=2,
    )
    assert result["schema"] == VALIDATION_MODULE.SCHEMA
    assert len(result["selected_cases"]) == 2
    assert len(set(result["selected_cases"])) == 2
    assert set(result["selected_cases"]) <= {8, 16, 22, 32}
    assert result["eligible_validation_cases"] == [8, 16, 22, 32]
    assert result["excluded_validation_rows"] == [
        {"case": 78, "reason": "missing_portfolio_case"}
    ]
    assert all(
        row["selection_role"]
        == "same_seed_validation_paired_canary_required"
        for row in result["selected_rows"]
    )
    assert result["same_seed_pair_required_before_capture"] is True
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["dataset_conversion_authorized"] is False
    assert result["dataset_merge_authorized"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["valid_for_training"] is False


def test_validation_selector_cli_imports_from_an_unrelated_cwd(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATION_SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--dynamic-selection" in result.stdout


def test_committed_validation_selection_is_exact_and_runtime_closed() -> None:
    raw = VALIDATION_EVIDENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VALIDATION_EVIDENCE_SHA256
    result = json.loads(raw)
    assert result["schema"] == VALIDATION_MODULE.SCHEMA
    assert result["selected_cases"] == [8, 16]
    assert result["eligible_validation_cases"] == [8, 16, 22, 32]
    assert result["excluded_validation_rows"] == [
        {"case": 78, "reason": "not_dynamically_quality_qualified"}
    ]
    assert result["checks"]["validation_train_disjoint"] is True
    assert result["checks"]["validation_holdout_disjoint"] is True
    assert result["same_seed_pair_required_before_capture"] is True
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "label_capture_authorized",
        "dataset_conversion_authorized",
        "dataset_merge_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert result[field] is False


def test_validation_selector_rejects_insufficient_dynamic_candidates(
    tmp_path,
) -> None:
    values = list(_fixture(tmp_path))
    values[1]["rows"] = [
        row for row in values[1]["rows"] if row["case"] not in {16, 22, 32}
    ]
    with pytest.raises(ValueError, match="not enough dynamically qualified"):
        VALIDATION_MODULE.build_validation_selection(
            *values[:4],
            portfolio_path=values[4],
            dynamic_path=values[5],
            split_path=values[6],
            conversion_audit_path=values[7],
            tranche_size=2,
        )


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
        values[1]["rows"][0]["checks"]["row_dynamic_quality"] = False
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
