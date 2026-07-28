import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT / "scripts/two_wheel_balance/prepare_case32_validation_selection.py"
)
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_selection_cpu_v1/selection.json"
)
SPEC = importlib.util.spec_from_file_location("case32_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_case32_replaces_retired_case16_without_changing_case8() -> None:
    result = MODULE.build_selection()
    assert result["passed"] is True
    assert result["selected_cases"] == [8, 32]
    assert result["retired_validation_cases"] == [16]
    assert [row["case"] for row in result["selected_rows"]] == [8, 32]
    assert result["selected_rows"][0]["plan_sha256"] == (
        "f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655"
    )
    case32 = result["selected_rows"][1]
    assert case32["replaces_case"] == 16
    assert case32["fresh_pair_required"] is True
    assert all(case32["checks"].values())


def test_selection_keeps_every_execution_and_learning_route_closed() -> None:
    result = MODULE.build_selection()
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


def test_selection_rejects_alternate_disposition_bytes(tmp_path: Path) -> None:
    alternate = tmp_path / "disposition.json"
    alternate.write_bytes(MODULE.DEFAULT_DISPOSITION.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="disposition sha256 mismatch"):
        MODULE.build_selection(disposition_path=alternate)


def test_committed_selection_matches_builder() -> None:
    assert json.loads(EVIDENCE.read_text(encoding="utf-8")) == (
        MODULE.build_selection()
    )


def test_cli_regenerates_lf_selection(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == EVIDENCE.read_bytes()
    assert b"\r\n" not in output.read_bytes()
