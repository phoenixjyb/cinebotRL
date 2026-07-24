import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "prepare_model_based_corrective_routes.py"
)
CATALOG = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_route_catalog_v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_model_based_corrective_routes", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_committed_catalog_matches_canonical_teacher_contract() -> None:
    result = MODULE.validate_catalog(ROOT, CATALOG)
    assert result["passed"] is True
    assert all(result["checks"].values())
    teacher = result["canonical_teacher_contract"]
    assert teacher["observation_dimension"] == 65
    assert teacher["action_dimension"] == 3
    assert teacher["residual_action_scales"] == [0.05, 0.05, 0.02]
    assert teacher["training_target"] == (
        "effective_post_supervisor_residual_action"
    )


def test_catalog_consolidates_pending_routes_without_authorization() -> None:
    result = MODULE.validate_catalog(ROOT, CATALOG)
    assert [route["key"] for route in result["routes"]] == [
        "case7_pair",
        "case8_validation_pair",
        "case16_validation_pair",
    ]
    assert [route["case"] for route in result["routes"]] == [7, 8, 16]
    assert [route["split"] for route in result["routes"]] == [
        "train",
        "validation",
        "validation",
    ]
    assert all(route["passed"] for route in result["routes"])


def test_catalog_rejects_open_authorization(tmp_path: Path) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["runtime_authorized"] = True
    alternate = tmp_path / "catalog.json"
    alternate.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.validate_catalog(ROOT, alternate)
    assert result["checks"]["canonical_path"] is False
    assert result["checks"]["catalog_learning_closed"] is False
    assert result["passed"] is False


def test_catalog_rejects_action_contract_drift(tmp_path: Path) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["canonical_teacher_contract"]["action_dimension"] = 4
    alternate = tmp_path / "catalog.json"
    alternate.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.validate_catalog(ROOT, alternate)
    assert result["checks"]["canonical_teacher_contract"] is False
    assert result["passed"] is False


def test_catalog_rejects_duplicate_case(tmp_path: Path) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["routes"][1]["case"] = 7
    alternate = tmp_path / "catalog.json"
    alternate.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.validate_catalog(ROOT, alternate)
    assert result["checks"]["unique_routes_and_cases"] is False
    assert result["passed"] is False


def test_unknown_route_fails_before_any_runtime_authorization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_run_preflight",
        lambda *_: {"passed": True},
    )
    result = MODULE.prepare_routes(
        ROOT,
        CATALOG,
        ["not_a_route"],
    )
    assert result["checks"]["known_nonempty_selection"] is False
    assert result["runtime_authorized"] is False
    assert result["gpu_launch_authorized"] is False
    assert result["training_started"] is False
    assert result["passed"] is False
