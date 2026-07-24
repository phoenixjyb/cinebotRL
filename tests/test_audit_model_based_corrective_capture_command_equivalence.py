import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
AUDITOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_capture_command_equivalence.py"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_capture_command_contract_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "corrective_capture_command_auditor",
        AUDITOR,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _manifest() -> dict:
    return json.loads(CONTRACT.read_text())


def _route(case: int) -> dict:
    return next(row for row in _manifest()["routes"] if row["case"] == case)


def test_current_routes_are_command_compatible_and_case30_is_quarantined() -> None:
    report = MODULE.build_report(ROOT, CONTRACT, enforce_repository=False)
    routes = {route["case"]: route for route in report["routes"]}
    assert report["passed"] is True
    assert routes[6]["current_command_compatible"] is True
    assert routes[23]["current_command_compatible"] is True
    assert routes[7]["current_command_compatible"] is True
    assert routes[30]["current_command_compatible"] is False
    assert routes[30]["mismatches"] == [
        "--corrective-teacher-capture-split",
        "playback_identity",
    ]
    assert routes[30]["classification_passed"] is True
    assert report["generic_runtime_wrapper_created"] is False
    assert report["runtime_authorized"] is False
    assert report["capture_started"] is False
    assert report["dataset_created"] is False
    assert report["training_started"] is False


def test_windows_repository_path_maps_to_wsl() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"


def test_case7_every_option_and_identity_is_bound() -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    result = MODULE.audit_route(ROOT, _route(7), manifest, defaults)
    assert result["current_command_compatible"] is True
    assert result["mismatches"] == []
    assert result["unknown_options"] == []
    assert all(result["io_checks"].values())
    assert all(result["contract_checks"].values())
    assert all(
        row["passed"] for row in result["semantic_options"].values()
    )
    assert all(
        row["passed"]
        for row in result["identity_option_bindings"].values()
    )
    assert result["playback_binding_passed"] is True


def test_controller_argument_drift_fails_closed(monkeypatch) -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    original = MODULE.parse_playback_options

    def drifted(source):
        options = original(source)
        options["--controller-wz-kp"] = "0.90"
        return options

    monkeypatch.setattr(MODULE, "parse_playback_options", drifted)
    result = MODULE.audit_route(ROOT, _route(7), manifest, defaults)
    assert result["current_command_compatible"] is False
    assert "--controller-wz-kp" in result["mismatches"]


def test_missing_case7_split_fails_closed(monkeypatch) -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    original = MODULE.parse_playback_options

    def missing_split(source):
        options = original(source)
        options.pop("--corrective-teacher-capture-split")
        return options

    monkeypatch.setattr(MODULE, "parse_playback_options", missing_split)
    result = MODULE.audit_route(ROOT, _route(7), manifest, defaults)
    assert result["current_command_compatible"] is False
    assert result["semantic_options"]["--corrective-teacher-capture-split"][
        "actual"
    ] is None


def test_unknown_runtime_option_fails_closed(monkeypatch) -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    original = MODULE.parse_playback_options

    def injected(source):
        options = original(source)
        options["--disable-phase-governor"] = True
        return options

    monkeypatch.setattr(MODULE, "parse_playback_options", injected)
    result = MODULE.audit_route(ROOT, _route(7), manifest, defaults)
    assert result["current_command_compatible"] is False
    assert result["unknown_options"] == ["--disable-phase-governor"]


def test_identity_path_drift_fails_closed(monkeypatch) -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    original = MODULE.parse_assignments

    def drifted(source):
        assignments = original(source)
        assignments["CORRECTIVE_PROFILE"] = (
            assignments["CORRECTIVE_PROFILE"] + ".forged"
        )
        return assignments

    monkeypatch.setattr(MODULE, "parse_assignments", drifted)
    result = MODULE.audit_route(ROOT, _route(7), manifest, defaults)
    assert result["current_command_compatible"] is False
    assert "--corrective-teacher-profile" in result["mismatches"]


def test_open_authorization_contract_fails_closed(tmp_path) -> None:
    manifest = _manifest()
    defaults = MODULE.parse_playback_defaults(ROOT / manifest["playback"])
    route = dict(_route(7))
    contract = json.loads((ROOT / route["contract"]).read_text())
    contract["runtime_authorized"] = True
    forged = tmp_path / "contract.json"
    forged.write_text(json.dumps(contract))
    route["contract"] = str(forged)
    result = MODULE.audit_route(ROOT, route, manifest, defaults)
    assert result["current_command_compatible"] is False
    assert result["contract_checks"]["authorization_closed"] is False
