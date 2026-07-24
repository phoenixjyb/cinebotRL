import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/finalize_model_based_corrective_capture.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "generic_corrective_capture_finalizer",
        FINALIZER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _route_fixture(
    tmp_path: Path,
    *,
    case: int = 7,
    split: str = "train",
) -> tuple[Path, Path]:
    root = tmp_path / "capture"
    root.mkdir()
    namespace = f"case-{case}-{split}-capture"
    contract = {
        "case": case,
        "split": split,
        "namespace": namespace,
        "identities": {
            f"case{case}_plan": {
                "path": f"case_{case:04d}.npz",
                "sha256": "a" * 64,
            },
            "corrective_profile": {
                "path": "profile.json",
                "sha256": "b" * 64,
            },
        },
    }
    admission = {
        "schema": MODULE.CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "case": case,
        "split": split,
        "namespace": namespace,
        "runtime_authorized": True,
        "label_capture_authorized": True,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    (root / "contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    return root, admission_path


@pytest.mark.parametrize(
    ("case", "split"),
    [(7, "train"), (8, "validation"), (16, "validation")],
)
def test_generic_finalizer_derives_case_split_and_archive_name(
    monkeypatch,
    tmp_path,
    case,
    split,
) -> None:
    root, admission = _route_fixture(
        tmp_path,
        case=case,
        split=split,
    )
    observed = {}

    def fake_summarize(*args, **kwargs):
        observed.update(kwargs)
        return {"passed": True, "case": case, "split": split}

    monkeypatch.setattr(MODULE, "summarize_capture", fake_summarize)
    result = MODULE.finalize(
        root,
        admission,
        runtime_commit="c" * 40,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert observed["expected_case"] == case
    assert observed["expected_split"] == split
    assert observed["expected_namespace"] == f"case-{case}-{split}-capture"
    assert observed["capture_name"] == (
        f"case_{case:04d}_corrective_teacher_capture_v2.npz"
    )
    assert observed["plan_identity_name"] == f"case{case}_plan"
    assert result["generic_finalizer"]["checks"]
    assert all(result["generic_finalizer"]["checks"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case", True),
        ("case", 0),
        ("split", "holdout"),
        ("runtime_authorized", False),
        ("label_capture_authorized", False),
        ("dataset_creation_authorized", True),
        ("bc_authorized", True),
        ("ppo_authorized", True),
        ("training_started", True),
    ],
)
def test_generic_finalizer_rejects_invalid_or_open_route(
    tmp_path,
    field,
    value,
) -> None:
    root, admission = _route_fixture(tmp_path)
    payload = json.loads(admission.read_text())
    payload[field] = value
    if field in {"case", "split"}:
        contract = json.loads((root / "contract.json").read_text())
        contract[field] = value
        (root / "contract.json").write_text(json.dumps(contract))
    admission.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="route mismatch"):
        MODULE.derive_route(root, admission)


def test_generic_finalizer_rejects_ambiguous_plan_identity(tmp_path) -> None:
    root, admission = _route_fixture(tmp_path)
    contract = json.loads((root / "contract.json").read_text())
    contract["identities"]["alternate_plan"] = contract["identities"][
        "case7_plan"
    ]
    (root / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="route mismatch"):
        MODULE.derive_route(root, admission)


def test_generic_finalizer_rejects_noncanonical_runtime_commit(tmp_path) -> None:
    root, admission = _route_fixture(tmp_path)
    with pytest.raises(ValueError, match="runtime commit"):
        MODULE.finalize(
            root,
            admission,
            runtime_commit="main",
            playback_exit_code=0,
            gpu_release_passed=True,
        )
