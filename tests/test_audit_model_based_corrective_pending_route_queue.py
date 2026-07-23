import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_pending_route_queue.py"
)
SPEC = importlib.util.spec_from_file_location("pending_route_queue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


HEAD = "1" * 40


def _closed_fields() -> dict[str, bool]:
    return {field: False for field in MODULE.FALSE_FIELDS}


def _preflight(spec, root: Path) -> Path:
    payload = {
        "schema": spec.schema,
        "case": spec.case,
        "split": spec.split,
        "namespace": spec.namespace,
        "runtime_commit": HEAD,
        "upstream_commit": HEAD,
        "identities": {
            f"identity_{index}": {"passed": True}
            for index in range(spec.identity_count)
        },
        "checks": {"healthy": True},
        "cpu_contract_ready": True,
        "runtime_route_contract_ready": True,
        "execution_route_complete": True,
        "authorization_file": None,
        "passed": True,
        **_closed_fields(),
    }
    if spec.kind == "cpu_conversion":
        payload["git"] = {"head": HEAD, "upstream": HEAD}
        payload["repository_checks"] = {"healthy": True}
        payload["contract_checks"] = {"healthy": True}
        payload["authorization_checks"] = {
            "authorization_file_present": False,
            "authorization_hash_matches": False,
        }
    path = root / f"{spec.key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _corpus(root: Path) -> Path:
    payload = {
        "schema": MODULE.CORPUS_SCHEMA,
        "passed": True,
        "converted_train_cases": [30],
        "converted_validation_cases": [],
        "pending_minimum_train_cases": [23, 6, 2],
        "pending_validation_cases": [8, 16],
        "next_bounded_action": (
            "authorize_exactly_one_case23_v4_cpu_conversion"
        ),
        "case23_conversion_authorized": False,
        "case23_conversion_output_created": False,
        "corpus_manifest_ready": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "valid_for_bc_admission_review": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    path = root / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit(tmp_path: Path):
    preflights = {
        spec.key: _preflight(spec, tmp_path) for spec in MODULE.ROUTES
    }
    return MODULE.audit_pending_route_queue(
        repo_root=tmp_path,
        corpus_intake_path=_corpus(tmp_path),
        preflight_paths=preflights,
        current_head=HEAD,
        upstream_head=HEAD,
        tracked_worktree_clean=True,
    ), preflights


def test_healthy_queue_is_ordered_and_closed(tmp_path: Path) -> None:
    result, _ = _audit(tmp_path)
    assert result["passed"] is True
    assert result["ready_route_count"] == 6
    assert result["execution_order"] == [
        "case23_conversion",
        "case6_pair",
        "case2_pair",
        "case7_pair",
        "case8_validation_pair",
        "case16_validation_pair",
    ]
    assert result["next_bounded_action"] == (
        "authorize_exactly_one_case23_v4_cpu_conversion"
    )
    assert all(route["passed"] for route in result["routes"])
    assert result["runtime_authorized"] is False
    assert result["dataset_conversion_authorized"] is False
    assert result["training_started"] is False


@pytest.mark.parametrize(
    ("route_key", "field", "value"),
    [
        ("case23_conversion", "conversion_authorized", True),
        ("case6_pair", "runtime_authorized", True),
        ("case7_pair", "dataset_conversion_authorized", True),
        ("case8_validation_pair", "training_started", True),
    ],
)
def test_queue_rejects_any_open_route(
    tmp_path: Path,
    route_key: str,
    field: str,
    value: object,
) -> None:
    _, preflights = _audit(tmp_path)
    payload = json.loads(preflights[route_key].read_text())
    payload[field] = value
    preflights[route_key].write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.audit_pending_route_queue(
        repo_root=tmp_path,
        corpus_intake_path=_corpus(tmp_path),
        preflight_paths=preflights,
        current_head=HEAD,
        upstream_head=HEAD,
        tracked_worktree_clean=True,
    )
    assert result["passed"] is False
    row = next(row for row in result["routes"] if row["key"] == route_key)
    assert row["checks"]["authorization_and_learning_closed"] is False


def test_queue_rejects_stale_preflight_commit(tmp_path: Path) -> None:
    _, preflights = _audit(tmp_path)
    payload = json.loads(preflights["case2_pair"].read_text())
    payload["runtime_commit"] = "2" * 40
    preflights["case2_pair"].write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.audit_pending_route_queue(
        repo_root=tmp_path,
        corpus_intake_path=_corpus(tmp_path),
        preflight_paths=preflights,
        current_head=HEAD,
        upstream_head=HEAD,
        tracked_worktree_clean=True,
    )
    row = next(row for row in result["routes"] if row["key"] == "case2_pair")
    assert row["checks"]["head_and_upstream_current"] is False
    assert result["passed"] is False


def test_queue_rejects_existing_runtime_namespace(tmp_path: Path) -> None:
    result, preflights = _audit(tmp_path)
    assert result["passed"] is True
    namespace = MODULE.ROUTES[2].namespace
    (tmp_path / "artifacts/two_wheel_riser" / namespace).mkdir(
        parents=True
    )
    result = MODULE.audit_pending_route_queue(
        repo_root=tmp_path,
        corpus_intake_path=_corpus(tmp_path),
        preflight_paths=preflights,
        current_head=HEAD,
        upstream_head=HEAD,
        tracked_worktree_clean=True,
    )
    row = next(row for row in result["routes"] if row["key"] == "case2_pair")
    assert row["checks"]["namespace_absent"] is False
    assert result["passed"] is False


def test_queue_rejects_corpus_state_drift(tmp_path: Path) -> None:
    _, preflights = _audit(tmp_path)
    corpus = _corpus(tmp_path)
    payload = json.loads(corpus.read_text())
    payload["converted_train_cases"] = [23, 30]
    corpus.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.audit_pending_route_queue(
        repo_root=tmp_path,
        corpus_intake_path=corpus,
        preflight_paths=preflights,
        current_head=HEAD,
        upstream_head=HEAD,
        tracked_worktree_clean=True,
    )
    assert result["checks"]["corpus_state"] is False
    assert result["passed"] is False
