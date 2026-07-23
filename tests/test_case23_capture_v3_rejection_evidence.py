import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v3_rejected_save_route"
)


def _json(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_rejection_manifest_hashes_every_preserved_runtime_file() -> None:
    manifest = _json("manifest.json")
    assert manifest["runtime_commit"] == (
        "71ed62558dc4588b4f9a39a3b598e3faf636bd5f"
    )
    assert manifest["authorization_consumed"] is True
    assert manifest["authorization_token_deleted"] is True
    assert manifest["retry_authorized"] is False
    for relative, expected_hash in manifest["files"].items():
        assert _sha256(EVIDENCE / relative) == expected_hash


def test_v3_completed_phase_but_archive_failure_created_no_teacher() -> None:
    manifest = _json("manifest.json")
    admission = _json("admission.json")
    gate = _json("case_0023.json")
    heartbeat = _json("runtime_heartbeat.json")
    final_status = _json("final_status.json")
    playback_log = (EVIDENCE / "logs/playback.log").read_text(encoding="utf-8")

    assert admission["case"] == 23
    assert admission["split"] == "train"
    assert admission["runtime_authorized"] is True
    assert admission["label_capture_authorized"] is True
    assert manifest["isaac_initialization_started"] is True
    assert heartbeat["completed_steps"] == 3273
    assert heartbeat["phase_time_s"] == heartbeat["execution_duration_s"]
    assert heartbeat["gate_result_written"] is False
    assert heartbeat["dataset_created"] is False
    assert heartbeat["valid_for_training"] is False
    assert gate["results"][0]["classification"] == "runtime_exception"
    assert gate["results"][0]["exception_message"] == (
        "corrective capture mixes or opens an unreviewed case"
    )
    assert "save_corrective_capture(" in playback_log
    assert "corrective capture mixes or opens an unreviewed case" in playback_log
    assert manifest["dynamic_gate_result_written"] is False
    assert manifest["dynamic_quality_passed"] is False
    assert manifest["capture_file_count"] == 0
    assert final_status["gate_checks"]["gpu_released"] is True
    assert final_status["normalized_training_dataset_created"] is False
    assert final_status["bc_authorized"] is False
    assert final_status["ppo_authorized"] is False
    assert final_status["training_started"] is False
    assert final_status["valid_for_training"] is False
    assert not list(EVIDENCE.rglob("*.npz"))


def test_v3_rejection_exposes_the_independent_finalizer_namespace_bug() -> None:
    manifest = _json("manifest.json")
    contract = _json("contract.json")
    final_status = _json("final_status.json")
    assert contract["namespace"].endswith("case23_capture_v3_exclusive")
    assert final_status["namespace"].endswith("case23_capture_v2_exclusive")
    assert final_status["contract_checks"]["namespace"] is False
    assert manifest["authorization_token_or_hash_committed"] is False
    assert contract["runtime_authorization_token_sha256"] == ""
