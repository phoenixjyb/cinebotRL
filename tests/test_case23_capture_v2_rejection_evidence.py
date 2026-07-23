import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v2_rejected_case_split"
)


def _json(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_rejection_manifest_hashes_every_preserved_runtime_file() -> None:
    manifest = _json("manifest.json")
    assert manifest["runtime_commit"] == (
        "526952133a784ad653f4cfebd3e618a23fd4b291"
    )
    assert manifest["authorization_consumed"] is True
    assert manifest["authorization_token_deleted"] is True
    assert manifest["retry_authorized"] is False
    for relative, expected_hash in manifest["files"].items():
        assert _sha256(EVIDENCE / relative) == expected_hash


def test_v2_rejection_is_pre_isaac_case_route_failure_without_labels() -> None:
    manifest = _json("manifest.json")
    admission = _json("admission.json")
    final_status = _json("final_status.json")
    playback_log = (EVIDENCE / "logs/playback.log").read_text(
        encoding="utf-8"
    )
    assert admission["case"] == 23
    assert admission["split"] == "train"
    assert admission["checks"]["case_split"] is True
    assert admission["runtime_authorized"] is True
    assert admission["label_capture_authorized"] is True
    assert "case_split': False" in playback_log
    assert manifest["isaac_initialization_started"] is False
    assert manifest["capture_file_count"] == 0
    assert final_status["playback_exit_code"] == 2
    assert final_status["gate_checks"]["gpu_released"] is True
    assert final_status["normalized_training_dataset_created"] is False
    assert final_status["bc_authorized"] is False
    assert final_status["ppo_authorized"] is False
    assert final_status["training_started"] is False
    assert final_status["valid_for_training"] is False
    assert not list(EVIDENCE.rglob("*.npz"))


def test_v2_rejection_commits_no_runtime_token_or_token_hash() -> None:
    manifest = _json("manifest.json")
    contract = _json("contract.json")
    assert manifest["authorization_token_or_hash_committed"] is False
    assert contract["runtime_authorization_token_sha256"] == ""
    assert not list(EVIDENCE.rglob("*.authorization-token"))
