import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_pair_v1"
)
EXPECTED_HASHES = {
    "admission.json": "2c159e802d829a134683060ad19edec1f68ce88018291b5fdf49f7230ea971d6",
    "baseline/case_0023.json": (
        "b2c4ef1f3bb39086e6bcaf015c10b8f9740c5497030f38321dfb4836b90ace72"
    ),
    "baseline/runtime_heartbeat.json": (
        "fd2086fe3ccb055b00101b3f03630aefa269be60762727d73a9c8dfea114020c"
    ),
    "candidate/case_0023.json": (
        "130da066f623eb588790fe2467ba44ee1f6918a69841bdeb328b28635c708474"
    ),
    "candidate/runtime_heartbeat.json": (
        "a026578249f53424ddb6ec40c16fbea31e0111f6305d219ff98fba4b3a32ddec"
    ),
    "contract.json": "e5b5b360efdb0334412fb156d77dba7e0a6eb605651c16bffc280a8076caa043",
    "final_status.json": (
        "67c8e99a0629a4b1cb4a2981abfe8360c5d9979c4757582dab6d4fb22cd00deb"
    ),
}


def _load(relative: str) -> dict[str, object]:
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def test_case23_pair_archive_hashes_are_exact() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((EVIDENCE / relative).read_bytes()).hexdigest()
        assert actual == expected
    manifest = {}
    for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    assert manifest == EXPECTED_HASHES


def test_case23_pair_passes_measurable_improvement_without_regression() -> None:
    baseline = _load("baseline/case_0023.json")
    candidate = _load("candidate/case_0023.json")
    final = _load("final_status.json")
    baseline_result = baseline["results"][0]
    candidate_result = candidate["results"][0]
    assert baseline["passed"] is True
    assert candidate["passed"] is True
    assert baseline_result["dynamic_quality_passed"] is True
    assert candidate_result["dynamic_quality_passed"] is True
    assert baseline_result["completed_steps"] == 3273
    assert candidate_result["completed_steps"] == 3273
    assert baseline_result["position_error_p95_m"] == pytest.approx(
        0.05938480224396858
    )
    assert candidate_result["position_error_p95_m"] == pytest.approx(
        0.053413363967263365
    )
    admission = final["paired_admission"]
    assert admission["position_p95_absolute_improvement_m"] == pytest.approx(
        0.005971438276705217
    )
    assert admission["position_p95_relative_improvement"] == pytest.approx(
        0.10055499136248629
    )
    assert all(admission["checks"].values())
    assert final["rollout_checks"]["gpu_released"] is True
    assert final["passed"] is True
    assert final["corrective_target_admission_passed"] is True


def test_case23_pair_keeps_capture_and_learning_closed() -> None:
    admission = _load("admission.json")
    final = _load("final_status.json")
    assert admission["runtime_commit"] == (
        "d77a1d494be79e442798e34368d865de1cf7ce25"
    )
    assert admission["runtime_authorized"] is True
    for phase in ("baseline", "candidate"):
        rollout = _load(f"{phase}/case_0023.json")
        result = rollout["results"][0]
        assert rollout["raw_teacher_capture_started"] is False
        assert rollout["normalized_dataset_capture_started"] is False
        assert rollout["policy_trace_started"] is False
        assert rollout["shadow_teacher_trace_started"] is False
        assert result["corrective_teacher_labels_captured"] is False
        assert result["executed_residual_dataset"] is None
        assert result["executed_raw_teacher_capture"] is None
    assert final["label_capture_authorized"] is False
    assert final["dataset_created"] is False
    assert final["bc_authorized"] is False
    assert final["ppo_authorized"] is False
    assert final["training_started"] is False
    assert final["valid_for_training"] is False
