import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "prepare_model_based_corrective_conversion_route.py"
)
CAPTURES = {
    30: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260722_case30_corrective_capture_v2"
    ),
    23: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4"
    ),
    6: (
        ROOT
        / "docs/03_training/two_wheel_balance/"
        "evidence_20260724_case6_corrective_capture_v1"
    ),
}


def _module():
    spec = importlib.util.spec_from_file_location(
        "corrective_conversion_route_preparer",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _healthy_repository():
    return {
        "head": "a" * 40,
        "upstream": "a" * 40,
        "checks": {
            "head_matches_upstream": True,
            "tracked_worktree_clean": True,
        },
        "passed": True,
    }


def _healthy_identity(repo: Path, path: Path):
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    return {
        "path": relative,
        "sha256": MODULE._sha256(path),
        "git_blob_sha1": MODULE._git_blob(path),
        "checks": {
            "inside_repository": True,
            "file_exists": True,
            "tracked": True,
            "committed_blob_matches": True,
        },
        "passed": True,
    }


def _capture_paths(case: int):
    root = CAPTURES[case]
    return (
        root / f"capture/case_{case:04d}_corrective_teacher_capture_v2.npz",
        root / "final_status.json",
    )


@pytest.mark.parametrize(
    ("case", "sample_count", "clipped_rows"),
    [
        (30, 11411, [200, 308, 333]),
        (23, 3273, [0, 0, 0]),
        (6, 7933, [0, 146, 0]),
    ],
)
def test_existing_admitted_captures_share_one_proposal_path(
    monkeypatch,
    case,
    sample_count,
    clipped_rows,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_repository_state",
        lambda repo: _healthy_repository(),
    )
    monkeypatch.setattr(MODULE, "_identity", _healthy_identity)
    capture, final = _capture_paths(case)
    result = MODULE.build_proposal(
        ROOT,
        capture,
        final,
        case=case,
        split="train",
    )
    assert result["passed"] is True
    assert result["proposal_ready"] is True
    assert all(result["proposal_checks"].values())
    assert result["metrics"]["sample_count"] == sample_count
    assert result["metrics"]["clipped_rows"] == clipped_rows
    assert result["metrics"]["observation_shape"] == [sample_count, 65]
    assert result["metrics"]["action_shape"] == [sample_count, 3]
    assert len(result["identities"]) == 6
    assert result["conversion_execution_implemented"] is False
    assert result["authorization_token_issued"] is False
    assert result["conversion_authorized"] is False
    assert result["output_created"] is False
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_case_mismatch_rejects_before_proposal(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_repository_state",
        lambda repo: _healthy_repository(),
    )
    monkeypatch.setattr(MODULE, "_identity", _healthy_identity)
    capture, final = _capture_paths(6)
    with pytest.raises(ValueError, match="unreviewed case"):
        MODULE.build_proposal(
            ROOT,
            capture,
            final,
            case=7,
            split="train",
        )


def test_dirty_or_unpushed_repository_keeps_proposal_closed(
    monkeypatch,
) -> None:
    repository = _healthy_repository()
    repository["checks"]["head_matches_upstream"] = False
    repository["passed"] = False
    monkeypatch.setattr(
        MODULE,
        "_repository_state",
        lambda repo: repository,
    )
    monkeypatch.setattr(MODULE, "_identity", _healthy_identity)
    capture, final = _capture_paths(6)
    result = MODULE.build_proposal(
        ROOT,
        capture,
        final,
        case=6,
        split="train",
    )
    assert result["proposal_ready"] is False
    assert result["passed"] is False
    assert result["conversion_authorized"] is False
    assert result["output_created"] is False


def test_source_identity_drift_keeps_proposal_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_repository_state",
        lambda repo: _healthy_repository(),
    )

    def drifted_identity(repo: Path, path: Path):
        row = _healthy_identity(repo, path)
        if path.name.endswith("capture_v2.npz"):
            row["checks"]["committed_blob_matches"] = False
            row["passed"] = False
        return row

    monkeypatch.setattr(MODULE, "_identity", drifted_identity)
    capture, final = _capture_paths(6)
    result = MODULE.build_proposal(
        ROOT,
        capture,
        final,
        case=6,
        split="train",
    )
    assert result["proposal_ready"] is False
    assert result["passed"] is False
    assert result["conversion_authorized"] is False
    assert result["output_created"] is False


def test_proposal_contains_no_runtime_authorization_secret(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_repository_state",
        lambda repo: _healthy_repository(),
    )
    monkeypatch.setattr(MODULE, "_identity", _healthy_identity)
    capture, final = _capture_paths(6)
    result = MODULE.build_proposal(
        ROOT,
        capture,
        final,
        case=6,
        split="train",
    )
    text = json.dumps(result)
    assert "authorization_token_sha256" not in text
    assert "authorization_file" not in text


def test_windows_and_wsl_unc_paths_map_to_same_wsl_file() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    assert MODULE._windows_path_to_wsl(
        r"\\wsl.localhost\Ubuntu\home\yanbo\proposal.json"
    ) == "/home/yanbo/proposal.json"
    with pytest.raises(ValueError):
        MODULE._windows_path_to_wsl(
            r"\\wsl.localhost\Other\home\yanbo\proposal.json"
        )
