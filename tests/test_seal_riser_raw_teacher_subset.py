import hashlib
import json
from pathlib import Path
import sys

from scripts.two_wheel_balance import seal_riser_raw_teacher_subset as sealer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/two_wheel_balance/seal_riser_raw_teacher_subset.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, forge_raw: bool = False) -> list[str]:
    selection = tmp_path / "selection.json"
    cases = list(range(1, 42))
    excluded = 42
    selection.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
                "rows": [
                    {"case": case, "plan_sha256": f"{case:064x}"}
                    for case in cases + [excluded]
                ],
            }
        ),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_smoothed_representative_admission_v1",
                "passed": True,
                "runtime_commit": "parent",
                "selection_sha256": _sha(selection),
                "requested_cases": cases + [excluded],
                "selected_plans": [
                    {"case": case, "plan_sha256": f"{case:064x}"}
                    for case in cases + [excluded]
                ],
                "raw_teacher_capture_authorized": True,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
            }
        ),
        encoding="utf-8",
    )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_raw_teacher42_progress_v1",
                "runtime_commit": "parent",
                "completed_cases": cases,
                "stopped_case": excluded,
                "reason": "runtime_or_physical_reject",
                "capture_admission_passed": False,
            }
        ),
        encoding="utf-8",
    )
    audit_dir = tmp_path / "audits"
    gate_dir = tmp_path / "gates"
    raw_dir = tmp_path / "raw"
    audit_dir.mkdir()
    gate_dir.mkdir()
    raw_dir.mkdir()
    for case in cases:
        gate = gate_dir / f"case_{case:04d}.json"
        gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
        raw = raw_dir / f"case_{case:04d}_executed_raw_teacher_v1.npz"
        raw.write_bytes(f"raw-{case}".encode())
        audit = {
            "case": case,
            "passed": True,
            "capture_admission_passed": True,
            "admission_sha256": _sha(parent),
            "selection_sha256": _sha(selection),
            "gate": str(gate),
            "gate_sha256": _sha(gate),
            "raw_case": str(raw),
            "raw_case_sha256": "0" * 64 if forge_raw and case == 7 else _sha(raw),
            "valid_for_training": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
        }
        (audit_dir / f"case_{case:04d}.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
    rejected_gate = gate_dir / f"case_{excluded:04d}.json"
    rejected_gate.write_text(
        json.dumps(
            {
                "passed": False,
                "dynamic_quality_passed": False,
                "training_started": False,
                "ppo_authorized": False,
                "results": [
                    {
                        "case": excluded,
                        "passed": False,
                        "dynamic_quality_passed": False,
                        "executed_residual_dataset": None,
                        "checks": {"position_p95_bounded": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return [
        sys.executable,
        str(SCRIPT),
        "--parent-admission",
        str(parent),
        "--selection",
        str(selection),
        "--progress-status",
        str(progress),
        "--case-audit-dir",
        str(audit_dir),
        "--gate-dir",
        str(gate_dir),
        "--raw-dir",
        str(raw_dir),
        "--excluded-cases",
        str(excluded),
        "--output",
        str(tmp_path / "subset.json"),
    ]


def _run(monkeypatch, command: list[str]) -> int:
    monkeypatch.setattr(
        sealer,
        "git_value",
        lambda *args: "" if args[0] == "status" else "a" * 40,
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *command[2:]])
    return sealer.main()


def test_seals_provenance_bound_subset(tmp_path: Path, monkeypatch) -> None:
    assert _run(monkeypatch, _fixture(tmp_path)) == 0
    payload = json.loads((tmp_path / "subset.json").read_text(encoding="utf-8"))
    assert payload["passed"]
    assert payload["retained_case_count"] == 41
    assert payload["excluded_cases"] == [42]
    assert payload["corpus_audit_authorized"]
    assert not payload["runtime_authorized"]
    assert not payload["new_raw_teacher_capture_authorized"]
    assert not payload["bc_authorized"]
    assert not payload["ppo_authorized"]


def test_rejects_forged_retained_raw_hash(tmp_path: Path, monkeypatch) -> None:
    assert _run(monkeypatch, _fixture(tmp_path, forge_raw=True)) != 0
    payload = json.loads((tmp_path / "subset.json").read_text(encoding="utf-8"))
    assert not payload["passed"]
    row = next(row for row in payload["retained_case_evidence"] if row["case"] == 7)
    assert not row["checks"]["raw_hash"]


def test_translates_windows_repository_path_for_internal_wsl_git() -> None:
    assert (
        sealer.windows_to_wsl_path(r"G:\wSpace\cinebotRL-two-wheel-riser")
        == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    )
