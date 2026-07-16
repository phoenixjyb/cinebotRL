import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_continuous_yaw_scope.py"


def _write_plan(path: Path, yaw_deg: list[float]) -> str:
    proxy = np.zeros((len(yaw_deg), 3), dtype=np.float64)
    proxy[:, 2] = np.deg2rad(yaw_deg)
    np.savez_compressed(path, proxy_gimbal_q=proxy)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scope_audit_preserves_multi_turn_semantics_and_reports_branch_risk(
    tmp_path: Path,
) -> None:
    yaw_by_case = {
        1: [170.0, 181.0, 190.0],
        2: [-170.0, -181.0, -190.0],
        3: [-541.0, -530.0, -520.0],
    }
    items = []
    for case, yaw in yaw_by_case.items():
        name = f"case_{case:04d}.npz"
        items.append(
            {
                "case": case,
                "file": name,
                "plan_sha256": _write_plan(tmp_path / name, yaw),
            }
        )
    manifest = {
        "kinematic_accepted_count": 3,
        "kinematic_accepted_cases": [1, 2, 3],
        "items": items,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "rows.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            manifest_hash,
            "--expected-accepted-count",
            "3",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=False,
    )
    summary = json.loads(output_json.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert summary["passed"]
    assert summary["affected_cases"] == [1, 2, 3]
    assert summary["canonical_crossing_cases"] == [1, 2, 3]
    assert summary["maximum_naive_branch_delta_case"] == 3
    assert summary["maximum_naive_branch_delta_deg"] == 720.0
    assert summary["maximum_unwrapped_step_deg"] == 11.0
    assert summary["maximum_unwrapped_step_cases"] == [1, 2, 3]
    assert summary["semantic_unwrapped_yaw_is_authoritative"]
    assert summary["nearest_equivalent_physics_branch_required"]
    assert not summary["multi_turn_semantic_plans_rejected"]
    assert not summary["valid_for_training"]
    assert output_csv.read_text(encoding="utf-8").count("\n") == 4
