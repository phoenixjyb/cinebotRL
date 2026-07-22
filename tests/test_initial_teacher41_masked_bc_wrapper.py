from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/two_wheel_balance/run_initial_teacher41_masked_bc.sh"


def test_wrapper_is_one_use_masked_validation_only_and_runtime_closed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "initial_teacher41_case78_31_5_5_v2_resealed.npz" in source
    assert "20260722_initial_teacher41_masked_bc_contract_v1_cpu" in source
    assert "--mask-previous-action-observations" in source
    assert "--seed 20260722" in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source
    assert source.index('rm -f "$AUTHORIZATION_FILE"') < source.index(
        '"$PY" -u -X utf8'
    )
    assert '"learned_rollout_authorized": False' in source
    assert '"ppo_authorized": False' in source
    assert '"holdout_opened": False' in source
    assert "report.get(\"dataset_case_count\") == 41" in source
    assert "report.get(\"dataset_row_count\") == 486619" in source


def test_execute_rejects_missing_authorization_before_output_creation() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing training Python" in result.stderr or "missing one-use" in result.stderr
