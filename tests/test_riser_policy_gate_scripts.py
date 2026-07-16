from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts/two_wheel_balance"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_bc_gate_requires_complete_quality_qualified_exact_source_dataset() -> None:
    source = _read("run_riser_residual_bc_gate.sh")
    assert 'summary.get("passed_case_count") == 79' in source
    assert 'dataset.get("action_clip_ratio") == [0.0, 0.0, 0.0]' in source
    assert 'dataset.get("trajectory_leakage") is False' in source
    assert 'dataset.get("source_action_labels_used") is False' in source
    assert '"physical_gimbal_labels_used_as_actions"' in source
    assert '"capture_admission_hash"' in source
    assert '"capture_admission_commit"' in source
    assert '"capture_admission_plan"' in source
    assert '"exact_source_contract"' in source
    assert '"exact_source_training_qualified"' in source
    assert '"exact_source_audit_passed"' in source
    assert 'merge-base --is-ancestor "$DATASET_COMMIT" HEAD' in source
    assert 'POLICY_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"' in source
    assert '--source-commit "$POLICY_COMMIT"' in source
    assert 'report.get("offline_gate_splits")' in source
    assert 'report.get("holdout_used_for_model_selection") is False' in source
    assert 'report.get("holdout_metrics_computed") is False' in source
    assert 'report.get("case_balanced_training_loss") is True' in source
    assert 'report.get("case_balanced_validation_gate") is True' in source


def test_all79_capture_is_bound_to_one_clean_source_revision() -> None:
    source = _read("run_riser_all79_dataset_gate.sh")
    assert "20260717_residual_all79_exact_source_v1" in source
    assert "20260717_all79_playback_exact_source_v1" in source
    assert "20260716_residual_all79_phase_v2" not in source
    assert "RISER_EXACT_SOURCE_MANIFEST_WSL" in source
    assert "RISER_EXACT_SOURCE_MANIFEST_SHA256" in source
    assert "validate_riser_exact_source_manifest.py" in source
    assert "--mode training" in source
    assert '"trajectory_integrity_contract": "exact_source_v1"' in source
    assert '"upstream_valid_for_training": True' in source
    assert 'git -C "$ROOT" diff --quiet' in source
    assert 'git -C "$ROOT" diff --cached --quiet' in source
    assert 'CAPTURE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"' in source
    assert "refusing to backfill admission onto existing capture artifacts" in source
    assert '"cinebotrl_two_wheel_riser_capture_admission_v1"' in source
    assert '"plan_manifest_sha256"' in source
    assert '"$CAPTURE_COMMIT"' in source


def test_holdout_gate_compares_teacher_zero_and_learned_sources() -> None:
    source = _read("run_riser_residual_holdout_gate.sh")
    assert "exact_source_v1" in source
    assert "20260717_all79_playback_exact_source_v1" in source
    assert "zero_policy_action_baseline" in source
    assert "torchscript_residual_policy" in source
    assert "gate_riser_residual_rollouts.py" in source
    assert "--mode holdout" in source
    assert 'split_cases.get("holdout", [])' in source
    assert 'split_cases.get("validation", [])' not in source
    assert 'len(cases) == 8' in source
    assert 'rev-parse HEAD)" == "$POLICY_COMMIT"' in source
    assert '"git_commit": sys.argv[4]' in source


def test_all79_policy_gate_requires_holdout_and_all_cases() -> None:
    source = _read("run_riser_residual_all79_policy_gate.sh")
    assert "exact_source_v1" in source
    assert "20260717_all79_playback_exact_source_v1" in source
    assert 'holdout.get("passed") is True' in source
    assert "for case_number in $(seq 1 79)" in source
    assert "gate_riser_residual_rollouts.py" in source
    assert "--mode all79" in source
    assert 'rev-parse HEAD)" == "$POLICY_COMMIT"' in source
    assert '"git_commit": sys.argv[3]' in source
    assert 'holdout.get("case_count") == 8' in source
