from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts/two_wheel_balance"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_bc_gate_requires_complete_quality_qualified_exact_source_dataset() -> None:
    source = _read("run_riser_residual_bc_gate.sh")
    assert "all79_residual_dataset_v2.npz" in source
    assert "all79_residual_dataset_v1.npz" not in source
    assert "20260717_residual_all79_exact_source_lookahead_v2" in source
    assert "20260717_residual_bc_exact_source_lookahead_v2" in source
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
    assert '"cinebotrl_two_wheel_riser_residual_merged_v2"' in source
    assert '"executed_state_with_execution_time_lookahead_v2"' in source
    assert '== [0.25, 0.5, 1.0]' in source


def test_all79_capture_is_bound_to_one_clean_source_revision() -> None:
    source = _read("run_riser_all79_dataset_gate.sh")
    assert "20260717_residual_all79_exact_source_lookahead_v2" in source
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
    assert "executed_residual_v2.npz" in source
    assert "executed_state_with_execution_time_lookahead_v2" in source
    assert '"lookahead_horizons_s": [0.25, 0.5, 1.0]' in source


def test_gate_c_canary_is_hash_bound_clean_pushed_and_label_free() -> None:
    source = _read("run_riser_gate_c_canary.sh")
    summarizer = _read("summarize_riser_gate_c_canary.py")
    assert "validate_riser_gate_c_portfolio.py" in source
    assert "851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a" in source
    assert 'git -C "$ROOT" diff --quiet' in source
    assert 'git -C "$ROOT" diff --cached --quiet' in source
    assert "rev-parse '@{u}'" in source
    assert "--plan-filename-template" in source
    assert "exact_source_riser_playback_v1.npz" in source
    assert "--dataset-dir" not in source
    assert "--residual-policy" not in source
    assert "--zero-policy-action" not in source
    assert '"residual_capture_started": False' in summarizer
    assert '"bc_started": False' in summarizer
    assert '"ppo_started": False' in summarizer
    assert "summarize_riser_gate_c_canary.py" in source
    assert "20260717_gate_c_canary_v3_exclusive_timing_resealed" in source
    assert "assert_exclusive_gpu" in source
    assert "refusing shared-GPU Gate C launch" in source
    assert "Gate C runtime JSON is not a sealed dynamic pass" in source
    assert '"riser_recovery_direction_v4"' in source
    assert '"tracking_direction_recovery_error_range_m"' in source


def test_case74_recovery_wrapper_requires_explicit_one_case_authorization() -> None:
    source = _read("run_riser_case74_recovery_canary.sh")
    assert 'RISER_CASE74_GPU_AUTHORIZATION' in source
    assert 'AUTHORIZED_CASE74_RECOVERY_V4' in source
    assert 'RISER_GATE_C_CASES="74"' in source
    assert 'case74_recovery_direction_v4_exclusive' in source
    assert "run_riser_gate_c_canary.sh" in source
    assert "1,52,74,77" not in source


def test_runtime_evidence_separates_source_and_execution_clocks() -> None:
    riser = _read("smoke_riser_reference_playback.py")
    whole_body = _read("smoke_all79_whole_body_playback.py")
    for source in (riser, whole_body):
        assert '"source_duration_s": source_duration_s' in source
        assert '"execution_duration_s": execution_duration_s' in source
        assert "phase_time_s >= execution_duration_s" in source
        assert "phase_time_s >= source_duration_s" not in source
    assert 'candidate["source_time_s"][-1]' in whole_body
    assert 'candidate["execution_time_s"][-1]' in whole_body
    assert "np.array_equal(time_s, execution_time_s)" in whole_body
    assert '"residual_label_envelope_rejection"' in riser
    assert "threading.Timer(60.0" in riser
    assert "failure_plan.source_time_s[-1]" in riser
    assert "failure_plan.time_s[-1]" in riser
    assert "if dataset_dir is not None:" in riser
    assert '"raw_residual_label_applied_to_commands": False' in riser
    assert '"dynamic_quality_passed": dynamic_quality_passed' in riser
    assert '"residual_label_envelope_passed": residual_label_envelope_ok' in riser


def test_holdout_gate_compares_teacher_zero_and_learned_sources() -> None:
    source = _read("run_riser_residual_holdout_gate.sh")
    assert "exact_source_v1" in source
    assert "20260717_residual_holdout_exact_source_lookahead_v2" in source
    assert '"cinebotrl_two_wheel_riser_residual_policy_v2"' in source
    assert '"executed_state_with_execution_time_lookahead_v2"' in source
    assert '== [0.25, 0.5, 1.0]' in source
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
    assert "20260717_residual_policy_all79_exact_source_lookahead_v2" in source
    assert '"cinebotrl_two_wheel_riser_residual_policy_v2"' in source
    assert '"executed_state_with_execution_time_lookahead_v2"' in source
    assert '== [0.25, 0.5, 1.0]' in source
    assert "20260717_all79_playback_exact_source_v1" in source
    assert 'holdout.get("passed") is True' in source
    assert "for case_number in $(seq 1 79)" in source
    assert "gate_riser_residual_rollouts.py" in source
    assert "--mode all79" in source
    assert 'rev-parse HEAD)" == "$POLICY_COMMIT"' in source
    assert '"git_commit": sys.argv[3]' in source
    assert 'holdout.get("case_count") == 8' in source
