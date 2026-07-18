from pathlib import Path
import os
import subprocess


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
    assert '"cinebotrl_two_wheel_riser_residual_bc_gate_v2"' in source
    assert '"state_shared_lookahead_fusion_v1"' in source


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
    assert '"leadshine_400w_first_order_monitor_v1"' in summarizer
    assert '"riser_thermal_force_observed"' in summarizer
    assert '"riser_thermal_load_bounded"' in summarizer
    assert '"riser_peak_force_bounded"' in summarizer


def test_smoothed_case74_gate_c_is_isolated_hash_bound_and_label_free() -> None:
    source = _read("run_riser_smoothed_gate_c_case74.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_V1" in source
    assert "73121d240ccf54fa65783fc1cf47eed4d805af3e6bedbdfff847719c92f2130b" in source
    assert "fee7fd2c2d9cccca8fa19b0141996a4c530840ef427513d941b57e3ff773c1a3" in source
    assert "20260718_gate_c_smoothed_case74_v1_representative_exclusive" in source
    assert "TIMEOUT_SECONDS=360" in source
    assert "--cases \"$CASE\"" in source
    assert "smoothed_riser_plan_v1.npz" in source
    assert "cinebotrl_two_wheel_riser_smoothed_plan_v1" in source
    assert '"dynamic_margin_retime_absent"' in source
    assert "22.446453095094938" in source
    assert '== 12.0' in source
    assert 'MAXIMUM_DURATION_SCALE="2.05"' in source
    assert '--maximum-duration-scale "$MAXIMUM_DURATION_SCALE"' in source
    assert 'gate.get("maximum_duration_scale") == 2.05' in source
    assert '"bounded_execution_duration_scale_v1"' in source
    assert "time_alias_unambiguous" in source
    assert "assert_exclusive_gpu" in source
    assert "/usr/lib/wsl/lib/nvidia-smi" in source
    assert "--query-compute-apps=pid,process_name" in source
    assert "wait_for_gpu_release" in source
    assert "source_manifest \"$SOURCE_MANIFEST_WSL\"" in source
    assert "robot_usd \"$ROBOT_USD\"" in source
    assert "playback_loader \"$LOADER\"" in source
    assert "rev-parse '@{u}'" in source
    assert "--dataset-dir" not in source
    assert "--residual-policy" not in source
    assert "--zero-policy-action" not in source


def test_smoothed_case74_gate_c_rejects_missing_authorization_before_runtime() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_case74.sh"
    for authorization in (
        None,
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE77_V5",
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE52_V2",
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_V0",
    ):
        env = os.environ.copy()
        if authorization is not None:
            env["RISER_SMOOTHED_GATE_C_AUTHORIZATION"] = authorization
        result = subprocess.run(
            ["bash", str(wrapper)], capture_output=True, text=True, env=env
        )
        assert result.returncode == 7
        assert "authorization is absent or unknown" in result.stderr


def test_case74_recovery_wrapper_has_sealed_one_case_runtime_authorization() -> None:
    source = _read("run_riser_case74_recovery_canary.sh")
    assert 'RISER_CASE74_GPU_AUTHORIZATION' in source
    assert 'AUTHORIZED_CASE74_RECOVERY_V4_RUNTIME_V2' in source
    assert "obsolete case-74 authorization is permanently rejected" in source
    assert "runtime authorization is absent or unknown" in source
    assert 'RISER_GATE_C_CASES="74"' in source
    assert 'case74_recovery_v4_runtime_v2_exclusive' in source
    assert 'RISER_GATE_C_CASE_TIMEOUT_SECONDS="$PINNED_CASE_TIMEOUT_SECONDS"' in source
    assert "1,52,74,77" not in source
    runner = _read("run_riser_gate_c_canary.sh")
    assert "authorized only as an isolated one-case canary" in runner
    assert "requires sealed contract admission evidence" in runner
    assert "timeout --signal=TERM --kill-after=30s" in runner


def test_case74_wrapper_rejects_obsolete_unknown_and_override_before_python(
    tmp_path: Path,
) -> None:
    wrapper = SCRIPTS / "run_riser_case74_recovery_canary.sh"
    base_env = os.environ.copy()
    for authorization in ("AUTHORIZED_CASE74_RECOVERY_V4", "future-token"):
        env = base_env | {"RISER_CASE74_GPU_AUTHORIZATION": authorization}
        result = subprocess.run(
            ["bash", str(wrapper)], capture_output=True, text=True, env=env
        )
        assert result.returncode == 7
    env = base_env | {"RISER_GATE_C_CASES": "74"}
    result = subprocess.run(
        ["bash", str(wrapper)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 7
    assert "rejects environment override" in result.stderr
    assert not list(tmp_path.iterdir())


def test_shared_runner_rejects_unsealed_case74_before_runtime_checks(tmp_path: Path) -> None:
    runner = SCRIPTS / "run_riser_gate_c_canary.sh"
    env = os.environ.copy() | {
        "RISER_GATE_C_CASES": "74",
        "RISER_ROOT": str(tmp_path / "missing-root"),
        "ISAAC_PYTHON": str(tmp_path / "missing-python"),
    }
    result = subprocess.run(
        ["bash", str(runner)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 7
    assert "requires the sealed runtime-v2 authorization" in result.stderr
    assert not (tmp_path / "missing-root").exists()


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
    assert '"cinebotrl_two_wheel_riser_residual_bc_gate_v2"' in source
    assert '"state_shared_lookahead_fusion_v1"' in source
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
    assert '"cinebotrl_two_wheel_riser_residual_bc_gate_v2"' in source
    assert '"state_shared_lookahead_fusion_v1"' in source
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
