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
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_RELIEF_WZKP105_V5" in source
    assert "0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c" in source
    assert "0acc088a695ff53f9eccfde73107b0748e5de12ffbb6b048efa467455071bf90" in source
    assert "20260718_gate_c_smoothed_case74_relief_wzkp105_v5_exclusive" in source
    assert "TIMEOUT_SECONDS=360" in source
    assert "--cases \"$CASE\"" in source
    assert "smoothed_riser_plan_v1.npz" in source
    assert "cinebotrl_two_wheel_riser_smoothed_plan_v1" in source
    assert '"dynamic_margin_retime_not_applied"' in source
    assert "22.29452723780125" in source
    assert '== 24.0' in source
    assert '== 0.75' in source
    assert '"localized_heading_relief"' in source
    assert '"case74_localized_heading_relief_v1"' in source
    assert 'relief.get("start_anchor") == 394' in source
    assert 'relief.get("end_anchor") == 572' in source
    assert 'relief.get("controller_changed") is False' in source
    assert 'relief.get("phase_governor_changed") is False' in source
    assert 'relief.get("thresholds_changed") is False' in source
    assert 'MAXIMUM_DURATION_SCALE="2.05"' in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert '--controller-wz-kp "$CONTROLLER_WZ_KP"' in source
    assert 'gate.get("controller_overrides") == {"wz_kp": 1.05}' in source
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
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_RELIEF_V2",
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE74_V1",
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


def test_smoothed_representative_gate_is_ordered_hash_bound_and_fail_fast() -> None:
    source = _read("run_riser_smoothed_gate_c_representative.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_REPRESENTATIVE_77_52_V1" in source
    assert "20260718_gate_c_smoothed_representative_77_52_wzkp105_v1_exclusive" in source
    assert 'CASES="77,52"' in source
    assert "for CASE in 77 52" in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert "TIMEOUT_SECONDS=480" in source
    assert "a45892c98311cdd6e6f2096b6821ef760759504138edc2f9c7caa9b1ac90f559" in source
    assert "fa90c7345be5763e1e66a55b4b111780dfe5df97f5a779ab2c6bb390f7a3cbce" in source
    assert '--cases "$CASES"' in source
    assert '--cases "$CASE"' in source
    assert "representative Gate C stopped on case %s" in source
    assert "[r]etarget_exact_source_v1_nonholonomic" in source
    assert "wait_for_gpu_release" in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'gate.get("dynamic_quality_passed") is True' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert "representative CPU/disk ownership is not exclusive" in source
    assert 'summary.get("dynamically_passed_cases") == [77, 52]' in source
    assert 'gate.get("controller_overrides") == {"wz_kp": 1.05}' in source
    assert 'result.get("executed_residual_dataset") is None' in source
    assert "--dataset-dir" not in source


def test_smoothed_representative_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_representative.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "representative Gate C authorization is absent or unknown" in result.stderr


def test_smoothed_tranche1_gate_is_ordered_hash_bound_and_fail_fast() -> None:
    source = _read("run_riser_smoothed_gate_c_tranche1.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_TRANCHE1_53_10_12_11_23_V1" in source
    assert "20260718_gate_c_smoothed_tranche1_53_10_12_11_23_wzkp105_v1_exclusive" in source
    assert 'CASES="53,10,12,11,23"' in source
    assert "for CASE in 53 10 12 11 23" in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert "TIMEOUT_SECONDS=480" in source
    for plan_sha in (
        "f4bcd19e6193fb5da18d1bb4d4e778bda90fceaf75a94b29b96abf0b8c6a1181",
        "d5bda3feefe64230d0f9577523832b88b09662ae9ffa741ce4874b90db09eeb1",
        "4f4f4ed45e618ce2ae350aba430e6e20e78d3d63b631dbed8a742a726023097b",
        "538ddf56b161f93388040284626a9eae01fadbc88cfac8405a5e7848654292b2",
        "ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6",
    ):
        assert plan_sha in source
    assert '--cases "$CASES"' in source
    assert '--cases "$CASE"' in source
    assert "tranche-1 Gate C stopped on case %s" in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'gate.get("dynamic_quality_passed") is True' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert "[r]etarget_exact_source_v1_nonholonomic" in source
    assert 'summary.get("dynamically_passed_cases") == [53, 10, 12, 11, 23]' in source
    assert 'gate.get("controller_overrides") == {"wz_kp": 1.05}' in source
    assert 'result.get("executed_residual_dataset") is None' in source
    assert "--dataset-dir" not in source


def test_smoothed_tranche1_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_tranche1.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "tranche-1 Gate C authorization is absent or unknown" in result.stderr


def test_smoothed_case10_horizon_gate_changes_only_runtime_bound() -> None:
    source = _read("run_riser_smoothed_gate_c_case10_horizon.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE10_HORIZON300_V1" in source
    assert "20260718_gate_c_smoothed_case10_horizon300_wzkp105_v1_exclusive" in source
    assert 'CASES="10"' in source
    assert "for CASE in 10" in source
    assert 'MAXIMUM_DURATION_SCALE="3.00"' in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert "d5bda3feefe64230d0f9577523832b88b09662ae9ffa741ce4874b90db09eeb1" in source
    assert '"execution_duration_s": 7.874601349284786' in source
    assert 'gate.get("maximum_duration_scale") == 3.0' in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert 'summary.get("dynamically_passed_cases") == [10]' in source
    assert "--dataset-dir" not in source


def test_smoothed_case10_horizon_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_case10_horizon.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "case10 horizon Gate C authorization is absent or unknown" in result.stderr


def test_smoothed_tranche1_tail_gate_is_ordered_hash_bound_and_fail_fast() -> None:
    source = _read("run_riser_smoothed_gate_c_tranche1_tail.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_TRANCHE1_TAIL_12_11_23_HORIZON300_V1" in source
    assert "20260718_gate_c_smoothed_tranche1_tail_12_11_23_horizon300_wzkp105_v1_exclusive" in source
    assert 'CASES="12,11,23"' in source
    assert "for CASE in 12 11 23" in source
    assert 'MAXIMUM_DURATION_SCALE="3.00"' in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert (
        r'PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"'
        in source
    )
    assert r'OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"' in source
    for plan_sha in (
        "4f4f4ed45e618ce2ae350aba430e6e20e78d3d63b631dbed8a742a726023097b",
        "538ddf56b161f93388040284626a9eae01fadbc88cfac8405a5e7848654292b2",
        "ad76ada4cdb9f874da615aa0c6e441be62d9a768b813c597c5dc4e20894042b6",
    ):
        assert plan_sha in source
    assert 'summary.get("dynamically_passed_cases") == [12, 11, 23]' in source
    assert 'gate.get("maximum_duration_scale") == 3.0' in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert "tranche-1 tail Gate C stopped on case %s" in source
    assert "[r]etarget_exact_source_v1_nonholonomic" in source
    assert 'result.get("executed_residual_dataset") is None' in source
    assert "--dataset-dir" not in source


def test_smoothed_tranche1_tail_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_tranche1_tail.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "tranche-1 tail Gate C authorization is absent or unknown" in result.stderr


def test_smoothed_tranche2_gate_is_ordered_hash_bound_and_fail_fast() -> None:
    source = _read("run_riser_smoothed_gate_c_tranche2.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_TRANCHE2_24_19_28_70_26_HORIZON300_V1" in source
    assert "20260718_gate_c_smoothed_tranche2_24_19_28_70_26_horizon300_wzkp105_v1_exclusive" in source
    assert 'CASES="24,19,28,70,26"' in source
    assert "for CASE in 24 19 28 70 26" in source
    assert 'MAXIMUM_DURATION_SCALE="3.00"' in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert (
        r'PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"'
        in source
    )
    assert r'OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"' in source
    for plan_sha in (
        "598493ee1ab6dd79223ef9f4a4591e6a4d447a978e87eb70d803a8eed6b4aee6",
        "8cf8bde298c73d1809c3dc7c0dae249446d7554ba77275a490d32fc1a6004b37",
        "e46de958090bea0a1f2a70b27d7ce9f5dbe62190407e121d74d6866fc9ecaee4",
        "bbdb8ec625daf650f36cd1703999989f5cbd219300066417991e3babe34d0390",
        "66d6f491da71521928a7b9012fd929cd7721fc677d2cf432d69cf224f470b415",
    ):
        assert plan_sha in source
    for duration in (
        "9.929693999999998",
        "12.028270780335086",
        "12.408033674341521",
        "12.763227468955776",
        "13.159482653904575",
    ):
        assert duration in source
    assert '28: (382, 7.272057, 12.408033674341521, "smoothed_preview_0.25m_g2.75")' in source
    assert '24: (506, 9.929694, 9.929693999999998, "smoothed_preview_0.05m_g2.75")' in source
    assert '== strategy' in source
    assert 'summary.get("dynamically_passed_cases") == [24, 19, 28, 70, 26]' in source
    assert 'gate.get("maximum_duration_scale") == 3.0' in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert "tranche-2 Gate C stopped on case %s" in source
    assert "[r]etarget_exact_source_v1_nonholonomic" in source
    assert 'result.get("executed_residual_dataset") is None' in source
    assert "--dataset-dir" not in source


def test_smoothed_tranche2_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_tranche2.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "tranche-2 Gate C authorization is absent or unknown" in result.stderr


def test_smoothed_tranche3_gate_is_ordered_hash_bound_and_fail_fast() -> None:
    source = _read("run_riser_smoothed_gate_c_tranche3.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_TRANCHE3_25_66_68_67_7_HORIZON300_V1" in source
    assert "20260718_gate_c_smoothed_tranche3_25_66_68_67_7_horizon300_wzkp105_v1_exclusive" in source
    assert 'CASES="25,66,68,67,7"' in source
    assert "for CASE in 25 66 68 67 7" in source
    assert 'MAXIMUM_DURATION_SCALE="3.00"' in source
    assert 'CONTROLLER_WZ_KP="1.05"' in source
    assert (
        r'PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"'
        in source
    )
    assert r'OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"' in source
    for plan_sha in (
        "ac47ef0a8fef58fc8face3e8800b63283fe2d72da27321daeb44d59b8fe555c7",
        "ebdaf9a2e60e66c6231931bec6087c0b36a0895e22d4ee659e2b056b9b21bc37",
        "4f4fc302402c53533f4bdbed33682bf52971a6f0cb93af3b42bd6da5ffeed142",
        "e7acb5b9ca748645d878d360f357feb82e89b968f92d86c2639f2b74e03950e0",
        "421f9f74a9f56cb79b49611355d9520489bf0bbe7204212ba169b84591fa4cd0",
    ):
        assert plan_sha in source
    for duration in (
        "13.159482674279998",
        "13.56287393451732",
        "13.562891285858488",
        "13.562899771959323",
        "13.582122465552235",
    ):
        assert duration in source
    assert source.count('"smoothed_preview_0.05m_g2.75"') == 5
    assert 'summary.get("dynamically_passed_cases") == [25, 66, 68, 67, 7]' in source
    assert 'gate.get("maximum_duration_scale") == 3.0' in source
    assert "case_gate_passed" in source
    assert 'case_$(printf \'%04d\' "$CASE").exit_code' in source
    assert 'isinstance(result.get("residual_label_envelope_passed"), bool)' in source
    assert "tranche-3 Gate C stopped on case %s" in source
    assert "[r]etarget_exact_source_v1_nonholonomic" in source
    assert 'result.get("executed_residual_dataset") is None' in source
    assert "--dataset-dir" not in source


def test_smoothed_tranche3_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_tranche3.sh"
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    assert result.returncode == 7
    assert "tranche-3 Gate C authorization is absent or unknown" in result.stderr


def test_camera_lever_arm_gate_is_ordered_bounded_and_training_closed() -> None:
    source = _read("run_riser_smoothed_gate_c_camera_lever_arm.sh")
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE68_66_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case68_66_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE67_7_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case67_7_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE7_DYNAMIC_RETIME_V1" in source
    assert "20260718_gate_c_smoothed_case7_dynamic_retime_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE2_3_V8_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case2_3_v8_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE4_5_V8_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case4_5_v8_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE6_8_V8_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case6_8_v8_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE8_DYNAMIC_RETIME_V1" in source
    assert "20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive" in source
    assert "20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu" in source
    assert "ac5da6ce721bd0af51b9b851ada86b08f587f190440c9de23172b115bad3c748" in source
    assert "f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_13_V9_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case9_13_v9_camera_lever_arm_v1_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_13_V9_CAMERA_LEVER_ARM_RETRY_V2" in source
    assert "20260718_gate_c_smoothed_case9_13_v9_camera_lever_arm_retry_v2_exclusive" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_DYNAMIC_RETIME_V1" in source
    assert "20260718_gate_c_smoothed_case9_dynamic_retime_v1_exclusive" in source
    assert "20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu" in source
    assert "229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1" in source
    assert "195249929b363e49fcc73a2600c2d7de9dc9d9fedf0bb9ed0718a44e76bf3fd3" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE13_V10_CAMERA_LEVER_ARM_V1" in source
    assert "20260718_gate_c_smoothed_case13_v10_camera_lever_arm_v1_exclusive" in source
    assert "0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE13_V10_CAMERA_LEVER_ARM_RETRY_V2" in source
    assert "20260718_gate_c_smoothed_case13_v10_camera_lever_arm_retry_v2_exclusive" in source
    assert "CASE_TIMEOUT_SECONDS=1600" in source
    assert 'payload["case_timeout_seconds"] = int(sys.argv[4])' in source
    assert 'timeout --signal=TERM --kill-after=30s "$CASE_TIMEOUT_SECONDS"' in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE14_V10_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case14_v10_camera_lever_arm_v1_exclusive" in source
    assert "e863db5bc93c25bf91f31ac6dbcbd11fa091830290aaf64c58a4a3982d5cae58" in source
    assert "CASE_TIMEOUT_SECONDS=2100" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE15_V10_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case15_v10_camera_lever_arm_v1_exclusive" in source
    assert "8626af7d6d2feeb22d0eb4b2136f0617f91f1fbd3dc87c639d0f459f3c38c25f" in source
    assert "CASE_TIMEOUT_SECONDS=1100" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_V10_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case16_v10_camera_lever_arm_v1_exclusive" in source
    assert "847d1302086dae794e009f23c2a90869a262a43ca77912d88544f0fdb7492c58" in source
    assert "CASE_TIMEOUT_SECONDS=1400" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_DYNAMIC_RETIME_V11" in source
    assert "20260719_gate_c_smoothed_case16_dynamic_retime_v11_exclusive" in source
    assert "20260719_smoothed_plan_all79_v11_case16_dynamic_retime_cpu" in source
    assert "56670dd0ecbdf0157361bef65af50f8d688a9e86bc3e0ff50768472b17474032" in source
    assert "8bcf14454ce4b087973e0c0d2c6efb3858edf75209e195dec7fc09fe7111c821" in source
    assert "CASE_TIMEOUT_SECONDS=1500" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_EXPLICIT_PREVIEW015_V12" in source
    assert "20260719_gate_c_smoothed_case16_explicit_preview015_v12_exclusive" in source
    assert "20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu" in source
    assert "59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581" in source
    assert "742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE17_V12_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case17_v12_camera_lever_arm_v1_exclusive" in source
    assert "e38228121caf797546ac0936fc522e84f61f04cd3740438e0b93469665fa938d" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE18_V12_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case18_v12_camera_lever_arm_v1_exclusive" in source
    assert "121b0f336dd1e236aaee2b9bf0b158466636624507c107e2d90935339edf2517" in source
    assert "CASE_TIMEOUT_SECONDS=2200" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE19_V12_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case19_v12_camera_lever_arm_v1_exclusive" in source
    assert "8cf8bde298c73d1809c3dc7c0dae249446d7554ba77275a490d32fc1a6004b37" in source
    assert "CASE_TIMEOUT_SECONDS=700" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE20_V12_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case20_v12_camera_lever_arm_v1_exclusive" in source
    assert (
        "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE20_V12_CAMERA_ERROR_GOVERNOR_V1"
        in source
    )
    assert (
        "20260719_gate_c_smoothed_case20_v12_camera_error_governor_v1_exclusive"
        in source
    )
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE21_V12_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case21_v12_camera_lever_arm_v1_exclusive" in source
    assert "85029afbbcce435ec8df27770b521b0ab57eae8d98ab4a2dc7f7b7680efaa9ba" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE21_V13_LOCALIZED_REVERSAL_V1" in source
    assert "20260719_gate_c_smoothed_case21_v13_localized_reversal_v1_exclusive" in source
    assert "40611139cb50c4431c238994f311e578c6b43f754ad07b700ec54576a8574e3e" in source
    assert "81c0da4be22d5b800978d1d46ca9705912f72007f7c615b31715c672dd86a1d4" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE22_V13_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case22_v13_camera_lever_arm_v1_exclusive" in source
    assert "b36626c23d41ecd647f91f9c98e1e06abeb1320fbc96a3a59aea052926a39b75" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE22_V14_LOCALIZED_REVERSAL_V1" in source
    assert "20260719_gate_c_smoothed_case22_v14_localized_reversal_v1_exclusive" in source
    assert "369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab" in source
    assert "8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE30_V14_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case30_v14_camera_lever_arm_v1_exclusive" in source
    assert "1722bfdc7c1aeabc5a9d3920cf6a47bc789afbc96e6ef5c8e540695dc3c97dcb" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE31_V14_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case31_v14_camera_lever_arm_v1_exclusive" in source
    assert "8ebc938eeb53b8f7dbf4382a085d3667ea38d5ea52e535dc3be409767737aefb" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V14_CAMERA_LEVER_ARM_V1" in source
    assert "20260719_gate_c_smoothed_case32_v14_camera_lever_arm_v1_exclusive" in source
    assert "45040c19379c0f56f68f44e6391033d2342769f3c034cc281d12f4e5f0cb35a1" in source
    assert "AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V14_CAMERA_ERROR_GOVERNOR_V1" in source
    assert "20260719_gate_c_smoothed_case32_v14_camera_error_governor_v1_exclusive" in source
    assert "CASE_TIMEOUT_SECONDS=1600" in source
    assert "CASE_TIMEOUT_SECONDS=1500" in source
    assert "ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d" in source
    assert "CASE_TIMEOUT_SECONDS=800" in source
    assert "20260718_smoothed_plan_all79_v8_case7_dynamic_retime_cpu" in source
    assert "0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2" in source
    assert "a83934dab6e4293cd830397d3c2ffb41d4f4d78545dddec7fdfa630fa0d22f41" in source
    assert 'CASE_LIST=("$CASE_A")' in source
    assert 'for CASE in "${CASE_LIST[@]}"' in source
    assert 'CAMERA_LEVER_ARM_GAIN="1.00"' in source
    assert 'MAXIMUM_CAMERA_LEVER_ARM_CORRECTION_M="0.05"' in source
    assert "--enable-camera-lever-arm-compensation" in source
    assert "--camera-lever-arm-compensation-gain" in source
    assert "--maximum-camera-lever-arm-correction-m" in source
    assert "--enable-camera-error-recovery-governor" in source
    assert 'CAMERA_RECOVERY_ERROR_START_M="0.13"' in source
    assert 'CAMERA_RECOVERY_ERROR_FULL_M="0.155"' in source
    assert 'MINIMUM_CAMERA_RECOVERY_SCALE="0.20"' in source
    assert "--require-camera-error-recovery-governor" in source
    assert "0.0 < recovery_numeric[0] <= 1.0" in source
    assert "riser_recovery_direction_v4_camera_lever_arm_v1" in source
    assert "measured_camera_to_base_xy_offset_v1" in source
    assert 'gate.get("controller_evidence_passed") is True' in source
    assert 'result.get("camera_lever_arm_telemetry_sample_count")' in source
    assert "0.0 <= correction_max <= 0.05 + 1e-9" in source
    assert "assert_exclusive_resources" in source
    assert "rev-parse '@{u}'" in source
    assert 'tracking_controller "$TRACKING"' in source
    assert 'riser_control "$RISER_CONTROL"' in source
    assert 'recovery_evidence "$RECOVERY_EVIDENCE"' in source
    assert 'playback_loader "$LOADER"' in source
    assert "camera lever-arm Gate C stopped on case %s" in source
    assert 'summary.get("dynamically_passed_cases") == expected_cases' in source
    assert "--dataset-dir" not in source
    assert "--residual-policy" not in source
    assert "--zero-policy-action" not in source


def test_camera_lever_arm_gate_rejects_missing_authorization() -> None:
    wrapper = SCRIPTS / "run_riser_smoothed_gate_c_camera_lever_arm.sh"
    for env in (
        {},
        {"RISER_CAMERA_LEVER_ARM_GATE_C_AUTHORIZATION": "future-token"},
    ):
        result = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 7
        assert (
            "camera lever-arm Gate C authorization is absent or unknown"
            in result.stderr
        )


def test_case74_localized_heading_relief_derivation_is_hash_bound_and_closed() -> None:
    source = _read("derive_riser_smoothed_case74_heading_relief.py")
    assert "cinebotrl_two_wheel_riser_case74_localized_heading_relief_v1" in source
    assert "build_case74_localized_heading_relief" in source
    assert 'failed == ["position_p95_bounded"]' in source
    assert '"parent_plan_sha256"' in source
    assert '"gate_json_sha256"' in source
    assert '"array_derivation_checks"' in source
    assert '"prospective_accepted_duration_median"' in source
    assert '"isaac_started": False' in source
    assert '"residual_capture_started": False' in source
    assert '"bc_started": False' in source
    assert '"ppo_started": False' in source
    assert '"valid_for_training": False' in source


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
