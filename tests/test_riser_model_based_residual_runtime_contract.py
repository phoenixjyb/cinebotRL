from pathlib import Path


ROOT = Path(__file__).parents[1]
PLAYBACK = ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"


def test_model_based_mode_is_explicit_and_pre_isaac_fail_closed() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    parse_index = source.index("args = parser.parse_args()")
    app_index = source.index("app = AppLauncher(args).app")
    model_validation_index = source.index(
        'if args.policy_command_base == "model_based_planner":', parse_index
    )
    assert model_validation_index < app_index
    assert 'choices=("phase_feedforward", "model_based_planner")' in source
    assert "requires scales 0.05,0.05,0.02" in source
    assert "model-based residual canary mode forbids every capture path" in source


def test_model_based_mode_adds_policy_above_complete_planner_command() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    deterministic_index = source.index("model_based_vx_ref = vx_ref")
    policy_index = source.index("policy_output = residual_policy(")
    apply_index = source.index("apply_model_based_policy_residual(")
    assert deterministic_index < policy_index < apply_index
    assert "model_based_vx_ref," in source[apply_index : apply_index + 500]
    assert "model_based_wz_ref," in source[apply_index : apply_index + 500]
    assert "model_based_riser_target," in source[apply_index : apply_index + 500]


def test_model_based_mode_emits_independent_command_telemetry() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    assert '"model_based_vx_reference_mps"' in source
    assert '"model_based_wz_reference_rad_s"' in source
    assert '"model_based_riser_target_m"' in source
    assert '"applied_policy_residual_action"' in source
    assert '"policy_command_base": args.policy_command_base' in source
    assert "MODEL_BASED_POLICY_RESIDUAL_CONTRACT" in source


def test_model_based_mode_has_distinct_zero_and_learned_sources() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    assert '"model_based_planner_plus_zero_policy_residual"' in source
    assert '"model_based_planner_plus_torchscript_residual"' in source


def test_corrective_teacher_capture_requires_separate_pre_app_admission() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    pre_app = source.split("app = AppLauncher(args).app", 1)[0]
    assert '"--corrective-teacher-profile"' in pre_app
    assert "load_corrective_teacher_profile" in pre_app
    assert "requires its one pinned case only" in pre_app
    assert "corrective teacher requires model-based zero-policy mode" in pre_app
    assert "model_based_zero_measurement" in pre_app
    assert '"--corrective-teacher-capture-admission"' in pre_app
    assert '"--corrective-teacher-capture-split"' in pre_app
    assert "load_capture_admission" in pre_app
    assert (
        "load_capture_admission(\n"
        "            args.corrective_teacher_capture_admission,\n"
        "            expected_case=corrective_teacher_case,\n"
        "            expected_split=args.corrective_teacher_capture_split,\n"
        "        )"
    ) in pre_app
    assert (
        "corrective capture directory, admission, and split are required together"
        in pre_app
    )
    assert "corrective capture is exclusive with every legacy capture path" in pre_app
    assert '"corrective_teacher_label_capture_authorized": (' in source
    assert "corrective_capture_admission is not None" in source
    assert '"corrective_teacher_labels_captured": corrective_capture_path is not None' in source


def test_corrective_teacher_is_computed_after_model_planner_and_before_apply() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    model_index = source.index("model_based_vx_ref = vx_ref")
    teacher_index = source.index("corrective_output = build_corrective_teacher_action(")
    apply_index = source.index("apply_model_based_policy_residual(", teacher_index)
    assert model_index < teacher_index < apply_index


def test_corrective_capture_records_requested_and_effective_supervised_commands() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    assert "requested_residual = corrective_output.applied_residual.copy()" in source
    assert "effective_residual = final_command - model_command" in source
    assert '"requested_corrective_residual_commands"' in source
    assert '"effective_corrective_residual_commands"' in source
    assert '"requested_vs_effective_residual_delta"' in source
    assert '"command_clipped"' in source
    assert "corrective_capture_dir," in source
    failure = source.split("def write_runtime_failure", 1)[1]
    assert '"model_based_planner_plus_corrective_teacher"' in failure
