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
