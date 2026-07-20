#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/wSpace/cinebotRL-two-wheel-riser"
WIN_ROOT='G:\wSpace\cinebotRL-two-wheel-riser'
PY="/mnt/g/isaaclab_venv/Scripts/python.exe"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
WINDOWS_POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v7_case74_relief_cpu"
MANIFEST_SHA256="0fe4b517d2629a1bca413162378708c2985cf5a42a1da8746de0a662f2fab00c"
SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
PLANNER_COMMIT="b0b0f300543bbc0e140f472ee4c9d3142284a906"
GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
ROBOT_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
CASE_TIMEOUT_SECONDS=480
MAXIMUM_DURATION_SCALE="3.00"
CONTROLLER_WZ_KP="1.05"
CAMERA_LEVER_ARM_GAIN="1.00"
MAXIMUM_CAMERA_LEVER_ARM_CORRECTION_M="0.05"
TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_v1"
ENABLE_CAMERA_ERROR_RECOVERY=0
CAMERA_RECOVERY_ERROR_START_M="0.13"
CAMERA_RECOVERY_ERROR_FULL_M="0.155"
MINIMUM_CAMERA_RECOVERY_SCALE="0.20"
REQUIRE_INITIALIZATION_PREROLL=0
USE_ROOT_VELOCITY_OUTER_FEEDBACK=0
REQUIRE_ZERO_PROGRESS_HOLD=0
TRACKING_MINIMUM_PROGRESS_SCALE=""
REQUIRE_RECOVERY_VELOCITY_CAP=0
TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS=""
REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT=0
REQUIRE_COMMANDED_BASE_PROGRESS_ERROR=0
REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET=0
VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS="0.05"
REVIEWED_CONTROLLER_PARENT_COMMIT="20ed7cb5e40c5eb8930e8df74e1842a162b1011b"

case "${RISER_CAMERA_LEVER_ARM_GATE_C_AUTHORIZATION:-}" in
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE68_66_CAMERA_LEVER_ARM_V1)
    CASE_A=68
    CASE_B=66
    CASE_A_PLAN_SHA256="4f4fc302402c53533f4bdbed33682bf52971a6f0cb93af3b42bd6da5ffeed142"
    CASE_B_PLAN_SHA256="ebdaf9a2e60e66c6231931bec6087c0b36a0895e22d4ee659e2b056b9b21bc37"
    STAMP="20260718_gate_c_smoothed_case68_66_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE67_7_CAMERA_LEVER_ARM_V1)
    CASE_A=67
    CASE_B=7
    CASE_A_PLAN_SHA256="e7acb5b9ca748645d878d360f357feb82e89b968f92d86c2639f2b74e03950e0"
    CASE_B_PLAN_SHA256="421f9f74a9f56cb79b49611355d9520489bf0bbe7204212ba169b84591fa4cd0"
    STAMP="20260718_gate_c_smoothed_case67_7_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE7_DYNAMIC_RETIME_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v8_case7_dynamic_retime_cpu"
    MANIFEST_SHA256="0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2"
    PLANNER_COMMIT="cbd4074d5caa76cc7dcb2277868e69430ad299e3"
    CASE_A=7
    CASE_B=""
    CASE_A_PLAN_SHA256="a83934dab6e4293cd830397d3c2ffb41d4f4d78545dddec7fdfa630fa0d22f41"
    CASE_B_PLAN_SHA256=""
    STAMP="20260718_gate_c_smoothed_case7_dynamic_retime_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE2_3_V8_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v8_case7_dynamic_retime_cpu"
    MANIFEST_SHA256="0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2"
    PLANNER_COMMIT="cbd4074d5caa76cc7dcb2277868e69430ad299e3"
    CASE_A=2
    CASE_B=3
    CASE_A_PLAN_SHA256="a2ad28cf4d353c59a9a642e39c8bbf484a0233df50a0b72b7ec18ca746c2cbe7"
    CASE_B_PLAN_SHA256="660384498d82b4c9752769e6d6319235f4c8d29164fd9b85ff8ea428c2264d51"
    STAMP="20260718_gate_c_smoothed_case2_3_v8_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE4_5_V8_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v8_case7_dynamic_retime_cpu"
    MANIFEST_SHA256="0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2"
    PLANNER_COMMIT="cbd4074d5caa76cc7dcb2277868e69430ad299e3"
    CASE_A=4
    CASE_B=5
    CASE_A_PLAN_SHA256="16e962e57b906d18561cc8640c4788c719bf95817492896c252affe6920e3ddb"
    CASE_B_PLAN_SHA256="90d84bea0731614f779a94f1a4f35b82be8fba9404a1f9460ae1e3fa6a80dec4"
    STAMP="20260718_gate_c_smoothed_case4_5_v8_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE6_8_V8_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v8_case7_dynamic_retime_cpu"
    MANIFEST_SHA256="0a6a9361095e3045b2835f2ea96520f2b6e1c378df4feaa394fb87627bc165b2"
    PLANNER_COMMIT="cbd4074d5caa76cc7dcb2277868e69430ad299e3"
    CASE_A=6
    CASE_B=8
    CASE_A_PLAN_SHA256="b8ac6a9bb226de47a2722f076efb6dcf9586fd3b85740cf6efb5926cd86568aa"
    CASE_B_PLAN_SHA256="2e5c51b293be2147b8a4095a28f2f960880059b25b5a9b8baf586ce56dce16ac"
    STAMP="20260718_gate_c_smoothed_case6_8_v8_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE8_DYNAMIC_RETIME_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu"
    MANIFEST_SHA256="ac5da6ce721bd0af51b9b851ada86b08f587f190440c9de23172b115bad3c748"
    PLANNER_COMMIT="b7917da1ba864647a252410ae06165815240aeb5"
    CASE_A=8
    CASE_B=""
    CASE_A_PLAN_SHA256="f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655"
    CASE_B_PLAN_SHA256=""
    STAMP="20260718_gate_c_smoothed_case8_dynamic_retime_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_13_V9_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu"
    MANIFEST_SHA256="ac5da6ce721bd0af51b9b851ada86b08f587f190440c9de23172b115bad3c748"
    PLANNER_COMMIT="b7917da1ba864647a252410ae06165815240aeb5"
    CASE_A=9
    CASE_B=13
    CASE_A_PLAN_SHA256="6e04791453a1a1f73eed52ffa469b5cf160cbf30e3eb8dff932453b54c00e716"
    CASE_B_PLAN_SHA256="0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228"
    STAMP="20260718_gate_c_smoothed_case9_13_v9_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_13_V9_CAMERA_LEVER_ARM_RETRY_V2)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v9_case8_dynamic_retime_cpu"
    MANIFEST_SHA256="ac5da6ce721bd0af51b9b851ada86b08f587f190440c9de23172b115bad3c748"
    PLANNER_COMMIT="b7917da1ba864647a252410ae06165815240aeb5"
    CASE_A=9
    CASE_B=13
    CASE_A_PLAN_SHA256="6e04791453a1a1f73eed52ffa469b5cf160cbf30e3eb8dff932453b54c00e716"
    CASE_B_PLAN_SHA256="0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228"
    STAMP="20260718_gate_c_smoothed_case9_13_v9_camera_lever_arm_retry_v2_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE9_DYNAMIC_RETIME_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=9
    CASE_B=""
    CASE_A_PLAN_SHA256="195249929b363e49fcc73a2600c2d7de9dc9d9fedf0bb9ed0718a44e76bf3fd3"
    CASE_B_PLAN_SHA256=""
    STAMP="20260718_gate_c_smoothed_case9_dynamic_retime_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE13_V10_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=13
    CASE_B=""
    CASE_A_PLAN_SHA256="0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228"
    CASE_B_PLAN_SHA256=""
    STAMP="20260718_gate_c_smoothed_case13_v10_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE13_V10_CAMERA_LEVER_ARM_RETRY_V2)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=13
    CASE_B=""
    CASE_A_PLAN_SHA256="0451bc312420b1d1a026afb89c23ddb0b325a8b9da10246918e42a067494a228"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1600
    STAMP="20260718_gate_c_smoothed_case13_v10_camera_lever_arm_retry_v2_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE14_V10_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=14
    CASE_B=""
    CASE_A_PLAN_SHA256="e863db5bc93c25bf91f31ac6dbcbd11fa091830290aaf64c58a4a3982d5cae58"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2100
    STAMP="20260719_gate_c_smoothed_case14_v10_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE15_V10_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=15
    CASE_B=""
    CASE_A_PLAN_SHA256="8626af7d6d2feeb22d0eb4b2136f0617f91f1fbd3dc87c639d0f459f3c38c25f"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1100
    STAMP="20260719_gate_c_smoothed_case15_v10_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_V10_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260718_smoothed_plan_all79_v10_case9_dynamic_retime_cpu"
    MANIFEST_SHA256="229a76e3003b2e31a0d1a7a7cd34cda208b292638e7039e79198c951e034cda1"
    PLANNER_COMMIT="c82e3658a7769a424a26eb22a203cd0b4ae39d52"
    CASE_A=16
    CASE_B=""
    CASE_A_PLAN_SHA256="847d1302086dae794e009f23c2a90869a262a43ca77912d88544f0fdb7492c58"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1400
    STAMP="20260719_gate_c_smoothed_case16_v10_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_DYNAMIC_RETIME_V11)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v11_case16_dynamic_retime_cpu"
    MANIFEST_SHA256="56670dd0ecbdf0157361bef65af50f8d688a9e86bc3e0ff50768472b17474032"
    PLANNER_COMMIT="a84d13ea006ebc7c6053a2ba5a63a287aa7fbd53"
    CASE_A=16
    CASE_B=""
    CASE_A_PLAN_SHA256="8bcf14454ce4b087973e0c0d2c6efb3858edf75209e195dec7fc09fe7111c821"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case16_dynamic_retime_v11_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE16_EXPLICIT_PREVIEW015_V12)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=16
    CASE_B=""
    CASE_A_PLAN_SHA256="742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case16_explicit_preview015_v12_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE17_V12_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=17
    CASE_B=""
    CASE_A_PLAN_SHA256="e38228121caf797546ac0936fc522e84f61f04cd3740438e0b93469665fa938d"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case17_v12_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE18_V12_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=18
    CASE_B=""
    CASE_A_PLAN_SHA256="121b0f336dd1e236aaee2b9bf0b158466636624507c107e2d90935339edf2517"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    STAMP="20260719_gate_c_smoothed_case18_v12_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE19_V12_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=19
    CASE_B=""
    CASE_A_PLAN_SHA256="8cf8bde298c73d1809c3dc7c0dae249446d7554ba77275a490d32fc1a6004b37"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=700
    STAMP="20260719_gate_c_smoothed_case19_v12_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE20_V12_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=20
    CASE_B=""
    CASE_A_PLAN_SHA256="ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=800
    STAMP="20260719_gate_c_smoothed_case20_v12_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE20_V12_CAMERA_ERROR_GOVERNOR_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=20
    CASE_B=""
    CASE_A_PLAN_SHA256="ec0bb2845c948d17daec8abef6b00b205f6f56fe6cb9e4c42aa9395c6b66336d"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=800
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_error_governor_v1"
    ENABLE_CAMERA_ERROR_RECOVERY=1
    STAMP="20260719_gate_c_smoothed_case20_v12_camera_error_governor_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE21_V12_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v12_case16_explicit_preview015_cpu"
    MANIFEST_SHA256="59e572712879e25a687bebd17be94b8464e7f0de08ef3d5ce2102bb9303a5581"
    PLANNER_COMMIT="27574668625e55f858fac72df401d6165775b948"
    CASE_A=21
    CASE_B=""
    CASE_A_PLAN_SHA256="85029afbbcce435ec8df27770b521b0ab57eae8d98ab4a2dc7f7b7680efaa9ba"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1100
    STAMP="20260719_gate_c_smoothed_case21_v12_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE21_V13_LOCALIZED_REVERSAL_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v13_case21_localized_reversal_cpu"
    MANIFEST_SHA256="40611139cb50c4431c238994f311e578c6b43f754ad07b700ec54576a8574e3e"
    PLANNER_COMMIT="823db97916f67ff0d0d7819d81c654ffaa889aab"
    CASE_A=21
    CASE_B=""
    CASE_A_PLAN_SHA256="81c0da4be22d5b800978d1d46ca9705912f72007f7c615b31715c672dd86a1d4"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1200
    STAMP="20260719_gate_c_smoothed_case21_v13_localized_reversal_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE22_V13_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v13_case21_localized_reversal_cpu"
    MANIFEST_SHA256="40611139cb50c4431c238994f311e578c6b43f754ad07b700ec54576a8574e3e"
    PLANNER_COMMIT="823db97916f67ff0d0d7819d81c654ffaa889aab"
    CASE_A=22
    CASE_B=""
    CASE_A_PLAN_SHA256="b36626c23d41ecd647f91f9c98e1e06abeb1320fbc96a3a59aea052926a39b75"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1100
    STAMP="20260719_gate_c_smoothed_case22_v13_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE22_V14_LOCALIZED_REVERSAL_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v14_case22_localized_reversal_cpu"
    MANIFEST_SHA256="369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab"
    PLANNER_COMMIT="2e3f769dc8800dd84fc6b6c221b68d0b9a5e7fc0"
    CASE_A=22
    CASE_B=""
    CASE_A_PLAN_SHA256="8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1200
    STAMP="20260719_gate_c_smoothed_case22_v14_localized_reversal_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE30_V14_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v14_case22_localized_reversal_cpu"
    MANIFEST_SHA256="369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab"
    PLANNER_COMMIT="2e3f769dc8800dd84fc6b6c221b68d0b9a5e7fc0"
    CASE_A=30
    CASE_B=""
    CASE_A_PLAN_SHA256="1722bfdc7c1aeabc5a9d3920cf6a47bc789afbc96e6ef5c8e540695dc3c97dcb"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case30_v14_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE31_V14_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v14_case22_localized_reversal_cpu"
    MANIFEST_SHA256="369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab"
    PLANNER_COMMIT="2e3f769dc8800dd84fc6b6c221b68d0b9a5e7fc0"
    CASE_A=31
    CASE_B=""
    CASE_A_PLAN_SHA256="8ebc938eeb53b8f7dbf4382a085d3667ea38d5ea52e535dc3be409767737aefb"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case31_v14_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V14_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v14_case22_localized_reversal_cpu"
    MANIFEST_SHA256="369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab"
    PLANNER_COMMIT="2e3f769dc8800dd84fc6b6c221b68d0b9a5e7fc0"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="45040c19379c0f56f68f44e6391033d2342769f3c034cc281d12f4e5f0cb35a1"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1600
    STAMP="20260719_gate_c_smoothed_case32_v14_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V14_CAMERA_ERROR_GOVERNOR_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v14_case22_localized_reversal_cpu"
    MANIFEST_SHA256="369e3294a45ef468979a81a8bf34b9012f9ec4f77a1d4489c4514930f2d79dab"
    PLANNER_COMMIT="2e3f769dc8800dd84fc6b6c221b68d0b9a5e7fc0"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="45040c19379c0f56f68f44e6391033d2342769f3c034cc281d12f4e5f0cb35a1"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1600
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_error_governor_v1"
    ENABLE_CAMERA_ERROR_RECOVERY=1
    STAMP="20260719_gate_c_smoothed_case32_v14_camera_error_governor_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V15_EXPLICIT_PREVIEW0175_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V15_EXPLICIT_PREVIEW0175_V2)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v2_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V15_EXPLICIT_PREVIEW0175_V3)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260719_gate_c_smoothed_case32_v15_explicit_preview0175_v3_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE32_V15_EXPLICIT_PREVIEW0175_V4)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=32
    CASE_B=""
    CASE_A_PLAN_SHA256="71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260720_gate_c_smoothed_case32_v15_explicit_preview0175_v4_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE33_V15_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=33
    CASE_B=""
    CASE_A_PLAN_SHA256="052b828587efef44e8f17bc6c8a4d73dcbfc2d35466ae02f5dd1a60f64af8d00"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1600
    STAMP="20260720_gate_c_smoothed_case33_v15_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE34_V15_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=34
    CASE_B=""
    CASE_A_PLAN_SHA256="e2b170f649f9e90542bfaa463c74fa802c0247273d7fad8c26f24922c212b9d4"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2000
    STAMP="20260720_gate_c_smoothed_case34_v15_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE36_V15_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260719_smoothed_plan_all79_v15_case32_explicit_preview0175_cpu"
    MANIFEST_SHA256="ef084a77e9f9fe633d8f6918d4e29808d7b339fe2e7db939c4aa826d597f1977"
    PLANNER_COMMIT="6bc1ad879ca27add191d3ebcc4ce961388524ba6"
    CASE_A=36
    CASE_B=""
    CASE_A_PLAN_SHA256="16b8d492571794b057a6747235ce37ce26173058c776773c2eaf717e38f1fe95"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260720_gate_c_smoothed_case36_v15_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE36_V16_EXPLICIT_PREVIEW055_G125_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
    MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
    PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
    CASE_A=36
    CASE_B=""
    CASE_A_PLAN_SHA256="d1e4da8ea73a26a8ac9f7b3d7063d2272569a7375f5ec8feed6e9a238a3c08ed"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=1500
    STAMP="20260720_gate_c_smoothed_case36_v16_explicit_preview055_g125_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE37_V16_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
    MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
    PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
    CASE_A=37
    CASE_B=""
    CASE_A_PLAN_SHA256="3bc3119fa210f1fd190d7fba11b9571caa74dc1bc4de02fb98296ecc9e8d2c1e"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2000
    STAMP="20260720_gate_c_smoothed_case37_v16_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE41_V16_CAMERA_LEVER_ARM_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
    MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
    PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
    CASE_A=41
    CASE_B=""
    CASE_A_PLAN_SHA256="cf3c1f35fbf20377c23dbc7ff3d24fbca8cdc9ef833cf1eff925d585295a4679"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2800
    STAMP="20260720_gate_c_smoothed_case41_v16_camera_lever_arm_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V17_STATIC_MARGIN_PREVIEW055_G125_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v17_case42_static_margin_preview055_g125_cpu"
    MANIFEST_SHA256="57f6eab7453d0d1d3f2c244b7d429bb5ac1fa95184f63b4212770dc9fefb1a51"
    PLANNER_COMMIT="8e78e6c7d825f178fd4b49eb30aedf1adab77393"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="b2057b0a23c9b5172f09a5373a020b18583cf118d55886624332f1d4e861a298"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2000
    STAMP="20260720_gate_c_smoothed_case42_v17_static_margin_preview055_g125_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V18_LOCALIZED_REVERSAL_RETIME_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v18_case42_localized_reversal_retime_cpu"
    MANIFEST_SHA256="b296a32f64a3e1f22b3a2fb51db2dd426baa2e1374d36cdf0eaaf5a5cdecd5a1"
    PLANNER_COMMIT="4d5db5a5a5c9977c53f25a5d0cc744f94962071e"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="df694d8e6702ac0712ff7e1ce597c79ac30e3fc08f072caf63245fe8740e6669"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2300
    STAMP="20260720_gate_c_smoothed_case42_v18_localized_reversal_retime_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V16_BASELINE_ROLLBACK_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v16_case36_explicit_preview055_g125_cpu"
    MANIFEST_SHA256="8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1"
    PLANNER_COMMIT="0391190f536a29f65b4c97968b764f29444c9f43"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="f737f0b2e1fe4877685ae2bc4a976c2179dce5ce8c30491146d14b3994eb4343"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    STAMP="20260720_gate_c_smoothed_case42_v16_baseline_rollback_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_INITIALIZATION_PREROLL2S_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    STAMP="20260720_gate_c_smoothed_case42_v20_initialization_preroll2s_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ROOT_VX_OUTER_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    USE_ROOT_VELOCITY_OUTER_FEEDBACK=1
    STAMP="20260720_gate_c_smoothed_case42_v20_root_vx_outer_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ZERO_PROGRESS_HOLD_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    REQUIRE_ZERO_PROGRESS_HOLD=1
    TRACKING_MINIMUM_PROGRESS_SCALE="0.0"
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1"
    STAMP="20260720_gate_c_smoothed_case42_v20_zero_progress_hold_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ZERO_PROGRESS_HOLD_CAP020_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    REQUIRE_ZERO_PROGRESS_HOLD=1
    TRACKING_MINIMUM_PROGRESS_SCALE="0.0"
    REQUIRE_RECOVERY_VELOCITY_CAP=1
    TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS="0.2"
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_velocity_cap_v1"
    STAMP="20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ZERO_PROGRESS_HOLD_CAP020_TOTAL_PITCH_LIMIT_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    REQUIRE_ZERO_PROGRESS_HOLD=1
    TRACKING_MINIMUM_PROGRESS_SCALE="0.0"
    REQUIRE_RECOVERY_VELOCITY_CAP=1
    TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS="0.2"
    REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT=1
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    STAMP="20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_total_pitch_limit_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ZERO_PROGRESS_HOLD_CAP020_TOTAL_PITCH_COMMANDED_BASE_PROGRESS_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    REQUIRE_ZERO_PROGRESS_HOLD=1
    TRACKING_MINIMUM_PROGRESS_SCALE="0.0"
    REQUIRE_RECOVERY_VELOCITY_CAP=1
    TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS="0.2"
    REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT=1
    REQUIRE_COMMANDED_BASE_PROGRESS_ERROR=1
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    STAMP="20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_total_pitch_commanded_base_progress_v1_exclusive"
    ;;
  AUTHORIZED_RISER_SMOOTHED_GATE_C_CASE42_V20_ZERO_PROGRESS_HOLD_CAP020_TOTAL_PITCH_COMMANDED_BASE_PROGRESS_OPPOSING_PI_RESET_V1)
    PORTFOLIO_STAMP="20260720_smoothed_plan_all79_v20_case42_initialization_preroll2s_cpu"
    MANIFEST_SHA256="3d7f9650a4f701f80a11948364a53ecd34641160bffb6bc3ed697d038d559b72"
    PLANNER_COMMIT="5a66e3deef01fceacc80fee37b199045705d7f02"
    CASE_A=42
    CASE_B=""
    CASE_A_PLAN_SHA256="ea2e54273c42efa3980eaa3ea9b161109702047467df131d4ad1d2604f063984"
    CASE_B_PLAN_SHA256=""
    CASE_TIMEOUT_SECONDS=2200
    REQUIRE_INITIALIZATION_PREROLL=1
    REQUIRE_ZERO_PROGRESS_HOLD=1
    TRACKING_MINIMUM_PROGRESS_SCALE="0.0"
    REQUIRE_RECOVERY_VELOCITY_CAP=1
    TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS="0.2"
    REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT=1
    REQUIRE_COMMANDED_BASE_PROGRESS_ERROR=1
    REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET=1
    REVIEWED_CONTROLLER_PARENT_COMMIT="35f775c39ed2d0c22b52be5dd8f9641354ee0b8f"
    TRACKING_PROFILE="riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    STAMP="20260720_gate_c_smoothed_case42_v20_zero_progress_hold_cap020_total_pitch_commanded_base_progress_opposing_pi_reset_v1_exclusive"
    ;;
  *)
    printf 'camera lever-arm Gate C authorization is absent or unknown\n' >&2
    exit 7
    ;;
esac
CASE_LIST=("$CASE_A")
[[ -z "$CASE_B" ]] || CASE_LIST+=("$CASE_B")
CASES="$(IFS=,; printf '%s' "${CASE_LIST[*]}")"

PORTFOLIO="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${PORTFOLIO_STAMP}"
SOURCE_MANIFEST="/mnt/g/wSpace/cinebotRL/data/gikWBC9DOF_exact_source_reference_all79_20260717/manifest.json"
OUTPUT="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="${WIN_ROOT}\\artifacts\\two_wheel_riser\\${STAMP}"
GAINS="$ROOT/docs/03_training/two_wheel_balance/evidence_20260714_28kg/lqr_gains.json"
GAINS_WIN="$WIN_ROOT\docs\03_training\two_wheel_balance\evidence_20260714_28kg\lqr_gains.json"
ROBOT_USD="$ROOT/assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.usd"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_smoothed_gate_c_canary.py"
PLAYBACK="$ROOT/scripts/two_wheel_balance/smoke_riser_reference_playback.py"
PLAYBACK_WIN="$WIN_ROOT\scripts\two_wheel_balance\smoke_riser_reference_playback.py"
SUMMARIZER="$ROOT/scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"
RUNNER="$ROOT/scripts/two_wheel_balance/run_riser_smoothed_gate_c_camera_lever_arm.sh"
LOADER="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_playback.py"
TRACKING="$ROOT/src/rl_platform/tasks/two_wheel_balance/whole_body_tracking.py"
RISER_CONTROL="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_control.py"
RECOVERY_EVIDENCE="$ROOT/src/rl_platform/tasks/two_wheel_balance/riser_recovery_evidence.py"
METRICS="$ROOT/src/rl_platform/tasks/two_wheel_balance/metrics.py"

assert_gpu_free() {
  local playback_owners compute_owners windows_owners
  playback_owners="$(ps -ef | grep -E '[p]ython(\.exe)? .*smoke_.*playback\.py' || true)"
  compute_owners="$($NVIDIA_SMI --query-compute-apps=pid,process_name --format=csv,noheader)"
  windows_owners="$(
    "$WINDOWS_POWERSHELL" -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "Stop"
      $queryProcessId = $PID
      Get-CimInstance Win32_Process |
        Where-Object {
          $_.ProcessId -ne $queryProcessId -and (
            $_.Name -eq "kit.exe" -or
            $_.CommandLine -match "smoke_.*playback|evaluate_cascade_robustness"
          )
        } |
        ForEach-Object { "{0}`t{1}" -f $_.ProcessId, $_.CommandLine }
    ' | tr -d '\r'
  )"
  if [[ -n "$playback_owners" || -n "$compute_owners" || -n "$windows_owners" ]]; then
    printf 'camera lever-arm Gate C GPU is not free\n' >&2
    [[ -z "$playback_owners" ]] || printf '%s\n' "$playback_owners" >&2
    [[ -z "$compute_owners" ]] || printf '%s\n' "$compute_owners" >&2
    [[ -z "$windows_owners" ]] || printf '%s\n' "$windows_owners" >&2
    return 1
  fi
}

assert_no_competing_cpu() {
  ! ps -ef | grep -qE '[r]etarget_exact_source_v1_nonholonomic\.py' || {
    printf 'camera lever-arm Gate C CPU/disk ownership is not exclusive\n' >&2
    return 1
  }
}

assert_exclusive_resources() {
  assert_gpu_free && assert_no_competing_cpu
}

wait_for_gpu_release() {
  local attempt
  for attempt in $(seq 1 90); do
    assert_gpu_free 2>/dev/null && return 0
    sleep 1
  done
  printf 'camera lever-arm Gate C GPU did not release within 90 seconds\n' >&2
  return 1
}

case_gate_passed() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" <<'PY'
import json
import math
from pathlib import Path
import sys

gate = json.loads(Path(sys.argv[1]).read_text())
case = int(sys.argv[2])
expected_tracking_profile = sys.argv[3]
require_camera_recovery = bool(int(sys.argv[4]))
require_initialization = bool(int(sys.argv[5]))
require_root_velocity_feedback = bool(int(sys.argv[6]))
require_zero_progress_hold = bool(int(sys.argv[7]))
require_recovery_velocity_cap = bool(int(sys.argv[8]))
expected_maximum_linear_velocity_mps = (
    float(sys.argv[9]) if require_recovery_velocity_cap else None
)
require_total_pitch_reference_limit = bool(int(sys.argv[10]))
require_commanded_base_progress_error = bool(int(sys.argv[11]))
require_opposing_vx_integral_deficit_reset = bool(int(sys.argv[12]))
expected_vx_integral_reset_deadband_mps = float(sys.argv[13])
result = gate.get("results", [{}])[0]
correction_max = result.get("camera_lever_arm_correction_max_m")
raw_max = result.get("camera_lever_arm_raw_correction_max_m")
saturation_ratio = result.get("camera_lever_arm_correction_saturation_ratio")
numeric = (correction_max, raw_max, saturation_ratio)
recovery_numeric = (
    result.get("camera_recovery_activation_ratio"),
    result.get("camera_recovery_progress_scale_min"),
    result.get("camera_recovery_progress_scale_mean"),
)
recovery_ok = not require_camera_recovery or (
    gate.get("camera_recovery_governor_enabled") is True
    and gate.get("camera_recovery_governor_contract")
    == "saturated_camera_error_continuous_phase_cap_v1"
    and gate.get("camera_recovery_error_range_m") == [0.13, 0.155]
    and gate.get("minimum_camera_recovery_scale") == 0.2
    and result.get("camera_recovery_governor_enabled") is True
    and result.get("camera_recovery_governor_contract")
    == "saturated_camera_error_continuous_phase_cap_v1"
    and result.get("camera_recovery_error_range_m") == [0.13, 0.155]
    and result.get("minimum_camera_recovery_scale") == 0.2
    and result.get("camera_recovery_telemetry_observed") is True
    and result.get("camera_recovery_telemetry_sample_count")
    == result.get("completed_steps")
    and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in recovery_numeric
    )
    and 0.0 < recovery_numeric[0] <= 1.0
    and 0.2 - 1e-12 <= recovery_numeric[1] <= recovery_numeric[2] <= 1.0
)
initialization_numeric = (
    result.get("initialization_terminal_base_error_m"),
    result.get("initialization_terminal_base_yaw_error_deg"),
    result.get("initialization_terminal_riser_error_m"),
    result.get("initialization_terminal_proxy_error_deg"),
    result.get("initialization_action_saturation_ratio"),
    result.get("initialization_riser_thermal_load_max"),
    result.get("initialization_riser_effort_max_n"),
)
initialization_ok = not require_initialization or (
    result.get("initialization_duration_s") == 2.0
    and result.get("initialization_steps") == 400
    and result.get("initialization_completed") is True
    and result.get("initialization_scored_as_source_tracking") is False
    and result.get("initialization_source_metric_samples") == 0
    and result.get("initialization_residual_label_samples") == 0
    and result.get("initialization_riser_thermal_sample_count") == 400
    and all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0
        for value in initialization_numeric
    )
    and result.get("checks", {}).get(
        "initialization_action_saturation_bounded"
    ) is True
    and result.get("checks", {}).get(
        "initialization_riser_thermal_force_observed"
    ) is True
    and result.get("checks", {}).get(
        "initialization_riser_thermal_load_bounded"
    ) is True
    and result.get("checks", {}).get(
        "initialization_riser_peak_force_bounded"
    ) is True
    and result.get("checks", {}).get(
        "initialization_source_metrics_clean"
    ) is True
)
velocity_feedback = result.get("velocity_feedback_telemetry")
root_velocity_feedback_ok = not require_root_velocity_feedback or (
    result.get("outer_velocity_feedback_source") == "root_link_vx"
    and result.get("velocity_feedback_telemetry_observed") is True
    and isinstance(velocity_feedback, dict)
    and velocity_feedback.get("schema")
    == "riser_root_vs_wheel_velocity_policy_rate_v1"
    and velocity_feedback.get("policy_rate_sample_count")
    == result.get("completed_steps")
    and result.get("checks", {}).get(
        "velocity_feedback_telemetry_observed"
    ) is True
)
hold_steps = result.get("progress_hold_step_count")
hold_ratio = result.get("progress_hold_ratio")
hold_segments = result.get("progress_hold_segment_count")
completed_steps = result.get("completed_steps")
expected_tracking_overrides = {"minimum_progress_scale": 0.0}
if require_recovery_velocity_cap:
    expected_tracking_overrides["maximum_linear_velocity_mps"] = (
        expected_maximum_linear_velocity_mps
    )
velocity_feedback = result.get("velocity_feedback_telemetry")
recovery_velocity_cap_ok = not require_recovery_velocity_cap or (
    gate.get("tracking_recovery_velocity_cap_enabled") is True
    and gate.get("maximum_linear_velocity_mps")
    == expected_maximum_linear_velocity_mps
    and result.get("maximum_linear_velocity_mps")
    == expected_maximum_linear_velocity_mps
    and isinstance(velocity_feedback, dict)
    and isinstance(
        velocity_feedback.get("effective_reference_abs_max_mps"), (int, float)
    )
    and math.isfinite(velocity_feedback["effective_reference_abs_max_mps"])
    and velocity_feedback["effective_reference_abs_max_mps"]
    <= expected_maximum_linear_velocity_mps + 1e-9
)
expected_total_pitch_limit_rad = math.radians(6.0)
total_pitch_reference_limit_ok = not require_total_pitch_reference_limit or (
    gate.get("total_pitch_reference_limit_enabled") is True
    and gate.get("total_pitch_reference_limit_rad") == expected_total_pitch_limit_rad
    and result.get("total_pitch_reference_limit_enabled") is True
    and result.get("total_pitch_reference_limit_rad") == expected_total_pitch_limit_rad
    and isinstance(velocity_feedback, dict)
    and isinstance(
        velocity_feedback.get("total_pitch_reference_abs_max_rad"), (int, float)
    )
    and math.isfinite(velocity_feedback["total_pitch_reference_abs_max_rad"])
    and expected_total_pitch_limit_rad - 1e-9
    <= velocity_feedback["total_pitch_reference_abs_max_rad"]
    <= expected_total_pitch_limit_rad + 1e-9
    and isinstance(velocity_feedback.get("pitch_reference_abs_max_rad"), (int, float))
    and math.isfinite(velocity_feedback["pitch_reference_abs_max_rad"])
    and velocity_feedback["pitch_reference_abs_max_rad"]
    > expected_total_pitch_limit_rad + 1e-6
)
expected_phase_governor_contract = (
    "commanded_base_and_camera_error_continuous_phase_scale_v1"
    if require_commanded_base_progress_error
    else "position_error_continuous_phase_scale_v1"
)
progress_base_error_numeric = (
    result.get("nominal_base_progress_error_p95_m"),
    result.get("nominal_base_progress_error_max_m"),
    result.get("commanded_base_progress_error_p95_m"),
    result.get("commanded_base_progress_error_max_m"),
    result.get("selected_base_progress_error_p95_m"),
    result.get("selected_base_progress_error_max_m"),
    result.get("selected_vs_nominal_base_progress_error_mean_delta_m"),
    result.get("selected_vs_nominal_base_progress_error_abs_max_delta_m"),
    result.get("maximum_commanded_base_progress_error_delta_m"),
)
commanded_base_progress_error_ok = (
    not require_commanded_base_progress_error
    or (
        gate.get("commanded_base_progress_error_enabled") is True
        and gate.get("progress_base_error_source")
        == "lever_compensated_commanded_base_target"
        and result.get("phase_governor_contract")
        == expected_phase_governor_contract
        and result.get("commanded_base_progress_error_enabled") is True
        and result.get("progress_base_error_source")
        == "lever_compensated_commanded_base_target"
        and result.get("progress_base_error_telemetry_observed") is True
        and result.get("progress_base_error_telemetry_sample_count")
        == result.get("completed_steps")
        and result.get("progress_base_error_selected_source_matches") is True
        and result.get("progress_base_error_command_delta_bounded") is True
        and result.get("checks", {}).get(
            "progress_base_error_telemetry_observed"
        ) is True
        and result.get("checks", {}).get(
            "progress_base_error_selected_source_matches"
        ) is True
        and result.get("checks", {}).get(
            "progress_base_error_command_delta_bounded"
        ) is True
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in progress_base_error_numeric
        )
        and all(value >= 0.0 for value in progress_base_error_numeric[:6])
        and progress_base_error_numeric[8] == 0.05
        and 0.0 <= progress_base_error_numeric[7] <= 0.05 + 1e-9
    )
)
longitudinal_authority = result.get("longitudinal_authority_telemetry")
authority_integer_fields = (
    "policy_rate_sample_count",
    "controller_update_count",
    "held_controller_command_step_count",
    "reference_sign_change_count",
    "opposing_integral_sign_change_count",
    "integral_reset_count",
    "velocity_deficit_step_count",
    "total_pitch_limit_step_count",
)
authority_numeric_fields = (
    "velocity_deficit_ratio",
    "velocity_deficit_mean_mps",
    "velocity_deficit_abs_max_mps",
    "deficit_pitch_contribution_mean",
    "deficit_pitch_rate_contribution_mean",
    "deficit_wheel_velocity_contribution_mean",
    "vx_integral_before_abs_max",
    "vx_integral_after_abs_max",
    "pitch_abs_max_rad",
    "pitch_rate_abs_max_rad_s",
    "total_pitch_reference_abs_max_rad",
    "common_action_abs_max",
    "reference_deadband_mps",
    "deficit_tolerance_mps",
)
longitudinal_authority_ok = (
    not require_opposing_vx_integral_deficit_reset
    or (
        gate.get("opposing_vx_integral_deficit_reset_enabled") is True
        and gate.get("vx_integral_reset_reference_deadband_mps")
        == expected_vx_integral_reset_deadband_mps
        and result.get("opposing_vx_integral_deficit_reset_enabled") is True
        and result.get("vx_integral_reset_reference_deadband_mps")
        == expected_vx_integral_reset_deadband_mps
        and result.get("longitudinal_authority_telemetry_observed") is True
        and result.get("checks", {}).get(
            "longitudinal_authority_telemetry_observed"
        ) is True
        and isinstance(longitudinal_authority, dict)
        and longitudinal_authority.get("schema")
        == "riser_longitudinal_authority_policy_rate_v1"
        and all(
            isinstance(longitudinal_authority.get(name), int)
            and longitudinal_authority[name] >= 0
            for name in authority_integer_fields
        )
        and all(
            isinstance(longitudinal_authority.get(name), (int, float))
            and math.isfinite(longitudinal_authority[name])
            for name in authority_numeric_fields
        )
        and longitudinal_authority["policy_rate_sample_count"]
        == result.get("completed_steps")
        and longitudinal_authority["controller_update_count"] > 0
        and longitudinal_authority["held_controller_command_step_count"]
        == longitudinal_authority["policy_rate_sample_count"]
        - longitudinal_authority["controller_update_count"]
        and 0 < longitudinal_authority["integral_reset_count"]
        <= longitudinal_authority["controller_update_count"]
        and longitudinal_authority["reference_deadband_mps"]
        == expected_vx_integral_reset_deadband_mps
        and 0.0 <= longitudinal_authority["velocity_deficit_ratio"] <= 1.0
    )
)
zero_progress_hold_ok = not require_zero_progress_hold or (
    gate.get("phase_governor_enabled") is True
    and gate.get("phase_governor_contract")
    == expected_phase_governor_contract
    and gate.get("minimum_progress_scale") == 0.0
    and gate.get("tracking_overrides") == expected_tracking_overrides
    and result.get("minimum_progress_scale") == 0.0
    and result.get("progress_scale_min") == 0.0
    and result.get("outer_velocity_feedback_source") == "wheel_derived_vx"
    and isinstance(completed_steps, int)
    and completed_steps > 0
    and isinstance(hold_steps, int)
    and 0 < hold_steps <= completed_steps
    and isinstance(hold_segments, int)
    and 0 < hold_segments <= hold_steps
    and isinstance(hold_ratio, (int, float))
    and math.isfinite(hold_ratio)
    and math.isclose(
        hold_ratio, hold_steps / completed_steps, rel_tol=0.0, abs_tol=1e-12
    )
)
expected_controller_overrides = {"wz_kp": 1.05}
if require_total_pitch_reference_limit:
    expected_controller_overrides["limit_total_pitch_reference"] = True
if require_opposing_vx_integral_deficit_reset:
    expected_controller_overrides.update(
        {
            "reset_opposing_vx_integral_on_directional_deficit": True,
            "vx_integral_reset_reference_deadband_mps": (
                expected_vx_integral_reset_deadband_mps
            ),
        }
    )
ok = (
    gate.get("cases") == [case]
    and len(gate.get("results", [])) == 1
    and gate.get("dynamic_quality_passed") is True
    and result.get("dynamic_quality_passed") is True
    and gate.get("thermal_admission_passed") is True
    and result.get("thermal_admission_passed") is True
    and gate.get("controller_evidence_passed") is True
    and result.get("controller_evidence_passed") is True
    and gate.get("controller_overrides") == expected_controller_overrides
    and gate.get("tracking_profile") == expected_tracking_profile
    and gate.get("camera_lever_arm_compensation_contract")
    == "measured_camera_to_base_xy_offset_v1"
    and gate.get("camera_lever_arm_compensation_enabled") is True
    and gate.get("camera_lever_arm_compensation_gain") == 1.0
    and gate.get("maximum_camera_lever_arm_correction_m") == 0.05
    and result.get("camera_lever_arm_compensation_enabled") is True
    and result.get("camera_lever_arm_compensation_gain") == 1.0
    and result.get("maximum_camera_lever_arm_correction_m") == 0.05
    and result.get("camera_lever_arm_telemetry_observed") is True
    and result.get("camera_lever_arm_telemetry_sample_count")
    == result.get("completed_steps")
    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric)
    and 0.0 <= correction_max <= 0.05 + 1e-9
    and raw_max + 1e-12 >= correction_max
    and 0.0 <= saturation_ratio <= 1.0
    and gate.get("trajectory_command_source") == "deterministic_teacher"
    and gate.get("residual_policy") is None
    and result.get("executed_residual_dataset") is None
    and result.get("raw_residual_label_applied_to_commands") is False
    and gate.get("training_started") is False
    and gate.get("ppo_authorized") is False
    and isinstance(result.get("residual_label_envelope_passed"), bool)
    and recovery_ok
    and initialization_ok
    and root_velocity_feedback_ok
    and zero_progress_hold_ok
    and recovery_velocity_cap_ok
    and total_pitch_reference_limit_ok
    and commanded_base_progress_error_ok
    and longitudinal_authority_ok
)
raise SystemExit(0 if ok else 6)
PY
}

[[ -x "$PY" && -x "$NVIDIA_SMI" && -x "$WINDOWS_POWERSHELL" ]] || exit 2
[[ ! -e "$OUTPUT" ]] || { printf 'refusing existing namespace: %s\n' "$OUTPUT" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || exit 2
[[ "$(sha256sum "$PORTFOLIO/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || exit 2
CASE_A_FILE="case_$(printf '%04d' "$CASE_A")_smoothed_riser_plan_v1.npz"
[[ "$(sha256sum "$PORTFOLIO/$CASE_A_FILE" | awk '{print $1}')" == "$CASE_A_PLAN_SHA256" ]] || exit 2
if [[ -n "$CASE_B" ]]; then
  CASE_B_FILE="case_$(printf '%04d' "$CASE_B")_smoothed_riser_plan_v1.npz"
  [[ "$(sha256sum "$PORTFOLIO/$CASE_B_FILE" | awk '{print $1}')" == "$CASE_B_PLAN_SHA256" ]] || exit 2
fi
[[ "$(sha256sum "$GAINS" | awk '{print $1}')" == "$GAINS_SHA256" ]] || exit 2
[[ "$(sha256sum "$ROBOT_USD" | awk '{print $1}')" == "$ROBOT_USD_SHA256" ]] || exit 2

git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make runtime provenance ambiguous\n' >&2
  exit 2
}
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || { printf 'runtime commit is not pushed\n' >&2; exit 2; }
git -C "$ROOT" cat-file -e "${REVIEWED_CONTROLLER_PARENT_COMMIT}^{commit}" \
  && git -C "$ROOT" merge-base --is-ancestor \
    "$REVIEWED_CONTROLLER_PARENT_COMMIT" "$COMMIT" || {
  printf 'reviewed controller parent is absent or not an ancestor\n' >&2
  exit 2
}
assert_exclusive_resources || exit 5

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO/manifest.json" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$SOURCE_SHA256" \
  --expected-planner-commit "$PLANNER_COMMIT" \
  --expected-count 79 --minimum-candidates 70 --cases "$CASES" \
  --output "$TEMP_ADMISSION" >/dev/null

IDENTITY_ARGS=(
  source_manifest "$SOURCE_MANIFEST"
  portfolio_manifest "$PORTFOLIO/manifest.json"
  case_a_plan "$PORTFOLIO/$CASE_A_FILE"
  lqr_gains "$GAINS"
  robot_usd "$ROBOT_USD"
  playback "$PLAYBACK"
  tracking_controller "$TRACKING"
  riser_control "$RISER_CONTROL"
  recovery_evidence "$RECOVERY_EVIDENCE"
  balance_controller "$METRICS"
  playback_loader "$LOADER"
  wrapper "$RUNNER"
  summarizer "$SUMMARIZER"
  validator "$VALIDATOR"
)
if [[ -n "$CASE_B" ]]; then
  IDENTITY_ARGS+=(case_b_plan "$PORTFOLIO/$CASE_B_FILE")
fi
python3 - "$TEMP_ADMISSION" "$COMMIT" "$STAMP" "$CASE_TIMEOUT_SECONDS" \
  "$TRACKING_PROFILE" "$ENABLE_CAMERA_ERROR_RECOVERY" \
  "$CAMERA_RECOVERY_ERROR_START_M" "$CAMERA_RECOVERY_ERROR_FULL_M" \
  "$MINIMUM_CAMERA_RECOVERY_SCALE" "$USE_ROOT_VELOCITY_OUTER_FEEDBACK" \
  "$REQUIRE_ZERO_PROGRESS_HOLD" "$TRACKING_MINIMUM_PROGRESS_SCALE" \
  "$REQUIRE_RECOVERY_VELOCITY_CAP" "$TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS" \
  "$REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT" \
  "$REQUIRE_COMMANDED_BASE_PROGRESS_ERROR" \
  "$REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET" \
  "$VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS" \
  "$REVIEWED_CONTROLLER_PARENT_COMMIT" \
  "${IDENTITY_ARGS[@]}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["runtime_commit"] = payload["upstream_commit"] = sys.argv[2]
payload["namespace"] = sys.argv[3]
payload["case_timeout_seconds"] = int(sys.argv[4])
payload["tracking_profile"] = sys.argv[5]
payload["camera_recovery_governor_enabled"] = bool(int(sys.argv[6]))
payload["camera_recovery_error_range_m"] = [float(sys.argv[7]), float(sys.argv[8])]
payload["minimum_camera_recovery_scale"] = float(sys.argv[9])
payload["root_velocity_outer_feedback_enabled"] = bool(int(sys.argv[10]))
payload["zero_progress_hold_required"] = bool(int(sys.argv[11]))
payload["minimum_progress_scale"] = (
    float(sys.argv[12]) if sys.argv[12] else 0.1
)
payload["recovery_velocity_cap_required"] = bool(int(sys.argv[13]))
payload["maximum_linear_velocity_mps"] = (
    float(sys.argv[14]) if sys.argv[14] else 0.4
)
payload["total_pitch_reference_limit_required"] = bool(int(sys.argv[15]))
payload["total_pitch_reference_limit_rad"] = math.radians(6.0)
payload["commanded_base_progress_error_required"] = bool(int(sys.argv[16]))
payload["phase_governor_contract"] = (
    "commanded_base_and_camera_error_continuous_phase_scale_v1"
    if payload["commanded_base_progress_error_required"]
    else "position_error_continuous_phase_scale_v1"
)
payload["progress_base_error_source"] = (
    "lever_compensated_commanded_base_target"
    if payload["commanded_base_progress_error_required"]
    else "nominal_base_target"
)
payload["maximum_commanded_base_progress_error_delta_m"] = 0.05
payload["opposing_vx_integral_deficit_reset_required"] = bool(int(sys.argv[17]))
payload["vx_integral_reset_reference_deadband_mps"] = float(sys.argv[18])
payload["longitudinal_authority_telemetry_schema"] = (
    "riser_longitudinal_authority_policy_rate_v1"
)
payload["reviewed_controller_parent_commit"] = sys.argv[19]
payload["reviewed_controller_parent_is_ancestor"] = True
payload["camera_recovery_governor_contract"] = (
    "saturated_camera_error_continuous_phase_cap_v1"
)
args = sys.argv[20:]
payload["runtime_identities"] = {
    args[index]: {
        "path": str(Path(args[index + 1]).resolve()),
        "sha256": hashlib.sha256(Path(args[index + 1]).read_bytes()).hexdigest(),
    }
    for index in range(0, len(args), 2)
}
payload["camera_lever_arm_compensation_contract"] = "measured_camera_to_base_xy_offset_v1"
payload["camera_lever_arm_compensation_gain"] = 1.0
payload["maximum_camera_lever_arm_correction_m"] = 0.05
payload["runtime_authorized"] = payload["passed"] is True
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

mkdir -p "$OUTPUT/gates" "$OUTPUT/logs"
mv "$TEMP_ADMISSION" "$OUTPUT/admission.json"

CAMERA_RECOVERY_ARGS=()
if [[ "$ENABLE_CAMERA_ERROR_RECOVERY" == 1 ]]; then
  CAMERA_RECOVERY_ARGS+=(
    --enable-camera-error-recovery-governor
    --camera-recovery-error-start-m "$CAMERA_RECOVERY_ERROR_START_M"
    --camera-recovery-error-full-m "$CAMERA_RECOVERY_ERROR_FULL_M"
    --minimum-camera-recovery-scale "$MINIMUM_CAMERA_RECOVERY_SCALE"
  )
fi
SUMMARY_RECOVERY_ARGS=()
if [[ "$ENABLE_CAMERA_ERROR_RECOVERY" == 1 ]]; then
  SUMMARY_RECOVERY_ARGS+=(--require-camera-error-recovery-governor)
fi
ROOT_VELOCITY_ARGS=()
if [[ "$USE_ROOT_VELOCITY_OUTER_FEEDBACK" == 1 ]]; then
  ROOT_VELOCITY_ARGS+=(--use-root-velocity-outer-feedback)
fi
PROGRESS_HOLD_ARGS=()
SUMMARY_HOLD_ARGS=()
if [[ "$REQUIRE_ZERO_PROGRESS_HOLD" == 1 ]]; then
  PROGRESS_HOLD_ARGS+=(
    --tracking-minimum-progress-scale "$TRACKING_MINIMUM_PROGRESS_SCALE"
  )
  SUMMARY_HOLD_ARGS+=(--require-zero-progress-hold)
fi
VELOCITY_CAP_ARGS=()
SUMMARY_VELOCITY_CAP_ARGS=()
if [[ "$REQUIRE_RECOVERY_VELOCITY_CAP" == 1 ]]; then
  VELOCITY_CAP_ARGS+=(
    --tracking-maximum-linear-velocity-mps "$TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS"
  )
  SUMMARY_VELOCITY_CAP_ARGS+=(
    --require-recovery-velocity-cap
    --expected-maximum-linear-velocity-mps "$TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS"
  )
fi
TOTAL_PITCH_REFERENCE_ARGS=()
SUMMARY_TOTAL_PITCH_REFERENCE_ARGS=()
if [[ "$REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT" == 1 ]]; then
  TOTAL_PITCH_REFERENCE_ARGS+=(--limit-total-pitch-reference)
  SUMMARY_TOTAL_PITCH_REFERENCE_ARGS+=(--require-total-pitch-reference-limit)
fi
COMMANDED_BASE_PROGRESS_ARGS=()
SUMMARY_COMMANDED_BASE_PROGRESS_ARGS=()
if [[ "$REQUIRE_COMMANDED_BASE_PROGRESS_ERROR" == 1 ]]; then
  COMMANDED_BASE_PROGRESS_ARGS+=(--use-commanded-base-progress-error)
  SUMMARY_COMMANDED_BASE_PROGRESS_ARGS+=(
    --require-commanded-base-progress-error
  )
fi
OPPOSING_VX_INTEGRAL_RESET_ARGS=()
SUMMARY_OPPOSING_VX_INTEGRAL_RESET_ARGS=()
if [[ "$REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET" == 1 ]]; then
  OPPOSING_VX_INTEGRAL_RESET_ARGS+=(
    --reset-opposing-vx-integral-on-directional-deficit
    --vx-integral-reset-reference-deadband-mps \
      "$VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS"
  )
  SUMMARY_OPPOSING_VX_INTEGRAL_RESET_ARGS+=(
    --require-opposing-vx-integral-deficit-reset
    --expected-vx-integral-reset-reference-deadband-mps \
      "$VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS"
  )
fi

for CASE in "${CASE_LIST[@]}"; do
  assert_exclusive_resources || exit 5
  STATUS=0
  timeout --signal=TERM --kill-after=30s "$CASE_TIMEOUT_SECONDS" \
    "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" --plan-dir "$PORTFOLIO_WIN" \
    --plan-filename-template 'case_{case:04d}_smoothed_riser_plan_v1.npz' \
    --cases "$CASE" --controller-wz-kp "$CONTROLLER_WZ_KP" \
    --maximum-duration-scale "$MAXIMUM_DURATION_SCALE" \
    --enable-camera-lever-arm-compensation \
    --camera-lever-arm-compensation-gain "$CAMERA_LEVER_ARM_GAIN" \
    --maximum-camera-lever-arm-correction-m "$MAXIMUM_CAMERA_LEVER_ARM_CORRECTION_M" \
    "${ROOT_VELOCITY_ARGS[@]}" \
    "${PROGRESS_HOLD_ARGS[@]}" \
    "${VELOCITY_CAP_ARGS[@]}" \
    "${TOTAL_PITCH_REFERENCE_ARGS[@]}" \
    "${COMMANDED_BASE_PROGRESS_ARGS[@]}" \
    "${OPPOSING_VX_INTEGRAL_RESET_ARGS[@]}" \
    "${CAMERA_RECOVERY_ARGS[@]}" \
    --output "$OUTPUT_WIN\gates\case_$(printf '%04d' "$CASE").json" --headless \
    >"$OUTPUT/logs/case_$(printf '%04d' "$CASE").log" 2>&1 || STATUS=$?
  printf '%s\n' "$STATUS" >"$OUTPUT/logs/case_$(printf '%04d' "$CASE").exit_code"
  wait_for_gpu_release || exit 5
  if [[ ! -s "$OUTPUT/gates/case_$(printf '%04d' "$CASE").json" ]] \
    || ! case_gate_passed "$OUTPUT/gates/case_$(printf '%04d' "$CASE").json" \
      "$CASE" "$TRACKING_PROFILE" "$ENABLE_CAMERA_ERROR_RECOVERY" \
      "$REQUIRE_INITIALIZATION_PREROLL" \
      "$USE_ROOT_VELOCITY_OUTER_FEEDBACK" \
      "$REQUIRE_ZERO_PROGRESS_HOLD" \
      "$REQUIRE_RECOVERY_VELOCITY_CAP" \
      "$TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS" \
      "$REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT" \
      "$REQUIRE_COMMANDED_BASE_PROGRESS_ERROR" \
      "$REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET" \
      "$VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS"; then
    python3 "$SUMMARIZER" --root "$OUTPUT" --git-commit "$COMMIT" --cases "$CASES" \
      --expected-tracking-profile "$TRACKING_PROFILE" \
      --require-camera-lever-arm-compensation "${SUMMARY_RECOVERY_ARGS[@]}" \
      "${SUMMARY_HOLD_ARGS[@]}" \
      "${SUMMARY_VELOCITY_CAP_ARGS[@]}" \
      "${SUMMARY_TOTAL_PITCH_REFERENCE_ARGS[@]}" \
      "${SUMMARY_COMMANDED_BASE_PROGRESS_ARGS[@]}" \
      "${SUMMARY_OPPOSING_VX_INTEGRAL_RESET_ARGS[@]}" \
      --output "$OUTPUT/summary.json" >/dev/null
    printf 'camera lever-arm Gate C stopped on case %s\n' "$CASE" >&2
    exit 4
  fi
done

python3 "$SUMMARIZER" --root "$OUTPUT" --git-commit "$COMMIT" --cases "$CASES" \
  --expected-tracking-profile "$TRACKING_PROFILE" \
  --require-camera-lever-arm-compensation "${SUMMARY_RECOVERY_ARGS[@]}" \
  "${SUMMARY_HOLD_ARGS[@]}" \
  "${SUMMARY_VELOCITY_CAP_ARGS[@]}" \
  "${SUMMARY_TOTAL_PITCH_REFERENCE_ARGS[@]}" \
  "${SUMMARY_COMMANDED_BASE_PROGRESS_ARGS[@]}" \
  "${SUMMARY_OPPOSING_VX_INTEGRAL_RESET_ARGS[@]}" \
  --output "$OUTPUT/summary.json" >/dev/null
python3 - "$OUTPUT/summary.json" "$CASES" \
  "$REQUIRE_INITIALIZATION_PREROLL" \
  "$USE_ROOT_VELOCITY_OUTER_FEEDBACK" \
  "$REQUIRE_ZERO_PROGRESS_HOLD" \
  "$REQUIRE_RECOVERY_VELOCITY_CAP" \
  "$TRACKING_MAXIMUM_LINEAR_VELOCITY_MPS" \
  "$REQUIRE_TOTAL_PITCH_REFERENCE_LIMIT" \
  "$REQUIRE_COMMANDED_BASE_PROGRESS_ERROR" \
  "$REQUIRE_OPPOSING_VX_INTEGRAL_DEFICIT_RESET" \
  "$VX_INTEGRAL_RESET_REFERENCE_DEADBAND_MPS" <<'PY'
import json
import math
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
expected_cases = [int(value) for value in sys.argv[2].split(",")]
require_initialization = bool(int(sys.argv[3]))
require_root_velocity_feedback = bool(int(sys.argv[4]))
require_zero_progress_hold = bool(int(sys.argv[5]))
require_recovery_velocity_cap = bool(int(sys.argv[6]))
expected_maximum_linear_velocity_mps = (
    float(sys.argv[7]) if require_recovery_velocity_cap else None
)
require_total_pitch_reference_limit = bool(int(sys.argv[8]))
require_commanded_base_progress_error = bool(int(sys.argv[9]))
require_opposing_vx_integral_deficit_reset = bool(int(sys.argv[10]))
expected_vx_integral_reset_deadband_mps = float(sys.argv[11])
gate_rows = summary.get("gate_rows", [])
initialization_ok = not require_initialization or (
    len(gate_rows) == 1
    and gate_rows[0].get("initialization_evidence_passed") is True
    and gate_rows[0].get("initialization_duration_s") == 2.0
    and gate_rows[0].get("initialization_steps") == 400
    and gate_rows[0].get("initialization_source_metric_samples") == 0
    and gate_rows[0].get("initialization_residual_label_samples") == 0
)
root_velocity_feedback_ok = not require_root_velocity_feedback or (
    len(gate_rows) == 1
    and gate_rows[0].get("outer_velocity_feedback_source") == "root_link_vx"
    and gate_rows[0].get("velocity_feedback_evidence_passed") is True
)
zero_progress_hold_ok = not require_zero_progress_hold or (
    len(gate_rows) == 1
    and gate_rows[0].get("zero_progress_hold_evidence_passed") is True
    and gate_rows[0].get("minimum_progress_scale") == 0.0
    and gate_rows[0].get("progress_scale_min") == 0.0
    and gate_rows[0].get("progress_hold_step_count", 0) > 0
    and gate_rows[0].get("progress_hold_segment_count", 0) > 0
    and gate_rows[0].get("outer_velocity_feedback_source") == "wheel_derived_vx"
)
recovery_velocity_cap_ok = not require_recovery_velocity_cap or (
    len(gate_rows) == 1
    and gate_rows[0].get("recovery_velocity_cap_evidence_passed") is True
    and gate_rows[0].get("maximum_linear_velocity_mps")
    == expected_maximum_linear_velocity_mps
)
total_pitch_reference_limit_ok = not require_total_pitch_reference_limit or (
    len(gate_rows) == 1
    and gate_rows[0].get("total_pitch_reference_limit_evidence_passed") is True
    and gate_rows[0].get("total_pitch_reference_limit_enabled") is True
    and gate_rows[0].get("total_pitch_reference_limit_rad") == math.radians(6.0)
)
commanded_base_progress_error_ok = (
    not require_commanded_base_progress_error
    or (
        len(gate_rows) == 1
        and summary.get("commanded_base_progress_error_required") is True
        and summary.get("expected_phase_governor_contract")
        == "commanded_base_and_camera_error_continuous_phase_scale_v1"
        and summary.get("expected_progress_base_error_source")
        == "lever_compensated_commanded_base_target"
        and summary.get("commanded_base_progress_error_evidence_passed") is True
        and gate_rows[0].get(
            "commanded_base_progress_error_evidence_passed"
        ) is True
        and gate_rows[0].get("commanded_base_progress_error_enabled") is True
        and gate_rows[0].get("progress_base_error_source")
        == "lever_compensated_commanded_base_target"
        and gate_rows[0].get(
            "progress_base_error_telemetry_sample_count"
        )
        == gate_rows[0].get("completed_steps")
        and gate_rows[0].get(
            "progress_base_error_selected_source_matches"
        ) is True
        and gate_rows[0].get(
            "progress_base_error_command_delta_bounded"
        ) is True
        and isinstance(
            gate_rows[0].get(
                "selected_vs_nominal_base_progress_error_abs_max_delta_m"
            ),
            (int, float),
        )
        and 0.0
        <= gate_rows[0][
            "selected_vs_nominal_base_progress_error_abs_max_delta_m"
        ]
        <= 0.05 + 1e-9
    )
)
longitudinal_authority_ok = (
    not require_opposing_vx_integral_deficit_reset
    or (
        len(gate_rows) == 1
        and summary.get("opposing_vx_integral_deficit_reset_required") is True
        and summary.get("expected_vx_integral_reset_reference_deadband_mps")
        == expected_vx_integral_reset_deadband_mps
        and summary.get("longitudinal_authority_evidence_passed") is True
        and gate_rows[0].get("longitudinal_authority_evidence_passed") is True
        and gate_rows[0].get(
            "opposing_vx_integral_deficit_reset_enabled"
        ) is True
        and gate_rows[0].get("vx_integral_reset_reference_deadband_mps")
        == expected_vx_integral_reset_deadband_mps
        and gate_rows[0].get("longitudinal_authority_telemetry", {}).get(
            "integral_reset_count", 0
        ) > 0
    )
)
ok = (
    summary.get("requested_cases") == expected_cases
    and summary.get("dynamically_passed_cases") == expected_cases
    and summary.get("first_dynamic_reject") is None
    and summary.get("dynamic_quality_passed") is True
    and summary.get("thermal_admission_passed") is True
    and summary.get("controller_evidence_passed") is True
    and summary.get("runtime_contract_passed") is True
    and summary.get("residual_capture_started") is False
    and summary.get("bc_started") is False
    and summary.get("ppo_started") is False
    and summary.get("valid_for_final_gate_c") is True
    and summary.get("valid_for_training") is False
    and initialization_ok
    and root_velocity_feedback_ok
    and zero_progress_hold_ok
    and recovery_velocity_cap_ok
    and total_pitch_reference_limit_ok
    and commanded_base_progress_error_ok
    and longitudinal_authority_ok
)
raise SystemExit(0 if ok else 6)
PY
printf 'camera lever-arm Gate C closed: %s\n' "$OUTPUT"
