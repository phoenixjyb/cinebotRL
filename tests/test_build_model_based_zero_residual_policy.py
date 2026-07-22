import copy

import pytest


torch = pytest.importorskip("torch")

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
    RiserResidualPolicy,
)
from scripts.two_wheel_balance.build_model_based_zero_residual_policy import (  # noqa: E402
    build_policy,
    translate_wsl_path_for_windows,
)


def _source_checkpoint():
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        masked_observation_indices=PREVIOUS_ACTION_INDICES,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(torch.randn_like(parameter))
    return {
        "schema": "cinebotrl_two_wheel_riser_residual_policy_v2",
        "policy_architecture": MASKED_PREVIOUS_ACTION_POLICY_ARCHITECTURE,
        "model_state_dict": policy.state_dict(),
        "state_hidden_sizes": [16],
        "lookahead_hidden_sizes": [8],
        "fusion_hidden_sizes": [16],
        "observation_names": OBSERVATION_NAMES,
        "action_names": ACTION_NAMES,
        "masked_observation_indices": list(PREVIOUS_ACTION_INDICES),
        "previous_action_observation_gains": [1.0, 1.0, 1.0],
    }


def _audit():
    return {
        "passed": True,
        "failed_dynamic_gate": "position_p95_bounded",
        "architecture_audit": {
            "required_contract_satisfied": False,
            "checkpoint_classification": "planner_imitation_bc_initialization_only",
        },
        "decision": {
            "bc_retraining_authorized": False,
            "ppo_authorized": False,
        },
    }


def test_builds_exact_zero_head_over_transferred_encoder() -> None:
    source = _source_checkpoint()
    target, checks = build_policy(source, _audit())
    assert all(checks.values())
    for name, value in target.state_dict().items():
        if name.startswith("action_head."):
            torch.testing.assert_close(value, torch.zeros_like(value))
        else:
            torch.testing.assert_close(value, source["model_state_dict"][name])
    with torch.inference_mode():
        output = target(torch.randn(64, len(OBSERVATION_NAMES)))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_rejects_wrong_source_architecture() -> None:
    source = _source_checkpoint()
    source["policy_architecture"] = "direct_wheel_effort_policy"
    with pytest.raises(ValueError, match="input contract failed"):
        build_policy(source, _audit())


def test_rejects_failure_audit_that_authorizes_training() -> None:
    audit = copy.deepcopy(_audit())
    audit["decision"]["ppo_authorized"] = True
    with pytest.raises(ValueError, match="input contract failed"):
        build_policy(_source_checkpoint(), audit)


def test_rejects_observation_contract_drift() -> None:
    source = _source_checkpoint()
    source["observation_names"] = source["observation_names"][:-1]
    with pytest.raises(ValueError, match="input contract failed"):
        build_policy(source, _audit())


def test_translates_wsl_worktree_git_dir_for_windows() -> None:
    assert translate_wsl_path_for_windows(
        "/mnt/g/wSpace/cinebotRL/.git/worktrees/cinebotRL-two-wheel-riser"
    ) == (
        "G:\\wSpace\\cinebotRL\\.git\\worktrees\\cinebotRL-two-wheel-riser"
    )


def test_rejects_non_wsl_git_dir_translation() -> None:
    with pytest.raises(ValueError, match="unsupported WSL Git metadata path"):
        translate_wsl_path_for_windows("/Users/yanbo/Projects/cinebotRL/.git")
