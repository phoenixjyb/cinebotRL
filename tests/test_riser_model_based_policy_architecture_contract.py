import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import (  # noqa: E402
    build_projection_aware_residual_policy,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_bc_contract import (  # noqa: E402
    DEFAULT_BC_TRAINING_CONFIG,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
)


ROOT = Path(__file__).parents[1]
CASE30_DATASET = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case30_effective_label_conversion_v1/"
    "case_0030_model_based_corrective_case_dataset_v1.npz"
)
CASE23_CAPTURE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v4/capture/"
    "case_0023_corrective_teacher_capture_v2.npz"
)
CASE23_CONVERSION_REVIEW = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_conversion_review_v1/summary.json"
)


def _load_observation_contract(path: Path) -> tuple[dict[str, object], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        shape = archive["observations"].shape
    return metadata, shape


def test_real_corrective_sources_match_the_admitted_65_to_3_contract() -> None:
    case30_metadata, case30_shape = _load_observation_contract(CASE30_DATASET)
    case23_metadata, case23_shape = _load_observation_contract(CASE23_CAPTURE)
    conversion_review = json.loads(
        CASE23_CONVERSION_REVIEW.read_text(encoding="utf-8")
    )

    assert case30_shape == (11411, len(OBSERVATION_NAMES))
    assert case23_shape == (3273, len(OBSERVATION_NAMES))
    assert conversion_review["prospective_dataset_metrics"][
        "observation_shape"
    ] == [3273, len(OBSERVATION_NAMES)]
    for metadata in (case30_metadata, case23_metadata):
        assert metadata["observation_names"] == list(OBSERVATION_NAMES)
        assert metadata["action_names"] == list(ACTION_NAMES)
        assert metadata["bc_authorized"] is False
        assert metadata["ppo_authorized"] is False
        assert metadata["training_started"] is False
    assert case30_metadata["action_scales"] == pytest.approx(
        MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist()
    )


def test_admitted_policy_layout_and_zero_initialization_are_exact() -> None:
    assert DEFAULT_BC_TRAINING_CONFIG["policy_architecture"] == (
        MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
    )
    assert DEFAULT_BC_TRAINING_CONFIG["observation_dimension"] == len(
        OBSERVATION_NAMES
    )
    assert DEFAULT_BC_TRAINING_CONFIG["base_observation_dimension"] == len(
        BASE_OBSERVATION_NAMES
    )
    assert DEFAULT_BC_TRAINING_CONFIG["lookahead_horizon_count"] == len(
        LOOKAHEAD_HORIZONS_S
    )
    assert DEFAULT_BC_TRAINING_CONFIG["lookahead_channel_count"] == len(
        LOOKAHEAD_CHANNEL_NAMES
    )
    assert DEFAULT_BC_TRAINING_CONFIG["action_dimension"] == len(ACTION_NAMES)
    assert DEFAULT_BC_TRAINING_CONFIG["zero_initialize_action_head"] is True
    model = build_projection_aware_residual_policy(
        np.zeros(len(OBSERVATION_NAMES), dtype=np.float32),
        np.ones(len(OBSERVATION_NAMES), dtype=np.float32),
    ).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    observations = torch.randn(5, len(OBSERVATION_NAMES))
    with torch.no_grad():
        eager = model(observations)
        scripted = torch.jit.script(model)(observations)
    assert parameter_count == 142019
    assert torch.count_nonzero(model.action_head.weight).item() == 0
    assert torch.count_nonzero(model.action_head.bias).item() == 0
    assert torch.count_nonzero(eager).item() == 0
    assert torch.equal(scripted, eager)
