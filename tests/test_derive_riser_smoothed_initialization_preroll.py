from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/two_wheel_balance"))

from derive_riser_smoothed_initialization_preroll import (
    initialization_state_and_metrics,
)


def arrays() -> dict[str, np.ndarray]:
    return {
        "base_xy_yaw": np.array([[0.0, 0.0, np.pi], [0.1, 0.0, np.pi]]),
        "riser_q": np.array([0.5, 0.51]),
        "proxy_gimbal_q": np.array([[0.1, 0.2, 3.0], [0.11, 0.21, 3.01]]),
        "feedforward_v_wz": np.array([[-0.08, -0.2]]),
        "feedforward_riser_velocity": np.array([0.04]),
        "feedforward_proxy_velocity": np.array([[0.1, 0.2, -0.3]]),
    }


def test_preroll_starts_near_rest_and_joins_first_execution_derivative() -> None:
    time_s, state, metrics = initialization_state_and_metrics(arrays(), 2.0, 200.0)
    assert len(time_s) == 401
    np.testing.assert_allclose(
        state[-1], [0.0, 0.0, np.pi, 0.5, 0.1, 0.2, 3.0], atol=1e-12
    )
    assert metrics["checks"]["starts_near_rest"] is True
    assert metrics["checks"]["terminal_rate_matches_execution"] is True
    assert metrics["checks"]["base_linear_velocity_bounded"] is True
    assert metrics["checks"]["proxy_rate_bounded"] is True
    json.dumps(metrics)


def test_preroll_rejects_overspeed_terminal_derivative() -> None:
    values = arrays()
    values["feedforward_v_wz"][0, 0] = 0.6
    with pytest.raises(ValueError, match="kinematic gate failed"):
        initialization_state_and_metrics(values, 2.0, 200.0)


def test_preroll_rejects_short_or_low_rate_clock() -> None:
    with pytest.raises(ValueError, match="too short"):
        initialization_state_and_metrics(arrays(), 0.5, 200.0)
    with pytest.raises(ValueError, match="policy rate"):
        initialization_state_and_metrics(arrays(), 2.0, 20.0)


def test_portfolio_composer_admits_preroll_replacement_schema() -> None:
    source = (
        ROOT
        / "scripts/two_wheel_balance/compose_riser_smoothed_plan_portfolio.py"
    ).read_text(encoding="utf-8")
    assert "cinebotrl_two_wheel_riser_initialization_preroll_derivation_v1" in source
    derivation = (
        ROOT
        / "scripts/two_wheel_balance/derive_riser_smoothed_initialization_preroll.py"
    ).read_text(encoding="utf-8")
    assert '"parent_plan_sha256": args.expected_parent_plan_sha256' in derivation
    assert '"item": replacement_item' in derivation
    assert '"source_manifest_sha256"' in derivation


def _load_composer():
    script = (
        ROOT
        / "scripts/two_wheel_balance/compose_riser_smoothed_plan_portfolio.py"
    )
    spec = importlib.util.spec_from_file_location("preroll_composer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def healthy_replacement_contract() -> tuple[dict[str, object], dict[str, object]]:
    checks = {
        "parent_initialization_was_empty": True,
        "initialization_separate_clock": True,
        "initialization_unscored": True,
        "initialization_kinematic_gate_passed": True,
    }
    item = {
        "checks": checks,
        "initialization_preroll": {
            "source_clock_advanced": False,
            "execution_clock_advanced": False,
            "scored_as_source_tracking": False,
        },
    }
    replacement = {
        "schema": "cinebotrl_two_wheel_riser_initialization_preroll_derivation_v1",
        "source_and_scored_arrays_immutable": {"source_time_s": True},
        "source_clock_advanced": False,
        "execution_clock_advanced": False,
        "initialization_metrics": {
            "duration_s": 2.0,
            "sample_count": 401,
            "checks": {"state_finite": True},
        },
    }
    return replacement, item


def test_composer_accepts_isolated_preroll_contract() -> None:
    replacement, item = healthy_replacement_contract()
    _load_composer()._validate_initialization_replacement(replacement, item)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload, item: payload.update(source_clock_advanced=True), "scored clock"),
        (
            lambda payload, item: payload["source_and_scored_arrays_immutable"].update(
                source_time_s=False
            ),
            "mutated source",
        ),
        (
            lambda payload, item: item["checks"].update(
                initialization_separate_empty=True
            ),
            "stale or incomplete",
        ),
        (
            lambda payload, item: item["initialization_preroll"].update(
                scored_as_source_tracking=True
            ),
            "not isolated",
        ),
    ],
)
def test_composer_rejects_invalid_preroll_contract(mutation, message: str) -> None:
    replacement, item = healthy_replacement_contract()
    mutation(replacement, item)
    with pytest.raises(ValueError, match=message):
        _load_composer()._validate_initialization_replacement(replacement, item)
