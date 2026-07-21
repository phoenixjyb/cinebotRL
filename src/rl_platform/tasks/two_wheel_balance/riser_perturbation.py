"""Deterministic, measurement-only perturbation contract for riser playback."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


PERTURBATION_SCHEMA = (
    "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1"
)
MAXIMUM_FORCE_ABS_N = 40.0
MAXIMUM_DURATION_STEPS = 50
MAXIMUM_APPLICATION_HEIGHT_M = 1.0
AUTHORIZED_MEASUREMENT_CASE = 30
PROFILE_FIELDS = {
    "schema",
    "case",
    "start_phase_time_s",
    "duration_steps",
    "force_body_x_n",
    "application_height_m",
}


@dataclass(frozen=True)
class DeterministicWrenchPulse:
    case: int
    start_phase_time_s: float
    duration_steps: int
    force_body_x_n: float
    application_height_m: float

    def validate(self) -> None:
        checks = {
            "case": (
                isinstance(self.case, int)
                and self.case == AUTHORIZED_MEASUREMENT_CASE
            ),
            "start_phase_time_s": (
                math.isfinite(self.start_phase_time_s)
                and self.start_phase_time_s >= 0.0
            ),
            "duration_steps": (
                isinstance(self.duration_steps, int)
                and not isinstance(self.duration_steps, bool)
                and 1 <= self.duration_steps <= MAXIMUM_DURATION_STEPS
            ),
            "force_body_x_n": (
                math.isfinite(self.force_body_x_n)
                and 0.0 < abs(self.force_body_x_n) <= MAXIMUM_FORCE_ABS_N
            ),
            "application_height_m": (
                math.isfinite(self.application_height_m)
                and 0.0 <= self.application_height_m
                <= MAXIMUM_APPLICATION_HEIGHT_M
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"invalid deterministic wrench profile: {checks}")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PERTURBATION_SCHEMA,
            "case": self.case,
            "start_phase_time_s": self.start_phase_time_s,
            "duration_steps": self.duration_steps,
            "force_body_x_n": self.force_body_x_n,
            "application_height_m": self.application_height_m,
        }


def load_deterministic_wrench_profile(
    path: Path,
) -> tuple[DeterministicWrenchPulse, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing deterministic wrench profile: {path}")
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict) or set(payload) != PROFILE_FIELDS:
        fields = sorted(payload) if isinstance(payload, dict) else []
        raise ValueError(f"unexpected deterministic wrench profile fields: {fields}")
    if payload["schema"] != PERTURBATION_SCHEMA:
        raise ValueError("unexpected deterministic wrench profile schema")
    if not isinstance(payload["case"], int) or isinstance(payload["case"], bool):
        raise ValueError("deterministic wrench profile case must be an integer")
    if not isinstance(payload["duration_steps"], int) or isinstance(
        payload["duration_steps"], bool
    ):
        raise ValueError("deterministic wrench duration_steps must be an integer")
    profile = DeterministicWrenchPulse(
        case=payload["case"],
        start_phase_time_s=float(payload["start_phase_time_s"]),
        duration_steps=payload["duration_steps"],
        force_body_x_n=float(payload["force_body_x_n"]),
        application_height_m=float(payload["application_height_m"]),
    )
    profile.validate()
    return profile, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


class DeterministicWrenchPulseRuntime:
    """Trigger one phase-indexed pulse with execution-step-bounded duration."""

    def __init__(self, profile: DeterministicWrenchPulse | None):
        self.profile = profile
        self.trigger_step: int | None = None
        self.trigger_phase_time_s: float | None = None
        self.active_steps = 0
        self.release_observed = False

    def command(self, *, step: int, phase_time_s: float) -> float:
        if step < 0 or not math.isfinite(phase_time_s) or phase_time_s < 0.0:
            raise ValueError("invalid perturbation runtime clock")
        if self.profile is None:
            return 0.0
        if (
            self.trigger_step is None
            and phase_time_s + 1e-12 >= self.profile.start_phase_time_s
        ):
            self.trigger_step = step
            self.trigger_phase_time_s = phase_time_s
        active = (
            self.trigger_step is not None
            and step < self.trigger_step + self.profile.duration_steps
        )
        if active:
            self.active_steps += 1
            return self.profile.force_body_x_n
        if self.trigger_step is not None:
            self.release_observed = True
        return 0.0

    def summary(self) -> dict[str, object]:
        enabled = self.profile is not None
        expected_steps = 0 if self.profile is None else self.profile.duration_steps
        return {
            "schema": "cinebotrl_two_wheel_riser_wrench_pulse_telemetry_v1",
            "enabled": enabled,
            "profile": None if self.profile is None else self.profile.as_dict(),
            "trigger_step": self.trigger_step,
            "trigger_phase_time_s": self.trigger_phase_time_s,
            "active_step_count": self.active_steps,
            "expected_active_step_count": expected_steps,
            "triggered": self.trigger_step is not None,
            "released_after_pulse": self.release_observed,
            "applied_to_planner_commands": False,
            "applied_to_policy_actions": False,
            "dataset_created": False,
        }

    def mark_released(self) -> None:
        if self.trigger_step is not None:
            self.release_observed = True

    def contract_checks(self) -> dict[str, bool]:
        if self.profile is None:
            return {
                "perturbation_disabled": True,
                "perturbation_not_required": True,
            }
        return {
            "perturbation_enabled": True,
            "perturbation_triggered": self.trigger_step is not None,
            "perturbation_exact_duration": (
                self.active_steps == self.profile.duration_steps
            ),
            "perturbation_released": self.release_observed,
        }
