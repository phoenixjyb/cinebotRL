import json
from pathlib import Path

from rl_platform.tasks.two_wheel_balance.runtime_heartbeat import (
    SCHEMA,
    write_runtime_heartbeat,
)


def test_runtime_heartbeat_atomically_replaces_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "runtime_heartbeat.json"
    write_runtime_heartbeat(
        path,
        {
            "case": 78,
            "stage": "execution",
            "completed_steps": 2000,
            "phase_time_s": 8.5,
        },
    )
    write_runtime_heartbeat(
        path,
        {
            "case": 78,
            "stage": "execution",
            "completed_steps": 4000,
            "phase_time_s": 16.25,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["completed_steps"] == 4000
    assert payload["phase_time_s"] == 16.25
    assert payload["dataset_created"] is False
    assert payload["valid_for_training"] is False
    assert not list(tmp_path.glob(".*.tmp"))


def test_runtime_heartbeat_forces_non_training_status(tmp_path: Path) -> None:
    path = tmp_path / "runtime_heartbeat.json"
    write_runtime_heartbeat(
        path,
        {
            "dataset_created": True,
            "valid_for_training": True,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_created"] is False
    assert payload["valid_for_training"] is False

