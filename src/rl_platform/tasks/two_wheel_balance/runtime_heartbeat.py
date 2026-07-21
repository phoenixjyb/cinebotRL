"""Atomic, non-training runtime progress evidence for bounded playbacks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA = "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"


def write_runtime_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one lightweight progress snapshot."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        **payload,
        "schema": SCHEMA,
        "emitted_epoch_s": time.time(),
        "dataset_created": False,
        "valid_for_training": False,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
