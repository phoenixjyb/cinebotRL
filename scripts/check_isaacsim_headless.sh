#!/usr/bin/env bash
# Basic health check for headless Isaac Sim install under WSL2.

set -euo pipefail

ENV_FILE="${ISAACSIM_ENV_FILE:-$HOME/.config/isaac-sim-wsl.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "[WARN] Isaac Sim env file not found at $ENV_FILE" >&2
fi

if [[ -z "${ISAACSIM_ROOT:-}" || ! -d "$ISAACSIM_ROOT" ]]; then
  echo "[ERROR] ISAACSIM_ROOT not set or directory missing." >&2
  exit 1
fi

PYTHON_SH="${ISAACSIM_PYTHON:-$ISAACSIM_ROOT/python.sh}"
if [[ ! -x "$PYTHON_SH" ]]; then
  echo "[ERROR] Isaac Sim python launcher missing: $PYTHON_SH" >&2
  exit 1
fi

echo "[INFO] Using ISAACSIM_ROOT=$ISAACSIM_ROOT" >&2

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.total --format=csv,noheader
else
  echo "[WARN] nvidia-smi not available; GPU telemetry skipped." >&2
fi

TMP_SCRIPT="$(mktemp --suffix .py)"
cat <<'PY' >"$TMP_SCRIPT"
from omni.isaac.kit import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import omni
from omni.isaac.core import World

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.reset()
for _ in range(10):
    world.step(render=False)

print("[CHECK] Isaac Sim headless world stepped successfully")

app.close()
PY

trap 'rm -f "$TMP_SCRIPT"' EXIT

"$PYTHON_SH" "$TMP_SCRIPT"

echo "[INFO] Isaac Sim headless smoke test completed." >&2
