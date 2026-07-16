#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
STAMP="${RISER_GATE_STAMP:-20260716_rs4_proxy_rerun}"
ARTIFACTS="$WIN_ROOT\\artifacts\\two_wheel_riser\\$STAMP"
ARTIFACTS_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
URDF="$WIN_ROOT\\assets_own\\recomoProto2_two_wheel_riser\\recomoProto2_two_wheel_riser.urdf"
USD="$WIN_ROOT\\assets_own\\recomoProto2_two_wheel_riser\\recomoProto2_two_wheel_riser.usd"
GAINS="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json"
PLAN_DIR="$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260716_riser_render_inputs"
D3D12_EXPERIENCE="${ISAAC_D3D12_EXPERIENCE:-G:\\isaaclab\\apps\\isaaclab.python.headless.rendering.d3d12.kit}"
D3D12_EXPERIENCE_WSL="${ISAAC_D3D12_EXPERIENCE_WSL:-/mnt/g/isaaclab/apps/isaaclab.python.headless.rendering.d3d12.kit}"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -f "$ROOT/scripts/convert_urdf_to_usd.py" ]] || {
  printf 'missing converter in remote worktree\n' >&2
  exit 2
}
[[ -f "$D3D12_EXPERIENCE_WSL" ]] || {
  printf 'missing D3D12 render experience: %s\n' "$D3D12_EXPERIENCE_WSL" >&2
  exit 2
}

require_json_pass() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"gate failed: {path}")
print(f"gate passed: {path}")
PY
}

cd "$ROOT"
"$PY" -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\build_riser_urdf.py"
"$PY" -X utf8 -c "import sys,pytest; sys.path[:0]=[r'$WIN_ROOT',r'$WIN_ROOT\\src']; raise SystemExit(pytest.main(['-q',r'$WIN_ROOT\\tests\\test_two_wheel_riser_rs4_attitude.py',r'$WIN_ROOT\\tests\\test_two_wheel_riser_reference.py',r'$WIN_ROOT\\tests\\test_two_wheel_riser_kinematics.py',r'$WIN_ROOT\\tests\\test_two_wheel_riser_playback.py',r'$WIN_ROOT\\tests\\test_build_riser_corrected_stage.py',r'$WIN_ROOT\\tests\\test_two_wheel_riser_control.py',r'$WIN_ROOT\\tests\\test_two_wheel_riser_asset.py','-k','not meshes_resolve_locally']))"

"$PY" -u -X utf8 "$WIN_ROOT\\scripts\\convert_urdf_to_usd.py" \
  --urdf "$URDF" --usd "$USD" --mesh-scale 1.0 \
  --default-drive-type none --headless

"$PY" -u -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_asset.py" \
  --urdf "$URDF" --usd "$USD" \
  --output "$ARTIFACTS\\gate0_asset.json" --headless
require_json_pass "$ARTIFACTS_WSL/gate0_asset.json"

"$PY" -u -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_static_heights.py" \
  --gains "$GAINS" --steps 2000 \
  --output "$ARTIFACTS\\gate1_static_heights.json" --headless
require_json_pass "$ARTIFACTS_WSL/gate1_static_heights.json"

"$PY" -u -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_dynamic.py" \
  --gains "$GAINS" --steps 10000 \
  --output "$ARTIFACTS\\gate2_riser_dynamic.json" --headless
require_json_pass "$ARTIFACTS_WSL/gate2_riser_dynamic.json"

"$PY" -u -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py" \
  --gains "$GAINS" --plan-dir "$PLAN_DIR" --cases 1,31,73 \
  --output "$ARTIFACTS\\gate3_representative_playback.json" --headless
require_json_pass "$ARTIFACTS_WSL/gate3_representative_playback.json"

for case in 1 31 73; do
  padded="$(printf '%04d' "$case")"
  # Kit can retain the previous GPU context briefly after a headless process.
  sleep 5
  "$PY" -u -X utf8 "$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py" \
    --gains "$GAINS" --plan-dir "$PLAN_DIR" --cases "$case" \
    --video-dir "$ARTIFACTS\\videos\\case_$padded" \
    --video-fps 200 \
    --output "$ARTIFACTS\\gate3_render_case_$padded.json" \
    --headless --enable_cameras --experience "$D3D12_EXPERIENCE"
  require_json_pass "$ARTIFACTS_WSL/gate3_render_case_$padded.json"
done

printf 'riser Gate 0-3 and representative renders passed; artifacts: %s\n' "$ARTIFACTS"
