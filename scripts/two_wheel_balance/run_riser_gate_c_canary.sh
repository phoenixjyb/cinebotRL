#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
PORTFOLIO_STAMP="${RISER_GATE_C_PORTFOLIO_STAMP:-20260717_exact_source_all79_portfolio_v4_threshold71}"
PORTFOLIO_WSL="$ROOT/artifacts/two_wheel_riser/$PORTFOLIO_STAMP"
PORTFOLIO_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$PORTFOLIO_STAMP"
EXPECTED_MANIFEST_SHA256="${RISER_GATE_C_MANIFEST_SHA256:-851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a}"
EXPECTED_SOURCE_SHA256="${RISER_GATE_C_SOURCE_SHA256:-f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d}"
CASES="${RISER_GATE_C_CASES:-1,52,74,77}"
STAMP="${RISER_GATE_C_STAMP:-20260717_gate_c_canary_v2_timing_resealed}"
OUTPUT_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
OUTPUT_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$STAMP"
GAINS_WIN="${RISER_GAINS_WIN:-$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json}"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_gate_c_portfolio.py"
PLAYBACK_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
SUMMARIZER="$ROOT/scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -s "$PORTFOLIO_WSL/manifest.json" ]] || { printf 'missing portfolio manifest\n' >&2; exit 2; }
[[ ! -e "$OUTPUT_WSL" ]] || { printf 'refusing existing Gate C namespace: %s\n' "$OUTPUT_WSL" >&2; exit 2; }
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make Gate C provenance ambiguous\n' >&2
  exit 2
}
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')"
[[ "$COMMIT" == "$UPSTREAM" ]] || {
  printf 'Gate C commit is not pushed to upstream: %s != %s\n' "$COMMIT" "$UPSTREAM" >&2
  exit 2
}

TEMP_ADMISSION="$(mktemp)"
trap 'rm -f "$TEMP_ADMISSION"' EXIT
python3 "$VALIDATOR" \
  --manifest "$PORTFOLIO_WSL/manifest.json" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$EXPECTED_SOURCE_SHA256" \
  --expected-count 79 \
  --minimum-candidates 70 \
  --cases "$CASES" \
  --output "$TEMP_ADMISSION" >/dev/null
mkdir -p "$OUTPUT_WSL/gates" "$OUTPUT_WSL/logs"
mv "$TEMP_ADMISSION" "$OUTPUT_WSL/admission.json"

IFS=',' read -r -a case_list <<< "$CASES"
for raw_case in "${case_list[@]}"; do
  case_number="$((10#$raw_case))"
  padded="$(printf '%04d' "$case_number")"
  printf 'Gate C canary case %s\n' "$padded"
  if ! "$PY" -u -X utf8 "$PLAYBACK_WIN" \
    --gains "$GAINS_WIN" \
    --plan-dir "$PORTFOLIO_WIN" \
    --plan-filename-template 'case_{case:04d}_exact_source_riser_playback_v1.npz' \
    --cases "$case_number" \
    --output "$OUTPUT_WIN\\gates\\case_$padded.json" \
    --headless >"$OUTPUT_WSL/logs/case_$padded.log" 2>&1; then
      python3 "$SUMMARIZER" \
        --root "$OUTPUT_WSL" \
        --git-commit "$COMMIT" \
        --cases "$CASES" \
        --output "$OUTPUT_WSL/summary.json" >/dev/null
      tail -n 100 "$OUTPUT_WSL/logs/case_$padded.log" >&2
      printf 'Gate C stopped on first dynamic reject: case %s\n' "$padded" >&2
      exit 4
  fi
  python3 - "$OUTPUT_WSL/gates/case_$padded.json" "$case_number" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("passed") is True
    and payload.get("dynamic_quality_passed") is True
    and payload.get("training_started") is False
    and payload.get("ppo_authorized") is False
    and payload.get("trajectory_command_source") == "deterministic_teacher"
    and payload.get("residual_policy") is None
    and payload.get("cases") == [case]
    and payload.get("passed_case_count") == 1
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("passed") is True
    and payload["results"][0].get("dynamic_quality_passed") is True
    and payload["results"][0].get("executed_residual_dataset") is None
)
raise SystemExit(0 if valid else 1)
PY
done

python3 "$SUMMARIZER" \
  --root "$OUTPUT_WSL" \
  --git-commit "$COMMIT" \
  --cases "$CASES" \
  --output "$OUTPUT_WSL/summary.json"

printf 'Gate C canary passed without labels: %s\n' "$OUTPUT_WSL"
