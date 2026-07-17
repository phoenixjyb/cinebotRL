#!/usr/bin/env bash
set -euo pipefail

ROOT="${RISER_ROOT:-/mnt/g/wSpace/cinebotRL-two-wheel-riser}"
WIN_ROOT="${RISER_WIN_ROOT:-G:\\wSpace\\cinebotRL-two-wheel-riser}"
PY="${ISAAC_PYTHON:-/mnt/g/isaaclab_venv/Scripts/python.exe}"
STAMP="${RISER_ALL79_STAMP:-20260717_residual_all79_exact_source_lookahead_v2}"
ARTIFACTS_WSL="$ROOT/artifacts/two_wheel_riser/$STAMP"
ARTIFACTS_WIN="$WIN_ROOT\\artifacts\\two_wheel_riser\\$STAMP"
PLAN_STAMP="${RISER_ALL79_PLAN_STAMP:-20260717_all79_playback_exact_source_v1}"
PLAN_DIR_WSL="$ROOT/artifacts/two_wheel_riser/$PLAN_STAMP"
PLAN_DIR_WIN="${RISER_ALL79_PLAN_DIR_WIN:-$WIN_ROOT\\artifacts\\two_wheel_riser\\$PLAN_STAMP}"
GAINS_WIN="${RISER_GAINS_WIN:-$WIN_ROOT\\docs\\03_training\\two_wheel_balance\\evidence_20260714_28kg\\lqr_gains.json}"
SCRIPT_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\smoke_riser_reference_playback.py"
MERGER_WIN="$WIN_ROOT\\scripts\\two_wheel_balance\\build_riser_residual_dataset.py"
ADMISSION="$ARTIFACTS_WSL/admission.json"
SOURCE_MANIFEST="${RISER_EXACT_SOURCE_MANIFEST_WSL:-}"
SOURCE_MANIFEST_SHA256="${RISER_EXACT_SOURCE_MANIFEST_SHA256:-}"
EXACT_SOURCE_AUDIT="$ARTIFACTS_WSL/exact_source_admission.json"
VALIDATOR="$ROOT/scripts/two_wheel_balance/validate_riser_exact_source_manifest.py"

[[ -x "$PY" ]] || { printf 'missing Isaac Python: %s\n' "$PY" >&2; exit 2; }
[[ -s "$PLAN_DIR_WSL/manifest.json" ]] || {
  printf 'missing all-79 playback manifest: %s\n' "$PLAN_DIR_WSL/manifest.json" >&2
  exit 2
}
[[ -n "$SOURCE_MANIFEST" && -s "$SOURCE_MANIFEST" ]] || {
  printf 'missing quality-qualified exact_source_v1 manifest; set RISER_EXACT_SOURCE_MANIFEST_WSL\n' >&2
  exit 2
}
[[ "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'set RISER_EXACT_SOURCE_MANIFEST_SHA256 to the admitted manifest hash\n' >&2
  exit 2
}
TEMP_AUDIT="$(mktemp)"
trap 'rm -f "$TEMP_AUDIT"' EXIT
python3 "$VALIDATOR" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-count 79 \
  --mode training \
  --expected-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --output "$TEMP_AUDIT" >/dev/null
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet || {
  printf 'tracked worktree changes make capture provenance ambiguous\n' >&2
  exit 2
}
CAPTURE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
if [[ ! -e "$ADMISSION" ]] && find \
  "$ARTIFACTS_WSL/cases" "$ARTIFACTS_WSL/gates" \
  -type f -print -quit 2>/dev/null | grep -q .; then
  printf 'refusing to backfill admission onto existing capture artifacts\n' >&2
  exit 2
fi
mkdir -p "$ARTIFACTS_WSL/cases" "$ARTIFACTS_WSL/gates" "$ARTIFACTS_WSL/logs"
if [[ -e "$EXACT_SOURCE_AUDIT" ]]; then
  cmp -s "$TEMP_AUDIT" "$EXACT_SOURCE_AUDIT" || {
    printf 'exact-source admission differs from existing partial run\n' >&2
    exit 2
  }
else
  mv "$TEMP_AUDIT" "$EXACT_SOURCE_AUDIT"
fi
python3 - "$ADMISSION" "$CAPTURE_COMMIT" "$PLAN_DIR_WSL/manifest.json" "$SOURCE_MANIFEST" "$EXACT_SOURCE_AUDIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
manifest = Path(sys.argv[3])
source_manifest = Path(sys.argv[4])
source_audit = Path(sys.argv[5])
expected = {
    "schema": "cinebotrl_two_wheel_riser_capture_admission_v1",
    "git_commit": sys.argv[2],
    "plan_manifest": str(manifest),
    "plan_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "trajectory_integrity_contract": "exact_source_v1",
    "exact_source_manifest": str(source_manifest),
    "exact_source_manifest_sha256": hashlib.sha256(
        source_manifest.read_bytes()
    ).hexdigest(),
    "exact_source_admission": str(source_audit),
    "exact_source_admission_sha256": hashlib.sha256(
        source_audit.read_bytes()
    ).hexdigest(),
    "upstream_valid_for_training": True,
    "cases": list(range(1, 80)),
    "tracking_profile": "riser_phase_consistent_v2",
    "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
    "observation_contract": "executed_state_with_execution_time_lookahead_v2",
    "lookahead_horizons_s": [0.25, 0.5, 1.0],
}
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise SystemExit("capture admission differs from existing partial run")
else:
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
PY

gate_is_resumable() {
  local case_number="$1"
  local padded="$2"
  local gate="$ARTIFACTS_WSL/gates/case_$padded.json"
  local dataset="$ARTIFACTS_WSL/cases/case_${padded}_executed_residual_v2.npz"
  [[ -s "$gate" && -s "$dataset" ]] || return 1
  python3 - "$gate" "$case_number" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
case = int(sys.argv[2])
valid = (
    payload.get("passed") is True
    and payload.get("tracking_profile") == "riser_phase_consistent_v2"
    and payload.get("phase_feedforward_contract")
    == "derivatives_scaled_by_progress_v1"
    and payload.get("passed_case_count") == 1
    and payload.get("cases") == [case]
    and len(payload.get("results", [])) == 1
    and payload["results"][0].get("passed") is True
)
raise SystemExit(0 if valid else 1)
PY
}

for case_number in $(seq 1 79); do
  padded="$(printf '%04d' "$case_number")"
  if gate_is_resumable "$case_number" "$padded"; then
    printf 'case %s already passed; resuming\n' "$padded"
    continue
  fi
  printf 'running case %s\n' "$padded"
  "$PY" -u -X utf8 "$SCRIPT_WIN" \
    --gains "$GAINS_WIN" \
    --plan-dir "$PLAN_DIR_WIN" \
    --cases "$case_number" \
    --dataset-dir "$ARTIFACTS_WIN\\cases" \
    --output "$ARTIFACTS_WIN\\gates\\case_$padded.json" \
    --headless >"$ARTIFACTS_WSL/logs/case_$padded.log" 2>&1 || {
      tail -n 80 "$ARTIFACTS_WSL/logs/case_$padded.log" >&2
      printf 'case %s process failed\n' "$padded" >&2
      exit 3
    }
  if ! gate_is_resumable "$case_number" "$padded"; then
    tail -n 80 "$ARTIFACTS_WSL/logs/case_$padded.log" >&2
    printf 'case %s dynamic or dataset gate failed\n' "$padded" >&2
    exit 4
  fi
  printf 'case %s passed\n' "$padded"
done

"$PY" -u -X utf8 "$MERGER_WIN" \
  --case-dir "$ARTIFACTS_WIN\\cases" \
  --output "$ARTIFACTS_WIN\\all79_residual_dataset_v2.npz" \
  --expected-count 79 >"$ARTIFACTS_WSL/merge.log" 2>&1

python3 - "$ARTIFACTS_WSL" "$CAPTURE_COMMIT" "$PLAN_DIR_WSL/manifest.json" "$ADMISSION" "$SOURCE_MANIFEST" "$EXACT_SOURCE_AUDIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
commit = sys.argv[2]
plan_manifest = Path(sys.argv[3])
admission = Path(sys.argv[4])
source_manifest = Path(sys.argv[5])
source_audit = Path(sys.argv[6])
gates = []
for path in sorted((root / "gates").glob("case_*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    gates.append(
        {
            "case": result["case"],
            "tracking_profile": payload.get("tracking_profile"),
            "phase_governor_enabled": payload.get("phase_governor_enabled"),
            "completed_steps": result["completed_steps"],
            "position_error_p95_m": result["position_error_p95_m"],
            "position_error_max_m": result["position_error_max_m"],
            "attitude_error_p95_deg": result["attitude_error_p95_deg"],
            "attitude_error_max_deg": result["attitude_error_max_deg"],
            "pitch_max_deg": result["pitch_max_deg"],
            "passed": result["passed"],
        }
    )
dataset_summary = json.loads(
    (root / "all79_residual_dataset_v2.summary.json").read_text(encoding="utf-8")
)
summary = {
    "schema": "cinebotrl_two_wheel_riser_all79_dynamic_dataset_gate_v1",
    "git_commit": commit,
    "capture_admission": str(admission),
    "capture_admission_sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
    "plan_manifest": str(plan_manifest),
    "plan_manifest_sha256": hashlib.sha256(plan_manifest.read_bytes()).hexdigest(),
    "trajectory_integrity_contract": "exact_source_v1",
    "exact_source_manifest": str(source_manifest),
    "exact_source_manifest_sha256": hashlib.sha256(
        source_manifest.read_bytes()
    ).hexdigest(),
    "exact_source_admission": str(source_audit),
    "exact_source_admission_sha256": hashlib.sha256(
        source_audit.read_bytes()
    ).hexdigest(),
    "upstream_valid_for_training": True,
    "training_started": False,
    "ppo_authorized": False,
    "tracking_profiles": sorted({item["tracking_profile"] for item in gates}),
    "phase_governor_enabled": all(
        item["phase_governor_enabled"] is True for item in gates
    ),
    "case_count": len(gates),
    "passed_case_count": sum(item["passed"] for item in gates),
    "total_steps": sum(item["completed_steps"] for item in gates),
    "worst_metrics": {
        key: max(item[key] for item in gates)
        for key in (
            "position_error_p95_m",
            "position_error_max_m",
            "attitude_error_p95_deg",
            "attitude_error_max_deg",
            "pitch_max_deg",
        )
    },
    "dataset": dataset_summary,
    "cases": gates,
    "passed": len(gates) == 79 and all(item["passed"] for item in gates),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["passed"] else 5)
PY

printf 'all-79 dynamic residual dataset gate passed: %s\n' "$ARTIFACTS_WSL"
