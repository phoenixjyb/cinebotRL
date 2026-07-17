#!/usr/bin/env bash
set -euo pipefail

readonly OBSOLETE_AUTHORIZATION="AUTHORIZED_CASE74_RECOVERY_V4"
readonly PINNED_CASE="74"
readonly PINNED_NAMESPACE="20260717_gate_c_case74_recovery_v4_contract_v1_exclusive"
readonly PINNED_PORTFOLIO="20260717_exact_source_all79_portfolio_v4_threshold71"
readonly PINNED_MANIFEST_SHA256="851a7b2751cd397ba35daf57d1a8c6971fb14ed0186683af48d3c6109090570a"
readonly PINNED_SOURCE_SHA256="f265aa1bdd1cd6c762fd6e5367c00c7abcb7b19dea76bb30c6311885d2f3237d"
readonly PINNED_PLAN_SHA256="146ad72b0fd5a4010cd41231cc8c076cc29bea2d52de1cf1603550a617ee0f5f"
readonly PINNED_GAINS_SHA256="2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6"
readonly PINNED_USD_SHA256="89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0"
readonly PINNED_TRACKING_PROFILE="riser_recovery_direction_v4"
readonly PINNED_RECOVERY_RANGE="0.20,0.40"

protected_variables=(
  RISER_ROOT
  RISER_WIN_ROOT
  ISAAC_PYTHON
  RISER_GATE_C_PORTFOLIO_STAMP
  RISER_GATE_C_MANIFEST_SHA256
  RISER_GATE_C_SOURCE_SHA256
  RISER_GATE_C_CASES
  RISER_GATE_C_STAMP
  RISER_GAINS_WIN
  RISER_CASE74_CONTRACT
)
for variable in "${protected_variables[@]}"; do
  if [[ -v "$variable" ]]; then
    printf 'case-74 contract rejects environment override: %s\n' "$variable" >&2
    exit 7
  fi
done

authorization="${RISER_CASE74_GPU_AUTHORIZATION:-}"
if [[ "$authorization" == "$OBSOLETE_AUTHORIZATION" ]]; then
  printf 'obsolete case-74 authorization is permanently rejected\n' >&2
  exit 7
fi
if [[ -n "$authorization" ]]; then
  printf 'no case-74 runtime authorization token exists in this revision\n' >&2
  exit 7
fi

printf '%s\n' \
  "case-74 recovery-v4 contract is CPU-review-only" \
  "case=$PINNED_CASE namespace=$PINNED_NAMESPACE" \
  "portfolio=$PINNED_PORTFOLIO manifest=$PINNED_MANIFEST_SHA256" \
  "source=$PINNED_SOURCE_SHA256 plan=$PINNED_PLAN_SHA256" \
  "gains=$PINNED_GAINS_SHA256 usd=$PINNED_USD_SHA256" \
  "profile=$PINNED_TRACKING_PROFILE recovery_range=$PINNED_RECOVERY_RANGE" >&2
exit 7
