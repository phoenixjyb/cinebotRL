#!/usr/bin/env python3
"""Build and verify the non-training ep4 duration-preserving time-warp package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.ep4_time_reparameterization import (  # noqa: E402
    TimeReparameterizationConfig,
    derive_ep4_time_warp_package,
    verify_ep4_time_warp_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--raw-integrity-seed-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-speed-cap-mps", type=float, default=0.40)
    parser.add_argument("--angular-speed-cap-radps", type=float, default=0.35)
    parser.add_argument("--minimum-interval-dt-s", type=float, default=1.0e-3)
    parser.add_argument("--diagnostic-transition-start-1based", type=int, default=190)
    parser.add_argument("--diagnostic-transition-end-1based", type=int, default=205)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TimeReparameterizationConfig(
        translation_speed_cap_mps=args.translation_speed_cap_mps,
        angular_speed_cap_radps=args.angular_speed_cap_radps,
        minimum_interval_dt_s=args.minimum_interval_dt_s,
        diagnostic_transition_start_1based=args.diagnostic_transition_start_1based,
        diagnostic_transition_end_1based=args.diagnostic_transition_end_1based,
    )
    derive_ep4_time_warp_package(
        args.reference_package,
        args.raw_integrity_seed_package,
        args.output_dir,
        config=config,
    )
    manifest = verify_ep4_time_warp_package(
        args.reference_package,
        args.raw_integrity_seed_package,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
