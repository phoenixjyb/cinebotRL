#!/usr/bin/env python3
"""Finalize one admitted corrective capture without case-specific scaffolding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.two_wheel_balance.summarize_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    summarize as summarize_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (  # noqa: E402
    CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
    CORRECTIVE_CAPTURE_ALLOWED_SPLITS,
)


SCHEMA = "cinebotrl_two_wheel_riser_generic_corrective_capture_finalizer_v1"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def derive_route(root: Path, admission_path: Path) -> dict[str, Any]:
    admission = _load_object(admission_path)
    contract_path = root / "contract.json"
    contract = _load_object(contract_path)
    case = admission.get("case")
    split = admission.get("split")
    namespace = admission.get("namespace")
    identities = contract.get("identities")
    plan_names = (
        sorted(name for name in identities if name.endswith("_plan"))
        if isinstance(identities, dict)
        else []
    )
    checks = {
        "admission_schema": admission.get("schema")
        == CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "case_is_positive_integer": isinstance(case, int)
        and not isinstance(case, bool)
        and case > 0,
        "split_is_allowed": split in CORRECTIVE_CAPTURE_ALLOWED_SPLITS,
        "namespace_is_nonempty": isinstance(namespace, str) and bool(namespace),
        "contract_case_split": contract.get("case") == case
        and contract.get("split") == split,
        "contract_namespace": contract.get("namespace") == namespace,
        "exactly_one_plan_identity": len(plan_names) == 1,
        "capture_is_authorized": admission.get("runtime_authorized") is True
        and admission.get("label_capture_authorized") is True,
        "normalized_dataset_is_closed": admission.get(
            "dataset_creation_authorized"
        )
        is False,
        "learning_is_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"generic capture finalizer route mismatch: {checks}")
    assert isinstance(case, int)
    assert isinstance(split, str)
    assert isinstance(namespace, str)
    return {
        "schema": SCHEMA,
        "case": case,
        "split": split,
        "namespace": namespace,
        "plan_identity_name": plan_names[0],
        "capture_name": (
            f"case_{case:04d}_corrective_teacher_capture_v2.npz"
        ),
        "checks": checks,
    }


def finalize(
    root: Path,
    admission_path: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    gpu_release_passed: bool,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None:
        raise ValueError("runtime commit must be an exact lowercase Git SHA-1")
    route = derive_route(root, admission_path)
    result = summarize_capture(
        root,
        admission_path,
        runtime_commit=runtime_commit,
        playback_exit_code=playback_exit_code,
        gpu_release_passed=gpu_release_passed,
        expected_case=int(route["case"]),
        expected_split=str(route["split"]),
        expected_namespace=str(route["namespace"]),
        capture_name=str(route["capture_name"]),
        plan_identity_name=str(route["plan_identity_name"]),
    )
    result["generic_finalizer"] = route
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument("--gpu-release-passed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        args.admission,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite final status: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
