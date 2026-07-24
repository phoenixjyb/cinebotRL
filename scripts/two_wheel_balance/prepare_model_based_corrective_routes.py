#!/usr/bin/env python3
"""Validate pending corrective routes with one canonical CPU-only command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    DATASET_SCHEMA,
    MODEL_BASED_POLICY_CONTROL_OWNERSHIP_CONTRACT,
    MODEL_BASED_POLICY_PREVIOUS_ACTION_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_route_preflight_v1"
)
CATALOG_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_route_catalog_v1"
)
CATALOG_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_route_catalog_v1.json"
)
CLOSED_FIELDS = (
    "runtime_authorized",
    "gpu_launch_authorized",
    "label_capture_authorized",
    "dataset_creation_authorized",
    "dataset_conversion_authorized",
    "dataset_merge_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
    "valid_for_training",
)
SHARED_PAIR_GATE_KEYS = (
    "rollout_order",
    "candidate_requires_baseline_dynamic_pass",
    "label_capture_during_pair",
    "dataset_creation_during_pair",
    "minimum_position_p95_improvement_m",
    "minimum_position_p95_relative_improvement",
    "maximum_position_error_regression_m",
    "maximum_attitude_error_regression_deg",
    "maximum_pitch_regression_deg",
    "maximum_riser_error_regression_m",
    "saturation_regression_allowed",
    "maximum_runtime_seconds_per_rollout",
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _windows_to_wsl(value: str) -> str:
    path = PureWindowsPath(value)
    if not path.is_absolute() or not path.drive:
        return value
    drive = path.drive[0].lower()
    relative = path.as_posix().split(":/", 1)[1]
    return f"/mnt/{drive}/{relative}"


def _git(repo: Path, *args: str, check: bool = True) -> str:
    if os.name == "nt":
        command = [
            "wsl.exe",
            "--exec",
            "git",
            "-C",
            _windows_to_wsl(str(repo.resolve())),
            *(_windows_to_wsl(value) for value in args),
        ]
    else:
        command = ["git", "-C", str(repo), *args]
    result = subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _all_true(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_all_true(item) for item in value.values())
    return value is True


def _closed(payload: Mapping[str, object]) -> bool:
    return all(
        payload.get(field) in (None, False)
        for field in CLOSED_FIELDS
    )


def canonical_teacher_contract() -> dict[str, object]:
    return {
        "dataset_schema": DATASET_SCHEMA,
        "observation_dimension": len(OBSERVATION_NAMES),
        "action_dimension": len(ACTION_NAMES),
        "action_names": list(ACTION_NAMES),
        "residual_action_scales": (
            MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist()
        ),
        "residual_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "control_ownership_contract": (
            MODEL_BASED_POLICY_CONTROL_OWNERSHIP_CONTRACT
        ),
        "previous_action_contract": (
            MODEL_BASED_POLICY_PREVIOUS_ACTION_CONTRACT
        ),
        "training_target": "effective_post_supervisor_residual_action",
        "source_execution_and_elapsed_clocks_required": True,
    }


def _identity_checks(
    repo: Path,
    identities: Mapping[str, object],
) -> bool:
    for value in identities.values():
        if not isinstance(value, Mapping):
            return False
        relative = value.get("path")
        if not isinstance(relative, str):
            return False
        path = repo / relative
        if not path.is_file():
            return False
        if _sha256(path) != value.get("sha256"):
            return False
        if _git(repo, "hash-object", str(path)) != value.get(
            "git_blob_sha1"
        ):
            return False
    return True


def _route_static_row(
    repo: Path,
    route: Mapping[str, object],
    catalog: Mapping[str, object],
) -> dict[str, object]:
    contract_path = repo / str(route.get("contract", ""))
    wrapper_path = repo / str(route.get("wrapper", ""))
    validator_path = repo / str(route.get("validator", ""))
    finalizer_path = repo / str(route.get("finalizer", ""))
    profile_path = repo / str(route.get("corrective_profile", ""))
    contract = _load(contract_path) if contract_path.is_file() else {}
    identities = contract.get("identities", {})
    controller = contract.get("controller_arguments", {})
    pair = contract.get("paired_experiment_contract", {})
    shared_controller = catalog.get("shared_controller_contract", {})
    shared_gates = catalog.get("shared_pair_gates", {})
    wrapper_source = (
        wrapper_path.read_text(encoding="utf-8")
        if wrapper_path.is_file()
        else ""
    )
    required_paths = {
        str(route.get("wrapper")),
        str(route.get("validator")),
        str(route.get("finalizer")),
        str(route.get("corrective_profile")),
    }
    identity_paths = {
        str(value.get("path"))
        for value in identities.values()
        if isinstance(value, Mapping)
    }
    checks = {
        "files_exist": all(
            path.is_file()
            for path in (
                contract_path,
                wrapper_path,
                validator_path,
                finalizer_path,
                profile_path,
            )
        ),
        "contract_case_split_namespace": (
            contract.get("schema") == route.get("contract_schema")
            and contract.get("case") == route.get("case")
            and contract.get("split") == route.get("split")
            and contract.get("namespace") == route.get("namespace")
        ),
        "identity_count": (
            isinstance(identities, Mapping)
            and len(identities) == route.get("identity_count")
        ),
        "identity_files_match": (
            isinstance(identities, Mapping)
            and _identity_checks(repo, identities)
        ),
        "route_files_identity_bound": required_paths <= identity_paths,
        "plan_identity_present": route.get("plan_identity") in identities,
        "canonical_residual_scales": (
            contract.get("residual_action_scales")
            == catalog.get("canonical_teacher_contract", {}).get(
                "residual_action_scales"
            )
        ),
        "shared_controller": all(
            controller.get(key) == value
            for key, value in shared_controller.items()
        )
        and controller.get("case") == route.get("case")
        and controller.get("reset_seed")
        == controller.get("configuration_seed") + route.get("case", 0),
        "shared_dynamic_thresholds": (
            contract.get("unchanged_dynamic_gate_thresholds")
            == catalog.get("shared_dynamic_thresholds")
        ),
        "shared_pair_gates": all(
            pair.get(key) == shared_gates.get(key)
            for key in SHARED_PAIR_GATE_KEYS
        ),
        "authorization_closed": (
            contract.get("authorization_token_issued") is False
            and contract.get("runtime_authorization_token_sha256") == ""
            and _closed(contract)
            and 'readonly AUTHORIZATION_SHA256=""' in wrapper_source
        ),
        "pair_does_not_capture_or_train": all(
            marker not in wrapper_source
            for marker in (
                "--dataset-dir",
                "--raw-teacher-dir",
                "--policy-trace-dir",
                "--shadow-teacher-trace-dir",
                "--corrective-teacher-capture-dir",
            )
        ),
    }
    return {
        "key": route.get("key"),
        "case": route.get("case"),
        "split": route.get("split"),
        "role": route.get("role"),
        "namespace": route.get("namespace"),
        "perturbation_mode": route.get("perturbation_mode"),
        "contract": {
            "path": str(route.get("contract")),
            "sha256": _sha256(contract_path)
            if contract_path.is_file()
            else None,
            "git_blob_sha1": _git(
                repo, "hash-object", str(contract_path)
            )
            if contract_path.is_file()
            else None,
        },
        "identity_count": len(identities)
        if isinstance(identities, Mapping)
        else 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_catalog(
    repo: Path,
    catalog_path: Path,
) -> dict[str, object]:
    catalog = _load(catalog_path)
    routes = catalog.get("routes", [])
    rows = [
        _route_static_row(repo, route, catalog)
        for route in routes
        if isinstance(route, Mapping)
    ]
    keys = [row["key"] for row in rows]
    cases = [row["case"] for row in rows]
    checks = {
        "schema": catalog.get("schema") == CATALOG_SCHEMA,
        "canonical_path": catalog_path.resolve()
        == (repo / CATALOG_RELATIVE_PATH).resolve(),
        "canonical_teacher_contract": (
            catalog.get("canonical_teacher_contract")
            == canonical_teacher_contract()
        ),
        "route_order": keys
        == ["case7_pair", "case8_validation_pair", "case16_validation_pair"],
        "unique_routes_and_cases": (
            len(keys) == len(set(keys)) == 3
            and len(cases) == len(set(cases)) == 3
        ),
        "train_then_validation": [
            row["split"] for row in rows
        ] == ["train", "validation", "validation"],
        "operation_boundaries": catalog.get("operation_boundaries")
        == {
            "cpu_preflight_requires_authorization": False,
            "gpu_pair_requires_explicit_authorization": True,
            "label_capture_requires_separate_authorization": True,
            "dataset_conversion_requires_separate_authorization": True,
            "corpus_merge_requires_separate_authorization": True,
            "bc_requires_separate_authorization": True,
            "ppo_requires_separate_authorization": True,
        },
        "catalog_learning_closed": _closed(catalog),
        "all_routes_static_ready": len(rows) == 3
        and all(row["passed"] for row in rows),
    }
    return {
        "catalog": {
            "path": CATALOG_RELATIVE_PATH,
            "sha256": _sha256(catalog_path),
            "git_blob_sha1": _git(
                repo, "hash-object", str(catalog_path)
            ),
        },
        "canonical_teacher_contract": canonical_teacher_contract(),
        "checks": checks,
        "routes": rows,
        "passed": all(checks.values()),
    }


def _preflight_command(
    repo: Path,
    route: Mapping[str, object],
) -> list[str]:
    wrapper = str(repo / str(route["wrapper"]))
    if os.name == "nt":
        return [
            "wsl.exe",
            "--exec",
            "bash",
            _windows_to_wsl(wrapper),
            "--preflight",
        ]
    return ["bash", wrapper, "--preflight"]


def _run_preflight(
    repo: Path,
    route: Mapping[str, object],
) -> dict[str, object]:
    result = subprocess.run(
        _preflight_command(repo, route),
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "exit_code": result.returncode,
            "stderr": result.stderr.strip(),
            "passed": False,
        }
    payload = json.loads(result.stdout)
    check_groups = {
        key: _all_true(value)
        for key, value in payload.items()
        if key.endswith("checks") and isinstance(value, Mapping)
    }
    identities = payload.get("identities", {})
    normalized_identities = {
        name: {
            "sha256": value.get("sha256"),
            "git_blob_sha1": value.get("git_blob_sha1"),
        }
        for name, value in sorted(identities.items())
        if isinstance(value, Mapping)
    }
    checks = {
        "exit_zero": result.returncode == 0,
        "schema_case_split_namespace": (
            payload.get("schema") == route.get("preflight_schema")
            and payload.get("case") == route.get("case")
            and payload.get("split") == route.get("split")
            and payload.get("namespace") == route.get("namespace")
        ),
        "preflight_passed": payload.get("passed") is True,
        "cpu_and_route_ready": (
            payload.get("cpu_contract_ready") is True
            and payload.get("runtime_route_contract_ready") is True
            and payload.get("execution_route_complete") is True
        ),
        "all_check_groups_passed": bool(check_groups)
        and all(check_groups.values()),
        "identity_count": len(normalized_identities)
        == route.get("identity_count"),
        "authorization_and_learning_closed": _closed(payload)
        and payload.get("authorization_file") is None,
    }
    return {
        "schema": payload.get("schema"),
        "runtime_commit": payload.get("runtime_commit"),
        "upstream_commit": payload.get("upstream_commit"),
        "identity_count": len(normalized_identities),
        "identities": normalized_identities,
        "check_groups": check_groups,
        "checks": checks,
        "passed": all(checks.values()),
    }


def prepare_routes(
    repo: Path,
    catalog_path: Path,
    selected_keys: Sequence[str],
) -> dict[str, object]:
    static = validate_catalog(repo, catalog_path)
    catalog = _load(catalog_path)
    routes = {
        str(route["key"]): route
        for route in catalog.get("routes", [])
        if isinstance(route, Mapping)
    }
    unknown = sorted(set(selected_keys) - set(routes))
    selected = [routes[key] for key in selected_keys if key in routes]
    preflights = [
        {
            "key": route["key"],
            "case": route["case"],
            "split": route["split"],
            **_run_preflight(repo, route),
        }
        for route in selected
    ]
    head = _git(repo, "rev-parse", "HEAD")
    upstream = _git(repo, "rev-parse", "@{upstream}")
    tracked_clean = (
        _git(
            repo,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        == ""
    )
    checks = {
        "catalog_passed": static["passed"] is True,
        "known_nonempty_selection": not unknown and bool(selected),
        "selection_preserves_catalog_order": [
            route["key"] for route in selected
        ] == list(selected_keys),
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "all_selected_preflights_pass": len(preflights) == len(selected)
        and all(row["passed"] for row in preflights),
    }
    return {
        "schema": SCHEMA,
        "git": {
            "head": head,
            "upstream": upstream,
            "tracked_worktree_clean": tracked_clean,
        },
        "catalog": static["catalog"],
        "canonical_teacher_contract": static[
            "canonical_teacher_contract"
        ],
        "selected_routes": list(selected_keys),
        "unknown_routes": unknown,
        "static_catalog_checks": static["checks"],
        "routes": preflights,
        "checks": checks,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=PROJECT_ROOT
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / CATALOG_RELATIVE_PATH,
    )
    parser.add_argument(
        "--routes",
        default="case7_pair,case8_validation_pair,case16_validation_pair",
        help="Comma-separated catalog route keys in execution-review order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [key for key in args.routes.split(",") if key]
    result = prepare_routes(
        args.repo_root.resolve(),
        args.catalog.resolve(),
        selected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
