#!/usr/bin/env python3
"""Audit the closed, ordered queue of pending corrective-data routes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_pending_route_queue_v1"
)
CORPUS_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_corpus_intake_v1"
)


@dataclass(frozen=True)
class RouteSpec:
    key: str
    schema: str
    case: int
    split: str
    namespace: str
    kind: str
    identity_count: int


ROUTES = (
    RouteSpec(
        key="case23_conversion",
        schema=(
            "cinebotrl_two_wheel_riser_case23_conversion_"
            "execution_admission_v1"
        ),
        case=23,
        split="train",
        namespace="20260723_model_based_corrective_case23_conversion_v1_cpu",
        kind="cpu_conversion",
        identity_count=9,
    ),
    RouteSpec(
        key="case6_pair",
        schema=(
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case6_pair_admission_v2"
        ),
        case=6,
        split="train",
        namespace=(
            "20260724_model_based_corrective_teacher_case6_pair_v2_exclusive"
        ),
        kind="train_paired_canary",
        identity_count=18,
    ),
    RouteSpec(
        key="case2_pair",
        schema=(
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case2_natural_error_pair_admission_v1"
        ),
        case=2,
        split="train",
        namespace=(
            "20260724_model_based_corrective_teacher_"
            "case2_natural_error_pair_v1_exclusive"
        ),
        kind="train_paired_canary",
        identity_count=19,
    ),
    RouteSpec(
        key="case7_pair",
        schema=(
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case7_pair_admission_v1"
        ),
        case=7,
        split="train",
        namespace=(
            "20260724_model_based_corrective_teacher_case7_pair_v1_exclusive"
        ),
        kind="train_paired_canary",
        identity_count=18,
    ),
    RouteSpec(
        key="case8_validation_pair",
        schema=(
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case8_validation_pair_admission_v1"
        ),
        case=8,
        split="validation",
        namespace=(
            "20260724_model_based_corrective_teacher_"
            "case8_validation_pair_v1_exclusive"
        ),
        kind="validation_paired_canary",
        identity_count=19,
    ),
    RouteSpec(
        key="case16_validation_pair",
        schema=(
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case16_validation_natural_error_pair_admission_v1"
        ),
        case=16,
        split="validation",
        namespace=(
            "20260724_model_based_corrective_teacher_"
            "case16_validation_natural_error_pair_v1_exclusive"
        ),
        kind="validation_paired_canary",
        identity_count=24,
    ),
)

TRUE_CHECK_GROUPS = (
    "checks",
    "document_checks",
    "repository_checks",
    "contract_checks",
    "selection_checks",
    "readiness_checks",
    "profile_proposal_checks",
    "corrective_profile_checks",
    "perturbation_checks",
)
FALSE_FIELDS = (
    "authorization_token_issued",
    "authorization_consumed_before_isaac",
    "authorization_consumed_before_conversion",
    "capture_authorized",
    "conversion_authorized",
    "runtime_authorized",
    "gpu_launch_authorized",
    "teacher_admission_authorized",
    "label_capture_authorized",
    "dataset_conversion_authorized",
    "dataset_creation_authorized",
    "dataset_merge_authorized",
    "merge_authorized",
    "output_created",
    "valid_for_case_merge",
    "merged_dataset_created",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
    "valid_for_training",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _all_true(mapping: object) -> bool:
    return (
        isinstance(mapping, Mapping)
        and bool(mapping)
        and all(value is True for value in mapping.values())
    )


def _identity_checks(payload: Mapping[str, object], count: int) -> bool:
    identities = payload.get("identities")
    return (
        isinstance(identities, Mapping)
        and len(identities) == count
        and all(
            isinstance(identity, Mapping)
            and identity.get("passed") is True
            for identity in identities.values()
        )
    )


def _route_commit(payload: Mapping[str, object]) -> tuple[object, object]:
    git = payload.get("git")
    if isinstance(git, Mapping):
        return git.get("head"), git.get("upstream")
    return payload.get("runtime_commit"), payload.get("upstream_commit")


def _closed(payload: Mapping[str, object]) -> bool:
    if any(
        field in payload and payload.get(field) is not False
        for field in FALSE_FIELDS
    ):
        return False
    if "authorization_file" in payload and payload["authorization_file"] is not None:
        return False
    authorization_checks = payload.get("authorization_checks")
    if isinstance(authorization_checks, Mapping) and any(
        value is not False for value in authorization_checks.values()
    ):
        return False
    return True


def _route_row(
    *,
    spec: RouteSpec,
    payload: Mapping[str, object],
    path: Path,
    current_head: str,
    repo_root: Path,
) -> dict[str, object]:
    route_head, route_upstream = _route_commit(payload)
    check_groups = {
        name: _all_true(payload[name])
        for name in TRUE_CHECK_GROUPS
        if name in payload
    }
    namespace_path = repo_root / "artifacts/two_wheel_riser" / spec.namespace
    checks = {
        "schema_case_split_namespace": (
            payload.get("schema") == spec.schema
            and payload.get("case") == spec.case
            and payload.get("split") == spec.split
            and payload.get("namespace") == spec.namespace
        ),
        "preflight_passed": payload.get("passed") is True,
        "cpu_contract_ready": payload.get("cpu_contract_ready") is True,
        "route_complete_if_runtime": (
            spec.kind == "cpu_conversion"
            or (
                payload.get("runtime_route_contract_ready") is True
                and payload.get("execution_route_complete") is True
            )
        ),
        "head_and_upstream_current": (
            route_head == current_head and route_upstream == current_head
        ),
        "all_check_groups_passed": bool(check_groups)
        and all(check_groups.values()),
        "identity_count_and_checks": _identity_checks(
            payload, spec.identity_count
        ),
        "authorization_and_learning_closed": _closed(payload),
        "namespace_absent": not namespace_path.exists(),
    }
    return {
        "key": spec.key,
        "priority": next(
            index for index, route in enumerate(ROUTES, start=1)
            if route.key == spec.key
        ),
        "case": spec.case,
        "split": spec.split,
        "kind": spec.kind,
        "namespace": spec.namespace,
        "preflight": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        },
        "identity_count": spec.identity_count,
        "check_groups": check_groups,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_pending_route_queue(
    *,
    repo_root: Path,
    corpus_intake_path: Path,
    preflight_paths: Mapping[str, Path],
    current_head: str,
    upstream_head: str,
    tracked_worktree_clean: bool,
) -> dict[str, object]:
    corpus = _load_object(corpus_intake_path)
    expected_keys = [route.key for route in ROUTES]
    rows = [
        _route_row(
            spec=spec,
            payload=_load_object(preflight_paths[spec.key]),
            path=preflight_paths[spec.key],
            current_head=current_head,
            repo_root=repo_root,
        )
        for spec in ROUTES
    ]
    corpus_checks = {
        "schema_and_passed": (
            corpus.get("schema") == CORPUS_SCHEMA
            and corpus.get("passed") is True
        ),
        "converted_state": (
            corpus.get("converted_train_cases") == [30]
            and corpus.get("converted_validation_cases") == []
            and corpus.get("pending_minimum_train_cases") == [23, 6, 2]
            and corpus.get("pending_validation_cases") == [8, 16]
        ),
        "case23_is_next": corpus.get("next_bounded_action")
        == "authorize_exactly_one_case23_v4_cpu_conversion",
        "learning_closed": all(
            corpus.get(field) is False
            for field in (
                "case23_conversion_authorized",
                "case23_conversion_output_created",
                "corpus_manifest_ready",
                "runtime_authorized",
                "gpu_launch_authorized",
                "label_capture_authorized",
                "dataset_conversion_authorized",
                "dataset_merge_authorized",
                "valid_for_bc_admission_review",
                "bc_authorized",
                "ppo_authorized",
                "training_started",
                "valid_for_training",
            )
        ),
    }
    checks = {
        "exact_route_set_and_order": list(preflight_paths) == expected_keys,
        "head_matches_upstream": current_head == upstream_head,
        "tracked_worktree_clean": tracked_worktree_clean,
        "corpus_state": all(corpus_checks.values()),
        "all_six_preflights_pass": len(rows) == 6
        and all(row["passed"] is True for row in rows),
        "namespaces_unique": len({row["namespace"] for row in rows}) == 6,
        "train_before_validation": [row["split"] for row in rows]
        == ["train", "train", "train", "train", "validation", "validation"],
    }
    return {
        "schema": SCHEMA,
        "git": {
            "head": current_head,
            "upstream": upstream_head,
            "tracked_worktree_clean": tracked_worktree_clean,
        },
        "corpus_intake": {
            "path": str(corpus_intake_path.resolve()),
            "sha256": _sha256(corpus_intake_path),
            "checks": corpus_checks,
        },
        "checks": checks,
        "routes": rows,
        "ready_route_count": sum(row["passed"] is True for row in rows),
        "execution_order": expected_keys,
        "next_bounded_action": (
            "authorize_exactly_one_case23_v4_cpu_conversion"
        ),
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


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus-intake", type=Path, required=True)
    for route in ROUTES:
        parser.add_argument(
            f"--{route.key.replace('_', '-')}",
            dest=route.key,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    head = _git(repo_root, "rev-parse", "HEAD")
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    tracked_clean = (
        _git(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        == ""
    )
    preflight_paths = {
        route.key: getattr(args, route.key).resolve() for route in ROUTES
    }
    result = audit_pending_route_queue(
        repo_root=repo_root,
        corpus_intake_path=args.corpus_intake.resolve(),
        preflight_paths=preflight_paths,
        current_head=head,
        upstream_head=upstream,
        tracked_worktree_clean=tracked_clean,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
