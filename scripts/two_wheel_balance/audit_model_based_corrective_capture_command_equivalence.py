#!/usr/bin/env python3
"""Audit corrective-capture wrappers without executing their runtime commands."""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


SCHEMA = "cinebotrl_two_wheel_riser_corrective_capture_command_audit_v1"
CONTRACT_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_capture_command_contract_v1"
)
CAPTURE_CONTRACT_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_contract_v2"
)
AUTHORIZATION_FIELDS = (
    "authorization_token_issued",
    "runtime_authorized",
    "gpu_launch_authorized",
    "label_capture_authorized",
    "dataset_creation_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
)
IO_OPTIONS = {
    "--cases",
    "--corrective-teacher-capture-dir",
    "--corrective-teacher-capture-admission",
    "--runtime-heartbeat",
    "--output",
}
ASSIGNMENT_PATTERN = re.compile(
    r"^readonly\s+([A-Z][A-Z0-9_]*)=(.*)$"
)
VARIABLE_PATTERN = re.compile(r"\$(?:\{([A-Z][A-Z0-9_]*)\}|([A-Z][A-Z0-9_]*))")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _windows_path_to_wsl(value: str) -> str:
    if len(value) >= 3 and value[1:3] in (":\\", ":/"):
        return f"/mnt/{value[0].lower()}/{value[3:].replace(chr(92), '/')}"
    raise ValueError(f"cannot map Windows path into WSL: {value}")


def _git_command(repo: Path, *args: str) -> list[str]:
    if os.name == "nt":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        return [
            str(windir / "System32/wsl.exe"),
            "git",
            "-C",
            _windows_path_to_wsl(str(repo)),
            *args,
        ]
    return ["git", "-C", str(repo), *args]


def _git_result(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _git_command(repo, *args),
        check=check,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = _git_result(repo, *args, check=check)
    return result.stdout.strip()


def _identity(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    tracked = (
        _git_result(
            repo,
            "ls-files",
            "--error-unmatch",
            relative,
            check=False,
        ).returncode
        == 0
    )
    actual_blob = _git_blob(path) if path.is_file() else None
    committed_blob = (
        _git(repo, "rev-parse", f"HEAD:{relative}", check=False)
        if tracked
        else ""
    )
    checks = {
        "file_exists": path.is_file(),
        "tracked": tracked,
        "committed_blob_matches": bool(
            tracked and actual_blob == committed_blob
        ),
    }
    return {
        "path": relative,
        "sha256": _sha256(path) if path.is_file() else None,
        "git_blob_sha1": actual_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_playback_defaults(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
        ):
            continue
        option = node.args[0].value
        default: Any = None
        action: str | None = None
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    default = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    default = None
            elif keyword.arg == "action":
                try:
                    action = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    action = None
        if action == "store_true":
            default = False
        defaults[option] = default
    return defaults


def parse_assignments(source: str) -> dict[str, str]:
    raw: dict[str, str] = {}
    for line in source.splitlines():
        match = ASSIGNMENT_PATTERN.match(line.strip())
        if match:
            raw[match.group(1)] = match.group(2).strip()

    resolved: dict[str, str] = {}

    def resolve(name: str, stack: tuple[str, ...] = ()) -> str:
        if name in resolved:
            return resolved[name]
        if name in stack or name not in raw:
            raise ValueError(f"unresolved shell variable: {name}")
        value = raw[name].replace('"', "").replace("'", "")

        def replace(match: re.Match[str]) -> str:
            dependency = match.group(1) or match.group(2)
            return resolve(dependency, (*stack, name))

        value = VARIABLE_PATTERN.sub(replace, value)
        while "\\\\" in value:
            value = value.replace("\\\\", "\\")
        resolved[name] = value
        return value

    for variable in raw:
        resolve(variable)
    return resolved


def parse_playback_options(source: str) -> dict[str, str | bool]:
    lines = source.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if '"$PY" -u -X utf8 "$PLAYBACK"' in line
        ),
        None,
    )
    if start is None:
        raise ValueError("playback command not found")
    options: dict[str, str | bool] = {}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith('>"$OUTPUT/logs/playback.log"'):
            break
        match = re.match(r"^(--[a-z0-9-]+)(?:\s+(.*?))?\s*\\$", stripped)
        if not match:
            continue
        option = match.group(1)
        if option in options:
            raise ValueError(f"duplicate playback option: {option}")
        raw_value = match.group(2)
        options[option] = (
            True
            if raw_value is None or not raw_value.strip()
            else raw_value.strip().strip('"').strip("'")
        )
    if not options:
        raise ValueError("playback options not found")
    return options


def _expand_option_value(
    value: str | bool,
    assignments: Mapping[str, str],
) -> str | bool:
    if isinstance(value, bool):
        return value

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1) or match.group(2)
        if variable not in assignments:
            raise ValueError(f"unresolved option variable: {variable}")
        return assignments[variable]

    expanded = VARIABLE_PATTERN.sub(replace, value)
    while "\\\\" in expanded:
        expanded = expanded.replace("\\\\", "\\")
    return expanded


def _repo_relative_windows_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "/cinebotRL-two-wheel-riser/"
    if marker not in normalized:
        raise ValueError(f"path is outside expected Windows repository: {value}")
    return normalized.split(marker, 1)[1]


def _semantic_value(
    option: str,
    options: Mapping[str, str | bool],
    defaults: Mapping[str, Any],
) -> tuple[Any, str]:
    if option in options:
        return options[option], "explicit"
    return defaults.get(option), "playback_default"


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    try:
        return Decimal(str(actual)) == Decimal(str(expected))
    except (InvalidOperation, ValueError):
        return str(actual) == str(expected)


def _identity_name(
    binding: Mapping[str, Any],
    identities: Mapping[str, Any],
) -> str | None:
    name = binding.get("identity")
    if isinstance(name, str):
        return name if name in identities else None
    suffix = binding.get("identity_suffix")
    matches = (
        sorted(key for key in identities if key.endswith(suffix))
        if isinstance(suffix, str)
        else []
    )
    return matches[0] if len(matches) == 1 else None


def audit_route(
    repo: Path,
    route: Mapping[str, Any],
    manifest: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    case = route.get("case")
    split = route.get("split")
    wrapper_relative = route.get("wrapper")
    contract_relative = route.get("contract")
    if (
        not isinstance(case, int)
        or isinstance(case, bool)
        or not isinstance(split, str)
        or not isinstance(wrapper_relative, str)
        or not isinstance(contract_relative, str)
    ):
        raise ValueError(f"invalid route specification: {route}")

    wrapper_path = repo / wrapper_relative
    contract_path = repo / contract_relative
    source = wrapper_path.read_text(encoding="utf-8")
    assignments = parse_assignments(source)
    raw_options = parse_playback_options(source)
    options = {
        name: _expand_option_value(value, assignments)
        for name, value in raw_options.items()
    }
    contract = _load_object(contract_path)
    identities = contract.get("identities")
    if not isinstance(identities, dict):
        identities = {}

    expected_values = manifest.get("canonical_value_options")
    expected_flags = manifest.get("canonical_flag_options")
    bindings = manifest.get("identity_option_bindings")
    if (
        not isinstance(expected_values, dict)
        or not isinstance(expected_flags, list)
        or not isinstance(bindings, dict)
    ):
        raise ValueError("malformed command contract")

    semantic: dict[str, Any] = {}
    mismatches: list[str] = []
    for option, expected in expected_values.items():
        actual, source_kind = _semantic_value(option, options, defaults)
        semantic[option] = {
            "expected": expected,
            "actual": actual,
            "source": source_kind,
            "passed": _values_equal(actual, expected),
        }
        if not _values_equal(actual, expected):
            mismatches.append(option)
    for option in expected_flags:
        actual, source_kind = _semantic_value(option, options, defaults)
        semantic[option] = {
            "expected": True,
            "actual": actual,
            "source": source_kind,
            "passed": actual is True,
        }
        if actual is not True:
            mismatches.append(option)

    path_bindings: dict[str, Any] = {}
    for option, binding in bindings.items():
        identity_name = _identity_name(binding, identities)
        identity = identities.get(identity_name, {}) if identity_name else {}
        expected_path = identity.get("path") if isinstance(identity, dict) else None
        if binding.get("path_mode") == "parent" and isinstance(
            expected_path, str
        ):
            expected_path = str(Path(expected_path).parent).replace("\\", "/")
        actual_value = options.get(option)
        try:
            actual_path = (
                _repo_relative_windows_path(actual_value)
                if isinstance(actual_value, str)
                else None
            )
        except ValueError:
            actual_path = None
        passed = bool(
            identity_name
            and isinstance(expected_path, str)
            and actual_path == expected_path
        )
        path_bindings[option] = {
            "identity": identity_name,
            "expected_path": expected_path,
            "actual_path": actual_path,
            "passed": passed,
        }
        if not passed:
            mismatches.append(option)

    playback_identity = identities.get("playback")
    playback_path = assignments.get("PLAYBACK")
    try:
        playback_relative = (
            _repo_relative_windows_path(playback_path)
            if isinstance(playback_path, str)
            else None
        )
    except ValueError:
        playback_relative = None
    playback_binding_passed = bool(
        isinstance(playback_identity, dict)
        and playback_relative == playback_identity.get("path")
        and playback_identity.get("sha256") == _sha256(repo / playback_relative)
    )
    if not playback_binding_passed:
        mismatches.append("playback_identity")

    expected_options = (
        set(expected_values)
        | set(expected_flags)
        | set(bindings)
        | IO_OPTIONS
    )
    unknown_options = sorted(set(options) - expected_options)
    if unknown_options:
        mismatches.extend(unknown_options)

    capture_dir = options.get("--corrective-teacher-capture-dir")
    admission = options.get("--corrective-teacher-capture-admission")
    heartbeat = options.get("--runtime-heartbeat")
    output = options.get("--output")
    namespace = contract.get("namespace")
    io_checks = {
        "case": options.get("--cases") == str(case),
        "capture_dir": isinstance(capture_dir, str)
        and capture_dir.endswith(f"{namespace}\\capture"),
        "capture_admission": isinstance(admission, str)
        and admission.endswith(f"{namespace}\\admission.json"),
        "runtime_heartbeat": isinstance(heartbeat, str)
        and heartbeat.endswith(f"{namespace}\\runtime_heartbeat.json"),
        "output": isinstance(output, str)
        and output.endswith(f"{namespace}\\case_{case:04d}.json"),
    }
    contract_checks = {
        "schema": contract.get("schema") == CAPTURE_CONTRACT_SCHEMA,
        "case_split": contract.get("case") == case
        and contract.get("split") == split,
        "namespace": assignments.get("NAMESPACE") == namespace,
        "contract_path": assignments.get("CONTRACT", "").replace("\\", "/").endswith(
            contract_relative
        ),
        "authorization_closed": all(
            contract.get(field) is False for field in AUTHORIZATION_FIELDS
        ),
        "capture_only": contract.get("execution_contract", {}).get(
            "capture_only"
        )
        is True,
        "runtime_limit": contract.get("execution_contract", {}).get(
            "maximum_runtime_seconds"
        )
        == 600,
    }
    known = sorted(route.get("known_current_incompatibilities", []))
    unique_mismatches = sorted(set(mismatches))
    current_compatible = (
        not unique_mismatches
        and playback_binding_passed
        and not unknown_options
        and all(io_checks.values())
        and all(contract_checks.values())
    )
    classification_passed = (
        current_compatible
        if route.get("status") != "historical_only"
        else (
            unique_mismatches == known
            and not unknown_options
            and all(io_checks.values())
            and all(contract_checks.values())
        )
    )
    return {
        "case": case,
        "split": split,
        "status": route.get("status"),
        "wrapper": _identity(repo, wrapper_relative),
        "capture_contract": _identity(repo, contract_relative),
        "semantic_options": semantic,
        "identity_option_bindings": path_bindings,
        "playback_binding_passed": playback_binding_passed,
        "io_checks": io_checks,
        "contract_checks": contract_checks,
        "unknown_options": unknown_options,
        "mismatches": unique_mismatches,
        "known_current_incompatibilities": known,
        "current_command_compatible": current_compatible,
        "classification_passed": classification_passed,
        "runtime_authorized": False,
        "capture_started": False,
        "dataset_created": False,
        "training_started": False,
    }


def build_report(
    repo: Path,
    contract_path: Path,
    *,
    enforce_repository: bool = True,
) -> dict[str, Any]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    manifest = _load_object(contract_path)
    if manifest.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected command-contract schema")
    playback_relative = manifest.get("playback")
    routes = manifest.get("routes")
    if not isinstance(playback_relative, str) or not isinstance(routes, list):
        raise ValueError("malformed command contract")
    defaults = parse_playback_defaults(repo / playback_relative)
    route_reports = [
        audit_route(repo, route, manifest, defaults) for route in routes
    ]
    repository = {
        "head": _git(repo, "rev-parse", "HEAD"),
        "upstream": _git(repo, "rev-parse", "@{upstream}"),
        "tracked_worktree_clean": (
            _git_result(repo, "diff", "--quiet", check=False).returncode
            == 0
            and _git_result(
                repo, "diff", "--cached", "--quiet", check=False
            ).returncode
            == 0
        ),
    }
    repository["passed"] = bool(
        repository["head"] == repository["upstream"]
        and repository["tracked_worktree_clean"]
    )
    pending = [
        route
        for route in route_reports
        if route["status"] == "pending_authorization"
    ]
    authorization_closed = all(
        manifest.get(field) is False
        for field in (
            "runtime_authorized",
            "label_capture_authorized",
            "dataset_creation_authorized",
            "conversion_authorized",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
        )
    )
    checks = {
        "contract_identity": (
            _identity(repo, contract_path.relative_to(repo).as_posix())["passed"]
            if enforce_repository
            else contract_path.is_file()
        ),
        "playback_identity": _identity(repo, playback_relative)["passed"],
        "repository": repository["passed"] if enforce_repository else True,
        "four_routes_classified": len(route_reports) == 4,
        "all_routes_fail_closed": all(
            route["classification_passed"] for route in route_reports
        ),
        "one_pending_route": len(pending) == 1 and pending[0]["case"] == 7,
        "pending_case7_command_compatible": len(pending) == 1
        and pending[0]["current_command_compatible"],
        "authorization_closed": authorization_closed,
    }
    return {
        "schema": SCHEMA,
        "repository": repository,
        "command_contract": _identity(
            repo, contract_path.relative_to(repo).as_posix()
        ),
        "playback": _identity(repo, playback_relative),
        "routes": route_reports,
        "checks": checks,
        "passed": all(checks.values()),
        "generic_runtime_wrapper_created": False,
        "runtime_authorized": False,
        "capture_started": False,
        "dataset_created": False,
        "conversion_started": False,
        "bc_started": False,
        "ppo_started": False,
        "training_started": False,
        "next_operation": "authorize_exactly_one_case7_corrective_label_capture",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    report = build_report(args.repo, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
