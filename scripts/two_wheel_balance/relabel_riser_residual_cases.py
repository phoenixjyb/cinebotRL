#!/usr/bin/env python3
"""Rebuild normalized labels from preserved raw commands without rerunning physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    DATASET_SCHEMA,
    validate_case_dataset,
)


def parse_scales(value: str) -> np.ndarray:
    scales = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise argparse.ArgumentTypeError("action scales must contain three positive values")
    return scales


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"wrong source schema: {path}")
    validate_case_dataset(payload, expected_case=int(metadata["case"]))
    return metadata, payload


def raw_residual(payload: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        (
            payload["teacher_commands"][:, 0] - payload["observations"][:, 18],
            payload["teacher_commands"][:, 1] - payload["observations"][:, 19],
            payload["teacher_commands"][:, 2] - payload["observations"][:, 15],
        )
    ).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case-dir", type=Path, required=True)
    parser.add_argument("--output-case-dir", type=Path, required=True)
    parser.add_argument("--action-scales", type=parse_scales, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.source_case_dir.glob("case_*_executed_residual_v1.npz"))
    if len(paths) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} source cases, found {len(paths)}")
    if args.output_case_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_case_dir}")
    args.output_case_dir.mkdir(parents=True)

    rows = []
    for source in paths:
        metadata, payload = load_source(source)
        residual = raw_residual(payload)
        normalized = residual / args.action_scales
        maximum = np.max(np.abs(normalized), axis=0)
        if np.any(maximum >= 1.0 - 1e-6):
            raise ValueError(
                f"case {metadata['case']} still exceeds proposed scales: {maximum}"
            )
        relabeled = dict(payload)
        relabeled["actions"] = normalized.astype(np.float32)
        validate_case_dataset(relabeled, expected_case=int(metadata["case"]))
        reconstructed = np.column_stack(
            (
                relabeled["observations"][:, 18]
                + args.action_scales[0] * relabeled["actions"][:, 0],
                relabeled["observations"][:, 19]
                + args.action_scales[1] * relabeled["actions"][:, 1],
                relabeled["observations"][:, 15]
                + args.action_scales[2] * relabeled["actions"][:, 2],
            )
        )
        reconstruction_error = float(
            np.max(np.abs(reconstructed - relabeled["teacher_commands"]))
        )
        if reconstruction_error > 2e-6:
            raise ValueError(f"case {metadata['case']} reconstruction failed")
        output = args.output_case_dir / source.name
        output_metadata = metadata | {
            "action_scales": args.action_scales.tolist(),
            "relabel_contract": "raw_teacher_command_reconstruction_v1",
            "source_action_scales": metadata["action_scales"],
            "source_file": source.name,
            "source_sha256": sha256(source),
        }
        np.savez_compressed(
            output,
            metadata_json=np.array(json.dumps(output_metadata, sort_keys=True)),
            **relabeled,
        )
        rows.append(
            {
                "case": int(metadata["case"]),
                "source_file": source.name,
                "source_sha256": sha256(source),
                "output_file": output.name,
                "output_sha256": sha256(output),
                "raw_residual_abs_max": np.max(np.abs(residual), axis=0).tolist(),
                "normalized_abs_max": maximum.tolist(),
                "reconstruction_max_error": reconstruction_error,
            }
        )
    cases = [row["case"] for row in rows]
    if len(cases) != len(set(cases)):
        raise ValueError("duplicate cases in source corpus")
    summary = {
        "schema": "cinebotrl_two_wheel_riser_residual_relabel_v1",
        "case_count": len(rows),
        "action_scales": args.action_scales.tolist(),
        "raw_residual_abs_max": np.max(
            np.asarray([row["raw_residual_abs_max"] for row in rows]), axis=0
        ).tolist(),
        "normalized_abs_max": np.max(
            np.asarray([row["normalized_abs_max"] for row in rows]), axis=0
        ).tolist(),
        "reconstruction_max_error": max(
            row["reconstruction_max_error"] for row in rows
        ),
        "rows": rows,
        "passed": True,
        "training_started": False,
        "ppo_authorized": False,
    }
    (args.output_case_dir.parent / "relabel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
