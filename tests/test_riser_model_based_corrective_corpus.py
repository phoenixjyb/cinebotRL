import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_corpus import (
    DEFAULT_RESERVED_HOLDOUT_CASES,
    MODEL_BASED_CORRECTIVE_CORPUS_MANIFEST_SCHEMA,
    MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA,
    build_corpus,
    load_corpus,
    save_corpus,
    validate_corpus,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (
    MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
    PREVIOUS_ACTION_CONTRACT,
    TRAINING_TARGET_CONTRACT,
    save_case_dataset,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


TRAIN_CASES = [30, 23, 6, 2]
VALIDATION_CASES = [8, 16]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case_dataset(path: Path, *, case: int, split: str) -> None:
    count = 4
    actions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.2],
            [0.2, -0.2, 0.3],
            [0.3, -0.1, 0.2],
        ],
        dtype=np.float32,
    )
    requested_actions = actions.copy()
    requested_actions[2, 0] += 0.05
    effective_residual = actions * MODEL_BASED_POLICY_RESIDUAL_SCALES
    requested_residual = requested_actions * MODEL_BASED_POLICY_RESIDUAL_SCALES
    delta = effective_residual - requested_residual
    model = np.tile(np.array([0.1, -0.05, 0.8]), (count, 1))
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[1:, PREVIOUS_ACTION_INDICES] = actions[:-1]
    payload = {
        "observations": observations,
        "actions": actions,
        "requested_actions_audit": requested_actions,
        "effective_residual_commands": effective_residual,
        "requested_residual_commands_audit": requested_residual,
        "model_based_commands": model,
        "final_high_level_commands": model + effective_residual,
        "requested_vs_effective_residual_delta": delta,
        "command_clipped": np.abs(delta) > 2e-7,
        "case_ids": np.full(count, case, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "execution_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "source_time_s": np.arange(count, dtype=np.float64) * 0.002,
    }
    case_hex = f"{case:064x}"
    metadata = {
        "schema": MODEL_BASED_CORRECTIVE_CASE_DATASET_SCHEMA,
        "case": case,
        "split": split,
        "sample_count": count,
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "training_target_contract": TRAINING_TARGET_CONTRACT,
        "previous_action_contract": PREVIOUS_ACTION_CONTRACT,
        "previous_action_rebuilt": True,
        "source_capture_schema": (
            "cinebotrl_two_wheel_riser_corrective_teacher_capture_v2"
        ),
        "source_capture_sha256": case_hex,
        "source_final_status_sha256": f"{case + 100:064x}",
        "source_runtime_commit": f"{case:040x}",
        "source_plan_sha256": f"{case + 200:064x}",
        "source_corrective_profile_sha256": f"{case + 300:064x}",
        "source_paired_final_status_sha256": f"{case + 400:064x}",
        "requested_actions_used_as_training_targets": False,
        "effective_actions_used_as_training_targets": True,
        "valid_for_case_merge": True,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    save_case_dataset(
        path,
        metadata,
        payload,
        expected_case=case,
        expected_split=split,
    )


def _write_manifest(tmp_path: Path) -> Path:
    entries = []
    for split, cases in (("train", TRAIN_CASES), ("validation", VALIDATION_CASES)):
        for case in cases:
            path = tmp_path / f"case_{case:04d}.npz"
            _write_case_dataset(path, case=case, split=split)
            entries.append(
                {
                    "case": case,
                    "split": split,
                    "path": path.name,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema": MODEL_BASED_CORRECTIVE_CORPUS_MANIFEST_SCHEMA,
        "command_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "action_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        "training_target_contract": TRAINING_TARGET_CONTRACT,
        "previous_action_contract": PREVIOUS_ACTION_CONTRACT,
        "reserved_holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        "case_datasets": entries,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_corpus_build_round_trip_preserves_effective_model_residual_contract(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    metadata, payload = build_corpus(manifest)
    assert metadata["schema"] == MODEL_BASED_CORRECTIVE_CORPUS_SCHEMA
    assert metadata["split_cases"] == {
        "train": sorted(TRAIN_CASES),
        "validation": sorted(VALIDATION_CASES),
    }
    assert metadata["reserved_holdout_cases"] == DEFAULT_RESERVED_HOLDOUT_CASES
    assert metadata["holdout_rows_present"] is False
    assert metadata["valid_for_bc_admission_review"] is True
    assert metadata["bc_authorized"] is False
    assert metadata["valid_for_training"] is False
    np.testing.assert_allclose(
        payload["effective_residual_commands"],
        payload["actions"] * MODEL_BASED_POLICY_RESIDUAL_SCALES,
    )
    output = tmp_path / "corpus.npz"
    save_corpus(output, metadata, payload)
    restored_metadata, restored = load_corpus(output)
    assert restored_metadata == metadata
    np.testing.assert_array_equal(restored["actions"], payload["actions"])
    with pytest.raises(FileExistsError, match="overwrite"):
        save_corpus(output, metadata, payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("holdout", "reserved holdout"),
        ("duplicate", "duplicate case"),
        ("bad_sha", "SHA-256 mismatch"),
        ("too_few_train", "at least four train"),
        ("too_few_validation", "at least two validation"),
    ],
)
def test_corpus_manifest_rejects_unreviewed_or_unsealed_routes(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["case_datasets"]
    if mutation == "holdout":
        entries[0]["case"] = DEFAULT_RESERVED_HOLDOUT_CASES[0]
    elif mutation == "duplicate":
        entries[1]["case"] = entries[0]["case"]
    elif mutation == "bad_sha":
        entries[0]["sha256"] = "0" * 64
    elif mutation == "too_few_train":
        manifest["case_datasets"] = [
            entry for entry in entries if entry["case"] != TRAIN_CASES[-1]
        ]
    else:
        manifest["case_datasets"] = [
            entry for entry in entries if entry["case"] != VALIDATION_CASES[-1]
        ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_corpus(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("recurrence", "recurrence mismatch"),
        ("source_leakage", "source leakage"),
        ("command", "final command mismatch"),
        ("metadata", "metadata fields mismatch"),
        ("training_open", "metadata failed"),
    ],
)
def test_corpus_validation_rejects_tampering(
    tmp_path: Path, mutation: str, message: str
) -> None:
    metadata, payload = build_corpus(_write_manifest(tmp_path))
    metadata = copy.deepcopy(metadata)
    payload = {name: value.copy() for name, value in payload.items()}
    if mutation == "recurrence":
        payload["observations"][1, PREVIOUS_ACTION_INDICES[0]] += 0.2
    elif mutation == "source_leakage":
        payload["source_index"][4] = payload["source_index"][0]
    elif mutation == "command":
        payload["final_high_level_commands"][0, 0] += 0.1
    elif mutation == "metadata":
        metadata["unexpected"] = True
    else:
        metadata["bc_authorized"] = True
    with pytest.raises(ValueError, match=message):
        validate_corpus(metadata, payload)


def test_builder_cli_preflights_without_output_and_execute_is_review_only(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "corpus.npz"
    script = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/build_model_based_corrective_corpus.py"
    )
    base = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    preflight = subprocess.run(base, check=True, capture_output=True, text=True)
    assert json.loads(preflight.stdout)["output_created"] is False
    assert not output.exists()
    executed = subprocess.run(
        base + ["--execute"], check=True, capture_output=True, text=True
    )
    result = json.loads(executed.stdout)
    assert result["output_created"] is True
    assert result["holdout_rows_present"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["valid_for_training"] is False
    load_corpus(output)


def test_bc_loader_accepts_two_split_model_based_corpus_without_authorizing_rollout(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from scripts.two_wheel_balance.train_riser_residual_bc import (
        dataset_action_semantics,
        load_dataset,
    )

    metadata, payload = build_corpus(_write_manifest(tmp_path))
    output = tmp_path / "corpus.npz"
    save_corpus(output, metadata, payload)
    loaded_metadata, arrays = load_dataset(output)
    assert set(np.unique(arrays["split_labels"]).tolist()) == {0, 1}
    assert loaded_metadata["reserved_holdout_cases"] == DEFAULT_RESERVED_HOLDOUT_CASES
    assert dataset_action_semantics(loaded_metadata) == {
        "policy_command_base": "model_based_planner",
        "policy_residual_contract": MODEL_BASED_POLICY_RESIDUAL_CONTRACT,
        "residual_action_scales": [0.05, 0.05, 0.02],
    }


def test_bc_entrypoint_refuses_review_only_model_based_corpus(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    metadata, payload = build_corpus(_write_manifest(tmp_path))
    dataset = tmp_path / "corpus.npz"
    save_corpus(dataset, metadata, payload)
    output = tmp_path / "policy"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/two_wheel_balance/train_riser_residual_bc.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--source-commit",
            "a" * 40,
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "admission-review-only" in result.stderr
    assert not output.exists()
