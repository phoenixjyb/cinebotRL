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
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_training_dataset import (
    CASE_BALANCING_CONTRACT,
    MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_SCHEMA,
    MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA,
    TRANSITION_CONTRACT,
    build_training_dataset,
    load_training_dataset,
    save_training_dataset,
    validate_training_dataset,
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
LOSS_MODULE = (
    Path(__file__).parents[1]
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_bc_loss.py"
)
LOSS_AUDIT = (
    Path(__file__).parents[1]
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_bc_loss_v1/summary.json"
)
PROMOTION_MODULE = (
    Path(__file__).parents[1]
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_corrective_training_dataset.py"
)
PROMOTION_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/"
    "promote_model_based_corrective_training_dataset.py"
)
PROMOTION_ADMISSION_TEMPLATE = (
    Path(__file__).parents[1]
    / "docs/03_training/two_wheel_balance/"
    "MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_TEMPLATE_20260723.json"
)


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


def _write_training_admission(
    path: Path,
    corpus: Path,
    *,
    approved: bool = True,
) -> None:
    admission = {
        "schema": MODEL_BASED_CORRECTIVE_TRAINING_ADMISSION_SCHEMA,
        "source_corpus_sha256": _sha256(corpus),
        "promotion_commit": "a" * 40,
        "loss_module_sha256": _sha256(LOSS_MODULE),
        "loss_audit_summary_sha256": _sha256(LOSS_AUDIT),
        "promotion_module_sha256": _sha256(PROMOTION_MODULE),
        "promotion_script_sha256": _sha256(PROMOTION_SCRIPT),
        "loss_contract": "model_based_projected_effective_action_bc_loss_v1",
        "projection_contract": "model_based_residual_safety_projection_v1",
        "requested_slew_regularization_contract": (
            "requested_physical_residual_slew_hinge_v1"
        ),
        "minimum_train_cases": 4,
        "minimum_validation_cases": 2,
        "reserved_holdout_cases": DEFAULT_RESERVED_HOLDOUT_CASES,
        "training_schema_promotion_approved": approved,
        "bc_authorized": False,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "training_started": False,
    }
    path.write_text(json.dumps(admission), encoding="utf-8")


def _build_review_corpus(tmp_path: Path) -> Path:
    metadata, payload = build_corpus(_write_manifest(tmp_path))
    corpus = tmp_path / "corpus.npz"
    save_corpus(corpus, metadata, payload)
    return corpus


def test_checked_in_training_promotion_template_is_unusable_by_default() -> None:
    admission = json.loads(
        PROMOTION_ADMISSION_TEMPLATE.read_text(encoding="utf-8")
    )
    assert admission["source_corpus_sha256"] is None
    assert admission["promotion_commit"] is None
    assert admission["training_schema_promotion_approved"] is False
    assert admission["bc_authorized"] is False
    assert admission["ppo_authorized"] is False
    assert admission["learned_rollout_authorized"] is False
    assert admission["training_started"] is False
    assert admission["promotion_module_sha256"] == _sha256(PROMOTION_MODULE)
    assert admission["promotion_script_sha256"] == _sha256(PROMOTION_SCRIPT)


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


def test_projection_training_promotion_derives_case_safe_mechanics(
    tmp_path: Path,
) -> None:
    corpus = _build_review_corpus(tmp_path)
    admission = tmp_path / "admission.json"
    _write_training_admission(admission, corpus)
    metadata, payload = build_training_dataset(
        corpus,
        admission,
        loss_module_path=LOSS_MODULE,
        loss_audit_summary_path=LOSS_AUDIT,
        promotion_module_path=PROMOTION_MODULE,
        promotion_script_path=PROMOTION_SCRIPT,
    )
    assert metadata["schema"] == MODEL_BASED_CORRECTIVE_TRAINING_DATASET_SCHEMA
    assert metadata["transition_contract"] == TRANSITION_CONTRACT
    assert metadata["case_balancing_contract"] == CASE_BALANCING_CONTRACT
    assert metadata["valid_for_projection_aware_bc_input"] is True
    assert metadata["valid_for_training"] is True
    assert metadata["bc_authorized"] is False
    assert metadata["ppo_authorized"] is False
    assert metadata["learned_rollout_authorized"] is False
    assert metadata["training_started"] is False
    assert metadata["holdout_rows_present"] is False
    cases = payload["case_ids"]
    for case in np.unique(cases):
        indices = np.flatnonzero(cases == case)
        assert payload["previous_row_index"][indices[0]] == -1
        assert payload["transition_valid"][indices[0]] is np.False_
        np.testing.assert_array_equal(
            payload["previous_row_index"][indices[1:]], indices[:-1]
        )
        assert np.all(payload["transition_valid"][indices[1:]])
        assert np.sum(payload["case_balanced_sample_weights"][indices]) == (
            pytest.approx(1.0)
        )
    output = tmp_path / "training.npz"
    save_training_dataset(output, metadata, payload)
    restored_metadata, restored = load_training_dataset(output)
    assert restored_metadata == metadata
    np.testing.assert_array_equal(
        restored["previous_row_index"], payload["previous_row_index"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("not_approved", "promotion"),
        ("wrong_corpus", "source"),
        ("wrong_loss", "loss_module"),
        ("open_bc", "learning_closed"),
    ],
)
def test_projection_training_promotion_rejects_forged_admission(
    tmp_path: Path, mutation: str, message: str
) -> None:
    corpus = _build_review_corpus(tmp_path)
    admission_path = tmp_path / "admission.json"
    _write_training_admission(admission_path, corpus)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if mutation == "not_approved":
        admission["training_schema_promotion_approved"] = False
    elif mutation == "wrong_corpus":
        admission["source_corpus_sha256"] = "0" * 64
    elif mutation == "wrong_loss":
        admission["loss_module_sha256"] = "0" * 64
    else:
        admission["bc_authorized"] = True
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_training_dataset(
            corpus,
            admission_path,
            loss_module_path=LOSS_MODULE,
            loss_audit_summary_path=LOSS_AUDIT,
            promotion_module_path=PROMOTION_MODULE,
            promotion_script_path=PROMOTION_SCRIPT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("previous", "previous-row"),
        ("transition", "transition mask"),
        ("delta", "derived array"),
        ("weights", "case weights"),
        ("holdout", "contains a holdout"),
        ("training_open", "learning_closed"),
    ],
)
def test_projection_training_dataset_rejects_derived_or_gate_tampering(
    tmp_path: Path, mutation: str, message: str
) -> None:
    corpus = _build_review_corpus(tmp_path)
    admission = tmp_path / "admission.json"
    _write_training_admission(admission, corpus)
    metadata, payload = build_training_dataset(
        corpus,
        admission,
        loss_module_path=LOSS_MODULE,
        loss_audit_summary_path=LOSS_AUDIT,
        promotion_module_path=PROMOTION_MODULE,
        promotion_script_path=PROMOTION_SCRIPT,
    )
    metadata = copy.deepcopy(metadata)
    payload = {name: value.copy() for name, value in payload.items()}
    if mutation == "previous":
        payload["previous_row_index"][1] = -1
    elif mutation == "transition":
        payload["transition_valid"][1] = False
    elif mutation == "delta":
        payload["delta_time_s"][1] = 0.0
    elif mutation == "weights":
        payload["case_balanced_sample_weights"][0] *= 2.0
    elif mutation == "holdout":
        original_case = int(payload["case_ids"][0])
        holdout_case = DEFAULT_RESERVED_HOLDOUT_CASES[0]
        payload["case_ids"][payload["case_ids"] == original_case] = holdout_case
        metadata["split_cases"]["train"] = [
            holdout_case if case == original_case else case
            for case in metadata["split_cases"]["train"]
        ]
        for source in metadata["source_datasets"]:
            if source["case"] == original_case:
                source["case"] = holdout_case
    else:
        metadata["bc_authorized"] = True
    with pytest.raises((ValueError, AssertionError), match=message):
        validate_training_dataset(metadata, payload)


def test_projection_training_cli_is_preflight_first_and_bc_stays_closed(
    tmp_path: Path,
) -> None:
    corpus = _build_review_corpus(tmp_path)
    admission = tmp_path / "admission.json"
    _write_training_admission(admission, corpus)
    output = tmp_path / "training.npz"
    script = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/"
        "promote_model_based_corrective_training_dataset.py"
    )
    base = [
        sys.executable,
        str(script),
        "--corpus",
        str(corpus),
        "--admission",
        str(admission),
        "--output",
        str(output),
    ]
    preflight = subprocess.run(
        base, check=True, capture_output=True, text=True
    )
    report = json.loads(preflight.stdout)
    assert report["output_created"] is False
    assert report["valid_for_projection_aware_bc_input"] is True
    assert report["bc_authorized"] is False
    assert report["training_started"] is False
    assert not output.exists()
    executed = subprocess.run(
        base + ["--execute"], check=True, capture_output=True, text=True
    )
    assert json.loads(executed.stdout)["output_created"] is True
    load_training_dataset(output)
    second = subprocess.run(
        base + ["--execute"], check=False, capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "already exists" in second.stderr

    trainer_output = tmp_path / "policy"
    trainer_base = [
        sys.executable,
        "scripts/two_wheel_balance/train_riser_residual_bc.py",
        "--dataset",
        str(output),
        "--output-dir",
        str(trainer_output),
        "--source-commit",
        "a" * 40,
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    trainer_preflight = subprocess.run(
        trainer_base + ["--preflight-only"],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert trainer_preflight.returncode == 0, trainer_preflight.stderr
    trainer_report = json.loads(trainer_preflight.stdout)
    assert trainer_report["preflight_passed"] is True
    assert trainer_report["preflight_only"] is True
    assert trainer_report["dataset_valid_for_projection_aware_bc_input"] is True
    assert trainer_report["requested_actions_used_as_training_targets"] is False
    assert trainer_report["effective_actions_remain_training_targets"] is True
    assert trainer_report["separate_hash_bound_bc_authorization_required"] is True
    assert trainer_report["bc_authorized"] is False
    assert trainer_report["training_started"] is False
    assert trainer_report["checkpoint_created"] is False
    assert not trainer_output.exists()

    rejected = subprocess.run(
        trainer_base,
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "separate hash-bound BC authorization" in rejected.stderr
    assert not trainer_output.exists()
