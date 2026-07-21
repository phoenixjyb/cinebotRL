import numpy as np

from scripts.two_wheel_balance.rank_riser_dagger_training_cases import rank_cases


def _dataset() -> tuple[dict, dict]:
    rows = []
    actions = []
    cases = []
    elapsed = []
    labels = []
    split_cases = {"train": [2, 6], "validation": [4], "holdout": [3]}
    for case, split, offset in ((2, 0, 0.02), (6, 0, 0.5), (4, 1, 0.0), (3, 2, 0.01)):
        for index in range(5):
            rows.append(np.full(65, offset + index * 0.01))
            actions.append(np.full(3, offset + index * 0.01))
            cases.append(case)
            elapsed.append(index * 0.005)
            labels.append(split)
    return (
        {
            "observations": np.asarray(rows),
            "actions": np.asarray(actions),
            "case_ids": np.asarray(cases),
            "elapsed_time_s": np.asarray(elapsed),
            "split_labels": np.asarray(labels),
        },
        {"split_cases": split_cases},
    )


def test_ranking_selects_nearest_training_case_and_excludes_other_splits() -> None:
    payload, metadata = _dataset()
    mask = np.ones(65)
    mask[23:26] = 0.0
    report = rank_cases(
        payload,
        metadata,
        np.zeros(65),
        np.ones(65),
        mask,
        reference_case=4,
        maximum_candidates=1,
    )
    assert report["selected_training_cases"] == [2]
    assert report["selected_candidates"][0]["split"] == "train"
    assert 4 not in report["selected_training_cases"]
    assert 3 not in report["selected_training_cases"]
    assert report["previous_action_channels_effective"] is False
    assert report["runtime_authorized"] is False
    assert report["dataset_created"] is False
    assert report["dagger_authorized"] is False
    assert report["valid_for_training"] is False


def test_ranking_rejects_non_validation_reference() -> None:
    payload, metadata = _dataset()
    try:
        rank_cases(
            payload,
            metadata,
            np.zeros(65),
            np.ones(65),
            np.ones(65),
            reference_case=2,
            maximum_candidates=1,
        )
    except ValueError as error:
        assert "validation split" in str(error)
    else:
        raise AssertionError("training reference was accepted")


def test_ranking_rejects_overlapping_splits() -> None:
    payload, metadata = _dataset()
    metadata["split_cases"]["holdout"].append(2)
    try:
        rank_cases(
            payload,
            metadata,
            np.zeros(65),
            np.ones(65),
            np.ones(65),
            reference_case=4,
            maximum_candidates=1,
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping split was accepted")
