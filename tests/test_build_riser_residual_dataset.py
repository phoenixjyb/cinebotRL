from scripts.two_wheel_balance.build_riser_residual_dataset import grouped_split


def test_grouped_split_is_deterministic_and_case_disjoint() -> None:
    first = grouped_split(list(range(1, 80)), 20260716, 0.1, 0.1)
    second = grouped_split(list(range(1, 80)), 20260716, 0.1, 0.1)
    assert first == second
    assert len(first["train"]) == 63
    assert len(first["validation"]) == 8
    assert len(first["holdout"]) == 8
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["holdout"]))
    assert not (set(first["validation"]) & set(first["holdout"]))
    assert set().union(*map(set, first.values())) == set(range(1, 80))
