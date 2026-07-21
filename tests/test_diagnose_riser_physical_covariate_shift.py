import numpy as np
import pytest

from scripts.two_wheel_balance.diagnose_riser_physical_covariate_shift import (
    align_teacher_to_learned,
    analyze_fields,
    flatten_trace,
    wrapped_delta,
)


def _trace(phases, *, base_offset=0.0, position_offset=0.0):
    rows = []
    for phase in phases:
        row = {
            "phase_time_s": phase,
            "elapsed_s": phase,
            "position_error_m": 0.05 + position_offset * phase,
        }
        from scripts.two_wheel_balance.diagnose_riser_physical_covariate_shift import (
            FIELD_SPECS,
            VECTOR_FIELDS,
        )

        vector_names = {name for names in VECTOR_FIELDS.values() for name in names}
        for name in FIELD_SPECS:
            if name not in vector_names:
                row[name] = 0.0
        for source, names in VECTOR_FIELDS.items():
            values = [0.0] * len(names)
            if source == "actual_base_xy_yaw":
                values[0] = base_offset * phase
            row[source] = values
        rows.append(row)
    return rows


def test_alignment_uses_phase_not_row_index() -> None:
    teacher = flatten_trace(_trace([0.0, 1.0, 2.0]))
    learned = flatten_trace(_trace([0.0, 0.5, 1.5, 2.0]))
    teacher["pitch_deg"] = np.array([0.0, 2.0, 4.0])
    aligned = align_teacher_to_learned(teacher, learned)
    np.testing.assert_allclose(aligned["pitch_deg"], [0.0, 1.0, 3.0, 4.0])


def test_alignment_rejects_uncovered_phase() -> None:
    teacher = flatten_trace(_trace([0.0, 1.0]))
    learned = flatten_trace(_trace([0.0, 1.5]))
    with pytest.raises(ValueError, match="outside"):
        align_teacher_to_learned(teacher, learned)


def test_wrapped_delta_handles_yaw_branch_cut() -> None:
    delta = wrapped_delta(np.array([-3.13]), np.array([3.13]))
    assert abs(delta[0]) < 0.03


def test_analysis_localizes_pre_peak_base_shift() -> None:
    teacher = flatten_trace(_trace([0.0, 1.0, 2.0, 3.0]))
    learned = flatten_trace(
        _trace([0.0, 1.0, 2.0, 3.0], base_offset=0.03, position_offset=0.04)
    )
    fields, groups, peak_index, excess = analyze_fields(teacher, learned)
    assert peak_index == 3
    assert groups["base"]["normalized_envelope_pre_peak_max"] >= 9.0
    assert groups["base"]["onset_phase_time_s"] == 1.0
    assert excess[-1] == pytest.approx(0.12)
    assert fields[0]["group"] == "base"
