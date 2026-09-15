"""The pre-registered Track 5 verdict (2026-09-14 design 8.2, amended by A1).

These tests pin the RULE, on synthetic frames, so that applying it to the real
data is mechanical and cannot be renegotiated after the numbers are in.
"""
import pandas as pd

from scripts import track5_verdict as tv


def _frame(rows):
    return pd.DataFrame(
        rows, columns=['arm_slug', 'M', 'split', 'k', 'n_trials_run', 'n_feasible'])


def test_delta_helps_when_the_mean_is_higher_and_the_majority_agrees():
    rows = []
    for split in range(10, 14):
        for k in (1, 2, 5, 9, 13, 17):
            rows.append(['joint-d000', 25, split, k, 100, 10])
            rows.append(['joint-d020', 25, split, k, 100, 20])

    result = tv.verdict(_frame(rows))

    assert result['condition_1'] is True
    assert result['cells_total'] == 24
    assert result['cells_favouring_d020'] == 24
    assert result['threshold'] == 13
    assert result['condition_2'] is True
    assert result['delta_helps'] is True


def test_a_higher_mean_carried_by_two_cells_does_not_pass():
    """Condition 2's whole reason for existing: the shape that produced the
    overstatement 8.4 had to walk back."""
    rows = []
    for split in range(10, 14):
        for k in (1, 2, 5, 9, 13, 17):
            rows.append(['joint-d000', 25, split, k, 100, 10])
            rows.append(['joint-d020', 25, split, k, 100, 9])
    # Two cells carry the mean.
    for i in (1, 3):
        rows[i * 2 + 1][5] = 100

    result = tv.verdict(_frame(rows))

    assert result['condition_1'] is True
    assert result['cells_favouring_d020'] == 2
    assert result['condition_2'] is False
    assert result['delta_helps'] is False


def test_ties_do_not_count_toward_the_majority():
    """'that sign is consistent' means strictly higher at d020. A tied cell
    supports neither arm and must not be counted as support."""
    rows = []
    for split in range(10, 14):
        for k in (1, 2, 5, 9, 13, 17):
            rows.append(['joint-d000', 25, split, k, 100, 10])
            rows.append(['joint-d020', 25, split, k, 100, 10])

    result = tv.verdict(_frame(rows))

    assert result['cells_favouring_d020'] == 0
    assert result['condition_2'] is False


def test_the_threshold_follows_the_split_count():
    """Amendment A1 fixed the threshold as a strict majority of the cells
    actually present: 13 of 24 at 4 splits, 10 of 18 at the 3-split OOM
    fallback."""
    assert tv.majority_threshold(24) == 13
    assert tv.majority_threshold(18) == 10


def test_only_M25_decides():
    """The verdict is stated at M = 25; M = 100 is the budget-dependence
    control and must not move it."""
    rows = []
    for split in range(10, 14):
        for k in (1, 2, 5, 9, 13, 17):
            rows.append(['joint-d000', 25, split, k, 100, 20])
            rows.append(['joint-d020', 25, split, k, 100, 10])
            rows.append(['joint-d000', 100, split, k, 100, 1])
            rows.append(['joint-d020', 100, split, k, 100, 99])

    result = tv.verdict(_frame(rows))

    assert result['delta_helps'] is False
