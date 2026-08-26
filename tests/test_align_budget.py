import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE
from src.training import align_budget as ab


def test_pooled_interval_count_is_the_common_refinement():
    """Two gap-free tilings of one feature pool into their common refinement:
    thresholds {10} and {5} give intervals (0,5),(6,10),(11,INF) -- three, not
    the four a tuple union would report."""
    r1 = [(0, 10), (11, INFINITE)]
    r2 = [(0, 5), (6, INFINITE)]
    assert ab.pooled_interval_count(r1, r2) == 3
    assert ab.pooled_interval_count(r1, r1) == 2
    assert ab.pooled_interval_count([(0, INFINITE)], [(0, INFINITE)]) == 1


def test_band_ceiling_is_the_highest_length_still_in_a_band():
    """Used by the scoring module to price how far past a boundary a run
    overshot -- bits below the ceiling of the band it landed in bought
    nothing."""
    from src.p4gen.evaluation import band_factor
    for factor in (1, 2, 3, 7):
        ceiling = ab.band_ceiling(factor)
        assert band_factor(ceiling) == factor
        assert band_factor(ceiling + 1) == factor + 1


def test_band_target_is_the_highest_length_one_band_cheaper():
    """Boundaries sit at length + 4 == 44k. band_target must land exactly on
    the highest length in the next band down, never one off."""
    for length in (41, 60, 84, 85, 128, 300):
        target = ab.band_target(length)
        from src.p4gen.evaluation import band_factor
        assert band_factor(target) == band_factor(length) - 1
        assert band_factor(target + 1) == band_factor(length)


def test_band_target_is_unreachable_in_the_first_band():
    """factor == 1 is already the cheapest band; the target must be negative so
    `target >= floor` is False for any non-negative floor."""
    assert ab.band_target(0) < 0
    assert ab.band_target(40) < 0


def test_codeword_floor_is_reached_when_both_models_already_agree():
    """L_floor is exact, not a heuristic: alignment only RELOCATES a threshold,
    never deletes one from its own model (threshold_alignment.py:106-111), so a
    common feature's pooled set can never drop below the larger of the two
    models' own counts. Constructive check: make the two identical and the
    floor must equal the actual pooled count."""
    intervals = {0: [(0, 10), (11, 20), (21, INFINITE)],
                 1: [(0, 5), (6, INFINITE)]}
    assert ab.codeword_floor(intervals, intervals) == 2 + 1


def test_codeword_floor_counts_exclusive_features_in_full():
    iv1 = {0: [(0, 10), (11, INFINITE)], 1: [(0, 7), (8, INFINITE)]}
    iv2 = {0: [(0, 4), (5, 9), (10, INFINITE)]}
    # feature 0 common: max(2, 3) - 1 == 2 ; feature 1 exclusive: 2 - 1 == 1
    assert ab.codeword_floor(iv1, iv2) == 3


def test_a_gated_budget_stops_spending_when_the_floor_blocks_the_band():
    budget = ab.BandBudget(codeword_length=100, floor=99, delta_rel=0.2, gated=True)
    assert not budget.spending()          # target 84 < floor 99
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_a_gated_budget_spends_while_the_band_is_reachable():
    budget = ab.BandBudget(codeword_length=100, floor=10, delta_rel=0.2, gated=True)
    assert budget.spending()              # target 84 >= floor 10
    assert budget.delta_for_candidate() == 0.2
    assert budget.spent_budget is True


def test_an_ungated_budget_always_hands_out_the_raw_delta():
    """The legacy path: BandBudget is still constructed so spent_budget has one
    definition, but it must never gate."""
    budget = ab.BandBudget(codeword_length=100, floor=99, delta_rel=0.2, gated=False)
    assert budget.spending()
    assert budget.delta_for_candidate() == 0.2


def test_a_zero_delta_is_not_recorded_as_spending_budget():
    """delta_rel == 0.0 gives nothing away, so it is not 'spending' even in the
    spending state -- otherwise S3's wasted-bit share is uninterpretable."""
    budget = ab.BandBudget(codeword_length=100, floor=10, delta_rel=0.0, gated=True)
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_an_unbounded_delta_is_recorded_as_spending_budget():
    budget = ab.BandBudget(codeword_length=100, floor=10, delta_rel=None, gated=True)
    assert budget.delta_for_candidate() is None
    assert budget.spent_budget is True


def test_shedding_across_the_boundary_retargets_the_next_band():
    budget = ab.BandBudget(codeword_length=90, floor=10, delta_rel=0.2, gated=True)
    assert not budget.crossed_band()
    budget.note_shed(10)                  # 90 -> 80, factor 3 -> 2
    assert budget.length == 80
    assert budget.crossed_band()
    assert budget.spending()              # next target is 40, still >= floor 10


def test_shedding_stops_spending_once_the_next_band_is_out_of_reach():
    budget = ab.BandBudget(codeword_length=90, floor=50, delta_rel=0.2, gated=True)
    assert budget.spending()              # target 84 >= 50
    budget.note_shed(10)                  # 80, next target 40 < floor 50
    assert not budget.spending()
    assert budget.delta_for_candidate() == 0.0
