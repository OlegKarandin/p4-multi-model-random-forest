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
    from src.p4gen.evaluation import codeword_bits_to_blocks
    for factor in (1, 2, 3, 7):
        ceiling = ab.band_ceiling(factor)
        assert codeword_bits_to_blocks(ceiling) == factor
        assert codeword_bits_to_blocks(ceiling + 1) == factor + 1


def test_band_target_is_the_highest_length_one_band_cheaper():
    """Boundaries sit at length + 4 == 44k. band_target must land exactly on
    the highest length in the next band down, never one off."""
    for length in (41, 60, 84, 85, 128, 300):
        target = ab.band_target(length)
        from src.p4gen.evaluation import codeword_bits_to_blocks
        assert codeword_bits_to_blocks(target) == codeword_bits_to_blocks(length) - 1
        assert codeword_bits_to_blocks(target + 1) == codeword_bits_to_blocks(length)


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


# BandBudget's own spending/delta/shed/crossed unit tests were deleted here
# 2026-09-07 (gate repair): BlockBudget replaces it as the wired gate, and
# BlockBudget carries the identical-shaped tests below (search "BlockBudget
# (design 2026-09-07"), so keeping both would be duplication, not coverage.
# BandBudget the CLASS was itself deleted from align_budget.py in a later
# commit of this same repair -- BlockBudget's own tests below are its
# replacement, not merely a stand-in for tests of a still-importable class.
#
# ---------------------------------------------------------------------------
# Byte-domain arithmetic (design 2026-08-30 §2.2): the twin of the block-domain
# functions above, stepping on the 64-byte ternary crossbar budget instead of
# the 44-bit key block.
# ---------------------------------------------------------------------------

def test_bits_to_next_byte_frees_exactly_one_byte_and_one_fewer_frees_none():
    """The whole premise of byte-aware targeting: a bit is worth 0 or a whole
    byte depending only on where the feature's width sits modulo 8."""
    for width in range(1, 41):
        step = ab.bits_to_next_byte(width)
        assert ab.byte_width(width - step) == ab.byte_width(width) - 1
        assert ab.byte_width(width - step + 1) == ab.byte_width(width)
    assert ab.bits_to_next_byte(24) == 8          # w = 8k costs a full byte
    assert ab.bits_to_next_byte(25) == 1          # w = 8k+1 costs one bit


def test_key_bytes_floor_is_reached_when_both_models_already_agree():
    """Exact for the same reason codeword_floor is: alignment relocates a
    threshold and never deletes one, so a common feature's pooled set can
    never drop below the larger of the two models' own counts."""
    intervals = {0: [(0, 10), (11, 20), (21, INFINITE)],
                 1: [(0, 5), (6, INFINITE)]}
    assert ab.key_bytes_floor(intervals, intervals) == ab.pooled_key_bytes(
        intervals, intervals)


def test_key_bytes_floor_counts_exclusive_features_in_full():
    iv1 = {0: [(0, 10), (11, INFINITE)], 1: [(0, 7), (8, INFINITE)]}
    iv2 = {0: [(0, 4), (5, 9), (10, INFINITE)]}
    # feature 0 common: max(2, 3) - 1 == 2 bits -> 1 byte
    # feature 1 exclusive: 2 - 1 == 1 bit -> 1 byte
    assert ab.key_bytes_floor(iv1, iv2) == 2


def test_bits_to_reach_prices_the_same_starting_width_very_differently():
    """Why the answer is not a function of B alone: two pairs can both sit at
    B = 30 and cost 9 bits or 72 bits to reach 21, depending only on where
    each feature's width sits modulo 8."""
    cheap_widths = {f: 17 for f in range(30)}          # 17 -> 3 bytes each
    cheap_floors = {f: 1 for f in range(30)}
    assert sum(ab.byte_width(w) for w in cheap_widths.values()) == 90
    assert ab.bits_to_reach(cheap_widths, cheap_floors, 81) == 9

    dear_widths = {f: 24 for f in range(30)}           # 24 -> 3 bytes each
    dear_floors = {f: 1 for f in range(30)}
    assert sum(ab.byte_width(w) for w in dear_widths.values()) == 90
    assert ab.bits_to_reach(dear_widths, dear_floors, 81) == 72


def test_bits_to_reach_returns_zero_when_already_at_or_below_target():
    """A negative `need` must not slice the cost list from the right and
    return a nonsense positive cost."""
    widths, floors = {0: 17}, {0: 1}
    assert ab.bits_to_reach(widths, floors, 3) == 0
    assert ab.bits_to_reach(widths, floors, 10) == 0


@pytest.mark.parametrize('seed', range(20))
def test_bits_to_reach_is_none_exactly_when_the_floor_blocks_the_target(seed):
    """E1c. `None` means unreachable at any delta, and must agree with
    key_bytes_floor > target -- otherwise the bound and the floor would
    disagree about which cells can win at all."""
    rng = np.random.default_rng(seed)
    widths = {f: int(rng.integers(1, 40)) for f in range(8)}
    floors = {f: int(rng.integers(1, widths[f] + 1)) for f in widths}
    floor_bytes = sum(ab.byte_width(w) for w in floors.values())
    for target in range(1, sum(ab.byte_width(w) for w in widths.values()) + 1):
        unreachable = ab.bits_to_reach(widths, floors, target) is None
        assert unreachable == (floor_bytes > target)


def test_factor_prices_a_width_dict_exactly_as_p4model_does():
    """The anchor for the whole repair: alignment's block arithmetic must BE
    the generator's, not a copy of it. A dict is what _pooled_widths produces;
    ternary_block_factor takes a sorted tuple, so the sort belongs here rather
    than at every call site."""
    from src.p4model.tables import ternary_block_factor
    widths = {'a': 11, 'b': 3, 'c': 84}
    assert ab._factor(widths) == ternary_block_factor((3, 11, 84))


def test_factor_ignores_the_dicts_own_key_order():
    """The dict's order is the generator's feature-emission order. The crossbar
    allocator is free to place fields where it likes and measurably does, so
    ternary_block_factor prices a MULTISET -- emission order must never reach
    the cost model."""
    assert ab._factor({'a': 11, 'b': 3}) == ab._factor({'b': 3, 'a': 11})


def test_factor_of_an_empty_width_dict_is_the_empty_key_factor():
    """Reachable, not hypothetical: neither forest split on any feature, which
    happens whenever min_samples_leaf approaches n_samples and every tree is a
    single leaf -- a real Optuna sample, not a synthetic corner case."""
    from src.p4model.tables import ternary_block_factor
    assert ab._factor({}) == ternary_block_factor(())


# ---------------------------------------------------------------------------
# BlockBudget (design 2026-09-07 §4.1). Widths below (3 fields of 40 bits
# each, vs. a floor of 8 bits each) put both models comfortably past a block
# boundary in ternary_block_factor -- band and crossbar arms agree exactly at
# these widths, so what matters is only that factor(current) > factor(floor),
# not which arm binds.

def _block_widths(*widths):
    return {i: w for i, w in enumerate(widths)}


def test_a_block_budget_spends_while_a_cheaper_factor_is_reachable():
    """spending() is factor(current) > factor(floor) -- an EXACT reachability
    test with no inverse of the step function required, because the floor
    width vector IS the best attainable case (alignment can relocate a
    threshold but never delete one)."""
    current = _block_widths(40, 40, 40)          # 15 bytes
    floor = _block_widths(8, 8, 8)               #  3 bytes
    budget = ab.BlockBudget(current, floor, 0.05)
    assert ab._factor(current) > ab._factor(floor)
    assert budget.spending() is True
    assert budget.delta_for_candidate() == 0.05
    assert budget.spent_budget is True


def test_a_block_budget_declines_when_the_floor_is_already_the_factor():
    """§8's risk as a unit test: a pair whose floor costs what it already
    costs can never authorise spending, however generous delta is. The run
    collapses to free moves, which is CORRECT behaviour."""
    widths = _block_widths(40, 40, 40)
    budget = ab.BlockBudget(widths, dict(widths), 0.05)
    assert budget.spending() is False
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_a_zero_delta_is_not_recorded_as_spending_block_budget():
    """Carried over verbatim from BandBudget: a delta of exactly 0.0 gives
    nothing away, so recording it would make the wasted-bit share
    uninterpretable."""
    budget = ab.BlockBudget(_block_widths(40, 40, 40), _block_widths(8, 8, 8), 0.0)
    assert budget.spending() is True
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_an_unbounded_delta_is_recorded_as_spending_block_budget():
    """delta_rel=None is the accept-everything anchor and gives away the most
    of all, so it must count as spending."""
    budget = ab.BlockBudget(_block_widths(40, 40, 40), _block_widths(8, 8, 8), None)
    assert budget.delta_for_candidate() is None
    assert budget.spent_budget is True


def test_note_shed_narrows_one_features_width_and_moves_the_factor():
    """note_shed gains a feature argument because the factor needs the width
    MULTISET; the VALUE passed is unchanged from BandBudget's, since
    width = intervals - 1 makes dwidth == dintervals."""
    budget = ab.BlockBudget(_block_widths(40, 40, 40), _block_widths(8, 8, 8), 0.05)
    before = budget.factor()
    budget.note_shed(0, 32)
    assert budget.widths[0] == 8
    assert budget.factor() <= before


def test_crossed_compares_against_the_entry_factor():
    """crossed() is the rollback's question: did this run buy a block? The
    baseline is the factor at ENTRY, captured once, never the floor."""
    budget = ab.BlockBudget(_block_widths(40, 40, 40), _block_widths(8, 8, 8), 0.05)
    assert budget.crossed() is False
    budget.note_shed(0, 32)
    budget.note_shed(1, 32)
    budget.note_shed(2, 32)
    assert budget.factor() < ab._factor(_block_widths(40, 40, 40))
    assert budget.crossed() is True


def test_the_floor_factor_is_immutable_across_shedding():
    """Invariant 4: nothing alignment does can lower blocks_floor. The floor
    widths are copied at entry and never touched by note_shed."""
    floor = _block_widths(8, 8, 8)
    budget = ab.BlockBudget(_block_widths(40, 40, 40), floor, 0.05)
    frozen = budget._floor_factor
    budget.note_shed(0, 32)
    budget.note_shed(1, 32)
    assert budget._floor_factor == frozen


def test_the_budget_does_not_alias_the_callers_width_dict():
    """align_rf_thresholds reuses pooled_widths for feature_order and for
    stats; a budget mutating it in place would silently rewrite both."""
    caller = _block_widths(40, 40, 40)
    budget = ab.BlockBudget(caller, _block_widths(8, 8, 8), 0.05)
    budget.note_shed(0, 32)
    assert caller[0] == 40
