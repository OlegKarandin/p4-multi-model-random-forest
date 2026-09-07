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


# BandBudget's own spending/delta/shed/crossed unit tests were deleted here
# 2026-09-07 (gate repair): BlockBudget replaces it as the wired gate, and
# BlockBudget carries the identical-shaped tests below (search "BlockBudget
# (design 2026-09-07"), so keeping both would be duplication, not coverage.
# BandBudget the CLASS is untouched and still importable -- only the wiring
# moved -- so nothing here is testing dead code by omission.
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


def test_tables_per_stage_applies_both_per_stage_caps():
    """64 // key_bytes alone would return 9 at 7 bytes, disagreeing with
    crossbar_stages_needed exactly in the narrow-key cells the design says
    matter most."""
    assert ab.tables_per_stage(35) == 1
    assert ab.tables_per_stage(32) == 2
    assert ab.tables_per_stage(21) == 3
    assert ab.tables_per_stage(16) == 4
    assert ab.tables_per_stage(64) == 1
    assert ab.tables_per_stage(7) == 8            # the 8-table cap, not 9
    assert ab.tables_per_stage(8) == 8
    assert ab.tables_per_stage(4) == 8


def test_tables_per_stage_never_returns_zero():
    """Callers divide by it. A key wider than a whole stage is rejected
    outright by evaluation.CrossbarKeyTooWide, not packed zero-per-stage."""
    assert ab.tables_per_stage(65) == 1
    assert ab.tables_per_stage(200) == 1


def test_tables_per_stage_handles_a_zero_byte_key():
    """Regression: a forest that never split at all (a real, reachable Optuna
    sample -- e.g. min_samples_leaf close to n_samples yields single-leaf
    trees) gives pooled_key_bytes == 0, and a bare `64 // key_bytes` raises
    ZeroDivisionError. A 0-byte key costs nothing on the crossbar byte
    budget, so only the table-count cap should bind -- exactly what
    evaluation.crossbar_stages_needed's own load() computes for a 0-width
    table (its byte term is 0 / 64 == 0, leaving only 1 / 8)."""
    from src.p4gen.evaluation import crossbar_stages_needed
    assert ab.tables_per_stage(0) == 8
    for n_tables in (1, 4, 8, 9, 20):
        assert ab.ternary_stages(0, n_tables) == \
            crossbar_stages_needed([(1, 0)] * n_tables).occupied


def test_ternary_stages_is_the_byte_domain_step_function():
    assert ab.ternary_stages(30, 4) == 2
    assert ab.ternary_stages(21, 4) == 2
    assert ab.ternary_stages(16, 4) == 1
    assert ab.ternary_stages(30, 6) == 3
    assert ab.ternary_stages(21, 6) == 2


def test_stage_step_target_is_tree_count_aware():
    """The correction the amendment exists for: a T-blind
    64 // (tables_per_stage(B) + 1) returns 21 in every row below, authorising
    accuracy spending at T=4 for a target that provably cannot change depth."""
    assert ab.stage_step_target(30, 4) == 16      # NOT 21: ceil(4/2) == ceil(4/3)
    assert ab.stage_step_target(30, 6) == 21      # there the same crossing pays
    assert ab.stage_step_target(30, 9) == 21
    assert ab.stage_step_target(35, 6) == 32      # the design's worked cell
    assert ab.stage_step_target(11, 40) == 10


def test_stage_step_target_is_none_when_nothing_can_be_bought():
    """None has one meaning everywhere: nothing this objective can buy.
    Already-at-one-stage is an absolute floor no byte shedding can breach;
    at 8 bytes and below the 8-table cap is exhausted."""
    assert ab.stage_step_target(11, 4) is None    # already one stage
    assert ab.stage_step_target(8, 40) is None    # cap exhausted
    assert ab.stage_step_target(7, 40) is None


@pytest.mark.parametrize('key_bytes', range(1, 65))
@pytest.mark.parametrize('n_tables', range(1, 21))
def test_stage_step_target_returns_the_largest_paying_width(key_bytes, n_tables):
    """Property: the target is the LARGEST B that pays, so one byte wider must
    not pay. Anything smaller would spend accuracy buying bytes that were not
    needed."""
    target = ab.stage_step_target(key_bytes, n_tables)
    now = ab.ternary_stages(key_bytes, n_tables)
    if target is None:
        for candidate in range(1, key_bytes):
            assert ab.ternary_stages(candidate, n_tables) == now
    else:
        assert target < key_bytes
        assert ab.ternary_stages(target, n_tables) < now
        assert ab.ternary_stages(target + 1, n_tables) == now


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


def test_a_stage_budget_spends_while_a_paying_step_is_reachable():
    budget = ab.StageBudget(key_bytes=35, floor=20, delta_rel=0.2, n_tables=6)
    assert budget.spending()              # target 32 >= floor 20
    assert budget.delta_for_candidate() == 0.2
    assert budget.spent_budget is True


def test_a_stage_budget_stops_spending_when_the_floor_blocks_the_step():
    budget = ab.StageBudget(key_bytes=35, floor=33, delta_rel=0.2, n_tables=6)
    assert not budget.spending()          # target 32 < floor 33
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_a_stage_budget_declines_when_no_step_pays_at_all():
    """The zero-payoff case the T-awareness correction exists for: at one
    stage nothing can be bought, and at 8 bytes the table cap is exhausted."""
    one_stage = ab.StageBudget(key_bytes=11, floor=1, delta_rel=0.2, n_tables=4)
    assert not one_stage.spending()
    assert one_stage.delta_for_candidate() == 0.0
    cap_exhausted = ab.StageBudget(key_bytes=8, floor=1, delta_rel=0.2, n_tables=40)
    assert not cap_exhausted.spending()


def test_a_zero_delta_is_not_recorded_as_spending_stage_budget():
    budget = ab.StageBudget(key_bytes=35, floor=1, delta_rel=0.0, n_tables=6)
    assert budget.delta_for_candidate() == 0.0
    assert budget.spent_budget is False


def test_an_unbounded_delta_is_recorded_as_spending_stage_budget():
    budget = ab.StageBudget(key_bytes=35, floor=1, delta_rel=None, n_tables=6)
    assert budget.delta_for_candidate() is None
    assert budget.spent_budget is True


def test_shedding_bytes_retargets_and_flips_crossed_step():
    """note_shed_bytes must change spending() state mid-run, exactly as
    BandBudget.note_shed does -- that is what makes the OR of the two gates
    worth having."""
    budget = ab.StageBudget(key_bytes=35, floor=1, delta_rel=0.2, n_tables=6)
    assert not budget.crossed_step()
    budget.note_shed_bytes(3)             # 35 -> 32, fit 1 -> 2, stages 6 -> 3
    assert budget.key_bytes == 32
    assert budget.crossed_step()
    assert budget.spending()              # next target is 21, still >= floor 1


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
    single leaf. tables_per_stage's docstring already records it as a real
    Optuna sample."""
    from src.p4model.tables import ternary_block_factor
    assert ab._factor({}) == ternary_block_factor(())


# ---------------------------------------------------------------------------
# BlockBudget (design 2026-09-07 §4.1). Widths chosen so the crossbar arm
# binds: 5 fields of 9 bits are 10 crossbar bytes (2 blocks) against a
# 45-bit band (band_factor 2) -- shedding one field to 8 bits drops it to
# 9 bytes and the factor to 2... so the fixtures below state their own
# arithmetic rather than relying on intuition about where a step lands.

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
