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
    codeword_to_blocks takes a sorted tuple, so the sort belongs here rather
    than at every call site."""
    from src.p4model.tables import codeword_to_blocks
    widths = {'a': 11, 'b': 3, 'c': 84}
    assert ab._factor(widths) == codeword_to_blocks((3, 11, 84))


def test_factor_ignores_the_dicts_own_key_order():
    """The dict's order is the generator's feature-emission order. The crossbar
    allocator is free to place fields where it likes and measurably does, so
    codeword_to_blocks prices a MULTISET -- emission order must never reach
    the cost model."""
    assert ab._factor({'a': 11, 'b': 3}) == ab._factor({'b': 3, 'a': 11})


def test_factor_of_an_empty_width_dict_is_the_empty_key_factor():
    """Reachable, not hypothetical: neither forest split on any feature, which
    happens whenever min_samples_leaf approaches n_samples and every tree is a
    single leaf -- a real Optuna sample, not a synthetic corner case."""
    from src.p4model.tables import codeword_to_blocks
    assert ab._factor({}) == codeword_to_blocks(())


# ---------------------------------------------------------------------------
# BlockBudget (design 2026-09-07 §4.1). Widths below (3 fields of 40 bits
# each, vs. a floor of 8 bits each) put both models comfortably past a block
# boundary in codeword_to_blocks -- band and crossbar arms agree exactly at
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


def test_the_budget_has_only_one_crossing_test():
    """Audit §8.2 item 8. BlockBudget.crossed() was dead and computed the same
    thing as crossed_a_boundary(stats); two copies of one predicate is how the
    superseded band gate survived a repair."""
    assert not hasattr(ab.BlockBudget, 'crossed')


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


def test_range_blocks_steps_at_the_measured_interval_capacity():
    """compiler_range_rows, delegated not restated. A range table is sized by
    p4c at COMPILE time from the DECLARED interval count, giving 206 intervals
    per block at this project's 16-bit keys -- never from the expanded physical
    row count, which answers a different question and lives in
    range_deployment_overflow.
    """
    assert ab.range_blocks(1) == 1
    assert ab.range_blocks(206) == 1
    assert ab.range_blocks(207) == 2


def test_total_blocks_equals_the_assembled_usage_on_every_joint_fixture_row():
    """The E1 sibling (design §7.4), and the prerequisite for trusting Option A
    at all: without it, alignment prices a table the switch does not build --
    the exact failure mode the superseded band_factor identity had.

    Joint rows only. A disjoint pool carries two models' differently-sized
    tables and two distinct key sets, so a single width dict cannot describe
    it; those rows are "not applicable", never "predicted wrong".

    joint_low_sd12 is the live witness that range blocks are NOT
    alignment-invariant: ONE feature and TWO range blocks, so its interval
    count exceeds the 206-per-block ladder. That is Gap 1, in the fixture.
    """
    from tests.test_resource_model_golden import load_fixture, rebuild_pool
    from src.p4model.usage import assemble_usage

    checked = 0
    for row in load_fixture()['rows']:
        if not row['row_id'].startswith('joint'):
            continue
        pool = rebuild_pool(row)
        usage, _range_plan, _ternary_plan = assemble_usage(pool)
        widths = {i: bits for i, bits in enumerate(pool['ternary_key_bits'][0])}
        multiplier = sum(1 for _ in pool['ternary_table_specs'])
        assert ab.total_blocks(widths, multiplier) == usage.blocks, row['row_id']
        checked += 1
    assert checked == 8


def test_the_range_half_equals_the_generators_own_range_accounting():
    """§7.4's first clause, stated separately from the total.

    The total could be right by two errors cancelling. This pins the range term
    against tables.range_matching_resource_usage on the same merged intervals,
    so a drift in either half is attributable.
    """
    from src.p4model.tables import range_matching_resource_usage
    from src.p4gen.build_p4_script import INFINITE

    # A gap-free tiling of n intervals, which is what the generator emits.
    def tiling(n):
        return [(i * 10, i * 10 + 9) for i in range(n - 1)] + [((n - 1) * 10,
                                                                INFINITE)]

    for counts in ([3, 40], [207], [206, 25, 300]):
        intervals = {'f{}'.format(i): tiling(n) for i, n in enumerate(counts)}
        _entries, generator_blocks, _specs = range_matching_resource_usage(
            intervals)
        widths = {name: len(rows) - 1 for name, rows in intervals.items()}
        assert sum(ab.range_blocks(w + 1)
                   for w in widths.values()) == generator_blocks, counts


def test_the_tree_multiplier_is_what_one_factor_step_is_worth():
    """audit §8.1. A block is memory, charged once per TREE. At this project's
    tree sizes every tree costs exactly one block-row, so the multiplier is the
    two forests' tree counts added together -- but the ceil is what stops that
    coincidence being baked in.
    """
    class _Tree:
        def __init__(self, leaves):
            self.tree_ = type('T', (), {
                'children_left': np.array([-1] * leaves)})()

    class _Forest:
        def __init__(self, *leaf_counts):
            self.estimators_ = [_Tree(n) for n in leaf_counts]

    assert ab.tree_multiplier(_Forest(10, 20), _Forest(30)) == 3
    assert ab.tree_multiplier(_Forest(513), _Forest(10)) == 3


def test_blocks_bought_by_finds_the_cheapest_block_buying_shed():
    """The quantity feature_order ranks on (Task 12). Returns what the shed
    BUYS and what it COSTS, so a feature that can buy nothing within its own
    room is (0, 0) rather than an arbitrary distance.
    """
    #  8 bits -> 1 byte; shedding to 0 frees that byte.
    widths = {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8}
    floors = {f: 0 for f in widths}
    bought, spent = ab.blocks_bought_by(widths, floors, 0, multiplier=4)
    assert bought > 0 and spent > 0

    # No room at all: nothing can be bought at any price.
    assert ab.blocks_bought_by(widths, dict(widths), 0, multiplier=4) == (0, 0)


def test_blocks_bought_by_sees_a_range_step_a_byte_rule_cannot():
    """Gap 1's mechanism, as a unit. A feature at 207 intervals (width 206) is
    ONE interval from a free range block -- a saving no byte-completion rule
    can see, because 206 is not a byte boundary and the ternary factor does not
    move.
    """
    widths = {0: 206, 1: 8}
    floors = {0: 0, 1: 0}
    bought, spent = ab.blocks_bought_by(widths, floors, 0, multiplier=1)
    assert (bought, spent) == (1, 1)
