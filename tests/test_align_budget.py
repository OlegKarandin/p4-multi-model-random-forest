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


# band_ceiling, band_target, codeword_floor and key_bytes_floor and their
# tests were deleted 2026-09-15 (task 14): each survived only as a stats
# column (or, for band_ceiling, in src/reporting/replay_scoring.py's since-
# also-pruned legacy_band_wasted_bits), with no caller reading it for a
# decision. The arithmetic they wrapped -- _own_floor_widths, byte_width --
# has its own tests below.

# BandBudget's own spending/delta/shed/crossed unit tests were deleted here
# 2026-09-07 (gate repair): BlockBudget replaced it as the wired gate and
# carried the identical-shaped tests. BandBudget the CLASS was deleted from
# align_budget.py in a later commit of the same repair. BlockBudget and its
# eight tests then went the same way on 2026-09-15 -- see the note further
# down this file -- so there is no budget class in this module at all now.
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
    """E1c. `None` means unreachable at any delta, and must agree with the
    floor byte width (sum(byte_width(w) for w in floors.values())) exceeding
    target -- otherwise the bound and the floor would disagree about which
    cells can win at all."""
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
# BlockBudget's own eight tests stood here until 2026-09-15 -- spending(),
# delta_for_candidate()'s zero/None cases, spent_budget, note_shed, the
# single-crossing-test guard, the immutable floor factor and the no-aliasing
# guard. The class is gone: Track 5's pre-registered live-Optuna trial returned
# delta_helps = FALSE (mean_d000 0.7956173344395895 vs mean_d020
# 0.7861922400433382, cells_favouring_d020 14/24), so there is no accuracy
# tolerance left for a budget to gate, and align_rf_thresholds now keeps the
# live width dict itself.
#
# Nothing here replaces them, and nothing is left untested by their removal:
# the arithmetic they exercised is _factor / total_blocks / _own_floor_widths,
# which have their own tests above and below, and the live-width bookkeeping is
# pinned end-to-end by test_threshold_alignment.py's
# test_the_per_move_sheds_sum_to_the_whole_runs_shed.


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

    Holds on the fixture (no joint fixture key has lane price != ladder,
    checked 2026-10-04); the general relation is ladder >= lane on compiled
    keys, see align_budget's module docstring.
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
