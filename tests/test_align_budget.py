import numpy as np
import pytest

from p4gen.build_p4_script import INFINITE
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


# The ladder (tables.codeword_to_blocks of the union key, the default alignment
# price until 2026-11-15), its per-tree variant 'ladder_pruned', the price=
# option, tree_multiplier and bits_to_reach / bits_to_next_byte / byte_width,
# and their tests, were deleted by Task 32 (G2: alignment prices with the
# crossbar port only). BlockBudget's tests went 2026-09-15 with the class.


def test_tree_factor_is_the_dearest_tree_keys_port_price():
    """factor_* are per-TREE since Task 32: the max port price over the tree
    keys, whatever the dict order; no keys, no factor."""
    from p4model.lanes import table_blocks
    widths = {'a': 11, 'b': 3, 'c': 84}
    keys = ((frozenset({'a'}), 2), (frozenset({'a', 'b', 'c'}), 1))
    assert ab.tree_factor(widths, keys) == max(table_blocks((11,)), table_blocks((3, 11, 84)))
    assert ab.tree_factor({'c': 84, 'b': 3, 'a': 11}, keys) == ab.tree_factor(widths, keys)
    assert ab.tree_factor(widths, ()) is None


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
    """The E1 sibling (design §7.4): alignment prices the tables the switch
    builds. Every fixture tree keys one key; total_blocks charges each tree
    its port price alone, production its allocation in its stage (a later
    table of the same key can reuse the earlier one's slots and cost less);
    on these 8 rows the two are equal.

    joint_low_sd12 is the live witness that range blocks are NOT
    alignment-invariant: ONE feature and TWO range blocks, so its interval
    count exceeds the 206-per-block step. That is Gap 1, in the fixture.
    """
    from tests.test_resource_model_golden import load_fixture, rebuild_pool
    from p4model.usage import assemble_usage

    checked = 0
    for row in load_fixture()['rows']:
        if not row['row_id'].startswith('joint'):
            continue
        pool = rebuild_pool(row)
        usage, _range_plan, _ternary_plan = assemble_usage(pool)
        widths = {i: bits for i, bits in enumerate(pool['ternary_key_bits'][0])}
        keys = ((frozenset(widths), sum(1 for _ in pool['ternary_table_specs'])),)
        assert ab.total_blocks(widths, keys) == usage.blocks, row['row_id']
        checked += 1
    assert checked == 8


def test_the_range_half_equals_the_generators_own_range_accounting():
    """§7.4's first clause, stated separately from the total.

    The total could be right by two errors cancelling. This pins the range term
    against tables.range_matching_resource_usage on the same merged intervals,
    so a drift in either half is attributable.
    """
    from p4model.tables import range_matching_resource_usage

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
        assert ab.total_blocks(widths, ()) == generator_blocks, counts


def test_tree_keys_counts_each_keys_512_row_words():
    """audit §8.1. A block is memory, charged once per TREE: per distinct
    split-feature set, the sum of its trees' ceil(leaves / 512)."""
    class _Tree:
        def __init__(self, feature, leaves):
            internal = 1 if feature is not None else 0
            self.tree_ = type('T', (), {
                'children_left': np.array([5] * internal + [-1] * leaves),
                'feature': np.array([feature if feature is not None else -2] * internal
                                    + [-2] * leaves)})()

    class _Forest:
        def __init__(self, *trees):
            self.estimators_ = [_Tree(*t) for t in trees]

    keys = ab.tree_keys(_Forest((0, 10), (0, 600)), _Forest((1, 30)))
    assert keys == ((frozenset({0}), 3), (frozenset({1}), 1))


def test_blocks_bought_by_finds_the_cheapest_block_buying_shed():
    """The quantity feature_order ranks on (Task 12). Returns what the shed
    BUYS and what it COSTS, so a feature that can buy nothing within its own
    room is (0, 0) rather than an arbitrary distance.
    """
    #  8 bits -> 1 byte; shedding to 0 frees that byte (6 bytes, 2 blocks -> 5, 1).
    widths = {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8}
    floors = {f: 0 for f in widths}
    keys = ((frozenset(widths), 4),)
    bought, spent = ab.blocks_bought_by(widths, floors, 0, keys)
    assert bought > 0 and spent > 0

    # No room at all: nothing can be bought at any price.
    assert ab.blocks_bought_by(widths, dict(widths), 0, keys) == (0, 0)


def test_blocks_bought_by_sees_a_range_step_a_byte_rule_cannot():
    """Gap 1's mechanism, as a unit. A feature at 207 intervals (width 206) is
    ONE interval from a free range block -- a saving no byte-completion rule
    can see, because 206 is not a byte boundary and no key price moves.
    """
    widths = {0: 206, 1: 8}
    floors = {0: 0, 1: 0}
    keys = ((frozenset({1}), 1),)
    bought, spent = ab.blocks_bought_by(widths, floors, 0, keys)
    assert (bought, spent) == (1, 1)


@pytest.mark.parametrize('key', [(3, 11, 84), (27, 52), (4, 19, 24, 53, 62)])
def test_port_price_is_the_ports_price(key):
    """port_price returns lanes.table_blocks (the production price) of the
    key, whatever the order."""
    from p4model.lanes import table_blocks
    widths = {i: w for i, w in enumerate(reversed(key))}
    one_tree = ((frozenset(widths), 1),)
    assert ab.tree_factor(widths, one_tree) == table_blocks(key)
    assert ab.port_price(key) == table_blocks(key)


def test_the_union_key_is_never_priced():
    """Interim review I1: the factor is the max per-tree price; the union key
    (here 9 x 60 bits, past what a key may be) is never priced."""
    from p4model.errors import CrossbarKeyTooWide
    from p4model.lanes import table_blocks
    widths = {f: 60 for f in range(9)}
    with pytest.raises(CrossbarKeyTooWide):
        ab.port_price(widths.values())
    keys = ((frozenset({0, 1}), 1), (frozenset({2, 3, 4}), 2))
    assert ab.tree_factor(widths, keys) == table_blocks((60, 60, 60))
    assert ab.total_blocks(widths, keys) > 0


def test_port_width_never_changes_the_ports_price():
    """port_width keeps every width's byte count and <= 4-bit-tail flag, which
    is all the port reads; checked against the uncached port."""
    import random
    from p4model import lanes
    rng = random.Random(9)
    for _ in range(200):
        key = tuple(sorted(rng.randint(1, 60) for _ in range(rng.randint(1, 8))))
        canon = tuple(sorted(ab.port_width(w) for w in key))
        assert (lanes._table_blocks.__wrapped__(key)
                == lanes._table_blocks.__wrapped__(canon)), key


def test_total_blocks_prices_each_tree_key_on_its_own():
    from p4model.lanes import table_blocks
    widths = {0: 30, 1: 9, 2: 3}
    keys = ((frozenset({0}), 2), (frozenset({0, 1, 2}), 1))
    ranges = sum(ab.range_blocks(w + 1) for w in widths.values())
    assert ab.total_blocks(widths, keys) == (
        ranges + 2 * table_blocks((30,)) + table_blocks((3, 9, 30)))


def test_a_stump_prices_at_the_empty_key_floor():
    from p4model.lanes import table_blocks
    assert ab.port_price(()) == table_blocks(()) == 1


# --- Task 30: TreeKeyPricer -- the plain scan's answers, fewer key prices ---

def _plain_bought_by(widths, floors, feature, keys):
    """blocks_bought_by as it was before Task 30: two total_blocks per probe."""
    room = widths[feature] - floors[feature]
    if room <= 0:
        return 0, 0
    before = ab.total_blocks(widths, keys)
    probe = dict(widths)
    for shed in range(1, room + 1):
        probe[feature] = widths[feature] - shed
        bought = before - ab.total_blocks(probe, keys)
        if bought > 0:
            return bought, shed
    return 0, 0


def _outcome(fn, *args):
    """fn's result, or its exception's type and message."""
    try:
        return fn(*args)
    except Exception as exc:  # the exception is part of the answer
        return type(exc).__name__, str(exc)


def _random_case(rng, n_features, n_keys, max_width):
    widths = {f: rng.randint(0, max_width) for f in range(n_features)}
    # A few widths near the range table's 206-interval step, so range terms move too.
    for f in rng.sample(range(n_features), 2):
        widths[f] = rng.choice((rng.randint(0, max_width), rng.randint(200, 212)))
    floors = {f: rng.randint(0, w) for f, w in widths.items()}
    keys = tuple((frozenset(rng.sample(range(n_features), rng.randint(1, n_features))),
                  rng.randint(1, 3)) for _ in range(n_keys))
    return widths, floors, keys


def test_tree_key_pricer_scans_exactly_as_the_plain_scan_across_steps():
    """One pricer kept across alignment-like steps (every feature scanned, then
    one feature's width drops) gives blocks_bought_by's plain answer every
    time, exceptions (an over-limit key) included."""
    import random
    rng = random.Random(30)
    raised = 0
    for case in range(25):
        widths, floors, keys = _random_case(rng, rng.randint(2, 9), rng.randint(1, 7),
                                            rng.choice((12, 40, 70)))
        pricer = ab.TreeKeyPricer(keys)
        for step in range(6):
            for f in sorted(widths):
                want = _outcome(_plain_bought_by, widths, floors, f, keys)
                got = _outcome(ab.blocks_bought_by, widths, floors, f, keys, pricer)
                assert got == want, (case, step, f)
                raised += isinstance(want[0], str)
            f = rng.choice(sorted(widths))
            widths[f] = max(floors[f], widths[f] - rng.randint(1, 12))
    assert raised > 0, 'no over-limit key: the exception path is untested'


def test_tree_key_pricer_reprices_only_the_trees_keying_a_changed_width(monkeypatch):
    """A step that changes one feature's width re-prices the trees that key it
    and no other; an unchanged second call prices nothing."""
    keys = ((frozenset({0, 1}), 1), (frozenset({2, 3}), 2), (frozenset({1}), 1))
    pricer = ab.TreeKeyPricer(keys)
    calls = []
    real = ab.port_price
    crossbar = ab.crossbar_blocks
    monkeypatch.setattr(ab, 'crossbar_blocks', lambda w: calls.append(tuple(w)) or crossbar(w))
    widths = {0: 30, 1: 9, 2: 17, 3: 3}
    first = list(pricer.prices(widths))
    assert len(calls) == 3
    assert first == [real(widths[f] for f in sorted(fs)) for fs, _ in keys]
    del calls[:]
    del calls[:]
    assert list(pricer.prices(dict(widths))) == first and calls == []
    widths[2] = 16                       # W container -> H container: tree 1 only
    second = list(pricer.prices(widths))
    assert calls == [(ab.port_width(3), 16)]
    assert second[0] == first[0] and second[2] == first[2]
    assert second[1] == real((3, 16))


def test_tree_key_pricer_prices_each_distinct_key_once():
    """Two trees with the same canonical key, and a width change that keeps
    every port_width, cost one port price, not three."""
    keys = ((frozenset({0, 1}), 1), (frozenset({2, 3}), 1))
    pricer = ab.TreeKeyPricer(keys)
    pricer.prices({0: 20, 1: 5, 2: 20, 3: 5})
    assert pricer.priced == 1
    assert ab.port_width(19) == ab.port_width(20)
    pricer.prices({0: 19, 1: 5, 2: 20, 3: 5})
    assert pricer.priced == 1


def test_port_floor_is_a_lower_bound_on_the_port():
    """port_floor never exceeds the uncached port (lanes._table_blocks), and
    _field_byte_counts counts the bytes xbar.field_key builds."""
    import random
    from p4model import lanes, xbar
    rng = random.Random(31)
    tight = 0
    for _ in range(400):
        key = tuple(sorted(rng.randint(1, rng.choice((8, 20, 40, 70)))
                           for _ in range(rng.randint(1, 12))))
        counts = [ab._field_byte_counts(w) for w in key]
        n, nib = sum(c for c, _ in counts), sum(c for _, c in counts)
        built = xbar.field_key(lanes.key_fields(key), {})
        assert n == len(built) and nib == sum(b.only_one_nibble() for b in built)
        try:
            price = lanes._table_blocks.__wrapped__(key)
        except Exception:
            continue
        floor = ab.port_floor(n, nib)
        assert floor is not None and floor <= price, key
        tight += floor == price
    assert tight > 100
    assert ab.port_floor(0, 0) == 0 and ab.port_floor(5, 0) == 1 and ab.port_floor(6, 0) == 2
    assert ab.port_floor(66 + 60, 0) is None
