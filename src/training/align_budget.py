"""Total-blocks arithmetic for cost-aware threshold alignment (C1).

The joint block cost alignment optimises is

    blocks = sum over features of range_blocks(intervals_f)
           + sum over distinct tree keys of rows_k * port_price(key_k's pooled widths)

-- see total_blocks, which is the whole objective. A tree key is the set of
features one tree splits on (tree_keys): production keys every joint tree with
only its own split features (decision 2.3, evaluation._tree_keys), each at its
POOLED width, and charges it rows x lanes.table_blocks of that key -- the
ternary crossbar port (xbar.alone_blocks of the key's pinned layout). Alignment
prices exactly that, so its own estimate and the production model can differ
only through stage sharing, which alignment does not see by design (G2, all
2,550 campaign_2026_11 joint-off DEV designs: the estimate equals production on
2492 / 2550, every miss a shared stage; model8/impl/g2_alignment.md).

PORT ONLY (G2, decided 2026-11-15, Task 32). Until then the default price was
"the ladder" -- tables.codeword_to_blocks, a byte-count rule of thumb, of the
UNION key of every feature, for every tree -- with the port and a per-tree
ladder ('ladder_pruned') as options. The ladder, its per-tree variant, the
union-key tree multiplier (tree_multiplier) and the bits_to_reach bound built
on the ladder are deleted; the alignment price is no longer a parameter. The
switch moved thresholds against the old default on 385 of the 2,550 designs and
production-priced twin blocks on 32 (sum 52,991 ladder vs 52,992 port, model).

factor_* statistics are per-TREE: the dearest tree key's port price
(tree_factor), never the union key's. The union key can be past a stage's
crossbar while every tree key fits (interim review I1), so it is never priced.

SUPERSEDED, 2026-09-14. This module used to claim range blocks were
alignment-invariant (206 intervals per block against real per-feature interval
counts of 25-50). joint_low_sd12 in tests/fixtures/resource_model_golden.json
is ONE feature costing TWO range blocks: at low k one feature carries the whole
forest's thresholds, and alignment's only move is to remove intervals -- so the
range term is exactly the one it can move. Audit 2026-09-13 Sec 2.

Imports from p4gen and p4model only, so both threshold_alignment and the replay
harness can use it without importing the mutation loop.

SPEED (Task 30, 2026-11-15). feature_order's shed scans go through
TreeKeyPricer, which keeps every tree key's price at the live widths and
re-prices only the keys whose keyed widths changed, prices a probe as the
difference on the keys that hold the probed feature, and skips probes whose
price inputs cannot have changed or that a byte-count lower bound on the port
(port_floor) shows buy nothing. Same answers, exceptions included
(tests/test_align_budget.py pins it against the plain total_blocks scan); the
crossbar port itself is untouched.
"""
import functools

from p4gen.build_p4_script import INFINITE
from p4gen.evaluation import (codeword_fields_to_bytes_from_bits,
                                  entries_across_trees_to_blocks)
from p4model.lanes import crossbar_blocks, layout
from p4model.ranges import compiler_range_rows
from p4model.tables import check_key_bits
from p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK


def port_width(width):
    """The smallest-information stand-in for a `width`-bit code field under
    the port: same byte count, same "tail of <= 4 bits" flag.

    lanes.layout gives a field its containers from ceil(w / 8) alone (every
    size boundary -- 8, 16, 32, and each W32 step above -- is a byte
    boundary), every slice but the top one is whole bytes, and the top slice's
    tail is w's own tail; the crossbar port reads a slice only as started
    bytes plus whether its last byte has <= 4 live bits (one nibble). So the
    port price of a key is a function of these two numbers per field, and
    mapping each width to 8 * (bytes - 1) + (4 or 8) lets lanes' cache hit on
    every width whose layout cannot differ -- the port is called only where
    the layout can change (byte boundaries, <= 4-bit tails). Pinned against
    the raw port on random keys by tests/test_align_budget.py.

    The stand-in has as many or MORE bits than `width`, so the live-bit check
    (tables.check_key_bits) is applied to the real widths, never to these
    (port_price, TreeKeyPricer._key)."""
    if width <= 0:
        return 0
    full = (width - 1) // 8
    tail = width - 8 * full
    return 8 * full + (4 if tail <= 4 else 8)


def port_price(widths):
    """lanes.table_blocks of an iterable of field widths: one tree key's
    per-word block price, CrossbarKeyTooWide where production refuses the key
    (p4c's two checks: the live bits of the real widths, then the crossbar fit
    of their port_width stand-ins, lanes.crossbar_blocks)."""
    widths = [int(w) for w in widths if w > 0]
    check_key_bits(sum(widths))
    return crossbar_blocks(tuple(sorted(port_width(w) for w in widths)))


def pooled_interval_count(ranges1, ranges2):
    """Intervals in the common refinement of two gap-free tilings of ONE
    feature.

    The per-feature counterpart of threshold_alignment.joint_interval_count,
    and the reason C1 is affordable: recomputing the whole joint count after
    every accepted move is O(total thresholds) and would be paid thousands of
    times per trial, while this is O(len(ranges1) + len(ranges2)) over lists
    that hold 25-50 entries at campaign scale.

    An interval's upper bound IS a split threshold (the tiling is gap-free, so
    interval i's hi is interval i+1's lo minus one); INFINITE terminates every
    tiling and is not a threshold, so it is excluded. n thresholds tile a
    feature into n + 1 intervals, hence the + 1.
    """
    bounds = {hi for _, hi in ranges1 if hi != INFINITE}
    bounds |= {hi for _, hi in ranges2 if hi != INFINITE}
    return len(bounds) + 1


def _pooled_widths(intervals1, intervals2):
    """Per-feature codeword width AFTER pooling, keyed as the inputs are.

    A common feature's width is the common refinement's interval count minus
    one; an exclusive feature keeps its own. This is the per-feature
    decomposition of what joint_interval_count totals.
    """
    common = set(intervals1) & set(intervals2)
    widths = {f: pooled_interval_count(intervals1[f], intervals2[f]) - 1
              for f in common}
    for source in (intervals1, intervals2):
        widths.update({f: len(v) - 1 for f, v in source.items()
                       if f not in common})
    return widths


def _own_floor_widths(intervals1, intervals2):
    """The per-feature width no alignment can go below.

    max(own1, own2) for a common feature -- perfect coincidence on every
    threshold is the best case, and alignment can never delete one of a
    model's own thresholds -- and the model's own width for an exclusive one.
    """
    common = set(intervals1) & set(intervals2)
    floors = {f: max(len(intervals1[f]), len(intervals2[f])) - 1
              for f in common}
    for source in (intervals1, intervals2):
        floors.update({f: len(v) - 1 for f, v in source.items()
                       if f not in common})
    return floors


def tree_factor(widths, keys):
    """The per-table block factor alignment reports (factor_*): the dearest
    tree key's port price at these pooled widths, max over `keys`
    (tree_keys), or None when there is no tree key. A per-TREE quantity, never
    the union key's price (see the module docstring)."""
    prices = [port_price(widths[f] for f in features) for features, _ in keys]
    return max(prices) if prices else None


def range_blocks(interval_count):
    """TCAM blocks p4c allocates for ONE feature's range table.

    Sized from the DECLARED interval count at COMPILE time, never from the
    expanded physical row count -- the distinction tables.range_deployment_overflow
    exists to keep apart. Delegates to compiler_range_rows rather than
    restating the quarter/worst-case rule: this module's whole 2026-09-07
    lesson is that a restated cost rule drifts. At this project's 16-bit keys
    a range table steps at 206 intervals.
    """
    return -(-compiler_range_rows(interval_count)
             // TERNARY_MATCHING_ENTRIES_PER_BLOCK)


_range_blocks_cached = functools.lru_cache(maxsize=4096)(range_blocks)


@functools.lru_cache(maxsize=None)
def _field_byte_counts(width):
    """(crossbar bytes, one-nibble bytes) of one `width`-bit code field in the
    pinned layout the port prices (lanes.layout; xbar.field_key's bytes: every
    started byte of every slice, one-nibble when it carries <= 4 live bits)."""
    n = nibbles = 0
    for _, used in layout(width):
        n += -(-used // 8)
        if used - 8 * ((used - 1) // 8) <= 4:
            nibbles += 1
    return n, nibbles


@functools.lru_cache(maxsize=None)
def port_floor(nbytes, nibbles):
    """A LOWER BOUND on the crossbar port's per-word blocks (xbar.alone_blocks)
    of any key of `nbytes` crossbar bytes of which `nibbles` carry <= 4 live
    bits, or None when no block count up to the stage's 12 groups admits it
    (then the key is priced, never bounded). Task 30; used only to skip probes
    that provably buy nothing, never as a price.

    Why it is a bound. An allocation xbar finds is priced by
    xbar.blocks_per_row = max(G, 2 * whole + half + 1), G groups touched, and
    `whole` / `half` midbytes holding a whole / one-nibble byte. Each group
    has 5 byte slots and each midbyte takes one byte (xbar's _free_mid places
    one byte per midbyte), there are 6 midbytes, and a half midbyte holds a
    one-nibble byte, so nbytes <= 5G + whole + half, half <= nibbles,
    whole + half <= 6. With b = that allocation's blocks, b >= G and
    b >= 2 * whole + half + 1, so (b, whole, half) satisfies the constraints
    below and the smallest b that does is <= the port's price. The bound drops
    every lane (container alignment) constraint, which can only lower it."""
    if nbytes <= 0:
        return 0
    for b in range(1, 13):
        best = 0
        for whole in range(0, min(6, (b - 1) // 2) + 1):
            half = max(0, min(nibbles, b - 1 - 2 * whole, 6 - whole))
            best = max(best, whole + half)
        if 5 * b + best >= nbytes:
            return b
    return None


class TreeKeyPricer:
    """Per-tree key port prices for ONE alignment run, kept across alignment
    steps (Task 30).

    `keys` is tree_keys(...) (((features, rows), ...)). The pricer holds every
    key's price at the widths it last saw and, on the next call, re-prices
    only the keys holding a feature whose width changed. A key's price is a
    function of its port_width classes (exact: see port_width), so every
    distinct canonical key is priced once per run (memo) and through
    lanes.crossbar_blocks, whose own cache stays in force; the live-bit check
    (tables.check_key_bits) runs on every key's real widths before the memo.

    bought_by is blocks_bought_by's scan, computed as a difference instead of
    two totals: a probe changes one feature f's width, so
    total(widths) - total(probe) = range term of f + sum over the keys holding
    f of rows * (price now - price at the probe) -- an integer identity, every
    other term cancels. A probe whose port_width and range blocks equal the
    previous probe's (or, for the first, the current width's) has the same
    total, so it cannot be the first block-buying shed and is skipped.

    Exceptions are those of the plain scan: the current prices are taken over
    every key in `keys` order on first use (total_blocks' order, so the same
    key raises first), a later call re-prices only keys whose widths changed
    (an unchanged key priced once already, at these very widths), and a probe
    prices the keys holding f in `keys` order, the order total_blocks would."""

    def __init__(self, keys, floors=True):
        self.features = [tuple(sorted(features)) for features, _ in keys]
        self.rows = [rows for _, rows in keys]
        self.by_feature = {}
        for i, features in enumerate(self.features):
            for f in features:
                self.by_feature.setdefault(f, []).append(i)
        self._floors = floors
        self._memo = {}
        self._seen = None            # the widths the current prices are for
        self._widths = [None] * len(self.features)
        self._prices = [None] * len(self.features)
        self._counts = [None] * len(self.features)
        self.priced = 0              # crossbar_blocks calls made (memo misses), for tests

    def _key(self, widths):
        """The price of one key from its raw widths (any order). The live-bit
        check reads the raw widths (port_width can only add bits), the memo the
        canonical ones."""
        widths = [int(w) for w in widths if w > 0]
        check_key_bits(sum(widths))
        key = tuple(sorted(port_width(w) for w in widths))
        price = self._memo.get(key)
        if price is None:
            price = crossbar_blocks(key)
            self.priced += 1
            self._memo[key] = price
        return price

    def prices(self, widths):
        """Every key's price at `widths`, re-pricing only keys whose keyed
        widths changed since the last call."""
        if self._seen is None:
            stale = range(len(self.features))
        else:
            seen = self._seen
            hit = set()
            for f, idx in self.by_feature.items():
                if widths[f] != seen[f]:
                    hit.update(idx)
            stale = sorted(hit)
        for i in stale:
            own = {f: int(widths[f]) for f in self.features[i]}
            self._prices[i] = self._key(own.values())
            self._widths[i] = own
            if self._floors:
                counts = [_field_byte_counts(port_width(w)) for w in own.values()]
                self._counts[i] = (sum(n for n, _ in counts), sum(b for _, b in counts))
        self._seen = {f: widths[f] for f in self.by_feature}
        return self._prices

    def bought_by(self, widths, floors, feature):
        """blocks_bought_by (same contract)."""
        room = widths[feature] - floors[feature]
        if room <= 0:
            return 0, 0
        now = self.prices(widths)
        width = widths[feature]
        holders = self.by_feature.get(feature, ())
        range_now = _range_blocks_cached(width + 1)
        last = (port_width(int(width)), range_now)
        for shed in range(1, room + 1):
            probe = width - shed
            cls, rng = port_width(int(probe)), _range_blocks_cached(probe + 1)
            if (cls, rng) == last:
                continue
            last = (cls, rng)
            bought = (self._bounded(now, holders, feature, probe, range_now - rng) if self._floors
                      else self._exact(now, holders, feature, probe, range_now - rng))
            if bought is not None and bought > 0:
                return bought, shed
        return 0, 0

    def _at_probe(self, i, feature, probe):
        own = self._widths[i]
        return self._key([probe if f == feature else w for f, w in own.items()])

    def _exact(self, now, holders, feature, probe, range_gain):
        """A probe's blocks bought, every holder priced (keys order)."""
        bought = range_gain
        for i in holders:
            bought += self.rows[i] * (now[i] - self._at_probe(i, feature, probe))
        return bought

    def _bounded(self, now, holders, feature, probe, range_gain):
        """A probe's blocks bought, or None when it provably buys nothing.

        port_floor bounds each holder's price at the probe from below, so
        `bound` = range_gain + sum rows * (now - floor) is >= the real bought.
        The holders whose floor is below their current price (the only ones
        that can add a block) are priced one at a time, keys order, each exact
        price replacing its floor in `bound`; as soon as `bound` <= 0 the probe
        cannot buy, and is left. Only if it still can are the other holders
        (now - price <= now - floor <= 0) priced, for the exact total. A holder
        with no floor (port_floor None) sends the probe to _exact."""
        add_n, add_nib = _field_byte_counts(port_width(int(probe)))
        floor = []
        for i in holders:
            n, nib = self._counts[i]
            old_n, old_nib = _field_byte_counts(port_width(self._widths[i][feature]))
            f = port_floor(n - old_n + add_n, nib - old_nib + add_nib)
            if f is None:
                return self._exact(now, holders, feature, probe, range_gain)
            floor.append(f)
        bound = range_gain + sum(self.rows[i] * (now[i] - f) for i, f in zip(holders, floor))
        if bound <= 0:
            return None
        exact = range_gain
        for i, f in zip(holders, floor):
            if now[i] > f:
                price = self._at_probe(i, feature, probe)
                bound -= self.rows[i] * (price - f)
                exact += self.rows[i] * (now[i] - price)
                if bound <= 0:
                    return None
        for i, f in zip(holders, floor):
            if now[i] <= f:
                exact += self.rows[i] * (now[i] - self._at_probe(i, feature, probe))
        return exact


def tree_keys(*forests):
    """Per DISTINCT set of features a tree splits on, the 512-row words its
    trees hold, ((frozenset(feature indices), rows), ...) in a deterministic
    order -- what one step of each tree key's price is worth, in blocks.

    A block is MEMORY, so it is charged once per TREE -- the exact opposite of
    the crossbar's byte slots, which a stage charges once however many tables
    read them (audit Sec 8.1). Without the rows a range step (1 block) and a
    ternary step could not be weighed against each other (audit Sec 8.2 item 6).

    Production keys every tree with only its own split features (decision
    2.3, evaluation._tree_keys), each at the POOLED width, so its price is
    rows * lanes.table_blocks(those widths). Rows are counted from leaves
    (children_left == -1), an UPPER BOUND on a tree's real entry count: the
    generator keys each tree's codewords by the codeword STRING, so two leaves
    with the identical codeword collapse into one entry; the two disagree only
    when a tree straddles a 512-entry boundary. Computed once at entry:
    alignment moves thresholds but never removes one, so a tree's split
    feature set and leaf count do not change."""
    rows = {}
    for forest in forests:
        for estimator in forest.estimators_:
            tree = estimator.tree_
            internal = tree.children_left != -1
            features = frozenset(int(f) for f in tree.feature[internal])
            rows[features] = rows.get(features, 0) + entries_across_trees_to_blocks(
                [int((~internal).sum())])
    return tuple(sorted(rows.items(), key=lambda item: sorted(item[0])))


def total_blocks(widths, keys):
    """Every TCAM block a joint design of these per-feature widths costs:
    range blocks plus, per distinct tree key of `keys` (tree_keys), rows x
    port_price of that key's pooled widths -- what production charges a joint
    tree standing alone.

    Option A of the audit's Sec 5, and THE quantity alignment optimises after
    2026-09-14. Both terms are blocks, so they ADD. A feature's width is its
    interval count minus one, hence the `+ 1`.

    Equals ResourceUsage.blocks wherever no two trees share a stage at a
    different price than alone (the stage simulation, which alignment does not
    see); measured on all 2,550 campaign_2026_11 joint-off DEV twins: equal on
    2492 (model8/impl/g2_alignment.md, Task 24).
    """
    ranges = sum(range_blocks(width + 1) for width in widths.values())
    return ranges + sum(rows * port_price(widths[f] for f in features)
                        for features, rows in keys)


def blocks_bought_by(widths, floors, feature, keys, pricer=None):
    """(blocks_bought, bits_spent) for the CHEAPEST block-buying shed on one
    feature, or (0, 0) when no shed within that feature's own room buys
    anything.

    The quantity feature_order ranks on. It is evaluated on the FULL width dict
    with this feature substituted, not on the feature alone: a key's port price
    depends on its whole width MULTISET (the crossbar allocates the key's bytes
    together), so no per-feature quantity can price a ternary step (design
    Sec 1.3).

    The scan is exact rather than closed-form because the two costs step for
    different reasons -- a byte completion or a freed nibble moves a key's
    price, a 206th interval moves a range block -- and the cheapest of the two
    is the answer. Bounded by this feature's own room (pooled width minus floor
    width).

    KNOWN BLIND SPOT (audit finding 2.4): the scan probes ONE feature's shed
    at a time, holding every other feature's width fixed. Two features that
    are each one bit over the same byte or range-block boundary can only buy
    that block TOGETHER -- shedding either alone leaves total_blocks
    unchanged, so both score (0, 0) here and feature_order falls back to its
    combined-interval-count tiebreak. This costs ranking quality only, not a
    lost block: the ranking only ORDERS features, it never skips a move, and
    every free move is accepted whichever feature comes first (spec 2026-09-28
    T3). Under the pre-alignment anchor the order can decide who spends the
    run's shared accuracy slack, which is feature_order's D1 caveat.

    The scan is TreeKeyPricer.bought_by (Task 30: the same answer as two
    total_blocks per probe, from the keys that hold `feature` only); `pricer`
    is the alignment run's TreeKeyPricer for `keys`, or None for a fresh one.
    """
    room = widths[feature] - floors[feature]
    if room <= 0:
        return 0, 0
    if pricer is None:
        pricer = TreeKeyPricer(keys)
    return pricer.bought_by(widths, floors, feature)


def pooled_key_bytes(intervals1, intervals2):
    """Crossbar byte width of the UNION key under joint encoding (a reported
    statistic, key_bytes_*; no tree is keyed on it since decision 2.3).

    MUST equal tables.codeword_fields_to_bytes on the joint intervals the
    generator emits from the same pooled thresholds -- required test E1.
    Delegates the byte-rounding rule rather than restating it (design Sec 5.1).
    """
    return codeword_fields_to_bytes_from_bits(
        _pooled_widths(intervals1, intervals2).values())


# DELETED 2026-11-15 (Task 32, G2): the ladder price ('ladder', the union key's
# tables.codeword_to_blocks for every tree, the default until then), its
# per-tree variant ('ladder_pruned', pruned_ladder_price), the price= option
# (PRICES / TREE_KEY_PRICES / prices_tree_keys / key_price), tree_multiplier
# (the union path's per-tree multiplier), _factor's union branch, and
# bits_to_reach / bits_to_next_byte / byte_width (a reported lower bound built
# on the ladder's crossbar_capacity; nothing read it for a decision).
#
# DELETED 2026-09-15: class BlockBudget. It decided, per candidate, whether
# alignment might spend accuracy while a cheaper block factor was still
# REACHABLE. Track 5's pre-registered live-Optuna trial returned delta_helps =
# FALSE (mean_d000 0.7956173344395895 vs mean_d020 0.7861922400433382,
# cells_favouring_d020 14/24), so there is no tolerance left to gate: alignment
# keeps the free moves and nothing else, and align_rf_thresholds carries the
# live width dict itself.
