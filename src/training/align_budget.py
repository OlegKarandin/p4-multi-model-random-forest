"""Total-blocks arithmetic for cost-aware threshold alignment (C1).

The joint block cost is

    blocks = sum over features of range_blocks(intervals_f)
           + factor(widths) * sum over trees of ceil(entries_t / 512)
    factor = tables.codeword_to_blocks(sorted per-feature field widths)   # the LADDER, not the production price

-- see total_blocks, which is the whole objective.

PRICES WITH THE LADDER (2026-10-04, spec Sec I.3.4 option b). The production
per-table price is lanes.table_blocks (since 2026-11-13 xbar.alone_blocks of
the key's pinned layout, the ternary crossbar port; the lane count from
2026-10-04 until then); alignment keeps the byte-count ladder
tables.codeword_to_blocks. Measured on 2,330 joint programs AGAINST THE LANE
COUNT (lane-era, 2026-10-04; not re-measured against the port): a
lane-priced alignment ends with different thresholds on 48 (2.1%), summed
total blocks 1,708 vs 1,706, accuracy differences +-0.001 that cancel, at
x1.42 the alignment time. align_stats' factor_* / total_blocks_* fields are
therefore LADDER quantities -- unless price='port' (decision 9.4, 2026-11-14,
off by default): then every total prices each tree's OWN key with the port
(tree_keys, port_price). G2 sample, 300 campaign_2026_11 joint-off DEV designs
(model): x1.34 (k<=8) / x1.80 (k>=9) the alignment time, production-priced
twin blocks equal on 297, summed blocks equal (model8/impl/g2_alignment.md).
price='ladder_pruned' (G2 option 3, 2026-11-15, off by default) keeps the
port path's per-tree structure but prices each tree key with the ladder
(pruned_ladder_price); no port call.

SUPERSEDED, 2026-09-14. This module used to claim range blocks were
alignment-invariant, on the grounds that a range table holds 206 intervals per
block against real per-feature interval counts of 25-50, and therefore that
minimising `factor` alone was exactly minimising blocks. The archive it cited
contradicts it: joint_low_sd12 in tests/fixtures/resource_model_golden.json is
ONE feature costing TWO range blocks. At low k a single feature carries the
whole forest's thresholds and clears 206 routinely, and alignment's only move
is to remove intervals -- so the term it was told to ignore is exactly the one
it can move. Audit 2026-09-13 §2.

SUPERSEDED, 2026-09-07. This module used to state the cost as
`range_blocks + n_trees * codeword_bits_to_blocks(L)` over the pooled
split-threshold count L, and gate spending on `band_target(L) >= floor`. That
identity is wrong wherever the ternary input crossbar binds, which is every
many-feature design measured: the crossbar allocates per key FIELD and
byte-rounds each one, so a 13-field 84-bit key really presents 16 bytes = 3
blocks where codeword_bits_to_blocks says 2. It looked right for years only
because on few wide fields byte-rounding is nearly a no-op.
codeword_bits_to_blocks no longer appears in this module at all -- band_ceiling
and band_target, its only callers here, were pruned once the block-domain
repair made them stats-only; codeword_bits_to_blocks itself (p4model/tables.py)
survives only as the empty-key floor codeword_to_blocks special-cases to, and
in src/reporting/replay_scoring.py's band_factor_before/after columns. It no
longer gates anything.

The byte-domain helpers below (byte_width, bits_to_next_byte, bits_to_reach,
pooled_key_bytes) are MORE central after the repair, not
less: blocks now live in the byte domain too. The stage-domain twins that
once sat beside them -- tables_per_stage, ternary_stages, stage_step_target,
StageBudget -- are gone, because classification-pool stages are derivable
from the block factor and step only when it steps (design 2026-09-07 §4.3,
pinned by tests/test_resource_model_golden.py).

Imports from p4gen only, so both threshold_alignment and the replay harness
can use it without importing the mutation loop.

SPEED (Task 30, 2026-11-15). Under the tree-key prices ('port',
'ladder_pruned') feature_order's shed scans go through TreeKeyPricer, which
keeps every tree key's price at the live widths and re-prices only the keys
whose keyed widths changed, prices a probe as the difference on the keys that
hold the probed feature, and skips probes whose price inputs cannot have
changed. Same answers, exceptions included (tests/test_align_budget.py pins it
against the plain total_blocks scan); the crossbar port itself is untouched.
"""
import functools

from p4gen.build_p4_script import INFINITE
from p4gen.evaluation import (codeword_fields_to_bytes_from_bits,
                                  codeword_to_blocks,
                                  entries_across_trees_to_blocks)
from p4model.lanes import layout, table_blocks
from p4model.ranges import compiler_range_rows
from p4model.tables import check_key_bytes
from p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK

# Decision 9.4 (2026-11-14): alignment may price keys with the crossbar port
# (lanes.table_blocks, the production per-table price) instead of the ladder.
# G2 option 3 (2026-11-15): 'ladder_pruned' has the port path's structure --
# each tree's OWN key at its pooled widths, times its 512-row words, plus range
# blocks -- but prices every key with the ladder (pruned_ladder_price).
# 'ladder' stays the default, so nothing changes unless a caller asks.
PRICES = ('ladder', 'port', 'ladder_pruned')
# The prices that charge each tree's own key (tree_keys), never the union key.
TREE_KEY_PRICES = ('port', 'ladder_pruned')


def _check_price(price):
    if price not in PRICES:
        raise ValueError('price must be one of {}, got {!r}'.format(PRICES, price))


def prices_tree_keys(price):
    """True when `price` charges each tree's own key (the multiplier is then
    tree_keys(...)), False for the union-key ladder."""
    _check_price(price)
    return price in TREE_KEY_PRICES


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
    the raw port on random keys by tests/test_align_budget.py."""
    if width <= 0:
        return 0
    full = (width - 1) // 8
    tail = width - 8 * full
    return 8 * full + (4 if tail <= 4 else 8)


def port_price(widths):
    """lanes.table_blocks of an iterable of field widths, via port_width."""
    return table_blocks(tuple(sorted(port_width(int(w)) for w in widths if w > 0)))


def pruned_ladder_price(widths):
    """The 'ladder_pruned' price of ONE tree key: tables.codeword_to_blocks
    (the ladder) of an iterable of field widths, with the production key limit.

    The ladder itself applies no key limit (it prices pooled keys past it, see
    codeword_to_blocks), so this applies the same refusal the port path does,
    tables.check_key_bytes (target.MAX_KEY_BYTES), before pricing: a tree key
    over the limit raises CrossbarKeyTooWide exactly where port_price would.
    The empty key (a stump) prices at the ladder's floor, as the port's does."""
    key = tuple(sorted(int(w) for w in widths if w > 0))
    check_key_bytes(codeword_fields_to_bytes_from_bits(key))
    return codeword_to_blocks(key)


def key_price(widths, price):
    """One tree key's per-word block price under a tree-key price."""
    if price == 'port':
        return port_price(widths)
    if price == 'ladder_pruned':
        return pruned_ladder_price(widths)
    raise ValueError('key_price prices tree keys; price must be one of {}, got {!r}'
                     .format(TREE_KEY_PRICES, price))


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


def byte_width(bits):
    """ceil(bits / 8) without importing math -- the crossbar allocates per
    FIELD, so every per-feature width is byte-rounded on its own before being
    summed (evaluation.codeword_fields_to_bytes)."""
    return -(-bits // 8)


def bits_to_next_byte(width):
    """Bits this feature must shed to free one whole crossbar byte.

    The byte-domain counterpart of "distance to the next band boundary", and
    the quantity that makes the two costs behave differently: codeword length
    is a plain sum, so a bit shed anywhere is worth the same, while key width
    is per-feature and quantised, so a bit is worth 0 or a whole byte
    depending only on where that feature's width sits modulo 8.

    Contract: width >= 1. A feature present in an interval dict was split on
    at least once, so it has at least two intervals and a width of at least 1;
    width 0 would return 8 here, which is meaningless rather than wrong -- a
    zero-width field costs no bytes and can shed nothing.
    """
    return ((width - 1) % 8) + 1


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


def _factor(widths, price='ladder', keys=None):
    """The per-table TCAM block factor of a pooled per-feature width dict.

    THE quantity alignment optimises after the 2026-09-07 repair, and the
    reason the objective collapses to one integer (design §3): under 'joint'
    encoding every classification table keys the same field set, so every table
    shares this factor, while alignment changes neither a tree's entry count
    nor the tree count. Minimising this minimises the LADDER's blocks; the
    price shipped on 2026-10-04 (lanes.table_blocks, then the lane count)
    agreed except on 13 of 2,330 post-alignment joint campaign keys (always
    lane lower) and 2 of 2,330 pre-alignment keys (lane higher). Lane-era
    measurement, not re-measured against the crossbar port (2026-11-13).

    Delegates to p4model rather than restating the rule, which is the point of
    the repair -- the superseded `blocks = range_blocks + n_trees *
    codeword_bits_to_blocks(L)` identity this module was built around drifted
    precisely because it was a restatement. Pinned by E1-blocks
    (tests/test_threshold_alignment.py).

    Sorted because codeword_to_blocks prices a MULTISET of field widths and
    documents its input as sorted (ternary_key_field_bits returns it that way).
    The dict's own key order is the generator's feature-emission order, and the
    crossbar allocator does not honour it.

    price='port' (decision 9.4) and price='ladder_pruned' (G2 option 3) never
    price the union key: production keys each joint tree with only its own
    split features, and a union key past the stage's MAX_KEY_BYTES crossbar
    bytes would raise CrossbarKeyTooWide although every tree's own key fits
    (interim review I1). They need `keys` (tree_keys) and return
    tree_factor(widths, keys, price).
    """
    if prices_tree_keys(price):
        if keys is None:
            raise ValueError("price={!r} prices tree keys: pass keys=tree_keys(...)"
                             .format(price))
        return tree_factor(widths, keys, price)
    return codeword_to_blocks(tuple(sorted(widths.values())))


def tree_factor(widths, keys, price):
    """A tree-key price's per-table block factor: the dearest tree key's
    key_price at these pooled widths, max over `keys` (tree_keys), or None
    when there is no tree key. Never prices the union key (see _factor)."""
    prices = [key_price((widths[f] for f in features), price) for features, _ in keys]
    return max(prices) if prices else None


def port_factor(widths, keys):
    """tree_factor under the port (decision 9.4)."""
    return tree_factor(widths, keys, 'port')


def range_blocks(interval_count):
    """TCAM blocks p4c allocates for ONE feature's range table.

    Sized from the DECLARED interval count at COMPILE time, never from the
    expanded physical row count -- the distinction tables.range_deployment_overflow
    exists to keep apart. Delegates to compiler_range_rows rather than
    restating the quarter/worst-case rule: this module's whole 2026-09-07
    lesson is that a restated cost rule drifts. At this project's 16-bit keys
    the ladder steps at 206 intervals.
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
    """Per-tree key prices for ONE alignment run under a tree-key price
    ('port' or 'ladder_pruned'), kept across alignment steps (Task 30).

    `keys` is tree_keys(...) (((features, rows), ...)). The pricer holds every
    key's price at the widths it last saw and, on the next call, re-prices
    only the keys holding a feature whose width changed. A key's price is a
    function of its CANONICAL widths -- port_width of each field under the
    port (exact: see port_width), the raw widths under the pruned ladder --
    so every distinct canonical key is priced once per run (memo) and through
    the same key_price as total_blocks, whose own caches (lanes.table_blocks)
    stay in force.

    bought_by is blocks_bought_by's scan, computed as a difference instead of
    two totals: a probe changes one feature f's width, so
    total(widths) - total(probe) = range term of f + sum over the keys holding
    f of rows * (price now - price at the probe) -- an integer identity, every
    other term cancels. A probe whose canonical width and range blocks equal
    the previous probe's (or, for the first, the current width's) has the same
    total, so it cannot be the first block-buying shed and is skipped.

    Exceptions are those of the plain scan: the current prices are taken over
    every key in `keys` order on first use (total_blocks' order, so the same
    key raises first), a later call re-prices only keys whose widths changed
    (an unchanged key priced once already, at these very widths), and a probe
    prices the keys holding f in `keys` order, the order total_blocks would."""

    def __init__(self, keys, price, floors=True):
        if not prices_tree_keys(price):
            raise ValueError('TreeKeyPricer prices tree keys; price must be one of {}, got {!r}'
                             .format(TREE_KEY_PRICES, price))
        self.price = price
        self.features = [tuple(sorted(features)) for features, _ in keys]
        self.rows = [rows for _, rows in keys]
        self.by_feature = {}
        for i, features in enumerate(self.features):
            for f in features:
                self.by_feature.setdefault(f, []).append(i)
        self._canon = port_width if price == 'port' else int
        # The port_floor bound is a bound on the crossbar port only.
        self._floors = price == 'port' and floors
        self._memo = {}
        self._seen = None            # the widths the current prices are for
        self._classes = [None] * len(self.features)
        self._prices = [None] * len(self.features)
        self._counts = [None] * len(self.features)
        self.priced = 0              # key_price calls made (memo misses), for tests

    def _key(self, canon_widths):
        """The price of one key from its canonical widths (any order)."""
        key = tuple(sorted(c for c in canon_widths if c > 0))
        price = self._memo.get(key)
        if price is None:
            price = key_price(key, self.price)
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
        canon = self._canon
        for i in stale:
            classes = {f: canon(int(widths[f])) for f in self.features[i]}
            self._prices[i] = self._key(classes.values())
            self._classes[i] = classes
            if self._floors:
                counts = [_field_byte_counts(c) for c in classes.values()]
                self._counts[i] = (sum(n for n, _ in counts), sum(b for _, b in counts))
        self._seen = {f: widths[f] for f in self.by_feature}
        return self._prices

    def bought_by(self, widths, floors, feature):
        """blocks_bought_by under this pricer's price (same contract)."""
        room = widths[feature] - floors[feature]
        if room <= 0:
            return 0, 0
        now = self.prices(widths)
        canon = self._canon
        width = widths[feature]
        holders = self.by_feature.get(feature, ())
        range_now = _range_blocks_cached(width + 1)
        last = (canon(int(width)), range_now)
        for shed in range(1, room + 1):
            probe = width - shed
            cls, rng = canon(int(probe)), _range_blocks_cached(probe + 1)
            if (cls, rng) == last:
                continue
            last = (cls, rng)
            bought = (self._bounded(now, holders, feature, cls, range_now - rng) if self._floors
                      else self._exact(now, holders, feature, cls, range_now - rng))
            if bought is not None and bought > 0:
                return bought, shed
        return 0, 0

    def _at_probe(self, i, feature, cls):
        classes = self._classes[i]
        return self._key([cls if f == feature else c for f, c in classes.items()])

    def _exact(self, now, holders, feature, cls, range_gain):
        """A probe's blocks bought, every holder priced (keys order)."""
        bought = range_gain
        for i in holders:
            bought += self.rows[i] * (now[i] - self._at_probe(i, feature, cls))
        return bought

    def _bounded(self, now, holders, feature, cls, range_gain):
        """A probe's blocks bought, or None when it provably buys nothing.

        port_floor bounds each holder's price at the probe from below, so
        `bound` = range_gain + sum rows * (now - floor) is >= the real bought.
        The holders whose floor is below their current price (the only ones
        that can add a block) are priced one at a time, keys order, each exact
        price replacing its floor in `bound`; as soon as `bound` <= 0 the probe
        cannot buy, and is left. Only if it still can are the other holders
        (now - price <= now - floor <= 0) priced, for the exact total. A holder
        with no floor (port_floor None) sends the probe to _exact."""
        add_n, add_nib = _field_byte_counts(cls)
        floor = []
        for i in holders:
            n, nib = self._counts[i]
            old_n, old_nib = _field_byte_counts(self._classes[i][feature])
            f = port_floor(n - old_n + add_n, nib - old_nib + add_nib)
            if f is None:
                return self._exact(now, holders, feature, cls, range_gain)
            floor.append(f)
        bound = range_gain + sum(self.rows[i] * (now[i] - f) for i, f in zip(holders, floor))
        if bound <= 0:
            return None
        exact = range_gain
        for i, f in zip(holders, floor):
            if now[i] > f:
                price = self._at_probe(i, feature, cls)
                bound -= self.rows[i] * (price - f)
                exact += self.rows[i] * (now[i] - price)
                if bound <= 0:
                    return None
        for i, f in zip(holders, floor):
            if now[i] <= f:
                exact += self.rows[i] * (now[i] - self._at_probe(i, feature, cls))
        return exact


def tree_multiplier(*forests):
    """How many blocks ONE step of the per-table block factor is worth.

    A block is MEMORY, so it is charged once per TREE -- the exact opposite of
    the crossbar's byte slots, which a stage charges once however many tables
    read them (audit §8.1).

    Counts leaves (children_left == -1), which is an UPPER BOUND on a tree's
    real entry count, not an exact one: the generator keys each tree's
    codewords in a dict by the codeword STRING (build_p4_script.py:540), so
    two leaves that happen to produce the identical codeword collapse into
    one entry there. len(codewords[tree]) <= this tree's leaf count always,
    with equality the common case; the two disagree only when a tree
    straddles a 512-entry boundary, where the leaves this over-counts could
    shift which side of that boundary -- and so which block-sharding step --
    the tree lands on.

    Computed ONCE at entry and never updated: alignment relocates thresholds
    and never deletes one from its own model, so it changes neither a tree's
    leaf count nor the tree count (invariant 4's sibling).

    Without this, a range step (1 block) and a ternary step (8-80 blocks across
    the golden fixture) cannot be weighed against each other at all -- audit
    §8.2 item 6.
    """
    return entries_across_trees_to_blocks(
        int((estimator.tree_.children_left == -1).sum())
        for forest in forests for estimator in forest.estimators_)


def tree_keys(*forests):
    """The port path's counterpart of tree_multiplier: per DISTINCT set of
    features a tree splits on, the 512-row words its trees hold,
    ((frozenset(feature indices), rows), ...) in a deterministic order.

    Production keys every tree with only its own split features (decision
    2.3, evaluation._tree_keys), each at the POOLED width, so its price is
    rows * lanes.table_blocks(those widths); the ladder path keeps pricing
    the union key. Rows are counted from leaves exactly as tree_multiplier
    counts them (an upper bound, same caveat). Computed once at entry:
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


def total_blocks(widths, multiplier, price='ladder'):
    """Every TCAM block a joint design of these per-feature widths costs.

    Option A of the audit's §5, and THE quantity alignment optimises after
    2026-09-14. Both terms are blocks, so they ADD -- this reintroduces no
    trade-off axis, unlike the retired blocks-vs-stages pair which lived in
    different domains.

    A feature's width is its interval count minus one, hence the `+ 1`.

    Equals ResourceUsage.blocks on every joint row of
    tests/fixtures/resource_model_golden.json (design §7.4) -- on the fixture,
    not in general: ResourceUsage prices with lanes.table_blocks and this with
    the ladder. Measured lane-era (2026-10-04, against the lane count, not
    re-measured against the crossbar port): they differed on 13 of 2,330
    post-alignment joint campaign keys (lane lower) and 2 of 2,330
    pre-alignment keys (lane higher).

    What this REPLACES is `_factor` alone, which was blind to range blocks
    entirely: a feature four intervals from a free range block could not open
    the budget gate, and could not outrank a feature one bit from a byte
    boundary that bought nothing (Gap 1).

    price='port' (decision 9.4): `multiplier` is then tree_keys(...) and the
    ternary term is sum over distinct tree keys of rows * port_price(that
    key's pooled widths) -- what production charges a joint tree.
    price='ladder_pruned' (G2 option 3): the same sum with each key priced by
    pruned_ladder_price (the ladder, plus the port path's key-limit refusal)
    -- production's structure, the ladder's per-key rule.
    """
    ranges = sum(range_blocks(width + 1) for width in widths.values())
    if prices_tree_keys(price):
        return ranges + sum(rows * key_price((widths[f] for f in features), price)
                            for features, rows in multiplier)
    return ranges + _factor(widths) * multiplier


def blocks_bought_by(widths, floors, feature, multiplier, price='ladder', pricer=None):
    """(blocks_bought, bits_spent) for the CHEAPEST block-buying shed on one
    feature, or (0, 0) when no shed within that feature's own room buys
    anything.

    The quantity feature_order ranks on. It is evaluated on the FULL width dict
    with this feature substituted, not on the feature alone, and that is forced
    rather than stylistic: codeword_to_blocks's isolation credit (Sec 2.3)
    depends on the width MULTISET -- one nibble-clean field's credit can move
    the whole table's price -- so no per-feature quantity can price a ternary
    step (design §1.3).

    The scan is exact rather than closed-form because the two ladders step for
    different reasons -- a byte completion moves the ternary factor, a 206th
    interval moves a range block -- and the cheapest of the two is the answer.
    Bounded by this feature's own room (pooled width minus floor width), one
    total_blocks evaluation per bit, each a pass over <= ~15 features. That is
    the same order as the `_factor` call it replaces, and negligible beside the
    model evaluation every candidate already pays for.

    KNOWN BLIND SPOT (audit finding 2.4): the scan probes ONE feature's shed
    at a time, holding every other feature's width fixed. Two features that
    are each one bit over the same byte or range-block boundary can only buy
    that block TOGETHER -- shedding either alone leaves total_blocks
    unchanged, so both score (0, 0) here and feature_order falls back to its
    combined-interval-count tiebreak instead of ranking them by the joint
    opportunity. This costs ranking quality only, not a lost block: the
    ranking only ORDERS features, it never skips a move, and every free move
    is accepted whichever feature comes first -- true under the old ratcheted
    guard and under the pre-alignment anchor alike (spec 2026-09-28 T3) -- so
    the block still gets bought once both features have been shed. (This
    used to credit "the budget gate", which was deleted with BlockBudget on
    2026-09-15 -- see the note at the end of this module.) Under the anchor
    the order can decide who spends the run's shared accuracy slack, which
    is feature_order's D1 caveat, not a lost block.

    Under a tree-key price the scan is TreeKeyPricer.bought_by (Task 30: the
    same answer from the keys that hold `feature` only); `pricer` is the
    alignment run's TreeKeyPricer for `multiplier`, or None for a fresh one.
    """
    room = widths[feature] - floors[feature]
    if room <= 0:
        return 0, 0
    if prices_tree_keys(price):
        if pricer is None:
            pricer = TreeKeyPricer(multiplier, price)
        return pricer.bought_by(widths, floors, feature)
    before = total_blocks(widths, multiplier, price)
    probe = dict(widths)
    for shed in range(1, room + 1):
        probe[feature] = widths[feature] - shed
        bought = before - total_blocks(probe, multiplier, price)
        if bought > 0:
            return bought, shed
    return 0, 0


def pooled_key_bytes(intervals1, intervals2):
    """Crossbar byte width of one classification table under joint encoding.

    MUST equal tables.codeword_fields_to_bytes on the joint intervals the
    generator emits from the same pooled thresholds -- required test E1. If it
    does not, this budget prices a table the switch does not build.

    Delegates the byte-rounding rule rather than restating it (design §5.1):
    this module's 2026-09-07 lesson was that a restated cost rule drifts.
    """
    return codeword_fields_to_bytes_from_bits(
        _pooled_widths(intervals1, intervals2).values())


def bits_to_reach(pooled_widths, own_floors, target_bytes):
    """Fewest bits that could bring sum(ceil(w/8)) down to target_bytes.

    A LOWER BOUND, not a prediction: it prices bits, and accept_alignment
    prices accuracy. A run needs at least this many admissible bits and
    generally more, because the cheapest bits are not necessarily the least
    damaging ones. Its use is negative -- when the bound already exceeds what
    any run plausibly sheds, the cell cannot win and need not be attempted.

    Why the answer is not a function of B alone: two pairs can both sit at
    B = 30 and cost 9 bits or 72 bits to reach B = 21, depending only on where
    each feature's width sits modulo 8.

    None means unreachable at any delta, and agrees exactly with
    sum(byte_width(w) for w in own_floors.values()) > target_bytes (required
    test E1c). Reported, never enforced: no run is skipped on this bound.
    """
    costs = []
    for feature, width in pooled_widths.items():
        room, shed = width - own_floors[feature], 0
        while True:
            step = bits_to_next_byte(width - shed)
            if shed + step > room:
                break
            costs.append(step)
            shed += step

    need = sum(byte_width(w) for w in pooled_widths.values()) - target_bytes
    if need <= 0:
        # Already at or below the target. Not reachable-for-free-by-luck: the
        # caller asked what it costs to get somewhere it already is. Guarding
        # here rather than slicing costs[:need] with a negative need, which
        # would drop the |need| DEAREST steps and return a positive cost.
        return 0
    costs.sort()
    return sum(costs[:need]) if need <= len(costs) else None


# DELETED 2026-09-15: class BlockBudget.
#
# It decided, per candidate, whether alignment might spend accuracy: offer the
# configured tolerance while a cheaper block factor was still REACHABLE
# (`factor(current) > factor(floor)`, exact because the floor width vector is
# the best attainable case), otherwise judge at delta = 0.0. It carried the live
# per-feature width dict, a `spent_budget` flag and `note_shed`, and its
# reachability test was necessary but not sufficient -- the candidate generator
# could run dry before a block was bought, which is what
# threshold_alignment.align_with_policy's rollback existed to undo.
#
# Track 5's pre-registered live-Optuna trial returned delta_helps = FALSE
# (mean_d000 0.7956173344395895 vs mean_d020 0.7861922400433382,
# cells_favouring_d020 14/24), so there is no tolerance left to gate: alignment
# keeps the free moves and nothing else. align_rf_thresholds now carries the
# live width dict itself, which is all the class did that anything still needs.
#
# _own_floor_widths / _factor / total_blocks survive -- they still feed
# factor_floor/total_blocks_floor in the stats and still underlie the width
# dicts feature_order and blocks_bought_by rank on. codeword_floor and
# key_bytes_floor -- the two convenience wrappers that summed
# _own_floor_widths into a single reported scalar -- were themselves pruned
# 2026-09-15 once they were confirmed to be stats-only (task 14: no caller
# read them for a decision, only stats['codeword_floor'] /
# stats['key_bytes_floor']).
