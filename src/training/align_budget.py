"""Total-blocks arithmetic for cost-aware threshold alignment (C1).

The joint block cost is

    blocks = sum over features of range_blocks(intervals_f)
           + factor(widths) * sum over trees of ceil(entries_t / 512)
    factor = tables.codeword_to_blocks(sorted per-feature field widths)

-- see total_blocks, which is the whole objective.

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
codeword_bits_to_blocks survives here as one ARM of codeword_to_blocks and
in src/reporting/replay_scoring.py's legacy columns; it no longer gates
anything.

The byte-domain helpers below (byte_width, bits_to_next_byte, bits_to_reach,
key_bytes_floor, pooled_key_bytes) are MORE central after the repair, not
less: blocks now live in the byte domain too. The stage-domain twins that
once sat beside them -- tables_per_stage, ternary_stages, stage_step_target,
StageBudget -- are gone, because classification-pool stages are derivable
from the block factor and step only when it steps (design 2026-09-07 §4.3,
pinned by tests/test_resource_model_golden.py).

Imports from p4gen only, so both threshold_alignment and the replay harness
can use it without importing the mutation loop.
"""
from src.p4gen.build_p4_script import INFINITE, TCAM_BLOCK_KEY_LENGTH
from src.p4gen.evaluation import (CODEWORD_KEY_OVERHEAD_BITS,
                                  codeword_bits_to_blocks,
                                  codeword_fields_to_bytes_from_bits,
                                  codeword_to_blocks,
                                  entries_across_trees_to_blocks)
from src.p4model.ranges import compiler_range_rows
from src.p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK


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


def codeword_floor(intervals1, intervals2):
    """The lowest codeword length ANY alignment of this pair could reach.

    Exact and invariant, not a heuristic. Alignment only ever relocates a
    threshold, never deletes one from its own model
    (threshold_alignment.py:106-111), so each model's own per-feature interval
    count is constant for the whole run and a common feature's pooled
    threshold set can never drop below the larger of the two. Perfect
    coincidence on every common feature is therefore the floor.

    Computed once at entry: nothing alignment does can move it.
    """
    # sum over common features of max(own1, own2) - 1, plus each exclusive
    # feature's own count - 1: exactly _own_floor_widths, in the bit domain.
    return sum(_own_floor_widths(intervals1, intervals2).values())


def band_ceiling(factor):
    """The highest codeword length that still fits in `factor` key blocks.

    codeword_bits_to_blocks(L) == ceil((L + 4) / 44), so the largest L in a given band
    satisfies L + 4 <= 44 * factor. src/reporting/replay_scoring.py uses this
    to price overshoot: bits shed below the ceiling of the band a run actually
    landed in bought nothing.
    """
    return TCAM_BLOCK_KEY_LENGTH * factor - CODEWORD_KEY_OVERHEAD_BITS


def band_target(codeword_length):
    """The highest codeword length that sits one band cheaper than this one.

    In the first band the result is negative, which correctly makes
    `target >= floor` False for every non-negative floor -- there is no
    cheaper band to reach.
    """
    return band_ceiling(codeword_bits_to_blocks(codeword_length) - 1)


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


def _factor(widths):
    """The per-table TCAM block factor of a pooled per-feature width dict.

    THE quantity alignment optimises after the 2026-09-07 repair, and the
    reason the objective collapses to one integer (design §3): under 'joint'
    encoding every classification table keys the same field set, so every table
    shares this factor, while alignment changes neither a tree's entry count
    nor the tree count. Minimising this IS minimising blocks.

    Delegates to p4model rather than restating the rule, which is the point of
    the repair -- the superseded `blocks = range_blocks + n_trees *
    codeword_bits_to_blocks(L)` identity this module was built around drifted
    precisely because it was a restatement. Pinned by E1-blocks
    (tests/test_threshold_alignment.py).

    Sorted because codeword_to_blocks prices a MULTISET of field widths and
    documents its input as sorted (ternary_key_field_bits returns it that way).
    The dict's own key order is the generator's feature-emission order, and the
    crossbar allocator does not honour it.
    """
    return codeword_to_blocks(tuple(sorted(widths.values())))


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


def total_blocks(widths, multiplier):
    """Every TCAM block a joint design of these per-feature widths costs.

    Option A of the audit's §5, and THE quantity alignment optimises after
    2026-09-14. Both terms are blocks, so they ADD -- this reintroduces no
    trade-off axis, unlike the retired blocks-vs-stages pair which lived in
    different domains.

    A feature's width is its interval count minus one, hence the `+ 1`.

    Equals ResourceUsage.blocks exactly on every joint row of
    tests/fixtures/resource_model_golden.json (design §7.4). Without that
    invariant this prices a table the switch does not build -- the exact
    failure mode the superseded band_factor identity had.

    What this REPLACES is `_factor` alone, which was blind to range blocks
    entirely: a feature four intervals from a free range block could not open
    the budget gate, and could not outrank a feature one bit from a byte
    boundary that bought nothing (Gap 1).
    """
    return (sum(range_blocks(width + 1) for width in widths.values())
            + _factor(widths) * multiplier)


def blocks_bought_by(widths, floors, feature, multiplier):
    """(blocks_bought, bits_spent) for the CHEAPEST block-buying shed on one
    feature, or (0, 0) when no shed within that feature's own room buys
    anything.

    The quantity feature_order ranks on. It is evaluated on the FULL width dict
    with this feature substituted, not on the feature alone, and that is forced
    rather than stylistic: version_block_penalty depends on the width MULTISET,
    so no per-feature quantity can price a ternary step (design §1.3).

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
    budget gate itself evaluates the full width dict once both features have
    actually been shed, not this function's per-feature probe, so the block
    still gets bought when both candidates are eventually accepted -- just
    possibly after spending accuracy budget on a feature that, ranked
    correctly, would not have gone first.
    """
    room = widths[feature] - floors[feature]
    if room <= 0:
        return 0, 0
    before = total_blocks(widths, multiplier)
    probe = dict(widths)
    for shed in range(1, room + 1):
        probe[feature] = widths[feature] - shed
        bought = before - total_blocks(probe, multiplier)
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


def key_bytes_floor(intervals1, intervals2):
    """The lowest key width ANY alignment at ANY delta could reach.

    Exact for the same reason codeword_floor is, and computed once at entry:
    nothing alignment does can move it.
    """
    return codeword_fields_to_bytes_from_bits(
        _own_floor_widths(intervals1, intervals2).values())


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
    key_bytes_floor > target_bytes (required test E1c). Reported, never
    enforced: no run is skipped on this bound.
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
# codeword_floor / _own_floor_widths / _factor / total_blocks survive -- the
# floor is still reported in the stats and still ranks features in
# feature_order.
