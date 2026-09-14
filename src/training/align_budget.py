"""Block-factor arithmetic for cost-aware threshold alignment (C1).

The joint block cost is

    blocks = range_blocks + sum over trees of ceil(entries_t / 512) * factor
    factor = tables.codeword_to_blocks(sorted per-feature field widths)

Alignment changes neither entries_t nor the tree count, and range blocks are
alignment-invariant (a range table is sized by p4c at COMPILE time from the
declared interval count, giving 206 intervals per block at this project's
16-bit keys, against real per-feature interval counts of 25-50). So
minimising `factor` is exactly minimising blocks, and the whole objective
collapses to one small integer -- see _factor.

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
                                  codeword_to_blocks)


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


def pooled_key_bytes(intervals1, intervals2):
    """Crossbar byte width of one classification table under joint encoding.

    MUST equal evaluation.codeword_fields_to_bytes on the joint intervals the
    generator emits from the same pooled thresholds -- required test E1. If it
    does not, this budget prices a table the switch does not build.
    """
    return sum(byte_width(w) for w in _pooled_widths(intervals1, intervals2).values())


def key_bytes_floor(intervals1, intervals2):
    """The lowest key width ANY alignment at ANY delta could reach.

    Exact for the same reason codeword_floor is, and computed once at entry:
    nothing alignment does can move it.
    """
    return sum(byte_width(w) for w in _own_floor_widths(intervals1, intervals2).values())


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


class BlockBudget:
    """Decides, per candidate, whether alignment may spend accuracy.

    The C1 rule, repriced (design 2026-09-07): spend the configured tolerance
    only while a CHEAPER BLOCK FACTOR is still reachable, otherwise judge
    candidates at delta = 0.0 and keep collecting the free moves. Replaces
    BandBudget, which gated on `band_target(L) >= floor` -- a step function the
    hardware does not have wherever the crossbar arm binds, which is every
    many-feature design measured (§1.3: 0 of 14 high-k key sets).

    Carries the per-feature WIDTH DICT rather than a scalar length, and that is
    forced rather than stylistic: version_block_penalty depends on the width
    MULTISET, so no scalar can carry the cost (§1.3).

    Reachability is simpler here than it was in the band domain, not harder.
    BandBudget needed band_target -- an INVERSE of the step function. Here the
    floor width vector IS the best attainable case (alignment relocates a
    threshold but never deletes one, so a common feature's pooled width can
    never drop below max(own1, own2)), so `factor(current) > factor(floor)` is
    an exact reachability test with no inverse required. It is also the most
    permissive CORRECT gate: it stays open while any block remains
    theoretically reachable.

    Reachability is NECESSARY, not sufficient -- the candidate generator can
    still run dry before a block is bought, which is what
    threshold_alignment.align_with_policy's rollback exists to undo.
    """

    def __init__(self, pooled_widths, floor_widths, delta_rel):
        # Copied, not aliased: align_rf_thresholds keeps its own pooled_widths
        # for feature_order and for the stats, and a budget mutating it in
        # place would silently rewrite both.
        self.widths = dict(pooled_widths)
        self.delta_rel = delta_rel
        self.spent_budget = False
        # Invariant 4: computed once at entry, never updated. Nothing
        # alignment does can lower it.
        self._floor_factor = _factor(floor_widths)
        self._start_factor = _factor(self.widths)

    def factor(self):
        return _factor(self.widths)

    def spending(self):
        return self.factor() > self._floor_factor

    def delta_for_candidate(self):
        """The delta the NEXT candidate is judged by. Records that real budget
        was offered -- a delta of exactly 0.0 gives nothing away and does not
        count, or the wasted-bit share becomes uninterpretable. Semantics
        carried over verbatim from BandBudget."""
        if not self.spending():
            return 0.0
        if self.delta_rel is None or self.delta_rel > 0.0:
            self.spent_budget = True
        return self.delta_rel

    def note_shed(self, feature, bits):
        """Record an accepted move's realised shed on the ONE feature that
        moved, so the next reachability test sees the current widths.

        The feature argument is what the width dict needs; the VALUE is
        unchanged from BandBudget's, because width = intervals - 1 makes
        dwidth == dintervals == pooled_before - pooled_after, which
        _rank_targets already hands back.
        """
        self.widths[feature] -= bits

    def crossed(self):
        """Did this run buy a block? Compared against the factor at ENTRY --
        never against the floor, which is what could still be reached."""
        return self.factor() < self._start_factor
