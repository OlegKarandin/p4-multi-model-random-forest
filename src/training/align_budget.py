"""Block-band and ternary-stage arithmetic for cost-aware threshold alignment (C1).

The joint block cost is

    blocks = range_blocks + n_trees * band_factor(L)

where L is the pooled split-threshold count -- which IS the classification
table's codeword length, since generate_codewords emits exactly
len(intervals_f) - 1 bits per feature (build_p4_script.py:514/520). Pinned by
tests/test_evaluation.py::test_codeword_length_is_the_pooled_threshold_count.

band_factor only steps every TCAM_BLOCK_KEY_LENGTH bits, so a bit shed
mid-band buys nothing while still costing whatever accuracy the acceptance
tolerance gave away for it. 57.8% of the campaign's shed bits were of that
kind. This module holds the three quantities that decide whether the next band
is worth spending for, and deliberately nothing else: it imports from p4gen
only, so both threshold_alignment and the replay harness can use it without
importing the mutation loop.

The byte-domain functions below are the twin quantities for the ternary crossbar
budget, stepping on the 64-byte per-stage cap instead of the 44-bit block. The
key distinction from block-domain arithmetic is that key width is per-feature
and byte-quantised: a bit is worth 0 or a whole byte depending only on where
that feature's width sits modulo 8, not a constant value across the payload.
"""
from src.p4gen.build_p4_script import (INFINITE, TCAM_BLOCK_KEY_LENGTH,
                                       TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
                                       TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)
from src.p4gen.evaluation import CODEWORD_KEY_OVERHEAD_BITS, band_factor


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

    band_factor(L) == ceil((L + 4) / 44), so the largest L in a given band
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
    return band_ceiling(band_factor(codeword_length) - 1)


def byte_width(bits):
    """ceil(bits / 8) without importing math -- the crossbar allocates per
    FIELD, so every per-feature width is byte-rounded on its own before being
    summed (evaluation.ternary_table_key_bytes)."""
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


def tables_per_stage(key_bytes):
    """Classification tables sharing one stage at this key width.

    Both per-stage caps: the 64-byte crossbar budget, and the 8-table hard cap
    that binds at 8 bytes and narrower. Never returns 0 -- a key wider than a
    whole stage is rejected outright by evaluation.CrossbarKeyTooWide, not
    packed zero-per-stage, and callers divide by it.

    key_bytes <= 0 -- every classification table's key is genuinely empty,
    which happens whenever neither forest split on any feature at all (a
    real, reachable Optuna sample: e.g. min_samples_leaf close to n_samples
    yields single-leaf trees) -- costs nothing on the byte budget, so only
    the table-count cap binds. This is not a guess: crossbar_stages_needed's
    own load() (evaluation.py) computes width / TERNARY_CROSSBAR_MAX_BYTES_
    PER_STAGE, which is exactly 0 at width 0, leaving the
    1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE term as the only bound -- so
    this is the real packer's own treatment of a 0-byte key, not a policy
    invented here.
    """
    if key_bytes <= 0:
        return TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
    return min(TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
               max(1, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE // key_bytes))


def ternary_stages(key_bytes, n_tables):
    """Stages the classification pool occupies: ceil(T / tables_per_stage(B)).

    The byte-domain counterpart of band_factor -- the step function alignment
    is stepping down. Agrees with evaluation.crossbar_stages_needed on the
    uniform-width table lists this design produces (every classification table
    keys on the same set of per-feature fields and therefore shares one width,
    evaluation.py:324-326) PROVIDED the per-stage BLOCK cap is not the binding
    one; this function models the crossbar caps only, so where blocks bind it
    is a lower bound rather than an equality. Pinned both ways by
    tests/test_threshold_alignment.py's E1b tests.
    """
    return -(-n_tables // tables_per_stage(key_bytes))


def stage_step_target(key_bytes, n_tables):
    """Largest key width that strictly reduces ternary_stages, or None.

    Walks forward to the first fit that genuinely PAYS rather than aiming at
    the next packing step: at T=4, B=30 it returns 16 (fit 4), because the
    2->3 crossing leaves ceil(4/2) == ceil(4/3) == 2; at T=6, B=30 it returns
    21, because there the same crossing IS worth a stage. A T-blind
    64 // (tables_per_stage(B) + 1) returns 21 in both, authorising accuracy
    spending for a target that provably cannot change depth.

    None has one meaning everywhere: nothing this objective can buy. It covers
    the already-at-one-stage case (an absolute floor no byte shedding can
    breach) and the cap-exhausted case. Callers treat None as 'do not spend'.
    """
    now = ternary_stages(key_bytes, n_tables)
    if now <= 1:
        return None
    for fit in range(tables_per_stage(key_bytes) + 1,
                     TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE + 1):
        if -(-n_tables // fit) < now:
            return TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE // fit
    return None


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


def pooled_key_bytes(intervals1, intervals2):
    """Crossbar byte width of one classification table under joint encoding.

    MUST equal evaluation.ternary_table_key_bytes on the joint intervals the
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


class BandBudget:
    """Decides, per candidate, whether alignment may spend accuracy.

    The C1 rule: spend the configured tolerance only while the next-cheaper
    band is still reachable (`band_target(L) >= floor`), otherwise judge
    candidates at delta = 0.0 and keep collecting the free moves. Reachability
    is NECESSARY, not sufficient -- the candidate generator can still run dry
    before the band is crossed, which is what threshold_alignment's
    align_with_policy rollback exists to undo.
    """

    def __init__(self, codeword_length, floor, delta_rel):
        self.length = codeword_length
        self.floor = floor
        self.delta_rel = delta_rel
        self.spent_budget = False
        self._start_factor = band_factor(codeword_length)

    def spending(self):
        return band_target(self.length) >= self.floor

    def delta_for_candidate(self):
        """The delta the NEXT candidate is judged by. Records that real budget
        was offered -- a delta of exactly 0.0 gives nothing away and does not
        count, or S3's wasted-bit share would be uninterpretable."""
        if not self.spending():
            return 0.0
        if self.delta_rel is None or self.delta_rel > 0.0:
            self.spent_budget = True
        return self.delta_rel

    def note_shed(self, bits):
        """Record an accepted move's realised shed, so the next reachability
        test sees the current length."""
        self.length -= bits

    def crossed_band(self):
        return band_factor(self.length) < self._start_factor


class StageBudget:
    """Decides, per candidate, whether alignment may spend accuracy for a
    STAGE step -- the byte-domain twin of BandBudget.

    A separate object rather than a mode on BandBudget, because a pair can be
    one bit from a cheaper block band and twelve bytes from a cheaper stage
    step, or the reverse. It has no `gated` flag: BandBudget carried one only
    for the deleted 'legacy' policy, and this class never had that history.

    n_tables is constant for the whole run -- alignment relocates thresholds
    and never changes the forests -- so it is passed once and never updated,
    unlike key_bytes, which note_shed_bytes decrements so spending() can
    change state mid-run.
    """

    def __init__(self, key_bytes, floor, delta_rel, n_tables):
        self.key_bytes = key_bytes
        self.floor = floor
        self.delta_rel = delta_rel
        self.n_tables = n_tables
        self.spent_budget = False
        self._start_stages = ternary_stages(key_bytes, n_tables)

    def spending(self):
        target = stage_step_target(self.key_bytes, self.n_tables)
        return target is not None and target >= self.floor

    def delta_for_candidate(self):
        """The delta the NEXT candidate is judged by. Records that real budget
        was offered -- a delta of exactly 0.0 gives nothing away and does not
        count, on identical terms to BandBudget."""
        if not self.spending():
            return 0.0
        if self.delta_rel is None or self.delta_rel > 0.0:
            self.spent_budget = True
        return self.delta_rel

    def note_shed_bytes(self, n_bytes):
        """Record an accepted move's realised byte shed, so the next
        reachability test sees the current width."""
        self.key_bytes -= n_bytes

    def crossed_step(self):
        return ternary_stages(self.key_bytes, self.n_tables) < self._start_stages
