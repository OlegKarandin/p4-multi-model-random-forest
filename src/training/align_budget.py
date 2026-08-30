"""Block-band arithmetic for cost-aware threshold alignment (C1).

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
"""
from src.p4gen.build_p4_script import INFINITE, TCAM_BLOCK_KEY_LENGTH
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
    common = set(intervals1) & set(intervals2)
    total = sum(max(len(intervals1[f]), len(intervals2[f])) - 1 for f in common)
    total += sum(len(v) - 1 for f, v in intervals1.items() if f not in common)
    total += sum(len(v) - 1 for f, v in intervals2.items() if f not in common)
    return total


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
