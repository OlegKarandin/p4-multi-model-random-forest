import random

import pytest

from src.p4gen.build_p4_script import INFINITE
from src.training import align_targets as at
from src.training import threshold_alignment as ta
from src.training.errors import AlignmentInvariantError
from tests.test_threshold_alignment import _golden_alignment_pair as _pair


def test_candidate_targets_lists_the_four_corners_intersection_first():
    """Intersection first so a pair with one admissible candidate reproduces
    the legacy choice exactly."""
    assert at.candidate_targets((41, 96), (33, 88)) == [
        (41, 88), (33, 96), (41, 96), (33, 88)]


def test_candidate_targets_dedupes_coincident_corners():
    assert at.candidate_targets((10, 20), (10, 30)) == [(10, 20), (10, 30)]
    assert at.candidate_targets((10, 20), (10, 20)) == [(10, 20)]


def test_boundary_moves_mirrors_the_sentinel_guards():
    """adjust_range_boundaries silently declines to move a boundary sitting ON
    a sentinel (0 or INFINITE), so a corner that asks only for sentinel moves
    is a no-op and must be dropped before it costs an oracle evaluation."""
    assert at.boundary_moves((41, 96), (33, 88)) == [(40, 32), (96, 88)]
    assert at.boundary_moves((0, 96), (33, 88)) == [(96, 88)]      # min on 0
    assert at.boundary_moves((41, INFINITE), (33, 88)) == [(40, 32)]
    assert at.boundary_moves((0, INFINITE), (33, 88)) == []
    assert at.boundary_moves((41, 96), (41, 96)) == []


def test_boundary_moves_agrees_with_adjust_range_boundaries(monkeypatch):
    """The predicate and the mutator must not drift: whatever
    adjust_range_boundaries actually writes is what boundary_moves must
    predict, over random pairs."""
    rf1, X1, y1, rf2, X2, y2 = _pair()
    index = ta.build_threshold_index(rf1)
    intervals = ta.extract_feature_intervals(rf1)
    feature = sorted(intervals)[0]
    ranges = intervals[feature]

    rng = random.Random(0)
    for _ in range(200):
        source = rng.choice(ranges)
        target = rng.choice(ranges)
        predicted = {old for old, _ in at.boundary_moves(source, target)}
        mods = ta.adjust_range_boundaries(rf1, feature, source, target, index)
        ta.restore_thresholds(rf1, mods)
        assert {old for _, _, old in mods} == predicted


def _threshold_index_for(ranges):
    """A threshold_index that actually reflects `ranges`' own boundaries.

    An empty dict makes update_neighboring_ranges_and_index's own
    update_threshold_index call raise AlignmentInvariantError for a "missing
    key" reason on almost every real boundary rewrite (measured: 307/500
    trials with `{}`) -- the same exception type as the inversion raise, so
    `except AlignmentInvariantError` cannot tell the two apart and the
    `except Exception: continue` branch below never fires. Populating the
    index with the target's own thresholds (the only ones this function ever
    looks up) isolates the raise to genuine inversions, which is what this
    wiring test is actually about."""
    idx = {}
    for lo, hi in ranges:
        t_min = lo - 1 if lo > 0 else lo
        if t_min != 0:
            idx[(0, t_min)] = [(0, 0)]
        if hi != INFINITE:
            idx[(0, hi)] = [(0, 0)]
    return idx


def test_target_admissible_agrees_with_whether_the_mutator_raises():
    """A wiring test -- they share neighbour_writes, so disagreement means one
    of them stopped calling it."""
    rng = random.Random(1)
    for _ in range(500):
        n = rng.randint(2, 6)
        cuts = sorted(rng.sample(range(1, 400), n - 1))
        ranges, lo = [], 0
        for cut in cuts:
            ranges.append((lo, cut))
            lo = cut + 1
        ranges.append((lo, INFINITE))

        idx = rng.randrange(len(ranges))
        old = ranges[idx]
        new = (rng.randrange(0, 400), rng.randrange(0, 400))

        admissible = at.target_admissible(list(ranges), idx, old, new)
        try:
            ta.update_neighboring_ranges_and_index(
                list(ranges), idx, old, new, 0, _threshold_index_for(ranges))
            raised = False
        except AlignmentInvariantError:
            raised = True
        except Exception:
            continue      # threshold-index bookkeeping, not the inversion path
        assert admissible != raised


def test_neighbour_writes_mutates_nothing():
    ranges = [(0, 10), (11, 20), (21, INFINITE)]
    snapshot = list(ranges)
    at.neighbour_writes(ranges, 1, (11, 20), (5, 20))
    assert ranges == snapshot


def test_a_widening_target_shrinks_its_neighbour_and_can_invert_it():
    """Union and mixed corners widen the aligned interval, so inversions are a
    real and expected outcome for them -- unlike the intersection, which only
    ever widens neighbours."""
    ranges = [(0, 10), (11, 20), (21, INFINITE)]
    assert at.target_admissible(ranges, 1, (11, 20), (5, 20))     # neighbour -> (0, 4)
    assert not at.target_admissible(ranges, 1, (11, 20), (0, 20))  # neighbour -> (0, -1)
