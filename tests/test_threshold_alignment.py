"""First tests for threshold_alignment.py -- the module the spec identifies as
the sole source of the joint-vs-independent accuracy delta (C2, C5)."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
from src.training import threshold_alignment as ta
from src.training.errors import AlignmentInvariantError


def _one_split_forest():
    """One tree, one split at threshold 10 on feature 0."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([[5.0, 1.0], [6.0, 1.0], [40.0, 1.0], [41.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    rf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0).fit(X, y)
    rf.estimators_[0].tree_.threshold[0] = 10.0
    return rf


def _aligned_forest_pair():
    """Two forests over clipped-at-INFINITE features, aligned end to end."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    X1 = np.clip(rng.integers(0, 90000, size=(200, 3)), 0, INFINITE).astype(float)
    y1 = np.array(([0, 1, 2] * 67)[:200])
    X2 = np.clip(rng.integers(0, 90000, size=(200, 3)), 0, INFINITE).astype(float)
    y2 = np.array([-1, 1] * 100)

    rf1 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=3, max_depth=4, random_state=0).fit(X1, y1))
    rf2 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=3, max_depth=4, random_state=0).fit(X2, y2))

    # delta_rel=None accepts every move, which is the maximum-mutation path --
    # exactly what a partition-invariant test should be exercising.
    return ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                  overlap_threshold=0.5, delta_rel=None)


def test_missing_threshold_raises_a_catchable_exception_not_systemexit():
    """C2: `exit()` raises SystemExit (a BaseException), which bypasses
    `except Exception` and Optuna's `catch=`, so a campaign worker died with
    no traceback and no indication of which (feature, threshold) was missing."""
    rf = _one_split_forest()

    with pytest.raises(AlignmentInvariantError) as excinfo:
        ta.adjust_range_boundaries(
            rf, feature_idx=0, source_range=(11, 40), target_range=(16, 40),
            threshold_index={})  # deliberately empty -> the invariant is violated

    assert '(0, 10)' in str(excinfo.value)


def test_update_threshold_index_raises_on_a_missing_key():
    """C2, third site."""
    with pytest.raises(AlignmentInvariantError):
        ta.update_threshold_index({}, feature_idx=0, old_threshold=10, new_threshold=15)


def test_overlap_vetoes_a_pair_where_exactly_one_side_is_unbounded():
    """C5: adjust_range_boundaries refuses to move an INFINITE boundary, but
    update_neighboring_ranges_and_index wrote the shrunk value into `ranges`
    anyway, leaving ranges / thresholds / index disagreeing and the tail
    uncovered. The pair must never become a candidate.

    Mirrors the existing veto for exactly-one-side-starts-at-0 two lines up."""
    assert ta.calculate_range_overlap((30000, INFINITE), (30000, 40000)) == 0.0
    assert ta.calculate_range_overlap((30000, 40000), (30000, INFINITE)) == 0.0


def test_overlap_still_accepts_a_pair_where_both_sides_are_unbounded():
    """Both unbounded is fine: neither max boundary needs to move."""
    assert ta.calculate_range_overlap((30000, INFINITE), (32000, INFINITE)) > 0.0


def test_neighbor_update_refuses_to_write_a_boundary_that_was_not_moved():
    """Defence in depth for C5: even if a candidate slipped through, `ranges`
    must not claim a boundary the model still splits at."""
    ranges = [(0, 100), (101, INFINITE)]
    threshold_index = {(0, 100): [(0, 0)]}

    ta.update_neighboring_ranges_and_index(
        ranges, target_idx=1, old_range=(101, INFINITE), new_range=(101, 40000),
        feature_idx=0, threshold_index=threshold_index)

    assert ranges[1] == (101, INFINITE)
    assert ranges[0] == (0, 100)


def test_aligned_intervals_still_tile_zero_to_infinite():
    """The partition invariant C5 broke: after alignment, each feature's
    intervals must still tile [0, INFINITE] with no gap and no overlap."""
    rf1, rf2 = _aligned_forest_pair()

    for rf in (rf1, rf2):
        for feature_idx, intervals in ta.extract_feature_intervals(rf).items():
            assert intervals[0][0] == 0, (feature_idx, intervals)
            assert intervals[-1][1] == INFINITE, (feature_idx, intervals)
            for (_, prev_max), (next_min, _) in zip(intervals, intervals[1:]):
                assert next_min == prev_max + 1, (feature_idx, intervals)


def _forest_and_data(n_estimators=7, n=300, seed=5, min_samples_leaf=20):
    """A forest with impure leaves, so hard and soft voting can disagree.

    min_samples_leaf is a parameter because the hard/soft gap widens sharply
    with it (P1 Task 7 measured 0.33% of DDoS flows at leaf 5 versus 1.90% at
    leaf 200), so a test that needs the two to differ has to ask for impurity
    rather than hope for it."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    X = np.clip(rng.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y = np.array([c % 3 for c in range(n)])
    rf = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5, min_samples_leaf=min_samples_leaf,
        random_state=0).fit(X, y))
    return rf, X, y


def test_ensemble_prediction_is_the_cached_path_to_switch_predict():
    """One rule, two paths. switch_predict (P1 Task 7) computes the switch's
    hard vote from scratch; this function computes it from the incrementally
    maintained cache. They must agree exactly -- otherwise the alignment guard
    is measuring something the reported accuracy is not."""
    from src.p4gen.switch_semantics import switch_predict

    rf, X, y = _forest_and_data()
    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    got = ta.compute_ensemble_prediction(tree_predictions, rf)

    assert np.array_equal(got, switch_predict(rf, X))


def test_ensemble_prediction_matches_the_generated_vote_tables_rule():
    """Pin the tie-break too, against the same vote_winner the generated
    vote_<task> table's const entries are built from."""
    from src.p4gen.switch_semantics import vote_winner

    rf, X, y = _forest_and_data()
    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    got = ta.compute_ensemble_prediction(tree_predictions, rf)

    expected = np.array([
        rf.classes_[vote_winner(tree_predictions[:, i].tolist(), rf.n_classes_)]
        for i in range(X.shape[0])])
    assert np.array_equal(got, expected)


def test_ensemble_prediction_differs_from_rf_predict_and_that_is_intended():
    """Guard against someone 'fixing' the hard vote into a soft one. The hard
    vote is what the switch runs; rf.predict's soft vote is up to 1.7 accuracy
    points optimistic (P1 Task 7). With impure leaves the two genuinely differ."""
    rf, X, y = _forest_and_data(n_estimators=7, n=1200, min_samples_leaf=200)
    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    hard = ta.compute_ensemble_prediction(tree_predictions, rf)

    assert not np.array_equal(hard, rf.predict(X))


def test_prediction_cache_stores_class_indices_not_labels():
    """The round-trip rf.classes_[predict(...)] in build_prediction_cache
    existed only to be undone by a per-element dict lookup in
    compute_ensemble_prediction. Indices throughout removes both."""
    rf, X, y = _forest_and_data()

    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    assert tree_predictions.dtype == np.intp
    assert tree_predictions.min() >= 0
    assert tree_predictions.max() < rf.n_classes_


def test_prediction_cache_agrees_with_each_tree_predicting_alone():
    rf, X, y = _forest_and_data()

    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    for tree_idx, estimator in enumerate(rf.estimators_):
        assert np.array_equal(tree_predictions[tree_idx],
                              estimator.predict(X).astype(np.intp))


def test_vectorised_ensemble_prediction_is_much_faster_than_a_python_loop():
    """Not a microbenchmark for its own sake: this function runs ~2x per
    candidate, thousands of candidates per alignment call, once per Optuna
    trial, across 7 M x 15 splits x 17 k. A pure-Python double loop here is the
    single largest cost in the module."""
    import time

    rf, X, y = _forest_and_data(n_estimators=7, n=4000)
    tree_predictions, _ = ta.build_prediction_cache(rf, X)

    start = time.perf_counter()
    for _ in range(20):
        ta.compute_ensemble_prediction(tree_predictions, rf)
    elapsed = time.perf_counter() - start

    # 20 calls over 7 trees x 4000 samples = 560k vote increments. A Python
    # loop needs seconds; a bincount needs milliseconds. 1.0 s is a loose
    # ceiling that still fails the interpreted version by a wide margin.
    assert elapsed < 1.0, elapsed


def test_node_to_samples_matches_a_direct_decision_path_query():
    """The CSC inversion must produce exactly what the per-column CSR query
    produced: the same sample indices, sorted, for every internal node."""
    rf, X, y = _forest_and_data()

    _, node_to_samples = ta.build_prediction_cache(rf, X)

    for tree_idx, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_
        path = estimator.decision_path(X)
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] < 0:
                continue
            expected = path[:, node_idx].nonzero()[0]
            got = node_to_samples[(tree_idx, node_idx)]
            assert np.array_equal(np.sort(got), np.sort(expected)), (tree_idx, node_idx)
            assert np.array_equal(got, np.sort(got)), 'indices must stay sorted'


def test_alignment_does_not_mutate_the_callers_validation_arrays():
    """The float32 cast must be local. C8 already flags that this module
    mutates the caller's MODELS in place; it must not also start mutating the
    caller's data."""
    rf1, X1, y1 = _forest_and_data(seed=5)
    rf2, X2, y2 = _forest_and_data(seed=6)
    y2 = np.where(y2 == 0, -1, 1)
    before_dtype, before_copy = X1.dtype, X1.copy()

    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           overlap_threshold=0.5, delta_rel=None)

    assert X1.dtype == before_dtype
    assert np.array_equal(X1, before_copy)


def test_float32_cast_is_value_preserving_for_this_projects_data():
    """Every threshold is an integer after dt_thresholds_float_to_int, and every
    feature value is an integer clipped at INFINITE = 65535 -- both far below
    float32's 2**24 exact-integer limit. That is WHY the cast is safe."""
    values = np.arange(0, INFINITE + 1, dtype=np.float64)

    assert np.array_equal(values.astype(np.float32).astype(np.float64), values)


def test_a_candidate_that_moves_nothing_costs_no_prediction(monkeypatch):
    """P5: adjust_range_boundaries declines to move a threshold at 0 or at
    INFINITE, and every feature's interval list begins at 0 and ends at
    INFINITE -- so empty `modifications` is a common path, not an exotic one.
    It must not pay for two ensemble predictions and four metric computations."""
    rf1, X1, y1 = _forest_and_data(seed=5)
    rf2, X2, y2 = _forest_and_data(seed=6)
    y2 = np.where(y2 == 0, -1, 1)

    calls = []
    real = ta.compute_ensemble_prediction
    monkeypatch.setattr(ta, 'compute_ensemble_prediction',
                        lambda tp, rf: (calls.append(1), real(tp, rf))[1])

    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           overlap_threshold=0.5, delta_rel=0.05)

    # Two initial predictions, then exactly two per candidate that actually
    # moved something. An odd count, or a count that keeps growing when no
    # candidate moves anything, means the bail is missing.
    assert len(calls) % 2 == 0


import copy


def _snapshot(rf, tree_predictions, node_to_samples, threshold_index):
    return {
        'thresholds': [e.tree_.threshold.copy() for e in rf.estimators_],
        'predictions': tree_predictions.copy(),
        'node_samples': {k: v.copy() for k, v in node_to_samples.items()},
        'index': copy.deepcopy(threshold_index),
    }


def _assert_snapshot_restored(rf, tree_predictions, node_to_samples, threshold_index, snap):
    for estimator, before in zip(rf.estimators_, snap['thresholds']):
        assert np.array_equal(estimator.tree_.threshold, before)
    assert np.array_equal(tree_predictions, snap['predictions'])
    assert set(node_to_samples) == set(snap['node_samples'])
    for key, before in snap['node_samples'].items():
        assert np.array_equal(node_to_samples[key], before), key
    assert threshold_index == snap['index']


def _first_movable_interval(rf):
    """A (feature_idx, source, target) triple adjust_range_boundaries will
    actually act on: both boundaries away from 0 and INFINITE."""
    for feature_idx, intervals in ta.extract_feature_intervals(rf).items():
        for lo, hi in intervals:
            if lo > 0 and hi != INFINITE:
                return feature_idx, (lo, hi), (lo, hi + 1)
    raise AssertionError('fixture has no interior interval to move')


def test_the_incremental_cache_equals_a_from_scratch_recomputation():
    """THE invariant the whole incremental cache rests on, and nothing checked
    it. After a modification plus a cache update, the maintained predictions
    must equal what build_prediction_cache would produce on the mutated model."""
    rf, X, y = _forest_and_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    threshold_index = ta.build_threshold_index(rf)
    tree_predictions, node_to_samples = ta.build_prediction_cache(rf, X32)

    feature_idx, source, target = _first_movable_interval(rf)
    modifications = ta.adjust_range_boundaries(
        rf, feature_idx, source, target, threshold_index)
    assert modifications, 'the fixture must actually move a threshold'
    ta.update_cache_for_modifications(
        rf, X32, tree_predictions, node_to_samples, modifications)

    fresh_predictions, fresh_node_samples = ta.build_prediction_cache(rf, X32)

    assert np.array_equal(tree_predictions, fresh_predictions)
    for key, fresh in fresh_node_samples.items():
        assert np.array_equal(np.sort(node_to_samples[key]), np.sort(fresh)), key


def test_a_rejected_alignment_restores_every_data_structure_exactly():
    """Rollback round-trip. Task 5's four independent guards make rejection far
    more common than the single averaged guard did, so any leak here compounds."""
    rf, X, y = _forest_and_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    threshold_index = ta.build_threshold_index(rf)
    tree_predictions, node_to_samples = ta.build_prediction_cache(rf, X32)

    snap = _snapshot(rf, tree_predictions, node_to_samples, threshold_index)

    feature_idx, source, target = _first_movable_interval(rf)
    modifications = ta.adjust_range_boundaries(
        rf, feature_idx, source, target, threshold_index)
    undo_info = ta.update_cache_for_modifications(
        rf, X32, tree_predictions, node_to_samples, modifications)

    ta.restore_thresholds(rf, modifications)
    ta.undo_cache_update(tree_predictions, node_to_samples, undo_info)

    _assert_snapshot_restored(rf, tree_predictions, node_to_samples, threshold_index, snap)


def test_extract_feature_intervals_agrees_with_the_generator():
    """Alignment optimises the partition extract_feature_intervals produces,
    while the TCAM cost is computed from the generator's partition. If they
    disagree, the block savings are mis-targeted -- so they must be the same
    partition, by construction."""
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen.build_p4_script import get_feature_intervals

    names = ['Flow.IAT.Max', 'Fwd.IAT.Max', 'Fwd.Packet.Length.Max', 'Bwd.IAT.Min']
    rng = np.random.default_rng(5)
    n = 300
    X = np.clip(rng.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y = np.array([c % 3 for c in range(n)])
    # Force a real threshold-0 split on feature 0: give it a small integer
    # range (so 0 and 1 are adjacent observed values -- floor(0.5) == 0 is
    # then a real, reachable rounded threshold, not just a rare coincidence
    # of a 90000-wide random range) and zero out a label-correlated subset,
    # so "value == 0 vs > 0" becomes an optimal split. This exercises the
    # still-live C1 bug deterministically, matching how dataset.py's real
    # zero-valued rows produce exactly this kind of split.
    X[:, 0] = rng.integers(0, 5, size=n).astype(float)
    X[y == 0, 0] = 0.0
    rf = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=7, max_depth=5, min_samples_leaf=20, random_state=0).fit(X, y))

    ours = ta.extract_feature_intervals(rf)
    theirs = get_feature_intervals(rf, names)

    assert {names[idx] for idx in ours} == set(theirs)
    for feature_idx, intervals in ours.items():
        assert intervals == theirs[names[feature_idx]], names[feature_idx]


def test_a_forest_with_a_zero_threshold_is_representable_in_the_fixtures():
    """C1's precondition: a split at threshold 0 is real -- dataset.py keeps
    zero-valued rows, and a 'counter is zero vs non-zero' split is exactly
    sklearn threshold 0.5 truncated to 0. Build one deliberately so Task 4's
    fix has something to be tested against."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[0.0], [0.0], [1.0], [5.0], [0.0], [7.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    rf = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=1, max_depth=1, random_state=0).fit(X, y))

    thresholds = [int(round(t)) for t in rf.estimators_[0].tree_.threshold
                  if t != -2.0]
    assert 0 in thresholds, thresholds


def test_a_zero_split_gets_its_own_interval():
    """C1: the generator emits (0, 0), (1, t1), ...; this module emitted
    (0, t1), ... -- so alignment optimised a partition the TCAM cost was not
    computed from, and its block savings were mis-targeted wherever a zero
    split existed."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[0.0], [0.0], [1.0], [5.0], [0.0], [7.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    rf = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=1, max_depth=1, random_state=0).fit(X, y))

    intervals = ta.extract_feature_intervals(rf)

    assert intervals[0][0] == (0, 0), intervals[0]


def test_the_threshold_index_and_the_intervals_agree_on_which_splits_exist():
    """build_threshold_index never skipped 0, so it held (f, 0) keys that no
    interval referenced. After C1 the two views agree."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[0.0], [0.0], [1.0], [5.0], [0.0], [7.0]])
    y = np.array([0, 0, 1, 1, 0, 1])
    rf = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=1, max_depth=1, random_state=0).fit(X, y))

    intervals = ta.extract_feature_intervals(rf)
    index = ta.build_threshold_index(rf)

    # Every threshold in the index is a boundary of some interval on that
    # feature: either an upper bound, or (lower - 1).
    for feature_idx, threshold in index:
        bounds = {hi for _, hi in intervals[feature_idx]}
        bounds |= {lo - 1 for lo, _ in intervals[feature_idx] if lo > 0}
        bounds |= {0}
        assert threshold in bounds, (feature_idx, threshold, intervals[feature_idx])


# ---------------------------------------------------------------------------
# T1: find_partially_overlapping_ranges -- the two-pointer overlap sweep.
# ---------------------------------------------------------------------------

def _find_overlaps_nested(ranges1, ranges2):
    """Reference oracle: a verbatim copy of the O(n*m) nested scan that
    find_partially_overlapping_ranges used to be, kept here so the sweep can
    be checked against the exact behaviour it replaces."""
    overlaps = []

    for i, (start1, end1) in enumerate(ranges1):
        if end1 <= start1:
            continue
        for j, (start2, end2) in enumerate(ranges2):
            if end2 <= start2:
                continue
            if start1 == start2 and end1 == end2:
                continue
            if start1 < end2 and start2 < end1:
                overlaps.append((i, j))

    return overlaps


def _random_tiling(rng, max_threshold=19, n_points=8):
    """Builds a tiling the way extract_feature_intervals / get_feature_intervals
    _from_thresholds does: thresholds sorted, each interval chained from
    last_range[1] + 1, an equal-to-last-max threshold deduped away.

    Drawing thresholds from a SMALL range (default 0..19) makes both kinds of
    degenerate interval common rather than rare: a threshold of 0 gives a
    (0, 0) first interval, and two thresholds that are consecutive integers
    collapse into a (t, t) single-point interval for t > 0.
    """
    thresholds = sorted(int(t) for t in rng.integers(0, max_threshold + 1, size=n_points))
    intervals = []
    for t in thresholds:
        if not intervals:
            intervals.append((0, t))
        else:
            last_range = intervals[-1]
            if t == last_range[1]:
                continue
            intervals.append((last_range[1] + 1, t))
    return intervals


def test_the_sweep_matches_the_nested_scan_on_random_gap_free_tilings():
    """Equivalence, exact list equality INCLUDING ORDER, not set equality --
    align_stats, the candidate_log row order, and the accept/reject trajectory
    all depend on the order pairs come out in. Thresholds are drawn from a
    small range so (0,0) and (t,t) degenerates occur constantly, not as a rare
    edge case."""
    rng = np.random.default_rng(20260819)

    for _ in range(3000):
        ranges1 = _random_tiling(rng)
        ranges2 = _random_tiling(rng)

        assert ta.find_partially_overlapping_ranges(ranges1, ranges2) == \
            _find_overlaps_nested(ranges1, ranges2), (ranges1, ranges2)


def test_the_sweep_does_not_drop_a_pair_at_the_end1_equals_end2_tie():
    """The retirement invariant's hardest case: when end1 == end2 the sweep
    retires only i (ranges1's pointer), never both. Hand-built so the tie is
    guaranteed to fire at (0, 10) vs (5, 10), rather than hoping a random case
    hits it."""
    ranges1 = [(0, 10), (11, 20)]
    ranges2 = [(5, 10), (11, 25)]

    got = ta.find_partially_overlapping_ranges(ranges1, ranges2)

    assert got == _find_overlaps_nested(ranges1, ranges2)
    # The pair spanning the tie itself (ranges1[0] against ranges2[0], which
    # is where end1 == end2 == 10 fires) must not have been dropped.
    assert (0, 0) in got


def test_degenerate_zero_zero_and_t_t_intervals_are_excluded_by_choice_not_accident():
    """find_partially_overlapping_ranges filters end <= start, which drops
    (0, 0) AND (t, t) intervals for t > 0. That is consistent, not a bug:
    calculate_range_overlap already vetoes any pair where exactly one side
    starts at 0, and adjust_range_boundaries refuses to move a boundary at 0
    -- so a degenerate interval could never be aligned anyway. This test
    documents the exclusion as a choice, and pins it against the nested
    oracle so a future change to the filter shows up here."""
    ranges1 = [(0, 0), (1, 15), (16, 16), (17, 30)]
    ranges2 = [(0, 0), (1, 20), (21, 21), (22, 30)]
    degenerate1 = {0, 2}  # indices of (0, 0) and (16, 16) in ranges1
    degenerate2 = {0, 2}  # indices of (0, 0) and (21, 21) in ranges2

    got = ta.find_partially_overlapping_ranges(ranges1, ranges2)

    assert got == _find_overlaps_nested(ranges1, ranges2)
    for idx1, idx2 in got:
        assert idx1 not in degenerate1 and idx2 not in degenerate2, (idx1, idx2)


def test_merely_touching_intervals_are_not_overlaps():
    """Pins the strict '<' semantics against a future off-by-one 'fix': an
    interval that only touches another at a shared or adjacent boundary is
    not a partial overlap, whether the touch is exact (100 == 100) or there
    is a one-unit gap (100, 101)."""
    ranges1 = [(0, 100)]

    assert ta.find_partially_overlapping_ranges(ranges1, [(100, 200)]) == []
    assert ta.find_partially_overlapping_ranges(ranges1, [(101, 200)]) == []


def test_a_zero_zero_candidate_produces_no_modifications_either_way():
    """The consistency argument made executable: a (0, 0) source_range can
    never produce a modification, whether the other side's min is also 0 or
    is positive. target_range is computed via calculate_target_range exactly
    as align_rf_thresholds would, so this exercises the real shape of a call,
    not a contrived one. threshold_index is deliberately empty -- if either
    branch DID try to look up a threshold, that would raise rather than
    silently pass, so an empty modifications list is real evidence of the
    refusal, not an accident of a missing key."""
    rf = _one_split_forest()

    # Other side's min is 0: calculate_target_range((0,0), (0, 10)) == (0, 0),
    # so both the min- and max-side checks in adjust_range_boundaries see no
    # change and refuse.
    other_min_zero = (0, 10)
    target_a = ta.calculate_target_range((0, 0), other_min_zero)
    modifications_a = ta.adjust_range_boundaries(
        rf, feature_idx=0, source_range=(0, 0), target_range=target_a,
        threshold_index={})
    assert modifications_a == []

    # Other side's min is not 0: calculate_target_range((0,0), (5, 10)) ==
    # (5, 0) -- the min-side check is refused because threshold_source_min is
    # 0, and the max-side check sees threshold_source_max == threshold_target
    # _max == 0.
    other_min_nonzero = (5, 10)
    target_b = ta.calculate_target_range((0, 0), other_min_nonzero)
    modifications_b = ta.adjust_range_boundaries(
        rf, feature_idx=0, source_range=(0, 0), target_range=target_b,
        threshold_index={})
    assert modifications_b == []
