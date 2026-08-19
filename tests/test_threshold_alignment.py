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
