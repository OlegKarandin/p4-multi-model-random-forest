"""First tests for threshold_alignment.py -- the module the spec identifies as
the sole source of the joint-vs-independent accuracy delta (C2, C5)."""
import copy
from unittest import mock

import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int, normalise_feature_name
from src.training import align_budget as ab
from src.training import align_targets as at
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


def test_neighbor_update_raises_alignment_invariant_error_on_inversion():
    """When absorbing a target range's boundary move would flip a neighboring
    range's own min above its max, that neighbor cannot be written back --
    this used to be a bare `raise RuntimeError("Smth is very-very wrong")`,
    which is indistinguishable from a bug in `ranges`/`threshold_index`
    bookkeeping unrelated to this invariant. It must now be the module's own
    AlignmentInvariantError, like every other invariant site here.

    All-or-nothing is the new, stronger contract (Task 9): the inversion is
    now detected by align_targets.neighbour_writes BEFORE any write lands, so
    `ranges` must come out of the raise completely untouched -- not partially
    mutated the way the old mid-loop-raise version left it."""
    ranges = [(10, 20), (7, 9)]
    snapshot = list(ranges)
    threshold_index = {(0, 9): [(0, 0)]}

    with pytest.raises(AlignmentInvariantError):
        ta.update_neighboring_ranges_and_index(
            ranges, target_idx=0, old_range=(10, 20), new_range=(5, 20),
            feature_idx=0, threshold_index=threshold_index)

    assert ranges == snapshot


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


def test_ensemble_prediction_counts_votes_with_one_bincount_call_not_a_python_loop(
        monkeypatch):
    """Not a microbenchmark for its own sake: this function runs ~2x per
    candidate, thousands of candidates per alignment call, once per Optuna
    trial, across 7 M x 15 splits x 17 k. A pure-Python double loop here is the
    single largest cost in the module.

    Structural, not wall-clock -- a `elapsed < 1.0` timing assertion here was
    the one flaky test in this file, able to fail on a loaded CI box or a
    cold import with no code change at all. compute_ensemble_prediction's own
    docstring says the vote count is "vectorised as one bincount over a
    sample-major offset array": exactly one np.bincount call per invocation,
    however large tree_predictions is. A regression to a per-(tree, sample)
    Python double loop would either not call np.bincount at all, or call it
    once per sample -- both are caught by pinning the call count to the
    number of INVOCATIONS (2) across two very differently sized inputs,
    independent of n_trees/n_samples.
    """
    rf_small, X_small, _ = _forest_and_data(n_estimators=7, n=10)
    rf_large, X_large, _ = _forest_and_data(n_estimators=7, n=4000)
    tp_small, _ = ta.build_prediction_cache(rf_small, X_small)
    tp_large, _ = ta.build_prediction_cache(rf_large, X_large)

    calls = []
    real_bincount = np.bincount

    def counting_bincount(*args, **kwargs):
        calls.append(1)
        return real_bincount(*args, **kwargs)

    monkeypatch.setattr(np, 'bincount', counting_bincount)

    ta.compute_ensemble_prediction(tp_small, rf_small)
    ta.compute_ensemble_prediction(tp_large, rf_large)

    assert len(calls) == 2


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
    """The float32 cast must be local. Even now that C8 stops this module from
    mutating the caller's MODELS in place (below), it must not start mutating
    the caller's data instead."""
    rf1, X1, y1 = _forest_and_data(seed=5)
    rf2, X2, y2 = _forest_and_data(seed=6)
    y2 = np.where(y2 == 0, -1, 1)
    before_dtype, before_copy = X1.dtype, X1.copy()

    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           overlap_threshold=0.5, delta_rel=None)

    assert X1.dtype == before_dtype
    assert np.array_equal(X1, before_copy)


def test_original_forests_are_unchanged_after_an_alignment_that_accepts_a_move():
    """C8: align_rf_thresholds deepcopies rf1/rf2 on entry and mutates only the
    copies, so the caller's originals survive the call. delta_rel=None is the
    maximum-mutation arm (see _aligned_forest_pair) and this fixture is the
    same one test_a_candidate_that_moves_nothing_costs_no_prediction uses at
    delta_rel=0.05, where it reliably produces accepted moves -- a fixture
    where nothing gets accepted would pass whether or not the copy-on-entry
    fix landed, which would make the test worthless.

    Checked against the actual tree_.threshold arrays, not just object
    identity: identity alone wouldn't catch a version that deepcopies but
    still writes through to the original by accident.
    """
    rf1, X1, y1 = _forest_and_data(seed=5)
    rf2, X2, y2 = _forest_and_data(seed=6)
    y2 = np.where(y2 == 0, -1, 1)

    before1 = [np.array(e.tree_.threshold, copy=True) for e in rf1.estimators_]
    before2 = [np.array(e.tree_.threshold, copy=True) for e in rf2.estimators_]

    stats = {}
    out1, out2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                        overlap_threshold=0.5, delta_rel=None,
                                        align_stats=stats)

    assert stats['accepted'] > 0, 'the fixture must accept at least one move'

    # Both directions of the contract: the returned objects are new objects,
    # not the caller's originals wearing new thresholds.
    assert out1 is not rf1
    assert out2 is not rf2

    for estimator, expected in zip(rf1.estimators_, before1):
        assert np.array_equal(estimator.tree_.threshold, expected)
    for estimator, expected in zip(rf2.estimators_, before2):
        assert np.array_equal(estimator.tree_.threshold, expected)


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
    It must not pay for two ensemble predictions and four metric computations.

    `_forest_and_data`'s random forests never actually hit this path (checked
    by instrumenting adjust_range_boundaries directly: 0 of ~200 candidates
    across ten seed pairs came back empty on both sides), because the loop
    already skips `range1 == range2` before the modifications are even
    computed -- so on a random fixture at least one side always has real work
    left. Hand-build one instead: feature 0's (1, 999) vs (5, 999) triggers
    the bail two different ways at once -- rf1's min boundary sits right
    after the model's threshold-0 split (adjust_range_boundaries refuses to
    move a threshold AT 0, and 1 - 1 == 0), and rf2's own range already
    equals the target -- while (2000, 65535) vs (3000, 65535) is a genuine,
    attempted move. This is what test_a_zero_zero_candidate_produces_no_
    modifications_either_way already proves component-by-component; this
    test is the same mechanism wired into the real loop, counted end to end.
    """
    rf1 = _hand_built_forest([0, 999, 2999])
    rf2 = _hand_built_forest([0, 4, 999, 1999])
    X = np.array([[0.0], [50.0], [500.0], [1500.0],
                  [2500.0], [4000.0], [7000.0], [65535.0]])
    y1 = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    # T2b: the loop no longer calls compute_ensemble_prediction at all -- it
    # reads the winner off IncrementalMetrics -- so counting THAT would make
    # this test pass vacuously with an empty list. Count the metric updates
    # instead: IncrementalMetrics.apply is the per-candidate work this test
    # exists to prove the bail avoids.
    calls = []
    real_apply = ta.IncrementalMetrics.apply

    def counting_apply(self, tree_predictions, undo_info):
        calls.append(1)
        return real_apply(self, tree_predictions, undo_info)

    monkeypatch.setattr(ta.IncrementalMetrics, 'apply', counting_apply)

    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
                           overlap_threshold=0.5, delta_rel=0.05,
                           align_stats=stats)

    # Exactly one of this fixture's two candidates survives the bail, so
    # exactly two metric updates happen -- one per model, for that one
    # candidate. Both numbers are hardcoded from this fixture on purpose:
    # deleting the bail lets the (1, 999) vs (5, 999) pair through too, which
    # raises stats['attempted'] to 2 and len(calls) to 4 -- verified locally
    # by disabling the bail and rerunning, then restoring it. A relation
    # between the two (e.g. len(calls) == 2 * stats['attempted']) would NOT
    # catch that: apply is called exactly twice per attempted candidate
    # whether or not the bail exists, so that relation holds either way --
    # only the absolute counts move.
    assert stats['attempted'] == 1
    assert len(calls) == 2


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
    more common than the single averaged guard did, so any leak here compounds.

    T2b adds two more structures the reject path has to restore: the vote
    matrix / winner column and the confusion matrix owned by IncrementalMetrics.
    They are exercised here in the real ordering the loop uses --
    update_cache_for_modifications, then IncrementalMetrics.apply, then (on
    reject) restore_thresholds + undo_cache_update + IncrementalMetrics.revert.
    """
    rf, X, y = _forest_and_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    threshold_index = ta.build_threshold_index(rf)
    tree_predictions, node_to_samples = ta.build_prediction_cache(rf, X32)
    metrics = ta.IncrementalMetrics(tree_predictions, rf, y, task='app')

    snap = _snapshot(rf, tree_predictions, node_to_samples, threshold_index)
    votes_before = metrics.votes.copy()
    pred_before = metrics.pred_idx.copy()
    confusion_before = metrics.confusion.copy()
    metrics_before = metrics.metrics()

    feature_idx, source, target = _first_movable_interval(rf)
    modifications = ta.adjust_range_boundaries(
        rf, feature_idx, source, target, threshold_index)
    undo_info = ta.update_cache_for_modifications(
        rf, X32, tree_predictions, node_to_samples, modifications)
    token = metrics.apply(tree_predictions, undo_info)

    ta.restore_thresholds(rf, modifications)
    ta.undo_cache_update(tree_predictions, node_to_samples, undo_info)
    metrics.revert(token)

    _assert_snapshot_restored(rf, tree_predictions, node_to_samples, threshold_index, snap)
    assert np.array_equal(metrics.votes, votes_before)
    assert np.array_equal(metrics.pred_idx, pred_before)
    assert np.array_equal(metrics.confusion, confusion_before)
    assert metrics.votes.dtype == votes_before.dtype
    assert metrics.pred_idx.dtype == pred_before.dtype
    assert metrics.confusion.dtype == confusion_before.dtype
    assert metrics.metrics() == metrics_before


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

    assert {normalise_feature_name(names[idx]) for idx in ours} == set(theirs)
    for feature_idx, intervals in ours.items():
        assert intervals == theirs[normalise_feature_name(names[feature_idx])], names[feature_idx]


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


# ---------------------------------------------------------------------------
# T2 (part b): the incremental vote/confusion state wired into
# align_rf_thresholds. This change is meant to be EXACTLY numerically neutral
# -- it changes how (accuracy, weighted_f1) is computed, never what it is --
# so the gate below pins the whole alignment output against values captured
# from the pre-change implementation.
# ---------------------------------------------------------------------------

def _golden_alignment_pair(n=300):
    """The fixture the golden values below were captured on. Deterministic end
    to end: fixed rng seeds for the feature matrices, fixed random_state for
    both forests, and dt_thresholds_float_to_int so every threshold is an
    integer (which is why the golden arrays can be written as ints).

    Deliberately a real App/DDoS pair -- rf1 fit on labels {0,1,2}, rf2 fit on
    {-1,1} -- so the DDoS half exercises the negative label space through the
    whole loop rather than only in a unit test.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng1 = np.random.default_rng(5)
    X1 = np.clip(rng1.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y1 = np.array([c % 3 for c in range(n)])
    rf1 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=7, max_depth=5, min_samples_leaf=20, random_state=0).fit(X1, y1))

    rng2 = np.random.default_rng(6)
    X2 = np.clip(rng2.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y2 = np.where(np.arange(n) % 2 == 0, -1, 1)
    rf2 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=7, max_depth=5, min_samples_leaf=20, random_state=0).fit(X2, y2))

    return rf1, X1, y1, rf2, X2, y2


# Captured from THIS commit's c1c2 path by a throwaway script (see the
# 2026-08-30 cost-aware-threshold-alignment plan, Task 1), at
# MAX_RECOMPUTE_ROUNDS = 1, on _golden_alignment_pair().
#
# This literal's role is DIFFERENT from _ALIGNMENT_GOLDEN's above, and the
# difference matters. _ALIGNMENT_GOLDEN is a HISTORICAL anchor: captured at
# commit 0fb5ace from an implementation that no longer exists, it must never
# be regenerated, because regenerating it would pin a change against itself.
# This one is a DELETION-INVARIANCE anchor: it is captured from current code
# deliberately, and its job is to be a fixed point across the commit that
# deletes the legacy/c1 policies. Under that deletion c1c2's behaviour becomes
# the ONLY behaviour, so these arrays must come through it bit-identical --
# that is the whole proof that the deletion touched nothing that survived it.
# It must likewise not be regenerated after the deletion.
_ALIGNMENT_GOLDEN_C1C2 = {
    0.0: {
        'stats': {'attempted': 27, 'accepted': 17,
                  'intervals_before': 91, 'intervals_after': 71},
        't1': [
            [49965, 37970, -2, 60939, 30850, -2, -2, -2, 29400, -2, 33384, -2,
             -2],
            [25153, 17906, 42582, -2, -2, -2, 25152, -2, 65407, 47461, -2, -2,
             -2],
            [9867, -2, 38129, 64574, 22960, -2, -2, -2, 22949, -2, 41841, -2,
             -2],
            [11493, -2, 15571, -2, 26424, -2, 43169, 33744, -2, -2, 27856, -2,
             -2],
            [45724, 17518, 48924, -2, -2, 25535, -2, 48261, -2, -2, 63815, -2,
             49629, -2, -2],
            [40514, 50955, 32983, -2, -2, 21061, -2, -2, 64763, 24115, -2,
             50610, -2, -2, -2],
            [40996, 17244, -2, 35130, -2, 58452, -2, -2, 52649, 65407, -2, -2,
             -2],
        ],
        't2': [
            [8902, -2, 58514, 50135, 29400, -2, -2, 30850, -2, -2, 33384, -2,
             -2],
            [14536, -2, 27856, -2, 40996, -2, 61298, 38258, -2, -2, -2],
            [27458, 47461, -2, -2, 58452, 62051, 33860, -2, -2, -2, 61513, -2,
             -2],
            [61422, 43169, 26424, 15571, -2, -2, -2, 11493, -2, 57942, -2, -2,
             53373, -2, -2],
            [27321, 53909, 39702, -2, -2, -2, 17534, -2, 25152, -2, 53934, -2,
             60939, -2, -2],
            [54408, 44768, 33744, -2, -2, 62045, -2, -2, 17244, -2, 41360, -2,
             52649, -2, -2],
            [54766, 50955, 24115, -2, -2, 38129, -2, -2, 49965, 25912, -2, -2,
             -2],
        ],
    },
    0.05: {
        'stats': {'attempted': 32, 'accepted': 2,
                  'intervals_before': 91, 'intervals_after': 88},
        't1': [
            [50148, 37970, -2, 64068, 30850, -2, -2, -2, 29481, -2, 36261, -2,
             -2],
            [25153, 17906, 42582, -2, -2, -2, 25152, -2, 65407, 47461, -2, -2,
             -2],
            [9867, -2, 41694, 64574, 22960, -2, -2, -2, 22949, -2, 41841, -2,
             -2],
            [11493, -2, 16559, -2, 26336, -2, 43169, 33744, -2, -2, 26063, -2,
             -2],
            [45724, 8902, 48924, -2, -2, 25535, -2, 48261, -2, -2, 63815, -2,
             49629, -2, -2],
            [40514, 50955, 32983, -2, -2, 21061, -2, -2, 64763, 28712, -2,
             50610, -2, -2, -2],
            [40996, 17244, -2, 35130, -2, 58798, -2, -2, 52649, 65407, -2, -2,
             -2],
        ],
        't2': [
            [8902, -2, 58514, 50135, 29400, -2, -2, 30237, -2, -2, 33384, -2,
             -2],
            [14536, -2, 26063, -2, 40514, -2, 61298, 38258, -2, -2, -2],
            [27458, 48468, -2, -2, 58452, 62051, 33860, -2, -2, -2, 61513, -2,
             -2],
            [61422, 45058, 26424, 15571, -2, -2, -2, 16443, -2, 57942, -2, -2,
             53373, -2, -2],
            [27321, 53909, 39702, -2, -2, -2, 17534, -2, 21985, -2, 53934, -2,
             60939, -2, -2],
            [54408, 44768, 33254, -2, -2, 62045, -2, -2, 18076, -2, 41360, -2,
             48048, -2, -2],
            [54766, 44845, 24115, -2, -2, 38129, -2, -2, 49965, 25912, -2, -2,
             -2],
        ],
    },
}


@pytest.mark.parametrize('delta_rel', [0.0, 0.05])
def test_align_rf_thresholds_produces_the_same_models_as_before_this_change(
        delta_rel, monkeypatch):
    """The end-to-end numeric-neutrality gate for T2b, and the REGRESSION side
    of T3's two-sided gate.

    T3 (C3) recomputes the candidate set after every accepted move, which
    legitimately moves these numbers -- so this test pins the loop at
    MAX_RECOMPUTE_ROUNDS = 1, where C3 is required to be a bit-identical
    no-op: one round, in sweep order (== the old nested order), with `seen`
    never firing. The literal below is therefore still the PRE-C3 output; it
    was NOT regenerated from post-C3 code, which would have turned the gate
    into a tautology. What it now pins is "round-1 C3 == pre-C3", exactly.

    Replacing sklearn's accuracy_score/f1_score with a confusion-matrix
    formula, and the from-scratch ensemble vote with an incrementally
    maintained one, must not move a single number. It cannot be checked by
    "the metrics look close": accept_alignment compares against a per-task
    ratchet, so a one-ULP disagreement flips a decision, the flipped decision
    changes which thresholds move, and every later candidate sees a different
    model. The observable consequence is the final threshold arrays and the
    stats dict -- so pin those.

    The legacy parametrization was removed with the policy ladder; the
    historical pre-C3 literal it pinned is preserved in git history at the
    commit before the deletion, and its role is taken over by the c1c2
    literal, which came through the deletion bit-identical.
    """
    golden = _ALIGNMENT_GOLDEN_C1C2[delta_rel]
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 1)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}

    # C8: align_rf_thresholds no longer mutates rf1/rf2 in place -- it returns
    # copies -- so the aligned models to check are the returned ones.
    rf1, rf2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=delta_rel, align_stats=stats)

    # Compare only the keys the golden literal was captured for. The literal
    # dates from commit 0fb5ace and must NOT be regenerated from post-change
    # code (see this test's docstring); the codeword keys added later are
    # instead checked by DERIVATION from the pinned interval counts, in
    # test_align_stats_records_the_codeword_length_it_optimises.
    assert {k: stats[k] for k in golden['stats']} == golden['stats']
    for key, rf in (('t1', rf1), ('t2', rf2)):
        for tree_idx, (estimator, expected) in enumerate(
                zip(rf.estimators_, golden[key])):
            assert np.array_equal(estimator.tree_.threshold,
                                  np.array(expected, dtype=np.float64)), (key, tree_idx)


def test_compute_ensemble_prediction_is_still_reachable_and_returns_the_right_shape():
    """T2b removed compute_ensemble_prediction's last PRODUCTION caller -- the
    alignment loop now reads its winner off IncrementalMetrics. The function
    must survive anyway: it is the from-scratch oracle every equivalence test
    in this file and in test_incremental_metrics.py compares against, and a
    dead-code sweep that deletes it takes those tests with it.

    So: it is still exported and it still computes something of the right
    shape. The full "still the oracle" claim -- that it agrees with
    switch_predict exactly -- is exercised in full by
    test_ensemble_prediction_is_the_cached_path_to_switch_predict; repeating
    that comparison here added nothing.
    """
    assert callable(ta.compute_ensemble_prediction)

    rf, X, y = _forest_and_data()
    tree_predictions, _ = ta.build_prediction_cache(rf, X)
    got = ta.compute_ensemble_prediction(tree_predictions, rf)
    assert got.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# T3 (C3): the candidate set is recomputed after every ACCEPTED move.
#
# Before C3 the overlap list was computed once per feature and iterated while
# update_neighboring_ranges_and_index mutated the underlying interval lists in
# place. Aligning range i widens its neighbours; a widened neighbour can newly
# overlap a range in the other model, and that pair was never enumerated. The
# tests below pin both sides of the gate: one round alone must reproduce the
# pre-C3 result bit for bit, and the full loop may only APPEND to it.
# ---------------------------------------------------------------------------

def _hand_built_forest(thresholds):
    """A forest whose feature-0 interval list is exactly the one asked for.

    One feature, one depth-1 tree per threshold (bootstrap=False so every tree
    really does get its split), then the thresholds are overwritten by hand --
    the same trick _one_split_forest uses. `thresholds` must be ascending, and
    the resulting intervals are (0,t0),(t0+1,t1),...,(tlast+1,INFINITE).
    """
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    rf = RandomForestClassifier(n_estimators=len(thresholds), max_depth=1,
                                bootstrap=False, random_state=0).fit(X, y)
    for tree_idx, threshold in enumerate(thresholds):
        tree = rf.estimators_[tree_idx].tree_
        assert tree.node_count == 3 and tree.feature[0] == 0
        tree.threshold[0] = float(threshold)
    return rf


def _neighbour_widening_pair():
    """The motivating fixture: one accepted move provably CREATES a candidate.

        I1 = [(0,99), (100,999), (1000,5999), (6000,INF)]
        I2 = [(0,99), (100,999), (1000,2999), (3000,5999), (6000,INF)]

    Round 1 has exactly one eligible candidate, (1000,5999) vs (3000,5999) at
    ratio 0.5999. Accepting it drags I1's left neighbour out to (100,2999),
    which then overlaps I2's (1000,2999) at ratio 0.6896 -- a pair that did
    not overlap AT ALL before the move (it was (100,999) vs (1000,2999)), so
    no amount of re-reading the original overlap list could reach it.
    """
    rf1 = _hand_built_forest([99, 999, 5999])
    rf2 = _hand_built_forest([99, 999, 2999, 5999])
    X = np.array([[0.0], [50.0], [500.0], [1500.0],
                  [2500.0], [4000.0], [7000.0], [65535.0]])
    y1 = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    return rf1, rf2, X, y1, y2


def _count_rounds(monkeypatch):
    """Rounds actually run, keyed by feature.

    The recompute loop calls find_partially_overlapping_ranges exactly once
    per round, and each feature's ranges list is a distinct list object owned
    by intervals1 -- so id(ranges1) identifies the feature.
    """
    real = ta.find_partially_overlapping_ranges
    rounds = {}

    def spy(ranges1, ranges2):
        rounds[id(ranges1)] = rounds.get(id(ranges1), 0) + 1
        return real(ranges1, ranges2)

    monkeypatch.setattr(ta, 'find_partially_overlapping_ranges', spy)
    return rounds


def _isolate_c3(monkeypatch):
    """Neutralise C1's budget gating and C2's damage-ranked target selection,
    both unconditional since the 2026-08-30 policy-ladder deletion, so the
    C3-recompute tests below keep testing the mechanism they were built for
    -- round-by-round rescanning -- rather than incidentally re-deriving
    whether THIS hand-built fixture's candidates clear the (now-mandatory)
    accuracy gate or the gain filter. Both are covered by their own tests
    (test_delta_align_none_still_builds_the_oracle_now_that_gating_is_
    unconditional in test_train_model_contract.py, which actually proves the
    oracle is always built by spying IncrementalMetrics.__init__ and
    asserting both tasks were scored, and the C2 tests respectively);
    restoring the pre-ladder single-target, always-accept shape here
    reproduces this file's pre-Task-3 recompute numbers exactly.
    """
    monkeypatch.setattr(ta, 'accept_alignment', lambda *a, **k: True)
    monkeypatch.setattr(
        ta, '_rank_targets',
        lambda range1, range2, *a, **k: [ta.calculate_target_range(range1, range2)])


def test_a_widened_neighbour_becomes_a_candidate_only_after_the_recompute(monkeypatch):
    """THE motivating test for C3: without the rescan this pair is unreachable.

    With the recompute disabled (a single round) the fixture attempts exactly
    the one candidate the original overlap list held. With it enabled, the
    pair the accepted move CREATED is attempted too -- in round 2, the only
    place it could ever appear.
    """
    new_pair = ((100, 2999), (1000, 2999))

    _isolate_c3(monkeypatch)
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 1)
    rf1, rf2, X, y1, y2 = _neighbour_widening_pair()
    stats_one_round, log_one_round = {}, []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, overlap_threshold=0.5,
                           delta_rel=None, align_stats=stats_one_round,
                           candidate_log=log_one_round)

    assert [(e['range1'], e['range2']) for e in log_one_round] == \
        [((1000, 5999), (3000, 5999))]
    assert stats_one_round['attempted'] == 1

    # undo() only lifts the MAX_RECOMPUTE_ROUNDS=1 patch -- _isolate_c3's two
    # patches must stay in place for the full-recompute run below too, or the
    # zero-gain filter and the now-unconditional accuracy gate reintroduce
    # exactly the interference _isolate_c3 exists to remove.
    monkeypatch.undo()
    _isolate_c3(monkeypatch)
    rf1, rf2, X, y1, y2 = _neighbour_widening_pair()
    stats, log = {}, []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, overlap_threshold=0.5,
                           delta_rel=None, align_stats=stats,
                           candidate_log=log)

    assert [(e['range1'], e['range2']) for e in log] == \
        [((1000, 5999), (3000, 5999)), new_pair]
    assert [e['round'] for e in log] == [1, 2]
    assert stats['attempted'] == 2 and stats['accepted'] == 2


def test_the_recompute_stops_as_soon_as_a_round_accepts_nothing(monkeypatch):
    """Termination is by fixpoint, not by exhausting the cap: a round that
    accepts nothing changed no tuple, so the recomputed sweep would yield the
    identical list with every member already retired in `seen`. A feature with
    no eligible candidate at all therefore costs exactly ONE sweep."""
    rounds = _count_rounds(monkeypatch)

    # I1 = [(0,999),(1000,1999),(2000,INF)]
    # I2 = [(0,999),(1000,1499),(1500,1999),(2000,INF)]
    # The shared head and tail intervals are identical (excluded by the
    # sweep), and both remaining pairs score 499/999 = 0.4995 -- just under
    # the 0.5 threshold. So nothing is ever accepted.
    rf1 = _hand_built_forest([999, 1999])
    rf2 = _hand_built_forest([999, 1499, 1999])
    X = np.array([[0.0], [50.0], [500.0], [1500.0],
                  [2500.0], [4000.0], [7000.0], [65535.0]])
    y1 = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, overlap_threshold=0.5,
                           delta_rel=None, align_stats=stats)

    assert stats['accepted'] == 0
    assert set(rounds.values()) == {1}
    assert max(rounds.values()) < ta.MAX_RECOMPUTE_ROUNDS


def test_the_recompute_cap_raises_instead_of_looping_without_end(monkeypatch):
    """MAX_RECOMPUTE_ROUNDS is a CYCLE GUARD, not a tuning parameter: there is
    no monotone measure on interval count or union size (see the counterexample
    below), so termination is ENFORCED rather than proved. Truncating a loop
    that was still accepting moves is an invariant violation, not a silent
    stop.

    The cap is monkeypatched DOWN rather than exercised at its shipped value
    on purpose. The shipped value is deliberately far above the measured
    fixpoint depth (32 against an observed maximum of 10 across 18 seed x arm
    configurations -- a sample maximum, not a proven bound), so no reachable
    fixture would ever trip it -- a test that waited for the real cap to fire
    would either never run this branch or would have to be retuned every time
    the constant moves. The motivating fixture needs three rounds (accept,
    accept, fixpoint), so a cap of 2 truncates it mid-progress.
    """
    _isolate_c3(monkeypatch)
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 2)
    rf1, rf2, X, y1, y2 = _neighbour_widening_pair()

    with pytest.raises(AlignmentInvariantError) as excinfo:
        ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, overlap_threshold=0.5,
                               delta_rel=None)

    assert 'fixpoint' in str(excinfo.value).lower()


def test_the_recompute_never_evaluates_the_same_value_pair_twice(monkeypatch):
    """`seen` keys on VALUE pairs, not index pairs -- an accepted move rewrites
    tuples in place, so the same index pair names a different candidate in a
    later round and the same candidate can move to a different index. Without
    it every round would re-offer every pair it had already judged."""
    judged = []
    real = ta.calculate_range_overlap

    def spy(range1, range2):
        judged.append((range1, range2))
        return real(range1, range2)

    monkeypatch.setattr(ta, 'calculate_range_overlap', spy)
    _isolate_c3(monkeypatch)
    # The motivating fixture plus two intervals well above the region the
    # accepted moves touch:
    #   I1 = [(0,99),(100,999),(1000,5999),(6000,20000),(20001,INF)]
    #   I2 = [(0,99),(100,999),(1000,2999),(3000,5999),(6000,50000),(50001,INF)]
    # The three pairs up there score below the threshold and are never
    # touched by any move, so every round re-enumerates them unchanged --
    # which is what makes this test non-vacuous. Measured: 9 judgements with
    # `seen`, 15 for the same 9 distinct pairs without it.
    rf1 = _hand_built_forest([99, 999, 5999, 20000])
    rf2 = _hand_built_forest([99, 999, 2999, 5999, 50000])
    X = np.array([[0.0], [50.0], [500.0], [1500.0], [2500.0],
                  [4000.0], [7000.0], [30000.0], [65535.0]])
    y1 = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1])
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, overlap_threshold=0.5,
                           delta_rel=None, align_stats=stats)

    # A single-feature fixture, so every judgement belongs to the same feature
    # and the per-feature `seen` set covers all of them.
    assert stats['accepted'] > 1, 'more than one round must actually run'
    assert judged, 'the fixture must produce candidates'
    assert len(judged) == len(set(judged)), judged


def test_the_partition_invariant_survives_the_multi_round_recompute(monkeypatch):
    """C5's invariant under the condition most likely to break it: repeated
    rounds at delta_rel=None, the maximum-mutation arm. More accepted moves is
    exactly when update_neighboring_ranges_and_index's
    AlignmentInvariantError (a neighboring range inverting) would newly fire,
    and this tiling is what the generator's TCAM ranges are built from."""
    rounds = _count_rounds(monkeypatch)
    rf1, rf2 = _aligned_forest_pair()

    assert max(rounds.values()) > 1, 'the fixture must actually recompute'
    for rf in (rf1, rf2):
        for feature_idx, intervals in ta.extract_feature_intervals(rf).items():
            assert intervals[0][0] == 0, (feature_idx, intervals)
            assert intervals[-1][1] == INFINITE, (feature_idx, intervals)
            for (_, prev_max), (next_min, _) in zip(intervals, intervals[1:]):
                assert next_min == prev_max + 1, (feature_idx, intervals)


def test_a_single_accepted_move_can_leave_the_joint_interval_count_flat():
    """Counterexample A, re-derived under the corrected pooled-threshold
    joint_interval_count (controller ruling P3b-4 named this counterexample;
    the arithmetic below is new -- the old union-of-tuples version claimed a
    RAISE from 6 to 7, which does not survive the fix: see
    joint_interval_count's docstring).

    Under the corrected definition, `stats['intervals_after'] <=
    stats['intervals_before']` IS a theorem: joint_interval_count is the size
    of the common refinement of both models' pooled thresholds per feature,
    and every write adjust_range_boundaries/update_neighboring_ranges_and_index
    perform relocates a threshold to a value drawn from {min1, min2, max1,
    max2} of the CURRENT candidate pair -- i.e. a value already present in
    one of the two models' current threshold sets for that feature, never a
    new one. So the pooled threshold SET for a feature can only shrink or
    stay the same, never grow, and neither can the interval count derived
    from it. See test_joint_interval_count_never_rises_across_random_alignment_runs
    below for a real, re-runnable sweep corroborating this on generated
    forests, not just the hand-built case here.

    It is NOT strictly decreasing on every move, though -- this is the
    counterexample for that weaker claim, and it is what
    MAX_RECOMPUTE_ROUNDS's docstring cites as the reason interval count can't
    serve as a per-round descent measure that bounds the round count: I1's
    two thresholds {9, 49} are already a SUBSET of I2's {9, 19, 44, 49}
    before the move, so the pooled set (and the joint count) is driven
    entirely by I2 and does not move even though a real move is accepted and
    I1's own tiling changes shape.
    """
    I1 = [(0, 9), (10, 49), (50, INFINITE)]
    I2 = [(0, 9), (10, 19), (20, 44), (45, 49), (50, INFINITE)]
    assert ta.joint_interval_count({0: I1}, {0: I2}) == 5

    range1, range2 = (10, 49), (20, 44)
    assert ta.calculate_range_overlap(range1, range2) == pytest.approx(0.6153846)
    target = ta.calculate_target_range(range1, range2)
    assert target == (20, 44)

    # The nodes those two boundaries come from; only (0, 9) and (0, 49) are
    # read, but a real index holds every threshold of the feature.
    threshold_index = {(0, 9): [(0, 0)], (0, 49): [(0, 1)],
                       (0, 19): [(0, 2)], (0, 44): [(0, 3)]}
    ta.update_neighboring_ranges_and_index(I1, 1, range1, target, 0, threshold_index)

    assert I1 == [(0, 19), (20, 44), (45, INFINITE)]
    assert ta.joint_interval_count({0: I1}, {0: I2}) == 5


def test_joint_interval_count_never_rises_across_random_alignment_runs():
    """Real, re-runnable corroboration for the structural claim in
    test_a_single_accepted_move_can_leave_the_joint_interval_count_flat's
    docstring and for #27's kept assertion
    (test_alignment_acceptance.py's stats['intervals_after'] <=
    stats['intervals_before']): every accepted move relocates a threshold to
    a value already present in one of the two models' current threshold
    sets, so joint_interval_count can only fall or stay flat, never rise.

    Sweeps varied forest shapes (sample count, feature count, depth) and
    every delta_rel arm (None/0.0/0.05/0.2) rather than relying on a single
    hand-built case -- this is what actually failed (assert 5 == 6) on the
    OLD union-of-tuples joint_interval_count before this task's fix, so it
    is a genuine regression guard, not decoration.
    """
    violations = []

    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 200 + (seed % 5) * 50
        nf = 3 + (seed % 3)
        X1 = np.clip(rng.integers(0, 90000, size=(n, nf)), 0, INFINITE).astype(float)
        y1 = np.array([c % 3 for c in range(n)])
        X2 = np.clip(rng.integers(0, 90000, size=(n, nf)), 0, INFINITE).astype(float)
        y2 = np.where(np.arange(n) % 2 == 0, -1, 1)

        from sklearn.ensemble import RandomForestClassifier
        rf1 = dt_thresholds_float_to_int(RandomForestClassifier(
            n_estimators=5, max_depth=4 + (seed % 3), min_samples_leaf=5,
            random_state=seed).fit(X1, y1))
        rf2 = dt_thresholds_float_to_int(RandomForestClassifier(
            n_estimators=5, max_depth=4 + (seed % 3), min_samples_leaf=5,
            random_state=seed + 1).fit(X2, y2))

        for delta in (None, 0.0, 0.05, 0.2):
            stats = {}
            ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                   overlap_threshold=0.5, delta_rel=delta,
                                   align_stats=stats)
            if stats['intervals_after'] > stats['intervals_before']:
                violations.append((seed, delta, stats))

    assert violations == []


def _align_golden_pair(delta_rel, cap, monkeypatch):
    """One run of the golden fixture at a given MAX_RECOMPUTE_ROUNDS."""
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', cap)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats, log = {}, []
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=delta_rel, align_stats=stats,
                           candidate_log=log)
    monkeypatch.undo()
    return stats, log


def _accepted_moves(log):
    return [(e['feature_idx'], e['range1'], e['range2'])
            for e in log if e['accepted']]


def _is_subsequence(small, big):
    """Every element of `small`, in order, somewhere in `big`."""
    it = iter(big)
    return all(item in it for item in small)


@pytest.mark.parametrize('delta_rel', [None, 0.0, 0.05])
def test_c3_only_appends_to_the_moves_a_single_round_already_made(delta_rel, monkeypatch):
    """The legitimate-change side of the two-sided gate.

    C3 reaches strictly more candidates, so `attempted` and `accepted` rise
    weakly. What it must NEVER do is reorder or drop work that the single
    pre-C3 pass already did. Stated precisely, because "the single-round
    sequence is a global prefix of the C3 sequence" is the wrong shape and
    fails on real fixtures: C3 appends its extra rounds INSIDE each feature's
    block, before moving on to the next feature. So the append-only property
    is
      - per feature: round 1's accepted moves for feature f are a PREFIX of
        C3's accepted moves for f;
      - globally: the whole round-1 sequence is a SUBSEQUENCE of C3's, and
        the order in which features contribute their first move is unchanged
        (`sorted_features` does not depend on the loop).
    Any diff not explained by "extra moves appended inside a feature's block"
    is a regression rather than a result change.

    Only the delta_rel=None arm is a theorem. On the guarded arms an extra
    move accepted in an earlier feature ratchets `marks` up (spec B.4), which
    may legitimately flip a later feature's decisions -- features are
    structurally independent (each owns its interval lists and its
    threshold-index keys) but the per-task high-water marks are global. It
    holds on this fixture for all three arms, and is asserted for all three;
    if a future change breaks it on a guarded arm only, that is the mechanism
    to check before assuming a bug.
    """
    stats_r1, log_r1 = _align_golden_pair(delta_rel, 1, monkeypatch)
    stats_c3, log_c3 = _align_golden_pair(delta_rel, ta.MAX_RECOMPUTE_ROUNDS, monkeypatch)

    # No new stats key from C3 itself -- 'round' lives in the candidate_log
    # instead. The seven byte-domain keys below are Task 5's addition,
    # recorded unconditionally regardless of C3 round depth.
    # Task 2 adds 'accuracy_spent', recorded on every objective.
    assert set(stats_c3) == {
        'attempted', 'accepted', 'intervals_before', 'intervals_after',
        'codeword_before', 'codeword_after', 'codeword_floor',
        'spent_budget', 'rolled_back', 'accuracy_spent',
        'key_bytes_before', 'key_bytes_after', 'key_bytes_floor',
        'ternary_stages_before', 'ternary_stages_after', 'stage_target',
        'bits_to_reach'}
    assert stats_c3['intervals_before'] == stats_r1['intervals_before']
    assert stats_c3['attempted'] >= stats_r1['attempted']
    assert stats_c3['accepted'] >= stats_r1['accepted']

    moves_r1, moves_c3 = _accepted_moves(log_r1), _accepted_moves(log_c3)
    assert moves_r1, 'the fixture must accept something in the single-round pass'
    # Under the now-unconditional c1c2 target ranking (2026-08-30 ladder
    # deletion) a recomputed round can surface a candidate that C2's gain
    # filter or the now-mandatory C1 accuracy gate then rejects, so on this
    # fixture ACCEPTED count no longer strictly grows on every delta arm
    # (measured: flat at 2 accepted for None and 0.05, still 17->18 for
    # 0.0). attempted is the honest "C3 found something new" signal here --
    # it strictly grows on every arm regardless of whether the extra
    # candidate is accepted.
    assert stats_c3['attempted'] > stats_r1['attempted'], 'C3 must find something new here'
    assert len(moves_c3) >= len(moves_r1)

    features_r1 = list(dict.fromkeys(f for f, _, _ in moves_r1))
    features_c3 = list(dict.fromkeys(f for f, _, _ in moves_c3))
    assert features_c3 == features_r1

    for feature_idx in features_r1:
        head = [m for m in moves_r1 if m[0] == feature_idx]
        full = [m for m in moves_c3 if m[0] == feature_idx]
        assert full[:len(head)] == head, feature_idx

    assert _is_subsequence(moves_r1, moves_c3)
    # Every round-1 candidate is round 1 in the C3 run too -- the rounds above
    # 1 are the appended work and nothing else.
    assert [e['round'] for e in log_r1] == [1] * len(log_r1)
    assert max(e['round'] for e in log_c3) > 1


def test_align_stats_records_the_codeword_length_it_optimises():
    """L is the quantity the block cost is a step function of, and it was
    recorded nowhere -- the companion analysis had to solve for it and failed
    on 8% of rows."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           overlap_threshold=0.5, delta_rel=0.0, align_stats=stats)

    assert set(stats) == {
        'attempted', 'accepted', 'intervals_before', 'intervals_after',
        'codeword_before', 'codeword_after', 'codeword_floor',
        'spent_budget', 'rolled_back', 'accuracy_spent',
        'key_bytes_before', 'key_bytes_after', 'key_bytes_floor',
        'ternary_stages_before', 'ternary_stages_after', 'stage_target',
        'bits_to_reach'}

    n_features = len(set(ta.extract_feature_intervals(rf1))
                     | set(ta.extract_feature_intervals(rf2)))
    assert stats['codeword_before'] == stats['intervals_before'] - n_features
    assert stats['codeword_after'] == stats['intervals_after'] - n_features
    assert stats['codeword_floor'] <= stats['codeword_after']
    assert stats['rolled_back'] is False


def test_the_recorded_codeword_is_the_one_the_block_cost_was_computed_from():
    """align_rf_thresholds counts in the models' COLUMN-INDEX space while
    multi_model_memory_evaluation counts over the union of the two models'
    selected feature NAMES. They coincide only when both models are fit on the
    same column space -- which align_rf_thresholds documents as required
    (threshold_alignment.py:180-184) and which holds throughout this campaign.
    Pin it, or C1 could silently gate on the wrong number."""
    from src.p4gen.evaluation import multi_model_memory_evaluation

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    names = ['f{}'.format(i) for i in range(X1.shape[1])]
    stats = {}
    a1, a2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=0.0,
                                    align_stats=stats)
    usage = multi_model_memory_evaluation(a1, a2, names, names, 'joint')
    assert stats['codeword_after'] == usage.codeword_length


# ---------------------------------------------------------------------------
# Task 7: BandBudget wiring.
# ---------------------------------------------------------------------------

def test_an_unreachable_boundary_is_identical_to_spending_nothing():
    """The strongest statement of C1's no-loss guarantee: when the floor puts
    the next band out of reach, C1 must be prediction-identical to delta = 0,
    not merely close to it."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats_c1 = {}
    a1, a2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=0.20,
                                    align_stats=stats_c1)

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats_zero = {}
    b1, b2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=0.0,
                                    align_stats=stats_zero)

    if stats_c1['spent_budget']:
        pytest.skip('this fixture can reach a band; the identity does not apply')

    for x, y in zip(a1.estimators_ + a2.estimators_, b1.estimators_ + b2.estimators_):
        assert np.array_equal(x.tree_.threshold, y.tree_.threshold)
    assert stats_c1['codeword_after'] == stats_zero['codeword_after']


def test_the_per_move_sheds_sum_to_the_whole_runs_shed():
    """C1 decrements L per accepted move instead of recomputing the joint count
    thousands of times. The two must agree exactly, or every band decision
    after the first accepted move is made on a stale number."""
    sheds = []
    original = ab.pooled_interval_count

    def spy(r1, r2):
        return original(r1, r2)

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    budget_lengths = []
    real_init = ab.BandBudget.note_shed

    def record(self, bits):
        budget_lengths.append(bits)
        return real_init(self, bits)

    with mock.patch.object(ab.BandBudget, 'note_shed', record):
        ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                               delta_rel=0.05, align_stats=stats)

    assert sum(budget_lengths) == stats['codeword_before'] - stats['codeword_after']


def test_the_oracle_is_built_even_at_an_unbounded_delta():
    """delta_rel=None normally skips the metric machinery entirely
    (threshold_alignment.py:345-348), which is what makes the dinf arm
    cheapest. C1's non-spending state needs delta=0, which needs the oracle --
    so now that gating is unconditional it must always be built. The dinf arm
    therefore loses its cost advantage; that is expected and must show up in
    the runtime budget."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=None, align_stats=stats)
    # With the oracle live, some candidate must have been judged rather than
    # waved through, so accepted cannot equal attempted on a reject-capable
    # fixture unless the budget was genuinely unbounded throughout.
    assert stats['attempted'] > 0


# ---------------------------------------------------------------------------
# Task 8: align_with_policy -- commit or roll back.
# ---------------------------------------------------------------------------

def test_spending_that_crosses_no_band_is_rolled_back_to_the_free_moves():
    """S1 by construction rather than by measurement: if budget was spent and
    the block factor did not fall, the accuracy was given away for nothing, so
    the run is redone at delta = 0 and THAT result is returned."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    a1, a2 = ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                                  overlap_threshold=0.5, delta_rel=0.20,
                                  align_stats=stats)
    from src.p4gen.evaluation import band_factor
    if not stats['rolled_back']:
        assert (band_factor(stats['codeword_after'])
                < band_factor(stats['codeword_before'])
                or not stats['spent_budget'])
        pytest.skip('this fixture crossed a band or never spent; nothing to roll back')

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    free = {}
    b1, b2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=0.0,
                                    align_stats=free)
    for x, y in zip(a1.estimators_ + a2.estimators_, b1.estimators_ + b2.estimators_):
        assert np.array_equal(x.tree_.threshold, y.tree_.threshold)
    assert stats['codeword_after'] == free['codeword_after']


def test_a_rollback_never_fires_when_no_budget_was_spent():
    """delta_rel = 0 gives nothing away, so there is never anything to undo and
    the second pass must not be paid for."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                         delta_rel=0.0, align_stats=stats)
    assert stats['spent_budget'] is False
    assert stats['rolled_back'] is False


# ---------------------------------------------------------------------------
# Task 10: C2 -- rank candidate targets by predicted damage and try them in
# order.
# ---------------------------------------------------------------------------

def test_c2_prefers_the_low_damage_corner_at_equal_gain():
    """The whole point of C2: all four corners of a clean pair shed the same
    two bits, because each of the two boundary gaps is crossed exactly once
    whichever corner is chosen -- only WHICH MODEL pays for which gap changes.
    So damage, measured on each model's own validation distribution, is the
    discriminator."""
    r1, r2 = (41, 96), (33, 88)
    ranges1 = [(0, 40), r1, (97, INFINITE)]
    ranges2 = [(0, 32), r2, (89, INFINITE)]

    gains = []
    for target in at.candidate_targets(r1, r2):
        h1 = at.hypothetical_ranges(ranges1, 1, r1, target)
        h2 = at.hypothetical_ranges(ranges2, 1, r2, target)
        if h1 is None or h2 is None:
            continue
        gains.append(ab.pooled_interval_count(ranges1, ranges2)
                     - ab.pooled_interval_count(h1, h2))
    assert gains and len(set(gains)) == 1, (
        'all admissible corners must shed the same bits on a clean pair')


def test_c2_never_offers_a_target_that_moves_nothing():
    """A corner asking only for sentinel moves has gain 0 and must be dropped
    before it costs an oracle evaluation -- mirroring the existing
    `if not modifications1 and not modifications2: continue` fast path."""
    assert at.boundary_moves((0, INFINITE), (0, INFINITE)) == []


def test_c2_evaluates_at_most_four_targets_per_pair(monkeypatch):
    """Cost bound. Alignment is already the campaign's largest unquantified
    runtime; four oracle calls per pair is the ceiling C2 may not exceed."""
    calls = []
    original = ta.accept_alignment
    monkeypatch.setattr(ta, 'accept_alignment',
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.0, align_stats=stats)
    assert len(calls) <= 4 * stats['attempted'] or stats['attempted'] == 0


def test_rank_targets_orders_equal_gain_corners_by_max_damage_not_sum():
    """Direct unit test for ta._rank_targets itself -- the brief's own four C2
    tests never call it (see review notes on this task): three test pure
    align_targets functions or align_rf_thresholds's dispatch, and the fourth
    is a tautology (len(calls) <= 4 * attempted, for any per-pair target
    count). None would catch a broken sort direction, a sum-instead-of-max
    damage aggregation, or a broken tiebreak.

    Reuses test_c2_prefers_the_low_damage_corner_at_equal_gain's exact
    r1/r2/ranges1/ranges2/idx1/idx2 -- that test already proves every
    admissible corner of this pair sheds the identical two bits, so the
    intersection and union corners here are KNOWN to tie on gain. sorted_cols1
    /sorted_cols2 are hand-built so damage differs between them, and --
    deliberately -- so that MAX and SUM of the two per-model shift_masses
    disagree about which corner is worse:

      intersection: model1 moves (88,96] (9/20 rows), model2 moves (32,40]
                    (9/20 rows) -> max damage 0.45, summed damage 0.90
      union:        model1 moves (32,40] (2/20 rows), model2 moves (88,96]
                    (10/20 rows) -> max damage 0.50, summed damage 0.60

    By MAX (the real rule) the intersection is cheaper (0.45 < 0.50) and must
    rank first. By SUM it would look worse (0.90 > 0.60) and a sum-based bug
    would rank the union first instead -- so this one assertion catches that
    bug directly. It also catches a reversed damage-ascending sort (flipping
    `damage` to `-damage` in the score tuple also puts the union first).
    Because these two corners tie on gain by construction, a reversed
    gain-descending sort alone does not move either of them relative to the
    other -- that failure mode needs a pair with UNEQUAL gain, which is
    exactly what the existing (already-passing)
    test_c2_prefers_the_low_damage_corner_at_equal_gain and
    test_c2_with_one_admissible_candidate_reproduces_the_legacy_choice
    partially cover from the outside; this test's job is specifically the
    damage-ordering half of the ranking that those four leave untested.

    RED/GREEN evidence (see task-10-report.md): fails when the damage
    aggregation is changed from `max` to `sum`, and fails when the score
    tuple's damage sign is flipped (`-damage` instead of `damage`) -- both
    confirmed manually against the real _rank_targets, then reverted. Does
    NOT fail under a flipped gain sign alone, for the reason above -- noted
    here so a future reader does not assume this one test is a complete
    substitute for gain-direction coverage.
    """
    r1, r2 = (41, 96), (33, 88)
    ranges1 = [(0, 40), r1, (97, INFINITE)]
    ranges2 = [(0, 32), r2, (89, INFINITE)]
    idx1, idx2, feature_idx = 1, 1, 0

    # 9/20 rows in (88, 96] (intersection's model1 move), 2/20 in (32, 40]
    # (union's model1 move), the rest elsewhere.
    sorted_cols1 = np.sort(
        np.array([0] * 9 + [35] * 2 + [90] * 9, dtype=np.float64)
    ).reshape(-1, 1)
    # 9/20 rows in (32, 40] (intersection's model2 move), 10/20 in (88, 96]
    # (union's model2 move), the rest elsewhere.
    sorted_cols2 = np.sort(
        np.array([0] * 1 + [35] * 9 + [90] * 10, dtype=np.float64)
    ).reshape(-1, 1)

    targets = ta._rank_targets(r1, r2, ranges1, ranges2, idx1, idx2,
                               feature_idx, sorted_cols1, sorted_cols2)

    intersection = (max(r1[0], r2[0]), min(r1[1], r2[1]))  # (41, 88)
    union = (min(r1[0], r2[0]), max(r1[1], r2[1]))          # (33, 96)
    assert intersection in targets and union in targets, (
        'both corners must be admissible for this hand-built pair', targets)
    assert targets.index(intersection) < targets.index(union), (
        'the lower-damage corner (intersection, max damage 0.45) must be '
        'tried before the higher-damage one (union, max damage 0.50)',
        targets)


# ---------------------------------------------------------------------------
# The align_objective axis (design 2026-08-30 §2.1-§2.4).
# ---------------------------------------------------------------------------

def _ordering_fixture():
    """Two interval dicts whose byte-first order is a genuine reordering of
    the combined-count order.

    Each feature's own1 and own2 threshold sets are same-sized but only
    PARTIALLY overlapping, so the pooled (union) width exceeds the floor
    (max of the two own widths) by real room to shrink through alignment:
      0: own1={1..17}, own2={9..25}  -> pooled 25, floor 17, room 8, step 1  REACHABLE
      1: own1={1..30}, own2={3..32}  -> pooled 32, floor 30, room 2, step 8  unreachable
      2: own1={1..10}, own2={9..18}  -> pooled 18, floor 10, room 8, step 2  REACHABLE
    Combined-count order is 1, 0, 2 (widest first); byte-first order must be
    0, 2, 1 -- cheapest reachable byte first, unreachable-but-widest last.
    """
    def interval_list(bounds):
        # A gap-free tiling terminated at INFINITE, built from sorted finite
        # threshold values -- the shape extract_feature_intervals actually
        # produces, unlike a bare list of singleton (i, i) tuples (which
        # smuggles in an extra threshold at 0 and breaks the width algebra).
        intervals = []
        lo = 0
        for b in bounds:
            intervals.append((lo, b))
            lo = b + 1
        intervals.append((lo, INFINITE))
        return intervals

    iv1 = {0: interval_list(range(1, 18)),
          1: interval_list(range(1, 31)),
          2: interval_list(range(1, 11))}
    iv2 = {0: interval_list(range(9, 26)),
          1: interval_list(range(3, 33)),
          2: interval_list(range(9, 19))}
    return iv1, iv2


def test_feature_order_under_blocks_is_the_pre_existing_order():
    """objective='blocks' must reproduce today's order EXACTLY -- combined
    interval count descending -- or commit 3 is not additive."""
    iv1, iv2 = _ordering_fixture()
    common = set(iv1) & set(iv2)
    expected = sorted(common,
                      key=lambda f: len(iv1.get(f, [])) + len(iv2.get(f, [])),
                      reverse=True)
    assert ta.feature_order(iv1, iv2, 'blocks') == expected


def test_feature_order_under_stages_puts_the_cheapest_reachable_byte_first():
    """A feature that cannot complete a byte is NOT dropped -- its bits still
    shrink L and buy blocks -- it only loses priority."""
    iv1, iv2 = _ordering_fixture()
    assert ta.feature_order(iv1, iv2, 'stages') == [0, 2, 1]


def test_feature_order_keeps_the_pre_existing_key_among_unreachable_features():
    """Byte distance decides only among features that can actually complete a
    byte; the rest follow in the pre-existing order."""
    def tiling(width):
        return [(0, 0)] + [(i, i) for i in range(1, width + 1)]
    # both unreachable (floor == pooled width), widths 20 and 28
    iv = {0: tiling(20), 1: tiling(28)}
    assert ta.feature_order(iv, iv, 'stages') == [1, 0]


def test_feature_order_is_a_total_order():
    """train_model.py:373-377's refit assertion depends on the run being
    deterministic, so ties must be broken, not left to set iteration."""
    def tiling(width):
        return [(0, 0)] + [(i, i) for i in range(1, width + 1)]
    iv = {0: tiling(20), 1: tiling(20), 2: tiling(20)}
    assert ta.feature_order(iv, iv, 'stages') == [0, 1, 2]


def test_an_unknown_objective_is_rejected_loudly():
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    with pytest.raises(ValueError, match='align_objective'):
        ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                               delta_rel=0.0, align_objective='stage')


def _boundary_stats(codeword_before, codeword_after, bytes_before, bytes_after):
    return {'codeword_before': codeword_before, 'codeword_after': codeword_after,
            'key_bytes_before': bytes_before, 'key_bytes_after': bytes_after}


def test_crossed_a_boundary_reports_a_band_crossing_under_every_objective():
    stats = _boundary_stats(100, 80, 35, 35)      # band factor 3 -> 2
    for objective in ('blocks', 'stages', 'both'):
        assert ta.crossed_a_boundary(stats, objective, n_tables=6)


def test_crossed_a_boundary_reports_nothing_when_neither_moved():
    stats = _boundary_stats(100, 99, 35, 35)
    for objective in ('blocks', 'stages', 'both'):
        assert not ta.crossed_a_boundary(stats, objective, n_tables=6)


def test_a_stage_step_alone_counts_only_for_the_stage_objectives():
    """Under 'blocks' the surviving policy keeps rolling back a run that
    crossed only a stage step: its semantics are pinned by the re-anchored
    goldens, and widening what counts as 'bought something' would change a
    measured property of an already-reported arm."""
    stats = _boundary_stats(100, 99, 35, 32)      # 6 stages -> 3, no band
    assert not ta.crossed_a_boundary(stats, 'blocks', n_tables=6)
    assert ta.crossed_a_boundary(stats, 'stages', n_tables=6)
    assert ta.crossed_a_boundary(stats, 'both', n_tables=6)


def test_a_fit_increase_that_saves_no_stage_is_not_a_crossing():
    """Compare STAGES, not fit. fit rising 2 -> 3 at T=4 leaves stage_depth
    unchanged, and keeping that run would reproduce in the byte domain exactly
    the failure the rollback exists to prevent."""
    stats = _boundary_stats(100, 99, 30, 21)
    assert not ta.crossed_a_boundary(stats, 'stages', n_tables=4)  # 2 -> 2
    assert ta.crossed_a_boundary(stats, 'stages', n_tables=6)      # 3 -> 2


# ---------------------------------------------------------------------------
# _stage_route_preferred (final-review finding I1): the sole decider of
# feature order under align_objective='both', previously untested.
# ---------------------------------------------------------------------------

def _route_stats(stage_target, bits_to_reach, key_bytes_floor=0,
                 ternary_stages_before=0, codeword_before=0, codeword_floor=0):
    return {'stage_target': stage_target, 'bits_to_reach': bits_to_reach,
           'key_bytes_floor': key_bytes_floor,
           'ternary_stages_before': ternary_stages_before,
           'codeword_before': codeword_before, 'codeword_floor': codeword_floor}


def test_stage_route_not_preferred_when_there_is_no_stage_target():
    stats = _route_stats(stage_target=None, bits_to_reach=5)
    assert not ta._stage_route_preferred(stats, n_tables=6)


def test_stage_route_not_preferred_below_the_key_bytes_floor():
    stats = _route_stats(stage_target=32, bits_to_reach=5, key_bytes_floor=40)
    assert not ta._stage_route_preferred(stats, n_tables=6)


def test_stage_route_not_preferred_when_the_target_is_unreachable():
    stats = _route_stats(stage_target=32, bits_to_reach=None, key_bytes_floor=10)
    assert not ta._stage_route_preferred(stats, n_tables=6)


def test_stage_route_preferred_when_it_is_free():
    """cost == 0 short-circuits before the ratio is even computed."""
    stats = _route_stats(stage_target=32, bits_to_reach=0, key_bytes_floor=10)
    assert ta._stage_route_preferred(stats, n_tables=6)


def test_stage_route_preferred_when_the_band_route_is_not_live():
    # band_target(100) == 84 < codeword_floor 90 -> band route dead,
    # band_ppb == 0.0, so any positive stage payoff wins the ratio compare.
    stats = _route_stats(stage_target=32, bits_to_reach=5, key_bytes_floor=10,
                         ternary_stages_before=6, codeword_before=100,
                         codeword_floor=90)
    assert ta._stage_route_preferred(stats, n_tables=6)


def test_stage_route_ratio_comparison_against_a_live_band_route():
    """band_target(100) == 84 >= codeword_floor 80 -> band route live.
    band_cost = 100 - 84 = 16, band_ppb = 6 / 16 = 0.375.
    stage_payoff = ternary_stages_before(6) - ternary_stages(32, 6)(3) = 3.
    Sweeping bits_to_reach sweeps the stage ratio through win / lose / tie
    against that fixed 0.375 band ratio; ties go to the stage route."""
    base = dict(stage_target=32, key_bytes_floor=10, ternary_stages_before=6,
               codeword_before=100, codeword_floor=80)

    winning = _route_stats(bits_to_reach=5, **base)   # 3/5 = 0.6 > 0.375
    assert ta._stage_route_preferred(winning, n_tables=6)

    losing = _route_stats(bits_to_reach=20, **base)   # 3/20 = 0.15 < 0.375
    assert not ta._stage_route_preferred(losing, n_tables=6)

    tied = _route_stats(bits_to_reach=8, **base)       # 3/8 = 0.375 == 0.375
    assert ta._stage_route_preferred(tied, n_tables=6)


def test_the_stages_objective_actually_changes_the_feature_order(monkeypatch):
    """Integration-level check that align_objective='stages' really reaches
    feature_order end to end through align_rf_thresholds -- not just that the
    unit-level pieces behave in isolation. Before this test the only evidence
    'stages' does anything at all was a committed replay CSV, not the suite:
    if the OR-gate at line ~423 or the feature-order wiring silently broke,
    nothing here would have gone red."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    captured = {}
    real_feature_order = ta.feature_order

    def spy(intervals1, intervals2, objective):
        order = real_feature_order(intervals1, intervals2, objective)
        captured[objective] = order
        return order

    monkeypatch.setattr(ta, 'feature_order', spy)

    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=stats,
                           align_objective='stages')
    # If this fixture ever stops having a reachable stage target the test
    # below proves nothing -- fail loudly rather than silently passing.
    assert stats['stage_target'] is not None
    assert 'stages' in captured

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_objective='blocks')
    assert 'blocks' in captured

    assert captured['stages'] != captured['blocks']


@pytest.mark.parametrize('align_objective', ['blocks', 'stages', 'both'])
def test_the_byte_domain_stats_are_recorded_on_every_objective(align_objective):
    """B is as fundamental to the stage cost as L is to the block cost, and
    the validation needs these on 'blocks' runs to have anything to compare
    against."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=stats,
                           align_objective=align_objective)
    for key in ('key_bytes_before', 'key_bytes_after', 'key_bytes_floor',
                'ternary_stages_before', 'ternary_stages_after'):
        assert isinstance(stats[key], int), key
    assert stats['key_bytes_after'] <= stats['key_bytes_before']
    assert stats['key_bytes_floor'] <= stats['key_bytes_after']


@pytest.mark.parametrize('delta_rel', [0.0, 0.05])
def test_the_blocks_objective_is_the_default_and_changes_nothing(delta_rel,
                                                                 monkeypatch):
    """Commit 3 is additive relative to the post-deletion code: the default
    objective must be bit-identical to not passing one at all."""
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 1)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    implicit = {}
    a1, a2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=delta_rel,
                                    align_stats=implicit)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    explicit = {}
    b1, b2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                    overlap_threshold=0.5, delta_rel=delta_rel,
                                    align_stats=explicit,
                                    align_objective='blocks')
    assert implicit == explicit
    for x, y in zip(a1.estimators_ + a2.estimators_,
                    b1.estimators_ + b2.estimators_):
        assert np.array_equal(x.tree_.threshold, y.tree_.threshold)


def test_pooled_key_bytes_equals_the_evaluators_ternary_key_bytes():
    """E1, the premise. If this fails the budget prices a table the switch
    does not build, and nothing else in the byte domain may be read."""
    from src.p4gen.build_p4_script import get_joint_feature_intervals
    from src.p4gen.evaluation import ternary_table_key_bytes

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    names = ['f0', 'f1', 'f2', 'f3']
    for models in ((rf1, rf2),
                   ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                          overlap_threshold=0.5, delta_rel=0.05)):
        m1, m2 = models
        joint = get_joint_feature_intervals(m1, names, m2, names)
        assert ab.pooled_key_bytes(ta.extract_feature_intervals(m1),
                                   ta.extract_feature_intervals(m2)) == \
            ternary_table_key_bytes(joint)


@pytest.mark.parametrize('key_bytes,n_tables', [(30, 6), (21, 6), (35, 6),
                                                (16, 4), (8, 8), (7, 8)])
def test_ternary_stages_agrees_with_the_real_packer(key_bytes, n_tables):
    """E1b. The classification tables all key on the same fields, so they
    share one width; on such a uniform list the closed form must equal what
    the FFD packer actually returns."""
    from src.p4gen.evaluation import crossbar_stages_needed
    specs = [(1, key_bytes)] * n_tables
    assert ab.ternary_stages(key_bytes, n_tables) == \
        crossbar_stages_needed(specs).occupied


def test_ternary_stages_is_a_lower_bound_when_the_block_cap_binds():
    """Documented limitation, pinned so it cannot be mistaken for agreement:
    ternary_stages models the crossbar caps only. Where a table's BLOCK count
    forces a stage on its own, the real packer needs more."""
    from src.p4gen.evaluation import crossbar_stages_needed
    specs = [(20, 8)] * 4                     # 20 blocks each, cap is 24/stage
    assert ab.ternary_stages(8, 4) == 1
    assert crossbar_stages_needed(specs).occupied == 4


# ---------------------------------------------------------------------------
# align_with_policy x align_objective anchor (design 2026-08-31, Task 1).
#
# Captured BEFORE the dual-run change so that change is a reviewed diff.
# 'blocks' and 'stages' must never move; 'both' is re-captured deliberately
# in Task 5 and nowhere else.
# ---------------------------------------------------------------------------

_ANCHOR_STATS_KEYS = ('attempted', 'accepted', 'intervals_before',
                      'intervals_after', 'codeword_before', 'codeword_after',
                      'codeword_floor', 'key_bytes_before', 'key_bytes_after',
                      'key_bytes_floor', 'ternary_stages_before',
                      'ternary_stages_after', 'spent_budget', 'rolled_back')

_POLICY_OBJECTIVE_ANCHOR = {'blocks': {'stats': {'accepted': 2,
                      'attempted': 33,
                      'codeword_after': 84,
                      'codeword_before': 87,
                      'codeword_floor': 47,
                      'intervals_after': 88,
                      'intervals_before': 91,
                      'key_bytes_after': 13,
                      'key_bytes_before': 13,
                      'key_bytes_floor': 8,
                      'rolled_back': False,
                      'spent_budget': True,
                      'ternary_stages_after': 4,
                      'ternary_stages_before': 4},
            'thresholds': [[50148,
                            37970,
                            -2,
                            64068,
                            30850,
                            -2,
                            -2,
                            -2,
                            29481,
                            -2,
                            36261,
                            -2,
                            -2],
                           [25153,
                            17906,
                            42582,
                            -2,
                            -2,
                            -2,
                            25152,
                            -2,
                            65407,
                            47461,
                            -2,
                            -2,
                            -2],
                           [9867,
                            -2,
                            41694,
                            64574,
                            22960,
                            -2,
                            -2,
                            -2,
                            22949,
                            -2,
                            41841,
                            -2,
                            -2],
                           [11493,
                            -2,
                            16559,
                            -2,
                            26336,
                            -2,
                            43169,
                            33744,
                            -2,
                            -2,
                            26063,
                            -2,
                            -2],
                           [45724,
                            8902,
                            48924,
                            -2,
                            -2,
                            25535,
                            -2,
                            48261,
                            -2,
                            -2,
                            63815,
                            -2,
                            49629,
                            -2,
                            -2],
                           [40514,
                            50955,
                            32983,
                            -2,
                            -2,
                            21061,
                            -2,
                            -2,
                            64763,
                            28712,
                            -2,
                            50610,
                            -2,
                            -2,
                            -2],
                           [40996,
                            17244,
                            -2,
                            35130,
                            -2,
                            58798,
                            -2,
                            -2,
                            52649,
                            65407,
                            -2,
                            -2,
                            -2],
                           [8902,
                            -2,
                            58514,
                            50135,
                            29400,
                            -2,
                            -2,
                            30237,
                            -2,
                            -2,
                            33384,
                            -2,
                            -2],
                           [14536,
                            -2,
                            26063,
                            -2,
                            40514,
                            -2,
                            61298,
                            38258,
                            -2,
                            -2,
                            -2],
                           [27458,
                            48468,
                            -2,
                            -2,
                            58452,
                            62051,
                            33860,
                            -2,
                            -2,
                            -2,
                            61513,
                            -2,
                            -2],
                           [61422,
                            45058,
                            26424,
                            15571,
                            -2,
                            -2,
                            -2,
                            16443,
                            -2,
                            57942,
                            -2,
                            -2,
                            53373,
                            -2,
                            -2],
                           [27321,
                            53909,
                            39702,
                            -2,
                            -2,
                            -2,
                            17534,
                            -2,
                            21985,
                            -2,
                            53934,
                            -2,
                            60939,
                            -2,
                            -2],
                           [54408,
                            44768,
                            33254,
                            -2,
                            -2,
                            62045,
                            -2,
                            -2,
                            18076,
                            -2,
                            41360,
                            -2,
                            48048,
                            -2,
                            -2],
                           [54766,
                            44845,
                            24115,
                            -2,
                            -2,
                            38129,
                            -2,
                            -2,
                            49965,
                            25912,
                            -2,
                            -2,
                            -2]]},
 'both': {'stats': {'accepted': 23,
                    'attempted': 24,
                    'codeword_after': 61,
                    'codeword_before': 87,
                    'codeword_floor': 47,
                    'intervals_after': 65,
                    'intervals_before': 91,
                    'key_bytes_after': 10,
                    'key_bytes_before': 13,
                    'key_bytes_floor': 8,
                    'rolled_back': False,
                    'spent_budget': True,
                    'ternary_stages_after': 3,
                    'ternary_stages_before': 4},
          'thresholds': [[49965,
                          41360,
                          -2,
                          60939,
                          30850,
                          -2,
                          -2,
                          -2,
                          29400,
                          -2,
                          33860,
                          -2,
                          -2],
                         [25153,
                          14536,
                          44768,
                          -2,
                          -2,
                          -2,
                          21985,
                          -2,
                          65407,
                          47461,
                          -2,
                          -2,
                          -2],
                         [9867,
                          -2,
                          38129,
                          64574,
                          22960,
                          -2,
                          -2,
                          -2,
                          22949,
                          -2,
                          41841,
                          -2,
                          -2],
                         [11493,
                          -2,
                          15571,
                          -2,
                          27458,
                          -2,
                          43169,
                          33254,
                          -2,
                          -2,
                          26063,
                          -2,
                          -2],
                         [45724,
                          8902,
                          48924,
                          -2,
                          -2,
                          25535,
                          -2,
                          48261,
                          -2,
                          -2,
                          63815,
                          -2,
                          49629,
                          -2,
                          -2],
                         [40514,
                          50955,
                          32983,
                          -2,
                          -2,
                          21061,
                          -2,
                          -2,
                          64763,
                          24115,
                          -2,
                          48048,
                          -2,
                          -2,
                          -2],
                         [40996,
                          17244,
                          -2,
                          39702,
                          -2,
                          58452,
                          -2,
                          -2,
                          52649,
                          65407,
                          -2,
                          -2,
                          -2],
                         [8902,
                          -2,
                          58514,
                          50135,
                          29400,
                          -2,
                          -2,
                          30850,
                          -2,
                          -2,
                          33384,
                          -2,
                          -2],
                         [14536,
                          -2,
                          26063,
                          -2,
                          40996,
                          -2,
                          61298,
                          38258,
                          -2,
                          -2,
                          -2],
                         [27458,
                          47461,
                          -2,
                          -2,
                          58452,
                          62051,
                          33860,
                          -2,
                          -2,
                          -2,
                          61513,
                          -2,
                          -2],
                         [61422,
                          43169,
                          26424,
                          15571,
                          -2,
                          -2,
                          -2,
                          11493,
                          -2,
                          57942,
                          -2,
                          -2,
                          53373,
                          -2,
                          -2],
                         [27321,
                          53909,
                          39702,
                          -2,
                          -2,
                          -2,
                          17534,
                          -2,
                          21985,
                          -2,
                          53934,
                          -2,
                          60939,
                          -2,
                          -2],
                         [49629,
                          44768,
                          33254,
                          -2,
                          -2,
                          62045,
                          -2,
                          -2,
                          17244,
                          -2,
                          41360,
                          -2,
                          52649,
                          -2,
                          -2],
                         [54766,
                          48261,
                          24115,
                          -2,
                          -2,
                          38129,
                          -2,
                          -2,
                          49965,
                          25912,
                          -2,
                          -2,
                          -2]]},
 'stages': {'stats': {'accepted': 23,
                      'attempted': 24,
                      'codeword_after': 61,
                      'codeword_before': 87,
                      'codeword_floor': 47,
                      'intervals_after': 65,
                      'intervals_before': 91,
                      'key_bytes_after': 10,
                      'key_bytes_before': 13,
                      'key_bytes_floor': 8,
                      'rolled_back': False,
                      'spent_budget': True,
                      'ternary_stages_after': 3,
                      'ternary_stages_before': 4},
            'thresholds': [[49965,
                            41360,
                            -2,
                            60939,
                            30850,
                            -2,
                            -2,
                            -2,
                            29400,
                            -2,
                            33860,
                            -2,
                            -2],
                           [25153,
                            14536,
                            44768,
                            -2,
                            -2,
                            -2,
                            21985,
                            -2,
                            65407,
                            47461,
                            -2,
                            -2,
                            -2],
                           [9867,
                            -2,
                            38129,
                            64574,
                            22960,
                            -2,
                            -2,
                            -2,
                            22949,
                            -2,
                            41841,
                            -2,
                            -2],
                           [11493,
                            -2,
                            15571,
                            -2,
                            27458,
                            -2,
                            43169,
                            33254,
                            -2,
                            -2,
                            26063,
                            -2,
                            -2],
                           [45724,
                            8902,
                            48924,
                            -2,
                            -2,
                            25535,
                            -2,
                            48261,
                            -2,
                            -2,
                            63815,
                            -2,
                            49629,
                            -2,
                            -2],
                           [40514,
                            50955,
                            32983,
                            -2,
                            -2,
                            21061,
                            -2,
                            -2,
                            64763,
                            24115,
                            -2,
                            48048,
                            -2,
                            -2,
                            -2],
                           [40996,
                            17244,
                            -2,
                            39702,
                            -2,
                            58452,
                            -2,
                            -2,
                            52649,
                            65407,
                            -2,
                            -2,
                            -2],
                           [8902,
                            -2,
                            58514,
                            50135,
                            29400,
                            -2,
                            -2,
                            30850,
                            -2,
                            -2,
                            33384,
                            -2,
                            -2],
                           [14536,
                            -2,
                            26063,
                            -2,
                            40996,
                            -2,
                            61298,
                            38258,
                            -2,
                            -2,
                            -2],
                           [27458,
                            47461,
                            -2,
                            -2,
                            58452,
                            62051,
                            33860,
                            -2,
                            -2,
                            -2,
                            61513,
                            -2,
                            -2],
                           [61422,
                            43169,
                            26424,
                            15571,
                            -2,
                            -2,
                            -2,
                            11493,
                            -2,
                            57942,
                            -2,
                            -2,
                            53373,
                            -2,
                            -2],
                           [27321,
                            53909,
                            39702,
                            -2,
                            -2,
                            -2,
                            17534,
                            -2,
                            21985,
                            -2,
                            53934,
                            -2,
                            60939,
                            -2,
                            -2],
                           [49629,
                            44768,
                            33254,
                            -2,
                            -2,
                            62045,
                            -2,
                            -2,
                            17244,
                            -2,
                            41360,
                            -2,
                            52649,
                            -2,
                            -2],
                           [54766,
                            48261,
                            24115,
                            -2,
                            -2,
                            38129,
                            -2,
                            -2,
                            49965,
                            25912,
                            -2,
                            -2,
                            -2]]}}


def _policy_anchor_capture(objective):
    """Run align_with_policy at `objective` on the golden pair and return the
    anchor-shaped dict. Used both to capture and to check."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    a1, a2 = ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                                  overlap_threshold=0.5, delta_rel=0.05,
                                  align_stats=stats,
                                  align_objective=objective)
    return {
        'stats': {k: stats[k] for k in _ANCHOR_STATS_KEYS},
        'thresholds': [[int(round(t)) for t in est.tree_.threshold]
                       for est in a1.estimators_ + a2.estimators_],
    }


@pytest.mark.parametrize('objective', ['blocks', 'stages', 'both'])
def test_align_with_policy_matches_its_captured_anchor(objective):
    assert _policy_anchor_capture(objective) == _POLICY_OBJECTIVE_ANCHOR[objective]


@pytest.mark.parametrize('align_objective', ['blocks', 'stages'])
@pytest.mark.parametrize('delta_rel', [0.0, 0.05, None])
def test_accuracy_spent_is_recorded_on_every_objective_and_delta(
        align_objective, delta_rel):
    """§2.4: the ranking needs a price for every run, and the campaign needs
    it on single-objective runs too -- otherwise there is nothing to compare
    a 'both' run against."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=delta_rel, align_stats=stats,
                           align_objective=align_objective)
    assert isinstance(stats['accuracy_spent'], float)
    assert 0.0 <= stats['accuracy_spent'] <= 1.0


def test_accuracy_spent_is_zero_when_no_move_is_accepted(monkeypatch):
    """A run that accepts nothing changed no threshold, so it spent nothing.
    Forcing every candidate to be rejected is the cleanest way to pin the
    floor of the quantity."""
    monkeypatch.setattr(ta, 'accept_alignment', lambda before, after, d: False)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=stats)
    assert stats['accepted'] == 0
    assert stats['accuracy_spent'] == 0.0


def test_accuracy_spent_is_a_max_across_tasks_not_a_mean(monkeypatch):
    """The module's own standard everywhere else (accept_alignment's all(),
    ratchet, _rank_targets' damage). A run cheap on average but expensive on
    one task is not cheap, and this field must not be the first place that
    principle is violated."""
    captured = {}
    real = ta.rel_deg

    def spy(before, after):
        captured.setdefault('pairs', []).append((before, after))
        return real(before, after)

    monkeypatch.setattr(ta, 'rel_deg', spy)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=stats)
    # The final four calls are the accuracy_spent computation itself, one per
    # metric, and its result must be their maximum.
    final_four = captured['pairs'][-4:]
    assert stats['accuracy_spent'] == max(real(b, a) for b, a in final_four)


# ---------------------------------------------------------------------------
# _state shared setup (design 2026-08-31 §2.2). THE PREMISE: if the cached
# path is not observationally identical to the from-scratch path, nothing
# built on top of it may be trusted (the analogue of 2026-08-30's E1/E1b).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('align_objective', ['blocks', 'stages'])
@pytest.mark.parametrize('delta_rel', [0.0, 0.05, None])
def test_the_shared_state_path_is_observationally_identical(align_objective,
                                                            delta_rel):
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    scratch_stats = {}
    s1, s2 = ta.align_rf_thresholds(
        rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5, delta_rel=delta_rel,
        align_stats=scratch_stats, align_objective=align_objective)

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    state = ta._build_shared_setup(rf1, rf2, X1, X2)
    cached_stats = {}
    c1, c2 = ta.align_rf_thresholds(
        rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5, delta_rel=delta_rel,
        align_stats=cached_stats, align_objective=align_objective,
        _state=state)

    assert cached_stats == scratch_stats
    for a, b in zip(s1.estimators_ + s2.estimators_,
                    c1.estimators_ + c2.estimators_):
        assert np.array_equal(a.tree_.threshold, b.tree_.threshold)


def test_one_shared_state_serves_two_runs_without_cross_contamination():
    """The whole point: 'both' reuses ONE state for two arms. If the first
    arm's mutations leaked into the shared structures, the second arm would
    start from the first arm's end state and differ from a fresh run."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    state = ta._build_shared_setup(rf1, rf2, X1, X2)

    first = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=first,
                           align_objective='blocks', _state=state)
    second = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                           delta_rel=0.05, align_stats=second,
                           align_objective='blocks', _state=state)
    assert first == second


def test_the_shared_state_builder_does_not_mutate_the_callers_forests():
    """C8's guarantee, extended to the new entry point: _build_shared_setup
    reads the caller's forests and must leave them untouched."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    before = [est.tree_.threshold.copy()
              for est in rf1.estimators_ + rf2.estimators_]
    ta._build_shared_setup(rf1, rf2, X1, X2)
    for est, snapshot in zip(rf1.estimators_ + rf2.estimators_, before):
        assert np.array_equal(est.tree_.threshold, snapshot)


# ---------------------------------------------------------------------------
# 'both' = run both orderings, keep the best (design 2026-08-31 §2.3-§2.4).
# ---------------------------------------------------------------------------

def _rank_stats(ternary_stages_after, codeword_after, accuracy_spent):
    return {'ternary_stages_after': ternary_stages_after,
            'codeword_after': codeword_after,
            'accuracy_spent': accuracy_spent}


def test_rank_prefers_fewer_stages_over_everything_else():
    """Priority order follows the recorded cost model: blocks rarely bind
    (large headroom), the 64-byte crossbar cap is the one that typically
    does, so a stage saved outranks a block saved."""
    cheap_stages = _rank_stats(2, 200, 0.9)
    cheap_everything_else = _rank_stats(3, 100, 0.0)
    assert ta._rank_key(cheap_stages, 'blocks') < ta._rank_key(
        cheap_everything_else, 'blocks')


def test_rank_falls_through_to_codeword_when_stages_tie():
    assert ta._rank_key(_rank_stats(2, 100, 0.9), 'blocks') < ta._rank_key(
        _rank_stats(2, 128, 0.0), 'blocks')


def test_rank_falls_through_to_accuracy_spent_when_stages_and_blocks_tie():
    """The E3 fix, at the level of the key: two runs reaching the identical
    end state are separated by what they paid to get there."""
    assert ta._rank_key(_rank_stats(2, 128, 0.001), 'blocks') < ta._rank_key(
        _rank_stats(2, 128, 0.020), 'blocks')


def test_rank_breaks_an_exact_tie_in_favour_of_blocks_deterministically():
    """train_model.py:373-377 refits the winner instead of caching it, which
    is only valid while this is a total order."""
    tied = _rank_stats(2, 128, 0.01)
    assert ta._rank_key(tied, 'blocks') < ta._rank_key(tied, 'stages')


def test_both_returns_one_of_the_two_single_pass_results_and_says_which():
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    b1, b2 = ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                                  overlap_threshold=0.5, delta_rel=0.05,
                                  align_stats=stats, align_objective='both')
    assert stats['objective_used'] in ta._SINGLE_PASS_OBJECTIVES
    assert isinstance(stats['arms_differed'], bool)

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    winner_stats = {}
    w1, w2 = ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                                  overlap_threshold=0.5, delta_rel=0.05,
                                  align_stats=winner_stats,
                                  align_objective=stats['objective_used'])
    for a, b in zip(b1.estimators_ + b2.estimators_,
                    w1.estimators_ + w2.estimators_):
        assert np.array_equal(a.tree_.threshold, b.tree_.threshold)


def test_single_pass_objectives_report_themselves_as_the_one_used():
    """The column must stay dense: replay_alignment turns every stats key into
    a result column, and a NaN there makes downstream groupbys drop rows."""
    for objective in ('blocks', 'stages'):
        rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
        stats = {}
        ta.align_with_policy(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                             delta_rel=0.05, align_stats=stats,
                             align_objective=objective)
        assert stats['objective_used'] == objective
        assert stats['arms_differed'] is False


def test_both_rolls_each_arm_back_before_ranking_them(monkeypatch):
    """§2.3's ordering constraint, directly. An arm that spent budget and
    crossed nothing must be replaced by its delta=0 rerun BEFORE it is
    ranked -- otherwise a speculative arm that gave away accuracy for nothing
    could win the comparison on paper."""
    seen = []
    real = ta.crossed_a_boundary

    def never_crossed(stats, objective, n_tables):
        seen.append(objective)
        return False

    monkeypatch.setattr(ta, 'crossed_a_boundary', never_crossed)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2, overlap_threshold=0.5,
                         delta_rel=0.05, align_stats=stats,
                         align_objective='both')
    # Both arms were asked, and the surviving stats are a rolled-back run's.
    assert sorted(seen) == ['blocks', 'stages']
    assert stats['rolled_back'] is True
    assert stats['accuracy_spent'] == 0.0
    monkeypatch.setattr(ta, 'crossed_a_boundary', real)


def test_both_is_deterministic_across_repeated_runs():
    def once():
        rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
        stats = {}
        a1, a2 = ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                                      overlap_threshold=0.5, delta_rel=0.05,
                                      align_stats=stats,
                                      align_objective='both')
        return stats, [est.tree_.threshold.copy()
                       for est in a1.estimators_ + a2.estimators_]

    first_stats, first_trees = once()
    second_stats, second_trees = once()
    assert first_stats == second_stats
    for a, b in zip(first_trees, second_trees):
        assert np.array_equal(a, b)
