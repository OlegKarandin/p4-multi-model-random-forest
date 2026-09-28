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


def _target_range(range1, range2):
    """Test-local stand-in for the pruned calculate_target_range (task 14):
    the intersection of two ranges, exactly as align_rf_thresholds's own
    _rank_targets/hypothetical_ranges machinery would compute it for the
    plain intersection case these fixtures exercise."""
    return (max(range1[0], range2[0]), min(range1[1], range2[1]))


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
                                  delta_rel=None)


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
                           delta_rel=None)

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
                                        delta_rel=None,
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
                           delta_rel=0.05,
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
    structurally_alignable already vetoes any pair where exactly one side
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
    is positive. target_range is computed via _target_range (the plain
    intersection, matching what a call from align_rf_thresholds would produce
    in this case), so this exercises the real shape of a call, not a
    contrived one. threshold_index is deliberately empty -- if either
    branch DID try to look up a threshold, that would raise rather than
    silently pass, so an empty modifications list is real evidence of the
    refusal, not an accident of a missing key."""
    rf = _one_split_forest()

    # Other side's min is 0: _target_range((0,0), (0, 10)) == (0, 0),
    # so both the min- and max-side checks in adjust_range_boundaries see no
    # change and refuse.
    other_min_zero = (0, 10)
    target_a = _target_range((0, 0), other_min_zero)
    modifications_a = ta.adjust_range_boundaries(
        rf, feature_idx=0, source_range=(0, 0), target_range=target_a,
        threshold_index={})
    assert modifications_a == []

    # Other side's min is not 0: _target_range((0,0), (5, 10)) ==
    # (5, 0) -- the min-side check is refused because threshold_source_min is
    # 0, and the max-side check sees threshold_source_max == threshold_target
    # _max == 0.
    other_min_nonzero = (5, 10)
    target_b = _target_range((0, 0), other_min_nonzero)
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
# It must likewise not be regenerated after the deletion -- EXCEPT where a
# real behavioural change legitimately moves it; see each arm's own comment.
#
# Regenerated a THIRD time 2026-09-14 by Task 7 (design D4): the ratio test
# `overlap_ratio < overlap_threshold` that used to gate admission is gone
# unconditionally, so both arms now consider candidates the 0.5 default used
# to reject -- there is no gate-state exemption left for either arm.
#   0.0:  attempted 27->46, accepted 17->26, intervals_after 71->62
#   0.05: attempted 29->40, accepted 10->27, intervals_after 81->62
#
# Regenerated a FOURTH time 2026-09-14 by Task 12: feature_order now ranks by
# blocks bought, recomputed per feature against the budget's live widths,
# instead of byte-completion order computed once at entry -- a real,
# documented visiting-order change (D6), so this MAX_RECOMPUTE_ROUNDS=1 gate
# legitimately moves. 0.0 barely moves (attempted 46->41, accepted 26->27,
# intervals_after unchanged at 62) since delta_rel == 0.0 forces
# effective_delta == 0.0 regardless of order. 0.05 moves hard the other way
# (attempted 40->117, accepted 27->10, intervals_after 62->81): D6's
# non-guarantee, realised on this exact pair -- the new order is a better
# HEURISTIC on average, not a per-row improvement, and this golden pin is the
# proof it is not one here.
_ALIGNMENT_GOLDEN_C1C2 = {
    0.0: {
        'stats': {'attempted': 41, 'accepted': 27,
                  'intervals_before': 91, 'intervals_after': 62},
        't1': [
            [50135, 33860, -2, 64068, 30850, -2, -2, -2, 29400, -2, 33384, -2,
             -2],
            [25153, 17906, 44768, -2, -2, -2, 25152, -2, 65407, 47461, -2, -2,
             -2],
            [9867, -2, 38129, 64574, 22960, -2, -2, -2, 22949, -2, 41841, -2,
             -2],
            [11493, -2, 15571, -2, 26424, -2, 43169, 33254, -2, -2, 26063, -2,
             -2],
            [45724, 17534, 48924, -2, -2, 25535, -2, 44845, -2, -2, 53373, -2,
             49629, -2, -2],
            [39753, 58452, 32983, -2, -2, 21061, -2, -2, 64763, 24115, -2,
             50610, -2, -2, -2],
            [40996, 17244, -2, 35130, -2, 58798, -2, -2, 60939, 65407, -2, -2,
             -2],
        ],
        't2': [
            [8902, -2, 58514, 50135, 29400, -2, -2, 30850, -2, -2, 33384, -2,
             -2],
            [14536, -2, 39753, -2, 40996, -2, 61298, 35130, -2, -2, -2],
            [27458, 47461, -2, -2, 58452, 65407, 33860, -2, -2, -2, 61513, -2,
             -2],
            [61422, 43169, 26424, 15571, -2, -2, -2, 11493, -2, 57942, -2, -2,
             53373, -2, -2],
            [27321, 53909, 49629, -2, -2, -2, 17534, -2, 25152, -2, 53934, -2,
             60939, -2, -2],
            [54408, 44768, 33254, -2, -2, 62045, -2, -2, 17244, -2, 41841, -2,
             50610, -2, -2],
            [54766, 48924, 24115, -2, -2, 38129, -2, -2, 49965, 26063, -2, -2,
             -2],
        ],
    },
    # The 0.05 arm was deleted here on 2026-09-15 with the delta_align
    # mechanism (Track 5: delta_helps = FALSE -- mean_d000 0.7956173344395895
    # vs mean_d020 0.7861922400433382, cells_favouring_d020 14/24). Its last
    # recorded state was attempted 117, accepted 10, intervals_after 81.
    #
    # Removing BlockBudget changes what a non-zero delta DOES: the budget used
    # to clamp the effective delta to 0.0 once the floor factor was reached,
    # and nothing clamps it now, so this arm would move to attempted 30,
    # accepted 29, intervals_after 62. That is a legitimate behavioural change
    # on an unreachable input -- no caller can supply a non-zero delta any more
    # -- but this literal predates the change it is supposed to gate and must
    # NOT be regenerated from post-change code (see the test's docstring), so
    # the arm is dropped rather than refreshed. The 0.0 arm, which is the only
    # one the pipeline can produce, comes through bit-identical and keeps the
    # gate's whole T2b/C3 job.
}


@pytest.mark.parametrize('delta_rel', [0.0])
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

    Parametrized over one value since 2026-09-15: the 0.05 arm went with the
    delta_align mechanism (see the literal's own comment). delta_rel = 0.0 is
    the only value any caller can now produce, and it came through that
    deletion bit-identical too.
    """
    golden = _ALIGNMENT_GOLDEN_C1C2[delta_rel]
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 1)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}

    # C8: align_rf_thresholds no longer mutates rf1/rf2 in place -- it returns
    # copies -- so the aligned models to check are the returned ones.
    rf1, rf2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
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

    find_partially_overlapping_ranges' one sweep over these lists reports TWO
    pairs at (1000,5999): first (1000,5999)&(1000,2999) (sweep order is (i
    ascending, j ascending)), then (1000,5999)&(3000,5999). Before Task 7
    (design D4) the ratio gate admitted only the second (ratio 0.5999 >=
    0.5's threshold; the first, at ratio 0.39997, was rejected before ever
    reaching still_overlaps), so round 1 had exactly one eligible candidate.
    With the gate gone, both are OFFERED in round 1, but only the first is
    ever ATTEMPTED: accepting it immediately shrinks (1000,5999) to
    (1000,2999) and widens the neighbour (6000,INF) left to (3000,INF), so
    the second pair, re-read afterward, no longer overlaps at all -- exactly
    Task 6's `still_overlaps` check, added for this reason, rejects it.

    The widened neighbour itself then creates further candidates only the
    recompute can reach -- see
    test_a_widened_neighbour_becomes_a_candidate_only_after_the_recompute for
    the full cascade this fixture now produces under delta_rel=None.
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
    (test_the_joint_arm_always_builds_the_metric_oracle in
    test_train_model_contract.py, which actually proves the oracle is always
    built by spying IncrementalMetrics.__init__ and asserting both tasks were
    scored, and the C2 tests respectively);
    restoring the pre-ladder single-target, always-accept shape here
    reproduces this file's pre-Task-3 recompute numbers exactly.
    """
    monkeypatch.setattr(ta, 'accept_alignment', lambda *a, **k: True)

    def _single_legacy_target(range1, range2, ranges1, ranges2, idx1, idx2,
                              *a, **k):
        # Same stand-in as before (the legacy intersection target, offered
        # unconditionally with no gain filter) reshaped to _rank_targets'
        # (before, [(target, after), ...]) contract: `before`/`after` are
        # computed for real via the same pure helpers _rank_targets itself
        # uses, not faked, so the caller's shed bookkeeping stays numerically
        # sound even though these C3 tests don't assert on it.
        target = _target_range(range1, range2)
        before = ta.pooled_interval_count(ranges1, ranges2)
        hypo1 = ta.hypothetical_ranges(ranges1, idx1, range1, target)
        hypo2 = ta.hypothetical_ranges(ranges2, idx2, range2, target)
        after = (ta.pooled_interval_count(hypo1, hypo2)
                 if hypo1 is not None and hypo2 is not None else before)
        return before, [(target, after)]

    monkeypatch.setattr(ta, '_rank_targets', _single_legacy_target)


def test_a_widened_neighbour_becomes_a_candidate_only_after_the_recompute(monkeypatch):
    """THE motivating test for C3: without the rescan later pairs are unreachable.

    With the recompute disabled (a single round) the fixture attempts exactly
    the one candidate still_overlaps admits after the first move (see
    _neighbour_widening_pair's docstring for why the sweep's second-listed
    pair is offered but never attempted here). With it enabled, the widened
    neighbour cascades into three further candidates across three more
    rounds -- reachable only because each round recomputes the overlap list
    from the just-mutated ranges.

    Regenerated 2026-09-14 by Task 7 (design D4): the ratio gate that used to
    make this fixture's story a clean two-candidate example is gone, so the
    single-round pass now attempts the ratio-0.39997 pair instead of the
    ratio-0.5999 one (both were always "eligible" by the surviving structural
    checks; the ratio was the only thing choosing between them), and the full
    recompute cascades to 4 rounds instead of 2. Captured directly from this
    commit's code via a throwaway script -- not hand-derived -- so it is a
    characterisation pin, not a re-derivation of the mechanism.
    """
    _isolate_c3(monkeypatch)
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 1)
    rf1, rf2, X, y1, y2 = _neighbour_widening_pair()
    stats_one_round, log_one_round = {}, []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
                           delta_rel=None, align_stats=stats_one_round,
                           candidate_log=log_one_round)

    assert [(e['range1'], e['range2']) for e in log_one_round] == \
        [((1000, 5999), (1000, 2999))]
    assert stats_one_round['attempted'] == 1

    # undo() only lifts the MAX_RECOMPUTE_ROUNDS=1 patch -- _isolate_c3's two
    # patches must stay in place for the full-recompute run below too, or the
    # zero-gain filter and the now-unconditional accuracy gate reintroduce
    # exactly the interference _isolate_c3 exists to remove.
    monkeypatch.undo()
    _isolate_c3(monkeypatch)
    rf1, rf2, X, y1, y2 = _neighbour_widening_pair()
    stats, log = {}, []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
                           delta_rel=None, align_stats=stats,
                           candidate_log=log)

    assert [(e['range1'], e['range2']) for e in log] == [
        ((1000, 5999), (1000, 2999)),
        ((3000, INFINITE), (6000, INFINITE)),
        ((1000, 5999), (3000, 5999)),
        ((100, 2999), (100, 999)),
    ]
    assert [e['round'] for e in log] == [1, 2, 3, 4]
    assert stats['attempted'] == 4 and stats['accepted'] == 4


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
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
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
        ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
                               delta_rel=None)

    assert 'fixpoint' in str(excinfo.value).lower()


def test_the_recompute_never_evaluates_the_same_value_pair_twice(monkeypatch):
    """`seen` keys on VALUE pairs, not index pairs -- an accepted move rewrites
    tuples in place, so the same index pair names a different candidate in a
    later round and the same candidate can move to a different index. Without
    it every round would re-offer every pair it had already judged."""
    judged = []
    real = ta.structurally_alignable

    def spy(range1, range2):
        judged.append((range1, range2))
        return real(range1, range2)

    # structurally_alignable is the last of the two unconditional correctness
    # checks the loop runs on a non-seen pair -- exactly the position the
    # since-pruned calculate_range_overlap (task 14) used to sit in, so
    # spying here counts the same "one judgement per non-seen candidate pair"
    # this test was built to pin.
    monkeypatch.setattr(ta, 'structurally_alignable', spy)
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
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2,
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
    target = _target_range(range1, range2)
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
                                   delta_rel=delta,
                                   align_stats=stats)
            if stats['intervals_after'] > stats['intervals_before']:
                violations.append((seed, delta, stats))

    assert violations == []


def _align_golden_pair(delta_rel, cap, monkeypatch):
    """One run of the golden fixture at a given MAX_RECOMPUTE_ROUNDS."""
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', cap)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats, log = {}, []
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
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

    Whether C3's extra rounds surface anything NEW on this fixture is a
    measured fact of the current feature order and admission rule, not a
    theorem. It has moved twice: under the pre-2026-09-07 combined-count
    order every arm reached round 2; under byte-completion order only
    delta_rel=0.0 did, because the ratio gate then in place made the
    None/0.05 round-1 candidate set already the recompute fixpoint. Task 7
    (design D4) removed that gate, and with it removed the fixpoint: all
    three arms now attempt more in the full recompute than in round 1 alone
    (measured: 40->43 for None, 46->49 for 0.0, 40->43 for 0.05).
    """
    stats_r1, log_r1 = _align_golden_pair(delta_rel, 1, monkeypatch)
    stats_c3, log_c3 = _align_golden_pair(delta_rel, ta.MAX_RECOMPUTE_ROUNDS, monkeypatch)

    # No new stats key from C3 itself -- 'round' lives in the candidate_log
    # instead. key_bytes_before/after and bits_to_reach are recorded
    # unconditionally regardless of C3 round depth. 'accuracy_spent' is
    # recorded on every run. factor_before/after/floor are the per-table
    # block-factor keys and total_blocks_before/after/floor are the block-total
    # keys (Task 11); the 'stages'-era ternary_stages_*/stage_target keys and
    # the objective axis itself are retired (design 2026-09-07 §4.3/§4.5),
    # 'spent_budget'/'rolled_back' went with the delta_align mechanism on
    # 2026-09-15 (Track 5: delta_helps = FALSE), and codeword_floor /
    # key_bytes_floor were pruned the same day as stats-only diagnostics
    # (task 14).
    assert set(stats_c3) == {
        'attempted', 'accepted', 'intervals_before', 'intervals_after',
        'codeword_before', 'codeword_after',
        'accuracy_spent',
        'key_bytes_before', 'key_bytes_after',
        'bits_to_reach',
        'factor_before', 'factor_after', 'factor_floor',
        'total_blocks_before', 'total_blocks_after', 'total_blocks_floor'}
    assert stats_c3['intervals_before'] == stats_r1['intervals_before']
    assert stats_c3['attempted'] >= stats_r1['attempted']
    assert stats_c3['accepted'] >= stats_r1['accepted']

    moves_r1, moves_c3 = _accepted_moves(log_r1), _accepted_moves(log_c3)
    assert moves_r1, 'the fixture must accept something in the single-round pass'
    # Under the now-unconditional c1c2 target ranking (2026-08-30 ladder
    # deletion) a recomputed round can surface a candidate that C2's gain
    # filter or the now-mandatory C1 accuracy gate then rejects, so ACCEPTED
    # count does not have to grow in lockstep with ATTEMPTED. attempted is
    # the honest "C3 found something new" signal, and as of Task 7 (design
    # D4) it does on every arm of this fixture -- there is no longer a
    # gate-produced fixpoint at round 1 (see this test's docstring).
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
    # As of Task 7 (design D4), all three arms find something in a later
    # round on this fixture -- see the docstring's measured note.
    assert max(e['round'] for e in log_c3) > 1


def test_align_stats_records_the_codeword_length_it_optimises():
    """L is the quantity the block cost is a step function of, and it was
    recorded nowhere -- the companion analysis had to solve for it and failed
    on 8% of rows."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           delta_rel=0.0, align_stats=stats)

    assert set(stats) == {
        'attempted', 'accepted', 'intervals_before', 'intervals_after',
        'codeword_before', 'codeword_after',
        'accuracy_spent',
        'key_bytes_before', 'key_bytes_after',
        'bits_to_reach',
        'factor_before', 'factor_after', 'factor_floor',
        'total_blocks_before', 'total_blocks_after', 'total_blocks_floor'}

    n_features = len(set(ta.extract_feature_intervals(rf1))
                     | set(ta.extract_feature_intervals(rf2)))
    assert stats['codeword_before'] == stats['intervals_before'] - n_features
    assert stats['codeword_after'] == stats['intervals_after'] - n_features


def test_accuracy_spent_is_zero_when_no_move_is_accepted(monkeypatch):
    """A run that accepts nothing changed no threshold, so it spent nothing.
    Forcing every candidate to be rejected is the cleanest way to pin the
    floor of the quantity."""
    monkeypatch.setattr(ta, 'accept_alignment', lambda before, after, d: False)
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
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
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           delta_rel=0.05, align_stats=stats)
    # The final four calls are the accuracy_spent computation itself, one per
    # metric, and its result must be their maximum.
    final_four = captured['pairs'][-4:]
    assert stats['accuracy_spent'] == max(0.0, max(real(b, a) for b, a in final_four))


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
                                    delta_rel=0.0,
                                    align_stats=stats)
    usage = multi_model_memory_evaluation(a1, a2, names, names, 'joint')
    assert stats['codeword_after'] == usage.codeword_length


# ---------------------------------------------------------------------------
# Budget wiring.
#
# Three tests stood here until 2026-09-15 and went with the mechanism (Track
# 5's pre-registered live-Optuna trial returned delta_helps = FALSE: mean_d000
# 0.7956173344395895 vs mean_d020 0.7861922400433382, cells_favouring_d020
# 14/24):
#
#   test_an_unreachable_boundary_is_identical_to_spending_nothing -- C1's
#     no-loss guarantee, that an unreachable boundary made a delta-0.20 run
#     prediction-identical to a delta-0 one. Vacuous once every run IS a
#     delta-0 run.
#   test_the_oracle_is_built_even_at_an_unbounded_delta -- delta_rel=None was
#     the only path that could have skipped the metric machinery, and it is no
#     longer reachable from any caller. The surviving statement of the same
#     property is test_train_model_contract.py's
#     test_the_joint_arm_always_builds_the_metric_oracle.
#   test_a_rollback_never_fires_when_no_budget_was_spent, and with it
#     test_spending_that_crosses_no_band_is_rolled_back_to_the_free_moves --
#     see the align_with_policy section below.
# ---------------------------------------------------------------------------

def test_the_per_move_sheds_sum_to_the_whole_runs_shed():
    """Alignment decrements its live per-feature width dict once per accepted
    move instead of recomputing the joint count thousands of times. The two
    must agree exactly, or every cost decision after the first accepted move is
    made on a stale number.

    The recorded VALUE has survived two rewrites unchanged -- BandBudget's
    scalar length, then BlockBudget.note_shed's (feature, bits) pair, now a
    plain dict decrement -- because width = intervals - 1 makes
    dwidth == dintervals, which is exactly what pooled_before - pooled_after
    already measures.

    There is no longer a method to patch, so the live dict is captured by
    identity instead: align_rf_thresholds hands it to feature_order as
    `widths=`, unchanged object, on every pass. Holding that reference lets the
    test read the dict's FINAL state and compare it against the from-scratch
    recomputation align_rf_thresholds reports -- which is a stronger statement
    than the old sum-of-sheds one, since it pins the whole dict's total rather
    than the decrements in isolation.

    codeword_before/after are exactly sum(widths): a feature's width is its
    pooled interval count minus one, and the interval count is summed over the
    same feature set the codeword length subtracts n_features from.
    """
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    captured = {}
    real_feature_order = ta.feature_order

    def spy(*args, **kwargs):
        captured['live_widths'] = kwargs['widths']
        return real_feature_order(*args, **kwargs)

    with mock.patch.object(ta, 'feature_order', spy):
        ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                               delta_rel=0.0, align_stats=stats)

    live_widths = captured['live_widths']
    assert sum(live_widths.values()) == stats['codeword_after']
    assert stats['codeword_after'] < stats['codeword_before'], (
        'the fixture must shed something, or this test is vacuous')


# ---------------------------------------------------------------------------
# align_with_policy.
#
# It was C1's commit-or-rollback wrapper until 2026-09-15. Two tests pinned
# that and are gone with it:
#
#   test_spending_that_crosses_no_band_is_rolled_back_to_the_free_moves
#   test_a_rollback_never_fires_when_no_budget_was_spent
#
# Neither can be written any more: the rollback fired only when budget was
# genuinely spent, which required a non-zero delta, and there is no longer a
# caller that can supply one. What the wrapper still guarantees -- that it
# clears the caller's stats dict before the run -- is pinned below.
# ---------------------------------------------------------------------------

def test_align_with_policy_clears_a_reused_stats_dict():
    """A caller passing the same dict twice must not read a previous run's keys
    back out of it. This is the whole of what the wrapper does now that the
    rollback is gone, so it is also the whole of what there is to pin."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {'left_over_from_an_earlier_run': 'stale'}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2, align_stats=stats)

    assert 'left_over_from_an_earlier_run' not in stats
    assert stats['intervals_after'] <= stats['intervals_before']


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
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
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

    _before, ranked = ta._rank_targets(r1, r2, ranges1, ranges2, idx1, idx2,
                                       feature_idx, sorted_cols1, sorted_cols2)
    targets = [target for target, _after in ranked]

    intersection = (max(r1[0], r2[0]), min(r1[1], r2[1]))  # (41, 88)
    union = (min(r1[0], r2[0]), max(r1[1], r2[1]))          # (33, 96)
    assert intersection in targets and union in targets, (
        'both corners must be admissible for this hand-built pair', targets)
    assert targets.index(intersection) < targets.index(union), (
        'the lower-damage corner (intersection, max damage 0.45) must be '
        'tried before the higher-damage one (union, max damage 0.50)',
        targets)


# ---------------------------------------------------------------------------
# Feature ordering after the align_objective axis was retired (design
# 2026-09-07): feature_order has one behaviour, not a choice between a
# 'blocks' and a 'stages' branch. As of Task 12 that one behaviour ranks by
# the blocks a shed would actually buy (audit Gaps 1+2), recomputed per
# feature against the live widths -- superseding the byte-completion order
# this comment used to describe. crossed_a_boundary was the other subject of
# this section and is gone (2026-09-15).
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


def test_legacy_combined_order_reproduces_the_pre_2026_09_07_order():
    """§9: `_legacy_combined_order` is the objective='blocks' branch this
    module used to run, kept private and used only by the characterisation
    tests so the switch to byte-completion ordering (below) has a measurable
    delta rather than a silent one. Repointed from the old
    test_feature_order_under_blocks_is_the_pre_existing_order, which asserted
    the same order through the now-removed `objective` parameter."""
    iv1, iv2 = _ordering_fixture()
    common = set(iv1) & set(iv2)
    expected = sorted(common,
                      key=lambda f: len(iv1.get(f, [])) + len(iv2.get(f, [])),
                      reverse=True)
    assert ta._legacy_combined_order(iv1, iv2) == expected


def test_align_rf_thresholds_no_longer_accepts_an_objective():
    """The axis is gone, not defaulted: a caller still passing it must fail
    loudly rather than have the argument silently ignored."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    with pytest.raises(TypeError):
        ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                               delta_rel=0.0,
                               align_objective='stages')


def test_crossed_a_boundary_is_gone():
    """Deleted 2026-09-15 with the rest of the delta_align mechanism (Track 5:
    delta_helps = FALSE -- mean_d000 0.7956173344395895 vs mean_d020
    0.7861922400433382, cells_favouring_d020 14/24).

    It answered "did this run buy a cheaper block FACTOR?" and existed only so
    align_with_policy could decide whether to roll a speculative run back. Both
    were unreachable without a non-zero delta. Asserted ABSENT rather than
    simply untested, so a future reviewer reinstating it as a gate has to argue
    for it rather than restore it quietly -- review finding 2.1 was that this
    test priced the factor where feature_order and the per-feature early exit
    price total blocks, and the resolution was deletion, not repricing.
    """
    assert not hasattr(ta, 'crossed_a_boundary')


def test_the_byte_domain_stats_are_recorded():
    """B (key bytes) IS the block cost after this repair -- there is no
    separate stage cost any more -- and the validation needs these recorded
    to have anything to compare against."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                         delta_rel=0.05, align_stats=stats)
    for key in ('key_bytes_before', 'key_bytes_after'):
        assert isinstance(stats[key], int), key
    assert stats['key_bytes_after'] <= stats['key_bytes_before']


def test_pooled_key_bytes_equals_the_evaluators_codeword_fields_to_bytes():
    """E1, the premise. If this fails the budget prices a table the switch
    does not build, and nothing else in the byte domain may be read."""
    from src.p4gen.build_p4_script import get_joint_feature_intervals
    from src.p4gen.evaluation import codeword_fields_to_bytes

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    names = ['f0', 'f1', 'f2', 'f3']
    for models in ((rf1, rf2),
                   ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                          delta_rel=0.05)):
        m1, m2 = models
        joint = get_joint_feature_intervals(m1, names, m2, names)
        assert ab.pooled_key_bytes(ta.extract_feature_intervals(m1),
                                   ta.extract_feature_intervals(m2)) == \
            codeword_fields_to_bytes(joint)


def test_the_block_factor_equals_the_evaluators_block_factor():
    """E1-blocks (design §5.2) -- the anchor that stops the cost model drifting
    again. Alignment's factor must equal what the generator's own merged
    intervals price, before AND after a run. If the generator's key layout or
    the block rule changes, this fails loudly rather than alignment silently
    going stale, which is exactly how the superseded band model survived for
    years."""
    from src.p4gen.build_p4_script import get_joint_feature_intervals
    from src.p4gen.evaluation import codeword_to_blocks, ternary_key_field_bits

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    names = ['f0', 'f1', 'f2', 'f3']
    for models in ((rf1, rf2),
                   ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                          delta_rel=0.05)):
        m1, m2 = models
        joint = get_joint_feature_intervals(m1, names, m2, names)
        widths = ab._pooled_widths(ta.extract_feature_intervals(m1),
                                   ta.extract_feature_intervals(m2))
        assert ab._factor(widths) == codeword_to_blocks(
            ternary_key_field_bits(joint))


@pytest.mark.parametrize('seed', range(50))
def test_the_block_factor_matches_p4model_on_random_width_vectors(seed):
    """The property half of E1-blocks. The fixture above pins four features on
    one pair; version_block_penalty's saturation clause only fires on a key
    whose bytes consume every midbyte its groups reach, which one pair will
    almost never produce. 1-15 fields of 1-60 bits is the design's own probe
    range (§1.3). Guards against anyone reimplementing _factor's arithmetic
    locally instead of delegating."""
    from src.p4model.tables import codeword_to_blocks
    rng = np.random.default_rng(seed)
    widths = {i: int(w) for i, w in
              enumerate(rng.integers(1, 61, size=int(rng.integers(1, 16))))}
    assert ab._factor(widths) == codeword_to_blocks(
        tuple(sorted(widths.values())))


@pytest.mark.parametrize('seed', range(200))
def test_shedding_a_bit_never_raises_the_block_factor(seed):
    """Invariant 7 (design §5.7). Alignment's only move sheds bits from one
    feature, so a non-monotone objective would create a perverse incentive: a
    run could be punished for a free saving, and the rollback would then be
    reasoning about a cost that moved the wrong way. Verified by the design
    over 200 000 random shapes; pinned here over a smaller sample so a future
    change to version_block_penalty that breaks monotonicity fails in this
    suite rather than in a campaign."""
    rng = np.random.default_rng(1000 + seed)
    widths = {i: int(w) for i, w in
              enumerate(rng.integers(1, 61, size=int(rng.integers(1, 16))))}
    victim = int(rng.integers(0, len(widths)))
    if widths[victim] <= 1:
        return                          # nothing to shed; not a counterexample
    before = ab._factor(widths)
    widths[victim] -= 1
    assert ab._factor(widths) <= before


@pytest.mark.parametrize('delta_rel', [0.0, 0.05, None])
def test_align_stats_records_the_block_factor_at_entry_exit_and_floor(delta_rel):
    """§4.6's three added keys (renamed factor_* by Task 11). factor_floor is
    what NO alignment of this pair could beat, factor_before what it starts
    at, factor_after what it reached -- so a run is sandwiched between them. A
    violation means either the floor is not a floor (invariant 4) or shedding
    raised the factor (invariant 7)."""
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                           delta_rel=delta_rel, align_stats=stats)
    for key in ('factor_before', 'factor_after', 'factor_floor'):
        assert isinstance(stats[key], int), key
    assert stats['factor_floor'] <= stats['factor_after'] <= stats['factor_before']


# Two more crossed_a_boundary tests stood here until 2026-09-15 and went with
# it: test_a_band_crossing_that_buys_no_block_no_longer_counts_as_crossing
# (§1.4's false positive -- a 44-bit band crossing that moved no block) and
# test_a_block_saving_without_a_band_crossing_now_counts_as_crossing (§1.4's
# false negative). Both pinned the 2026-09-07 band-to-factor repair of a gate
# that no longer exists.


def _order_fixture_pair():
    """Three features whose byte-completion order is the exact REVERSE of the
    legacy combined-count order, so the two are unambiguously distinguishable.

    Verified against the real helpers, not asserted from intuition -- an
    interval list's width is the POOLED bound count minus one, while its floor
    is max(len(list1), len(list2)) minus one, and the two move independently:

      f0  width 10  floor  5  step 2  room 5  reachable   combined 12
      f1  width 16  floor  8  step 8  room 8  reachable   combined 18
      f2  width 10  floor 10  step 2  room 0  UNreachable combined 22
    """
    def lst(bounds):
        # A gap-free tiling terminated by INFINITE, which is not a threshold.
        return [(0, b) for b in bounds] + [(0, INFINITE)]

    intervals1 = {0: lst([10, 20, 30, 40, 50]),
                  1: lst([1, 2, 3, 4, 5, 6, 7, 8]),
                  2: lst(range(10, 110, 10))}
    intervals2 = {0: lst([11, 21, 31, 41, 51]),
                  1: lst([11, 12, 13, 14, 15, 16, 17, 18]),
                  2: lst(range(10, 110, 10))}
    return intervals1, intervals2


def test_feature_order_puts_the_feature_that_buys_the_most_blocks_first():
    """Gap 2's fix (audit §3), and Gap 1's (§2) -- they are ONE edit: a ranking
    by achieved cost is meaningless on stale widths, and a ranking by byte
    distance is blind to both version_block_penalty and range steps.

    Feature 0 is one bit from completing a crossbar byte (width 9 -> 8 drops
    the key from 2 bytes to 1, so the ternary factor steps and the saving is
    worth the MULTIPLIER). Feature 1 is one interval from a free range block
    (207 intervals -> 206, the measured per-block capacity), worth exactly 1
    block however many trees there are.

    Both cost one bit, so the multiplier alone decides -- which is precisely
    what no byte-domain rule can see. Verified numerically before this plan was
    written: at multiplier 1 the keys are (-1, 1, -20, 0) and (-1, 1, -414, 1),
    so the larger combined count breaks the tie toward feature 1; at multiplier
    8 they are (-8, 1, -20, 0) and (-1, 1, -414, 1).
    """
    iv1 = {0: [(0, 1)] * 10, 1: [(0, 1)] * 207}
    iv2 = dict(iv1)
    widths = {0: 9, 1: 206}
    floors = {0: 0, 1: 0}

    assert ta.feature_order(iv1, iv2, multiplier=1,
                            widths=widths, floors=floors)[0] == 1
    assert ta.feature_order(iv1, iv2, multiplier=8,
                            widths=widths, floors=floors)[0] == 0


def test_feature_order_is_a_total_order():
    """Invariant 5: train_model.py's refit assertion depends on alignment being
    a deterministic function of (models, data, params), so the trailing feature
    index must break every tie."""
    same = {0: [(0, 1)] * 5, 1: [(0, 1)] * 5, 2: [(0, 1)] * 5}
    order = ta.feature_order(same, dict(same), multiplier=4)
    assert order == [0, 1, 2]
    assert order == ta.feature_order(same, dict(same), multiplier=4)


def test_feature_order_ranks_only_the_features_it_is_given():
    """The loop takes [0] from a FRESH call per feature, against the CURRENT
    widths -- audit §8.2 item 9. `features` is what is left to visit."""
    iv = {0: [(0, 1)] * 5, 1: [(0, 1)] * 9, 2: [(0, 1)] * 5}
    assert set(ta.feature_order(iv, dict(iv), multiplier=4,
                                features={1, 2})) == {1, 2}


def test_feature_order_keeps_the_combined_count_key_when_nothing_buys_a_block():
    """A feature that can buy nothing is NOT dropped -- it only loses priority,
    and among such features the pre-existing combined-count key still decides,
    so the fallback order is the one the archive was produced under."""
    iv1 = {0: [(0, 1)] * 3, 1: [(0, 1)] * 5}
    iv2 = {0: [(0, 1)] * 3, 1: [(0, 1)] * 5}
    floors = {0: 2, 1: 4}          # no room anywhere
    widths = {0: 2, 1: 4}
    assert ta.feature_order(iv1, iv2, multiplier=4,
                            widths=widths, floors=floors) == [1, 0]


def test_feature_order_no_longer_takes_an_objective():
    """The axis is gone, not defaulted -- a caller still passing it must fail
    loudly rather than have the argument silently ignored."""
    intervals1, intervals2 = _order_fixture_pair()
    with pytest.raises(TypeError):
        ta.feature_order(intervals1, intervals2, 'blocks')


def test_still_overlaps_rejects_a_pair_an_earlier_move_pulled_apart():
    """Audit §8.3's undocumented third job, and the crash.

    find_partially_overlapping_ranges is computed once per ROUND and its pairs
    are RE-READ inside the round, so an accepted move earlier in the same round
    can have rewritten either tuple. The stale pair below produces the
    empty-intersection target (701, 700), which inverts the tiling when
    committed. Until now nothing revalidated this except, incidentally, the
    similarity ratio -- so the one setting that disabled the heuristic
    (overlap_threshold=0.0) disabled the correctness check with it.
    """
    assert ta.still_overlaps((581, 1005), (701, 1005))
    assert not ta.still_overlaps((581, 700), (701, 1005))
    assert not ta.still_overlaps((0, 100), (100, 200))     # touching, not overlapping


def test_structurally_alignable_vetoes_a_lone_sentinel():
    """The zero-side and INFINITE-side vetoes, which adjust_range_boundaries
    physically cannot satisfy: a boundary sitting on a sentinel is never moved,
    so `ranges` would claim a move the model refused to make (the C5 bug).
    dataset.py clips every feature at INFINITE, so (m, INFINITE) is common.
    """
    assert ta.structurally_alignable((0, 100), (0, 200))
    assert ta.structurally_alignable((10, INFINITE), (20, INFINITE))
    assert not ta.structurally_alignable((0, 100), (10, 200))
    assert not ta.structurally_alignable((10, INFINITE), (20, 900))


def test_target_is_well_formed_rejects_the_inverted_target():
    """The live crash, as a unit. neighbour_writes validates that NEIGHBOURING
    intervals do not invert; nothing validated the target itself.
    """
    assert ta.target_is_well_formed((701, 1005))
    assert ta.target_is_well_formed((700, 700))
    assert not ta.target_is_well_formed(
        _target_range((581, 700), (701, 1005)))     # (701, 700)


def test_the_stale_pair_leaves_the_tiling_and_the_index_intact():
    """The regression the three checks exist for: rejected, not committed.

    Asserts on the STRUCTURES, not just on the absence of a raise -- a
    committed inverted target is what desynchronises `ranges` from
    threshold_index, and a test that only catches the eventual
    AlignmentInvariantError would pass on a version that corrupted them both
    consistently.
    """
    ranges = [(0, 580), (581, 700), (701, 1005), (1006, INFINITE)]
    before = list(ranges)
    index = {(0, 580): [(0, 1)], (0, 700): [(0, 2)], (0, 1005): [(0, 3)]}
    index_before = {k: list(v) for k, v in index.items()}

    assert not ta.still_overlaps(ranges[1], ranges[2])
    assert ranges == before
    assert index == index_before


def test_a_real_fitted_pair_aligns_without_an_invariant_error():
    """Audit §8.3: zero AlignmentInvariantErrors across 324 replayed runs with
    the ratio test disabled, versus a hard crash at a literal
    overlap_threshold=0.0 today. Pinned here on the golden pair so the fix
    cannot silently regress. Unparametrised as of Task 7: the ratio test is
    gone entirely, so there is no threshold left to vary.
    """
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2,
                         delta_rel=0.0,
                         align_stats=stats)
    assert stats['intervals_after'] <= stats['intervals_before']


@pytest.mark.parametrize('delta_rel', [0.0, 0.05, None])
def test_align_stats_records_the_factor_and_the_total_separately(delta_rel):
    """Audit §8.2 item 7. `blocks_*` held the ternary FACTOR, not a block
    count, and archived campaign CSVs carry it under that name. Renaming those
    three to factor_* and adding total_blocks_* as NEW names means an old CSV
    can never be read as though it held totals -- a silent repurpose would make
    pre- and post-repair campaigns look comparable when they are not.
    """
    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
    stats = {}
    ta.align_with_policy(rf1, rf2, X1, y1, X2, y2, delta_rel=delta_rel,
                         align_stats=stats)

    assert 'blocks_before' not in stats
    for key in ('factor_before', 'factor_after', 'factor_floor',
                'total_blocks_before', 'total_blocks_after',
                'total_blocks_floor'):
        assert isinstance(stats[key], int), key

    # Sandwiched between entry and floor in BOTH domains: a violation means
    # either the floor is not a floor (invariant 4) or shedding raised the cost
    # (invariant 7).
    assert stats['factor_floor'] <= stats['factor_after'] <= stats['factor_before']
    assert (stats['total_blocks_floor'] <= stats['total_blocks_after']
            <= stats['total_blocks_before'])

    # The total is never below the factor's own contribution.
    assert stats['total_blocks_after'] >= stats['factor_after']


def _block_purchase_then_more_pair():
    """One feature where round 1 BUYS a block and round 2 still sheds.

        rf1 cuts {3, 4, 27, 31} + F,   rf2 cuts {32, 56, 57} + F,
        F = 100..133 (34 cuts both models share, so they never form a pair
        but do count toward the pooled width).

    Pooled width 41 bits = 6 crossbar bytes (factor 2). Round 1 accepts two
    moves (targets (0, 3) and (58, 100)), bringing the width to 39 = 5 bytes,
    factor 1: a real purchase, priced by the real total_blocks. Only round 2
    then reaches the pair that the round-1 moves created, target (5, 27),
    shedding one more bit (width 38). Found by a random search over small
    hand-built cut sets, then padded with F to sit on the 40-bit ladder step.
    """
    filler = list(range(100, 134))
    rf1 = _hand_built_forest([3, 4, 27, 31] + filler)
    rf2 = _hand_built_forest([32, 56, 57] + filler)
    X = np.array([[0.0], [50.0], [500.0], [1500.0],
                  [2500.0], [4000.0], [7000.0], [65535.0]])
    y1 = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    return rf1, rf2, X, y1, y2


def test_a_feature_keeps_shedding_after_it_buys_a_block():
    """Spec 2026-09-28 T4: the per-feature early exit is gone, so a feature
    runs to its fixpoint even after a round in which it bought a block.

    Under the deleted exit this fixture stopped after round 1 at width 39:
    the purchase (41 -> 39 bits, factor 2 -> 1) retired the feature before
    round 2 could reach the (5, 27) move. Now round 2 runs and sheds it.
    """
    rf1, rf2, X, y1, y2 = _block_purchase_then_more_pair()
    stats, log = {}, []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, delta_rel=None,
                           align_stats=stats, candidate_log=log)

    accepted = [(e['round'], e['target']) for e in log if e['accepted']]
    assert accepted == [(1, (0, 3)), (1, (58, 100)), (2, (5, 27))]
    assert stats['codeword_before'] == 41 and stats['factor_before'] == 2
    # The block was bought in round 1 (39 bits is already factor 1) ...
    assert stats['factor_after'] == 1
    # ... and the feature still shed past it.
    assert stats['codeword_after'] == 38


def test_the_truncation_guard_still_raises_on_a_feature_that_bought_a_block(
        monkeypatch):
    """The fixpoint check is now simply `progressed and rounds > 1`. With the
    cap patched to 2 the fixture above is cut off in round 2 while still
    accepting (round 2 accepts (5, 27)), so it must raise -- the purchase in
    round 1 no longer exempts it, because nothing leaves a feature early any
    more."""
    monkeypatch.setattr(ta, 'MAX_RECOMPUTE_ROUNDS', 2)
    rf1, rf2, X, y1, y2 = _block_purchase_then_more_pair()
    with pytest.raises(AlignmentInvariantError) as excinfo:
        ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, delta_rel=None)
    assert 'fixpoint' in str(excinfo.value).lower()
