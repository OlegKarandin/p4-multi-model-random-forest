"""Spec B.4: the acceptance test is per task; spec 2026-09-28 T3: it is
anchored at the pre-alignment scores for the whole run (no ratchet)."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import (
    INFINITE, dt_thresholds_float_to_int, get_feature_intervals_from_thresholds)
from src.training import threshold_alignment as ta

# (acc_app, f1_app, acc_ddos, f1_ddos). App error ~0.22, DDoS error ~0.04 --
# a 19-23 point accuracy gap that is genuine task difficulty, not class
# imbalance: app is 8717/8968/9537 across 3 classes and ddos is 10000/10000.
BEFORE = (0.780, 0.778, 0.960, 0.959)


def test_the_motivating_case_is_accepted_by_the_average_and_rejected_per_task():
    """Spec B.4's worked case: a move costing DDoS 0.009 while gaining App
    0.001 drops the MEAN by 0.0040 -- inside the old 0.005 tolerance, so
    accepted. Per task it is 0.009 / 0.040 = 22.5% of DDoS's error."""
    after = (0.781, 0.779, 0.951, 0.950)

    mean_before = (BEFORE[0] + BEFORE[2]) / 2
    mean_after = (after[0] + after[2]) / 2
    assert mean_before - mean_after == pytest.approx(0.0040, abs=1e-6)

    assert ta.accept_alignment(BEFORE, after, delta_rel=0.05) is False
    assert ta.accept_alignment(BEFORE, after, delta_rel=0.20) is False
    assert ta.accept_alignment(BEFORE, after, delta_rel=0.25) is True


def test_all_four_metrics_are_guarded_independently():
    """Accuracy and weighted F1, for both tasks. F1 can move further than
    accuracy per flip when flips concentrate in one class, so it needs its own
    guard rather than riding along on accuracy's."""
    for position, name in enumerate(('acc_app', 'f1_app', 'acc_ddos', 'f1_ddos')):
        after = list(BEFORE)
        after[position] -= 0.02

        assert ta.accept_alignment(BEFORE, tuple(after), delta_rel=0.01) is False, name
        assert ta.accept_alignment(BEFORE, tuple(after), delta_rel=0.9) is True, name


def test_the_same_absolute_drop_is_judged_differently_per_task():
    """0.005 is 2.3% of App's error and 12.5% of DDoS's. Averaging accuracy
    treats them as equal; the relative-error scale does not."""
    app_hit = (0.775, 0.778, 0.960, 0.959)
    ddos_hit = (0.780, 0.778, 0.955, 0.959)

    assert ta.accept_alignment(BEFORE, app_hit, delta_rel=0.05) is True
    assert ta.accept_alignment(BEFORE, ddos_hit, delta_rel=0.05) is False


def test_delta_zero_accepts_a_move_that_changes_nothing():
    """delta_align = 0 is not 'reject everything' -- it is 'reject anything that
    costs a task anything'. A move that flips no prediction must pass."""
    assert ta.accept_alignment(BEFORE, BEFORE, delta_rel=0.0) is True


def test_delta_zero_rejects_any_loss_however_small():
    after = (0.780, 0.778, 0.9599, 0.959)

    assert ta.accept_alignment(BEFORE, after, delta_rel=0.0) is False


def test_an_improvement_on_one_task_never_licenses_a_loss_on_the_other():
    """The substitution the reviewer objected to, at the level of a single
    move: no amount of App gain may pay for a DDoS loss."""
    after = (0.900, 0.900, 0.940, 0.940)

    assert ta.accept_alignment(BEFORE, after, delta_rel=0.05) is False


def test_delta_none_accepts_everything_including_a_catastrophic_move():
    after = (0.10, 0.10, 0.10, 0.10)

    assert ta.accept_alignment(BEFORE, after, delta_rel=None) is True


def test_the_ratchet_is_gone():
    """Spec 2026-09-28 T3: the guard is anchored at the PRE-ALIGNMENT scores
    for the whole run. The running high-water mark it replaced let a noise
    gain on val_align become the floor for every later move."""
    assert not hasattr(ta, 'ratchet')


class _ScriptedMetrics:
    """Stands in for IncrementalMetrics inside align_rf_thresholds.

    Each apply() moves this model's (accuracy, f1) to the next scripted value
    (both metrics move together); once the script runs out a move changes
    nothing. revert() restores the value apply() replaced, exactly the
    contract the real class honours. `start` is what metrics() reports before
    any move."""

    scripts = {}
    starts = {}

    def __init__(self, tree_predictions, rf, y_true, task):
        self.task = task
        self.script = list(self.scripts.get(task, []))
        self.value = self.starts[task]

    def metrics(self):
        return (self.value, self.value)

    def apply(self, tree_predictions, undo_info):
        token = self.value
        if self.script:
            self.value = self.script.pop(0)
        return token

    def revert(self, token):
        self.value = token


def _scripted_run(monkeypatch, app_script, ddos_script):
    from tests.test_threshold_alignment import _block_purchase_then_more_pair

    monkeypatch.setattr(_ScriptedMetrics, 'scripts',
                        {'app': list(app_script), 'ddos': list(ddos_script)})
    monkeypatch.setattr(_ScriptedMetrics, 'starts',
                        {'app': 0.9500, 'ddos': 0.9000})
    monkeypatch.setattr(ta, 'IncrementalMetrics', _ScriptedMetrics)
    rf1, rf2, X, y1, y2 = _block_purchase_then_more_pair()
    log = []
    ta.align_rf_thresholds(rf1, rf2, X, y1, X, y2, delta_rel=0.0,
                           candidate_log=log)
    return log


def test_a_gain_then_a_partial_giveback_is_accepted(monkeypatch):
    """0.9500 -> 0.9510 -> 0.9505: the second move gives back half of an
    accidental gain but stays above where the run STARTED, so it is free.
    Under the retired ratchet it was judged against 0.9510 and rejected."""
    log = _scripted_run(monkeypatch, [0.9510, 0.9505], [])
    assert [e['accepted'] for e in log[:2]] == [True, True]


def test_a_move_below_the_start_on_one_metric_is_rejected_however_much_the_others_rise(
        monkeypatch):
    """Still per metric (accept_alignment unchanged): App rises to 0.9600 on
    the same move that takes DDoS to 0.8999, one ten-thousandth below its own
    pre-alignment 0.9000 -- rejected. And a later move that drops App below
    its START after an earlier gain is rejected too, even though it is only
    a partial giveback of that gain relative to the gain's peak."""
    log = _scripted_run(monkeypatch, [0.9600], [0.8999])
    assert log[0]['accepted'] is False

    log = _scripted_run(monkeypatch, [0.9510, 0.9499], [])
    assert [e['accepted'] for e in log[:2]] == [True, False]


def test_the_final_val_align_metrics_never_fall_below_the_start():
    """The run-level statement T3's fixed anchor guarantees: on the data the
    guard judges (val_align), every one of the four metrics ends >= where it
    started. Measured on the real metric machinery, not a script."""
    from tests.test_threshold_alignment import _golden_alignment_pair
    from src.training.incremental_metrics import IncrementalMetrics

    rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()

    def four(m1, m2):
        p1, _ = ta.build_prediction_cache(m1, np.asarray(X1, dtype=np.float32))
        p2, _ = ta.build_prediction_cache(m2, np.asarray(X2, dtype=np.float32))
        return (IncrementalMetrics(p1, m1, y1, task='app').metrics()
                + IncrementalMetrics(p2, m2, y2, task='ddos').metrics())

    stats = {}
    a1, a2 = ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2, delta_rel=0.0,
                                    align_stats=stats)
    assert stats['accepted'] > 0, 'the fixture must actually move something'
    before, after = four(rf1, rf2), four(a1, a2)
    assert all(a >= b for b, a in zip(before, after)), (before, after)


def test_alignment_reports_its_acceptance_rate_and_interval_counts():
    """align_attempted / align_accepted are what turn the delta frontier from a
    black box into a mechanism: the acceptance rate should rise with delta as
    blocks fall. intervals_before/after is the resource-side counterpart."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(11)
    X1 = np.clip(rng.integers(0, 90000, size=(300, 4)), 0, INFINITE).astype(float)
    y1 = np.array([c % 3 for c in range(300)])
    X2 = np.clip(rng.integers(0, 90000, size=(300, 4)), 0, INFINITE).astype(float)
    y2 = np.array([-1, 1] * 150)
    mk = lambda X, y, s: dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=5, max_depth=5, min_samples_leaf=10, random_state=s).fit(X, y))

    stats_strict, stats_loose = {}, {}
    ta.align_rf_thresholds(mk(X1, y1, 0), mk(X2, y2, 1), X1, y1, X2, y2,
                           delta_rel=0.0,
                           align_stats=stats_strict)
    ta.align_rf_thresholds(mk(X1, y1, 0), mk(X2, y2, 1), X1, y1, X2, y2,
                           delta_rel=None,
                           align_stats=stats_loose)

    for stats in (stats_strict, stats_loose):
        assert set(stats) == {
            'attempted', 'accepted', 'intervals_before', 'intervals_after',
            'codeword_before', 'codeword_after',
            'key_bytes_before', 'key_bytes_after',
            'bits_to_reach', 'accuracy_spent',
            'factor_before', 'factor_after', 'factor_floor',
            'total_blocks_before', 'total_blocks_after', 'total_blocks_floor'}
        assert stats['accepted'] <= stats['attempted']
        # #27: under the OLD union-of-interval-tuples joint_interval_count this
        # read like a theorem but was not one (a single accepted move could
        # RAISE it -- see the since-rewritten
        # test_a_single_accepted_move_can_leave_the_joint_interval_count_flat
        # in test_threshold_alignment.py). Under the corrected pooled-threshold
        # definition it genuinely IS an invariant, not merely a fixture
        # coincidence: every write in align_rf_thresholds relocates a
        # threshold to a value already present in one of the two models'
        # CURRENT threshold sets for that feature (never a new one), so the
        # pooled threshold set -- and the interval count derived from it --
        # can only shrink or stay flat, per feature and therefore in total.
        # Kept as a real regression guard, not removed.
        assert stats['intervals_after'] <= stats['intervals_before']

    # NOT delta-invariant in general, and that's expected rather than a bug.
    # Since C3 (P3b T3) the candidate set is RE-DERIVED from the current,
    # already-mutated interval lists after every accepted move, and the loop
    # runs rounds until one accepts nothing. So the whole candidate set --
    # not just whether a candidate still produces a non-empty `modifications`
    # -- depends on the accept/reject trajectory, and therefore on delta_rel:
    # an accepted move widens the aligned range's neighbours, and a widened
    # neighbour can overlap a range in the other model that nothing overlapped
    # before. Those pairs used to be unreachable (the overlap list was
    # computed once per feature and never refreshed); now they are attempted,
    # which is why `attempted` and `accepted` are both larger than they were
    # before C3.
    assert stats_loose['accepted'] >= stats_strict['accepted']


def test_joint_interval_count_prefers_the_common_refinement_over_the_sum():
    """The bug this replaces: summing each model's OWN interval count can
    never move, since alignment relocates a threshold, it never deletes one.
    The TCAM-relevant quantity, for a feature both models split on, is the
    size of the COMMON REFINEMENT of both models' thresholds -- i.e. what you
    get by pooling both models' threshold values for that feature and
    re-partitioning, exactly as evaluation.py's joint-encoding cost model
    does over the merged tree set (NOT the union of the two models' interval
    TUPLES, which overcounts: {10}-only and {5}-only both cut the same axis
    into thirds once pooled -- (0,5),(6,10),(11,INF) -- but as tuples
    (0,10) != (0,5) and (11,INF) != (6,INF), so a naive union sees 4 distinct
    tuples where the true partition has 3)."""
    # Feature 0: both models split it, currently DIFFERENT ranges (no sharing
    # possible yet). Pooled thresholds {100, 150} -> 3 intervals
    # ((0,100),(101,150),(151,65535)), not the 4-tuple union.
    before1 = {0: [(0, 100), (101, 65535)]}
    before2 = {0: [(0, 150), (151, 65535)]}
    assert ta.joint_interval_count(before1, before2) == 3

    # After alignment succeeds, both models split feature 0 identically --
    # pooled thresholds collapse to {100} -> 2 intervals. This is the real
    # savings signal a flat per-model sum (2 + 2 = 4, unchanged) could never
    # show.
    after1 = {0: [(0, 100), (101, 65535)]}
    after2 = {0: [(0, 100), (101, 65535)]}
    assert ta.joint_interval_count(after1, after2) == 2

    # A feature only one model splits on can't be shared -- added directly,
    # not pooled away.
    only1 = {0: [(0, 100), (101, 65535)], 1: [(0, 65535)]}
    only2 = {0: [(0, 100), (101, 65535)]}
    assert ta.joint_interval_count(only1, only2) == 2 + 1


def test_joint_interval_count_review_counterexample_10_and_5():
    """The review's worked counterexample, kept verbatim as a unit case:
    thresholds {10} and {5} on the same feature pool into 3 intervals under
    the true common refinement -- (0,5),(6,10),(11,INF) -- not 4."""
    intervals1 = {0: [(0, 10), (11, INFINITE)]}
    intervals2 = {0: [(0, 5), (6, INFINITE)]}
    assert ta.joint_interval_count(intervals1, intervals2) == 3


def test_joint_interval_count_matches_the_pooled_threshold_cost_model():
    """The stat and the real joint-encoding cost model
    (evaluation.py's multi_model_memory_evaluation, which pools both models'
    thresholds through get_feature_intervals_from_thresholds over the merged
    tree set and returns a ResourceUsage) must agree BY CONSTRUCTION: build the expected value completely
    independently -- read every (feature, threshold) split straight off both
    forests' raw trees, pool them, and run them through the exact function
    the cost model calls -- rather than trusting joint_interval_count's own
    internals to have gotten it right."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(21)
    n = 300
    X1 = np.clip(rng.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y1 = np.array([c % 3 for c in range(n)])
    X2 = np.clip(rng.integers(0, 90000, size=(n, 4)), 0, INFINITE).astype(float)
    y2 = np.where(np.arange(n) % 2 == 0, -1, 1)

    rf1 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=5, max_depth=5, min_samples_leaf=10, random_state=2).fit(X1, y1))
    rf2 = dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=5, max_depth=5, min_samples_leaf=10, random_state=3).fit(X2, y2))

    # Independently reconstruct the pooled (feature, threshold) list straight
    # from both forests' raw tree arrays -- deliberately NOT going through
    # extract_feature_intervals or joint_interval_count for this half of the
    # computation, so the expected value is derived a genuinely different way
    # than the code under test.
    pooled = []
    for estimator in list(rf1.estimators_) + list(rf2.estimators_):
        tree = estimator.tree_
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:
                pooled.append((int(tree.feature[node_idx]),
                               int(round(tree.threshold[node_idx]))))
    pooled.sort()
    expected = sum(len(v) for v in
                   get_feature_intervals_from_thresholds(pooled).values())

    intervals1 = ta.extract_feature_intervals(rf1)
    intervals2 = ta.extract_feature_intervals(rf2)
    assert ta.joint_interval_count(intervals1, intervals2) == expected
