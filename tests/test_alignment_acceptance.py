"""Spec B.4: the acceptance test and the ratchet become per-task."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
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


def test_the_ratchet_is_per_task_and_elementwise():
    """Today only the mean was ratcheted, so a sequence where App improves while
    DDoS degrades kept the mean flat, no single move tripped the guard, and
    DDoS drifted arbitrarily far. Four independent marks bound each task's
    total drift from ITS OWN best."""
    marks = ta.ratchet(BEFORE, (0.790, 0.788, 0.955, 0.954))

    assert marks == (0.790, 0.788, 0.960, 0.959)


def test_a_ratcheted_sequence_cannot_let_one_task_drift_past_delta():
    """The regression test for the drift: App climbs 0.01 per step while DDoS
    slides 0.003 per step. Under a mean-only ratchet every step looks free.
    Under per-task marks, DDoS's cumulative loss is measured from its own best
    and the sequence is cut off."""
    delta = 0.05
    marks = BEFORE
    ddos_acc = BEFORE[2]
    app_acc = BEFORE[0]
    accepted_steps = 0

    for _ in range(20):
        app_acc += 0.010
        ddos_acc -= 0.003
        candidate = (app_acc, app_acc - 0.002, ddos_acc, ddos_acc - 0.001)
        if not ta.accept_alignment(marks, candidate, delta):
            break
        marks = ta.ratchet(marks, candidate)
        accepted_steps += 1

    # delta = 5% of DDoS's 0.040 error = 0.002 absolute, and each step costs
    # 0.003 -- so not even the first step may pass.
    assert accepted_steps == 0
    total_ddos_loss = BEFORE[2] - ddos_acc
    assert total_ddos_loss > delta * (1 - BEFORE[2])


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
                           overlap_threshold=0.5, delta_rel=0.0,
                           align_stats=stats_strict)
    ta.align_rf_thresholds(mk(X1, y1, 0), mk(X2, y2, 1), X1, y1, X2, y2,
                           overlap_threshold=0.5, delta_rel=None,
                           align_stats=stats_loose)

    for stats in (stats_strict, stats_loose):
        assert set(stats) == {'attempted', 'accepted', 'intervals_before', 'intervals_after'}
        assert stats['accepted'] <= stats['attempted']
        assert stats['intervals_after'] <= stats['intervals_before']

    # NOT delta-invariant in general, and that's expected rather than a bug:
    # the candidate INDEX-PAIR list (find_partially_overlapping_ranges) is
    # computed once per feature from the untouched initial intervals, so it
    # doesn't depend on delta_rel -- but whether a given candidate still
    # produces a non-empty `modifications` (and so counts as "attempted")
    # depends on current_ranges1/current_ranges2, which EARLIER accepted
    # moves in the same feature mutate in place via
    # update_neighboring_ranges_and_index. So attempted-count depends on the
    # accept/reject trajectory whenever a feature has more than one overlap --
    # pre-existing behaviour from Tasks 1-4, not something this task's
    # accept_alignment/ratchet swap introduced.
    assert stats_loose['accepted'] >= stats_strict['accepted']
