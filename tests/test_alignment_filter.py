"""shift_mass as a pure diagnostic (no filtering happens on it -- the
shift_mass-based veto was removed outright in P3 Task 8), plus
align_rf_thresholds' candidate_log.

endpoint_ratio (the endpoint-ratio-cap heuristic's replaced quantity) and
calculate_range_overlap (its remaining structural-only vetoes, since
duplicated by structurally_alignable -- see
test_structurally_alignable_vetoes_a_lone_sentinel in
test_threshold_alignment.py) were pruned outright 2026-09-15 (task 14) once
neither had any admission role or reader left."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE
from src.training import threshold_alignment as ta


def test_shift_mass_counts_the_validation_rows_that_change_side():
    """sklearn sends x <= threshold left, so the affected set is (lo, hi]."""
    col = np.sort(np.array([0.0, 5.0, 10.0, 10.0, 20.0, 30.0, 65535.0]))

    assert ta.shift_mass(col, 10, 10) == 0.0
    assert ta.shift_mass(col, 10, 20) == pytest.approx(1 / 7)
    assert ta.shift_mass(col, 20, 10) == pytest.approx(1 / 7)
    assert ta.shift_mass(col, 4, 10) == pytest.approx(3 / 7)


def test_shift_mass_is_enormous_at_the_clip_atom():
    """dataset.py clips every feature at INFINITE = 65535, so a large fraction
    of rows sit at exactly that value: moving the clip boundary relocates
    almost the whole atom, which is a large physical effect the (since-pruned)
    endpoint ratio measured as barely more than 1.0 -- backwards, which is why
    shift_mass replaced it as the damage predictor."""
    col = np.sort(np.array([1.0, 2.0] + [65535.0] * 98))

    assert ta.shift_mass(col, 65534, 65535) == pytest.approx(0.98)


def test_the_candidate_log_records_one_row_per_candidate_with_shift_mass():
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen.build_p4_script import dt_thresholds_float_to_int

    rng = np.random.default_rng(7)
    X1 = np.clip(rng.integers(0, 90000, size=(300, 4)), 0, INFINITE).astype(float)
    y1 = np.array([c % 3 for c in range(300)])
    X2 = np.clip(rng.integers(0, 90000, size=(300, 4)), 0, INFINITE).astype(float)
    y2 = np.array([-1, 1] * 150)
    mk = lambda X, y, s: dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=5, max_depth=5, min_samples_leaf=10, random_state=s).fit(X, y))

    log = []
    ta.align_rf_thresholds(mk(X1, y1, 0), mk(X2, y2, 1), X1, y1, X2, y2,
                           delta_rel=0.05,
                           candidate_log=log)

    assert log, 'the fixture must produce candidates'
    # Every entry carries shift_mass regardless of how it was decided.
    # overlap_ratio and endpoint_ratio were dropped from this dict 2026-09-15
    # (task 14) along with the functions that computed them.
    for entry in log:
        assert set(entry) == {'feature_idx', 'range1', 'range2', 'target',
                             'shift_mass_1',
                             'shift_mass_2', 'rel_deg', 'accepted', 'error_app',
                             'error_ddos', 'round'}
        assert 0.0 <= entry['shift_mass_1'] <= 1.0
        # C3's recompute round this candidate was found in. It lives here and
        # not in align_stats deliberately: the stats dict's key set is pinned
        # exactly, while candidate_log is the diagnostic structure meant to
        # grow. Round 1 is the pre-C3 candidate set; anything above 1 is a
        # candidate an accepted move created.
        assert entry['round'] >= 1
        assert len(entry['rel_deg']) == 4
        assert isinstance(entry['accepted'], bool)


def test_the_candidate_log_is_off_by_default():
    import inspect

    signature = inspect.signature(ta.align_rf_thresholds)
    assert signature.parameters['candidate_log'].default is None
