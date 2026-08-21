"""endpoint_ratio and shift_mass as pure diagnostics (no filtering happens on
them -- the endpoint-ratio-cap heuristic was replaced in Task 7 and the
shift_mass-based veto was removed outright in P3 Task 8), plus
calculate_range_overlap's remaining structural-only vetoes and
align_rf_thresholds' candidate_log."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE
from src.training.config import TrainConfig
from src.training import threshold_alignment as ta


def test_the_cap_waves_through_a_far_larger_absolute_shift():
    """(1000, 4000) vs (3900, 4000): endpoint ratio 3.9, so the cap passes it,
    despite a 2900-unit drag on a 3%-overlapping pair. Backwards near zero."""
    assert ta.endpoint_ratio((1000, 4000), (3900, 4000)) < 5.0
    assert ta.endpoint_ratio((1, 100), (10, 100)) > 5.0


def test_the_cap_is_maximally_permissive_at_the_clip_atom():
    """dataset.py clips every feature at INFINITE = 65535, so a large fraction
    of rows sit at exactly that value. Moving 65534 -> 65535 has an endpoint
    ratio of ~1.00002 -- the most permissive reading available -- while
    relocating the entire clip atom. No tuned constant fixes this."""
    assert ta.endpoint_ratio((40000, 65534), (40000, INFINITE)) < 1.001


def test_shift_mass_counts_the_validation_rows_that_change_side():
    """sklearn sends x <= threshold left, so the affected set is (lo, hi]."""
    col = np.sort(np.array([0.0, 5.0, 10.0, 10.0, 20.0, 30.0, 65535.0]))

    assert ta.shift_mass(col, 10, 10) == 0.0
    assert ta.shift_mass(col, 10, 20) == pytest.approx(1 / 7)
    assert ta.shift_mass(col, 20, 10) == pytest.approx(1 / 7)
    assert ta.shift_mass(col, 4, 10) == pytest.approx(3 / 7)


def test_shift_mass_is_enormous_at_the_clip_atom():
    """The property endpoint ratio gets exactly backwards."""
    col = np.sort(np.array([1.0, 2.0] + [65535.0] * 98))

    assert ta.shift_mass(col, 65534, 65535) == pytest.approx(0.98)
    assert ta.endpoint_ratio((0, 65534), (0, 65535)) < 1.001


def test_the_candidate_log_records_one_row_per_candidate_with_both_predictors():
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
                           overlap_threshold=0.5, delta_rel=0.05,
                           candidate_log=log)

    assert log, 'the fixture must produce candidates'
    # Every entry carries both predictors regardless of how it was decided.
    for entry in log:
        assert set(entry) == {'feature_idx', 'range1', 'range2', 'overlap_ratio',
                             'endpoint_ratio', 'shift_mass_1', 'shift_mass_2',
                             'rel_deg', 'accepted', 'error_app', 'error_ddos',
                             'round'}
        assert 0.0 <= entry['shift_mass_1'] <= 1.0
        # C3's recompute round this candidate was found in. It lives here and
        # not in align_stats deliberately: the stats dict's key set is pinned
        # exactly, while candidate_log is the diagnostic structure meant to
        # grow. Round 1 is the pre-C3 candidate set; anything above 1 is a
        # candidate an accepted move created.
        assert entry['round'] >= 1
        assert len(entry['rel_deg']) == 4
        assert isinstance(entry['accepted'], bool)


def test_calculate_range_overlap_is_a_pure_measurement_again():
    """The heuristic veto is gone; only the two STRUCTURAL vetoes remain --
    adjust_range_boundaries genuinely cannot move a boundary at 0 or at
    INFINITE, so those pairs can never be aligned."""
    import inspect

    assert 'endpoint_ratio_cap' not in inspect.signature(ta.calculate_range_overlap).parameters
    assert ta.calculate_range_overlap((1, 100), (10, 100)) > 0.9
    assert ta.calculate_range_overlap((0, 100), (10, 100)) == 0.0
    assert ta.calculate_range_overlap((10, INFINITE), (10, 40000)) == 0.0


def test_the_candidate_log_is_off_by_default():
    import inspect

    signature = inspect.signature(ta.align_rf_thresholds)
    assert signature.parameters['candidate_log'].default is None
