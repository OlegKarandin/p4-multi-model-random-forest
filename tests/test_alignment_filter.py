"""The endpoint_ratio_cap: made observable here, replaced in Task 7."""
import numpy as np
import pytest

from src.p4gen.build_p4_script import INFINITE
from src.training.config import TrainConfig
from src.training import threshold_alignment as ta


def test_the_cap_vetoes_a_pair_the_stated_criterion_would_accept():
    """(1, 100) vs (10, 100): endpoint ratio 10, so vetoed -- yet the true
    overlap is 0.909, far above overlap_threshold = 0.5, and the move is a
    9-unit nudge. The cap is a second acceptance criterion hiding inside a
    function whose name promises a measurement."""
    assert ta.calculate_range_overlap((1, 100), (10, 100), endpoint_ratio_cap=5.0) == 0.0
    assert ta.calculate_range_overlap((1, 100), (10, 100), endpoint_ratio_cap=None) > 0.9


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


def test_disabling_the_cap_lets_every_candidate_reach_the_oracle():
    """Required for the instrumented run, and required by the sweep: any filter
    used during the sweep must widen at least as fast as the oracle so that it
    never binds. A fixed cap would make the frontier saturate because of `5`
    rather than because of anything about alignment."""
    assert ta.calculate_range_overlap((1, 100), (10, 100), endpoint_ratio_cap=None) > 0.0
    assert TrainConfig(endpoint_ratio_cap=None).endpoint_ratio_cap is None
    assert TrainConfig().endpoint_ratio_cap == 5.0


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
                           endpoint_ratio_cap=None, candidate_log=log)

    assert log, 'the fixture must produce candidates'
    for entry in log:
        assert set(entry) == {'feature_idx', 'range1', 'range2', 'overlap_ratio',
                             'endpoint_ratio', 'shift_mass_1', 'shift_mass_2',
                             'rel_deg', 'accepted', 'error_app', 'error_ddos'}
        assert 0.0 <= entry['shift_mass_1'] <= 1.0
        assert len(entry['rel_deg']) == 4
        assert isinstance(entry['accepted'], bool)


def test_the_candidate_log_is_off_by_default_and_costs_nothing():
    """It runs inside the hot path, so it must be strictly opt-in."""
    import inspect

    signature = inspect.signature(ta.align_rf_thresholds)
    assert signature.parameters['candidate_log'].default is None
    assert signature.parameters['endpoint_ratio_cap'].default == 5.0
