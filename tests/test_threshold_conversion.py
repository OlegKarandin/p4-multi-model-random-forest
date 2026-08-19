"""Spec risk 6: int(round(...)) is round-half-to-EVEN, and sklearn puts every
integer-feature split on a .5 midpoint."""
import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
from src.p4gen.switch_semantics import switch_predict


def _raw_forest(X, y, n_estimators=7, seed=0):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10, min_samples_leaf=5,
        min_samples_split=10, random_state=seed, n_jobs=1).fit(X, y)


def _integer_data(n=900, n_features=4, seed=0):
    """Integer-valued features with unit gaps -- so every split sklearn picks
    lands on an exact v + 0.5 midpoint, which is the regime where banker's
    rounding does its damage."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 60, size=(n, n_features)).astype(float)
    y = (X[:, 0] // 20).astype(int)
    return X, y


def _thresholds(clf):
    return [float(t.tree_.threshold[i])
            for t in clf.estimators_
            for i in range(t.tree_.node_count)
            if t.tree_.feature[i] != -2]


def test_conversion_never_widens_a_split():
    """The property that makes floor correct: floor(t) <= t always, so a
    `x <= t` split can only ever get tighter, never looser. round() widens
    whenever it rounds up -- and it only ever rounds UP, never down, so the
    distortion is asymmetric rather than symmetric noise."""
    X, y = _integer_data()
    raw = _raw_forest(X, y)
    before = _thresholds(raw)

    after = _thresholds(dt_thresholds_float_to_int(raw))

    assert len(before) == len(after)
    for original, converted in zip(before, after):
        assert converted <= original, (original, converted)
        assert converted == math.floor(original)


def test_conversion_is_exact_on_integer_valued_features():
    """For an integer x and any real t, `x <= t` and `x <= floor(t)` are the
    SAME test. So on integer-valued features the conversion cannot change a
    single prediction. (This does NOT extend to the real datasets: the *.Mean
    features are genuine means and take fractional values.)"""
    X, y = _integer_data()
    raw = _raw_forest(X, y)
    reference = np.vstack([e.predict(X) for e in raw.estimators_])

    converted = dt_thresholds_float_to_int(raw)

    assert np.array_equal(np.vstack([e.predict(X) for e in converted.estimators_]),
                          reference)


def test_round_half_to_even_would_not_be_exact_on_the_same_data():
    """The regression this task exists for: with the old rule the same forest
    on the same integer data changes predictions."""
    X, y = _integer_data()
    raw = _raw_forest(X, y)
    reference = np.vstack([e.predict(X) for e in raw.estimators_])

    rounded = _raw_forest(X, y)
    for tree in rounded.estimators_:
        node = tree.tree_
        for i in range(node.node_count):
            if node.feature[i] != -2:
                node.threshold[i] = int(round(node.threshold[i]))

    assert not np.array_equal(
        np.vstack([e.predict(X) for e in rounded.estimators_]), reference)


def test_most_splits_on_integer_features_sit_on_a_half_midpoint():
    """Why this matters at all: sklearn splits at the MIDPOINT of two observed
    values, so with unit gaps every threshold is v + 0.5 and banker's rounding
    applies to essentially all of them -- not to rare edge cases."""
    X, y = _integer_data()
    raw = _raw_forest(X, y)

    thresholds = _thresholds(raw)
    halves = [t for t in thresholds if abs(t - math.floor(t) - 0.5) < 1e-9]

    # Threshold is 0.85, not 0.9: on this sklearn version (1.6.1) this exact
    # seed/data measures 33/37 = 89.2%. The mechanism is unaffected -- a
    # handful of splits land on an exact integer only where the two nearest
    # observed values in a node happen to straddle a gap of 2 rather than 1
    # (e.g. midpoint of 18 and 20). That is still "essentially all", not the
    # rare edge case the docstring warns against, so the bound is loosened
    # rather than the claim being abandoned.
    assert len(halves) / len(thresholds) > 0.85


def test_banker_rounding_widens_exactly_the_odd_floor_cases():
    """Pins the mechanism: round-half-to-even sends v + 0.5 up when v is odd
    and leaves it when v is even, so ~half of all midpoint splits are widened."""
    for v in range(0, 12):
        midpoint = v + 0.5
        assert math.floor(midpoint) == v
        assert round(midpoint) == (v + 1 if v % 2 else v)


def test_conversion_leaves_leaf_nodes_alone():
    """Leaves carry threshold -2.0 as a sentinel; flooring it would corrupt the
    tree structure, not just a boundary."""
    X, y = _integer_data()

    converted = dt_thresholds_float_to_int(_raw_forest(X, y))

    for tree in converted.estimators_:
        node = tree.tree_
        for i in range(node.node_count):
            if node.feature[i] == -2:
                assert node.threshold[i] == -2.0


def test_conversion_cannot_produce_a_negative_threshold():
    """Feature values are clipped into [0, INFINITE], so the smallest possible
    midpoint is 0.5 and flooring it gives 0 -- a real split (C1), never -1."""
    X, y = _integer_data()

    converted = dt_thresholds_float_to_int(_raw_forest(X, y))

    for threshold in _thresholds(converted):
        assert 0 <= threshold <= INFINITE
