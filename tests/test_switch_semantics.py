"""What the generated P4 program computes -- which is NOT rf.predict."""
import numpy as np
import pytest
from itertools import product
from sklearn.ensemble import RandomForestClassifier

from src.p4gen import switch_semantics as ss
from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int, generate_voting_code


def _forest(n_estimators=7, n_classes=3, min_samples_leaf=200, n=1200, seed=0):
    """Impure leaves on purpose: min_samples_leaf up to 200 at max_depth <= 10 is
    the search's normal regime, and it is where hard and soft voting diverge."""
    rng = np.random.default_rng(seed)
    X = np.clip(rng.integers(0, 90000, size=(n, 5)), 0, INFINITE).astype(float)
    y = np.array([c % n_classes for c in range(n)])
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10, min_samples_leaf=min_samples_leaf,
        min_samples_split=10, random_state=seed, n_jobs=1).fit(X, y)
    return dt_thresholds_float_to_int(rf), X, y


def test_per_tree_indices_are_what_the_switch_carries():
    """meta.class_tree_<task>_<i> holds one class per tree -- the leaf's
    argmax, which is exactly what a sub-estimator's predict returns."""
    rf, X, y = _forest()

    idx = ss.per_tree_class_indices(rf, X)

    assert idx.shape == (len(rf.estimators_), X.shape[0])
    assert idx.dtype == np.intp
    for tree_idx, estimator in enumerate(rf.estimators_):
        assert np.array_equal(idx[tree_idx], estimator.predict(X).astype(np.intp))


def test_vote_winner_breaks_ties_toward_the_smallest_class_index():
    """Order-independence is the reason this rule is the specification:
    statistics.mode returns the FIRST-ENCOUNTERED mode, so its winner depends on
    tree ordering, which is arbitrary."""
    assert ss.vote_winner([1, 0, 2], 3) == 0
    assert ss.vote_winner([2, 1, 0], 3) == 0
    assert ss.vote_winner([0, 2, 1, 1, 2], 3) == 1
    assert ss.vote_winner([1, 1, 0], 3) == 1
    assert ss.vote_winner([0], 3) == 0


def test_vote_winner_is_invariant_to_tree_order():
    for arr in ([1, 0, 2], [0, 2, 1, 1, 2], [2, 2, 0, 1, 1, 1, 0]):
        expected = ss.vote_winner(arr, 3)
        for permuted in (list(reversed(arr)), sorted(arr), sorted(arr, reverse=True)):
            assert ss.vote_winner(permuted, 3) == expected, arr


def test_the_generated_vote_table_uses_exactly_vote_winner():
    """The switch's rule and this module's rule must be one rule. Parse the
    generated const entries back out and compare every one."""
    for num_trees, num_classes in ((3, 3), (5, 3), (7, 3), (1, 2), (3, 2)):
        table, _ = generate_voting_code(num_trees, num_classes, 'app')

        entries = {}
        for line in table.splitlines():
            line = line.strip()
            if not line.startswith('('):
                continue
            key_text, action = line.split(') : ')
            key = tuple(int(v) for v in key_text.lstrip('(').split(', '))
            entries[key] = int(action.split('(')[1].rstrip(');').rstrip(')'))

        assert len(entries) == num_classes ** num_trees, (num_trees, num_classes)
        for key, winner in entries.items():
            assert winner == ss.vote_winner(list(key), num_classes), (key, winner)


def test_switch_predict_is_the_hard_vote_over_the_per_tree_classes():
    rf, X, y = _forest()

    got = ss.switch_predict(rf, X)

    idx = ss.per_tree_class_indices(rf, X)
    expected = np.array([rf.classes_[ss.vote_winner(idx[:, i].tolist(), rf.n_classes_)]
                         for i in range(X.shape[0])])
    assert np.array_equal(got, expected)


def test_switch_predict_diverges_from_rf_predict_on_impure_leaves():
    """The defect this task exists for: rf.predict averages predict_proba (SOFT)
    while the switch votes hard. Measured on the real app dataset at 7 trees /
    min_samples_leaf=200 this is 6.6% of flows and 1.2 accuracy points."""
    rf, X, y = _forest(n_estimators=7, n_classes=3, min_samples_leaf=200)

    assert not np.array_equal(ss.switch_predict(rf, X), rf.predict(X))


def test_a_single_tree_forest_cannot_diverge():
    """With one tree the hard vote IS the tree's own argmax, which is what
    rf.predict reduces to. A useful sanity anchor: any divergence here would
    mean the per-tree class extraction is wrong."""
    rf, X, y = _forest(n_estimators=1)

    assert np.array_equal(ss.switch_predict(rf, X), rf.predict(X))


def test_switch_accuracy_scorer_matches_accuracy_of_switch_predict():
    from sklearn.metrics import accuracy_score

    rf, X, y = _forest()

    assert ss.switch_accuracy_scorer(rf, X, y) == pytest.approx(
        accuracy_score(y, ss.switch_predict(rf, X)))


def test_the_scorer_is_usable_by_permutation_importance():
    """P2 Task 5 passes it to permutation_importance, which calls a scorer as
    scorer(estimator, X, y) -- so the signature has to be exactly that."""
    from sklearn.inspection import permutation_importance

    rf, X, y = _forest(n=400)

    result = permutation_importance(
        rf, X, y, scoring=ss.switch_accuracy_scorer, n_repeats=2, random_state=0)

    assert result.importances_mean.shape == (X.shape[1],)
