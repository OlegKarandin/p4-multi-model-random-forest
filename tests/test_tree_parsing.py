"""Semantic tests for the tree -> interval -> codeword parse chain.

Unlike the rest of the P4-generator tests (which assert on emitted TEXT),
these assert on MEANING: they run real fitted forests through the generator's
own parse chain and compare the resulting table semantics against sklearn's
own predict(). A defect anywhere in export_text parsing, threshold->interval
derivation, or codeword construction shows up here as a misclassified input.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.p4gen import build_p4_script as bps


def _fit(n_features, max_depth, seed, value_hi=(2 ** 16) - 2, n_samples=6000,
         n_classes=3, n_estimators=1):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, value_hi + 1, size=(n_samples, n_features))
    y = ((X[:, 0] // max(1, value_hi // 9)) +
         (X[:, 1 % n_features] // max(1, value_hi // 5))) % n_classes
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=seed, bootstrap=False).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf), X


def _parse(clf, features):
    trees = bps.get_tree_textual_representation(clf, features)
    tree_nodes = {t: bps.get_nodes(trees[t]) for t in trees}
    intervals = bps.get_feature_intervals_from_thresholds(
        bps.get_feature_thresholds(tree_nodes))
    paths = bps.get_root_to_leaf_paths(tree_nodes)
    return tree_nodes, intervals, bps.generate_codewords(paths, intervals)


def _thermometer(intervals, value):
    """The exact code get_table_entries writes for the interval containing
    `value`: original-order interval i -> '0'*i + '1'*(n-1-i)."""
    n = len(intervals)
    for i, (lo, hi) in enumerate(intervals):
        if lo <= value <= hi:
            return "0" * i + "1" * (n - 1 - i)
    raise AssertionError("value {} in no interval of {}".format(value, intervals))


def _simulate(codewords_for_tree, intervals, features, sample):
    bits = "".join(_thermometer(intervals[f], sample[f]) for f in intervals)
    hits = [cls for cw, cls in codewords_for_tree.items()
            if len(cw) == len(bits) and all(p == "*" or p == b for p, b in zip(cw, bits))]
    return hits


def _assert_matches_sklearn(clf, features, probes, intervals, codewords):
    sk = np.array([e.predict(probes) for e in clf.estimators_])
    for k in range(len(probes)):
        sample = {f: int(probes[k, i]) for i, f in enumerate(features)}
        for t in codewords:
            hits = _simulate(codewords[t], intervals, features, sample)
            assert hits, "input {} matched no entry in tree {}".format(sample, t)
            assert int(float(hits[0])) == int(sk[t, k]), (
                "tree {} input {} -> {} but sklearn says {}".format(
                    t, sample, hits[0], sk[t, k]))


# ---------------------------------------------------------------------------
# export_text truncation
#
# sklearn's export_text defaults to max_depth=10 and renders anything deeper
# as "|--- truncated branch of depth N" lines, which contain neither "class"
# nor "<=" -- so get_nodes() drops them with no error at all. Measured before
# the fix: a real depth-12 tree with 350 leaves parsed as 168, and on a
# depth-14 tree 2516 of 4000 probe inputs matched no table entry.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [12, 14])
def test_deep_tree_parses_every_leaf(depth):
    features = ["Feat_0", "Feat_1", "Feat_2"]
    clf, _ = _fit(len(features), depth, seed=depth, n_samples=20000)
    assert clf.estimators_[0].get_depth() >= 12, "fixture did not build a deep tree"

    tree_nodes, _, _ = _parse(clf, features)

    parsed_leaves = sum(1 for n in tree_nodes[0].values() if n["is_leaf"])
    assert parsed_leaves == clf.estimators_[0].get_n_leaves()


def test_deep_tree_classifies_identically_to_sklearn():
    features = ["Feat_0", "Feat_1", "Feat_2"]
    clf, _ = _fit(len(features), 13, seed=99, n_samples=20000)
    assert clf.estimators_[0].get_depth() >= 12, "fixture did not build a deep tree"
    _, intervals, codewords = _parse(clf, features)

    rng = np.random.RandomState(5)
    probes = rng.randint(0, (2 ** 16) - 1, size=(400, len(features)))

    _assert_matches_sklearn(clf, features, probes, intervals, codewords)


# ---------------------------------------------------------------------------
# threshold == 0
#
# get_feature_intervals_from_thresholds used to skip a split at 0 outright
# ("avoid creating a [0,0] interval"), while generate_codewords still saw
# that condition in every leaf path. The ">" branch then matched no interval
# bound at all and stayed fully wildcarded, so it also matched values it
# should have excluded. dataset.py keeps zero-valued rows, and a "counter is
# zero vs non-zero" split lands on sklearn threshold 0.5 -> int 0, so this is
# reachable on the real data, not a synthetic edge case.
# ---------------------------------------------------------------------------

def _zero_split_forest():
    rng = np.random.RandomState(3)
    X = rng.randint(0, 4, size=(4000, 2))
    y = (X[:, 0] == 0).astype(int) * 2 + (X[:, 1] >= 2).astype(int)
    clf = RandomForestClassifier(n_estimators=1, max_depth=4, random_state=0,
                                 bootstrap=False).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


def test_zero_threshold_is_kept_as_its_own_interval():
    features = ["Feat_0", "Feat_1"]
    clf = _zero_split_forest()
    tree_nodes, intervals, _ = _parse(clf, features)

    thresholds = bps.get_feature_thresholds(tree_nodes)
    assert any(t == 0 for _, t in thresholds), "fixture produced no split at 0"

    # A split at 0 must produce a real [0, 0] interval, so that "<= 0" and
    # "> 0" are distinguishable in the codeword.
    zero_features = {f for f, t in thresholds if t == 0}
    for f in zero_features:
        assert intervals[f][0] == (0, 0)


def test_zero_threshold_classifies_identically_to_sklearn():
    features = ["Feat_0", "Feat_1"]
    clf = _zero_split_forest()
    _, intervals, codewords = _parse(clf, features)

    probes = np.array([[a, b] for a in range(4) for b in range(4)])

    _assert_matches_sklearn(clf, features, probes, intervals, codewords)


# ---------------------------------------------------------------------------
# Regression guard for the shallow case that already worked, so a fix to
# either defect above cannot silently break the common path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shallow_multi_tree_forest_classifies_identically_to_sklearn(seed):
    features = ["Feat_0", "Feat_1", "Feat_2"]
    clf, _ = _fit(len(features), 5, seed=seed, n_estimators=3)
    _, intervals, codewords = _parse(clf, features)

    rng = np.random.RandomState(100 + seed)
    probes = rng.randint(0, (2 ** 16) - 1, size=(300, len(features)))

    _assert_matches_sklearn(clf, features, probes, intervals, codewords)
