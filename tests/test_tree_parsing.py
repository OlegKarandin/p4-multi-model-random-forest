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
    # Task 15: production now derives tree_nodes straight off each
    # estimator's tree_ arrays via get_nodes(estimator, features), one
    # estimator at a time -- no more export_text round-trip. Mirrors the
    # exact call shape build_p4_script.get_feature_intervals etc. use.
    tree_nodes = {i: bps.get_nodes(est, features)
                  for i, est in enumerate(clf.estimators_)}
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
# Deep trees
#
# Historically this exercised export_text's truncation hazard directly,
# since the old get_nodes() parsed export_text's rendered text (default
# max_depth=10, anything deeper rendered as "|--- truncated branch of depth
# N" lines containing neither "class" nor "<=", silently dropped -- measured
# before the fix: a real depth-12 tree with 350 leaves parsed as 168, and on
# a depth-14 tree 2516 of 4000 probe inputs matched no table entry). Task 15
# replaced get_nodes() with a direct estimator.tree_ walk that has no
# rendered text to under-size, so these two tests are now a depth-stress
# regression guard for that implementation rather than a truncation
# regression test -- the truncation hazard itself is characterised
# explicitly, against both implementations, in the
# "old vs. new get_nodes" section below.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [12, 14])
def test_deep_tree_parses_every_leaf(depth):
    features = ["feat_0", "feat_1", "feat_2"]
    clf, _ = _fit(len(features), depth, seed=depth, n_samples=20000)
    assert clf.estimators_[0].get_depth() >= 12, "fixture did not build a deep tree"

    tree_nodes, _, _ = _parse(clf, features)

    parsed_leaves = sum(1 for n in tree_nodes[0].values() if n["is_leaf"])
    assert parsed_leaves == clf.estimators_[0].get_n_leaves()


def test_deep_tree_classifies_identically_to_sklearn():
    features = ["feat_0", "feat_1", "feat_2"]
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
    features = ["feat_0", "feat_1"]
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
    features = ["feat_0", "feat_1"]
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
    features = ["feat_0", "feat_1", "feat_2"]
    clf, _ = _fit(len(features), 5, seed=seed, n_estimators=3)
    _, intervals, codewords = _parse(clf, features)

    rng = np.random.RandomState(100 + seed)
    probes = rng.randint(0, (2 ** 16) - 1, size=(300, len(features)))

    _assert_matches_sklearn(clf, features, probes, intervals, codewords)


# ---------------------------------------------------------------------------
# Old vs. new get_nodes (Task 15)
#
# get_nodes() used to render each tree with export_text and re-parse the
# text (a state machine keyed on "|" counts and the literal strings "class"
# and "<="). It now reads estimator.tree_'s C-level arrays directly. The old
# implementation is kept as _get_nodes_from_text (fed pre-rendered text) so
# this suite can compare the two against real fitted forests and prove they
# agree, node dict for node dict, before _get_nodes_from_text is deleted in
# the following commit.
#
# Two shape differences are deliberate and reconciled explicitly below:
#   - "class": _get_nodes_from_text keeps export_text's printed STRING
#     ("2.0"); get_nodes returns int(np.argmax(...)), a class INDEX. Asserted
#     via int(old["class"]) == new["class"], not skipped.
#   - node ids: _get_nodes_from_text numbers nodes in export_text's print
#     order (a fresh counter); get_nodes uses sklearn's own tree_ node ids.
#     Both are pre-order traversals of the same tree, so they coincide --
#     proved below by asserting the two node dicts have IDENTICAL key sets
#     across every forest tested, not assumed.
#
# A third, non-obvious difference was found while writing this suite and is
# reconciled the same way: "depth" is off by a constant +1 in the old
# scheme. export_text's rendered root line already contains one "|"
# character (from its "|--- " branch marker), and _get_nodes_from_text's
# depth is literally line.count("|"), so its root is depth 1; tree_'s own
# depth is 0-based, so get_nodes' root is depth 0 -- every node down the
# tree inherits the same +1 offset. This is NOT one of the two shape
# differences the brief names, so it is called out here rather than folded
# in silently. It cannot affect any observable output: grepping the whole
# tree for node["depth"]/node['depth'] usage outside get_nodes and
# _get_nodes_from_text themselves (both here and across build_p4_script.py,
# evaluation.py, main.py) turns up nothing -- "depth" is write-only
# bookkeeping neither implementation's own callers ever read back.
# ---------------------------------------------------------------------------

def _assert_same_node_dicts(old_nodes, new_nodes):
    assert set(old_nodes.keys()) == set(new_nodes.keys()), (
        "node id sets differ -- the pre-order coincidence between "
        "_get_nodes_from_text's print-order numbering and get_nodes' raw "
        "tree_ indices does not hold for this tree")

    for node_id in new_nodes:
        old, new = old_nodes[node_id], new_nodes[node_id]

        assert old["is_leaf"] == new["is_leaf"], (node_id, old, new)
        assert old["father_node"] == new["father_node"], (node_id, old, new)
        # Reconciled depth-numbering-base difference, see module comment above.
        assert old["depth"] - 1 == new["depth"], (node_id, old, new)

        if new["is_leaf"]:
            # Reconciled class-as-string-vs-index difference (brief's shape
            # difference #1), asserted explicitly rather than skipped.
            assert int(float(old["class"])) == new["class"], (node_id, old, new)
        else:
            assert old["feature"] == new["feature"], (node_id, old, new)
            assert old["threshold"] == new["threshold"], (node_id, old, new)
            assert old["left_child"] == new["left_child"], (node_id, old, new)
            assert old["right_child"] == new["right_child"], (node_id, old, new)


def _old_and_new_nodes(clf, features):
    trees = bps.get_tree_textual_representation(clf, features)
    old_nodes = {i: bps._get_nodes_from_text(trees[i]) for i in trees}
    new_nodes = {i: bps.get_nodes(est, features)
                 for i, est in enumerate(clf.estimators_)}
    return old_nodes, new_nodes


@pytest.mark.parametrize("n_estimators,max_depth,seed", [
    (1, 3, 0),
    (1, 5, 1),
    (3, 5, 2),
    (5, 8, 3),
    (1, 10, 4),   # exactly export_text's own truncation default
    (1, 12, 12),  # deep enough to trigger export_text's default truncation
    (3, 14, 14),  # deep, multi-tree
    (1, None, 7),  # fully unbounded depth
])
def test_get_nodes_agrees_with_text_parse(n_estimators, max_depth, seed):
    features = ["feat_0", "feat_1", "feat_2"]
    clf, _ = _fit(len(features), max_depth, seed=seed, n_samples=20000,
                  n_estimators=n_estimators)

    old_nodes, new_nodes = _old_and_new_nodes(clf, features)

    assert set(old_nodes.keys()) == set(new_nodes.keys())
    for tree_id in new_nodes:
        _assert_same_node_dicts(old_nodes[tree_id], new_nodes[tree_id])


def test_get_nodes_agrees_with_text_parse_on_zero_threshold_forest():
    # The threshold==0 fixture above exercises a real corner of
    # get_feature_intervals_from_thresholds/generate_codewords, not of
    # get_nodes itself -- but it is still worth confirming the two
    # implementations agree on it, since a threshold of exactly 0 is where
    # int(math.floor(...)) (new) and int(float(...)) (old, on export_text's
    # printed "0.50" text after dt_thresholds_float_to_int has already
    # floored it to "0.00") would be most likely to disagree if they ever did.
    features = ["feat_0", "feat_1"]
    clf = _zero_split_forest()
    old_nodes, new_nodes = _old_and_new_nodes(clf, features)
    for tree_id in new_nodes:
        _assert_same_node_dicts(old_nodes[tree_id], new_nodes[tree_id])


# ---------------------------------------------------------------------------
# The truncation hazard itself, exploited directly
#
# get_tree_textual_representation always sizes export_text's max_depth to
# the tree's OWN depth, so production never actually hit the truncation
# bug even before Task 15 -- but the bug class it existed to guard against
# is real: export_text's own default (max_depth=10, undocumented-by-keyword
# but confirmed by sklearn's source) truncates anything deeper. This test
# calls export_text directly, the way get_tree_textual_representation would
# WITHOUT its explicit max_depth= argument, to prove that gap concretely:
# _get_nodes_from_text silently loses leaves under it, while get_nodes
# (reading tree_ directly, no text involved) is immune by construction.
# ---------------------------------------------------------------------------

def test_export_text_default_max_depth_truncates_a_deep_tree():
    from sklearn.tree import export_text

    features = ["feat_0", "feat_1", "feat_2"]
    clf, _ = _fit(len(features), 13, seed=99, n_samples=20000)
    tree = clf.estimators_[0]
    assert tree.get_depth() > 10, "fixture did not build a tree deeper than export_text's default"

    truncated_text = export_text(tree, feature_names=features)  # no max_depth= -> sklearn default of 10
    truncated_nodes = bps._get_nodes_from_text(truncated_text)
    truncated_leaves = sum(1 for n in truncated_nodes.values() if n["is_leaf"])

    new_nodes = bps.get_nodes(tree, features)
    new_leaves = sum(1 for n in new_nodes.values() if n["is_leaf"])

    assert new_leaves == tree.get_n_leaves(), (
        "get_nodes must recover every real leaf regardless of tree depth")
    assert truncated_leaves < new_leaves, (
        "fixture must actually exercise export_text's truncation for this test to "
        "prove anything -- if this fails, the tree was not deep enough")
