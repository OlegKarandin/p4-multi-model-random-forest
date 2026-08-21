"""Tests for src/training/incremental_metrics.py -- the exact confusion-matrix
metric functions that replace accuracy_metrics's sklearn calls in the
threshold-alignment loop (P3b Task 2a).

The bar throughout is exact bit equality with sklearn (`accuracy_score`,
`f1_score(..., average='weighted')`), never `pytest.approx`: these numbers
decide alignment accept/reject, and the trajectory is path-dependent, so a
one-ULP disagreement can flip a decision and cascade.
"""
import random
import warnings

import numpy as np
import pytest
import sklearn.metrics as mt
from sklearn.exceptions import UndefinedMetricWarning

from src.training import incremental_metrics as im

LABEL_SPACES = ([0, 1, 2], [-1, 1])  # App, DDoS -- this project's only two tasks


def _acc_and_f1(lab, y_true, y_pred, classes):
    """Test-side glue mirroring how a caller is expected to use the module:
    build the universe once, map y_true/y_pred into it, count, then read off
    both metrics. Not part of the module under test -- Task 2b owns the real
    incremental version of this plumbing.
    """
    universe = im.label_universe(lab, y_true, classes)
    k = len(universe)
    pos = {label: i for i, label in enumerate(universe)}
    y_uni = np.array([pos[v] for v in y_true], dtype=np.intp)
    pred_uni = np.array([pos[v] for v in y_pred], dtype=np.intp)
    confusion = im.confusion_from_predictions(y_uni, pred_uni, k)
    acc = im.accuracy_from_confusion(confusion, len(y_true))
    f1 = im.weighted_f1_from_confusion(confusion, len(lab))
    return acc, f1


def _sklearn_acc_and_f1(lab, y_true, y_pred):
    """sklearn's own numbers for the same inputs, silencing
    UndefinedMetricWarning -- the ONE declared behavioural difference
    (see incremental_metrics.weighted_f1_from_confusion's docstring): the
    VALUE is identical, only sklearn's warning is absent from our path.
    """
    acc = mt.accuracy_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        f1 = mt.f1_score(y_true, y_pred, labels=lab, average='weighted')
    return acc, f1


def test_weighted_f1_and_accuracy_are_bit_identical_to_sklearn_over_random_class_subsets():
    """Property test over both label spaces, n in [1,40], with the alphabet
    y_true is drawn from and the alphabet y_pred is drawn from sampled
    INDEPENDENTLY -- so a class can vanish from y_true only, from y_pred
    only, from both, or from neither. The planning run did 20 000 draws with
    0 mismatches (p3b-design-reference.md section 2); this runs a few
    thousand so it stays fast.
    """
    rng = random.Random(20260820)
    n_cases = 4000
    for _ in range(n_cases):
        lab = rng.choice(LABEL_SPACES)
        n = rng.randint(1, 40)
        true_alphabet = rng.sample(lab, k=rng.randint(1, len(lab)))
        pred_alphabet = rng.sample(lab, k=rng.randint(1, len(lab)))
        y_true = [rng.choice(true_alphabet) for _ in range(n)]
        y_pred = [rng.choice(pred_alphabet) for _ in range(n)]

        acc, f1 = _acc_and_f1(lab, y_true, y_pred, classes=lab)
        expected_acc, expected_f1 = _sklearn_acc_and_f1(lab, y_true, y_pred)

        assert acc == expected_acc, (lab, y_true, y_pred)
        assert f1 == expected_f1, (lab, y_true, y_pred)


def test_a_class_with_zero_support_matches_sklearn_and_a_perfect_match_scores_one():
    lab = [0, 1, 2]

    y_true = [0, 0, 1]
    y_pred = [2, 2, 2]
    acc, f1 = _acc_and_f1(lab, y_true, y_pred, classes=lab)
    expected_acc, expected_f1 = _sklearn_acc_and_f1(lab, y_true, y_pred)
    assert acc == expected_acc
    assert f1 == expected_f1

    y_true2 = [0, 0, 0]
    y_pred2 = [0, 0, 0]
    acc2, f1_2 = _acc_and_f1(lab, y_true2, y_pred2, classes=lab)
    assert acc2 == 1.0
    assert f1_2 == 1.0


def test_no_scored_label_present_takes_the_unweighted_average_branch_without_raising():
    """true_sum.sum() == 0: none of `lab`'s labels occur in y_true at all.
    sklearn reaches this via _nanaverage catching ZeroDivisionError (weights
    summing to zero) and retrying unweighted -- verify we take the same
    branch, land on the same value, and that no ZeroDivisionError escapes
    (an uncaught exception here would fail this test on its own, but the
    intent is spelled out explicitly).
    """
    lab = [0, 1, 2]
    y_true = [3, 3]  # neither label is in lab -- true_sum for [0,1,2] is [0,0,0]
    y_pred = [3, 3]

    try:
        acc, f1 = _acc_and_f1(lab, y_true, y_pred, classes=lab)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError escaped weighted_f1_from_confusion")

    expected_acc, expected_f1 = _sklearn_acc_and_f1(lab, y_true, y_pred)
    assert acc == expected_acc == 1.0
    assert f1 == expected_f1


def test_accuracy_from_confusion_matches_accuracy_score_exactly_including_a_true_label_outside_lab():
    lab = [0, 1, 2]
    y_true = [3, 3]  # true label outside lab entirely
    y_pred = [3, 3]
    acc, _ = _acc_and_f1(lab, y_true, y_pred, classes=lab)
    assert acc == mt.accuracy_score(y_true, y_pred) == 1.0

    # A plain in-domain case too, so this test isn't only exercising the edge.
    lab2 = [-1, 1]
    y_true2 = [-1, 1, -1, 1, 1]
    y_pred2 = [-1, -1, -1, 1, 1]
    acc2, _ = _acc_and_f1(lab2, y_true2, y_pred2, classes=lab2)
    assert acc2 == mt.accuracy_score(y_true2, y_pred2)


def test_accuracy_from_confusion_rejects_zero_samples():
    # `n_samples == 0` is out of scope (Ruling P3b-1): asserted, not defined.
    # It used to be a bare `assert`, which `python -O` strips entirely,
    # silently returning nan from a 0/0 division instead of failing loudly.
    confusion = np.zeros((2, 2), dtype=np.intp)
    with pytest.raises(ValueError):
        im.accuracy_from_confusion(confusion, 0)


def test_the_ddos_label_space_is_not_treated_as_array_indices():
    """Regression test for the specific risk called out in the task: -1 is a
    valid label but an invalid raw array index (or a silently-wrong one --
    numpy would happily wrap it to the last element). Every lookup in this
    module must go through the universe's position mapping, never index an
    array with a label directly. rf.classes_ == [-1, 1] but only 1 is ever
    predicted, so class -1 has support but zero predictions.
    """
    lab = [-1, 1]
    classes = [-1, 1]
    y_true = [-1, 1, 1, -1, 1]
    y_pred = [1, 1, 1, 1, 1]

    acc, f1 = _acc_and_f1(lab, y_true, y_pred, classes)
    expected_acc, expected_f1 = _sklearn_acc_and_f1(lab, y_true, y_pred)
    assert acc == expected_acc
    assert f1 == expected_f1


def test_confusion_from_predictions_agrees_with_a_from_scratch_python_count():
    """This is the oracle Task 2b's incremental confusion-matrix maintenance
    will be checked against, so it needs to be right in its own name -- not
    merely agree with sklearn's end-to-end numbers.
    """
    rng = random.Random(11)
    k = 4
    n = 200
    y_uni = [rng.randrange(k) for _ in range(n)]
    pred_uni = [rng.randrange(k) for _ in range(n)]

    confusion = im.confusion_from_predictions(y_uni, pred_uni, k)

    expected = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_uni, pred_uni):
        expected[t, p] += 1

    assert confusion.shape == (k, k)
    assert np.array_equal(confusion, expected)


def test_label_universe_places_lab_first_in_order_then_appends_other_labels_sorted():
    """Section 1's construction: lab verbatim (its order matters -- it fixes
    which universe rows/cols weighted_f1_from_confusion reads), then every
    other label seen in y_true or predictable by the model, sorted."""
    lab = [2, 0, 1]
    y_true = [5, 0, -3]
    classes = [2, 0, 1, 9]
    universe = im.label_universe(lab, y_true, classes)
    assert universe == [2, 0, 1, -3, 5, 9]


# ---------------------------------------------------------------------------
# Task 2b: IncrementalMetrics -- the mutable state that maintains the vote
# matrix, the per-sample winner and the confusion matrix in O(#changed) as
# align_rf_thresholds applies and rolls back candidate threshold moves.
#
# The oracle everywhere below is the pair
# `accuracy_metrics(y, compute_ensemble_prediction(tree_predictions, rf), task)`
# -- the exact expression the alignment loop evaluated before this change.
# Comparison is with `==`, never approx, for the same reason as above.
# ---------------------------------------------------------------------------

from src.p4gen.evaluation import accuracy_metrics
from src.training import threshold_alignment as ta
from src.training.incremental_metrics import IncrementalMetrics


class _FakeForest:
    """The only two attributes IncrementalMetrics and
    compute_ensemble_prediction ever read off a RandomForestClassifier:
    `classes_` (the label alphabet, an ndarray, in sklearn's sorted order) and
    `n_classes_` (its length, which is also the width of the vote matrix).

    Using a stub rather than a fitted forest is what lets the property tests
    below drive thousands of ARBITRARY (n_trees, n_samples) prediction blocks
    -- including class alphabets a fitted forest would never produce, such as
    a model that can only emit two of its task's three labels. The end-to-end
    tests in test_threshold_alignment.py cover the real-forest path.
    """

    def __init__(self, classes):
        self.classes_ = np.asarray(classes)
        self.n_classes_ = len(self.classes_)


def test_incremental_metrics_rejects_zero_samples():
    # Same guard as accuracy_from_confusion, and for the same reason: this
    # used to be a bare `assert`, silently disabled under `python -O`.
    rf = _FakeForest([0, 1, 2])
    tree_predictions = np.zeros((3, 0), dtype=np.intp)
    with pytest.raises(ValueError):
        IncrementalMetrics(tree_predictions, rf, np.array([]), 'app')


def _oracle(y_true, tree_predictions, rf, task):
    """The exact expression align_rf_thresholds evaluated per candidate before
    T2b. UndefinedMetricWarning is silenced for the same reason as in
    _sklearn_acc_and_f1 above -- it is the one declared behavioural difference,
    and these property tests deliberately generate the degenerate cases that
    trigger it thousands of times."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        return accuracy_metrics(
            y_true, ta.compute_ensemble_prediction(tree_predictions, rf), task)


def _mutate(rng, tree_predictions, n_classes, n_trees_touched=None):
    """Flip a random subset of (tree, sample) entries IN PLACE and return the
    `undo_info`-shaped dict that update_cache_for_modifications would have
    produced for exactly that change: `{'predictions': {tree_idx:
    (sample_indices, old_predictions)}}`, sample indices sorted-unique (which
    is what update_cache_for_modifications:820's np.unique guarantees today).

    The replacement class is drawn freely, so a "flip" that leaves the value
    unchanged happens often -- that is deliberate: it exercises apply's
    `new != old` filter and its no-op / None-token path.
    """
    n_trees, n_samples = tree_predictions.shape
    undo_info = {'predictions': {}, 'node_samples': {}}
    if n_trees_touched is None:
        n_trees_touched = rng.integers(1, n_trees + 1)
    for tree_idx in rng.choice(n_trees, size=int(n_trees_touched), replace=False):
        tree_idx = int(tree_idx)
        count = int(rng.integers(1, n_samples + 1))
        idx = np.unique(rng.integers(0, n_samples, size=count)).astype(np.intp)
        old = tree_predictions[tree_idx, idx].copy()
        tree_predictions[tree_idx, idx] = rng.integers(0, n_classes, size=idx.size)
        undo_info['predictions'][tree_idx] = (idx, old)
    return undo_info


def test_the_incremental_metrics_equal_the_from_scratch_oracle_after_every_step():
    """The central property test: after EVERY mutation step, the incrementally
    maintained (accuracy, weighted_f1) must equal, bit for bit, what the
    alignment loop used to compute from scratch --
    accuracy_metrics(y, compute_ensemble_prediction(...), task).

    Trials deliberately include forests whose `classes_` omits a label that
    `lab` scores (e.g. an App model that can only ever emit 0 and 2): that
    class then has support but zero predictions, which is the case where
    sklearn's zero-division branch and the weighted average interact.
    """
    rng = np.random.default_rng(20260820)

    for _ in range(400):
        lab = [0, 1, 2] if rng.random() < 0.5 else [-1, 1]
        task = 'app' if lab == [0, 1, 2] else 'ddos'
        # Sometimes drop a label from the model's alphabet, but never from the
        # true labels -- y_true always spans the whole task label space.
        classes = list(lab)
        if len(lab) > 2 and rng.random() < 0.4:
            dropped = int(rng.choice(lab))  # hoisted: one draw, not one per element
            classes = [c for c in lab if c != dropped]
        rf = _FakeForest(classes)

        n_trees = int(rng.integers(1, 8))
        n_samples = int(rng.integers(1, 40))
        y_true = np.array([lab[int(i)] for i in rng.integers(0, len(lab), size=n_samples)])
        tree_predictions = rng.integers(
            0, rf.n_classes_, size=(n_trees, n_samples)).astype(np.intp)

        state = IncrementalMetrics(tree_predictions, rf, y_true, task)
        assert state.metrics() == _oracle(y_true, tree_predictions, rf, task)

        for _ in range(6):
            undo_info = _mutate(rng, tree_predictions, rf.n_classes_)
            state.apply(tree_predictions, undo_info)
            assert state.metrics() == _oracle(y_true, tree_predictions, rf, task), (
                lab, classes, tree_predictions)


def test_revert_restores_votes_pred_and_confusion_byte_for_byte():
    """Rollback must be byte-identical, not merely numerically equal -- dtypes
    included, since a silent int32 -> int64 promotion of the vote matrix would
    grow with every rejected candidate.

    Reverts are interleaved with ACCEPTED steps so they happen from non-initial
    states: a rollback that only ever ran from the freshly constructed state
    would not exercise the token at all.
    """
    rng = np.random.default_rng(4242)

    for _ in range(120):
        lab = [0, 1, 2] if rng.random() < 0.5 else [-1, 1]
        task = 'app' if lab == [0, 1, 2] else 'ddos'
        rf = _FakeForest(lab)
        n_trees = int(rng.integers(1, 8))
        n_samples = int(rng.integers(1, 40))
        y_true = np.array([lab[int(i)] for i in rng.integers(0, len(lab), size=n_samples)])
        tree_predictions = rng.integers(
            0, rf.n_classes_, size=(n_trees, n_samples)).astype(np.intp)
        state = IncrementalMetrics(tree_predictions, rf, y_true, task)

        for _ in range(5):
            # An accepted step first, so the next rejected one rolls back to a
            # mutated state rather than to construction.
            state.apply(tree_predictions, _mutate(rng, tree_predictions, rf.n_classes_))

            votes_before = state.votes.copy()
            pred_before = state.pred_idx.copy()
            confusion_before = state.confusion.copy()
            metrics_before = state.metrics()
            dtypes_before = (state.votes.dtype, state.pred_idx.dtype,
                             state.confusion.dtype)

            undo_info = _mutate(rng, tree_predictions, rf.n_classes_)
            token = state.apply(tree_predictions, undo_info)
            # Undo the prediction block too -- exactly as undo_cache_update does
            # on the reject path, and independently of state.revert.
            for tree_idx, (idx, old) in undo_info['predictions'].items():
                tree_predictions[tree_idx, idx] = old
            state.revert(token)

            assert np.array_equal(state.votes, votes_before)
            assert np.array_equal(state.pred_idx, pred_before)
            assert np.array_equal(state.confusion, confusion_before)
            assert (state.votes.dtype, state.pred_idx.dtype,
                    state.confusion.dtype) == dtypes_before
            assert state.metrics() == metrics_before


def test_apply_returns_None_and_is_a_no_op_when_no_tree_vote_actually_flipped():
    """A threshold can move without any per-tree prediction changing -- the
    common case when the move only reshuffles samples between nodes that
    happen to predict the same class. apply must recognise that, return None,
    and leave every array untouched; revert(None) must then be a no-op rather
    than an unpacking error.
    """
    rf = _FakeForest([0, 1, 2])
    y_true = np.array([0, 1, 2, 0, 1, 2])
    tree_predictions = np.array([[0, 1, 2, 0, 1, 2],
                                 [0, 0, 2, 2, 1, 1],
                                 [1, 1, 2, 0, 0, 2]], dtype=np.intp)
    state = IncrementalMetrics(tree_predictions, rf, y_true, 'app')
    before = (state.votes.copy(), state.pred_idx.copy(), state.confusion.copy())
    metrics_before = state.metrics()

    # The sample indices really are re-predicted, but every tree re-predicts
    # them to the value it already held -- so `new != old` is empty.
    idx = np.array([1, 3, 4], dtype=np.intp)
    undo_info = {'predictions': {1: (idx, tree_predictions[1, idx].copy())},
                 'node_samples': {}}

    token = state.apply(tree_predictions, undo_info)

    assert token is None
    assert np.array_equal(state.votes, before[0])
    assert np.array_equal(state.pred_idx, before[1])
    assert np.array_equal(state.confusion, before[2])
    assert state.metrics() == metrics_before

    state.revert(token)  # must not raise
    assert state.metrics() == metrics_before


def test_a_sample_touched_by_several_trees_is_counted_once_per_tree():
    """The duplicate-index hazard. `undo_info` is keyed by tree, so the SAME
    sample index legitimately appears under several tree_idx entries -- the
    concatenated index array genuinely has duplicates. apply's
    searchsorted+bincount handles that by construction; plain fancy indexing
    (`votes[rows, new] += 1`) would apply only ONE of the duplicate updates
    and silently break the row sum.

    This is the test that fails if someone "optimises" that away.
    """
    rf = _FakeForest([0, 1, 2])
    n_trees, n_samples = 5, 4
    y_true = np.array([0, 1, 2, 1])
    tree_predictions = np.zeros((n_trees, n_samples), dtype=np.intp)
    state = IncrementalMetrics(tree_predictions, rf, y_true, 'app')

    # Sample 2 is touched by three different trees, all flipping 0 -> 1.
    idx = np.array([2], dtype=np.intp)
    undo_info = {'predictions': {t: (idx.copy(), np.array([0], dtype=np.intp))
                                 for t in (0, 1, 2)},
                 'node_samples': {}}
    for t in (0, 1, 2):
        tree_predictions[t, 2] = 1

    state.apply(tree_predictions, undo_info)

    assert np.array_equal(state.votes.sum(axis=1), np.full(n_samples, n_trees))
    assert np.array_equal(state.votes[2], np.array([2, 3, 0]))
    assert state.metrics() == _oracle(y_true, tree_predictions, rf, 'app')


def test_the_winner_is_recomputed_over_the_whole_row_not_just_the_moved_classes():
    """[3,3,1] -> [3,4,1]: a flip from class 2 to class 1 never touches the
    incumbent winner's own count (class 0 stays at 3), yet the winner changes
    from 0 to 1. Any shortcut that compares the incumbent only against the
    classes that moved, or that keeps the incumbent while it is still
    maximal, gets this wrong.
    """
    rf = _FakeForest([0, 1, 2])
    y_true = np.array([0])
    # 8 trees: three vote 0, three vote 1, two vote 2 -> votes [3, 3, 2].
    tree_predictions = np.array([[0], [0], [0], [1], [1], [1], [2], [2]],
                                dtype=np.intp)
    state = IncrementalMetrics(tree_predictions, rf, y_true, 'app')
    assert np.array_equal(state.votes[0], np.array([3, 3, 2]))
    assert state.pred_idx[0] == 0

    idx = np.array([0], dtype=np.intp)
    undo_info = {'predictions': {7: (idx, np.array([2], dtype=np.intp))},
                 'node_samples': {}}
    tree_predictions[7, 0] = 1

    state.apply(tree_predictions, undo_info)

    assert np.array_equal(state.votes[0], np.array([3, 4, 1]))
    assert state.pred_idx[0] == 1
    assert state.metrics() == _oracle(y_true, tree_predictions, rf, 'app')


def test_ties_break_to_the_smallest_class_index_after_an_incremental_update():
    """np.argmax returns the FIRST maximal index, which is the smallest class
    index, which is exactly switch_semantics.vote_winner -- the rule the
    generated vote_<task> table's const entries are built from. An
    incumbent-preserving shortcut diverges the moment a flip creates a tie
    with a smaller-indexed class, so drive a row into a tie through apply and
    check all three views agree.
    """
    from src.p4gen.switch_semantics import vote_winner

    rf = _FakeForest([0, 1, 2])
    y_true = np.array([2])
    # votes [1, 2, 3] -> winner 2. Flip one class-2 tree to class 0 -> [2,2,2].
    tree_predictions = np.array([[0], [1], [1], [2], [2], [2]], dtype=np.intp)
    state = IncrementalMetrics(tree_predictions, rf, y_true, 'app')
    assert state.pred_idx[0] == 2

    idx = np.array([0], dtype=np.intp)
    undo_info = {'predictions': {5: (idx, np.array([2], dtype=np.intp))},
                 'node_samples': {}}
    tree_predictions[5, 0] = 0

    state.apply(tree_predictions, undo_info)

    assert np.array_equal(state.votes[0], np.array([2, 2, 2]))
    from_scratch = np.bincount(tree_predictions[:, 0], minlength=rf.n_classes_)
    assert state.pred_idx[0] == int(np.argmax(from_scratch)) == 0
    assert state.pred_idx[0] == vote_winner(tree_predictions[:, 0].tolist(),
                                            rf.n_classes_)


def test_vote_rows_always_sum_to_the_tree_count():
    """Free invariant: every sample gets exactly one vote from every tree, so
    each row of the vote matrix sums to n_trees -- at construction, after any
    apply, and after any revert. A broken delta shows up here immediately.
    """
    rng = np.random.default_rng(97)

    rf = _FakeForest([-1, 1])
    n_trees, n_samples = 7, 25
    y_true = np.array([-1 if i % 2 else 1 for i in range(n_samples)])
    tree_predictions = rng.integers(
        0, rf.n_classes_, size=(n_trees, n_samples)).astype(np.intp)
    state = IncrementalMetrics(tree_predictions, rf, y_true, 'ddos')

    expected = np.full(n_samples, n_trees)
    assert np.array_equal(state.votes.sum(axis=1), expected)

    for _ in range(20):
        undo_info = _mutate(rng, tree_predictions, rf.n_classes_)
        token = state.apply(tree_predictions, undo_info)
        assert np.array_equal(state.votes.sum(axis=1), expected)
        state.revert(token)
        for tree_idx, (idx, old) in undo_info['predictions'].items():
            tree_predictions[tree_idx, idx] = old
        assert np.array_equal(state.votes.sum(axis=1), expected)


def test_both_label_spaces_work():
    """App [0, 1, 2] and DDoS [-1, 1]. The DDoS space is the regression risk:
    -1 is a valid label but an invalid raw array index, and numpy would
    silently wrap it to the last row/column rather than raise. Every lookup
    has to go through the universe position mapping.
    """
    cases = [
        ('app', [0, 1, 2], np.array([0, 1, 2, 0, 1, 2, 2, 0]),
         np.array([[0, 1, 2, 0, 1, 1, 2, 0],
                   [0, 1, 1, 2, 1, 2, 2, 1],
                   [1, 1, 2, 0, 0, 2, 2, 0]], dtype=np.intp)),
        ('ddos', [-1, 1], np.array([-1, 1, 1, -1, 1, -1, -1, 1]),
         np.array([[0, 1, 1, 0, 1, 1, 0, 0],
                   [1, 1, 0, 0, 1, 0, 0, 1],
                   [0, 0, 1, 1, 1, 0, 1, 1]], dtype=np.intp)),
    ]

    for task, lab, y_true, tree_predictions in cases:
        rf = _FakeForest(lab)
        state = IncrementalMetrics(tree_predictions, rf, y_true, task)
        assert state.metrics() == _oracle(y_true, tree_predictions, rf, task), task

        idx = np.array([0, 3, 7], dtype=np.intp)
        undo_info = {'predictions': {1: (idx, tree_predictions[1, idx].copy())},
                     'node_samples': {}}
        tree_predictions[1, idx] = (tree_predictions[1, idx] + 1) % rf.n_classes_

        state.apply(tree_predictions, undo_info)
        assert state.metrics() == _oracle(y_true, tree_predictions, rf, task), task
