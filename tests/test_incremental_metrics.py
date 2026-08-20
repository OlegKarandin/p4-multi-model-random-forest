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
