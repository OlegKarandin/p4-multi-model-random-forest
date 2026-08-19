"""Spec B.5: 55/15/15/15, with val_align and val_select disjoint."""
import numpy as np
import pytest

from src.training.splits import TaskSplits, make_task_splits


def _data(n=4000, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 500, size=(n, 5)).astype(float)
    y = np.array([c % n_classes for c in range(n)])
    return X, y


def test_the_four_buckets_partition_the_dataset():
    X, y = _data()

    s = make_task_splits(X, y, random_state=42)

    all_idx = np.concatenate([s.idx_train, s.idx_val_align, s.idx_val_select, s.idx_test])
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)
    assert set(all_idx.tolist()) == set(range(len(y)))


def test_val_align_and_val_select_are_disjoint():
    """The whole reason for splitting validation in half: alignment fits the
    threshold grid to val_align, and permutation_importance then measures
    importance on val_select with the aligned model. Sharing one set inflates
    the apparent importance of exactly the features alignment touched."""
    X, y = _data()

    s = make_task_splits(X, y, random_state=42)

    assert set(s.idx_val_align.tolist()).isdisjoint(s.idx_val_select.tolist())


def test_test_set_is_disjoint_from_everything_that_is_selected_on():
    X, y = _data()

    s = make_task_splits(X, y, random_state=42)
    test = set(s.idx_test.tolist())

    assert test.isdisjoint(s.idx_train.tolist())
    assert test.isdisjoint(s.idx_val_align.tolist())
    assert test.isdisjoint(s.idx_val_select.tolist())


def test_proportions_are_55_15_15_15():
    X, y = _data(n=27222)

    s = make_task_splits(X, y, random_state=42)
    n = len(y)

    assert len(s.idx_train) / n == pytest.approx(0.55, abs=0.005)
    assert len(s.idx_val_align) / n == pytest.approx(0.15, abs=0.005)
    assert len(s.idx_val_select) / n == pytest.approx(0.15, abs=0.005)
    assert len(s.idx_test) / n == pytest.approx(0.15, abs=0.005)


def test_training_set_is_larger_than_the_old_effective_one():
    """Today: 70% train, of which 3-fold CV uses 2/3 per fold -> ~46.7%
    effective. New: 55% used whole. That is the point of dropping CV."""
    X, y = _data(n=27222)

    s = make_task_splits(X, y, random_state=42)

    assert len(s.idx_train) > 0.70 * 2 / 3 * len(y)


def test_every_bucket_is_stratified():
    X, y = _data(n=9000, n_classes=3)

    s = make_task_splits(X, y, random_state=42)

    overall = np.bincount(y) / len(y)
    for bucket in (s.y_train, s.y_val_align, s.y_val_select, s.y_test):
        assert np.allclose(np.bincount(bucket, minlength=3) / len(bucket), overall, atol=0.02)


def test_binary_labels_with_minus_one_are_supported():
    """DDoS labels are [-1, 1], not [0, 1] -- np.bincount would reject them, so
    the implementation must not assume non-negative classes."""
    rng = np.random.default_rng(1)
    X = rng.integers(0, 500, size=(2000, 5)).astype(float)
    y = np.array([-1, 1] * 1000)

    s = make_task_splits(X, y, random_state=42)

    for bucket in (s.y_train, s.y_val_align, s.y_val_select, s.y_test):
        assert set(np.unique(bucket).tolist()) == {-1, 1}


def test_the_same_seed_gives_the_same_split():
    X, y = _data()

    first = make_task_splits(X, y, random_state=42)
    second = make_task_splits(X, y, random_state=42)

    assert np.array_equal(first.idx_train, second.idx_train)
    assert np.array_equal(first.idx_val_align, second.idx_val_align)


def test_different_seeds_give_different_splits():
    """Split-level replication is what controls the variance the delta sweep's
    feature-path divergence introduces, so the seed must actually matter."""
    X, y = _data()

    first = make_task_splits(X, y, random_state=42)
    second = make_task_splits(X, y, random_state=43)

    assert not np.array_equal(first.idx_train, second.idx_train)


def test_the_materialised_arrays_match_their_indices():
    X, y = _data()

    s = make_task_splits(X, y, random_state=42)

    assert np.array_equal(s.X_train, X[s.idx_train])
    assert np.array_equal(s.y_val_align, y[s.idx_val_align])
    assert np.array_equal(s.X_test, X[s.idx_test])
