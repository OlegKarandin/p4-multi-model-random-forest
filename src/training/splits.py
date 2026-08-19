"""The 55/15/15/15 data split (spec B.5).

Two changes from the old 70/15/15:

1. Validation is 30% split into two disjoint halves. X_val previously had two
   consumers that interacted: align_rf_thresholds fits the threshold grid to
   it, then permutation_importance measures importance on the same set with
   the aligned model -- inflating the apparent importance of exactly the
   features alignment touched. val_align now serves alignment and nothing
   else; val_select serves the Optuna objective and permutation_importance.

2. Training drops to 55% but the EFFECTIVE training set grows ~18%, because
   B.2 stops discarding a third of it to 3-fold CV (70% x 2/3 = 46.7%).

X_test is touched by none of this, so nothing here can inflate a reported
metric. The residual concern is adaptive overfitting of the validation halves,
which degrades choice quality rather than reported accuracy -- and which
biases AGAINST the joint arm, since alignment runs only there.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

TEST_FRACTION = 0.15
VAL_FRACTION = 0.30


@dataclass(frozen=True)
class TaskSplits:
    """One task's four buckets, plus the row indices that produced them.

    The indices are carried so disjointness is directly assertable in tests and
    so a split is reproducible from an artifact rather than only from a seed.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val_align: np.ndarray
    y_val_align: np.ndarray
    X_val_select: np.ndarray
    y_val_select: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val_align: np.ndarray
    idx_val_select: np.ndarray
    idx_test: np.ndarray


def make_task_splits(X, y, random_state):
    """Stratified 55% train / 15% val_align / 15% val_select / 15% test.

    Cut three times rather than once so each cut can be stratified on the
    labels that survive the previous one. The second fraction is
    VAL_FRACTION / (1 - TEST_FRACTION) because it is taken from what remains
    after the test cut.
    """
    indices = np.arange(len(y))

    idx_temp, idx_test = train_test_split(
        indices, test_size=TEST_FRACTION, random_state=random_state, stratify=y)

    idx_train, idx_val = train_test_split(
        idx_temp, test_size=VAL_FRACTION / (1.0 - TEST_FRACTION),
        random_state=random_state, stratify=y[idx_temp])

    idx_val_align, idx_val_select = train_test_split(
        idx_val, test_size=0.5, random_state=random_state, stratify=y[idx_val])

    return TaskSplits(
        X_train=X[idx_train], y_train=y[idx_train],
        X_val_align=X[idx_val_align], y_val_align=y[idx_val_align],
        X_val_select=X[idx_val_select], y_val_select=y[idx_val_select],
        X_test=X[idx_test], y_test=y[idx_test],
        idx_train=idx_train, idx_val_align=idx_val_align,
        idx_val_select=idx_val_select, idx_test=idx_test)
