"""Exact confusion-matrix metric functions, plus the incremental state that
maintains them across threshold moves -- P3b plan Task T2 (parts a and b).

`align_rf_thresholds` (threshold_alignment.py) repeatedly proposes a
threshold move, measures (accuracy, weighted_f1) on a validation set, and
accepts or rejects the move by comparing metrics before/after. Until T2b that
measurement went through `src.p4gen.evaluation.accuracy_metrics`,
which calls sklearn's `accuracy_score` + `f1_score` -- measured at 4022us per
call at n=4000, against 256us for the prediction those calls measure. Almost
all of that is sklearn's fixed per-call overhead (`validate_params`,
`_check_targets`, `unique_labels`, `LabelEncoder`, `multilabel_confusion_matrix`),
not O(n) work (p3b-design-reference.md's cost table). This module replaces
the measurement with the same numbers computed directly from a confusion
matrix -- ~14.6x end to end even recomputed from scratch every candidate.

The module is in two layers. The four functions below are PURE: no mutable
state, no knowledge of the alignment loop, and they are the from-scratch
oracle the layer above them is tested against. `IncrementalMetrics` is that
layer -- one instance per model, owning the vote matrix, the per-sample
winner and the confusion matrix, and updating all three in O(#changed) as
`align_rf_thresholds` applies and rolls back candidate moves. Together they
buy a further ~8x on top of the ~14.6x above, because the per-candidate cost
stops scaling with the validation set at all.

The bar is exact bit equality with sklearn's `accuracy_score` and
`f1_score(..., average='weighted')`, not approximate agreement: these
numbers decide accept/reject, and the accept/reject trajectory is
path-dependent, so even a one-ULP disagreement can flip a decision and
cascade into a different final model. Every formula here is traced against
sklearn 1.6.1's source (the version installed in this project's env) --
see p3b-design-reference.md sections 1-2 for the full trace and a
20 000-draw verification with 0 mismatches.
"""
import numpy as np


def label_universe(lab, y_true, classes):
    """The label set sklearn's `multilabel_confusion_matrix` actually counts
    over. It does
    `labels = concat([labels, setdiff1d(present_labels, labels)])`,
    bincounts fp/fn contributions over ALL samples, and only THEN retains the
    first `len(lab)` rows/cols. Mirror that: `lab` first, in `lab`'s given
    order (this fixes which universe rows/cols `weighted_f1_from_confusion`'s
    `n_labels` slice reads), followed by every other label that appears in
    `y_true` or that the model can predict (`classes`), sorted.

    `classes` rather than the live `y_pred`: on this project every
    prediction is `rf.classes_[argmax(votes)]`
    (threshold_alignment.compute_ensemble_prediction), so the predicted-label
    alphabet is exactly `rf.classes_`, which is fixed for a whole alignment
    run while `y_true` and the predictions vary candidate to candidate.
    Building the universe from `classes` instead of `y_pred` is what lets
    Task 2b compute it ONCE per run rather than rebuild it every candidate.

    Why this matters at all (Ruling P3b-1): sklearn counts a label's fp/fn
    over ALL samples, so `pred_sum[c]` for `c` in `lab` must include samples
    whose TRUE label is outside `lab`, and `true_sum[c]` must include samples
    PREDICTED outside `lab`. A matrix spanning only `lab` would silently drop
    those contributions. On this project's two tasks `lab` already covers
    every label the data or the model can produce, so `universe == list(lab)`
    in practice today -- but nothing here assumes that, so it keeps holding
    if that ever stops being true.
    """
    lab = list(lab)
    extra = (set(np.unique(np.asarray(y_true)).tolist()) | set(classes)) - set(lab)
    return lab + sorted(extra)


def confusion_from_predictions(y_uni, pred_uni, k):
    """From-scratch confusion-matrix crosstab: `confusion[t, p]` = count of
    samples whose true universe index is `t` and predicted universe index is
    `p`. `y_uni` and `pred_uni` must already be mapped into universe indices
    in `[0, k)` -- e.g. via `label_universe`'s returned list.

    This is the ORACLE Task 2b's incremental confusion-matrix maintenance
    (updated in O(#changed) on each threshold move) is checked against, so it
    counts the plain way -- one bincount over a flattened `row * k + col`
    index -- rather than sharing any machinery with that incremental path.
    """
    y_uni = np.asarray(y_uni, dtype=np.intp)
    pred_uni = np.asarray(pred_uni, dtype=np.intp)
    return np.bincount(y_uni * k + pred_uni, minlength=k * k).reshape(k, k)


def weighted_f1_from_confusion(confusion, n_labels):
    """Reproduces sklearn 1.6.1's
    `f1_score(y_true, y_pred, labels=lab, average='weighted')` bit for bit.
    `confusion` must be indexed in `label_universe` order; `n_labels` is
    `len(lab)` -- only the first `n_labels` rows/cols of `confusion` are the
    labels being scored (`label_universe`'s trailing "extra" labels exist
    only to make their fp/fn contributions to THOSE rows/cols correct).

    Traced through sklearn 1.6.1 source (matches the version installed in
    this project's env -- see p3b-design-reference.md section 2):

    - `f1_score` -> `fbeta_score(beta=1.0)` -> `precision_recall_fscore_support`
      (`_classification.py:1872-1880`): `denom = beta2*true_sum + pred_sum`
      (`beta2 == 1.0`, so `denom == true_sum + pred_sum`);
      `f_score = _prf_divide((1+beta2)*tp_sum, denom, ..., zero_division)`.
      There is NO separate precision/recall combination step in 1.6 -- the
      older `2PR/(P+R)` form is gone from this sklearn version, so
      replicating that form would silently be wrong.
    - `_prf_divide` (`:1531`): `mask = denominator == 0`;
      `denominator[mask] = 1`; `result = numerator / denominator`;
      `result[mask] = zero_division_value`, and `_check_zero_division('warn')
      == 0.0` -- this is the `where=denom > 0` branch below (a class with no
      true samples and no predictions, or a support/prediction mismatch that
      zeroes the denominator, scores 0.0 rather than raising or NaN-ing).
    - `_nanaverage` (`utils/extmath.py:1207`): no NaNs are possible here (the
      mask above already replaced them with 0.0), so this is plain
      `numpy.average(a, weights)`; `except ZeroDivisionError: return
      _average(a)` when the weights (here `true_sum`) sum to zero -- the
      `true_sum.sum() == 0` branch below, taken when none of `lab`'s labels
      have any true samples at all.
    - `2.0 * tp` and `true_sum + pred_sum` both promote int64 -> float64
      exactly for counts under 2**53 (this project's tables are many orders
      of magnitude below that), so computing the numerator/denominator in
      integer counts and letting `/` promote gives bit-identical results to
      sklearn's own float64 path.

    ONE DECLARED BEHAVIOURAL DIFFERENCE from sklearn: `_warn_prf` emits
    `UndefinedMetricWarning` when the zero-division mask fires (a class with
    no true samples and no predictions, or one only on one side). This
    function never warns -- the returned VALUE is identical bit for bit,
    only sklearn's warning is absent. Checked (`grep -rn
    "UndefinedMetricWarning|filterwarnings" src/ scripts/ tests/`): nothing
    in this repo asserts on or filters that specific warning -- the one hit
    outside this module is `feature_selection.py`'s blanket
    `warnings.filterwarnings('ignore')` (silences everything, not
    conditioned on this warning), and a test comment explaining why a
    fixture is shaped to AVOID triggering it, not asserting its presence.
    """
    tp = np.diagonal(confusion)[:n_labels].astype(np.int64)
    true_sum = confusion.sum(axis=1)[:n_labels]
    pred_sum = confusion.sum(axis=0)[:n_labels]

    denom = (true_sum + pred_sum).astype(np.float64)
    f = np.divide(2.0 * tp, denom, out=np.zeros_like(denom), where=denom > 0)

    if true_sum.sum() == 0:
        return float(np.average(f))
    return float(np.average(f, weights=true_sum))


def accuracy_from_confusion(confusion, n_samples):
    """`accuracy_score(y_true, y_pred)` is `float(np.mean(y_true == y_pred))`:
    an exactly-integral float64 sum divided by `n`. `trace(confusion)` counts
    exactly the samples whose true and predicted universe index agree --
    identical to that sum for every `n_samples < 2**53` (this project's
    tables are many orders of magnitude below that bound).

    `n_samples == 0` is out of scope (Ruling P3b-1): asserted, not defined.
    """
    if n_samples <= 0:
        raise ValueError("accuracy_from_confusion: n_samples must be > 0")
    return float(np.trace(confusion)) / n_samples


# The label sets `evaluation.accuracy_metrics` scores each task over, mirrored
# here so IncrementalMetrics takes the same `task` string its callers already
# pass. Duplicated rather than imported because accuracy_metrics hardcodes them
# inline in an if/elif; the equality is pinned by every equivalence test in
# tests/test_incremental_metrics.py, which compares against accuracy_metrics
# itself for both tasks.
TASK_LABELS = {'app': [0, 1, 2], 'ddos': [-1, 1]}


class IncrementalMetrics:
    """(accuracy, weighted_f1) for one model on one validation set, maintained
    in O(#changed) across the threshold moves `align_rf_thresholds` proposes.

    WHY A CLASS, rather than the free functions above. The state is five
    coupled arrays -- `votes (n, C)`, `pred_idx (n,)`, `confusion (k, k)`, plus
    the static `y_uni` and `class_to_uni` maps -- with two invariants that must
    hold BETWEEN calls, not merely inside one:

        pred_idx  == argmax(votes, axis=1)
        confusion == crosstab(y_uni, class_to_uni[pred_idx])

    Threading those through free functions would add ~10 parameters at three
    call sites times two models inside a loop body that is already ~120 lines,
    and the branch most likely to update one array and not the other is the
    `delta_rel is None` early-accept branch that skips the metric path
    entirely. One object per model keeps the coupling in one place.

    ORDERING CONTRACT -- `align_rf_thresholds` must respect all three:

    1. `apply` runs AFTER `update_cache_for_modifications`: it reads the NEW
       per-tree predictions out of `tree_predictions` and the OLD ones out of
       `undo_info['predictions']`, so both have to be in their post-update
       state.
    2. `apply` runs BEFORE `undo_cache_update`, for the same reason -- once the
       cache is rolled back, `tree_predictions` no longer holds the new values.
    3. `revert(token)` undoes exactly the `apply` that produced `token`, and
       only from the state that `apply` left behind. It is independent of
       `undo_cache_update` (it restores its own arrays from a stored copy, not
       from `tree_predictions`), so the two may run in either order -- but a
       token must not be held across a second `apply`.

    `n_samples == 0` is out of scope, as it is for `accuracy_from_confusion`.
    """

    def __init__(self, tree_predictions, rf, y_true, task):
        _, n_samples = tree_predictions.shape
        if n_samples <= 0:
            raise ValueError("IncrementalMetrics: n_samples must be > 0")

        # Unconditional: y_val reaches align_rf_thresholds as splits.py's
        # `y[idx_val_align]` today (a 1-D ndarray, so this is a no-op), but the
        # positional indexing below would silently misbehave on a pandas Series
        # with a non-positional index, and this costs nothing.
        y_true = np.asarray(y_true)

        if task not in TASK_LABELS:
            raise ValueError(
                "IncrementalMetrics: unknown task {!r}; expected one of {}".format(
                    task, sorted(TASK_LABELS)))
        lab = TASK_LABELS[task]
        self.n_labels = len(lab)
        self.n_samples = n_samples

        # Static for the whole alignment run: neither rf.classes_ nor y_true
        # changes under threshold mutation, only which class each tree votes
        # for. So the universe and both index maps are built once here rather
        # than per candidate -- that is most of what makes the metric path
        # cheap (see incremental_metrics' module docstring).
        universe = label_universe(lab, y_true, rf.classes_)
        self.k = len(universe)
        pos = {label: i for i, label in enumerate(universe)}
        self.y_uni = np.array([pos[v] for v in y_true.tolist()], dtype=np.intp)
        self.class_to_uni = np.array([pos[c] for c in rf.classes_.tolist()],
                                     dtype=np.intp)

        n_classes = rf.n_classes_
        # Same sample-major offset trick as compute_ensemble_prediction: give
        # every sample its own length-n_classes slot so the entire
        # (n_trees, n_samples) block is counted in one bincount pass.
        offsets = np.arange(n_samples, dtype=np.intp) * n_classes
        self.votes = np.bincount(
            (offsets[None, :] + tree_predictions).ravel(),
            minlength=n_samples * n_classes
        ).reshape(n_samples, n_classes).astype(np.int32)

        self.pred_idx = np.argmax(self.votes, axis=1).astype(np.intp)
        self.confusion = confusion_from_predictions(
            self.y_uni, self.class_to_uni[self.pred_idx], self.k)

    def metrics(self):
        """(accuracy, weighted_f1) -- the same pair, in the same order, that
        `evaluation.accuracy_metrics(y_true, y_pred, task)` returns, and
        bit-identical to it."""
        return (accuracy_from_confusion(self.confusion, self.n_samples),
                weighted_f1_from_confusion(self.confusion, self.n_labels))

    def apply(self, tree_predictions, undo_info):
        """Fold one candidate's per-tree prediction changes into the votes,
        the winners and the confusion matrix.

        Returns an opaque token to hand to `revert`, or None when no per-tree
        vote actually flipped (a threshold can move samples between nodes that
        predict the same class, in which case there is nothing to update and
        nothing to roll back).
        """
        rows_all, old_all, new_all = [], [], []
        for tree_idx, (sample_indices, old) in undo_info['predictions'].items():
            new = tree_predictions[tree_idx, sample_indices]
            flipped = new != old
            if not flipped.any():
                continue
            rows_all.append(sample_indices[flipped])
            old_all.append(old[flipped])
            new_all.append(new[flipped])

        if not rows_all:
            return None

        changed_rows = np.concatenate(rows_all)
        old_classes = np.concatenate(old_all)
        new_classes = np.concatenate(new_all)

        # The affected SAMPLES, sorted and deduplicated.
        rows = np.unique(changed_rows)
        # Fancy indexing already materialises a copy, so the rollback block is
        # free. See revert for why rollback is by stored copy, not by inverse
        # delta.
        old_block = self.votes[rows]
        token = (rows, old_block, self.pred_idx[rows].copy(), self.confusion.copy())

        # searchsorted + bincount rather than np.add.at / plain fancy indexing.
        # `undo_info` is keyed by TREE, so one sample can appear under several
        # tree_idx entries and `changed_rows` genuinely has duplicates:
        # `votes[rows, new] += 1` would apply only one of them. bincount sums
        # duplicates by construction and leans on no uniqueness assumption
        # about undo_info (update_cache_for_modifications happens to np.unique
        # per tree today, but that is an invariant of a different function).
        # np.add.at would also be correct, but is unbuffered and ~10x slower.
        position = np.searchsorted(rows, changed_rows)
        m, n_classes = len(rows), self.votes.shape[1]
        slot = position * n_classes
        delta = (np.bincount(slot + new_classes, minlength=m * n_classes)
                 - np.bincount(slot + old_classes, minlength=m * n_classes)
                 ).reshape(m, n_classes)
        # int32 votes + int64 delta promotes to int64, and the assignment
        # same-kind-downcasts back. Safe: every count is bounded by n_trees.
        self.votes[rows] = old_block + delta

        # Re-argmax the WHOLE row, never a shortcut against the classes that
        # moved. The winner can change while the incumbent winner's own count
        # is untouched -- [3,3,2] with a tree flipping 2 -> 1 becomes [3,4,1],
        # moving the winner from 0 to 1. And ties must break to the smallest
        # class index (np.argmax's first-maximal rule, which is exactly
        # switch_semantics.vote_winner and the generated vote_<task> table);
        # "keep the incumbent while it is still maximal" diverges the moment a
        # flip creates a tie with a smaller-indexed class.
        #
        # Restricting the re-argmax to `rows` is complete because argmax is a
        # pure function of the row, so a row with no flipped tree vote cannot
        # change winner. That is the correctness licence for the whole design.
        new_pred = np.argmax(self.votes[rows], axis=1).astype(np.intp)
        old_pred = self.pred_idx[rows]      # fancy index -> a copy, so the
        self.pred_idx[rows] = new_pred      # write below cannot clobber it

        # Only rows whose WINNER moved contribute a net change here; rows where
        # it did not cancel exactly between the two bincounts.
        true_uni, k = self.y_uni[rows], self.k
        self.confusion += (
            np.bincount(true_uni * k + self.class_to_uni[new_pred], minlength=k * k)
            - np.bincount(true_uni * k + self.class_to_uni[old_pred], minlength=k * k)
        ).reshape(k, k)

        return token

    def revert(self, token):
        """Undo the `apply` that returned `token`, byte for byte.

        Restores from the stored pre-change copies rather than re-applying the
        delta with the signs flipped. The vote block was already copied for
        free by `apply`'s fancy index, and `confusion` is (k, k) int64 with k
        at most 5 on this project's two tasks -- 200 bytes. So the copy is
        cheap, and it makes the rollback exact BY
        DEFINITION rather than by an algebraic argument that would have to be
        re-proved every time the delta changes. A rejected candidate must leave
        the state bit-identical: alignment's accept/reject trajectory is
        path-dependent, so drift here compounds silently across candidates.
        """
        if token is None:
            return
        rows, votes_block, pred_block, confusion = token
        self.votes[rows] = votes_block
        self.pred_idx[rows] = pred_block
        self.confusion = confusion
