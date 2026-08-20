"""Exact confusion-matrix metric functions -- P3b plan Task T2, part (a) only
(`task-2a-brief.md`, Ruling P3b-2).

`align_rf_thresholds` (threshold_alignment.py) repeatedly proposes a
threshold move, measures (accuracy, weighted_f1) on a validation set, and
accepts or rejects the move by comparing metrics before/after. That
measurement currently goes through `src.p4gen.evaluation.accuracy_metrics`,
which calls sklearn's `accuracy_score` + `f1_score` -- measured at 4022us per
call at n=4000, against 256us for the prediction those calls measure. Almost
all of that is sklearn's fixed per-call overhead (`validate_params`,
`_check_targets`, `unique_labels`, `LabelEncoder`, `multilabel_confusion_matrix`),
not O(n) work (p3b-design-reference.md's cost table). This module replaces
the measurement with the same numbers computed directly from a confusion
matrix -- ~14.6x end to end even recomputed from scratch every candidate.

These are PURE FUNCTIONS ONLY: no mutable state, nothing wired into the
alignment loop. Task 2b adds the incremental state class that maintains the
confusion matrix in O(#changed) as moves are accepted/rejected, and does the
wiring; this module does not.

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
    assert n_samples > 0, "accuracy_from_confusion: n_samples must be > 0"
    return float(np.trace(confusion)) / n_samples
