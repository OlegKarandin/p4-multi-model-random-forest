"""scripts/capacity_ceiling.py's own measurement (`collect`) rather than the
reporting half already covered by tests/test_figures.py's
`_synthetic_ceiling_csv` (which builds a CSV in the script's schema directly,
bypassing `collect()` entirely).

Task 21: `collect()`'s row-building used to set `joint_within_limit` /
`disjoint_within_limit` from the codeword length alone, even though
`measure()` can ALSO return None for a completely separate reason -- a
per-stage crossbar byte-width rejection (F3, `CrossbarKeyTooWide`),
independent of codeword length. This is a regression test for that: it
drives the real `collect()` loop (with datasets/splits/fitting/feature-
interval and evaluator calls monkeypatched to a fast synthetic scenario) and
asserts a codeword-length-fine-but-crossbar-rejected cell is NOT reported as
within_limit.
"""
import types

import numpy as np
import pandas as pd

import scripts.capacity_ceiling as capacity_ceiling
import src.main as main_module
from src.p4gen.evaluation import CrossbarKeyTooWide


def test_collect_marks_crossbar_rejected_cell_as_not_within_limit(monkeypatch):
    """A cell whose codeword length is comfortably under MAX_CODEWORD_LENGTH
    but whose table the crossbar rejects (F3) must NOT be reported as
    within_limit -- `measure()` correctly returns None for that cell (it
    catches CrossbarKeyTooWide), and the *_within_limit flag has to reflect
    that None, not just re-check the codeword length that was never the
    problem.
    """
    # Shrink the grid to exactly one cell / one split / one corner, so the
    # real collect() loop body runs (and both encodings' measure() calls
    # happen) exactly once, keeping this fast.
    monkeypatch.setattr(capacity_ceiling, 'N_TREES_GRID', (1,))
    monkeypatch.setattr(capacity_ceiling, 'MAX_DEPTH_GRID', (2,))
    monkeypatch.setattr(capacity_ceiling, 'SPLIT_INDICES', (10,))
    monkeypatch.setattr(capacity_ceiling, 'CORNERS', (capacity_ceiling.PRUNED,))

    # Fake datasets: collect() only needs a Label column and a length out of
    # these before handing them to remove_correlated_features_both_datasets,
    # which is mocked below.
    fake_df = pd.DataFrame({'Label': [0, 1, 0]})
    monkeypatch.setattr(capacity_ceiling, 'read_app_dataset', lambda *a, **k: fake_df)
    monkeypatch.setattr(capacity_ceiling, 'read_DDOS_dataset', lambda *a, **k: fake_df)
    # collect() does `from src.main import remove_correlated_features_both_datasets`
    # INSIDE the function body, so patching the attribute on src.main (rather
    # than on capacity_ceiling) is what that late-bound import will pick up.
    monkeypatch.setattr(
        main_module, 'remove_correlated_features_both_datasets',
        lambda df_app, df_ddos, threshold=0.95: (
            np.zeros((3, 2)), np.zeros((3, 2)), ['f0']))

    fake_split = types.SimpleNamespace(
        X_train=np.zeros((3, 2)), y_train=np.array([0, 1, 0]))
    monkeypatch.setattr(capacity_ceiling, 'make_task_splits', lambda X, y, seed: fake_split)

    # fit() would train a real RandomForestClassifier; a sentinel is enough
    # since get_feature_intervals/get_joint_feature_intervals and
    # multi_model_memory_evaluation (mocked below) never inspect it.
    monkeypatch.setattr(capacity_ceiling, 'fit', lambda *a, **k: object())

    # Fixed, tiny feature intervals: codeword length is deliberately WAY
    # under MAX_CODEWORD_LENGTH (512) under both encodings -- 5 bits.
    small_intervals = {'f0': [0, 1, 2, 3, 4, 5]}
    monkeypatch.setattr(capacity_ceiling, 'get_joint_feature_intervals',
                         lambda *a, **k: small_intervals)
    monkeypatch.setattr(capacity_ceiling, 'get_feature_intervals',
                         lambda *a, **k: small_intervals)

    # The crossbar (F3) rejects this cell's table under BOTH encodings, even
    # though the codeword length is fine -- exactly the scenario the bug
    # mishandles: measure() correctly returns None each time, but (pre-fix)
    # the *_within_limit flag looked only at codeword length and missed it.
    def fake_evaluation(clf_app, clf_ddos, features_app, features_ddos, encoding,
                         use_default_action_discount=False):
        raise CrossbarKeyTooWide('crossbar key too wide', 999)
    monkeypatch.setattr(capacity_ceiling, 'multi_model_memory_evaluation', fake_evaluation)

    frame = capacity_ceiling.collect()

    assert len(frame) == 1
    row = frame.iloc[0]

    # The codeword length was never the problem.
    assert row.joint_codeword_length <= capacity_ceiling.MAX_CODEWORD_LENGTH
    assert row.disjoint_codeword_length <= capacity_ceiling.MAX_CODEWORD_LENGTH
    # measure() returned None for both encodings (crossbar rejection), so no
    # block/stage counts exist for this row.
    assert pd.isna(row.joint_stages)
    assert pd.isna(row.disjoint_stages)

    # The bug: within_limit was computed from codeword length alone, so it
    # would say True here even though the table is uncompilable.
    assert row.joint_within_limit == False, (
        'codeword length was within the limit, but the crossbar rejected '
        'the table (measure() returned None) -- joint_within_limit must be '
        'False, not just codeword-length-derived')
    assert row.disjoint_within_limit == False, (
        'codeword length was within the limit, but the crossbar rejected '
        'the table (measure() returned None) -- disjoint_within_limit must '
        'be False, not just codeword-length-derived')
