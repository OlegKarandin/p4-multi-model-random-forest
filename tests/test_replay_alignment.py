import numpy as np
import pandas as pd
import pytest

import scripts.replay_alignment as ra


def _frame():
    return pd.DataFrame([
        {'arm': 'joint', 'arm_slug': 'joint-d000', 'M': 25, 'split': 10, 'k': 17,
         'features_app': 'a;b', 'features_ddos': 'a;b', 'blocks': 20},
        {'arm': 'joint', 'arm_slug': 'joint-d000', 'M': 25, 'split': 11, 'k': 17,
         'features_app': 'a;b', 'features_ddos': 'a;b', 'blocks': 21},
        {'arm': 'joint', 'arm_slug': 'joint-d020', 'M': 25, 'split': 10, 'k': 9,
         'features_app': 'a', 'features_ddos': 'a', 'blocks': 12},
        {'arm': 'independent', 'arm_slug': 'independent', 'M': 25, 'split': 10,
         'k': 17, 'features_app': 'a;b', 'features_ddos': 'a;b', 'blocks': 30},
    ])


def test_select_rows_filters_on_every_axis():
    out = ra.select_rows(_frame(), arms=['joint-d000'], m_values=[25],
                         k_values=[17], n_splits=1)
    assert len(out) == 1
    assert out.iloc[0]['split'] == 10


def test_select_rows_takes_the_lowest_split_ids_for_reproducibility():
    """Sampling must be deterministic: the same invocation must select the same
    rows, so the ladder's policies are compared on one fixed pair set."""
    out = ra.select_rows(_frame(), arms=['joint-d000'], m_values=[25],
                         k_values=[17], n_splits=2)
    assert sorted(out['split']) == [10, 11]


def test_select_rows_never_returns_an_unaligned_arm():
    """Alignment runs on joint arms only, and joint-off skips the call
    entirely, so neither can source a model pair for an alignment replay."""
    out = ra.select_rows(_frame(), arms=None, m_values=None, k_values=None,
                         n_splits=10)
    assert set(out['arm_slug']) == {'joint-d000', 'joint-d020'}


def test_column_indices_maps_names_back_to_columns():
    assert ra.column_indices(['a', 'b', 'c'], 'c;a') == [2, 0]


def test_column_indices_rejects_a_name_the_dataset_does_not_have():
    """A silent mismatch would replay a different feature set than the row
    records, which is exactly the confound this harness exists to remove."""
    with pytest.raises(ValueError, match='not in the loaded dataset'):
        ra.column_indices(['a', 'b'], 'a;zzz')


def test_split_random_state_is_the_campaign_formula():
    """feature_selection.py:640 -- the CSV's `split` column holds split_idx
    (feature_selection.py:333, ids running from 10), and the split seed is
    random_state + split_idx with random_state fixed at 42 (main.py:380)."""
    assert ra.split_random_state(10) == 52
    assert ra.split_random_state(24) == 66
