from unittest import mock

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


def _replay_row_fixture():
    """A row/refit_pair pairing that never touches real models or data --
    refit_pair is mocked out, so replay_row only needs opaque sentinels it
    threads through to (also mocked) run_one_policy."""
    row = {'arm_slug': 'joint-d000', 'M': 25, 'split': 10, 'k': 17,
           'features_app': 'a;b', 'features_ddos': 'a;b', 'blocks': 20}
    refit_result = ('model_app', 'model_ddos', 'app', 'ddos',
                    'cols_app', 'cols_ddos')
    return row, refit_result


def test_replay_row_skips_a_failing_swept_cell_but_keeps_the_rest(capsys):
    """One (policy, overlap_threshold) cell raising (e.g. CrossbarKeyTooWide
    at a combination the campaign never validated) must not lose the other
    cells already computed for this row."""
    row, refit_result = _replay_row_fixture()

    def fake_run_one_policy(models, app, ddos, cols_app, cols_ddos,
                            names_app, names_ddos, policy, delta_rel,
                            overlap_threshold):
        if overlap_threshold == 0.25:
            # Stand-in for a real evaluation.CrossbarKeyTooWide -- any
            # exception at this call site must be caught, so the fixture
            # doesn't need the real exception class.
            raise RuntimeError('table key is 67 crossbar bytes')
        return {'policy': policy, 'overlap_threshold': overlap_threshold,
                'blocks': 10}

    with mock.patch.object(ra, 'refit_pair', return_value=refit_result), \
         mock.patch.object(ra, 'run_one_policy', side_effect=fake_run_one_policy):
        results = ra.replay_row(row, data=None, policies=['legacy'],
                                overlap_thresholds=[0.25, 0.5],
                                ladder_delta=0.20, verify=False)

    assert len(results) == 1
    assert results[0]['overlap_threshold'] == 0.5
    out = capsys.readouterr().out
    assert 'SKIPPED (error)' in out
    assert 'VERIFY FAILED' not in out
    assert 'joint-d000' in out and 'overlap=0.25' in out
    assert 'RuntimeError' in out


def test_replay_row_reports_a_failing_verify_call_distinctly(capsys):
    """A failure reproducing the row's OWN recorded settings is a broken
    determinism claim, not an ordinary infeasible sweep cell -- it must be
    logged with a visually distinct prefix and must not crash replay_row,
    and the (empty) sweep loop that follows must still run cleanly."""
    row, refit_result = _replay_row_fixture()

    with mock.patch.object(ra, 'refit_pair', return_value=refit_result), \
         mock.patch.object(ra, 'run_one_policy',
                           side_effect=RuntimeError('boom')):
        results = ra.replay_row(row, data=None, policies=[],
                                overlap_thresholds=[], ladder_delta=0.20,
                                verify=True)

    assert results == []
    out = capsys.readouterr().out
    assert 'VERIFY FAILED (error)' in out
    assert 'SKIPPED (error)' not in out
    assert 'joint-d000' in out
    assert 'RuntimeError: boom' in out


def test_replay_row_verify_failure_does_not_block_later_swept_cells(capsys):
    """A broken verify call for a row must not prevent that same row's
    ordinary swept cells from being attempted."""
    row, refit_result = _replay_row_fixture()

    # replay_row always issues the verify call before the sweep loop, so the
    # first call is the (failing) verify call and the second is the swept
    # cell -- no need to distinguish by policy/overlap value.
    calls = []

    def side_effect(models, app, ddos, cols_app, cols_ddos, names_app,
                    names_ddos, policy, delta_rel, overlap_threshold):
        calls.append((policy, overlap_threshold))
        if len(calls) == 1:
            raise RuntimeError('verify broke')
        return {'policy': policy, 'overlap_threshold': overlap_threshold,
                'blocks': 10}

    with mock.patch.object(ra, 'refit_pair', return_value=refit_result), \
         mock.patch.object(ra, 'run_one_policy', side_effect=side_effect):
        results = ra.replay_row(row, data=None, policies=['legacy'],
                                overlap_thresholds=[0.5], ladder_delta=0.20,
                                verify=True)

    assert len(calls) == 2
    assert len(results) == 1
    assert results[0]['overlap_threshold'] == 0.5
    out = capsys.readouterr().out
    assert 'VERIFY FAILED (error)' in out
