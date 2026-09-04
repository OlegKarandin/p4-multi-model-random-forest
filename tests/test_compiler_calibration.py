"""Unit tests for scripts/compiler_calibration.py.

Pure-function tests only (sample construction, gap arithmetic, resume
logic, and run_one_row's F2 degrade path via monkeypatching) -- the real
compile is exercised by the plan's Task 4 gate, not by this suite (spec
§4)."""
import json
import os

import pandas as pd
import pytest

import scripts.compiler_calibration as cc
from scripts.compiler_calibration import (
    GROUPS,
    K_BANDS,
    STRATA_STAGE_DEPTHS,
    add_group_and_band,
    build_sample,
    k_band,
    stratum_row,
)


def test_k_band_low_at_and_below_5():
    assert k_band(0) == 'low'
    assert k_band(5) == 'low'


def test_k_band_high_at_and_above_13():
    assert k_band(13) == 'high'
    assert k_band(17) == 'high'


def test_k_band_none_in_the_middle():
    assert k_band(6) is None
    assert k_band(12) is None


def _synthetic_frame(rows):
    """rows: list of dicts with at least arm_slug, k, stage_depth. Fills in
    every other column build_sample/run_one_row might touch with a cheap
    placeholder, so tests can pass minimal per-row overrides."""
    base = {'M': 100, 'split': 10, 'best_params': '{}',
            'features_app': 'f1', 'features_ddos': 'f1', 'blocks': 10}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_add_group_and_band_maps_independent_and_joint_d000():
    frame = _synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'joint-d000', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'joint-d020', 'k': 2, 'stage_depth': 5},
    ])
    out = add_group_and_band(frame)
    assert list(out['group']) == ['independent', 'joint', None]


def test_stratum_row_raises_on_an_empty_stratum():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
    ]))
    with pytest.raises(ValueError, match='stage_depth=8'):
        stratum_row(frame, 'independent', 'low', 8)


def test_stratum_row_returns_the_lowest_split_on_a_tie():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5, 'split': 12},
        {'arm_slug': 'independent', 'k': 3, 'stage_depth': 5, 'split': 10},
    ]))
    row = stratum_row(frame, 'independent', 'low', 5)
    assert row['split'] == 10


def test_build_sample_raise_mode_raises_on_first_missing_cell():
    # Only ONE of the 24 target cells is populated.
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
    ]))
    with pytest.raises(ValueError):
        build_sample(frame, strata=STRATA_STAGE_DEPTHS, on_missing='raise')


def test_build_sample_skip_mode_reports_missing_cells_and_returns_the_rest():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 6},
    ]))
    rows, missing = build_sample(
        frame, strata=(5, 6), groups=('independent',), k_bands=('low',),
        on_missing='skip')
    assert len(rows) == 2
    assert missing == []
    rows2, missing2 = build_sample(
        frame, strata=(5, 6, 7), groups=('independent',), k_bands=('low',),
        on_missing='skip')
    assert len(rows2) == 2
    assert missing2 == [('independent', 'low', 7)]


def test_build_sample_row_id_is_unique_per_cell():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 14, 'stage_depth': 5},
    ]))
    rows, _ = build_sample(frame, strata=(5,), on_missing='skip')
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids))


def test_build_sample_invalid_on_missing_raises():
    frame = add_group_and_band(_synthetic_frame([]))
    with pytest.raises(ValueError, match='on_missing'):
        build_sample(frame, on_missing='bogus')


import math

from scripts.compiler_calibration import gap_blocks, gap_stages, is_void


def test_gap_stages_computes_real_minus_predicted():
    row = {'stage_depth': 6, 'stages_real': 9}
    assert gap_stages(row) == 3


def test_gap_stages_none_when_stages_real_is_none():
    assert gap_stages({'stage_depth': 6, 'stages_real': None}) is None


def test_gap_stages_none_when_stages_real_is_nan():
    assert gap_stages({'stage_depth': 6, 'stages_real': float('nan')}) is None


def test_gap_blocks_computes_real_minus_predicted():
    row = {'blocks': 4, 'tcam_real': 4}
    assert gap_blocks(row) == 0


def test_gap_blocks_none_when_tcam_real_is_none():
    assert gap_blocks({'blocks': 4, 'tcam_real': None}) is None


def test_is_void_false_when_compile_errors_is_zero():
    assert is_void({'compile_errors': 0}) is False


def test_is_void_true_when_compile_errors_is_a_positive_int():
    assert is_void({'compile_errors': 2}) is True


def test_is_void_true_for_the_unavailable_reason_string():
    assert is_void({'compile_errors': "no FEATURE_REGISTER_CATALOG entry for 'x'"}) is True


def test_is_void_false_when_compile_errors_is_none_or_nan():
    assert is_void({'compile_errors': None}) is False
    assert is_void({'compile_errors': float('nan')}) is False


def test_is_void_true_int_string_survives_csv_round_trip_with_a_void_row(tmp_path):
    """Regression test: pandas reads an entire mixed int/string column back as
    str once any row has a non-numeric value. A numeric string like '0' must
    still be recognised as a real, non-void error count after a CSV round-trip.

    Scenario: one row with compile_errors=0 (real success) and one row with
    compile_errors="unavailable reason" (F2 degrade path). Write to CSV, read
    back with pd.read_csv, and verify is_void classifies both correctly."""
    import io

    # Create a small DataFrame with one successful row (0 errors) and one void row
    df = pd.DataFrame({
        'row_id': ['success', 'void_row'],
        'compile_errors': [0, "no FEATURE_REGISTER_CATALOG entry for 'x'"],
    })

    # Write to CSV
    csv_file = tmp_path / 'test.csv'
    df.to_csv(csv_file, index=False)

    # Read back -- pandas will coerce the entire compile_errors column to str
    df_read = pd.read_csv(csv_file)

    # Verify that compile_errors is now entirely string type
    assert df_read['compile_errors'].dtype == object  # pandas uses object for str columns

    # Extract the rows and test is_void on each
    success_row = df_read[df_read['row_id'] == 'success'].iloc[0].to_dict()
    void_row = df_read[df_read['row_id'] == 'void_row'].iloc[0].to_dict()

    # The numeric string '0' should NOT be void (it's a real 0-error count)
    assert is_void(success_row) is False
    # The non-numeric reason string should be void
    assert is_void(void_row) is True


import scripts.compiler_calibration as cc
from scripts.compiler_calibration import already_done, run_one_row


def test_already_done_empty_when_out_path_missing(tmp_path):
    assert already_done(str(tmp_path / 'nope.csv')) == set()


def test_already_done_reads_recorded_row_ids(tmp_path):
    out = tmp_path / 'out.csv'
    pd.DataFrame([{'row_id': 'independent_low_sd5', 'stage_depth': 5},
                 {'row_id': 'joint_high_sd6', 'stage_depth': 6}]).to_csv(out, index=False)
    assert already_done(str(out)) == {'independent_low_sd5', 'joint_high_sd6'}


def test_run_one_row_records_the_unavailable_reason_on_a_generate_p4_code_valueerror(tmp_path, monkeypatch):
    archived_row = pd.Series({
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10,
        'best_params': json.dumps({'n_estimators_A': 3, 'n_estimators_B': 3}),
        'features_app': 'f1', 'features_ddos': 'f1',
    })

    class _FakeUsage:
        stage_depth, blocks, stages = 5, 10, 3
        range_entries = ternary_entries = register_depth = register_count = 0

    monkeypatch.setattr(cc, 'refit_pair', lambda row, data: (
        object(), object(), None, None, [0], [0]))
    monkeypatch.setattr(cc, 'multi_model_memory_evaluation', lambda *a, **kw: _FakeUsage())
    monkeypatch.setattr(cc, 'get_feature_intervals', lambda *a, **kw: {})

    def _raise(*a, **kw):
        raise ValueError("no FEATURE_REGISTER_CATALOG entry for 'f1'")
    monkeypatch.setattr(cc, 'generate_P4_code', _raise)

    row = run_one_row('independent_low_sd5', 'independent', archived_row, None, str(tmp_path))
    assert row['stages_real'] is None
    assert row['compile_errors'] == "no FEATURE_REGISTER_CATALOG entry for 'f1'"


def test_run_one_row_creates_compiles_dir_before_calling_compile_p4(tmp_path, monkeypatch):
    """Regression test: compile_p4 requires its output_dir's immediate
    parent to already exist (p4c creates only the final path segment).
    Nothing else creates output_root/compiles, so run_one_row itself must,
    before calling compile_p4 -- otherwise every real compile fails
    forever, even on retry."""
    archived_row = pd.Series({
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10,
        'best_params': json.dumps({'n_estimators_A': 3, 'n_estimators_B': 3}),
        'features_app': 'f1', 'features_ddos': 'f1',
    })

    class _FakeUsage:
        stage_depth, blocks, stages = 5, 10, 3
        range_entries = ternary_entries = register_depth = register_count = 0

    class _FakeCompileResult:
        stages = tcam = sram = map_ram = 0
        errors = 0

    monkeypatch.setattr(cc, 'refit_pair', lambda row, data: (
        object(), object(), None, None, [0], [0]))
    monkeypatch.setattr(cc, 'multi_model_memory_evaluation', lambda *a, **kw: _FakeUsage())
    monkeypatch.setattr(cc, 'get_feature_intervals', lambda *a, **kw: {})
    monkeypatch.setattr(cc, 'generate_P4_code',
                         lambda *a, **kw: str(tmp_path / 'p4_src' / 'fake.p4'))

    def _fake_compile_p4(written_path, output_dir_arg):
        assert os.path.isdir(os.path.dirname(output_dir_arg))
        return _FakeCompileResult()
    monkeypatch.setattr(cc, 'compile_p4', _fake_compile_p4)

    output_root = str(tmp_path)
    row = run_one_row('independent_low_sd5', 'independent', archived_row, None, output_root)

    assert os.path.isdir(os.path.join(output_root, 'compiles'))
    assert row['stages_real'] == 0
    assert row['compile_errors'] == 0


from scripts.compiler_calibration import collect


def test_collect_skips_rows_already_recorded_at_out(tmp_path, monkeypatch):
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 6},
    ]))
    monkeypatch.setattr(cc, 'load_backup', lambda campaign_dir: frame)
    monkeypatch.setattr(cc, 'load_campaign_data', lambda: None)

    calls = []
    def _fake_run_one_row(row_id, group, archived_row, data, output_root):
        calls.append(row_id)
        return {'row_id': row_id, 'group': group, 'stage_depth': 5,
                'stages_real': 6, 'blocks': 1, 'tcam_real': 1,
                'compile_errors': 0}
    monkeypatch.setattr(cc, 'run_one_row', _fake_run_one_row)

    out = str(tmp_path / 'out.csv')
    root = str(tmp_path / 'root')
    pd.DataFrame([{'row_id': 'independent_low_sd5', 'stage_depth': 5,
                   'stages_real': 6, 'blocks': 1, 'tcam_real': 1,
                   'compile_errors': 0, 'group': 'independent'}]).to_csv(out, index=False)

    result_frame, missing = collect('unused', out, root, strata=(5, 6),
                                    groups=('independent',), k_bands=('low',))
    assert calls == ['independent_low_sd6']
    assert set(result_frame['row_id']) == {'independent_low_sd5', 'independent_low_sd6'}
