"""Unit tests for scripts/compiler_calibration.py.

Pure-function tests only (sample construction, gap arithmetic, resume
logic, and run_one_row's F2 degrade path via monkeypatching) -- the real
compile is exercised by the plan's Task 4 gate, not by this suite (spec
§4)."""
import collections
import json
import os

import pandas as pd
import pytest

import scripts.compiler_calibration as cc
from src.p4gen import evaluation as ev
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
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10, 'stage_depth': 5,
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
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10, 'stage_depth': 5,
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


def test_run_one_row_passes_generate_p4_code_a_p4_dir_with_a_trailing_separator(tmp_path, monkeypatch):
    """Regression test: generate_P4_code (build_p4_script.py) builds its
    final path via raw string concatenation -- output_dir + output_filename,
    NOT os.path.join -- an established project convention (feature_selection.py's
    real caller passes its output dir with a trailing slash for the same
    reason). If run_one_row's p4_dir lacks a trailing separator, the
    concatenation glues the directory name straight onto the filename (e.g.
    'p4_srcindependent_low_sd5.p4' landing directly under output_root instead
    of inside p4_src/), and a real p4c compile then can't find the file.

    This fake mimics the real concatenation exactly, so it fails against the
    old code (no trailing separator) and passes once p4_dir carries one."""
    archived_row = pd.Series({
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10, 'stage_depth': 5,
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

    received = {}

    def _fake_generate_P4_code(*a, output_dir, output_filename, **kw):
        # Mirrors build_p4_script.py:1623's real concatenation exactly.
        received['output_dir'] = output_dir
        return output_dir + output_filename

    def _fake_compile_p4(written_path, output_dir_arg):
        received['written_path'] = written_path
        return _FakeCompileResult()

    monkeypatch.setattr(cc, 'generate_P4_code', _fake_generate_P4_code)
    monkeypatch.setattr(cc, 'compile_p4', _fake_compile_p4)

    output_root = str(tmp_path)
    row_id = 'independent_low_sd5'
    row = run_one_row(row_id, 'independent', archived_row, None, output_root)

    # The fake was handed a directory with a trailing separator, so its
    # own output_dir + output_filename concatenation (mirroring the real
    # build_p4_script.py:1623) lands the file INSIDE p4_src/ ...
    expected_p4_dir = os.path.join(output_root, 'p4_src') + os.sep
    assert received['output_dir'] == expected_p4_dir
    assert os.path.dirname(received['written_path']) == os.path.dirname(expected_p4_dir)
    assert os.path.basename(received['written_path']) == row_id + '.p4'
    # ... instead of the garbled 'p4_srcindependent_low_sd5.p4' the old
    # (no-trailing-separator) code produced directly under output_root.
    garbled_filename = 'p4_src' + row_id + '.p4'
    assert os.path.basename(received['written_path']) != garbled_filename
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

    result_frame, missing, failed = collect('unused', out, root, strata=(5, 6),
                                            groups=('independent',), k_bands=('low',))
    assert calls == ['independent_low_sd6']
    assert set(result_frame['row_id']) == {'independent_low_sd5', 'independent_low_sd6'}
    assert failed == []


def test_collect_isolates_a_row_whose_run_one_row_raises(tmp_path, monkeypatch):
    """A row whose run_one_row raises must NOT land in the output CSV (it
    would be permanently skipped via already_done on the next call), must
    be reported back in `failed`, and must not block the other rows in the
    same batch from being recorded normally."""
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 6},
    ]))
    monkeypatch.setattr(cc, 'load_backup', lambda campaign_dir: frame)
    monkeypatch.setattr(cc, 'load_campaign_data', lambda: None)

    def _fake_run_one_row(row_id, group, archived_row, data, output_root):
        if row_id == 'independent_low_sd5':
            raise RuntimeError('simulated p4c timeout')
        return {'row_id': row_id, 'group': group, 'stage_depth': 6,
                'stages_real': 6, 'blocks': 1, 'tcam_real': 1,
                'compile_errors': 0}
    monkeypatch.setattr(cc, 'run_one_row', _fake_run_one_row)

    out = str(tmp_path / 'out.csv')
    root = str(tmp_path / 'root')

    result_frame, missing, failed = collect(
        'unused', out, root, strata=(5, 6),
        groups=('independent',), k_bands=('low',))

    # (c) the failing row is reported back
    assert failed == [('independent_low_sd5', 'RuntimeError: simulated p4c timeout')]
    # (a) NOT written to the output CSV
    assert 'independent_low_sd5' not in set(result_frame['row_id'])
    # (d) the other row in the same batch is still recorded normally
    assert 'independent_low_sd6' in set(result_frame['row_id'])
    # (b) NOT in already_done on the next call, so it's retried
    assert 'independent_low_sd5' not in already_done(out)
    assert 'independent_low_sd6' in already_done(out)


def test_run_one_row_records_stage_depth_archived_and_warns_on_mismatch(tmp_path, monkeypatch, capsys):
    archived_row = pd.Series({
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10, 'stage_depth': 12,
        'best_params': json.dumps({'n_estimators_A': 3, 'n_estimators_B': 3}),
        'features_app': 'f1', 'features_ddos': 'f1',
    })

    class _FakeUsage:
        stage_depth, blocks, stages = 6, 10, 3  # deliberately != archived (12)
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
    monkeypatch.setattr(cc, 'compile_p4', lambda *a, **kw: _FakeCompileResult())

    row = run_one_row('independent_low_sd12', 'independent', archived_row, None, str(tmp_path))

    assert row['stage_depth_archived'] == 12
    assert row['stage_depth'] == 6
    captured = capsys.readouterr()
    assert 'independent_low_sd12' in captured.out
    assert '12' in captured.out and '6' in captured.out
    assert 'WARNING' in captured.out


def test_run_one_row_no_warning_when_archived_matches_recomputed(tmp_path, monkeypatch, capsys):
    archived_row = pd.Series({
        'arm_slug': 'independent', 'M': 100, 'k': 2, 'split': 10, 'stage_depth': 5,
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
    monkeypatch.setattr(cc, 'compile_p4', lambda *a, **kw: _FakeCompileResult())

    row = run_one_row('independent_low_sd5', 'independent', archived_row, None, str(tmp_path))

    assert row['stage_depth_archived'] == 5
    captured = capsys.readouterr()
    assert 'WARNING' not in captured.out


from scripts.compiler_calibration import report


def _report_frame(rows):
    return pd.DataFrame(rows)


def test_report_v1_not_established_when_zero_measured_rows(capsys):
    # Two rows, both void (compile_errors is a positive error count) -- so
    # `measured` (Finding 1's fix) is empty even though `frame` is not.
    frame = _report_frame([
        {'row_id': 'r1', 'compile_errors': 2, 'stage_depth': 5, 'stages_real': None,
         'blocks': 1, 'tcam_real': None, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
        {'row_id': 'r2', 'compile_errors': 3, 'stage_depth': 6, 'stages_real': None,
         'blocks': 1, 'tcam_real': None, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
    ])
    report(frame, [])
    out = capsys.readouterr().out
    assert 'NOT ESTABLISHED' in out
    assert 'OK -- gap_stages' not in out


def test_report_unmeasured_row_excluded_from_v1_v2_v5_counts(capsys):
    # r1 is a real measured row (gap_stages=0, gap_blocks=0). r2 has
    # is_void()==False (compile_errors is missing/NaN) but no real
    # stages_real/tcam_real -- Finding 1's "unmeasured" case. It must not
    # inflate V1/V2/V5's n, and must not be mistaken for a gap_blocks!=0
    # violation (None != 0 bug).
    frame = _report_frame([
        {'row_id': 'r1', 'compile_errors': 0, 'stage_depth': 6, 'stages_real': 6,
         'blocks': 4, 'tcam_real': 4, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
        {'row_id': 'r2', 'compile_errors': None, 'stage_depth': 5, 'stages_real': None,
         'blocks': 3, 'tcam_real': None, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
    ])
    report(frame, [])
    out = capsys.readouterr().out
    assert 'OK -- gap_stages >= 0 on all 1 compiled, non-void, measured rows' in out
    assert 'OK -- gap_blocks == 0 on all 1 rows' in out
    assert 'r2' in out  # reported under V6 as unmeasured, not silently dropped
    assert 'gap_blocks != 0' not in out


def test_report_stages_only_degrade_row_excluded_from_v5_nonzero_list(capsys):
    # r2 is the real degrade shape seen on compiler_calibration_v6.csv
    # (independent_high_sd12, joint_high_sd8): the compiler reports a real
    # stages_real (table_summary.log gets written even when the backend
    # never allocates resources), but tcam_real is NaN (no mau.resources.log
    # "Allocated Resource Usage" section) -- unlike test_report_unmeasured_
    # row_excluded_from_v1_v2_v5_counts's r2, which has NEITHER. gap_stages
    # is real (0) so r2 correctly counts in V1/V2; gap_blocks must stay None
    # (no tcam_real to compare against), not get compared as `None != 0`
    # (pandas: NaN != 0 is True), which mis-flagged both rows as block
    # divergences in the fresh compiler_calibration_v6 run.
    frame = _report_frame([
        {'row_id': 'r1', 'compile_errors': 0, 'stage_depth': 6, 'stages_real': 6,
         'blocks': 4, 'tcam_real': 4, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
        {'row_id': 'r2', 'compile_errors': 0, 'stage_depth': 13, 'stages_real': 13,
         'blocks': 55, 'tcam_real': None, 'group': 'joint', 'k': 13,
         'n_estimators_A': 3, 'n_estimators_B': 3},
    ])
    report(frame, [])
    out = capsys.readouterr().out
    assert 'OK -- gap_stages >= 0 on all 2 compiled, non-void, measured rows' in out
    assert 'OK -- gap_blocks == 0 on all 1 rows' in out
    assert 'gap_blocks != 0' not in out


def test_report_accepts_two_positional_args_for_backward_compatibility():
    """report(frame, missing) with no `failed` arg must still work --
    existing callers (and this suite's earlier tests) don't pass it."""
    frame = _report_frame([
        {'row_id': 'r1', 'compile_errors': 0, 'stage_depth': 6, 'stages_real': 6,
         'blocks': 4, 'tcam_real': 4, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
    ])
    report(frame, [])  # should not raise


def test_report_prints_failed_rows_under_v6(capsys):
    frame = _report_frame([
        {'row_id': 'r1', 'compile_errors': 0, 'stage_depth': 6, 'stages_real': 6,
         'blocks': 4, 'tcam_real': 4, 'group': 'independent', 'k': 2,
         'n_estimators_A': 3, 'n_estimators_B': 3},
    ])
    report(frame, [], [('independent_low_sd5', 'RuntimeError: simulated p4c timeout')])
    out = capsys.readouterr().out
    assert 'independent_low_sd5' in out
    assert 'simulated p4c timeout' in out


# ---------------------------------------------------------------------------
# Stage-depth replay against the real compiles (2026-09-05).
#
# replay_stage_depth feeds the estimator's packer the REAL per-table facts
# read out of a row's generated P4 and its committed compile logs -- which
# crossbar fields each table keys on, each field's width, each table's
# physical block count, each range table's readiness level -- so the only
# thing under test is stage placement, not the block model or the refit.
# The artifacts are gitignored, so these skip unless the study has been run.
# ---------------------------------------------------------------------------

_ARTIFACTS = os.path.join('results', 'compiler_calibration')

# (predicted stage_depth, committed compiler stage count) per row, measured
# with the corrected crossbar-sharing + readiness-origin model. Hardcoded per
# row rather than checked as an aggregate so a packer change that moves any
# single row is caught instead of averaged away.
_REPLAY_EXPECTED = {
    'independent_high_sd10': (12, 12),
    'independent_high_sd6': (11, 11),
    'independent_high_sd7': (11, 11),
    'independent_high_sd8': (11, 12),
    'independent_low_sd10': (11, 11),
    # Exact since the 2026-09-27 any-order fit rule (reviews/
    # final_model_check_2026-09-27.md section 1b): the crowded-stage rule
    # still charges the ddos trees +1 in the 40 + 20 = 60-byte stage p4c
    # shared for free, but fits() now only requires SOME ordering of the
    # stage's keys to pack column-wise, and one does (ddos key first, no
    # margin; app key second, +1). Requiring EVERY ordering to fit used to
    # reject that placement and push the design to 13 stages -- see the
    # golden fixture's known_findings 'crowded_stage_rule_2026_09_25' and
    # 'any_order_fit_rule_2026_09_27'.
    'independent_low_sd12': (12, 12),
    'independent_low_sd5': (8, 8),
    'independent_low_sd6': (8, 11),
    'independent_low_sd7': (9, 10),
    'independent_low_sd8': (9, 12),
    'joint_high_sd6': (11, 11),
    'joint_high_sd7': (11, 12),
    'joint_high_sd8': (12, 12),
    'joint_low_sd10': (10, 12),
    'joint_low_sd12': (11, 12),
    'joint_low_sd5': (8, 8),
    'joint_low_sd6': (8, 9),
    'joint_low_sd7': (9, 10),
}

def _row_features(row_id):
    """The row's real selected features, read back off its generated P4: the
    raw feature behind each range table's meta.<f>_val key, deduplicated
    (under 'independent' both models' tables can name the same feature, and
    the register behind it is still emitted once)."""
    tables, _, _ = cc._p4_table_keys(
        os.path.join(_ARTIFACTS, 'p4_src', row_id + '.p4'))
    features = []
    for name, keys in tables.items():
        if name.startswith('table_') and keys:
            feature = keys[0][:-len('_val')]
            if feature not in features:
                features.append(feature)
    return features


_no_artifacts = pytest.mark.skipif(
    not os.path.isdir(os.path.join(_ARTIFACTS, 'compiles')),
    reason='needs results/compiler_calibration/ (gitignored; run collect() first)')


@_no_artifacts
@pytest.mark.parametrize('row_id', sorted(_REPLAY_EXPECTED))
def test_replayed_stage_depth_matches_the_pinned_calibration_value(row_id):
    assert cc.replay_stage_depth(row_id, _ARTIFACTS) == _REPLAY_EXPECTED[row_id]


@_no_artifacts
def test_replayed_stage_depth_still_under_predicts_by_at_most_three():
    # Successive corrections against this sample: mean 2.84 (original) ->
    # 1.56 (crossbar field sharing + readiness origin + vote epilogue) ->
    # 1.00 (the stateful-ALU register schedule) -> 0.94 (Mechanism B, the
    # gated register sub-block) -> 0.78 (Mechanism C, the 2x12 TCAM column
    # geometry). The error is NOT closed and what remains is in the UNSAFE
    # direction: 9 of 18 rows still predict fewer stages than the compiler
    # uses. Every one of those nine is Mechanism A -- PHV container conflicts,
    # worth up to 3 stages on these archived compiles, which predate the
    # @pa_solitary fix that removes it from the generated P4. Pinned so the
    # residual cannot silently grow back.
    residuals = [real - predicted
                 for predicted, real in (cc.replay_stage_depth(row_id, _ARTIFACTS)
                                          for row_id in _REPLAY_EXPECTED)]
    assert max(residuals) <= 3
    # independent_low_sd12 used to be the one deliberate exception here, over
    # by 1 under the crowded-stage rule alone (see _REPLAY_EXPECTED); the
    # 2026-09-27 any-order fit rule closed it to exact, so no row may
    # over-count a committed placement any more.
    over = {row_id: real - predicted for row_id, (predicted, real) in
            ((row_id, cc.replay_stage_depth(row_id, _ARTIFACTS))
             for row_id in _REPLAY_EXPECTED) if real < predicted}
    assert over == {}
    assert sum(abs(r) for r in residuals) / len(residuals) <= 0.84
    assert sum(r == 0 for r in residuals) >= 8


@_no_artifacts
@pytest.mark.parametrize('row_id', sorted(_REPLAY_EXPECTED))
def test_the_column_geometry_violates_no_committed_placement(row_id):
    # Mechanism C is a TIGHTENING of the packer, so the thing that could go
    # wrong is over-counting: a rule that forbids a stage the compiler
    # actually built would turn a real design infeasible. Checked directly --
    # every stage of every pool of every row packs into 2 columns of 12 with
    # each table's blocks inside one column. No exceptions in 18 rows.
    #
    # This is also where the old "refutation" of the column rule died. It read
    # independent_low_sd10 as fitting 3 tables where 2*floor(12/w) allows 2,
    # but that shortcut needs one uniform w and the stage in question holds
    # three 2-BLOCK tables. Judged as a packing rather than a quotient,
    # nothing here refutes anything.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    blocks = cc._committed_blocks(logs)
    stages = cc.committed_table_stages(logs)

    per_stage = collections.defaultdict(list)
    for name, count in blocks.items():
        if name.startswith('table_') or name.startswith('get_classification_tree'):
            per_stage[(stages[name], name.startswith('table_'))].append(count)

    assert per_stage
    for (stage, _is_range), widths in sorted(per_stage.items()):
        assert ev.fits_two_columns(widths), (row_id, stage, sorted(widths))


@_no_artifacts
@pytest.mark.parametrize('row_id', sorted(_REPLAY_EXPECTED))
def test_register_schedule_reproduces_the_compilers_own_register_depth(row_id):
    # The direct evidence for evaluation.METER_ALUS_PER_STAGE, independent of
    # anything the packer then does with the levels: schedule the row's real
    # feature set and compare the last register's stage against the one the
    # compiler committed to. Exact on every row. Chain depth alone reports 5
    # on each of the six k>=13 rows where the compiler needs 7.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    real = cc.committed_register_stages(logs)
    features = _row_features(row_id)

    placed = ev.register_stage_schedule(features)

    assert set(placed) == set(real)
    assert max(placed.values()) == max(real.values())
    # ...and the reason it lands there: no stage ever runs more than four.
    by_stage = collections.Counter(real.values())
    assert max(by_stage.values()) <= ev.METER_ALUS_PER_STAGE


@_no_artifacts
@pytest.mark.parametrize('row_id', sorted(_REPLAY_EXPECTED))
def test_predicted_interior_stages_match_the_range_pools_real_holes(row_id):
    # Mechanism B's whole content, checked directly against the compiler
    # rather than through the stage count it produces. A stage the placer
    # spends fully INSIDE a gated register block can hold no table from the
    # outer sequence, so it shows up as a hole in the range pool's committed
    # occupancy. Measured across all 18 rows: 5 have such a hole, 13 have
    # none, and the schedule's interior stages agree on every one -- so the
    # model reproduces both which rows lose a stage and which index it is,
    # with no per-row constant anywhere.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    tables, _, _ = cc._p4_table_keys(
        os.path.join(_ARTIFACTS, 'p4_src', row_id + '.p4'))
    committed = cc.committed_table_stages(logs)

    occupied = sorted({committed[name] for name in tables
                       if name.startswith('table_') and name in committed})
    holes = frozenset(stage for stage in range(min(occupied), max(occupied))
                      if stage not in occupied)

    assert ev.gated_block_interior_stages(_row_features(row_id)) == holes


# ---------------------------------------------------------------------------
# Item 16 (reviews/open_issues.md), resolved 2026-09-05: the +1 residual on
# independent_high_sd8 and joint_high_sd7 is Mechanism A -- PHV container write
# conflicts in the RANGE pool -- not readiness-level accuracy and not a new
# mechanism. Pinned here because the attribution was wrong once already.
# ---------------------------------------------------------------------------

_ITEM_16_ROWS = ('independent_high_sd8', 'joint_high_sd7')


@_no_artifacts
@pytest.mark.parametrize('row_id', _ITEM_16_ROWS)
def test_these_rows_never_ran_the_nocc_try_counterfactual(row_id):
    # The premise that ruled Mechanism A out on these two rows was "their
    # NOCC_TRY delta is 0". It is 0 because p4c only ever ran ONE placement
    # round on them, so there is no container-conflicts-disabled placement to
    # difference against -- absence of the round, not absence of the effect.
    # Rows where the delta IS evidence (independent_low_sd6 and friends) have
    # a NOCC_TRY round; these do not.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    assert cc.placement_round_states(logs) == ['INITIAL']


@_no_artifacts
@pytest.mark.parametrize('row_id', _ITEM_16_ROWS)
def test_the_extra_range_stage_holds_only_phv_advanced_tables(row_id):
    # The whole +1: the model's range pool ends one stage before the
    # compiler's, and every table in the compiler's extra stage is one p4c
    # logged verbatim as advanced there by an action dependency "due to PHV
    # allocation". independent_high_sd8's stage 9 holds exactly
    # table_13_ddos_bwd_iat_min and table_0_app_bwd_iat_max; joint_high_sd7's
    # holds exactly table_2_bwd_packet_length_max. Nothing else lands there,
    # so nothing else has to be explained.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    committed = cc.committed_table_stages(logs)
    range_stages = {name: stage for name, stage in committed.items()
                    if name.startswith('table_')}
    last = max(range_stages.values())
    on_last = {name for name, stage in range_stages.items() if stage == last}

    advanced = cc.phv_advanced_tables(logs)

    assert on_last
    assert on_last <= set(advanced)
    assert all(advanced[name] == last for name in on_last)


@_no_artifacts
@pytest.mark.parametrize('row_id', _ITEM_16_ROWS)
def test_readiness_levels_are_not_what_costs_these_rows_their_stage(row_id):
    # The hypothesis item 16 filed and this kills: that readiness_levels_for
    # calls a range table ready a stage before the compiler does, letting the
    # eager packer fill a stage the compiler could not. Substituting a perfect
    # oracle -- levels read off the compiler's OWN committed register
    # placement, one stage past each feature's last register -- moves the
    # predicted depth on neither row (and on none of the other 16 either).
    # Per-table the model is already right: over all 18 rows the committed
    # stage equals the predicted level 96 times, exceeds it 67, and is BELOW
    # it 11 times, so the error is symmetric tie-break noise inside a level,
    # not a systematic offset.
    logs = os.path.join(_ARTIFACTS, 'compiles', row_id, 'pipe', 'logs')
    features = _row_features(row_id)
    real = cc.committed_register_stages(logs)

    oracle = {}
    for feature in features:
        entry = ev.FEATURE_REGISTER_CATALOG.get(ev.normalise_feature_name(feature))
        if entry is None or not entry['registers']:
            oracle[feature] = ev.feature_readiness_level(feature)
        else:
            oracle[feature] = max(real[r['name']] for r in entry['registers']) + 1

    modelled = dict(zip(features, ev.readiness_levels_for(features)))
    assert oracle != modelled          # the oracle really is a different input
    assert (cc.replay_stage_depth(row_id, _ARTIFACTS, readiness_levels=oracle)
            == _REPLAY_EXPECTED[row_id])


@_no_artifacts
def test_replay_rejects_a_row_the_backend_never_allocated():
    # independent_high_sd12's placement needs 13 stages against Tofino's 12,
    # so p4c's assembler refused it ("tofino supports up to 12 stages, using
    # 13") and mau.resources.log has no allocation at all. There is nothing
    # to replay, and silently treating it as a zero-resource program would
    # turn an infeasible design into a cheap-looking one.
    with pytest.raises(ValueError):
        cc.replay_stage_depth('independent_high_sd12', _ARTIFACTS)


# ---------------------------------------------------------------------------
# Mechanism G (2026-09-06): the current-generator rows.
#
# The 18 rows above predate the @pa_solitary generator fix and still pay
# Mechanism A's 1-3 stage PHV tax, so they can only ever pin the residual.
# These 11 were produced by TODAY's generator, which is what the model is
# supposed to predict, and on them it is exact everywhere -- including
# independent_low_sd9, the last divergence the study had open, closed by the
# version-block charge (evaluation.crossbar_stages_needed's key_field_bits).
# ---------------------------------------------------------------------------

_CURRENT_ARTIFACTS = {
    'independent_low_sd9': os.path.join('results', 'compiler_calibration_extra'),
    'independent_low_sd11': os.path.join('results', 'compiler_calibration_extra'),
    'joint_low_sd9': os.path.join('results', 'compiler_calibration_extra'),
    'joint_low_sd11': os.path.join('results', 'compiler_calibration_extra'),
    'joint_high_sd9': os.path.join('results', 'compiler_calibration_extra'),
    'independent_high_sd6': os.path.join('results', 'compiler_calibration_v5'),
    'independent_high_sd7': os.path.join('results', 'compiler_calibration_v5'),
    'independent_high_sd8': os.path.join('results', 'compiler_calibration_v5'),
    'independent_high_sd10': os.path.join('results', 'compiler_calibration_v5'),
    'joint_high_sd6': os.path.join('results', 'compiler_calibration_v5'),
    'joint_high_sd7': os.path.join('results', 'compiler_calibration_v5'),
}


@pytest.mark.parametrize('row_id', sorted(_CURRENT_ARTIFACTS))
def test_the_model_is_exact_on_every_current_generator_row(row_id):
    root = _CURRENT_ARTIFACTS[row_id]
    if not os.path.isdir(os.path.join(root, 'compiles', row_id)):
        pytest.skip('needs %s (gitignored; run collect() first)' % root)
    predicted, real = cc.replay_stage_depth(row_id, root)
    # Exact, not merely safe. An inequality here would let the group-offset
    # penalty drift into over-prediction unnoticed, and never over-predicting
    # is the property the 12-stage feasibility gate is built on.
    assert predicted == real


def test_independent_low_sd9_costs_the_stage_the_version_charge_buys():
    # The row the charge exists for. Its app trees key 179 + 204 bits (49
    # crossbar bytes, 9 blocks) and its ddos trees 37 + 49 bits (12 bytes, 3
    # blocks). 2 app + 2 ddos is 24 blocks and packs both columns cleanly as
    # 9+3 | 9+3, so every limit this model knows says they may share a stage --
    # and p4c still refuses.
    #
    # What reproduces that today is the stage-sharing MARGIN (spec Sec 13.2
    # "Effect 4", src/p4model/packing.py's `charged`): a table whose key is not
    # the first distinct key in its stage, and whose standalone price leaves no
    # spare whole-byte crossbar slot, pays one extra TCAM block for the 2-bit
    # --version-- field. The app key is exactly saturated (crossbar_capacity(9)
    # == 49 bytes), so a mixed stage prices at 26, not 24, and the two tasks
    # stay apart. Measured on the same key by scripts/tcam_stretch_sweep.py: 9
    # blocks alone (ragged_ax1_bx5), 10 sharing (ragged_ax1_bx4).
    #
    # Note what the committed artifact does NOT show: any table at 10 blocks.
    # resources.json has all five app trees at 9 and all five ddos at 3, in
    # four stages (5 ddos | 2 app | 2 app | 1 app). The margin decides the
    # PLACEMENT, and the placement it settles on gives every stage a single
    # key, where nothing is owed. Reading that artifact is also what falsified
    # the older "ragged key at an odd group offset" rule, which claimed these
    # tables cost 10 unconditionally -- that mechanism is retired outright
    # (rewrite design Sec 2). The margin is pinned in
    # tests/test_p4model_guards.py's
    # test_two_different_ragged_keys_do_not_share_a_stage.
    root = _CURRENT_ARTIFACTS['independent_low_sd9']
    if not os.path.isdir(os.path.join(root, 'compiles', 'independent_low_sd9')):
        pytest.skip('needs %s (gitignored; run collect() first)' % root)
    assert cc.replay_stage_depth('independent_low_sd9', root) == (11, 11)


# ---------------------------------------------------------------------------
# Held-out designs: results/compiler_calibration_extra/ was compiled but never
# used to fit or re-tune the block model (the per-table harvest and the golden
# fixture read compiler_calibration_v6 only). replay_design prices every table
# with the MODEL's own per-table prices and packs them with the model's own
# packer, so this is the only end-to-end check on data the model never saw.
# The CSV's blocks/stage_depth columns are stale predictions from the model
# that was live when the rows were collected; only the *_real columns are
# ground truth here.
# ---------------------------------------------------------------------------

_HELDOUT_ROOT = os.path.join('results', 'compiler_calibration_extra')
_HELDOUT_CSV = os.path.join('results', 'compiler_calibration_extra.csv')


def _heldout_rows():
    if not os.path.isfile(_HELDOUT_CSV):
        return []
    frame = pd.read_csv(_HELDOUT_CSV)
    return [(r.row_id, r.tcam_real, int(r.stages_real))
            for r in frame.itertuples()]


# independent_low_sd9 is the design the stage-sharing margin exists for. The
# model's packing puts one app tree in a stage with ddos trees -- 49 + 12 = 61
# bytes, a crowded stage -- and the one key order that FITS (fits() only
# requires some ordering to pack column-wise, since the 2026-09-27 any-order
# fit rule: reviews/final_model_check_2026-09-27.md section 1b) charges the
# non-first table +1; p4c instead keeps the tasks in separate stages, at the
# same depth (exact) and no extra block. So blocks read 65 against 64 -- down
# from 67 under the retired "every order must fit" rule, which charged every
# non-first table in the stage rather than just the one order p4c could
# actually place. Safe direction; pinned by name so it cannot grow.
_HELDOUT_KNOWN_BLOCK_OVERS = {'independent_low_sd9': 1}


@pytest.mark.parametrize('row_id,tcam_real,stages_real', _heldout_rows())
def test_the_model_matches_p4c_on_every_held_out_design(row_id, tcam_real,
                                                         stages_real):
    if not os.path.isdir(os.path.join(_HELDOUT_ROOT, 'compiles', row_id)):
        pytest.skip('needs %s (gitignored)' % _HELDOUT_ROOT)
    predicted_depth, predicted_blocks = cc.replay_design(row_id, _HELDOUT_ROOT)
    # Depth exact on all 8, including the 3 p4c rejected as too deep (13-14
    # stages): the model calls them infeasible for the right reason.
    assert predicted_depth == stages_real
    if not pd.isna(tcam_real):
        assert (predicted_blocks - int(tcam_real)
                == _HELDOUT_KNOWN_BLOCK_OVERS.get(row_id, 0))


# ---------------------------------------------------------------------------
# 16 REAL campaign designs compiled specifically to exercise the stage-sharing
# margin and the crowded-stage rule (scripts/tcam_margin_screen.py): disjoint
# designs whose two keys can share a stage, half of them at 59-64 combined
# bytes. Chosen adversarially, so over-predictions are expected here; what
# must never happen is an under-prediction.
# ---------------------------------------------------------------------------

_MARGIN_ROOT = os.path.join('results', 'tcam_margin_screen')
_MARGIN_CSV = os.path.join('results', 'tcam_margin_screen_compiled.csv')


def _margin_rows():
    if not os.path.isfile(_MARGIN_CSV):
        return []
    return [(r.row_id, r.tcam_real, int(r.stages_real))
            for r in pd.read_csv(_MARGIN_CSV).itertuples()]


@pytest.mark.parametrize('row_id,tcam_real,stages_real', _margin_rows())
def test_the_model_never_under_predicts_a_crowded_real_design(row_id, tcam_real,
                                                              stages_real):
    if not os.path.isdir(os.path.join(_MARGIN_ROOT, 'compiles', row_id)):
        pytest.skip('needs %s (gitignored)' % _MARGIN_ROOT)
    depth, blocks = cc.replay_design(row_id, _MARGIN_ROOT)
    assert stages_real <= depth <= stages_real + 1
    if not pd.isna(tcam_real):
        assert int(tcam_real) <= blocks <= int(tcam_real) + 3
