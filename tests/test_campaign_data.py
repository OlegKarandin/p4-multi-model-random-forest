"""P7a: loading and pairing campaign results (spec C.3-C.5's frame contract).

`load_and_combine_data` (main.py) constructs literal
`feature_selection_comparison_results_by_k_-1_-1_{M}.csv` names, which the
current pipeline never writes -- it writes `rf_t{n_trees}_d{max_depth}_M{M}_
{arm_slug}.csv` (`arm_result_path`, main.py) instead. `load_campaign` replaces
it with a glob-and-parse loader.

Two traps this module exists to close, neither of which raises on its own:
infeasible rows carry NaN accuracies, and every NaN comparison is False, so a
NaN point is never dominated and lands on every Pareto front computed
downstream unless infeasible rows are filtered at load; and `delta_align` is
a string column ('', '0', '0.05', 'inf') -- compared numerically as loaded,
'0.05' < '0.1' is a string comparison that happens to be True, which survives
a casual test.

No real campaign CSV exists yet (the pilot cell hasn't run) -- every test here
builds a synthetic frame with a known answer and writes it to `tmp_path`.
"""
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.reporting.campaign_data import (
    MislabelledArtifactError,
    load_campaign,
    pair_arms,
)


# ---------------------------------------------------------------------------
# Row builders -- mirror the exact schema src/training/feature_selection.py
# and src/main.py's compare_independent_joint_mapping actually write.
# ---------------------------------------------------------------------------

def _feasible_row(arm='joint', method='multi', split=10, k=17, M=25,
                   acc_app=0.9, acc_ddos=0.85, blocks=40, stages=3,
                   alignment_enabled=True, delta_align='0.05',
                   overlap_threshold='0.5'):
    return {
        'arm': arm, 'method': method, 'split': split, 'k': k, 'M': M,
        'acc_app': acc_app, 'f1_app': 0.88, 'acc_ddos': acc_ddos, 'f1_ddos': 0.83,
        'acc_sel_app': 0.89, 'acc_sel_ddos': 0.84,
        'stages': stages, 'blocks': blocks,
        'infeasible': '',
        'stages_real': '', 'tcam_real': '', 'compile_errors': '',
        'features_app': 'F1;F2', 'features_ddos': 'F1;F2',
        'best_params': json.dumps({'n_estimators': 11}),
        'rel_shortfall': 0.01, 'n_trials_run': 120, 'n_feasible': 30,
        'align_attempted': 4, 'align_accepted': 3,
        'intervals_before': 10, 'intervals_after': 9,
        'alignment_enabled': alignment_enabled, 'delta_align': delta_align,
        'delta_select': 0.02, 'overlap_threshold': overlap_threshold,
    }


def _infeasible_row(arm='joint', method='multi', split=10, k=1,
                     alignment_enabled=True, delta_align='0.05'):
    return {
        'arm': arm, 'method': method, 'split': split, 'k': k,
        'acc_app': '', 'f1_app': '', 'acc_ddos': '', 'f1_ddos': '',
        'acc_sel_app': '', 'acc_sel_ddos': '',
        'stages': '', 'blocks': '',
        'infeasible': 'NoFeasibleSolution: no trial met the block budget',
        'stages_real': '', 'tcam_real': '', 'compile_errors': '',
        'features_app': 'F1', 'features_ddos': 'F1',
        'best_params': '',
        'rel_shortfall': '', 'n_trials_run': '', 'n_feasible': '',
        'align_attempted': '', 'align_accepted': '',
        'intervals_before': '', 'intervals_after': '',
        'alignment_enabled': alignment_enabled, 'delta_align': delta_align,
        'delta_select': 0.02, 'overlap_threshold': '0.5' if alignment_enabled else '',
    }


def _write_arm_file(tmp_path, n_trees, max_depth, M, arm_slug, rows,
                     results_dir='results'):
    frame = pd.DataFrame(rows)
    frame['M'] = M
    frame['n_trees'] = n_trees
    frame['max_depth'] = max_depth
    out_dir = tmp_path / results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'rf_t{n_trees}_d{max_depth}_M{M}_{arm_slug}.csv'
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Trap 1: infeasible rows must not reach the returned frame.
# ---------------------------------------------------------------------------

def test_load_campaign_drops_an_infeasible_row_so_its_nan_accuracy_cannot_reach_a_front(tmp_path):
    rows = [
        _feasible_row(k=17),
        _infeasible_row(k=1),
    ]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert len(df) == 1
    assert (df['infeasible'] == '').all()
    assert not df['acc_app'].isna().any()
    assert not df['acc_ddos'].isna().any()


def test_load_campaign_returns_numeric_dtype_for_acc_app_not_object_strings(tmp_path):
    rows = [_feasible_row(k=17), _infeasible_row(k=1)]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert pd.api.types.is_float_dtype(df['acc_app'])
    assert pd.api.types.is_float_dtype(df['acc_ddos'])
    assert pd.api.types.is_float_dtype(df['blocks'])


# ---------------------------------------------------------------------------
# F5/F6: `stage_depth` is a column added AFTER every real campaign CSV on
# disk was written, so the loader must tolerate its header being absent
# entirely (not merely '' on some rows, which is `stages_real`'s situation).
# ---------------------------------------------------------------------------

def test_load_campaign_gives_nan_stage_depth_when_the_column_is_absent_from_every_file(tmp_path):
    """_feasible_row/_infeasible_row above predate `stage_depth` too (by
    construction -- neither builder sets it), so a file built from them has
    no `stage_depth` header at all, exactly like a real pre-Task-13 CSV."""
    rows = [_feasible_row(k=17), _infeasible_row(k=1)]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert 'stage_depth' in df.columns
    assert pd.api.types.is_float_dtype(df['stage_depth'])
    assert df['stage_depth'].isna().all()


def test_load_campaign_parses_a_real_on_disk_csv_missing_the_stage_depth_column(tmp_path):
    """The real campaign result files under results/rf_t11_d14_M25_*.csv
    predate `stage_depth` (F5/F6) -- their header has no such column at all.
    load_campaign must still load them without raising, with `stage_depth`
    present and NaN throughout, not silently missing from the frame."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_files = sorted(glob.glob(
        os.path.join(repo_root, 'results', 'rf_t11_d14_M25_*.csv')))
    assert real_files, 'expected at least one real rf_t11_d14_M25_*.csv on disk'
    for path in real_files:
        with open(path, encoding='utf-8') as f:
            header = f.readline()
        assert 'stage_depth' not in header, (
            '{} already has a stage_depth column -- this test no longer '
            'exercises the column-absent path it is meant to prove'.format(path))

    out_dir = tmp_path / 'results'
    out_dir.mkdir()
    for path in real_files:
        shutil.copy(path, out_dir / os.path.basename(path))

    df = load_campaign(results_dir=str(out_dir))

    assert len(df) > 0
    assert 'stage_depth' in df.columns
    assert pd.api.types.is_float_dtype(df['stage_depth'])
    assert df['stage_depth'].isna().all()
    # The pre-existing stages/blocks columns must still parse as real numbers
    # -- proof this is additive, not a regression on the columns that were
    # already there.
    assert not df['stages'].isna().all()
    assert not df['blocks'].isna().all()


# ---------------------------------------------------------------------------
# Trap 2: delta_align is a string column, must be parsed not string-compared.
# ---------------------------------------------------------------------------

def test_delta_align_empty_string_parses_to_nan_and_is_not_flagged_inf(tmp_path):
    rows = [_feasible_row(k=17, arm='independent', method='single',
                           alignment_enabled=False, delta_align='',
                           overlap_threshold='')]
    _write_arm_file(tmp_path, 11, 14, 25, 'independent', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert np.isnan(df['delta_align_num'].iloc[0])
    assert df['delta_align_is_inf'].iloc[0] == False  # noqa: E712


def test_delta_align_zero_parses_to_numeric_zero_not_a_falsy_empty_string(tmp_path):
    rows = [_feasible_row(k=17, delta_align='0')]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d000', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert df['delta_align_num'].iloc[0] == 0.0
    assert df['delta_align_is_inf'].iloc[0] == False  # noqa: E712


def test_delta_align_decimal_value_parses_to_the_correct_float(tmp_path):
    rows = [_feasible_row(k=17, delta_align='0.05')]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert df['delta_align_num'].iloc[0] == pytest.approx(0.05)
    assert df['delta_align_is_inf'].iloc[0] == False  # noqa: E712


def test_delta_align_inf_is_distinguishable_from_any_numeric_value(tmp_path):
    rows = [_feasible_row(k=17, delta_align='inf')]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-dinf', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert df['delta_align_is_inf'].iloc[0] == True  # noqa: E712
    # inf must not collapse onto a numeric value (e.g. 0.0) via NaN-that-
    # looks-like-zero or similar coercion bugs.
    assert np.isnan(df['delta_align_num'].iloc[0])


def test_delta_align_num_holds_the_true_parsed_float_for_each_row(tmp_path):
    # NOT a test of string-vs-numeric ORDERING in general -- an earlier
    # version of this comment claimed '{:g}'-formatted values in [0, 1)
    # always sort the same way lexicographically and numerically. That
    # claim was wrong ('{:g}' switches to scientific notation under about
    # 1e-4, e.g. '{:g}'.format(5.19e-05) == '5.19e-05', which sorts
    # lexicographically ABOVE an ordinary '0.78...' string while being
    # numerically far below it) and is not repeated here, corrected or
    # otherwise -- it explains nothing this test or load_campaign needs.
    #
    # What this test actually establishes: load_campaign parses
    # delta_align unconditionally through pd.to_numeric, so the resulting
    # delta_align_num holds the correct float value regardless of how the
    # raw strings would sort -- there is no code path in this module where
    # a raw string comparison stands in for it. The real hazard on this
    # column (see module docstring) is non-numeric sentinels ('', 'inf')
    # and pandas' CSV dtype inference silently turning an all-'inf' column
    # into float64 infinity, both covered by the tests above.
    rows_lo = [_feasible_row(k=17, split=10, delta_align='0.2')]
    rows_hi = [_feasible_row(k=17, split=11, delta_align='0.1')]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d020', rows_lo)
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d010', rows_hi)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    row_020 = df[df['delta_align'] == '0.2'].iloc[0]
    row_010 = df[df['delta_align'] == '0.1'].iloc[0]
    assert row_020['delta_align_num'] == pytest.approx(0.2)
    assert row_010['delta_align_num'] == pytest.approx(0.1)
    assert row_020['delta_align_num'] > row_010['delta_align_num']


# ---------------------------------------------------------------------------
# Cross-check: a mislabelled artifact must fail loudly.
# ---------------------------------------------------------------------------

def test_a_file_whose_filename_arm_slug_disagrees_with_its_in_file_arm_column_fails_loudly(tmp_path):
    # Filename claims joint-d005, but every row inside is actually the
    # independent arm -- a mislabelled artifact (e.g. from a bad rename or a
    # copy-paste of the wrong file).
    rows = [_feasible_row(arm='independent', method='single',
                           alignment_enabled=False, delta_align='',
                           overlap_threshold='')]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    with pytest.raises(MislabelledArtifactError):
        load_campaign(results_dir=str(tmp_path / 'results'))


def test_a_file_whose_filename_M_disagrees_with_its_in_file_M_column_fails_loudly(tmp_path):
    rows = [_feasible_row()]
    path = _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)
    # Corrupt the in-file M column after writing, simulating a mislabelled
    # or hand-edited artifact whose filename no longer matches its contents.
    frame = pd.read_csv(path)
    frame['M'] = 40
    frame.to_csv(path, index=False)

    with pytest.raises(MislabelledArtifactError):
        load_campaign(results_dir=str(tmp_path / 'results'))


def test_a_file_whose_filename_n_trees_disagrees_with_its_in_file_column_fails_loudly(tmp_path):
    rows = [_feasible_row()]
    path = _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)
    frame = pd.read_csv(path)
    frame['n_trees'] = 7
    frame.to_csv(path, index=False)

    with pytest.raises(MislabelledArtifactError):
        load_campaign(results_dir=str(tmp_path / 'results'))


def test_load_campaign_accepts_a_correctly_labelled_file_without_raising(tmp_path):
    rows = [_feasible_row()]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert len(df) == 1
    assert df['arm_slug'].iloc[0] == 'joint-d005'


def test_load_campaign_raises_a_clear_error_when_no_files_match(tmp_path):
    (tmp_path / 'results').mkdir()

    with pytest.raises(FileNotFoundError):
        load_campaign(results_dir=str(tmp_path / 'results'))


def test_load_campaign_ignores_the_legacy_by_k_filename_pattern(tmp_path):
    # The old feature_selection_comparison_results_by_k_-1_-1_{M}.csv files
    # must not be picked up by the new glob.
    out_dir = tmp_path / 'results'
    out_dir.mkdir()
    pd.DataFrame([_feasible_row()]).to_csv(
        out_dir / 'feature_selection_comparison_results_by_k_-1_-1_25.csv', index=False)
    rows = [_feasible_row()]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert len(df) == 1


# ---------------------------------------------------------------------------
# pair_arms: inner join on (M, split, k), keyed on arm_slug not method.
# ---------------------------------------------------------------------------

def _build_paired_campaign(tmp_path):
    treatment_rows = [
        _feasible_row(split=10, k=17, M=25),
        _feasible_row(split=10, k=15, M=25),
        _feasible_row(split=11, k=17, M=25),
        # This one has no counterpart in the baseline at M=25 (a
        # deliberately missing cell) -- inner join must drop it.
        _feasible_row(split=12, k=17, M=25),
        # Same (split, k) as the first row, but at a DIFFERENT M -- must
        # not collapse onto the M=25 pairing.
        _feasible_row(split=10, k=17, M=40),
    ]
    baseline_rows = [
        _feasible_row(arm='independent', method='single',
                      alignment_enabled=False, delta_align='', overlap_threshold='',
                      split=10, k=17, M=25),
        _feasible_row(arm='independent', method='single',
                      alignment_enabled=False, delta_align='', overlap_threshold='',
                      split=10, k=15, M=25),
        _feasible_row(arm='independent', method='single',
                      alignment_enabled=False, delta_align='', overlap_threshold='',
                      split=11, k=17, M=25),
        # No M=40 baseline row for split=10,k=17 is added deliberately below
        # in the "missing cell" variant of these tests; this one IS present
        # so the M-distinctness test has a real pair to find.
        _feasible_row(arm='independent', method='single',
                      alignment_enabled=False, delta_align='', overlap_threshold='',
                      split=10, k=17, M=40),
    ]
    _write_arm_file(tmp_path, 11, 14, 25, 'joint-d005',
                     [r for r in treatment_rows if r['M'] == 25])
    _write_arm_file(tmp_path, 11, 14, 40, 'joint-d005',
                     [r for r in treatment_rows if r['M'] == 40])
    _write_arm_file(tmp_path, 11, 14, 25, 'independent',
                     [r for r in baseline_rows if r['M'] == 25])
    _write_arm_file(tmp_path, 11, 14, 40, 'independent',
                     [r for r in baseline_rows if r['M'] == 40])
    return load_campaign(results_dir=str(tmp_path / 'results'))


def test_pair_arms_inner_joins_on_M_split_k_and_drops_a_cell_missing_from_the_baseline(tmp_path):
    df = _build_paired_campaign(tmp_path)

    paired = pair_arms(df, treatment='joint-d005', baseline='independent')

    # (split=12, k=17, M=25) exists only in the treatment arm -- must be
    # dropped by the inner join, not carried through with a NaN baseline.
    assert not ((paired['split'] == 12) & (paired['k'] == 17) & (paired['M'] == 25)).any()


def test_pair_arms_does_not_collapse_rows_that_differ_only_in_M(tmp_path):
    df = _build_paired_campaign(tmp_path)

    paired = pair_arms(df, treatment='joint-d005', baseline='independent')

    same_split_k = paired[(paired['split'] == 10) & (paired['k'] == 17)]
    # Two distinct M values (25 and 40) both pair split=10,k=17 -- keying
    # the join on (split, k) only (the old perform_statistical_analysis bug)
    # would collapse these into one row, last-wins.
    assert set(same_split_k['M']) == {25, 40}
    assert len(same_split_k) == 2


def test_pair_arms_keys_on_arm_slug_not_on_the_legacy_method_column(tmp_path):
    # Both the treatment and baseline rows share method values that overlap
    # in principle ('multi' vs 'single' is a legacy duplicate of arm/arm_slug,
    # not the real identity) -- pairing must select by arm_slug regardless.
    df = _build_paired_campaign(tmp_path)

    paired = pair_arms(df, treatment='joint-d005', baseline='independent')

    assert (paired['arm_slug_treatment'] == 'joint-d005').all()
    assert (paired['arm_slug_baseline'] == 'independent').all()


def test_pair_arms_returns_empty_frame_for_an_arm_slug_present_in_neither_arm(tmp_path):
    df = _build_paired_campaign(tmp_path)

    paired = pair_arms(df, treatment='joint-dinf', baseline='independent')

    assert len(paired) == 0
