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
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.reporting.campaign_data import (
    MislabelledArtifactError,
    _expected_arm_slug,
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
        'range_entries': 15, 'ternary_entries': 8,
        'register_depth': 2, 'register_count': 3,
        'infeasible': '',
        'stages_real': '', 'tcam_real': '', 'sram_real': '', 'map_ram_real': '',
        'compile_errors': '',
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
        'range_entries': '', 'ternary_entries': '',
        'register_depth': '', 'register_count': '',
        'infeasible': 'NoFeasibleSolution: no trial met the block budget',
        'stages_real': '', 'tcam_real': '', 'sram_real': '', 'map_ram_real': '',
        'compile_errors': '',
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
    """`tests/fixtures/rf_t11_d14_M25_historical.csv` is a frozen copy of one
    of the pilot campaign's real result files, taken before `stage_depth`
    (F5/F6) or the six Task-8 columns existed -- its header has none of
    them. Unlike the files under results/, which Phase 7 will eventually
    overwrite with a rerun that DOES carry these columns, this fixture is
    checked in and never regenerated, so the column-absent path it exercises
    stays provable forever. load_campaign must still load it without
    raising, with every one of these columns present and NaN throughout, not
    silently missing from the frame."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixture_path = os.path.join(
        repo_root, 'tests', 'fixtures', 'rf_t11_d14_M25_historical.csv')
    assert os.path.exists(fixture_path), (
        'expected the frozen fixture {} to exist'.format(fixture_path))

    new_columns = [
        'stage_depth', 'range_entries', 'ternary_entries', 'register_depth',
        'register_count', 'sram_real', 'map_ram_real',
    ]
    with open(fixture_path, encoding='utf-8') as f:
        header = f.readline()
    for col in new_columns:
        assert col not in header, (
            '{} already has a {!r} column -- this test no longer exercises '
            'the column-absent path it is meant to prove'.format(
                fixture_path, col))

    # Copy under the identity-matching basename (the fixture's own arm data
    # says 'independent'): load_campaign cross-checks the filename-encoded
    # identity against the in-file columns and raises MislabelledArtifactError
    # on a mismatch, so the file on disk must be named to match its content,
    # not the fixture's own storage name.
    out_dir = tmp_path / 'results'
    out_dir.mkdir()
    shutil.copy(fixture_path, out_dir / 'rf_t11_d14_M25_independent.csv')

    df = load_campaign(results_dir=str(out_dir))

    assert len(df) > 0
    for col in new_columns:
        assert col in df.columns
        assert pd.api.types.is_float_dtype(df[col])
        assert df[col].isna().all()
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
# overlap_threshold: arm_slug suffixes only away from the 0.5 default, so
# _expected_arm_slug must check the column against the filename to keep that
# implicit default safe rather than merely convenient.
# ---------------------------------------------------------------------------

def test_a_file_whose_overlap_column_contradicts_its_filename_is_rejected(tmp_path):
    """arm_slug suffixes only away from 0.5, so the default is implicit in the
    filename. This check is what makes that safe: an artifact named
    joint-d020 that actually ran at overlap 0.25 is mislabelled, and reading
    it would silently attribute one treatment's results to another."""
    path = tmp_path / 'rf_t11_d14_M25_joint-d020.csv'
    pd.DataFrame([{
        'arm': 'joint', 'method': 'multi', 'split': 10, 'k': 3,
        'alignment_enabled': True, 'delta_align': 0.2, 'delta_select': 0.02,
        'M': 25, 'n_trees': 11, 'max_depth': 14,
        'overlap_threshold': 0.25,          # contradicts the filename
        'blocks': 10, 'stages': 3, 'stage_depth': 4, 'infeasible': '',
    }]).to_csv(path, index=False)

    with pytest.raises(MislabelledArtifactError):
        load_campaign(results_dir=str(tmp_path))


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


def test_an_archived_slug_still_validates_from_its_own_overlap_column():
    """Archived filenames carry the -o suffix and archived rows carry the
    column that reproduces it. TrainConfig no longer writes either, but the
    check that a file named joint-d020-o025 really holds
    overlap_threshold=0.25 must keep working, or the archive stops being
    readable."""
    assert _expected_arm_slug('joint', True, '0.2', '0.25') == 'joint-d020-o025'
    assert _expected_arm_slug('joint', True, '0.2', '0.5') == 'joint-d020'


def test_a_fresh_file_without_the_column_validates_too():
    """The other half. A frame written after 2026-09-14 has no
    overlap_threshold at all; requiring it would make every new campaign
    unreadable. `''` is the raw CSV text for a suppressed arm and `None`/NaN is
    what an absent column becomes -- both must mean "no suffix"."""
    assert _expected_arm_slug('joint', True, '0.2', '') == 'joint-d020'
    assert _expected_arm_slug('joint', True, '0.2', None) == 'joint-d020'


def test_an_archived_slug_still_validates_from_its_own_delta_align_column():
    """The 2026-09-15 archive boundary, the same shape as the overlap one
    above. campaign_backup_20260825's files are named joint-d000 ... joint-dinf
    and each row carries the delta_align label that reproduces the name.
    TrainConfig no longer has the field and src/main.py no longer writes the
    column (Track 5: delta_helps = FALSE), but that reconstruction must keep
    working or the whole archive stops being readable."""
    assert _expected_arm_slug('joint', True, '0') == 'joint-d000'
    assert _expected_arm_slug('joint', True, '0.05') == 'joint-d005'
    assert _expected_arm_slug('joint', True, '0.2') == 'joint-d020'
    assert _expected_arm_slug('joint', True, 'inf') == 'joint-dinf'


def test_a_fresh_aligned_row_without_a_delta_column_is_the_plain_joint_arm():
    """The other half. A file written after 2026-09-15 has no delta_align
    column, so the label arrives as '' (or NaN) and the slug is plain `joint`
    -- TrainConfig.arm_slug's own answer for the same config. No collision
    with the archived branch: an archived ALIGNED row never carried '', since
    the retired delta_align_label only returned '' for the independent arm or
    for alignment_enabled=False, both of which return earlier."""
    assert _expected_arm_slug('joint', True, '') == 'joint'
    assert _expected_arm_slug('joint', True, None) == 'joint'
    assert _expected_arm_slug('joint', False, '') == 'joint-off'
    assert _expected_arm_slug('independent', True, '') == 'independent'


def test_a_fresh_campaign_file_loads_without_a_delta_align_column(tmp_path):
    """End-to-end for the same boundary: nothing writes delta_align any more,
    so load_campaign must not require it -- and must still expose the three
    delta columns, so a downstream reader's shape does not depend on which
    files happened to be loaded."""
    rows = [_feasible_row(k=17), _feasible_row(k=9)]
    for row in rows:
        del row['delta_align']
        del row['overlap_threshold']
    _write_arm_file(tmp_path, 11, 14, 25, 'joint', rows)

    df = load_campaign(results_dir=str(tmp_path / 'results'))

    assert len(df) == 2
    assert (df['arm_slug'] == 'joint').all()
    assert (df['delta_align'] == '').all()
    assert df['delta_align_num'].isna().all()
    assert (~df['delta_align_is_inf']).all()


def test_pair_arms_returns_empty_frame_for_an_arm_slug_present_in_neither_arm(tmp_path):
    df = _build_paired_campaign(tmp_path)

    paired = pair_arms(df, treatment='joint-dinf', baseline='independent')

    assert len(paired) == 0


# ---------------------------------------------------------------------------
# Task 12: a compiler-verified RUN (rows/, verification.csv, run_manifest.json)
# ---------------------------------------------------------------------------

from src.reporting.campaign_data import (  # noqa: E402
    EnvironmentDriftError,
    UnverifiedRowsError,
    _parse_filename,
    load_verification,
)
from src.verify.runner import VERIFICATION_COLUMNS  # noqa: E402

_RUN_IMAGE = 'ghcr.io/example/p4c:1'
_RUN_P4STUDIO = 'abc123'


def _run_row(split=0, k=5, M=35, stage_depth=9, blocks=30, infeasible=''):
    """One row of a fresh per-split CSV (Task 9 schema): no delta_align or
    overlap_threshold column, a row_id on every row, M '' when unbudgeted."""
    budgeted = M != 'inf'
    token = '{:03d}'.format(M) if budgeted else 'inf'
    row = _feasible_row(arm='joint', split=split, k=k, M=M if budgeted else '',
                        blocks=blocks, alignment_enabled=True)
    del row['delta_align'], row['overlap_threshold']
    row.update({
        'budgeted': budgeted, 'stage_depth': stage_depth,
        'n_trees': 7, 'max_depth': 14,
        'row_id': 'joint_M{}_s{:02d}_k{:02d}'.format(token, split, k),
    })
    if infeasible:
        row.update({'infeasible': infeasible, 'acc_app': '', 'blocks': '',
                    'stage_depth': ''})
    return row


def _ver(row_id, verdict='EXACT', p4c_stage_depth=10, p4c_blocks=32,
         over_budget=False, over_stages=False, unverified=False, M=35, **extra):
    """One verification.csv line, every column of the verifier's real list."""
    line = {col: '' for col in VERIFICATION_COLUMNS}
    line.update({
        'row_id': row_id, 'M': M, 'verdict': verdict,
        'model_stage_depth': 9, 'model_blocks': 30,
        'model_paths_differ': False,
        'p4c_stage_depth': '' if unverified else p4c_stage_depth,
        'p4c_blocks': '' if unverified else p4c_blocks,
        'p4c_sram': '' if unverified else 12,
        'p4c_map_ram': '' if unverified else 4,
        'p4c_phv_containers': '' if unverified else 100,
        'p4c_over_budget': over_budget, 'p4c_over_stages': over_stages,
        'unverified': unverified, 'tables_differing': '[]',
        'p4c_image': _RUN_IMAGE, 'open_p4studio_commit': _RUN_P4STUDIO,
    })
    line.update(extra)
    return line


def _write_run(tmp_path, rows, verification, manifest=None, name='run'):
    """Write a run directory: rows/ per-split CSVs grouped by (M, split),
    named as split_csv_name names them (rf_t7_d14_M035_joint_s00.csv, and an
    Minf file for unbudgeted rows), plus verification.csv in the verifier's
    own column order and run_manifest.json."""
    run = tmp_path / name
    (run / 'rows').mkdir(parents=True)
    groups = {}
    for row in rows:
        token = 'inf' if row['M'] == '' else '{:03d}'.format(int(row['M']))
        groups.setdefault((token, row['split']), []).append(row)
    for (token, split), group in groups.items():
        pd.DataFrame(group).to_csv(
            run / 'rows' / 'rf_t7_d14_M{}_joint_s{:02d}.csv'.format(token, split),
            index=False)
    pd.DataFrame(verification, columns=list(VERIFICATION_COLUMNS)).to_csv(
        run / 'verification.csv', index=False)
    if manifest is None:
        manifest = {'M_values': [35, 'inf'], 'p4c_image': _RUN_IMAGE,
                    'open_p4studio_commit': _RUN_P4STUDIO}
    (run / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return str(run)


def test_parse_filename_round_trips_per_split_minf_and_legacy_names():
    assert _parse_filename('rf_t7_d14_M035_joint_s00.csv') == {
        'n_trees': 7, 'max_depth': 14, 'M': 35, 'arm_slug': 'joint', 'split': 0}
    inf = _parse_filename('rf_t7_d14_Minf_joint-off_s13.csv')
    assert inf['M'] == float('inf') and inf['split'] == 13
    assert inf['arm_slug'] == 'joint-off'
    assert _parse_filename('rf_t11_d14_M25_joint-d005.csv') == {
        'n_trees': 11, 'max_depth': 14, 'M': 25, 'arm_slug': 'joint-d005',
        'split': None}


def test_a_split_file_whose_rows_name_another_split_is_mislabelled(tmp_path):
    run = _write_run(tmp_path, [_run_row(split=0)],
                     [_ver('joint_M035_s00_k05')])
    path = os.path.join(run, 'rows', 'rf_t7_d14_M035_joint_s00.csv')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame['split'] = '1'
    frame.to_csv(path, index=False)
    with pytest.raises(MislabelledArtifactError):
        load_campaign(run)


def test_a_run_reports_p4c_numbers_and_keeps_the_model_numbers_beside_them(tmp_path):
    run = _write_run(tmp_path, [_run_row(stage_depth=9, blocks=30)],
                     [_ver('joint_M035_s00_k05', verdict='OVER',
                           p4c_stage_depth=10, p4c_blocks=32)])
    df = load_campaign(run)
    assert len(df) == 1
    row = df.iloc[0]
    assert row['stage_depth'] == 10 and row['blocks'] == 32
    assert row['model_stage_depth'] == 9 and row['model_blocks'] == 30
    assert row['p4c_sram'] == 12 and row['p4c_phv_containers'] == 100
    assert row['verdict'] == 'OVER'
    assert not row['flagged']
    assert not row['p4c_over_budget'] and not row['unverified']
    assert pd.api.types.is_float_dtype(df['blocks'])
    assert pd.api.types.is_float_dtype(df['model_blocks'])
    for col in ('flagged', 'unverified', 'p4c_over_budget', 'p4c_over_stages'):
        assert pd.api.types.is_bool_dtype(df[col]), col


def test_p4c_infeasible_designs_leave_the_frame_but_stay_in_load_verification(tmp_path):
    rows = [_run_row(k=5), _run_row(k=6), _run_row(k=7)]
    ver = [_ver('joint_M035_s00_k05'),
           _ver('joint_M035_s00_k06', verdict='FALSE_FEASIBLE', over_budget=True),
           _ver('joint_M035_s00_k07', verdict='UNDER', over_budget=True)]
    run = _write_run(tmp_path, rows, ver)
    df = load_campaign(run)
    assert df['row_id'].tolist() == ['joint_M035_s00_k05']
    verification = load_verification(run)
    assert set(verification['row_id']) == {
        'joint_M035_s00_k05', 'joint_M035_s00_k06', 'joint_M035_s00_k07'}
    k7 = verification[verification['row_id'] == 'joint_M035_s00_k07'].iloc[0]
    assert k7['arm_slug'] == 'joint' and k7['M'] == 35
    assert k7['split'] == 0 and k7['k'] == 7


def test_an_over_stages_design_is_excluded_too(tmp_path):
    run = _write_run(tmp_path, [_run_row(k=5), _run_row(k=6)],
                     [_ver('joint_M035_s00_k05'),
                      _ver('joint_M035_s00_k06', verdict='UNDER', over_stages=True)])
    assert load_campaign(run)['row_id'].tolist() == ['joint_M035_s00_k05']


def test_a_compile_error_row_keeps_the_model_numbers_and_is_flagged(tmp_path):
    run = _write_run(tmp_path, [_run_row(k=5, stage_depth=9, blocks=30),
                                _run_row(k=6)],
                     [_ver('joint_M035_s00_k05', verdict='COMPILE_ERROR',
                           unverified=True, failure='p4c_errors'),
                      _ver('joint_M035_s00_k06')])
    df = load_campaign(run).set_index('row_id')
    bad = df.loc['joint_M035_s00_k05']
    assert bad['flagged']
    assert bad['stage_depth'] == 9 and bad['blocks'] == 30
    assert bad['model_stage_depth'] == 9 and bad['model_blocks'] == 30
    assert bad['verdict'] == 'COMPILE_ERROR'
    assert not df.loc['joint_M035_s00_k06', 'flagged']


def test_a_design_row_without_a_verification_line_raises_naming_it(tmp_path):
    run = _write_run(tmp_path, [_run_row(k=5), _run_row(k=6),
                                _run_row(k=1, infeasible='NoFeasibleSolution: x')],
                     [_ver('joint_M035_s00_k05')])
    with pytest.raises(UnverifiedRowsError) as excinfo:
        load_campaign(run)
    assert excinfo.value.row_ids == ['joint_M035_s00_k06']


def test_require_verified_false_loads_a_partly_verified_run(tmp_path):
    run = _write_run(tmp_path, [_run_row(k=5), _run_row(k=6)],
                     [_ver('joint_M035_s00_k05')])
    df = load_campaign(run, require_verified=False).set_index('row_id')
    assert df.loc['joint_M035_s00_k06', 'flagged']
    assert df.loc['joint_M035_s00_k06', 'blocks'] == 30
    assert df.loc['joint_M035_s00_k05', 'blocks'] == 32


def test_an_image_tag_mismatch_raises_environment_drift(tmp_path):
    run = _write_run(tmp_path, [_run_row()],
                     [_ver('joint_M035_s00_k05', p4c_image='ghcr.io/example/p4c:2')])
    with pytest.raises(EnvironmentDriftError):
        load_campaign(run)


def test_a_manifest_without_an_image_skips_the_drift_check(tmp_path):
    run = _write_run(tmp_path, [_run_row()],
                     [_ver('joint_M035_s00_k05', p4c_image='anything')],
                     manifest={'M_values': [35], 'p4c_image': None,
                               'open_p4studio_commit': None})
    assert len(load_campaign(run)) == 1


def test_an_minf_file_loads_inf_and_unbudgeted(tmp_path):
    run = _write_run(tmp_path, [_run_row(M='inf'), _run_row(M=35)],
                     [_ver('joint_Minf_s00_k05', M='inf'),
                      _ver('joint_M035_s00_k05')])
    df = load_campaign(run).set_index('row_id')
    assert df['M'].dtype == np.float64
    assert df['budgeted'].dtype == bool
    assert df.loc['joint_Minf_s00_k05', 'M'] == float('inf')
    assert not df.loc['joint_Minf_s00_k05', 'budgeted']
    assert df.loc['joint_M035_s00_k05', 'M'] == 35.0
    assert df.loc['joint_M035_s00_k05', 'budgeted']
    ver = load_verification(run).set_index('row_id')
    assert ver.loc['joint_Minf_s00_k05', 'M'] == float('inf')


def test_an_minf_file_whose_rows_say_budgeted_is_mislabelled(tmp_path):
    run = _write_run(tmp_path, [_run_row(M='inf')],
                     [_ver('joint_Minf_s00_k05', M='inf')])
    path = os.path.join(run, 'rows', 'rf_t7_d14_Minf_joint_s00.csv')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame['budgeted'] = 'True'
    frame.to_csv(path, index=False)
    with pytest.raises(MislabelledArtifactError):
        load_campaign(run)


def test_a_legacy_load_is_unflagged_with_a_float_M(tmp_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixture_path = os.path.join(
        repo_root, 'tests', 'fixtures', 'rf_t11_d14_M25_historical.csv')
    out_dir = tmp_path / 'results'
    out_dir.mkdir()
    shutil.copy(fixture_path, out_dir / 'rf_t11_d14_M25_independent.csv')
    df = load_campaign(results_dir=str(out_dir))
    assert (~df['flagged']).all()
    assert (df['verdict'] == '').all()
    assert df['M'].dtype == np.float64 and (df['M'] == 25.0).all()
    assert df['budgeted'].all()
