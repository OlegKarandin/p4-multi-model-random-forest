"""Unit tests for scripts/feasibility_frontier.py.

Covers the pure, module-level helpers (Task 3): build_grid, cfg_for_arm,
trial_violation_type, summarize_trials, any_feasible_at, reach,
load_cell_features, print_reach_table, print_violation_breakdown.

It ALSO exercises already_done's on-disk resume logic and collect()'s
ProcessPoolExecutor-based orchestration end-to-end against real, spawned
worker processes writing real CSV files under tmp_path (resume/skip,
initializer-based dataset passing, per-future exception isolation, the
--limit / --max-workers plumbing) -- with run_one_point and
load_campaign_data monkeypatched to cheap, deterministic stand-ins so this
still never touches Optuna, sklearn, or the real campaign search itself.
main() is touched only via a stubbed collect() (see
test_main_timing_reports_the_resolved_worker_count_when_max_workers_omitted),
not run against real data.
"""
import os
import types

import optuna
import pandas as pd
import pytest

import scripts.feasibility_frontier as feasibility_frontier
from scripts.feasibility_frontier import (
    ARM_NAMES,
    CELLS,
    FULL_GRID_SIZE,
    SPLIT_INDICES,
    T_VALUES,
    already_done,
    any_feasible_at,
    build_grid,
    cfg_for_arm,
    collect,
    load_cell_features,
    print_violation_breakdown,
    reach,
    summarize_trials,
    trial_violation_type,
)
from src.training import early_stopping


# --------------------------------------------------------------------------
# build_grid
# --------------------------------------------------------------------------

def test_build_grid_has_exactly_576_points():
    assert FULL_GRID_SIZE == 576
    assert len(build_grid()) == 576


def test_build_grid_has_no_duplicate_points():
    grid = build_grid()
    as_tuples = [tuple(sorted(point.items())) for point in grid]
    assert len(set(as_tuples)) == len(as_tuples)


def test_build_grid_covers_every_cell_T_arm_split_combination():
    grid = build_grid()
    seen = {(point['M'], point['k'], point['T'], point['arm'], point['split'])
            for point in grid}
    expected = {(M, k, T, arm, split)
                for (M, k) in CELLS for T in T_VALUES
                for arm in ARM_NAMES for split in SPLIT_INDICES}
    assert seen == expected


# --------------------------------------------------------------------------
# cfg_for_arm
# --------------------------------------------------------------------------

def test_cfg_for_arm_control_disables_alignment_and_ccp_alpha():
    cfg = cfg_for_arm('control', T=3)
    assert cfg.alignment_enabled is False
    assert cfg.ccp_alpha_max == 0.0
    assert cfg.n_trees_min == 3
    assert cfg.n_trees == 3


def test_cfg_for_arm_aligned_only_enables_alignment_not_ccp_alpha():
    cfg = cfg_for_arm('aligned_only', T=4)
    assert cfg.alignment_enabled is True
    assert cfg.delta_align == 0.20
    assert cfg.overlap_threshold == 0.5
    assert cfg.align_objective == 'blocks'
    assert cfg.ccp_alpha_max == 0.0
    assert cfg.n_trees_min == 4
    assert cfg.n_trees == 4


def test_cfg_for_arm_ccp_alpha_only_enables_ccp_alpha_not_alignment():
    cfg = cfg_for_arm('ccp_alpha_only', T=5)
    assert cfg.alignment_enabled is False
    assert cfg.ccp_alpha_max == feasibility_frontier.CCP_ALPHA_STUDY_MAX
    assert cfg.n_trees_min == 5
    assert cfg.n_trees == 5


def test_cfg_for_arm_both_enables_alignment_and_ccp_alpha():
    cfg = cfg_for_arm('both', T=6)
    assert cfg.alignment_enabled is True
    assert cfg.delta_align == 0.20
    assert cfg.overlap_threshold == 0.5
    assert cfg.align_objective == 'blocks'
    assert cfg.ccp_alpha_max == feasibility_frontier.CCP_ALPHA_STUDY_MAX
    assert cfg.n_trees_min == 6
    assert cfg.n_trees == 6


@pytest.mark.parametrize('T', T_VALUES)
def test_cfg_for_arm_pins_n_trees_min_equal_to_n_trees_equal_to_T(T):
    """spec 2.5: T is pinned via n_trees_min == n_trees == T, for every arm."""
    for arm in ARM_NAMES:
        cfg = cfg_for_arm(arm, T)
        assert cfg.n_trees_min == cfg.n_trees == T


def test_cfg_for_arm_rejects_unknown_arm():
    with pytest.raises(ValueError, match='unknown arm'):
        cfg_for_arm('not_a_real_arm', T=1)


# --------------------------------------------------------------------------
# trial_violation_type
# --------------------------------------------------------------------------

@pytest.mark.parametrize('violated_attr', early_stopping._VIOLATION_ATTRS)
def test_trial_violation_type_reads_the_one_nonzero_attribute(violated_attr):
    user_attrs = {name: 0.0 for name in early_stopping._VIOLATION_ATTRS}
    user_attrs[violated_attr] = 1.5
    assert trial_violation_type(user_attrs) == violated_attr


def test_trial_violation_type_all_zero_is_none():
    user_attrs = {name: 0.0 for name in early_stopping._VIOLATION_ATTRS}
    assert trial_violation_type(user_attrs) is None


def test_trial_violation_type_missing_attributes_is_none():
    """A trial that never reached objective()'s attribute-setting code has no
    user_attrs at all -- trial_violation_type (the F6 "which violation type"
    diagnostic) must not crash, and reports "no violation type" (None).

    This is NOT the same as being feasible: summarize_trials deliberately
    does not use this function's None to decide feasibility -- see
    test_summarize_trials_missing_attrs_trial_counts_as_infeasible below,
    which is the actual feasibility-decision test for this case."""
    assert trial_violation_type({}) is None


# --------------------------------------------------------------------------
# summarize_trials
# --------------------------------------------------------------------------

def _trial(state=optuna.trial.TrialState.COMPLETE, **violations):
    """A types.SimpleNamespace stand-in for an optuna.trial.FrozenTrial,
    duck-typed to expose only what summarize_trials/trial_violation_type
    read: .state and .user_attrs."""
    user_attrs = {name: 0.0 for name in early_stopping._VIOLATION_ATTRS}
    user_attrs.update(violations)
    return types.SimpleNamespace(state=state, user_attrs=user_attrs)


def test_summarize_trials_counts_feasible_and_infeasible_trials():
    trials = [
        _trial(),  # feasible
        _trial(),  # feasible
        _trial(blocks_violation=2.0),  # infeasible: blocks
        _trial(codeword_violation=1.0),  # infeasible: codeword
    ]
    out = summarize_trials(trials)
    assert out['n_trials_run'] == 4
    assert out['n_feasible'] == 2
    assert out['any_feasible'] is True
    assert out['n_blocks_violation'] == 1
    assert out['n_codeword_violation'] == 1
    assert out['n_crossbar_violation'] == 0
    assert out['n_stages_violation'] == 0


def test_summarize_trials_ignores_non_complete_trials():
    """A PRUNED or RUNNING trial contributes to n_trials_run but never to
    n_feasible or a violation count -- summarize_trials only inspects
    user_attrs for COMPLETE trials."""
    trials = [
        _trial(),  # feasible, COMPLETE
        _trial(state=optuna.trial.TrialState.PRUNED, blocks_violation=99.0),
    ]
    out = summarize_trials(trials)
    assert out['n_trials_run'] == 2
    assert out['n_feasible'] == 1
    assert out['n_blocks_violation'] == 0


def test_summarize_trials_dominant_violation_type_is_the_most_common():
    trials = [
        _trial(blocks_violation=1.0),
        _trial(blocks_violation=1.0),
        _trial(codeword_violation=1.0),
    ]
    out = summarize_trials(trials)
    assert out['dominant_violation_type'] == 'blocks_violation'


def test_summarize_trials_no_trials_at_all():
    out = summarize_trials([])
    assert out['n_trials_run'] == 0
    assert out['n_feasible'] == 0
    assert out['any_feasible'] is False
    assert out['dominant_violation_type'] is None


def test_summarize_trials_all_feasible_dominant_violation_type_is_none():
    trials = [_trial(), _trial()]
    out = summarize_trials(trials)
    assert out['any_feasible'] is True
    assert out['dominant_violation_type'] is None


def test_summarize_trials_missing_attrs_trial_counts_as_infeasible():
    """Code review Finding 3: feasibility must come from the public
    early_stopping.is_feasible, not from trial_violation_type's own
    missing-attribute-defaults-to-0.0 logic. A COMPLETE trial with NO
    violation user_attrs at all (never reached objective()'s
    attribute-setting code -- e.g. it failed before that point) must read as
    INFEASIBLE here, per early_stopping.constraint_values' own documented
    convention (missing attribute -> inf -> infeasible), even though
    trial_violation_type({}) itself reports None ("no violation type")."""
    trial = types.SimpleNamespace(state=optuna.trial.TrialState.COMPLETE, user_attrs={})
    out = summarize_trials([trial])
    assert out['n_trials_run'] == 1
    assert out['n_feasible'] == 0
    assert out['any_feasible'] is False
    # Still reports "no diagnosable violation type" for this trial, since
    # trial_violation_type genuinely can't tell WHICH constraint was hit when
    # no attributes were ever set -- but it must not count as feasible.
    assert all(out['n_' + name] == 0 for name in early_stopping._VIOLATION_ATTRS)


def test_summarize_trials_mixed_missing_and_real_violations():
    """A more realistic mix: one genuinely feasible trial, one COMPLETE trial
    with a real violation, and one COMPLETE trial with no violation
    attributes at all (must count as infeasible, not feasible, per Finding
    3) -- n_feasible must reflect only the genuinely feasible trial."""
    trials = [
        _trial(),  # feasible
        _trial(blocks_violation=2.0),  # infeasible: blocks
        types.SimpleNamespace(state=optuna.trial.TrialState.COMPLETE, user_attrs={}),
    ]
    out = summarize_trials(trials)
    assert out['n_trials_run'] == 3
    assert out['n_feasible'] == 1
    assert out['any_feasible'] is True
    assert out['n_blocks_violation'] == 1


# --------------------------------------------------------------------------
# any_feasible_at / reach
# --------------------------------------------------------------------------

def _frame(rows):
    """Build a synthetic results table using bracket/column-name access only
    -- frame['T'], never frame.T, since DataFrame.T is pandas' own transpose
    property and silently shadows a column literally named 'T'."""
    return pd.DataFrame(rows)


def test_any_feasible_at_true_when_a_matching_row_is_feasible():
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 10, 'any_feasible': True},
    ])
    assert any_feasible_at(frame, M=25, k=9, T=2, arm='control') is True


def test_any_feasible_at_false_when_no_matching_row_is_feasible():
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 10, 'any_feasible': False},
        {'M': 25, 'k': 9, 'T': 3, 'arm': 'control', 'split': 10, 'any_feasible': True},
    ])
    assert any_feasible_at(frame, M=25, k=9, T=2, arm='control') is False


def test_any_feasible_at_ors_across_splits():
    """A T is feasible if ANY recorded split at that T was feasible, even if
    others were not."""
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 10, 'any_feasible': False},
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 11, 'any_feasible': True},
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 12, 'any_feasible': False},
    ])
    assert any_feasible_at(frame, M=25, k=9, T=2, arm='control') is True


def test_any_feasible_at_no_matching_rows_is_false():
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'control', 'split': 10, 'any_feasible': True},
    ])
    assert any_feasible_at(frame, M=999, k=999, T=999, arm='control') is False


def _non_monotone_frame(M=25, k=9, arm='control'):
    """T=1 infeasible, T=2 feasible, T=3 infeasible, T=4 feasible -- reach()
    must not assume monotonicity in T (spec 2.1)."""
    feasibility_by_T = {1: False, 2: True, 3: False, 4: True}
    rows = []
    for T, feasible in feasibility_by_T.items():
        for split in SPLIT_INDICES:
            rows.append({'M': M, 'k': k, 'T': T, 'arm': arm, 'split': split,
                         'any_feasible': feasible})
    return _frame(rows)


def test_reach_and_any_feasible_at_are_not_monotone_in_T():
    frame = _non_monotone_frame()
    assert reach(frame, M=25, k=9, arm='control') == 4
    assert any_feasible_at(frame, M=25, k=9, T=3, arm='control') is False
    assert any_feasible_at(frame, M=25, k=9, T=2, arm='control') is True
    assert any_feasible_at(frame, M=25, k=9, T=1, arm='control') is False
    assert any_feasible_at(frame, M=25, k=9, T=4, arm='control') is True


def test_reach_ors_across_splits():
    """reach() must credit a T as feasible even when only one of the three
    recorded splits at that T was feasible."""
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 1, 'arm': 'control', 'split': 10, 'any_feasible': False},
        {'M': 25, 'k': 9, 'T': 1, 'arm': 'control', 'split': 11, 'any_feasible': False},
        {'M': 25, 'k': 9, 'T': 1, 'arm': 'control', 'split': 12, 'any_feasible': True},
    ])
    assert reach(frame, M=25, k=9, arm='control') == 1


def test_reach_returns_none_when_nothing_feasible_anywhere():
    rows = [{'M': 25, 'k': 9, 'T': T, 'arm': 'control', 'split': split,
             'any_feasible': False}
            for T in T_VALUES for split in SPLIT_INDICES]
    frame = _frame(rows)
    assert reach(frame, M=25, k=9, arm='control') is None


def test_reach_returns_none_for_a_cell_arm_with_no_rows_at_all():
    frame = _frame([
        {'M': 25, 'k': 9, 'T': 1, 'arm': 'control', 'split': 10, 'any_feasible': True},
    ])
    assert reach(frame, M=999, k=999, arm='control') is None


# --------------------------------------------------------------------------
# load_cell_features
# --------------------------------------------------------------------------

def test_load_cell_features_reads_the_archived_joint_d000_row():
    """M=25, k=9, split=10 is a real, confirmed-present row in
    results/campaign_backup_20260825/rf_t11_d14_M25_joint-d000.csv."""
    feat_app, feat_ddos = load_cell_features(
        feasibility_frontier.CAMPAIGN_BACKUP_DIR, M=25, k=9, split_idx=10)

    assert feat_app == (
        'Fwd.Packet.Length.Max;Fwd.Packet.Length.Min;Fwd.Packet.Length.Mean;'
        'Bwd.Packet.Length.Max;Bwd.Packet.Length.Min;Bwd.Packet.Length.Mean;'
        'Flow.IAT.Min;Bwd.IAT.Mean;Packet.Length.Mean')
    assert feat_ddos == feat_app
    # k=9 selected features: exactly 9 ';'-joined names each side.
    assert len(feat_app.split(';')) == 9
    assert len(feat_ddos.split(';')) == 9


def test_load_cell_features_unknown_split_raises():
    with pytest.raises(ValueError, match='no archived row'):
        load_cell_features(
            feasibility_frontier.CAMPAIGN_BACKUP_DIR, M=25, k=9, split_idx=999)


def test_load_cell_features_unknown_k_raises():
    with pytest.raises(ValueError, match='no archived row'):
        load_cell_features(
            feasibility_frontier.CAMPAIGN_BACKUP_DIR, M=25, k=999, split_idx=10)


# --------------------------------------------------------------------------
# already_done
# --------------------------------------------------------------------------

def test_already_done_empty_when_file_does_not_exist(tmp_path):
    out_path = str(tmp_path / 'does_not_exist.csv')
    assert already_done(out_path) == set()


def test_already_done_reads_the_five_key_columns_as_tuples(tmp_path):
    out_path = str(tmp_path / 'results.csv')
    frame = pd.DataFrame([
        {'M': 25, 'k': 9, 'T': 1, 'arm': 'control', 'split': 10,
         'any_feasible': True},
        {'M': 25, 'k': 9, 'T': 2, 'arm': 'aligned_only', 'split': 11,
         'any_feasible': False},
    ])
    frame.to_csv(out_path, index=False)

    assert already_done(out_path) == {
        (25, 9, 1, 'control', 10),
        (25, 9, 2, 'aligned_only', 11),
    }


def test_already_done_ignores_a_torn_row_with_null_any_feasible(tmp_path):
    """Code review Finding 6: pd.to_csv(mode='a') isn't atomic, so a crash
    mid-write can leave a torn, truncated trailing row -- pd.read_csv pads
    its missing trailing column(s) with NaN, even though the row's key
    columns (M/k/T/arm/split) are still intact. Such a row must NOT count as
    "done": it never actually completed cleanly, so it must be retried, not
    permanently poison the resume checkpoint."""
    out_path = str(tmp_path / 'torn.csv')
    with open(out_path, 'w') as f:
        f.write('M,k,T,arm,split,any_feasible\n')
        f.write('25,9,1,control,10,True\n')
        f.write('25,9,2,control,11,\n')  # torn: any_feasible truncated away

    assert already_done(out_path) == {(25, 9, 1, 'control', 10)}


# --------------------------------------------------------------------------
# collect: resume + parallelism
#
# _fake_run_one_point is a real MODULE-LEVEL function (not a closure, and not
# a monkeypatch applied in place to feasibility_frontier.run_one_point)
# because collect() ships it to ProcessPoolExecutor workers, which on this
# platform are separate spawned processes: pickle serializes a function by
# (module, qualname) reference, not by value, so the worker re-imports THIS
# test module and looks up this exact name. A closure has no importable
# qualname and cannot be pickled at all; and even though the test below does
# `monkeypatch.setattr(feasibility_frontier, 'run_one_point',
# _fake_run_one_point)`, what gets pickled at the executor.submit() call
# site is this function object itself (its own __module__/__qualname__ are
# unaffected by being assigned to a *different* name on a *different*
# module), so the worker finds the real stand-in below rather than silently
# falling back to the genuine (expensive, Optuna-calling) run_one_point.
# Verified empirically (see test_collect_parallel_actually_uses_worker_processes)
# by having each row record os.getpid() and confirming it differs from the
# test process's own pid.
#
# Code review Finding 2: collect() no longer passes the dataset as a per-call
# run_one_point argument through the pool -- it loads it once and hands it to
# ProcessPoolExecutor's `initializer` (_init_worker), which sets each worker
# process's own _worker_data global exactly once. So collect() must now
# submit run_one_point with data=None, and _fake_run_one_point asserts
# exactly that (data is None) -- a regression back to piping the whole
# dataset through per-call would fail this assertion in every test below
# that uses it. It also reports whether THIS (genuinely separate, freshly
# re-imported) worker process's feasibility_frontier._worker_data matches
# what _fake_load_campaign_data() would produce, confirming the initializer
# really did run and really did set the global inside the worker -- not just
# that data=None was passed.
# --------------------------------------------------------------------------

def _fake_run_one_point(point, data, campaign_dir):
    assert data is None, (
        'collect() must submit run_one_point through the pool with data=None '
        'and rely on the ProcessPoolExecutor initializer instead of piping '
        'the (real, ~95MB) dataset through as a per-call argument (Finding 2)')
    row = dict(point)
    row.update({'any_feasible': True, 'n_trials_run': 1, 'n_feasible': 1,
                'dominant_violation_type': None, 'pid': os.getpid(),
                'worker_data_is_expected':
                    feasibility_frontier._worker_data == _fake_load_campaign_data()})
    return row


def _fake_load_campaign_data():
    return (None, None, None, None, None)


def test_collect_resume_skips_already_done_points_and_appends_only_new_rows(
        tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    out_path = str(tmp_path / 'resume.csv')

    frame1, _started1, n_done1, _workers1 = collect(
        'unused_campaign_dir', out_path, limit=3, max_workers=1)
    assert n_done1 == 3
    assert len(frame1) == 3

    # Second invocation, same --out: must skip the 3 points already recorded
    # and process the NEXT 3 (not re-select the first 3).
    frame2, _started2, n_done2, _workers2 = collect(
        'unused_campaign_dir', out_path, limit=3, max_workers=1)
    assert n_done2 == 3
    assert len(frame2) == 6

    keys = list(zip(frame2['M'], frame2['k'], frame2['T'], frame2['arm'], frame2['split']))
    assert len(keys) == len(set(keys)), 'duplicate (M,k,T,arm,split) rows across both invocations'


def test_collect_with_no_remaining_points_is_a_noop(tmp_path, monkeypatch):
    """Re-running collect() once every grid point is already recorded prints
    a nothing-to-do message and returns n_done=0, without touching --out."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:2])
    out_path = str(tmp_path / 'exhausted.csv')

    frame1, _started1, n_done1, _workers1 = collect('unused_campaign_dir', out_path, max_workers=1)
    assert n_done1 == 2
    assert len(frame1) == 2

    frame2, _started2, n_done2, _workers2 = collect('unused_campaign_dir', out_path, max_workers=1)
    assert n_done2 == 0
    assert len(frame2) == 2


def test_collect_parallel_actually_uses_worker_processes(tmp_path, monkeypatch):
    """max_workers=2 against 4 synthetic points: exactly 4 rows come back, no
    duplicates, and at least one row's pid differs from this test process's
    own pid -- confirming run_one_point genuinely executed in a separate
    ProcessPoolExecutor worker rather than silently degrading to serial,
    in-process execution.

    Also confirms Finding 2's initializer-based data passing genuinely works
    across those real, separate worker processes: every row's
    worker_data_is_expected (computed inside its own worker, from that
    worker's own feasibility_frontier._worker_data global) must be True."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:4])
    out_path = str(tmp_path / 'parallel.csv')

    frame, _started, n_done, resolved_max_workers = collect(
        'unused_campaign_dir', out_path, max_workers=2)

    assert n_done == 4
    assert len(frame) == 4
    assert resolved_max_workers == 2
    keys = list(zip(frame['M'], frame['k'], frame['T'], frame['arm'], frame['split']))
    assert len(keys) == len(set(keys))
    assert frame['worker_data_is_expected'].all(), (
        'a worker process never saw the dataset the ProcessPoolExecutor '
        'initializer was supposed to set into its _worker_data global')

    this_pid = os.getpid()
    worker_pids = set(frame['pid'])
    assert all(pid != this_pid for pid in worker_pids), (
        'a row ran in the test process itself, not a ProcessPoolExecutor worker')


# --------------------------------------------------------------------------
# collect: per-future exception isolation (code review Finding 1)
#
# _fake_run_one_point_with_one_failure is, like _fake_run_one_point above, a
# real module-level function (not a closure) for the same picklability reason
# documented above it: ProcessPoolExecutor workers on this platform are
# spawned, fresh-imported processes, so only a by-reference-importable
# function survives being shipped to one.
# --------------------------------------------------------------------------

def _fake_run_one_point_with_one_failure(point, data, campaign_dir):
    """Raises for exactly one grid point (arm='aligned_only', split=10) and
    delegates to _fake_run_one_point for every other point -- lets a test
    confirm that one bad point doesn't abort collection of the rest."""
    if point['arm'] == 'aligned_only' and point['split'] == 10:
        raise RuntimeError('synthetic failure for point {}'.format(point))
    return _fake_run_one_point(point, data, campaign_dir)


def test_collect_isolates_a_single_point_exception_and_continues(tmp_path, monkeypatch, capsys):
    """A run_one_point failure on exactly one grid point must not abort
    collection of the rest -- mirrors src/training/feature_selection.py's
    compare_feature_selection_approaches_parallel (~line 783-788), which
    wraps `future.result()` in try/except, logs which split failed, and
    `continue`s so the ProcessPoolExecutor's `with` block still completes
    and the other futures' results still get written, rather than letting
    the exception propagate out of the as_completed loop and having
    __exit__'s `shutdown(wait=True)` discard every other in-flight future's
    result. The failed point should simply be ABSENT from the output (it
    stays outside already_done()'s set and is retried on the next
    invocation -- the same resumability collect() already provides for a
    crash/interrupt).

    Also (code review Finding 4): the per-future exception log must include
    the exception TYPE and a full traceback (not just str(e), which for e.g.
    a bare KeyError('study') would print as just the word "study" -- useless
    10 hours later with no other context), and collect() must print a
    one-line summary of how many points failed this invocation."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point_with_one_failure)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:4])
    out_path = str(tmp_path / 'partial_failure.csv')

    frame, _started, n_done, _resolved_max_workers = collect(
        'unused_campaign_dir', out_path, max_workers=2)

    # build_grid()[:4] for CELLS[0] == (25, 9): 3 'control' points (one per
    # split) then the first 'aligned_only' point (split=10), which is the one
    # rigged to fail -- so exactly 3 of the 4 submitted points should survive.
    assert n_done == 3
    assert len(frame) == 3
    keys = list(zip(frame['M'], frame['k'], frame['T'], frame['arm'], frame['split']))
    assert (25, 9, 1, 'aligned_only', 10) not in keys
    assert len(keys) == len(set(keys))

    captured = capsys.readouterr()
    assert 'RuntimeError' in captured.out, 'exception TYPE must be logged, not just str(e)'
    assert 'Traceback (most recent call last)' in captured.out, 'full traceback must be logged'
    assert '1 points failed this run' in captured.out
    assert 'retried on the next invocation' in captured.out


# --------------------------------------------------------------------------
# collect: resolved max_workers exposed to callers (code review Finding 2)
# --------------------------------------------------------------------------

def test_collect_returns_the_auto_computed_worker_count_when_max_workers_omitted(
        tmp_path, monkeypatch):
    """When max_workers=None (the --max-workers-omitted default-use case),
    collect() must return the worker count it actually resolved to and used
    -- min(len(points), max(1, os.cpu_count() - 1)) -- not just echo back the
    None it was passed. Patches os.cpu_count (via the feasibility_frontier
    module's own `os` reference, restored by monkeypatch) to a small,
    deterministic value so the expected worker count differs from both 1 and
    len(points), ruling out a fix that accidentally still hardcodes either."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:4])
    monkeypatch.setattr(feasibility_frontier.os, 'cpu_count', lambda: 3)
    out_path = str(tmp_path / 'auto_workers.csv')

    frame, _started, n_done, resolved_max_workers = collect(
        'unused_campaign_dir', out_path, max_workers=None)

    assert n_done == 4
    assert len(frame) == 4
    assert resolved_max_workers == 2  # min(4 points, max(1, 3 - 1)) == 2
    assert resolved_max_workers != 1
    assert resolved_max_workers != len(frame)


def test_main_timing_reports_the_resolved_worker_count_when_max_workers_omitted(
        tmp_path, monkeypatch, capsys):
    """main()'s --timing line used to compute `args.max_workers or 1`, which
    is always 1 whenever --max-workers is omitted (the documented default
    use case) since collect() resolves its own worker count internally and
    main() never learned it. Stubs collect() itself so this test is a pure
    main()-level check on the printed message, independent of the
    lower-level collect() test above."""
    fake_frame = pd.DataFrame([{'any_feasible': True}])

    def _fake_collect(campaign_dir, out, limit, max_workers):
        assert max_workers is None  # --max-workers omitted on the CLI
        return fake_frame, 1000.0, 4, 3  # started, n_done=4, resolved_max_workers=3

    monkeypatch.setattr(feasibility_frontier, 'collect', _fake_collect)
    monkeypatch.setattr(feasibility_frontier, 'report', lambda frame: None)
    monkeypatch.setattr(feasibility_frontier.time, 'time', lambda: 1040.0)

    out_path = str(tmp_path / 'main_out.csv')
    feasibility_frontier.main(['--out', out_path, '--timing'])

    captured = capsys.readouterr()
    assert 'at 3 workers' in captured.out
    assert 'at 1 workers' not in captured.out


# --------------------------------------------------------------------------
# collect: --limit 0 means "run nothing" (code review Finding 7)
# --------------------------------------------------------------------------

def test_collect_limit_zero_runs_nothing(tmp_path, monkeypatch, capsys):
    """--limit 0 must be indistinguishable from "run nothing", not from
    "--limit omitted" -- `if limit:` (truthiness) cannot tell 0 apart from
    None, so it would silently run the full remaining grid instead. Uses the
    normal (successful, not raising) _fake_run_one_point, so a regression
    back to truthiness-based limit handling would genuinely make all 4
    synthetic points run and get recorded, failing the assertions below for
    real rather than incidentally via an unrelated exception."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:4])
    out_path = str(tmp_path / 'limit_zero.csv')

    frame, _started, n_done, _resolved_max_workers = collect(
        'unused_campaign_dir', out_path, limit=0, max_workers=1)

    assert n_done == 0
    assert len(frame) == 0
    assert not os.path.exists(out_path), '--limit 0 must not write anything to --out'
    captured = capsys.readouterr()
    assert 'nothing to do' in captured.out


# --------------------------------------------------------------------------
# collect: every submitted point failing this invocation (code review
# Finding 8) -- out is never created, so the final read must not raise a
# bare FileNotFoundError.
# --------------------------------------------------------------------------

def _fake_run_one_point_always_fails(point, data, campaign_dir):
    raise RuntimeError('synthetic failure for point {}'.format(point))


def test_collect_returns_empty_frame_when_every_point_fails_and_out_never_created(
        tmp_path, monkeypatch, capsys):
    """If every future raised this invocation (Finding 4's per-future
    exception handling catches each one), `out` is never created -- the final
    `pd.read_csv(out)` must be guarded the same way the nothing-to-do
    early-return path already guards it, returning an empty DataFrame instead
    of a bare, unguarded FileNotFoundError."""
    monkeypatch.setattr(feasibility_frontier, 'run_one_point', _fake_run_one_point_always_fails)
    monkeypatch.setattr(feasibility_frontier, 'load_campaign_data', _fake_load_campaign_data)
    monkeypatch.setattr(feasibility_frontier, 'build_grid', lambda: build_grid()[:2])
    out_path = str(tmp_path / 'all_failed.csv')

    frame, _started, n_done, _resolved_max_workers = collect(
        'unused_campaign_dir', out_path, max_workers=1)

    assert n_done == 0
    assert len(frame) == 0
    assert not os.path.exists(out_path)
    captured = capsys.readouterr()
    assert '2 points failed this run' in captured.out


# --------------------------------------------------------------------------
# report(): the genuinely-empty, column-less frame that collect() can now
# return (Finding 8) must not crash print_reach_table/print_violation_
# breakdown, which both index frame['M'] etc. Discovered live: the first
# real Codespace run hit exactly this path (every point failed because
# results/campaign_backup_20260825/ hadn't been copied yet) and main()
# crashed with KeyError: 'M' inside report() after printing the graceful
# "N points failed" message -- the crash happened one line further down
# than any existing test looked.
# --------------------------------------------------------------------------

def test_report_on_a_genuinely_empty_frame_does_not_raise(capsys):
    feasibility_frontier.report(pd.DataFrame())
    captured = capsys.readouterr()
    assert 'no rows to report' in captured.out


# --------------------------------------------------------------------------
# print_violation_breakdown: per-(M, k, T, arm) grouping, not per-row
# (code review Finding 5)
# --------------------------------------------------------------------------

def _violation_row(M, k, T, arm, split, any_feasible, dominant_violation_type):
    return {'M': M, 'k': k, 'T': T, 'arm': arm, 'split': split,
            'any_feasible': any_feasible,
            'dominant_violation_type': dominant_violation_type}


def test_print_violation_breakdown_counts_each_infeasible_cell_once(capsys):
    """A cell infeasible on all 3 of its recorded splits must contribute
    exactly ONE count to the table, not 3 -- spec 2.1/F6 wants one count per
    infeasible (M, k, T, arm), not per raw row."""
    rows = [
        _violation_row(25, 9, 1, 'control', 10, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 11, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 12, False, 'blocks_violation'),
    ]
    frame = pd.DataFrame(rows)
    print_violation_breakdown(frame)
    captured = capsys.readouterr()
    assert '| blocks_violation | 1 |' in captured.out


def test_print_violation_breakdown_excludes_a_cell_feasible_on_any_split(capsys):
    """A cell feasible on even one of its 3 splits is not "infeasible" at all
    (any_feasible_at/reach OR across splits) -- it must not appear in the F6
    breakdown, even though 2 of its 3 rows individually recorded
    any_feasible=False."""
    rows = [
        _violation_row(25, 9, 1, 'control', 10, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 11, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 12, True, None),
    ]
    frame = pd.DataFrame(rows)
    print_violation_breakdown(frame)
    captured = capsys.readouterr()
    assert 'no infeasible points in this run' in captured.out


def test_print_violation_breakdown_counts_two_distinct_infeasible_cells_separately(capsys):
    """Two distinct (M, k, T, arm) cells, each infeasible on all 3 splits with
    a different dominant violation type, must produce two separate rows in
    the table -- 1 count each, not merged."""
    rows = (
        [_violation_row(25, 9, 1, 'control', s, False, 'blocks_violation')
         for s in SPLIT_INDICES]
        + [_violation_row(25, 9, 2, 'control', s, False, 'codeword_violation')
           for s in SPLIT_INDICES]
    )
    frame = pd.DataFrame(rows)
    print_violation_breakdown(frame)
    captured = capsys.readouterr()
    assert '| blocks_violation | 1 |' in captured.out
    assert '| codeword_violation | 1 |' in captured.out


def test_print_violation_breakdown_does_not_crash_on_a_torn_row(capsys):
    """A torn/corrupted row (NaN any_feasible, from a crash mid pd.to_csv
    append -- see already_done's fix for Finding 6) must not crash this table
    with a TypeError on `~` against an object-dtype column; grouping and
    `.any()`-based feasibility must tolerate it."""
    import numpy as np
    rows = [
        _violation_row(25, 9, 1, 'control', 10, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 11, False, 'blocks_violation'),
        _violation_row(25, 9, 1, 'control', 12, np.nan, np.nan),
    ]
    frame = pd.DataFrame(rows)
    print_violation_breakdown(frame)  # must not raise
    captured = capsys.readouterr()
    assert '| blocks_violation | 1 |' in captured.out
