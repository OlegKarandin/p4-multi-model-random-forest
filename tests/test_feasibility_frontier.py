"""Unit tests for scripts/feasibility_frontier.py's pure, module-level
functions (Task 3): build_grid, cfg_for_arm, trial_violation_type,
summarize_trials, any_feasible_at, reach, load_cell_features.

Per spec 2.1's task-4 scope, this does NOT run the script end-to-end (that's
a later gate) -- only the side-effect-free helpers that don't touch Optuna,
sklearn, or the campaign search itself.
"""
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
    any_feasible_at,
    build_grid,
    cfg_for_arm,
    load_cell_features,
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
    user_attrs at all -- must read as feasible-shaped (None), not crash."""
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
