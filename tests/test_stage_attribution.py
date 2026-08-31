import pandas as pd
import pytest

from src.reporting import stage_attribution as sa


def _row(source_arm='joint-d020', M=25, split=10, k=17, overlap_threshold=0.5,
        policy='none', register_depth=(4, 4), range_depth=(1, 1),
        ternary_depth=(6, 6)):
    """One replay row's decomposition columns. stage_depth is derived the
    same way multi_model_memory_evaluation derives it -- max(range, ternary)
    -- so fixtures can't accidentally violate the real invariant."""
    j_reg, cf_reg = register_depth
    j_range, cf_range = range_depth
    j_tern, cf_tern = ternary_depth
    return {'source_arm': source_arm, 'M': M, 'split': split, 'k': k,
           'overlap_threshold': overlap_threshold, 'policy': policy,
           'joint_register_depth': j_reg,
           'counterfactual_disjoint_register_depth': cf_reg,
           'joint_range_depth': j_range,
           'counterfactual_disjoint_range_depth': cf_range,
           'joint_ternary_depth': j_tern,
           'counterfactual_disjoint_ternary_depth': cf_tern,
           'joint_stage_depth': max(j_range, j_tern),
           'counterfactual_disjoint_stage_depth': max(cf_range, cf_tern)}


def test_derive_columns_satisfies_the_encoding_identity_exactly():
    # Spec Sec 5.1: Δ_encoding = Δ_range + Δ_ternary_spill, EXACT, not
    # approximate -- the whole point of decomposing rather than reading
    # stage_depth alone.
    frame = pd.DataFrame([_row(range_depth=(2, 3), ternary_depth=(6, 7))])
    out = sa.derive_columns(frame)
    row = out.iloc[0]
    assert row['delta_range'] == 1        # 3 - 2
    assert row['delta_ternary_spill'] == 0   # (7-3) - (6-2) == 4 - 4
    assert row['delta_encoding'] == 1        # 7 - 6
    assert row['delta_range'] + row['delta_ternary_spill'] == row['delta_encoding']


def test_d1_passes_when_register_depth_agrees_on_every_row():
    frame = pd.DataFrame([_row(), _row(k=2)])
    verdict = sa.score(frame)
    assert verdict['D1']['passed'] is True
    assert verdict['D1']['n'] == 2
    assert verdict['D1']['n_mismatched'] == 0


def test_d1_fails_and_reports_the_mismatch_count():
    frame = pd.DataFrame([_row(), _row(k=2, register_depth=(4, 5))])
    verdict = sa.score(frame)
    assert verdict['D1']['passed'] is False
    assert verdict['D1']['n_mismatched'] == 1


def test_score_does_not_compute_d2_d3_d4_when_d1_fails():
    frame = pd.DataFrame([_row(register_depth=(4, 5))])
    verdict = sa.score(frame)
    assert verdict['D2']['n_cells'] == 0
    assert 'D1 failed' in verdict['D2']['detail']
    assert verdict['D3']['n_cells'] == 0
    assert verdict['D4'] == {}


def test_d2_is_reported_per_cell_and_not_pooled():
    # Cell (M=25, k=17): mean delta_encoding = +2 -> passes.
    # Cell (M=25, k=2):  mean delta_encoding = -0.5 -> fails.
    # A pooled mean across all four rows would be (+1+3-1+0)/4 = +0.75,
    # a false pass -- this test pins that D2 does NOT do that.
    rows = [
        _row(k=17, split=10, ternary_depth=(6, 7)),   # delta +1
        _row(k=17, split=11, ternary_depth=(5, 8)),   # delta +3
        _row(k=2, split=10, ternary_depth=(6, 5)),    # delta -1
        _row(k=2, split=11, ternary_depth=(6, 6)),    # delta 0
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)
    cells = verdict['D2']['cells'].set_index(['M', 'k'])
    assert cells.loc[(25, 17), 'passed'] == True
    assert cells.loc[(25, 17), 'n'] == 2
    assert cells.loc[(25, 2), 'passed'] == False
    assert verdict['D2']['fraction_passing'] == 0.5
    assert verdict['D2']['n_cells'] == 2


def test_d2_reports_no_rows_for_a_policy_that_was_never_swept():
    frame = pd.DataFrame([_row(policy='legacy')])
    verdict = sa.score(frame, policy='none')
    assert verdict['D2']['n_cells'] == 0
    assert 'no rows for policy' in verdict['D2']['detail']


def test_d3_requires_both_directions_fixed_in_advance():
    # Cell A: delta_range > 0 AND delta_ternary_spill < 0 on both rows -> passes.
    # Cell B: one row has delta_range < 0 -> fails.
    rows = [
        _row(k=17, split=10, range_depth=(2, 5), ternary_depth=(6, 6)),
        _row(k=17, split=11, range_depth=(2, 4), ternary_depth=(6, 6)),
        _row(k=2, split=10, range_depth=(2, 1), ternary_depth=(6, 7)),
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)
    cells = verdict['D3']['cells'].set_index(['M', 'k'])
    assert cells.loc[(25, 17), 'passed'] == True
    assert cells.loc[(25, 2), 'passed'] == False


def test_score_excludes_verify_rows_from_d1_count_and_from_d4_keys():
    """--verify injects an extra row with policy='verify' re-running
    'legacy' at the row's own recorded arm delta (not the ladder delta
    being swept). It must not inflate D1's paired-row count with a
    duplicate of the 'legacy' pair already present, and must never appear
    as its own key in D4 (which would compare it against the wrong
    alignment budget)."""
    rows = [
        _row(k=17, split=10, policy='none', ternary_depth=(8, 9)),
        _row(k=17, split=10, policy='legacy', ternary_depth=(6, 8)),
        _row(k=17, split=10, policy='verify', ternary_depth=(6, 8)),
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)
    assert verdict['D1']['n'] == 2
    assert 'verify' not in verdict['D4']
    assert set(verdict['D4']) == {'legacy'}


def test_d2_and_d3_dedupe_none_rows_replayed_at_multiple_overlap_thresholds():
    """policy='none' ignores overlap_threshold entirely (run_one_policy
    skips align_with_policy, the only place it's used, for 'none'), so
    replaying the SAME pair at two swept --overlap-thresholds values
    produces two byte-identical 'none' rows differing only in that column.
    The cell's n must count independent model pairs (2), not sweep cells
    (3)."""
    rows = [
        _row(k=17, split=10, overlap_threshold=0.3, policy='none',
            ternary_depth=(6, 7)),
        _row(k=17, split=10, overlap_threshold=0.7, policy='none',
            ternary_depth=(6, 7)),
        _row(k=17, split=11, overlap_threshold=0.5, policy='none',
            ternary_depth=(5, 8)),
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)
    d2_cells = verdict['D2']['cells'].set_index(['M', 'k'])
    d3_cells = verdict['D3']['cells'].set_index(['M', 'k'])
    assert d2_cells.loc[(25, 17), 'n'] == 2
    assert d3_cells.loc[(25, 17), 'n'] == 2


def test_d1_failure_returns_independent_d2_and_d3_objects():
    """D2 and D3 must not alias the same dict/DataFrame on the D1-failure
    path -- a caller mutating one must not corrupt the other."""
    frame = pd.DataFrame([_row(register_depth=(4, 5))])
    verdict = sa.score(frame)
    assert verdict['D2'] is not verdict['D3']
    assert verdict['D2']['cells'] is not verdict['D3']['cells']


def test_empty_cells_schema_matches_populated_cells_schema():
    """The D1-failure path's 'cells' frame must carry the same columns as a
    populated 'cells' frame (M, k, ..., n, passed) so callers can uniformly
    do result['cells']['passed'].sum() without a KeyError on empty data."""
    frame = pd.DataFrame([_row(register_depth=(4, 5))])
    verdict = sa.score(frame)
    assert 'passed' in verdict['D2']['cells'].columns
    assert verdict['D2']['cells']['passed'].sum() == 0


def test_d4_compares_none_against_every_aligned_policy_present():
    """stage_attribution is genuinely policy-name-agnostic (it takes the
    'none' label as a parameter and treats every other value as an aligned
    policy to compare against it), so this exercises D4's multi-label
    grouping logic in the abstract. 'aligned' is the one real policy name
    REPLAY_POLICIES can produce post-ladder-deletion; 'synthetic_other_arm'
    is a made-up second label with no real-world counterpart, used only to
    prove D4 keeps two distinct aligned-policy labels separate rather than
    merging them -- a scenario REPLAY_POLICIES = ('none', 'aligned') cannot
    itself produce, since there is now only one real aligned policy name.
    """
    rows = [
        # pair 1 (k=17): aligned shrinks joint depth by 2, cf by only 1 -> passes.
        _row(k=17, split=10, policy='none', ternary_depth=(8, 9)),
        _row(k=17, split=10, policy='aligned', ternary_depth=(6, 8)),
        # pair 2 (k=2): aligned changes nothing -> fails (delta_align == 0).
        _row(k=2, split=20, policy='none', ternary_depth=(5, 5)),
        _row(k=2, split=20, policy='aligned', ternary_depth=(5, 5)),
        # a second (synthetic, made-up) aligned policy label, present at only
        # one pair -- must appear as its own key in D4's output, not merged
        # with 'aligned'.
        _row(k=17, split=10, policy='synthetic_other_arm', ternary_depth=(7, 9)),
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)

    assert set(verdict['D4']) == {'aligned', 'synthetic_other_arm'}

    aligned_cells = verdict['D4']['aligned']['cells'].set_index(['M', 'k'])
    assert aligned_cells.loc[(25, 17), 'passed'] == True
    assert aligned_cells.loc[(25, 2), 'passed'] == False

    other_cells = verdict['D4']['synthetic_other_arm']['cells'].set_index(['M', 'k'])
    assert len(other_cells) == 1   # only the k=17 pair has a row to pair with


def test_score_refuses_a_multi_objective_frame():
    """Task 6 made 'objective' part of a replay row's identity -- three can
    appear per model pair in one CSV (Task 7's committed replay data).
    PAIR_KEYS does not name 'objective', so a multi-objective frame would
    triplicate every aligned pair in D1/D4 -- score() must raise loudly
    rather than silently pooling across it."""
    frame = pd.DataFrame([_row(), _row(k=2)])
    frame['objective'] = ['blocks', 'stages']
    with pytest.raises(ValueError, match='objective'):
        sa.score(frame)


def test_score_accepts_a_frame_with_a_single_objective_value():
    """The guard must not fire on a frame that merely carries the
    'objective' column -- only on one with more than one distinct value."""
    frame = pd.DataFrame([_row(), _row(k=2)])
    frame['objective'] = 'blocks'
    verdict = sa.score(frame)
    baseline = sa.score(pd.DataFrame([_row(), _row(k=2)]))
    assert verdict['D1'] == baseline['D1']
    assert verdict['D2']['fraction_passing'] == baseline['D2']['fraction_passing']
    assert verdict['D2']['n_cells'] == baseline['D2']['n_cells']
