import pandas as pd

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


def test_d4_compares_none_against_every_aligned_policy_present():
    rows = [
        # pair 1 (k=17): legacy shrinks joint depth by 2, cf by only 1 -> passes.
        _row(k=17, split=10, policy='none', ternary_depth=(8, 9)),
        _row(k=17, split=10, policy='legacy', ternary_depth=(6, 8)),
        # pair 2 (k=2): legacy changes nothing -> fails (delta_align == 0).
        _row(k=2, split=20, policy='none', ternary_depth=(5, 5)),
        _row(k=2, split=20, policy='legacy', ternary_depth=(5, 5)),
        # a second aligned policy, present at only one pair -- must appear
        # as its own key in D4's output, not merged with 'legacy'.
        _row(k=17, split=10, policy='c1', ternary_depth=(7, 9)),
    ]
    frame = pd.DataFrame(rows)
    verdict = sa.score(frame)

    assert set(verdict['D4']) == {'legacy', 'c1'}

    legacy_cells = verdict['D4']['legacy']['cells'].set_index(['M', 'k'])
    assert legacy_cells.loc[(25, 17), 'passed'] == True
    assert legacy_cells.loc[(25, 2), 'passed'] == False

    c1_cells = verdict['D4']['c1']['cells'].set_index(['M', 'k'])
    assert len(c1_cells) == 1   # only the k=17 pair has a 'c1' row to pair with
