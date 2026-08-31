"""E2-E7 scoring, on synthetic frames -- no fitting, so the arithmetic can be
debugged without repeating a ten-minute run."""
import pandas as pd
import pytest

import scripts.score_objective_replay as so


def _row(objective, **kw):
    base = {'policy': 'aligned', 'objective': objective, 'source_arm': 'joint-d020',
            'M': 100, 'split': 10, 'k': 13, 'overlap_threshold': 0.5,
            'joint_stage_depth': 11, 'joint_blocks': 60,
            'acc_app': 0.90, 'acc_ddos': 0.95,
            'align_key_bytes_before': 35, 'align_key_bytes_after': 34,
            'align_key_bytes_floor': 26, 'align_ternary_stages_before': 6,
            'align_ternary_stages_after': 6, 'align_stage_target': 32.0,
            'align_bits_to_reach': 9.0, 'align_codeword_before': 234,
            'align_codeword_after': 216}
    base.update(kw)
    return base


def test_e2_counts_cells_where_stages_lowers_stage_depth():
    frame = pd.DataFrame([
        _row('blocks', joint_stage_depth=11),
        _row('stages', joint_stage_depth=8),
        _row('blocks', k=17, joint_stage_depth=9),
        _row('stages', k=17, joint_stage_depth=9),
    ])
    out = so.score_objectives(frame)
    assert out['E2']['value'] == 1
    assert out['E2']['n_cells'] == 2


def test_e2_is_zero_when_no_cell_moves_and_that_is_the_falsification():
    frame = pd.DataFrame([_row('blocks'), _row('stages')])
    out = so.score_objectives(frame)
    assert out['E2']['value'] == 0
    assert not out['E2']['passed']
    assert 'falsif' in out['E2']['detail'].lower()


def test_e4_reports_a_block_regression_rather_than_hiding_it():
    """A result that trades 1 stage for 3 blocks is a trade-off to report,
    not a win."""
    frame = pd.DataFrame([
        _row('blocks', joint_blocks=60, joint_stage_depth=11),
        _row('stages', joint_blocks=63, joint_stage_depth=8),
    ])
    out = so.score_objectives(frame)
    assert out['E4']['value'] == 3
    assert not out['E4']['passed']


def test_e7_counts_cells_a_tree_blind_target_would_have_spent_in():
    """Scored against the design's own population: 18 of 25 cells save nothing
    at the floor. A cell where the T-blind target is reachable and the T-aware
    one is None is a cell where tolerance would have bought nothing."""
    frame = pd.DataFrame([
        _row('stages', align_key_bytes_before=11, align_key_bytes_floor=7,
             align_ternary_stages_before=1, align_stage_target=float('nan')),
        _row('stages', k=17, align_key_bytes_before=35, align_stage_target=32.0),
    ])
    out = so.score_objectives(frame, n_tables=4)
    assert out['E7']['value'] == 1


def test_pairing_drops_a_cell_missing_one_objective():
    """The whole reason the replay exists: never pool across pairs."""
    frame = pd.DataFrame([_row('blocks'), _row('stages'), _row('both', k=17)])
    out = so.score_objectives(frame)
    assert out['E2']['n_cells'] == 1
    assert out['E6']['n_cells'] == 0
