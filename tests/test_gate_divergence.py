"""The measurement that justifies the 2026-09-07 cost-model repair: how often
the superseded band gate and the corrected block gate disagree."""
import pandas as pd

from src.reporting.gate_divergence import score_gate_divergence


def _row(codeword_before, codeword_after, blocks_before, blocks_after,
         spent=True, policy='aligned', blocks_floor=1):
    return {'policy': policy, 'align_spent_budget': spent,
            'align_codeword_before': codeword_before,
            'align_codeword_after': codeword_after,
            'align_blocks_before': blocks_before,
            'align_blocks_after': blocks_after,
            'align_blocks_floor': blocks_floor}


def test_a_band_crossing_that_buys_no_block_is_a_false_positive():
    """§1.4's first failure: the run crosses a 44-bit band, the crossbar arm
    still binds, crossed_a_boundary returns True, the rollback declines to
    fire, and the accuracy is spent for nothing."""
    # 88 -> 40 crosses a band (band_factor 3 -> 1); the factor does not move.
    out = score_gate_divergence(pd.DataFrame([_row(88, 40, 6, 6)]))
    assert out['false_positive'] == 1
    assert out['false_negative'] == 0


def test_a_block_saving_without_a_band_crossing_is_a_false_negative():
    """§1.4's second failure: a move drops some feature's own ceil(w/8), saves
    a real block, crosses no band -- so the run is discarded and re-run at
    delta = 0, throwing the real saving away."""
    out = score_gate_divergence(pd.DataFrame([_row(88, 86, 6, 5)]))
    assert out['false_negative'] == 1
    assert out['false_positive'] == 0


def test_rows_that_never_spent_budget_are_excluded():
    """The rollback is unreachable when spent_budget is False -- _run_one_arm
    returns the speculative result before consulting crossed_a_boundary -- so
    a free-moves row was never judged by either gate and would inflate both
    counts."""
    out = score_gate_divergence(pd.DataFrame([_row(88, 40, 6, 6, spent=False)]))
    assert out['n'] == 0
    assert out['false_positive'] == 0


def test_unaligned_control_rows_are_excluded():
    """policy='none' skips align_with_policy entirely and carries no align_*
    columns in a real replay; excluding it here matches replay_scoring."""
    out = score_gate_divergence(pd.DataFrame([_row(88, 40, 6, 6, policy='none')]))
    assert out['n'] == 0


def test_the_gate_can_open_only_where_the_floor_is_strictly_cheaper():
    """§8's risk, answered directly: BlockBudget.spending() is
    factor(current) > factor(floor), so a pair already at its floor can never
    authorise spending however generous delta is."""
    frame = pd.DataFrame([_row(88, 88, 6, 6, blocks_floor=6),
                          _row(88, 88, 6, 6, blocks_floor=4)])
    assert score_gate_divergence(frame)['gate_can_open'] == 1
