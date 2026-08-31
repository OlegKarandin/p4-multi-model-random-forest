import pandas as pd
import pytest

from src.reporting import replay_scoring as rs


def _replay_frame():
    """Two model pairs, an unaligned control row and an aligned row each.
    Pair 0 crosses a band (90 -> 84, factor 3 -> 2); pair 1 sheds five bits
    and crosses nothing."""
    rows = []
    for pair, (before, after_aligned) in enumerate([(90, 84), (100, 95)]):
        for policy, after, blocks in (('none', before, 30),
                                      ('aligned', after_aligned, 28)):
            rows.append({'source_arm': 'joint-d020', 'M': 25, 'split': 10,
                         'k': pair, 'overlap_threshold': 0.5, 'policy': policy,
                         'align_codeword_before': before,
                         'align_codeword_after': after, 'blocks': blocks,
                         'acc_app': 0.90, 'acc_ddos': 0.95})
    return pd.DataFrame(rows)


def test_derive_columns_computes_the_band_factors_and_wasted_bits():
    out = rs.derive_columns(_replay_frame())
    first = out[(out['policy'] == 'aligned') & (out['k'] == 0)].iloc[0]
    assert first['factor_before'] == 3      # 90 + 4 = 94 -> 3
    assert first['factor_after'] == 2       # 84 + 4 = 88 -> 2
    assert first['bits_shed'] == 6
    assert first['wasted_bits'] == 0        # the band was crossed


def test_wasted_bits_counts_every_bit_when_no_band_is_crossed():
    out = rs.derive_columns(_replay_frame())
    second = out[(out['policy'] == 'aligned') & (out['k'] == 1)].iloc[0]
    assert second['factor_before'] == second['factor_after']
    assert second['wasted_bits'] == second['bits_shed']


def test_s5_uses_the_tightest_swept_threshold_as_the_reference_not_literal_0_5():
    """The reference ("tight") threshold must be derived from the data --
    the maximum swept overlap_threshold -- not require the literal value 0.5
    to be present in the sweep."""
    rows = []
    for threshold, after, blocks in ((0.3, 90, 28), (0.7, 96, 28)):
        rows.append({'source_arm': 'joint-d020', 'M': 25, 'split': 10,
                     'k': 0, 'overlap_threshold': threshold, 'policy': 'aligned',
                     'align_codeword_before': 100,
                     'align_codeword_after': after, 'blocks': blocks,
                     'acc_app': 0.90, 'acc_ddos': 0.95})
    frame = pd.DataFrame(rows)
    verdict = rs.score(rs.derive_columns(frame))
    assert verdict['S5']['passed'] is True
    assert '0.3' in verdict['S5']['detail']
    assert '0.7' in verdict['S5']['detail']
    assert '0.5' not in verdict['S5']['detail']


def test_s5_reports_no_data_when_only_one_threshold_was_swept():
    """A genuinely single-threshold sweep has nothing to compare against --
    that must still fail honestly, distinct from the reference simply not
    being 0.5."""
    frame = _replay_frame()  # every row uses overlap_threshold=0.5 only
    verdict = rs.score(rs.derive_columns(frame))
    assert verdict['S5']['passed'] is False
    assert 'no loosened-threshold rows to compare' in verdict['S5']['detail']


def test_derive_columns_drops_none_policy_rows_without_crashing():
    """policy='none' rows carry no align_* columns (run_one_policy skips
    align_with_policy for 'none') -- band_factor(NaN) must not be reached,
    and the aligned rows' scoring must be unaffected by 'none' rows being
    present in the input frame."""
    frame = _replay_frame()
    none_rows = pd.DataFrame([
        {'source_arm': 'joint-d020', 'M': 25, 'split': 10, 'k': pair,
         'overlap_threshold': 0.5, 'policy': 'none',
         'align_codeword_before': float('nan'),
         'align_codeword_after': float('nan'), 'blocks': 30,
         'acc_app': 0.90, 'acc_ddos': 0.95}
        for pair in (0, 1)
    ])
    mixed = pd.concat([frame, none_rows], ignore_index=True)

    out = rs.derive_columns(mixed)
    assert 'none' not in out['policy'].unique()

    verdict = rs.score(out)
    assert verdict == rs.score(rs.derive_columns(frame))


def test_score_refuses_a_multi_objective_frame():
    """Task 6 made 'objective' part of a replay row's identity -- three can
    appear per model pair in one CSV (Task 7's committed replay data). S3 and
    S5 key only on PAIR_KEYS, which does not name 'objective', so handing
    score() a multi-objective frame must raise loudly rather than silently
    triple-counting bits_shed / fanning out the S5 self-merge."""
    blocks = _replay_frame()
    blocks['objective'] = 'blocks'
    stages = _replay_frame()
    stages['objective'] = 'stages'
    frame = pd.concat([blocks, stages], ignore_index=True)

    with pytest.raises(ValueError, match='objective'):
        rs.score(rs.derive_columns(frame))


def test_score_accepts_a_frame_with_a_single_objective_value():
    """The guard must not fire on a frame that merely carries the 'objective'
    column -- only on one with more than one distinct value in it."""
    frame = _replay_frame()
    frame['objective'] = 'blocks'
    verdict = rs.score(rs.derive_columns(frame))
    assert verdict == rs.score(rs.derive_columns(_replay_frame()))
