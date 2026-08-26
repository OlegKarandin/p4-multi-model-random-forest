import pandas as pd

from src.reporting import replay_scoring as rs


def _replay_frame():
    """Two model pairs, two policies each. Pair 0 crosses a band (90 -> 84,
    factor 3 -> 2); pair 1 sheds five bits and crosses nothing."""
    rows = []
    for pair, (before, after_c1) in enumerate([(90, 84), (100, 95)]):
        for policy, after, blocks in (('legacy', before, 30),
                                      ('c1', after_c1, 28)):
            rows.append({'source_arm': 'joint-d020', 'M': 25, 'split': 10,
                         'k': pair, 'overlap_threshold': 0.5, 'policy': policy,
                         'align_codeword_before': before,
                         'align_codeword_after': after, 'blocks': blocks,
                         'acc_app': 0.90, 'acc_ddos': 0.95})
    return pd.DataFrame(rows)


def test_derive_columns_computes_the_band_factors_and_wasted_bits():
    out = rs.derive_columns(_replay_frame())
    first = out[(out['policy'] == 'c1') & (out['k'] == 0)].iloc[0]
    assert first['factor_before'] == 3      # 90 + 4 = 94 -> 3
    assert first['factor_after'] == 2       # 84 + 4 = 88 -> 2
    assert first['bits_shed'] == 6
    assert first['wasted_bits'] == 0        # the band was crossed


def test_wasted_bits_counts_every_bit_when_no_band_is_crossed():
    out = rs.derive_columns(_replay_frame())
    second = out[(out['policy'] == 'c1') & (out['k'] == 1)].iloc[0]
    assert second['factor_before'] == second['factor_after']
    assert second['wasted_bits'] == second['bits_shed']


def test_s1_fails_when_c1_loses_accuracy_against_the_zero_delta_baseline():
    frame = _replay_frame()
    frame.loc[frame['policy'] == 'c1', 'acc_ddos'] = 0.80
    verdict = rs.score(rs.derive_columns(frame))
    assert verdict['S1']['passed'] is False


def test_s1_passes_when_accuracy_is_held():
    verdict = rs.score(rs.derive_columns(_replay_frame()))
    assert verdict['S1']['passed'] is True


def test_scoring_is_paired_and_ignores_unmatched_rows():
    """The entire point of the replay is pairing; a policy missing for a pair
    must drop that pair, never silently compare across pairs."""
    frame = _replay_frame()
    frame = frame[~((frame['policy'] == 'c1') & (frame['k'] == 1))]
    out = rs.score(rs.derive_columns(frame))
    assert out['S1']['detail'].startswith('1 paired')
