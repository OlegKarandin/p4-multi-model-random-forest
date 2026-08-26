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


def test_s5_uses_the_tightest_swept_threshold_as_the_reference_not_literal_0_5():
    """The reference ("tight") threshold must be derived from the data --
    the maximum swept overlap_threshold -- not require the literal value 0.5
    to be present in the sweep."""
    rows = []
    for threshold, after, blocks in ((0.3, 90, 28), (0.7, 96, 28)):
        rows.append({'source_arm': 'joint-d020', 'M': 25, 'split': 10,
                     'k': 0, 'overlap_threshold': threshold, 'policy': 'c1',
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


def test_s5_and_s6_agree_on_the_best_available_policy_when_c1c2_is_absent():
    """A run with only legacy/c1 (no c1c2) -- the exact shape of Task 6's
    baseline sweep -- must have S5 and S6 pick the SAME fallback policy
    (c1), not silently disagree (S5 falling back to legacy, S6 to c1)."""
    rows = []
    for policy, threshold, after, blocks in (
            ('legacy', 0.3, 100, 30), ('legacy', 0.7, 100, 30),
            ('c1', 0.3, 90, 28), ('c1', 0.7, 96, 28)):
        rows.append({'source_arm': 'joint-d020', 'M': 25, 'split': 10,
                     'k': 0, 'overlap_threshold': threshold, 'policy': policy,
                     'align_codeword_before': 100,
                     'align_codeword_after': after, 'blocks': blocks,
                     'acc_app': 0.90, 'acc_ddos': 0.95})
    frame = pd.DataFrame(rows)
    verdict = rs.score(rs.derive_columns(frame))
    # S5 compared c1's own thresholds (not legacy's), so it found the
    # bits-shed widening; S6 pairs c1 against legacy, i.e. 'c1' appears in
    # its detail string.
    assert verdict['S5']['passed'] is True
    assert verdict['S6']['detail'].startswith('2 paired') or \
        'c1 saves' in verdict['S6']['detail']
