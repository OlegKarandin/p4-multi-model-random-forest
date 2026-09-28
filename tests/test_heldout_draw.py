"""Selection and exclusion logic of scripts/heldout_draw.py (the frozen
held-out batch). Pure-function tests on synthetic frames: no campaign data,
no sklearn, no p4c."""
import json
import os

import pandas as pd
import pytest

from scripts import heldout_draw as hd


def _row(arm, M, k, split, feats='a;b', params=None):
    return {'arm_slug': arm, 'M': M, 'k': k, 'split': split,
            'features_app': feats, 'features_ddos': feats,
            'best_params': json.dumps(params or {'n_estimators_A': 1, 'x': M * 1000 + k}),
            'infeasible': float('nan')}


def _frame(rows):
    return pd.DataFrame(rows)


def test_split_counts_thirty_is_fifteen_ten_five():
    assert hd.split_counts(30) == (15, 10, 5)
    assert sum(hd.split_counts(31)) == 31


def test_family_maps_every_joint_arm_to_joint():
    assert hd.family('independent') == 'independent'
    for arm in ('joint-off', 'joint-d000', 'joint-dinf'):
        assert hd.family(arm) == 'joint'


def test_content_key_ignores_param_key_order_and_arm_within_a_family():
    a = _row('joint-d000', 25, 3, 10, params={'p': 1, 'q': 2})
    b = _row('joint-off', 25, 3, 10, params={'q': 2, 'p': 1})
    c = _row('independent', 25, 3, 10, params={'p': 1, 'q': 2})
    assert hd.content_key(a) == hd.content_key(b)
    assert hd.content_key(a) != hd.content_key(c)


def test_compiled_identities_reads_csvs_and_margin_dirs(tmp_path):
    pd.DataFrame([{'row_id': 'joint_low_sd5', 'source_arm_slug': 'joint-d000',
                   'M': 25, 'k': 1, 'split': 10}]).to_csv(
        tmp_path / 'compiler_calibration_v6.csv', index=False)
    for name in ('joint_low_sd5', 'margin_independent_M250_k2_s12', 'dsp01', 'w003'):
        os.makedirs(tmp_path / 'somearchive' / 'compiles' / name)
    ids = hd.compiled_identities(str(tmp_path))
    assert ids == {('joint-d000', 25, 1, 10), ('independent', 250, 2, 12)}


def test_compiled_identities_refuses_an_unresolvable_stratum_dir(tmp_path):
    os.makedirs(tmp_path / 'x' / 'compiles' / 'independent_high_sd99')
    with pytest.raises(ValueError, match='independent_high_sd99'):
        hd.compiled_identities(str(tmp_path))


def test_eligible_candidates_excludes_by_identity_and_by_content():
    rows = [
        _row('independent', 25, 1, 10),                      # compiled identity
        _row('joint-d000', 25, 1, 10, params={'z': 1}),      # compiled identity
        _row('joint-d005', 25, 1, 10, params={'z': 1}),      # same content -> out
        _row('joint-d005', 50, 2, 11),                       # fresh
        _row('independent', 50, 2, 11),                      # fresh (other family)
    ]
    compiled = {('independent', 25, 1, 10), ('joint-d000', 25, 1, 10)}
    out, stats = hd.eligible_candidates(_frame(rows), compiled)
    ids = {hd.identity(r) for _, r in out.iterrows()}
    assert ids == {('joint-d005', 50, 2, 11), ('independent', 50, 2, 11)}
    assert stats['excluded_compiled_identity'] == 2
    assert stats['excluded_compiled_content'] == 1
    assert stats['candidates'] == 2


def test_eligible_candidates_collapses_content_duplicates_keeping_the_first():
    rows = [_row('joint-d010', 50, 2, 11, params={'z': 9}),
            _row('joint-d002', 50, 2, 11, params={'z': 9})]
    out, stats = hd.eligible_candidates(_frame(rows), set())
    assert len(out) == 1 and out.iloc[0]['arm_slug'] == 'joint-d002'
    assert stats['collapsed_content_duplicates'] == 1


def test_eligible_candidates_drops_rows_without_params_or_marked_infeasible():
    rows = [_row('independent', 25, 1, 10), _row('independent', 25, 2, 10),
            _row('independent', 25, 3, 10)]
    rows[0]['best_params'] = None
    rows[1]['infeasible'] = 'yes'
    out, stats = hd.eligible_candidates(_frame(rows), set())
    assert len(out) == 1 and stats['not_refittable'] == 2


def _synthetic_candidates(n):
    return pd.DataFrame([_row('independent', 25, k, 10 + k) for k in range(n)])


def _fake_predict(row):
    """depth band on k % 3 == 0, bytes band on k % 3 == 1, neither otherwise;
    k == 7 is not realisable."""
    k = int(row['k'])
    if k == 7:
        return None
    return {'stage_depth': 12 if k % 3 == 0 else 6,
            'combined_key_bytes': 60 if k % 3 == 1 else 20}


def test_draw_fills_every_stratum_with_band_members_and_never_repeats():
    picks, skipped, examined = hd.draw_designs(_synthetic_candidates(200),
                                               _fake_predict, seed=1, n=30)
    cats = [c for c, _, _ in picks]
    assert cats.count('depth') == 15 and cats.count('bytes') == 10
    assert cats.count('uniform') == 5
    for category, _, prediction in picks:
        if category == 'depth':
            assert 11 <= prediction['stage_depth'] <= 13
        if category == 'bytes':
            assert 55 <= prediction['combined_key_bytes'] <= 64
    ks = [int(r['k']) for _, r, _ in picks]
    assert len(ks) == len(set(ks))
    assert 7 not in ks


def test_draw_takes_the_uniform_stratum_first_without_looking_at_the_band():
    calls = []

    def predict(row):
        calls.append(int(row['k']))
        return {'stage_depth': 1, 'combined_key_bytes': 1}   # in no band
    picks, _, examined = hd.draw_designs(_synthetic_candidates(40), predict,
                                         seed=3, n=30)
    # Only the uniform stratum can fill; the rest of the walk passes over all.
    assert [c for c, _, _ in picks] == ['uniform'] * 5
    assert examined == 40


def test_draw_is_reproducible_by_seed_and_moves_with_it():
    frame = _synthetic_candidates(200)
    ids = lambda picks: [(c, int(r['k'])) for c, r, _ in picks]  # noqa: E731
    first = ids(hd.draw_designs(frame, _fake_predict, seed=5)[0])
    again = ids(hd.draw_designs(frame, _fake_predict, seed=5)[0])
    other = ids(hd.draw_designs(frame, _fake_predict, seed=6)[0])
    assert first == again
    assert first != other


def test_draw_prefers_depth_when_a_row_is_in_both_bands():
    frame = _synthetic_candidates(60)
    picks, _, _ = hd.draw_designs(
        frame, lambda row: {'stage_depth': 12, 'combined_key_bytes': 60},
        seed=0, n=30)
    cats = [c for c, _, _ in picks]
    # depth fills before any bytes pick is taken
    assert cats.index('bytes') > max(i for i, c in enumerate(cats) if c == 'depth')


def test_combined_key_bytes_counts_a_shared_key_once():
    app = frozenset({(('f1', (0, 5)), 3), (('f2', (0, 9)), 2)})
    ddos = frozenset({(('f3', (0, 1)), 4)})
    assert hd.combined_key_bytes([app, app, ddos]) == 9
    assert hd.combined_key_bytes([app] * 4) == 5


def test_stage_orders_respects_first_or_not():
    assert hd.stage_orders((8,), [], True) == [((), ())]
    assert hd.stage_orders((8,), [(16,)], False) == [((), ((16,),))]
    assert hd.stage_orders((8,), [(16,)], True) == [(((16,),), ())]
    later = hd.stage_orders((8,), [(16,), (24,)], True)
    assert all(ahead for ahead, _ in later)
    assert (((16,),), ((24,),)) in later and (((24,), (16,)), ()) in later
