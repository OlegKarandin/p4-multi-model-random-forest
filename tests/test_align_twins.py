import copy
import dataclasses
import hashlib
import json
import os

import numpy as np
import pytest

from src.p4gen.evaluation import multi_model_memory_evaluation
from src.training import align_twins
from src.training import row_artifacts as ra
from src.training.campaign_run import row_id as make_row_id
from tests.test_campaign_runner import _synthetic_data
from tests.test_row_artifacts import _forests
from tests.test_train_model_seed import _NAMES


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _source(tmp_path, split=0, k=3, M=35):
    """A joint-off source design in a run dir: artifacts + the row dict."""
    fa, fd = _forests()
    usage = multi_model_memory_evaluation(fa, fd, _NAMES, _NAMES, 'joint')
    rid = make_row_id('joint-off', M, split, k)
    ctx = ra.RowContext(str(tmp_path), 'joint-off', M, 'abc123')
    ra.write_row_artifacts(ctx, rid, fa, fd, _NAMES, _NAMES, 'joint', usage)
    row = {'row_id': rid, 'arm': 'joint', 'method': 'multi', 'split': split, 'k': k,
           'M': M, 'budgeted': True, 'alignment_enabled': False,
           'features_app': ';'.join(_NAMES), 'features_ddos': ';'.join(_NAMES),
           'blocks': usage.blocks, 'stage_depth': usage.stage_depth, 'stages': usage.stages,
           'range_entries': usage.range_entries, 'ternary_entries': usage.ternary_entries,
           'register_depth': usage.register_depth, 'register_count': usage.register_count,
           'acc_app': 0.5, 'f1_app': 0.5, 'acc_ddos': 0.5, 'f1_ddos': 0.5,
           'acc_sel_app': 0.5, 'acc_sel_ddos': 0.5, 'best_params': '{}',
           'align_attempted': '', 'align_accepted': '', 'intervals_before': '',
           'intervals_after': '', 'infeasible': '', 'source_row_id': '', 'twin_identical': ''}
    return ctx, row


def test_twin_row_id_swaps_the_arm_prefix_only():
    assert align_twins.twin_row_id('joint-off_M035_s07_k09') == 'joint-off-al_M035_s07_k09'
    assert align_twins.twin_row_id('joint-off_Minf_s24_k17') == 'joint-off-al_Minf_s24_k17'
    with pytest.raises(ValueError):
        align_twins.twin_row_id('independent_M035_s07_k09')
    with pytest.raises(ValueError):
        align_twins.twin_row_id('joint_M035_s07_k09')


def test_build_twin_writes_artifacts_and_never_costs_blocks(tmp_path):
    ctx, source = _source(tmp_path)
    forests = ra.load_forests(os.path.join(str(tmp_path), 'forests', source['row_id'] + '.joblib'))
    twin = align_twins.build_twin(ctx, source, forests, _synthetic_data())

    tid = 'joint-off-al_M035_s00_k03'
    assert twin['row_id'] == tid and twin['source_row_id'] == source['row_id']
    assert twin['infeasible'] == ''
    assert twin['blocks'] <= source['blocks'] and twin['stage_depth'] <= source['stage_depth']
    assert twin['alignment_enabled'] is False and twin['alignment_postprocess'] is True
    assert isinstance(twin['align_accepted'], int) and twin['intervals_after'] <= twin['intervals_before']
    for ext in ('.p4', '.model.json'):
        assert os.path.isfile(os.path.join(str(tmp_path), 'designs', tid + ext))
    assert os.path.isfile(os.path.join(str(tmp_path), 'forests', tid + '.joblib'))
    model = json.load(open(os.path.join(str(tmp_path), 'designs', tid + '.model.json')))
    assert model['source_row_id'] == source['row_id']
    same = _sha(os.path.join(str(tmp_path), 'designs', tid + '.p4')) == \
        _sha(os.path.join(str(tmp_path), 'designs', source['row_id'] + '.p4'))
    assert twin['twin_identical'] is same
    assert model.get('identical_to') == (source['row_id'] if same else None)
    for key in ('acc_app', 'f1_app', 'acc_ddos', 'f1_ddos', 'acc_sel_app', 'acc_sel_ddos'):
        assert 0.0 <= twin[key] <= 1.0


def test_build_twin_with_no_accepted_move_is_identical_and_marked(tmp_path, monkeypatch):
    def unchanged(a, d, *args, align_stats=None, **kw):
        align_stats.update(attempted=0, accepted=0, intervals_before=7, intervals_after=7)
        return a, d
    monkeypatch.setattr(align_twins, 'align_with_policy', unchanged)
    ctx, source = _source(tmp_path)
    forests = ra.load_forests(os.path.join(str(tmp_path), 'forests', source['row_id'] + '.joblib'))
    twin = align_twins.build_twin(ctx, source, forests, _synthetic_data())
    assert twin['twin_identical'] is True and twin['align_accepted'] == 0
    model = json.load(open(os.path.join(str(tmp_path), 'designs', twin['row_id'] + '.model.json')))
    assert model['identical_to'] == source['row_id']


def test_build_twin_with_no_accepted_move_but_a_changed_program_is_not_marked_identical(
        tmp_path, monkeypatch, capsys):
    # Review focus 2: the sha decides, not the aligner's count.
    def lies(a, d, *args, align_stats=None, **kw):
        align_stats.update(attempted=1, accepted=0, intervals_before=7, intervals_after=7)
        a2 = copy.deepcopy(a)
        # The program only reads interval COUNTS, not threshold values, so
        # make every split threshold distinct to change the declared tables.
        value = 1000
        for est in a2.estimators_:
            tree = est.tree_
            for node in np.flatnonzero(tree.children_left != -1):
                value += 7
                tree.threshold[node] = value
        return a2, d
    monkeypatch.setattr(align_twins, 'align_with_policy', lies)
    ctx, source = _source(tmp_path)
    forests = ra.load_forests(os.path.join(str(tmp_path), 'forests', source['row_id'] + '.joblib'))
    twin = align_twins.build_twin(ctx, source, forests, _synthetic_data())
    assert twin['twin_identical'] is False
    model = json.load(open(os.path.join(str(tmp_path), 'designs', twin['row_id'] + '.model.json')))
    assert model.get('identical_to') is None
    assert 'accepted no move but the program differs' in capsys.readouterr().out


def test_build_twin_invariant_violation_writes_nothing_and_marks_the_row(tmp_path, monkeypatch):
    real = align_twins._usage
    def inflated(*args):
        u = real(*args)
        return dataclasses.replace(u, blocks=u.blocks + 1)
    monkeypatch.setattr(align_twins, '_usage', inflated)
    ctx, source = _source(tmp_path)
    forests = ra.load_forests(os.path.join(str(tmp_path), 'forests', source['row_id'] + '.joblib'))
    twin = align_twins.build_twin(ctx, source, forests, _synthetic_data())
    assert twin['infeasible'].startswith('TwinInvariantViolation')
    assert not os.path.exists(os.path.join(str(tmp_path), 'designs', twin['row_id'] + '.p4'))
    assert not os.path.exists(os.path.join(str(tmp_path), 'designs', twin['row_id'] + '.model.json'))
