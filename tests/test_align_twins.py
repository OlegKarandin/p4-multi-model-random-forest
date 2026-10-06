import copy
import dataclasses
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src.p4gen.evaluation import multi_model_memory_evaluation
from src.training import align_twins
from src.training import row_artifacts as ra
from src.training.campaign_run import (atomic_write_text, canonical_json, run_paths,
                                       split_csv_name)
from src.training.campaign_run import row_id as make_row_id
from src.verify import runner as verify_runner
from src.verify.verdicts import classify
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


def _serial(max_workers):
    return ThreadPoolExecutor(max_workers=1)


def _verified_run(tmp_path, ks=(3, 4), split=0, M=35):
    """A run with `len(ks)` joint-off designs in one (M, split) file, every one verified."""
    rows = []
    for k in ks:
        ctx, row = _source(tmp_path, split=split, k=k, M=M)
        row.update({'n_trees': 7, 'max_depth': 14, 'delta_select': 0.02,
                    'selection_rule': 'tied_cheapest', 'select_alpha': 0.05,
                    'alignment_postprocess': False})
        rows.append(row)
    paths = run_paths(str(tmp_path)).ensure()
    pd.DataFrame(rows).to_csv(os.path.join(paths.rows, split_csv_name(7, 14, M, 'joint-off', split)),
                              index=False)
    for row in rows:
        model = json.load(open(os.path.join(paths.designs, row['row_id'] + '.model.json')))
        rec = verify_runner._record(model, row['row_id'],
                                    os.path.join(paths.designs, row['row_id'] + '.p4'),
                                    classify(model, None, M, 'COMPILE_ERROR'), failure='p4c_timeout')
        atomic_write_text(os.path.join(paths.verify, row['row_id'] + '.json'), canonical_json(rec))
    return str(tmp_path), rows


def _twin_csv(run, M=35, split=0):
    return os.path.join(run, 'rows', split_csv_name(7, 14, M, 'joint-off-al', split))


def test_run_refuses_while_a_source_is_unverified(tmp_path, capsys):
    run, rows = _verified_run(tmp_path)
    os.remove(os.path.join(run, 'verify', rows[0]['row_id'] + '.json'))
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 2
    assert rows[0]['row_id'] in capsys.readouterr().out
    assert os.listdir(os.path.join(run, 'rows')) == [split_csv_name(7, 14, 35, 'joint-off', 0)]


def test_run_writes_one_twin_file_per_split_and_copies_identical_records(tmp_path):
    run, rows = _verified_run(tmp_path)
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    twins = pd.read_csv(_twin_csv(run), keep_default_na=False, dtype=str)
    source_cols = list(pd.read_csv(os.path.join(run, 'rows', split_csv_name(7, 14, 35, 'joint-off', 0)),
                                   nrows=0).columns)
    assert list(twins.columns)[:len(source_cols)] == source_cols
    assert {'source_row_id', 'twin_identical', 'alignment_postprocess'} <= set(twins.columns)
    assert sorted(twins['row_id']) == sorted(align_twins.twin_row_id(r['row_id']) for r in rows)
    assert set(twins['alignment_postprocess']) == {'True'}
    assert set(twins['arm']) == {'joint'} and set(twins['alignment_enabled']) == {'False'}
    for _, t in twins.iterrows():
        record = os.path.join(run, 'verify', t['row_id'] + '.json')
        if t['twin_identical'] == 'True':
            assert json.load(open(record))['copied_from'] == t['source_row_id']
        else:
            assert not os.path.exists(record)   # left for the compiler
    from src.reporting.campaign_data import load_campaign
    df = load_campaign(run, require_verified=False)
    assert sorted(set(df['arm_slug'])) == ['joint-off', 'joint-off-al']


def test_run_is_resumable_and_does_not_rewrite_a_finished_split(tmp_path):
    run, rows = _verified_run(tmp_path)
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    twin_csv = _twin_csv(run)
    before = os.stat(twin_csv).st_mtime_ns
    records = sorted(os.listdir(os.path.join(run, 'verify')))
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    assert os.stat(twin_csv).st_mtime_ns == before
    assert sorted(os.listdir(os.path.join(run, 'verify'))) == records


def test_run_after_a_partial_first_run_skips_finished_splits_and_keeps_records(tmp_path):
    # Two (M, split) files; the first finishes, the second is missing (crash).
    run, rows35 = _verified_run(tmp_path, ks=(3,), split=0, M=35)
    _, rows50 = _verified_run(tmp_path, ks=(3,), split=0, M=50)
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    done_csv, other_csv = _twin_csv(run, 35), _twin_csv(run, 50)
    os.remove(other_csv)
    stamp = os.stat(done_csv).st_mtime_ns
    record_files = {n: os.stat(os.path.join(run, 'verify', n)).st_mtime_ns
                    for n in os.listdir(os.path.join(run, 'verify'))}
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    assert os.path.isfile(other_csv)
    assert os.stat(done_csv).st_mtime_ns == stamp
    for name, mtime in record_files.items():     # no existing record rewritten
        assert os.stat(os.path.join(run, 'verify', name)).st_mtime_ns == mtime


def test_run_marks_an_invariant_violation_infeasible_and_still_finishes(tmp_path, monkeypatch):
    real = align_twins._usage

    def inflated(*args):
        u = real(*args)
        return dataclasses.replace(u, blocks=u.blocks + 1)
    monkeypatch.setattr(align_twins, '_usage', inflated)
    run, rows = _verified_run(tmp_path)
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 0
    twins = pd.read_csv(_twin_csv(run), keep_default_na=False, dtype=str)
    assert all(v.startswith('TwinInvariantViolation') for v in twins['infeasible'])
    assert not any(name.startswith('joint-off-al') for name in os.listdir(os.path.join(run, 'designs')))
    assert not any(name.startswith('joint-off-al') for name in os.listdir(os.path.join(run, 'verify')))
    # The loader treats them like any infeasible row: they never reach the frame.
    from src.reporting.campaign_data import load_campaign
    df = load_campaign(run, require_verified=False)
    assert 'joint-off-al' not in set(df['arm_slug'])
    assert not any(str(r).startswith('joint-off-al') for r in df['row_id'])


def test_run_returns_1_when_a_split_fails(tmp_path, monkeypatch):
    run, _ = _verified_run(tmp_path)

    def boom(*args, **kw):
        raise RuntimeError('boom')
    monkeypatch.setattr(align_twins, 'build_twin', boom)
    assert align_twins.run(run, data=_synthetic_data(), executor_factory=_serial) == 1
    assert not os.path.exists(_twin_csv(run))
