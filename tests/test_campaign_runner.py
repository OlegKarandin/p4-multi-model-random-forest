"""Task 9: the per-(arm, M, split) job driver and the run-level manifest."""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src.reporting import manifest
from src.training import campaign_runner as runner
from src.training.campaign_run import INF
from src.training.config import TrainConfig
from src.training.feature_selection import SplitResult

ARMS = [
    ('independent', TrainConfig()),
    ('joint', TrainConfig(alignment_enabled=False)),
    ('joint', TrainConfig()),
]
JOINT = ('joint', TrainConfig())


def _threads(max_workers):
    return ThreadPoolExecutor(max_workers=max_workers)


def _dummy_data():
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    return X, X, y, y, ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# plan_jobs
# ---------------------------------------------------------------------------

def test_plan_jobs_order_is_M_then_arm_then_split(tmp_path):
    jobs = runner.plan_jobs(ARMS, [35, INF], [0, 1], str(tmp_path))
    assert len(jobs) == 12
    keys = [(j.M, j.slug, j.split) for j in jobs]
    expected = [(M, cfg.arm_slug('joint' if arm == 'joint' else 'disjoint'), s)
                for M in (35, INF) for arm, cfg in ARMS for s in (0, 1)]
    assert keys == expected


def test_job_csv_path_names_the_split_file(tmp_path):
    job = runner.Job(arm='joint', cfg=TrainConfig(), M=INF, split=1)
    assert job.slug == 'joint'
    path = job.csv_path(str(tmp_path))
    assert os.path.basename(path) == 'rf_t7_d14_Minf_joint_s01.csv'
    assert os.path.dirname(path) == os.path.join(str(tmp_path), 'rows')


def test_plan_jobs_skips_a_job_whose_csv_exists(tmp_path):
    existing = runner.Job(arm='joint', cfg=TrainConfig(), M=INF, split=1)
    path = existing.csv_path(str(tmp_path))
    os.makedirs(os.path.dirname(path))
    open(path, 'w').close()

    jobs = runner.plan_jobs(ARMS, [35, INF], [0, 1], str(tmp_path))
    assert len(jobs) == 11
    assert existing not in jobs
    # --redo: skip_existing=False recomputes it.
    assert len(runner.plan_jobs(ARMS, [35, INF], [0, 1], str(tmp_path),
                                skip_existing=False)) == 12


@pytest.mark.parametrize("bad", [15.5, 2000, -1])
def test_plan_jobs_rejects_an_M_before_any_training(tmp_path, bad):
    with pytest.raises(ValueError):
        runner.plan_jobs(ARMS, [35, bad], [0], str(tmp_path))


# ---------------------------------------------------------------------------
# run_jobs
# ---------------------------------------------------------------------------

def _stub_worker(error=None):
    calls = []

    def fake(split_idx, X_app, X_ddos, y_app, y_ddos, max_blocks, feature_names,
             random_state, arm='independent', cfg=None, row_context=None, **kw):
        calls.append({'split': split_idx, 'M': max_blocks, 'arm': arm, 'cfg': cfg,
                      'random_state': random_state, 'row_context': row_context})
        rows = [{'row_id': 'r{}'.format(k), 'arm': arm, 'split': split_idx, 'k': k}
                for k in (3, 2)]
        return SplitResult(split_idx=split_idx, results=rows, error=error)

    return fake, calls


@pytest.mark.parametrize("M,want_M,want_budgeted", [(INF, '', False), (35, 35, True)])
def test_run_jobs_writes_one_csv_with_M_and_budgeted(tmp_path, monkeypatch,
                                                     M, want_M, want_budgeted):
    fake, calls = _stub_worker()
    monkeypatch.setattr(runner, '_process_single_split', fake)
    job = runner.Job(arm='joint', cfg=TrainConfig(), M=M, split=0)

    summary = runner.run_jobs([job], _dummy_data(), str(tmp_path), 'abc123', 1,
                              executor_factory=_threads)

    assert summary.failed == {}
    assert len(summary.done) == 1
    frame = pd.read_csv(job.csv_path(str(tmp_path)), keep_default_na=False)
    assert len(frame) == 2
    if want_M == '':
        assert (frame['M'] == '').all()
    else:
        assert (frame['M'].astype(int) == want_M).all()
    assert (frame['budgeted'].astype(str) == str(want_budgeted)).all()
    assert (frame['alignment_enabled'].astype(str) == 'True').all()
    assert (frame['n_trees'].astype(int) == 7).all()
    assert (frame['max_depth'].astype(int) == 14).all()
    assert 'delta_select' in frame.columns
    # The worker got this job's identity and the fixed data seed.
    (call,) = calls
    assert call['random_state'] == 42
    assert call['row_context'] == runner.RowContext(str(tmp_path), 'joint', M, 'abc123')
    assert not glob.glob(os.path.join(str(tmp_path), 'rows', '*.partial'))


def test_run_jobs_independent_arm_rows_say_alignment_off(tmp_path, monkeypatch):
    fake, _ = _stub_worker()
    monkeypatch.setattr(runner, '_process_single_split', fake)
    job = runner.Job(arm='independent', cfg=TrainConfig(), M=35, split=0)
    runner.run_jobs([job], _dummy_data(), str(tmp_path), 'abc', 1,
                    executor_factory=_threads)
    frame = pd.read_csv(job.csv_path(str(tmp_path)))
    assert (~frame['alignment_enabled']).all()


def test_a_job_with_an_error_writes_no_file_and_is_reported(tmp_path, monkeypatch):
    fake, _ = _stub_worker(error='boom')
    monkeypatch.setattr(runner, '_process_single_split', fake)
    job = runner.Job(arm='joint', cfg=TrainConfig(), M=35, split=0)

    summary = runner.run_jobs([job], _dummy_data(), str(tmp_path), 'abc', 1,
                              executor_factory=_threads)

    assert not os.path.exists(job.csv_path(str(tmp_path)))
    assert summary.done == []
    assert len(summary.failed) == 1
    assert 'boom' in next(iter(summary.failed.values()))


def test_a_worker_that_raises_is_reported_not_fatal(tmp_path, monkeypatch):
    def explode(*a, **kw):
        raise RuntimeError('worker died')
    monkeypatch.setattr(runner, '_process_single_split', explode)
    jobs = runner.plan_jobs([JOINT], [35], [0, 1], str(tmp_path))

    summary = runner.run_jobs(jobs, _dummy_data(), str(tmp_path), 'abc', 2,
                              executor_factory=_threads)

    assert len(summary.failed) == 2
    assert all('worker died' in v for v in summary.failed.values())


# ---------------------------------------------------------------------------
# write_campaign_manifest
# ---------------------------------------------------------------------------

def test_manifest_appends_a_batch_per_invocation(tmp_path, monkeypatch):
    monkeypatch.setenv('THESIS_P4C_IMAGE', 'img:1')
    monkeypatch.delenv('THESIS_P4STUDIO_COMMIT', raising=False)
    run_dir = str(tmp_path)

    first = manifest.write_campaign_manifest(run_dir, ARMS, [35, INF], [0, 1], 10, 20)
    second = manifest.write_campaign_manifest(run_dir, ARMS, [35, INF], [2], 10, 20)

    assert len(first['batches']) == 1
    assert [b['splits'] for b in second['batches']] == [[0, 1], [2]]
    with open(os.path.join(run_dir, 'run_manifest.json')) as f:
        on_disk = json.load(f)
    assert on_disk['batches'] == second['batches']
    assert on_disk['run_dir'] == run_dir
    assert on_disk['seed_rules'] == {'data': '42 + split', 'optuna': '1000 * split + k'}
    assert on_disk['p4c_image'] == 'img:1'
    assert on_disk['open_p4studio_commit'] is None
    assert 'env_hash' in on_disk
    assert on_disk['dataset_rows'] == {'app': 10, 'ddos': 20}
    assert {'started_utc', 'git'} <= set(on_disk['batches'][0])


def test_manifest_refuses_a_different_grid(tmp_path):
    run_dir = str(tmp_path)
    manifest.write_campaign_manifest(run_dir, ARMS, [35, INF], [0], 10, 20)
    with pytest.raises(ValueError):
        manifest.write_campaign_manifest(run_dir, ARMS, [35, 50], [1], 10, 20)
    with pytest.raises(ValueError):
        manifest.write_campaign_manifest(run_dir, ARMS[:1], [35, INF], [1], 10, 20)


def test_env_hash_never_raises_when_conda_is_absent(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError('conda')
    monkeypatch.setattr(manifest.subprocess, 'check_output', missing)
    assert manifest.env_hash() is None


# ---------------------------------------------------------------------------
# Determinism (spec §5.4)
# ---------------------------------------------------------------------------

def _synthetic_data():
    from tests.test_train_model_seed import _NAMES, _task
    parts = []
    for n_classes, seed in ((3, 0), (2, 1)):
        X_tr, y_tr, (X_al, y_al), (X_sel, y_sel) = _task(n_classes, seed)
        parts.append((np.vstack([X_tr, X_al, X_sel]),
                      np.concatenate([y_tr, y_al, y_sel])))
    (X_app, y_app), (X_ddos, y_ddos) = parts
    return X_app, X_ddos, y_app, y_ddos, list(_NAMES)


def _read_all(pattern):
    return {os.path.basename(p): open(p, 'rb').read() for p in sorted(glob.glob(pattern))}


def test_one_split_twice_is_byte_identical(tmp_path):
    data = _synthetic_data()
    cfg = TrainConfig(n_trials=12, min_feasible_before_stop=5, lookback=4)
    snapshots = []
    started = time.time()
    for name in ('a', 'b'):
        run_dir = str(tmp_path / name)
        job = runner.Job(arm='joint', cfg=cfg, M=35, split=0)
        summary = runner.run_jobs([job], data, run_dir, 'deadbeef', 1,
                                  executor_factory=_threads)
        assert summary.failed == {}, summary.failed
        snapshots.append({
            'rows': _read_all(os.path.join(run_dir, 'rows', '*.csv')),
            'p4': _read_all(os.path.join(run_dir, 'designs', '*.p4')),
            'model': _read_all(os.path.join(run_dir, 'designs', '*.model.json')),
        })
    print('determinism test ran in {:.1f} s'.format(time.time() - started))
    first, second = snapshots
    assert first['rows'] and first['p4'] and first['model']
    for kind in ('rows', 'p4', 'model'):
        assert first[kind].keys() == second[kind].keys(), kind
        for name in first[kind]:
            assert first[kind][name] == second[kind][name], (kind, name)


def test_stamp_records_the_selection_rule_and_alpha():
    import pandas as pd
    from src.training.campaign_runner import Job, _stamp
    from src.training.config import TrainConfig
    job = Job(arm='joint', cfg=TrainConfig(select_alpha=0.1), M=35, split=0)
    frame = _stamp(pd.DataFrame([{'k': 5}]), job)
    assert frame.loc[0, 'selection_rule'] == 'tied_cheapest'
    assert frame.loc[0, 'select_alpha'] == 0.1


def test_stamp_marks_every_trained_row_as_not_postprocessed():
    cfg = TrainConfig(n_trials=12, min_feasible_before_stop=5, lookback=4)
    job = runner.Job(arm='joint', cfg=cfg, M=35, split=0)
    frame = runner._stamp(pd.DataFrame([{'k': 3}]), job)
    assert frame['alignment_postprocess'].tolist() == [False]


def test_result_row_carries_the_twin_columns_as_empty_text():
    from src.training.feature_selection import _build_result_row
    row = _build_result_row('joint', 'multi', 0, 3, ['a'], ['a'])
    assert row['source_row_id'] == '' and row['twin_identical'] == ''
