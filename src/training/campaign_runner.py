"""The compiler-verified campaign's job driver: one job per (arm, M, split).

A job trains every k of one split for one arm at one TCAM budget and writes
one CSV, rows/<split_csv_name>. That file is the unit of work and of resume:
it is written atomically and only when the split finished without an error,
so its existence means "done" (spec §5, §9).

All jobs of an invocation go through ONE process pool, so a small machine
stays busy across cells instead of draining at the end of each (arm, M).
"""

import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from src.training.campaign_run import (
    INF, m_token, run_paths, split_csv_name)
from src.training.config import TrainConfig
from src.training.feature_selection import _process_single_split
from src.training.row_artifacts import RowContext

# The data seed is DATA_RANDOM_STATE + split (make_task_splits, spec §5.4).
DATA_RANDOM_STATE = 42


def _encoding(arm):
    return 'joint' if arm == 'joint' else 'disjoint'


@dataclasses.dataclass(frozen=True)
class Job:
    arm: str
    cfg: TrainConfig
    M: float
    split: int

    @property
    def slug(self):
        return self.cfg.arm_slug(_encoding(self.arm))

    @property
    def key(self):
        """Identity used in logs and in RunSummary: slug_M<token>_s<split>."""
        return '{}_M{}_s{:02d}'.format(self.slug, m_token(self.M), self.split)

    def csv_path(self, run_dir):
        return os.path.join(
            run_paths(run_dir).rows,
            split_csv_name(self.cfg.n_trees, self.cfg.max_depth, self.M,
                           self.slug, self.split))


@dataclasses.dataclass
class RunSummary:
    done: List[str]
    failed: Dict[str, str]


def plan_jobs(arms, M_values, splits, run_dir, skip_existing=True):
    """Jobs in manifest grid order: M, then arm, then split.

    Every M is validated through m_token first, so a budget that cannot be
    named in a row_id fails here, before any training, not hours in."""
    for M in M_values:
        try:
            m_token(M)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ValueError('invalid M {!r}: {}'.format(M, exc)) from None
    jobs = []
    for M in M_values:
        for arm, cfg in arms:
            for split in splits:
                job = Job(arm=arm, cfg=cfg, M=M, split=split)
                if skip_existing and os.path.exists(job.csv_path(run_dir)):
                    continue
                jobs.append(job)
    return jobs


def _stamp(frame, job):
    """The row columns the worker cannot know: the arm's config and budget."""
    budgeted = job.M != INF
    frame['alignment_enabled'] = bool(job.cfg.alignment_enabled and job.arm == 'joint')
    frame['delta_select'] = job.cfg.delta_select
    frame['selection_rule'] = job.cfg.selection_rule
    frame['select_alpha'] = job.cfg.select_alpha
    frame['M'] = int(job.M) if budgeted else ''
    frame['budgeted'] = budgeted
    frame['n_trees'] = job.cfg.n_trees
    frame['max_depth'] = job.cfg.max_depth
    return frame


def _write_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_jobs(jobs, data, run_dir, git_commit, max_workers, executor_factory=None):
    """Run every job through one pool and write each finished split's CSV.

    data : (X_app, X_ddos, y_app, y_ddos, feature_names).
    max_workers : None means min(len(jobs), cpu_count - 1).
    executor_factory : callable(max_workers) -> Executor; defaults to
        ProcessPoolExecutor (tests inject a ThreadPoolExecutor).

    A job whose SplitResult carries an error writes NO file -- even if some
    k rows completed -- so the next invocation retries it; it is reported in
    RunSummary.failed. A job that finishes with zero rows is treated the same
    way, since an empty file would read as "done" forever.
    """
    import pandas as pd

    summary = RunSummary(done=[], failed={})
    if not jobs:
        return summary
    if executor_factory is None:
        executor_factory = ProcessPoolExecutor
    if max_workers is None:
        max_workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    run_paths(run_dir).ensure()
    X_app, X_ddos, y_app, y_ddos, feature_names = data

    with executor_factory(max_workers) as executor:
        futures = {}
        for job in jobs:
            context = RowContext(run_dir, job.slug, job.M, git_commit)
            future = executor.submit(
                _process_single_split, job.split, X_app, X_ddos, y_app, y_ddos,
                job.M, feature_names, DATA_RANDOM_STATE, job.arm, job.cfg,
                row_context=context)
            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                summary.failed[job.key] = '{}: {}'.format(type(exc).__name__, exc)
                print('FAILED {}: {}'.format(job.key, exc))
                continue
            if result.error:
                summary.failed[job.key] = result.error
                print('FAILED {} after {} rows: {}'.format(
                    job.key, len(result.results), result.error))
                continue
            if not result.results:
                summary.failed[job.key] = 'no rows produced'
                print('FAILED {}: no rows produced'.format(job.key))
                continue
            frame = _stamp(pd.DataFrame(result.results), job)
            _write_csv(frame, job.csv_path(run_dir))
            summary.done.append(job.key)
            print('done {} ({} rows)'.format(job.key, len(frame)))

    return summary
