"""Feasibility-frontier study: does the joint search find a deployable model
at all, across a grid of (M, k, T, arm, split) (spec 2.1).

Research question. Every prior campaign script measures QUALITY among
feasible cells (accuracy, blocks, stages) or a capacity CEILING under a
placeholder search (capacity_ceiling.py's no-Optuna grid). Neither answers
whether the real, full Optuna search (`train_multi_RF_Optuna_multi_constrained`)
can find ANY feasible model at a given forest size T, under a given block
budget M and feature count k -- and whether that reach depends on which of
two orthogonal treatments is active: threshold alignment (which sheds blocks
by merging thresholds across the two models) and a ccp_alpha search dimension
(which sheds blocks by pruning each tree). This script measures reach(cell,
arm) = max{T : some trial at that T is feasible}, over a 2x2 factorial of
those two treatments (arm in {control, aligned_only, ccp_alpha_only, both}),
for every cell in an 8-cell x 6-T x 4-arm x 3-split grid (576 points total),
and records, for every point, WHICH constraint the search's own trials hit
when a point turns out infeasible (F6) -- codeword length, block budget, the
crossbar's byte width, or the pipeline-depth ceiling.

Two decisions already resolved during planning:

1. CCP_ALPHA_STUDY_MAX = 0.05. `TrainConfig.ccp_alpha_max` is a raw
   cost-complexity threshold, not a fraction; the campaign's archived
   winners (results/campaign_backup_20260825) never approach this value at
   any (M, k) this grid touches, so 0.05 gives the ccp_alpha-enabled arms
   genuine room to prune without a priori assuming an answer.
2. Fixed feature set per cell. `run_one_point` does not re-run feature
   selection: it reads the SAME (features_app, features_ddos) pair, archived
   for that (M, k, split) under the campaign's own joint-d000 (unaligned)
   arm, for all 4 arms of this grid. This isolates the two treatments under
   test (alignment, ccp_alpha) from feature-selection noise -- exactly the
   same reasoning `replay_alignment.py`'s module docstring gives for refitting
   one pair per row rather than letting each policy pick its own features.

Every grid point is a REAL Optuna search (spec 2.1: this must measure what
the actual campaign pipeline can find, not a replay of previously-recorded
best_params), so a full 576-point grid is expensive -- use --limit and
--timing to measure per-point cost before committing to it.

Run (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/feasibility_frontier.py --limit 5 --timing

The real, unattended Codespace run (full 576-point grid, one worker process
per core on a 4-core machine):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/feasibility_frontier.py --max-workers 4

Re-running that exact command with the same --out path RESUMES rather than
restarts: already-recorded (M, k, T, arm, split) points are skipped (see
already_done/collect), so an interrupted or crashed run can simply be
re-invoked to pick up where it left off.
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Running a file inside scripts/ puts scripts/ on sys.path, not the repo root,
# so `src` would not import under the command in the docstring above.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import pandas as pd

from scripts.replay_alignment import column_indices, split_random_state
from src.main import load_campaign_data
from src.training import early_stopping
from src.training.config import TrainConfig
from src.training.errors import NoFeasibleSolution
from src.training.splits import make_task_splits
from src.training.train_model import train_multi_RF_Optuna_multi_constrained

CELLS = ((25, 9), (50, 5), (50, 9), (100, 5), (100, 9), (100, 13), (150, 9), (250, 9))
T_VALUES = (1, 2, 3, 4, 5, 6)
SPLIT_INDICES = (10, 11, 12)
ARM_NAMES = ('control', 'aligned_only', 'ccp_alpha_only', 'both')
CAMPAIGN_BACKUP_DIR = 'results/campaign_backup_20260825'
CCP_ALPHA_STUDY_MAX = 0.05
FULL_GRID_SIZE = len(CELLS) * len(T_VALUES) * len(ARM_NAMES) * len(SPLIT_INDICES)  # 576


def cfg_for_arm(arm, T):
    """One TrainConfig per 2x2-factorial arm (spec 2.2); T pinned via
    n_trees_min == n_trees == T (spec 2.5). 'Alignment on' uses the design's
    established production setting: delta_align=0.20, overlap_threshold=0.5,
    align_objective='blocks'."""
    common = dict(n_trees_min=T, n_trees=T)
    if arm == 'control':
        return TrainConfig(alignment_enabled=False, **common)
    if arm == 'aligned_only':
        return TrainConfig(alignment_enabled=True, delta_align=0.20,
                           overlap_threshold=0.5, align_objective='blocks', **common)
    if arm == 'ccp_alpha_only':
        return TrainConfig(alignment_enabled=False, ccp_alpha_max=CCP_ALPHA_STUDY_MAX, **common)
    if arm == 'both':
        return TrainConfig(alignment_enabled=True, delta_align=0.20,
                           overlap_threshold=0.5, align_objective='blocks',
                           ccp_alpha_max=CCP_ALPHA_STUDY_MAX, **common)
    raise ValueError('unknown arm {!r}, expected one of {}'.format(arm, ARM_NAMES))


def build_grid():
    """Every (M, k, T, arm, split) point -- FULL_GRID_SIZE (576) points."""
    return [{'M': M, 'k': k, 'T': T, 'arm': arm, 'split': split}
            for (M, k) in CELLS for T in T_VALUES
            for arm in ARM_NAMES for split in SPLIT_INDICES]


def already_done(out_path):
    """(M, k, T, arm, split) tuples already recorded at out_path, or an
    empty set if it doesn't exist yet -- lets collect() resume a partial
    run instead of losing everything on a crash/interrupt (Task 5's gate
    found no checkpointing existed, and the full grid is a ~10h
    single-threaded / ~2.5h at 4 workers run under Task 7's Codespace)."""
    if not os.path.exists(out_path):
        return set()
    frame = pd.read_csv(out_path)
    # A crash mid-write (pd.to_csv(mode='a') isn't atomic) can leave a torn,
    # truncated trailing row; pd.read_csv pads its missing trailing columns
    # with NaN. Such a row's key columns (M/k/T/arm/split) can still look
    # intact while any_feasible and the metric columns are garbage -- only
    # count a row as "done" when any_feasible is present, so a torn row gets
    # silently retried instead of permanently poisoning the resume checkpoint
    # (and, via the resulting NaN/object-dtype any_feasible column, later
    # crashing report()'s boolean logic).
    frame = frame[frame['any_feasible'].notna()]
    return set(zip(frame['M'], frame['k'], frame['T'], frame['arm'], frame['split']))


def load_cell_features(campaign_dir, M, k, split_idx):
    """(features_app, features_ddos) as ';'-joined strings, read once from
    this cell's archived joint-d000 row and reused across all 4 arms (see
    module docstring / plan's feature-set decision)."""
    path = os.path.join(campaign_dir, 'rf_t11_d14_M{}_joint-d000.csv'.format(M))
    frame = pd.read_csv(path)
    rows = frame[(frame.split == split_idx) & (frame.k == k)]
    if not len(rows):
        raise ValueError('no archived row for M={} k={} split={} in {}'.format(
            M, k, split_idx, path))
    row = rows.iloc[0]
    return row['features_app'], row['features_ddos']


def trial_violation_type(user_attrs):
    """Which of early_stopping._VIOLATION_ATTRS is >0 for one trial, or None
    (no violation attribute is set to a positive value).

    This is the F6 "which violation type" DIAGNOSTIC only -- it is deliberately
    NOT the feasibility decision (see summarize_trials, which uses the public
    early_stopping.is_feasible for that). A missing attribute reads as 0.0
    here, so a trial with no user_attrs at all comes back None ("no violation
    type to report"), never a crash -- but summarize_trials must not mistake
    that None for "this trial is feasible": early_stopping.constraint_values'
    own docstring is explicit that a missing attribute means the objective
    never ran far enough to set it (failed/pruned/still running), and that
    must read as INFEASIBLE, not feasible-by-default.
    """
    for name in early_stopping._VIOLATION_ATTRS:
        if user_attrs.get(name, 0.0) > 0:
            return name
    return None


def summarize_trials(trials):
    """Pure aggregation (spec 2.1/F6) over trial-like objects exposing .state
    and .user_attrs -- duck-typed like early_stopping.is_feasible, so it is
    unit-testable with plain stand-ins.

    The FEASIBLE count and any_feasible boolean come from the public
    early_stopping.is_feasible(trial) -- the same function
    ParetoStagnationStopper itself uses -- not from trial_violation_type's
    missing-attribute-defaults-to-0.0 logic, so a trial with no violation
    attributes at all (never reached objective()'s attribute-setting code)
    reads as infeasible here, consistent with early_stopping's own documented
    convention, instead of silently counting as feasible. trial_violation_type
    is used only for the separate F6 diagnostic -- which violation a NOT
    -feasible trial hit -- computed for every trial is_feasible rejects."""
    n_feasible = 0
    counts = {name: 0 for name in early_stopping._VIOLATION_ATTRS}
    n_run = 0
    for trial in trials:
        n_run += 1
        if early_stopping.is_feasible(trial):
            n_feasible += 1
            continue
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        vtype = trial_violation_type(trial.user_attrs)
        if vtype is not None:
            counts[vtype] += 1
    dominant = max(counts, key=counts.get) if any(counts.values()) else None
    out = {'n_trials_run': n_run, 'n_feasible': n_feasible,
           'any_feasible': n_feasible > 0, 'dominant_violation_type': dominant}
    out.update({'n_' + name: c for name, c in counts.items()})
    return out


def any_feasible_at(frame, M, k, T, arm):
    # frame['T'], NOT frame.T -- DataFrame.T is the transpose property, which
    # shadows a column literally named 'T' under attribute access. Using
    # frame.T here silently compares the transposed frame to a scalar instead
    # of filtering a column, so every subset comes back empty.
    subset = frame[(frame['M'] == M) & (frame['k'] == k)
                   & (frame['T'] == T) & (frame['arm'] == arm)]
    return bool(subset['any_feasible'].any())


def reach(frame, M, k, arm):
    """reach(cell, arm) = max{T : any_feasible(cell, T, arm)} (spec 2.1), OR'd
    across recorded splits. Not assumed monotone in T -- callers reading
    any_feasible_at() per T see the raw per-T record."""
    feasible_Ts = [T for T in T_VALUES if any_feasible_at(frame, M, k, T, arm)]
    return max(feasible_Ts) if feasible_Ts else None


# Set once per worker process by _init_worker (ProcessPoolExecutor's
# `initializer`), never mutated afterward -- see run_one_point.
_worker_data = None


def _init_worker(data):
    """ProcessPoolExecutor `initializer`: load the ~95MB campaign dataset
    (load_campaign_data()'s 5-tuple) into a process-global exactly ONCE per
    worker process, at pool-startup time.

    Without this, collect() would pass `data` as a per-call argument to
    run_one_point, and ProcessPoolExecutor.submit pickles each call's
    arguments independently as tasks are queued -- so submitting `data` to
    each of the grid's up to 576 tasks would pickle/unpickle the whole ~95MB
    dataset 576 times (~55GB of IPC), instead of once per worker process."""
    global _worker_data
    _worker_data = data


def run_one_point(point, data, campaign_dir):
    """One grid point. Runs CONCURRENTLY as a ProcessPoolExecutor worker task
    (collect() submits this function once per remaining grid point).

    Captures the underlying optuna.Study by temporarily reassigning
    optuna.create_study (same technique used throughout
    tests/test_train_model_contract.py's monkeypatch-based tests, applied here
    without pytest's monkeypatch since this is script, not test, code).
    Necessary because NoFeasibleSolution discards the local `study` before
    this caller ever sees it, and even on success TrainResult exposes only
    summary counts (n_trials_run, n_feasible), not F6's per-violation-type
    breakdown.

    That reassignment is a mutation of a PROCESS-GLOBAL (the `optuna` module's
    own `create_study` attribute), which is safe here ONLY because each
    concurrently-running call executes in its own separate OS process (a
    ProcessPoolExecutor worker): every process gets its own private copy of
    the `optuna` module, so one call's reassign-then-restore of
    optuna.create_study can never race against another call's. This function
    must NEVER be moved to a ThreadPoolExecutor -- concurrently-running
    threads share one process's `optuna` module object, so two calls'
    reassignments of the same global attribute would race, and one call could
    end up capturing (or restoring) another call's Study.

    `data` is the (X_app, X_ddos, y_app, y_ddos, names) 5-tuple from
    load_campaign_data(). Pass it explicitly for any DIRECT (non-pool) call --
    e.g. a `--limit 1` sanity check, or a test that calls run_one_point
    itself. When collect() submits this function to a ProcessPoolExecutor, it
    passes data=None and this function falls back to the process-global
    `_worker_data`, which _init_worker already set once at pool-startup time
    (see _init_worker's docstring for why)."""
    if data is None:
        data = _worker_data
    X_app, X_ddos, y_app, y_ddos, names = data
    M, k, T, arm, split_idx = (point['M'], point['k'], point['T'],
                               point['arm'], point['split'])

    feat_app, feat_ddos = load_cell_features(campaign_dir, M, k, split_idx)
    cols_app = column_indices(names, feat_app)
    cols_ddos = column_indices(names, feat_ddos)

    seed = split_random_state(split_idx)
    app = make_task_splits(X_app, y_app, seed)
    ddos = make_task_splits(X_ddos, y_ddos, seed)
    cfg = cfg_for_arm(arm, T)

    captured = {}
    real_create_study = optuna.create_study
    optuna.create_study = lambda *a, **kw: captured.setdefault('study', real_create_study(*a, **kw))
    try:
        try:
            result = train_multi_RF_Optuna_multi_constrained(
                app.X_train[:, cols_app], app.y_train,
                ddos.X_train[:, cols_ddos], ddos.y_train,
                (app.X_val_align[:, cols_app], app.y_val_align),
                (ddos.X_val_align[:, cols_ddos], ddos.y_val_align),
                (app.X_val_select[:, cols_app], app.y_val_select),
                (ddos.X_val_select[:, cols_ddos], ddos.y_val_select),
                feat_app.split(';'), feat_ddos.split(';'),
                M, 'joint', cfg)
        except NoFeasibleSolution:
            result = None
    finally:
        optuna.create_study = real_create_study

    row = dict(point)
    row.update(summarize_trials(captured['study'].trials))
    if result is not None:
        row.update({'acc_app': result.acc_sel_A, 'acc_ddos': result.acc_sel_B,
                    'blocks': result.blocks, 'stages': result.stages,
                    'stage_depth': result.stage_depth,
                    'ccp_alpha_A': result.best_params.get('ccp_alpha_A'),
                    'ccp_alpha_B': result.best_params.get('ccp_alpha_B')})
    else:
        row.update({field: None for field in
                    ('acc_app', 'acc_ddos', 'blocks', 'stages', 'stage_depth',
                     'ccp_alpha_A', 'ccp_alpha_B')})
    return row


def collect(campaign_dir, out, limit=None, max_workers=None):
    """Run remaining grid points in parallel via ProcessPoolExecutor, writing
    each row to `out` as its future completes (checkpointing: Task 5's gate
    found the old serial, write-once-at-the-end collect() would lose an
    entire ~10h run on any crash/interrupt).

    Points already recorded at `out` (per already_done) are skipped, so
    re-invoking with the same --out resumes a partial run instead of redoing
    it -- mirroring src/main.py's compare_independent_joint_mapping
    (skip_existing) and src/training/feature_selection.py's
    compare_feature_selection_approaches_parallel (ProcessPoolExecutor +
    as_completed), just at CSV-row granularity instead of one-file-per-cell.

    --limit counts only REMAINING (not-yet-done) points, since resuming a
    partial run with the same --limit should make progress, not re-select
    already-completed points. `limit=0` genuinely means "run nothing" (checked
    via `is not None`, not truthiness) -- a natural way to ask for just the
    current report/state without doing any new work.

    The dataset (`load_campaign_data()`'s ~95MB 5-tuple) is loaded once here
    and handed to the ProcessPoolExecutor as its `initializer`'s argument, so
    each worker process loads it into its own process-global exactly once,
    rather than it being pickled/unpickled separately as a per-call argument
    for every one of the (up to 576) submitted tasks -- see run_one_point and
    _init_worker's docstrings.

    A future that raises is caught, logged (type + full traceback), and
    skipped -- that point is simply absent from `out` this invocation, so it
    is retried on the next one (same mechanism as an interrupt/crash). A
    summary count of how many points failed this invocation is printed after
    the pool closes.

    Returns (frame, started, n_done, resolved_max_workers): frame is read back
    from `out` after writing, so it reflects the FULL accumulated file across
    every resume, not just this invocation's new rows -- or an empty
    DataFrame if `out` was never actually created (e.g. every submitted point
    failed this invocation). n_done is how many points THIS invocation
    actually ran -- distinct from len(frame) (everything ever recorded) and
    from limit/FULL_GRID_SIZE -- so callers computing per-search timing divide
    by n_done, not by any of those three (which would silently overcount on a
    resumed run). resolved_max_workers is the actual worker-process count this
    invocation used -- equal to `max_workers` when the caller passed one
    explicitly, or the auto-computed `min(len(points), max(1, os.cpu_count() -
    1))` otherwise -- so callers printing a "per-search cost at N workers"
    figure report the real N instead of guessing.
    """
    data = load_campaign_data()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    done = already_done(out)
    points = [p for p in build_grid()
              if (p['M'], p['k'], p['T'], p['arm'], p['split']) not in done]
    if limit is not None:
        points = points[:limit]

    if not points:
        print('nothing to do -- every grid point already recorded at {}'.format(out))
        frame = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        return frame, time.time(), 0, max_workers

    if max_workers is None:
        max_workers = min(len(points), max(1, os.cpu_count() - 1))
    print('Using {} parallel workers for {} remaining points'.format(max_workers, len(points)))

    file_exists = os.path.exists(out) and os.path.getsize(out) > 0
    started = time.time()
    n_done = 0
    n_failed = 0
    # initializer/initargs load the ~95MB dataset into each worker process's
    # own _worker_data global exactly ONCE at pool-startup time; the per-call
    # `data` argument below is always None so it never gets independently
    # pickled for each of the (up to 576) submitted tasks (see _init_worker
    # and run_one_point's docstrings).
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_init_worker, initargs=(data,)) as executor:
        futures = {executor.submit(run_one_point, point, None, campaign_dir): point
                   for point in points}
        for future in as_completed(futures):
            point = futures[future]
            try:
                row = future.result()
            except Exception:
                n_failed += 1
                print('  point M={} k={} T={} arm={} split={} raised an exception:\n{}'.format(
                    point['M'], point['k'], point['T'], point['arm'], point['split'],
                    traceback.format_exc()))
                continue
            n_done += 1
            pd.DataFrame([row]).to_csv(out, mode='a', header=not file_exists, index=False)
            file_exists = True
            print('  [{}/{}] M={} k={} T={} arm={} split={} any_feasible={} ({:.1f}s elapsed)'.format(
                n_done, len(points), point['M'], point['k'], point['T'], point['arm'],
                point['split'], row['any_feasible'], time.time() - started))

    if n_failed:
        print('{} points failed this run and were not recorded -- they will be '
              'retried on the next invocation'.format(n_failed))

    frame = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    return frame, started, n_done, max_workers


def print_reach_table(frame):
    print('\n### reach(cell, arm) -- largest T with any feasible trial (spec 2.1)\n')
    header = ['cell'] + list(ARM_NAMES)
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for M, k in CELLS:
        line = ['M={} k={}'.format(M, k)]
        for arm in ARM_NAMES:
            r = reach(frame, M, k, arm)
            line.append(str(r) if r is not None else 'none')
        print('| ' + ' | '.join(line) + ' |')


def _cell_dominant_violation_type(group):
    """One (M, k, T, arm) cell's dominant violation type: the most common
    per-split `dominant_violation_type` across the cell's up to 3 recorded
    splits. Series.mode() drops NaN and is stable (first-encountered order),
    so a torn/corrupted row (NaN dominant_violation_type) is ignored rather
    than winning a tie, and a genuine 3-way disagreement resolves to whichever
    type appears first among the group's rows -- an arbitrary but deterministic
    tie-break, since spec's F6 doesn't define what "dominant" means when a
    cell's splits disagree."""
    modes = group['dominant_violation_type'].mode()
    return modes.iloc[0] if len(modes) else None


def print_violation_breakdown(frame):
    print('\n### Dominant violation type per infeasible grid point (F6)\n')
    # spec 2.1/F6 wants one count per infeasible (cell, T, arm) -- i.e. one
    # per (M, k, T, arm) -- not one per raw row. The grid records up to 3
    # splits per such group (any_feasible_at/reach OR across them), so
    # counting raw rows would count a cell infeasible on all 3 splits three
    # times over. A group counts as infeasible here only when NONE of its
    # recorded splits were feasible; `.any()` (not `~`, which raises on the
    # object-dtype column a torn/NaN any_feasible row produces) also means a
    # corrupted row doesn't crash this table, just gets folded in harmlessly.
    cell_types = []
    for _, group in frame.groupby(['M', 'k', 'T', 'arm']):
        if group['any_feasible'].any():
            continue  # this (cell, T, arm) is feasible overall (>=1 split worked)
        cell_types.append(_cell_dominant_violation_type(group))
    if not cell_types:
        print('no infeasible points in this run')
        return
    print('| violation type | count |\n|---|---|')
    for name, count in pd.Series(cell_types).value_counts(dropna=False).items():
        print('| {} | {} |'.format(name, count))


def report(frame):
    """Split from main() so src/reporting/figures.py could replay this output
    from results/feasibility_frontier.csv alone, mirroring
    scripts/capacity_ceiling.py's report().

    Guards the genuinely-empty case (no columns at all) separately from a
    merely-small one: collect() returns a bare pd.DataFrame() when `out`
    was never created (Finding 8) -- e.g. every point in this invocation
    raised (Finding 4's per-future isolation means that no longer aborts
    the run, it just leaves nothing to report). print_reach_table/
    print_violation_breakdown both index frame['M'] etc., which raises
    KeyError on a column-less frame rather than reporting "nothing found".
    """
    if frame.empty:
        print('\nno rows to report -- every point in this run failed, or '
              'nothing was run. Re-run to retry (checkpointing means '
              'already-recorded points are skipped, so this is safe).')
        return
    print_reach_table(frame)
    print_violation_breakdown(frame)


def _parse_max_workers(value):
    """--max-workers' argparse type: positive integer worker-process count.
    Rejects zero or negative values with an error message that names the
    flag -- mirrors src/main.py's own _parse_max_workers."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('--max-workers must be positive, got {!r}'.format(value))
    return n


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', default=CAMPAIGN_BACKUP_DIR)
    parser.add_argument('--out', default='results/feasibility_frontier.csv')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--timing', action='store_true',
                        help='print a per-search timing summary and an extrapolation')
    parser.add_argument(
        '--max-workers', dest='max_workers', type=_parse_max_workers, default=None,
        help='number of parallel worker processes. Defaults to min(remaining points, '
             'cpu_count - 1) when omitted; pass this to use every core on a small '
             'machine (e.g. --max-workers 4 on a 4-core Codespace)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame, started, n_done, resolved_max_workers = collect(
        args.campaign_dir, args.out, args.limit, args.max_workers)
    print('{} total rows on disk at {}'.format(len(frame), args.out))
    if args.timing and n_done:
        per_search = (time.time() - started) / n_done
        workers = resolved_max_workers or 1
        print('per-search cost: {:.1f}s -- a full {}-search grid would take '
              '{:.1f}h single-threaded ({:.1f}h at {} workers)'.format(
                  per_search, FULL_GRID_SIZE,
                  per_search * FULL_GRID_SIZE / 3600,
                  per_search * FULL_GRID_SIZE / 3600 / workers,
                  workers))
    report(frame)


if __name__ == '__main__':
    main()
