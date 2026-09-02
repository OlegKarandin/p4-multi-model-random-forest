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
"""
import argparse
import os
import sys
import time
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
    (feasible, or never reached objective()'s attribute-setting code)."""
    for name in early_stopping._VIOLATION_ATTRS:
        if user_attrs.get(name, 0.0) > 0:
            return name
    return None


def summarize_trials(trials):
    """Pure aggregation (spec 2.1/F6) over trial-like objects exposing .state
    and .user_attrs -- duck-typed like early_stopping.is_feasible, so it is
    unit-testable with plain stand-ins."""
    n_feasible = 0
    counts = {name: 0 for name in early_stopping._VIOLATION_ATTRS}
    n_run = 0
    for trial in trials:
        n_run += 1
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        vtype = trial_violation_type(trial.user_attrs)
        if vtype is None:
            n_feasible += 1
        else:
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


def run_one_point(point, data, campaign_dir):
    """One grid point. Captures the underlying optuna.Study by temporarily
    reassigning optuna.create_study (same technique used throughout
    tests/test_train_model_contract.py's monkeypatch-based tests, applied here
    without pytest's monkeypatch since this is script, not test, code).
    Necessary because NoFeasibleSolution discards the local `study` before
    this caller ever sees it, and even on success TrainResult exposes only
    summary counts (n_trials_run, n_feasible), not F6's per-violation-type
    breakdown. Serial loop, not thread-safe -- matches capacity_ceiling.py and
    replay_alignment.py's own precedent."""
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
    already-completed points.

    Returns (frame, started, n_done): frame is read back from `out` after
    writing, so it reflects the FULL accumulated file across every resume,
    not just this invocation's new rows. n_done is how many points THIS
    invocation actually ran -- distinct from len(frame) (everything ever
    recorded) and from limit/FULL_GRID_SIZE -- so callers computing
    per-search timing divide by n_done, not by any of those three (which
    would silently overcount on a resumed run).
    """
    data = load_campaign_data()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    done = already_done(out)
    points = [p for p in build_grid()
              if (p['M'], p['k'], p['T'], p['arm'], p['split']) not in done]
    if limit:
        points = points[:limit]

    if not points:
        print('nothing to do -- every grid point already recorded at {}'.format(out))
        frame = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        return frame, time.time(), 0

    if max_workers is None:
        max_workers = min(len(points), max(1, os.cpu_count() - 1))
    print('Using {} parallel workers for {} remaining points'.format(max_workers, len(points)))

    file_exists = os.path.exists(out) and os.path.getsize(out) > 0
    started = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_point, point, data, campaign_dir): point
                   for point in points}
        for future in as_completed(futures):
            point = futures[future]
            row = future.result()
            n_done += 1
            pd.DataFrame([row]).to_csv(out, mode='a', header=not file_exists, index=False)
            file_exists = True
            print('  [{}/{}] M={} k={} T={} arm={} split={} any_feasible={} ({:.1f}s elapsed)'.format(
                n_done, len(points), point['M'], point['k'], point['T'], point['arm'],
                point['split'], row['any_feasible'], time.time() - started))

    return pd.read_csv(out), started, n_done


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


def print_violation_breakdown(frame):
    print('\n### Dominant violation type per infeasible grid point (F6)\n')
    infeasible = frame[~frame['any_feasible']]
    if not len(infeasible):
        print('no infeasible points in this run')
        return
    print('| violation type | count |\n|---|---|')
    for name, count in infeasible['dominant_violation_type'].value_counts().items():
        print('| {} | {} |'.format(name, count))


def report(frame):
    """Split from main() so src/reporting/figures.py could replay this output
    from results/feasibility_frontier.csv alone, mirroring
    scripts/capacity_ceiling.py's report()."""
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
    frame, started, n_done = collect(args.campaign_dir, args.out, args.limit, args.max_workers)
    print('{} total rows on disk at {}'.format(len(frame), args.out))
    if args.timing and n_done:
        per_search = (time.time() - started) / n_done
        print('per-search cost: {:.1f}s -- a full {}-search grid would take '
              '{:.1f}h single-threaded ({:.1f}h at {} workers)'.format(
                  per_search, FULL_GRID_SIZE,
                  per_search * FULL_GRID_SIZE / 3600,
                  per_search * FULL_GRID_SIZE / 3600 / (args.max_workers or 1),
                  args.max_workers or 1))
    report(frame)


if __name__ == '__main__':
    main()
