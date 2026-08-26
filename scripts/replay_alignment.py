"""Paired replay of threshold-alignment policies over an existing campaign.

WHY THIS EXISTS. Every arm of the campaign ran its own Optuna search, so
joint-d000 and joint-dinf differ in two things at once -- the alignment budget
AND the models the search happened to select. Every alignment effect is
confounded with search noise, which is why the companion analysis could not
attribute a single block to alignment. This harness refits ONE model pair per
campaign row and runs EVERY policy on THAT pair, so the only thing that varies
is the policy. Cost is one fit-pair per row instead of a whole campaign.

Determinism is what makes it legitimate: random_state is fixed at 42 inside
rf_params_from_params, the split seed is a pure function of the row's `split`
column, and align_rf_thresholds is a deterministic function of (models, data,
params). The same determinism already backs the refit assertion at
train_model.py:373-377. `--verify` re-runs each row under its OWN recorded arm
settings and checks the resulting block count against the row's `blocks`
column -- if that disagrees, the replay is not reproducing the campaign and no
number below it means anything.

SCALE. The 40-file backup holds 5185 joint-arm rows with alignment. Replaying
all of them across the ladder is days of compute, so --k / --n-splits / --arms
/ --M select a deterministic subset. Start with --limit 5 --timing to measure
per-row cost before committing to a grid.

Run (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \
      scripts/replay_alignment.py --limit 5 --timing
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import sklearn

from src.main import load_campaign_data
from src.p4gen.build_p4_script import dt_thresholds_float_to_int
from src.p4gen.evaluation import accuracy_metrics, multi_model_memory_evaluation
from src.p4gen.switch_semantics import switch_predict
from src.training.splits import make_task_splits
from src.training.threshold_alignment import align_rf_thresholds
from src.training.train_model import rf_params_from_params

from sklearn.ensemble import RandomForestClassifier


# align_rf_thresholds now accepts align_policy directly (Task 7). The real
# commit-or-rollback align_with_policy wrapper lands in Task 8; this alias
# keeps the harness runnable in between.
align_with_policy = align_rf_thresholds


# main.py:380 fixes random_state=42 for every campaign invocation, and
# feature_selection.py:640 derives the per-split seed from it.
CAMPAIGN_RANDOM_STATE = 42

# joint-off never calls align_rf_thresholds and independent is disjoint, so
# neither can source a pair for an ALIGNMENT replay.
ALIGNED_ARM_SLUGS = ('joint-d000', 'joint-d002', 'joint-d005',
                     'joint-d010', 'joint-d020', 'joint-dinf')

# The delta_align each arm slug was run at, needed by --verify to reproduce a
# row under its own settings. 'joint-dinf' is the accept-everything anchor.
ARM_DELTA = {'joint-d000': 0.0, 'joint-d002': 0.02, 'joint-d005': 0.05,
             'joint-d010': 0.10, 'joint-d020': 0.20, 'joint-dinf': None}

DEFAULT_K = (17, 13, 9, 5, 2)


def split_random_state(split_idx):
    return CAMPAIGN_RANDOM_STATE + int(split_idx)


def column_indices(all_names, joined_names):
    """Column indices for a row's ';'-joined feature list."""
    out = []
    for name in joined_names.split(';'):
        if name not in all_names:
            raise ValueError(
                '{!r} is not in the loaded dataset feature list {!r}'.format(
                    name, all_names))
        out.append(all_names.index(name))
    return out


def load_backup(results_dir):
    """Every campaign CSV in `results_dir`, with arm_slug from the filename."""
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, 'rf_t*_d*_M*_*.csv'))):
        frame = pd.read_csv(path)
        base = os.path.basename(path)[:-len('.csv')]
        frame['arm_slug'] = base.split('_M')[1].split('_', 1)[1]
        frames.append(frame)
    if not frames:
        raise ValueError('no campaign CSVs found under {!r}'.format(results_dir))
    return pd.concat(frames, ignore_index=True)


def select_rows(frame, arms, m_values, k_values, n_splits):
    """A deterministic subset of replayable rows.

    Deterministic by construction -- lowest split ids first, no RNG -- because
    the ladder's policies must be compared on ONE fixed pair set. Infeasible
    rows are dropped: they carry no best_params to refit from.
    """
    out = frame[frame['arm_slug'].isin(arms or ALIGNED_ARM_SLUGS)]
    if 'infeasible' in out.columns:
        out = out[out['infeasible'].isna() | (out['infeasible'] == '')]
    if m_values:
        out = out[out['M'].isin(m_values)]
    if k_values:
        out = out[out['k'].isin(k_values)]
    keep = sorted(out['split'].unique())[:n_splits]
    out = out[out['split'].isin(keep)]
    return out.sort_values(['arm_slug', 'M', 'split', 'k']).reset_index(drop=True)


def refit_pair(row, data):
    """The campaign's own model pair for this row, refit from best_params."""
    X_app, X_ddos, y_app, y_ddos, names = data
    params = json.loads(row['best_params'])

    cols_app = column_indices(names, row['features_app'])
    cols_ddos = column_indices(names, row['features_ddos'])

    seed = split_random_state(row['split'])
    app = make_task_splits(X_app, y_app, seed)
    ddos = make_task_splits(X_ddos, y_ddos, seed)

    with sklearn.config_context(assume_finite=True):
        model_app = RandomForestClassifier(
            **rf_params_from_params(params, 'A'), n_jobs=1).fit(
                app.X_train[:, cols_app], app.y_train)
        model_ddos = RandomForestClassifier(
            **rf_params_from_params(params, 'B'), n_jobs=1).fit(
                ddos.X_train[:, cols_ddos], ddos.y_train)

    return (dt_thresholds_float_to_int(model_app),
            dt_thresholds_float_to_int(model_ddos),
            app, ddos, cols_app, cols_ddos)


def run_one_policy(models, app, ddos, cols_app, cols_ddos, names_app, names_ddos,
                   policy, delta_rel, overlap_threshold):
    """One policy on one already-fit pair. Returns a result dict."""
    model_app, model_ddos = models
    stats = {}
    started = time.time()
    aligned_app, aligned_ddos = align_with_policy(
        model_app, model_ddos,
        app.X_val_align[:, cols_app], app.y_val_align,
        ddos.X_val_align[:, cols_ddos], ddos.y_val_align,
        overlap_threshold=overlap_threshold,
        delta_rel=delta_rel,
        align_policy=policy,
        align_stats=stats)
    elapsed = time.time() - started

    usage = multi_model_memory_evaluation(
        aligned_app, aligned_ddos, names_app, names_ddos, 'joint')

    # switch_predict, NOT model.predict: these must be the numbers the deployed
    # switch produces -- rf.predict's soft vote is up to 1.7 points optimistic
    # (P1 Task 7), and on TEST, matching the campaign's acc_app/acc_ddos.
    with sklearn.config_context(assume_finite=True):
        acc_app, f1_app = accuracy_metrics(
            app.y_test, switch_predict(aligned_app, app.X_test[:, cols_app]),
            task='app')
        acc_ddos, f1_ddos = accuracy_metrics(
            ddos.y_test, switch_predict(aligned_ddos, ddos.X_test[:, cols_ddos]),
            task='ddos')

    out = {'policy': policy, 'overlap_threshold': overlap_threshold,
           'delta_rel': 'inf' if delta_rel is None else delta_rel,
           'blocks': int(usage.blocks), 'stages': int(usage.stages),
           'codeword_length': int(usage.codeword_length),
           'acc_app': acc_app, 'f1_app': f1_app,
           'acc_ddos': acc_ddos, 'f1_ddos': f1_ddos,
           'runtime_s': elapsed}
    out.update({'align_' + k: v for k, v in stats.items()})
    return out


def replay_row(row, data, policies, overlap_thresholds, ladder_delta, verify):
    """Every (policy, overlap_threshold) cell for one campaign row."""
    model_app, model_ddos, app, ddos, cols_app, cols_ddos = refit_pair(row, data)
    names_app = row['features_app'].split(';')
    names_ddos = row['features_ddos'].split(';')
    models = (model_app, model_ddos)

    results = []
    if verify:
        # Reproduce the row under ITS OWN arm settings. If `blocks` disagrees,
        # the refit is not reproducing the campaign and nothing below is valid.
        # NOT `or 0.5`: a NaN is truthy, and a NaN overlap_threshold would make
        # `overlap_ratio < overlap_threshold` False for every pair, silently
        # disabling the candidate gate instead of failing.
        recorded = row.get('overlap_threshold')
        verify_overlap = 0.5 if pd.isna(recorded) else float(recorded)
        try:
            own = run_one_policy(models, app, ddos, cols_app, cols_ddos,
                                 names_app, names_ddos, 'legacy',
                                 ARM_DELTA[row['arm_slug']], verify_overlap)
        except Exception as exc:
            # This row's OWN recorded settings failed to reproduce -- that is
            # a break in the harness's determinism claim, not merely "this
            # swept combination happened to be infeasible", so it gets a
            # visually distinct diagnostic from the swept-cell skip below.
            print('  VERIFY FAILED (error): {} M={} split={} k={} '
                  'policy=legacy overlap={} -- {}: {}'.format(
                      row['arm_slug'], row['M'], row['split'], row['k'],
                      verify_overlap, type(exc).__name__, exc))
        else:
            own['policy'] = 'verify'
            own['blocks_recorded'] = int(row['blocks'])
            own['blocks_reproduced'] = own['blocks'] == int(row['blocks'])
            results.append(own)

    for policy in policies:
        for overlap in overlap_thresholds:
            try:
                result = run_one_policy(
                    models, app, ddos, cols_app, cols_ddos, names_app,
                    names_ddos, policy, ladder_delta, overlap)
            except Exception as exc:
                # An exploratory sweep cell landed outside the feasible
                # region for this refit pair (e.g. CrossbarKeyTooWide at a
                # (policy, overlap_threshold) the campaign never validated).
                # Skip the cell rather than losing every row already
                # computed -- matches select_rows' "drop infeasible rows"
                # semantics for the campaign's own data.
                print('  SKIPPED (error): {} M={} split={} k={} policy={} '
                      'overlap={} -- {}: {}'.format(
                          row['arm_slug'], row['M'], row['split'], row['k'],
                          policy, overlap, type(exc).__name__, exc))
                continue
            results.append(result)

    for result in results:
        result.update({'source_arm': row['arm_slug'], 'M': int(row['M']),
                       'split': int(row['split']), 'k': int(row['k'])})
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results/campaign_backup_20260825')
    parser.add_argument('--out', default='results/replay_alignment.csv')
    parser.add_argument('--arms', default=None,
                        help='comma-separated arm slugs (default: all aligned joint arms)')
    parser.add_argument('--M', default=None, help='comma-separated block budgets')
    parser.add_argument('--k', default=','.join(str(k) for k in DEFAULT_K))
    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--policies', default='legacy')
    parser.add_argument('--overlap-thresholds', default='0.5')
    parser.add_argument('--ladder-delta', default='0.20',
                        help="delta_rel every policy in the ladder runs at; 'inf' for accept-all")
    parser.add_argument('--verify', action='store_true',
                        help="reproduce each row under its own arm settings and check `blocks`")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--timing', action='store_true',
                        help='print a per-row timing summary and an extrapolation')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ints = lambda s: [int(v) for v in s.split(',')] if s else None
    floats = lambda s: [float(v) for v in s.split(',')]

    frame = load_backup(args.results_dir)
    rows = select_rows(frame, args.arms.split(',') if args.arms else None,
                       ints(args.M), ints(args.k), args.n_splits)
    if args.limit:
        rows = rows.head(args.limit)
    print('replaying {} rows x {} policies x {} overlap thresholds'.format(
        len(rows), len(args.policies.split(',')),
        len(args.overlap_thresholds.split(','))))

    data = load_campaign_data()
    ladder_delta = None if args.ladder_delta == 'inf' else float(args.ladder_delta)

    out, started = [], time.time()
    for i, (_, row) in enumerate(rows.iterrows()):
        out.extend(replay_row(row, data, args.policies.split(','),
                              floats(args.overlap_thresholds), ladder_delta,
                              args.verify))
        print('  [{}/{}] {} M={} split={} k={}  ({:.1f}s elapsed)'.format(
            i + 1, len(rows), row['arm_slug'], row['M'], row['split'], row['k'],
            time.time() - started))

    frame_out = pd.DataFrame(out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    frame_out.to_csv(args.out, index=False)
    print('wrote {} rows to {}'.format(len(frame_out), args.out))

    if args.verify and 'blocks_reproduced' in frame_out.columns:
        checked = frame_out[frame_out['policy'] == 'verify']
        bad = checked[~checked['blocks_reproduced'].astype(bool)]
        print('determinism: {}/{} rows reproduced their recorded blocks'.format(
            len(checked) - len(bad), len(checked)))
        if len(bad):
            print(bad[['source_arm', 'M', 'split', 'k',
                       'blocks_recorded', 'blocks']].to_string())

    if args.timing:
        per_row = (time.time() - started) / max(len(rows), 1)
        print('per-row cost: {:.1f}s -- a full 5185-row replay would take '
              '{:.1f}h single-threaded'.format(per_row, per_row * 5185 / 3600))


if __name__ == '__main__':
    main()
