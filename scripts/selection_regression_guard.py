"""Regression guard (spec 2026-10-04 Sec II.7): with selection_rule='balanced'
the new code must reproduce a campaign_2026_10 row exactly, on a row whose
whole search is price-invariant (no trial key has lanes.table_blocks !=
tables.codeword_to_blocks). campaign_2026_10 was trained with the ladder.

Usage: python scripts/selection_regression_guard.py --run results/campaign_2026_10 ROW_ID [ROW_ID ...]
Each ROW_ID must be a first-k row (k = every feature; no warm start). Prefer
joint / joint-off rows: one key per pool, so no lane stage simulation.
Exit 0: a price-invariant row matched. 2: every row touched a differing key.
1: a price-invariant row did not match.
"""
import argparse
import dataclasses
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.main import PRIMARY_ARMS, load_campaign_data  # noqa: E402
from src.p4gen.evaluation import accuracy_metrics  # noqa: E402
from src.p4gen.switch_semantics import switch_predict  # noqa: E402
from src.p4model import lanes  # noqa: E402
from src.p4model.tables import codeword_to_blocks  # noqa: E402
from src.training.campaign_run import optuna_seed, split_csv_name  # noqa: E402
from src.training.splits import make_task_splits  # noqa: E402
from src.training.train_model import train_multi_RF_Optuna_multi_constrained  # noqa: E402

ROW = re.compile(r'^(?P<slug>.+)_M(?P<m>\d{3}|inf)_s(?P<split>\d{2})_k(?P<k>\d{2})$')
COMPARED = ('best_params', 'blocks', 'stage_depth', 'acc_sel_app', 'acc_sel_ddos',
            'acc_app', 'f1_app', 'acc_ddos', 'f1_ddos', 'n_trials_run', 'n_feasible')
DATA_RANDOM_STATE = 42


def _arm(slug):
    for arm, cfg in PRIMARY_ARMS:
        encoding = 'joint' if arm == 'joint' else 'disjoint'
        if cfg.arm_slug(encoding) == slug:
            return arm, encoding, dataclasses.replace(cfg, selection_rule='balanced')
    raise SystemExit('unknown arm slug {!r}'.format(slug))


def _stored_row(run, slug, M, split, row_id, cfg):
    path = os.path.join(run, 'rows', split_csv_name(cfg.n_trees, cfg.max_depth, M, slug, split))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame[frame['row_id'] == row_id].iloc[0]


def _same(name, stored, fresh):
    if name == 'best_params':
        # Tolerance, not ==: a log-sampled float (ccp_alpha) can differ in the
        # last ulp between the environment that wrote the CSV and this one;
        # every discrete choice (ints) must still be identical.
        want = json.loads(stored)
        return want.keys() == fresh.keys() and all(
            math.isclose(float(want[key]), float(fresh[key]), rel_tol=1e-12, abs_tol=0.0)
            if isinstance(want[key], float) else want[key] == fresh[key]
            for key in want)
    return float(stored) == float(fresh)


def guard(run, row_id, data):
    match = ROW.match(row_id)
    if not match:
        raise SystemExit('bad row id {!r}'.format(row_id))
    slug, split, k = match['slug'], int(match['split']), int(match['k'])
    M = float('inf') if match['m'] == 'inf' else int(match['m'])
    arm, encoding, cfg = _arm(slug)
    X_app, X_ddos, y_app, y_ddos, names = data
    if k != len(names):
        raise SystemExit('{} is not a first-k row (k={}, features={})'.format(row_id, k, len(names)))
    app = make_task_splits(X_app, y_app, DATA_RANDOM_STATE + split)
    ddos = make_task_splits(X_ddos, y_ddos, DATA_RANDOM_STATE + split)

    differing = set()
    real = lanes.table_blocks

    def watched(widths):
        price = real(widths)
        key = tuple(sorted(widths))
        if price != codeword_to_blocks(key):
            differing.add(key)
        return price

    lanes.table_blocks = watched
    try:
        result = train_multi_RF_Optuna_multi_constrained(
            app.X_train, app.y_train, ddos.X_train, ddos.y_train,
            (app.X_val_align, app.y_val_align), (ddos.X_val_align, ddos.y_val_align),
            (app.X_val_select, app.y_val_select), (ddos.X_val_select, ddos.y_val_select),
            list(names), list(names), M, encoding, cfg,
            None, optuna_seed=optuna_seed(split, k))
    finally:
        lanes.table_blocks = real
    if differing:
        print('{}: NOT price-invariant ({} differing keys, e.g. {}) -- not a valid guard'.format(
            row_id, len(differing), sorted(differing)[:3]))
        return 2
    with sklearn.config_context(assume_finite=True):
        acc_app, f1_app = accuracy_metrics(app.y_test, switch_predict(result.model_A, app.X_test), task='app')
        acc_ddos, f1_ddos = accuracy_metrics(ddos.y_test, switch_predict(result.model_B, ddos.X_test), task='ddos')
    fresh = {'best_params': result.best_params, 'blocks': result.blocks,
             'stage_depth': result.stage_depth, 'acc_sel_app': result.acc_sel_A,
             'acc_sel_ddos': result.acc_sel_B, 'acc_app': acc_app, 'f1_app': f1_app,
             'acc_ddos': acc_ddos, 'f1_ddos': f1_ddos,
             'n_trials_run': result.n_trials_run, 'n_feasible': result.n_feasible}
    stored = _stored_row(run, slug, M, split, row_id, cfg)
    bad = [n for n in COMPARED if not _same(n, stored[n], fresh[n])]
    for n in bad:
        print('{}: {} stored {!r} fresh {!r}'.format(row_id, n, stored[n], fresh[n]))
    print('{}: {}'.format(row_id, 'MATCH' if not bad else 'MISMATCH'))
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('row_ids', nargs='+')
    args = parser.parse_args(argv)
    data = load_campaign_data()
    for row_id in args.row_ids:
        code = guard(args.run, row_id, data)
        if code != 2:
            return code
    return 2


if __name__ == '__main__':
    sys.exit(main())
