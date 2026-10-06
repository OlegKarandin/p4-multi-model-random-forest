"""Aligned twins (spec docs/superpowers/specs/2026-10-06-alignment-twins-and-two-arm-campaign-design.md).

A twin is a `joint-off` design's SAVED forest with its thresholds aligned on
`val_align` after selection -- the production aligner, the same data, no
search -- regenerated as a P4 program under the arm slug `joint-off-al` and
compiled like any design. The paired difference twin - source is what the
paper reports for threshold alignment.

Two kinds of twin: one whose program is byte-identical to its source
(alignment accepted nothing that changed a threshold) inherits the source's
verification record (`verify.runner.copy_record`); one whose program differs
is left for `python -m src.verify --run` to compile.
"""
import argparse
import copy
import csv
import hashlib
import io
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import sklearn
from sklearn.metrics import accuracy_score

from src.p4gen.evaluation import accuracy_metrics, multi_model_memory_evaluation
from src.p4gen.switch_semantics import switch_predict
from src.training.campaign_run import INF, atomic_write_text, run_paths, split_csv_name
from src.training.campaign_run import row_id as make_row_id
from src.training.row_artifacts import RowContext, load_forests, write_row_artifacts
from src.training.splits import make_task_splits
from src.training.threshold_alignment import align_with_policy
from src.verify.runner import copy_record

CAMPAIGN_RANDOM_STATE = 42          # campaign_runner.DATA_RANDOM_STATE
SOURCE_ARM_SLUG = 'joint-off'
TWIN_ARM_SLUG = 'joint-off-al'
_ROW_ID = re.compile(r'^(?P<arm>.+)_M(?P<M>\d{3}|inf)_s(?P<split>\d+)_k(?P<k>\d+)$')


def split_random_state(split_idx):
    return CAMPAIGN_RANDOM_STATE + int(split_idx)


def column_indices(all_names, joined_names):
    """Column indices for a row's ';'-joined feature list."""
    out = []
    for name in joined_names.split(';'):
        if name not in all_names:
            raise ValueError(
                '{!r} is not in the loaded dataset feature list {!r}'.format(name, all_names))
        out.append(all_names.index(name))
    return out


def twin_row_id(source_row_id):
    m = _ROW_ID.match(source_row_id)
    if m is None or m.group('arm') != SOURCE_ARM_SLUG:
        raise ValueError(f'{source_row_id!r} is not a {SOURCE_ARM_SLUG} row id')
    M = float('inf') if m.group('M') == 'inf' else int(m.group('M'))
    return make_row_id(TWIN_ARM_SLUG, M, int(m.group('split')), int(m.group('k')))


def _usage(model_A, model_B, names_app, names_ddos):
    return multi_model_memory_evaluation(model_A, model_B, names_app, names_ddos, 'joint')


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def build_twin(ctx, source, forests, data, write_artifacts=True):
    """One twin row from one source row (spec §3). `forests` is
    `row_artifacts.load_forests(...)`; `data` is `(X_app, X_ddos, y_app,
    y_ddos, feature_names)` as `main.load_campaign_data` returns it."""
    X_app, X_ddos, y_app, y_ddos, names = data
    names_app = forests['features_app']
    names_ddos = forests['features_ddos']
    cols_app = column_indices(list(names), ';'.join(names_app))
    cols_ddos = column_indices(list(names), ';'.join(names_ddos))
    seed = split_random_state(source['split'])
    app = make_task_splits(X_app, y_app, seed)
    ddos = make_task_splits(X_ddos, y_ddos, seed)

    stats = {}
    model_A, model_B = align_with_policy(
        copy.deepcopy(forests['app']), copy.deepcopy(forests['ddos']),
        app.X_val_align[:, cols_app], app.y_val_align,
        ddos.X_val_align[:, cols_ddos], ddos.y_val_align,
        align_stats=stats)
    usage = _usage(model_A, model_B, names_app, names_ddos)

    tid = twin_row_id(source['row_id'])
    row = dict(source)
    row.update({'row_id': tid, 'source_row_id': source['row_id'],
                'alignment_enabled': False, 'alignment_postprocess': True,
                'align_attempted': stats.get('attempted'), 'align_accepted': stats.get('accepted'),
                'intervals_before': stats.get('intervals_before'),
                'intervals_after': stats.get('intervals_after')})

    src_blocks, src_depth = int(source['blocks']), int(source['stage_depth'])
    if usage.blocks > src_blocks or usage.stage_depth > src_depth:
        row['infeasible'] = ('TwinInvariantViolation: blocks {} > {} or stage_depth {} > {}'
                             .format(usage.blocks, src_blocks, usage.stage_depth, src_depth))
        row['twin_identical'] = ''
        return row

    with sklearn.config_context(assume_finite=True):
        acc_app, f1_app = accuracy_metrics(
            app.y_test, switch_predict(model_A, app.X_test[:, cols_app]), task='app')
        acc_ddos, f1_ddos = accuracy_metrics(
            ddos.y_test, switch_predict(model_B, ddos.X_test[:, cols_ddos]), task='ddos')
        sel_app = float(accuracy_score(
            app.y_val_select, switch_predict(model_A, app.X_val_select[:, cols_app])))
        sel_ddos = float(accuracy_score(
            ddos.y_val_select, switch_predict(model_B, ddos.X_val_select[:, cols_ddos])))
    row.update({'acc_app': acc_app, 'f1_app': f1_app, 'acc_ddos': acc_ddos, 'f1_ddos': f1_ddos,
                'acc_sel_app': sel_app, 'acc_sel_ddos': sel_ddos,
                'stages': int(usage.stages), 'blocks': int(usage.blocks),
                'stage_depth': int(usage.stage_depth),
                'range_entries': int(usage.range_entries),
                'ternary_entries': int(usage.ternary_entries),
                'register_depth': int(usage.register_depth),
                'register_count': int(usage.register_count), 'infeasible': ''})

    if not write_artifacts:
        row['twin_identical'] = ''
        return row
    designs = os.path.join(ctx.run_dir, 'designs')
    # Write the program first, compare, then rewrite model.json with the
    # verdict of the comparison: write_row_artifacts generates the program
    # and model.json in one call, so the second call only adds `identical_to`.
    write_row_artifacts(ctx, tid, model_A, model_B, names_app, names_ddos, 'joint', usage,
                        extra={'source_row_id': source['row_id'], 'identical_to': None})
    identical = (_sha256(os.path.join(designs, tid + '.p4'))
                 == _sha256(os.path.join(designs, source['row_id'] + '.p4')))
    if identical:
        write_row_artifacts(ctx, tid, model_A, model_B, names_app, names_ddos, 'joint', usage,
                            extra={'source_row_id': source['row_id'],
                                   'identical_to': source['row_id']})
    elif stats.get('accepted') == 0:
        print(f'{tid}: alignment accepted no move but the program differs from '
              f'{source["row_id"]} -- compiling it rather than copying', flush=True)
    row['twin_identical'] = identical
    return row


_INT_FIELDS = ('split', 'k', 'blocks', 'stage_depth', 'n_trees', 'max_depth')
_EXTRA_COLUMNS = ('source_row_id', 'twin_identical', 'alignment_postprocess')


def source_rows(run_dir):
    """Every joint-off design row (infeasible == ''), grouped by (M, split).
    M is float('inf') for the unbudgeted cell; values stay CSV text except
    the integer columns in `_INT_FIELDS`."""
    paths = run_paths(run_dir)
    groups = {}
    for name in sorted(os.listdir(paths.rows)):
        if not name.endswith('.csv') or f'_{SOURCE_ARM_SLUG}_s' not in name:
            continue
        with open(os.path.join(paths.rows, name), newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if (row.get('infeasible') or '').strip():
                    continue
                for field in _INT_FIELDS:
                    row[field] = int(float(row[field]))
                M = INF if row['M'].strip() in ('', 'inf') else float(row['M'])
                groups.setdefault((M, row['split']), []).append(row)
    return groups


def unverified_sources(run_dir, groups):
    """Source row ids with no verify/<id>.json."""
    paths = run_paths(run_dir)
    return [row['row_id'] for rows in groups.values() for row in rows
            if not os.path.isfile(os.path.join(paths.verify, row['row_id'] + '.json'))]


def twin_csv_path(run_dir, M, split, n_trees, max_depth):
    return os.path.join(run_paths(run_dir).rows,
                        split_csv_name(n_trees, max_depth, M, TWIN_ARM_SLUG, split))


def _source_columns(run_dir, M, split, n_trees, max_depth):
    path = os.path.join(run_paths(run_dir).rows,
                        split_csv_name(n_trees, max_depth, M, SOURCE_ARM_SLUG, split))
    with open(path, newline='', encoding='utf-8') as handle:
        return list(next(csv.reader(handle)))


def twins_for_split(run_dir, M, split, rows, data, git_commit):
    """Build every twin of one (M, split) group and write its CSV atomically
    (last, so the file is the done marker). Returns the path; a split whose
    CSV exists is skipped. A crash before the CSV leaves artifacts and maybe
    copied records behind: the re-run rebuilds the artifacts and keeps any
    record already copied."""
    paths = run_paths(run_dir).ensure()
    out = twin_csv_path(run_dir, M, split, rows[0]['n_trees'], rows[0]['max_depth'])
    if os.path.isfile(out):
        return out
    ctx = RowContext(run_dir, TWIN_ARM_SLUG, M, git_commit)
    twins = []
    for source in rows:
        forests = load_forests(os.path.join(paths.forests, source['row_id'] + '.joblib'))
        twin = build_twin(ctx, source, forests, data)
        if (twin['infeasible'] == '' and twin['twin_identical'] is True
                and not os.path.isfile(os.path.join(paths.verify, twin['row_id'] + '.json'))):
            copy_record(run_dir, twin['row_id'], source['row_id'])
        twins.append(twin)
    fieldnames = _source_columns(run_dir, M, split, rows[0]['n_trees'], rows[0]['max_depth'])
    fieldnames += [c for c in _EXTRA_COLUMNS if c not in fieldnames]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    for twin in twins:
        writer.writerow({k: ('' if v is None else v) for k, v in twin.items()})
    atomic_write_text(out, buffer.getvalue())
    return out


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def run(run_dir, data=None, workers=1, git_commit=None, executor_factory=None):
    """0 = all splits done, 1 = at least one split failed, 2 = refused
    (nothing to do, or a source design is unverified; nothing written)."""
    groups = source_rows(run_dir)
    if not groups:
        print(f'{run_dir}: no {SOURCE_ARM_SLUG} design rows', flush=True)
        return 2
    missing = unverified_sources(run_dir, groups)
    if missing:
        print(f'refusing: {len(missing)} {SOURCE_ARM_SLUG} designs have no verification record '
              f'(run python -m src.verify --run {run_dir} first):', flush=True)
        for row_id in missing[:20]:
            print(f'  {row_id}', flush=True)
        return 2
    if data is None:
        from src.main import load_campaign_data
        data = load_campaign_data()
    if git_commit is None:
        git_commit = _git_commit()
    executor_factory = executor_factory or ProcessPoolExecutor
    failed = {}
    with executor_factory(max(1, workers)) as pool:
        futures = {pool.submit(twins_for_split, run_dir, M, split, rows, data, git_commit): (M, split)
                   for (M, split), rows in sorted(groups.items())}
        for future in as_completed(futures):
            key = futures[future]
            try:
                print(f'done {key}: {future.result()}', flush=True)
            except Exception as exc:          # one bad split must not stop the others
                failed[key] = f'{type(exc).__name__}: {exc}'
                print(f'FAILED {key}: {failed[key]}', flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m src.training.align_twins')
    parser.add_argument('--run', required=True, metavar='DIR')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    return run(args.run, workers=args.workers)


if __name__ == '__main__':
    sys.exit(main())
