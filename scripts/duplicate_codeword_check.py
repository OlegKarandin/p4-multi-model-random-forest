"""M6 -- do two leaves with the SAME codeword but DIFFERENT predicted classes
actually occur in any real trained forest? (docs/2026-09-20-version-block-findings.md)

WHY THIS EXISTS. `build_p4_script.generate_codewords` keys its per-tree
output dict by codeword STRING and silently keeps whichever leaf was written
last (build_p4_script.py:540-545's own "Known open item" comment). This is a
correctness question, not a cost one: if it ever fires on a real forest, the
deployed P4 table would classify some inputs differently than the trained
sklearn model does, silently. This is the only item on the 2026-09-20
findings list that would change reported ACCURACY, not TCAM usage -- hence
its priority in that doc's Part 4 (step 6, "highest thesis-validity payoff").

Not a compile -- a pure Python check over real, deterministically-refit
campaign forests (the same `scripts.replay_alignment.refit_pair` machinery
`scripts/compiler_calibration.py` uses to get real trained
RandomForestClassifiers without re-running Optuna).

METHOD. `generate_codewords` cannot be probed after the fact (the dict
overwrite already happened), so this calls it once PER LEAF -- passing a
{tree: {leaf_id: leaf}} singleton each time -- which reuses the exact
production codeword-generation logic with no duplication and no dict ever
overwrites anything, then groups the resulting (tree, codeword) -> {classes}
by hand. A collision is a (tree, codeword) pair whose class set has more
than one member.

Checks BOTH real encodings this project ships: 'disjoint' (each task's own
feature_intervals, matching single_model_memory_evaluation) and 'joint' (one
merged tree_nodes/feature_intervals set across both tasks, matching
_pool_inputs's 'joint' branch) -- a collision is possible under 'joint' even
when neither task collides alone, since joint widens every feature's
interval set to the union of what both tasks split on.

Run it (from the repository root; needs the PolimiML sklearn environment --
patch_sklearn() elsewhere in this env breaks RandomForestClassifier.fit(),
so use this python.exe exactly, matching every other script here that trains
real forests):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/duplicate_codeword_check.py --n-rows 30

Resumable: rows already present in --out are skipped.
"""
import argparse
import collections
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.compiler_calibration import CAMPAIGN_BACKUP_DIR
from scripts.replay_alignment import column_indices, load_backup, refit_pair
from src.main import load_campaign_data
from src.p4gen.build_p4_script import (feature_intervals_from_nodes,
                                        generate_codewords, get_root_to_leaf_paths,
                                        merge_tree_nodes, tree_nodes_for)

DEFAULT_OUT = 'results/duplicate_codeword_check.csv'


def all_codewords(paths_leaf_nodes_per_tree, feature_intervals):
    """[(tree, leaf_id, codeword, class)] for EVERY leaf -- no dict overwrite,
    since generate_codewords is called on a one-leaf slice each time and its
    own per-leaf codeword logic (build_p4_script.py:484-545) is otherwise
    untouched."""
    records = []
    for tree, leaves in paths_leaf_nodes_per_tree.items():
        for leaf_id, leaf in leaves.items():
            singleton = {tree: {leaf_id: leaf}}
            codewords = generate_codewords(singleton, feature_intervals)
            codeword, cls = next(iter(codewords[tree].items()))
            records.append((tree, leaf_id, codeword, cls))
    return records


def find_collisions(records):
    """{(tree, codeword): {classes}} for every (tree, codeword) pair whose
    leaves do NOT all agree on the class -- the generator-correctness bug
    this check exists to find, if it exists at all."""
    by_key = collections.defaultdict(set)
    for tree, _leaf_id, codeword, cls in records:
        by_key[(tree, codeword)].add(cls)
    return {k: v for k, v in by_key.items() if len(v) > 1}


def check_one_encoding(tree_nodes, feature_intervals):
    paths = get_root_to_leaf_paths(tree_nodes)
    records = all_codewords(paths, feature_intervals)
    collisions = find_collisions(records)
    return {
        'n_trees': len(tree_nodes),
        'n_leaves': len(records),
        'n_distinct_codewords': len({(t, c) for t, _, c, _ in records}),
        'n_collisions': len(collisions),
        'collision_examples': json.dumps(
            [{'tree': t, 'codeword': c, 'classes': sorted(str(x) for x in v)}
             for (t, c), v in list(collisions.items())[:5]]),
    }


def run_row(row, data):
    model_app, model_ddos, app, ddos, cols_app, cols_ddos = refit_pair(row, data)
    names = data[4]
    names_app = [names[i] for i in cols_app]
    names_ddos = [names[i] for i in cols_ddos]

    tree_nodes_app = tree_nodes_for(model_app, names_app)
    tree_nodes_ddos = tree_nodes_for(model_ddos, names_ddos)

    out = {
        'row_id': row.get('row_id', row.name),
        'arm_slug': row['arm_slug'], 'M': row['M'], 'k': row['k'],
        'split': row['split'],
        'n_features_app': len(names_app), 'n_features_ddos': len(names_ddos),
    }

    disjoint_app = check_one_encoding(
        tree_nodes_app, feature_intervals_from_nodes(tree_nodes_app))
    disjoint_ddos = check_one_encoding(
        tree_nodes_ddos, feature_intervals_from_nodes(tree_nodes_ddos))
    for prefix, result in (('disjoint_app', disjoint_app),
                           ('disjoint_ddos', disjoint_ddos)):
        for key, value in result.items():
            out['%s_%s' % (prefix, key)] = value

    merged = merge_tree_nodes(tree_nodes_app, tree_nodes_ddos)
    joint = check_one_encoding(merged, feature_intervals_from_nodes(merged))
    for key, value in joint.items():
        out['joint_%s' % key] = value

    return out


def select_sample(frame, n_rows, seed):
    """A deterministic, encoding-diverse sample: half from 'independent'
    (disjoint encoding in production), half spread across the joint arms --
    picking distinct (arm_slug, M, k, split) combinations rather than n
    duplicates of the same shape."""
    if 'infeasible' in frame.columns:
        frame = frame[frame['infeasible'].isna() | (frame['infeasible'] == '')]
    frame = frame.dropna(subset=['best_params'])
    independent = frame[frame['arm_slug'] == 'independent']
    joint = frame[frame['arm_slug'].str.startswith('joint-')]

    def pick(pool, n):
        pool = pool.drop_duplicates(subset=['arm_slug', 'M', 'k', 'split'])
        return pool.sample(n=min(n, len(pool)), random_state=seed)

    half = n_rows // 2
    sample = pd.concat([pick(independent, half), pick(joint, n_rows - half)])
    return sample.reset_index().rename(columns={'index': 'row_id'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--backup-dir', default=CAMPAIGN_BACKUP_DIR)
    parser.add_argument('--n-rows', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    print('loading campaign backup from %s ...' % args.backup_dir, flush=True)
    frame = load_backup(args.backup_dir)
    sample = select_sample(frame, args.n_rows, args.seed)
    print('sampled %d rows: %s' %
          (len(sample), sample['arm_slug'].value_counts().to_dict()), flush=True)

    print('loading campaign dataset ...', flush=True)
    data = load_campaign_data()

    done = set()
    rows = []
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        rows = existing.to_dict('records')
        done = {int(r) for r in existing['row_id']}

    total_collisions = 0
    for _, row in sample.iterrows():
        if int(row['row_id']) in done:
            continue
        print('=== row %s (%s, M=%s, k=%s, split=%s) ...' %
              (row['row_id'], row['arm_slug'], row['M'], row['k'], row['split']),
              flush=True)
        started = time.time()
        try:
            out = run_row(row, data)
        except Exception:
            traceback.print_exc()
            out = {'row_id': int(row['row_id']), 'arm_slug': row['arm_slug'],
                  'error': 1}
        out['seconds'] = round(time.time() - started, 1)
        rows.append(out)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        n_coll = (out.get('disjoint_app_n_collisions', 0)
                 + out.get('disjoint_ddos_n_collisions', 0)
                 + out.get('joint_n_collisions', 0))
        total_collisions += n_coll if n_coll == n_coll else 0   # skip NaN
        print('    disjoint_app=%s leaves/%s collisions  disjoint_ddos=%s/%s  '
              'joint=%s/%s  (%.1fs)' % (
                  out.get('disjoint_app_n_leaves'), out.get('disjoint_app_n_collisions'),
                  out.get('disjoint_ddos_n_leaves'), out.get('disjoint_ddos_n_collisions'),
                  out.get('joint_n_leaves'), out.get('joint_n_collisions'),
                  out['seconds']), flush=True)

    frame_out = pd.DataFrame(rows)
    print('\nwrote %s (%d rows)' % (args.out, len(frame_out)))
    for col in ('disjoint_app_n_collisions', 'disjoint_ddos_n_collisions',
               'joint_n_collisions'):
        if col in frame_out.columns:
            print('  total %s across all rows: %d' % (col, frame_out[col].fillna(0).sum()))
    any_collisions = frame_out.get('disjoint_app_n_collisions', pd.Series(dtype=float)).fillna(0).sum() \
                    + frame_out.get('disjoint_ddos_n_collisions', pd.Series(dtype=float)).fillna(0).sum() \
                    + frame_out.get('joint_n_collisions', pd.Series(dtype=float)).fillna(0).sum()
    print('\n%s: duplicate-codeword-different-class collisions %s across %d real '
          'trained-forest rows.' % (
              'FOUND' if any_collisions else 'NOT FOUND',
              'were found' if any_collisions else 'were not found', len(frame_out)))


if __name__ == '__main__':
    main()
