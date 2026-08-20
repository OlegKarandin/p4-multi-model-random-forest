"""Capacity-ceiling rederivation (P4, spec B.7 / F10i).

`TrainConfig.n_trees` and `TrainConfig.max_depth` are *inclusive search
bounds*, not fixed hyperparameters: `train_model.rf_params` suggests
`n_estimators in [1, cfg.n_trees]` and `max_depth in [2, cfg.max_depth]`
(train_model.py:96-99). Their values were placeholders, justified in the thesis
only as "chosen manually because larger values gave overly long codewords".
This script replaces that sentence with a table: it measures, over a grid of
bounds, where the 512-bit codeword limit (`MAX_CODEWORD_LENGTH`) starts to
bind, and applies the adoption rule to the measurement.

Method. For every cell of the grid `n_trees x max_depth`, and for each of the
first 3 campaign splits, one App forest and one DDoS forest are fitted on the
full training split at the full selected-feature set, and the codeword length,
TCAM blocks and TCAM stages are recorded under BOTH encodings. No Optuna, no
alignment, no feature elimination -- this measures the capacity of the *bound*,
not the quality of a model.

Worst-case corner. Each cell is measured at the corner of the Optuna search box
that produces the LARGEST trees: `n_estimators = n_trees`, `max_depth =
max_depth`, `min_samples_leaf = 5`, `min_samples_split = 10` (the lower ends of
`rf_params`' ranges). Codeword length grows with the number of distinct splits,
so this is the longest codeword the box admits. A cell that fits at its corner
therefore admits its whole box, and a cell that does not fit is one whose
corner Optuna would only ever sample and discard.

The measurement is corner-conditional, and the script says so. Re-running it
with `MIN_SAMPLES_LEAF = 200` / `MIN_SAMPLES_SPLIT = 400` -- the OPPOSITE
corner, where every tree is heavily pruned -- moves the ceiling up by roughly a
factor of three in depth: only (15, >=10) exceeds the limit there, and the rule
would then select (11, 14) at cardinality 143 instead of (15, 4) at 45. The
large-tree corner is the reading taken here because a bound is a promise about
the LARGEST forest the search may request, and a cell whose own corner cannot be
compiled is a cell Optuna can only sample and discard.

Codeword length above the limit. `multi_model_memory_evaluation` raises
`RuntimeError("Codewords are too long", codeword_length)` rather than returning
when the limit is exceeded (evaluation.py:176), so over-512 cells would be lost
-- and a table containing only feasible cells cannot locate a ceiling. The
length is therefore derived directly from the feature intervals (it is
`sum(len(intervals) - 1)` over features, exactly what `generate_codewords`
emits per leaf) so that it exists for every cell, and is cross-checked against
`e.args[1]` on every cell that raises.

Adoption rule (Ruling P4-1). Among cells whose JOINT-encoding codeword length
stays within the limit on all 3 splits, rank by admissible search-space
cardinality `n_trees * (max_depth - 1)` -- the number of `(n_estimators,
max_depth)` pairs the Optuna bounds admit -- take the maximum, and break ties
toward the smaller `n_trees`. The script prints the cardinality column and the
justifying row so the adopted values are traceable to the table.

Writes results/capacity_ceiling.csv (one row per cell per split) and prints the
markdown tables.

Run:
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/capacity_ceiling.py
"""
import os
import sys
import time

# Running a file inside scripts/ puts scripts/ on sys.path, not the repo root,
# so `src` would not import under the command in the docstring above.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.main import remove_correlated_features_both_datasets
from src.p4gen.build_p4_script import (
    INFINITE, MAX_CODEWORD_LENGTH, dt_thresholds_float_to_int,
    get_feature_intervals_from_thresholds, get_feature_thresholds, get_nodes,
    get_tree_textual_representation)
from src.p4gen.evaluation import multi_model_memory_evaluation
from src.training.dataset import read_app_dataset, read_DDOS_dataset
from src.training.splits import make_task_splits

SELECTED_FEATURES = [
    'Fwd.Packet.Length.Max', 'Fwd.Packet.Length.Min', 'Fwd.Packet.Length.Mean',
    'Bwd.Packet.Length.Max', 'Bwd.Packet.Length.Min', 'Bwd.Packet.Length.Mean',
    'Flow.IAT.Mean', 'Flow.IAT.Max', 'Flow.IAT.Min',
    'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min',
    'Bwd.IAT.Mean', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
    'Min.Packet.Length', 'Max.Packet.Length', 'Packet.Length.Mean']

N_TREES_GRID = (1, 3, 5, 7, 9, 11, 15)
MAX_DEPTH_GRID = (2, 4, 6, 8, 10, 12, 14)

# The campaign's own first three splits: feature_selection.py submits
# split_idx in range(10, 10 + n_splits) and each worker seeds its splits with
# random_state + split_idx (feature_selection.py:447, 580) at random_state=42.
SPLIT_INDICES = (10, 11, 12)
SPLIT_RANDOM_STATE = 42

# The large-tree corner of rf_params' search box (train_model.py:96-99).
MIN_SAMPLES_LEAF = 5
MIN_SAMPLES_SPLIT = 10
RF_RANDOM_STATE = 42


def tree_nodes_of(clf, feature_names):
    """One model's parsed nodes, via the same path evaluation.py uses."""
    trees = get_tree_textual_representation(clf, feature_names)
    return {tree: get_nodes(trees[tree]) for tree in trees}


def joint_tree_nodes(clf_app, clf_ddos, feature_names):
    """Both models' nodes merged into one dict, mirroring
    `multi_model_memory_evaluation`'s 'joint' branch offset trick exactly --
    the merged set is what the shared discretization is derived from."""
    nodes = tree_nodes_of(clf_app, feature_names)
    offset = len(nodes)
    for tree, tree_node in tree_nodes_of(clf_ddos, feature_names).items():
        nodes[tree + offset] = tree_node
    return nodes


def codeword_length_of(tree_nodes):
    """Bits in one codeword for the discretization these nodes induce.

    `generate_codewords` emits `len(feature_intervals[f]) - 1` characters per
    feature f, for every leaf of every tree (build_p4_script.py:397, 405), so
    the length is a property of the intervals alone. Deriving it here rather
    than reading it off a return value is what lets an over-512 cell be
    recorded instead of lost; `measure` asserts it against the length the
    evaluator itself reports whenever the evaluator raises.
    """
    intervals = get_feature_intervals_from_thresholds(
        get_feature_thresholds(tree_nodes))
    return sum(len(ranges) - 1 for ranges in intervals.values())


def measure(clf_app, clf_ddos, feature_names, encoding, codeword_length):
    """(stages, blocks) for one encoding, or (None, None) when the codeword
    limit makes the pair uncompilable. Returns Nones rather than propagating,
    so an infeasible cell still contributes its codeword length to the table."""
    try:
        return multi_model_memory_evaluation(
            clf_app, clf_ddos, feature_names, feature_names, encoding)
    except RuntimeError as e:
        reported = e.args[1]
        if encoding == 'joint' and reported != codeword_length:
            raise AssertionError(
                'joint codeword length {} disagrees with the evaluator\'s {}'
                .format(codeword_length, reported))
        if encoding == 'disjoint' and reported > codeword_length:
            # Under disjoint encoding the evaluator raises on whichever model
            # it reaches first, so its number is one of the two per-model
            # lengths -- never longer than the larger of them.
            raise AssertionError(
                'disjoint codeword length {} is below the evaluator\'s {}'
                .format(codeword_length, reported))
        return None, None


def fit(X, y, n_estimators, max_depth):
    return dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RF_RANDOM_STATE, n_jobs=1).fit(X, y))


def collect():
    """One row per (n_trees, max_depth, split)."""
    df_app = read_app_dataset(SELECTED_FEATURES, INFINITE)
    df_ddos = read_DDOS_dataset(SELECTED_FEATURES, INFINITE)
    X_app, X_ddos, names = remove_correlated_features_both_datasets(df_app, df_ddos)
    y_app = df_app.Label.to_numpy()
    y_ddos = df_ddos.Label.to_numpy()

    rows = []
    for split_idx in SPLIT_INDICES:
        seed = SPLIT_RANDOM_STATE + split_idx
        app = make_task_splits(X_app, y_app, seed)
        ddos = make_task_splits(X_ddos, y_ddos, seed)
        print('\nsplit_idx={} (seed {}): {} app / {} ddos training rows, '
              '{} features'.format(split_idx, seed, len(app.y_train),
                                   len(ddos.y_train), len(names)))

        for n_trees in N_TREES_GRID:
            for max_depth in MAX_DEPTH_GRID:
                start = time.perf_counter()
                clf_app = fit(app.X_train, app.y_train, n_trees, max_depth)
                clf_ddos = fit(ddos.X_train, ddos.y_train, n_trees, max_depth)

                joint_len = codeword_length_of(
                    joint_tree_nodes(clf_app, clf_ddos, names))
                app_len = codeword_length_of(tree_nodes_of(clf_app, names))
                ddos_len = codeword_length_of(tree_nodes_of(clf_ddos, names))
                # Each model gets its own ternary table under disjoint
                # encoding, so the limit binds per model: the pair is
                # compilable iff the LONGER of the two fits.
                disjoint_len = max(app_len, ddos_len)

                joint_stages, joint_blocks = measure(
                    clf_app, clf_ddos, names, 'joint', joint_len)
                disjoint_stages, disjoint_blocks = measure(
                    clf_app, clf_ddos, names, 'disjoint', disjoint_len)

                rows.append({
                    'n_trees': n_trees,
                    'max_depth': max_depth,
                    'cardinality': n_trees * (max_depth - 1),
                    'split_idx': split_idx,
                    'split_seed': seed,
                    'joint_codeword_length': joint_len,
                    'joint_within_limit': joint_len <= MAX_CODEWORD_LENGTH,
                    'joint_stages': joint_stages,
                    'joint_blocks': joint_blocks,
                    'disjoint_codeword_length_app': app_len,
                    'disjoint_codeword_length_ddos': ddos_len,
                    'disjoint_codeword_length': disjoint_len,
                    'disjoint_within_limit': disjoint_len <= MAX_CODEWORD_LENGTH,
                    'disjoint_stages': disjoint_stages,
                    'disjoint_blocks': disjoint_blocks,
                    'seconds': time.perf_counter() - start,
                })

                print('  t={:<3} d={:<3} joint cw={:<5}{} blocks={:<6} | '
                      'disjoint cw={:<5}{} blocks={:<6} | {:.1f}s'.format(
                          n_trees, max_depth, joint_len,
                          '!' if joint_len > MAX_CODEWORD_LENGTH else ' ',
                          '-' if joint_blocks is None else joint_blocks,
                          disjoint_len,
                          '!' if disjoint_len > MAX_CODEWORD_LENGTH else ' ',
                          '-' if disjoint_blocks is None else disjoint_blocks,
                          rows[-1]['seconds']))

    frame = pd.DataFrame(rows)
    # None marks "no block/stage count exists because the codeword is too
    # long". Plain object columns would render those counts as 2.0 in the CSV;
    # the nullable integer dtype keeps them integers and leaves the cell empty.
    for column in ('joint_stages', 'joint_blocks',
                   'disjoint_stages', 'disjoint_blocks'):
        frame[column] = frame[column].astype('Int64')
    return frame


def per_cell(frame):
    """Collapse the 3 splits into the worst case per cell -- the adoption rule
    asks for cells that stay within the limit on ALL splits, so the maximum
    over splits is the deciding statistic."""
    frame = frame.copy()
    for column in ('joint_stages', 'joint_blocks',
                   'disjoint_stages', 'disjoint_blocks'):
        # None (the infeasible marker) makes the column dtype object, which
        # groupby.max cannot order; NaN is skipped instead.
        frame[column] = pd.to_numeric(frame[column], errors='coerce')

    cells = frame.groupby(['n_trees', 'max_depth'], as_index=False).agg(
        cardinality=('cardinality', 'max'),
        joint_cw_max=('joint_codeword_length', 'max'),
        joint_cw_min=('joint_codeword_length', 'min'),
        joint_blocks_max=('joint_blocks', 'max'),
        joint_within_limit_all_splits=('joint_within_limit', 'all'),
        disjoint_cw_max=('disjoint_codeword_length', 'max'),
        disjoint_blocks_max=('disjoint_blocks', 'max'),
        disjoint_within_limit_all_splits=('disjoint_within_limit', 'all'))

    # A cell that exceeds the limit on ANY split has no usable block count --
    # reporting the surviving split's number would read as a feasible cell.
    cells.loc[~cells.joint_within_limit_all_splits, 'joint_blocks_max'] = float('nan')
    cells.loc[~cells.disjoint_within_limit_all_splits, 'disjoint_blocks_max'] = float('nan')
    return cells


def print_grid(cells, column, title, note):
    print('\n### {}\n'.format(title))
    print('{}\n'.format(note))
    header = ['n_trees \\ max_depth'] + [str(d) for d in MAX_DEPTH_GRID]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for n_trees in N_TREES_GRID:
        line = ['**{}**'.format(n_trees)]
        for max_depth in MAX_DEPTH_GRID:
            row = cells[(cells.n_trees == n_trees) & (cells.max_depth == max_depth)]
            value = row[column].iloc[0]
            if pd.isna(value):
                line.append('-')
                continue
            value = int(value)
            over = column.endswith('cw_max') and value > MAX_CODEWORD_LENGTH
            line.append('**{}**'.format(value) if over else str(value))
        print('| ' + ' | '.join(line) + ' |')


def print_cell_table(cells):
    print('\n### Per-cell summary, ranked by admissible search-space '
          'cardinality (Ruling P4-1)\n')
    print('`cardinality = n_trees * (max_depth - 1)` is the number of '
          '`(n_estimators, max_depth)` pairs the Optuna bounds admit. '
          '"feasible" means the joint codeword length stayed within {} bits on '
          'all {} splits.\n'.format(MAX_CODEWORD_LENGTH, len(SPLIT_INDICES)))
    header = ['n_trees', 'max_depth', 'cardinality', 'joint cw (worst split)',
              'joint blocks', 'disjoint cw (worst split)', 'disjoint blocks',
              'feasible']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    ordered = cells.sort_values(
        ['joint_within_limit_all_splits', 'cardinality', 'n_trees'],
        ascending=[False, False, True])
    for _, row in ordered.iterrows():
        print('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            int(row.n_trees), int(row.max_depth), int(row.cardinality),
            int(row.joint_cw_max),
            '-' if pd.isna(row.joint_blocks_max) else int(row.joint_blocks_max),
            int(row.disjoint_cw_max),
            '-' if pd.isna(row.disjoint_blocks_max) else int(row.disjoint_blocks_max),
            'yes' if row.joint_within_limit_all_splits else 'NO'))


def print_where_the_limit_binds(cells):
    print('\n### Where the {}-bit limit starts to bind\n'.format(MAX_CODEWORD_LENGTH))
    print('| n_trees | smallest max_depth exceeding the limit (joint) | '
          'smallest max_depth exceeding the limit (disjoint) |')
    print('|---|---|---|')
    for n_trees in N_TREES_GRID:
        row = cells[cells.n_trees == n_trees]
        cell = []
        for column in ('joint_within_limit_all_splits',
                       'disjoint_within_limit_all_splits'):
            over = row[~row[column]].max_depth
            cell.append(str(int(over.min())) if len(over) else
                        'never within this grid')
        print('| {} | {} | {} |'.format(n_trees, cell[0], cell[1]))


def select(cells):
    """Ruling P4-1: maximum admissible cardinality among cells within the limit
    on all splits, ties broken toward the smaller n_trees."""
    feasible = cells[cells.joint_within_limit_all_splits]
    if not len(feasible):
        raise RuntimeError('no grid cell stays within the codeword limit')

    best = feasible.cardinality.max()
    tied = feasible[feasible.cardinality == best].sort_values('n_trees')
    chosen = tied.iloc[0]

    print('\n### Adopted values\n')
    print('{} of {} grid cells keep the joint codeword within {} bits on all '
          '{} splits. The largest admissible search space among them has '
          'cardinality n_trees * (max_depth - 1) = {}, attained by {}.'.format(
              len(feasible), len(cells), MAX_CODEWORD_LENGTH,
              len(SPLIT_INDICES), int(best),
              ', '.join('({}, {})'.format(int(r.n_trees), int(r.max_depth))
                        for _, r in tied.iterrows())))
    if len(tied) > 1:
        print('Ties are broken toward the smaller n_trees (Ruling P4-1).')
    print('\n**Adopted: n_trees = {}, max_depth = {}** -- justified by the row '
          '(n_trees={}, max_depth={}) of the per-cell table: cardinality {}, '
          'joint codeword length {} bits at its worst split (limit {}), '
          '{} joint blocks.'.format(
              int(chosen.n_trees), int(chosen.max_depth),
              int(chosen.n_trees), int(chosen.max_depth),
              int(chosen.cardinality), int(chosen.joint_cw_max),
              MAX_CODEWORD_LENGTH,
              '-' if pd.isna(chosen.joint_blocks_max) else int(chosen.joint_blocks_max)))
    return int(chosen.n_trees), int(chosen.max_depth)


def main():
    frame = collect()
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', 'capacity_ceiling.csv')
    frame.to_csv(path, index=False)
    print('\nwrote {} ({} rows)'.format(path, len(frame)))

    cells = per_cell(frame)
    print_grid(cells, 'joint_cw_max',
               'Joint-encoding codeword length, worst of {} splits'.format(
                   len(SPLIT_INDICES)),
               'Bold exceeds the {}-bit limit, so the pair does not '
               'compile.'.format(MAX_CODEWORD_LENGTH))
    print_grid(cells, 'disjoint_cw_max',
               'Disjoint-encoding codeword length (longer of the two models), '
               'worst of {} splits'.format(len(SPLIT_INDICES)),
               'Bold exceeds the {}-bit limit.'.format(MAX_CODEWORD_LENGTH))
    print_grid(cells, 'joint_blocks_max',
               'Joint-encoding TCAM blocks, worst of {} splits'.format(
                   len(SPLIT_INDICES)),
               '"-" is a cell whose codeword exceeds the limit, so no block '
               'count exists.')
    print_grid(cells, 'disjoint_blocks_max',
               'Disjoint-encoding TCAM blocks, worst of {} splits'.format(
                   len(SPLIT_INDICES)),
               '"-" is a cell whose codeword exceeds the limit, so no block '
               'count exists.')
    print_where_the_limit_binds(cells)
    print_cell_table(cells)
    select(cells)


if __name__ == '__main__':
    main()
