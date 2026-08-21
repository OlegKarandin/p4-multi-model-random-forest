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

Two corners, because the ceiling is not a property of (n_trees, max_depth)
alone. A cell is a BOUND, so measuring "its" codeword length means fixing the
regularization that `rf_params` also searches -- `min_samples_leaf` over
[5, 200] step 10 (Optuna's `suggest_int` therefore clips the reachable set to
[5, 195]; 200 is never actually selectable) and `min_samples_split` over
[10, 400] step 10 (400 IS reachable: 10 + 10*39) -- and the answer depends on
which end you fix it at. Both ends are measured for every cell:

  pruned      min_samples_leaf=195, min_samples_split=400 -- the smallest
              forests the box admits. A cell feasible here is a cell the
              campaign can genuinely reach, with pruning.
  large-tree  min_samples_leaf=5, min_samples_split=10 -- the largest forests
              the box admits, i.e. the longest codeword anywhere in the box.

Which corner decides (Ruling P4-2): the PRUNED one. A cell counts as feasible
when ANY configuration the search can actually reach there is feasible.
Deciding on the large-tree corner would instead demand that the ENTIRE box
compile -- and that guarantee is not real: at (15, 4), feasible under that
reading, the pair still needs 257 TCAM blocks against a campaign M grid of
25-100, so the box contains infeasible points either way. That reading pays for
frontier truncation and does not receive the guarantee it paid for. The search
is built for boxes with infeasible regions in them: `train_model`'s objective
records violation MAGNITUDES rather than booleans so the constrained sampler
can order infeasible trials by how badly they miss, and
`early_stopping.is_feasible` keys stopping on feasible-front movement, so
infeasible trials cannot end a search early. Both staircases are printed
anyway, so a reader can see that the ceiling moves with pruning rather than
being a property of the two bounds alone.

A box with an infeasible corner is established practice here, not a new risk.
The bounds are per-axis and independent: at the old (7, 10) placeholders,
`rf_params` suggested `n_estimators` from {1, 3, 5, 7} and `max_depth` from
[2, 10] separately, so the joint corner (7 trees at depth 10) -- 1599 bits in
the table below -- was suggestable but never selectable, and the campaign's
deployed models were combinations like (7, 4) or (3, 10). Nothing about those
runs was invalid; what was missing is the measurement, since the bounds were
placeholders carrying a comment saying P4 would derive them and no one had ever
located the ceiling. That is why this table exists. It also means the
worst-case-corner reading would impose a stricter requirement than this project
has ever applied to itself.

Codeword length above the limit. `multi_model_memory_evaluation` raises
`RuntimeError("Codewords are too long", codeword_length)` rather than returning
when the limit is exceeded (evaluation.py:176), so over-512 cells would be lost
-- and a table containing only feasible cells cannot locate a ceiling. The
length is therefore derived directly from the feature intervals (it is
`sum(len(intervals) - 1)` over features, exactly what `generate_codewords`
emits per leaf) so that it exists for every cell, and is cross-checked against
`e.args[1]` on every cell that raises.

Adoption rule (Rulings P4-1 and P4-3). Among cells whose JOINT-encoding
codeword length stays within the limit on all 3 splits at the deciding corner,
rank by admissible search-space cardinality, take the maximum, and break ties
toward the smaller `n_trees`. Cardinality is
`ceil(n_trees / 2) * (max_depth - 1)`: `rf_params` suggests `n_estimators` with
`step=2` from 1, so only every other tree count is reachable and the tree axis
offers `ceil(n_trees / 2)` values, not `n_trees` -- Ruling P4-3 correcting
P4-1's formula, which overcounted that axis ~2x uniformly (being monotone in
the same direction, it changes no ranking). The script prints the cardinality
column and the justifying row, so the adopted values are traceable to a row.

Writes results/capacity_ceiling.csv (one row per cell per split per corner) and
prints the markdown tables.

Run:
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/capacity_ceiling.py
"""
import os
import sys
import time
from collections import namedtuple

# Running a file inside scripts/ puts scripts/ on sys.path, not the repo root,
# so `src` would not import under the command in the docstring above.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# `src.main` is imported inside `collect()`, not here. It is needed only by
# the MEASUREMENT half of this script; the REPORTING half (per_cell /
# print_* / select / report) is imported by `src/reporting/figures.py` to
# persist this script's markdown without re-running the ~10-minute
# measurement, and keeping `src.main` out of this module's own top-level
# imports means that reporting-only import path never pulls in training/P4
# codegen dependencies it does not need. (P7d retired `src.reporting.
# plotting`, whose global matplotlib state mutation at import -- `plt.style.
# use('default')` / `sns.set_palette` -- used to be the sharper reason this
# mattered; that module is gone, but the import-cost argument stands on its
# own.)
from src.p4gen.build_p4_script import (
    INFINITE, MAX_CODEWORD_LENGTH, dt_thresholds_float_to_int,
    get_feature_intervals, get_joint_feature_intervals)
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

# The two ends of rf_params' regularization ranges (train_model.py:96-99).
# min_samples_leaf is suggested as suggest_int(5, 200, step=10), which Optuna
# clips to the reachable set [5, 195] -- 200 is never actually selectable, so
# the pruned corner uses 195, the true edge of the search space. Effect is
# nil in practice: the deciding corner (11, 14) measures 413 bits against the
# 512-bit limit either way, comfortable margin regardless of which of the two
# values is used.
Corner = namedtuple('Corner', 'name min_samples_leaf min_samples_split')
PRUNED = Corner('pruned', 195, 400)
LARGE_TREE = Corner('large-tree', 5, 10)
CORNERS = (PRUNED, LARGE_TREE)

# Ruling P4-2: a cell is feasible when the search can reach ANY feasible
# configuration there, which is what the pruned corner witnesses.
DECIDING_CORNER = PRUNED

RF_RANDOM_STATE = 42


def cardinality_of(n_trees, max_depth):
    """Reachable `(n_estimators, max_depth)` pairs inside the bounds.

    `n_estimators` is suggested with step=2 from 1, so the tree axis offers
    ceil(n_trees / 2) values, not n_trees (Ruling P4-3). `max_depth` is
    suggested with the default step of 1 from 2, so its axis offers
    max_depth - 1 values.
    """
    return -(-n_trees // 2) * (max_depth - 1)


def codeword_length_of(feature_intervals):
    """Bits in one codeword for this discretization.

    `generate_codewords` emits `len(feature_intervals[f]) - 1` characters per
    feature f, for every leaf of every tree (build_p4_script.py:397, 405), so
    the length is a property of the intervals alone. Deriving it here rather
    than reading it off a return value is what lets an over-512 cell be
    recorded instead of lost; `measure` asserts it against the length the
    evaluator itself reports whenever the evaluator raises.
    """
    return sum(len(ranges) - 1 for ranges in feature_intervals.values())


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


def fit(X, y, n_estimators, max_depth, corner):
    return dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=corner.min_samples_leaf,
        min_samples_split=corner.min_samples_split,
        random_state=RF_RANDOM_STATE, n_jobs=1).fit(X, y))


def collect():
    """One row per (n_trees, max_depth, split, corner)."""
    from src.main import remove_correlated_features_both_datasets

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
                for corner in CORNERS:
                    start = time.perf_counter()
                    clf_app = fit(app.X_train, app.y_train, n_trees, max_depth, corner)
                    clf_ddos = fit(ddos.X_train, ddos.y_train, n_trees, max_depth, corner)

                    joint_len = codeword_length_of(
                        get_joint_feature_intervals(clf_app, names, clf_ddos, names))
                    app_len = codeword_length_of(get_feature_intervals(clf_app, names))
                    ddos_len = codeword_length_of(get_feature_intervals(clf_ddos, names))
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
                        'cardinality': cardinality_of(n_trees, max_depth),
                        'corner': corner.name,
                        'min_samples_leaf': corner.min_samples_leaf,
                        'min_samples_split': corner.min_samples_split,
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

                    print('  t={:<3} d={:<3} {:<11} joint cw={:<5}{} blocks={:<6} | '
                          'disjoint cw={:<5}{} blocks={:<6} | {:.1f}s'.format(
                              n_trees, max_depth, corner.name, joint_len,
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
    """Collapse the 3 splits into the worst case per (cell, corner) -- the
    adoption rule asks for cells within the limit on ALL splits, so the maximum
    over splits is the deciding statistic."""
    frame = frame.copy()
    for column in ('joint_stages', 'joint_blocks',
                   'disjoint_stages', 'disjoint_blocks'):
        # Int64's pd.NA does not compare, so groupby.max cannot order it; NaN
        # is skipped instead.
        frame[column] = pd.to_numeric(frame[column], errors='coerce')

    cells = frame.groupby(['n_trees', 'max_depth', 'corner'], as_index=False).agg(
        cardinality=('cardinality', 'max'),
        # Carried through rather than read off the `Corner` constant: these
        # are what the ROWS were actually fit with, which is not
        # necessarily what the constant says today if it changed after this
        # CSV was written. `min_samples_leaf`/`min_samples_split` are
        # constant within one (corner, this CSV) by construction (`fit`
        # always reads them off the same `Corner` namedtuple for every row
        # of a `collect()` run), so 'first' just recovers that constant --
        # it is not really an aggregation.
        min_samples_leaf=('min_samples_leaf', 'first'),
        min_samples_split=('min_samples_split', 'first'),
        joint_cw_max=('joint_codeword_length', 'max'),
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


def at_corner(cells, corner, n_trees=None, max_depth=None):
    rows = cells[cells.corner == corner.name]
    if n_trees is not None:
        rows = rows[rows.n_trees == n_trees]
    if max_depth is not None:
        rows = rows[rows.max_depth == max_depth]
    return rows


def corner_params(cells, corner):
    """The min_samples_leaf/min_samples_split a corner's rows in `cells`
    were ACTUALLY fit with, read from the data -- never from the `Corner`
    constant. If the constant changes after a CSV was measured, a printed
    header built from the constant would disagree with the table beneath
    it; reading the value off the rows being rendered makes that
    impossible; whatever produced the file is what gets printed.

    Raises if a CSV somehow mixes two values for one corner name, since
    that would mean the file blends two measurements under one label and no
    single header could honestly describe it.
    """
    rows = cells[cells.corner == corner.name]
    if not len(rows):
        raise ValueError(
            'corner_params: no rows for corner {!r} in this data'.format(
                corner.name))
    leaf = rows['min_samples_leaf'].unique()
    split = rows['min_samples_split'].unique()
    if len(leaf) > 1 or len(split) > 1:
        raise ValueError(
            'corner_params: corner {!r} has inconsistent min_samples_leaf '
            '({}) or min_samples_split ({}) across its own rows -- this '
            'file mixes measurements taken under different constants and '
            'cannot be rendered under one header.'.format(
                corner.name, sorted(leaf), sorted(split)))
    return int(leaf[0]), int(split[0])


def print_grid(cells, corner, column, title, note):
    leaf, split = corner_params(cells, corner)
    print('\n### {} ({} corner: min_samples_leaf={}, min_samples_split={})\n'
          .format(title, corner.name, leaf, split))
    print('{}\n'.format(note))
    header = ['n_trees \\ max_depth'] + [str(d) for d in MAX_DEPTH_GRID]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for n_trees in N_TREES_GRID:
        line = ['**{}**'.format(n_trees)]
        for max_depth in MAX_DEPTH_GRID:
            value = at_corner(cells, corner, n_trees, max_depth)[column].iloc[0]
            if pd.isna(value):
                line.append('-')
                continue
            value = int(value)
            over = column.endswith('cw_max') and value > MAX_CODEWORD_LENGTH
            line.append('**{}**'.format(value) if over else str(value))
        print('| ' + ' | '.join(line) + ' |')


def print_where_the_limit_binds(cells):
    print('\n### Where the {}-bit limit starts to bind, under both corners\n'
          .format(MAX_CODEWORD_LENGTH))
    print('The smallest max_depth in the grid whose codeword exceeds the limit '
          'on at least one split. The two staircases differ by a factor of ~3 '
          'in depth, which is the point: the ceiling is a property of '
          '(n_trees, max_depth, pruning), not of the two bounds alone. Ruling '
          'P4-2 makes the {} corner the deciding one.\n'
          .format(DECIDING_CORNER.name))
    header = ['n_trees'] + ['{} / {}'.format(corner.name, encoding)
                            for corner in CORNERS
                            for encoding in ('joint', 'disjoint')]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for n_trees in N_TREES_GRID:
        line = [str(n_trees)]
        for corner in CORNERS:
            rows = at_corner(cells, corner, n_trees)
            for column in ('joint_within_limit_all_splits',
                           'disjoint_within_limit_all_splits'):
                over = rows[~rows[column]].max_depth
                line.append(str(int(over.min())) if len(over) else 'never')
        print('| ' + ' | '.join(line) + ' |')


def print_cell_table(cells):
    print('\n### Per-cell summary, ranked by admissible search-space '
          'cardinality (Rulings P4-1 and P4-3)\n')
    print('`cardinality = ceil(n_trees / 2) * (max_depth - 1)` is the number of '
          '`(n_estimators, max_depth)` pairs the Optuna bounds actually reach '
          '(n_estimators has step=2). "feasible" is the {} corner\'s joint '
          'codeword staying within {} bits on all {} splits -- the deciding '
          'criterion (Ruling P4-2); the large-tree columns are shown alongside '
          'so the cost of the stricter reading is visible.\n'.format(
              DECIDING_CORNER.name, MAX_CODEWORD_LENGTH, len(SPLIT_INDICES)))
    header = ['n_trees', 'max_depth', 'cardinality',
              'pruned joint cw', 'pruned joint blocks',
              'pruned disjoint cw', 'pruned disjoint blocks',
              'FEASIBLE (pruned)',
              'large-tree joint cw', 'large-tree joint blocks',
              'feasible (large-tree)']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')

    def number(value):
        return '-' if pd.isna(value) else str(int(value))

    deciding = at_corner(cells, DECIDING_CORNER).sort_values(
        ['joint_within_limit_all_splits', 'cardinality', 'n_trees'],
        ascending=[False, False, True])
    for _, row in deciding.iterrows():
        other = at_corner(cells, LARGE_TREE, row.n_trees, row.max_depth).iloc[0]
        print('| ' + ' | '.join([
            str(int(row.n_trees)), str(int(row.max_depth)),
            str(int(row.cardinality)),
            number(row.joint_cw_max), number(row.joint_blocks_max),
            number(row.disjoint_cw_max), number(row.disjoint_blocks_max),
            'yes' if row.joint_within_limit_all_splits else 'NO',
            number(other.joint_cw_max), number(other.joint_blocks_max),
            'yes' if other.joint_within_limit_all_splits else 'NO',
        ]) + ' |')


def select(cells):
    """Rulings P4-1/P4-2/P4-3: maximum admissible cardinality among cells whose
    joint codeword is within the limit at the deciding corner on all splits,
    ties broken toward the smaller n_trees."""
    deciding = at_corner(cells, DECIDING_CORNER)
    feasible = deciding[deciding.joint_within_limit_all_splits]
    if not len(feasible):
        raise RuntimeError('no grid cell stays within the codeword limit')

    best = feasible.cardinality.max()
    tied = feasible[feasible.cardinality == best].sort_values('n_trees')
    chosen = tied.iloc[0]
    strict = at_corner(cells, LARGE_TREE, chosen.n_trees, chosen.max_depth).iloc[0]

    deciding_leaf, deciding_split = corner_params(cells, DECIDING_CORNER)
    print('\n### Adopted values\n')
    print('The {} corner decides (Ruling P4-2): a cell counts as feasible when '
          'ANY configuration the search can reach there compiles, and pruning '
          'is inside the search space -- min_samples_leaf up to {}, '
          'min_samples_split up to {} (the values the rows below were '
          'actually fit with, read from the measurement rather than from '
          'the `Corner` constant in code, so this stays correct even if the '
          'constant changes after a CSV is measured). Requiring the whole '
          'box to compile (the {} corner) would truncate the reachable '
          'frontier without buying a real guarantee, since the block budget '
          'binds inside the box regardless.'.format(
              DECIDING_CORNER.name, deciding_leaf, deciding_split,
              LARGE_TREE.name))
    print('\n{} of {} grid cells keep the joint codeword within {} bits on all '
          '{} splits at that corner. The largest admissible search space among '
          'them has cardinality ceil(n_trees / 2) * (max_depth - 1) = {}, '
          'attained by {}.'.format(
              len(feasible), len(deciding), MAX_CODEWORD_LENGTH,
              len(SPLIT_INDICES), int(best),
              ', '.join('({}, {})'.format(int(r.n_trees), int(r.max_depth))
                        for _, r in tied.iterrows())))
    if len(tied) > 1:
        print('Ties are broken toward the smaller n_trees (Ruling P4-1).')
    print('\n**Adopted: n_trees = {}, max_depth = {}** -- justified by the row '
          '(n_trees={}, max_depth={}) of the per-cell table: cardinality {}, '
          'joint codeword length {} bits at its worst split under pruning '
          '(limit {}), {} joint blocks. The same cell measures {} bits at the '
          'large-tree corner, so part of the box is out of reach on codeword '
          'length. That is normal and was already true of the (7, 10) '
          'placeholders, whose own corner measures 1599 bits: the bounds are '
          'per-axis, so a joint corner can be suggestable without ever being '
          'selectable, and the objective scores such trials by violation '
          'magnitude rather than letting them stop the search.'.format(
              int(chosen.n_trees), int(chosen.max_depth),
              int(chosen.n_trees), int(chosen.max_depth),
              int(chosen.cardinality), int(chosen.joint_cw_max),
              MAX_CODEWORD_LENGTH,
              '-' if pd.isna(chosen.joint_blocks_max) else int(chosen.joint_blocks_max),
              int(strict.joint_cw_max)))
    print('\nRuling P4-4: at the adopted n_trees, the pruned-corner codeword '
          'and block counts can look nearly identical across the top of the '
          'max_depth grid (e.g. the per-cell table above), because under '
          'heavy pruning the forests stop growing well before the depth '
          'bound is reached -- raising max_depth buys almost nothing AT THAT '
          'CORNER. The bound is kept at the top of the grid anyway: '
          'saturation is a property of the pruned corner only, not of the '
          'bound. At low min_samples_leaf the search explores every depth in '
          'the box as a genuinely different model, so truncating max_depth '
          'to where the pruned corner saturates would cut off models the '
          'large-tree end of the search can still reach.')
    return int(chosen.n_trees), int(chosen.max_depth)


def report(cells):
    """Print every markdown table this script reports, and return the
    adopted (n_trees, max_depth).

    Split out of `main()` so the reporting half can be replayed from
    `results/capacity_ceiling.csv` alone -- `src/reporting/figures.py`
    captures this output to persist appendix 6 (spec C.5 deliverable 6),
    which was previously printed and then lost. Nothing here re-measures
    anything: `collect()` is the measurement and is called only by `main()`.
    """
    splits = len(SPLIT_INDICES)
    for corner in CORNERS:
        print_grid(cells, corner, 'joint_cw_max',
                   'Joint-encoding codeword length, worst of {} splits'.format(splits),
                   'Bold exceeds the {}-bit limit, so the pair does not '
                   'compile.'.format(MAX_CODEWORD_LENGTH))
        print_grid(cells, corner, 'disjoint_cw_max',
                   'Disjoint-encoding codeword length (longer of the two '
                   'models), worst of {} splits'.format(splits),
                   'Bold exceeds the {}-bit limit.'.format(MAX_CODEWORD_LENGTH))
        print_grid(cells, corner, 'joint_blocks_max',
                   'Joint-encoding TCAM blocks, worst of {} splits'.format(splits),
                   '"-" is a cell whose codeword exceeds the limit, so no block '
                   'count exists. Compare against the campaign M grid of '
                   '25-100: blocks bind well before codeword bits do at high '
                   'tree counts.')
        print_grid(cells, corner, 'disjoint_blocks_max',
                   'Disjoint-encoding TCAM blocks, worst of {} splits'.format(splits),
                   '"-" is a cell whose codeword exceeds the limit, so no block '
                   'count exists.')
    print_where_the_limit_binds(cells)
    print_cell_table(cells)
    return select(cells)


def main():
    frame = collect()
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', 'capacity_ceiling.csv')
    frame.to_csv(path, index=False)
    print('\nwrote {} ({} rows)'.format(path, len(frame)))
    report(per_cell(frame))


if __name__ == '__main__':
    main()
