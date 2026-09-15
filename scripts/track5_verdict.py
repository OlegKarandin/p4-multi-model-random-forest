"""Score Track 5 (live-Optuna delta_align sweep) against the pre-registered
verdict rule, so applying the rule to real numbers is mechanical and cannot be
renegotiated after seeing them.

Rule, quoted from the 2026-09-14 design 8.2, amended ONLY by Amendment A1
(this wording -- majority threshold and cell count -- was written down BEFORE
any Track 5 datum existed):

    `delta` "helps" iff BOTH hold at `M = 25`:

    1. mean feasible fraction is strictly higher at `joint-d020` than at
       `joint-d000`; and
    2. that sign is consistent in a strict majority of the 24 tight cells
       (>= 13 of 24) -- 6 k-values x 4 splits, per Amendment A1. (At the
       3-split fallback the threshold reverts to >= 10 of 18.)

Condition 2 exists because a mean alone can be carried by one or two cells,
which is the shape that produced the overstatement 8.4 had to walk back.
`joint-dinf` is an anchor, not part of the test -- 8.4 already measured its
accuracy tail (accuracy_spent max 5.77) and it is not a candidate default
under any outcome.

Run (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/track5_verdict.py
"""
import argparse
import glob
import os
import re
import sys

# Running a file inside scripts/ puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

DEFAULT_RESULTS_GLOB = 'results/rf_t7_d14_M*_joint-*.csv'
DEFAULT_OUT = 'results/track5_verdict.csv'
_D000 = 'joint-d000'
_D020 = 'joint-d020'

# The pre-registered grid (design 8.1): "M in {25, 100} x k in {1, 2, 5, 9,
# 13, 17} x splits" -- these 6 are the "tight cells" the verdict rule counts.
# compare_feature_selection_approaches_parallel's k-ladder actually sweeps ALL
# 17 k-values (dropping one feature at a time), so the archived result CSVs
# carry 17 k-values per split; only these 6 are the pre-registered sample.
TIGHT_K_VALUES = (1, 2, 5, 9, 13, 17)

# Matches the arm slug out of a filename like
# 'results/rf_t7_d14_M25_joint-d000.csv' -> 'joint-d000'.
_ARM_SLUG_RE = re.compile(r'_M\d+_(.+)\.csv$')


def majority_threshold(n_cells):
    """A strict majority of n_cells: 13 of 24, 10 of 18."""
    return n_cells // 2 + 1


def feasible_fraction(frame):
    """Return a copy of `frame` with a `feasible_fraction` column
    (n_feasible / n_trials_run). Raises if any n_trials_run == 0 -- a zero
    denominator is a broken cell (the search never ran), not a zero
    fraction."""
    if (frame['n_trials_run'] == 0).any():
        raise ValueError(
            'n_trials_run == 0 for at least one row -- a zero denominator is '
            'a broken cell, not a zero feasible fraction')
    out = frame.copy()
    out['feasible_fraction'] = out['n_feasible'] / out['n_trials_run']
    return out


def _pivot_by_cell(frame):
    """One row per (k, split) cell, with a column per arm_slug holding its
    mean feasible fraction. Columns for `_D000`/`_D020` are always present
    (NaN-filled when that arm has no rows at all for a cell), so callers can
    index them unconditionally."""
    pivot = frame.pivot_table(index=['k', 'split'], columns='arm_slug',
                               values='feasible_fraction', aggfunc='mean')
    for col in (_D000, _D020):
        if col not in pivot.columns:
            pivot[col] = float('nan')
    return pivot.reset_index()


def verdict(frame):
    """Score `frame` (rows with at least arm_slug, M, split, k, n_trials_run,
    n_feasible) against the pre-registered rule. Returns a dict with keys
    mean_d000, mean_d020, condition_1, cells_total, cells_favouring_d020,
    threshold, condition_2, delta_helps."""
    sub = frame[(frame['M'] == 25) & (frame['arm_slug'].isin((_D000, _D020)))
                & (frame['k'].isin(TIGHT_K_VALUES))]
    sub = feasible_fraction(sub)

    mean_d000 = sub.loc[sub['arm_slug'] == _D000, 'feasible_fraction'].mean()
    mean_d020 = sub.loc[sub['arm_slug'] == _D020, 'feasible_fraction'].mean()
    condition_1 = bool(mean_d020 > mean_d000)

    pivot = _pivot_by_cell(sub)
    both = pivot.dropna(subset=[_D000, _D020])
    cells_total = int(len(both))
    if cells_total not in (24, 18):
        raise ValueError(
            'cells_total={} -- expected 24 (6 k x 4 splits) or 18 (6 k x 3 '
            'splits, the OOM fallback); the rule must not be silently applied '
            'to a different grid than it was pre-registered against'.format(
                cells_total))

    cells_favouring_d020 = int((both[_D020] > both[_D000]).sum())
    threshold = majority_threshold(cells_total)
    condition_2 = bool(cells_favouring_d020 >= threshold)
    delta_helps = bool(condition_1 and condition_2)

    return {
        'mean_d000': mean_d000,
        'mean_d020': mean_d020,
        'condition_1': condition_1,
        'cells_total': cells_total,
        'cells_favouring_d020': cells_favouring_d020,
        'threshold': threshold,
        'condition_2': condition_2,
        'delta_helps': delta_helps,
    }


def arm_slug_from_filename(path):
    match = _ARM_SLUG_RE.search(os.path.basename(path))
    if not match:
        raise ValueError('cannot derive arm_slug from filename {!r}'.format(path))
    return match.group(1)


def load_results(pattern):
    """Load and concatenate every file matching `pattern`, adding an
    `arm_slug` column derived from each file's own name (the real CSVs' own
    `arm` column is always 'joint' -- it does not distinguish delta_align
    variants)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError('no files matched --results-glob {!r}'.format(pattern))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame['arm_slug'] = arm_slug_from_filename(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _favours(row):
    d000, d020 = row[_D000], row[_D020]
    if pd.isna(d000) or pd.isna(d020):
        return 'missing'
    if d020 > d000:
        return _D020
    if d000 > d020:
        return _D000
    return 'tie'


def build_cell_table(frame):
    """One row per (k, split) cell at M=25, with both arms' feasible
    fractions and which arm the sign favours -- written to `results/
    track5_verdict.csv` for a human to inspect the raw cells behind the
    verdict."""
    sub = frame[(frame['M'] == 25) & (frame['arm_slug'].isin((_D000, _D020)))
                & (frame['k'].isin(TIGHT_K_VALUES))]
    sub = feasible_fraction(sub)
    pivot = _pivot_by_cell(sub)
    pivot['favours'] = pivot.apply(_favours, axis=1)
    return pivot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-glob', default=DEFAULT_RESULTS_GLOB)
    parser.add_argument('--out', default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame = load_results(args.results_glob)

    result = verdict(frame)
    cell_table = build_cell_table(frame)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cell_table.to_csv(args.out, index=False)

    print(result)
    print('DELTA HELPS' if result['delta_helps'] else 'DELTA DOES NOT HELP')
    return result


if __name__ == '__main__':
    main()
