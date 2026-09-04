"""Compiler calibration study: measures the gap between the analytic
resource model's predicted `stage_depth` and the real Tofino compiler's
whole-program stage count (`stages_real`), on a stratified sample of
archived, deterministically-refit campaign models (spec
docs/superpowers/specs/2026-09-03-compiler-calibration-design.md).

WHY THIS EXISTS. Every stage/block/feasibility number this project has
produced comes from `multi_model_memory_evaluation`'s analytic model. The
one calibration point on record (M2) has the model predicting stage_depth=6
where the real compiler reports 9 -- and `train_model.objective`'s hard
reject at TOFINO_PIPELINE_STAGES (12) states its own check is "necessary
but not sufficient". This script runs the real WSL2 p4c compiler (already
built for `feature_selection._kickoff_hardware_validation`, just never
turned on for a controlled, stratified sample) over 20-24 archived winners
and reports whether the gap is constant, and whether it differs between
'independent' (P4-gen encoding 'disjoint') and 'joint' models -- the
consequential finding, since every published joint-vs-independent depth
comparison assumes a shared bias that cancels.

SCOPE. Replays the archive (results/campaign_backup_20260825) via
scripts.replay_alignment.refit_pair -- no new Optuna search, no production
code touched. See the plan's Global Constraints for the verified fact that
4 of the 24 target (group, k_band, stage_depth) cells are empty in this
archive; build_sample reports them by name rather than raising when told to.

TERMINOLOGY. This study's own factor is called `group` ('independent' /
'joint') throughout, to avoid colliding with `generate_P4_code` /
`multi_model_memory_evaluation`'s own `encoding` parameter, which takes the
DIFFERENT values 'disjoint' / 'joint'. group='independent' maps to
p4_encoding='disjoint'; group='joint' maps to p4_encoding='joint'. See
`run_one_row` for where that mapping happens.

Run the 3-row gate (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/compiler_calibration.py --groups independent --k-bands low \\
      --strata 5,8,12 --limit 3

The full remaining batch, after the gate passes (resumable: already-recorded
row_ids in --out are skipped):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/compiler_calibration.py
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.replay_alignment import load_backup, refit_pair
from src.main import load_campaign_data
from src.p4gen.build_p4_script import (
    generate_P4_code,
    get_feature_intervals,
    get_joint_feature_intervals,
)
from src.p4gen.evaluation import multi_model_memory_evaluation
from src.p4gen.p4_compile import compile_p4

GROUPS = ('independent', 'joint')
K_BANDS = ('low', 'high')
STRATA_STAGE_DEPTHS = (5, 6, 7, 8, 10, 12)
JOINT_ARM_SLUG = 'joint-d000'
K_LOW_MAX = 5
K_HIGH_MIN = 13
CAMPAIGN_BACKUP_DIR = 'results/campaign_backup_20260825'
DEFAULT_OUT = 'results/compiler_calibration.csv'
DEFAULT_OUTPUT_ROOT = 'results/compiler_calibration'


def k_band(k):
    """'low' at k<=K_LOW_MAX, 'high' at k>=K_HIGH_MIN, None in between (the
    mid-range is deliberately excluded from this study's k factor, spec
    §2.2)."""
    if k <= K_LOW_MAX:
        return 'low'
    if k >= K_HIGH_MIN:
        return 'high'
    return None


def add_group_and_band(frame):
    """Adds 'group' ('independent' / 'joint' / None) and 'k_band' ('low' /
    'high' / None) columns, derived from 'arm_slug' and 'k'. Rows outside
    both this study's group set and its k bands get None in the
    corresponding column rather than being dropped here -- callers filter
    explicitly (stratum_row/build_sample)."""
    frame = frame.copy()
    if len(frame) > 0:
        frame['group'] = frame['arm_slug'].map(
            lambda s: 'independent' if s == 'independent'
            else ('joint' if s == JOINT_ARM_SLUG else None))
        frame['k_band'] = frame['k'].map(k_band)
    else:
        frame['group'] = pd.Series([], dtype=object)
        frame['k_band'] = pd.Series([], dtype=object)
    return frame


def stratum_row(frame, group, band, stage_depth):
    """The archived row for one (group, k_band, stage_depth) cell --
    deterministic (lowest split first, then M, then k), matching
    replay_alignment.select_rows' own tie-break convention. Raises
    ValueError naming the empty cell rather than returning nothing, so a
    caller building a fixed-size sample finds out immediately which
    stratum is missing (spec §4's pure-function contract)."""
    subset = frame[(frame['group'] == group) & (frame['k_band'] == band)
                   & (frame['stage_depth'] == stage_depth)]
    if not len(subset):
        raise ValueError(
            'no archived row for group={!r} k_band={!r} stage_depth={} -- '
            'this stratum is empty in the source archive'.format(
                group, band, stage_depth))
    return subset.sort_values(['split', 'M', 'k']).iloc[0]


def build_sample(frame, strata=STRATA_STAGE_DEPTHS, groups=GROUPS,
                  k_bands=K_BANDS, on_missing='raise'):
    """One row per (group, k_band, stage_depth) cell in the cross product
    of groups x k_bands x strata.

    on_missing='raise' (default): the first empty cell raises ValueError,
    naming it. on_missing='skip': empty cells are recorded in the returned
    `missing` list instead, so a caller that already knows some cells are
    structurally absent from the archive (see this plan's Global
    Constraints -- 4 of the 24 target cells are, verified 2026-09-04) can
    still run the cells that DO exist.

    Returns (rows, missing): rows is a list of
    (row_id, group, k_band, stage_depth, archived_row) tuples, row_id a
    string unique per cell ('{group}_{k_band}_sd{stage_depth}'); missing is
    a list of (group, k_band, stage_depth) tuples, empty unless
    on_missing='skip' and a cell really is absent.
    """
    if on_missing not in ('raise', 'skip'):
        raise ValueError("on_missing must be 'raise' or 'skip', got {!r}".format(on_missing))
    rows, missing = [], []
    for group in groups:
        for band in k_bands:
            for stage_depth in strata:
                try:
                    row = stratum_row(frame, group, band, stage_depth)
                except ValueError:
                    if on_missing == 'raise':
                        raise
                    missing.append((group, band, stage_depth))
                    continue
                row_id = '{}_{}_sd{}'.format(group, band, stage_depth)
                rows.append((row_id, group, band, stage_depth, row))
    return rows, missing


def _is_missing(value):
    """True for None and for a float NaN (pandas' representation of a
    missing cell after a CSV round-trip); False for any real value,
    0 included."""
    return value is None or (isinstance(value, float) and pd.isna(value))


def gap_stages(row):
    """stages_real - stage_depth (V1/V2's subject), or None when there is
    no real compile result to compare against (void row)."""
    if _is_missing(row['stages_real']):
        return None
    return row['stages_real'] - row['stage_depth']


def gap_blocks(row):
    """tcam_real - blocks (V5, confirmatory), or None when there is no
    real compile result to compare against (void row)."""
    if _is_missing(row['tcam_real']):
        return None
    return row['tcam_real'] - row['blocks']


def is_void(row):
    """True when this row has no usable real-compiler comparison: either
    compile_errors is a positive int (the real p4c reported errors) or a
    non-numeric string (the F2 'unavailable' degrade-path reason -- see
    run_one_row). A void row is excluded from every V1-V5 aggregate and
    reported separately by name (V6).

    compile_errors may arrive as a genuine int (fresh from run_one_row) OR
    as a string after a CSV round-trip: once any row in the accumulated
    output has a non-numeric reason string, pandas reads the WHOLE
    compile_errors column back as str, so a numeric string like '0' must
    still be recognised as a real, non-void error count."""
    errors = row.get('compile_errors')
    if _is_missing(errors):
        return False
    if isinstance(errors, str):
        try:
            errors = int(errors)
        except ValueError:
            return True
    return int(errors) > 0


def run_one_row(row_id, group, archived_row, data, output_root):
    """Compiles ONE stratified sample row: refit -> generate P4 -> real
    p4c compile -> parse -> one result dict. Raises on a genuine toolchain
    failure (compile_p4's RuntimeError, e.g. a timeout or a crash with no
    parseable error/warning line) -- the caller (collect()) is responsible
    for catching and recording that as a void row, mirroring
    feasibility_frontier.collect's per-point exception isolation.

    group is 'independent' (P4-gen encoding 'disjoint') or 'joint'
    (P4-gen encoding 'joint') -- see the module docstring for why the two
    vocabularies are kept distinct.

    The 'predicted' side of every comparison (stage_depth, blocks, stages,
    ...) is computed via multi_model_memory_evaluation on THIS row's own
    refit pair -- not read from the archive's recorded columns -- so it is
    guaranteed to describe the exact program that gets compiled.
    """
    p4_encoding = 'joint' if group == 'joint' else 'disjoint'
    model_app, model_ddos, app, ddos, cols_app, cols_ddos = refit_pair(archived_row, data)
    names_app = archived_row['features_app'].split(';')
    names_ddos = archived_row['features_ddos'].split(';')

    usage = multi_model_memory_evaluation(
        model_app, model_ddos, names_app, names_ddos, p4_encoding)

    params = json.loads(archived_row['best_params'])
    row = {
        'row_id': row_id, 'group': group,
        'source_arm_slug': archived_row['arm_slug'],
        'M': int(archived_row['M']), 'k': int(archived_row['k']),
        'split': int(archived_row['split']),
        'stage_depth': usage.stage_depth, 'blocks': usage.blocks,
        'stages': usage.stages, 'range_entries': usage.range_entries,
        'ternary_entries': usage.ternary_entries,
        'register_depth': usage.register_depth,
        'register_count': usage.register_count,
        'n_estimators_A': params.get('n_estimators_A'),
        'n_estimators_B': params.get('n_estimators_B'),
    }

    if group == 'joint':
        feature_intervals = get_joint_feature_intervals(
            model_app, names_app, model_ddos, names_ddos)
        feature_intervals_app = feature_intervals_ddos = feature_intervals
    else:
        feature_intervals_app = get_feature_intervals(model_app, names_app)
        feature_intervals_ddos = get_feature_intervals(model_ddos, names_ddos)

    p4_dir = os.path.join(output_root, 'p4_src') + os.sep
    os.makedirs(p4_dir, exist_ok=True)
    filename = row_id + '.p4'
    try:
        written_path = generate_P4_code(
            3, 2, model_app, model_ddos,
            feature_intervals_app=feature_intervals_app,
            feature_intervals_ddos=feature_intervals_ddos,
            output_dir=p4_dir, output_filename=filename,
            selected_features_app=names_app, selected_features_ddos=names_ddos)
    except ValueError as e:
        # F2 degrade path: a selected feature has no FEATURE_REGISTER_CATALOG
        # entry -- no compile to run. Mirrors
        # feature_selection._kickoff_hardware_validation's own
        # ('unavailable', reason) handling.
        row.update({'stages_real': None, 'tcam_real': None,
                    'sram_real': None, 'map_ram_real': None,
                    'compile_errors': str(e)})
        return row

    compile_dir = os.path.join(output_root, 'compiles', row_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an already-existing dir (see
        # compile_p4's docstring) -- a stale dir here means a prior
        # attempt at this row started but never finished.
        shutil.rmtree(compile_dir)
    result = compile_p4(written_path, compile_dir)

    row.update({'stages_real': result.stages, 'tcam_real': result.tcam,
                'sram_real': result.sram, 'map_ram_real': result.map_ram,
                'compile_errors': result.errors})
    return row


def already_done(out_path):
    """row_ids already recorded at out_path, or an empty set if it doesn't
    exist yet -- lets collect() resume a partial run."""
    if not os.path.exists(out_path):
        return set()
    return set(pd.read_csv(out_path)['row_id'])


def collect(campaign_dir, out, output_root, strata=STRATA_STAGE_DEPTHS,
            groups=GROUPS, k_bands=K_BANDS, limit=None):
    """Compiles remaining sample rows SEQUENTIALLY, writing each to `out`
    as it completes.

    Sequential, not a ProcessPoolExecutor pool like feasibility_frontier's:
    compiles are the scarce, long-running resource here and N is at most
    24, so the parallelism that mattered for a 576-point Optuna grid is
    overkill -- a plain try/except per row gets the same "one bad row
    doesn't lose the whole run" property with far less machinery.

    Re-invoking with the same --out resumes: rows already recorded (by
    row_id) are skipped. `limit` caps how many NOT-YET-DONE rows this
    invocation compiles -- the mechanism the plan's Task 4 gate uses
    (--limit 3) before committing to the rest.

    Returns (frame, missing): frame is read back from `out` after writing
    (the full accumulated file across every resume); missing is
    build_sample's list of (group, k_band, stage_depth) cells with no
    archived row at all (reported by report(), not silently dropped).
    """
    backup = load_backup(campaign_dir)
    if 'infeasible' in backup.columns:
        # Mirrors replay_alignment.select_rows' own guard: real campaign
        # CSVs carry this column, but a synthetic/minimal frame (tests)
        # may not -- absence means nothing to filter, not "everything
        # infeasible".
        backup = backup[backup['infeasible'].isna() | (backup['infeasible'] == '')]
    frame = add_group_and_band(backup)
    rows, missing = build_sample(frame, strata=strata, groups=groups,
                                 k_bands=k_bands, on_missing='skip')
    if missing:
        print('missing strata (no archived row for these cells) -- '
              'proceeding without them:')
        for group, band, stage_depth in missing:
            print('  group={} k_band={} stage_depth={}'.format(group, band, stage_depth))

    done = already_done(out)
    remaining = [r for r in rows if r[0] not in done]
    if limit is not None:
        remaining = remaining[:limit]

    if not remaining:
        print('nothing to do -- every sampled row already recorded at {}'.format(out))
        frame_out = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
        return frame_out, missing

    data = load_campaign_data()
    os.makedirs(output_root, exist_ok=True)
    file_exists = os.path.exists(out) and os.path.getsize(out) > 0
    n_failed = 0
    for row_id, group, band, stage_depth, archived_row in remaining:
        print('compiling {} ...'.format(row_id))
        started = time.time()
        try:
            result_row = run_one_row(row_id, group, archived_row, data, output_root)
        except Exception:
            n_failed += 1
            print('  {} raised an exception:\n{}'.format(row_id, traceback.format_exc()))
            continue
        pd.DataFrame([result_row]).to_csv(out, mode='a', header=not file_exists, index=False)
        file_exists = True
        print('  [{}] done in {:.1f}s -- stage_depth={} stages_real={}'.format(
            row_id, time.time() - started, result_row['stage_depth'],
            result_row.get('stages_real')))

    if n_failed:
        print('{} row(s) raised a toolchain exception and were not recorded '
              '-- they will be retried on the next invocation'.format(n_failed))

    frame_out = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    return frame_out, missing


def report(frame, missing):
    """V1-V6 against the accumulated frame (spec §5). Split from main() so
    a caller could replay this report from results/compiler_calibration.csv
    alone, mirroring scripts/feasibility_frontier.py's report()."""
    if frame.empty:
        print('\nno rows to report -- nothing has been compiled yet.')
        return

    void_mask = frame.apply(is_void, axis=1)
    live = frame[~void_mask].copy()
    void = frame[void_mask]
    live['gap_stages'] = live.apply(gap_stages, axis=1)
    live['gap_blocks'] = live.apply(gap_blocks, axis=1)

    print('\n### V1 -- gap_stages >= 0 on every row (headline)\n')
    negative = live[live['gap_stages'] < 0]
    if len(negative):
        print('FALSIFIED -- {} row(s) with a negative gap_stages:'.format(len(negative)))
        print(negative[['row_id', 'stage_depth', 'stages_real', 'gap_stages']].to_string(index=False))
    else:
        print('OK -- gap_stages >= 0 on all {} compiled, non-void rows'.format(len(live)))

    print('\n### V2 -- gap_stages distribution\n')
    if len(live):
        g = live['gap_stages']
        constant = (g.max() - g.min()) <= 1
        print('mean={:.2f} min={} max={} n={} -- {}'.format(
            g.mean(), g.min(), g.max(), len(g),
            'CONSTANT (max-min<=1)' if constant else 'SCALES (max-min>1)'))

    print('\n### V3 -- gap_stages by group (independent vs joint)\n')
    if len(live):
        by_group = live.groupby('group')['gap_stages'].agg(['mean', 'count'])
        print(by_group.to_string())
        if {'independent', 'joint'}.issubset(set(by_group.index)):
            diff = by_group.loc['joint', 'mean'] - by_group.loc['independent', 'mean']
            print('joint - independent mean gap_stages = {:.2f}'.format(diff))

    print('\n### V4 -- gap_stages correlation with k and T\n')
    if len(live) > 2:
        t = live[['n_estimators_A', 'n_estimators_B']].mean(axis=1)
        print('corr(gap_stages, k) = {:.3f}  (n={})'.format(
            live['gap_stages'].corr(live['k']), len(live)))
        print('corr(gap_stages, T) = {:.3f}  (n={})'.format(
            live['gap_stages'].corr(t), len(live)))

    print('\n### V5 -- gap_blocks == 0 on every row (confirmatory)\n')
    if len(live):
        nonzero = live[live['gap_blocks'] != 0]
        if len(nonzero):
            print('{} row(s) with gap_blocks != 0 (reopens a question '
                  'treated as closed):'.format(len(nonzero)))
            print(nonzero[['row_id', 'blocks', 'tcam_real', 'gap_blocks']].to_string(index=False))
        else:
            print('OK -- gap_blocks == 0 on all {} rows'.format(len(live)))

    print('\n### V6 -- void rows and missing strata (diagnostic)\n')
    if len(void):
        print(void[['row_id', 'compile_errors']].to_string(index=False))
    else:
        print('no void rows')
    if missing:
        for group, band, stage_depth in missing:
            print('  missing stratum: group={} k_band={} stage_depth={}'.format(
                group, band, stage_depth))
    else:
        print('no missing strata')


def _parse_int_tuple(value):
    return tuple(int(v) for v in value.split(','))


def _parse_str_tuple(value):
    return tuple(value.split(','))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', default=CAMPAIGN_BACKUP_DIR)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--strata', type=_parse_int_tuple,
                        default=STRATA_STAGE_DEPTHS)
    parser.add_argument('--groups', type=_parse_str_tuple, default=GROUPS)
    parser.add_argument('--k-bands', dest='k_bands', type=_parse_str_tuple,
                        default=K_BANDS)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame, missing = collect(args.campaign_dir, args.out, args.output_root,
                             strata=args.strata, groups=args.groups,
                             k_bands=args.k_bands, limit=args.limit)
    print('{} total rows on disk at {}'.format(len(frame), args.out))
    report(frame, missing)


if __name__ == '__main__':
    main()
