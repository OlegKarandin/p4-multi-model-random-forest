"""Compiler calibration study: measures the gap between the analytic
resource model's predicted `stage_depth` and the real Tofino compiler's
whole-program stage count (`stages_real`), on a stratified sample of
archived, deterministically-refit campaign models (spec
docs/superpowers/specs/2026-09-03-compiler-calibration-design.md).

WHY THIS EXISTS. Every stage/block/feasibility number this project has
produced comes from `multi_model_memory_evaluation`'s analytic model. The
one calibration point on record (M2) had the model predicting stage_depth=6
where the real compiler reports 9 -- and `train_model.objective`'s hard
reject at TOFINO_PIPELINE_STAGES (12) states its own check is "necessary
but not sufficient". This script runs the real WSL2 p4c compiler (already
built for `feature_selection._kickoff_hardware_validation`, just never
turned on for a controlled, stratified sample) over 20-24 archived winners

OUTCOME (2026-09-05). The study ran, and its results changed the model: the
19 committed placements showed the readiness-level origin was 2 stages early
and that the ternary crossbar charges the union of distinct key FIELDS
rather than the sum of per-table key widths. Both are fixed in
src/p4gen/evaluation.py, M2 now predicts 9 against the compiler's 9, and
`replay_stage_depth` below re-checks all 19 rows without recompiling.
stage_depth still under-predicts the real compiler by 0-3 stages -- PHV
container conflicts and TCAM column geometry remain unmodelled -- so the
TOFINO_PIPELINE_STAGES check is still necessary rather than sufficient.
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
import math
import os
import re
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
from src.p4gen.evaluation import (
    VOTE_EPILOGUE_STAGES,
    crossbar_stages_needed,
    gated_block_interior_stages,
    multi_model_memory_evaluation,
    readiness_levels_for,
)
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
    for catching this. It does NOT write such a row to the output CSV (a
    CSV-recorded row is treated as "done" and permanently skipped on
    resume, which would be wrong for a row that should be retried);
    instead it records the row_id and a one-line exception summary for
    report()'s V6, mirroring feasibility_frontier.collect's per-point
    exception isolation.

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
    stage_depth_archived = int(archived_row['stage_depth'])
    if stage_depth_archived != usage.stage_depth:
        print('WARNING: {} -- archived stage_depth={} (used to select this '
              'stratum) disagrees with the recomputed stage_depth={} on the '
              'refit pair; refit determinism may not hold for this '
              'row'.format(row_id, stage_depth_archived, usage.stage_depth))
    row = {
        'row_id': row_id, 'group': group,
        'source_arm_slug': archived_row['arm_slug'],
        'M': int(archived_row['M']), 'k': int(archived_row['k']),
        'split': int(archived_row['split']),
        'stage_depth': usage.stage_depth,
        'stage_depth_archived': stage_depth_archived,
        'blocks': usage.blocks,
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

    Returns (frame, missing, failed): frame is read back from `out` after
    writing (the full accumulated file across every resume); missing is
    build_sample's list of (group, k_band, stage_depth) cells with no
    archived row at all (reported by report(), not silently dropped);
    failed is a list of (row_id, summary) pairs for rows whose
    run_one_row raised a toolchain exception this invocation -- NOT
    written to `out` (so they remain retryable on the next invocation)
    but reported by report()'s V6 by name.
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
        return frame_out, missing, []

    data = load_campaign_data()
    os.makedirs(output_root, exist_ok=True)
    file_exists = os.path.exists(out) and os.path.getsize(out) > 0
    failed = []
    for row_id, group, band, stage_depth, archived_row in remaining:
        print('compiling {} ...'.format(row_id))
        started = time.time()
        try:
            result_row = run_one_row(row_id, group, archived_row, data, output_root)
        except Exception as e:
            summary = '{}: {}'.format(type(e).__name__, str(e))[:200]
            failed.append((row_id, summary))
            print('  {} raised an exception:\n{}'.format(row_id, traceback.format_exc()))
            continue
        pd.DataFrame([result_row]).to_csv(out, mode='a', header=not file_exists, index=False)
        file_exists = True
        print('  [{}] done in {:.1f}s -- stage_depth={} stages_real={}'.format(
            row_id, time.time() - started, result_row['stage_depth'],
            result_row.get('stages_real')))

    if failed:
        print('{} row(s) raised a toolchain exception and were not recorded '
              '-- they will be retried on the next invocation'.format(len(failed)))

    frame_out = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    return frame_out, missing, failed


def report(frame, missing, failed=()):
    """V1-V6 against the accumulated frame (spec §5). Split from main() so
    a caller could replay this report from results/compiler_calibration.csv
    alone, mirroring scripts/feasibility_frontier.py's report().

    void rows (is_void()==True) and unmeasured rows (is_void()==False but
    with no real stages_real/tcam_real -- e.g. a row whose compile_errors
    is missing/NaN) are BOTH excluded from every V1-V5 aggregate: only
    `measured` rows (gap_stages not null) feed V1-V5. void and unmeasured
    are reported separately, by name, under V6, alongside `failed`
    (toolchain-exception row_ids from collect() that never made it into
    `frame` at all)."""
    if frame.empty:
        print('\nno rows to report -- nothing has been compiled yet.')
        return

    void_mask = frame.apply(is_void, axis=1)
    live = frame[~void_mask].copy()
    void = frame[void_mask]
    live['gap_stages'] = live.apply(gap_stages, axis=1)
    live['gap_blocks'] = live.apply(gap_blocks, axis=1)

    measured_mask = live['gap_stages'].notna()
    measured = live[measured_mask].copy()
    unmeasured = live[~measured_mask]

    print('\n### V1 -- gap_stages >= 0 on every row (headline)\n')
    if len(measured):
        negative = measured[measured['gap_stages'] < 0]
        if len(negative):
            print('FALSIFIED -- {} row(s) with a negative gap_stages:'.format(len(negative)))
            print(negative[['row_id', 'stage_depth', 'stages_real', 'gap_stages']].to_string(index=False))
        else:
            print('OK -- gap_stages >= 0 on all {} compiled, non-void, measured rows'.format(len(measured)))
    else:
        print('NOT ESTABLISHED -- no compiled, non-void rows to check (0 measured)')

    print('\n### V2 -- gap_stages distribution\n')
    if len(measured):
        g = measured['gap_stages']
        constant = (g.max() - g.min()) <= 1
        print('mean={:.2f} min={} max={} n={} -- {}'.format(
            g.mean(), g.min(), g.max(), len(g),
            'CONSTANT (max-min<=1)' if constant else 'SCALES (max-min>1)'))

    print('\n### V3 -- gap_stages by group (independent vs joint)\n')
    if len(measured):
        by_group = measured.groupby('group')['gap_stages'].agg(['mean', 'count'])
        print(by_group.to_string())
        if {'independent', 'joint'}.issubset(set(by_group.index)):
            diff = by_group.loc['joint', 'mean'] - by_group.loc['independent', 'mean']
            print('joint - independent mean gap_stages = {:.2f}'.format(diff))

    print('\n### V4 -- gap_stages correlation with k and T\n')
    if len(measured) > 2:
        t = measured[['n_estimators_A', 'n_estimators_B']].mean(axis=1)
        print('corr(gap_stages, k) = {:.3f}  (n={})'.format(
            measured['gap_stages'].corr(measured['k']), len(measured)))
        print('corr(gap_stages, T) = {:.3f}  (n={})'.format(
            measured['gap_stages'].corr(t), len(measured)))

    print('\n### V5 -- gap_blocks == 0 on every row (confirmatory)\n')
    # A row can have a real stages_real (table_summary.log is written
    # regardless) but no tcam_real at all (no mau.resources.log "Allocated
    # Resource Usage" section -- the same degrade path is_void() does not
    # catch, since compile_errors is still 0): gap_blocks() already returns
    # None there, but `measured` is keyed off gap_stages, so filtering must
    # happen again here or such a row's None gets compared as `None != 0`
    # (pandas: NaN != 0 is True) and misreports as a block divergence.
    blocks_measured = measured[measured['gap_blocks'].notna()]
    if len(blocks_measured):
        nonzero = blocks_measured[blocks_measured['gap_blocks'] != 0]
        if len(nonzero):
            print('{} row(s) with gap_blocks != 0 (reopens a question '
                  'treated as closed):'.format(len(nonzero)))
            print(nonzero[['row_id', 'blocks', 'tcam_real', 'gap_blocks']].to_string(index=False))
        else:
            print('OK -- gap_blocks == 0 on all {} rows'.format(len(blocks_measured)))
    else:
        print('NOT ESTABLISHED -- no row has both stages_real and tcam_real to compare')

    print('\n### V6 -- void rows, unmeasured rows, failed rows, and missing strata (diagnostic)\n')
    if len(void):
        print(void[['row_id', 'compile_errors']].to_string(index=False))
    else:
        print('no void rows')
    if len(unmeasured):
        print('{} row(s) with is_void()==False but no real measurement '
              '(missing stages_real/tcam_real despite a non-void '
              'compile_errors):'.format(len(unmeasured)))
        print(unmeasured[['row_id']].to_string(index=False))
    else:
        print('no unmeasured rows')
    if failed:
        print('{} row(s) raised a toolchain exception during collect() and '
              'were not recorded (retryable):'.format(len(failed)))
        for row_id, summary in failed:
            print('  {}: {}'.format(row_id, summary))
    else:
        print('no failed rows')
    if missing:
        for group, band, stage_depth in missing:
            print('  missing stratum: group={} k_band={} stage_depth={}'.format(
                group, band, stage_depth))
    else:
        print('no missing strata')


_RESOURCE_TABLE_ROW = re.compile(
    r"^\|\s*(\S+)\s*\|\s*(-?\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|")


def _p4_table_keys(p4_path):
    """(table name -> [match key field names], field -> byte width,
    field -> BIT width) for one generated program.

    The widths come from the metadata declarations (`bit<N> code_<f>;` /
    `bit<N> <f>_val;`), byte-rounded per FIELD because that is how the ternary
    crossbar allocates -- see evaluation.codeword_fields_to_bytes. The raw bit
    widths come back too because a field that is not a whole number of bytes
    hands the crossbar a part-used byte, which is what decides whether a
    midbyte nibble survives for the version field
    (evaluation.version_block_penalty)."""
    with open(p4_path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    widths, bits = {}, {}
    for pattern in (r"bit<(\d+)>\s+(code_\w+)\s*;", r"bit<(\d+)>\s+(\w+_val)\s*;"):
        for match in re.finditer(pattern, text):
            widths[match.group(2)] = math.ceil(int(match.group(1)) / 8)
            bits[match.group(2)] = int(match.group(1))
    tables, current = {}, None
    for line in text.splitlines():
        opened = re.match(r"\s*table\s+(\w+)\s*\{", line)
        if opened:
            current = opened.group(1)
            tables[current] = []
            continue
        key = re.match(r"\s*meta\.(\w+)\s*:\s*(ternary|range)\s*;", line)
        if key and current:
            tables[current].append(key.group(1))
    return tables, widths, bits


def _committed_blocks(logs_dir):
    """table basename -> physical TCAM block count, from the COMMITTED
    allocation in mau.resources.log. Returns None when the backend never got
    as far as allocating resources (see replay_stage_depth's raise)."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    blocks = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = _RESOURCE_TABLE_ROW.match(line)
        if match and not match.group(1).endswith('$action'):
            blocks[match.group(1).split('.')[-1]] = int(match.group(7))
    return blocks


def committed_register_stages(logs_dir):
    """register base name -> the stage the compiler ran its RegisterAction in,
    from the COMMITTED allocation in mau.resources.log.

    The direct measurement behind evaluation.METER_ALUS_PER_STAGE: a stage's
    register count never exceeds 4 in any of these compiles, and the
    Percentage table reports 4 as 100% of the stage's Meter ALUs."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    stages = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = _RESOURCE_TABLE_ROW.match(line)
        if match and match.group(1).endswith('_reg'):
            name = match.group(1).split('.')[-1]
            stages[name[:-len('_reg')]] = int(match.group(2))
    return stages


def committed_table_stages(logs_dir):
    """table basename -> the stage the compiler placed it in, from the
    COMMITTED allocation in mau.resources.log.

    The measurement Mechanism B is checked against: a stage the placer spends
    entirely inside a gated register block can hold no table of the outer
    sequence, so it shows up here as a HOLE in the range pool's occupancy --
    a stage index between two occupied ones that no table landed in. Returns
    None when the backend never allocated resources, exactly as
    _committed_blocks does."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    stages = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = _RESOURCE_TABLE_ROW.match(line)
        if match and not match.group(1).endswith('$action'):
            stages[match.group(1).split('.')[-1]] = int(match.group(2))
    return stages


def placement_round_states(logs_dir):
    """The placement rounds p4c actually ran, in order, e.g. ['INITIAL'] or
    ['INITIAL', 'NOCC_TRY1', 'REDO_PHV1'].

    NOCC_TRY is a container-conflicts-DISABLED re-placement of the identical
    program -- a free controlled experiment for Mechanism A, and the
    measurement its "PHV delta" column comes from. But p4c only runs it when
    the initial round leaves it something to retry, so a row with no NOCC_TRY
    round has no counterfactual at all and its delta is 0 by absence, not by
    measurement. Reading that 0 as "no PHV effect here" is what mis-attributed
    open_issues.md item 16; this function makes the distinction checkable."""
    path = os.path.join(logs_dir, 'table_summary.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    return re.findall(r'Table allocation done \d+ time\(s\), state = (\S+)', text)


def _latest_round_log(logs_dir, prefix):
    """The HIGHEST-numbered <prefix>_<N>.log in logs_dir.

    These logs exist once per placement round and round 0 is the DISCARDED
    initial allocation, exactly as table_summary.log's first stage count is
    (see committed_stages_real). Reading the lowest instead is what cost the
    first calibration pass its "+1 range packing" conclusion."""
    numbered = []
    for name in os.listdir(logs_dir):
        match = re.fullmatch(re.escape(prefix) + r'_(\d+)\.log', name)
        if match:
            numbered.append((int(match.group(1)), name))
    if not numbered:
        raise ValueError('no {}_<N>.log in {}'.format(prefix, logs_dir))
    return os.path.join(logs_dir, max(numbered)[1])


_PHV_ADVANCE_RE = re.compile(
    r'action dependency between (\S+) and table (\S+) due to PHV allocation '
    r'advances stage to (\d+)')


def phv_advanced_tables(logs_dir):
    """table basename -> the stage p4c's own placement log says a PHV-induced
    action dependency pushed it to (the LAST such stage, if it was pushed more
    than once).

    This is Mechanism A stated by the compiler rather than inferred: a Tofino
    stage has one action ALU per PHV CONTAINER, so two match tables whose
    actions write fields the allocator packed into one container cannot share
    a stage, and p4c logs the resulting advance verbatim:

        - action dependency between table_14_ddos_bwd_packet_length_max_0 and
          table table_13_ddos_bwd_iat_min_0 due to PHV allocation advances
          stage to 9

    The SECOND table named is the one that moves. Placement-log names carry a
    trailing `_<n>` suffix that mau.resources.log's do not (the generator's own
    table names never end in a digit -- they end in a feature name), so it is
    stripped to keep both keyed the same way."""
    with open(_latest_round_log(logs_dir, 'table_placement'),
              encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    advanced = {}
    for _blocker, moved, stage in _PHV_ADVANCE_RE.findall(text):
        name = re.sub(r'_\d+$', '', moved)
        advanced[name] = max(int(stage), advanced.get(name, 0))
    return advanced


def committed_stages_real(logs_dir):
    """The compiler's FINAL stage count for a program.

    table_summary.log holds one "Number of stages in table allocation" line
    per placement round (INITIAL, then NOCC_TRY/REDO_PHV retries), and only
    the LAST is the allocation the compiler commits to -- cross-checked
    against mau.resources.log, which only ever reflects the committed one.
    Reading the first instead (a bare re.search) over-reported 4 of this
    study's 19 rows by 1-2 stages."""
    path = os.path.join(logs_dir, 'table_summary.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        last_round = handle.read().split('Table allocation done ')[-1]
    return int(re.search(r'stages in table allocation:\s*(\d+)',
                          last_round).group(1))


def replay_stage_depth(row_id, artifacts_root, readiness_levels=None):
    """Returns (predicted_stage_depth, committed_stages_real) for one
    already-compiled calibration row.

    Feeds the estimator's packer the REAL per-table facts from the row's own
    generated P4 and committed compile logs -- key field identities and
    widths, physical block counts, per-feature readiness levels -- so the
    only thing being compared is stage PLACEMENT. Deliberately does not
    refit the archived model: that would fold the block model's own error
    (see this module's V5) into a stage measurement.

    readiness_levels (optional) overrides the model's own per-feature levels
    with a {raw_feature_name: level} mapping, so a caller can substitute a
    different level source and see what the packer then does with it. It
    exists for one measurement: feeding in an ORACLE -- levels read off the
    compiler's own committed register placement, one stage past each feature's
    last register -- moves the predicted depth on none of the 18 rows, which
    is what killed open_issues.md item 16's readiness-accuracy hypothesis.
    Default None keeps readiness_levels_for, i.e. the real model.

    Raises ValueError when the backend produced no resource allocation,
    which is how a program that does not fit the chip shows up (see
    independent_high_sd12: "tofino supports up to 12 stages, using 13").
    Returning a plausible number there would price an infeasible design as a
    cheap one."""
    logs_dir = os.path.join(artifacts_root, 'compiles', row_id, 'pipe', 'logs')
    blocks = _committed_blocks(logs_dir)
    if blocks is None:
        raise ValueError(
            '%s: the compiler never allocated resources for this program '
            '(mau.resources.log has no "Allocated Resource Usage" section), so '
            'there is no committed placement to replay. Its table placement '
            'needs %d stages against Tofino\'s 12.'
            % (row_id, committed_stages_real(logs_dir)))

    tables, widths, bits = _p4_table_keys(
        os.path.join(artifacts_root, 'p4_src', row_id + '.p4'))

    # A range table keys on exactly one meta.<raw_feature>_val field. Levels
    # come from one schedule over the row's WHOLE feature set, not per
    # feature: the stateful-ALU cap is a property of the set (see
    # evaluation.register_stage_schedule). Two tables of different models may
    # name the same feature -- the register behind it is emitted once, so it
    # appears in the schedule once.
    row_features = []
    for name, keys in tables.items():
        if name.startswith('table_') and name in blocks and keys:
            feature = keys[0][:-len('_val')]
            if feature not in row_features:
                row_features.append(feature)
    if readiness_levels is None:
        levels = dict(zip(row_features, readiness_levels_for(row_features)))
    else:
        missing = [f for f in row_features if f not in readiness_levels]
        if missing:
            raise ValueError(
                '%s: readiness_levels names no level for %s; every feature with '
                'a range table must have one or the packer would place it at an '
                'arbitrary stage' % (row_id, ', '.join(sorted(missing))))
        levels = readiness_levels
    # Mechanism B: stages the placer spends wholly inside a gated register
    # block hold no table from the outer sequence, however empty they are.
    interior = gated_block_interior_stages(row_features)

    range_specs, range_fields, range_levels = [], [], []
    ternary_specs, ternary_fields, ternary_key_bits = [], [], []
    for name, keys in tables.items():
        if name not in blocks or not keys:
            continue
        fields = frozenset((key, widths[key]) for key in keys)
        width = sum(field_bytes for _, field_bytes in fields)
        if name.startswith('table_'):
            range_specs.append((blocks[name], width))
            range_fields.append(fields)
            range_levels.append(levels[keys[0][:-len('_val')]])
        elif name.startswith('get_classification_tree'):
            # blocks[name] is the count p4c committed, which is exactly the
            # key's cost at the offset it actually got -- and every committed
            # classification key in this corpus sits at crossbar group 0. That
            # is the same basis crossbar_stages_needed assumes for a declared
            # spec, so it is fed in unmodified: the packer adds only
            # version_block_delta, i.e. what a DIFFERENT offset would change.
            key_bits = tuple(sorted(bits[key] for key in keys))
            ternary_specs.append((blocks[name], width))
            ternary_fields.append(fields)
            ternary_key_bits.append(key_bits)

    range_plan = crossbar_stages_needed(range_specs, readiness_levels=range_levels,
                                         key_fields=range_fields,
                                         unavailable_stages=interior)
    ternary_level = range_plan.depth if range_specs else 0
    # key_field_bits matters even though the block counts here are p4c's own.
    # The version charge is what makes a MIXED stage infeasible: in
    # independent_low_sd9 two 9-block app tables and two 3-block ddos tables
    # pack cleanly into 9+3 | 9+3 = 24 blocks, and p4c still refuses, because
    # whichever key it hands the later crossbar groups pays a version block
    # and 26 > 24. The committed placement then puts each key at offset 0, so
    # nothing actually pays -- the charge decides the PLACEMENT, not the
    # invoice.
    ternary_plan = crossbar_stages_needed(
        ternary_specs, readiness_levels=[ternary_level] * len(ternary_specs),
        key_fields=ternary_fields, unavailable_stages=interior,
        key_field_bits=ternary_key_bits)

    predicted = (max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES)
    return predicted, committed_stages_real(logs_dir)


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
    frame, missing, failed = collect(args.campaign_dir, args.out, args.output_root,
                                     strata=args.strata, groups=args.groups,
                                     k_bands=args.k_bands, limit=args.limit)
    print('{} total rows on disk at {}'.format(len(frame), args.out))
    report(frame, missing, failed)


if __name__ == '__main__':
    main()
