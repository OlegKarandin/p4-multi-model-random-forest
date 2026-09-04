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
