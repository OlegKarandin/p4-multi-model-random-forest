"""P7a: loading and pairing campaign results (spec C.3-C.5).

Replaces `load_and_combine_data` (`src/main.py:386-409`), which constructs
literal `feature_selection_comparison_results_by_k_-1_-1_{M}.csv` filenames --
a schema the current pipeline does not write. Two layouts are read:

* A compiler-verified RUN (`src/training/campaign_run.py`'s layout), detected
  by a `<results_dir>/rows/` directory. One CSV per (arm, M, split), named by
  `campaign_run.split_csv_name`:

      <run>/rows/rf_t{n_trees}_d{max_depth}_M{token}_{arm_slug}_s{NN}.csv

  `token` is the zero-padded budget (`035`) or `inf` for the unbudgeted cell,
  whose rows carry an EMPTY `M` column and `budgeted == False`. The run's
  `verification.csv` (written by `src/verify/runner.py`) is joined on
  `row_id`: p4c's numbers replace the model's in `stage_depth`/`blocks`,
  p4c-infeasible designs are dropped, and an unverified row is flagged.
* The LEGACY flat layout, one file per (arm, M) cell and no verification:

      results/rf_t{n_trees}_d{max_depth}_M{M}_{arm_slug}.csv

`arm_slug` is `TrainConfig.arm_slug` (e.g. `independent`, `joint`,
`joint-off`, archived `joint-d005`/`joint-dinf`). Legacy run manifests are
separate JSON files under `results/manifests/`, so a non-recursive
`results/*.csv` glob does not need to exclude them by name -- confirmed by
extension alone (manifests carry a `.json` suffix, never `.csv`).

Twin columns (2026-10-06): `source_row_id` str, the `joint-off` row a twin
was built from, '' otherwise; `twin_identical` 'True'/'False' text on twin
rows, '' otherwise; `alignment_postprocess` bool.

Two silent-corruption traps this module exists to close -- neither raises on
its own, both produce a plausible wrong answer:

1. Infeasible rows (`NoFeasibleSolution` at some k) carry `''` for every
   accuracy/blocks/diagnostic column, and a naive parse turns that into NaN.
   Every NaN comparison is False, so a NaN point is never dominated and
   lands on EVERY Pareto front computed downstream. `load_campaign` filters
   `infeasible == ''` before any numeric coercion happens, so the ten
   diagnostic columns' OWN `''` (meaning "not applicable", e.g. alignment
   never ran for the independent arm) never gets confused with an
   infeasible row's `''` (meaning "this k was infeasible, ignore this row
   entirely").
2. `delta_align` is a string column (`''`, `'0'`, `'0.05'`, `'inf'`, written
   by the retired `TrainConfig.delta_align_label`) carrying two non-numeric
   sentinels: `''` (alignment did not run, or -- on a file written after
   2026-09-15 -- ran with no tolerance axis to record) and `'inf'` (accept
   every move, the accept-all anchor, not a numeric value). It is OPTIONAL,
   the second archive boundary after `overlap_threshold`: nothing has written
   it since Track 5's delta_helps = FALSE verdict, and `load_campaign`
   materialises it as `''` when no loaded file has it, so the three
   delta columns are always present. `load_campaign` never compares
   this raw string on any code path: it parses unconditionally into
   `delta_align_num` (float, NaN for both sentinels) plus
   `delta_align_is_inf` (bool, the only way to tell the two sentinels
   apart once `delta_align_num` is NaN for both). That parse contract does
   not rest on any claim about how the raw strings would sort -- ordering
   is simply never evaluated on this column, so no such claim is needed to
   justify it. (An earlier draft of this docstring made one anyway, about
   `'{:g}'`-formatted values in `[0, 1)` always sorting the same
   lexicographically and numerically; that claim was wrong -- `'{:g}'`
   switches to scientific notation under about `1e-4`, e.g.
   `'{:g}'.format(5.19e-05) == '5.19e-05'`, which sorts lexicographically
   *above* an ordinary `'0.78...'` string while being numerically far
   below it. The retired `TrainConfig.delta_align` field also enforced no
   upper bound beyond >= 0, so no fixed domain could have supported the
   claim regardless. Corrected here rather than repeated.)
   The genuine, reproducible hazard on this column is different: pandas'
   own CSV dtype inference silently turns a column that is ENTIRELY the
   literal text `'inf'` (true of every real `joint-dinf` file, since the
   value is stamped identically onto every row) into float64 infinity
   before any of this module's own parsing even runs -- which is why
   `load_campaign` reads the whole file as `dtype=str` first (see that
   comment below), not as an optional hardening step.

Frame contract -- every column `load_campaign` returns, and its dtype after
parsing. Downstream modules (P7b `claims.py`, P7c `figures.py`) should treat
this as the interface:

Identity / provenance
    arm            str   'independent' or 'joint' -- NOT the full arm
                          identity (every joint sensitivity arm shares
                          arm == 'joint'; see arm_slug below).
    method         str   legacy derived duplicate of arm ('single'/'multi').
                          Never key on this -- arm_slug is the real identity.
    arm_slug       str   added by load_campaign from the filename, e.g.
                          'independent', 'joint-off', 'joint-d005',
                          'joint-dinf'. This is "arm + the parsed delta"
                          collapsed to one string -- the correct join/group
                          key.
    source_file    str   basename of the CSV this row came from.
    split          int64
    k              int64
    M              float64 TCAM block budget for this cell, `inf` for the
                          unbudgeted cell (a run's `Minf` files). Part of the
                          join key -- pair_arms requires it explicitly because
                          the legacy perform_statistical_analysis silently
                          collapsed all seven M files by keying on
                          (split, k) alone.
    budgeted       bool  False iff M is inf. Legacy files: always True.
    row_id         str   a run's `{arm_slug}_M{token}_s{NN}_k{NN}`; absent or
                          '' on legacy files.
    n_trees        int64
    max_depth      int64

Alignment configuration
    alignment_enabled bool
    delta_align       str    OPTIONAL (the 2026-09-15 archive boundary): the
                              raw label as written ('', '0', '0.05', 'inf'),
                              or '' for EVERY row when no loaded file's
                              header has the column -- true of every file
                              written after Track 5's delta_helps = FALSE
                              verdict retired the axis. Kept for provenance /
                              display; never compare this numerically.
    delta_align_num   float64  NaN when delta_align is '' (alignment did not
                              run) or 'inf' (accept-all anchor); otherwise
                              the parsed float.
    delta_align_is_inf bool   True iff delta_align == 'inf'. The only way to
                              distinguish the accept-all anchor from "not
                              applicable" once delta_align_num is NaN for
                              both.
    delta_select      float64
    overlap_threshold float64  OPTIONAL (Task 9's archive boundary): NaN when
                              alignment did not run for this row's arm/config
                              (independent arm, or joint-off), and NaN for
                              EVERY row when the loaded file's header omits
                              the column entirely -- true of every file
                              written after 2026-09-14, once the tunable
                              itself was retired (design D4). The same
                              missing-column tolerance `stage_depth` etc.
                              already have.

Outcome metrics -- present on every row because infeasible rows (whose
accuracy fields were '') have already been filtered out by the time
load_campaign returns:
    acc_app, f1_app, acc_ddos, f1_ddos   float64
    acc_sel_app, acc_sel_ddos            float64
    stages, blocks                       float64
    stage_depth   float64, NaN on any file written before this column
                  existed (real on-disk `results/rf_t11_d14_M25_*.csv` files
                  predate it) -- the loader adds it as all-NaN when a loaded
                  file's header omits it outright, the same "tolerate a
                  missing column" contract `stages_real` already has, except
                  `stages_real` is present in every real file's header with
                  '' values, while `stage_depth` may be missing the header
                  ENTIRELY. `stages`, `stage_depth` and `stages_real` are
                  THREE DIFFERENT quantities that must never be compared or
                  plotted as if they were the same thing:
                    stages      : occupied match-table stage COUNT (model).
                    stage_depth : pipeline DEPTH, what the hard 12-stage
                                  Tofino-1 ceiling reads (model, F5/F6).
                    stages_real : the real compiler's whole-program stage
                                  count, below.
    range_entries, ternary_entries       float64, NaN on any file written
                  before these columns existed (same missing-column
                  tolerance as `stage_depth` above). The model-side entry
                  counts off the refit's final ResourceUsage (see
                  TrainResult's docstring, src/training/train_model.py).
    register_depth, register_count   float64, same missing-column
                  tolerance. `register_depth` is a STAGE count (register-
                  side pipeline depth), NOT to be confused with
                  `stage_depth` above, which is the match-table side's
                  depth. `register_sram_bits` was dropped 2026-09-06 (Task
                  1, no downstream consumer read it) and is no longer part
                  of this contract; archived campaign CSVs written before
                  that date still carry a `register_sram_bits` column, but
                  it is simply ignored on load -- the missing-column
                  tolerance handles the reverse direction (newer files
                  loaded by older code), and an unrecognised extra column
                  needs no special handling at all.

Compiler verification (a run; legacy loads carry NaN / '' / False here)
    stage_depth, blocks   are p4c's numbers on a verified row (see above for
                          what each quantity means); the model's on a
                          flagged row.
    model_stage_depth, model_blocks   float64, the model's numbers (the
                          CSV's own stage_depth/blocks before substitution).
    p4c_sram, p4c_map_ram, p4c_phv_containers   float64, NaN when unverified.
    verdict        str   verification.csv's verdict ('' on legacy loads).
                          Never FALSE_FEASIBLE: p4c-infeasible designs
                          (FALSE_FEASIBLE, p4c_over_stages, p4c_over_budget)
                          are dropped, exactly as model-infeasible rows are;
                          they remain visible through load_verification.
    p4c_over_budget, p4c_over_stages, unverified, model_paths_differ   bool
    flagged        bool  True iff the row is unverified (compile error,
                          timeout, or -- with require_verified=False -- no
                          verification line): its stage_depth/blocks are the
                          MODEL's. False on every legacy row.

Feasibility
    infeasible     str   always '' after load_campaign's filter. Kept
                          (rather than dropped) so a caller can assert on it
                          directly instead of trusting the filter blindly.

Diagnostics (Task 8/9 columns) -- float64, NaN means "not applicable" for
THIS row's arm/config (e.g. alignment never ran), which is a different
meaning from 0 (alignment ran and accepted/attempted nothing) and must stay
distinguishable:
    rel_shortfall, n_trials_run, n_feasible                     float64
    align_attempted, align_accepted                             float64
    intervals_before, intervals_after                           float64
    stages_real, tcam_real   float64, NaN when hardware validation was not
                              run for this row (the campaign's default). The
                              REAL compiler's whole-program stage count
                              (registers/orientation/vote overhead included)
                              -- NOT the same quantity as `stages` or
                              `stage_depth` above; see the module docstring's
                              Outcome metrics section.
    sram_real, map_ram_real   float64, NaN under the same condition as
                              `stages_real`/`tcam_real` above (hardware
                              validation not run for this row) -- the real
                              compiler's SRAM / map-RAM block usage,
                              parsed from `p4_compile`'s result object
                              alongside `stages_real`/`tcam_real`.

Other
    features_app, features_ddos   str   ';'-joined feature names.
    best_params                   str   JSON-encoded dict (always present,
                                          since infeasible rows -- where this
                                          was '' -- are filtered out).
    compile_errors                 str   raw string, '' when hardware
                                          validation was not run.
"""
import glob
import json
import os
import re

import pandas as pd

# The value every archived row was written at. NOT a config default any more:
# the tunable is gone (design D4) and nothing writes this column. It survives
# here because arm slugs suffix only AWAY from 0.5, so reconstructing an
# archived filename from an archived row still needs to know where the
# suppressed point was. Frozen: a future change to it would silently rename
# files that already exist.
ARCHIVED_OVERLAP_DEFAULT = 0.5


class MislabelledArtifactError(ValueError):
    """A result file's filename-encoded identity disagrees with the identity
    recorded in its own columns -- e.g. a bad rename or a copy-pasted file.
    Raised instead of silently trusting either source."""


class UnverifiedRowsError(RuntimeError):
    """A run has design rows (feasible under the model) with no line in its
    verification.csv, and the caller required every one to be verified.
    `.row_ids` lists them, sorted."""

    def __init__(self, row_ids):
        self.row_ids = sorted(row_ids)
        super().__init__(
            '{} design row(s) have no verification line: {}'.format(
                len(self.row_ids), ', '.join(self.row_ids[:10])
                + (' ...' if len(self.row_ids) > 10 else '')))


class EnvironmentDriftError(RuntimeError):
    """verification.csv records a p4c image or open-p4studio commit that
    differs from the one run_manifest.json pinned for the run."""


_FILENAME_RE = re.compile(
    r'^rf_t(?P<n_trees>\d+)_d(?P<max_depth>\d+)_M(?P<M>\d+|inf)_(?P<arm_slug>.+?)(?:_s(?P<split>\d{2}))?\.csv$')

# A run's row identity, `campaign_run.row_id`: {arm_slug}_M{token}_s{NN}_k{NN}.
_ROW_ID_RE = re.compile(
    r'^(?P<arm_slug>.+)_M(?P<M>\d+|inf)_s(?P<split>\d+)_k(?P<k>\d+)$')

# Identity / join-key columns: always fully populated with true integers
# (stamped uniformly per file, or set per row regardless of feasibility), so
# these are forced to int64 -- a clean, unsurprising dtype for join keys.
# `M` is deliberately NOT here: the unbudgeted cell is inf, so it is coerced
# separately to float64 (see _coerce_M).
_INTEGER_KEY_COLUMNS = ['n_trees', 'max_depth', 'split', 'k']

# verification.csv columns carried into load_campaign's frame (besides the
# p4c numbers substituted into stage_depth/blocks).
_VERIFICATION_BOOL_COLUMNS = ('p4c_over_budget', 'p4c_over_stages', 'unverified',
                              'model_paths_differ')
_VERIFICATION_NUMERIC_COLUMNS = ('p4c_stage_depth', 'p4c_blocks', 'p4c_sram',
                                 'p4c_map_ram', 'p4c_phv_containers')
# Provenance fields that must match run_manifest.json's pin.
_ENVIRONMENT_FIELDS = ('p4c_image', 'open_p4studio_commit')

# Outcome/diagnostic columns coerced to numeric (float64, NaN for "not
# applicable") AFTER infeasible rows have been dropped. Forced to float64
# explicitly -- not left to pd.to_numeric's natural int-when-no-NaN
# inference -- so a column's dtype does not depend on which particular
# files happened to be loaded (e.g. align_attempted is int-valued whenever
# only joint arms are present, but NaN-valued as soon as an independent-arm
# file is mixed in; forcing float64 makes that composition-independent).
# Deliberately excludes `delta_align`, which is parsed separately into
# delta_align_num / delta_align_is_inf (trap 2 in the module docstring)
# precisely because it must never be compared as a plain numeric column.
_FLOAT_COLUMNS = [
    'acc_app', 'f1_app', 'acc_ddos', 'f1_ddos', 'acc_sel_app', 'acc_sel_ddos',
    'stages', 'blocks', 'stage_depth',
    'range_entries', 'ternary_entries', 'register_depth', 'register_count',
    'rel_shortfall', 'n_trials_run', 'n_feasible',
    'align_attempted', 'align_accepted', 'intervals_before', 'intervals_after',
    'stages_real', 'tcam_real', 'sram_real', 'map_ram_real',
    'delta_select',
    'model_stage_depth', 'model_blocks', 'p4c_sram', 'p4c_map_ram',
    'p4c_phv_containers',
]

# The archive boundary (Task 9): the tuple also holds the 2026-10-04
# columns, which older runs lack. overlap_threshold is written by every
# archived campaign CSV but by none written after 2026-09-14 -- TrainConfig
# no longer has the tunable (design D4, Task 7). Kept out of _FLOAT_COLUMNS'
# required set and coerced separately below so a fresh file's missing column
# degrades the same way an old file's missing `stage_depth`/etc. does: an
# all-NaN float64 column, never a KeyError.
OPTIONAL_COLUMNS = ('overlap_threshold',
                    # spec 2026-10-04 Part II; absent from runs before it
                    'select_alpha', 'chosen_trial', 'ref_trial', 'n_tied',
                    'ref_blocks', 'ref_stage_depth', 'ref_acc_sel_app',
                    'ref_acc_sel_ddos', 'ref_acc_app', 'ref_acc_ddos',
                    'ref_f1_app', 'ref_f1_ddos')

# The SECOND archive boundary, 2026-09-15: delta_align, on Track 5's
# pre-registered verdict delta_helps = FALSE (mean_d000 0.7956173344395895 vs
# mean_d020 0.7861922400433382, cells_favouring_d020 14/24). Every archived
# campaign CSV carries it; nothing written after that date does, because
# TrainConfig no longer has the field and src/main.py no longer stamps the
# column.
#
# Handled separately from OPTIONAL_COLUMNS rather than added to it, because it
# is a STRING column with two non-numeric sentinels (trap 2 in the module
# docstring) and must never go through pd.to_numeric as loaded. When a file's
# header omits it, every row of that file reads as '' -- the same value an
# archived independent / joint-off row already carries, and the same value
# _expected_arm_slug already reads as "no delta in this slug".
ARCHIVED_DELTA_ALIGN_COLUMN = 'delta_align'

# The alignment-twin arm (spec 2026-10-06): a `joint-off` design's forest
# aligned AFTER selection. Carried by one identity column,
# `alignment_postprocess`, which only ever reads True on a twin row.
TWIN_ARM_SLUG = 'joint-off-al'
TWIN_SOURCE_ARM_SLUG = 'joint-off'
POSTPROCESS_COLUMN = 'alignment_postprocess'


def _is_true(value):
    """CSV text or a bool: 'True'/True -> True, everything else False."""
    return value is True or value == 'True'


def _parse_filename(path):
    """Parse (n_trees, max_depth, M, arm_slug, split) out of a
    `rf_t{n}_d{n}_M{n|inf}_{slug}[_s{NN}].csv` basename. `M` is an int, or
    `float('inf')` for a run's unbudgeted `Minf` file; `split` is an int for a
    run's per-split file and None for a legacy per-cell file. Raises
    ValueError -- loudly, not a warning -- if the filename does not match,
    since the whole point of the glob is that the filename is
    self-describing."""
    basename = os.path.basename(path)
    m = _FILENAME_RE.match(basename)
    if not m:
        raise ValueError(
            "Filename does not match "
            "rf_t<n_trees>_d<max_depth>_M<M>_<arm_slug>[_s<NN>].csv: "
            "{!r}".format(basename))
    split = m.group('split')
    return {
        'n_trees': int(m.group('n_trees')),
        'max_depth': int(m.group('max_depth')),
        'M': float('inf') if m.group('M') == 'inf' else int(m.group('M')),
        'arm_slug': m.group('arm_slug'),
        'split': None if split is None else int(split),
    }


def _parse_bool_column(series, column, where):
    """'True'/'False' text -> bool. Never `bool(text)`: bool('False') is True.
    Anything else raises, naming the column and file."""
    mapping = {'True': True, 'False': False}
    bad = sorted(set(series) - set(mapping))
    if bad:
        raise ValueError('{}: column {!r} must be True/False, got {}'.format(
            where, column, bad))
    return series.map(mapping).astype(bool)


def _expected_arm_slug(arm, alignment_enabled, delta_align_label,
                        overlap_threshold_label='', alignment_postprocess=''):
    """Recompute the arm slug from the in-file identity columns, mirroring
    `TrainConfig.arm_slug` (src/training/config.py) and the two label helpers
    it has since lost -- `delta_align_label` and `overlap_threshold_label` --
    without needing a TrainConfig instance: the row only carries the
    already-labelled columns, not the config object that produced them.

    The aligned joint arm has TWO correct answers, and which one applies is
    decided by the row, not by a flag. A fresh file (2026-09-15 onward) has no
    `delta_align` column at all, so its label arrives as '' and the slug is
    plain `joint`. An archived file always carried a numeric or 'inf' label on
    its aligned rows, so it reconstructs to `joint-d{:03d}` / `joint-dinf` as
    it always did. The two cannot collide: an archived aligned row never
    carried '' (TrainConfig.delta_align_label only returned '' for the
    independent arm or for alignment_enabled=False, both of which return
    above).

    `alignment_postprocess` (2026-10-06) is the twin marker: True turns
    `joint-off` into `joint-off-al`. It is only legal on arm='joint' with
    alignment disabled -- a twin is an unaligned search's forest aligned
    afterwards -- so any other combination is a mislabelled file.
    """
    postprocess = _is_true(alignment_postprocess)
    if arm == 'independent':
        if postprocess:
            raise MislabelledArtifactError(
                "alignment_postprocess=True on the independent arm: a twin "
                "needs the joint encoding")
        return 'independent'
    if arm != 'joint':
        raise MislabelledArtifactError(
            "in-file 'arm' column has unrecognised value {!r} (expected "
            "'independent' or 'joint')".format(arm))
    if not alignment_enabled:
        return TWIN_ARM_SLUG if postprocess else 'joint-off'
    if postprocess:
        raise MislabelledArtifactError(
            "alignment_postprocess=True together with alignment_enabled=True: "
            "a twin is built from an UNALIGNED search's forest")
    if pd.isna(delta_align_label) or delta_align_label == '':
        # Post-2026-09-15: no tolerance axis, so no suffix.
        slug = 'joint'
    elif delta_align_label == 'inf':
        slug = 'joint-dinf'
    else:
        try:
            delta = float(delta_align_label)
        except (TypeError, ValueError):
            raise MislabelledArtifactError(
                "arm='joint' with alignment_enabled=True must carry an empty, "
                "numeric or 'inf' delta_align, got {!r}".format(
                    delta_align_label))
        slug = 'joint-d{:03d}'.format(int(round(delta * 100)))
    return slug + _expected_overlap_suffix(overlap_threshold_label)


def _expected_overlap_suffix(overlap_threshold_label):
    """Mirrors the retired `TrainConfig._overlap_suffix` (removed Task 7,
    design D4) without a TrainConfig instance: '' at the historical 0.5
    default, '-o{:03d}' anywhere else. Nothing writes this column any more --
    this function exists only to read archived rows.

    The suppressed-arm value -- '' (raw CSV text, what the retired
    `overlap_threshold_label` wrote for `independent`/`joint-off`) or NaN
    (what the same value becomes once `_FLOAT_COLUMNS` has coerced it through
    `pd.to_numeric`) -- means "no suffix". The NaN check has to come first:
    NaN compares unequal to everything, including itself, so a `== ''` check
    alone would not catch it and a float comparison against
    ARCHIVED_OVERLAP_DEFAULT would silently be False forever instead of
    matching.
    """
    if pd.isna(overlap_threshold_label) or overlap_threshold_label == '':
        return ''
    value = float(overlap_threshold_label)
    if abs(value - ARCHIVED_OVERLAP_DEFAULT) < 1e-9:
        return ''
    return '-o{:03d}'.format(int(round(value * 100)))


def _cross_check_identity(path, parsed, file_df):
    """Raise MislabelledArtifactError if this file's filename-encoded
    identity disagrees with the identity recorded in its own columns.

    What is checked, and why: n_trees/max_depth/M are stamped onto every row
    of a file uniformly by the writer (`campaign_runner._stamp` for a run),
    so they must match the filename exactly and be constant within the file.
    An unbudgeted (`Minf`) file must carry an EMPTY `M` on every row and
    `budgeted == 'False'`; a budgeted file an integer `M` and (when the
    column exists -- legacy files predate it) `budgeted == 'True'`. A run's
    per-split file must carry its filename's split on every row. arm_slug is
    not stored directly -- it is recomputed from the four columns that
    together determine it (arm, alignment_enabled, delta_align,
    overlap_threshold), which are exactly the columns TrainConfig.arm_slug and
    its two retired label helpers were derived from, so a mismatch here can
    only mean the file was mislabelled (wrong filename) or corrupted
    (inconsistent columns), not a legitimate new arm shape. Two of the four
    are archive boundaries, stood in for by '' when a fresh file omits them
    entirely; see _expected_arm_slug.
    """
    def _mismatch(field, col, expected):
        return MislabelledArtifactError(
            "{}: filename says {}={} but in-file column {!r} has {}".format(
                path, field, expected, col, sorted(set(file_df[col].tolist()))))

    def _constant_int(col, expected):
        values = pd.unique(file_df[col])
        try:
            return len(values) == 1 and int(values[0]) == expected
        except ValueError:
            return False

    for field in ('n_trees', 'max_depth'):
        if not _constant_int(field, parsed[field]):
            raise _mismatch(field, field, parsed[field])

    unbudgeted = parsed['M'] == float('inf')
    if unbudgeted:
        if not (file_df['M'] == '').all():
            raise _mismatch('M', 'M', 'inf (empty M column)')
    elif not _constant_int('M', parsed['M']):
        raise _mismatch('M', 'M', parsed['M'])
    if 'budgeted' in file_df.columns:
        expected_budgeted = 'False' if unbudgeted else 'True'
        if not (file_df['budgeted'] == expected_budgeted).all():
            raise _mismatch('M', 'budgeted', '{} (budgeted={})'.format(
                'inf' if unbudgeted else parsed['M'], expected_budgeted))
    elif unbudgeted:
        raise MislabelledArtifactError(
            "{}: an Minf file must carry a 'budgeted' column".format(path))

    if parsed.get('split') is not None and not _constant_int('split', parsed['split']):
        raise _mismatch('split', 'split', parsed['split'])

    arm_values = pd.unique(file_df['arm'])
    align_values = pd.unique(file_df['alignment_enabled'])
    # overlap_threshold (Task 9) and delta_align (2026-09-15) are the two
    # archive boundaries: every archived file has both, no fresh file has
    # either. '' -- the same "no suffix" sentinel _expected_overlap_suffix and
    # _expected_arm_slug already treat a suppressed arm's raw CSV text as
    # meaning -- stands in for the whole column when it is absent, rather than
    # this check reading a column that is not there.
    has_delta = 'delta_align' in file_df.columns
    delta_values = pd.unique(file_df['delta_align']) if has_delta else ['']
    has_overlap = 'overlap_threshold' in file_df.columns
    overlap_values = pd.unique(file_df['overlap_threshold']) if has_overlap else ['']
    has_post = POSTPROCESS_COLUMN in file_df.columns
    post_values = pd.unique(file_df[POSTPROCESS_COLUMN]) if has_post else ['']
    if (len(arm_values) != 1 or len(align_values) != 1 or len(delta_values) != 1
            or len(overlap_values) != 1 or len(post_values) != 1):
        raise MislabelledArtifactError(
            "{}: file mixes more than one (arm, alignment_enabled, delta_align, "
            "overlap_threshold, alignment_postprocess) combination -- expected "
            "exactly one per file (arm={}, alignment_enabled={}, delta_align={}, "
            "overlap_threshold={}, alignment_postprocess={})".format(
                path, list(arm_values), list(align_values), list(delta_values),
                list(overlap_values), list(post_values)))

    expected_slug = _expected_arm_slug(
        arm_values[0], bool(align_values[0]), delta_values[0], overlap_values[0],
        post_values[0])
    if expected_slug != parsed['arm_slug']:
        raise MislabelledArtifactError(
            "{}: filename says arm_slug={!r} but in-file columns "
            "(arm={!r}, alignment_enabled={!r}, delta_align={!r}, "
            "overlap_threshold={!r}) recompute to {!r}".format(
                path, parsed['arm_slug'], arm_values[0], align_values[0],
                delta_values[0], overlap_values[0], expected_slug))


def _is_run(results_dir):
    return os.path.isdir(os.path.join(results_dir, 'rows'))


def _read_verification(results_dir):
    """verification.csv as literal strings ('' for None), or None if absent."""
    path = os.path.join(results_dir, 'verification.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[], dtype=str)


def _check_environment(results_dir, verification):
    """Raise EnvironmentDriftError if verification.csv's p4c image or
    open-p4studio commit differs from run_manifest.json's pin. A manifest
    value of None (a locally trained run, no pinned image) skips that field;
    so does a missing manifest."""
    path = os.path.join(results_dir, 'run_manifest.json')
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    for field in _ENVIRONMENT_FIELDS:
        pinned = manifest.get(field)
        if pinned is None or field not in verification.columns:
            continue
        seen = sorted(set(verification[field]) - {pinned})
        if seen:
            raise EnvironmentDriftError(
                '{}: run_manifest.json pins {}={!r} but verification.csv also '
                'records {}'.format(results_dir, field, pinned, seen))


def _join_verification(df, results_dir, require_verified):
    """Join a run's verification.csv onto its (already model-feasible) rows,
    substitute p4c's numbers, drop p4c-infeasible designs (O1) and flag
    unverified rows. `df` holds string cells; numeric coercion follows."""
    verification = _read_verification(results_dir)
    if verification is None:
        verification = pd.DataFrame(columns=['row_id', 'verdict']
                                    + list(_VERIFICATION_BOOL_COLUMNS)
                                    + list(_VERIFICATION_NUMERIC_COLUMNS))
    else:
        _check_environment(results_dir, verification)

    missing = sorted(set(df['row_id']) - set(verification['row_id']))
    if missing and require_verified:
        raise UnverifiedRowsError(missing)

    carried = (['row_id', 'verdict'] + list(_VERIFICATION_BOOL_COLUMNS)
               + list(_VERIFICATION_NUMERIC_COLUMNS))
    right = verification[carried].copy()
    for col in _VERIFICATION_BOOL_COLUMNS:
        # model_paths_differ may be '' (training path absent); read as False.
        text = right[col].replace('', 'False') if col == 'model_paths_differ' \
            else right[col]
        right[col] = _parse_bool_column(text, col, 'verification.csv')
    df = df.drop(columns=[c for c in carried if c != 'row_id' and c in df.columns])
    df = df.merge(right, on='row_id', how='left', indicator='_verified')
    has_line = df.pop('_verified') == 'both'

    # Rows with no line (only reachable with require_verified=False) are
    # unverified: model numbers kept, verdict '', flags False, flagged True.
    df['verdict'] = df['verdict'].where(has_line, '')
    for col in _VERIFICATION_BOOL_COLUMNS:
        df[col] = df[col].where(has_line, False).astype(bool)
    for col in _VERIFICATION_NUMERIC_COLUMNS:
        df[col] = df[col].where(has_line, '')

    # O1: a design p4c finds infeasible leaves the frame, exactly as a
    # model-infeasible row does. load_verification still reports it.
    excluded = has_line & ((df['verdict'] == 'FALSE_FEASIBLE')
                           | df['p4c_over_stages'] | df['p4c_over_budget'])
    df, has_line = df[~excluded], has_line[~excluded]

    flagged = df['unverified'] | ~has_line
    df['model_stage_depth'] = df['stage_depth']
    df['model_blocks'] = df['blocks']
    df['stage_depth'] = df['p4c_stage_depth'].where(~flagged, df['model_stage_depth'])
    df['blocks'] = df['p4c_blocks'].where(~flagged, df['model_blocks'])
    df['flagged'] = flagged.astype(bool)
    return df.drop(columns=['p4c_stage_depth', 'p4c_blocks']).reset_index(drop=True)


def _coerce_M(series):
    """M text -> float64: '' (an unbudgeted run row) and 'inf' -> inf."""
    return pd.to_numeric(series.replace({'': 'inf'}), errors='raise').astype('float64')


def load_verification(results_dir):
    """The raw verification.csv of a run, every line -- including designs
    load_campaign drops as p4c-infeasible (O1) -- with each row's
    `arm_slug`, `M` (float64, inf when unbudgeted), `split` and `k` parsed
    from its `row_id` ({arm_slug}_M{token}_s{NN}_k{NN}). Every other column
    stays the literal CSV text ('True'/'False', '' for None). The agreement
    table reads this.

    Raises FileNotFoundError if the run has no verification.csv.
    """
    verification = _read_verification(results_dir)
    if verification is None:
        raise FileNotFoundError(
            'No verification.csv in {!r}'.format(results_dir))
    parts = verification['row_id'].str.extract(_ROW_ID_RE)
    bad = verification['row_id'][parts['arm_slug'].isna()].tolist()
    if bad:
        raise ValueError('verification.csv row_id(s) not of the form '
                         '{{arm_slug}}_M{{token}}_s{{NN}}_k{{NN}}: {}'.format(bad))
    verification = verification.drop(columns=['M']).copy() \
        if 'M' in verification.columns else verification.copy()
    verification['arm_slug'] = parts['arm_slug']
    verification['M'] = _coerce_M(parts['M'])
    verification['split'] = parts['split'].astype('int64')
    verification['k'] = parts['k'].astype('int64')
    return verification


def load_campaign(results_dir='results', require_verified=None):
    """Load a campaign as one frame. See the module docstring for the full
    column contract of the returned frame.

    Layout: if `<results_dir>/rows/` exists this is a compiler-verified run --
    glob `rows/*.csv`, and `require_verified` defaults to True. Otherwise glob
    `results_dir` for legacy `rf_t*_d*_M*_*.csv` files, with
    `require_verified` defaulting to False (legacy loads never read a
    verification file).

    Each file's filename identity is cross-checked against its own columns
    (MislabelledArtifactError on disagreement), infeasible rows are filtered
    and delta_align is parsed. For a run, verification.csv is then joined on
    `row_id` for every design row (`infeasible == ''`):

    * a design row with no verification line raises UnverifiedRowsError when
      `require_verified` (else it is kept, model numbers, `flagged=True`);
    * a p4c image / open-p4studio commit differing from run_manifest.json
      raises EnvironmentDriftError (skipped where the manifest has None);
    * `stage_depth`/`blocks` move to `model_stage_depth`/`model_blocks` and
      take p4c's numbers;
    * designs p4c finds infeasible (FALSE_FEASIBLE, p4c_over_stages or
      p4c_over_budget) are dropped (O1);
    * unverified rows (compile error / timeout) keep the model's numbers and
      get `flagged=True`; all others `flagged=False`.

    Raises FileNotFoundError if no files match -- an empty campaign frame is
    never a useful silent result for downstream analysis.
    """
    is_run = _is_run(results_dir)
    if require_verified is None:
        require_verified = is_run
    if is_run:
        pattern = os.path.join(results_dir, 'rows', '*.csv')
    else:
        pattern = os.path.join(results_dir, 'rf_t*_d*_M*_*.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            "No campaign result files matched {!r}".format(pattern))

    frames = []
    for path in paths:
        parsed = _parse_filename(path)
        # keep_default_na=False, na_values=[]: this schema uses '' itself as
        # a meaningful "not applicable" / infeasible-row marker (trap 1 in
        # the module docstring). Letting pandas' default NA sniffing turn
        # '' into NaN at read time would make infeasible == '' unusable and
        # would make a genuine NaN indistinguishable from "not applicable".
        #
        # dtype=str: forces EVERY column to be read as a literal string,
        # bypassing pandas' per-column type inference entirely. Without
        # this, a `delta_align` column whose every row is the literal text
        # 'inf' (true of every real joint-dinf file, since the value is
        # stamped once per file onto every row) gets silently inferred as
        # float64 infinity instead of the string 'inf' -- a second, sharper
        # form of trap 2, invisible in a mixed-value test fixture and only
        # surfacing on a real single-arm file. Every column this module
        # cares about is re-parsed explicitly below (numeric via
        # pd.to_numeric, alignment_enabled via an explicit 'True'/'False'
        # map), so forcing str at read time costs nothing and removes the
        # inference landmine uniformly rather than column-by-column.
        file_df = pd.read_csv(
            path, keep_default_na=False, na_values=[], dtype=str)
        file_df['alignment_enabled'] = file_df['alignment_enabled'] == 'True'
        _cross_check_identity(path, parsed, file_df)
        file_df[POSTPROCESS_COLUMN] = (file_df[POSTPROCESS_COLUMN] == 'True'
                                       if POSTPROCESS_COLUMN in file_df.columns
                                       else False)
        if 'budgeted' in file_df.columns:
            file_df['budgeted'] = _parse_bool_column(
                file_df['budgeted'], 'budgeted', path)
        else:
            file_df['budgeted'] = True
        file_df['arm_slug'] = parsed['arm_slug']
        file_df['source_file'] = os.path.basename(path)
        frames.append(file_df)

    df = pd.concat(frames, ignore_index=True)

    # Trap 1: drop infeasible rows BEFORE any numeric coercion. Order
    # matters -- the ten diagnostic columns also carry '' on feasible rows
    # where they are simply not applicable (e.g. alignment never ran for the
    # independent arm), so coercing to numeric before filtering would not
    # distinguish "infeasible, discard this row" from "feasible, alignment
    # not applicable, coerce to NaN and keep the row" -- both would produce
    # NaN either way, but only the filter step is allowed to decide which
    # rows disappear entirely.
    df = df[df['infeasible'] == ''].reset_index(drop=True)

    if is_run:
        df = _join_verification(df, results_dir, require_verified)
    else:
        # Legacy loads carry no verification: nothing is flagged.
        df['verdict'] = ''
        df['flagged'] = False
        for col in _VERIFICATION_BOOL_COLUMNS:
            df[col] = False

    for col in _INTEGER_KEY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='raise').astype('int64')
    df['M'] = _coerce_M(df['M'])

    for col in _FLOAT_COLUMNS + list(OPTIONAL_COLUMNS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        else:
            # A column absent from EVERY loaded file's header entirely --
            # not merely '' on some rows -- e.g. `stage_depth` (F5/F6) on any
            # file written before this column existed, including every real
            # on-disk results/rf_t11_d14_M25_*.csv today. `col in df.columns`
            # above only catches per-VALUE absence ('' -> NaN via coerce);
            # this branch is what makes per-COLUMN absence degrade the same
            # way, to an all-NaN float64 column, rather than the column
            # missing from the returned frame altogether.
            df[col] = float('nan')

    # The 2026-09-15 archive boundary: nothing writes delta_align any more, so
    # a frame built entirely from fresh files has no such column. Materialise
    # it as '' -- "alignment ran, at no tolerance" -- so the parse below, and
    # every downstream reader of delta_align / delta_align_num /
    # delta_align_is_inf, sees the same three columns whatever the frame was
    # built from. Done BEFORE the parse rather than special-cased inside it,
    # for the same reason the missing-column branch above exists: a column's
    # presence must not depend on which files happened to be loaded.
    if ARCHIVED_DELTA_ALIGN_COLUMN not in df.columns:
        df[ARCHIVED_DELTA_ALIGN_COLUMN] = ''

    # Trap 2: delta_align is a string column; never compare it numerically
    # as loaded. Parse into a nullable-by-NaN float plus an explicit is_inf
    # flag, so 'inf' cannot be silently coerced into some numeric sentinel
    # and so string ordering never substitutes for numeric ordering.
    df['delta_align_is_inf'] = df['delta_align'] == 'inf'
    delta_for_numeric = df['delta_align'].mask(df['delta_align_is_inf'], '')
    # .astype('float64') explicitly: if every loaded row happens to share
    # one integer-valued delta (e.g. a frame built from a single joint-d000
    # file, no '' / 'inf' rows present at all), pd.to_numeric would
    # otherwise infer int64 -- the same composition-dependent surprise
    # _FLOAT_COLUMNS guards against above, so delta_align_num gets the same
    # treatment for the same reason.
    df['delta_align_num'] = pd.to_numeric(
        delta_for_numeric, errors='coerce').astype('float64')

    return df


def pair_arms(df, treatment, baseline):
    """Inner-join the treatment arm's rows against the baseline arm's rows
    on (M, split, k) -- the join key spec C.3 requires for every paired
    claim. Replaces the pattern in the legacy `perform_statistical_analysis`
    (`analysis.py:238-240`), which keyed on (split, k) only and silently
    collapsed the seven M files into one, last-wins.

    treatment, baseline : `arm_slug` values (e.g. 'joint-d005',
        'independent', 'joint-off'). Keying on arm_slug -- not `arm` or
        `method` -- is required: `arm` only distinguishes independent/joint
        (every joint sensitivity arm shares arm == 'joint'), and `method` is
        a legacy derived duplicate of arm. arm_slug is "arm + the parsed
        delta" collapsed to the one string that is the real per-arm
        identity.

    Returns a frame with one row per (M, split, k) present in BOTH arms.
    Every column other than the join key is suffixed `_treatment` /
    `_baseline`. A cell present in only one arm (a deliberately missing
    cell, or a k an elimination run never reached) is dropped, not carried
    through with a NaN partner.
    """
    key = ['M', 'split', 'k']
    treatment_rows = df[df['arm_slug'] == treatment]
    baseline_rows = df[df['arm_slug'] == baseline]
    return treatment_rows.merge(
        baseline_rows, on=key, how='inner', suffixes=('_treatment', '_baseline'))
