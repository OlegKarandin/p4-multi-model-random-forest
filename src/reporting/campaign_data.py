"""P7a: loading and pairing campaign results (spec C.3-C.5).

Replaces `load_and_combine_data` (`src/main.py:386-409`), which constructs
literal `feature_selection_comparison_results_by_k_-1_-1_{M}.csv` filenames --
a schema the current pipeline does not write. What the campaign actually
writes, after phases P4/P5 (`src/main.py`'s `arm_result_path`,
`src/training/feature_selection.py`'s `_run_elimination`, and
`compare_independent_joint_mapping`'s per-frame column stamping):

    results/rf_t{n_trees}_d{max_depth}_M{M}_{arm_slug}.csv

one file per (arm, M) cell, `arm_slug` from `TrainConfig.arm_slug` (e.g.
`independent`, `joint-off`, `joint-d005`, `joint-dinf`). Run manifests are
separate JSON files under `results/manifests/`, so a non-recursive
`results/*.csv` glob does not need to exclude them by name -- confirmed by
extension alone (manifests carry a `.json` suffix, never `.csv`).

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
2. `delta_align` is a string column (`''`, `'0'`, `'0.05'`, `'inf'`,
   `TrainConfig.delta_align_label`) carrying two non-numeric sentinels:
   `''` (alignment did not run) and `'inf'` (accept every move -- the
   accept-all anchor, not a numeric value). `load_campaign` never compares
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
   below it. `TrainConfig.delta_align` (`src/training/config.py`) also
   enforces no upper bound beyond >= 0, so no fixed domain could have
   supported the claim regardless. Corrected here rather than repeated.)
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
    M              int64 TCAM block budget for this cell. Part of the join
                          key -- pair_arms requires it explicitly because the
                          legacy perform_statistical_analysis silently
                          collapsed all seven M files by keying on
                          (split, k) alone.
    n_trees        int64
    max_depth      int64

Alignment configuration
    alignment_enabled bool
    delta_align       str    raw label as written ('', '0', '0.05', 'inf').
                              Kept for provenance / display; never compare
                              this numerically.
    delta_align_num   float64  NaN when delta_align is '' (alignment did not
                              run) or 'inf' (accept-all anchor); otherwise
                              the parsed float.
    delta_align_is_inf bool   True iff delta_align == 'inf'. The only way to
                              distinguish the accept-all anchor from "not
                              applicable" once delta_align_num is NaN for
                              both.
    delta_select      float64
    overlap_threshold float64  NaN when alignment did not run for this row's
                              arm/config (independent arm, or joint-off).

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

Other
    features_app, features_ddos   str   ';'-joined feature names.
    best_params                   str   JSON-encoded dict (always present,
                                          since infeasible rows -- where this
                                          was '' -- are filtered out).
    compile_errors                 str   raw string, '' when hardware
                                          validation was not run.
"""
import glob
import os
import re

import pandas as pd


class MislabelledArtifactError(ValueError):
    """A result file's filename-encoded identity disagrees with the identity
    recorded in its own columns -- e.g. a bad rename or a copy-pasted file.
    Raised instead of silently trusting either source."""


_FILENAME_RE = re.compile(
    r'^rf_t(?P<n_trees>\d+)_d(?P<max_depth>\d+)_M(?P<M>\d+)_(?P<arm_slug>.+)\.csv$')

# Identity / join-key columns: always fully populated with true integers
# (stamped uniformly per file, or set per row regardless of feasibility), so
# these are forced to int64 -- a clean, unsurprising dtype for join keys.
_INTEGER_KEY_COLUMNS = ['M', 'n_trees', 'max_depth', 'split', 'k']

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
    'rel_shortfall', 'n_trials_run', 'n_feasible',
    'align_attempted', 'align_accepted', 'intervals_before', 'intervals_after',
    'stages_real', 'tcam_real',
    'delta_select', 'overlap_threshold',
]


def _parse_filename(path):
    """Parse (n_trees, max_depth, M, arm_slug) out of a
    `rf_t{n}_d{n}_M{n}_{slug}.csv` basename. Raises ValueError -- loudly,
    not a warning -- if the filename does not match, since the whole point
    of the glob is that the filename is self-describing."""
    basename = os.path.basename(path)
    m = _FILENAME_RE.match(basename)
    if not m:
        raise ValueError(
            "Filename does not match rf_t<n_trees>_d<max_depth>_M<M>_<arm_slug>.csv: "
            "{!r}".format(basename))
    return {
        'n_trees': int(m.group('n_trees')),
        'max_depth': int(m.group('max_depth')),
        'M': int(m.group('M')),
        'arm_slug': m.group('arm_slug'),
    }


def _expected_arm_slug(arm, alignment_enabled, delta_align_label):
    """Recompute the arm slug from the in-file identity columns, mirroring
    `TrainConfig.arm_slug` / `delta_align_label`'s own logic (src/training/
    config.py) without needing a TrainConfig instance -- the row only carries
    the already-labelled columns, not the config object that produced them."""
    if arm == 'independent':
        return 'independent'
    if arm != 'joint':
        raise MislabelledArtifactError(
            "in-file 'arm' column has unrecognised value {!r} (expected "
            "'independent' or 'joint')".format(arm))
    if not alignment_enabled:
        return 'joint-off'
    if delta_align_label == 'inf':
        return 'joint-dinf'
    try:
        delta = float(delta_align_label)
    except (TypeError, ValueError):
        raise MislabelledArtifactError(
            "arm='joint' with alignment_enabled=True must carry a numeric "
            "or 'inf' delta_align, got {!r}".format(delta_align_label))
    return 'joint-d{:03d}'.format(int(round(delta * 100)))


def _cross_check_identity(path, parsed, file_df):
    """Raise MislabelledArtifactError if this file's filename-encoded
    identity disagrees with the identity recorded in its own columns.

    What is checked, and why: n_trees/max_depth/M are stamped onto every row
    of a file uniformly by compare_independent_joint_mapping, so they must
    match the filename exactly and be constant within the file. arm_slug is
    not stored directly -- it is recomputed from the three columns that
    together determine it (arm, alignment_enabled, delta_align), which are
    exactly the columns TrainConfig.arm_slug itself was derived from, so a
    mismatch here can only mean the file was mislabelled (wrong filename) or
    corrupted (inconsistent columns), not a legitimate new arm shape.
    """
    for field, col in (('n_trees', 'n_trees'), ('max_depth', 'max_depth'), ('M', 'M')):
        values = pd.unique(file_df[col])
        if len(values) != 1 or int(values[0]) != parsed[field]:
            raise MislabelledArtifactError(
                "{}: filename says {}={} but in-file column {!r} has {}".format(
                    path, field, parsed[field], col, sorted(set(values.tolist()))))

    arm_values = pd.unique(file_df['arm'])
    align_values = pd.unique(file_df['alignment_enabled'])
    delta_values = pd.unique(file_df['delta_align'])
    if len(arm_values) != 1 or len(align_values) != 1 or len(delta_values) != 1:
        raise MislabelledArtifactError(
            "{}: file mixes more than one (arm, alignment_enabled, delta_align) "
            "combination -- expected exactly one per file (arm={}, "
            "alignment_enabled={}, delta_align={})".format(
                path, list(arm_values), list(align_values), list(delta_values)))

    expected_slug = _expected_arm_slug(arm_values[0], bool(align_values[0]), delta_values[0])
    if expected_slug != parsed['arm_slug']:
        raise MislabelledArtifactError(
            "{}: filename says arm_slug={!r} but in-file columns "
            "(arm={!r}, alignment_enabled={!r}, delta_align={!r}) recompute to "
            "{!r}".format(path, parsed['arm_slug'], arm_values[0], align_values[0],
                          delta_values[0], expected_slug))


def load_campaign(results_dir='results'):
    """Glob `results_dir` for `rf_t*_d*_M*_*.csv` campaign result files,
    parse each filename's identity, cross-check it against the file's own
    columns (raising MislabelledArtifactError loudly on disagreement),
    filter infeasible rows, and parse delta_align. See the module docstring
    for the full column contract of the returned frame.

    Raises FileNotFoundError if no files match -- an empty campaign frame is
    never a useful silent result for downstream analysis.
    """
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

    for col in _INTEGER_KEY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='raise').astype('int64')

    for col in _FLOAT_COLUMNS:
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
