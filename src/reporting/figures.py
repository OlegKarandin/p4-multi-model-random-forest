"""P7c: the thesis deliverables (spec C.5) -- seven from the original plan,
deliverable 8 (S3.3, T12's entries-vs-blocks question), and, for a
compiler-verified RUN only, deliverables 9 and 10 (spec section 7.3).

Everything upstream of this module exists to make these artifacts correct.
`campaign_data.load_campaign` supplies the frame, `claims.py` supplies
every statistic, and this module does one thing: render them.

| # | artifact | answers |
|---|---|---|
| 1 | per-task accuracy vs blocks plus the acc_app-vs-acc_ddos trade plane (D8), all arms overlaid | the headline comparison, never averaged |
| 2 | delta frontier: d blocks and d rel-error per task vs delta, mean +/- CI | what the tolerance buys |
| 3 | substitution scatter with quadrants, accuracy and F1 rows per arm column | the reviewer's objection, answered directly |
| 4 | paired per-task test table, Holm-corrected -- superiority AND non-inferiority families | significance |
| 5 | ablation table: constraint cost vs alignment cost | where the savings come from |
| 6 | appendix: capacity-ceiling rederivation (B.7) | replaces "chosen manually" |
| 7 | appendix: elimination order per split | reproducibility |
| 8 | entries vs blocks, faceted by k (S3.3) | T12: is the joint-mapping saving real, or an artifact of TCAM block quantization? |
| 9 | model-vs-p4c agreement per arm x M, plus every miss (run only) | does the cost model the numbers were selected by agree with the compiler? |
| 10 | budget binding per arm x M (run only) | does each block budget M actually bind? |
| 11 | alignment twins: paired twin - source, ladder, counts (run with twins only) | what does alignment buy as a post-processing step? |

Four, five, eight, nine and ten are tables and six is a replay, so `Deliverable`
carries a `figure` that is None for those; every deliverable is
independently callable, because `main.py --mode plot` (P7d) and the pilot
cell both need to invoke them one at a time.

**Nothing here averages the two tasks.** The entire rerun exists because
the old `analysis.py` reported one mean accuracy over App and DDoS -- the
number that hides a model excellent on one task and useless on the other.
Two panels, two rows, two columns; never one mean. The one place a reader
might expect an average and not find one is figure 2's relative-error
panels: the same accuracy drop is a different relative error on each task
because the error denominators differ (App errs around 0.24, DDoS around
0.04), which is exactly why the pooled number was misleading.

**Nothing here recomputes a `claims.py` statistic.** A figure that
disagrees with the table beside it is the failure P7d exists to eliminate,
and the old code contained two independent copies of the same averaging
rule (`plotting.py:375-378` inline, plus `extract_approach_data`) for
precisely that reason. Fronts, coverage, correlations, quadrants,
confidence intervals, Wilcoxon tests, Holm correction and the ablation
contrasts are all imported, never re-derived. The two quantities this
module does compute itself are the ones `claims.py` does not own:

* the RELATIVE error change `(e_treatment - e_baseline) / e_baseline`
  per task (spec's difficulty-normalised scale, section 9's measurement
  log), built on top of `campaign_data.pair_arms` and then aggregated by
  `claims.delta_frontier` so the interval machinery stays single-sourced;
* the elimination order, which is a re-reading of the `features_app` /
  `features_ddos` columns across descending k, not a statistic.

State hygiene -- the five leaks in `plotting.py` this module must not
reproduce:

* `plt.style.use('default')` and `sns.set_palette` at import
  (`plotting.py:7-8`), and a global `font.family = 'Times New Roman'`
  (`:325`). Nothing here writes to `matplotlib.rcParams` at all: figures
  are built as bare `matplotlib.figure.Figure` objects and every visual
  property is passed per-artist. `matplotlib.use('Agg')` at import is the
  single deliberate global call, mandated by the plan as the headless
  guarantee; it selects a file backend and changes no styling.
* `pyplot` is never imported, which is a stronger guarantee than "no
  `plt.show()`": a figure that never enters pyplot's figure manager can
  neither be shown nor leaked, so a batch run cannot block
  (`plotting.py:438` and `:527` both call `plt.show()`) and cannot
  accumulate figures.
* A 3x3 subplot grid assuming exactly nine k values (`:327-328`) and
  hardcoded axis limits (`:416-429`). Panel counts here are derived from
  the arms present and limits are left to matplotlib.
* A `k == 17 -> drop acc < 0.8` special case (`:356-361`). There is no k
  filter and no accuracy filter anywhere in this module; feasibility
  filtering already happened once, at load.

Arm ordering follows `claims.JOINT_ARM_SLUGS` -- the 3-arm design since
Task 13: `independent` first, then `joint-off`, then `joint-off-al`; a legacy `joint` arm is appended -- and any arm
slug this module has never heard of is appended at the end rather than
dropped: an unrecognised arm is a thing to see in the figure, not to hide.
That includes an ARCHIVED 7-arm frame's delta arms (`joint-d005`,
`joint-dinf`, ...): the figures that overlay arms (1, 2, 7, 8) still draw
them, after the current arms, but the claims families (3, 4, 5) run only
over `claims.JOINT_ARM_SLUGS`, so on archived data those deliverables cover
`joint-off` alone (and the default family-size gate refuses them).

**M may be infinite.** A run's unbudgeted cell has `M = inf` (float64), so
every label that prints an M goes through `format_M` ('25', or the infinity
sign), never `str(M)`, which would print '25.0' or 'inf'. The CSVs keep M as
a number (`inf` round-trips through pandas); only human-facing text is
formatted.

**Flagged rows.** In a run, a row p4c could not verify (compile error,
timeout) is `flagged` and carries the MODEL's numbers. Every scatter splits
its rows on `flagged` and draws the flagged ones with `FLAGGED_MARKER` and
`gid='flagged:{arm}'`, so a reader can always tell a compiler-checked point
from an unchecked one.
"""
import contextlib
import io
import json
import os
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import matplotlib

# The headless guarantee the plan requires. Selects a file backend for the
# whole process; it does not touch styling, and this module never uses
# pyplot, so nothing here depends on which backend is active.
matplotlib.use('Agg')

from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure     # noqa: E402  (must follow use())
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from src.reporting import claims         # noqa: E402
from src.reporting.campaign_data import pair_arms   # noqa: E402


DEFAULT_FIGURE_DIR = os.path.join('results', 'figures')
DEFAULT_CEILING_CSV = os.path.join('results', 'capacity_ceiling.csv')

# (task key, accuracy column, short name, panel title, feature-set column).
# The single place the two tasks are enumerated -- add a third task here and
# every per-task panel and per-task appendix follows, with no per-task
# branching anywhere else in the module.
TASKS = (
    ('app', 'acc_app', 'App', 'Application identification', 'features_app'),
    ('ddos', 'acc_ddos', 'DDoS', 'DDoS detection', 'features_ddos'),
)

# Figure 2's reported quantities: one accuracy relative-error change per
# task, one F1 relative-error change per task, plus the block delta. Derived
# from TASKS rather than spelled out, so a third task would gain a panel
# here too and the lists cannot drift apart. Deliberately one panel per task
# per metric rather than one shared axis.
#
# `TASKS` (the single place tasks are enumerated) has no F1 column field --
# widening its 5-tuple shape would ripple into every site that unpacks it --
# so the F1 column name is derived from each task's `key` by the same
# string convention the campaign CSV already uses (`acc_<key>` / `f1_<key>`):
# `F1_REL_ERROR_PREFIX + key` reads as `'f1_' + key`. This is a SEPARATE
# edit from `claims.DEFAULT_METRICS` gaining F1 (Task 11): that constant
# drives `claims.py`'s own statistics, while `FRONTIER_METRICS` drives only
# this module's panel layout and is derived from `TASKS`, not from
# `DEFAULT_METRICS`.
REL_ERROR_PREFIX = 'rel_error_change_'
F1_REL_ERROR_PREFIX = 'rel_error_change_f1_'
BLOCKS_DELTA = 'd_blocks'
REL_ERROR_METRICS = tuple(REL_ERROR_PREFIX + key for key, _, _, _, _ in TASKS)
F1_REL_ERROR_METRICS = tuple(F1_REL_ERROR_PREFIX + key for key, _, _, _, _ in TASKS)
FRONTIER_METRICS = REL_ERROR_METRICS + F1_REL_ERROR_METRICS + (BLOCKS_DELTA,)

# Per-panel size in inches, for a SMALL grid -- multiplied by the panel
# counts a given frame implies, never a fixed canvas for a fixed grid.
_PANEL_WIDTH = 6.0
_PANEL_HEIGHT = 4.2

# A panel below this stops being legible, so `_make_figure` never scales a
# grid down past it, no matter how many rows/columns the frame implies.
_MIN_PANEL_SIZE = 1.8

# Total-figure caps, per axis. Chosen as 3x the small-grid per-panel
# constant above, which has two effects at once: (1) any grid of <=3
# columns/rows -- the 1-3-panel shapes _PANEL_WIDTH/_PANEL_HEIGHT were
# originally sized for (figure 3's 2-row grid, figure 1's 3-row trade-plane
# grid, an old single-arm figure) -- divides out to exactly the original
# per-panel size, unchanged; (2) a wider/taller grid trades panel size down
# to keep the TOTAL figure at this cap instead of growing without bound.
# At this branch's largest real grid (deliverable 2: 9 odd-k columns x 5
# FRONTIER_METRICS rows), this caps the rendered PDF at 18.0in x 12.6in
# instead of the (_PANEL_WIDTH * 9, _PANEL_HEIGHT * 5) = 54.0in x 21.0in an
# unconditional per-panel multiply produces -- unusable as a thesis figure.
_MAX_FIGURE_WIDTH = 3 * _PANEL_WIDTH
_MAX_FIGURE_HEIGHT = 3 * _PANEL_HEIGHT

# Marker cycle, used together with the colour cycle so that a campaign with
# more arms than the qualitative colormap has colours stays readable.
_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>')

# A flagged (unverified) point: p4c never confirmed its numbers, so it is
# drawn differently from every compiler-checked point, in every scatter.
FLAGGED_MARKER = 'x'

_INFINITY_SIGN = '\u221e'


def format_M(M):
    """An M value as a label: '25' for a whole number (never '25.0'), the
    infinity sign for the unbudgeted cell (never 'inf'), '{:g}' otherwise.
    Every figure and table label that prints an M goes through this."""
    value = float(M)
    if np.isinf(value):
        return _INFINITY_SIGN
    if value.is_integer():
        return str(int(value))
    return '{:g}'.format(value)


def _flagged(frame, column='flagged'):
    """`frame[column]` as a bool Series (all False when absent): a run's
    parsed bools, or the literal 'True'/'False' text."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values
    return values.map(lambda value: str(value).strip().lower() == 'true')


def _flagged_sentence(n_flagged, n_rows, unit='rows'):
    """The caption sentence for flagged points, or '' when there are none."""
    if not n_flagged:
        return ''
    return (' {} of {} {} are FLAGGED and drawn as "{}": p4c did not verify '
            'them (a compile error or a timeout), so they carry the MODEL\'s '
            'numbers, not the compiler\'s.'.format(
                n_flagged, n_rows, unit, FLAGGED_MARKER))


@dataclass
class Deliverable:
    """One §C.5 artifact: its identity, the exact table behind it, the
    figure if it has one, and the files written for it.

    `data` is the table the artifact renders, not a summary of it -- it is
    written alongside as CSV so a reader can check any drawn point against
    the number it came from, and so the tests can assert that the two
    agree.
    """
    number: int
    slug: str
    title: str
    caption: str
    data: Optional[pd.DataFrame] = None
    figure: Optional[Figure] = None
    markdown_body: Optional[str] = None
    paths: Tuple[str, ...] = field(default=())
    # Further tables written beside `data` as `<stem>_<suffix>.csv`, e.g.
    # deliverable 9's misses next to its summary.
    extra_data: Tuple[Tuple[str, pd.DataFrame], ...] = field(default=())


def _log(message):
    """Log a message to stdout. Used to announce which facets were computed
    but not shown in the deliverable figures."""
    print(message)


# ---------------------------------------------------------------------------
# Arms, labels and per-arm styling
# ---------------------------------------------------------------------------

def ordered_arms(df, include_baseline=True,
                 baseline=claims.INDEPENDENT_ARM_SLUG):
    """The arm slugs present in `df`, in design order.

    Known arms come first in `claims.JOINT_ARM_SLUGS` order (`joint-off`,
    then `joint-off-al`; a legacy `joint` arm is appended), so tables and figures read left to right as the design.
    An arm slug this module does not recognise -- including an archived
    delta arm such as `joint-d005` -- is APPENDED rather than dropped: a
    campaign that grew an arm should show up in the figure, not vanish from
    it.
    """
    present = list(dict.fromkeys(df['arm_slug'].tolist()))
    known = list(claims.JOINT_ARM_SLUGS)
    if include_baseline:
        known = [baseline] + known
    ordered = [slug for slug in known if slug in present]
    extras = [slug for slug in present
              if slug not in known and slug != baseline]
    return tuple(ordered + extras)


def require_baseline(df, baseline, where):
    """Refuse to render a paired artifact when the baseline arm is absent.

    Every paired figure pairs each treatment arm against `baseline` on
    (M, split, k); with no baseline rows, every join is empty and the figure
    renders BLANK -- axes, ticks, caption and all, with nothing plotted. A
    blank figure that looks like a finished one is worse than no figure, so
    this raises the way `claims.paired_tests` already raises on a contrast
    with no paired cells.
    """
    if baseline not in set(df['arm_slug'].unique()):
        raise ValueError(
            '{}: baseline arm {!r} is not present in the frame, so every '
            'pairing on (M, split, k) would be empty and the figure would '
            'render blank. Arms found: {}.'.format(
                where, baseline, sorted(df['arm_slug'].unique().tolist())))


def _delta_tick_label(arm_slug, delta_num, is_inf):
    """The x-axis label for one arm on the joint-arm axis.

    Non-numeric arms keep their own identity instead of being given an
    invented numeric position: `joint-off` (sharing only) never ran alignment
    at all (not delta 0) and `joint-off-al` is sharing plus post-selection
    threshold alignment (no tolerance axis since 2026-09-15; `joint` is only
    a legacy slug). An archived frame's `joint-dinf` (the accept-all anchor,
    not a large number) keeps 'inf', and its numeric delta arms their delta.
    """
    if bool(is_inf):
        return 'inf'
    if pd.isna(delta_num):
        return arm_slug.replace('joint-', '')
    return '{:g}'.format(delta_num)


def _arm_styles(arms):
    """A (colour, marker) per arm, drawn from a qualitative colormap sized
    to the arms actually present. Beyond the colormap's length colours
    repeat, which is why the marker cycles at a different period."""
    colormap = matplotlib.colormaps['tab10']
    return {arm: (colormap(index % colormap.N),
                  _MARKERS[index % len(_MARKERS)])
            for index, arm in enumerate(arms)}


def _make_figure(n_rows, n_columns):
    """A `Figure` sized from the grid a deliverable's data implies, never a
    literal panel count (see the module docstring's state-hygiene rules).

    Per-panel size is `min(_PANEL_WIDTH, _MAX_FIGURE_WIDTH / n_columns)`
    (and the height equivalent): at <=3 columns/rows this divides out to
    exactly `_PANEL_WIDTH`/`_PANEL_HEIGHT` unchanged (see
    `_MAX_FIGURE_WIDTH`'s comment for why), and beyond that the panel
    shrinks just enough to hold the total figure at the cap -- down to
    `_MIN_PANEL_SIZE`, past which a panel stops being legible and the total
    figure is allowed to exceed the cap rather than render unreadable
    panels.
    """
    n_rows = max(n_rows, 1)
    n_columns = max(n_columns, 1)
    panel_width = max(_MIN_PANEL_SIZE,
                      min(_PANEL_WIDTH, _MAX_FIGURE_WIDTH / n_columns))
    panel_height = max(_MIN_PANEL_SIZE,
                       min(_PANEL_HEIGHT, _MAX_FIGURE_HEIGHT / n_rows))
    return Figure(figsize=(panel_width * n_columns, panel_height * n_rows))


def _facet_k_values(k_series, context):
    """Which k values get their own panel column, and which are computed
    but not shown (D7): facet at ODD k only -- 1, 3, 5, ..., 17 -- matching
    the paper's presentation. Even-k rows are still computed and still feed
    every pooled statistic and paired test; they simply get no column here,
    and that omission is announced through `_log()` rather than left for
    the reader to notice on their own.

    Falls back to showing every k present when a frame is scoped entirely
    to even k (e.g. a partial campaign): rendering zero columns would be a
    worse failure than a figure that, on that one unusual input, shows an
    even k after all. `k=None` in the returned `shown` list means "no k
    column present at all -- do not filter by k".
    """
    values = (sorted({int(k) for k in k_series.dropna()})
             if k_series is not None else [])
    if not values:
        return [None], []
    shown = [k for k in values if k % 2 == 1]
    dropped = [k for k in values if k % 2 == 0]
    if not shown:
        shown, dropped = dropped, []
    if dropped:
        # The shown range is derived from `shown` itself (never a literal
        # "1..17"), so this stays honest if the k-grid ever widens beyond
        # today's real max of 17.
        _log('{}: {} computed but not shown (facet is odd k only, {}..{}) '
             '-- still pooled into every paired test and every pooled '
             'statistic.'.format(
                 context, ', '.join('k={}'.format(k) for k in dropped),
                 min(shown), max(shown)))
    return shown, dropped


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _markdown_cell(value, float_format):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return '' if pd.isna(value) else float_format.format(float(value))
    if value is None:
        return ''
    text = str(value)
    return text.replace('|', r'\|')


def _markdown_table(frame, float_format='{:.6g}'):
    """A GitHub-flavoured markdown table. Written here rather than through
    `DataFrame.to_markdown` because that requires `tabulate`, which is not
    installed in this environment and is not worth a new dependency for one
    table renderer."""
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(str(column) for column in columns) + ' |',
             '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(
            _markdown_M_cell(row[column]) if column == 'M'
            else _markdown_cell(row[column], float_format)
            for column in columns) + ' |')
    return '\n'.join(lines)


def _markdown_M_cell(value):
    """An `M` table cell through `format_M` ('25', the infinity sign)."""
    if value is None or (isinstance(value, (float, np.floating))
                         and np.isnan(value)):
        return ''
    try:
        return format_M(value)
    except (TypeError, ValueError):
        return _markdown_cell(value, '{:.6g}')


def _project_columns(table, columns):
    """Select `columns` from `table` in order, silently dropping any that
    are absent -- shared by every markdown-table renderer so a table missing
    an optional column (e.g. a partial campaign) still renders instead of
    raising a KeyError."""
    return table.loc[:, [column for column in columns if column in table.columns]]


def _write(deliverable, output_dir):
    """Write one deliverable's artifacts and return it with `paths` filled.

    `output_dir=None` writes nothing, so a caller (or a test) can build a
    figure and inspect it without touching the filesystem. Every deliverable
    gets a `.md` carrying its caption -- captions are part of the artifact,
    not decoration, and figure 2's in particular is a spec requirement.
    """
    if output_dir is None:
        return deliverable
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir,
                        '{:02d}_{}'.format(deliverable.number, deliverable.slug))
    paths = []

    if deliverable.figure is not None:
        pdf_path = stem + '.pdf'
        deliverable.figure.savefig(pdf_path, bbox_inches='tight')
        paths.append(pdf_path)

    if deliverable.data is not None:
        csv_path = stem + '.csv'
        deliverable.data.to_csv(csv_path, index=False)
        paths.append(csv_path)

    for suffix, frame in deliverable.extra_data:
        extra_path = '{}_{}.csv'.format(stem, suffix)
        frame.to_csv(extra_path, index=False)
        paths.append(extra_path)

    markdown_path = stem + '.md'
    sections = ['# Figure/Table {}. {}'.format(deliverable.number,
                                               deliverable.title),
                '', deliverable.caption]
    if deliverable.markdown_body:
        sections += ['', deliverable.markdown_body]
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(sections) + '\n')
    paths.append(markdown_path)

    return replace(deliverable, paths=tuple(paths))


# ---------------------------------------------------------------------------
# Deliverable 1 -- per-task accuracy vs blocks plus the trade plane (D8),
# all arms overlaid
# ---------------------------------------------------------------------------

def figure_1_accuracy_vs_blocks(df, output_dir=DEFAULT_FIGURE_DIR,
                                baseline=claims.INDEPENDENT_ARM_SLUG):
    """A grid -- the top two rows are the two tasks, a third row is D8's
    trade plane, and columns are odd feature counts k (D7: facet at
    k = 1, 3, 5, ..., 17) -- with every arm overlaid in every panel. Row and
    column counts are both derived (`len(TASKS) + 1` rows, `len(shown_k)`
    columns), never a literal panel count.

    Each arm contributes its own 3-D Pareto front, computed ONCE by
    `claims.pareto_front_3d` on `(acc_app, acc_ddos, -blocks)` pooling
    EVERY M, split AND k for that arm, and shown through
    `claims.pareto_projections`. The drawn line in the top two rows is a
    PROJECTION of that 3-D front, not a front recomputed inside the plane: a
    point can look dominated in one panel and still be non-dominated
    overall, and dropping it would hide exactly the App-versus-DDoS trade
    the thesis is about. Every cell is also scattered faintly behind the
    fronts, so the front is visibly a subset of the data rather than the
    only data shown. Faceting by k only slices WHICH points of that one
    pooled front and pooled cell set land in which column -- it does not
    recompute the front per k, and an even-k point still contributes to it
    even though it gets no column of its own (`_facet_k_values` logs which
    k those were).

    The third row is the trade plane (acc_app vs acc_ddos) that the top two
    rows cannot show directly: a point can look dominated in EVERY
    accuracy-vs-blocks panel while still sitting on the pooled 3-D front,
    because that front also weighs the OTHER task's accuracy. Colour there
    is `blocks` (a continuous gradient, not per-arm identity -- this row's
    whole point is the memory cost of a trade, not which arm made it), and
    an open ring marks front membership instead of a connecting line: unlike
    blocks on the x-axis above, acc_app and acc_ddos carry no ordering for a
    line to imply.

    There is no averaged accuracy anywhere in this figure, which is part of
    why it has one row per task rather than one panel overall.
    """
    arms = ordered_arms(df, baseline=baseline)
    styles = _arm_styles(arms)

    # Finding 1: `claims.hypervolume_2d` existed with no production caller.
    # Computed once here, per (arm, M, task) -- pooling every split and k at
    # that (arm, M), Pareto-filtered to a genuine 2-D front before
    # `hypervolume_2d` sees it (see `claims._pareto_front_2d`'s docstring for
    # why that filtering is not optional) -- and merged into `.data` below,
    # never recomputed: this module recomputes no `claims.py` statistic (see
    # the module docstring).
    hv_table = claims.hypervolume_by_arm(df, baseline=baseline, arms=arms)

    shown_k, _dropped_k = _facet_k_values(
        df['k'] if 'k' in df.columns else None, 'figure 1')

    # D8's trade plane: a THIRD row below the two task rows, same k columns
    # as the rest of the grid -- never a fixed literal panel count. Colour
    # there encodes `blocks` (a continuous cost gradient) rather than arm
    # identity, so it needs its own colormap/normalisation, built once here
    # over every block count in the frame so the gradient reads consistently
    # across every arm and every k column.
    trade_row = len(TASKS)
    trade_cmap = matplotlib.colormaps['viridis']
    if 'blocks' in df.columns and len(df):
        trade_norm = Normalize(vmin=float(df['blocks'].min()),
                               vmax=float(df['blocks'].max()))
    else:
        trade_norm = Normalize(vmin=0.0, vmax=1.0)

    figure = _make_figure(len(TASKS) + 1, len(shown_k))
    axes = figure.subplots(len(TASKS) + 1, len(shown_k), squeeze=False)

    front_frames = []
    coverage = {}
    baseline_front = None
    # Note: unlike figure_3 and figure_2, this function does NOT call
    # require_baseline. It overlays arms independently rather than pairing them,
    # so it stays meaningful without a baseline -- the figure shows the fronts
    # of whatever arms are present.
    if baseline in arms:
        baseline_front = claims.pareto_front_3d(df[df['arm_slug'] == baseline])

    for arm in arms:
        arm_rows = df[df['arm_slug'] == arm]
        arm_flagged = _flagged(arm_rows)
        front = claims.pareto_front_3d(arm_rows)
        projections = claims.pareto_projections(front)
        colour, marker = styles[arm]

        for row_index, (_, accuracy_column, _, _, _) in enumerate(TASKS):
            plane = projections['{}_vs_blocks'.format(accuracy_column)]
            for col_index, k in enumerate(shown_k):
                axis = axes[row_index][col_index]
                if k is None:
                    cell_rows, plane_k = arm_rows, plane
                else:
                    cell_rows = arm_rows[arm_rows['k'] == k]
                    plane_k = (plane[plane['k'] == k]
                              if 'k' in plane.columns else plane)
                # Split on `flagged` the way the trade row below splits on
                # front membership: an unverified cell (model numbers, not
                # p4c's) is drawn as FLAGGED_MARKER, never as an ordinary
                # cell.
                cell_flagged = arm_flagged.loc[cell_rows.index].to_numpy()
                plain_rows = cell_rows[~cell_flagged]
                flagged_rows = cell_rows[cell_flagged]
                axis.scatter(plain_rows['blocks'], plain_rows[accuracy_column],
                             s=10, alpha=0.25, color=colour, linewidths=0,
                             gid='cells:{}'.format(arm))
                if len(flagged_rows):
                    axis.scatter(flagged_rows['blocks'],
                                 flagged_rows[accuracy_column],
                                 s=36, color=colour, marker=FLAGGED_MARKER,
                                 linewidths=1.4,
                                 gid='flagged:{}'.format(arm))
                axis.plot(plane_k['blocks'], plane_k[accuracy_column],
                          marker=marker, color=colour, linewidth=1.6,
                          markersize=5, label=arm, gid='front:{}'.format(arm))

        # D8's trade plane: acc_app vs acc_ddos, coloured by blocks rather
        # than by arm (that's the whole point of this row -- the memory
        # cost of a trade-off, not which arm made it), with front
        # membership shown by marker style (open ring vs faint fill)
        # instead of a connecting line -- unlike the two rows above, there
        # is no natural ordering between acc_app and acc_ddos for a line to
        # imply. `front` (this arm's pooled 3-D front, computed once above)
        # decides ring membership per point via its preserved index; a
        # front point can still look "dominated" in this plane's 2-D
        # picture, which is exactly the case this panel exists to explain.
        front_index = set(front.index)
        for col_index, k in enumerate(shown_k):
            axis = axes[trade_row][col_index]
            cell_rows = arm_rows if k is None else arm_rows[arm_rows['k'] == k]
            on_front = cell_rows.index.isin(front_index)
            cell_flagged = arm_flagged.loc[cell_rows.index].to_numpy()
            # A flagged point keeps its ring when it sits on the front (front
            # membership is a fact about the numbers it carries), but is
            # never drawn as an ordinary faint cell: it gets FLAGGED_MARKER.
            off_rows = cell_rows[~on_front & ~cell_flagged]
            on_rows = cell_rows[on_front]
            flagged_rows = cell_rows[cell_flagged]
            if len(flagged_rows):
                axis.scatter(flagged_rows['acc_app'], flagged_rows['acc_ddos'],
                             c=flagged_rows['blocks'], cmap=trade_cmap,
                             norm=trade_norm, s=40, marker=FLAGGED_MARKER,
                             linewidths=1.4, gid='flagged:{}'.format(arm))
            if len(off_rows):
                axis.scatter(off_rows['acc_app'], off_rows['acc_ddos'],
                             c=off_rows['blocks'], cmap=trade_cmap,
                             norm=trade_norm, s=16, alpha=0.35, marker='o',
                             linewidths=0, gid='trade-cells:{}'.format(arm))
            if len(on_rows):
                ring_colours = trade_cmap(trade_norm(
                    on_rows['blocks'].to_numpy()))
                axis.scatter(on_rows['acc_app'], on_rows['acc_ddos'],
                             facecolors='none', edgecolors=ring_colours,
                             s=70, marker='o', linewidths=1.8,
                             gid='trade-front:{}'.format(arm))

        # 'stages' here is the MODEL's occupied match-table stage count
        # (campaign_data._FLOAT_COLUMNS) -- not `stage_depth` (pipeline
        # depth, what the 12-stage ceiling reads) and not `stages_real` (the
        # real compiler's whole-program count). Never plot/compare 'stages'
        # against 'stages_real' as if they measured the same thing -- see
        # evaluation.multi_model_memory_evaluation's ResourceUsage docstring for the full
        # three-quantity (stages, stage_depth, stages_real) disambiguation.
        # F1 rides along in the CSV only (D4): it is a tested metric, not a
        # Pareto axis, so it must reach `.data` without the front itself
        # -- computed above on `claims.FRONT_OBJECTIVES`, still 3-D -- ever
        # seeing it. Do not add an F1 panel to this figure; that would imply
        # a front this code does not compute.
        carried = [column for column in
                   ('arm_slug', 'M', 'split', 'k', 'blocks', 'stages',
                    'acc_app', 'acc_ddos', 'f1_app', 'f1_ddos')
                   if column in front.columns]
        arm_frame = front.loc[:, carried].copy()

        # Zitzler's C metric is asymmetric: C(A, B) != C(B, A). Reporting
        # only "how much of the baseline this arm covers" (as a previous
        # refactor, P7d, left it) hides the other, equally load-bearing
        # direction -- "how much of this arm the baseline covers" -- which
        # is exactly the number that shows a joint arm's front is not
        # merely competitive but strictly dominates the baseline's (see the
        # caption below). Both are computed and both land in `.data`, not
        # just the caption string, so a reader of the CSV alone still sees
        # the asymmetry.
        if baseline_front is not None and arm != baseline:
            coverage_of_baseline = claims.coverage_ratio_3d(front, baseline_front)
            coverage_by_baseline = claims.coverage_ratio_3d(baseline_front, front)
            coverage[arm] = (coverage_of_baseline, coverage_by_baseline)
            arm_frame['coverage_of_baseline'] = coverage_of_baseline
            arm_frame['coverage_by_baseline'] = coverage_by_baseline
        else:
            # The baseline's own rows (and any arm when there is no
            # baseline front to compare against): coverage of/by itself is
            # not a meaningful quantity (`coverage_ratio_3d(A, A) == 0` by
            # construction, which would misleadingly read as "the baseline
            # covers none of itself"), so leave it undefined rather than
            # print a number nobody should compare.
            arm_frame['coverage_of_baseline'] = float('nan')
            arm_frame['coverage_by_baseline'] = float('nan')
        front_frames.append(arm_frame)

    for row_index, (_, _, short_name, panel_title, _) in enumerate(TASKS):
        for col_index, k in enumerate(shown_k):
            axis = axes[row_index][col_index]
            axis.set_xlabel('TCAM blocks')
            axis.set_ylabel('{} accuracy'.format(short_name))
            axis.set_title(panel_title if k is None
                           else '{} (k={})'.format(panel_title, k))
            axis.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize='small', title='arm')

    for col_index, k in enumerate(shown_k):
        axis = axes[trade_row][col_index]
        axis.set_xlabel('App accuracy (acc_app)')
        axis.set_ylabel('DDoS accuracy (acc_ddos)')
        axis.set_title('Trade plane: acc_ddos vs acc_app' if k is None
                       else 'Trade plane: acc_ddos vs acc_app (k={})'.format(k))
        axis.grid(True, alpha=0.3)
    figure.tight_layout()

    coverage_sentence = ''
    if baseline not in arms:
        coverage_sentence = (
            ' NOTE: baseline arm {!r} is not present in this frame, so no '
            'coverage ratios are computed.'.format(baseline))
    elif coverage:
        coverage_sentence = (
            ' Coverage ratio (Zitzler C, 3-D, strict) is asymmetric -- '
            'C(A, B) != C(B, A) -- so both directions are reported for '
            'each joint arm against the {} baseline: {}. A high forward '
            'value paired with a near-zero reverse value (e.g. covers 83% '
            'of the baseline while being covered by 0% of it) is a much '
            'stronger claim than either number alone -- it means the arm\'s '
            'front strictly dominates the baseline\'s, not merely that it '
            'is competitive.'.format(
                baseline,
                ', '.join(
                    '{} covers {:.0%} of {} and is covered by {:.0%} of it'
                    .format(arm, of_baseline, baseline, by_baseline)
                    for arm, (of_baseline, by_baseline) in coverage.items())))

    facet_sentence = ''
    if _dropped_k:
        facet_sentence = (
            ' EVEN k IS STILL COMPUTED -- {} -- and is pooled into every '
            'front, coverage ratio and paired statistic exactly like every '
            'odd k below; it is simply not drawn as its own column (see '
            'the run log for the full list of k omitted this way).'.format(
                ', '.join('k={}'.format(k) for k in _dropped_k)))

    # Finding 1: report hypervolume as GAIN relative to the baseline arm
    # (D5, amended by A2), pooled over every M and task into one summary
    # number per arm so the caption stays readable -- the per-(arm, M, task)
    # numbers it is pooled from are the `hypervolume_gain_<task>` columns
    # merged into `.data` above, so any reader can unpool it.
    hypervolume_sentence = ''
    if len(hv_table):
        gain_by_arm = {}
        for arm in arms:
            if arm == baseline:
                continue
            arm_gains = hv_table.loc[hv_table['arm_slug'] == arm,
                                     'hypervolume_gain'].dropna()
            if len(arm_gains):
                gain_by_arm[arm] = float(arm_gains.mean())
        if gain_by_arm:
            hypervolume_sentence = (
                ' Hypervolume (D5, amended by A2) is computed PER TASK on '
                '(acc_<task>, blocks) pairs -- never on the two tasks\' '
                'accuracies averaged together -- from each arm\'s 2-D '
                'Pareto front at each block budget M, against a reference '
                'point (0.5, M) THAT TRACKS M rather than the published '
                'Fig. results_2a\'s fixed (0.5, 100); this diverges from '
                'the published numbers on purpose (a fixed 100 would '
                'silently discard most of the front once M sweeps past '
                'it) and the divergence is disclosed here rather than '
                'hidden. Reported as gain relative to the {} baseline, '
                'pooled over every M and task (`hypervolume_gain_app` / '
                '`hypervolume_gain_ddos` in the data carry the '
                'unpooled per-(M, task) numbers): {}.'.format(
                    baseline,
                    ', '.join(
                        '{} averages {:.2f}x the baseline\'s hypervolume'
                        .format(arm, gain)
                        for arm, gain in gain_by_arm.items())))

    caption = (
        'Per-task accuracy against TCAM blocks: the top two rows are the '
        'two tasks, never averaged, and columns are odd feature counts k '
        '(1, 3, 5, ..., 17), with every arm of the design overlaid in every '
        'panel. A single mean accuracy hides a model that is excellent on '
        'one task and unusable on the other, and pooling every k into one '
        'panel hides how the trade-off moves as k grows.{} '
        'Faint points in a panel are that panel\'s (M, split, k) cells; '
        'the joined markers are the same k-slice of each arm\'s Pareto '
        'front, computed ONCE in 3-D on (acc_app, acc_ddos, -blocks) over '
        'EVERY M, split AND k for that arm and PROJECTED into each panel '
        '-- a projected point may look dominated within its panel while '
        'being non-dominated overall, and removing it would hide the very '
        'trade between the two tasks this figure exists to show. EACH '
        'FRONT POOLS EVERY SPLIT AND EVERY BLOCK BUDGET M for that arm '
        'into one 3-D Pareto computation, so a point from one split can '
        'dominate a point from another and the front is not a front of '
        'anything replicated; the coverage figure below is therefore over '
        'that pooled surface, not a per-split or per-k comparison.{} '
        'The THIRD row is the trade plane itself, acc_app against acc_ddos, '
        'one panel per shown k: colour is TCAM blocks (a continuous cost '
        'gradient, not arm identity), and an open ring marks a point that '
        'sits on that arm\'s pooled 3-D Pareto front -- the only place this '
        'figure can show WHY a front point exists even when it looks '
        'dominated in both accuracy-vs-blocks rows above. It carries no '
        'connecting line: unlike blocks on the x-axis of the rows above, '
        'acc_app and acc_ddos have no ordering between them for a line to '
        'imply.{}{}'.format(facet_sentence, coverage_sentence,
                            hypervolume_sentence,
                            _flagged_sentence(int(_flagged(df).sum()), len(df))))

    data = (pd.concat(front_frames, ignore_index=True)
            if front_frames else pd.DataFrame())

    # Merge the per-(arm, M, task) hypervolume rows onto `.data`, pivoted
    # wide by task (`hypervolume_app` / `hypervolume_ddos` etc.) so each
    # merges cleanly onto the one-row-per-(arm, M, front-point) shape `data`
    # already has -- a reader can look up any front point's (arm_slug, M)
    # and see the hypervolume numbers the caption's pooled sentence above
    # was computed from.
    if len(data) and len(hv_table):
        hv_pieces = []
        for task, task_rows in hv_table.groupby('task'):
            piece = task_rows[['arm_slug', 'M', 'hypervolume',
                               'baseline_hypervolume', 'hypervolume_gain']]
            piece = piece.rename(columns={
                'hypervolume': 'hypervolume_{}'.format(task),
                'baseline_hypervolume': 'baseline_hypervolume_{}'.format(task),
                'hypervolume_gain': 'hypervolume_gain_{}'.format(task),
            })
            hv_pieces.append(piece.set_index(['arm_slug', 'M']))
        hv_wide = pd.concat(hv_pieces, axis=1).reset_index()
        data = data.merge(hv_wide, on=['arm_slug', 'M'], how='left')

    return _write(Deliverable(
        number=1, slug='accuracy_vs_blocks_per_task',
        title='Per-task accuracy against TCAM blocks, all arms',
        caption=caption, data=data, figure=figure), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 2 -- the delta frontier
# ---------------------------------------------------------------------------

def _relative_error_change(baseline_accuracy, treatment_accuracy):
    """`(e_treatment - e_baseline) / e_baseline`, the difficulty-normalised
    scale the spec reports (section 9's measurement log).

    Accuracy points are not comparable across the two tasks: App errs around
    0.24 and DDoS around 0.04, so the same 0.005 accuracy drop is a 2%
    relative degradation on one and a 12% one on the other. Positive means
    the treatment arm made MORE errors.

    A baseline cell with perfect accuracy has no relative scale (division by
    a zero error), and yields NaN rather than an infinity that would
    dominate any mean built on it.
    """
    baseline_error = 1.0 - baseline_accuracy
    change = (baseline_accuracy - treatment_accuracy) / baseline_error
    return change.where(baseline_error > 0, np.nan)


def paired_delta_frame(df, baseline=claims.INDEPENDENT_ARM_SLUG, arms=None):
    """One row per (arm, M, split, k) cell paired against `baseline`, with
    the block delta and the two per-task relative-error changes on BOTH
    accuracy and F1.

    `d_blocks` comes from `claims.arm_deltas` -- the module that owns paired
    differences and the `(M, split, k)` join key -- and the relative-error
    columns are computed here from the same `pair_arms` join, because a
    ratio is not a difference and `claims.py` does not compute it. The F1
    column per task is named by string convention from the task's `key`
    (`'f1_' + key`, matching the campaign CSV's `acc_<key>` / `f1_<key>`
    naming), not a new `TASKS` field. The two are merged back on the join
    key with `validate='one_to_one'`, so a duplicated cell fails loudly
    instead of silently multiplying rows.
    """
    require_baseline(df, baseline, 'paired_delta_frame')
    if arms is None:
        arms = ordered_arms(df, include_baseline=False, baseline=baseline)

    frames = []
    for arm in arms:
        deltas = claims.arm_deltas(df, arm, baseline, metrics=('blocks',))
        paired = pair_arms(df, arm, baseline)
        if len(paired) == 0:
            continue
        relative = pd.DataFrame({
            'M': paired['M'], 'split': paired['split'], 'k': paired['k'],
        })
        for key, accuracy_column, _, _, _ in TASKS:
            relative[REL_ERROR_PREFIX + key] = _relative_error_change(
                paired['{}_baseline'.format(accuracy_column)],
                paired['{}_treatment'.format(accuracy_column)])
            f1_column = 'f1_' + key
            relative[F1_REL_ERROR_PREFIX + key] = _relative_error_change(
                paired['{}_baseline'.format(f1_column)],
                paired['{}_treatment'.format(f1_column)])
        merged = deltas.merge(relative, on=['M', 'split', 'k'], how='inner',
                              validate='one_to_one')
        merged.insert(0, 'arm_slug', arm)
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=['arm_slug', 'M', 'split', 'k']
                            + list(FRONTIER_METRICS))
    long = pd.concat(frames, ignore_index=True)
    return claims.attach_delta_columns(long, df)


def delta_frontier_table(df, baseline=claims.INDEPENDENT_ARM_SLUG,
                         confidence=0.95, arms=None):
    """Mean and confidence interval per (arm, k) for each of figure 2's
    five quantities (`FRONTIER_METRICS`: accuracy and F1 relative-error
    change per task, plus the block delta), aggregated over SPLITS.

    Each (arm, k, split) is collapsed to its own mean difference first, so
    the interval `claims.delta_frontier` then builds has exactly one
    observation per split within each (arm, k) group. Feeding it the raw
    cells instead would put many correlated cells from one training split
    into the same interval and make it too narrow -- which is why
    `delta_frontier` refuses that shape outright unless the caller says it
    means it. k is kept as a real grouping column, not averaged away
    alongside M: the paper's conclusion is k-dependent (joint dominates at
    k>=11, parity at 5-9, independent wins at k<=5 -- main.tex:591), and a
    single mean over k could not reproduce that headline. M IS still
    averaged away here -- there is no per-M facet, only per-k -- matching
    the split-level convention `claims.ablation_decomposition` uses.
    """
    long = paired_delta_frame(df, baseline=baseline, arms=arms)
    if len(long) == 0:
        return long

    group_columns = ['arm_slug', 'k', 'split']
    split_means = long.groupby(group_columns, as_index=False)[
        list(FRONTIER_METRICS)].mean()

    split_means = claims.attach_delta_columns(split_means, long)

    return claims.delta_frontier(
        split_means, metrics=FRONTIER_METRICS,
        group_columns=('arm_slug', 'k'), confidence=confidence)


def figure_2_delta_frontier(df, output_dir=DEFAULT_FIGURE_DIR,
                            baseline=claims.INDEPENDENT_ARM_SLUG,
                            confidence=0.95):
    """What each joint arm buys against the baseline: block saving and
    per-task relative error change (on accuracy AND F1) per arm, mean +/- CI
    across splits. (The slug and function name keep "delta" from the retired
    tolerance sweep; the x axis is now the joint arms.)

    A grid of panels: rows are the five reported quantities (the two tasks'
    accuracy relative-error change, the two tasks' F1 relative-error change,
    plus the block delta -- `FRONTIER_METRICS`), because pooling the two
    tasks would reintroduce the defect this rerun exists to fix, and columns
    are odd feature counts k (D7), because the paper's conclusion is
    k-dependent and a single k-pooled point cannot reproduce it. F1 is
    reported here because it is a tested metric wherever accuracy is
    (D2 admits F1 for exactly this reason -- a minority-class collapse is
    what accuracy alone hides). The x axis within each panel is categorical
    in `ordered_arms` order rather than numeric: `joint-off` (alignment
    never ran) and `joint` (aligned) are arms, not numbers, and placing them
    on a numeric axis would require inventing coordinates for them. An
    archived frame's delta arms are appended after them, labelled by their
    delta (`joint-dinf` as 'inf').
    """
    table = delta_frontier_table(df, baseline=baseline, confidence=confidence)
    arms = [arm for arm in ordered_arms(df, include_baseline=False,
                                        baseline=baseline)
            if arm in set(table['arm_slug'])] if len(table) else []
    positions = {arm: index for index, arm in enumerate(arms)}

    labels = {REL_ERROR_PREFIX + key: '{}: rel. error change vs {}'.format(
        short_name, baseline) for key, _, short_name, _, _ in TASKS}
    labels.update({
        F1_REL_ERROR_PREFIX + key: '{}: F1 rel. error change vs {}'.format(
            short_name, baseline) for key, _, short_name, _, _ in TASKS})
    labels[BLOCKS_DELTA] = 'TCAM blocks: change vs {}'.format(baseline)

    # One tick label per arm, built once from the parsed delta columns
    # `claims.delta_frontier` carried through -- never from the raw
    # `delta_align` string, which must not be ordered or compared.
    # `arms` is empty whenever `table` is, so these lookups only ever run on
    # a populated table.
    per_arm = table.drop_duplicates('arm_slug').set_index('arm_slug') if len(table) else table
    tick_labels = [
        _delta_tick_label(
            arm,
            per_arm['delta_align_num'].get(arm, np.nan)
            if 'delta_align_num' in per_arm.columns else np.nan,
            per_arm['delta_align_is_inf'].get(arm, False)
            if 'delta_align_is_inf' in per_arm.columns else False)
        for arm in arms]
    x = [positions[arm] for arm in arms]

    shown_k, _dropped_k = _facet_k_values(
        table['k'] if len(table) and 'k' in table.columns else None,
        'figure 2')

    figure = _make_figure(len(FRONTIER_METRICS), len(shown_k))
    axes = figure.subplots(len(FRONTIER_METRICS), len(shown_k), squeeze=False)

    for row_index, metric in enumerate(FRONTIER_METRICS):
        metric_rows = table[table['metric'] == metric] if len(table) else table
        for col_index, k in enumerate(shown_k):
            axis = axes[row_index][col_index]
            rows = metric_rows
            if k is not None and len(rows) and 'k' in rows.columns:
                rows = rows[rows['k'] == k]
            rows = rows.set_index('arm_slug').reindex(arms) if len(rows) else rows
            if len(rows):
                means = rows['mean'].to_numpy(dtype='float64')
                lower = means - rows['ci_low'].to_numpy(dtype='float64')
                upper = rows['ci_high'].to_numpy(dtype='float64') - means
                axis.errorbar(x, means, yerr=np.vstack([lower, upper]),
                              marker='o', capsize=4, linewidth=1.6,
                              gid='frontier:{}'.format(metric))
            axis.axhline(0.0, color='0.4', linewidth=1.0, linestyle=':')
            axis.set_xticks(x)
            axis.set_xticklabels(tick_labels)
            axis.set_xlabel('joint arm')
            axis.set_ylabel(labels[metric])
            if k is not None:
                axis.set_title('k={}'.format(k))
            axis.grid(True, alpha=0.3)
    figure.tight_layout()

    # Derived from the JOINED frame (`pair_arms`' inner join on
    # (M, split, k), via `paired_delta_frame`), not from `df` directly.
    # `--M` and `--n-splits` let the campaign be chunked and resumed
    # (main.py's `skip_existing`), so different arms can end up with
    # different M grids on disk; deriving from raw `df` would report every
    # M/k present anywhere in the file even when the join dropped some of
    # them for the arms actually plotted here, which is exactly the average
    # this sentence claims to describe.
    joined = paired_delta_frame(df, baseline=baseline)
    pooled_m = (sorted(joined['M'].unique().tolist())
               if len(joined) and 'M' in joined.columns else [])
    pooled_k = (sorted(int(value) for value in joined['k'].unique())
               if len(joined) and 'k' in joined.columns else [])
    pooling_sentence = (
        'EACH POINT POOLS ACROSS BLOCK BUDGETS, not one operating point: '
        'cell differences are paired on (M, split, k), averaged within '
        'each split across every block budget M ({}), and the interval is '
        'then taken across those per-split means. A block change read off '
        'this figure is therefore an average over the M budget grid at a '
        'fixed k, and can hide a saving that is much larger at one budget '
        'than another; Figure 1 shows the per-budget spread that this '
        'averages over. FEATURE COUNT k IS NOT POOLED HERE -- each column '
        'is one value of k ({}), shown separately, because the paper\'s '
        'conclusion is k-dependent (main.tex:591) and a k-pooled mean '
        'cannot reproduce it. '.format(
            ', '.join(format_M(value) for value in pooled_m) or 'none present',
            ', '.join(str(value) for value in pooled_k) or 'none present'))

    caption = (
        'The joint arms against the baseline: block change and per-task '
        'relative error change, on BOTH accuracy and F1, one point per joint '
        'arm, each a mean over splits with a {:.0%} Student-t confidence '
        'interval, paired against the {} arm on (M, split, k). {}The two '
        'tasks are '
        'shown on separate panels for each metric and are never averaged; '
        'relative error ((e_arm - e_base) / e_base) is reported because '
        'the tasks have very different error scales, so equal accuracy (or '
        'F1) losses are not equal degradations. F1 is shown alongside '
        'accuracy because a minority-class collapse is exactly what '
        'accuracy alone can hide. The arms carry no numeric position and are '
        'labelled as themselves: "off" (joint-off) shares one encoding but never '
        'ran threshold alignment, and "off-al" (joint-off-al) is sharing plus '
        'post-selection threshold alignment. '
        'THE FEATURE SETS DIFFER ACROSS ARMS BY CONSTRUCTION -- sharing and '
        'alignment change which thresholds, and hence which intervals and '
        'which eliminated features, each arm ends up with, so the arms are not '
        'evaluated on identical inputs. Split-level replication is what '
        'controls the resulting variance: each interval is built over '
        'per-split mean differences, one observation per split, so the '
        'spread of feature sets across splits is inside the interval rather '
        'than being assumed away.'.format(confidence, baseline,
                                          pooling_sentence))

    return _write(Deliverable(
        number=2, slug='delta_frontier',
        title='Joint-arm frontier: block and per-task relative-error change',
        caption=caption, data=table, figure=figure), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 3 -- substitution scatter with quadrants
# ---------------------------------------------------------------------------

_QUADRANT_ANCHORS = {
    'quadrant_both_up': (0.97, 0.97, 'right', 'top', 'both up'),
    'quadrant_app_down_ddos_up': (0.03, 0.97, 'left', 'top',
                                  'App down / DDoS up'),
    'quadrant_app_up_ddos_down': (0.97, 0.03, 'right', 'bottom',
                                  'App up / DDoS down'),
    'quadrant_both_down': (0.03, 0.03, 'left', 'bottom', 'both down'),
}


def figure_3_substitution_scatter(df, output_dir=DEFAULT_FIGURE_DIR,
                                  baseline=claims.INDEPENDENT_ARM_SLUG,
                                  alpha=0.05, expected_family_size=None):
    """A grid -- the top row is accuracy, a second row is F1 (Task 19 Part
    5), and columns are joint arms -- the paired per-task deltas against
    each other, with the sign quadrants annotated on the accuracy row.

    This answers the reviewer's objection directly. Substitution -- one task
    paying for the other's gain -- is a NEGATIVE correlation between the two
    deltas, and the mass in the two off-diagonal quadrants is what it looks
    like. Every number annotated on the ACCURACY row comes from
    `claims.substitution_test_all_arms`: the Pearson r, the partial r
    controlling for the block delta (two accuracy deltas can correlate
    purely because both track how much TCAM the cell was allowed), the
    Holm-corrected one-sided p across the joint arms
    (`claims.SUBSTITUTION_FAMILY_SIZE`, one test per arm in
    `claims.JOINT_ARM_SLUGS`), and the quadrant fractions.

    The F1 row is DESCRIPTIVE ONLY -- the same question (does one task's
    gain come at the other's expense?) is live for F1 too, since a
    minority-class collapse is exactly what accuracy alone can hide (R3
    §IV(d), the reason D2 admits F1 at all) -- but it adds nothing to either
    Holm family: `claims.SUBSTITUTION_FAMILY_SIZE` stays one test per joint
    arm (the accuracy test count), never doubled, and there is no parallel
    `claims.substitution_test_all_arms`-style correlation test run on F1.
    The tested F1 comparisons live in deliverable 4
    (`claims.paired_tests`' superiority family already includes `f1_app` /
    `f1_ddos`); this row is a visual aid only, built from
    `claims.arm_deltas(df, arm, baseline)` -- which already carries
    `d_f1_app` / `d_f1_ddos` via `claims.DEFAULT_METRICS` -- with no new
    statistic computed here.

    The test runs at every joint arm, so the claim defended is "no task
    sacrifices itself under either joint arm" rather than "at one operating
    point". A pair with either side `flagged` (unverified, model numbers) is
    drawn with `FLAGGED_MARKER` and `gid='flagged:{arm}'` on both rows; the
    statistics above still include it, exactly as `claims.py` computes them.

    `expected_family_size` defaults to None so a partial campaign (the pilot
    cell) still renders; pass `claims.SUBSTITUTION_FAMILY_SIZE` to turn a
    family shrunk by a missing arm into an error, mirroring deliverable 4's
    two Holm-family gates -- see `claims.substitution_test_all_arms`.
    """
    require_baseline(df, baseline, 'figure_3_substitution_scatter')
    table = claims.substitution_test_all_arms(
        df, baseline=baseline, alpha=alpha,
        expected_family_size=expected_family_size)
    arms = list(table['treatment']) if len(table) else []
    accuracy_row, f1_row = 0, 1
    n_columns = max(len(arms), 1)

    figure = _make_figure(2, n_columns)
    axes = figure.subplots(2, n_columns, squeeze=False)
    if not arms:
        for axis in axes.ravel():
            figure.delaxes(axis)

    n_pairs_total, n_pairs_flagged = 0, 0
    for col_index, arm in enumerate(arms):
        record = table[table['treatment'] == arm].iloc[0]
        deltas = claims.arm_deltas(df, arm, baseline)
        # `arm_deltas` is built row-for-row from `pair_arms` (same join,
        # index reset), so the pair's flags line up by position: a pair is
        # flagged when EITHER side carries the model's numbers.
        paired = pair_arms(df, arm, baseline)
        pair_flagged = (_flagged(paired, 'flagged_treatment')
                        | _flagged(paired, 'flagged_baseline')).to_numpy()
        plain = deltas[~pair_flagged]
        marked = deltas[pair_flagged]
        n_pairs_total += len(deltas)
        n_pairs_flagged += len(marked)

        axis = axes[accuracy_row][col_index]
        axis.scatter(plain['d_acc_app'], plain['d_acc_ddos'],
                     s=14, alpha=0.55, linewidths=0,
                     gid='substitution:{}'.format(arm))
        if len(marked):
            axis.scatter(marked['d_acc_app'], marked['d_acc_ddos'],
                         s=30, color='C3', marker=FLAGGED_MARKER,
                         linewidths=1.2, gid='flagged:{}'.format(arm))
        axis.axhline(0.0, color='0.3', linewidth=1.0)
        axis.axvline(0.0, color='0.3', linewidth=1.0)
        for column, (x, y, ha, va, name) in _QUADRANT_ANCHORS.items():
            axis.text(x, y, '{} {:.2f}'.format(name, record[column]),
                      transform=axis.transAxes, fontsize='small',
                      horizontalalignment=ha, verticalalignment=va)
        axis.set_title(
            '{}\nr = {:.3f}, partial r = {:.3f}, Holm p = {:.3g}'.format(
                arm, record['pearson_r'], record['partial_pearson_r'],
                record['pearson_p_negative_one_sided_holm']),
            fontsize='medium')
        axis.set_xlabel('delta App accuracy')
        axis.set_ylabel('delta DDoS accuracy')
        axis.grid(True, alpha=0.3)

        # F1 row: purely descriptive, same scatter shape, no test statistics
        # (none are computed for F1 here -- see the docstring). Quadrant
        # zero lines are kept for visual continuity with the row above.
        f1_axis = axes[f1_row][col_index]
        f1_axis.scatter(plain['d_f1_app'], plain['d_f1_ddos'],
                        s=14, alpha=0.55, linewidths=0,
                        gid='substitution-f1:{}'.format(arm))
        if len(marked):
            f1_axis.scatter(marked['d_f1_app'], marked['d_f1_ddos'],
                            s=30, color='C3', marker=FLAGGED_MARKER,
                            linewidths=1.2, gid='flagged:{}'.format(arm))
        f1_axis.axhline(0.0, color='0.3', linewidth=1.0)
        f1_axis.axvline(0.0, color='0.3', linewidth=1.0)
        f1_axis.set_title('{} (F1, descriptive)'.format(arm), fontsize='medium')
        f1_axis.set_xlabel('delta App F1')
        f1_axis.set_ylabel('delta DDoS F1')
        f1_axis.grid(True, alpha=0.3)
    figure.tight_layout()

    detected = (list(table.loc[table['substitution_detected_holm'], 'treatment'])
                if len(table) else [])
    caption = (
        'Paired per-task deltas against the {} arm, one column per joint '
        'arm: the top row is accuracy, the second is F1. Substitution -- '
        'one task gaining at the other\'s expense -- is a negative '
        'correlation, i.e. mass in the two off-diagonal quadrants ("App '
        'down / DDoS up" and "App up / DDoS down"); cells where either task '
        'did not move at all are counted separately and are in none of the '
        'four. The ACCURACY row reports the Pearson r, the partial r '
        'controlling for the block delta (two accuracy deltas can move '
        'together simply because both track the cell\'s block budget), the '
        'one-sided p for rho < 0 after Holm-Bonferroni correction across '
        'the {} arms tested, and the quadrant fractions. Arms where '
        'substitution is detected at alpha = {:g} after correction: {}. The '
        'test is run at every joint arm, so the claim is about the whole '
        'design and not one operating point. Cells within a split share a '
        'training split, so these p-values are anti-conservative relative to '
        'the number of independent splits. The F1 row is DESCRIPTIVE ONLY -- '
        'the same substitution question is live for F1 (a minority-class '
        'collapse is exactly what accuracy alone can hide), but no '
        'correlation test is run on it here and it adds nothing to either '
        'Holm family; the tested F1 comparisons are in Table 4.{}'.format(
            baseline, len(arms), alpha,
            ', '.join(detected) if detected else 'none',
            _flagged_sentence(n_pairs_flagged, n_pairs_total,
                              'paired cells (either side flagged)')))

    return _write(Deliverable(
        number=3, slug='substitution_scatter',
        title='Substitution: per-task accuracy deltas against each other',
        caption=caption, data=table, figure=figure), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 4 -- the paired per-task test table
# ---------------------------------------------------------------------------

_PAIRED_TEST_MARKDOWN_COLUMNS = (
    'family', 'subset', 'unit', 'contrast', 'metric', 'alternative', 'n_pairs',
    'n_splits', 'median_diff', 'mean_diff_split_level', 'ci_low', 'ci_high',
    'p_value', 'p_holm', 'significant_holm')


def table_4_paired_tests(df, output_dir=DEFAULT_FIGURE_DIR,
                         baseline=claims.INDEPENDENT_ARM_SLUG,
                         margin=0.0, alpha=0.05, units=('pair', 'split'),
                         expected_family_size=None,
                         expected_noninferiority_family_size=None):
    """The pre-registered paired tests, Holm-corrected -- rendered, not
    recomputed. Every number is `claims.paired_tests_robustness`' (whose
    `'all'` subset IS `claims.paired_tests`) or
    `claims.noninferiority_tests`'.

    Two INDEPENDENT test families sit in this one table, discriminated by
    the `family` column, plus the superiority family's robustness lines:

    * `family='superiority'` -- `claims.paired_tests`' pre-registered
      family (`claims.PRE_REGISTERED_FAMILY_SIZE`, one test per metric in
      `claims.DEFAULT_METRICS` for each arm in `claims.JOINT_ARM_SLUGS`):
      "no detectable loss" (or non-inferiority within `margin` when `margin
      > 0`) on `acc_app`, `f1_app`, `acc_ddos`, `f1_ddos`, plus a two-sided
      test on `blocks`, for each joint arm against `baseline`.
    * `family='robustness'` -- directly under it, the same tests re-run by
      `claims.paired_tests_robustness` on its extra subsets, labelled by the
      `subset` column: `no_flagged` (without unverified rows; only when any
      row is flagged) and `heldout_splits` (without the development splits).
      Checks on the headline, not families of their own: each is
      Holm-corrected over what it ran and none is held to the family size.
      A subset with no rows is skipped, and the markdown says so.
    * `family='noninferiority'` -- `claims.noninferiority_tests`' D13 family
      (`claims.NONINFERIORITY_FAMILY_SIZE`): non-inferiority of each
      joint arm to `baseline` on `acc_app`/`acc_ddos` ONLY, at a margin sized
      per row as a FRACTION of that row's own baseline error (see that
      function's docstring for why this cannot reuse `paired_tests(...,
      margin=...)`). D13 requires this reported ALONGSIDE the superiority
      family above, never in its place, so both land in this one table
      rather than the non-inferiority family living only in a separate
      artifact a reader could miss.

    One row per (family, unit, contrast, metric), and `acc_app` and
    `acc_ddos` are separate rows throughout: there is no pooled accuracy
    test, because a pooled test is exactly what let a loss on one task hide
    behind a gain on the other.

    Ruling P7-3: `unit='pair'` -- one difference per `(M, split, k)` cell --
    is the spec-mandated primary, but those cells are not independent (the
    same split recurs across every M and k), so its p-values are
    anti-conservative. `unit='split'` -- one mean difference per split -- is
    the statistically clean check: valid under split-level replication, far
    less powerful. The ruling requires BOTH be visible wherever the primary
    appears, not the primary alone with the split-level number folded into a
    confidence interval elsewhere. So this table stacks both, for EACH
    family: `units` is called through `claims.paired_tests` and through
    `claims.noninferiority_tests` once per unit, each call independently
    Holm-corrected over its own family (mixing units, or mixing families,
    into one Holm pass would correct p-values from different questions
    against each other, which is not what either correction means), and the
    results are concatenated with the `family` and `unit` columns
    identifying which is which. `claims.noninferiority_tests` documents
    that it returns "the same column set `paired_tests` emits" specifically
    so this concatenation is a plain `pd.concat`, never a column remap.

    `expected_family_size` gates the superiority family and
    `expected_noninferiority_family_size` gates the non-inferiority family;
    both default to None so a partial campaign (the pilot cell) still
    produces a table. That is a real weakening -- Holm over fewer
    comparisons is a laxer correction than Holm over the pre-registered
    size -- so the rendered markdown always states how many comparisons
    were actually corrected over, for EACH family, and what the
    pre-registered family sizes are. Pass
    `expected_family_size=claims.PRE_REGISTERED_FAMILY_SIZE` and
    `expected_noninferiority_family_size=claims.NONINFERIORITY_FAMILY_SIZE`
    on the complete campaign to turn either shrunken family into an error.
    """
    superiority_tables, robustness_tables = [], []
    for unit in units:
        stacked = claims.paired_tests_robustness(
            df, baseline=baseline, metrics=claims.DEFAULT_METRICS,
            margin=margin, alpha=alpha, unit=unit,
            expected_family_size=expected_family_size)
        main = stacked[stacked['subset'] == 'all'].reset_index(drop=True)
        extra = stacked[stacked['subset'] != 'all'].reset_index(drop=True)
        main.insert(0, 'family', 'superiority')
        extra.insert(0, 'family', 'robustness')
        superiority_tables.append(main)
        robustness_tables.append(extra)
    noninferiority_tables = [
        claims.noninferiority_tests(
            df, baseline=baseline, alpha=alpha, unit=unit,
            expected_family_size=expected_noninferiority_family_size)
        for unit in units
    ]
    for one_table in noninferiority_tables:
        one_table.insert(0, 'family', 'noninferiority')
        one_table.insert(1, 'subset', 'all')
    # The robustness lines sit directly UNDER the main family they check.
    table = pd.concat(superiority_tables + robustness_tables
                      + noninferiority_tables, ignore_index=True)

    # Which robustness subsets `paired_tests_robustness` would run on this
    # frame, and which it skipped for having no rows -- said on the face of
    # the table, so a missing line reads as "empty", not as "forgotten".
    expected_subsets = []
    if _flagged(df).any():
        expected_subsets.append('no_flagged')
    expected_subsets.append('heldout_splits')
    ran_subsets = set(table.loc[table['family'] == 'robustness', 'subset'])
    skipped_subsets = [name for name in expected_subsets
                       if name not in ran_subsets]
    robustness_note = (
        'ROBUSTNESS LINES (`family` = robustness, labelled by `subset`) '
        're-run the superiority family on: {}. Each is Holm-corrected over '
        'the comparisons it ran and none is held to the pre-registered '
        'size; they check the headline, they are not families of their '
        'own.{}'.format(
            ', '.join('`{}`'.format(name) for name in expected_subsets
                      if name in ran_subsets) or 'none',
            ' Skipped for having no rows: {}.'.format(
                ', '.join('`{}`'.format(name) for name in skipped_subsets))
            if skipped_subsets else ''))

    # n_comparisons is the same family size for every unit within a family
    # (it counts contrasts x metrics, not pairs), so one note per family
    # covers both units.
    n_superiority = (int(superiority_tables[0]['n_comparisons'].iloc[0])
                     if len(superiority_tables[0]) else 0)
    n_noninferiority = (int(noninferiority_tables[0]['n_comparisons'].iloc[0])
                        if len(noninferiority_tables[0]) else 0)
    family_note = (
        'TWO INDEPENDENTLY HOLM-CORRECTED FAMILIES are reported below, '
        'discriminated by the `family` column, per D13: non-inferiority is '
        'reported ALONGSIDE the superiority tests, never instead of them. '
        '`superiority`: {} comparisons were Holm-corrected within EACH unit '
        '(pair and split are corrected independently of each other); the '
        'pre-registered family is {} ({} joint arms x {} tests). {} '
        '`noninferiority`: {} comparisons were Holm-corrected within EACH '
        'unit; the pre-registered family is {} ({} joint arms x {} accuracy '
        'metrics -- F1 is not retested here, it stays in the superiority '
        'family). {}'.format(
            n_superiority, claims.PRE_REGISTERED_FAMILY_SIZE,
            len(claims.JOINT_ARM_SLUGS), len(claims.DEFAULT_METRICS),
            'The superiority family is complete.'
            if n_superiority == claims.PRE_REGISTERED_FAMILY_SIZE else
            'The superiority family is INCOMPLETE, so this correction is '
            'weaker than the pre-registered one and its adjusted p-values '
            'below are correspondingly optimistic.',
            n_noninferiority, claims.NONINFERIORITY_FAMILY_SIZE,
            len(claims.JOINT_ARM_SLUGS), len(claims.NONINFERIORITY_METRICS),
            'The non-inferiority family is complete.'
            if n_noninferiority == claims.NONINFERIORITY_FAMILY_SIZE else
            'The non-inferiority family is INCOMPLETE, so this correction '
            'is weaker than the pre-registered one and its adjusted '
            'p-values below are correspondingly optimistic.'))

    caption = (
        'Paired Wilcoxon signed-rank tests, one per (family, unit, contrast, '
        'task) and one per (family, unit, contrast) on blocks. Two '
        'INDEPENDENT test families are stacked in this one table (the '
        '`family` column): `superiority` is the pre-registered '
        '{}-comparison family ("no detectable loss", or non-inferiority '
        'within `margin` when `margin` > 0) and `noninferiority` is D13\'s '
        '{}-comparison family testing non-inferiority of accuracy at a '
        'margin sized per row as a fraction of that row\'s own baseline '
        'error -- reported ALONGSIDE the superiority tests, never in their '
        'place. The superiority family\'s robustness lines (`family` = '
        'robustness) follow it directly, one block per `subset`. Within each '
        'family, two units are reported for every comparison, per Ruling '
        'P7-3: `pair` tests one difference per (M, split, k) cell -- the '
        'spec-mandated primary, paired exactly as spec C.3 requires -- but '
        'cells within a split share a training split, so its p-values are '
        'anti-conservative relative to the number of independent splits. '
        '`split` collapses each split to its mean difference first -- the '
        'statistically clean check, far less powerful, valid under '
        'split-level replication. Neither supersedes the other; '
        'disagreement between them is itself the diagnostic. Each '
        '(family, unit) combination is Holm-corrected independently over '
        'its own family, never pooled with any other. The superiority '
        'accuracy/F1 tests are one-sided with alternative "greater" applied '
        'to {}, so a small p-value is the positive finding: the joint arm '
        'shows no detectable loss. The block test is two-sided, because '
        'alignment adds intervals before it merges any and sharing can cost '
        'blocks as well as save them. The non-inferiority tests are '
        'one-sided in the same direction, against a per-row margin rather '
        'than a flat one. '
        '{} {}'.format(
            claims.PRE_REGISTERED_FAMILY_SIZE,
            claims.NONINFERIORITY_FAMILY_SIZE,
            'd + {:g}'.format(margin) if margin > 0 else 'd', family_note,
            robustness_note))

    markdown = _markdown_table(
        _project_columns(table, _PAIRED_TEST_MARKDOWN_COLUMNS))
    body = '\n'.join([markdown, '', family_note, '', robustness_note, '',
                      'Hypotheses, verbatim from `claims.paired_tests` '
                      '(family=superiority/robustness) and '
                      '`claims.noninferiority_tests` '
                      '(family=noninferiority):', ''] +
                     ['* `{}` / `{}` / `{}` / `{}` / `{}`: {}'.format(
                         row['family'], row['subset'], row['unit'],
                         row['contrast'], row['metric'], row['hypothesis'])
                      for _, row in table.iterrows()])

    return _write(Deliverable(
        number=4, slug='paired_tests', title='Paired per-task tests, Holm-corrected',
        caption=caption, data=table, markdown_body=body), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 5 -- the ablation table
# ---------------------------------------------------------------------------

_ABLATION_MARKDOWN_COLUMNS = (
    'component', 'contrast', 'metric', 'n_pairs', 'n_splits',
    'mean_diff_split_level', 'ci_low', 'ci_high', 'median_diff_pairwise')


def table_5_ablation(df, output_dir=DEFAULT_FIGURE_DIR, confidence=0.95):
    """Where the savings come from: the sharing constraint or the threshold
    alignment. Rendered from `claims.ablation_decomposition`.

    Two components, and the second's baseline is the point of the whole
    table: `sharing` is `joint-off - independent`, and `alignment` is
    `joint-off-al - joint-off`. Measuring alignment against `independent`
    instead would re-count the sharing effect inside every alignment number
    and the two components would not add up.
    """
    table = claims.ablation_decomposition(
        df, metrics=claims.DEFAULT_METRICS, confidence=confidence)

    caption = (
        'Ablation of the joint arm\'s effect into its two causes, per task '
        'and on blocks, never pooled across tasks. "sharing" is '
        'joint-off minus independent: joint-off skips threshold alignment '
        'entirely, so the contrast isolates the cost of sharing one feature '
        'encoding. "alignment" is the aligned twin of each joint-off design minus that design (joint-off-al minus joint-off), measured '
        'against joint-off rather than against independent so that the '
        'sharing effect is not counted twice and the two components add up. '
        'Descriptive only: no p-values, because testing these contrasts too '
        'would enlarge the multiplicity family of Table 4 without enlarging '
        'the claim. Intervals are {:.0%} Student-t over split-level mean '
        'differences, since cells inside one split are not independent '
        'observations.'.format(confidence))

    body = _markdown_table(
        _project_columns(table, _ABLATION_MARKDOWN_COLUMNS))

    return _write(Deliverable(
        number=5, slug='ablation_decomposition',
        title='Ablation: sharing constraint cost against alignment cost',
        caption=caption, data=table, markdown_body=body), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 6 -- the capacity-ceiling appendix
# ---------------------------------------------------------------------------

def appendix_6_capacity_ceiling(ceiling_csv=DEFAULT_CEILING_CSV,
                                output_dir=DEFAULT_FIGURE_DIR):
    """Persist the capacity-ceiling rederivation (spec B.7) as markdown.

    `scripts/capacity_ceiling.py` already measured this -- roughly ten
    minutes of forest fitting -- and wrote `results/capacity_ceiling.csv`,
    but its markdown tables were printed to a terminal and then lost. This
    replays the script's OWN reporting half (`per_cell` and `report`) over
    that CSV and captures the output, so the appendix and the script can
    never disagree about the adoption rule: there is one implementation of
    it and this is not a second copy.

    The measurement is never re-run. `report` calls no fitting code, and
    `collect` -- the only function that does -- is not called from here.

    Raises FileNotFoundError when the CSV is absent: the appendix's purpose
    is to replace "chosen manually" with a measurement, and there is no
    honest way to render it from nothing.
    """
    if not os.path.exists(ceiling_csv):
        raise FileNotFoundError(
            'capacity-ceiling appendix: {!r} does not exist. Run '
            '`python scripts/capacity_ceiling.py` once to produce it (it '
            'takes about ten minutes); this appendix only re-renders that '
            'measurement and never repeats it.'.format(ceiling_csv))

    # Imported inside the function: `scripts/` is a script directory, not a
    # dependency of the reporting path, and its import pulls in
    # src.p4gen.build_p4_script (sklearn) for MAX_CODEWORD_LENGTH.
    from scripts.capacity_ceiling import per_cell, report

    frame = pd.read_csv(ceiling_csv)
    cells = per_cell(frame)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        n_trees, max_depth = report(cells)

    caption = (
        'Capacity-ceiling rederivation (spec B.7), replayed from {} -- the '
        'measurement itself is not repeated here. n_trees and max_depth are '
        'inclusive search bounds, not fixed hyperparameters, and were '
        'previously justified only as "chosen manually because larger values '
        'gave overly long codewords". The tables below locate where the '
        '512-bit codeword limit actually binds, over a grid of bounds, at '
        'both ends of the regularisation range the search also explores, and '
        'apply the adoption rule to the measurement. Adopted: n_trees = {}, '
        'max_depth = {}.'.format(ceiling_csv, n_trees, max_depth))

    return _write(Deliverable(
        number=6, slug='capacity_ceiling',
        title='Appendix: capacity-ceiling rederivation',
        caption=caption, data=cells,
        markdown_body=captured.getvalue()), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 7 -- elimination order per split
# ---------------------------------------------------------------------------

_ELIMINATION_KEYS = ('arm_slug', 'M', 'split')
_FEATURE_COLUMNS = tuple((key, features_column)
                        for key, _, _, _, features_column in TASKS)


def _feature_list(value):
    return [name for name in str(value).split(';') if name]


def elimination_order(df):
    """The order in which features were eliminated, per arm, M, split and
    task.

    No new computation: `features_app` / `features_ddos` carry the surviving
    feature set at every k, so the feature eliminated at each step is the
    set difference between consecutive k values, read downwards.

    Two honesty constraints:

    * The two tasks are reported separately even for the joint arm, where
      the two sets are identical by construction -- collapsing them would
      make the appendix's shape depend on the arm.
    * Infeasible rows are dropped at load, so a step can span more than one
      k. The features lost across such a step share one `elimination_rank`
      and carry `n_dropped_in_step > 1`, because their relative order is
      simply not recoverable from the surviving rows and inventing one would
      be a fabricated result. They are sorted by name within the step, for
      determinism only.

    The features still standing at the smallest k reached are emitted as
    `event='retained'` rows with no rank, so the final set is visible rather
    than having to be inferred from what is missing.
    """
    records = []
    for keys, group in df.groupby(list(_ELIMINATION_KEYS), sort=True):
        base = dict(zip(_ELIMINATION_KEYS, keys))
        for task, column in _FEATURE_COLUMNS:
            ordered = group.sort_values('k', ascending=False)
            previous_features, previous_k, rank = None, None, 0
            for _, row in ordered.iterrows():
                current = _feature_list(row[column])
                if previous_features is not None:
                    dropped = sorted(set(previous_features) - set(current))
                    if dropped:
                        rank += 1
                        for feature in dropped:
                            records.append(dict(
                                base, task=task, event='eliminated',
                                elimination_rank=rank, feature=feature,
                                from_k=previous_k, to_k=int(row['k']),
                                n_dropped_in_step=len(dropped)))
                        rank += len(dropped) - 1
                previous_features, previous_k = current, int(row['k'])
            for feature in previous_features or []:
                records.append(dict(
                    base, task=task, event='retained',
                    elimination_rank=np.nan, feature=feature,
                    from_k=np.nan, to_k=previous_k, n_dropped_in_step=np.nan))

    columns = list(_ELIMINATION_KEYS) + [
        'task', 'event', 'elimination_rank', 'feature', 'from_k', 'to_k',
        'n_dropped_in_step']
    return pd.DataFrame(records, columns=columns)


def appendix_7_elimination_order(df, output_dir=DEFAULT_FIGURE_DIR):
    """Appendix: which features each split eliminated, in which order.

    A reproducibility artifact rather than a claim: recursive feature
    elimination ranks by permutation importance measured on that split's own
    validation half, so the order legitimately differs between splits, and a
    reader comparing two runs needs to see the orders rather than be told
    they agree.

    The CSV is the complete record (one row per elimination event, plus the
    retained set); the markdown collapses each (arm, M, split, task) to its
    ordered sequence, which is what a reader scans.
    """
    events = elimination_order(df)

    sequences = []
    if len(events):
        for keys, group in events.groupby(
                list(_ELIMINATION_KEYS) + ['task'], sort=True):
            eliminated = group[group['event'] == 'eliminated'].sort_values(
                ['elimination_rank', 'feature'])
            retained = group[group['event'] == 'retained'].sort_values('feature')
            sequences.append(dict(
                zip(list(_ELIMINATION_KEYS) + ['task'], keys),
                eliminated_first_to_last=' > '.join(eliminated['feature']),
                retained_at_k=int(retained['to_k'].iloc[0])
                if len(retained) else np.nan,
                retained=' ; '.join(retained['feature'])))
    sequence_table = pd.DataFrame(sequences)

    caption = (
        'Elimination order per split. Recursive elimination drops the least '
        'important surviving feature at each k, ranked by permutation '
        'importance measured on that split\'s own selection half with the '
        'switch\'s hard-vote semantics, so the order is a per-split result '
        'and is expected to differ between splits; it is reported rather '
        'than summarised for exactly that reason. App and DDoS are listed '
        'separately throughout -- for the joint arm the two sets coincide by '
        'construction, and showing both makes that visible instead of '
        'assumed. Where a step spans more than one k (an infeasible k was '
        'dropped at load), the features lost in that step share a rank and '
        'carry n_dropped_in_step > 1: their relative order is not '
        'recoverable and is not invented.')

    return _write(Deliverable(
        number=7, slug='elimination_order',
        title='Appendix: elimination order per split',
        caption=caption, data=events,
        markdown_body=_markdown_table(sequence_table) if len(sequence_table)
        else None), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 8 -- entries vs blocks (T12, reviews/todo.md:487-499)
# ---------------------------------------------------------------------------

_ENTRIES_VS_BLOCKS_SUMMARY_COLUMNS = (
    'arm_slug', 'k', 'n_pairs', 'mean_d_range_entries',
    'mean_d_ternary_entries', 'mean_d_blocks', 'mean_entries_saving',
    'mean_blocks_saving', 'mean_rounding_loss')


def entries_vs_blocks_frame(df, baseline=claims.INDEPENDENT_ARM_SLUG,
                            arms=None):
    """One row per (arm, M, split, k) cell paired against `baseline`: the
    entries and blocks deltas, plus T12's rounding loss between them.

    Built directly on `campaign_data.pair_arms`, not `claims.arm_deltas`
    alone: `arm_deltas` only returns the DIFFERENCE, and this deliverable's
    whole point is a RATIO (entries-saving, blocks-saving) that needs the
    raw baseline value as its denominator too.

    `d_range_entries`, `d_ternary_entries`, `d_blocks` are signed
    `treatment - baseline`, matching `arm_deltas`'s convention -- negative
    means the treatment SAVED. `entries_saving` and `blocks_saving` are the
    two ratios T12 (reviews/todo.md:487-494) asks to be paired:
    `(baseline - treatment) / baseline` on, respectively, the SUM of
    range_entries + ternary_entries (a smooth, continuous physical-row
    count) and on `blocks` (an integer count, quantized in steps of
    `build_p4_script.TERNARY_MATCHING_ENTRIES_PER_BLOCK` physical rows per
    block). `rounding_loss` is
    `entries_saving - blocks_saving`: positive means entries saved
    proportionally MORE than blocks did -- quantization ate part of the
    saving. A baseline of 0 in either denominator yields NaN, never a
    division-by-zero infinity.
    """
    require_baseline(df, baseline, 'entries_vs_blocks_frame')
    if arms is None:
        arms = ordered_arms(df, include_baseline=False, baseline=baseline)

    frames = []
    for arm in arms:
        paired = pair_arms(df, arm, baseline)
        if len(paired) == 0:
            continue
        range_baseline = paired['range_entries_baseline'].astype('float64')
        range_treatment = paired['range_entries_treatment'].astype('float64')
        ternary_baseline = paired['ternary_entries_baseline'].astype('float64')
        ternary_treatment = paired['ternary_entries_treatment'].astype('float64')
        blocks_baseline = paired['blocks_baseline'].astype('float64')
        blocks_treatment = paired['blocks_treatment'].astype('float64')

        total_entries_baseline = range_baseline + ternary_baseline
        total_entries_treatment = range_treatment + ternary_treatment

        entries_saving = ((total_entries_baseline - total_entries_treatment)
                          / total_entries_baseline).where(
                              total_entries_baseline > 0, np.nan)
        blocks_saving = ((blocks_baseline - blocks_treatment)
                         / blocks_baseline).where(blocks_baseline > 0, np.nan)

        frame = pd.DataFrame({
            'arm_slug': arm,
            'M': paired['M'], 'split': paired['split'], 'k': paired['k'],
            'range_entries_baseline': range_baseline,
            'range_entries_treatment': range_treatment,
            'ternary_entries_baseline': ternary_baseline,
            'ternary_entries_treatment': ternary_treatment,
            'blocks_baseline': blocks_baseline,
            'blocks_treatment': blocks_treatment,
            'd_range_entries': range_treatment - range_baseline,
            'd_ternary_entries': ternary_treatment - ternary_baseline,
            'd_blocks': blocks_treatment - blocks_baseline,
            'entries_saving': entries_saving,
            'blocks_saving': blocks_saving,
            'rounding_loss': entries_saving - blocks_saving,
        })
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[
            'arm_slug', 'M', 'split', 'k',
            'range_entries_baseline', 'range_entries_treatment',
            'ternary_entries_baseline', 'ternary_entries_treatment',
            'blocks_baseline', 'blocks_treatment',
            'd_range_entries', 'd_ternary_entries', 'd_blocks',
            'entries_saving', 'blocks_saving', 'rounding_loss'])
    long = pd.concat(frames, ignore_index=True)
    return claims.attach_delta_columns(long, df)


def figure_8_entries_vs_blocks(df, output_dir=DEFAULT_FIGURE_DIR,
                               baseline=claims.INDEPENDENT_ARM_SLUG):
    """T12 (reviews/todo.md:487-499): does joint mapping's memory saving
    survive TCAM block quantization, or is it an artifact of rounding?

    Pairs two saving ratios computed from the SAME cells: entries-saving (a
    smooth, continuous physical-row count) against blocks-saving (the same
    quantity after the campaign CSV's `blocks` column has already rounded it
    up in steps of `build_p4_script.TERNARY_MATCHING_ENTRIES_PER_BLOCK`
    rows per block). The gap between them, `rounding_loss`, is descriptive
    only (D3) -- T12's question is
    mechanistic ("where does the gap happen"), not "is there a gap", and
    entries and blocks are too collinear for a second significance test to
    spend Table 4's multiplicity budget on.

    CSV + markdown only, no PDF: `entries_vs_blocks_frame`'s per-cell table
    already answers T12's question directly, and a new scatter/plot type
    would need its own defence of what "distance from the diagonal" means
    that a table does not (matching deliverables 4, 5 and 7, which are also
    table-only -- plan finding V7). The per-cell frame is `.data` (so a
    reader can check any summary number against the row it came from); the
    markdown body is a per-(arm, k) summary, faceted at odd k only (D7,
    matching figures 1 and 2's `_facet_k_values` pattern) -- even k is still
    in `.data` and still pooled into the caption's headline sentence, it
    simply gets no row in the printed summary.
    """
    # Imported here, not at module scope: build_p4_script pulls in sklearn
    # at import (`from sklearn.tree import export_text`), which this
    # reporting module otherwise never needs -- the same reason
    # appendix_6_capacity_ceiling (`:1090`, this file) imports
    # scripts.capacity_ceiling lazily rather than at module scope. This is
    # the single source of truth for the TCAM block boundary; no literal
    # 512 is duplicated here.
    from src.p4gen.build_p4_script import (
        TERNARY_MATCHING_ENTRIES_PER_BLOCK as tcam_rows_per_block)

    long = entries_vs_blocks_frame(df, baseline=baseline)

    shown_k, dropped_k = _facet_k_values(
        long['k'] if len(long) and 'k' in long.columns else None,
        'figure 8')

    summary = pd.DataFrame(columns=['arm_slug', 'k'])
    if len(long):
        summary = long.groupby(['arm_slug', 'k'], as_index=False).agg(
            n_pairs=('rounding_loss', 'size'),
            mean_d_range_entries=('d_range_entries', 'mean'),
            mean_d_ternary_entries=('d_ternary_entries', 'mean'),
            mean_d_blocks=('d_blocks', 'mean'),
            mean_entries_saving=('entries_saving', 'mean'),
            mean_blocks_saving=('blocks_saving', 'mean'),
            mean_rounding_loss=('rounding_loss', 'mean'))
    shown_summary = (summary[summary['k'].isin(shown_k)]
                     if len(summary) and shown_k and shown_k[0] is not None
                     else summary)

    overall_entries_saving = (float(long['entries_saving'].mean())
                              if len(long) else float('nan'))
    overall_blocks_saving = (float(long['blocks_saving'].mean())
                             if len(long) else float('nan'))
    overall_rounding_loss = (float(long['rounding_loss'].mean())
                             if len(long) else float('nan'))

    # `overall_entries_saving`/`overall_rounding_loss` are NaN whenever
    # `long['entries_saving']` is entirely NaN -- the real, documented case
    # where every source CSV predates the range_entries/ternary_entries
    # columns (campaign_data.py:120-124) and `entries_vs_blocks_frame` had
    # nothing to divide. A partially-NaN column still yields a real mean
    # over the non-NaN rows (`.mean()` already skips NaN), so this only
    # triggers on the fully-missing case -- it must never interpolate NaN
    # into prose that claims a specific percentage.
    entries_data_unavailable = bool(np.isnan(overall_entries_saving))
    # Distinct from `entries_data_unavailable`: that case has paired rows
    # whose entries columns are unusable, so `overall_blocks_saving` is
    # still a real number. Here there is no paired data AT ALL -- `long`
    # itself has zero rows (e.g. only the baseline arm is present, so
    # `pair_arms` never found a treatment row to join) -- and
    # `overall_blocks_saving` is ALSO NaN, so formatting it as blocks_pct
    # below would print "nan%" for blocks too.
    no_paired_data = len(long) == 0

    if no_paired_data:
        pooled_sentence = (
            'No (arm, M, split, k) cell paired against {baseline} in this '
            'campaign at all -- only the baseline arm (or no arm) is '
            'present in this frame -- so there is nothing to report: '
            'neither entries-saving, blocks-saving nor rounding_loss can '
            'be computed, and the table below is empty.'.format(
                baseline=baseline))
    elif entries_data_unavailable:
        pooled_sentence = (
            'This run\'s source CSVs do not carry usable range_entries / '
            'ternary_entries values (NaN on every paired cell -- written '
            'before those columns existed, campaign_data.py:120-124), so '
            'entries-saving and rounding_loss are UNAVAILABLE for this run '
            'and cannot be computed or reported. Only blocks-saving is '
            'available: pooled over every joint arm, M, split and k paired '
            'against {baseline} in this campaign, the block column moves '
            'by {blocks_pct:.1%} on average; the table below (faceted by '
            'k, D7; even k is still pooled into this sentence but gets no '
            'row of its own) reports blocks-saving broken out by k, with '
            'the entries columns blank.'.format(
                baseline=baseline, blocks_pct=overall_blocks_saving))
    else:
        pooled_sentence = (
            'Pooled over every joint arm, M, split and k paired against '
            '{baseline} in this campaign: joint mapping removes '
            '{entries_pct:.1%} of table entries on average; the block '
            'column only moves by {blocks_pct:.1%}; the {gap_pct:.1%} gap '
            'between them is quantization, and the table below (faceted by '
            'k, D7; even k is still pooled into this sentence but gets no '
            'row of its own) shows where it concentrates.'.format(
                baseline=baseline, entries_pct=overall_entries_saving,
                blocks_pct=overall_blocks_saving,
                gap_pct=overall_rounding_loss))

    caption = (
        'T12 (reviews/todo.md:487-499): does joint mapping\'s memory saving '
        'survive TCAM block quantization, or is it partly an artefact of '
        'rounding? Per (arm, M, split, k) cell paired against the {baseline} '
        'arm on `campaign_data.pair_arms`\'s (M, split, k) join key, this '
        'compares two saving ratios computed from the SAME cells: '
        'entries-saving = (baseline_entries - treatment_entries) / '
        'baseline_entries (summing range_entries + ternary_entries, the '
        'expanded PHYSICAL TCAM row counts -- a smooth, continuous '
        'quantity), against blocks-saving = (baseline_blocks - '
        'treatment_blocks) / baseline_blocks, where blocks is already '
        'those same rows rounded UP in steps of {block_size} physical rows '
        'per block (TERNARY_MATCHING_ENTRIES_PER_BLOCK, '
        'src/p4gen/build_p4_script.py:21) -- a quantized, step-function '
        'quantity. rounding_loss = entries-saving - blocks-saving: a '
        'POSITIVE value means entries saved proportionally MORE than '
        'blocks did, i.e. quantization ate part of the saving. '.format(
            baseline=baseline, block_size=tcam_rows_per_block)
        + pooled_sentence +
        ' Descriptive only (D3): no p-value is '
        'reported here and none should be -- this is a mechanistic "where" '
        'question, not a "does it differ" one, and entries and blocks are '
        'too collinear for a second test to add information Table 4\'s '
        'blocks test does not already carry.')

    body_lines = [
        'Block boundary: TERNARY_MATCHING_ENTRIES_PER_BLOCK = {0} physical '
        'TCAM rows per block (src/p4gen/build_p4_script.py:21) is the '
        'rounding unit behind blocks-saving below. blocks is NOT '
        'ceil(sum(entries) / {0}): range_matching_resource_usage and '
        'ternary_matching_resource_usage (src/p4gen/evaluation.py) round '
        'up PER FEATURE (range) and PER TREE (ternary) independently '
        'before summing, and the ternary side further multiplies each '
        'tree by a codeword-width factor -- so blocks can move for '
        'reasons entries alone does not capture, on top of the '
        'block-size rounding itself.'.format(tcam_rows_per_block),
        '',
        (_markdown_table(_project_columns(
            shown_summary, _ENTRIES_VS_BLOCKS_SUMMARY_COLUMNS))
         if len(shown_summary) else '(no paired cells)'),
    ]
    if dropped_k:
        body_lines += ['', (
            'k = {} are still computed above (see the full per-cell CSV) '
            'and pooled into the caption\'s headline sentence, but are not '
            'broken out as their own row in this summary (facet is odd k '
            'only, D7).'.format(', '.join(str(k) for k in dropped_k)))]

    return _write(Deliverable(
        number=8, slug='entries_vs_blocks',
        title='Entries against blocks: the TCAM quantization gap',
        caption=caption, data=long,
        markdown_body='\n'.join(body_lines)), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 9 -- model-vs-p4c agreement (compiler-verified runs only)
# ---------------------------------------------------------------------------

def _tables_differing_text(tables):
    """`tables_differing` (a list of {table, model, p4c}) as one readable
    markdown cell: 'tbl_x: model 4, p4c 6; ...'."""
    return '; '.join('{}: model {}, p4c {}'.format(
        entry.get('table'), entry.get('model'), entry.get('p4c'))
        for entry in tables)


def table_9_agreement(verification, output_dir=DEFAULT_FIGURE_DIR):
    """Model-vs-p4c agreement per arm x M, and every miss -- rendered from
    `claims.agreement_table`, not recomputed (spec section 7.3).

    `verification` is `campaign_data.load_verification`'s frame: EVERY
    verified design, including the ones `load_campaign` drops as
    p4c-infeasible, because this table is about the cost model, not about
    the reported designs. `.data` is the summary (one row per arm x M); the
    misses go beside it as `<stem>_misses.csv` (with `tables_differing` as
    JSON text), and the markdown carries both frames.
    """
    summary, misses = claims.agreement_table(verification)

    misses_csv = misses.copy()
    misses_csv['tables_differing'] = [
        json.dumps(tables, sort_keys=True) for tables in misses['tables_differing']]
    misses_markdown = misses.copy()
    misses_markdown['tables_differing'] = [
        _tables_differing_text(tables) for tables in misses['tables_differing']]

    n_rows = int(summary['n'].sum()) if len(summary) else 0
    caption = (
        'Agreement between the cost model and p4c, per arm and block budget '
        'M, over every design of the run p4c was asked to compile ({} rows) '
        '-- including designs the reported frame drops because p4c found '
        'them infeasible: this table is about the model, not about the '
        'reported designs. `stage_depth_exact` and `blocks_exact` count the '
        'rows whose model number equals p4c\'s exactly; `blocks_na` counts '
        'verified rows where p4c never allocated blocks (over 12 stages), '
        'so blocks agreement reads out of `n - blocks_na` at most. An '
        'unverified row (compile error or timeout) is exact on neither. '
        'Below the summary, every row whose verdict is not EXACT, with both '
        'numbers and the tables that differ.'.format(n_rows))

    body = '\n'.join([
        _markdown_table(summary) if len(summary) else '(no verified rows)',
        '', '## Misses ({})'.format(len(misses)), '',
        _markdown_table(misses_markdown) if len(misses_markdown)
        else '(none: every verified row is EXACT)'])

    return _write(Deliverable(
        number=9, slug='agreement',
        title='Model-vs-p4c agreement per arm and block budget',
        caption=caption, data=summary, markdown_body=body,
        extra_data=(('misses', misses_csv),)), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 10 -- budget binding (compiler-verified runs only)
# ---------------------------------------------------------------------------

def table_10_budget_binding(df, output_dir=DEFAULT_FIGURE_DIR, share=0.9):
    """Per arm x M, the share of rows whose blocks reach `share * M` --
    rendered from `claims.budget_binding` (spec section 7.3).

    `df` is `load_campaign`'s frame, whose `blocks` holds p4c's numbers for
    every verified row and the MODEL's for a flagged one; the caption says
    how many flagged rows there are.
    """
    table = claims.budget_binding(df, share=share)
    flagged = _flagged(df)
    caption = (
        'Budget binding: for each arm and block budget M, the share of '
        'designs whose TCAM blocks reach {:g} x M. A budget that binds shows '
        'a high share; M = 75 near zero and indistinguishable from the '
        'unbudgeted cell confirms the top of the grid, and any other '
        'non-binding M is reported as such rather than hidden. The '
        'unbudgeted cell (M = {}) has no budget to bind and its share is '
        'left blank. Blocks are p4c\'s numbers for every verified design; '
        '{} of {} rows are flagged (unverified) and carry the model\'s '
        'numbers instead.'.format(share, format_M(float('inf')),
                                  int(flagged.sum()), len(df)))
    body = _markdown_table(table) if len(table) else '(no rows)'
    return _write(Deliverable(
        number=10, slug='budget_binding',
        title='Budget binding per arm and block budget',
        caption=caption, data=table, markdown_body=body), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 11 -- alignment twins (compiler-verified runs with twins only)
# ---------------------------------------------------------------------------

def table_11_alignment_twins(df, verification, output_dir=DEFAULT_FIGURE_DIR,
                             confidence=0.95):
    """Threshold alignment as a post-selection step: each joint-off design
    against its aligned twin (spec 2026-10-06 section 6). Rendered from
    `claims.twin_effect` (paired deltas per M, per k group and pooled),
    `claims.twin_ladder` (independent -> joint-off -> joint-off-al) and
    `claims.twin_counts` (identical / compiled / rescued twins). A rescued
    twin -- feasible where its source was not -- is counted and never paired.
    """
    effect = claims.twin_effect(df, confidence=confidence)
    ladder = claims.twin_ladder(df, confidence=confidence)
    counts = pd.DataFrame([claims.twin_counts(verification)])
    alignment = claims.twin_alignment_stats(df)
    c = counts.iloc[0]
    caption = (
        'Threshold alignment applied after selection: every joint-off design '
        'paired with its aligned twin (same forest, thresholds aligned on '
        'val_align, regenerated and compiled), p4c numbers. Negative blocks and '
        'stage-depth deltas are savings; F1 in points of the 0-1 scale. '
        'Intervals are {:.0%} Student-t over split-level mean differences. '
        '{} twins: {} byte-identical to their source (verification record '
        'copied, not compiled), {} compiled ({} EXACT against the cost model), '
        '{} rescued (feasible where the source was not; counted, never paired), '
        '{} infeasible. The ladder table gives independent -> joint-off -> '
        'joint-off-al means per M on the cells where all three exist. '
        'p_saves / p_costs are the shares of pairs with a strictly better / worse '
        'delta (for integer blocks and stages, P(saves >= 1 block) and P(costs >= '
        '1 block); P(-1 stage) is the stage_depth row). The alignment table gives '
        'what the step did per M: mean intervals removed, mean accepted / attempted '
        'alignments, and the share of twins where none was accepted.'.format(
            confidence, int(c['n_twins']), int(c['n_identical']), int(c['n_compiled']),
            int(c['n_compiled_exact']), int(c['n_rescued']), int(c['n_twin_infeasible'])))
    # `group` holds an M on the 'M' rows: print it through `format_M`.
    shown = effect.copy()
    if len(shown):
        shown['group'] = [format_M(g) if kind == 'M' else g
                          for kind, g in zip(shown['group_kind'], shown['group'])]
    shown_alignment = alignment.copy()
    shown_alignment['M'] = [m if m == 'all' else format_M(m) for m in shown_alignment['M']]
    body = '\n'.join([
        '## Paired twin - source', '', _markdown_table(shown) if len(effect) else '(no pairs)',
        '', '## Ladder', '', _markdown_table(ladder) if len(ladder) else '(no complete cells)',
        '', '## Alignment statistics', '',
        _markdown_table(shown_alignment) if len(alignment) else '(no alignment columns)',
        '', '## Counts', '', _markdown_table(counts)])
    return _write(Deliverable(
        number=11, slug='alignment_twins',
        title='Threshold alignment as post-processing: aligned twins against their sources',
        caption=caption, data=effect, markdown_body=body,
        extra_data=(('ladder', ladder), ('alignment_stats', alignment),
                    ('counts', counts))), output_dir)


# ---------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------

def render_all(df, output_dir=DEFAULT_FIGURE_DIR,
               ceiling_csv=DEFAULT_CEILING_CSV,
               baseline=claims.INDEPENDENT_ARM_SLUG,
               expected_family_size=None,
               expected_noninferiority_family_size=None,
               expected_substitution_family_size=None,
               verification=None):
    """Render every §C.5 deliverable and return them in order.

    `verification` is `campaign_data.load_verification`'s frame for a
    compiler-verified RUN (a directory with `rows/`); passing it adds
    deliverables 9 (agreement) and 10 (budget binding), which only a run
    can support, and 11 (alignment twins) when the frame has twin rows. None -- a legacy flat directory -- omits both.

    `ceiling_csv=None` omits deliverable 6 -- the capacity-ceiling appendix
    replays a measurement that either exists on disk or does not, and a
    campaign frame contains nothing from which it could be reconstructed.
    Every other deliverable comes from `df` alone.

    `expected_family_size` and `expected_noninferiority_family_size` gate
    deliverable 4's two independent Holm families (superiority and
    non-inferiority respectively) -- see `table_4_paired_tests`.
    `expected_substitution_family_size` gates deliverable 3's separate
    substitution family -- see `figure_3_substitution_scatter` and
    `claims.substitution_test_all_arms`.
    """
    deliverables = [
        figure_1_accuracy_vs_blocks(df, output_dir=output_dir,
                                    baseline=baseline),
        figure_2_delta_frontier(df, output_dir=output_dir, baseline=baseline),
        figure_3_substitution_scatter(
            df, output_dir=output_dir, baseline=baseline,
            expected_family_size=expected_substitution_family_size),
        table_4_paired_tests(
            df, output_dir=output_dir, baseline=baseline,
            expected_family_size=expected_family_size,
            expected_noninferiority_family_size=expected_noninferiority_family_size),
        table_5_ablation(df, output_dir=output_dir),
    ]
    if ceiling_csv is not None:
        deliverables.append(appendix_6_capacity_ceiling(
            ceiling_csv=ceiling_csv, output_dir=output_dir))
    deliverables.append(
        appendix_7_elimination_order(df, output_dir=output_dir))
    deliverables.append(
        figure_8_entries_vs_blocks(df, output_dir=output_dir,
                                   baseline=baseline))
    if verification is not None:
        deliverables.append(
            table_9_agreement(verification, output_dir=output_dir))
        deliverables.append(table_10_budget_binding(df, output_dir=output_dir))
        if claims.TWIN_ARM_SLUG in set(df['arm_slug']):
            deliverables.append(
                table_11_alignment_twins(df, verification, output_dir=output_dir))
    return tuple(sorted(deliverables, key=lambda item: item.number))
