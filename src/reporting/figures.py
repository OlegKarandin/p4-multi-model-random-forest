"""P7c: the thesis deliverables (spec C.5) -- seven from the original plan
plus deliverable 8 (S3.3, T12's entries-vs-blocks question).

Everything upstream of this module exists to make these artifacts correct.
`campaign_data.load_campaign` supplies the frame, `claims.py` supplies
every statistic, and this module does one thing: render them.

| # | artifact | answers |
|---|---|---|
| 1 | per-task accuracy vs blocks plus the acc_app-vs-acc_ddos trade plane (D8), all arms overlaid | the headline comparison, never averaged |
| 2 | delta frontier: d blocks and d rel-error per task vs delta, mean +/- CI | what the tolerance buys |
| 3 | substitution scatter with quadrants, one panel per arm | the reviewer's objection, answered directly |
| 4 | paired per-task test table, Holm-corrected | significance |
| 5 | ablation table: constraint cost vs alignment cost | where the savings come from |
| 6 | appendix: capacity-ceiling rederivation (B.7) | replaces "chosen manually" |
| 7 | appendix: elimination order per split | reproducibility |
| 8 | entries vs blocks, faceted by k (S3.3) | T12: is the joint-mapping saving real, or an artifact of TCAM block quantization? |

Four, five and eight are tables and six is a replay, so `Deliverable`
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

Arm ordering follows `claims.JOINT_ARM_SLUGS` (the sweep order), with the
independent baseline first and any arm slug this module has never heard of
appended at the end rather than dropped -- an unrecognised arm is a thing
to see in the figure, not to hide.
"""
import contextlib
import io
import math
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

# Figure 2's reported quantities: one relative-error change per task, plus
# the block delta. Derived from TASKS rather than spelled out, so a third
# task would gain a panel here too and the two lists cannot drift apart.
# Deliberately one panel per task rather than one shared axis.
REL_ERROR_PREFIX = 'rel_error_change_'
BLOCKS_DELTA = 'd_blocks'
REL_ERROR_METRICS = tuple(REL_ERROR_PREFIX + key for key, _, _, _, _ in TASKS)
FRONTIER_METRICS = REL_ERROR_METRICS + (BLOCKS_DELTA,)

# Per-panel size in inches. Multiplied by the panel counts a given frame
# implies -- never a fixed canvas for a fixed grid.
_PANEL_WIDTH = 6.0
_PANEL_HEIGHT = 4.2

# Marker cycle, used together with the colour cycle so that a campaign with
# more arms than the qualitative colormap has colours stays readable.
_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>')


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


def _log(message):
    """Log a message to stdout. Used to announce which facets were computed
    but not shown in the deliverable figures."""
    print(message)


# ---------------------------------------------------------------------------
# Arms, labels and per-arm styling
# ---------------------------------------------------------------------------

def ordered_arms(df, include_baseline=True,
                 baseline=claims.INDEPENDENT_ARM_SLUG):
    """The arm slugs present in `df`, in sweep order.

    Known arms come first in `claims.JOINT_ARM_SLUGS` order (the two
    anchors, then increasing delta), so tables and figures read left to
    right as the sweep. An arm slug this module does not recognise is
    APPENDED rather than dropped: a campaign that grew an arm should show
    up in the figure, not vanish from it.
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
    """The x-axis label for one arm on the delta sweep.

    Both non-numeric arms keep their own identity instead of being given an
    invented numeric position: `joint-dinf` is the accept-all anchor (not a
    large number) and `joint-off` never ran alignment at all (not delta 0).
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


def _panel_grid(n_panels):
    """Rows and columns for `n_panels`, as square as possible. The old
    module hardcoded 3x3 and silently mis-rendered any other count."""
    columns = int(math.ceil(math.sqrt(n_panels)))
    rows = int(math.ceil(n_panels / columns))
    return rows, columns


def _make_figure(n_rows, n_columns):
    return Figure(figsize=(_PANEL_WIDTH * n_columns,
                           _PANEL_HEIGHT * n_rows))


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
        _log('{}: {} computed but not shown (facet is odd k only, 1..17) '
             '-- still pooled into every paired test and every pooled '
             'statistic.'.format(
                 context, ', '.join('k={}'.format(k) for k in dropped)))
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
            _markdown_cell(row[column], float_format)
            for column in columns) + ' |')
    return '\n'.join(lines)


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
                axis.scatter(cell_rows['blocks'], cell_rows[accuracy_column],
                             s=10, alpha=0.25, color=colour, linewidths=0,
                             gid='cells:{}'.format(arm))
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
            off_rows = cell_rows[~on_front]
            on_rows = cell_rows[on_front]
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
        carried = [column for column in
                   ('arm_slug', 'M', 'split', 'k', 'blocks', 'stages',
                    'acc_app', 'acc_ddos')
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

    caption = (
        'Per-task accuracy against TCAM blocks: the top two rows are the '
        'two tasks, never averaged, and columns are odd feature counts k '
        '(1, 3, 5, ..., 17), with every arm of the sweep overlaid in every '
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
        'imply.'.format(facet_sentence, coverage_sentence))

    data = (pd.concat(front_frames, ignore_index=True)
            if front_frames else pd.DataFrame())
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
    the block delta and the two per-task relative-error changes.

    `d_blocks` comes from `claims.arm_deltas` -- the module that owns paired
    differences and the `(M, split, k)` join key -- and the relative-error
    columns are computed here from the same `pair_arms` join, because a
    ratio is not a difference and `claims.py` does not compute it. The two
    are merged back on the join key with `validate='one_to_one'`, so a
    duplicated cell fails loudly instead of silently multiplying rows.
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
    three quantities, aggregated over SPLITS.

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
    """What the alignment tolerance buys: block saving and per-task relative
    error change against delta, mean +/- CI across splits.

    A grid of panels: rows are the three reported quantities (the two
    tasks' relative error change plus the block delta), because pooling the
    two tasks would reintroduce the defect this rerun exists to fix, and
    columns are odd feature counts k (D7), because the paper's conclusion
    is k-dependent and a single k-pooled point cannot reproduce it. The x
    axis within each panel is categorical in sweep order rather than
    numeric: `joint-off` (alignment never ran) and `joint-dinf` (accept
    every move) are anchors, not numbers, and placing them on a numeric
    axis would require inventing coordinates for them.
    """
    table = delta_frontier_table(df, baseline=baseline, confidence=confidence)
    arms = [arm for arm in ordered_arms(df, include_baseline=False,
                                        baseline=baseline)
            if arm in set(table['arm_slug'])] if len(table) else []
    positions = {arm: index for index, arm in enumerate(arms)}

    labels = {REL_ERROR_PREFIX + key: '{}: rel. error change vs {}'.format(
        short_name, baseline) for key, _, short_name, _, _ in TASKS}
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
            axis.set_xlabel('alignment tolerance delta')
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
            ', '.join(str(value) for value in pooled_m) or 'none present',
            ', '.join(str(value) for value in pooled_k) or 'none present'))

    caption = (
        'The alignment tolerance sweep: block change and per-task relative '
        'error change against delta, each point a mean over splits with a '
        '{:.0%} Student-t confidence interval, paired against the {} arm on '
        '(M, split, k). {}The two tasks are shown on separate panels and are '
        'never averaged; relative error ((e_delta - e_base) / e_base) is '
        'reported because the tasks have very different error scales, so '
        'equal accuracy losses are not equal degradations. The two anchors '
        'carry no numeric delta and are labelled as themselves: "off" never '
        'ran alignment at all, and "inf" accepts every move. '
        'THE FEATURE SETS DIFFER ACROSS DELTA BY CONSTRUCTION -- alignment '
        'changes which thresholds, and hence which intervals and which '
        'eliminated features, each arm ends up with, so the arms are not '
        'evaluated on identical inputs. Split-level replication is what '
        'controls the resulting variance: each interval is built over '
        'per-split mean differences, one observation per split, so the '
        'spread of feature sets across splits is inside the interval rather '
        'than being assumed away.'.format(confidence, baseline,
                                          pooling_sentence))

    return _write(Deliverable(
        number=2, slug='delta_frontier',
        title='Delta frontier: block and per-task relative-error change',
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
                                  alpha=0.05):
    """One panel per joint arm: the paired per-task accuracy deltas against
    each other, with the sign quadrants annotated.

    This answers the reviewer's objection directly. Substitution -- one task
    paying for the other's gain -- is a NEGATIVE correlation between the two
    deltas, and the mass in the two off-diagonal quadrants is what it looks
    like. Every number annotated comes from `claims.substitution_test_all_arms`:
    the Pearson r, the partial r controlling for the block delta (two
    accuracy deltas can correlate purely because both track how much TCAM
    the cell was allowed), the Holm-corrected one-sided p across the seven
    arms, and the quadrant fractions.

    The test runs at every arm, not just the largest delta, so the claim
    defended is "no task sacrifices itself at any tolerance" rather than "at
    one operating point".
    """
    require_baseline(df, baseline, 'figure_3_substitution_scatter')
    table = claims.substitution_test_all_arms(df, baseline=baseline,
                                              alpha=alpha)
    arms = list(table['treatment']) if len(table) else []
    rows, columns = _panel_grid(max(len(arms), 1))

    figure = _make_figure(rows, columns)
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for axis in axes[len(arms):]:
        figure.delaxes(axis)

    for axis, arm in zip(axes, arms):
        record = table[table['treatment'] == arm].iloc[0]
        deltas = claims.arm_deltas(df, arm, baseline)
        axis.scatter(deltas['d_acc_app'], deltas['d_acc_ddos'],
                     s=14, alpha=0.55, linewidths=0,
                     gid='substitution:{}'.format(arm))
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
    figure.tight_layout()

    detected = (list(table.loc[table['substitution_detected_holm'], 'treatment'])
                if len(table) else [])
    caption = (
        'Paired per-task accuracy deltas against the {} arm, one panel per '
        'joint arm, with the sign quadrants and their fractions. '
        'Substitution -- one task gaining at the other\'s expense -- is a '
        'negative correlation, i.e. mass in the two off-diagonal quadrants '
        '("App down / DDoS up" and "App up / DDoS down"); cells where either '
        'task did not move at all are counted '
        'separately and are in none of the four. Each panel reports the '
        'Pearson r, the partial r controlling for the block delta (two '
        'accuracy deltas can move together simply because both track the '
        'cell\'s block budget), and the one-sided p for rho < 0 after '
        'Holm-Bonferroni correction across the {} arms tested. Arms where '
        'substitution is detected at alpha = {:g} after correction: {}. The '
        'test is run at every tolerance, so the claim is about the whole '
        'sweep and not one operating point. Cells within a split share a '
        'training split, so these p-values are anti-conservative relative to '
        'the number of independent splits.'.format(
            baseline, len(arms), alpha,
            ', '.join(detected) if detected else 'none'))

    return _write(Deliverable(
        number=3, slug='substitution_scatter',
        title='Substitution: per-task accuracy deltas against each other',
        caption=caption, data=table, figure=figure), output_dir)


# ---------------------------------------------------------------------------
# Deliverable 4 -- the paired per-task test table
# ---------------------------------------------------------------------------

_PAIRED_TEST_MARKDOWN_COLUMNS = (
    'unit', 'contrast', 'metric', 'alternative', 'n_pairs', 'n_splits',
    'median_diff', 'mean_diff_split_level', 'ci_low', 'ci_high',
    'p_value', 'p_holm', 'significant_holm')


def table_4_paired_tests(df, output_dir=DEFAULT_FIGURE_DIR,
                         baseline=claims.INDEPENDENT_ARM_SLUG,
                         margin=0.0, alpha=0.05, units=('pair', 'split'),
                         expected_family_size=None):
    """The pre-registered paired tests, Holm-corrected -- rendered, not
    recomputed. Every number is `claims.paired_tests`'.

    One row per (unit, contrast, metric), and `acc_app` and `acc_ddos` are
    separate rows throughout: there is no pooled accuracy test, because a
    pooled test is exactly what let a loss on one task hide behind a gain on
    the other.

    Ruling P7-3: `unit='pair'` -- one difference per `(M, split, k)` cell --
    is the spec-mandated primary, but those cells are not independent (the
    same split recurs across every M and k), so its p-values are
    anti-conservative. `unit='split'` -- one mean difference per split -- is
    the statistically clean check: valid under split-level replication, far
    less powerful. The ruling requires BOTH be visible wherever the primary
    appears, not the primary alone with the split-level number folded into a
    confidence interval elsewhere. So this table stacks both: `units` is
    called through `claims.paired_tests` once per unit, each with its own
    independently Holm-corrected family (mixing the two units into one
    Holm family would correct pair-level and split-level p-values against
    each other, which is not what either correction means), and the results
    are concatenated with the `unit` column identifying which is which.

    `expected_family_size` is passed straight through to every unit's call
    and defaults to None so a partial campaign (the pilot cell) still
    produces a table. That is a real weakening -- Holm over 9 comparisons is
    a laxer correction than Holm over the pre-registered 35 -- so the
    rendered markdown always states how many comparisons were actually
    corrected over and what the pre-registered family size is. Pass
    `expected_family_size=claims.PRE_REGISTERED_FAMILY_SIZE` on the complete
    campaign to turn a shrunken family into an error.
    """
    tables = [
        claims.paired_tests(
            df, baseline=baseline, metrics=claims.DEFAULT_METRICS,
            margin=margin, alpha=alpha, unit=unit,
            expected_family_size=expected_family_size)
        for unit in units
    ]
    table = pd.concat(tables, ignore_index=True)

    # n_comparisons is the same family size for every unit (it counts
    # contrasts x metrics, not pairs), so one note covers all of them.
    n_comparisons = int(tables[0]['n_comparisons'].iloc[0]) if len(tables[0]) else 0
    family_note = (
        '{} comparisons were Holm-corrected within EACH unit below (pair and '
        'split are corrected independently of each other); the pre-registered '
        'family is {} (7 joint arms x 5 tests). {}'.format(
            n_comparisons, claims.PRE_REGISTERED_FAMILY_SIZE,
            'The family is complete.'
            if n_comparisons == claims.PRE_REGISTERED_FAMILY_SIZE else
            'The family is INCOMPLETE, so this correction is weaker than the '
            'pre-registered one and the adjusted p-values below are '
            'correspondingly optimistic.'))

    caption = (
        'Paired Wilcoxon signed-rank tests, one per (unit, contrast, task) '
        'and one per (unit, contrast) on blocks. Two units are reported for '
        'every comparison, per Ruling P7-3: `pair` tests one difference per '
        '(M, split, k) cell -- the spec-mandated primary, paired exactly as '
        'spec C.3 requires -- but cells within a split share a training '
        'split, so its p-values are anti-conservative relative to the '
        'number of independent splits. `split` collapses each split to its '
        'mean difference first -- the statistically clean check, far less '
        'powerful, valid under split-level replication. Neither supersedes '
        'the other; disagreement between them is itself the diagnostic. '
        'Each unit is Holm-corrected independently over its own family, '
        'never pooled with the other. The accuracy tests are one-sided with '
        'alternative "greater" applied to {}, so a small p-value is the '
        'positive finding: the joint arm shows no detectable loss. The '
        'block test is two-sided, because alignment adds intervals before '
        'it merges any and sharing can cost blocks as well as save them. '
        '{}'.format(
            'd + {:g}'.format(margin) if margin > 0 else 'd', family_note))

    markdown = _markdown_table(
        _project_columns(table, _PAIRED_TEST_MARKDOWN_COLUMNS))
    body = '\n'.join([markdown, '', family_note, '',
                      'Hypotheses, verbatim from `claims.paired_tests`:', ''] +
                     ['* `{}` / `{}` / `{}`: {}'.format(
                         row['unit'], row['contrast'], row['metric'],
                         row['hypothesis'])
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
    `joint-<delta> - joint-off`. Measuring alignment against `independent`
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
        'encoding. "alignment" is each swept delta minus joint-off, measured '
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

# The physical TCAM block boundary this deliverable exists to expose --
# `src/p4gen/build_p4_script.py:21`'s TERNARY_MATCHING_ENTRIES_PER_BLOCK.
# Not imported directly: build_p4_script pulls in sklearn at module level
# (the same reason appendix_6 imports scripts.capacity_ceiling lazily
# instead of at module scope), which this reporting module otherwise never
# needs. The value is stable -- it is a Tofino hardware constant, not a
# tunable -- so duplicating it as a literal here is safe.
_TCAM_ROWS_PER_BLOCK = 512

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
    `_TCAM_ROWS_PER_BLOCK` physical rows per block). `rounding_loss` is
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
    up in steps of `_TCAM_ROWS_PER_BLOCK` rows per block). The gap between
    them, `rounding_loss`, is descriptive only (D3) -- T12's question is
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
        'blocks did, i.e. quantization ate part of the saving. Pooled over '
        'every joint arm, M, split and k paired against {baseline} in this '
        'campaign: joint mapping removes {entries_pct:.1%} of table '
        'entries on average; the block column only moves by '
        '{blocks_pct:.1%}; the {gap_pct:.1%} gap between them is '
        'quantization, and the table below (faceted by k, D7; even k is '
        'still pooled into this sentence but gets no row of its own) shows '
        'where it concentrates. Descriptive only (D3): no p-value is '
        'reported here and none should be -- this is a mechanistic "where" '
        'question, not a "does it differ" one, and entries and blocks are '
        'too collinear for a second test to add information Table 4\'s '
        'blocks test does not already carry.'.format(
            baseline=baseline, block_size=_TCAM_ROWS_PER_BLOCK,
            entries_pct=overall_entries_saving,
            blocks_pct=overall_blocks_saving,
            gap_pct=overall_rounding_loss))

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
        'block-size rounding itself.'.format(_TCAM_ROWS_PER_BLOCK),
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
# The whole set
# ---------------------------------------------------------------------------

def render_all(df, output_dir=DEFAULT_FIGURE_DIR,
               ceiling_csv=DEFAULT_CEILING_CSV,
               baseline=claims.INDEPENDENT_ARM_SLUG,
               expected_family_size=None):
    """Render every §C.5 deliverable and return them in order.

    `ceiling_csv=None` omits deliverable 6 -- the capacity-ceiling appendix
    replays a measurement that either exists on disk or does not, and a
    campaign frame contains nothing from which it could be reconstructed.
    Every other deliverable comes from `df` alone.
    """
    deliverables = [
        figure_1_accuracy_vs_blocks(df, output_dir=output_dir,
                                    baseline=baseline),
        figure_2_delta_frontier(df, output_dir=output_dir, baseline=baseline),
        figure_3_substitution_scatter(df, output_dir=output_dir,
                                      baseline=baseline),
        table_4_paired_tests(df, output_dir=output_dir, baseline=baseline,
                             expected_family_size=expected_family_size),
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
    return tuple(sorted(deliverables, key=lambda item: item.number))
