"""P7c: the seven thesis deliverables (spec C.5).

Everything upstream of this module exists to make these seven artifacts
correct. `campaign_data.load_campaign` supplies the frame, `claims.py`
supplies every statistic, and this module does one thing: render them.

| # | artifact | answers |
|---|---|---|
| 1 | per-task accuracy vs blocks, two panels, all arms overlaid | the headline comparison, never averaged |
| 2 | delta frontier: d blocks and d rel-error per task vs delta, mean +/- CI | what the tolerance buys |
| 3 | substitution scatter with quadrants, one panel per arm | the reviewer's objection, answered directly |
| 4 | paired per-task test table, Holm-corrected | significance |
| 5 | ablation table: constraint cost vs alignment cost | where the savings come from |
| 6 | appendix: capacity-ceiling rederivation (B.7) | replaces "chosen manually" |
| 7 | appendix: elimination order per split | reproducibility |

Four and five are tables and six is a replay, so `Deliverable` carries a
`figure` that is None for those; every deliverable is independently
callable, because `main.py --mode plot` (P7d) and the pilot cell both need
to invoke them one at a time.

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

# Figure 2's three reported quantities: two per-task relative-error changes
# and the block delta. Deliberately three panels rather than two, so the
# tasks never share an axis.
REL_ERROR_APP = 'rel_error_change_app'
REL_ERROR_DDOS = 'rel_error_change_ddos'
BLOCKS_DELTA = 'd_blocks'
FRONTIER_METRICS = (REL_ERROR_APP, REL_ERROR_DDOS, BLOCKS_DELTA)

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
# Deliverable 1 -- per-task accuracy vs blocks, two panels, all arms overlaid
# ---------------------------------------------------------------------------

def figure_1_accuracy_vs_blocks(df, output_dir=DEFAULT_FIGURE_DIR,
                                baseline=claims.INDEPENDENT_ARM_SLUG):
    """Two panels -- App accuracy vs blocks, DDoS accuracy vs blocks -- with
    every arm overlaid in both.

    Each arm contributes its own 3-D Pareto front, computed by
    `claims.pareto_front_3d` on `(acc_app, acc_ddos, -blocks)` and shown
    through `claims.pareto_projections`. The drawn line is a PROJECTION of
    that 3-D front, not a front recomputed inside the plane: a point can
    look dominated in one panel and still be non-dominated overall, and
    dropping it would hide exactly the App-versus-DDoS trade the thesis is
    about. Every cell is also scattered faintly behind the fronts, so the
    front is visibly a subset of the data rather than the only data shown.

    There is no averaged accuracy anywhere in this figure, which is the
    whole reason it has two panels.
    """
    arms = ordered_arms(df, baseline=baseline)
    styles = _arm_styles(arms)

    figure = _make_figure(1, len(TASKS))
    axes = figure.subplots(1, len(TASKS), squeeze=False)[0]

    front_frames = []
    coverage = {}
    baseline_front = None
    if baseline in arms:
        baseline_front = claims.pareto_front_3d(df[df['arm_slug'] == baseline])

    for arm in arms:
        arm_rows = df[df['arm_slug'] == arm]
        front = claims.pareto_front_3d(arm_rows)
        projections = claims.pareto_projections(front)
        colour, marker = styles[arm]

        for axis, (_, accuracy_column, _, _, _) in zip(axes, TASKS):
            plane = projections['{}_vs_blocks'.format(accuracy_column)]
            axis.scatter(arm_rows['blocks'], arm_rows[accuracy_column],
                         s=10, alpha=0.25, color=colour, linewidths=0,
                         gid='cells:{}'.format(arm))
            axis.plot(plane['blocks'], plane[accuracy_column],
                      marker=marker, color=colour, linewidth=1.6,
                      markersize=5, label=arm, gid='front:{}'.format(arm))

        carried = [column for column in
                   ('arm_slug', 'M', 'split', 'k', 'blocks', 'stages',
                    'acc_app', 'acc_ddos')
                   if column in front.columns]
        front_frames.append(front.loc[:, carried])

        if baseline_front is not None and arm != baseline:
            coverage[arm] = claims.coverage_ratio_3d(front, baseline_front)

    for axis, (_, _, short_name, panel_title, _) in zip(axes, TASKS):
        axis.set_xlabel('TCAM blocks')
        axis.set_ylabel('{} accuracy'.format(short_name))
        axis.set_title(panel_title)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize='small', title='arm')
    figure.tight_layout()

    coverage_sentence = ''
    if coverage:
        coverage_sentence = (
            ' Coverage ratio (Zitzler C, 3-D, strict) of the {} front by each '
            'joint arm: {}.'.format(
                baseline,
                ', '.join('{} {:.0%}'.format(arm, value)
                          for arm, value in coverage.items())))

    caption = (
        'Per-task accuracy against TCAM blocks, one panel per task, with '
        'every arm of the sweep overlaid. The two tasks are reported '
        'separately and are never averaged: a single mean accuracy hides a '
        'model that is excellent on one task and unusable on the other. '
        'Faint points are all (M, split, k) cells; the joined markers are '
        'each arm\'s Pareto front, computed in 3-D on '
        '(acc_app, acc_ddos, -blocks) and PROJECTED into each panel -- a '
        'projected point may look dominated within its panel while being '
        'non-dominated overall, and removing it would hide the very trade '
        'between the two tasks this figure exists to show.{}'.format(
            coverage_sentence))

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
            REL_ERROR_APP: _relative_error_change(
                paired['acc_app_baseline'], paired['acc_app_treatment']),
            REL_ERROR_DDOS: _relative_error_change(
                paired['acc_ddos_baseline'], paired['acc_ddos_treatment']),
        })
        merged = deltas.merge(relative, on=['M', 'split', 'k'], how='inner',
                              validate='one_to_one')
        merged.insert(0, 'arm_slug', arm)
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=['arm_slug', 'M', 'split', 'k']
                            + list(FRONTIER_METRICS))
    long = pd.concat(frames, ignore_index=True)

    delta_columns = [column for column in
                     ('delta_align_num', 'delta_align_is_inf')
                     if column in df.columns]
    if delta_columns:
        mapping = df.loc[:, ['arm_slug'] + delta_columns].drop_duplicates()
        long = long.merge(mapping, on='arm_slug', how='left')
    return long


def delta_frontier_table(df, baseline=claims.INDEPENDENT_ARM_SLUG,
                         confidence=0.95, arms=None):
    """Mean and confidence interval per arm for each of figure 2's three
    quantities, aggregated over SPLITS.

    Each split is collapsed to its own mean difference first, so the
    interval `claims.delta_frontier` then builds has exactly one
    observation per split. Feeding it the raw cells instead would put many
    correlated cells from one training split into the same interval and make
    it too narrow -- which is why `delta_frontier` refuses that shape
    outright unless the caller says it means it. The split-level convention
    matches `claims.ablation_decomposition`.
    """
    long = paired_delta_frame(df, baseline=baseline, arms=arms)
    if len(long) == 0:
        return long

    group_columns = ['arm_slug', 'split']
    split_means = long.groupby(group_columns, as_index=False)[
        list(FRONTIER_METRICS)].mean()

    delta_columns = [column for column in
                     ('delta_align_num', 'delta_align_is_inf')
                     if column in long.columns]
    if delta_columns:
        mapping = long.loc[:, ['arm_slug'] + delta_columns].drop_duplicates()
        split_means = split_means.merge(mapping, on='arm_slug', how='left')

    return claims.delta_frontier(
        split_means, metrics=FRONTIER_METRICS, group_columns=('arm_slug',),
        confidence=confidence)


def figure_2_delta_frontier(df, output_dir=DEFAULT_FIGURE_DIR,
                            baseline=claims.INDEPENDENT_ARM_SLUG,
                            confidence=0.95):
    """What the alignment tolerance buys: block saving and per-task relative
    error change against delta, mean +/- CI across splits.

    Three panels, because the two tasks get one each and pooling them would
    reintroduce the defect this rerun exists to fix. The x axis is
    categorical in sweep order rather than numeric: `joint-off` (alignment
    never ran) and `joint-dinf` (accept every move) are anchors, not
    numbers, and placing them on a numeric axis would require inventing
    coordinates for them.
    """
    table = delta_frontier_table(df, baseline=baseline, confidence=confidence)
    arms = [arm for arm in ordered_arms(df, include_baseline=False,
                                        baseline=baseline)
            if arm in set(table['arm_slug'])] if len(table) else []
    positions = {arm: index for index, arm in enumerate(arms)}

    labels = {
        REL_ERROR_APP: 'App: rel. error change vs {}'.format(baseline),
        REL_ERROR_DDOS: 'DDoS: rel. error change vs {}'.format(baseline),
        BLOCKS_DELTA: 'TCAM blocks: change vs {}'.format(baseline),
    }

    # One tick label per arm, built once from the parsed delta columns
    # `claims.delta_frontier` carried through -- never from the raw
    # `delta_align` string, which must not be ordered or compared.
    # `arms` is empty whenever `table` is, so these lookups only ever run on
    # a populated table.
    per_arm = table.drop_duplicates('arm_slug').set_index('arm_slug')         if len(table) else table
    tick_labels = [
        _delta_tick_label(
            arm,
            per_arm['delta_align_num'].get(arm, np.nan)
            if 'delta_align_num' in per_arm.columns else np.nan,
            per_arm['delta_align_is_inf'].get(arm, False)
            if 'delta_align_is_inf' in per_arm.columns else False)
        for arm in arms]
    x = [positions[arm] for arm in arms]

    figure = _make_figure(1, len(FRONTIER_METRICS))
    axes = figure.subplots(1, len(FRONTIER_METRICS), squeeze=False)[0]

    for axis, metric in zip(axes, FRONTIER_METRICS):
        rows = table[table['metric'] == metric] if len(table) else table
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
        axis.grid(True, alpha=0.3)
    figure.tight_layout()

    caption = (
        'The alignment tolerance sweep: block change and per-task relative '
        'error change against delta, each point a mean over splits with a '
        '{:.0%} Student-t confidence interval, paired against the {} arm on '
        '(M, split, k). The two tasks are shown on separate panels and are '
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
        'than being assumed away.'.format(confidence, baseline))

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
    'contrast', 'metric', 'alternative', 'n_pairs', 'n_splits',
    'median_diff', 'mean_diff_split_level', 'ci_low', 'ci_high',
    'p_value', 'p_holm', 'significant_holm')


def table_4_paired_tests(df, output_dir=DEFAULT_FIGURE_DIR,
                         baseline=claims.INDEPENDENT_ARM_SLUG,
                         margin=0.0, alpha=0.05, unit='pair',
                         expected_family_size=None):
    """The pre-registered paired tests, Holm-corrected -- rendered, not
    recomputed. Every number is `claims.paired_tests`'.

    One row per (contrast, metric), and `acc_app` and `acc_ddos` are
    separate rows throughout: there is no pooled accuracy test, because a
    pooled test is exactly what let a loss on one task hide behind a gain on
    the other.

    `expected_family_size` is passed straight through and defaults to None
    so a partial campaign (the pilot cell) still produces a table. That is a
    real weakening -- Holm over 9 comparisons is a laxer correction than
    Holm over the pre-registered 21 -- so the rendered markdown always
    states how many comparisons were actually corrected over and what the
    pre-registered family size is. Pass
    `expected_family_size=claims.PRE_REGISTERED_FAMILY_SIZE` on the complete
    campaign to turn a shrunken family into an error.
    """
    table = claims.paired_tests(
        df, baseline=baseline, metrics=claims.DEFAULT_METRICS, margin=margin,
        alpha=alpha, unit=unit, expected_family_size=expected_family_size)

    n_comparisons = len(table)
    family_note = (
        '{} comparisons were Holm-corrected here; the pre-registered family '
        'is {} (7 joint arms x 3 tests). {}'.format(
            n_comparisons, claims.PRE_REGISTERED_FAMILY_SIZE,
            'The family is complete.'
            if n_comparisons == claims.PRE_REGISTERED_FAMILY_SIZE else
            'The family is INCOMPLETE, so this correction is weaker than the '
            'pre-registered one and the adjusted p-values below are '
            'correspondingly optimistic.'))

    caption = (
        'Paired Wilcoxon signed-rank tests, one per (contrast, task) and one '
        'per contrast on blocks, over cells paired on (M, split, k). The '
        'accuracy tests are one-sided with alternative "greater" applied to '
        '{}, so a small p-value is the positive finding: the joint arm shows '
        'no detectable loss. The block test is two-sided, because alignment '
        'adds intervals before it merges any and sharing can cost blocks as '
        'well as save them. p_holm is Holm-Bonferroni across the whole '
        'family. {} Cells within a split share a training split, so the '
        'p-values are anti-conservative relative to the number of '
        'independent splits; n_splits is reported beside n_pairs so the gap '
        'is visible.'.format(
            'd + {:g}'.format(margin) if margin > 0 else 'd', family_note))

    markdown = _markdown_table(
        table.loc[:, [column for column in _PAIRED_TEST_MARKDOWN_COLUMNS
                      if column in table.columns]])
    body = '\n'.join([markdown, '', family_note, '',
                      'Hypotheses, verbatim from `claims.paired_tests`:', ''] +
                     ['* `{}` / `{}`: {}'.format(row['contrast'], row['metric'],
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
        table.loc[:, [column for column in _ABLATION_MARKDOWN_COLUMNS
                      if column in table.columns]])

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
    return tuple(sorted(deliverables, key=lambda item: item.number))
