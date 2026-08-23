"""P7c: the seven thesis deliverables (spec C.5).

Every test builds a synthetic frame with a known answer. The pilot cell's
real CSV does not exist yet, and a figure function that can only be
exercised on real data cannot be tested at all -- which is how the old
`plotting.py` reached 614 lines with no test and two `plt.show()` calls in
it.

What these tests deliberately assert, and why -- testing plots is awkward,
so the choice matters more than the count:

* **Nothing is averaged across the two tasks.** Every frame here gives App
  and DDoS deliberately different values whose MEAN is a third, distinct
  number; `_drawn_values` then harvests every coordinate any artist
  actually carries and the tests assert the mean never appears while both
  task values do. This is the one defect the whole rerun exists to fix, so
  it is asserted structurally (on the artists) rather than by reading code.
* **The numbers drawn are `claims.py`'s numbers.** Rather than trusting
  that a figure calls the right function, the tests recompute the expected
  coordinates from `claims.py` (or from a hand-computed constant injected
  into the frame) and compare against the artist data.
* **Panel counts follow the data.** The old module hardcoded a 3x3 grid
  for exactly nine k values; the tests run frames with unusual arm and k
  counts and assert the panel count tracks them.
* **The headless property.** Backend is Agg, the module never touches
  `pyplot` (so it can neither `show()` nor leak a figure into pyplot's
  manager), and a full render leaves global `rcParams` byte-identical.

What these tests deliberately do NOT assert: anything about how a figure
LOOKS -- colours, marker shapes, legend placement, tick formatting. Pinning
appearance would freeze cosmetic choices into the suite without protecting
any claim. Nor do they assert that a file merely appeared: every write
assertion also checks the file's content.
"""
import os

import numpy as np
import pandas as pd
import pytest

import matplotlib

from src.reporting import claims, figures
from src.reporting.claims import INDEPENDENT_ARM_SLUG, JOINT_ARM_SLUGS


# ---------------------------------------------------------------------------
# Frame builders -- the post-`load_campaign` column contract, nothing more.
# ---------------------------------------------------------------------------

_DELTA_BY_SLUG = {
    'independent': (float('nan'), False),
    'joint-off': (float('nan'), False),
    'joint-d000': (0.0, False),
    'joint-d002': (0.02, False),
    'joint-d005': (0.05, False),
    'joint-d010': (0.10, False),
    'joint-d020': (0.20, False),
    'joint-dinf': (float('nan'), True),
}

_FEATURES = ('Flow.IAT.Max', 'Flow.IAT.Min', 'Fwd.IAT.Mean',
             'Bwd.IAT.Mean', 'Min.Packet.Length', 'Max.Packet.Length')


def _feature_order(split, task):
    """A per-(split, task) feature order, so a test can tell a genuinely
    per-split elimination order from one that silently collapsed splits."""
    shift = split + (0 if task == 'app' else 3)
    return [_FEATURES[(i + shift) % len(_FEATURES)] for i in range(len(_FEATURES))]


def _features_at(split, task, k):
    """The k features still standing at k, given `_feature_order`: the
    elimination order is therefore order[k-1] first, order[0] retained."""
    return ';'.join(_feature_order(split, task)[:k])


def _row(arm_slug='joint-d005', M=25, split=0, k=5,
         acc_app=0.90, acc_ddos=0.50, blocks=40.0, stages=3.0,
         f1_app=None, f1_ddos=None):
    if f1_app is None:
        f1_app = acc_app - 0.02
    if f1_ddos is None:
        f1_ddos = acc_ddos - 0.02
    delta_num, is_inf = _DELTA_BY_SLUG[arm_slug]
    return {
        'arm_slug': arm_slug,
        'arm': 'independent' if arm_slug == INDEPENDENT_ARM_SLUG else 'joint',
        'method': 'single' if arm_slug == INDEPENDENT_ARM_SLUG else 'multi',
        'M': M, 'split': split, 'k': k,
        'acc_app': acc_app, 'acc_ddos': acc_ddos,
        'f1_app': f1_app, 'f1_ddos': f1_ddos,
        'blocks': blocks, 'stages': stages,
        'delta_align_num': delta_num, 'delta_align_is_inf': is_inf,
        'features_app': _features_at(split, 'app', k),
        'features_ddos': _features_at(split, 'ddos', k),
        'infeasible': '',
    }


_COLUMNS = list(_row().keys())


def _frame(rows):
    return pd.DataFrame(rows, columns=_COLUMNS)


# The two per-task accuracies below are deliberately far apart, so their
# mean is a third number that must appear nowhere in any artifact.
BASE_ACC_APP = 0.90
BASE_ACC_DDOS = 0.50
BASE_ACC_MEAN = (BASE_ACC_APP + BASE_ACC_DDOS) / 2.0    # 0.70


def _constant_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS,
                       n_splits=4, m_values=(25, 50), k_values=(4, 5),
                       joint_d_app=-0.10, joint_d_ddos=-0.10,
                       joint_d_blocks=-5.0):
    """Every arm x every (M, split, k) cell, with EXACTLY the injected
    per-task deltas on every joint arm.

    Constant deltas make every mean, median and confidence interval a
    number the test knows in closed form:

        d_blocks              = joint_d_blocks on every cell
        rel-error change app  = -joint_d_app / (1 - BASE_ACC_APP)
        rel-error change ddos = -joint_d_ddos / (1 - BASE_ACC_DDOS)

    and the two rel-error changes differ (the errors have different
    denominators), which is the whole point of never averaging them.
    """
    rows = []
    for arm in arms:
        joint = arm != INDEPENDENT_ARM_SLUG
        for M in m_values:
            for split in range(n_splits):
                for k in k_values:
                    rows.append(_row(
                        arm_slug=arm, M=M, split=split, k=k,
                        acc_app=BASE_ACC_APP + (joint_d_app if joint else 0.0),
                        acc_ddos=BASE_ACC_DDOS + (joint_d_ddos if joint else 0.0),
                        blocks=40.0 + (joint_d_blocks if joint else 0.0)))
    return _frame(rows)


def _spread_campaign(arms=(INDEPENDENT_ARM_SLUG, 'joint-off', 'joint-d005'),
                     n_splits=3, m_values=(25, 50, 75), k_values=(3, 4, 5),
                     seed=11):
    """A frame whose accuracies and blocks vary with (M, k, split), so the
    3-D Pareto front is non-trivial and the two tasks never coincide."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm_index, arm in enumerate(arms):
        joint = arm != INDEPENDENT_ARM_SLUG
        for M in m_values:
            for split in range(n_splits):
                for k in k_values:
                    rows.append(_row(
                        arm_slug=arm, M=M, split=split, k=k,
                        acc_app=0.70 + 0.02 * k + 0.01 * arm_index
                                + rng.normal(0, 0.002),
                        acc_ddos=0.40 + 0.03 * k - 0.01 * arm_index
                                 + rng.normal(0, 0.002),
                        blocks=float(M) - 3.0 * k - (2.0 if joint else 0.0)))
    return _frame(rows)


# ---------------------------------------------------------------------------
# Artist harvesting -- what a figure actually drew, not what it meant to.
# ---------------------------------------------------------------------------

def _drawn_values(figure):
    """Every finite coordinate carried by every artist on every axis.

    Covers lines (including the ones `errorbar` creates for caps and bars),
    scatter offsets, and LineCollection segments, because a quantity that
    must not be plotted must not reach ANY of them. NaNs are dropped:
    matplotlib fills absent error bars with NaN, and a NaN is not a value
    the reader sees.
    """
    values = []
    for ax in figure.axes:
        for line in ax.lines:
            values.extend(np.asarray(line.get_xdata(), dtype='float64').ravel())
            values.extend(np.asarray(line.get_ydata(), dtype='float64').ravel())
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets(), dtype='float64')
            if offsets.size:
                values.extend(offsets.ravel())
            segments = getattr(collection, 'get_segments', None)
            if segments is not None:
                for segment in segments():
                    values.extend(np.asarray(segment, dtype='float64').ravel())
    values = np.asarray(values, dtype='float64')
    return values[np.isfinite(values)]


def _lines_by_gid(figure, gid):
    return [line for ax in figure.axes for line in ax.lines
            if line.get_gid() == gid]


def _texts(figure):
    return [t.get_text() for ax in figure.axes for t in ax.texts] + \
           [ax.get_title() for ax in figure.axes] + \
           [ax.get_xlabel() for ax in figure.axes] + \
           [ax.get_ylabel() for ax in figure.axes]


def _contains(values, target, tol=1e-9):
    return bool(np.any(np.isclose(values, target, atol=tol, rtol=0.0)))


# ---------------------------------------------------------------------------
# Headlessness and global-state hygiene
# ---------------------------------------------------------------------------

def test_importing_the_figures_module_selects_the_headless_agg_backend():
    assert matplotlib.get_backend().lower() == 'agg'


def test_the_figures_module_never_reaches_for_pyplot_so_it_cannot_call_show():
    """A figure that never enters pyplot's manager can neither be shown nor
    leaked, and a module that never names `rcParams` or `style.use` cannot
    restyle anyone else's figures.

    Asserted on the parsed module rather than on a substring search (which
    the module docstring's own explanation of these rules would trip) AND
    rather than only on a before/after snapshot: this test module imports
    `figures` at its own import, so any snapshot taken inside a test body is
    taken after import-time mutation has already happened and would miss
    exactly the leak `plotting.py:7-8` demonstrates. The AST check catches it
    at the level the rule is stated.
    """
    import ast
    tree = ast.parse(open(figures.__file__, encoding='utf-8').read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or '')
            imported.update('{}.{}'.format(node.module or '', alias.name)
                            for alias in node.names)
        elif isinstance(node, ast.Attribute):
            # `matplotlib.use('Agg')` is the one sanctioned global call, so
            # `.use` is only rejected when it hangs off `.style`. Naming
            # `rcParams` at all is rejected: reading it is harmless, but the
            # module has no reason to, and the assignment form that is NOT
            # harmless is spelled the same way.
            assert node.attr not in ('show', 'rcParams')
            assert not (node.attr == 'use'
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == 'style')
        elif isinstance(node, ast.Name):
            # catches `from matplotlib import rcParams; rcParams[...] = ...`,
            # which carries no Attribute node to catch above
            assert node.id != 'rcParams'
    assert not any('pyplot' in name or 'seaborn' in name or 'rcParams' in name
                   for name in imported)


def test_rendering_every_deliverable_leaves_no_figure_in_pyplots_manager(tmp_path):
    """The old module built figures through pyplot, which keeps a global
    reference to every one of them; a batch run leaks them all."""
    import matplotlib.pyplot as plt
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    before = list(plt.get_fignums())
    figures.render_all(_spread_campaign(), output_dir=None,
                       ceiling_csv=csv_path)
    assert list(plt.get_fignums()) == before


def test_rendering_every_deliverable_leaves_global_rcparams_untouched(tmp_path):
    """`plotting.py` sets `font.family = 'Times New Roman'` globally at
    `:325` and `plt.style.use` / `sns.set_palette` at import (`:7-8`),
    silently restyling every later figure in the process. Appendix 6 reaches
    into `scripts/capacity_ceiling.py`, so this also pins that that import
    path does not drag the old module in behind it."""
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    before = dict(matplotlib.rcParams)
    figures.render_all(_spread_campaign(), output_dir=None,
                       ceiling_csv=csv_path)
    after = dict(matplotlib.rcParams)
    changed = {key for key in before
               if repr(before[key]) != repr(after.get(key))}
    assert changed == set()


# ---------------------------------------------------------------------------
# Deliverable 1 -- per-task accuracy vs blocks, two panels, all arms overlaid
# ---------------------------------------------------------------------------

def test_deliverable_1_facets_rows_by_task_and_columns_by_odd_k():
    """D7: facet at odd k only. `_spread_campaign()`'s default k values are
    (3, 4, 5) -- k=4 is even and gets no column, k=3 and k=5 do -- so the
    grid is (2 task rows + 1 trade-plane row) x 2 k columns (D8's trade
    plane gets its own row below the two task rows, one panel per shown
    k, rather than a fixed literal panel count)."""
    df = _spread_campaign()
    deliverable = figures.figure_1_accuracy_vs_blocks(df, output_dir=None)
    axes = deliverable.figure.axes
    assert len(axes) == (len(figures.TASKS) + 1) * 2
    labels = [ax.get_ylabel() for ax in axes]
    assert any('App' in label for label in labels)
    assert any('DDoS' in label for label in labels)
    titles = [ax.get_title() for ax in axes]
    assert any('k=3' in title for title in titles)
    assert any('k=5' in title for title in titles)
    assert not any('k=4' in title for title in titles)


def test_deliverable_1_has_a_trade_plane_panel_with_no_connecting_line():
    """D8: the only panel that can show WHY a front point exists when that
    point looks dominated in both memory panels. A line through this plane
    would imply an ordering that does not exist."""
    fig = figures.figure_1_accuracy_vs_blocks(_spread_campaign()).figure
    titles = [t for t in _texts(fig)]
    assert any('acc_ddos' in t and 'acc_app' in t for t in titles)
    assert _lines_by_gid(fig, 'trade-line:') == []


def test_deliverable_1_never_plots_the_average_of_the_two_task_accuracies():
    deliverable = figures.figure_1_accuracy_vs_blocks(
        _constant_campaign(), output_dir=None)
    values = _drawn_values(deliverable.figure)
    assert _contains(values, BASE_ACC_APP)
    assert _contains(values, BASE_ACC_DDOS)
    assert not _contains(values, BASE_ACC_MEAN)


def test_deliverable_1_overlays_every_arm_present_including_unknown_ones():
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG, 'joint-off',
                                'joint-d005', 'joint-dinf'))
    df = pd.concat([df, df[df.arm_slug == 'joint-off'].assign(
        arm_slug='joint-experimental')], ignore_index=True)
    deliverable = figures.figure_1_accuracy_vs_blocks(df, output_dir=None)
    # One front line per (task, odd-k) panel -- default k values (3, 4, 5)
    # give 2 odd k columns, so 2 tasks x 2 k columns = 4 lines per arm.
    n_odd_k = len({int(k) for k in df['k'].unique() if int(k) % 2 == 1})
    expected = len(figures.TASKS) * n_odd_k
    for arm in ('independent', 'joint-off', 'joint-d005', 'joint-dinf',
                'joint-experimental'):
        assert len(_lines_by_gid(deliverable.figure, 'front:' + arm)) == expected


def test_deliverable_1_draws_the_3d_pareto_front_claims_computed_not_a_2d_one():
    """A point excellent on App and poor on DDoS belongs on the 3-D front
    and vanishes from a per-plane 2-D front; drawing the plane's own front
    would hide exactly the trade the thesis is about. The front itself is
    computed ONCE, pooling every k for the arm (unaffected by faceting);
    each k column only shows that k's slice of the one pooled front, which
    is what this test checks for k=3, the first (smallest odd) column."""
    df = _spread_campaign()
    arm = 'joint-d005'
    k = 3
    pooled = claims.pareto_projections(
        claims.pareto_front_3d(df[df.arm_slug == arm]))['acc_app_vs_blocks']
    expected = pooled[pooled['k'] == k]

    deliverable = figures.figure_1_accuracy_vs_blocks(df, output_dir=None)
    # Row 0 (App) x column 0 (k=3, the first odd value in sweep order) is
    # `figure.axes[0]` in the row-major flattened axes list `subplots`
    # produces.
    app_k3_axis = deliverable.figure.axes[0]
    line = [line for line in app_k3_axis.lines
           if line.get_gid() == 'front:' + arm][0]
    assert np.allclose(np.sort(line.get_xdata()),
                       np.sort(expected['blocks'].to_numpy()))
    assert np.allclose(np.sort(line.get_ydata()),
                       np.sort(expected['acc_app'].to_numpy()))


def test_deliverable_1_keeps_a_low_accuracy_cell_at_k_17():
    """`plotting.py:356-361` drops any front point below 0.8 accuracy, but
    only when k == 17 -- a magic filter on a magic k."""
    df = _spread_campaign(k_values=(16, 17))
    df.loc[(df.k == 17) & (df.arm_slug == 'joint-d005'), 'acc_app'] = 0.31
    deliverable = figures.figure_1_accuracy_vs_blocks(df, output_dir=None)
    assert _contains(_drawn_values(deliverable.figure), 0.31)


def test_deliverable_1_caption_discloses_that_the_front_pools_splits_and_m():
    """Figure 1 is the headline deliverable, computing `claims.pareto_front_3d`
    over every (M, split, k) row of an arm at once -- a point from one split
    can dominate a point from another, so the resulting 'front' is not a
    front of anything replicated. Figure 2 carries three sentences of this
    disclosure; Figure 1 must carry at least one, matching that treatment,
    so the quoted coverage percentage is not read as a per-split result."""
    caption = figures.figure_1_accuracy_vs_blocks(
        _spread_campaign(), output_dir=None).caption.lower()
    assert 'pool' in caption
    assert 'split' in caption
    assert ('block budget' in caption or ' m ' in caption or 'm,' in caption)


def test_deliverable_1_reports_coverage_in_both_directions():
    """The paper commits in writing: 'C(A,B) != C(B,A), making it essential
    to evaluate both directions' (main.tex:593). Reporting one is a
    regression against a published methodological commitment."""
    deliverable = figures.figure_1_accuracy_vs_blocks(_spread_campaign())
    caption = deliverable.caption
    assert 'covers' in caption and 'covered by' in caption
    assert deliverable.data is not None
    assert {'coverage_of_baseline', 'coverage_by_baseline'} <= set(
        deliverable.data.columns)


def test_deliverable_1_writes_a_pdf_a_data_csv_and_a_caption(tmp_path):
    deliverable = figures.figure_1_accuracy_vs_blocks(
        _spread_campaign(), output_dir=str(tmp_path))
    suffixes = {os.path.splitext(p)[1] for p in deliverable.paths}
    assert suffixes == {'.pdf', '.csv', '.md'}
    for path in deliverable.paths:
        assert os.path.getsize(path) > 0
    written = pd.read_csv([p for p in deliverable.paths
                           if p.endswith('.csv')][0])
    assert 'acc_app' in written.columns and 'acc_ddos' in written.columns
    assert not any('avg' in column for column in written.columns)


def test_the_renderer_says_which_k_values_it_computed_but_did_not_show(capsys):
    """D7: a silently truncated facet reads as full coverage. Even-k rows
    still enter every paired test and every pooled statistic -- they just
    get no panel -- and the reader has to be told."""
    figures.figure_1_accuracy_vs_blocks(_spread_campaign(k_values=(3, 4, 5)))
    out = capsys.readouterr().out
    assert 'k=4' in out and 'not shown' in out


# ---------------------------------------------------------------------------
# Deliverable 2 -- the delta frontier
# ---------------------------------------------------------------------------

def test_deliverable_2_caption_states_the_two_comparability_facts():
    """Spec A.5: without both sentences the figure invites the objection
    that the arms are not comparable."""
    caption = figures.figure_2_delta_frontier(
        _constant_campaign(), output_dir=None).caption.lower()
    assert 'feature set' in caption
    assert 'by construction' in caption
    assert 'split' in caption and 'replicat' in caption
    assert 'varian' in caption


def test_deliverable_2_reports_relative_error_per_task_never_pooled():
    delta_app, delta_ddos = -0.10, -0.10
    deliverable = figures.figure_2_delta_frontier(
        _constant_campaign(joint_d_app=delta_app, joint_d_ddos=delta_ddos),
        output_dir=None)
    # Same accuracy drop, different error denominators -> different relative
    # error change. Averaging the tasks would erase precisely this.
    expected_app = -delta_app / (1.0 - BASE_ACC_APP)
    expected_ddos = -delta_ddos / (1.0 - BASE_ACC_DDOS)
    assert not np.isclose(expected_app, expected_ddos)

    table = deliverable.data
    app_means = table[table.metric == 'rel_error_change_app']['mean']
    ddos_means = table[table.metric == 'rel_error_change_ddos']['mean']
    assert np.allclose(app_means, expected_app)
    assert np.allclose(ddos_means, expected_ddos)

    values = _drawn_values(deliverable.figure)
    assert _contains(values, expected_app)
    assert _contains(values, expected_ddos)
    assert not _contains(values, (expected_app + expected_ddos) / 2.0)


def test_deliverable_2_caption_discloses_that_each_point_pools_over_m_only():
    """A point is an average over every M inside a split at a FIXED k, not
    one operating point -- an examiner reading a block saving off this
    figure has to know that before believing it applies at their budget.
    k, unlike M, is no longer pooled away (D7): the caption must name the
    individual k values shown as separate columns, not a collapsed range."""
    df = _constant_campaign(m_values=(25, 100), k_values=(5, 7))
    deliverable = figures.figure_2_delta_frontier(df, output_dir=None)
    caption = deliverable.caption
    assert 'pools' in caption.lower()
    assert '25, 100' in caption
    assert '5, 7' in caption
    assert '5-7' not in caption
    # and the table really has collapsed M away but kept k as a real column,
    # which is what the sentence is disclosing
    assert 'M' not in deliverable.data.columns
    assert 'k' in deliverable.data.columns
    assert set(deliverable.data['k']) == {5, 7}


def test_deliverable_2_pooling_sentence_reflects_the_joined_grid_not_the_raw_frame():
    """`--M` and `--n-splits` let a campaign be chunked and resumed
    (main.py's `skip_existing`), so a baseline run at M in {25, 40} can be
    paired against a joint arm run only at M=25 -- the inner join in
    `pair_arms` drops M=40 for that contrast. The pooling sentence must name
    the M/k values the figure actually averaged over (the joined grid), not
    every M/k anywhere in the raw frame, which here would wrongly include
    the M=40 the join silently discarded for the plotted arm."""
    rows = []
    for M in (25, 40):
        for split in range(3):
            for k in (4, 5):
                rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, M=M,
                                 split=split, k=k, acc_app=BASE_ACC_APP,
                                 acc_ddos=BASE_ACC_DDOS, blocks=40.0))
    for M in (25,):    # the joint arm only ran at M=25
        for split in range(3):
            for k in (4, 5):
                rows.append(_row(arm_slug='joint-d005', M=M, split=split, k=k,
                                 acc_app=BASE_ACC_APP - 0.10,
                                 acc_ddos=BASE_ACC_DDOS - 0.10, blocks=35.0))
    df = _frame(rows)

    caption = figures.figure_2_delta_frontier(
        df, output_dir=None, baseline=INDEPENDENT_ARM_SLUG).caption
    assert '25' in caption
    assert '40' not in caption


def test_deliverable_2_has_one_panel_per_reported_quantity_two_of_them_per_task():
    deliverable = figures.figure_2_delta_frontier(
        _constant_campaign(), output_dir=None)
    labels = [ax.get_ylabel() for ax in deliverable.figure.axes]
    assert len(deliverable.figure.axes) == 3
    assert len({label for label in labels}) == 3
    assert sum('App' in label for label in labels) == 1
    assert sum('DDoS' in label for label in labels) == 1
    assert sum('block' in label.lower() for label in labels) == 1


def test_deliverable_2_plots_the_block_saving_that_was_injected():
    deliverable = figures.figure_2_delta_frontier(
        _constant_campaign(joint_d_blocks=-7.0), output_dir=None)
    blocks = deliverable.data[deliverable.data.metric == 'd_blocks']
    assert np.allclose(blocks['mean'], -7.0)
    assert _contains(_drawn_values(deliverable.figure), -7.0)


def test_deliverable_2_uses_split_level_replication_for_its_confidence_interval():
    """Cells inside one split share a training split, so the interval must
    be built over split-level means -- `n` is the split count, not the cell
    count."""
    df = _constant_campaign(n_splits=4, m_values=(25, 50), k_values=(4, 5))
    deliverable = figures.figure_2_delta_frontier(df, output_dir=None)
    assert set(deliverable.data['n']) == {4}
    assert set(deliverable.data['n_splits']) == {4}


def test_deliverable_2_places_the_two_non_numeric_arms_without_inventing_a_delta():
    """`joint-off` (alignment never ran) and `joint-dinf` (accept-all) both
    carry a NaN parsed delta and must not be dropped or given a made-up
    numeric position."""
    deliverable = figures.figure_2_delta_frontier(
        _constant_campaign(), output_dir=None)
    ticks = [t.get_text() for t in deliverable.figure.axes[0].get_xticklabels()]
    assert 'inf' in ticks
    assert any('off' in tick for tick in ticks)
    assert set(deliverable.data['arm_slug']) == set(JOINT_ARM_SLUGS)


def test_deliverable_2_facets_by_k_instead_of_averaging_it_away():
    """The paper's conclusion is k-dependent -- joint dominates at k>=11,
    parity at 5-9, independent wins at k<=5 (main.tex:591). Averaging over k
    cannot reproduce the headline."""
    table = figures.delta_frontier_table(_spread_campaign(k_values=(3, 5)))
    assert 'k' in table.columns
    assert set(table['k']) == {3, 5}


# ---------------------------------------------------------------------------
# Deliverable 3 -- substitution scatter with quadrants, one panel per arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_arms', [2, 5, 7])
def test_deliverable_3_draws_one_panel_per_arm_whatever_the_arm_count(n_arms):
    """The old module hardcoded a 3x3 grid for exactly nine k values."""
    arms = (INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS[:n_arms]
    deliverable = figures.figure_3_substitution_scatter(
        _spread_campaign(arms=arms), output_dir=None)
    assert len(deliverable.figure.axes) == n_arms


def test_deliverable_3_scatters_the_per_task_deltas_claims_paired():
    df = _spread_campaign()
    arm = 'joint-d005'
    expected = claims.arm_deltas(df, arm, INDEPENDENT_ARM_SLUG)
    deliverable = figures.figure_3_substitution_scatter(df, output_dir=None)
    panel = [ax for ax in deliverable.figure.axes if arm in ax.get_title()][0]
    offsets = np.asarray(panel.collections[0].get_offsets(), dtype='float64')
    assert np.allclose(np.sort(offsets[:, 0]),
                       np.sort(expected['d_acc_app'].to_numpy()))
    assert np.allclose(np.sort(offsets[:, 1]),
                       np.sort(expected['d_acc_ddos'].to_numpy()))


def test_deliverable_3_annotates_the_correlation_and_quadrants_claims_computed():
    df = _spread_campaign()
    expected = claims.substitution_test_all_arms(df, INDEPENDENT_ARM_SLUG)
    deliverable = figures.figure_3_substitution_scatter(df, output_dir=None)
    texts = ' | '.join(_texts(deliverable.figure))
    for _, row in expected.iterrows():
        assert '{:.3f}'.format(row['pearson_r']) in texts
        assert '{:.2f}'.format(row['quadrant_app_up_ddos_down']) in texts
    assert set(deliverable.data['treatment']) == set(expected['treatment'])


def test_deliverable_3_marks_the_two_substitution_quadrants_with_both_axes():
    deliverable = figures.figure_3_substitution_scatter(
        _spread_campaign(), output_dir=None)
    for ax in deliverable.figure.axes:
        y_zeros = [line for line in ax.lines
                   if np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)]
        x_zeros = [line for line in ax.lines
                   if np.allclose(np.asarray(line.get_xdata(), dtype=float), 0.0)]
        assert y_zeros and x_zeros


def test_a_paired_figure_refuses_to_render_blank_when_the_baseline_is_absent():
    """With no baseline rows every (M, split, k) join is empty, so the figure
    would render complete but with nothing plotted -- the plausible-looking
    wrong artifact. `claims.paired_tests` already raises here; so do these."""
    df = _spread_campaign(arms=('joint-off', 'joint-d005'))
    for render in (figures.figure_2_delta_frontier,
                   figures.figure_3_substitution_scatter):
        with pytest.raises(ValueError, match='baseline'):
            render(df, output_dir=None)
    with pytest.raises(ValueError):
        figures.render_all(df, output_dir=None, ceiling_csv=None)


# ---------------------------------------------------------------------------
# Deliverables 4 and 5 -- the two tables
# ---------------------------------------------------------------------------

def test_deliverable_4_is_exactly_the_holm_corrected_table_claims_produced(tmp_path):
    """Ruling P7-3: the table carries BOTH units, each independently
    Holm-corrected against `claims.paired_tests`' own per-unit output."""
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS)
    expected_pair = claims.paired_tests(df, unit='pair')
    expected_split = claims.paired_tests(df, unit='split')
    deliverable = figures.table_4_paired_tests(df, output_dir=str(tmp_path))

    assert set(deliverable.data['unit']) == {'pair', 'split'}
    got_pair = deliverable.data[deliverable.data['unit'] == 'pair'].reset_index(drop=True)
    got_split = deliverable.data[deliverable.data['unit'] == 'split'].reset_index(drop=True)

    assert list(got_pair['contrast']) == list(expected_pair['contrast'])
    assert np.allclose(got_pair['p_holm'], expected_pair['p_holm'])
    assert np.allclose(got_pair['p_value'], expected_pair['p_value'])

    assert list(got_split['contrast']) == list(expected_split['contrast'])
    assert np.allclose(got_split['p_holm'], expected_split['p_holm'])
    assert np.allclose(got_split['p_value'], expected_split['p_value'])


def test_deliverable_4_keeps_the_two_tasks_on_separate_rows(tmp_path):
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS)
    deliverable = figures.table_4_paired_tests(df, output_dir=str(tmp_path))
    metrics = set(deliverable.data['metric'])
    assert {'acc_app', 'acc_ddos'} <= metrics
    assert not any('avg' in metric for metric in metrics)


def test_deliverable_4_markdown_records_how_many_comparisons_were_corrected(tmp_path):
    """A shrunken family weakens Holm for every comparison in it, so the
    count has to be on the face of the table, not implicit. The family size
    is per-unit (contrasts x metrics), not the doubled row count."""
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS[:3])
    deliverable = figures.table_4_paired_tests(df, output_dir=str(tmp_path))
    markdown = open([p for p in deliverable.paths if p.endswith('.md')][0],
                    encoding='utf-8').read()
    n_comparisons = int(deliverable.data.loc[
        deliverable.data['unit'] == 'pair', 'n_comparisons'].iloc[0])
    assert str(n_comparisons) in markdown
    assert str(claims.PRE_REGISTERED_FAMILY_SIZE) in markdown


def test_deliverable_4_passes_the_expected_family_size_through_to_claims(tmp_path):
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS[:3])
    with pytest.raises(ValueError):
        figures.table_4_paired_tests(
            df, output_dir=str(tmp_path),
            expected_family_size=claims.PRE_REGISTERED_FAMILY_SIZE)


def test_deliverable_5_is_the_ablation_decomposition_claims_produced(tmp_path):
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS)
    expected = claims.ablation_decomposition(df)
    deliverable = figures.table_5_ablation(df, output_dir=str(tmp_path))
    assert list(deliverable.data['contrast']) == list(expected['contrast'])
    assert np.allclose(deliverable.data['mean_diff_split_level'],
                       expected['mean_diff_split_level'], equal_nan=True)
    assert set(deliverable.data['component']) == {'sharing', 'alignment'}
    assert {'acc_app', 'acc_ddos'} <= set(deliverable.data['metric'])
    assert not any('avg' in metric for metric in deliverable.data['metric'])


def test_deliverable_5_markdown_renders_a_row_for_every_contrast(tmp_path):
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS)
    deliverable = figures.table_5_ablation(df, output_dir=str(tmp_path))
    markdown = open([p for p in deliverable.paths if p.endswith('.md')][0],
                    encoding='utf-8').read()
    body = [line for line in markdown.splitlines() if line.startswith('| ')]
    assert len(body) == len(deliverable.data) + 1    # + the header row
    for contrast in deliverable.data['contrast'].unique():
        assert contrast in markdown


# ---------------------------------------------------------------------------
# Deliverable 6 -- the capacity-ceiling appendix
# ---------------------------------------------------------------------------

def _synthetic_ceiling_csv(path, pruned_min_samples_leaf=None):
    """A capacity-ceiling CSV in `scripts/capacity_ceiling.py`'s own schema,
    covering its full grid so the appendix has every cell it renders.

    `pruned_min_samples_leaf`, when given, overrides the recorded value on
    every pruned-corner row -- simulating a CSV measured under a DIFFERENT
    `PRUNED.min_samples_leaf` than whatever the constant holds today, the
    scenario the printed header must render honestly."""
    from scripts.capacity_ceiling import (
        CORNERS, MAX_DEPTH_GRID, N_TREES_GRID, SPLIT_INDICES, cardinality_of)
    rows = []
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            for corner in CORNERS:
                for split_idx in SPLIT_INDICES:
                    # Codeword length grows with the box; the pruned corner
                    # stays well inside the limit, the large-tree corner runs
                    # over it at the top of the grid.
                    scale = 1 if corner.name == 'pruned' else 9
                    length = scale * n_trees * max_depth
                    within = length <= 512
                    leaf = corner.min_samples_leaf
                    if corner.name == 'pruned' and pruned_min_samples_leaf is not None:
                        leaf = pruned_min_samples_leaf
                    rows.append({
                        'n_trees': n_trees, 'max_depth': max_depth,
                        'cardinality': cardinality_of(n_trees, max_depth),
                        'corner': corner.name,
                        'min_samples_leaf': leaf,
                        'min_samples_split': corner.min_samples_split,
                        'split_idx': split_idx, 'split_seed': 42 + split_idx,
                        'joint_codeword_length': length,
                        'joint_within_limit': within,
                        'joint_stages': 2, 'joint_blocks': 5 * n_trees,
                        'disjoint_codeword_length_app': length,
                        'disjoint_codeword_length_ddos': length,
                        'disjoint_codeword_length': length,
                        'disjoint_within_limit': within,
                        'disjoint_stages': 2, 'disjoint_blocks': 8 * n_trees,
                        'seconds': 0.1})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_deliverable_6_renders_the_ceiling_appendix_without_remeasuring(
        tmp_path, monkeypatch):
    """The measurement takes ~10 minutes; the appendix reads the CSV it
    already wrote. `collect` is booby-trapped to prove it is never called."""
    import scripts.capacity_ceiling as capacity_ceiling

    def _explode(*args, **kwargs):
        raise AssertionError('the appendix must not re-run the measurement')

    monkeypatch.setattr(capacity_ceiling, 'collect', _explode)
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    deliverable = figures.appendix_6_capacity_ceiling(
        ceiling_csv=csv_path, output_dir=str(tmp_path))
    markdown = open([p for p in deliverable.paths if p.endswith('.md')][0],
                    encoding='utf-8').read()
    assert 'Adopted' in markdown
    assert 'cardinality' in markdown
    assert '512' in markdown


def test_deliverable_6_persists_the_markdown_the_script_only_ever_printed(tmp_path):
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    deliverable = figures.appendix_6_capacity_ceiling(
        ceiling_csv=csv_path, output_dir=str(tmp_path))
    assert deliverable.markdown_body.count('|') > 100
    assert len(deliverable.data) > 0


def test_deliverable_6_says_so_when_the_measurement_has_never_been_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        figures.appendix_6_capacity_ceiling(
            ceiling_csv=str(tmp_path / 'absent.csv'), output_dir=str(tmp_path))


def test_deliverable_6_pruned_corner_header_reflects_the_csv_not_the_constant(tmp_path):
    """`at_corner` selects rows by the string label 'pruned', not by
    matching `min_samples_leaf` -- so if `PRUNED.min_samples_leaf` changes
    after a CSV was measured, a header built from the CONSTANT would
    disagree with the rows it sits above (a CSV genuinely fit at one value,
    labelled with another). The header must be read off the data being
    rendered instead, so it is correct for whatever produced the file --
    this constructs exactly that mismatch and would fail against a header
    built from `PRUNED.min_samples_leaf` directly."""
    from scripts.capacity_ceiling import PRUNED
    stale_value = PRUNED.min_samples_leaf + 5    # not today's constant
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv',
                                      pruned_min_samples_leaf=stale_value)
    deliverable = figures.appendix_6_capacity_ceiling(
        ceiling_csv=csv_path, output_dir=str(tmp_path))
    markdown = deliverable.markdown_body
    assert 'min_samples_leaf={}'.format(stale_value) in markdown
    assert 'min_samples_leaf={}'.format(PRUNED.min_samples_leaf) not in markdown


# ---------------------------------------------------------------------------
# Deliverable 7 -- elimination order per split
# ---------------------------------------------------------------------------

def test_deliverable_7_recovers_the_elimination_order_of_each_split():
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,), n_splits=3,
                          m_values=(25,), k_values=(1, 2, 3, 4, 5, 6))
    deliverable = figures.appendix_7_elimination_order(df, output_dir=None)
    for split in (0, 1, 2):
        for task in ('app', 'ddos'):
            rows = deliverable.data[
                (deliverable.data.split == split)
                & (deliverable.data.task == task)
                & (deliverable.data.event == 'eliminated')
            ].sort_values('elimination_rank')
            expected = list(reversed(_feature_order(split, task)[1:]))
            assert list(rows['feature']) == expected


def test_deliverable_7_reports_the_two_tasks_separately_never_one_shared_order():
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,), n_splits=2,
                          m_values=(25,), k_values=(3, 4, 5))
    deliverable = figures.appendix_7_elimination_order(df, output_dir=None)
    assert set(deliverable.data['task']) == {'app', 'ddos'}
    app = deliverable.data[deliverable.data.task == 'app']
    ddos = deliverable.data[deliverable.data.task == 'ddos']
    assert list(app['feature']) != list(ddos['feature'])


def test_deliverable_7_flags_a_step_that_dropped_more_than_one_feature():
    """Infeasible rows are filtered at load, so a k can be missing; the
    relative order inside such a step is not recoverable and must not be
    invented."""
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,), n_splits=1,
                          m_values=(25,), k_values=(3, 4, 6))
    deliverable = figures.appendix_7_elimination_order(df, output_dir=None)
    gap = deliverable.data[deliverable.data.from_k == 6]
    assert set(gap['n_dropped_in_step']) == {2}
    assert gap['elimination_rank'].nunique() == 1


def test_deliverable_7_records_the_features_that_survived_to_the_smallest_k():
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,), n_splits=1,
                          m_values=(25,), k_values=(2, 3, 4))
    deliverable = figures.appendix_7_elimination_order(df, output_dir=None)
    retained = deliverable.data[(deliverable.data.event == 'retained')
                                & (deliverable.data.task == 'app')]
    assert set(retained['feature']) == set(_feature_order(0, 'app')[:2])


def test_deliverable_7_writes_one_row_per_arm_M_and_split(tmp_path):
    df = _spread_campaign(arms=(INDEPENDENT_ARM_SLUG, 'joint-d005'),
                          n_splits=2, m_values=(25, 50), k_values=(3, 4, 5))
    deliverable = figures.appendix_7_elimination_order(df,
                                                       output_dir=str(tmp_path))
    written = pd.read_csv([p for p in deliverable.paths
                           if p.endswith('.csv')][0])
    assert {'arm_slug', 'M', 'split', 'task', 'feature'} <= set(written.columns)
    assert written.groupby(['arm_slug', 'M', 'split', 'task']).ngroups == 2 * 2 * 2 * 2


# ---------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------

def test_render_all_produces_the_seven_deliverables_numbered_one_to_seven(tmp_path):
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    deliverables = figures.render_all(
        _spread_campaign(arms=(INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS),
        output_dir=str(tmp_path), ceiling_csv=csv_path)
    assert [d.number for d in deliverables] == [1, 2, 3, 4, 5, 6, 7]
    for deliverable in deliverables:
        assert deliverable.paths
        for path in deliverable.paths:
            assert os.path.getsize(path) > 0


def test_render_all_writes_nothing_when_no_output_directory_is_given(tmp_path):
    deliverables = figures.render_all(_spread_campaign(), output_dir=None,
                                      ceiling_csv=None)
    assert os.listdir(str(tmp_path)) == []
    assert all(d.paths == () for d in deliverables)
    # ceiling_csv=None means "the measurement is not available here", and the
    # appendix is then omitted rather than fabricated.
    assert [d.number for d in deliverables] == [1, 2, 3, 4, 5, 7]


def test_render_all_never_draws_the_average_of_the_two_task_accuracies(tmp_path):
    csv_path = _synthetic_ceiling_csv(tmp_path / 'capacity_ceiling.csv')
    deliverables = figures.render_all(_constant_campaign(),
                                      output_dir=str(tmp_path),
                                      ceiling_csv=csv_path)
    for deliverable in deliverables:
        if deliverable.figure is None:
            continue
        assert not _contains(_drawn_values(deliverable.figure), BASE_ACC_MEAN)


def test_the_log_helper_prints_to_stdout(capsys):
    """The `_log()` helper wraps print so Task 16 has a single call site
    to use for announcing dropped facets."""
    figures._log('test message')
    out = capsys.readouterr().out
    assert 'test message' in out
