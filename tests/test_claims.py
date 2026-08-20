"""P7b: the statistical claims layer (spec C.3).

Every test here builds a synthetic frame with a known answer -- the pilot
cell's real CSV does not exist yet, and a claim function that can only be
checked against real data cannot be checked at all.

The frames are built directly in the post-`load_campaign` column contract
(see `src/reporting/campaign_data.py`'s module docstring), because that is
what `claims.py` consumes: `arm_slug` as the real per-arm identity,
`(M, split, k)` as the pairing key, and numeric `acc_app` / `acc_ddos` /
`blocks`.
"""
import numpy as np
import pandas as pd
import pytest

from src.reporting.claims import (
    INDEPENDENT_ARM_SLUG,
    JOINT_ARM_SLUGS,
    PRE_REGISTERED_FAMILY_SIZE,
    ablation_decomposition,
    arm_deltas,
    coverage_ratio_3d,
    default_contrast_family,
    delta_frontier,
    holm_bonferroni,
    paired_tests,
    pareto_front_3d,
    pareto_projections,
    substitution_test,
    substitution_test_all_arms,
)


# ---------------------------------------------------------------------------
# Frame builders -- the post-load column contract, nothing more.
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


def _row(arm_slug='joint-d005', M=25, split=0, k=5,
         acc_app=0.90, acc_ddos=0.85, blocks=40.0, stages=3.0):
    delta_num, is_inf = _DELTA_BY_SLUG[arm_slug]
    return {
        'arm_slug': arm_slug,
        'arm': 'independent' if arm_slug == 'independent' else 'joint',
        'method': 'single' if arm_slug == 'independent' else 'multi',
        'M': M, 'split': split, 'k': k,
        'acc_app': acc_app, 'acc_ddos': acc_ddos,
        'blocks': blocks, 'stages': stages,
        'delta_align_num': delta_num, 'delta_align_is_inf': is_inf,
    }


_COLUMNS = list(_row().keys())


def _frame(rows):
    """An empty frame still carries the full column contract -- an empty
    campaign is a frame with no rows, never a frame with no columns."""
    return pd.DataFrame(rows, columns=_COLUMNS)


def _points_frame(points):
    """points: iterable of (acc_app, acc_ddos, blocks)."""
    return _frame([_row(split=i, acc_app=a, acc_ddos=d, blocks=b)
                   for i, (a, d, b) in enumerate(points)])


def _paired_frame(d_app, d_ddos, d_blocks, treatment='joint-d005',
                  baseline=INDEPENDENT_ARM_SLUG, M=25, k=5):
    """One baseline row and one treatment row per observation, differing by
    exactly the requested delta, so the deltas `claims.py` recovers are the
    ones injected."""
    base_app, base_ddos, base_blocks = 0.90, 0.85, 40.0
    rows = []
    for i, (da, dd, db) in enumerate(zip(d_app, d_ddos, d_blocks)):
        rows.append(_row(arm_slug=baseline, M=M, split=i, k=k,
                         acc_app=base_app, acc_ddos=base_ddos, blocks=base_blocks))
        rows.append(_row(arm_slug=treatment, M=M, split=i, k=k,
                         acc_app=base_app + da, acc_ddos=base_ddos + dd,
                         blocks=base_blocks + db))
    return _frame(rows)


def _correlated_pair(rho, n, seed):
    """Draw (x, y) with population correlation exactly `rho`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + np.sqrt(1.0 - rho ** 2) * z
    return x, y


def _full_campaign_frame(n_splits=4, m_values=(25, 50), k_values=(5, 9),
                         seed=7):
    """Every arm x every (M, split, k) cell -- the shape the real campaign
    produces, so family-size assertions mean something."""
    rng = np.random.default_rng(seed)
    rows = []
    for slug in (INDEPENDENT_ARM_SLUG,) + JOINT_ARM_SLUGS:
        # joint arms are given a small genuine block saving and a tiny
        # accuracy edge, so directions in the assertions are unambiguous.
        joint = slug != INDEPENDENT_ARM_SLUG
        for M in m_values:
            for split in range(n_splits):
                for k in k_values:
                    rows.append(_row(
                        arm_slug=slug, M=M, split=split, k=k,
                        acc_app=0.90 + (0.01 if joint else 0.0) + rng.normal(0, 0.001),
                        acc_ddos=0.85 + (0.01 if joint else 0.0) + rng.normal(0, 0.001),
                        blocks=40.0 - (5.0 if joint else 0.0) + rng.normal(0, 0.2)))
    return _frame(rows)


# ---------------------------------------------------------------------------
# pareto_front_3d
# ---------------------------------------------------------------------------

# Hand-computed non-dominated set. Objectives: maximize acc_app, maximize
# acc_ddos, minimize blocks.
#   A (0.90, 0.80, 40) -- non-dominated
#   B (0.85, 0.85, 30) -- non-dominated (fewest blocks, best acc_ddos)
#   C (0.80, 0.75, 50) -- dominated by A on all three
#   D (0.90, 0.80, 45) -- dominated by A (equal accuracies, more blocks)
#   E (0.95, 0.70, 60) -- non-dominated (best acc_app)
_A = (0.90, 0.80, 40.0)
_B = (0.85, 0.85, 30.0)
_C = (0.80, 0.75, 50.0)
_D = (0.90, 0.80, 45.0)
_E = (0.95, 0.70, 60.0)


def test_pareto_front_3d_returns_exactly_the_hand_computed_non_dominated_set():
    front = pareto_front_3d(_points_frame([_A, _B, _C, _D, _E]))

    assert sorted(map(tuple, front[['acc_app', 'acc_ddos', 'blocks']].to_numpy())) == \
        sorted([_A, _B, _E])


def test_pareto_front_3d_drops_a_point_beaten_only_on_blocks_with_equal_accuracies():
    """D differs from A only by using five more blocks. Weak dominance on the
    two accuracy axes plus a strict win on the third is still dominance."""
    front = pareto_front_3d(_points_frame([_A, _D]))

    assert len(front) == 1
    assert front.iloc[0]['blocks'] == 40.0


def test_pareto_front_3d_keeps_a_point_that_a_per_task_2d_front_would_discard():
    """E is terrible on DDoS but best on App. A 2-D (blocks, acc_ddos) front
    would drop it; the 3-D front must not, because the trade it represents is
    exactly what the thesis is measuring."""
    front = pareto_front_3d(_points_frame([_A, _E]))

    assert len(front) == 2


def test_pareto_front_3d_keeps_both_copies_of_an_exactly_duplicated_point():
    """Identical points do not dominate each other (dominance requires a
    strict win somewhere), so neither may be silently dropped."""
    front = pareto_front_3d(_points_frame([_A, _A, _C]))

    assert len(front) == 2


def test_pareto_front_3d_raises_rather_than_absorbing_a_nan_objective():
    """Every NaN comparison is False, so a NaN point is never dominated and
    would land on EVERY front. load_campaign filters these out, but the front
    must not depend on that having happened."""
    df = _points_frame([_A, _C])
    df.loc[1, 'acc_ddos'] = float('nan')

    with pytest.raises(ValueError, match='(?i)nan'):
        pareto_front_3d(df)


def test_pareto_front_3d_raises_on_an_infinite_objective_too():
    df = _points_frame([_A, _C])
    df.loc[1, 'blocks'] = float('inf')

    with pytest.raises(ValueError):
        pareto_front_3d(df)


def test_pareto_front_3d_preserves_the_identity_columns_for_plotting():
    front = pareto_front_3d(_points_frame([_A, _B, _C, _D, _E]))

    assert 'arm_slug' in front.columns
    assert 'split' in front.columns


def test_pareto_front_3d_on_an_empty_frame_returns_an_empty_frame():
    front = pareto_front_3d(_points_frame([]))

    assert len(front) == 0


# ---------------------------------------------------------------------------
# pareto_projections
# ---------------------------------------------------------------------------

def test_pareto_projections_expose_all_three_planes():
    front = pareto_front_3d(_points_frame([_A, _B, _E]))

    projections = pareto_projections(front)

    assert set(projections) == {'acc_app_vs_blocks', 'acc_ddos_vs_blocks',
                                'acc_ddos_vs_acc_app'}


def test_pareto_projections_keep_points_a_2d_front_in_that_plane_would_drop():
    """E projects to (60 blocks, 0.70 acc_ddos), which A's (40, 0.80) beats in
    that plane. The projection is of the 3-D front, not a recomputed 2-D
    front, so E must still be there."""
    front = pareto_front_3d(_points_frame([_A, _B, _E]))

    plane = pareto_projections(front)['acc_ddos_vs_blocks']

    assert len(plane) == 3
    assert (plane['acc_ddos'] == 0.70).any()


def test_pareto_projections_are_sorted_on_the_x_axis_so_a_line_plot_is_valid():
    front = pareto_front_3d(_points_frame([_A, _B, _E]))

    plane = pareto_projections(front)['acc_app_vs_blocks']

    assert list(plane['blocks']) == sorted(plane['blocks'])


# ---------------------------------------------------------------------------
# coverage_ratio_3d
# ---------------------------------------------------------------------------

def test_coverage_ratio_3d_is_one_when_every_point_of_b_is_dominated():
    a = _points_frame([_A, _B, _E])
    b = _points_frame([_C, _D])

    assert coverage_ratio_3d(a, b) == 1.0


def test_coverage_ratio_3d_is_zero_in_the_other_direction():
    a = _points_frame([_A, _B, _E])
    b = _points_frame([_C, _D])

    assert coverage_ratio_3d(b, a) == 0.0


def test_coverage_ratio_3d_counts_each_dominated_point_of_b_once():
    a = _points_frame([_A])
    b = _points_frame([_C, _D, _E])

    assert coverage_ratio_3d(a, b) == pytest.approx(2.0 / 3.0)


def test_coverage_ratio_3d_of_a_set_against_itself_is_zero_under_strict_dominance():
    """Strict dominance, deliberately: a point cannot dominate its own copy,
    so C(A, A) = 0 reads as `A does not beat itself`. The weak-dominance
    variant of the C metric would report 1.0 here."""
    a = _points_frame([_A, _B, _E])

    assert coverage_ratio_3d(a, a) == 0.0


def test_coverage_ratio_3d_is_nan_when_b_is_empty_rather_than_a_misleading_zero():
    a = _points_frame([_A])

    assert np.isnan(coverage_ratio_3d(a, _points_frame([])))


def test_coverage_ratio_3d_raises_rather_than_absorbing_a_nan_point():
    a = _points_frame([_A])
    b = _points_frame([_C, _D])
    b.loc[0, 'acc_app'] = float('nan')

    with pytest.raises(ValueError):
        coverage_ratio_3d(a, b)


# ---------------------------------------------------------------------------
# arm_deltas / pairing
# ---------------------------------------------------------------------------

def test_arm_deltas_pairs_on_M_split_k_and_drops_a_deliberately_missing_cell():
    rows = []
    for M in (25, 50):
        for split in (0, 1):
            for k in (5, 9):
                rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, M=M, split=split, k=k))
                if not (M == 50 and split == 1 and k == 9):
                    rows.append(_row(arm_slug='joint-d005', M=M, split=split, k=k))

    deltas = arm_deltas(_frame(rows), 'joint-d005', INDEPENDENT_ARM_SLUG)

    assert len(deltas) == 7
    assert not ((deltas['M'] == 50) & (deltas['split'] == 1) & (deltas['k'] == 9)).any()


def test_arm_deltas_does_not_collapse_cells_that_differ_only_in_M():
    """The legacy perform_statistical_analysis keyed on (split, k) alone and
    silently merged the seven M files, last-wins."""
    rows = []
    for M in (25, 50, 75):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, M=M, split=0, k=5, blocks=40.0))
        rows.append(_row(arm_slug='joint-d005', M=M, split=0, k=5, blocks=30.0))

    deltas = arm_deltas(_frame(rows), 'joint-d005', INDEPENDENT_ARM_SLUG)

    assert len(deltas) == 3
    assert sorted(deltas['M']) == [25, 50, 75]


def test_arm_deltas_signs_the_difference_as_treatment_minus_baseline():
    df = _paired_frame([0.02], [-0.01], [-5.0])

    deltas = arm_deltas(df, 'joint-d005', INDEPENDENT_ARM_SLUG)

    assert deltas['d_acc_app'].iloc[0] == pytest.approx(0.02)
    assert deltas['d_acc_ddos'].iloc[0] == pytest.approx(-0.01)
    assert deltas['d_blocks'].iloc[0] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# substitution_test
# ---------------------------------------------------------------------------

def test_substitution_test_detects_an_injected_correlation_of_minus_zero_point_eight():
    x, y = _correlated_pair(-0.8, n=300, seed=11)
    df = _paired_frame(0.01 * x, 0.01 * y, np.zeros(300))

    result = substitution_test(df, 'joint-d005')

    assert result['pearson_r'] == pytest.approx(-0.8, abs=0.08)
    assert result['spearman_rho'] < -0.6
    assert result['substitution_detected'] is True


def test_substitution_test_does_not_fire_when_the_two_task_deltas_are_independent():
    x, y = _correlated_pair(0.0, n=300, seed=12)
    df = _paired_frame(0.01 * x, 0.01 * y, np.zeros(300))

    result = substitution_test(df, 'joint-d005')

    assert abs(result['pearson_r']) < 0.15
    assert result['substitution_detected'] is False


def test_substitution_test_does_not_fire_on_a_strong_positive_correlation():
    """A one-sided test for substitution must not be triggered by the two
    tasks improving together -- that is the opposite finding."""
    x, y = _correlated_pair(0.8, n=300, seed=13)
    df = _paired_frame(0.01 * x, 0.01 * y, np.zeros(300))

    result = substitution_test(df, 'joint-d005')

    assert result['pearson_r'] > 0.6
    assert result['substitution_detected'] is False


def test_substitution_test_partial_correlation_removes_a_shared_blocks_driver():
    """Both task deltas are driven by the same block delta and are otherwise
    independent. The raw correlation is strongly positive; controlling for
    the block delta must remove essentially all of it."""
    rng = np.random.default_rng(14)
    n = 400
    d_blocks = rng.standard_normal(n)
    d_app = 0.9 * d_blocks + 0.1 * rng.standard_normal(n)
    d_ddos = 0.9 * d_blocks + 0.1 * rng.standard_normal(n)
    df = _paired_frame(0.01 * d_app, 0.01 * d_ddos, d_blocks)

    result = substitution_test(df, 'joint-d005')

    assert result['pearson_r'] > 0.9
    assert abs(result['partial_pearson_r']) < 0.2


def test_substitution_test_quadrant_fractions_sum_to_one_and_count_the_trade_offs():
    df = _paired_frame(
        d_app=[+0.01, +0.01, -0.01, -0.01, 0.0],
        d_ddos=[+0.01, -0.01, +0.01, -0.01, 0.0],
        d_blocks=[0.0] * 5)

    quadrants = substitution_test(df, 'joint-d005')['quadrants']

    assert quadrants['both_up'] == pytest.approx(0.2)
    assert quadrants['app_up_ddos_down'] == pytest.approx(0.2)
    assert quadrants['app_down_ddos_up'] == pytest.approx(0.2)
    assert quadrants['both_down'] == pytest.approx(0.2)
    assert quadrants['on_axis'] == pytest.approx(0.2)
    assert sum(quadrants.values()) == pytest.approx(1.0)


def test_substitution_test_returns_nan_correlations_when_a_delta_is_constant():
    """Every difference identical means zero variance; a correlation is
    undefined, not zero."""
    df = _paired_frame([0.0] * 20, np.linspace(0, 0.01, 20), [0.0] * 20)

    result = substitution_test(df, 'joint-d005')

    assert np.isnan(result['pearson_r'])
    assert result['substitution_detected'] is False


def test_substitution_test_all_arms_covers_every_joint_arm_present():
    """The claim is `no task sacrifices itself at any tolerance`, so the test
    runs at every arm rather than only at the extreme delta."""
    table = substitution_test_all_arms(_full_campaign_frame())

    assert list(table['treatment']) == list(JOINT_ARM_SLUGS)
    assert (table['baseline'] == INDEPENDENT_ARM_SLUG).all()
    assert len(table) == 7


# ---------------------------------------------------------------------------
# delta_frontier
# ---------------------------------------------------------------------------

def test_delta_frontier_reports_the_split_level_mean_and_a_t_based_ci():
    """n = 3 splits: mean 0.84, sd 0.04, sem 0.0230940, t(2, 0.975) = 4.302653
    -> half-width 0.0993654. A normal-approximation CI would be far too
    narrow at this n, which is why the t quantile is used."""
    rows = [_row(arm_slug='joint-d005', M=25, split=s, k=5, acc_app=a)
            for s, a in enumerate([0.80, 0.84, 0.88])]

    table = delta_frontier(_frame(rows), metrics=('acc_app',))

    row = table.iloc[0]
    assert row['n_splits'] == 3
    assert row['mean'] == pytest.approx(0.84)
    assert row['sd'] == pytest.approx(0.04)
    assert row['ci_low'] == pytest.approx(0.84 - 0.0993654, abs=1e-6)
    assert row['ci_high'] == pytest.approx(0.84 + 0.0993654, abs=1e-6)


def test_delta_frontier_groups_by_arm_and_M_and_k_so_arms_are_never_pooled():
    table = delta_frontier(_full_campaign_frame(m_values=(25, 50), k_values=(5, 9)),
                           metrics=('blocks',))

    assert len(table) == 8 * 2 * 2
    assert set(table['arm_slug']) == {INDEPENDENT_ARM_SLUG} | set(JOINT_ARM_SLUGS)


def test_delta_frontier_carries_the_parsed_delta_so_the_sweep_can_be_ordered():
    table = delta_frontier(_full_campaign_frame(), metrics=('blocks',))

    dinf = table[table['arm_slug'] == 'joint-dinf'].iloc[0]
    d005 = table[table['arm_slug'] == 'joint-d005'].iloc[0]

    assert d005['delta_align_num'] == pytest.approx(0.05)
    assert bool(dinf['delta_align_is_inf']) is True
    assert np.isnan(dinf['delta_align_num'])


def test_delta_frontier_refuses_to_average_over_a_group_with_repeated_splits():
    """Pooling k inside one group makes the `one observation per split`
    assumption behind the CI false, so it must be an explicit choice."""
    df = _full_campaign_frame()

    with pytest.raises(ValueError, match='(?i)split'):
        delta_frontier(df, metrics=('blocks',), group_columns=('arm_slug', 'M'))


def test_delta_frontier_allows_pooling_when_the_caller_says_so_explicitly():
    df = _full_campaign_frame()

    table = delta_frontier(df, metrics=('blocks',), group_columns=('arm_slug', 'M'),
                           allow_repeated_splits=True)

    assert len(table) == 8 * 2


def test_delta_frontier_leaves_a_single_observation_groups_ci_undefined():
    rows = [_row(arm_slug='joint-d005', M=25, split=0, k=5, acc_app=0.9)]

    table = delta_frontier(_frame(rows), metrics=('acc_app',))

    assert table.iloc[0]['n_splits'] == 1
    assert np.isnan(table.iloc[0]['ci_low'])


# ---------------------------------------------------------------------------
# ablation_decomposition
# ---------------------------------------------------------------------------

def test_ablation_decomposition_reports_the_sharing_and_the_alignment_contrast():
    table = ablation_decomposition(_full_campaign_frame())

    assert set(table['component']) == {'sharing', 'alignment'}
    sharing = table[table['component'] == 'sharing']
    assert set(sharing['treatment']) == {'joint-off'}
    assert set(sharing['baseline']) == {INDEPENDENT_ARM_SLUG}


def test_ablation_decomposition_measures_alignment_against_joint_off_not_independent():
    """`joint@off - independent` isolates the sharing constraint;
    `joint@delta - joint@off` isolates threshold alignment. Measuring the
    second against `independent` would re-count the sharing effect."""
    table = ablation_decomposition(_full_campaign_frame())

    alignment = table[table['component'] == 'alignment']

    assert set(alignment['baseline']) == {'joint-off'}
    assert set(alignment['treatment']) == set(JOINT_ARM_SLUGS) - {'joint-off'}


def test_ablation_decomposition_recovers_an_exactly_injected_block_saving():
    rows = []
    for split in range(4):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, blocks=40.0))
        rows.append(_row(arm_slug='joint-off', split=split, blocks=34.0))
        rows.append(_row(arm_slug='joint-d005', split=split, blocks=30.0))

    table = ablation_decomposition(_frame(rows), metrics=('blocks',))

    sharing = table[(table['component'] == 'sharing')].iloc[0]
    alignment = table[(table['treatment'] == 'joint-d005')].iloc[0]

    assert sharing['mean_diff_split_level'] == pytest.approx(-6.0)
    assert alignment['mean_diff_split_level'] == pytest.approx(-4.0)


def test_ablation_decomposition_ci_is_built_over_splits_not_over_every_cell():
    """Cells within one split share a training split, so they are not
    independent observations; the CI is formed over split-level means."""
    rows = []
    for split in range(3):
        for k in (5, 9, 13):
            rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, k=k, blocks=40.0))
            rows.append(_row(arm_slug='joint-off', split=split, k=k,
                             blocks=40.0 - [4.0, 6.0, 8.0][split]))

    table = ablation_decomposition(_frame(rows), metrics=('blocks',))
    row = table[table['component'] == 'sharing'].iloc[0]

    assert row['n_pairs'] == 9
    assert row['n_splits'] == 3
    assert row['mean_diff_split_level'] == pytest.approx(-6.0)
    assert row['sd_split_level'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

def test_holm_bonferroni_matches_a_hand_computed_adjustment():
    """p sorted ascending: 0.005, 0.01, 0.03, 0.04 with n = 4.
    Raw step-down: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04.
    Enforcing monotonicity (running maximum): 0.02, 0.03, 0.06, 0.06."""
    adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.005])

    assert list(np.round(adjusted, 10)) == [0.03, 0.06, 0.06, 0.02]


def test_holm_bonferroni_clips_at_one():
    adjusted = holm_bonferroni([0.4, 0.5, 0.6])

    assert adjusted.max() <= 1.0
    assert adjusted[0] == pytest.approx(1.0)


def test_holm_bonferroni_matches_statsmodels():
    """statsmodels is not a dependency of this environment; where it is
    available its `multipletests(method='holm')` is the reference."""
    multipletests = pytest.importorskip(
        'statsmodels.stats.multitest').multipletests
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 21)

    assert holm_bonferroni(p) == pytest.approx(multipletests(p, method='holm')[1])


def test_holm_bonferroni_adjusted_p_reproduces_the_classical_sequential_rejection():
    """statsmodels is absent here, so the primary independent reference is
    Holm's procedure in its ORIGINAL form -- walk the p-values in ascending
    order and stop at the first i where p_(i) * (n - i + 1) >= alpha,
    rejecting everything before it -- checked against the adjusted-p form
    `p_holm < alpha` on random families of the real size."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.uniform(0, 0.2, PRE_REGISTERED_FAMILY_SIZE)
        alpha = 0.05
        n = len(p)
        order = np.argsort(p)
        classical = np.zeros(n, dtype=bool)
        for rank, index in enumerate(order):
            if p[index] * (n - rank) < alpha:
                classical[index] = True
            else:
                break

        assert list(holm_bonferroni(p) < alpha) == list(classical)


def test_holm_bonferroni_refuses_a_nan_p_value_rather_than_shrinking_the_family():
    with pytest.raises(ValueError):
        holm_bonferroni([0.01, float('nan')])


# ---------------------------------------------------------------------------
# paired_tests
# ---------------------------------------------------------------------------

def test_default_contrast_family_is_the_seven_joint_arms_against_independent():
    family = default_contrast_family(_full_campaign_frame())

    assert family == tuple((slug, INDEPENDENT_ARM_SLUG) for slug in JOINT_ARM_SLUGS)


def test_paired_tests_runs_exactly_the_pre_registered_twenty_one_comparisons():
    table = paired_tests(_full_campaign_frame(),
                         expected_family_size=PRE_REGISTERED_FAMILY_SIZE)

    assert len(table) == 21
    assert PRE_REGISTERED_FAMILY_SIZE == 21
    assert table['treatment'].nunique() == 7
    assert set(table['metric']) == {'acc_app', 'acc_ddos', 'blocks'}


def test_paired_tests_raises_when_the_family_is_not_the_size_the_caller_expected():
    df = _full_campaign_frame()
    df = df[df['arm_slug'] != 'joint-d020']

    with pytest.raises(ValueError, match='(?i)famil'):
        paired_tests(df, expected_family_size=PRE_REGISTERED_FAMILY_SIZE)


def test_paired_tests_uses_a_one_sided_alternative_on_each_accuracy_metric():
    table = paired_tests(_full_campaign_frame())

    accuracy = table[table['metric'].isin(['acc_app', 'acc_ddos'])]

    assert set(accuracy['alternative']) == {'greater'}


def test_paired_tests_uses_a_two_sided_alternative_on_blocks():
    """Sharing could plausibly help or hurt the block count, so a direction
    must not be assumed."""
    table = paired_tests(_full_campaign_frame())

    blocks = table[table['metric'] == 'blocks']

    assert set(blocks['alternative']) == {'two-sided'}


def test_paired_tests_one_sided_accuracy_test_rejects_when_the_joint_arm_is_better():
    """The alternative is `median(joint - independent) > -margin`, so a joint
    arm that is uniformly better must produce a small p-value. Reversing the
    alternative would make this p-value ~1."""
    rows = []
    for split in range(20):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, acc_app=0.80))
        rows.append(_row(arm_slug='joint-d005', split=split, acc_app=0.85))

    table = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',))

    assert table.iloc[0]['p_value'] < 0.001


def test_paired_tests_one_sided_accuracy_test_does_not_reject_when_the_joint_arm_is_worse():
    """The mirror of the previous test: a uniformly WORSE joint arm must
    yield a large p-value, never a small one. A reversed alternative fails
    exactly here, which is the whole point of testing both directions."""
    rows = []
    for split in range(20):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, acc_app=0.85))
        rows.append(_row(arm_slug='joint-d005', split=split, acc_app=0.80))

    table = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',))

    assert table.iloc[0]['p_value'] > 0.99


def test_paired_tests_non_inferiority_margin_shifts_the_null_it_tests():
    """A joint arm 0.005 worse on accuracy is inferior at margin 0 but
    non-inferior at a margin of 0.02."""
    rows = []
    for split in range(20):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, acc_app=0.850))
        rows.append(_row(arm_slug='joint-d005', split=split, acc_app=0.845))

    strict = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',))
    lenient = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',),
                           margin=0.02)

    assert strict.iloc[0]['p_value'] > 0.99
    assert lenient.iloc[0]['p_value'] < 0.001


def test_paired_tests_two_sided_blocks_test_fires_in_either_direction():
    better = []
    worse = []
    for split in range(20):
        better.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, blocks=40.0))
        better.append(_row(arm_slug='joint-d005', split=split, blocks=30.0))
        worse.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, blocks=30.0))
        worse.append(_row(arm_slug='joint-d005', split=split, blocks=40.0))

    p_better = paired_tests(_frame(better), arms=('joint-d005',),
                            metrics=('blocks',)).iloc[0]['p_value']
    p_worse = paired_tests(_frame(worse), arms=('joint-d005',),
                           metrics=('blocks',)).iloc[0]['p_value']

    assert p_better < 0.001
    assert p_worse < 0.001


def test_paired_tests_holm_column_corrects_over_the_whole_family_it_ran():
    table = paired_tests(_full_campaign_frame())

    assert (table['p_holm'] >= table['p_value'] - 1e-12).all()
    assert table['n_comparisons'].eq(21).all()
    assert holm_bonferroni(table['p_value'].to_numpy()) == \
        pytest.approx(table['p_holm'].to_numpy())


def test_paired_tests_holm_makes_a_marginal_result_non_significant():
    """A p just under 0.05 in a family of 21 must not survive the correction;
    this is the whole reason the correction exists."""
    table = paired_tests(_full_campaign_frame())
    marginal = 0.04

    assert holm_bonferroni([marginal] + [0.9] * 20)[0] > 0.05
    assert len(table) == 21


def test_paired_tests_reports_the_pair_count_and_the_split_count_per_contrast():
    table = paired_tests(_full_campaign_frame(n_splits=4, m_values=(25, 50),
                                              k_values=(5, 9)))

    assert table['n_pairs'].eq(2 * 4 * 2).all()
    assert table['n_splits'].eq(4).all()


def test_paired_tests_raises_when_a_contrast_has_no_paired_cells_at_all():
    """A contrast that silently contributes nothing would shrink the family
    without the reader noticing."""
    rows = [_row(arm_slug=INDEPENDENT_ARM_SLUG, M=25, split=0, k=5),
            _row(arm_slug='joint-d005', M=50, split=0, k=5)]

    with pytest.raises(ValueError, match='(?i)no paired'):
        paired_tests(_frame(rows), arms=('joint-d005',))


def test_paired_tests_split_level_unit_collapses_cells_before_testing():
    """An explicitly available robustness variant: cells inside a split are
    not independent, so `unit='split'` tests one mean difference per split."""
    table = paired_tests(_full_campaign_frame(n_splits=12), unit='split')

    assert table['n_tested'].eq(12).all()
    assert table['unit'].eq('split').all()


def test_paired_tests_default_unit_is_the_cell_the_spec_pairs_on():
    table = paired_tests(_full_campaign_frame(n_splits=4))

    assert table['unit'].eq('pair').all()
    assert table['n_tested'].eq(table['n_pairs']).all()


def test_paired_tests_raises_when_the_frame_contains_no_treatment_arm_at_all():
    """An empty family would produce an empty table and a vacuous
    correction."""
    rows = [_row(arm_slug=INDEPENDENT_ARM_SLUG, split=s) for s in range(3)]

    with pytest.raises(ValueError, match='(?i)no treatment arms'):
        paired_tests(_frame(rows))


def test_delta_frontier_names_a_missing_metric_column_instead_of_failing_inside_a_group():
    df = _full_campaign_frame().drop(columns=['acc_app'])

    with pytest.raises(KeyError, match='acc_app'):
        delta_frontier(df, metrics=('acc_app',))
