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
import math

import numpy as np
import pandas as pd
import pytest

from src.reporting import claims

from src.reporting.claims import (
    INDEPENDENT_ARM_SLUG,
    JOINT_ARM_SLUGS,
    METRIC_ALTERNATIVE,
    NONINFERIORITY_FAMILY_SIZE,
    PRE_REGISTERED_FAMILY_SIZE,
    SUBSTITUTION_FAMILY_SIZE,
    UNBUDGETED_REFERENCE_BLOCKS,
    ablation_decomposition,
    agreement_table,
    budget_binding,
    arm_deltas,
    coverage_ratio_3d,
    default_contrast_family,
    delta_frontier,
    holm_bonferroni,
    hypervolume_2d,
    hypervolume_by_arm,
    noninferiority_tests,
    paired_tests,
    paired_tests_robustness,
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
    # The compiler-verified campaign's aligned arm: no tolerance axis.
    'joint': (float('nan'), False),
    'joint-off-al': (float('nan'), False),
}

# The archived seven-arm sweep the pre-registration used to name. The
# 3-arm design (spec 2026-09-29 section 7.3) replaced it with
# JOINT_ARM_SLUGS = ('joint-off', 'joint-off-al'); tests that exercise the Holm
# correction over a LARGE family keep using these seven via an explicit
# `arms=`, so their numbers (raw p, Holm-adjusted p) are unchanged.
_ARCHIVED_SEVEN_SLUGS = (
    'joint-off', 'joint-d000', 'joint-d002', 'joint-d005',
    'joint-d010', 'joint-d020', 'joint-dinf',
)


def _row(arm_slug='joint-d005', M=25, split=0, k=5,
         acc_app=0.90, acc_ddos=0.85, blocks=40.0, stages=3.0,
         f1_app=None, f1_ddos=None):
    if f1_app is None:
        f1_app = acc_app - 0.02
    if f1_ddos is None:
        f1_ddos = acc_ddos - 0.02
    delta_num, is_inf = _DELTA_BY_SLUG[arm_slug]
    return {
        'arm_slug': arm_slug,
        'arm': 'independent' if arm_slug == 'independent' else 'joint',
        'method': 'single' if arm_slug == 'independent' else 'multi',
        'M': M, 'split': split, 'k': k,
        'acc_app': acc_app, 'acc_ddos': acc_ddos,
        'f1_app': f1_app, 'f1_ddos': f1_ddos,
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


def _noninferiority_frame(acc_app_base=0.90, acc_ddos_base=0.85,
                          app_relative_degradation=0.0,
                          ddos_relative_degradation=0.0,
                          treatment='joint-off-al', baseline=INDEPENDENT_ARM_SLUG,
                          n_splits=6, M=25, k=5):
    """One baseline row and one treatment row per split, where the
    treatment's accuracy is offset from the baseline by a stated FRACTION
    of the baseline's own error -- `acc_treatment = acc_base -
    relative_degradation * (1 - acc_base)` -- so `noninferiority_tests`'s
    per-row relative margin and the delta it tests are both exactly
    checkable from the inputs given here."""
    app_treatment = acc_app_base - app_relative_degradation * (1.0 - acc_app_base)
    ddos_treatment = acc_ddos_base - ddos_relative_degradation * (1.0 - acc_ddos_base)
    rows = []
    for split in range(n_splits):
        rows.append(_row(arm_slug=baseline, M=M, split=split, k=k,
                         acc_app=acc_app_base, acc_ddos=acc_ddos_base))
        rows.append(_row(arm_slug=treatment, M=M, split=split, k=k,
                         acc_app=app_treatment, acc_ddos=ddos_treatment))
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
# hypervolume_2d (D5 as amended by A2: per-M reference point)
# ---------------------------------------------------------------------------

def test_hypervolume_2d_is_the_area_above_the_per_M_reference_point():
    """A2: reference point (0.5, M), not the published (0.5, 100). At M=150
    a fixed 100 would silently discard most of the front. Consequence: the
    paper's Fig. results_2a numbers are NOT reproducible by this code."""
    front = [(0.9, 10.0), (0.8, 5.0)]
    # (0.9-0.5)*(25-10) + (0.8-0.5)*(10-5) = 6.0 + 1.5
    assert hypervolume_2d(front, reference=(0.5, 25)) == pytest.approx(7.5)


def test_a_solution_exactly_on_the_reference_budget_is_kept():
    """The ported function filtered with a strict `mem < ref`, silently
    dropping every solution landing exactly on budget at M=100."""
    assert hypervolume_2d([(0.9, 25.0)], reference=(0.5, 25)) == 0.0
    assert hypervolume_2d([(0.9, 24.0)], reference=(0.5, 25)) > 0.0


def test_an_empty_front_has_an_undefined_hypervolume_not_zero():
    """coverage_ratio_3d's convention: NaN for undefined, not a number that
    reads as a real measurement. The original returned integer 0."""
    assert math.isnan(hypervolume_2d([], reference=(0.5, 25)))


def test_a_non_empty_front_that_never_reaches_the_reference_is_a_real_zero_not_nan():
    """Distinct from the empty-front case above. An empty `front` means no
    data was measured at all (undefined -- NaN). A front with points, none
    of which reach the reference budget, means the measurement WAS taken
    and the answer is a genuine zero -- exactly `coverage_ratio_3d`'s
    "`a` dominates nothing" case, which returns 0.0 rather than NaN even
    though `a` contributed nothing."""
    front = [(0.4, 10.0), (0.3, 30.0)]  # both below ref_accuracy=0.5
    result = hypervolume_2d(front, reference=(0.5, 25))
    assert result == 0.0
    assert not math.isnan(result)


# ---------------------------------------------------------------------------
# hypervolume_by_arm (Finding 1: wires hypervolume_2d to a real caller, via a
# genuine 2-D Pareto filter -- hypervolume_2d's own contract assumes `front`
# is already non-dominated in the (accuracy, blocks) plane).
# ---------------------------------------------------------------------------

def test_hypervolume_by_arm_matches_a_hand_computed_value():
    """Small synthetic frame, one arm's app-task front hand-verified against
    the same numbers `test_hypervolume_2d_is_the_area_above_the_per_M_
    reference_point` already checks (front=[(0.9, 10), (0.8, 5)], neither
    point dominates the other, reference (0.5, 25) -> hv = 7.5), plus a
    baseline row whose own (acc_app, blocks) = (0.6, 20) gives a
    hand-computable baseline hypervolume of (25 - 20) * (0.6 - 0.5) = 0.5,
    so the gain (7.5 / 0.5 = 15.0) is checkable too."""
    df = _frame([
        _row(arm_slug=INDEPENDENT_ARM_SLUG, M=25, split=0, k=5,
             acc_app=0.6, acc_ddos=0.6, blocks=20.0),
        _row(arm_slug='joint-d005', M=25, split=0, k=5,
             acc_app=0.9, acc_ddos=0.6, blocks=10.0),
        _row(arm_slug='joint-d005', M=25, split=1, k=5,
             acc_app=0.8, acc_ddos=0.6, blocks=5.0),
    ])

    table = hypervolume_by_arm(df, baseline=INDEPENDENT_ARM_SLUG)

    row = table[(table['arm_slug'] == 'joint-d005') & (table['M'] == 25)
               & (table['task'] == 'app')].iloc[0]
    assert row['hypervolume'] == pytest.approx(7.5)
    assert row['baseline_hypervolume'] == pytest.approx(0.5)
    assert row['hypervolume_gain'] == pytest.approx(15.0)


def test_hypervolume_by_arm_pareto_filters_before_summing_so_a_dominated_point_cannot_corrupt_it():
    """`hypervolume_2d` assumes its `front` argument is already a genuine
    non-dominated 2-D front; the sweep silently gives a WRONG number if a
    dominated point sneaks in. Here G=(acc=0.95, blocks=5) strictly
    dominates D=(acc=0.94, blocks=10) (better accuracy AND fewer blocks), so
    a correct implementation must drop D before summing. Hand-computed:

    Filtered front {G, F=(0.96, 20)} at reference (0.5, 25):
      (25-20)*(0.96-0.5) + (20-5)*(0.95-0.5) = 5*0.46 + 15*0.45 = 2.3 + 6.75
      = 9.05
    The raw, UNFILTERED three points would instead give 8.95 (verified by
    hand and cross-checked against a direct `hypervolume_2d` call below) --
    a different, wrong number that this regression test guards against ever
    reappearing once a caller exists."""
    df = _frame([
        _row(arm_slug=INDEPENDENT_ARM_SLUG, M=25, split=0, k=5,
             acc_app=0.5, acc_ddos=0.6, blocks=25.0),
        _row(arm_slug='joint-d005', M=25, split=0, k=5,
             acc_app=0.95, acc_ddos=0.6, blocks=5.0),
        _row(arm_slug='joint-d005', M=25, split=1, k=5,
             acc_app=0.94, acc_ddos=0.6, blocks=10.0),
        _row(arm_slug='joint-d005', M=25, split=2, k=5,
             acc_app=0.96, acc_ddos=0.6, blocks=20.0),
    ])

    table = hypervolume_by_arm(df, baseline=INDEPENDENT_ARM_SLUG)
    row = table[(table['arm_slug'] == 'joint-d005') & (table['M'] == 25)
               & (table['task'] == 'app')].iloc[0]

    unfiltered_raw = hypervolume_2d(
        [(0.95, 5.0), (0.94, 10.0), (0.96, 20.0)], reference=(0.5, 25))
    assert unfiltered_raw == pytest.approx(8.95)
    assert row['hypervolume'] == pytest.approx(9.05)
    assert row['hypervolume'] != pytest.approx(unfiltered_raw)


def test_hypervolume_gain_is_nan_not_inf_when_the_baseline_hypervolume_is_zero_or_missing():
    """A baseline front that never reaches the (0.5, M) reference gives a
    real, measured hypervolume of 0.0 (not NaN, per `hypervolume_2d`'s own
    contract) -- dividing by it must not raise or produce inf."""
    df = _frame([
        _row(arm_slug=INDEPENDENT_ARM_SLUG, M=25, split=0, k=5,
             acc_app=0.2, acc_ddos=0.6, blocks=25.0),  # below ref accuracy
        _row(arm_slug='joint-d005', M=25, split=0, k=5,
             acc_app=0.9, acc_ddos=0.6, blocks=10.0),
    ])

    table = hypervolume_by_arm(df, baseline=INDEPENDENT_ARM_SLUG)
    row = table[(table['arm_slug'] == 'joint-d005') & (table['M'] == 25)
               & (table['task'] == 'app')].iloc[0]

    assert row['baseline_hypervolume'] == 0.0
    assert math.isnan(row['hypervolume_gain'])
    assert not math.isinf(row['hypervolume_gain'])


def test_hypervolume_gain_is_nan_when_the_baseline_arm_is_entirely_absent_at_that_M():
    """No baseline rows at all -> baseline hypervolume is NaN (an empty
    front, per `hypervolume_2d`'s own empty-front contract) -> the gain must
    be NaN, not a crash or an inf from dividing by NaN."""
    df = _frame([
        _row(arm_slug='joint-d005', M=25, split=0, k=5,
             acc_app=0.9, acc_ddos=0.6, blocks=10.0),
    ])

    table = hypervolume_by_arm(df, baseline=INDEPENDENT_ARM_SLUG)
    row = table[(table['arm_slug'] == 'joint-d005') & (table['M'] == 25)
               & (table['task'] == 'app')].iloc[0]

    assert math.isnan(row['baseline_hypervolume'])
    assert math.isnan(row['hypervolume_gain'])


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
    assert len(table) == 2      # 3-arm design: joint-off and joint


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

    assert len(table) == 3 * 2 * 2      # 3-arm design: 3 arms x 2 M x 2 k
    assert set(table['arm_slug']) == {INDEPENDENT_ARM_SLUG} | set(JOINT_ARM_SLUGS)


def test_delta_frontier_carries_the_parsed_delta_so_the_sweep_can_be_ordered():
    # The 3-arm grid has no delta arms any more, so archived ones are built
    # explicitly: the parsed delta must still be carried when they appear.
    rows = [_row(arm_slug=slug, M=25, split=s, k=5)
            for slug in ('joint-d005', 'joint-dinf') for s in range(2)]
    table = delta_frontier(_frame(rows), metrics=('blocks',))

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

    assert len(table) == 3 * 2      # 3-arm design: 3 arms x 2 M


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
        # 3-arm design: the aligned arm is `joint` (was archived `joint-d005`).
        rows.append(_row(arm_slug='joint-off-al', split=split, blocks=30.0))

    table = ablation_decomposition(_frame(rows), metrics=('blocks',))

    sharing = table[(table['component'] == 'sharing')].iloc[0]
    alignment = table[(table['treatment'] == 'joint-off-al')].iloc[0]

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


def test_paired_tests_runs_exactly_the_pre_registered_ten_comparisons():
    """D2: F1 joins the tested family (joint arms x 5 metrics), answering R3
    section IV(d) -- accuracy hides minority classes. 3-arm design (spec
    2026-09-29 section 7.3): 2 joint arms x 5 tests = 10, down from the
    archived sweep's 7 x 5 = 35."""
    table = paired_tests(_full_campaign_frame(),
                         expected_family_size=PRE_REGISTERED_FAMILY_SIZE)

    assert PRE_REGISTERED_FAMILY_SIZE == 10
    assert len(table) == 10
    assert table['treatment'].nunique() == 2
    assert set(table['metric']) == {'acc_app', 'f1_app', 'acc_ddos', 'f1_ddos', 'blocks'}


def test_f1_is_one_sided_greater_like_accuracy():
    """Same direction and same 'small p is the positive finding' reading."""
    assert METRIC_ALTERNATIVE['f1_app'] == 'greater'
    assert METRIC_ALTERNATIVE['f1_ddos'] == 'greater'


def test_paired_tests_raises_when_the_family_is_not_the_size_the_caller_expected():
    df = _full_campaign_frame()
    df = df[df['arm_slug'] != 'joint-off-al']

    with pytest.raises(ValueError, match='(?i)famil'):
        paired_tests(df, expected_family_size=PRE_REGISTERED_FAMILY_SIZE)


def test_the_family_sizes_are_derived_from_the_three_arm_grid():
    assert PRE_REGISTERED_FAMILY_SIZE == 10
    assert SUBSTITUTION_FAMILY_SIZE == 2
    assert NONINFERIORITY_FAMILY_SIZE == 4
    assert JOINT_ARM_SLUGS == ('joint-off', 'joint-off-al')


def test_a_frame_with_only_joint_off_still_raises_under_the_family_of_ten():
    """Spec section 7.3: 'a missing arm still raises'."""
    df = _full_campaign_frame()
    df = df[df['arm_slug'].isin([INDEPENDENT_ARM_SLUG, 'joint-off'])]

    with pytest.raises(ValueError, match='(?i)famil'):
        paired_tests(df, expected_family_size=10)


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
    assert table['n_comparisons'].eq(PRE_REGISTERED_FAMILY_SIZE).all()
    assert holm_bonferroni(table['p_value'].to_numpy()) == \
        pytest.approx(table['p_holm'].to_numpy())


def test_paired_tests_holm_makes_a_marginal_result_non_significant():
    """A p just under 0.05 in a family of 10 must not survive the correction;
    this is the whole reason the correction exists. (3-arm design: the
    family is 10, so 0.04 * 10 = 0.4; under the archived 35 it was 1.4.)"""
    table = paired_tests(_full_campaign_frame())
    marginal = 0.04

    assert holm_bonferroni([marginal] + [0.9] * (PRE_REGISTERED_FAMILY_SIZE - 1))[0] > 0.05
    assert len(table) == PRE_REGISTERED_FAMILY_SIZE


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


# ---------------------------------------------------------------------------
# Review fix 1: the emitted `hypothesis` column must not claim a margin
# nobody set.
# ---------------------------------------------------------------------------

def _hypothesis_at(margin):
    rows = []
    for split in range(20):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, acc_app=0.85))
        rows.append(_row(arm_slug='joint-d005', split=split, acc_app=0.84))
    table = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',),
                         margin=margin)
    return table.iloc[0]['hypothesis']


def test_hypothesis_column_says_no_detectable_loss_when_no_margin_was_set():
    """The default margin is 0, so there is no non-inferiority margin to be
    non-inferior to. The results table must say what the test actually did."""
    hypothesis = _hypothesis_at(0.0)

    assert 'no detectable loss' in hypothesis
    assert 'no non-inferiority margin was set' in hypothesis


def test_hypothesis_column_does_not_render_a_negative_zero_margin():
    """`'{:g}'.format(0.0)` behind a literal minus renders `-0`, which reads
    in a results table as though some margin exists."""
    hypothesis = _hypothesis_at(0.0)

    assert '-0' not in hypothesis


def test_hypothesis_column_claims_non_inferiority_only_once_a_margin_exists():
    hypothesis = _hypothesis_at(0.02)

    assert 'non-inferiority of joint-d005 to independent' in hypothesis
    assert 'margin of 0.02' in hypothesis
    assert 'no detectable loss' not in hypothesis


def test_zero_difference_counts_are_reported_before_and_after_the_margin_shift():
    """`n_zero_differences` describes the raw deltas; `n_zero_in_test` is what
    zsplit actually had to split. They diverge as soon as margin > 0."""
    rows = []
    for split in range(20):
        rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=split, acc_app=0.85))
        rows.append(_row(arm_slug='joint-d005', split=split, acc_app=0.85))
    table = paired_tests(_frame(rows), arms=('joint-d005',), metrics=('acc_app',),
                         margin=0.01)

    assert table.iloc[0]['n_zero_differences'] == 20
    assert table.iloc[0]['n_zero_in_test'] == 0


# ---------------------------------------------------------------------------
# Review fix 2: the substitution family is separate from the 35, and
# corrected within itself.
# ---------------------------------------------------------------------------

def _substitution_sweep_frame(seed, correlated_arm, rho, n=30,
                              slugs=JOINT_ARM_SLUGS):
    """Every joint arm against one shared independent baseline. Only
    `correlated_arm` carries a genuine negative association; the rest are
    independent draws."""
    rows = []
    for i, slug in enumerate(slugs):
        r = rho if slug == correlated_arm else 0.0
        rng = np.random.default_rng(seed + i)
        x = rng.standard_normal(n)
        z = rng.standard_normal(n)
        y = r * x + np.sqrt(1.0 - r ** 2) * z
        for j in range(n):
            if i == 0:
                rows.append(_row(arm_slug=INDEPENDENT_ARM_SLUG, split=j,
                                 acc_app=0.90, acc_ddos=0.85, blocks=40.0))
            rows.append(_row(arm_slug=slug, split=j,
                             acc_app=0.90 + 0.01 * x[j],
                             acc_ddos=0.85 + 0.01 * y[j],
                             blocks=40.0 + j * 0.01))
    return _frame(rows)


def test_substitution_sweep_exposes_a_holm_column_over_its_own_joint_arms():
    table = substitution_test_all_arms(_full_campaign_frame())

    assert 'pearson_p_negative_one_sided_holm' in table.columns
    assert 'substitution_detected_holm' in table.columns
    assert table['n_substitution_comparisons'].eq(SUBSTITUTION_FAMILY_SIZE).all()
    assert SUBSTITUTION_FAMILY_SIZE == 2      # 3-arm design: 2 joint arms


def test_substitution_sweep_holm_column_is_the_holm_adjustment_of_the_raw_column():
    table = substitution_test_all_arms(_full_campaign_frame())

    assert holm_bonferroni(table['pearson_p_negative_one_sided'].to_numpy()) == \
        pytest.approx(table['pearson_p_negative_one_sided_holm'].to_numpy())


def test_substitution_sweep_holm_demotes_a_flag_that_only_fired_because_seven_ran():
    """One arm carries a real but modest negative correlation: raw
    p = 0.0079 fires at alpha = 0.05, Holm over the seven arms lifts it to
    0.055 and it does not. Reading the seven raw flags across the sweep as
    one test is exactly the error this column exists to prevent."""
    # The archived seven arms, passed explicitly: the point is Holm over a
    # seven-member family, which the 3-arm grid no longer has on its own.
    table = substitution_test_all_arms(
        _substitution_sweep_frame(seed=1, correlated_arm='joint-d010', rho=-0.35,
                                  slugs=_ARCHIVED_SEVEN_SLUGS),
        arms=_ARCHIVED_SEVEN_SLUGS)

    assert table['substitution_detected'].sum() == 1
    assert table['substitution_detected_holm'].sum() == 0


def test_substitution_sweep_corrected_flag_never_fires_where_the_raw_one_did_not():
    """The correction may only ever demote; a Holm-adjusted p is never below
    its raw p."""
    table = substitution_test_all_arms(
        _substitution_sweep_frame(seed=4, correlated_arm='joint-d020', rho=-0.9,
                                  slugs=_ARCHIVED_SEVEN_SLUGS),
        arms=_ARCHIVED_SEVEN_SLUGS)

    assert (table['pearson_p_negative_one_sided_holm']
            >= table['pearson_p_negative_one_sided'] - 1e-12).all()
    assert not (table['substitution_detected_holm']
                & ~table['substitution_detected']).any()
    assert table['substitution_detected_holm'].sum() == 1


def test_substitution_sweep_excludes_an_undefined_arm_from_its_correction_family():
    """An arm whose deltas are constant produced no test at all, not a null
    result; counting it would inflate the family with a comparison nobody
    ran."""
    df = _substitution_sweep_frame(seed=2, correlated_arm='joint-d010', rho=-0.5,
                                   slugs=_ARCHIVED_SEVEN_SLUGS)
    flat = df['arm_slug'] == 'joint-d002'
    df.loc[flat, 'acc_app'] = 0.90
    df.loc[flat, 'acc_ddos'] = 0.85

    table = substitution_test_all_arms(df, arms=_ARCHIVED_SEVEN_SLUGS)

    assert table['n_substitution_comparisons'].eq(6).all()
    flat_row = table[table['treatment'] == 'joint-d002'].iloc[0]
    assert np.isnan(flat_row['pearson_r'])
    assert bool(flat_row['substitution_detected_holm']) is False


def test_substitution_test_refuses_a_family_shrunk_by_a_MISSING_arm():
    """paired_tests already closed this for its own family. It goes live
    precisely because the campaign runs in chunks, where a missing arm is
    the normal intermediate state -- and a smaller family is a WEAKER Holm
    correction applied silently."""
    frame = _substitution_sweep_frame(seed=3, correlated_arm='joint-off-al', rho=-0.8)
    frame = frame[frame['arm_slug'] != 'joint-off-al']
    with pytest.raises(ValueError, match='family'):
        substitution_test_all_arms(
            frame, expected_family_size=SUBSTITUTION_FAMILY_SIZE)


def test_substitution_test_reports_the_split_count_next_to_the_pair_count():
    """The cell-dependence caveat is only checkable if both numbers are
    there."""
    result = substitution_test(_full_campaign_frame(n_splits=4), 'joint-off-al')

    assert result['n_pairs'] == 16
    assert result['n_splits'] == 4


# ---------------------------------------------------------------------------
# noninferiority_tests -- D13: non-inferiority at a 5% relative-error margin
# ---------------------------------------------------------------------------

def test_non_inferiority_margin_is_relative_to_each_task_own_error():
    """D13: 0.5 accuracy points is ~1.7% relative for App but ~17% for DDoS.
    The margin uses the same currency as delta_frontier and delta_align."""
    frame = _noninferiority_frame(acc_app_base=0.70, acc_ddos_base=0.97)
    table = noninferiority_tests(frame)
    app = table[table['metric'] == 'acc_app'].iloc[0]
    ddos = table[table['metric'] == 'acc_ddos'].iloc[0]
    assert app['margin'] == pytest.approx(0.05 * 0.30)
    assert ddos['margin'] == pytest.approx(0.05 * 0.03)


def test_the_non_inferiority_family_is_corrected_separately_from_the_superiority_family():
    """Mirrors how `pair` and `split` units are already corrected
    independently. Leaves the superiority family's power untouched."""
    table = noninferiority_tests(_full_campaign_frame())
    assert NONINFERIORITY_FAMILY_SIZE == 4      # 3-arm design: 2 arms x 2 accuracies
    assert len(table) == 4
    assert set(table['metric']) == {'acc_app', 'acc_ddos'}
    assert (table['n_comparisons'] == 4).all()


def test_the_margin_is_capable_of_failing():
    """A margin nothing can fail is worthless. 5% sits inside the +12.07%
    DDoS relative degradation the pre-fix 2975-cell evidence measured."""
    frame = _noninferiority_frame(ddos_relative_degradation=0.1207)
    table = noninferiority_tests(frame)
    ddos = table[table['metric'] == 'acc_ddos'].iloc[0]
    assert not ddos['significant_holm']


# ---------------------------------------------------------------------------
# Task 13: M = inf hypervolume, agreement table, budget binding, robustness
# ---------------------------------------------------------------------------

def test_the_unbudgeted_reference_is_the_whole_pipe():
    assert UNBUDGETED_REFERENCE_BLOCKS == 24 * 12 == 288


def test_hypervolume_by_arm_at_M_inf_uses_the_whole_pipe_as_reference():
    """O2: (0.5, inf) would be an infinite reference; the pipe's 288 blocks
    stand in for it. Finite M keeps (0.5, M)."""
    inf = float('inf')
    rows = [
        _row(arm_slug=INDEPENDENT_ARM_SLUG, M=inf, split=0, acc_app=0.8, blocks=100.0),
        _row(arm_slug='joint-off-al', M=inf, split=0, acc_app=0.9, blocks=40.0),
        _row(arm_slug='joint-off-al', M=inf, split=1, acc_app=0.7, blocks=20.0),
        _row(arm_slug='joint-off-al', M=25.0, split=0, acc_app=0.9, blocks=20.0),
    ]
    table = hypervolume_by_arm(_frame(rows))

    at_inf = table[(table['arm_slug'] == 'joint-off-al') & (table['M'] == inf)
                   & (table['task'] == 'app')].iloc[0]
    expected = hypervolume_2d([(0.9, 40.0), (0.7, 20.0)], (0.5, 288))
    assert np.isfinite(at_inf['hypervolume'])
    assert at_inf['hypervolume'] == pytest.approx(expected)
    baseline = hypervolume_2d([(0.8, 100.0)], (0.5, 288))
    assert at_inf['hypervolume_gain'] == pytest.approx(expected / baseline)

    finite = table[(table['arm_slug'] == 'joint-off-al') & (table['M'] == 25.0)
                   & (table['task'] == 'app')].iloc[0]
    assert finite['hypervolume'] == pytest.approx(
        hypervolume_2d([(0.9, 20.0)], (0.5, 25.0)))


def _verification_line(arm_slug, M, split, k=5, verdict='EXACT',
                       model_stage_depth=9, p4c_stage_depth=9,
                       model_blocks=30, p4c_blocks=30, unverified=False,
                       tables_differing='[]'):
    """One row as `campaign_data.load_verification` returns it: literal CSV
    text ('True'/'False', '' for None) except the parsed arm_slug / M
    (float64, inf when unbudgeted) / split / k."""
    token = 'inf' if M == float('inf') else '{:03d}'.format(int(M))

    def cell(value):
        return '' if value is None else str(value)

    return {
        'row_id': '{}_M{}_s{:02d}_k{:02d}'.format(arm_slug, token, split, k),
        'verdict': verdict,
        'model_stage_depth': cell(model_stage_depth),
        'p4c_stage_depth': cell(None if unverified else p4c_stage_depth),
        'model_blocks': cell(model_blocks),
        'p4c_blocks': cell(None if unverified else p4c_blocks),
        'p4c_over_budget': 'False', 'p4c_over_stages': 'False',
        'unverified': str(bool(unverified)),
        'tables_differing': tables_differing,
        'arm_slug': arm_slug, 'M': float(M), 'split': split, 'k': k,
    }


def _verification_frame(lines):
    frame = pd.DataFrame(lines)
    frame['M'] = frame['M'].astype('float64')
    return frame


def test_agreement_table_counts_exact_out_of_n_per_arm_and_M():
    inf = float('inf')
    differing = '[{"model":4,"p4c":6,"table":"tbl_x"}]'
    lines = [
        _verification_line('joint', 25, 0),
        _verification_line('joint', 25, 1, model_blocks=30, p4c_blocks=32,
                           verdict='UNDER', tables_differing=differing),
        _verification_line('joint', 25, 2, model_stage_depth=11,
                           p4c_stage_depth=13, p4c_blocks=None,
                           verdict='FALSE_FEASIBLE'),
        _verification_line('joint', inf, 0),
        _verification_line('joint', inf, 1, unverified=True,
                           verdict='COMPILE_ERROR'),
        _verification_line('joint-off', 25, 0),
    ]
    summary, misses = agreement_table(_verification_frame(lines))

    assert list(summary.columns) == ['arm_slug', 'M', 'n', 'stage_depth_exact',
                                     'blocks_exact', 'blocks_na']
    cell = summary[(summary['arm_slug'] == 'joint') & (summary['M'] == 25)].iloc[0]
    assert (cell['n'], cell['stage_depth_exact'], cell['blocks_exact'],
            cell['blocks_na']) == (3, 2, 1, 1)
    cell = summary[(summary['arm_slug'] == 'joint') & (summary['M'] == inf)].iloc[0]
    assert (cell['n'], cell['stage_depth_exact'], cell['blocks_exact'],
            cell['blocks_na']) == (2, 1, 1, 0)
    cell = summary[summary['arm_slug'] == 'joint-off'].iloc[0]
    assert (cell['n'], cell['stage_depth_exact'], cell['blocks_exact']) == (1, 1, 1)

    assert list(misses.columns) == ['row_id', 'verdict', 'model_stage_depth',
                                    'p4c_stage_depth', 'model_blocks',
                                    'p4c_blocks', 'tables_differing']
    assert sorted(misses['verdict']) == ['COMPILE_ERROR', 'FALSE_FEASIBLE', 'UNDER']
    under = misses[misses['verdict'] == 'UNDER'].iloc[0]
    assert under['tables_differing'] == [{'model': 4, 'p4c': 6, 'table': 'tbl_x'}]
    assert under['model_blocks'] == 30 and under['p4c_blocks'] == 32
    false_feasible = misses[misses['verdict'] == 'FALSE_FEASIBLE'].iloc[0]
    assert false_feasible['row_id'] == 'joint_M025_s02_k05'
    assert false_feasible['p4c_stage_depth'] == 13
    assert np.isnan(false_feasible['p4c_blocks'])


def test_agreement_table_accepts_parsed_bools_as_well_as_csv_text():
    lines = [_verification_line('joint', 25, 0),
             _verification_line('joint', 25, 1, unverified=True,
                                verdict='TIMEOUT')]
    frame = _verification_frame(lines)
    frame['unverified'] = frame['unverified'] == 'True'
    summary, _ = agreement_table(frame)

    cell = summary.iloc[0]
    assert (cell['n'], cell['stage_depth_exact'], cell['blocks_na']) == (2, 1, 0)


def test_budget_binding_is_the_share_of_rows_at_ninety_percent_of_M():
    inf = float('inf')
    rows = [
        _row(arm_slug='joint-off-al', M=50.0, split=0, blocks=45.0),   # = 0.9 M: binding
        _row(arm_slug='joint-off-al', M=50.0, split=1, blocks=49.0),   # binding
        _row(arm_slug='joint-off-al', M=50.0, split=2, blocks=44.9),
        _row(arm_slug='joint-off-al', M=50.0, split=3, blocks=20.0),
        _row(arm_slug='joint-off-al', M=inf, split=0, blocks=200.0),
        _row(arm_slug='joint-off-al', M=inf, split=1, blocks=20.0),
    ]
    table = budget_binding(_frame(rows))

    assert list(table.columns) == ['arm_slug', 'M', 'n', 'binding_share']
    finite = table[table['M'] == 50.0].iloc[0]
    assert finite['n'] == 4
    assert finite['binding_share'] == pytest.approx(0.5)
    unbudgeted = table[table['M'] == inf].iloc[0]
    assert unbudgeted['n'] == 2
    assert np.isnan(unbudgeted['binding_share'])


def _robustness_frame(flagged_split=None):
    df = _full_campaign_frame(n_splits=16, m_values=(25,), k_values=(5,))
    df['flagged'] = False
    if flagged_split is not None:
        df.loc[df['split'] == flagged_split, 'flagged'] = True
    return df


def test_paired_tests_robustness_runs_heldout_splits_and_omits_no_flagged_when_clean():
    table = paired_tests_robustness(
        _robustness_frame(), expected_family_size=PRE_REGISTERED_FAMILY_SIZE)

    assert set(table['subset']) == {'all', 'heldout_splits'}
    heldout = table[table['subset'] == 'heldout_splits']
    assert len(heldout) == PRE_REGISTERED_FAMILY_SIZE
    assert heldout['n_splits'].eq(12).all()        # 16 splits minus 10-13
    assert table[table['subset'] == 'all']['n_splits'].eq(16).all()


def test_paired_tests_robustness_adds_no_flagged_when_any_row_is_flagged():
    table = paired_tests_robustness(_robustness_frame(flagged_split=3))

    assert set(table['subset']) == {'all', 'no_flagged', 'heldout_splits'}
    assert table[table['subset'] == 'no_flagged']['n_splits'].eq(15).all()


def test_paired_tests_robustness_holds_only_all_to_the_family_size():
    df = _robustness_frame()
    df = df[df['arm_slug'] != 'joint-off-al']
    with pytest.raises(ValueError, match='(?i)famil'):
        paired_tests_robustness(df, expected_family_size=PRE_REGISTERED_FAMILY_SIZE)


def test_paired_tests_robustness_skips_an_empty_subset_instead_of_crashing():
    """Task 14: a run holding only development splits leaves `heldout_splits`
    empty, and a run with every row flagged leaves `no_flagged` empty. An
    empty robustness subset has nothing to test; it is skipped (absent from
    `subset`), never allowed to crash the headline `all` family."""
    df = _robustness_frame()
    df = df[df['split'].isin([10, 11, 12, 13])].copy()
    df['flagged'] = True

    table = paired_tests_robustness(df)

    assert set(table['subset']) == {'all'}


def _three_arm_frame(n_splits=3):
    rows = []
    for arm_index, arm in enumerate(('independent', 'joint-off', 'joint-off-al')):
        for M in (25, 50):
            for split in range(n_splits):
                for k in (3, 5):
                    rows.append({'arm_slug': arm, 'M': float(M), 'split': split, 'k': k,
                                 'acc_app': 0.8 + 0.01 * arm_index, 'f1_app': 0.78 + 0.01 * arm_index,
                                 'acc_ddos': 0.9, 'f1_ddos': 0.88,
                                 'blocks': float(M - 2 * arm_index), 'stage_depth': 10.0})
    return pd.DataFrame(rows)


def test_joint_arm_slugs_are_sharing_then_twins_and_the_families_keep_their_size():
    assert claims.JOINT_ARM_SLUGS == ('joint-off', 'joint-off-al')
    assert claims.LEGACY_ARM_SLUGS == ('joint',)
    assert claims.PRE_REGISTERED_FAMILY_SIZE == 10
    assert claims.SUBSTITUTION_FAMILY_SIZE == 2
    assert claims.NONINFERIORITY_FAMILY_SIZE == 4


def test_ablation_decomposition_alignment_contrast_is_twins_minus_joint_off():
    df = _three_arm_frame()
    table = claims.ablation_decomposition(df)
    contrasts = set(table['contrast'])
    assert 'joint-off - independent' in contrasts
    assert 'joint-off-al - joint-off' in contrasts
    assert not any(c.startswith('joint -') for c in contrasts)


def _twin_frame():
    """independent/joint-off/joint-off-al on 2 M x 3 splits x k in (3, 8, 14).
    Twins save 2 blocks on k=8 cells only, cost nothing, and leave F1 alone."""
    rows = []
    for arm in ('independent', 'joint-off', 'joint-off-al'):
        for M in (25, 50):
            for split in range(3):
                for k in (3, 8, 14):
                    blocks = M - k
                    if arm != 'independent':
                        blocks -= 1
                    if arm == 'joint-off-al' and k == 8:
                        blocks -= 2
                    rows.append({'arm_slug': arm, 'M': float(M), 'split': split, 'k': k,
                                 'blocks': float(blocks), 'stage_depth': 10.0,
                                 'acc_app': 0.8, 'f1_app': 0.75, 'acc_ddos': 0.9, 'f1_ddos': 0.88})
    return pd.DataFrame(rows)


def test_twin_effect_reports_the_injected_saving_per_group():
    table = claims.twin_effect(_twin_frame())
    blocks_all = table[(table.group_kind == 'all') & (table.metric == 'blocks')].iloc[0]
    assert blocks_all['n_pairs'] == 18 and blocks_all['mean_diff_pairwise'] == pytest.approx(-2 / 3)
    assert blocks_all['p_saves'] == pytest.approx(1 / 3) and blocks_all['p_costs'] == 0.0
    mid = table[(table.group_kind == 'k_group') & (table.group == 'mid 6-11') & (table.metric == 'blocks')].iloc[0]
    assert mid['mean_diff_pairwise'] == -2.0 and mid['p_saves'] == 1.0
    f1 = table[(table.group_kind == 'all') & (table.metric == 'f1_app')].iloc[0]
    assert f1['mean_diff_pairwise'] == 0.0 and f1['p_saves'] == 0.0 and f1['p_costs'] == 0.0


def test_twin_ladder_adds_up():
    ladder = claims.twin_ladder(_twin_frame())
    row = ladder[ladder.M == 'all'].iloc[0]
    assert row['step_sharing_blocks'] == pytest.approx(-1.0)
    assert row['step_alignment_blocks'] == pytest.approx(-2 / 3)
    assert row['mean_joint-off-al_blocks'] == pytest.approx(
        row['mean_independent_blocks'] + row['step_sharing_blocks'] + row['step_alignment_blocks'])


def _rescued_verification(blank):
    return pd.DataFrame([
        {'row_id': 'joint-off_M025_s00_k03', 'arm_slug': 'joint-off', 'verdict': 'FALSE_FEASIBLE',
         'p4c_over_stages': True, 'p4c_over_budget': False, 'copied_from': blank, 'unverified': False},
        {'row_id': 'joint-off-al_M025_s00_k03', 'arm_slug': 'joint-off-al', 'verdict': 'EXACT',
         'p4c_over_stages': False, 'p4c_over_budget': False, 'copied_from': blank, 'unverified': False},
        {'row_id': 'joint-off_M025_s00_k08', 'arm_slug': 'joint-off', 'verdict': 'EXACT',
         'p4c_over_stages': False, 'p4c_over_budget': False, 'copied_from': blank, 'unverified': False},
        {'row_id': 'joint-off-al_M025_s00_k08', 'arm_slug': 'joint-off-al', 'verdict': 'EXACT',
         'p4c_over_stages': False, 'p4c_over_budget': False,
         'copied_from': 'joint-off_M025_s00_k08', 'unverified': False},
        {'row_id': 'joint-off-al_M025_s00_k14', 'arm_slug': 'joint-off-al', 'verdict': 'UNDER',
         'p4c_over_stages': False, 'p4c_over_budget': True, 'copied_from': blank, 'unverified': False},
    ])


@pytest.mark.parametrize('blank', ['', None, float('nan')])
def test_twin_pairs_excludes_a_rescued_twin_and_counts_it(blank):
    # Review focus 4: the source was dropped as p4c-infeasible, so the twin has no partner.
    df = _twin_frame()
    df = df[~((df.arm_slug == 'joint-off') & (df.M == 25.0) & (df.split == 0) & (df.k == 3))]
    pairs = claims.twin_pairs(df)
    assert len(pairs) == 17
    assert not ((pairs.M == 25.0) & (pairs.split == 0) & (pairs.k == 3)).any()
    # Neither the effect table nor the ladder sees the rescued cell.
    effect = claims.twin_effect(df)
    assert effect[(effect.group_kind == 'all') & (effect.metric == 'blocks')].iloc[0]['n_pairs'] == 17
    low = effect[(effect.group_kind == 'k_group') & (effect.group == 'low 1-5')
                 & (effect.metric == 'blocks')].iloc[0]
    assert low['n_pairs'] == 5
    ladder = claims.twin_ladder(df)
    assert ladder[ladder.M == 25.0].iloc[0]['n_cells'] == 8
    assert ladder[ladder.M == 'all'].iloc[0]['n_cells'] == 17
    counts = claims.twin_counts(_rescued_verification(blank))
    assert counts == {'n_twins': 3, 'n_identical': 1, 'n_compiled': 2, 'n_rescued': 1,
                      'n_twin_infeasible': 1, 'n_compiled_exact': 1,
                      'exact_rate_compiled': pytest.approx(0.5)}


def _text_flag_verification(over_stages='False', over_budget='False', verdict='EXACT'):
    """verification.csv as `load_verification` really returns it: flags are the
    literal text 'True'/'False' (dtype=str), copied_from '' for None."""
    rows = []
    for k in (3, 8):
        rows.append({'row_id': f'joint-off_M025_s00_k{k:02d}', 'arm_slug': 'joint-off',
                     'verdict': 'EXACT', 'p4c_over_stages': over_stages,
                     'p4c_over_budget': over_budget, 'copied_from': '', 'unverified': 'False'})
        rows.append({'row_id': f'joint-off-al_M025_s00_k{k:02d}', 'arm_slug': 'joint-off-al',
                     'verdict': verdict, 'p4c_over_stages': over_stages,
                     'p4c_over_budget': over_budget, 'copied_from': '', 'unverified': 'False'})
    return pd.DataFrame(rows)


def test_twin_counts_reads_literal_text_flags_from_the_csv():
    # Regression: 'False'.astype(bool) is True, so every twin looked p4c-infeasible.
    counts = claims.twin_counts(_text_flag_verification())
    assert counts['n_twins'] == 2
    assert counts['n_twin_infeasible'] == 0 and counts['n_rescued'] == 0


@pytest.mark.parametrize('over_stages, over_budget, infeasible', [
    ('True', 'False', 2), ('False', 'True', 2), (True, False, 2), (False, False, 0),
    ('False', False, 0), (False, 'True', 2)])
def test_twin_counts_text_and_bool_flags_agree(over_stages, over_budget, infeasible):
    counts = claims.twin_counts(_text_flag_verification(over_stages, over_budget))
    assert counts['n_twin_infeasible'] == infeasible


def _align_twin_frame():
    df = _twin_frame()
    twin = (df.arm_slug == 'joint-off-al').to_numpy()
    df['align_attempted'] = np.where(twin, 4.0, np.nan)
    df['align_accepted'] = np.where(twin, np.where(df.k == 3, 0.0, 2.0), np.nan)
    df['intervals_before'] = np.where(twin, 20.0, np.nan)
    df['intervals_after'] = np.where(twin, np.where(df.k == 3, 20.0, 17.0), np.nan)
    return df


def test_twin_alignment_stats_per_m_and_pooled():
    stats = claims.twin_alignment_stats(_align_twin_frame())
    assert list(stats.columns) == list(claims.ALIGNMENT_STATS_COLUMNS)
    assert list(stats['M']) == [25.0, 50.0, 'all']
    row = stats[stats.M == 'all'].iloc[0]
    assert row['n'] == 18
    assert row['mean_intervals_removed'] == pytest.approx(2.0)   # 3 on 2/3 of twins
    assert row['mean_align_attempted'] == 4.0
    assert row['mean_align_accepted'] == pytest.approx(4 / 3)
    assert row['share_no_accepted'] == pytest.approx(1 / 3)


def test_twin_alignment_stats_without_the_align_columns_is_empty():
    assert claims.twin_alignment_stats(_twin_frame()).empty
