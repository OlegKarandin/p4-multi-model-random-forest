"""P7b: the statistical claims layer (spec C.3).

Every number the results chapter states about the joint encoding comes
through this module. It consumes the frame contract documented in
`src/reporting/campaign_data.py` and turns it into five families of claim:

1. `pareto_front_3d` / `coverage_ratio_3d` -- is the joint arm's
   accuracy/accuracy/blocks trade-off surface better than the independent
   arm's?
2. `substitution_test` -- does either task pay for the other's gain?
3. `delta_frontier` -- how does each outcome move as the alignment
   tolerance delta is swept?
4. `ablation_decomposition` -- how much of the effect is the sharing
   constraint and how much is threshold alignment?
5. `paired_tests` -- are the accuracy claims survivable under a
   multiplicity correction?

Design decisions that a reader has to be able to audit
-----------------------------------------------------

**Why the front is 3-D.** The objectives are `(acc_app, acc_ddos, blocks)`
jointly, maximising the first two and minimising the third. Computing a 2-D
front per task independently would admit a point that is excellent on App
and terrible on DDoS -- exactly the trade this thesis exists to rule out.
`pareto_projections` exposes the three 2-D planes for plotting, but they are
PROJECTIONS of the 3-D front, never fronts recomputed inside a plane: a
projected point may look dominated in its plane and still belong on the
front, and hiding it would hide the trade.

**Why NaN is an error, not a filter.** Every NaN comparison is False, so a
NaN point is never dominated and would land on EVERY front. `load_campaign`
already drops infeasible rows (the only source of NaN accuracies in the
campaign), so this module should never see one -- which is precisely why it
raises instead of quietly dropping: a NaN arriving here means an upstream
invariant broke, and silently absorbing it would turn that break into a
plausible-looking wrong front.

**Why the Pareto relation is strict.** `a` dominates `b` iff `a` is no worse
on all three objectives and strictly better on at least one. Two identical
points therefore do not dominate each other and both stay on the front, and
`coverage_ratio_3d(A, A) == 0`. Zitzler's C metric is sometimes defined with
WEAK dominance, under which a set covers itself completely (C(A, A) == 1);
that convention makes "the joint arm covers X% of the independent arm's
front" unreadable when the two fronts share points, so the strict convention
is used here and `strict=False` is available for anyone who wants the other.

**Why one observation per split.** Replication in this campaign is at the
split level: every (M, k) cell inside one split was trained on the same data
split, so those cells are not independent observations. Wherever this module
forms a confidence interval it therefore aggregates to one number per split
first and puts the interval around the split-level mean, with a Student-t
quantile (not a normal one -- at 3-5 splits the difference is large). The
paired hypothesis tests are the deliberate exception, discussed below.

**One-sided versus two-sided.** Accuracy is tested one-sided and blocks
two-sided, and the two are not interchangeable:

* `acc_app`, `acc_ddos`: NO DETECTABLE LOSS. The alternative is
  `H1: median(joint - independent) > -margin`, i.e. `alternative='greater'`
  applied to `d + margin`. REJECTING H0 is the positive finding -- it says
  the joint arm is not worse by more than the margin. With the default
  `margin = 0` this reduces to `H1: median(d) > 0`: the joint arm shows no
  detectable loss against the baseline. That is deliberately NOT called
  non-inferiority, because non-inferiority is a claim against a
  pre-registered margin and no margin has been set; the word is used, in the
  emitted `hypothesis` column as well as here, only on the `margin > 0`
  branch, where it is earned. Reversing this to `alternative='less'` would
  test whether the joint arm IS worse, and reporting a large p-value from
  that as "no loss established" is a confident wrong answer, which is why
  `tests/test_claims.py` asserts the p-value in BOTH directions.
* `blocks`: two-sided. Sharing feature intervals could plausibly cost blocks
  as well as save them (alignment adds intervals before it merges any), so
  assuming a direction here would be assuming the result.

**The correction family, stated explicitly.** The compiler-verified campaign
(spec 2026-09-29, section 7.3) runs one independent arm plus TWO joint arms
(`JOINT_ARM_SLUGS`): `joint-off` (sharing only) and `joint-off-al` (sharing plus
post-selection threshold alignment: the aligned twin of each joint-off design -- the delta_align tolerance axis was deleted on
2026-09-15, so the seven archived `joint-d*` arms no longer exist). The
family is therefore

    2 contrasts   joint-off, joint-off-al -- each against `independent`
  x 5 tests       acc_app (one-sided), f1_app (one-sided),
                  acc_ddos (one-sided), f1_ddos (one-sided),
                  blocks (two-sided)
  = 10 comparisons

and every family size below is DERIVED from `JOINT_ARM_SLUGS` and asserted,
never written as a literal (the stale 35 = 7 x 5 of the archived sweep is
exactly the mistake a literal invites). The archived seven-arm frames can
still be analysed by passing `arms=` explicitly.

`PRE_REGISTERED_FAMILY_SIZE` is that 10, `default_contrast_family` builds
those contrasts, and `paired_tests` reports `n_comparisons` on every
row so a reader can check the correction covered what was actually run.
Pass `expected_family_size=PRE_REGISTERED_FAMILY_SIZE` to make a shrunken
family (an arm missing from the frame) an error rather than a quietly weaker
correction. Holm-Bonferroni is applied across all 10 at once -- not per task,
not per arm. `f1_app`/`f1_ddos` read the same one-sided direction as their
accuracy counterparts -- "small p is the positive finding" -- because a
per-class F1 collapse that raw accuracy hides (spec R3 section IV(d)) is
exactly the failure mode this family exists to catch.

**The substitution tests are a SECOND, SEPARATE family, and they are not
part of the pre-registered 10.** `substitution_test` returns six p-value
fields, and `substitution_test_all_arms` runs it at every joint arm; taken
raw that is six uncorrected p-values and one uncorrected decision flag per
arm (on the archived seven-arm sweep, at least one of seven flags at alpha =
0.05 fired under the null roughly 30% of the time). They are kept out of the
pre-registered family deliberately -- folding
them in would dilute the Holm correction protecting the primary accuracy
claims with tests that answer a different question -- but
kept out is not the same as unreported, so:

* `substitution_test_all_arms` emits `pearson_p_negative_one_sided_holm`
  and `substitution_detected_holm`, Holm-corrected across the joint arms
  (`SUBSTITUTION_FAMILY_SIZE`), with `n_substitution_comparisons` recording
  how many arms actually yielded a defined test. Report the corrected flag.
* The other five p-value fields on each row stay uncorrected diagnostics
  and must be read as such.
* The correction direction here is self-penalising in a way the primary
  family is not: a false positive argues AGAINST the thesis, so an
  uncorrected flag errs towards over-reporting substitution rather than
  towards hiding it. That is a reason to read the flags carefully, not a
  reason to skip the correction.
* `substitution_test_all_arms` takes the same `expected_family_size` guard
  as `paired_tests` (Task 23): pass `expected_family_size=
  SUBSTITUTION_FAMILY_SIZE` to turn a family shrunk by an arm MISSING from
  the frame into an error, rather than a silently weaker Holm correction.
  This is checked against how many arms were actually run, which is a
  different count from `n_substitution_comparisons` (arms that yielded a
  DEFINED test) -- a missing arm and a constant-delta arm's undefined test
  are different conditions, and the guard fires only on the former.
* The same `(M, split, k)` dependence caveat that applies to
  `paired_tests` applies here: the correlations are computed over cells
  that share a training split within a split, so their p-values are
  anti-conservative relative to the number of independent splits.

**`noninferiority_tests` is a THIRD, SEPARATE family (D13), and it is not
part of the pre-registered 10 either.** It answers a different question from
`paired_tests`' default `margin=0`: not "is there no detectable loss" but
"is any loss small enough to call the two arms equivalent". Equivalence
needs a margin, and D13 sizes it as **5% of each row's OWN baseline
error**, per `(M, split, k)` cell -- not the flat accuracy-point constant
`paired_tests`' `margin` parameter adds, which would test App (around 0.70
accuracy, about 30% error) and DDoS (around 0.97, about 3% error) at
wildly different strictness. The family is the two joint arms x
`acc_app`/`acc_ddos` only (`f1_app`/`f1_ddos` stay in the superiority
family; they are not retested here) = 4 comparisons
(`NONINFERIORITY_FAMILY_SIZE`), Holm-corrected on its own -- never mixed
with the superiority family's p-values, because
the two families answer different questions (superiority vs equivalence)
and pooling their corrections would weaken both. See
`noninferiority_tests` for the exact H0/H1 and the pre-registration
constraint on the margin itself.

`ablation_decomposition` deliberately reports NO p-values. Its two contrasts
(`joint-off - independent` and `joint-delta - joint-off`) are a descriptive
decomposition of where the effect comes from; adding tests there would
enlarge the multiplicity family without enlarging the claim, so it reports
effect sizes with split-level confidence intervals instead.

**Where the paired tests knowingly bend the independence assumption.** Spec
C.3 pairs on `(M, split, k)`, so `paired_tests` defaults to one paired
observation per cell (`unit='pair'`). Cells within a split share a training
split, so the effective sample size is smaller than `n_pairs` and the
p-values are anti-conservative. That is the spec's choice and the default
here, but `unit='split'` runs the same tests on one mean difference per
split -- valid under the split-level replication argument, at much lower
power -- as an available robustness check. Both `n_pairs` and `n_splits`
are reported on every row so the gap is visible.

**Libraries.** scipy only (`wilcoxon`, `pearsonr`, `spearmanr`, `rankdata`,
Student-t quantiles). statsmodels is not installed in this environment, so
Holm-Bonferroni and the partial correlation are implemented here; the test
suite checks Holm against a hand-computed table always, and against
`statsmodels.stats.multitest.multipletests` wherever statsmodels happens to
be importable.

**`hypervolume_2d` and the per-M reference point (D5, amended by A2).** An
earlier note here said `calculate_hypervolume_2d` (`analysis.py`) was
deliberately not ported, because its reference point -- the published fixed
`(0.5, 100)` -- was arbitrary. A2 dissolves that objection instead of
accepting it: `hypervolume_2d` takes a `reference` argument, and the caller
passes `(0.5, M)` per M-value rather than a single constant. A fixed 100
would silently discard most of the front once M sweeps up to this plan's
150-block grid point; a per-M budget stays meaningful at every M. The
deliberate cost is comparability -- this module's hypervolume numbers do
NOT reproduce the published Fig. results_2a, which used the fixed (0.5,
100) reference, and that divergence is reported, not hidden. Hypervolume
is computed per task on `(acc_<task>, blocks)` pairs (`TASK_FRONT_OBJECTIVES`),
never on the two tasks' accuracies averaged together -- that averaging
(`extract_approach_data`'s `avg_accuracy = (acc_app + acc_ddos) / 2` in the
deleted module) is exactly what this whole rerun exists to undo, since it
hides the k-dependent, per-task finding behind a single blended number.
Report hypervolume as gain relative to the independent arm, matching the
paper's framing, even though the absolute values differ from what is
published.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.p4model.target import TCAM_BLOCKS_PER_STAGE, TOFINO_PIPELINE_STAGES
from src.reporting.campaign_data import pair_arms, TWIN_ARM_SLUG
from src.training.campaign_run import DEVELOPMENT_SPLITS

INDEPENDENT_ARM_SLUG = 'independent'

# The compiler-verified campaign's joint arms (spec 2026-09-29 section 7.3),
# in sweep order: sharing alone, then sharing plus threshold alignment.
# Replaces the archived seven-arm sweep (`joint-off`, `joint-d000` ...
# `joint-dinf`), whose six delta arms can no longer be produced since the
# 2026-09-15 deletion of the delta_align axis (Track 5: delta_helps = FALSE).
# Every family size below is derived from this tuple, so the pre-registered
# family follows the grid instead of silently drifting from it.
# `figures.ordered_arms` APPENDS an unrecognised slug rather than dropping it,
# so an archived `joint-d*` arm still appears in every figure.
JOINT_ARM_SLUGS = ('joint-off', TWIN_ARM_SLUG)
# The in-search aligned arm `joint` of campaign_2026_10 is a legacy slug
# (spec 2026-10-06 decision 4): still loadable, appended by
# `figures.ordered_arms`, never a family member.
LEGACY_ARM_SLUGS = ('joint',)

# Blocks available to a whole Tofino pipe. The hypervolume reference budget
# for the unbudgeted cell (M = inf), where `(0.5, M)` would be an infinite
# reference (O2): the pipe's own capacity is the natural finite stand-in.
UNBUDGETED_REFERENCE_BLOCKS = TCAM_BLOCKS_PER_STAGE * TOFINO_PIPELINE_STAGES

# The three outcomes of spec C.3, in front-objective order.
FRONT_OBJECTIVES = ('acc_app', 'acc_ddos', 'blocks')

# True = larger is better. Blocks are a cost, so the front is computed on
# (acc_app, acc_ddos, -blocks) as the spec states.
FRONT_MAXIMIZE = (True, True, False)

# Per-task 2-D objective pairs for `hypervolume_2d` (D5, amended by A2).
# Deliberately SEPARATE from FRONT_OBJECTIVES, which stays 3-D (D4) -- this
# is not a reuse or a mutation of it. Each task gets its own (accuracy,
# blocks) pair rather than the deleted module's averaged accuracy, because
# averaging the two tasks together is exactly what this rerun exists to undo.
TASK_FRONT_OBJECTIVES = {
    'app': ('acc_app', 'blocks'),
    'ddos': ('acc_ddos', 'blocks'),
}

# True = larger is better, applied to each TASK_FRONT_OBJECTIVES pair:
# maximize accuracy, minimize blocks.
TASK_FRONT_MAXIMIZE = (True, False)

DEFAULT_METRICS = ('acc_app', 'f1_app', 'acc_ddos', 'f1_ddos', 'blocks')

# Which alternative each metric's paired test encodes. See the module
# docstring -- getting this table backwards is the expensive mistake.
METRIC_ALTERNATIVE = {
    'acc_app': 'greater',
    'f1_app': 'greater',
    'acc_ddos': 'greater',
    'f1_ddos': 'greater',
    'blocks': 'two-sided',
}

# Joint arms x 5 tests, derived from the grid (spec section 7.3) and asserted
# so a changed grid cannot quietly change the family size.
PRE_REGISTERED_FAMILY_SIZE = len(JOINT_ARM_SLUGS) * len(DEFAULT_METRICS)
assert PRE_REGISTERED_FAMILY_SIZE == 10

# The SEPARATE substitution family: one one-sided correlation test per joint
# arm. Explicitly not folded into the pre-registered family -- see the module
# docstring -- but Holm-corrected across its own arms so the per-arm
# decision flags are not read raw.
SUBSTITUTION_FAMILY_SIZE = len(JOINT_ARM_SLUGS)
assert SUBSTITUTION_FAMILY_SIZE == 2

# D13's non-inferiority family: acc_app and acc_ddos only. F1 stays in the
# pre-registered superiority family -- it is not retested here.
NONINFERIORITY_METRICS = ('acc_app', 'acc_ddos')

# D13's pre-registered margin: 5% of EACH ROW's OWN baseline error, not an
# absolute accuracy-point constant. See `noninferiority_tests`'s docstring
# for why `paired_tests`' `margin` parameter cannot be reused for this.
NONINFERIORITY_MARGIN = 0.05

# Joint arms x 2 accuracy metrics (NONINFERIORITY_METRICS). A THIRD family,
# separate from PRE_REGISTERED_FAMILY_SIZE's and SUBSTITUTION_FAMILY_SIZE's --
# see the module docstring and `noninferiority_tests`.
NONINFERIORITY_FAMILY_SIZE = len(JOINT_ARM_SLUGS) * len(NONINFERIORITY_METRICS)
assert NONINFERIORITY_FAMILY_SIZE == 4

# Wilcoxon zero handling. The default 'wilcox' DISCARDS tied pairs, which
# throws away the observations that most directly support a no-detectable-loss
# claim and shrinks n; 'pratt' keeps them but raises outright when every
# difference is zero (a perfectly possible outcome for the joint-off arm on
# a metric it cannot move). 'zsplit' keeps the zeros and splits their ranks
# between the two sides, so it is both the conservative choice and the only
# one that stays defined in the degenerate case.
_ZERO_METHOD = 'zsplit'

_PROJECTION_PLANES = {
    'acc_app_vs_blocks': ('blocks', 'acc_app'),
    'acc_ddos_vs_blocks': ('blocks', 'acc_ddos'),
    'acc_ddos_vs_acc_app': ('acc_app', 'acc_ddos'),
}

_IDENTITY_COLUMNS = ('arm_slug', 'M', 'split', 'k')


# ---------------------------------------------------------------------------
# Pareto front and coverage
# ---------------------------------------------------------------------------

def _objective_matrix(df, objectives, maximize, label):
    """Extract the objective columns as a float matrix already sign-flipped
    so that LARGER IS BETTER on every column, and refuse anything non-finite.

    The refusal is the point: NaN compares False against everything, so a NaN
    row is never dominated and lands on every front. `load_campaign` filters
    the campaign's only source of NaN accuracies (infeasible rows) at load,
    so a NaN reaching here means that invariant broke upstream and the right
    response is to say so, not to guess.
    """
    missing = [c for c in objectives if c not in df.columns]
    if missing:
        raise KeyError(
            '{}: missing objective column(s) {}'.format(label, missing))

    matrix = df.loc[:, list(objectives)].to_numpy(dtype='float64')
    if matrix.size:
        finite = np.isfinite(matrix)
        if not finite.all():
            bad_rows = df.index[~finite.all(axis=1)].tolist()
            raise ValueError(
                '{}: objective columns {} contain NaN or infinite values at '
                'index {} -- a NaN is never dominated and would land on every '
                'Pareto front, so it is rejected rather than silently kept. '
                'Infeasible rows should already have been dropped by '
                'load_campaign.'.format(label, list(objectives), bad_rows))

    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    return matrix * signs


def _dominates(better, worse):
    """`better[i]` dominates `worse[j]` iff it is no worse on every objective
    and strictly better on at least one (strict Pareto dominance). Both
    inputs are already sign-flipped to larger-is-better."""
    if better.size == 0 or worse.size == 0:
        return np.zeros((better.shape[0], worse.shape[0]), dtype=bool)
    no_worse = (better[:, None, :] >= worse[None, :, :]).all(axis=2)
    strictly_better = (better[:, None, :] > worse[None, :, :]).any(axis=2)
    return no_worse & strictly_better


def pareto_front_3d(df, objectives=FRONT_OBJECTIVES, maximize=FRONT_MAXIMIZE):
    """The non-dominated subset of `df` on `(acc_app, acc_ddos, -blocks)`.

    Computed in 3-D on purpose: a 2-D front per task would admit a point that
    is excellent on one task and terrible on the other, which is the trade
    the thesis exists to rule out. Use `pareto_projections` to get the 2-D
    planes for plotting.

    Returns the original rows (all columns, original index preserved), so the
    caller keeps `arm_slug` / `M` / `split` / `k` for colouring and joining.
    Exactly duplicated points are all kept -- neither copy dominates the
    other. Raises ValueError if any objective value is NaN or infinite.
    """
    points = _objective_matrix(df, objectives, maximize, 'pareto_front_3d')
    if len(df) == 0:
        return df.copy()
    dominated = _dominates(points, points).any(axis=0)
    return df.loc[~dominated].copy()


def _pareto_front_2d(df, objectives, maximize, label):
    """Same dominance logic as `pareto_front_3d`, generalized to any
    objective count -- used here for the 2-D (accuracy, blocks) front
    `hypervolume_2d` needs, which is intentionally NOT the 3-D front
    `pareto_front_3d` computes (a point can be 2-D-dominated on one task's
    plane while remaining on the pooled 3-D front; hypervolume is a
    per-task, per-M statistic, not a restatement of the 3-D front).
    """
    points = _objective_matrix(df, objectives, maximize, label)
    if len(df) == 0:
        return df.copy()
    dominated = _dominates(points, points).any(axis=0)
    return df.loc[~dominated].copy()


def pareto_projections(front):
    """The three 2-D planes of a 3-D front, for plotting.

    These are PROJECTIONS of the 3-D front, not fronts recomputed within each
    plane. A point can look dominated in one plane and still belong on the
    3-D front -- e.g. a solution with the best App accuracy but poor DDoS
    accuracy disappears from a (blocks, acc_ddos) 2-D front while remaining a
    genuine non-dominated trade-off. Dropping it would hide the trade.

    The three planes are fixed by `_PROJECTION_PLANES` and are not
    re-targetable -- an earlier signature took an `objectives` argument it
    never read, which would have told the figures task otherwise.

    Returns {plane_name: DataFrame}, each sorted ascending on its x axis so a
    line plot through the points is well defined, and carrying whichever of
    `arm_slug` / `M` / `split` / `k` are present.
    """
    carried = [c for c in _IDENTITY_COLUMNS if c in front.columns]
    projections = {}
    for name, (x_col, y_col) in _PROJECTION_PLANES.items():
        if x_col not in front.columns or y_col not in front.columns:
            continue
        columns = [x_col, y_col] + [c for c in carried if c not in (x_col, y_col)]
        projections[name] = front.loc[:, columns].sort_values(
            [x_col, y_col]).reset_index(drop=True)
    return projections


def coverage_ratio_3d(a, b, objectives=FRONT_OBJECTIVES, maximize=FRONT_MAXIMIZE,
                      strict=True):
    """Fraction of `b`'s points dominated by at least one point of `a`
    (Zitzler's C metric, in 3-D).

    `strict=True` (the default) uses strict Pareto dominance, so a point
    never covers its own copy and `coverage_ratio_3d(A, A) == 0`. The weak
    variant (`strict=False`) counts "no worse on every objective" as coverage,
    under which a set covers itself completely; it is offered because the
    literature uses both, but the strict reading is what the thesis reports,
    because "the joint front covers X% of the independent front" is only
    interpretable if shared points do not inflate X.

    Returns NaN when `b` is empty -- the ratio is undefined, and returning 0
    would read as "a dominates nothing", which is a different statement.
    Raises ValueError if either set contains a NaN or infinite objective.
    """
    a_points = _objective_matrix(a, objectives, maximize, 'coverage_ratio_3d(a)')
    b_points = _objective_matrix(b, objectives, maximize, 'coverage_ratio_3d(b)')
    if b_points.shape[0] == 0:
        return float('nan')
    if a_points.shape[0] == 0:
        return 0.0
    if strict:
        covered = _dominates(a_points, b_points).any(axis=0)
    else:
        covered = (a_points[:, None, :] >= b_points[None, :, :]).all(axis=2).any(axis=0)
    return float(covered.mean())


def hypervolume_2d(front, reference):
    """2-D hypervolume of `front` above `reference`, in the (accuracy,
    blocks) plane -- accuracy maximised, blocks minimised (D5, amended by
    A2).

    `front` is a plain sequence of `(accuracy, blocks)` pairs -- this
    function is data-source-agnostic; a caller extracts the pairs for one
    task from a campaign frame before calling it. `reference` is
    `(min_accuracy, max_blocks)`: the area credited is the region that
    beats the reference on BOTH axes, i.e. `accuracy >= reference[0] and
    blocks <= reference[1]`.

    This recovers the shape of the deleted `analysis.calculate_hypervolume_2d`
    but fixes three defects rather than reproducing them:

    1. Boundary-inclusive, not strict. The deleted version filtered with
       `acc > ref[0]` and `mem < ref[1]`, silently dropping any solution
       landing exactly on the reference budget. A solution exactly on the
       boundary is kept here; it simply contributes zero width (or zero
       height) to the sum, which falls out of the arithmetic on its own
       rather than needing a special case.
    2. NaN, not 0, for an EMPTY front. Matches `coverage_ratio_3d`'s
       convention for its empty-`b` case: 0 would read as "a real
       measurement of zero gain", which is a different, false claim from "no
       front was measured here" -- e.g. every cell at this (M, task)
       infeasible. This is distinct from a NON-empty front where no point
       reaches the reference: that IS a real, measurable 0 (no solution
       beats the budget on both axes), exactly `coverage_ratio_3d`'s
       "`a` dominates nothing" case, and is returned as `0.0`, not NaN.
    3. No hardcoded reference. The deleted version defaulted to the fixed,
       published `(0.5, 100)`. A2 requires a PER-M reference `(0.5, M)`
       instead -- the caller passes it in; this function does not assume a
       value.

    Returns a float. `float('nan')` only when `front` itself is empty;
    `0.0` (not NaN) when `front` has points but none reach the reference on
    both axes.
    """
    if not front:
        return float('nan')

    ref_accuracy, ref_blocks = reference
    valid = [(accuracy, blocks) for accuracy, blocks in front
             if accuracy >= ref_accuracy and blocks <= ref_blocks]
    if not valid:
        return 0.0

    # Sort by blocks ascending, then sweep from the most expensive (closest
    # to the reference budget) down to the cheapest, accumulating the
    # rectangle each point adds over the previous one's blocks level.
    sorted_front = sorted(valid, key=lambda point: point[1])
    hv = 0.0
    prev_blocks = ref_blocks
    for accuracy, blocks in reversed(sorted_front):
        width = prev_blocks - blocks
        height = accuracy - ref_accuracy
        hv += width * height
        prev_blocks = blocks
    return hv


def _reference_blocks(M):
    """The hypervolume reference budget at `M`: `M` itself, or
    `UNBUDGETED_REFERENCE_BLOCKS` for the unbudgeted cell (O2), where an
    infinite reference would make every hypervolume infinite."""
    return UNBUDGETED_REFERENCE_BLOCKS if np.isinf(M) else M


def hypervolume_by_arm(df, baseline=INDEPENDENT_ARM_SLUG, arms=None):
    """Per-(arm, M, task) 2-D hypervolume, reported as gain relative to
    `baseline` (D5, amended by A2 -- see the module docstring's
    "`hypervolume_2d` and the per-M reference point" section for the full
    amendment, including why the reference is `(0.5, M)` per M-value rather
    than the published fixed `(0.5, 100)`).

    For each `M` present in `df['M'].unique()`, each arm (`arms`, or --
    when not given -- every arm present in `df`, `baseline` first then
    `JOINT_ARM_SLUGS` in sweep order, mirroring `figures.ordered_arms`), and
    each task in `TASK_FRONT_OBJECTIVES`: pools every split and k at that
    `(arm, M)` cell (matching how the rest of this module pools
    replication -- see `pareto_front_3d`'s own per-arm pooling), reduces it
    to a genuine 2-D Pareto front via `_pareto_front_2d` on that task's
    `TASK_FRONT_OBJECTIVES` pair -- deliberately NOT `pareto_front_3d`'s
    3-D front, see `_pareto_front_2d`'s docstring -- and calls
    `hypervolume_2d` on the filtered `(accuracy, blocks)` pairs against the
    per-M reference `(0.5, M)` -- or `(0.5, UNBUDGETED_REFERENCE_BLOCKS)`
    (288, the whole pipe) at `M = inf` (O2). Pareto-filtering first is not optional:
    `hypervolume_2d`'s sweep assumes the front it is given is already
    non-dominated, and a raw, unfiltered set of cells would silently give a
    wrong hypervolume.

    Returns a tidy frame, one row per `(arm_slug, M, task)`:

    * `hypervolume` -- this arm's 2-D hypervolume at that (M, task).
    * `baseline_hypervolume` -- `baseline`'s hypervolume at the SAME
      (M, task), computed the same way (pooled, then Pareto-filtered).
    * `hypervolume_gain` -- `hypervolume / baseline_hypervolume`. NaN
      (never inf, never a ZeroDivisionError) when the baseline hypervolume
      is `0.0` or NaN -- matching this module's existing NaN-for-undefined
      convention (e.g. `coverage_ratio_3d`'s empty-`b` case): a ratio over
      an undefined or zero denominator is not a real gain number and must
      not be reported as one.

    `task` is `'app'` or `'ddos'` (the two `TASK_FRONT_OBJECTIVES` keys),
    never the two tasks' accuracies averaged together -- see the module
    docstring for why that averaging is exactly what this rerun exists to
    undo.
    """
    if len(df) == 0 or 'M' not in df.columns:
        m_values = []
    else:
        m_values = sorted(df['M'].unique().tolist())

    if arms is None:
        present = set(df['arm_slug'].unique()) if len(df) else set()
        known = [baseline] + list(JOINT_ARM_SLUGS)
        ordered = [slug for slug in known if slug in present]
        extras = [slug for slug in present if slug not in known]
        arms = tuple(ordered + extras)
    else:
        arms = tuple(arms)

    def _hv(arm, M, task):
        objectives = TASK_FRONT_OBJECTIVES[task]
        cell = df[(df['arm_slug'] == arm) & (df['M'] == M)]
        front = _pareto_front_2d(cell, objectives, TASK_FRONT_MAXIMIZE,
                                 'hypervolume_by_arm')
        pairs = list(zip(front[objectives[0]], front[objectives[1]]))
        return hypervolume_2d(pairs, reference=(0.5, _reference_blocks(M)))

    rows = []
    for M in m_values:
        baseline_hv_by_task = {task: _hv(baseline, M, task)
                               for task in TASK_FRONT_OBJECTIVES}
        for arm in arms:
            for task in TASK_FRONT_OBJECTIVES:
                hv = _hv(arm, M, task)
                baseline_hv = baseline_hv_by_task[task]
                if not np.isfinite(baseline_hv) or baseline_hv == 0.0:
                    gain = float('nan')
                else:
                    gain = hv / baseline_hv
                rows.append({
                    'arm_slug': arm,
                    'M': M,
                    'task': task,
                    'hypervolume': hv,
                    'baseline_hypervolume': baseline_hv,
                    'hypervolume_gain': gain,
                })

    return pd.DataFrame(rows, columns=[
        'arm_slug', 'M', 'task', 'hypervolume', 'baseline_hypervolume',
        'hypervolume_gain'])


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def arm_deltas(df, treatment, baseline=INDEPENDENT_ARM_SLUG, metrics=DEFAULT_METRICS):
    """Paired treatment-minus-baseline differences, one row per `(M, split,
    k)` cell present in BOTH arms.

    Pairing goes through `campaign_data.pair_arms`, whose join key includes
    `M`. That is not optional: the legacy `perform_statistical_analysis`
    keyed on `(split, k)` alone and silently collapsed the seven M files into
    one, last-wins.

    Returns columns `M`, `split`, `k`, and `d_<metric>` for each metric,
    signed as `treatment - baseline` throughout -- so on `blocks`, a negative
    delta means the treatment arm SAVED blocks.
    """
    paired = pair_arms(df, treatment, baseline)
    out = pd.DataFrame({
        'M': paired['M'] if len(paired) else pd.Series(dtype='float64'),
        'split': paired['split'] if len(paired) else pd.Series(dtype='int64'),
        'k': paired['k'] if len(paired) else pd.Series(dtype='int64'),
    })
    for metric in metrics:
        treatment_col = '{}_treatment'.format(metric)
        baseline_col = '{}_baseline'.format(metric)
        if len(paired) == 0:
            out['d_{}'.format(metric)] = pd.Series(dtype='float64')
        else:
            out['d_{}'.format(metric)] = (
                paired[treatment_col].astype('float64')
                - paired[baseline_col].astype('float64'))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def _safe_correlation(x, y, corr_func):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan'), float('nan')
    result = corr_func(x, y)
    return float(result[0]), float(result[1])


def _safe_pearson(x, y):
    return _safe_correlation(x, y, stats.pearsonr)


def _safe_spearman(x, y):
    return _safe_correlation(x, y, stats.spearmanr)


def _partial_correlation(x, y, z):
    """Pearson partial correlation of x and y controlling for z, with a
    two-sided p-value.

    Uses the closed form r_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 -
    r_yz^2)), tested as t = r sqrt((n - 3) / (1 - r^2)) on n - 3 degrees of
    freedom (n - 2 - 1 controlled variable). Returns NaN rather than 0 when
    any input is constant or n < 4 -- an undefined correlation is not a zero
    one.
    """
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return float('nan'), float('nan')
    r_xy = float(stats.pearsonr(x, y)[0])
    r_xz = float(stats.pearsonr(x, z)[0])
    r_yz = float(stats.pearsonr(y, z)[0])
    denominator = np.sqrt((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2))
    if denominator == 0:
        return float('nan'), float('nan')
    r = (r_xy - r_xz * r_yz) / denominator
    r = float(np.clip(r, -1.0, 1.0))
    if abs(r) >= 1.0:
        return r, 0.0
    dof = n - 3
    t_stat = r * np.sqrt(dof / (1.0 - r ** 2))
    p = float(2.0 * stats.t.sf(abs(t_stat), dof))
    return r, p


def _one_sided_negative_p(r, two_sided_p):
    """Convert a two-sided correlation p-value into the one-sided p for the
    alternative `rho < 0` (the substitution direction). A positive sample
    correlation gives a p-value above 0.5, which is what makes the test
    unable to fire on a positive association."""
    if np.isnan(r) or np.isnan(two_sided_p):
        return float('nan')
    return two_sided_p / 2.0 if r < 0 else 1.0 - two_sided_p / 2.0


def _quadrant_fractions(d_app, d_ddos):
    """Sign-quadrant fractions of the two task deltas.

    Exact ties (either delta exactly zero, which accuracy differences produce
    often) are put in their own `on_axis` bucket rather than being forced
    into a quadrant by a sign convention -- a cell where one task did not
    move is not evidence of substitution in either direction. All five
    fractions are over the same denominator and sum to 1, provided no input
    delta is NaN -- a NaN delta falls into none of the five buckets (since
    NaN comparisons are all False) and the fractions sum to less than 1 in
    that case.
    """
    n = len(d_app)
    if n == 0:
        return {k: float('nan') for k in
                ('both_up', 'both_down', 'app_up_ddos_down',
                 'app_down_ddos_up', 'on_axis')}
    app_up, app_down = d_app > 0, d_app < 0
    ddos_up, ddos_down = d_ddos > 0, d_ddos < 0
    return {
        'both_up': float(np.mean(app_up & ddos_up)),
        'both_down': float(np.mean(app_down & ddos_down)),
        'app_up_ddos_down': float(np.mean(app_up & ddos_down)),
        'app_down_ddos_up': float(np.mean(app_down & ddos_up)),
        'on_axis': float(np.mean((d_app == 0) | (d_ddos == 0))),
    }


def substitution_test(df, treatment, baseline=INDEPENDENT_ARM_SLUG, alpha=0.05):
    """Does one task's gain come at the other's expense in `treatment`?

    Correlates the paired per-cell deltas `(d_acc_app, d_acc_ddos)`. A
    NEGATIVE correlation is the substitution signature: cells where App
    improves are cells where DDoS degrades. The flag `substitution_detected`
    encodes the ONE-SIDED alternative `rho < 0` at level `alpha` -- a strong
    POSITIVE correlation (both tasks moving together) must not trigger it,
    which is the opposite finding and is asserted as such in the tests.

    `partial_pearson_r` controls for `d_blocks`: two accuracy deltas can be
    correlated purely because both track how much TCAM the cell was allowed,
    and that shared driver is not substitution. `partial_spearman_r` is the
    same statistic computed on midranks, so a monotone-but-nonlinear relation
    or a heavy-tailed delta does not have to be trusted to the Pearson form;
    its p-value is the same parametric approximation applied to ranks, so
    treat it as indicative, not exact.

    All correlations are NaN (never 0) when a delta is constant or the pair
    count is too small -- an undefined correlation must not be reported as
    "no association", and `substitution_detected` is False in that case
    because nothing was detected.

    TWO CAVEATS ON THE p-VALUES THIS RETURNS, both of which the caller owns:

    1. They are UNCORRECTED. This function returns six p-value fields, and
       `substitution_test_all_arms` runs it at every joint arm; none of
       those values, and not `substitution_detected` either, belong to the
       pre-registered family that `paired_tests` corrects (see the module
       docstring for why they are kept separate). Prefer
       `substitution_test_all_arms`, which adds a Holm-corrected flag across
       the joint arms.
    2. The pairs are `(M, split, k)` cells, and cells inside one split share
       a training split, so they are not independent observations. The
       effective sample size is smaller than `n_pairs` and the p-values are
       anti-conservative -- the same caveat `paired_tests` carries.
       `n_splits` is returned alongside `n_pairs` so the gap is visible.
    """
    deltas = arm_deltas(df, treatment, baseline)
    d_app = deltas['d_acc_app'].to_numpy(dtype='float64')
    d_ddos = deltas['d_acc_ddos'].to_numpy(dtype='float64')
    d_blocks = deltas['d_blocks'].to_numpy(dtype='float64')

    pearson_r, pearson_p = _safe_pearson(d_app, d_ddos)
    spearman_rho, spearman_p = _safe_spearman(d_app, d_ddos)
    partial_r, partial_p = _partial_correlation(d_app, d_ddos, d_blocks)
    partial_rank_r, partial_rank_p = _partial_correlation(
        stats.rankdata(d_app), stats.rankdata(d_ddos), stats.rankdata(d_blocks))

    pearson_negative_p = _one_sided_negative_p(pearson_r, pearson_p)
    detected = bool(
        not np.isnan(pearson_r) and pearson_r < 0
        and not np.isnan(pearson_negative_p) and pearson_negative_p < alpha)

    return {
        'treatment': treatment,
        'baseline': baseline,
        'n_pairs': int(len(deltas)),
        'n_splits': int(deltas['split'].nunique()) if len(deltas) else 0,
        'pearson_r': pearson_r,
        'pearson_p_two_sided': pearson_p,
        'pearson_p_negative_one_sided': pearson_negative_p,
        'spearman_rho': spearman_rho,
        'spearman_p_two_sided': spearman_p,
        'spearman_p_negative_one_sided': _one_sided_negative_p(spearman_rho, spearman_p),
        'partial_pearson_r': partial_r,
        'partial_pearson_p_two_sided': partial_p,
        'partial_spearman_r': partial_rank_r,
        'partial_spearman_p_two_sided': partial_rank_p,
        'quadrants': _quadrant_fractions(d_app, d_ddos),
        'alpha': alpha,
        'substitution_detected': detected,
    }


def substitution_test_all_arms(df, baseline=INDEPENDENT_ARM_SLUG, arms=None,
                               alpha=0.05, expected_family_size=None):
    """`substitution_test` at EVERY joint arm, one row per arm.

    Run at every arm rather than only at the largest delta because the claim
    being defended is "no task sacrifices itself for the other at ANY
    tolerance"; testing only the extreme would leave the interesting middle
    of the sweep unexamined. Arms absent from `df` are skipped, so a partial
    campaign still produces a table -- unless `expected_family_size` is
    given, mirroring `paired_tests`: passing
    `expected_family_size=SUBSTITUTION_FAMILY_SIZE` turns a family shrunk by
    a MISSING arm (fewer arms in `df` than `JOINT_ARM_SLUGS`) into an
    error rather than a quietly weaker Holm correction. This is checked
    against how many arms were actually run (`len(arms)`), not against
    `n_substitution_comparisons`: an arm that ran but produced no DEFINED
    test (a constant-delta arm) is a different condition from an arm that
    never appeared in the frame at all, and only the latter is what this
    guard exists to catch.

    THIS IS A SEPARATE FAMILY FROM THE PRE-REGISTERED ONE. Running one
    one-sided test per arm means one decision flag per arm (on the archived
    seven-arm sweep, at least one fired under the null roughly 30% of the
    time at alpha = 0.05), so the raw
    `substitution_detected` must not be read across the sweep as if it were
    a single test. Two extra columns fix that:

    * `pearson_p_negative_one_sided_holm` -- the flag-driving p-value,
      Holm-corrected across the arms in THIS table only. It is deliberately
      not pooled with `paired_tests`' family: folding a different question into
      that family would dilute the correction protecting the primary
      accuracy claims.
    * `substitution_detected_holm` -- the corrected decision. Report this
      one; `substitution_detected` is kept alongside as the uncorrected
      per-arm result, not as a second opinion.

    `n_substitution_comparisons` records how many arms yielded a DEFINED
    test and were therefore corrected over: an arm whose deltas were
    constant produced no test at all (NaN, not a null result), so including
    it would inflate the family with a comparison nobody ran. That count is
    `SUBSTITUTION_FAMILY_SIZE` on a complete campaign.

    The `(M, split, k)` dependence caveat from `substitution_test` applies to
    every p-value here, corrected or not: cells inside a split share a
    training split, so these p-values are anti-conservative relative to the
    number of independent splits.
    """
    arms = _arms_present(df, arms)
    if expected_family_size is not None and len(arms) != expected_family_size:
        raise ValueError(
            'substitution_test_all_arms: found {} arm(s) but the expected '
            'correction family size is {}. Running a subset of the '
            'pre-registered family weakens the Holm correction for every '
            'comparison in it, so this is refused rather than silently '
            'accepted. Arms found: {}.'
            .format(len(arms), expected_family_size, list(arms)))
    rows = []
    for arm in arms:
        result = substitution_test(df, arm, baseline, alpha=alpha)
        quadrants = result.pop('quadrants')
        result.update({'quadrant_{}'.format(k): v for k, v in quadrants.items()})
        rows.append(result)
    table = pd.DataFrame(rows)
    if len(table) == 0:
        return table

    raw = table['pearson_p_negative_one_sided'].to_numpy(dtype='float64')
    defined = np.isfinite(raw)
    corrected = np.full(raw.shape, float('nan'))
    if defined.any():
        corrected[defined] = holm_bonferroni(raw[defined])
    table['pearson_p_negative_one_sided_holm'] = corrected
    table['n_substitution_comparisons'] = int(defined.sum())
    table['substitution_detected_holm'] = (
        table['substitution_detected'] & (corrected < alpha))
    return table


# ---------------------------------------------------------------------------
# Delta frontier
# ---------------------------------------------------------------------------

def _t_interval(values, confidence):
    """Mean and Student-t confidence interval of a set of split-level
    observations. t rather than z because the campaign replicates over a
    handful of splits, where the normal quantile is materially too narrow
    (at n = 3 the 95% t quantile is 4.30 against z's 1.96). NaN interval when
    n < 2, where the spread is unestimable."""
    values = np.asarray(values, dtype='float64')
    n = len(values)
    mean = float(np.mean(values)) if n else float('nan')
    if n < 2:
        return mean, float('nan'), float('nan'), float('nan'), float('nan')
    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(n)
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sem
    return mean, sd, sem, mean - half, mean + half


def delta_frontier(df, metrics=DEFAULT_METRICS,
                   group_columns=('arm_slug', 'M', 'k'), confidence=0.95,
                   allow_repeated_splits=False):
    """Aggregate each outcome across splits, with a mean and a CI per group.

    The default grouping is `(arm_slug, M, k)`, which leaves exactly one
    observation per split inside a group -- the assumption the interval rests
    on. Grouping more coarsely (say by `(arm_slug, M)`, pooling k) puts many
    correlated cells from the same split into one interval and makes it too
    narrow, so that is refused unless `allow_repeated_splits=True` says the
    caller means it.

    Returns long format: one row per group per metric, with `n` (observations
    in the interval), `n_splits` (distinct splits behind them -- equal to `n`
    unless pooling was allowed), `mean`, `sd`, `sem`, `ci_low`, `ci_high`.
    `delta_align_num` / `delta_align_is_inf` are carried through when grouping
    by `arm_slug`, so the sweep can be ordered numerically without ever
    parsing the raw `delta_align` string.
    """
    group_columns = list(group_columns)
    missing = [c for c in group_columns + ['split'] + list(metrics)
               if c not in df.columns]
    if missing:
        raise KeyError('delta_frontier: missing column(s) {}'.format(missing))

    if not allow_repeated_splits:
        repeated = df.duplicated(subset=group_columns + ['split'])
        if repeated.any():
            offending = df.loc[repeated, group_columns].drop_duplicates()
            raise ValueError(
                'delta_frontier: group_columns {} leave more than one row per '
                'split (first offending groups:\n{}\n). Cells inside one split '
                'are not independent observations, so the confidence interval '
                'would be too narrow. Add the missing grouping column (usually '
                "k), or pass allow_repeated_splits=True to accept that."
                .format(group_columns, offending.head().to_string(index=False)))

    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].to_numpy(dtype='float64')
            mean, sd, sem, low, high = _t_interval(values, confidence)
            row = dict(base)
            row.update({
                'metric': metric,
                'n': int(len(values)),
                'n_splits': int(group['split'].nunique()),
                'mean': mean, 'sd': sd, 'sem': sem,
                'ci_low': low, 'ci_high': high,
                'confidence': confidence,
            })
            rows.append(row)

    table = pd.DataFrame(rows)
    if 'arm_slug' in group_columns:
        table = attach_delta_columns(table, df)
    return table


def attach_delta_columns(table, df):
    """Carry the parsed delta (`delta_align_num`, `delta_align_is_inf`) onto
    an arm-keyed table. Raises if an arm_slug carries more than one parsed
    delta, which would mean two different treatments were filed under one arm
    identity."""
    delta_columns = [c for c in ('delta_align_num', 'delta_align_is_inf')
                     if c in df.columns]
    if not delta_columns:
        return table
    mapping = df.loc[:, ['arm_slug'] + delta_columns].drop_duplicates()
    if mapping['arm_slug'].duplicated().any():
        raise ValueError(
            'attach_delta_columns: an arm_slug carries more than one parsed '
            'delta_align value:\n{}'.format(mapping.to_string(index=False)))
    return table.merge(mapping, on='arm_slug', how='left')


# ---------------------------------------------------------------------------
# Ablation decomposition
# ---------------------------------------------------------------------------

def _arms_present(df, arms):
    if arms is None:
        present = set(df['arm_slug'].unique())
        return tuple(slug for slug in JOINT_ARM_SLUGS if slug in present)
    return tuple(arms)


def ablation_decomposition(df, metrics=DEFAULT_METRICS, confidence=0.95):
    """Split the joint arm's effect into its two causes.

    Two contrasts, and the second's baseline is the point of the whole
    function:

    * `sharing`   : `joint-off - independent`. `joint-off` skips the
      `align_rf_thresholds` call entirely, so it is prediction-identical to
      the unaligned models and the difference isolates the SHARING
      constraint on its own.
    * `alignment` : `joint-off-al - joint-off`, the aligned twin against its
      own source forest.
      Measured against `joint-off`, NOT against `independent` -- against
      `independent` it would re-count the sharing effect inside every
      alignment number and the two components would not add up.

    Descriptive only: no p-values. These contrasts decompose where the effect
    comes from; testing them too would enlarge the multiplicity family
    (`paired_tests`) without enlarging the claim.

    The confidence interval is built over SPLIT-LEVEL mean differences, not
    over every `(M, split, k)` cell, because cells inside one split share a
    training split and are not independent. `mean_diff_pairwise` and
    `median_diff_pairwise` are reported alongside for transparency; they
    differ from `mean_diff_split_level` whenever the design is unbalanced.
    """
    contrasts = [('sharing', 'joint-off', INDEPENDENT_ARM_SLUG)]
    present = set(df['arm_slug'].unique())
    for slug in _arms_present(df, None):
        if slug == 'joint-off':
            continue
        contrasts.append(('alignment', slug, 'joint-off'))

    rows = []
    for component, treatment, baseline in contrasts:
        if treatment not in present or baseline not in present:
            continue
        deltas = arm_deltas(df, treatment, baseline, metrics=metrics)
        for metric in metrics:
            column = 'd_{}'.format(metric)
            values = deltas[column]
            split_means = deltas.groupby('split')[column].mean().to_numpy() \
                if len(deltas) else np.array([])
            mean, sd, sem, low, high = _t_interval(split_means, confidence)
            rows.append({
                'component': component,
                'contrast': '{} - {}'.format(treatment, baseline),
                'treatment': treatment,
                'baseline': baseline,
                'metric': metric,
                'n_pairs': int(len(deltas)),
                'n_splits': int(len(split_means)),
                'mean_diff_split_level': mean,
                'sd_split_level': sd,
                'sem_split_level': sem,
                'ci_low': low,
                'ci_high': high,
                'confidence': confidence,
                'mean_diff_pairwise': float(values.mean()) if len(values) else float('nan'),
                'median_diff_pairwise': float(values.median()) if len(values) else float('nan'),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiplicity correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down adjusted p-values.

    Sort ascending, multiply the i-th smallest (0-based) by `n - i`, enforce
    monotonicity with a running maximum, clip at 1, and restore the input
    order. Comparing the result against alpha is equivalent to the classical
    step-down procedure and matches
    `statsmodels.stats.multitest.multipletests(method='holm')[1]`, which the
    test suite checks wherever statsmodels is importable (it is not a
    dependency of this environment, so the primary check is a hand-computed
    table).

    Raises on NaN rather than dropping it: a dropped p-value would shrink the
    family and weaken the correction for every other comparison, which is
    exactly the failure mode the correction exists to prevent.
    """
    p = np.asarray(pvalues, dtype='float64')
    if p.ndim != 1:
        raise ValueError('holm_bonferroni expects a 1-D sequence of p-values')
    if p.size == 0:
        return p
    if not np.isfinite(p).all():
        raise ValueError(
            'holm_bonferroni: p-values contain NaN or infinity {} -- dropping '
            'one would silently shrink the correction family'.format(p.tolist()))
    if ((p < 0) | (p > 1)).any():
        raise ValueError('holm_bonferroni: p-values must lie in [0, 1]')

    n = p.size
    order = np.argsort(p, kind='stable')
    ascending = p[order]
    scaled = ascending * (n - np.arange(n))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(n, dtype='float64')
    adjusted[order] = adjusted_sorted
    return adjusted


# ---------------------------------------------------------------------------
# Paired hypothesis tests
# ---------------------------------------------------------------------------

def default_contrast_family(df=None, arms=None, baseline=INDEPENDENT_ARM_SLUG):
    """The pre-registered contrast family: each joint arm
    (`JOINT_ARM_SLUGS`) against `independent`.

    Two contrasts x five tests (`acc_app`, `f1_app`, `acc_ddos`, `f1_ddos`,
    `blocks`) is the 10-comparison family `PRE_REGISTERED_FAMILY_SIZE` names.
    When `df` is given, only arms actually present in it are returned, so a
    partial campaign yields a smaller -- and explicitly smaller -- family.
    """
    if arms is not None:
        selected = tuple(arms)
    elif df is None:
        selected = JOINT_ARM_SLUGS
    else:
        selected = _arms_present(df, None)
    return tuple((arm, baseline) for arm in selected)


def _wilcoxon(differences, alternative):
    """Wilcoxon signed-rank test with this module's zero handling, returning
    a defined result when every difference is exactly zero (which
    `zero_method='zsplit'` supports and 'wilcox'/'pratt' do not)."""
    differences = np.asarray(differences, dtype='float64')
    if differences.size == 0:
        return float('nan'), float('nan')
    if np.all(differences == 0):
        return 0.0, 1.0 if alternative == 'two-sided' else 0.5
    result = stats.wilcoxon(differences, zero_method=_ZERO_METHOD,
                            alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def paired_tests(df, baseline=INDEPENDENT_ARM_SLUG, arms=None,
                 metrics=DEFAULT_METRICS, margin=0.0, alpha=0.05,
                 unit='pair', confidence=0.95, expected_family_size=None):
    """Paired Wilcoxon tests over the whole contrast family, Holm-corrected.

    The family, stated so it can be checked (see the module docstring): the
    joint arms of `JOINT_ARM_SLUGS`, each against `independent`, times
    five tests -- `acc_app`, `f1_app`, `acc_ddos`, `f1_ddos`, `blocks` -- for
    10 comparisons. Holm-Bonferroni is applied across ALL of them at once;
    `n_comparisons` on every row records how many were actually corrected
    over, and `expected_family_size=PRE_REGISTERED_FAMILY_SIZE` turns a
    shrunken family (an arm missing from `df`) into an error rather than a
    quietly weaker correction.

    Which alternative each test encodes -- the expensive thing to get wrong:

    * `acc_app`, `f1_app`, `acc_ddos`, `f1_ddos`: ONE-SIDED,
      `alternative='greater'` applied to `d + margin` where
      `d = joint - independent`. The null is
      `median(d) <= -margin` and the alternative is `median(d) > -margin`, so
      a SMALL p-value is the positive finding: the joint arm is not worse by
      more than `margin`. With the default `margin = 0` this is
      `H1: median(d) > 0` -- NO DETECTABLE LOSS, which is what the emitted
      `hypothesis` column says at that default. It is not called
      non-inferiority there: non-inferiority is a claim against a
      pre-registered margin, and `margin = 0` sets none. Pass `margin > 0`
      and the column says non-inferiority and names the margin. Reversing the
      test to `alternative='less'` would ask whether the joint arm IS worse,
      and a large p-value from that establishes nothing.
    * `blocks`: TWO-SIDED. Alignment adds intervals before it merges any, so
      sharing can cost blocks as well as save them and no direction may be
      assumed. `margin` is not applied to `blocks`.

    `unit='pair'` (default) tests one difference per `(M, split, k)` cell, as
    spec C.3 pairs; those cells share a training split within a split, so the
    p-values are anti-conservative relative to the number of independent
    splits. `unit='split'` collapses each split to its mean difference first
    -- valid under split-level replication, far less powerful -- as a
    robustness check. Both `n_pairs` and `n_splits` are always reported.

    Raises ValueError if any contrast has no paired cells at all: a contrast
    contributing nothing would shrink the family without the reader noticing.
    """
    if unit not in ('pair', 'split'):
        raise ValueError("paired_tests: unit must be 'pair' or 'split', got {!r}".format(unit))

    family = default_contrast_family(df, arms=arms, baseline=baseline)
    if not family:
        raise ValueError(
            'paired_tests: no treatment arms found in the frame (expected some '
            'of {}), so there is no family to correct over.'.format(
                list(JOINT_ARM_SLUGS)))
    rows = []
    for treatment, contrast_baseline in family:
        deltas = arm_deltas(df, treatment, contrast_baseline, metrics=metrics)
        if len(deltas) == 0:
            raise ValueError(
                'paired_tests: contrast {!r} - {!r} has no paired (M, split, k) '
                'cells, which would silently shrink the correction family. '
                'Check that both arms were run over the same grid.'
                .format(treatment, contrast_baseline))
        n_pairs = int(len(deltas))
        n_splits = int(deltas['split'].nunique())
        for metric in metrics:
            column = 'd_{}'.format(metric)
            if unit == 'split':
                values = deltas.groupby('split')[column].mean().to_numpy(dtype='float64')
            else:
                values = deltas[column].to_numpy(dtype='float64')

            alternative = METRIC_ALTERNATIVE.get(metric, 'two-sided')
            if alternative == 'greater':
                tested = values + margin
                if margin > 0:
                    # Only here is "non-inferiority" earned: a margin was
                    # actually set, so the claim is against something.
                    hypothesis = (
                        'H0: median({0}) <= -{1:g}  vs  H1: median({0}) > -{1:g}  '
                        '(non-inferiority of {2} to {3} within a margin of '
                        '{1:g})').format(column, margin, treatment,
                                         contrast_baseline)
                else:
                    # margin == 0: the null is against zero, so render it as
                    # plain `0` -- '{:g}'.format(0.0) prefixed by a literal
                    # minus gives `-0`, which reads in a results table as
                    # though some margin exists. And the claim is "no
                    # detectable loss", not non-inferiority, because no
                    # margin was pre-registered.
                    hypothesis = (
                        'H0: median({0}) <= 0  vs  H1: median({0}) > 0  '
                        '(no detectable loss for {1} against {2}; no '
                        'non-inferiority margin was set)').format(
                            column, treatment, contrast_baseline)
                applied_margin = margin
            else:
                tested = values
                hypothesis = ('H0: median({0}) == 0  vs  H1: median({0}) != 0  '
                              '(two-sided: sharing may help or hurt)').format(column)
                applied_margin = 0.0

            statistic, p_value = _wilcoxon(tested, alternative)
            mean, sd, sem, low, high = _t_interval(
                deltas.groupby('split')[column].mean().to_numpy(dtype='float64'),
                confidence)
            rows.append({
                'contrast': '{} - {}'.format(treatment, contrast_baseline),
                'treatment': treatment,
                'baseline': contrast_baseline,
                'metric': metric,
                'alternative': alternative,
                'hypothesis': hypothesis,
                'margin': applied_margin,
                'unit': unit,
                'n_pairs': n_pairs,
                'n_splits': n_splits,
                'n_tested': int(len(values)),
                # Two counts, because they differ once margin > 0:
                # n_zero_differences describes the raw deltas, while
                # n_zero_in_test is what zsplit actually had to split.
                'n_zero_differences': int(np.sum(values == 0)),
                'n_zero_in_test': int(np.sum(tested == 0)),
                'median_diff': float(np.median(values)) if len(values) else float('nan'),
                'mean_diff_split_level': mean,
                'ci_low': low,
                'ci_high': high,
                'statistic': statistic,
                'p_value': p_value,
            })

    table = pd.DataFrame(rows)
    n_comparisons = len(table)
    if expected_family_size is not None and n_comparisons != expected_family_size:
        raise ValueError(
            'paired_tests: ran {} comparisons but the expected correction '
            'family size is {}. Running a subset of the pre-registered family '
            'weakens the Holm correction for every comparison in it, so this '
            'is refused rather than silently accepted. Arms found: {}.'
            .format(n_comparisons, expected_family_size,
                    [treatment for treatment, _ in family]))

    table['n_comparisons'] = n_comparisons
    table['p_holm'] = holm_bonferroni(table['p_value'].to_numpy())
    table['alpha'] = alpha
    table['significant_holm'] = table['p_holm'] < alpha
    return table


# ---------------------------------------------------------------------------
# Compiler-verified campaign: agreement, budget binding, robustness (spec 7.3)
# ---------------------------------------------------------------------------

def _as_bool(series):
    """A flag column as bools, whether it holds parsed bools or the literal
    verification.csv text ('True'/'False', '' for None -> False)."""
    if series.dtype == bool:
        return series
    return series.map(lambda v: v is True or (isinstance(v, (bool, np.bool_)) and bool(v))
                      or (isinstance(v, str) and v.strip().lower() == 'true'))


def _as_number(series):
    """A numeric column as float64; '' / None (p4c never got that far) -> NaN."""
    return pd.to_numeric(series.replace('', np.nan), errors='coerce').astype('float64')


def _as_list(value):
    """`tables_differing` as a list: JSON text as written by the verifier, a
    list already, or [] for empty / missing."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == '':
        return []
    return json.loads(value)


def _cell_order(frame):
    """(arm_slug, M) cells in reading order: the independent baseline, then
    JOINT_ARM_SLUGS, then any other arm; M ascending (inf last)."""
    known = [INDEPENDENT_ARM_SLUG] + list(JOINT_ARM_SLUGS)
    rank = {slug: i for i, slug in enumerate(known)}
    return frame.assign(
        _arm_rank=frame['arm_slug'].map(lambda a: rank.get(a, len(known)))
    ).sort_values(['_arm_rank', 'arm_slug', 'M']).drop(columns='_arm_rank') \
        .reset_index(drop=True)


def agreement_table(verification):
    """Model-vs-p4c agreement per arm x M (spec section 7.3), from the frame
    `campaign_data.load_verification` returns -- EVERY verified design,
    including the ones `load_campaign` drops as p4c-infeasible (O1): the
    agreement table is about the model, not about the reported designs.

    Returns `(summary, misses)`:

    * `summary`, one row per `(arm_slug, M)`: `n` rows, `stage_depth_exact`
      rows whose model stage depth equals p4c's, `blocks_exact` rows whose
      model blocks equal p4c's, and `blocks_na` VERIFIED rows where p4c
      never allocated (`p4c_blocks` empty -- over 12 stages), so the blocks
      column reads as `blocks_exact / (n - blocks_na - unverified)` at most.
      An unverified row (compile error / timeout) is exact on neither.
    * `misses`, every row whose `verdict` is not `EXACT`, with both numbers
      and `tables_differing` parsed to a list of `{table, model, p4c}`.

    Accepts the literal CSV text load_verification keeps ('True'/'False',
    '' for None) or parsed bools and numbers; `M` may be inf.
    """
    frame = verification.copy()
    for column in ('model_stage_depth', 'p4c_stage_depth', 'model_blocks',
                   'p4c_blocks'):
        frame[column] = _as_number(frame[column])
    unverified = _as_bool(frame['unverified']) if 'unverified' in frame.columns \
        else pd.Series(False, index=frame.index)
    frame['_stage_exact'] = frame['model_stage_depth'] == frame['p4c_stage_depth']
    frame['_blocks_exact'] = frame['model_blocks'] == frame['p4c_blocks']
    frame['_blocks_na'] = ~unverified & frame['p4c_blocks'].isna()

    rows = []
    for (arm, M), cell in frame.groupby(['arm_slug', 'M'], sort=False):
        rows.append({
            'arm_slug': arm, 'M': float(M), 'n': int(len(cell)),
            'stage_depth_exact': int(cell['_stage_exact'].sum()),
            'blocks_exact': int(cell['_blocks_exact'].sum()),
            'blocks_na': int(cell['_blocks_na'].sum()),
        })
    summary = pd.DataFrame(rows, columns=['arm_slug', 'M', 'n', 'stage_depth_exact',
                                          'blocks_exact', 'blocks_na'])
    if len(summary):
        summary = _cell_order(summary)

    miss_columns = ['row_id', 'verdict', 'model_stage_depth', 'p4c_stage_depth',
                    'model_blocks', 'p4c_blocks', 'tables_differing']
    misses = frame.loc[frame['verdict'] != 'EXACT', miss_columns].copy()
    misses['tables_differing'] = [_as_list(v) for v in misses['tables_differing']]
    return summary, misses.reset_index(drop=True)


def budget_binding(df, share=0.9):
    """Per `(arm_slug, M)` cell, the share of rows whose (p4c) `blocks` reach
    `share * M` (spec section 7.3): `M = 75` near 0 and identical to `inf`
    confirms the grid; a non-binding M other than 75/inf is reported as such.

    `df` is `load_campaign`'s frame, whose `blocks` already holds p4c's
    numbers. `n` counts the rows with a defined `blocks`; `binding_share` is
    NaN for `M = inf` (there is no budget to bind, and `0.9 * inf` is not a
    threshold) and for a cell with no defined `blocks`.
    """
    frame = df[['arm_slug', 'M']].copy()
    frame['M'] = frame['M'].astype('float64')
    frame['blocks'] = _as_number(df['blocks'])
    frame = frame[frame['blocks'].notna()]

    rows = []
    for (arm, M), cell in frame.groupby(['arm_slug', 'M'], sort=False):
        n = int(len(cell))
        if np.isinf(M) or n == 0:
            binding = float('nan')
        else:
            binding = float((cell['blocks'] >= share * M).mean())
        rows.append({'arm_slug': arm, 'M': float(M), 'n': n,
                     'binding_share': binding})
    table = pd.DataFrame(rows, columns=['arm_slug', 'M', 'n', 'binding_share'])
    return _cell_order(table) if len(table) else table


def paired_tests_robustness(df, **paired_kwargs):
    """`paired_tests` on three subsets, stacked with a `subset` column (spec
    section 7.3's robustness lines):

    * `'all'` -- every row; the only subset held to `expected_family_size`.
    * `'no_flagged'` -- without `flagged` rows (unverified designs carrying
      the model's numbers); run only when any row is flagged.
    * `'heldout_splits'` -- only splits not used during development
      (`split not in DEVELOPMENT_SPLITS`, i.e. excluding 10-13).

    The two robustness subsets are checks on the headline result, not new
    families, so `expected_family_size` is dropped for them; each is still
    Holm-corrected over the comparisons it ran. A robustness subset with no
    rows (only development splits present, or every row flagged) has
    nothing to test and is SKIPPED -- absent from `subset` -- rather than
    allowed to crash the headline `'all'` family.
    """
    subsets = [('all', df)]
    if 'flagged' in df.columns:
        flagged = _as_bool(df['flagged'])
        if flagged.any():
            subsets.append(('no_flagged', df[~flagged]))
    subsets.append(('heldout_splits', df[~df['split'].isin(DEVELOPMENT_SPLITS)]))

    robustness_kwargs = {key: value for key, value in paired_kwargs.items()
                         if key != 'expected_family_size'}
    tables = []
    for name, subset in subsets:
        if name != 'all' and len(subset) == 0:
            continue
        kwargs = paired_kwargs if name == 'all' else robustness_kwargs
        table = paired_tests(subset, **kwargs)
        table.insert(0, 'subset', name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------------
# Non-inferiority at a per-row relative-error margin (D13)
# ---------------------------------------------------------------------------

def noninferiority_tests(df, baseline=INDEPENDENT_ARM_SLUG, alpha=0.05,
                         unit='pair', expected_family_size=None):
    """D13: is each joint arm non-inferior to `independent` on accuracy at a
    margin of 5% of EACH ROW's OWN baseline error?

    Why this cannot reuse `paired_tests(..., margin=...)`. That machinery
    (`tested = values + margin`) adds a single CONSTANT to every row's
    accuracy-point delta. App sits near 0.70 accuracy (about 30% error) and
    DDoS near 0.97 (about 3% error), so one flat absolute margin would test
    the two tasks at wildly different strictness -- generous for App,
    punishing for DDoS. D13 instead sizes the margin PER ROW, as
    `NONINFERIORITY_MARGIN * (1 - baseline_accuracy)` -- the same FRACTION
    of each row's own error -- computed from `campaign_data.pair_arms`
    directly rather than `arm_deltas`: `arm_deltas` only returns the delta
    `d_<metric>` and drops the raw baseline value this margin needs.

    H0 / H1, stated so the sign is checkable against the code below:

        d = joint_accuracy - independent_accuracy          (per row)
        margin_row = NONINFERIORITY_MARGIN * (1 - independent_accuracy)

        H0: median(d) <= -margin_row   (joint increases this row's relative
                                         error by MORE than the margin)
        H1: median(d) >  -margin_row   (it does not)

    REJECTING H0 -- a small p-value -- is the POSITIVE finding: it is what
    licenses the word "equivalent" in the results chapter. This mirrors the
    statistical SHAPE of `paired_tests`' `margin > 0` branch exactly --
    Wilcoxon one-sided (`alternative='greater'`) on `d + margin` via
    `_wilcoxon` -- only the margin's construction differs, from a constant
    to a per-row fraction of baseline error.

    Scope is deliberately just `NONINFERIORITY_METRICS` (`acc_app`,
    `acc_ddos`). F1 is NOT tested here -- it stays in `paired_tests`'
    superiority family, which answers a different question.

    The family is the two joint arms x these two metrics = 4
    (`NONINFERIORITY_FAMILY_SIZE`), Holm-corrected on its OWN: this
    function's p-values are never mixed into `paired_tests`' family -- see the
    module docstring for why pooling the two would weaken both. Pass
    `expected_family_size=NONINFERIORITY_FAMILY_SIZE` to turn a shrunken
    family (an arm missing from `df`) into an error rather than a quietly
    weaker correction, exactly as `paired_tests` does.

    The emitted `margin` column reports the MEAN of the per-row margins
    over the cells that fed the test (a summary figure only -- the test
    itself uses the full per-row margin, not this mean). In this project's
    synthetic tests the baseline accuracy is constant across rows within a
    contrast, so the mean equals the single value it is averaging.

    `unit='pair'` (default) tests one row per `(M, split, k)` cell, as spec
    C.3 pairs; `unit='split'` collapses to one mean per split first, as the
    lower-power robustness check `paired_tests` also offers. Both `n_pairs`
    and `n_splits` are always reported. The returned frame carries the same
    column set as `paired_tests` so downstream table/deliverable code can
    concatenate the two families and treat their rows uniformly.

    PRE-REGISTRATION NOTICE. `NONINFERIORITY_MARGIN = 0.05` is a
    pre-registration decision, written down here BEFORE the first real
    campaign cell has been run. It is legitimate only because of that
    ordering: choosing, widening, or narrowing a non-inferiority margin
    AFTER seeing results turns the test into a foregone conclusion,
    defeating the entire point of pre-registering it. Once real campaign
    data exists, this margin must NOT be revisited to make a result come
    out either way.

    Raises ValueError if any contrast has no paired cells at all: a
    contrast contributing nothing would shrink the family without the
    reader noticing.
    """
    if unit not in ('pair', 'split'):
        raise ValueError(
            "noninferiority_tests: unit must be 'pair' or 'split', got {!r}"
            .format(unit))

    family = default_contrast_family(df, baseline=baseline)
    if not family:
        raise ValueError(
            'noninferiority_tests: no treatment arms found in the frame '
            '(expected some of {}), so there is no family to correct over.'
            .format(list(JOINT_ARM_SLUGS)))

    rows = []
    for treatment, contrast_baseline in family:
        for metric in NONINFERIORITY_METRICS:
            treatment_col = '{}_treatment'.format(metric)
            baseline_col = '{}_baseline'.format(metric)
            paired = pair_arms(df, treatment, contrast_baseline)
            if len(paired) == 0:
                raise ValueError(
                    'noninferiority_tests: contrast {!r} - {!r} has no '
                    'paired (M, split, k) cells, which would silently '
                    'shrink the correction family. Check that both arms '
                    'were run over the same grid.'
                    .format(treatment, contrast_baseline))

            baseline_values = paired[baseline_col].astype('float64').to_numpy()
            treatment_values = paired[treatment_col].astype('float64').to_numpy()
            d = treatment_values - baseline_values
            row_margin = NONINFERIORITY_MARGIN * (1.0 - baseline_values)
            tested_full = d + row_margin

            n_pairs = int(len(paired))
            n_splits = int(paired['split'].nunique())
            split_ids = paired['split'].to_numpy()

            if unit == 'split':
                grouped = pd.DataFrame({
                    'split': split_ids, 'd': d, 'tested': tested_full,
                }).groupby('split').mean()
                values = grouped['d'].to_numpy(dtype='float64')
                tested = grouped['tested'].to_numpy(dtype='float64')
            else:
                values = d
                tested = tested_full

            statistic, p_value = _wilcoxon(tested, 'greater')
            split_means = (
                pd.DataFrame({'split': split_ids, 'd': d})
                .groupby('split')['d'].mean().to_numpy(dtype='float64'))
            mean, sd, sem, low, high = _t_interval(split_means, 0.95)

            mean_margin = float(np.mean(row_margin))
            hypothesis = (
                'H0: median(d_{0}) <= -{1:g} * (1 - baseline_{0})  vs  '
                'H1: median(d_{0}) > -{1:g} * (1 - baseline_{0})  '
                '(non-inferiority of {2} to {3} within a margin of {1:g} '
                "of {3}'s own baseline error, computed per (M, split, k) "
                'row; rejecting H0 licenses "equivalent")'
            ).format(metric, NONINFERIORITY_MARGIN, treatment, contrast_baseline)

            rows.append({
                'contrast': '{} - {}'.format(treatment, contrast_baseline),
                'treatment': treatment,
                'baseline': contrast_baseline,
                'metric': metric,
                'alternative': 'greater',
                'hypothesis': hypothesis,
                'margin': mean_margin,
                'unit': unit,
                'n_pairs': n_pairs,
                'n_splits': n_splits,
                'n_tested': int(len(values)),
                'n_zero_differences': int(np.sum(values == 0)),
                'n_zero_in_test': int(np.sum(tested == 0)),
                'median_diff': float(np.median(values)) if len(values) else float('nan'),
                'mean_diff_split_level': mean,
                'ci_low': low,
                'ci_high': high,
                'statistic': statistic,
                'p_value': p_value,
            })

    table = pd.DataFrame(rows)
    n_comparisons = len(table)
    if expected_family_size is not None and n_comparisons != expected_family_size:
        raise ValueError(
            'noninferiority_tests: ran {} comparisons but the expected '
            'correction family size is {}. Running a subset of the '
            'pre-registered family weakens the Holm correction for every '
            'comparison in it, so this is refused rather than silently '
            'accepted. Arms found: {}.'
            .format(n_comparisons, expected_family_size,
                    [treatment for treatment, _ in family]))

    table['n_comparisons'] = n_comparisons
    table['p_holm'] = holm_bonferroni(table['p_value'].to_numpy())
    table['alpha'] = alpha
    table['significant_holm'] = table['p_holm'] < alpha
    return table


# ---------------------------------------------------------------------------
# Alignment twins (spec 2026-10-06 section 6)
# ---------------------------------------------------------------------------

K_GROUPS = (('low 1-5', 1, 5), ('mid 6-11', 6, 11), ('high 12-17', 12, 17))
TWIN_METRICS = ('blocks', 'stage_depth', 'f1_app', 'f1_ddos')
# For blocks/stage_depth a negative delta is a saving; for F1 a positive one is.
_SAVING_SIGN = {'blocks': -1, 'stage_depth': -1, 'f1_app': 1, 'f1_ddos': 1}
_LADDER_ARMS = (INDEPENDENT_ARM_SLUG, 'joint-off', TWIN_ARM_SLUG)


def k_group(k):
    for label, lo, hi in K_GROUPS:
        if lo <= k <= hi:
            return label
    return 'other'


def twin_pairs(df):
    """Every (M, split, k) cell with BOTH a twin and its joint-off source, as
    `pair_arms` builds it, plus `d_<metric>` = twin - source and the k group.
    A twin whose source was dropped (rescued) has no partner and is absent by
    construction."""
    pairs = pair_arms(df, TWIN_ARM_SLUG, 'joint-off')
    for metric in TWIN_METRICS:
        pairs['d_' + metric] = pairs[metric + '_treatment'] - pairs[metric + '_baseline']
    pairs['k_group'] = pairs['k'].map(k_group)
    return pairs


def _effect_rows(pairs, group_kind, group, confidence):
    rows = []
    for metric in TWIN_METRICS:
        d = pairs['d_' + metric]
        split_means = (pairs.groupby('split')['d_' + metric].mean().to_numpy()
                       if len(pairs) else np.array([]))
        mean, _, _, low, high = _t_interval(split_means, confidence)
        sign = _SAVING_SIGN[metric]
        rows.append({'group_kind': group_kind, 'group': group, 'metric': metric,
                     'n_pairs': int(len(pairs)), 'n_splits': int(len(split_means)),
                     'mean_diff_split_level': mean, 'ci_low': low, 'ci_high': high,
                     'mean_diff_pairwise': float(d.mean()) if len(d) else float('nan'),
                     'p_saves': float((sign * d > 0).mean()) if len(d) else float('nan'),
                     'p_costs': float((sign * d < 0).mean()) if len(d) else float('nan')})
    return rows


def twin_effect(df, confidence=0.95):
    """Twin minus source, overall, per M and per k group (split-level t CI)."""
    pairs = twin_pairs(df)
    rows = _effect_rows(pairs, 'all', 'all', confidence)
    for M in sorted(pairs['M'].unique()):
        rows += _effect_rows(pairs[pairs['M'] == M], 'M', M, confidence)
    for label, _, _ in K_GROUPS:
        rows += _effect_rows(pairs[pairs['k_group'] == label], 'k_group', label, confidence)
    return pd.DataFrame(rows)


def twin_ladder(df, confidence=0.95):
    """independent -> joint-off -> joint-off-al on the cells where all three
    exist: per-arm means and the two step deltas with split-level t CIs."""
    key = ['M', 'split', 'k']
    wide = None
    for arm in _LADDER_ARMS:
        part = df[df['arm_slug'] == arm].set_index(key)[list(TWIN_METRICS)]
        part.columns = [f'{m}__{arm}' for m in TWIN_METRICS]
        wide = part if wide is None else wide.join(part, how='inner')
    wide = wide.reset_index()
    rows = []
    for M in list(sorted(wide['M'].unique())) + ['all']:
        cells = wide if M == 'all' else wide[wide['M'] == M]
        row = {'M': M, 'n_cells': int(len(cells))}
        for metric in TWIN_METRICS:
            for arm in _LADDER_ARMS:
                row[f'mean_{arm}_{metric}'] = float(cells[f'{metric}__{arm}'].mean())
            for step, (a, b) in (('sharing', ('joint-off', INDEPENDENT_ARM_SLUG)),
                                 ('alignment', (TWIN_ARM_SLUG, 'joint-off'))):
                d = cells[f'{metric}__{a}'] - cells[f'{metric}__{b}']
                _, _, _, low, high = _t_interval(
                    d.groupby(cells['split']).mean().to_numpy(), confidence)
                row[f'step_{step}_{metric}'] = float(d.mean())
                row[f'step_{step}_{metric}_ci_low'] = low
                row[f'step_{step}_{metric}_ci_high'] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _p4c_feasible(verification):
    return ~((verification['verdict'] == 'FALSE_FEASIBLE')
             | _as_bool(verification['p4c_over_stages'])
             | _as_bool(verification['p4c_over_budget']))


def twin_counts(verification):
    """Counts deliverable 11 reports from `load_verification`'s frame.
    Rescued = twin p4c-feasible while its joint-off source is not; a source
    absent from the frame counts as feasible."""
    feasible = dict(zip(verification['row_id'], _p4c_feasible(verification)))
    twins = verification[verification['arm_slug'] == TWIN_ARM_SLUG]
    copied = twins['copied_from'].fillna('').astype(str) != ''
    compiled = twins[~copied]
    source_ids = twins['row_id'].str.replace(f'^{TWIN_ARM_SLUG}_', 'joint-off_', regex=True)
    twin_ok = twins['row_id'].map(feasible).astype(bool)
    source_ok = source_ids.map(lambda rid: feasible.get(rid, True)).astype(bool)
    return {'n_twins': int(len(twins)), 'n_identical': int(copied.sum()),
            'n_compiled': int(len(compiled)),
            'n_rescued': int((twin_ok & ~source_ok).sum()),
            'n_twin_infeasible': int((~twin_ok).sum()),
            'n_compiled_exact': int((compiled['verdict'] == 'EXACT').sum()),
            'exact_rate_compiled': (float((compiled['verdict'] == 'EXACT').mean())
                                    if len(compiled) else float('nan'))}
