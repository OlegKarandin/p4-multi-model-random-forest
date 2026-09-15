"""Spec A.2's arm grid and C.2's file naming."""
import re

from src import main as m
from src.training.config import TrainConfig


def test_primary_grid_is_three_arms_two_anchors_plus_delta_zero():
    """independent, joint with alignment OFF, and joint at delta = 0. The two
    constant anchors bracket the frontier: `off` is a genuine skip of the
    align_rf_thresholds call, provably prediction-identical to the unaligned
    models, and it doubles as the ablation the reviewer asked for.

    Until 2026-09-14 the third arm was swept across three overlap thresholds
    (design §2.4), giving five primary arms. Task 7 (design D4) removed the
    overlap_threshold tunable those three arms varied, so there is now only
    one delta-zero joint arm.
    """
    slugs = [cfg.arm_slug('disjoint' if arm == 'independent' else 'joint')
             for arm, cfg in m.PRIMARY_ARMS]

    assert slugs == ['independent', 'joint-off', 'joint-d000']


def test_sensitivity_grid_is_the_five_swept_tolerances():
    """5 delta_align values (0.01 is deliberately absent: it permits at most
    one DDoS sample to flip -- one flip = 0.83% relative error at val_align
    ~3000, error ~0.04, so it is operationally identical to delta = 0).

    Until 2026-09-14 each was also swept across three overlap thresholds
    (design §2.4), giving fifteen sensitivity arms; that axis is gone (Task
    7, design D4)."""
    slugs = [cfg.arm_slug('joint') for arm, cfg in m.SENSITIVITY_ARMS]

    assert slugs == [
        'joint-d002', 'joint-d005', 'joint-d010', 'joint-d020', 'joint-dinf',
    ]


def test_every_sensitivity_arm_is_a_joint_arm():
    assert all(arm == 'joint' for arm, _ in m.SENSITIVITY_ARMS)


def test_delta_select_is_identical_across_every_arm():
    """It is a constant of the setup, not a treatment: it moves the baseline as
    well as the treatment, so any variation across arms would shift the
    comparison under its own control variable."""
    every = m.PRIMARY_ARMS + m.SENSITIVITY_ARMS

    assert {cfg.delta_select for _, cfg in every} == {0.02}


def test_result_paths_are_self_describing_and_unique_per_arm():
    paths = {m.arm_result_path(arm, cfg, 25)
             for arm, cfg in m.PRIMARY_ARMS + m.SENSITIVITY_ARMS}

    assert len(paths) == 8
    assert any(p.endswith('rf_t7_d14_M25_independent.csv') for p in paths)
    assert any(p.endswith('rf_t7_d14_M25_joint-d002.csv') for p in paths)
    assert any(p.endswith('rf_t7_d14_M25_joint-dinf.csv') for p in paths)


def test_result_paths_record_the_effective_search_bounds():
    """Replaces feature_selection_comparison_results_by_k_-1_-1_25.csv, whose
    sentinel recorded neither the effective n_trees nor max_depth (F10i)."""
    path = m.arm_result_path('joint', TrainConfig(n_trees=5, max_depth=8), 40)

    assert path.endswith('rf_t5_d8_M40_joint-d000.csv')
    assert '-1' not in path


def test_arms_flag_defaults_to_primary():
    assert m.parse_args([]).arms == 'primary'


def test_arms_flag_selects_the_grid():
    assert m.parse_args(['--arms', 'sensitivity']).arms == 'sensitivity'
    assert m.parse_args(['--arms', 'all']).arms == 'all'


def test_select_arms_returns_the_requested_grid():
    assert m.select_arms('primary') == m.PRIMARY_ARMS
    assert m.select_arms('sensitivity') == m.SENSITIVITY_ARMS
    assert m.select_arms('all') == m.PRIMARY_ARMS + m.SENSITIVITY_ARMS


def test_the_grid_is_eight_arms_with_eight_distinct_slugs():
    """independent (1) + joint-off (1) + 6 delta_align (6).

    Until 2026-09-14 each of the 6 delta_align values (join-off excepted) was
    also swept across 3 overlap thresholds, giving 20 arms (independent (1) +
    joint-off (1) + 6 delta_align x 3 overlap (18)). Task 7 (design D4)
    removed that axis; the real multiplier over the archive's 8 arms is now
    1x, not 2.5x."""
    from src.main import select_arms

    arms = select_arms('all')
    assert len(arms) == 8

    slugs = {cfg.arm_slug('joint' if arm == 'joint' else 'disjoint')
             for arm, cfg in arms}
    assert len(slugs) == 8


def test_every_campaign_arm_enables_ccp_alpha():
    """It carries the larger of the two measured feasibility effects (+15.4pp
    vs alignment's +5.3pp), never costs feasibility, and no archived row has
    it. The 0.05 cap does not bind (0.2% of winners within 2x), so raising it
    would only add dead space at the degenerate-pruning end."""
    from src.main import select_arms, CAMPAIGN_CCP_ALPHA_MAX

    assert CAMPAIGN_CCP_ALPHA_MAX == 0.05
    for _, cfg in select_arms('all'):
        assert cfg.ccp_alpha_max == CAMPAIGN_CCP_ALPHA_MAX


def test_align_objective_stays_at_blocks_on_every_arm():
    """Trivially true since design 2026-09-07 §4.3 retired 'stages' and
    'both' -- 'blocks' is now the only value ALIGN_OBJECTIVES has. Kept as a
    guard against the campaign grid ever passing an explicit align_objective
    that would then need its own arm-slug handling (see
    test_align_objective_does_not_enter_the_arm_slug)."""
    from src.main import select_arms

    assert {cfg.align_objective for _, cfg in select_arms('all')} == {'blocks'}


def test_delta_select_stays_out_of_the_sweep():
    """It moves the baseline and the treatment together, so it cannot be a
    treatment in a joint-vs-independent comparison."""
    from src.main import select_arms

    assert {cfg.delta_select for _, cfg in select_arms('all')} == {0.02}


def test_the_arm_grid_lost_the_overlap_axis():
    """D4: 21 aligned arms collapse to 6, one per delta value. The overlap axis
    multiplied every delta by three thresholds; with the tunable gone there is
    one alignment behaviour and the product term disappears.

    (And to 1 if Track 5 retires delta too -- a separate plan, gated on a
    pre-registered live-Optuna verdict.)

    The brief's own draft of this test asserted `not any('-o' in slug for
    slug in slugs)`, which false-positives on 'joint-off' ('-o' is a
    substring of '-off') regardless of whether the retired overlap-suffix
    axis is really gone. The old suffix was `-o{:03d}` -- three digits after
    `-o`, e.g. '-o025' -- so check for that shape specifically.
    """
    from src import main

    assert not hasattr(main, 'OVERLAP_THRESHOLDS')
    assert len(main.PRIMARY_ARMS) == 3
    assert len(main.SENSITIVITY_ARMS) == len(main.DELTA_ALIGNS) - 1 == 5

    slugs = [cfg.arm_slug('joint' if arm == 'joint' else 'disjoint')
             for arm, cfg in main.select_arms('all')]
    assert len(slugs) == len(set(slugs)), slugs
    assert not any(re.search(r'-o\d{3}\b', slug) for slug in slugs), slugs
    # 'joint-off' is the genuine skip-alignment anchor, not a retired
    # overlap-threshold suffix -- the precise check above must not flag it.
    assert 'joint-off' in slugs


def test_the_default_M_grid_is_the_archive_grid():
    """Comparability: C3 and C4 are matched-(M,k) comparisons against the
    archive, and main.py's old default shared only {25,50,100} with it."""
    from src.main import parse_args
    import src.main as main_mod

    args = parse_args([])
    M = args.M if args.M is not None else main_mod.DEFAULT_M_GRID
    assert M == [25, 50, 100, 150, 250]


def test_select_arm_slugs_returns_the_named_arms_in_the_order_asked():
    """Track 5's arm set straddles both presets, and its cell ORDER is
    pre-registered (spec 2.4), so selection must preserve the caller's order
    rather than the catalogue's."""
    chosen = m.select_arm_slugs(['joint-dinf', 'joint-d000', 'joint-d020'])

    assert [cfg.arm_slug('joint') for _arm, cfg in chosen] == [
        'joint-dinf', 'joint-d000', 'joint-d020',
    ]
    assert all(arm == 'joint' for arm, _cfg in chosen)


def test_select_arm_slugs_can_name_the_independent_arm():
    """The independent arm's slug ignores the alignment fields, so it must be
    reachable by its own name and must carry encoding 'disjoint'."""
    chosen = m.select_arm_slugs(['independent'])

    assert len(chosen) == 1
    arm, cfg = chosen[0]
    assert arm == 'independent'
    assert cfg.arm_slug('disjoint') == 'independent'


def test_select_arm_slugs_rejects_an_unknown_slug_and_names_the_known_ones():
    """A typo in a six-hour campaign launch must fail at argument parse time,
    not silently run a different arm than the pre-registered one."""
    import pytest

    with pytest.raises(ValueError) as excinfo:
        m.select_arm_slugs(['joint-d20'])

    message = str(excinfo.value)
    assert 'joint-d20' in message
    assert 'joint-d020' in message


def test_every_track5_arm_slug_resolves():
    """The three slugs Task 2 and Task 10 pass on the command line."""
    slugs = ['joint-d000', 'joint-d020', 'joint-dinf']

    assert [cfg.arm_slug('joint')
            for _arm, cfg in m.select_arm_slugs(slugs)] == slugs
