"""Spec A.2's arm grid and C.2's file naming."""
import re

from src import main as m
from src.training.config import TrainConfig


def test_primary_grid_is_three_arms():
    """independent, joint with alignment OFF, and joint. `off` is a genuine
    skip of the align_rf_thresholds call, provably prediction-identical to the
    unaligned models, and it doubles as the ablation the reviewer asked for.

    Until 2026-09-14 the third arm was swept across three overlap thresholds
    (design §2.4), giving five primary arms; Task 7 (design D4) removed that
    tunable. Until 2026-09-15 it was additionally the delta = 0 member of a
    six-value tolerance sweep, so its slug read 'joint-d000'; Track 5 returned
    delta_helps = FALSE and the sweep is gone, leaving one aligned joint arm
    whose slug is plain 'joint'.
    """
    slugs = [cfg.arm_slug('disjoint' if arm == 'independent' else 'joint')
             for arm, cfg in m.PRIMARY_ARMS]

    assert slugs == ['independent', 'joint-off', 'joint']


def test_delta_select_is_identical_across_every_arm():
    """It is a constant of the setup, not a treatment: it moves the baseline as
    well as the treatment, so any variation across arms would shift the
    comparison under its own control variable."""
    assert {cfg.delta_select for _, cfg in m.PRIMARY_ARMS} == {0.02}


def test_result_paths_are_self_describing_and_unique_per_arm():
    paths = {m.arm_result_path(arm, cfg, 25) for arm, cfg in m.PRIMARY_ARMS}

    assert len(paths) == 3
    assert any(p.endswith('rf_t7_d14_M25_independent.csv') for p in paths)
    assert any(p.endswith('rf_t7_d14_M25_joint-off.csv') for p in paths)
    assert any(p.endswith('rf_t7_d14_M25_joint.csv') for p in paths)


def test_result_paths_record_the_effective_search_bounds():
    """Replaces feature_selection_comparison_results_by_k_-1_-1_25.csv, whose
    sentinel recorded neither the effective n_trees nor max_depth (F10i)."""
    path = m.arm_result_path('joint', TrainConfig(n_trees=5, max_depth=8), 40)

    assert path.endswith('rf_t5_d8_M40_joint.csv')
    assert '-1' not in path


def test_arms_flag_defaults_to_primary():
    assert m.parse_args([]).arms == 'primary'


def test_arms_flag_rejects_the_retired_sensitivity_grids():
    """'sensitivity' and 'all' existed only to name the delta_align sweep. A
    launch script still passing them must fail at parse time rather than
    silently run the primary grid instead."""
    import pytest

    for retired in ('sensitivity', 'all'):
        with pytest.raises(SystemExit):
            m.parse_args(['--arms', retired])


def test_select_arms_returns_the_requested_grid():
    import pytest

    assert m.select_arms('primary') == m.PRIMARY_ARMS
    with pytest.raises(ValueError):
        m.select_arms('all')


def test_the_grid_is_three_arms_with_three_distinct_slugs():
    """independent (1) + joint-off (1) + joint (1).

    Until 2026-09-14 the aligned arms were also swept across 3 overlap
    thresholds, and until 2026-09-15 across 6 delta_align tolerances -- 20 arms
    at the peak (independent + joint-off + 6 deltas x 3 overlaps). Both axes
    are gone, so the grid is smaller than the archive's 8, not larger."""
    from src.main import select_arms

    arms = select_arms('primary')
    assert len(arms) == 3

    slugs = {cfg.arm_slug('joint' if arm == 'joint' else 'disjoint')
             for arm, cfg in arms}
    assert len(slugs) == 3


def test_every_campaign_arm_enables_ccp_alpha():
    """It carries the larger of the two measured feasibility effects (+15.4pp
    vs alignment's +5.3pp), never costs feasibility, and no archived row has
    it. The 0.05 cap does not bind (0.2% of winners within 2x), so raising it
    would only add dead space at the degenerate-pruning end."""
    from src.main import select_arms, CAMPAIGN_CCP_ALPHA_MAX

    assert CAMPAIGN_CCP_ALPHA_MAX == 0.05
    for _, cfg in select_arms('primary'):
        assert cfg.ccp_alpha_max == CAMPAIGN_CCP_ALPHA_MAX


def test_align_objective_stays_at_blocks_on_every_arm():
    """Trivially true since design 2026-09-07 §4.3 retired 'stages' and
    'both' -- 'blocks' is now the only value ALIGN_OBJECTIVES has. Kept as a
    guard against the campaign grid ever passing an explicit align_objective
    that would then need its own arm-slug handling (see
    test_align_objective_does_not_enter_the_arm_slug)."""
    from src.main import select_arms

    assert {cfg.align_objective for _, cfg in select_arms('primary')} == {'blocks'}


def test_delta_select_stays_out_of_the_sweep():
    """It moves the baseline and the treatment together, so it cannot be a
    treatment in a joint-vs-independent comparison."""
    from src.main import select_arms

    assert {cfg.delta_select for _, cfg in select_arms('primary')} == {0.02}


def test_the_arm_grid_lost_the_overlap_and_delta_axes():
    """D4: 21 aligned arms collapsed to 6, one per delta value, when the
    overlap axis went. Track 5's delta_helps = FALSE verdict then collapsed
    those 6 to 1: there is one alignment behaviour, free moves only, and both
    product terms are gone.

    The brief's own draft of this test asserted `not any('-o' in slug for
    slug in slugs)`, which false-positives on 'joint-off' ('-o' is a
    substring of '-off') regardless of whether the retired overlap-suffix
    axis is really gone. The old suffix was `-o{:03d}` -- three digits after
    `-o`, e.g. '-o025' -- so check for that shape specifically.
    """
    from src import main

    assert not hasattr(main, 'OVERLAP_THRESHOLDS')
    assert not hasattr(main, 'DELTA_ALIGNS')
    assert not hasattr(main, 'SENSITIVITY_ARMS')
    assert len(main.PRIMARY_ARMS) == 3

    slugs = [cfg.arm_slug('joint' if arm == 'joint' else 'disjoint')
             for arm, cfg in main.select_arms('primary')]
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
    """Track 5's arm set straddled both presets and its cell ORDER was
    pre-registered (spec 2.4), so selection must preserve the caller's order
    rather than the catalogue's. Its own three slugs are gone with the delta
    axis; the ordering contract they motivated is not."""
    chosen = m.select_arm_slugs(['joint', 'joint-off'])

    assert [cfg.arm_slug('joint') for _arm, cfg in chosen] == [
        'joint', 'joint-off',
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
        m.select_arm_slugs(['joint-d020'])

    message = str(excinfo.value)
    assert 'joint-d020' in message
    assert 'joint-off' in message


def test_the_retired_delta_slugs_no_longer_resolve():
    """The three slugs Track 5 launched its cells with. They must fail loudly,
    not quietly resolve to the one surviving aligned arm -- a stale launch
    script would otherwise run something the pre-registration never named."""
    import pytest

    for retired in ('joint-d000', 'joint-d020', 'joint-dinf'):
        with pytest.raises(ValueError):
            m.select_arm_slugs([retired])
