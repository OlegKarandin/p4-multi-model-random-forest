"""Spec A.2's arm grid and C.2's file naming."""
from src import main as m
from src.training.config import TrainConfig


def test_primary_grid_is_five_arms_two_anchors_plus_the_delta_zero_overlap_sweep():
    """independent, joint with alignment OFF, and joint at delta = 0 swept
    across the three overlap thresholds (§2.4). The two constant anchors
    bracket the frontier: `off` is a genuine skip of the align_rf_thresholds
    call, provably prediction-identical to the unaligned models, and it
    doubles as the ablation the reviewer asked for."""
    slugs = [cfg.arm_slug('disjoint' if arm == 'independent' else 'joint')
             for arm, cfg in m.PRIMARY_ARMS]

    assert slugs == ['independent', 'joint-off', 'joint-d000',
                      'joint-d000-o025', 'joint-d000-o010']


def test_sensitivity_grid_is_the_fifteen_swept_tolerances_by_overlap():
    """5 delta_align values (0.01 is deliberately absent: it permits at most
    one DDoS sample to flip -- one flip = 0.83% relative error at val_align
    ~3000, error ~0.04, so it is operationally identical to delta = 0) x 3
    overlap thresholds (§2.4)."""
    slugs = [cfg.arm_slug('joint') for arm, cfg in m.SENSITIVITY_ARMS]

    assert slugs == [
        'joint-d002', 'joint-d002-o025', 'joint-d002-o010',
        'joint-d005', 'joint-d005-o025', 'joint-d005-o010',
        'joint-d010', 'joint-d010-o025', 'joint-d010-o010',
        'joint-d020', 'joint-d020-o025', 'joint-d020-o010',
        'joint-dinf', 'joint-dinf-o025', 'joint-dinf-o010',
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

    assert len(paths) == 20
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


def test_the_grid_is_twenty_arms_with_twenty_distinct_slugs():
    """independent (1) + joint-off (1) + 6 delta_align x 3 overlap (18).

    NOT 22: design §2.7's cost arithmetic says '7 joint arms x 3', which
    double-counts joint-off -- §2.4 excludes it from the overlap axis because
    it never calls align_rf_thresholds. The real multiplier over the archive's
    8 arms is 2.5x, not 2.75x."""
    from src.main import select_arms

    arms = select_arms('all')
    assert len(arms) == 20

    slugs = {cfg.arm_slug('joint' if arm == 'joint' else 'disjoint')
             for arm, cfg in arms}
    assert len(slugs) == 20


def test_every_campaign_arm_enables_ccp_alpha():
    """It carries the larger of the two measured feasibility effects (+15.4pp
    vs alignment's +5.3pp), never costs feasibility, and no archived row has
    it. The 0.05 cap does not bind (0.2% of winners within 2x), so raising it
    would only add dead space at the degenerate-pruning end."""
    from src.main import select_arms, CAMPAIGN_CCP_ALPHA_MAX

    assert CAMPAIGN_CCP_ALPHA_MAX == 0.05
    for _, cfg in select_arms('all'):
        assert cfg.ccp_alpha_max == CAMPAIGN_CCP_ALPHA_MAX


def test_overlap_is_swept_on_the_aligned_joint_arms_only():
    """independent has no alignment so no overlap axis; joint-off likewise."""
    from src.main import select_arms, OVERLAP_THRESHOLDS

    assert set(OVERLAP_THRESHOLDS) == {0.5, 0.25, 0.1}

    swept = [cfg for arm, cfg in select_arms('all')
             if arm == 'joint' and cfg.alignment_enabled]
    assert len(swept) == 18
    assert {cfg.overlap_threshold for cfg in swept} == set(OVERLAP_THRESHOLDS)
    assert {cfg.delta_align for cfg in swept} == {0.0, 0.02, 0.05, 0.10, 0.20, None}


def test_align_objective_stays_at_blocks_on_every_arm():
    """Design §2.3: 'both' is real but narrow (11 of 50 replay cells, ~1.82x
    cost) and adds a third factor to a design already gaining one."""
    from src.main import select_arms

    assert {cfg.align_objective for _, cfg in select_arms('all')} == {'blocks'}


def test_delta_select_stays_out_of_the_sweep():
    """It moves the baseline and the treatment together, so it cannot be a
    treatment in a joint-vs-independent comparison."""
    from src.main import select_arms

    assert {cfg.delta_select for _, cfg in select_arms('all')} == {0.02}


def test_the_default_M_grid_is_the_archive_grid():
    """Comparability: C3 and C4 are matched-(M,k) comparisons against the
    archive, and main.py's old default shared only {25,50,100} with it."""
    from src.main import parse_args
    import src.main as main_mod

    args = parse_args([])
    M = args.M if args.M is not None else main_mod.DEFAULT_M_GRID
    assert M == [25, 50, 100, 150, 250]
