"""Spec A.2's arm grid and C.2's file naming."""
from src import main as m
from src.training.config import TrainConfig


def test_primary_grid_is_the_three_arms_the_headline_comparison_needs():
    """independent, joint with alignment OFF, joint at delta = 0. The two
    anchors bracket the frontier: `off` is a genuine skip of the
    align_rf_thresholds call, provably prediction-identical to the unaligned
    models, and it doubles as the ablation the reviewer asked for."""
    slugs = [cfg.arm_slug('disjoint' if arm == 'independent' else 'joint')
             for arm, cfg in m.PRIMARY_ARMS]

    assert slugs == ['independent', 'joint-off', 'joint-d000']


def test_sensitivity_grid_is_the_five_swept_tolerances():
    """0.01 is deliberately absent: it permits at most one DDoS sample to flip
    (one flip = 0.83% relative error at val_align ~3000, error ~0.04), so it is
    operationally identical to delta = 0."""
    slugs = [cfg.arm_slug('joint') for arm, cfg in m.SENSITIVITY_ARMS]

    assert slugs == ['joint-d002', 'joint-d005', 'joint-d010', 'joint-d020', 'joint-dinf']


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
    assert any(p.endswith('rf_t11_d14_M25_independent.csv') for p in paths)
    assert any(p.endswith('rf_t11_d14_M25_joint-d002.csv') for p in paths)
    assert any(p.endswith('rf_t11_d14_M25_joint-dinf.csv') for p in paths)


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
