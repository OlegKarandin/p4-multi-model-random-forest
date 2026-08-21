"""TrainConfig is the arm definition: spec A.2's grid is a list of these."""
import pytest

from src.training.config import TrainConfig


def test_defaults_are_the_primary_joint_arm_at_delta_zero():
    cfg = TrainConfig()

    assert cfg.delta_align == 0.0
    assert cfg.alignment_enabled is True
    assert cfg.delta_select == 0.02
    assert cfg.overlap_threshold == 0.5
    # Rederived from the measured capacity ceiling, not chosen by hand:
    # (11, 14) is the grid cell with the largest reachable search space,
    # ceil(n_trees / 2) * (max_depth - 1) = 78, whose joint codeword stays
    # within 512 bits on all 3 splits at a configuration the search can
    # reach (scripts/capacity_ceiling.py, results/capacity_ceiling.csv).
    assert cfg.n_trees == 11
    assert cfg.max_depth == 14
    assert cfg.n_trials == 1000
    assert cfg.min_feasible_before_stop == 25
    assert cfg.lookback == 20


def test_config_is_frozen_so_a_worker_cannot_mutate_the_arm_under_itself():
    cfg = TrainConfig()

    with pytest.raises(Exception):
        cfg.delta_align = 0.5


def test_arm_slug_matches_the_spec_c2_filenames():
    """Spec C.2 names the files rf_t11_d14_M25_<slug>.csv, and the slug is the
    only thing that identifies which arm an artifact came from."""
    assert TrainConfig().arm_slug('disjoint') == 'independent'
    assert TrainConfig(alignment_enabled=False).arm_slug('joint') == 'joint-off'
    assert TrainConfig(delta_align=0.0).arm_slug('joint') == 'joint-d000'
    assert TrainConfig(delta_align=0.02).arm_slug('joint') == 'joint-d002'
    assert TrainConfig(delta_align=0.05).arm_slug('joint') == 'joint-d005'
    assert TrainConfig(delta_align=0.10).arm_slug('joint') == 'joint-d010'
    assert TrainConfig(delta_align=0.20).arm_slug('joint') == 'joint-d020'
    assert TrainConfig(delta_align=None).arm_slug('joint') == 'joint-dinf'


def test_independent_arm_slug_ignores_the_alignment_fields():
    """Alignment runs in the joint arm only, so delta_align must not leak into
    an independent arm's identity -- two independent runs differing only in
    delta_align would otherwise write to different files and look like two
    treatments."""
    assert TrainConfig(delta_align=0.2).arm_slug('disjoint') == 'independent'
    assert TrainConfig(alignment_enabled=False).arm_slug('disjoint') == 'independent'


def test_delta_align_label_is_what_goes_in_the_row():
    """Spec C.1: delta_align is a float, or "inf", or "" for independent."""
    assert TrainConfig(delta_align=0.05).delta_align_label() == '0.05'
    assert TrainConfig(delta_align=None).delta_align_label() == 'inf'
    assert TrainConfig(alignment_enabled=False).delta_align_label() == ''


def test_delta_align_label_disjoint_encoding_suppresses_it_like_arm_slug():
    """Mirrors arm_slug('disjoint'): the independent arm never runs alignment,
    so its row must not carry the joint arm's default alignment_enabled=True,
    delta_align=0.0 -- even though those are TrainConfig()'s defaults."""
    cfg = TrainConfig()
    assert cfg.alignment_enabled is True
    assert cfg.delta_align == 0.0

    assert cfg.delta_align_label('disjoint') == ''
    assert cfg.delta_align_label('joint') == '0'
    assert cfg.delta_align_label() == '0'


def test_overlap_threshold_label_is_what_goes_in_the_row():
    """Spec C.1: overlap_threshold is a float, or "" when alignment did not
    run -- mirrors delta_align_label, since overlap_threshold is only
    consulted by align_rf_thresholds, which is never called for the
    independent arm or the joint-off ablation."""
    assert TrainConfig(overlap_threshold=0.5).overlap_threshold_label() == '0.5'
    assert TrainConfig(overlap_threshold=0.75).overlap_threshold_label('joint') == '0.75'
    assert TrainConfig(alignment_enabled=False).overlap_threshold_label() == ''


def test_overlap_threshold_label_disjoint_encoding_suppresses_it_like_arm_slug():
    """Mirrors arm_slug('disjoint') and delta_align_label('disjoint'): the
    independent arm never runs alignment, so its row must not carry the
    joint arm's overlap_threshold setting -- even though TrainConfig()'s
    default (0.5) is shared by both arms' configs."""
    cfg = TrainConfig()
    assert cfg.overlap_threshold == 0.5

    assert cfg.overlap_threshold_label('disjoint') == ''
    assert cfg.overlap_threshold_label('joint') == '0.5'
    assert cfg.overlap_threshold_label() == '0.5'


def test_negative_tolerances_are_rejected():
    with pytest.raises(ValueError, match='delta_align'):
        TrainConfig(delta_align=-0.01)
    with pytest.raises(ValueError, match='delta_select'):
        TrainConfig(delta_select=-0.01)


def test_unknown_encoding_is_rejected_by_arm_slug():
    with pytest.raises(ValueError, match='encoding'):
        TrainConfig().arm_slug('mixed')


def test_unknown_encoding_is_rejected_by_delta_align_label():
    # delta_align_label and overlap_threshold_label used to fall through
    # their if/else on any unrecognized `encoding` -- including a typo --
    # and silently return the joint-arm value instead of failing.
    with pytest.raises(ValueError, match='encoding'):
        TrainConfig().delta_align_label('mixed')


def test_unknown_encoding_is_rejected_by_overlap_threshold_label():
    with pytest.raises(ValueError, match='encoding'):
        TrainConfig().overlap_threshold_label('mixed')
