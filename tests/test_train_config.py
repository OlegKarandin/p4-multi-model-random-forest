"""TrainConfig is the arm definition: spec A.2's grid is a list of these."""
import pytest

from src.training.config import TrainConfig


def test_defaults_are_the_primary_joint_arm_at_delta_zero():
    cfg = TrainConfig()

    assert cfg.delta_align == 0.0
    assert cfg.alignment_enabled is True
    assert cfg.delta_select == 0.02
    # n_trees is set by utilisation (design 2026-09-03 spec 2.1(b)), not by
    # the measured capacity ceiling: the archive never reaches 11 trees and
    # p75 is 3. max_depth is kept at its previous 14 -- the re-derived grid
    # (scripts/capacity_ceiling.py, results/capacity_ceiling.csv) found no
    # cell within the measured n_trees<=15/max_depth<=14 box that exceeded
    # the 512-bit codeword limit, so raising max_depth was not ruled out by
    # the measurement, but is deliberately not adopted here (see config.py's
    # docstring).
    assert cfg.n_trees == 7
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


def test_negative_tolerances_are_rejected():
    with pytest.raises(ValueError, match='delta_align'):
        TrainConfig(delta_align=-0.01)
    with pytest.raises(ValueError, match='delta_select'):
        TrainConfig(delta_select=-0.01)


def test_unknown_encoding_is_rejected_by_arm_slug():
    with pytest.raises(ValueError, match='encoding'):
        TrainConfig().arm_slug('mixed')


def test_unknown_encoding_is_rejected_by_delta_align_label():
    # delta_align_label used to fall through its if/else on any unrecognized
    # `encoding` -- including a typo -- and silently return the joint-arm
    # value instead of failing.
    with pytest.raises(ValueError, match='encoding'):
        TrainConfig().delta_align_label('mixed')


def test_align_objective_defaults_to_blocks():
    assert TrainConfig().align_objective == 'blocks'


@pytest.mark.parametrize('retired', ['stages', 'both', 'stage'])
def test_the_retired_align_objectives_are_rejected(retired):
    """Design 2026-09-07 §4.3: 'stages' and 'both' are gone. The FIELD stays,
    so a config or manifest that recorded 'blocks' still loads unchanged, but
    the validator now refuses the retired values rather than silently running
    an objective that no longer exists."""
    with pytest.raises(ValueError, match='align_objective'):
        TrainConfig(align_objective=retired)


def test_align_objective_does_not_enter_the_arm_slug():
    """Deliberate: the slug format is what load_backup's filename parsing and
    every existing analysis read. ccp_alpha_max is a convenient stand-in field
    that (like align_objective) plays no part in the slug -- a campaign
    sweeping it would overwrite its own output and must change the slug
    first."""
    a = TrainConfig(delta_align=0.20, ccp_alpha_max=0.0)
    b = TrainConfig(delta_align=0.20, ccp_alpha_max=0.01)
    assert a.arm_slug('joint') == b.arm_slug('joint') == 'joint-d020'


def test_n_trees_min_and_ccp_alpha_max_defaults():
    """Both fields default to today's unmodified behavior."""
    cfg = TrainConfig()
    assert cfg.n_trees_min == 1
    assert cfg.ccp_alpha_max == 0.0


@pytest.mark.parametrize('n_trees_min_value', [0, 8])
def test_n_trees_min_out_of_range_rejected(n_trees_min_value):
    """n_trees_min must be in [1, n_trees], where n_trees defaults to 7."""
    with pytest.raises(ValueError, match='n_trees_min'):
        TrainConfig(n_trees_min=n_trees_min_value)


def test_n_trees_min_equal_to_n_trees_accepted():
    """n_trees_min == n_trees is accepted (the T-pinning mechanism)."""
    cfg = TrainConfig(n_trees=11, n_trees_min=11)
    assert cfg.n_trees_min == 11


def test_ccp_alpha_max_negative_rejected():
    """ccp_alpha_max must be >= 0.0."""
    with pytest.raises(ValueError, match='ccp_alpha_max'):
        TrainConfig(ccp_alpha_max=-0.01)


def test_ccp_alpha_max_zero_accepted():
    """ccp_alpha_max == 0.0 is accepted (off-by-one guard)."""
    cfg = TrainConfig(ccp_alpha_max=0.0)
    assert cfg.ccp_alpha_max == 0.0


def test_the_default_overlap_keeps_every_archived_slug_reproducible():
    """campaign_backup_20260825's 40 files were all written at 0.5. Suffixing
    unconditionally would rename their expected slugs and make
    campaign_data._expected_arm_slug reject the whole archive."""
    for delta, expected in [(0.0, 'joint-d000'), (0.02, 'joint-d002'),
                            (0.05, 'joint-d005'), (0.10, 'joint-d010'),
                            (0.20, 'joint-d020'), (None, 'joint-dinf')]:
        assert TrainConfig(delta_align=delta).arm_slug('joint') == expected


def test_the_overlap_threshold_tunable_is_gone():
    """D4. The ratio test never protected accuracy (delta_align does) or
    runtime (the whole sweep costs tenths of a second) -- it only masked the
    correctness gap at its extreme setting, which is now closed
    unconditionally. Its loosest setting strictly dominated: one real fitted
    pair moved from 193 key bytes (factor 36) at the 0.5 default to 176 bytes
    (factor 32) with no ratio test, at accuracy_spent exactly 0.0.

    The arm slug loses its conditional suffix with it: there is one alignment
    behaviour now, so `joint-d020` is unambiguous.
    """
    import inspect
    from src.training import config as cfg_mod
    from src.training import threshold_alignment as ta

    assert not hasattr(cfg_mod, 'DEFAULT_OVERLAP_THRESHOLD')
    assert 'overlap_threshold' not in cfg_mod.TrainConfig.__dataclass_fields__
    assert not hasattr(cfg_mod.TrainConfig, 'overlap_threshold_label')
    assert not hasattr(cfg_mod.TrainConfig, '_overlap_suffix')
    with pytest.raises(TypeError):
        cfg_mod.TrainConfig(overlap_threshold=0.5)

    assert cfg_mod.TrainConfig(delta_align=0.20).arm_slug('joint') == 'joint-d020'

    for fn in (ta.align_rf_thresholds, ta.align_with_policy):
        assert 'overlap_threshold' not in inspect.signature(fn).parameters, fn
