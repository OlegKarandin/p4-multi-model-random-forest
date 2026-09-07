"""TrainConfig is the arm definition: spec A.2's grid is a list of these."""
import pytest

from src.training.config import TrainConfig


def test_defaults_are_the_primary_joint_arm_at_delta_zero():
    cfg = TrainConfig()

    assert cfg.delta_align == 0.0
    assert cfg.alignment_enabled is True
    assert cfg.delta_select == 0.02
    assert cfg.overlap_threshold == 0.5
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


def test_overlap_threshold_label_is_what_goes_in_the_row():
    """Spec C.1: overlap_threshold is a float, or "" when alignment did not
    run -- mirrors delta_align_label, since overlap_threshold is only
    consulted by align_with_policy, which is never called for the
    independent arm or the joint-off ablation.

    Also covers the disjoint-encoding suppression that arm_slug and
    delta_align_label each get their own test for (mirrors
    arm_slug('disjoint') / delta_align_label('disjoint')): the independent
    arm never runs alignment, so its row must not carry the joint arm's
    overlap_threshold setting, even though TrainConfig()'s default (0.5) is
    shared by both arms' configs.
    """
    assert TrainConfig(overlap_threshold=0.5).overlap_threshold_label() == '0.5'
    assert TrainConfig(overlap_threshold=0.75).overlap_threshold_label('joint') == '0.75'
    assert TrainConfig(alignment_enabled=False).overlap_threshold_label() == ''
    assert TrainConfig().overlap_threshold_label('disjoint') == ''


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


def test_overlap_threshold_enters_the_slug_when_it_leaves_the_default():
    """Design 2026-09-03 §3: a campaign sweeping overlap_threshold must
    distinguish its arms in the filename. main.py's skip_existing treats an
    existing path as 'cell done', so colliding arms are silently SKIPPED --
    quieter than an overwrite and just as wrong."""
    assert TrainConfig(delta_align=0.20).arm_slug('joint') == 'joint-d020'
    assert TrainConfig(delta_align=0.20, overlap_threshold=0.25).arm_slug(
        'joint') == 'joint-d020-o025'
    assert TrainConfig(delta_align=0.20, overlap_threshold=0.1).arm_slug(
        'joint') == 'joint-d020-o010'
    assert TrainConfig(delta_align=None, overlap_threshold=0.1).arm_slug(
        'joint') == 'joint-dinf-o010'


def test_the_three_overlap_values_give_three_distinct_paths_per_delta():
    """Spec §4's slug-uniqueness test, over the full swept grid."""
    slugs = {TrainConfig(delta_align=d, overlap_threshold=o).arm_slug('joint')
             for d in (0.0, 0.02, 0.05, 0.10, 0.20, None)
             for o in (0.5, 0.25, 0.1)}

    assert len(slugs) == 18


def test_the_default_overlap_keeps_every_archived_slug_reproducible():
    """campaign_backup_20260825's 40 files were all written at 0.5. Suffixing
    unconditionally would rename their expected slugs and make
    campaign_data._expected_arm_slug reject the whole archive."""
    for delta, expected in [(0.0, 'joint-d000'), (0.02, 'joint-d002'),
                            (0.05, 'joint-d005'), (0.10, 'joint-d010'),
                            (0.20, 'joint-d020'), (None, 'joint-dinf')]:
        assert TrainConfig(delta_align=delta).arm_slug('joint') == expected


def test_arms_without_alignment_never_carry_an_overlap_suffix():
    """overlap_threshold only governs which range pairs align_rf_thresholds
    considers, so it is meaningless where that function is never called --
    suppressed the same way overlap_threshold_label suppresses the column."""
    assert TrainConfig(overlap_threshold=0.1).arm_slug('disjoint') == 'independent'
    assert TrainConfig(alignment_enabled=False,
                       overlap_threshold=0.1).arm_slug('joint') == 'joint-off'
