"""Task 5: the `independent-tree` arm (encoding `disjoint_per_tree`) in training, CLI and reporting."""
import json
import re

import pytest

from p4model.program import DISJOINT_PER_TREE_ENCODING
from src.reporting import campaign_data as cd
from src.training import config, row_artifacts as ra
from src.training.campaign_run import row_id
from tests.test_row_artifacts import _forests
from tests.test_train_model_seed import _NAMES

SLUG = 'independent-tree'


def test_arm_maps_to_the_encoding_and_back():
    assert config.INDEPENDENT_TREE_ARM == SLUG
    assert config.arm_encoding(SLUG) == DISJOINT_PER_TREE_ENCODING
    assert config.TrainConfig(alignment_enabled=False).arm_slug(DISJOINT_PER_TREE_ENCODING) == SLUG
    # alignment fields are ignored, as for the other per-key arms
    assert config.TrainConfig().arm_slug(DISJOINT_PER_TREE_ENCODING) == SLUG


def test_select_arm_slugs_knows_independent_tree():
    from src.main import select_arm_slugs
    (arm, cfg), = select_arm_slugs([SLUG])
    assert arm == SLUG and cfg.alignment_enabled is False
    assert cfg.arm_slug(config.arm_encoding(arm)) == SLUG


def test_elimination_is_per_task_for_the_arm():
    from src.training.feature_selection import eliminates_shared
    assert eliminates_shared(SLUG) is False
    assert eliminates_shared('independent') is False
    for shared in ('joint', 'joint-task', 'joint-tree'):
        assert eliminates_shared(shared) is True


def test_campaign_data_accepts_the_arm_and_refuses_alignment_on_it():
    assert cd.INDEPENDENT_TREE_ARM_SLUG == SLUG
    assert cd._expected_arm_slug(SLUG, False, '', '') == SLUG
    for kwargs in ({'alignment_enabled': True}, {'alignment_postprocess': 'True'}):
        args = dict(arm=SLUG, alignment_enabled=False, delta_align_label='')
        args.update(kwargs)
        with pytest.raises(cd.MislabelledArtifactError):
            cd._expected_arm_slug(**args)


def test_row_id_round_trips():
    rid = row_id(SLUG, 35, 7, 9)
    assert rid == 'independent-tree_M035_s07_k09'
    m = cd._ROW_ID_RE.match(rid)
    assert (m.group('arm_slug'), m.group('M'), m.group('split'), m.group('k')) == (SLUG, '035', '07', '09')
    assert cd._ROW_ID_RE.match(row_id(SLUG, float('inf'), 0, 3)).group('arm_slug') == SLUG


def test_prefixes_do_not_misfile_the_arm():
    from scripts import paper_resource_baseline as prb
    rid = row_id(SLUG, 35, 7, 9)
    assert prb.arm_of(rid) == SLUG
    assert SLUG in prb.ARM_ORDER
    assert prb.arm_of(row_id('independent', 35, 7, 9)) == 'independent'
    # no joint-arm predicate accepts it
    assert not rid.startswith('joint')
    from src.reporting.claims import JOINT_ARM_SLUGS, REPORTED_JOINT_ARM_SLUGS
    assert SLUG not in JOINT_ARM_SLUGS and SLUG not in REPORTED_JOINT_ARM_SLUGS
    # calibration-archive fallback stays exact
    assert prb.arm_of('heldout_independent_M150_k7_s11') == 'independent'
    assert prb.arm_of('independent-tree_calib_x') != 'independent'


def test_row_artifacts_generate_the_new_encoding(tmp_path):
    from p4gen.evaluation import multi_model_memory_evaluation
    from p4gen.p4_replay import model_breakdown_text
    fa, fd = _forests()
    usage = multi_model_memory_evaluation(fa, fd, _NAMES, _NAMES, DISJOINT_PER_TREE_ENCODING)
    ctx = ra.RowContext(str(tmp_path), SLUG, 35, 'abc123')
    rid = row_id(SLUG, 35, 0, 3)
    model = ra.write_row_artifacts(ctx, rid, fa, fd, _NAMES, _NAMES, DISJOINT_PER_TREE_ENCODING, usage)
    p4 = (tmp_path / 'designs' / (rid + '.p4'))
    assert p4.is_file()
    on_disk = json.loads((tmp_path / 'designs' / (rid + '.model.json')).read_text())
    assert on_disk == model
    replay = model_breakdown_text(p4.read_text(encoding='utf-8'), rid)
    assert (model['stage_depth'], model['blocks']) == (replay['stage_depth'], replay['blocks'])
    assert model['training_blocks'] == usage.blocks
