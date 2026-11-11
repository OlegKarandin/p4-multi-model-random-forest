"""The replay of a packed per-tree program (spec docs/superpowers/specs/
2026-11-11-packed-codes-stage-pin-design.md Sec 3.1, 4.3, 4.5, 5): it rebuilds every tree's
key bytes from the program text alone (the code_pk_ slices and their @pa_container_size
pins), prices them as training does, and checks the program's @stage pins against the
tables its own placement moved."""
import os
import re

import pytest

from p4gen import build_p4_script as bps
from p4gen import evaluation
from p4gen.p4_replay import (model_breakdown, parse_program_text, range_action_bytes,
                             tree_key_bytes)
from p4model import packed, packing
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING)
from tests.test_packed_program import _specs
from tests.test_per_task_codes import _own, _rng_design

_TREE = re.compile(r"^get_classification_tree_(\w+)_(\d+)$")


def _generate(tmp_path, seed, encoding=PER_TREE_ENCODING, **kw):
  app, ddos, names = _rng_design(seed)
  specs = _specs(app, ddos, names, names)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    specs = _specs(app, ddos, names, names, joint, joint)
  path = bps.generate_p4_program(specs, output_dir=str(tmp_path) + os.sep, encoding=encoding,
                                 **kw)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


def _layout(seed):
  app, ddos, names = _rng_design(seed)
  specs = _specs(app, ddos, names, names)
  table_plan, _, own = bps.resolve_per_tree_plan(specs)
  return bps.per_tree_layout(table_plan, own, specs)


def _pins(seed):
  app, ddos, names = _rng_design(seed)
  return evaluation.tree_stage_pins(app, ddos, names, names, PER_TREE_ENCODING)


@pytest.mark.parametrize("seed", range(6))
def test_replay_rebuilds_the_layouts_key_bytes_from_the_text(tmp_path, seed):
  program = parse_program_text(_generate(tmp_path, seed))
  layout = _layout(seed)
  trees = [name for name in program.tables if _TREE.match(name)]
  assert trees
  for name in trees:
    task, index = _TREE.match(name).groups()
    expected = packed.key_bytes(layout, task, int(index))
    assert sorted(tree_key_bytes(program, name)) == sorted(expected), name
    assert tree_key_bytes(program, name) == expected, name      # the order is recoverable too


def test_the_program_parses_into_containers_slices_and_pins(tmp_path):
  text = _generate(tmp_path, 0, stage_pins={"get_classification_tree_ddos_1": 10})
  program = parse_program_text(text)
  layout = _layout(0)
  assert program.containers == {"code_pk_" + feature: containers
                                for feature, containers in layout.containers.items()}
  for (task, tree, feature) in layout.place:
    name = "get_classification_tree_%s_%d" % (task, tree)
    assert (program.key_slices[(name, "code_pk_" + feature)]
            == packed.key_slices(layout, task, tree, feature))
  assert program.stage_pins == {"get_classification_tree_ddos_1": 10}


@pytest.mark.parametrize("seed", range(6))
def test_training_prices_exactly_what_the_generated_program_replays(tmp_path, seed):
  app, ddos, names = _rng_design(seed)
  usage = evaluation.multi_model_memory_evaluation(app, ddos, names, names, PER_TREE_ENCODING)
  pins = _pins(seed)
  replay = model_breakdown(parse_program_text(_generate(tmp_path, seed, stage_pins=pins)))
  assert (replay["stage_depth"], replay["blocks"]) == (usage.stage_depth, usage.blocks)
  assert replay["stage_pins_consistent"] is True


@pytest.mark.parametrize("seed", [0, 2, 3, 5])
def test_moved_tables_replay_where_training_put_them(tmp_path, monkeypatch, seed):
  """A forced one-group budget moves tables (as tests/test_packed_pool.py does): the replay,
  reading the same budget, moves the same ones, and with their pins in the program it prices
  what training priced and calls the pins consistent; without them it flags the program."""
  monkeypatch.setattr(packing, "TERNARY_CROSSBAR_GROUPS_PER_STAGE", 1)
  app, ddos, names = _rng_design(seed)
  usage = evaluation.multi_model_memory_evaluation(app, ddos, names, names, PER_TREE_ENCODING)
  pins = _pins(seed)
  assert pins
  replay = model_breakdown(parse_program_text(_generate(tmp_path, seed, stage_pins=pins)))
  assert (replay["stage_depth"], replay["blocks"]) == (usage.stage_depth, usage.blocks)
  assert replay["stage_pins_consistent"] is True
  stages = {table["table"]: table["stage"] for table in replay["tables"]}
  assert all(stages[name] == stage for name, stage in pins.items())
  unpinned = model_breakdown(parse_program_text(_generate(tmp_path, seed)))
  assert unpinned["stage_pins_consistent"] is False
  assert (unpinned["stage_depth"], unpinned["blocks"]) == (usage.stage_depth, usage.blocks)


def test_a_program_whose_pins_disagree_with_the_model_is_flagged(tmp_path):
  # Review Focus 4: app_0 is not a moved table (seed 3 moves none), yet the program pins it.
  assert _pins(3) == {}
  text = _generate(tmp_path, 3, stage_pins={"get_classification_tree_app_0": 11})
  breakdown = model_breakdown(parse_program_text(text))
  assert breakdown["stage_pins_consistent"] is False
  # The pin is p4c's earliest stage for the table (spec Sec 4.3), so the replay places it
  # no earlier: what the program would cost, not what the model wanted.
  stages = {table["table"]: table["stage"] for table in breakdown["tables"]}
  assert stages["get_classification_tree_app_0"] == 11
  assert breakdown["stage_depth"] == 11 + 1 + 1


@pytest.mark.parametrize("seed", range(6))
def test_range_action_bytes_of_a_packed_program_are_the_fields_container_bytes(tmp_path, seed):
  program = parse_program_text(_generate(tmp_path, seed))
  layout = _layout(seed)
  written = range_action_bytes(program)
  ranges = [name for name, keys in program.tables.items() if name.startswith("table_") and keys]
  assert ranges and set(written) == set(ranges)
  for name in ranges:
    feature = re.match(r"^table_\d+_(\w+)$", name).group(1)
    assert written[name] == packed.bus_bytes(layout, feature) == sum(layout.containers[feature]) // 8


def _generate_as_recorded(tmp_path, seed, encoding):
  """The program the values below were recorded from: generate_P4_code, as
  tests/test_action_bus_and_block_order.py calls it."""
  app, ddos, names = _rng_design(seed)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    intervals = (joint, joint)
  else:
    intervals = (_own(app, names), _own(ddos, names))
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=intervals[0],
                              feature_intervals_ddos=intervals[1],
                              output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                              selected_features_app=names, selected_features_ddos=names,
                              encoding=encoding)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


# model_breakdown of the joint / per_task / disjoint programs of _rng_design(0..5), recorded by
# running p4_replay on 39d83d6 itself (and checked identical on a88e8f2, before this change).
# Per seed: (stage_depth, blocks, "table:blocks@stage ..." in model_breakdown's table order).
_RECORDED_ON_39D83D6 = {
    'joint': {
        0: (10, 36,
            'table_1_bwd_packet_length_max:1@5 table_2_flow_iat_max:1@5 table_3_flow_iat_min:1@5 '
            'table_0_bwd_iat_mean:1@6 get_classification_tree_app_0:4@7 '
            'get_classification_tree_app_1:4@7 get_classification_tree_app_2:4@7 '
            'get_classification_tree_app_3:4@7 get_classification_tree_ddos_0:4@7 '
            'get_classification_tree_ddos_1:4@7 get_classification_tree_ddos_2:4@8 '
            'get_classification_tree_ddos_3:4@8'),
        1: (9, 7,
            'table_0_bwd_iat_mean:1@6 get_classification_tree_app_0:1@7 '
            'get_classification_tree_app_1:1@7 get_classification_tree_ddos_0:1@7 '
            'get_classification_tree_ddos_1:1@7 get_classification_tree_ddos_2:1@7 '
            'get_classification_tree_ddos_3:1@7'),
        2: (14, 91,
            'table_1_bwd_packet_length_max:1@5 table_0_bwd_iat_mean:1@6 '
            'table_2_fwd_packet_length_max:1@6 get_classification_tree_app_0:8@7 '
            'get_classification_tree_app_1:8@7 get_classification_tree_app_2:8@8 '
            'get_classification_tree_app_3:8@8 get_classification_tree_app_4:8@9 '
            'get_classification_tree_app_5:8@9 get_classification_tree_app_6:8@10 '
            'get_classification_tree_ddos_0:8@10 get_classification_tree_ddos_1:8@11 '
            'get_classification_tree_ddos_2:8@11 get_classification_tree_ddos_3:8@12'),
        3: (8, 3,
            'table_0_flow_iat_max:1@5 get_classification_tree_app_0:1@6 '
            'get_classification_tree_ddos_0:1@6'),
        4: (8, 4,
            'table_0_flow_iat_min:1@5 get_classification_tree_app_0:1@6 '
            'get_classification_tree_app_1:1@6 get_classification_tree_ddos_0:1@6'),
        5: (9, 34,
            'table_0_flow_iat_min:1@5 table_1_fwd_packet_length_max:1@5 '
            'get_classification_tree_app_0:4@6 get_classification_tree_app_1:4@6 '
            'get_classification_tree_app_2:4@6 get_classification_tree_app_3:4@6 '
            'get_classification_tree_app_4:4@6 get_classification_tree_app_5:4@6 '
            'get_classification_tree_app_6:4@7 get_classification_tree_ddos_0:4@7'),
    },
    'per_task': {
        0: (9, 24,
            'table_1_bwd_packet_length_max:1@5 table_2_flow_iat_max:1@5 table_3_flow_iat_min:1@5 '
            'table_0_bwd_iat_mean:1@6 get_classification_tree_app_0:4@7 '
            'get_classification_tree_app_1:4@7 get_classification_tree_app_2:4@7 '
            'get_classification_tree_app_3:4@7 get_classification_tree_ddos_0:1@7 '
            'get_classification_tree_ddos_1:1@7 get_classification_tree_ddos_2:1@7 '
            'get_classification_tree_ddos_3:1@7'),
        1: (9, 7,
            'table_0_bwd_iat_mean:1@6 get_classification_tree_app_0:1@7 '
            'get_classification_tree_app_1:1@7 get_classification_tree_ddos_0:1@7 '
            'get_classification_tree_ddos_1:1@7 get_classification_tree_ddos_2:1@7 '
            'get_classification_tree_ddos_3:1@7'),
        2: (11, 57,
            'table_1_bwd_packet_length_max:1@5 table_0_bwd_iat_mean:1@6 '
            'table_2_fwd_packet_length_max:1@6 get_classification_tree_app_5:6@7 '
            'get_classification_tree_app_6:6@7 get_classification_tree_ddos_0:3@7 '
            'get_classification_tree_ddos_1:3@7 get_classification_tree_ddos_2:3@7 '
            'get_classification_tree_ddos_3:3@7 get_classification_tree_app_1:6@8 '
            'get_classification_tree_app_2:6@8 get_classification_tree_app_3:6@8 '
            'get_classification_tree_app_4:6@8 get_classification_tree_app_0:6@9'),
        3: (8, 3,
            'table_0_flow_iat_max:1@5 get_classification_tree_app_0:1@6 '
            'get_classification_tree_ddos_0:1@6'),
        4: (8, 4,
            'table_0_flow_iat_min:1@5 get_classification_tree_app_0:1@6 '
            'get_classification_tree_app_1:1@6 get_classification_tree_ddos_0:1@6'),
        5: (8, 24,
            'table_0_flow_iat_min:1@5 table_1_fwd_packet_length_max:1@5 '
            'get_classification_tree_app_0:3@6 get_classification_tree_app_1:3@6 '
            'get_classification_tree_app_2:3@6 get_classification_tree_app_3:3@6 '
            'get_classification_tree_app_4:3@6 get_classification_tree_app_5:3@6 '
            'get_classification_tree_app_6:3@6 get_classification_tree_ddos_0:1@6'),
    },
    'disjoint': {
        0: (9, 28,
            'table_1_app_bwd_packet_length_max:1@5 table_2_app_flow_iat_max:1@5 '
            'table_3_app_flow_iat_min:1@5 table_5_ddos_bwd_packet_length_max:1@5 '
            'table_6_ddos_flow_iat_max:1@5 table_7_ddos_flow_iat_min:1@5 '
            'table_0_app_bwd_iat_mean:1@6 table_4_ddos_bwd_iat_mean:1@6 '
            'get_classification_tree_app_0:4@7 get_classification_tree_app_1:4@7 '
            'get_classification_tree_app_2:4@7 get_classification_tree_app_3:4@7 '
            'get_classification_tree_ddos_0:1@7 get_classification_tree_ddos_1:1@7 '
            'get_classification_tree_ddos_2:1@7 get_classification_tree_ddos_3:1@7'),
        1: (9, 8,
            'table_0_app_bwd_iat_mean:1@6 table_1_ddos_bwd_iat_mean:1@6 '
            'get_classification_tree_app_0:1@7 get_classification_tree_app_1:1@7 '
            'get_classification_tree_ddos_0:1@7 get_classification_tree_ddos_1:1@7 '
            'get_classification_tree_ddos_2:1@7 get_classification_tree_ddos_3:1@7'),
        2: (11, 60,
            'table_1_app_bwd_packet_length_max:1@5 table_4_ddos_bwd_packet_length_max:1@5 '
            'table_0_app_bwd_iat_mean:1@6 table_2_app_fwd_packet_length_max:1@6 '
            'table_3_ddos_bwd_iat_mean:1@6 table_5_ddos_fwd_packet_length_max:1@6 '
            'get_classification_tree_app_5:6@7 get_classification_tree_app_6:6@7 '
            'get_classification_tree_ddos_0:3@7 get_classification_tree_ddos_1:3@7 '
            'get_classification_tree_ddos_2:3@7 get_classification_tree_ddos_3:3@7 '
            'get_classification_tree_app_1:6@8 get_classification_tree_app_2:6@8 '
            'get_classification_tree_app_3:6@8 get_classification_tree_app_4:6@8 '
            'get_classification_tree_app_0:6@9'),
        3: (8, 4,
            'table_0_app_flow_iat_max:1@5 table_1_ddos_flow_iat_max:1@5 '
            'get_classification_tree_app_0:1@6 get_classification_tree_ddos_0:1@6'),
        4: (8, 5,
            'table_0_app_flow_iat_min:1@5 table_1_ddos_flow_iat_min:1@5 '
            'get_classification_tree_app_0:1@6 get_classification_tree_app_1:1@6 '
            'get_classification_tree_ddos_0:1@6'),
        5: (8, 26,
            'table_0_app_flow_iat_min:1@5 table_1_app_fwd_packet_length_max:1@5 '
            'table_2_ddos_flow_iat_min:1@5 table_3_ddos_fwd_packet_length_max:1@5 '
            'get_classification_tree_app_0:3@6 get_classification_tree_app_1:3@6 '
            'get_classification_tree_app_2:3@6 get_classification_tree_app_3:3@6 '
            'get_classification_tree_app_4:3@6 get_classification_tree_app_5:3@6 '
            'get_classification_tree_app_6:3@6 get_classification_tree_ddos_0:1@6'),
    },
}


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING])
def test_other_encodings_replay_as_before(tmp_path, encoding):
  for seed in range(6):
    breakdown = model_breakdown(parse_program_text(_generate_as_recorded(tmp_path, seed, encoding)))
    tables = " ".join("%s:%d@%d" % (t["table"], t["blocks"], t["stage"])
                      for t in breakdown["tables"])
    assert (breakdown["stage_depth"], breakdown["blocks"], tables) == \
        _RECORDED_ON_39D83D6[encoding][seed], seed
    assert breakdown["stage_pins_consistent"] is True
