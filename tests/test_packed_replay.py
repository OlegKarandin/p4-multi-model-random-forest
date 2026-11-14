"""The replay of a packed per-tree program (spec docs/superpowers/specs/
2026-11-11-packed-codes-stage-pin-design.md Sec 3.1, 4.3, 4.5, 5): it rebuilds every tree's
key bytes from the program text alone (the code_pk_ slices and their @pa_container_size
pins) and prices them as training does. A program with @stage pins (one generated before
the crossbar port of spec 2026-11-13) is refused."""
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


def _shared_bytes(layout):
  """{(feature, field byte): [code bits]} for every byte holding codes of two or more trees."""
  byte_codes = {}
  for (_, _, feature), (offset, bits) in layout.place.items():
    for byte in range(offset // 8, (offset + bits - 1) // 8 + 1):
      byte_codes.setdefault((feature, byte), []).append(bits)
  return {key: codes for key, codes in byte_codes.items() if len(codes) > 1}


def test_replay_rebuilds_shared_bytes(tmp_path):
  """Decision 4.4 (Review Focus 4): codes of <= 8 bits share bytes, so the replay rebuilds
  each code's BIT offset from its slices; on 50 designs every tree's rebuilt key equals
  packed.key_bytes, container identities included. Seed 1 puts six codes of 1-2 bits (two
  app trees, four ddos trees) into one byte."""
  shared_designs = 0
  for seed in range(50):
    program = parse_program_text(_generate(tmp_path, seed))
    layout = _layout(seed)
    shared_designs += bool(_shared_bytes(layout))
    for name in (name for name in program.tables if _TREE.match(name)):
      task, index = _TREE.match(name).groups()
      assert tree_key_bytes(program, name) == packed.key_bytes(layout, task, int(index)), (seed, name)
  assert shared_designs >= 25
  assert any(len(codes) >= 3 and max(codes) <= 4 for codes in _shared_bytes(_layout(1)).values())


def test_a_straddling_slice_is_refused(tmp_path):
  text = _generate(tmp_path, 1)
  program = parse_program_text(text)
  name, key = next((table, key) for (table, key), runs in program.key_slices.items()
                   if runs[0][0] - runs[0][1] < 7 and runs[0][1] % 8 < 7)
  hi, lo = program.key_slices[(name, key)][0]
  moved = text.replace("meta.%s[%d:%d] : ternary;" % (key, hi, lo),
                       "meta.%s[%d:%d] : ternary;" % (key, 8 * (lo // 8) + 8, 8 * (lo // 8) + 7))
  assert moved != text
  with pytest.raises(ValueError, match="straddles"):
    tree_key_bytes(parse_program_text(moved), name)


def test_the_program_parses_into_containers_and_slices(tmp_path):
  program = parse_program_text(_generate(tmp_path, 0))
  layout = _layout(0)
  assert program.containers == {"code_pk_" + feature: containers
                                for feature, containers in layout.containers.items()}
  for (task, tree, feature) in layout.place:
    name = "get_classification_tree_%s_%d" % (task, tree)
    assert (program.key_slices[(name, "code_pk_" + feature)]
            == packed.key_slices(layout, task, tree, feature))


@pytest.mark.parametrize("seed", range(6))
def test_training_prices_exactly_what_the_generated_program_replays(tmp_path, seed):
  app, ddos, names = _rng_design(seed)
  usage = evaluation.multi_model_memory_evaluation(app, ddos, names, names, PER_TREE_ENCODING)
  replay = model_breakdown(parse_program_text(_generate(tmp_path, seed)))
  assert (replay["stage_depth"], replay["blocks"]) == (usage.stage_depth, usage.blocks)


def test_a_program_with_stage_pins_is_refused(tmp_path):
  # An archived pinned program (spec 2026-11-11 Sec 4.3, e.g. heldout8) is replayed at the
  # commit that generated it; the model no longer prices pins, so it never prices one silently.
  text = _generate(tmp_path, 3)
  pinned = re.sub(r"(\n)(\s*@placement_priority\(\d+\)\s*\n\s*table get_classification_tree_app_0 \{)",
                  r"\1    @stage(11)\n\2", text, count=1)
  assert pinned != text
  model_breakdown(parse_program_text(text))
  with pytest.raises(ValueError, match="@stage"):
    parse_program_text(pinned)


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


# 2026-11-13: a single-key (joint) pool is now placed by the crossbar simulation, in p4c's
# order (ddos trees first, the table listed last first), not by the declared-price packer:
# seeds 0, 2 and 5 keep their stage depth and blocks; only which tree sits in which stage
# moves (packing.crossbar_stages_needed, "EVERY classification pool is simulated").
_JOINT_ORDER_2026_11_13 = {
    0: ('table_1_bwd_packet_length_max:1@5 table_2_flow_iat_max:1@5 '
        'table_3_flow_iat_min:1@5 table_0_bwd_iat_mean:1@6 '
        'get_classification_tree_app_2:4@7 get_classification_tree_app_3:4@7 '
        'get_classification_tree_ddos_0:4@7 get_classification_tree_ddos_1:4@7 '
        'get_classification_tree_ddos_2:4@7 get_classification_tree_ddos_3:4@7 '
        'get_classification_tree_app_0:4@8 get_classification_tree_app_1:4@8'),
    2: ('table_1_bwd_packet_length_max:1@5 table_0_bwd_iat_mean:1@6 '
        'table_2_fwd_packet_length_max:1@6 get_classification_tree_ddos_2:8@7 '
        'get_classification_tree_ddos_3:8@7 get_classification_tree_ddos_0:8@8 '
        'get_classification_tree_ddos_1:8@8 get_classification_tree_app_5:8@9 '
        'get_classification_tree_app_6:8@9 get_classification_tree_app_3:8@10 '
        'get_classification_tree_app_4:8@10 get_classification_tree_app_1:8@11 '
        'get_classification_tree_app_2:8@11 get_classification_tree_app_0:8@12'),
    5: ('table_0_flow_iat_min:1@5 table_1_fwd_packet_length_max:1@5 '
        'get_classification_tree_app_2:4@6 get_classification_tree_app_3:4@6 '
        'get_classification_tree_app_4:4@6 get_classification_tree_app_5:4@6 '
        'get_classification_tree_app_6:4@6 get_classification_tree_ddos_0:4@6 '
        'get_classification_tree_app_0:4@7 get_classification_tree_app_1:4@7'),
}


# Decisions 1a / 7.1 option B (2026-11-14): one register schedule, no gated-block order.
# Seed 2's fwd_packet_length_max no longer waits for the bwd registers, so its range
# table(s) move from stage 6 to 5 in every encoding; stage depth and blocks unchanged.
_OPTION_B_2026_11_14 = {
    (JOINT_ENCODING, 2): (
        'table_1_bwd_packet_length_max:1@5 table_2_fwd_packet_length_max:1@5 '
        'table_0_bwd_iat_mean:1@6 get_classification_tree_ddos_2:8@7 '
        'get_classification_tree_ddos_3:8@7 get_classification_tree_ddos_0:8@8 '
        'get_classification_tree_ddos_1:8@8 get_classification_tree_app_5:8@9 '
        'get_classification_tree_app_6:8@9 get_classification_tree_app_3:8@10 '
        'get_classification_tree_app_4:8@10 get_classification_tree_app_1:8@11 '
        'get_classification_tree_app_2:8@11 get_classification_tree_app_0:8@12'),
    (PER_TASK_ENCODING, 2): (
        'table_1_bwd_packet_length_max:1@5 table_2_fwd_packet_length_max:1@5 '
        'table_0_bwd_iat_mean:1@6 get_classification_tree_app_5:6@7 '
        'get_classification_tree_app_6:6@7 get_classification_tree_ddos_0:3@7 '
        'get_classification_tree_ddos_1:3@7 get_classification_tree_ddos_2:3@7 '
        'get_classification_tree_ddos_3:3@7 get_classification_tree_app_1:6@8 '
        'get_classification_tree_app_2:6@8 get_classification_tree_app_3:6@8 '
        'get_classification_tree_app_4:6@8 get_classification_tree_app_0:6@9'),
    (DISJOINT_ENCODING, 2): (
        'table_1_app_bwd_packet_length_max:1@5 table_2_app_fwd_packet_length_max:1@5 '
        'table_4_ddos_bwd_packet_length_max:1@5 table_5_ddos_fwd_packet_length_max:1@5 '
        'table_0_app_bwd_iat_mean:1@6 table_3_ddos_bwd_iat_mean:1@6 '
        'get_classification_tree_app_5:6@7 get_classification_tree_app_6:6@7 '
        'get_classification_tree_ddos_0:3@7 get_classification_tree_ddos_1:3@7 '
        'get_classification_tree_ddos_2:3@7 get_classification_tree_ddos_3:3@7 '
        'get_classification_tree_app_1:6@8 get_classification_tree_app_2:6@8 '
        'get_classification_tree_app_3:6@8 get_classification_tree_app_4:6@8 '
        'get_classification_tree_app_0:6@9'),
}


# Key pruning (decision 2.3, 2026-11-14): every tree keys only the code fields of the
# features it splits on. Seed 0 is the one design here where some tree splits fewer
# features than its task: its ddos trees 0, 1 and 3 now read only stage-5 range tables, so
# they move into stage 6 beside table_0 (rule E per tree), and joint keys narrow (36 -> 31
# blocks, stage depth 10 -> 9). Per-task and disjoint keep their totals.
_KEY_PRUNING_2026_11_14 = {
    (JOINT_ENCODING, 0): (9, 31, (
        'table_1_bwd_packet_length_max:1@5 table_2_flow_iat_max:1@5 table_3_flow_iat_min:1@5 '
        'get_classification_tree_ddos_0:1@6 get_classification_tree_ddos_1:4@6 '
        'get_classification_tree_ddos_3:1@6 table_0_bwd_iat_mean:1@6 '
        'get_classification_tree_app_0:5@7 get_classification_tree_app_1:5@7 '
        'get_classification_tree_app_2:4@7 get_classification_tree_app_3:4@7 '
        'get_classification_tree_ddos_2:3@7')),
    (PER_TASK_ENCODING, 0): (9, 24, (
        'table_1_bwd_packet_length_max:1@5 table_2_flow_iat_max:1@5 table_3_flow_iat_min:1@5 '
        'get_classification_tree_ddos_0:1@6 get_classification_tree_ddos_1:1@6 '
        'get_classification_tree_ddos_3:1@6 table_0_bwd_iat_mean:1@6 '
        'get_classification_tree_app_0:4@7 get_classification_tree_app_1:4@7 '
        'get_classification_tree_app_2:4@7 get_classification_tree_app_3:4@7 '
        'get_classification_tree_ddos_2:1@7')),
    (DISJOINT_ENCODING, 0): (9, 28, (
        'table_1_app_bwd_packet_length_max:1@5 table_2_app_flow_iat_max:1@5 '
        'table_3_app_flow_iat_min:1@5 table_5_ddos_bwd_packet_length_max:1@5 '
        'table_6_ddos_flow_iat_max:1@5 table_7_ddos_flow_iat_min:1@5 '
        'get_classification_tree_ddos_0:1@6 get_classification_tree_ddos_1:1@6 '
        'get_classification_tree_ddos_3:1@6 table_0_app_bwd_iat_mean:1@6 '
        'table_4_ddos_bwd_iat_mean:1@6 get_classification_tree_app_0:4@7 '
        'get_classification_tree_app_1:4@7 get_classification_tree_app_2:4@7 '
        'get_classification_tree_app_3:4@7 get_classification_tree_ddos_2:1@7')),
}


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING])
def test_other_encodings_replay_as_before(tmp_path, encoding):
  for seed in range(6):
    breakdown = model_breakdown(parse_program_text(_generate_as_recorded(tmp_path, seed, encoding)))
    tables = " ".join("%s:%d@%d" % (t["table"], t["blocks"], t["stage"])
                      for t in breakdown["tables"])
    recorded = _RECORDED_ON_39D83D6[encoding][seed]
    if encoding == JOINT_ENCODING and seed in _JOINT_ORDER_2026_11_13:
      recorded = recorded[:2] + (_JOINT_ORDER_2026_11_13[seed],)
    if (encoding, seed) in _OPTION_B_2026_11_14:
      recorded = recorded[:2] + (_OPTION_B_2026_11_14[(encoding, seed)],)
    recorded = _KEY_PRUNING_2026_11_14.get((encoding, seed), recorded)
    assert (breakdown["stage_depth"], breakdown["blocks"], tables) == recorded, seed
