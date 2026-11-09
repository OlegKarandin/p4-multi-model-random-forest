"""The range pool's action-data bus budget and the gated register blocks' order (2026-11-09,
report results/encoding_evidence_2026_11/model3/report.md, I1a and "I1b follow-up")."""
import dataclasses
import os

import pytest

from p4gen import build_p4_script as bps
from p4gen import evaluation as ev
from p4gen.p4_replay import model_breakdown, parse_program_text, range_action_bits
from p4model.packing import crossbar_stages_needed
from p4model.program import (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)
from p4model.registers import register_stage_schedule
from p4model.tables import action_ram_entry_bytes
from p4model.target import (ACTION_DATA_BUS_BUDGET, ACTION_DATA_BUS_BYTES,
                            ACTION_RAM_WORD_BYTES, PER_TREE_MAX_ACTION_BYTES)
from tests.test_per_task_codes import _own, _rng_design


def test_the_budget_is_the_bus_less_one_ram_word():
  assert (ACTION_DATA_BUS_BYTES, ACTION_RAM_WORD_BYTES) == (128, 16)
  assert ACTION_DATA_BUS_BUDGET == 112 == PER_TREE_MAX_ACTION_BYTES


@pytest.mark.parametrize("bits,entry", [
    (0, 0), (1, 1), (8, 1), (9, 2), (16, 2), (17, 4), (32, 4), (33, 8), (64, 8),
    (65, 16), (128, 16), (129, 32), (256, 32), (257, 48), (600, 80)])
def test_an_action_ram_entry_is_a_power_of_two_bytes_then_whole_words(bits, entry):
  assert action_ram_entry_bytes(bits) == entry


def _range_pool(action_bytes, level=4):
  n = len(action_bytes)
  return crossbar_stages_needed([(1, 2)] * n, readiness_levels=[level] * n,
                                key_fields=[frozenset({(("val", i), 2)}) for i in range(n)],
                                action_bytes=action_bytes)


def test_range_tables_share_a_stage_up_to_the_bus_budget():
  # M075_s03_k03's stage 5 in p4c: 64 + 80 bytes refused, three stages
  assert _range_pool([64, 80]).table_stages == (4, 5)
  assert _range_pool([64, 80, 80]).table_stages == (4, 5, 6)
  # 112 is allowed, one byte more is not
  assert _range_pool([64, 48]).table_stages == (4, 4)
  assert _range_pool([64, 32, 16, 1]).table_stages == (4, 4, 4, 5)
  # without the budget, the old placement: everything in one stage
  plan = crossbar_stages_needed([(1, 2)] * 2, readiness_levels=[4, 4],
                                key_fields=[frozenset({(("val", i), 2)}) for i in range(2)])
  assert plan.table_stages == (4, 4)


def test_a_table_alone_always_takes_an_empty_stage():
  # its own limit is per_tree.pins_for's, not the stage budget's
  assert _range_pool([128]).table_stages == (4,)
  assert _range_pool([128, 128]).table_stages == (4, 5)


def test_the_budget_is_for_the_range_pool_only():
  with pytest.raises(ValueError, match="range pool"):
    crossbar_stages_needed([(1, 2)], readiness_levels=[4],
                           key_fields=[frozenset({("code", 2)})],
                           key_field_bits=[(16,)], action_bytes=[8])
  with pytest.raises(ValueError, match="aligned"):
    crossbar_stages_needed([(1, 2)] * 2, readiness_levels=[4, 4], action_bytes=[8])


def test_the_replay_counts_the_keyed_bits_each_range_table_writes(tmp_path):
  app, ddos, names = _rng_design(3)
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=_own(app, names),
                              feature_intervals_ddos=_own(ddos, names),
                              output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                              selected_features_app=names, selected_features_ddos=names,
                              encoding=PER_TASK_ENCODING)
  with open(path, encoding="utf-8") as handle:
    program = parse_program_text(handle.read())
  written = range_action_bits(program)
  for name, bits in written.items():
    feature = name.split("_", 2)[2]
    expected = sum(program.bits[field] for field in program.bits
                   if field in ("code_tk_app_" + feature, "code_tk_ddos_" + feature))
    assert bits == expected > 0, name


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING,
                                      PER_TREE_ENCODING])
@pytest.mark.parametrize("seed", range(6))
def test_training_and_replay_give_each_range_table_the_same_bus_bytes(tmp_path, encoding, seed):
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
    program = parse_program_text(handle.read())
  pool = ev._pool_inputs(app, ddos, names, names, encoding)
  written = range_action_bits(program)
  ranges = [name for name, keys in program.tables.items() if name.startswith("table_") and keys]
  assert sorted(pool["range_action_bytes"]) == sorted(
      action_ram_entry_bytes(written[name]) for name in ranges)


# --- the gated register blocks' order -------------------------------------------------

# fwd_iat_max and bwd_iat_min: each a dependency register plus a value register, so the two
# gated blocks tie on chain depth; fwd_packet_length_max: one value register.
_TIED = ["fwd_iat_max", "bwd_iat_min", "flow_iat_max"]
_FWD_DEEPER = ["fwd_iat_max", "bwd_packet_length_max", "flow_iat_max"]
_BWD_DEEPER = ["fwd_packet_length_max", "bwd_iat_min", "flow_iat_max"]


def _last(placed, prefix):
  return max(stage for name, stage in placed.items() if name.startswith(prefix))


def test_tied_gated_blocks_keep_the_emission_order():
  # p4c's stage depth on all 272 pinned pilot designs; ties to bwd over-priced 15 at k>=9
  pinned = register_stage_schedule(_TIED)
  assert pinned == register_stage_schedule(_TIED, fwd_pinned=False)
  assert _last(pinned, "fwd") < _last(pinned, "bwd")


def test_the_deeper_gated_block_goes_first():
  # bwd deeper: first with the pin (fwd floored at its last stage), second in emission order
  assert register_stage_schedule(_BWD_DEEPER) == {
      "flow_forward_srcaddr": 3, "flow_last_arrival_time": 3, "flow_iat_max": 4,
      "bwd_last_arrival_time": 4, "bwd_iat_min": 5, "fwd_packet_length_max": 5}
  assert register_stage_schedule(_BWD_DEEPER, fwd_pinned=False) == {
      "flow_forward_srcaddr": 3, "flow_last_arrival_time": 3, "flow_iat_max": 4,
      "fwd_packet_length_max": 4, "bwd_last_arrival_time": 4, "bwd_iat_min": 5}
  # fwd deeper: first either way
  for fwd_pinned in (True, False):
    placed = register_stage_schedule(_FWD_DEEPER, fwd_pinned=fwd_pinned)
    assert _last(placed, "fwd") <= min(s for n, s in placed.items() if n.startswith("bwd"))


def test_the_replay_reads_the_order_from_the_programs_fwd_pin(tmp_path):
  app, ddos, names = _rng_design(2)
  joint = bps.get_joint_feature_intervals(app, names, ddos, names)
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=joint,
                              feature_intervals_ddos=joint,
                              output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                              selected_features_app=names, selected_features_ddos=names)
  with open(path, encoding="utf-8") as handle:
    text = handle.read()
  pinned = parse_program_text(text)
  unpinned = parse_program_text(text.replace('@pa_no_overlay("ingress", "ig_md.fwd")\n', ""))
  assert pinned.fwd_pinned and not unpinned.fwd_pinned
  assert model_breakdown(unpinned) == model_breakdown(
      dataclasses.replace(pinned, fwd_pinned=False))
