"""The range pool's action-data bus budget (2026-11-09, report
results/encoding_evidence_2026_11/model3/report.md, I1a) and the one-block register
schedule (decisions 1a, 7.1 option B, 2026-11-14)."""
import inspect
import os

import pytest

from p4gen import build_p4_script as bps
from p4gen import evaluation as ev
from p4gen.p4_replay import (model_breakdown, parse_program_text, price_program,
                             range_action_bytes, replay_plans)
from p4model.errors import ProgramShapeRefused
from p4model.packing import crossbar_stages_needed
from p4model.program import (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)
from p4model.registers import register_stage_schedule
from p4model import target
from tests.test_per_task_codes import _own, _rng_design


def test_the_bus_is_eight_words_of_sixteen_bytes():
  assert target.ACTION_DATA_BUS_WORDS == 8 and not hasattr(target, "ACTION_DATA_BUS_BUDGET")


@pytest.mark.parametrize("bytes_per_table, stages", [
    ([64, 64], 1),          # 4 + 4 words
    ([48, 48, 32], 1),      # 3 + 3 + 2
    ([48, 48, 48], 2),      # 9 words
    ([63, 62, 53], 2),      # 4 + 4 + 4: the pilot's M075_s03_k03
    ([1] * 8, 1),           # eight small tables: one word each
])
def test_range_tables_share_a_stage_up_to_eight_words(bytes_per_table, stages):
  specs = [(1, 2)] * len(bytes_per_table)
  plan = crossbar_stages_needed(specs, readiness_levels=[5] * len(specs),
                                action_bytes=bytes_per_table)
  assert plan.occupied == stages


def _range_pool(action_bytes, level=4):
  n = len(action_bytes)
  return crossbar_stages_needed([(1, 2)] * n, readiness_levels=[level] * n,
                                key_fields=[frozenset({(("val", i), 2)}) for i in range(n)],
                                action_bytes=action_bytes)


def test_a_table_alone_always_takes_an_empty_stage():
  # its own limit is packed.layout_for's (a field over the bus is refused), not the stage budget's
  assert _range_pool([128]).table_stages == (4,)
  assert _range_pool([128, 128]).table_stages == (4, 5)


def test_the_budget_is_for_the_range_pool_only():
  with pytest.raises(ValueError, match="range pool"):
    crossbar_stages_needed([(1, 2)], readiness_levels=[4],
                           key_fields=[frozenset({("code", 2)})],
                           key_field_bits=[(16,)], action_bytes=[8])
  with pytest.raises(ValueError, match="aligned"):
    crossbar_stages_needed([(1, 2)] * 2, readiness_levels=[4, 4], action_bytes=[8])


def test_the_replay_counts_the_container_bytes_each_range_table_writes(tmp_path):
  # S14 / decision 4.3: the bus carries whole containers, so a range action costs
  # the containers of every field it writes, not its codes' bits pooled.
  from p4model import lanes
  app, ddos, names = _rng_design(3)
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=_own(app, names),
                              feature_intervals_ddos=_own(ddos, names),
                              output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                              selected_features_app=names, selected_features_ddos=names,
                              encoding=PER_TASK_ENCODING)
  with open(path, encoding="utf-8") as handle:
    program = parse_program_text(handle.read())
  written = range_action_bytes(program)
  for name, action_bytes in written.items():
    feature = name.split("_", 2)[2]
    fields = [field for field in ("code_tk_app_" + feature, "code_tk_ddos_" + feature)
              if field in program.bits]
    assert fields and program.action_writes["set_code_" + feature] == fields
    assert action_bytes == sum(sum(program.code_containers[field]) // 8 for field in fields)
    assert action_bytes == sum(lanes.container_bytes(program.bits[field]) for field in fields)


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
  written = range_action_bytes(program)
  ranges = [name for name, keys in program.tables.items() if name.startswith("table_") and keys]
  assert sorted(pool["range_action_bytes"]) == sorted(written[name] for name in ranges)


# --- the register schedule: one block (decisions 1a, 7.1 option B) ----------------------

def test_schedule_single_block():
  from p4model import registers
  from p4model.program import ORIENTATION_REGISTER
  import p4model.program as program
  # The block order, cursor floor and gated interiors are gone.
  assert not hasattr(registers, "gated_block_interior_stages")
  assert not hasattr(program, "REGISTER_BLOCK_ORDER")
  for function in (registers.register_stage_schedule, registers.readiness_levels_for):
    assert "fwd_pinned" not in inspect.signature(function).parameters
  # fwd and bwd chains share stages: no bwd register waits for the fwd ones.
  assert register_stage_schedule(["fwd_iat_max", "bwd_iat_max"]) == {
      ORIENTATION_REGISTER: 3, "fwd_last_arrival_time": 4, "bwd_last_arrival_time": 4,
      "fwd_iat_max": 5, "bwd_iat_max": 5}
  # Four stateful ALUs per stage; ready registers ranked longer tail chain first, then
  # more dependent trees, then later in program order (S5).
  features = ["min_packet_length", "packet_length_mean", "max_packet_length", "flow_iat_max",
              "fwd_packet_length_max"]
  assert register_stage_schedule(features) == {
      ORIENTATION_REGISTER: 3, "flow_last_arrival_time": 3, "max_packet_length": 3,
      "packet_length_mean": 3, "min_packet_length": 4, "flow_iat_max": 4,
      "fwd_packet_length_max": 4}
  favoured = register_stage_schedule(features, dependents={"min_packet_length": 5})
  assert favoured["min_packet_length"] == 3 and favoured["packet_length_mean"] == 4


def test_a_direction_owned_register_still_waits_for_the_orientation_register():
  # Its body reads meta.fwd. Four ungated two-register chains with more dependents
  # than the orientation register push it to the second register stage; the
  # fwd-owned register follows it, one stage later.
  from p4model.program import ORIENTATION_REGISTER
  catalog = {"u%d" % i: {"registers": [{"name": "u%d_dep" % i}, {"name": "u%d_val" % i}],
                         "gated_by": None} for i in range(4)}
  catalog["g"] = {"registers": [{"name": "g_val"}], "gated_by": "fwd"}
  features = ["g", "u0", "u1", "u2", "u3"]
  placed = register_stage_schedule(features, catalog=catalog,
                                   dependents={"u%d" % i: 9 for i in range(4)})
  assert placed[ORIENTATION_REGISTER] == 4
  assert placed["g_val"] == 5
  for features in (["fwd_iat_max", "bwd_packet_length_min", "flow_iat_min"],
                   ["bwd_iat_mean", "fwd_packet_length_mean", "packet_length_mean"]):
    placed = register_stage_schedule(features)
    for feature in features:
      entry = ev.FEATURE_REGISTER_CATALOG[feature]
      if entry["gated_by"] is not None:
        first = entry["registers"][0]["name"]
        assert placed[first] > placed[ORIENTATION_REGISTER], (features, first)


def test_ties_follow_call_site_order_not_feature_order():
  # Controller ruling 2026-11-14 (S5): p4c's "default choice" is the later CALL SITE. Here
  # the feature order lists bwd_packet_length_max first, but its call site follows every
  # fwd-owned one (ungated, fwd, bwd). Five value registers tie at stage 4 for four ALUs:
  # the earliest call site, fwd_packet_length_max, waits; by feature order it would have
  # been bwd_packet_length_max.
  from p4model import registers
  features = ["bwd_packet_length_max", "fwd_packet_length_max", "fwd_packet_length_min",
              "fwd_packet_length_mean", "bwd_packet_length_min"]
  assert registers.register_call_order(features) == [
      "flow_forward_srcaddr", "fwd_packet_length_max", "fwd_packet_length_min",
      "fwd_packet_length_mean", "bwd_packet_length_max", "bwd_packet_length_min"]
  placed = register_stage_schedule(features)
  assert placed["fwd_packet_length_max"] == 5
  assert all(placed[name] == 4 for name in features if name != "fwd_packet_length_max")
  # An explicit call order (the replay's, read from the text) decides instead.
  reversed_calls = ["flow_forward_srcaddr"] + list(reversed(features))
  assert register_stage_schedule(features, call_order=reversed_calls)["bwd_packet_length_min"] == 5


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING,
                                      PER_TREE_ENCODING])
def test_training_and_replay_read_the_same_call_site_order(tmp_path, encoding):
  # Training ranks by the generator's emission order (register_call_order over the
  # features); the replay reads the call sites from the text. They must agree.
  from p4model.registers import register_call_order
  for seed in range(6):
    app, ddos, names = _rng_design(seed)
    if encoding == JOINT_ENCODING:
      joint = bps.get_joint_feature_intervals(app, names, ddos, names)
      intervals = (joint, joint)
    else:
      intervals = (_own(app, names), _own(ddos, names))
    path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=intervals[0],
                                feature_intervals_ddos=intervals[1],
                                output_dir=str(tmp_path) + os.sep, output_filename="c.p4",
                                selected_features_app=names, selected_features_ddos=names,
                                encoding=encoding)
    with open(path, encoding="utf-8") as handle:
      program = parse_program_text(handle.read())
    features = []
    for name, keys in program.tables.items():
      if name.startswith("table_") and keys and keys[0][:-len("_val")] not in features:
        features.append(keys[0][:-len("_val")])
    pool = ev._pool_inputs(app, ddos, names, names, encoding)
    assert list(program.register_calls) == register_call_order(features), (encoding, seed)
    assert list(program.register_calls) == register_call_order(pool["emitted_features"]), (
        encoding, seed)


def test_the_model_ignores_the_programs_fwd_pin(tmp_path):
  app, ddos, names = _rng_design(2)
  joint = bps.get_joint_feature_intervals(app, names, ddos, names)
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=joint,
                              feature_intervals_ddos=joint,
                              output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                              selected_features_app=names, selected_features_ddos=names)
  with open(path, encoding="utf-8") as handle:
    text = handle.read()
  # The generator no longer writes the pin (gate G3, 3.1: free on 160/160 p4c
  # compiles). Adding it back changes no price, but the shape check (decision
  # 8.2) refuses the text: it is an older generator's program.
  pin = '@pa_no_overlay("ingress", "ig_md.fwd")\n'
  assert pin not in text
  assert not hasattr(parse_program_text(text), "fwd_pinned")
  first = text.index("@pa_")
  pinned = text[:first] + pin + text[first:]
  assert price_program(parse_program_text(pinned)) == price_program(parse_program_text(text))
  assert model_breakdown(parse_program_text(pinned))["refusal_reason"] == (
      "unexpected_shape:pin:pa_no_overlay:fwd")


def test_replay_refuses_a_gated_register_block():
  # A program generated before decision 1a wrapped direction-owned registers'
  # .execute() calls in `if (meta.fwd == 1) {...}` / `if (meta.fwd == 0) {...}`;
  # the model no longer prices that placement.
  text = ("control SwitchIngress() {\n\tapply {\n"
          "\t\t\tmeta.fwd = flow_orientation_action.execute(meta.flow_hash);\n\n"
          "\t\t\tif (meta.fwd == %d) {\n"
          "\t\t\t\tmeta.current_iat = x_action.execute(meta.flow_hash);\n"
          "\t\t\t}\n\t}\n}\n")
  for owner in (0, 1):
    program = parse_program_text(text % owner)
    assert program.gated_register_blocks
    breakdown = model_breakdown(program)
    assert (breakdown["refusal"], breakdown["refusal_reason"]) == (
        "program_shape", "gated_register_block")
    assert breakdown["stage_depth"] is None and breakdown["blocks"] is None
    # replay_plans refuses it too, for a caller that skips check_program
    with pytest.raises(ProgramShapeRefused, match="gated_register_block"):
      replay_plans("", program, {})
  # a direction test inside a RegisterAction body is not a gated block
  body = ("\tRegisterAction<bit<16>, bit<32>, bit<16>>(x_reg) x_action = {\n"
          "\t\tvoid apply(inout bit<16> value, out bit<16> rv) {\n"
          "\t\t\tif (meta.fwd == 1) {\n"
          "\t\t\t\tvalue = x_halve_unit.execute(value + meta.current_iat);\n"
          "\t\t\t}\n\t\t\trv = value;\n\t\t}\n\t};\n")
  assert not parse_program_text(body).gated_register_blocks
