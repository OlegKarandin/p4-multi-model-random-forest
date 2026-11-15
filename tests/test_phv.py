"""PHV containers counted the same way in training and in the replay of a program text,
and refused over the ingress budget in every encoding (design review decision 8.3,
inventory V5)."""
import os
import re

import pytest

from p4gen import build_p4_script as bps
from p4gen import evaluation
from p4gen.p4_replay import model_breakdown, parse_program, parse_program_text, price_program
from p4model import phv
from p4model.errors import PhvContainersExhausted
from p4model.program import (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)
from p4model.target import PHV_INGRESS_CONTAINERS, PHV_TEMPLATE_FIELDS
from tests.test_per_task_codes import FEATURES, _own, _rng_design

ENCODINGS = [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING, PER_TREE_ENCODING]


def _generate(tmp_path, encoding, seed):
  app, ddos, names = _rng_design(seed)
  if encoding == JOINT_ENCODING:
    ia = idd = bps.get_joint_feature_intervals(app, names, ddos, names)
  else:
    ia, idd = _own(app, names), _own(ddos, names)
  path = bps.generate_P4_code(
      3, 2, app, ddos, feature_intervals_app=ia, feature_intervals_ddos=idd,
      output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
      selected_features_app=names, selected_features_ddos=names, encoding=encoding)
  return app, ddos, names, path


def test_count_adds_the_template_the_own_fields_and_the_code_containers():
  used = phv.count(n_vals=3, n_scratch=2, n_trees=5, code_containers=[(32, 8), (16,), (8,)])
  assert used == {8: len(PHV_TEMPLATE_FIELDS[8]) + 5 + 2,
                  16: len(PHV_TEMPLATE_FIELDS[16]) + 3 + 2 + 1,
                  32: len(PHV_TEMPLATE_FIELDS[32]) + 1}


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_the_coarse_clock_takes_two_32_bit_containers_on_both_paths(tmp_path, encoding):
  """Task 38: a program with a duration computes meta.now_coarse (one W, pinned
  pa_solitary) from ingress_mac_tstamp[47:16] (another W): +2 32-bit containers in
  training and in the replay; a program without one has neither."""
  assert phv.count(1, 0, 0, (), coarse_clock=True)[32] == phv.count(1, 0, 0, ())[32] + 2
  for seed, duration in ((0, True), (3, False)):     # seed 0 has bwd_iat_total, seed 3 none
    app, ddos, names, path = _generate(tmp_path, encoding, seed)
    program = parse_program(path)
    assert ("now_coarse" in program.declared) == duration == bps.uses_coarse_clock(names)
    pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
    assert pool["phv_fields"]["coarse_clock"] == duration
    assert phv.containers_used(pool) == phv.containers_used(program)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_the_fine_clock_takes_two_8_and_four_16_bit_containers_where_it_is_computed(tmp_path, encoding):
  """Task 39 (Task 38 review minor 4): calc_timestamp writes meta.now_pseudo_us only
  when an IAT Max / Min keeps a last-arrival register, and p4c then holds it and its
  global_tstamp[41:10] source in 2 B + 4 H (105 / 105 recompiles). Counted only then, in
  training and in the replay alike; the template declares the field in every program,
  so the replay reads calc_timestamp's writes, not the declaration."""
  base = phv.count(1, 0, 0, ())
  fine = phv.count(1, 0, 0, (), fine_clock=True)
  assert (fine[8] - base[8], fine[16] - base[16], fine[32] - base[32]) == (2, 4, 0)
  assert phv.FINE_CLOCK_CONTAINERS == (8, 8, 16, 16, 16, 16)
  seen = set()
  for seed in (0, 3, 14, 1, 2, 4, 5):
    app, ddos, names, path = _generate(tmp_path, encoding, seed)
    program = parse_program(path)
    computed = any(f == "now_pseudo_us" for f, _ in program.timestamp_writes)
    assert computed == bps.uses_timestamp(names)
    pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
    assert pool["phv_fields"]["fine_clock"] == computed
    assert phv.containers_used(pool) == phv.containers_used(program)
    seen.add(computed)
  assert seen == {True, False}, "the seeds must cover a program with and one without the fine clock"


def test_check_names_the_size_and_the_shortfall():
  used = {8: 10, 16: PHV_INGRESS_CONTAINERS[16] + 3, 32: PHV_INGRESS_CONTAINERS[32] + 1}
  with pytest.raises(PhvContainersExhausted) as refused:
    phv.check(used)
  assert refused.value.args[1] == 4
  assert "16-bit" in refused.value.args[0] and "32-bit" in refused.value.args[0]
  phv.check(dict(PHV_INGRESS_CONTAINERS))             # exactly the budget fits


def test_phv_count_joint(tmp_path):
  """One small joint program, counted by hand from its text."""
  app, ddos, names, path = _generate(tmp_path, JOINT_ENCODING, 3)
  text = open(path, encoding="utf-8").read()
  code = re.findall(r"bit<(\d+)>\s+code_(\w+);", text)
  vals = re.findall(r"bit<16>\s+\w+_val;", text)
  scratch = re.findall(r"bit<16>\s+\w+_current_iat;", text)
  classes = re.findall(r"bit<\d+>\s+class_tree_\w+;", text)
  assert vals and code
  by_hand = {size: len(PHV_TEMPLATE_FIELDS[size]) for size in (8, 16, 32)}
  # C1f's Min read `+ 1`: one 16-bit container per constant (Task 33)
  constants = re.findall(r"_action\.execute\(meta\.flow_hash\) \+ 1;", text)
  by_hand[16] += len(vals) + len(scratch) + len(constants)
  by_hand[8] += len(classes)
  # the fine clock, 2 B + 4 H, when calc_timestamp writes it (Task 39); the coarse one, 2 W
  if re.search(r"meta\.now_pseudo_us = ig_prsr_md\.global_tstamp", text):
    by_hand[8] += 2
    by_hand[16] += 4
  if re.search(r"meta\.now_coarse = ", text):
    by_hand[32] += 2
  for bits, name in code:                              # each field sits in its pinned containers
    pin = re.search(r'@pa_container_size\("ingress", "ig_md\.code_%s", ([\d, ]+)\)' % name, text)
    assert int(bits) == sum(int(size) for size in pin.group(1).split(","))
    for size in pin.group(1).split(","):
      by_hand[int(size)] += 1
  program = parse_program(path)
  assert phv.containers_used(program) == by_hand
  pool = evaluation._pool_inputs(app, ddos, names, names, JOINT_ENCODING)
  assert phv.containers_used(pool) == by_hand
  # one class field per tree of a task with more than one tree (a one-tree task has
  # none since review A2)
  assert len(classes) == phv.class_fields([len(app.estimators_), len(ddos.estimators_)])
  assert phv.class_fields([1, 1]) == 0 and phv.class_fields([1, 3]) == 3


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("seed", [0, 3, 14])
def test_training_and_replay_count_the_same_containers(tmp_path, encoding, seed):
  app, ddos, names, path = _generate(tmp_path, encoding, seed)
  pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
  assert phv.containers_used(pool) == phv.containers_used(parse_program(path))


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_phv_refusal_every_encoding(tmp_path, monkeypatch, encoding):
  """A design over the container budget is refused by training and by the replay
  of its program, whatever the encoding."""
  app, ddos, names, path = _generate(tmp_path, encoding, 3)
  used = phv.containers_used(parse_program(path))
  phv.check(used)                                      # fits as it is
  tight = dict(PHV_INGRESS_CONTAINERS)
  tight[16] = used[16] - 1                             # one 16-bit container short
  monkeypatch.setattr(phv, "PHV_INGRESS_CONTAINERS", tight)
  breakdown = model_breakdown(parse_program(path))
  assert breakdown["refusal"] == "phv_containers_short"
  assert breakdown["stage_depth"] is None and breakdown["blocks"] is None
  with pytest.raises(PhvContainersExhausted) as replayed:
    price_program(parse_program(path))
  with pytest.raises(PhvContainersExhausted) as trained:
    evaluation._pool_inputs(app, ddos, names, names, encoding)
  assert replayed.value.args[1] >= 1 and trained.value.args[1] >= 1
  assert replayed.value.magnitude == replayed.value.args[1]
  assert (replayed.value.constraint, trained.value.constraint) == (
      "phv_containers_short", "phv_containers_short")
  assert "16-bit" in replayed.value.args[0]


def test_the_scratch_fields_are_counted_per_chain():
  program = parse_program_text(_text_with_scratch())
  assert program.scratch_fields == ("flow_current_iat", "fwd_current_iat")


def test_each_min_read_constant_takes_a_16_bit_container():
  """C1f (Task 34): `<f>_val = <f>_action.execute(...) + 1` -- p4c keeps the 1 in a
  16-bit PHV container (Task 33: +1 container per Min feature), in training and in the
  replay alike."""
  assert phv.count(1, 0, 0, (), n_constants=2)[16] == phv.count(1, 0, 0, ())[16] + 2
  features = {"flow_iat_min": None, "fwd_iat_min": None, "bwd_iat_max": None,
              "flow_iat_max": None}
  assert bps.call_site_constants(features) == ["flow_iat_min", "fwd_iat_min"]
  assert bps.call_site_constants({"flow_iat_max": None, "min_packet_length": None}) == []
  _, _, apply_code, _ = bps.generate_P4_registers_and_apply(features)
  program = parse_program_text("struct metadata_t {\n\tbit<16> x_val;\n}\n" + apply_code)
  assert program.call_constants == ("flow_iat_min_val", "fwd_iat_min_val")
  assert parse_program_text(_text_with_scratch()).call_constants == ()


def _text_with_scratch():
  return ("struct metadata_t {\n\tbit<16> flow_current_iat;\n\tbit<16> fwd_current_iat;\n"
          "\tbit<2> class_tree_app_0;\n\tbit<16> x_val;\n}\n")
