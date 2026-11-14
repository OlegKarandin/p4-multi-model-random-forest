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
  assert vals and classes and code
  by_hand = {size: len(PHV_TEMPLATE_FIELDS[size]) for size in (8, 16, 32)}
  by_hand[16] += len(vals) + len(scratch)
  by_hand[8] += len(classes)
  for bits, name in code:                              # each field sits in its pinned containers
    pin = re.search(r'@pa_container_size\("ingress", "ig_md\.code_%s", ([\d, ]+)\)' % name, text)
    assert int(bits) == sum(int(size) for size in pin.group(1).split(","))
    for size in pin.group(1).split(","):
      by_hand[int(size)] += 1
  program = parse_program(path)
  assert phv.containers_used(program) == by_hand
  pool = evaluation._pool_inputs(app, ddos, names, names, JOINT_ENCODING)
  assert phv.containers_used(pool) == by_hand
  assert len(classes) == len(app.estimators_) + len(ddos.estimators_)


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


def _text_with_scratch():
  return ("struct metadata_t {\n\tbit<16> flow_current_iat;\n\tbit<16> fwd_current_iat;\n"
          "\tbit<2> class_tree_app_0;\n\tbit<16> x_val;\n}\n")
