"""The generator's packed per-tree program (spec docs/superpowers/specs/
2026-11-11-packed-codes-stage-pin-design.md Sec 3.1, 4.3, 4.5): one field code_pk_<f> per
feature holding every tree's code at its bit offset (p4model.packed), each tree keying its
own slices of it. No program carries an @stage line."""
import os
import re

import pytest

from p4gen import build_p4_script as bps
from p4gen import evaluation
from p4model import packed
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING, TaskSpec)
from tests.test_per_task_codes import FEATURES, _own, _rng_design


def _specs(app, ddos, names_app=FEATURES, names_ddos=FEATURES, intervals_app=None,
           intervals_ddos=None):
  return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), names_ddos,
                   _own(ddos, names_ddos) if intervals_ddos is None else intervals_ddos),
          TaskSpec(APP_TASK, 3, list(app.estimators_), names_app,
                   _own(app, names_app) if intervals_app is None else intervals_app)]


def _text(tmp_path, seed=3, **kw):
  app, ddos, names = _rng_design(seed)
  return open(bps.generate_p4_program(_specs(app, ddos, names, names), output_dir=str(tmp_path) + os.sep,
                                      encoding=PER_TREE_ENCODING, **kw), encoding="utf-8").read()


def test_one_code_field_per_feature_and_no_per_tree_field(tmp_path):
  text = _text(tmp_path)
  fields = re.findall(r"bit<(\d+)> code_pk_(\w+);", text)
  assert fields and "code_tr_" not in text
  for bits, feature in fields:
    sizes = re.search(r'@pa_container_size\("ingress", "ig_md\.code_pk_%s", ([\d, ]+)\)' % feature, text).group(1)
    assert sum(int(s) for s in sizes.split(", ")) == int(bits)
    assert '@pa_no_overlay("ingress", "ig_md.code_pk_%s")' % feature in text
    assert '@pa_solitary("ingress", "ig_md.code_pk_%s")' % feature not in text


def test_each_range_action_writes_one_value(tmp_path):
  text = _text(tmp_path)
  for bits, feature in re.findall(r"bit<(\d+)> code_pk_(\w+);", text):
    assert re.search(r"action set_code_%s \(bit<%s> code\) \{\s+meta\.code_pk_%s = code;\s+\}" % (feature, bits, feature), text)


def test_tree_keys_are_slices_of_at_most_64_bits_and_trees_do_not_overlap(tmp_path):
  text = _text(tmp_path)
  owner = {}                                     # (feature, bit) -> the tree table keying it
  for table, body in re.findall(r"table (get_classification_tree_\w+) \{(.*?)size = ", text, re.S):
    for feature, hi, lo in re.findall(r"meta\.code_pk_(\w+)\[(\d+):(\d+)\] : ternary;", body):
      assert int(hi) - int(lo) < 64
      for bit in range(int(lo), int(hi) + 1):
        assert owner.setdefault((feature, bit), table) == table    # no bit keyed by two trees
  assert owner


def test_the_per_tree_program_has_no_stage_line(tmp_path):
  assert "@stage(" not in _text(tmp_path)


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING])
def test_the_other_encodings_have_no_stage_line(tmp_path, encoding):
  app, ddos, names = _rng_design(3)
  specs = _specs(app, ddos, names, names)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    specs = _specs(app, ddos, names, names, joint, joint)

  def text(**kw):
    path = bps.generate_p4_program(specs, output_dir=str(tmp_path) + os.sep,
                                   encoding=encoding, **kw)
    with open(path, encoding="utf-8") as handle:
      return handle.read()

  assert "@stage(" not in text()


def test_a_range_table_no_tree_codes_is_refused(tmp_path):
  app, ddos, names = _rng_design(3)
  intervals = dict(_own(app, names))
  extra = next(f for f in FEATURES if f not in intervals)
  intervals[extra] = [(0, 5), (6, 65535)]           # a threshold no app tree splits on
  with pytest.raises(ValueError, match="no tree has a code"):
    bps.generate_p4_program(_specs(app, ddos, names, names, intervals_app=intervals),
                            output_dir=str(tmp_path) + os.sep, encoding=PER_TREE_ENCODING)


@pytest.mark.parametrize("seed", range(6))
def test_the_generator_builds_the_layout_the_training_pool_builds(tmp_path, monkeypatch, seed):
  """Ruling R3: layout_for depends on the feature order and the task order (a tree's lane
  counts carry over from one feature to the next), so the program and the pool price the same
  key only if both call it with the same orders. Spy on every call and compare."""
  real, calls = packed.layout_for, []

  def spy(widths, features, task_order):
    layout = real(widths, features, task_order)
    calls.append((widths, [f for f in features if f in layout.containers], list(task_order),
                  layout))
    return layout

  monkeypatch.setattr(packed, "layout_for", spy)
  app, ddos, names = _rng_design(seed)
  evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  bps.generate_p4_program(_specs(app, ddos, names, names), output_dir=str(tmp_path) + os.sep,
                          encoding=PER_TREE_ENCODING)
  assert len(calls) == 2
  (w_pool, f_pool, t_pool, pool), (w_gen, f_gen, t_gen, gen) = calls
  assert (w_gen, f_gen, t_gen) == (w_pool, f_pool, t_pool)
  assert t_gen == [APP_TASK, DDOS_TASK]
  assert list(gen.containers.items()) == list(pool.containers.items())
  assert gen.place == pool.place
