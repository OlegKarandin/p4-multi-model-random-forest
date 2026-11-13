"""The generator's packed per-tree program (spec docs/superpowers/specs/
2026-11-11-packed-codes-stage-pin-design.md Sec 3.1, 4.3, 4.5): one field code_pk_<f> per
feature holding every tree's code at its byte offset (p4model.packed), each tree keying its
own slices of it, and @stage(n) on the tree tables the model moved."""
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
    assert '@pa_no_overlay("ingress", "ig_md.code_pk_%s")' % feature not in text
    assert '@pa_solitary("ingress", "ig_md.code_pk_%s")' % feature not in text


def test_each_range_action_writes_one_value(tmp_path):
  text = _text(tmp_path)
  for bits, feature in re.findall(r"bit<(\d+)> code_pk_(\w+);", text):
    assert re.search(r"action set_code_%s \(bit<%s> code\) \{\s+meta\.code_pk_%s = code;\s+\}" % (feature, bits, feature), text)


def test_tree_keys_are_slices_of_at_most_64_bits_and_trees_do_not_overlap(tmp_path):
  text = _text(tmp_path)
  owner = {}                                     # (feature, byte) -> the tree table keying it
  for table, body in re.findall(r"table (get_classification_tree_\w+) \{(.*?)size = ", text, re.S):
    for feature, hi, lo in re.findall(r"meta\.code_pk_(\w+)\[(\d+):(\d+)\] : ternary;", body):
      assert int(hi) - int(lo) < 64
      for byte in range(int(lo) // 8, int(hi) // 8 + 1):
        assert owner.setdefault((feature, byte), table) == table   # no byte keyed by two trees
  assert owner


def test_stage_pins_are_written_above_the_named_tables_only(tmp_path):
  text = _text(tmp_path, stage_pins={"get_classification_tree_app_0": 9})
  assert re.search(r"@stage\(9\)\s*\n\s*@placement_priority\(\d+\)\s*\n\s*table get_classification_tree_app_0 \{", text)
  assert text.count("@stage(") == 1
  with pytest.raises(ValueError):
    _text(tmp_path, stage_pins={"no_such_table": 3})


def test_no_pins_and_empty_pins_write_the_same_per_tree_program(tmp_path):
  assert _text(tmp_path) == _text(tmp_path, stage_pins={})
  assert "@stage(" not in _text(tmp_path)


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING])
def test_the_other_encodings_take_no_stage_pins_and_write_none(tmp_path, encoding):
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

  assert text() == text(stage_pins={})
  assert "@stage(" not in text()
  with pytest.raises(ValueError):
    text(stage_pins={"get_classification_tree_app_0": 4})


def test_generate_P4_code_passes_the_pins_through(tmp_path):
  app, ddos, names = _rng_design(3)
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=_own(app, names),
                              feature_intervals_ddos=_own(ddos, names),
                              output_dir=str(tmp_path) + os.sep, output_filename="design.p4",
                              selected_features_app=names, selected_features_ddos=names,
                              encoding=PER_TREE_ENCODING,
                              stage_pins={"get_classification_tree_ddos_0": 7})
  with open(path, encoding="utf-8") as handle:
    text = handle.read()
  assert re.search(r"@stage\(7\)\s*\n\s*@placement_priority\(\d+\)\s*\n\s*"
                   r"table get_classification_tree_ddos_0 \{", text)


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
