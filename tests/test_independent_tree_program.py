"""The generator's disjoint_per_tree program (arm independent-tree; spec
2026-11-17-independent-tree-encoding-design.md): disjoint's range tables and names
(resolve_feature_plan), each writing ONE packed field code_pk_<resolved> holding the
per-tree codes of the trees of its reader tasks, each tree keying its own slices of it
(p4model.packed). The four old encodings' texts are pinned byte for byte by
tests/fixtures/independent_tree_identity.json (hashed before the encoding existed)."""
import hashlib
import json
import os
import re

import pytest

from p4gen import build_p4_script as bps
from p4gen.p4_replay import parse_program
from p4gen.program_checks import check_program
from p4model import phv
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING, TaskSpec)
from tests.test_per_task_codes import FEATURES, _forest, _own, _rng_design, _stair_forest

NEW = "disjoint_per_tree"
OLD_ENCODINGS = (JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING, PER_TREE_ENCODING)
IDENTITY = os.path.join(os.path.dirname(__file__), "fixtures", "independent_tree_identity.json")
# Its four program-text digests were re-frozen 2026-11-17 for the overlay-pin change: each text
# is tests/overlay_pin_rewrite.current_pins of its previous one (24/24); entries and pools unchanged.

APP_NAMES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_total"]
DDOS_NAMES = ["flow_iat_max", "flow_iat_min", "bwd_packet_length_max"]


def _specs(app, ddos, names_app, names_ddos, intervals_app=None, intervals_ddos=None):
  return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_) if ddos is not None else [], names_ddos,
                   (_own(ddos, names_ddos) if ddos is not None else {})
                   if intervals_ddos is None else intervals_ddos),
          TaskSpec(APP_TASK, 3, list(app.estimators_), names_app,
                   _own(app, names_app) if intervals_app is None else intervals_app)]


def old_encoding_text(tmp_path, seed, encoding):
  """The program text of _rng_design(seed) under an old encoding (joint: the joint
  intervals for both tasks; the others: each task's own)."""
  app, ddos, names = _rng_design(seed)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    specs = _specs(app, ddos, names, names, joint, joint)
  else:
    specs = _specs(app, ddos, names, names)
  out = os.path.join(str(tmp_path), "%s_%d" % (encoding, seed)) + os.sep
  with open(bps.generate_p4_program(specs, output_dir=out, encoding=encoding),
            encoding="utf-8") as handle:
    return handle.read()


def _digest(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _overlap_design():
  """Two forests with different feature lists sharing flow_iat_max, both splitting it."""
  app = _forest(3, 4, APP_NAMES)
  ddos = _forest(2, 6, DDOS_NAMES)
  return _specs(app, ddos, APP_NAMES, DDOS_NAMES)


def _merge_design():
  """Both tasks split one feature at identical thresholds: one shared entry `f`."""
  names = FEATURES[:1]
  return _specs(_stair_forest(3, 3), _stair_forest(3, 2), names, names)


def _no_ddos_trees_design():
  return _specs(_forest(3, 4, APP_NAMES), None, APP_NAMES, None)


def _program(tmp_path, specs, name="design"):
  out = os.path.join(str(tmp_path), name) + os.sep
  path = bps.generate_p4_program(specs, output_dir=out, encoding=NEW)
  with open(path, encoding="utf-8") as handle:
    return path, handle.read()


DESIGNS = {"overlap": _overlap_design, "merge": _merge_design, "no_ddos_trees": _no_ddos_trees_design}


def test_overlap_design_really_overlaps():
  ddos, app = _overlap_design()
  shared = set(app.intervals) & set(ddos.intervals)
  assert "flow_iat_max" in shared and set(app.intervals) != set(ddos.intervals)


def test_plan_uses_disjoint_names_and_tables():
  specs = _overlap_design()
  table_plan, field_plan, tree_intervals = bps.resolve_disjoint_per_tree_plan(specs)
  resolved = bps.resolve_feature_plan([(t.name, t.intervals) for t in specs])
  assert list(table_plan) == list(resolved)
  assert list(field_plan) == ["pk_" + name for name in resolved]
  for name, (raw, intervals, readers) in resolved.items():
    union, writers = table_plan[name]
    assert union == intervals
    assert field_plan["pk_" + name] == (raw, intervals, readers)
    # writers: the reader tasks' trees splitting the raw feature, program order then index
    expected = [("%s_%d" % (t.name, i), len(own[raw]) - 1)
                for t in reversed(specs) if t.name in readers
                for i, own in enumerate(bps.per_tree_intervals(t)) if raw in own]
    assert writers == expected and writers
  for task in specs:
    for i, own in enumerate(bps.per_tree_intervals(task)):
      mapped = tree_intervals[task.name][i]
      assert list(mapped.values()) == list(own.values())
      for name in mapped:
        assert task.name in resolved[name][2]
      assert [resolved[name][0] for name in mapped] == list(own)


def test_identical_thresholds_merge_into_one_table_read_by_both_tasks():
  table_plan, field_plan, _ = bps.resolve_disjoint_per_tree_plan(_merge_design())
  feature = FEATURES[0]
  assert list(table_plan) == [feature]
  assert field_plan["pk_" + feature][2] == {APP_TASK, DDOS_TASK}
  writers = [writer for writer, _ in table_plan[feature][1]]
  assert any(w.startswith(APP_TASK + "_") for w in writers)
  assert any(w.startswith(DDOS_TASK + "_") for w in writers)


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_program_passes_check_program_and_pins_every_packed_field(tmp_path, design):
  path, text = _program(tmp_path, DESIGNS[design]())
  check_program(parse_program(path))
  fields = re.findall(r"bit<(\d+)> code_pk_(\w+);", text)
  assert fields
  for bits, name in fields:
    assert '@pa_no_overlay("ingress", "ig_md.code_pk_%s")' % name not in text  # removed 2026-11-17
    sizes = re.search(r'@pa_container_size\("ingress", "ig_md\.code_pk_%s", ([\d, ]+)\)'
                      % name, text).group(1)
    assert sum(int(s) for s in sizes.split(", ")) == int(bits)
    assert re.search(r"action set_code_%s \(bit<%s> code\) \{\s+meta\.code_pk_%s = code;\s+\}"
                     % (name, bits, name), text)
  assert "code_tk_" not in text


def test_range_tables_key_the_raw_value_field(tmp_path):
  specs = _overlap_design()
  _, text = _program(tmp_path, specs)
  resolved = bps.resolve_feature_plan([(t.name, t.intervals) for t in specs])
  assert any(name != raw for name, (raw, _, _) in resolved.items())     # some namespaced entry
  for index, (name, (raw, _, _)) in enumerate(resolved.items()):
    body = re.search(r"table table_%d_%s \{(.*?)\n\t\}" % (index, name), text, re.S).group(1)
    assert "meta.%s_val" % raw in body and "set_code_%s" % name in body


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_val_fields_are_counted_once_per_raw_feature(tmp_path, design):
  specs = DESIGNS[design]()
  path, text = _program(tmp_path, specs)
  raws = {raw for raw, _, _ in bps.resolve_feature_plan(
      [(t.name, t.intervals) for t in specs]).values()}
  program = parse_program(path)
  vals = sum(1 for name in program.bits if name.endswith("_val") and not name.startswith("code_"))
  assert vals == len(raws)
  assert len(re.findall(r"bit<16> \w+_val;", text)) == len(raws)
  # phv.from_program counts them, and agrees with the layout's own budget input
  assert phv.from_program(program)


def test_a_task_without_trees_generates(tmp_path):
  specs = _no_ddos_trees_design()
  table_plan, _, _ = bps.resolve_disjoint_per_tree_plan(specs)
  path, text = _program(tmp_path, specs)
  check_program(parse_program(path))
  for _, name in re.findall(r"bit<(\d+)> code_pk_(\w+);", text):
    assert table_plan[name][1]                     # every declared field has a writer
  assert "get_classification_tree_ddos" not in text


def test_trees_key_only_their_own_task_fields(tmp_path):
  specs = _overlap_design()
  _, text = _program(tmp_path, specs)
  resolved = bps.resolve_feature_plan([(t.name, t.intervals) for t in specs])
  for table, task, body in re.findall(
      r"table (get_classification_tree_(\w+?)_\d+) \{(.*?)size = ", text, re.S):
    names = set(re.findall(r"meta\.code_pk_(\w+)\[\d+:\d+\]", body))
    assert names and all(task in resolved[name][2] for name in names), table


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("encoding", OLD_ENCODINGS)
def test_existing_encodings_unchanged(tmp_path, seed, encoding):
  with open(IDENTITY, encoding="utf-8") as handle:
    pinned = json.load(handle)
  assert _digest(old_encoding_text(tmp_path, seed, encoding)) == pinned[encoding][str(seed)]


def write_identity_fixture(tmp_dir):
  """Hash the four old encodings' texts on seeds 0-5 (run once, before the new
  encoding touched the generator)."""
  pinned = {encoding: {str(seed): _digest(old_encoding_text(tmp_dir, seed, encoding))
                       for seed in range(6)}
            for encoding in OLD_ENCODINGS}
  with open(IDENTITY, "w", encoding="utf-8") as handle:
    json.dump(pinned, handle, indent=1, sort_keys=True)
    handle.write("\n")
