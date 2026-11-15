"""The control-plane entries of the packed per-tree program (spec docs/superpowers/specs/
2026-11-11-packed-codes-stage-pin-design.md Sec 3.1, 3.2, 8): each range entry writes ONE
value, every tree's code at its bit offset in code_pk_<f> (packed.range_value); each tree
table keys its own slices of that field (packed.key_slices), value and mask cut at the code's
bits.

Asserted on MEANING, like test_per_task_codes' entry tests: the entries are played as the
switch plays them (a range table's action writes the field, every tree table matches
value/mask on each key slice of it) and must give every tree's own class."""
import json
import os
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.switch_semantics import per_tree_class_indices, vote_winner
from p4model import packed
from p4model.errors import ActionDataTooWide, NoSplitTree
from p4model.program import APP_TASK, DDOS_TASK, PER_TREE_ENCODING, TaskSpec
from p4model.target import ACTION_DATA_BUS_BYTES
from tests.test_packed_program import _specs
from tests.test_per_task_codes import FEATURES, _forest, _own, _rng_design

_SLICE = re.compile(r"^code_pk_(\w+?)_(\d+)_(\d+)_$")
DEEP_FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_mean"]


def _deep_forest(n_classes, seed, depth, n_trees=2):
  # the label follows feature 0 in fine steps, so deep trees split it many times: codes wider
  # than 64 bits
  rng = np.random.RandomState(seed)
  X = rng.randint(0, 5000, size=(4000, len(DEEP_FEATURES)))
  y = ((X[:, 0] // 37) + (X[:, 1] > 2500) + (X[:, 2] // 1700)) % n_classes
  return bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=n_trees, max_depth=depth, random_state=seed).fit(X, y))


def _deep_specs(n_app=2):
  """Deep app trees: codes wider than 64 bits (keyed as two slices) next to narrow ones. At
  depth 10 app tree 0's 84-bit code on flow_iat_max sits at byte 9 of its field, keyed as
  [135:72] and [155:136] (depth 9 puts every two-slice code at offset 0)."""
  app, ddos = _deep_forest(3, 0, depth=10, n_trees=n_app), _deep_forest(2, 1, depth=3)
  return _specs(app, ddos, DEEP_FEATURES, DEEP_FEATURES), app, ddos


def _design(case):
  """(tasks, app forest, ddos forest, feature names, input range) of a case: an int is
  test_per_task_codes' random design of that seed, 'deep' the deep design."""
  if case == "deep":
    tasks, app, ddos = _deep_specs()
    return tasks, app, ddos, DEEP_FEATURES, 5100
  app, ddos, names = _rng_design(case)
  return _specs(app, ddos, names, names), app, ddos, names, 2100


def _program(tmp_path, tasks, **kwargs):
  path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                                 encoding=PER_TREE_ENCODING, **kwargs)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


def _program_tables(text):
  """{table: [(bf_rt key name, match kind)]} of a program text, in key order. A sliced key
  carries its name as @name (review B1), which must be the bf_rt name of the expression."""
  tables = {}
  for name, block in re.findall(r"table (\w+) \{\s*key = \{(.*?)\}", text, re.S):
    keys = []
    for expression, kind, annotated in re.findall(
        r'(meta\.[\w\[\]:]+)\s*:\s*(\w+)\s*(?:@name\("(\w+)"\))?\s*;', block):
      assert annotated == ("" if "[" not in expression else bps.bfrt_key_name(expression))
      keys.append((annotated or bps.bfrt_key_name(expression), kind))
    tables[name] = keys
  return tables


def _play(entries, row):
  """{tree table: class} for one input, the entries interpreted as the switch does: a range
  table's matching entry writes its `code` into meta.code_pk_<f>; a tree entry matches when
  every key slice of a field, masked, equals its value."""
  fields, by_table = {}, {}
  for entry in entries:
    by_table.setdefault(entry["table_name"], []).append(entry)
  for table, rows in by_table.items():
    if not table.startswith("table_"):
      continue
    feature = rows[0]["action"][len("set_code_"):]
    hits = [e for e in rows if int(e["key_fields"][feature + "_val"]["start"]) <= row[feature]
            <= int(e["key_fields"][feature + "_val"]["end"])]
    assert len(hits) == 1, (table, row)
    assert list(hits[0]["action_params"]) == ["code"]
    fields[feature] = int(hits[0]["action_params"]["code"])
  classes = {}
  for table, rows in by_table.items():
    if not table.startswith("get_classification_tree_"):
      continue

    def matches(entry):
      for name, spec in entry["key_fields"].items():
        feature, hi, lo = _SLICE.match(name).groups()
        value = (fields[feature] >> int(lo)) & ((1 << (int(hi) - int(lo) + 1)) - 1)
        if value & int(spec["mask"], 16) != int(spec["value"], 16):
          return False
      return True
    hits = [e for e in rows if not e["is_default_action"] and matches(e)]
    assert len(hits) <= 1, (table, row)
    hits = hits or [e for e in rows if e["is_default_action"]]
    assert len(hits) == 1, (table, row)
    classes[table] = int(hits[0]["action_params"]["class"])
  return classes


def _inputs(tasks, names, top, seed):
  """Every interval bound of every task (and one past it), a few edges, and random rows."""
  bounds = sorted({v for task in tasks for iv in (task.intervals or {}).values()
                   for lo, hi in iv for v in (lo, hi, hi + 1) if v <= 65535})
  rows = [[v] * len(names) for v in bounds + [0, 1, top - 1, 65535]]
  rng = np.random.RandomState(seed)
  return np.vstack([np.array(rows), rng.randint(0, top, size=(250, len(names)))])


def _multi_slice_keys_at_an_offset(tasks):
  """[(task, tree, feature)] whose code sits at a non-zero (bit) offset and is keyed as more
  than one slice."""
  table_plan, _, own = bps.resolve_per_tree_plan(tasks)
  layout = bps.per_tree_layout(table_plan, own, tasks)
  return [key for key, (offset, _) in layout.place.items()
          if offset > 0 and len(packed.key_slices(layout, *key)) > 1]


def test_the_deep_design_has_a_multi_slice_key_at_a_non_zero_offset():
  """Without one the meaning test below would not exercise a slice cut away from bit 0."""
  assert _multi_slice_keys_at_an_offset(_deep_specs()[0])


# Case 0 under both discounts reaches every line and branch cases 0-5 did
# (results/encoding_evidence_2026_11/model8/out/tests_parametrize.csv); "deep" adds a
# multi-slice key at a non-zero offset; case 1 six codes of 1-2 bits sharing one byte (decision 4.4).
@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("case", [0, 1, "deep"])
def test_per_tree_entries_classify_every_input_exactly_as_each_tree(case, discount):
  tasks, app, ddos, names, top = _design(case)
  entries = bps.generate_table_entries(tasks, use_default_action_discount=discount,
                                       encoding=PER_TREE_ENCODING)
  X = _inputs(tasks, names, top, 7 if case == "deep" else case)
  trees = {APP_TASK: per_tree_class_indices(app, X), DDOS_TASK: per_tree_class_indices(ddos, X)}
  n_app = len(app.estimators_)
  for s, values in enumerate(X):
    got = _play(entries, dict(zip(names, values)))
    want = {"get_classification_tree_%s_%d" % (t, i): int(c) for t in trees
            for i, c in enumerate(trees[t][:, s])}
    assert got == want, (case, values)
    assert vote_winner([got["get_classification_tree_app_%d" % i] for i in range(n_app)], 3) == \
        vote_winner(list(trees[APP_TASK][:, s]), 3)


@pytest.mark.parametrize("case", [0, 1, 3, "deep"])
def test_each_tree_writes_its_own_thermometer_code_at_its_offset(case):
  tasks = _design(case)[0]
  table_plan, _, own = bps.resolve_per_tree_plan(tasks)
  layout = bps.per_tree_layout(table_plan, own, tasks)
  for entry in bps.generate_table_entries(tasks, encoding=PER_TREE_ENCODING):
    if not entry["table_name"].startswith("table_"):
      continue
    feature = entry["action"][len("set_code_"):]
    x = int(entry["key_fields"][feature + "_val"]["start"])
    value = int(entry["action_params"]["code"])
    assert value < 1 << packed.field_bits(layout, feature)
    expected = 0
    for (task, tree, f), (offset, bits) in layout.place.items():
      if f != feature:
        continue
      iv = own[task][tree][feature]
      idx = next(j for j, (lo, hi) in enumerate(iv) if lo <= x <= hi)
      code = int(bps.thermometer_code(len(iv), idx), 2)
      assert (value >> offset) & ((1 << bits) - 1) == code
      expected |= code << offset
    assert value == expected        # nothing written outside the trees' codes


@pytest.mark.parametrize("case", [0, 3, "deep"])
def test_entry_key_names_are_the_programs_keys_as_bf_rt_names_them(tmp_path, case):
  tasks = _design(case)[0]
  declared = _program_tables(_program(tmp_path, tasks))
  entries = bps.generate_table_entries(tasks, use_default_action_discount=True,
                                       encoding=PER_TREE_ENCODING)
  seen = set()
  for entry in entries:
    if entry["is_default_action"]:
      assert entry["key_fields"] == {}
      continue
    assert list(entry["key_fields"]) == [name for name, _ in declared[entry["table_name"]]], entry
    seen.add(entry["table_name"])
  trees = {t for t in declared if t.startswith("get_classification_tree_")}
  ranges = {t for t in declared if t.startswith("table_")}
  assert trees and ranges and seen == trees | ranges
  assert all(kind == "range" for t in ranges for _, kind in declared[t])
  assert all(kind == "ternary" for t in trees for _, kind in declared[t])
  # every tree key is a slice of a packed field, named as bf_rt names it
  assert all(_SLICE.match(name) for t in trees for name, _ in declared[t])


def test_a_slice_is_named_with_its_field_bounds():
  assert bps.bfrt_key_name("meta.code_pk_flow_iat_max[19:8]") == "code_pk_flow_iat_max_19_8_"
  assert bps.bfrt_key_name("meta.code_pk_bwd_iat_mean[127:64]") == "code_pk_bwd_iat_mean_127_64_"
  assert bps.bfrt_key_name("meta.flow_iat_max_val") == "flow_iat_max_val"


@pytest.mark.parametrize("case", [0, 3, "deep"])
def test_every_range_entry_writes_exactly_the_one_parameter_the_action_declares(tmp_path, case):
  tasks = _design(case)[0]
  text = _program(tmp_path, tasks)
  actions = {name: [(p.split()[1], int(p.split()[0][4:-1])) for p in params.split(",") if p.strip()]
             for name, params in re.findall(r"action (\w+)\s*\(([^)]*)\)", text)}
  entries = bps.generate_table_entries(tasks, use_default_action_discount=True,
                                       encoding=PER_TREE_ENCODING)
  ranges = [e for e in entries if e["table_name"].startswith("table_")]
  assert ranges
  for entry in ranges:
    assert list(entry["action_params"]) == ["code"]
  for entry in entries:
    declared = actions[entry["action"]]
    assert list(entry["action_params"]) == [name for name, _ in declared], entry["action"]
    for (name, width), value in zip(declared, entry["action_params"].values()):
      assert 0 <= int(value) < (1 << width), (entry["action"], name, width, value)


def test_generate_p4_program_writes_the_entries_of_its_encoding(tmp_path):
  tasks = _deep_specs()[0]
  _program(tmp_path, tasks, entries_filename="entries.json", use_default_action_discount=True)
  with open(tmp_path / "entries.json", encoding="utf-8") as handle:
    written = json.load(handle)
  assert written
  assert written == bps.generate_table_entries(tasks, use_default_action_discount=True,
                                               encoding=PER_TREE_ENCODING)
  assert written == bps.generate_per_tree_table_entries(tasks, use_default_action_discount=True)


def test_entries_with_one_task_having_no_trees(tmp_path):
  app = _forest(3, 2)
  tasks = [TaskSpec(DDOS_TASK, 2, [], None, {}),
           TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, _own(app))]
  declared = _program_tables(_program(tmp_path, tasks))
  entries = bps.generate_table_entries(tasks, encoding=PER_TREE_ENCODING)
  assert entries and {e["table_name"] for e in entries} <= set(declared)
  assert not any("ddos" in e["table_name"] for e in entries)
  X = np.random.RandomState(1).randint(0, 2100, size=(40, len(FEATURES)))
  want = per_tree_class_indices(app, X)
  for s, values in enumerate(X):
    got = _play(entries, dict(zip(FEATURES, values)))
    assert got == {"get_classification_tree_app_%d" % i: int(c) for i, c in enumerate(want[:, s])}


def test_the_entries_writer_refuses_what_the_program_writer_refuses():
  """The layout's refusals, raised before any entry is written."""
  X = np.zeros((60, len(FEATURES)), dtype=int)
  flat = bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=2, max_depth=3, random_state=0).fit(X, np.arange(60) % 3))
  with pytest.raises(NoSplitTree) as stump:
    bps.generate_table_entries(_specs(flat, _forest(2, 1)), encoding=PER_TREE_ENCODING)
  assert stump.value.args[1] == "app_0"
  with pytest.raises(ActionDataTooWide) as wide:
    bps.generate_table_entries(_deep_specs(n_app=20)[0], encoding=PER_TREE_ENCODING)
  assert wide.value.args[1] > ACTION_DATA_BUS_BYTES
