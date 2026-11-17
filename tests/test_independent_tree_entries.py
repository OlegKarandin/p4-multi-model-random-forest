"""The control-plane entries of the disjoint_per_tree program (arm independent-tree; spec
2026-11-17-independent-tree-encoding-design.md): disjoint's range tables, each keyed on its RAW
feature's value and writing ONE packed value (every reading tree's code at its bit offset); each
tree table keys slices of code_pk_<resolved>, its codewords built against raw feature names.

Asserted on MEANING, as tests/test_packed_entries.py does: the entries are played as the switch
plays them and must give every tree's own class. per_tree's entries are pinned by hash
(tests/fixtures/independent_tree_identity.json, taken before this encoding's entries existed)."""
import hashlib
import json
import os
import re

import numpy as np
import pytest

from p4gen import build_p4_script as bps
from p4gen.switch_semantics import per_tree_class_indices
from p4model.program import APP_TASK, DDOS_TASK, DISJOINT_PER_TREE_ENCODING as NEW, PER_TREE_ENCODING
from tests.test_independent_tree_program import (DESIGNS, IDENTITY, _merge_design, _overlap_design,
                                                 _program, _specs)
from tests.test_packed_entries import _program_tables
from tests.test_per_task_codes import _rng_design, switch_equivalent_rows

_SLICE = re.compile(r"^code_pk_(\w+?)_(\d+)_(\d+)_$")


def _union_names(specs):
  return list(dict.fromkeys(n for t in reversed(specs) for n in (t.features or [])))


class _Fitted:
  """The estimators of a TaskSpec, as per_tree_class_indices reads a forest."""
  def __init__(self, trees):
    self.estimators_ = trees


def _play(entries, row):
  """{tree table: class} of one input row {raw feature: value}: a range table's matching entry
  writes `code` into the field of its RESOLVED name; a tree entry matches when every key slice
  of a field, masked, equals its value."""
  fields, by_table = {}, {}
  for entry in entries:
    by_table.setdefault(entry["table_name"], []).append(entry)
  for table, rows in by_table.items():
    if not table.startswith("table_"):
      continue
    resolved = table.split("_", 2)[2]
    assert rows[0]["action"] == "set_code_" + resolved
    (key,) = rows[0]["key_fields"]
    raw = key[:-len("_val")]
    reading = bps.switch_reading(raw, row[raw])
    hits = [e for e in rows if int(e["key_fields"][key]["start"]) <= reading
            <= int(e["key_fields"][key]["end"])]
    assert len(hits) == 1, (table, row)
    assert list(hits[0]["action_params"]) == ["code"]
    fields[resolved] = int(hits[0]["action_params"]["code"])
  classes = {}
  for table, rows in by_table.items():
    if not table.startswith("get_classification_tree_"):
      continue

    def matches(entry):
      for name, spec in entry["key_fields"].items():
        resolved, hi, lo = _SLICE.match(name).groups()
        value = (fields[resolved] >> int(lo)) & ((1 << (int(hi) - int(lo) + 1)) - 1)
        if value & int(spec["mask"], 16) != int(spec["value"], 16):
          return False
      return True
    hits = [e for e in rows if not e["is_default_action"] and matches(e)]
    assert len(hits) <= 1, (table, row)
    hits = hits or [e for e in rows if e["is_default_action"]]
    assert len(hits) == 1, (table, row)
    classes[table] = int(hits[0]["action_params"]["class"])
  return classes


def _inputs(specs, names, seed):
  bounds = sorted({v for t in specs for iv in (t.intervals or {}).values()
                   for lo, hi in iv for v in (lo, hi, hi + 1) if v <= 65535})
  rows = [[v] * len(names) for v in bounds + [0, 1, 2099]]
  rng = np.random.RandomState(seed)
  return np.vstack([np.array(rows), rng.randint(0, 2100, size=(200, len(names)))])


def _check_meaning(specs, discount):
  names = _union_names(specs)
  entries = bps.generate_table_entries(specs, use_default_action_discount=discount, encoding=NEW)
  X = _inputs(specs, names, 3)
  same = switch_equivalent_rows(X, names)
  want = {}
  for task in specs:
    if not task.trees:
      continue
    columns = [names.index(n) for n in task.features]
    classes = per_tree_class_indices(_Fitted(task.trees), same[:, columns])
    for i in range(len(task.trees)):
      want["get_classification_tree_%s_%d" % (task.name, i)] = classes[i]
  assert want
  for s, values in enumerate(X):
    got = _play(entries, dict(zip(names, values)))
    assert got == {k: int(v[s]) for k, v in want.items()}, values


@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_entries_classify_every_input_exactly_as_each_tree(design, discount):
  _check_meaning(DESIGNS[design](), discount)


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_entry_key_names_are_the_programs_keys(tmp_path, design):
  specs = DESIGNS[design]()
  _, text = _program(tmp_path, specs)
  declared = _program_tables(text)
  entries = bps.generate_table_entries(specs, use_default_action_discount=True, encoding=NEW)
  seen = set()
  for entry in entries:
    if entry["is_default_action"]:
      assert entry["key_fields"] == {}
      continue
    assert list(entry["key_fields"]) == [n for n, _ in declared[entry["table_name"]]], entry
    seen.add(entry["table_name"])
  trees = {t for t in declared if t.startswith("get_classification_tree_")}
  ranges = {t for t in declared if t.startswith("table_")}
  assert trees and ranges and seen == trees | ranges


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_every_range_entry_writes_the_one_parameter_the_action_declares(tmp_path, design):
  specs = DESIGNS[design]()
  _, text = _program(tmp_path, specs)
  actions = {name: [(p.split()[1], int(p.split()[0][4:-1])) for p in params.split(",") if p.strip()]
             for name, params in re.findall(r"action (\w+)\s*\(([^)]*)\)", text)}
  entries = bps.generate_table_entries(specs, use_default_action_discount=True, encoding=NEW)
  assert any(e["table_name"].startswith("table_") for e in entries)
  for entry in entries:
    if entry["table_name"].startswith("table_"):
      assert list(entry["action_params"]) == ["code"]
    declared = actions[entry["action"]]
    assert list(entry["action_params"]) == [n for n, _ in declared], entry["action"]
    for (name, width), value in zip(declared, entry["action_params"].values()):
      assert 0 <= int(value) < (1 << width), (entry["action"], name, width, value)


def test_overlap_range_tables_key_the_raw_feature():
  entries = bps.generate_table_entries(_overlap_design(), encoding=NEW)
  keyed = {e["table_name"].split("_", 2)[2]: list(e["key_fields"])[0]
           for e in entries if e["table_name"].startswith("table_")}
  # the tasks split flow_iat_max differently: namespaced tables, one raw value key
  assert {"app_flow_iat_max", "ddos_flow_iat_max"} <= set(keyed)
  assert keyed["app_flow_iat_max"] == keyed["ddos_flow_iat_max"] == "flow_iat_max_val"
  for resolved, key in keyed.items():
    assert key.endswith("_val") and resolved.endswith(key[:-len("_val")])


def test_merged_table_values_carry_both_tasks_codes():
  specs = _merge_design()
  table_plan, _, own, layout = bps.packed_program_plan(specs, NEW)
  (feature,) = table_plan
  assert {task for task, _, f in layout.place if f == feature} == {APP_TASK, DDOS_TASK}
  entries = [e for e in bps.generate_table_entries(specs, encoding=NEW)
             if e["table_name"].startswith("table_")]
  assert entries
  for entry in entries:
    x = int(entry["key_fields"][feature + "_val"]["start"])
    value = int(entry["action_params"]["code"])
    expected = 0
    for (task, i, f), (offset, bits) in layout.place.items():
      iv = own[task][i][f]
      idx = next(j for j, interval in enumerate(iv)
                 for lo, hi in [bps.switch_range(f, interval)] if lo <= x <= hi)
      code = int(bps.thermometer_code(len(iv), idx), 2)
      assert (value >> offset) & ((1 << bits) - 1) == code
      expected |= code << offset
    assert value == expected


def test_a_task_with_no_trees_writes_no_entries_of_its_own():
  specs = DESIGNS["no_ddos_trees"]()
  entries = bps.generate_table_entries(specs, encoding=NEW)
  assert entries and not any("ddos" in e["table_name"] for e in entries)
  _check_meaning(specs, False)


def test_generate_p4_program_writes_the_entries_of_its_encoding(tmp_path):
  specs = _overlap_design()
  out = os.path.join(str(tmp_path), "w") + os.sep
  bps.generate_p4_program(specs, output_dir=out, output_filename="e.p4", encoding=NEW,
                          entries_filename="entries.json", use_default_action_discount=True)
  with open(out + "entries.json", encoding="utf-8") as handle:
    assert json.load(handle) == bps.generate_table_entries(
        specs, use_default_action_discount=True, encoding=NEW)


def _entries_digest(seed, discount):
  app, ddos, names = _rng_design(seed)
  entries = bps.generate_table_entries(_specs(app, ddos, names, names),
                                       use_default_action_discount=discount,
                                       encoding=PER_TREE_ENCODING)
  return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("discount", [False, True])
def test_per_tree_entries_unchanged(seed, discount):
  with open(IDENTITY, encoding="utf-8") as handle:
    pinned = json.load(handle)
  key = "per_tree_entries_discount" if discount else "per_tree_entries"
  assert _entries_digest(seed, discount) == pinned[key][str(seed)]
