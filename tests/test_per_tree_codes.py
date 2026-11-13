"""Per-tree codes (program.PER_TREE_ENCODING) after the packed-codes change (spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md): the encoding's name and
plan, the generator's one `encoding` argument, the program and replay facts the packed
program shares with the per-task one, the arm joint-tree and its training path (the row
writer pins the moved tables, Sec 4.5).

The packed layout, program, entries, pool and replay have their own files
(tests/test_packed_layout.py, test_packed_program.py, test_packed_entries.py,
test_packed_pool.py, test_packed_placement.py, test_packed_replay.py); nothing here repeats
them. The old per-tree rules (the 44-field cap and its spill, the 112-byte action budget,
the Spilled width marker, phv_demand, rule A's padded declarations) are deleted (Sec 7)."""
import dataclasses
import hashlib
import json
import os
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.p4_replay import model_breakdown, parse_program_text
from p4model import lanes, packed
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, ENCODINGS, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING, TaskSpec,
                             feature_of_per_tree_field)
from p4model.target import ACTION_DATA_BUS_BYTES
from tests.test_per_task_codes import FEATURES, _forest, _own, _rng_design, _three_tasks

DEEP_FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_mean"]


def _deep_forest(n_classes, seed, depth, n_trees=2):
  # the label follows feature 0 in fine steps, so deep trees split it many times: codes wider
  # than 64 bits
  rng = np.random.RandomState(seed)
  X = rng.randint(0, 5000, size=(4000, len(DEEP_FEATURES)))
  y = ((X[:, 0] // 37) + (X[:, 1] > 2500) + (X[:, 2] // 1700)) % n_classes
  return bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=n_trees, max_depth=depth, random_state=seed).fit(X, y))


def test_the_encoding_is_named_and_listed():
  assert PER_TREE_ENCODING == "per_tree" and PER_TREE_ENCODING in ENCODINGS


def test_an_archived_per_tree_field_name_still_maps_to_its_feature():
  """Programs written before 2026-11-11 named one code per tree, code_tr_<task>_<i>_<f>;
  the replay still maps such a key to its range table (p4_replay.code_written_by)."""
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app_0") == "f"
  assert feature_of_per_tree_field("code_tk_app_f", "app") is None
  assert feature_of_per_tree_field("code_tr_ddos_2_f", "app") is None
  assert feature_of_per_tree_field("code_pk_f", "app") is None


def test_widths_are_interval_counts_minus_one_and_skip_featureless_entries():
  ivs = {"app": [{"f": [0, 5, 9], "g": [0]}], "ddos": [{"f": list(range(10))}]}
  assert packed.widths_of(ivs) == {"app": [{"f": 2}], "ddos": [{"f": 9}]}


def test_bare_widths_price_as_before():
  # the lane price of a plain key, now that the Spilled branch is gone
  for key in [(3,), (3, 3, 20), (1, 2, 3, 40, 52), (27, 52), (9, 9, 9, 9)]:
    assert lanes.table_blocks(key) == lanes.table_blocks(list(reversed(key)))
  assert lanes.table_blocks([]) == lanes.table_blocks(())
  assert lanes.table_blocks((3,) * 4) == 1


# --- the generator's per-tree program ---------------------------------------------

def _specs(app, ddos, names_app=FEATURES, names_ddos=FEATURES, intervals_app=None,
           intervals_ddos=None):
  return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), names_ddos,
                   _own(ddos, names_ddos) if intervals_ddos is None else intervals_ddos),
          TaskSpec(APP_TASK, 3, list(app.estimators_), names_app,
                   _own(app, names_app) if intervals_app is None else intervals_app)]


def _deep_specs(n_app=2):
  """Deep app trees on one feature: codes wider than 64 bits and narrow ones."""
  app, ddos = _deep_forest(3, 0, depth=9, n_trees=n_app), _deep_forest(2, 1, depth=3)
  return _specs(app, ddos, DEEP_FEATURES, DEEP_FEATURES)


def _program(tmp_path, tasks, name="tree.p4", encoding=PER_TREE_ENCODING, **kwargs):
  path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename=name,
                                 encoding=encoding, **kwargs)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


def test_plan_writes_one_code_per_tree_with_a_threshold_and_the_table_holds_the_union():
  app, ddos = _forest(3, 2), _forest(2, 1)
  tasks = _specs(app, ddos)
  tables, fields, trees = bps.resolve_per_tree_plan(tasks)
  for task in tasks:
    # one dict per tree: the generator's interval rule on that tree alone, features it never
    # splits left out
    assert trees[task.name] == bps.per_tree_intervals(task)
    assert isinstance(trees[task.name], list) and len(trees[task.name]) == len(task.trees)
    for i, tree in enumerate(task.trees):
      own = bps.get_feature_intervals(bps._Forest([tree]), task.features)
      assert trees[task.name][i] == {f: iv for f, iv in own.items() if len(iv) >= 2}
  for feature, (_, written) in tables.items():
    # program order: app's trees, then ddos', each tree index ascending; (writer, code bits)
    assert written == [("%s_%d" % (t, i), len(trees[t][i][feature]) - 1)
                       for t in (APP_TASK, DDOS_TASK) for i in range(len(trees[t]))
                       if feature in trees[t][i]]
  # one packed field per feature, in the tables' order, read by the tasks with a code on it
  assert list(fields) == [bps.packed_field(f) for f in tables]
  for name, (feature, union, readers) in fields.items():
    assert name == "pk_" + feature and union == tables[feature][0]
    assert readers == {t for t in (APP_TASK, DDOS_TASK)
                       if any(feature in own for own in trees[t])}


def test_plan_union_equals_the_per_task_plans():
  app, ddos = _forest(3, 2), _forest(2, 1)
  tasks = _specs(app, ddos)
  tables, _, _ = bps.resolve_per_tree_plan(tasks)
  per_task, _ = bps.resolve_per_task_plan([(t.name, t.intervals) for t in tasks])
  assert list(tables) == list(per_task)
  assert ({f: union for f, (union, _) in tables.items()}
          == {f: union for f, (union, _) in per_task.items()})
  for feature, (_, written) in per_task.items():
    assert [t for t, _, _ in written] == list(dict.fromkeys(
        w.rsplit("_", 1)[0] for w, _ in tables[feature][1]))


def test_a_task_with_no_trees_generates(tmp_path):
  app = _forest(3, 2)
  tasks = [TaskSpec(DDOS_TASK, 2, [], None, {}),
           TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, _own(app))]
  text = _program(tmp_path, tasks)
  assert "get_classification_tree_app_0" in text and "code_pk_" in text
  assert "get_classification_tree_ddos" not in text and "vote_ddos" not in text


def test_a_task_with_trees_and_no_feature_list_raises(tmp_path):
  app, ddos = _forest(3, 2), _forest(2, 1)
  tasks = [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), None, _own(ddos)),
           TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, _own(app))]
  with pytest.raises(ValueError, match="features"):
    _program(tmp_path, tasks)


def test_each_tree_keys_only_its_own_codes_in_feature_order(tmp_path):
  app, ddos = _forest(3, 2), _forest(2, 1)
  tasks = _specs(app, ddos)
  _, _, trees = bps.resolve_per_tree_plan(tasks)
  tables = parse_program_text(_program(tmp_path, tasks)).tables
  for t in (APP_TASK, DDOS_TASK):
    for i, own in enumerate(trees[t]):
      assert tables["get_classification_tree_%s_%d" % (t, i)] == [
          "code_" + bps.packed_field(f) for f in sorted(own)]


@pytest.mark.parametrize("deep", [False, True])
def test_everything_but_the_codes_matches_the_per_task_program(tmp_path, deep):
  """Registers, range tables, table order and sizes are the per-task design's, so a
  difference between the two encodings on p4c can only come from the code fields."""
  tasks = _deep_specs() if deep else _specs(_forest(3, 2), _forest(2, 1))
  tree = _program(tmp_path, tasks, "tree.p4")
  task = _program(tmp_path, tasks, "task.p4", encoding=PER_TASK_ENCODING)

  def without_codes(text):
    return [line for line in text.split("\n") if "code_tk_" not in line and "code_pk_" not in line
            and "set_code_" not in line and not re.match(r"^\s*\}\s*$", line)]
  assert without_codes(tree) == without_codes(task)
  assert parse_program_text(tree).sizes == parse_program_text(task).sizes


# --- one encoding argument ----------------------------------------------------------

# sha256 of the program text (LF) each design got from the generator's old per_task_codes
# flag, frozen 2026-11-08 before `encoding` replaced it (e1b7a97 plus the replay parser
# commits). ("joint", seed): joint intervals, flag off; ("disjoint", seed): each task's own
# intervals, flag off; ("per_task", seed): own intervals, flag on; "three": _three_tasks().
_OLD_FLAG_SHAS = {
    ("joint", 0): "44d42d7103c79514911511d7dcd65d2dbe93989fccbb55055803f6c347a6f695",
    ("disjoint", 0): "dd999db55ff49de5399654a65aa9435ca22c00fbca248ef0aef79c17ceb59816",
    ("per_task", 0): "e6e66869cc7d22a5f97ecef6baf95f7b22957e6ffb8a136ccb2dfe3d06ed78b6",
    ("joint", 1): "b5d7652026f991a38fee04f49879150587adadbf641740282bf3f3d4b8e02744",
    ("disjoint", 1): "1ee1db4bf06b3c0687fefcd9474d01199e35f4e55368dce183d80ed5651eb515",
    ("per_task", 1): "3e05607d7dc185784352dcca25fbac4c53b534c3f28d3d531afc6b9f70cc5170",
    ("joint", 2): "9c00eedc1465771df1dfa1153128982aa27d4f07ee005a163fd15ad58395d8c1",
    ("disjoint", 2): "769c2b26f0d036aef06dfe655d01c2413018326b0694a90331b4af4fe3e90d59",
    ("per_task", 2): "bc6d4626fcec05bd93e7bc5b83fd0a7234823ce07467042cf706ef9a065f7242",
    ("joint", 3): "3fe693dd280e151ba36332b34a668c514295c9df47615281f489bd6ccee6b17a",
    ("disjoint", 3): "f2863d59d205e2645d14083ca9454bc1fd9fe79548a21bc4c8e848435e6d4cf0",
    ("per_task", 3): "f7d0a2526da4dda5aa01bbf2568ba733c5d9a5f9714ecd834af762752de6551e",
    ("disjoint", "three"): "a5291e04a877b44c517726efa349e9b965da43e2f66fb9e69f690279af7c5119",
    ("per_task", "three"): "6ae2e0375e4b81694027d6096354201943218337d2adc83d758072716fd241f3",
}


# Every program pins meta.fwd out of PHV overlays since 2026-11-09 (build_p4_script;
# test_every_encoding_pins_fwd_once): the frozen programs above plus that one line.
_FWD_PIN = '@pa_no_overlay("ingress", "ig_md.fwd")\n'


def _sha(text):
  return hashlib.sha256(text.replace(_FWD_PIN, "", 1).encode("utf-8")).hexdigest()


def _two_task(tmp_path, app, ddos, names, intervals, **kwargs):
  path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=intervals[0],
                              feature_intervals_ddos=intervals[1],
                              output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
                              selected_features_app=names, selected_features_ddos=names, **kwargs)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING])
def test_each_encoding_writes_the_program_the_old_flag_wrote(tmp_path, encoding):
  for seed in range(4):
    app, ddos, names = _rng_design(seed)
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    own = (_own(app, names), _own(ddos, names))
    # joint and disjoint are both the flag-off path, where the intervals decide
    cases = ({"per_task": own} if encoding == PER_TASK_ENCODING
             else {"joint": (joint, joint), "disjoint": own})
    for frozen, intervals in cases.items():
      text = _two_task(tmp_path, app, ddos, names, intervals, encoding=encoding)
      assert _sha(text) == _OLD_FLAG_SHAS[(frozen, seed)], (frozen, seed)
  tasks, _, _ = _three_tasks()
  frozen = "per_task" if encoding == PER_TASK_ENCODING else "disjoint"
  assert (_sha(_program(tmp_path, tasks, "three.p4", encoding=encoding))
          == _OLD_FLAG_SHAS[(frozen, "three")])


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING,
                                      PER_TREE_ENCODING])
def test_every_encoding_pins_fwd_once(tmp_path, encoding):
  app, ddos, names = _rng_design(0)
  own = (_own(app, names), _own(ddos, names))
  text = _two_task(tmp_path, app, ddos, names, own, encoding=encoding)
  assert text.count(_FWD_PIN) == 1
  # ahead of every other PHV pragma, where the 45-design pin batch put it
  assert text.index(_FWD_PIN) < text.index("@pa_solitary")


def test_the_default_encoding_is_joint(tmp_path):
  app, ddos, names = _rng_design(0)
  joint = bps.get_joint_feature_intervals(app, names, ddos, names)
  assert _sha(_two_task(tmp_path, app, ddos, names, (joint, joint))) == _OLD_FLAG_SHAS[("joint", 0)]
  tasks, _, _ = _three_tasks()
  path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename="t.p4")
  with open(path, encoding="utf-8") as handle:
    assert _sha(handle.read()) == _OLD_FLAG_SHAS[("disjoint", "three")]


def test_an_unknown_encoding_raises(tmp_path):
  tasks, _, _ = _three_tasks()
  with pytest.raises(ValueError) as e:
    _program(tmp_path, tasks, "t.p4", encoding="per_forest")
  assert all(repr(name) in str(e.value) for name in ENCODINGS)
  app, ddos = _forest(3, 2), _forest(2, 1)
  with pytest.raises(ValueError):
    _two_task(tmp_path, app, ddos, FEATURES, (_own(app), _own(ddos)), encoding="perTask")
  assert not (tmp_path / "t.p4").exists() and not (tmp_path / "d.p4").exists()
  # the control-plane entries refuse it the same way, and name the same encodings
  with pytest.raises(ValueError) as e:
    bps.generate_table_entries(tasks, encoding="per_forest")
  assert all(repr(name) in str(e.value) for name in ENCODINGS)


# --- the control-plane entries -------------------------------------------------------
#
# The other encodings' entries, frozen. The per-tree (packed) entries and their meaning tests
# are in tests/test_packed_entries.py.

_ENTRIES_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "entries_by_encoding.json")


def _entry_tasks(encoding, seed):
  """The seed's synthetic design as tasks: both tasks share one joint dict under joint, each
  has its own intervals otherwise (the intervals decide joint or disjoint, as in the program)."""
  app, ddos, names = _rng_design(seed)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    return _specs(app, ddos, names, names, joint, joint)
  return _specs(app, ddos, names, names)


def _frozen_hash(entries):
  text = json.dumps(entries, sort_keys=True, separators=(",", ":"))
  return {"entries": len(entries), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("discount", [False, True])
def test_each_encoding_writes_exactly_the_frozen_entries(encoding, seed, discount):
  """Gate B5: the entries the old per_task_codes flag wrote, frozen 2026-11-08 before
  `encoding` replaced it (joint and disjoint were both flag off; the intervals decide)."""
  with open(_ENTRIES_FIXTURE, encoding="utf-8") as handle:
    frozen = json.load(handle)
  entries = bps.generate_table_entries(_entry_tasks(encoding, seed),
                                       use_default_action_discount=discount, encoding=encoding)
  assert _frozen_hash(entries) == frozen["%s/seed%d/discount_%s" % (
      encoding, seed, "on" if discount else "off")]


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING])
def test_generate_p4_program_writes_the_entries_of_its_encoding(tmp_path, encoding):
  tasks = _entry_tasks(encoding, 0)
  _program(tmp_path, tasks, encoding=encoding, entries_filename="entries.json",
           use_default_action_discount=True)
  with open(tmp_path / "entries.json", encoding="utf-8") as handle:
    written = json.load(handle)
  assert written == bps.generate_table_entries(tasks, use_default_action_discount=True,
                                               encoding=encoding)
  assert written
  # each encoding addresses its own code parameters: code_<task> per task
  params = {name for e in written if e["table_name"].startswith("table_")
            for name in e["action_params"]}
  if encoding == PER_TASK_ENCODING:
    assert all(re.match(r"^code_(app|ddos)$", name) for name in params)


# --- the model: training path and replay ----------------------------------------------
#
# That training prices exactly what the generated program replays, and the pins' round trip,
# are in tests/test_packed_replay.py.

def _tree_text(tmp_path, app, ddos, names_app, names_ddos=None):
  names_ddos = names_app if names_ddos is None else names_ddos
  return _program(tmp_path, _specs(app, ddos, names_app, names_ddos))


def _evaluate(app, ddos, names_app, names_ddos=None):
  from p4gen.evaluation import multi_model_memory_evaluation
  names_ddos = names_app if names_ddos is None else names_ddos
  return multi_model_memory_evaluation(app, ddos, names_app, names_ddos, PER_TREE_ENCODING)


def _pool(app, ddos, names_app, names_ddos=None):
  from p4gen.evaluation import _pool_inputs
  names_ddos = names_app if names_ddos is None else names_ddos
  return _pool_inputs(app, ddos, names_app, names_ddos, PER_TREE_ENCODING)


def _many_trees():
  """10 trees a task over five features: several codes on each packed field."""
  return _forest(3, 2, n_trees=10, depth=3), _forest(2, 1, n_trees=10, depth=3), FEATURES


def test_replay_labels_a_reader_of_a_packed_code_by_task(tmp_path):
  from p4gen.p4_replay import code_written_by, table_tasks
  from p4model.program import SHARED_TASK
  # app splits features 0-2, ddos 2-4: an app-only, a ddos-only and a shared feature
  app = _forest(3, 2, FEATURES[:3], depth=4)
  ddos = _forest(2, 1, FEATURES[2:], depth=4)
  tables = parse_program_text(_tree_text(tmp_path, app, ddos, FEATURES[:3], FEATURES[2:])).tables
  labels = table_tasks(tables)
  pool = _pool(app, ddos, FEATURES[:3], FEATURES[2:])
  union = pool["emitted_features"]
  assert [n for n in tables if n.startswith("table_")] == [
      "table_%d_%s" % (i, f) for i, f in enumerate(union)]
  assert pool["range_task"] == [labels["table_%d_%s" % (i, f)] for i, f in enumerate(union)]
  assert {APP_TASK, DDOS_TASK} <= set(pool["range_task"])
  trees = [(re.match(r"^get_classification_tree_(app|ddos)_(\d+)$", n), keys)
           for n, keys in tables.items() if n.startswith("get_classification_tree_")]
  for i, f in enumerate(union):
    readers = {m[1] for m, keys in trees if "code_pk_" + f in keys}
    assert labels["table_%d_%s" % (i, f)] == (SHARED_TASK if len(readers) == 2 else readers.pop())
  assert all(labels[m[0]] == m[1] for m, _ in trees)
  # a tree key names the range table of its feature, whichever encoding wrote it
  assert code_written_by("code_pk_flow_iat_max", APP_TASK) == "code_flow_iat_max"
  assert code_written_by("code_tr_app_3_flow_iat_max", APP_TASK) == "code_flow_iat_max"
  assert code_written_by("code_tk_app_flow_iat_max", APP_TASK) == "code_flow_iat_max"
  assert code_written_by("code_flow_iat_max", APP_TASK) == "code_flow_iat_max"


def test_a_tree_waits_for_the_range_tables_it_reads_as_the_replay_maps_them(tmp_path):
  """Rule E (usage.tree_readiness_levels): a tree waits for the range tables whose codes it
  keys, and never precedes the pool's last range table. Training must hand the packer the
  same reads the replay maps from the program's keys, or the two paths drift."""
  from p4gen.p4_replay import tree_range_reads
  app, ddos, names = _many_trees()
  program = parse_program_text(_tree_text(tmp_path, app, ddos, names))
  reads = tree_range_reads(program.tables)
  assert reads is not None
  ranges = [n for n in program.tables if n.startswith("table_")]
  trees = [n for n in program.tables if n.startswith("get_classification_tree_")]
  pool = _pool(app, ddos, names)
  assert [sorted(r) for r in pool["tree_reads"]] == [
      sorted(ranges.index(t) for t in reads[name]) for name in trees]
  own = {APP_TASK: bps.per_tree_intervals(_specs(app, ddos)[1]),
         DDOS_TASK: bps.per_tree_intervals(_specs(app, ddos)[0])}
  union = pool["emitted_features"]
  expected = [sorted(union.index(f) for f in own[t][i])
              for t in (APP_TASK, DDOS_TASK) for i in range(len(own[t]))]
  assert [sorted(r) for r in pool["tree_reads"]] == expected
  # some tree reads fewer range tables than its task writes codes for, or the test proves nothing
  labels = [t for t in (APP_TASK, DDOS_TASK) for _ in own[t]]
  task_reads = {t: set().union(*(set(r) for r, label in zip(expected, labels) if label == t))
                for t in (APP_TASK, DDOS_TASK)}
  assert any(set(r) < task_reads[label] for r, label in zip(expected, labels))
  # the readers per range table and per feature (p4c's tie orders) are the replay's too
  readers = {}
  for name in trees:
    for table in reads[name]:
      readers[table] = readers.get(table, 0) + 1
  assert pool["range_dependents"] == [readers.get(t, 0) for t in ranges]


def test_two_trees_on_identical_intervals_stay_distinct_keys():
  # two identical trees of one task: equal codes, each at its own bytes of the packed field
  rng = np.random.RandomState(0)
  X = rng.randint(0, 2000, size=(600, len(FEATURES)))
  y = (X[:, 0] > 900).astype(int) + (X[:, 1] > 1200)
  twins = bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=2, bootstrap=False, max_features=None, max_depth=3, random_state=0).fit(X, y))
  tasks = _specs(twins, _forest(2, 1))
  table_plan, _, own = bps.resolve_per_tree_plan(tasks)
  assert own[APP_TASK][0] == own[APP_TASK][1] and own[APP_TASK][0]
  layout = bps.per_tree_layout(table_plan, own, tasks)
  for feature in own[APP_TASK][0]:
    (a, bits_a), (b, bits_b) = (layout.place[(APP_TASK, i, feature)] for i in (0, 1))
    assert bits_a == bits_b
    assert not (set(range(a, a + -(-bits_a // 8))) & set(range(b, b + -(-bits_b // 8))))
  # in the pool: two keys of equal length, never sharing a crossbar field
  pool = _pool(twins, _forest(2, 1), FEATURES)
  assert len(pool["ternary_key_bytes"][0]) == len(pool["ternary_key_bytes"][1])
  assert pool["ternary_fields"] is None


def test_pool_inputs_raises_on_a_tree_with_no_splits(tmp_path):
  # a constant input leaves the tree nothing to split on: its root is a leaf
  X = np.zeros((60, len(FEATURES)), dtype=int)
  y = np.arange(60) % 3
  flat = bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=2, max_depth=3, random_state=0).fit(X, y))
  with pytest.raises(NoSplitTree) as e:
    _pool(flat, _forest(2, 1), FEATURES)
  assert e.value.args[1] == "app_0"
  with pytest.raises(NoSplitTree) as generated:
    _tree_text(tmp_path, flat, _forest(2, 1), FEATURES)
  assert generated.value.args == e.value.args


def test_the_range_pool_is_the_per_task_arms():
  app, ddos, names = _many_trees()
  pool = _pool(app, ddos, names)
  _, _, trees = bps.resolve_per_tree_plan(_specs(app, ddos, names, names))
  assert pool["ternary_task"] == [t for t in (APP_TASK, DDOS_TASK) for _ in trees[t]]
  # one range table per feature over the union, as under per_task
  from p4gen.evaluation import _pool_inputs
  per_task = _pool_inputs(app, ddos, names, names, PER_TASK_ENCODING)
  for key in ("range_table_specs", "range_levels", "range_task", "range_fields",
              "emitted_features", "register_names", "range_entries", "range_blocks"):
    assert pool[key] == per_task[key], key


# sha256 of json.dumps(model_breakdown(...), sort_keys=True) for the program each other encoding
# writes for test_per_task_codes' random design of the seed, frozen 2026-11-08 at 64ab25c,
# before the replay learned per-tree codes.
_OTHER_REPLAY_SHAS = {
    ("joint", 0): "0ddd86d1ae9800c435020184bbaf44ae0d76173977070b097ea3403716f42ced",  # 2026-11-13: joint pool simulated, same totals
    ("disjoint", 0): "9726ece6130b4ef6e4b80481f68291bcf6a26a8afadd4b1dd7181d14e8da25e7",
    ("per_task", 0): "84ee7109c89fe0ebd142f5416f40c7797799531d1c78a68fb2bdcdd2ce972e8c",
    ("joint", 1): "62d1be543bcea0b8242e3c7213f9f82898577b147ab475b8db0df1a8a9760b02",
    ("disjoint", 1): "dd83f6bc9962048dd56275f8142cbb10e8b4889d7eba877c93ca0ff80a5d4e93",
    ("per_task", 1): "62d1be543bcea0b8242e3c7213f9f82898577b147ab475b8db0df1a8a9760b02",
    ("joint", 2): "0056e70905fc70ca91bf79b92d6313d2367347adb61dacfb859360f768dde64a",  # 2026-11-13: joint pool simulated, same totals
    ("disjoint", 2): "16296479245e1f310425f1b7e388fed3761953cbf424d0d43536d3ad2d51ae39",
    ("per_task", 2): "57ddde2b9eb72d69671ea259cee648d41def369736a7291a8977b19788bd5a13",
    ("joint", 3): "d653f8b2df9ee85839c15378ffe683191867be0f4efb9310afc1f068e8316788",
    ("disjoint", 3): "80649da6da555fcdc4bd5589ee39e14af6a21012e3a9df6c7eb4ca4a28e6e155",
    ("per_task", 3): "d653f8b2df9ee85839c15378ffe683191867be0f4efb9310afc1f068e8316788",
    ("joint", 4): "0090942bc443b22f4d69add18e555be29718999f43bb32bf4addfaa71f0013bb",
    ("disjoint", 4): "5b6f0a75cd2f73b77d489796e1905b508d78073e61a777b8ff0b1f86cb442187",
    ("per_task", 4): "0090942bc443b22f4d69add18e555be29718999f43bb32bf4addfaa71f0013bb",
    ("joint", 5): "4a5389c4885dea09e728fcc879bf9ab709c21f944975690133d6ece7ff9a2a1d",  # 2026-11-13: joint pool simulated, same totals
    ("disjoint", 5): "b3b00aebf004fc1e89bf674d9d9655ca3ea3c3be197fbf4f7568c6e5f1a77748",
    ("per_task", 5): "4e62db9f47fb3831cead683627f98f2c9b2b2f0bb6c6b1c00c0c6c2a9590943e",
}


@pytest.mark.parametrize("seed", range(6))
def test_one_code_and_per_task_programs_replay_unchanged(tmp_path, seed):
  app, ddos, names = _rng_design(seed)
  joint = bps.get_joint_feature_intervals(app, names, ddos, names)
  own = (_own(app, names), _own(ddos, names))
  for encoding, intervals in ((JOINT_ENCODING, (joint, joint)), (DISJOINT_ENCODING, own),
                              (PER_TASK_ENCODING, own)):
    program = parse_program_text(_two_task(tmp_path, app, ddos, names, intervals,
                                           encoding=encoding))
    assert all(type(bits) is int for bits in program.bits.values())
    # The programs carry the meta.fwd pin since 2026-11-09, which switches the replay to
    # p4c's chain order for the gated register blocks (test_the_fwd_pin_switches_the_
    # register_order); read without it, everything else is the frozen replay, the
    # range pool's action-data budget included.
    assert program.fwd_pinned
    program = dataclasses.replace(program, fwd_pinned=False)
    breakdown = json.dumps(model_breakdown(program), sort_keys=True)
    assert _sha(breakdown) == _OTHER_REPLAY_SHAS[(encoding, seed)], (encoding, seed)


# --- the arm joint-tree (spec 2026-11-06 Sec 9.1, 9.3, 9.4; 2026-11-11 Sec 4.5) ----------

def test_arm_identity_flows_through_config_row_id_jobs_and_the_loader():
  from src.main import PER_TREE_ARM_ENTRY, select_arm_slugs
  from src.reporting import claims
  from src.reporting.campaign_data import PER_TREE_ARM_SLUG, _expected_arm_slug
  from src.training.campaign_run import row_id, split_csv_name
  from src.training.campaign_runner import Job, plan_jobs
  from src.training.config import ARM_ENCODINGS, PER_TREE_ARM, arm_encoding
  assert PER_TREE_ARM == PER_TREE_ARM_SLUG == "joint-tree"
  assert ARM_ENCODINGS[PER_TREE_ARM] == PER_TREE_ENCODING
  (arm, cfg), = select_arm_slugs(["joint-tree"])
  assert (arm, cfg) == PER_TREE_ARM_ENTRY
  assert arm == "joint-tree" and arm_encoding(arm) == PER_TREE_ENCODING
  assert cfg.arm_slug(PER_TREE_ENCODING) == "joint-tree"
  assert not cfg.alignment_enabled
  assert cfg == dict(select_arm_slugs(["joint-off"]))["joint"]     # only the encoding differs
  assert cfg == select_arm_slugs(["joint-task"])[0][1]
  job = Job(arm=arm, cfg=cfg, M=35, split=7)
  assert job.slug == "joint-tree" and job.key == "joint-tree_M035_s07"
  assert os.path.basename(job.csv_path("run")) == split_csv_name(
      cfg.n_trees, cfg.max_depth, 35, "joint-tree", 7)
  assert row_id("joint-tree", 35, 7, 9) == "joint-tree_M035_s07_k09"
  assert [j.key for j in plan_jobs([(arm, cfg)], [15, 75], [0], "no_such_dir")] == [
      "joint-tree_M015_s00", "joint-tree_M075_s00"]
  assert _expected_arm_slug("joint-tree", False, "") == "joint-tree"
  assert "joint-tree" in claims.REPORTED_JOINT_ARM_SLUGS
  assert "joint-tree" not in claims.JOINT_ARM_SLUGS      # outside the pre-registered family


def test_the_primary_preset_is_unchanged():
  from src.main import select_arms
  from src.training.config import arm_encoding
  assert [cfg.arm_slug(arm_encoding(arm)) for arm, cfg in select_arms("primary")] == [
      "independent", "joint-off", "joint"]


def test_row_artifacts_of_the_arm_replay_to_the_trainings_numbers(tmp_path):
  from src.training import row_artifacts as ra
  app, ddos = _forest(3, 2), _forest(2, 1)
  usage = _evaluate(app, ddos, FEATURES)
  ctx = ra.RowContext(str(tmp_path), "joint-tree", 35, "abc")
  row = "joint-tree_M035_s00_k05"
  model = ra.write_row_artifacts(ctx, row, app, ddos, FEATURES, FEATURES, PER_TREE_ENCODING, usage)
  assert model["model_paths_differ"] is False
  assert model["generator_error"] is None
  assert (model["stage_depth"], model["blocks"]) == (usage.stage_depth, usage.blocks)
  assert "code_pk_" in (tmp_path / "designs" / (row + ".p4")).read_text(encoding="utf-8")


def test_the_row_writer_writes_no_stage_pin(tmp_path):
  """Since the crossbar simulation (spec 2026-11-13) no table is moved for the model: the program carries
  no @stage line and model.json has no pin field."""
  from src.training import row_artifacts as ra
  app, ddos, names = _rng_design(3)
  usage = _evaluate(app, ddos, names)
  ctx = ra.RowContext(str(tmp_path), "joint-tree", 35, "abc")
  row = "joint-tree_M035_s00_k05"
  model = ra.write_row_artifacts(ctx, row, app, ddos, names, names, PER_TREE_ENCODING, usage)
  text = (tmp_path / "designs" / (row + ".p4")).read_text(encoding="utf-8")
  assert "@stage(" not in text
  assert model["model_paths_differ"] is False
  assert (model["stage_depth"], model["blocks"]) == (usage.stage_depth, usage.blocks)
  stored = json.loads((tmp_path / "designs" / (row + ".model.json")).read_text(encoding="utf-8"))
  assert not {"stage_pins", "stage_pin_first_refusal", "unpinnable_tables"} & set(stored)


@pytest.mark.parametrize("error", ["CodewordTooLong", "CrossbarKeyTooWide"])
def test_a_pricing_refusal_while_generating_is_a_generator_error_row(tmp_path, monkeypatch, error):
  """Defensive (unreachable today: training refuses these designs before selection)."""
  from p4model import errors
  from src.training import row_artifacts as ra

  def refuse(*args, **kwargs):
    raise getattr(errors, error)("refused for the test", 99)

  monkeypatch.setattr(ra, "generate_P4_code", refuse)
  app, ddos, names = _rng_design(3)
  row, model = _generator_error_row(tmp_path, app, ddos, names)
  assert "refused for the test" in model["generator_error"]
  assert not (tmp_path / "designs" / (row + ".p4")).exists()


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, PER_TASK_ENCODING])
def test_other_arms_rows_record_no_pins(tmp_path, encoding):
  from p4gen.evaluation import multi_model_memory_evaluation
  from src.training import row_artifacts as ra
  app, ddos = _forest(3, 2), _forest(2, 1)
  usage = multi_model_memory_evaluation(app, ddos, FEATURES, FEATURES, encoding)
  ctx = ra.RowContext(str(tmp_path), "x", 35, "abc")
  model = ra.write_row_artifacts(ctx, "x_M035_s00_k05", app, ddos, FEATURES, FEATURES,
                                 encoding, usage)
  assert model["model_paths_differ"] is False
  assert "stage_pins" not in model
  assert "@stage(" not in (tmp_path / "designs" / "x_M035_s00_k05.p4").read_text(encoding="utf-8")


def test_the_loader_refuses_alignment_on_the_arm():
  from src.reporting.campaign_data import MislabelledArtifactError, _expected_arm_slug
  with pytest.raises(MislabelledArtifactError, match="alignment on the joint-tree arm"):
    _expected_arm_slug("joint-tree", True, "")
  with pytest.raises(MislabelledArtifactError, match="alignment on the joint-tree arm"):
    _expected_arm_slug("joint-tree", False, "", "", True)       # a twin of it
  # not an alignment refusal: "joint-tree-off" is no arm at all
  with pytest.raises(MislabelledArtifactError, match="unrecognised value 'joint-tree-off'"):
    _expected_arm_slug("joint-tree-off", False, "")


def test_ablation_has_a_sharing_row_for_the_arm_and_no_alignment_row():
  import pandas as pd
  from src.reporting.claims import ablation_decomposition
  rows = []
  for split in range(3):
    for slug, blocks in (("independent", 40.0), ("joint-off", 34.0), ("joint-task", 32.0),
                         ("joint-tree", 30.0)):
      rows.append({"arm_slug": slug, "M": 25, "split": split, "k": 5, "acc_app": 0.9,
                   "acc_ddos": 0.85, "f1_app": 0.88, "f1_ddos": 0.83, "blocks": blocks,
                   "stages": 3.0})
  table = ablation_decomposition(pd.DataFrame(rows), metrics=("blocks",))
  assert set(table["contrast"]) == {"joint-off - independent", "joint-task - independent",
                                    "joint-tree - independent"}
  assert set(table["component"]) == {"sharing"}
  tree = table[table["treatment"] == "joint-tree"].iloc[0]
  assert tree["baseline"] == "independent"
  assert tree["mean_diff_split_level"] == pytest.approx(-10.0)


def _generator_error_row(tmp_path, app, ddos, names):
  from src.training import row_artifacts as ra
  usage = _evaluate(_forest(3, 2), _forest(2, 1), FEATURES)      # a harmless design's numbers
  ctx = ra.RowContext(str(tmp_path), "joint-tree", 35, "abc")
  row = "joint-tree_M035_s00_k05"
  model = ra.write_row_artifacts(ctx, row, app, ddos, names, names, PER_TREE_ENCODING, usage)
  return row, model


def test_a_field_wider_than_the_bus_is_recorded_as_a_generator_error_not_a_crash(tmp_path):
  # 20 deep app trees: flow_iat_max's packed field needs more than the bus's 128 bytes
  app, ddos = _deep_forest(3, 0, depth=10, n_trees=20), _deep_forest(2, 1, depth=3)
  row, model = _generator_error_row(tmp_path, app, ddos, DEEP_FEATURES)
  assert "action data bus" in model["generator_error"]
  assert model["stage_depth"] is None and model["blocks"] is None
  assert model["hw_feasible"] is False and model["budget_feasible"] is False
  assert (tmp_path / "designs" / (row + ".generator_error.txt")).read_text() == \
      model["generator_error"]
  assert not (tmp_path / "designs" / (row + ".p4")).exists()


def test_a_tree_with_no_splits_is_recorded_as_a_generator_error_not_a_crash(tmp_path):
  X = np.zeros((60, len(FEATURES)), dtype=int)
  flat = bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=2, max_depth=3, random_state=0).fit(X, np.arange(60) % 3))
  row, model = _generator_error_row(tmp_path, flat, _forest(2, 1), FEATURES)
  assert "no threshold" in model["generator_error"]
  assert model["hw_feasible"] is False


def _search(monkeypatch, replacement, n_trials=6):
  """The trials of a real per-tree search whose evaluation is `replacement(real, call)`
  (call = how many evaluations came before); returns (study trials, train result or None).
  4000 rows, because forests trained on fewer are mostly stumps (NoSplitTree)."""
  import optuna
  import src.training.train_model as tm
  from src.training.config import TrainConfig
  from src.training.errors import NoFeasibleSolution
  from tests.test_train_model_contract import _call
  captured = {}
  real_create = optuna.create_study

  def capture(*args, **kwargs):
    captured["study"] = real_create(*args, **kwargs)
    return captured["study"]

  monkeypatch.setattr(tm.optuna, "create_study", capture)
  real = tm.multi_model_memory_evaluation
  calls = []

  def evaluate(*args, **kwargs):
    calls.append(1)
    return replacement(real, len(calls) - 1)(*args, **kwargs)

  monkeypatch.setattr(tm, "multi_model_memory_evaluation", evaluate)
  cfg = TrainConfig(n_trials=n_trials, min_feasible_before_stop=3, lookback=2)
  try:
    result = _call(encoding=PER_TREE_ENCODING, cfg=cfg, max_blocks=200, n=4000)
  except NoFeasibleSolution:
    result = None
  return captured["study"].trials, result


def test_a_field_wider_than_the_bus_is_an_infeasible_trial_with_its_overflow(monkeypatch):
  from src.training import early_stopping

  def replacement(real, call):
    if call == 0:
      def refuse(*args, **kwargs):
        raise ActionDataTooWide("too wide", ACTION_DATA_BUS_BYTES + 5)
      return refuse
    return real

  trials, result = _search(monkeypatch, replacement)
  first = trials[0]
  assert first.user_attrs["crossbar_violation"] == 5.0
  for attr in ("codeword_violation", "blocks_violation", "stages_violation"):
    assert first.user_attrs[attr] == 0.0
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert "no_split_tree" not in first.user_attrs
  assert len(trials) > 1                          # the search continued
  assert result is not None                       # and found a feasible design


def test_a_phv_refusal_is_an_infeasible_trial_with_its_overflow(monkeypatch):
  from src.training import early_stopping

  def replacement(real, call):
    if call == 0:
      def refuse(*args, **kwargs):
        raise PhvContainersExhausted("no PHV container left", 4)
      return refuse
    return real

  trials, result = _search(monkeypatch, replacement)
  first = trials[0]
  assert first.user_attrs["crossbar_violation"] == 4.0
  for attr in ("codeword_violation", "blocks_violation", "stages_violation"):
    assert first.user_attrs[attr] == 0.0
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert len(trials) > 1                          # the search continued
  assert result is not None                       # and found a feasible design


def test_a_bus_overflow_is_floored_at_one(monkeypatch):
  def replacement(real, call):
    def refuse(*args, **kwargs):
      raise ActionDataTooWide("too wide", ACTION_DATA_BUS_BYTES)
    return refuse

  trials, result = _search(monkeypatch, replacement, n_trials=3)
  assert result is None
  assert {t.user_attrs["crossbar_violation"] for t in trials} == {1}


def test_a_no_split_tree_is_an_infeasible_trial_and_counted(monkeypatch):
  from src.training import early_stopping

  def replacement(real, call):
    if call == 0:
      def refuse(*args, **kwargs):
        raise NoSplitTree("tree app_0 has no split", "app_0")
      return refuse
    return real

  trials, result = _search(monkeypatch, replacement)
  first = trials[0]
  assert first.user_attrs["no_split_tree"] is True
  assert first.user_attrs["crossbar_violation"] == 1.0
  for attr in ("codeword_violation", "blocks_violation", "stages_violation"):
    assert first.user_attrs[attr] == 0.0
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert result is not None
  rows = {row["number"]: row for row in result.trial_table}
  assert rows[0]["no_split_tree"] is True and rows[0]["feasible"] is False
  # the column is the trials' own attribute (the search also meets stumps by itself)
  assert {number for number, row in rows.items() if row["no_split_tree"]} == {
      t.number for t in trials if t.user_attrs.get("no_split_tree")}
  assert all(row["feasible"] is False for row in rows.values() if row["no_split_tree"])


def test_other_arms_trial_tables_keep_their_columns(tmp_path):
  from src.training import row_artifacts as ra
  from tests.test_train_model_contract import _call
  header = ("number,params,feasible,acc_sel_app,acc_sel_ddos,blocks,stage_depth,"
            "b_app,c_app,b_ddos,c_ddos,tied")
  for encoding in (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING):
    result = _call(encoding=encoding)
    assert list(result.trial_table[0]) == header.split(",")
    assert "no_split_tree" not in result.trial_table[0]
    ctx = ra.RowContext(str(tmp_path / encoding), "x", 35, "abc")
    path = ra.write_trial_table(ctx, "x_M035_s00_k01", result.trial_table)
    with open(path, encoding="utf-8", newline="") as handle:
      assert handle.readline().rstrip("\n") == header
  result = _call(encoding=PER_TREE_ENCODING, n=4000)
  assert list(result.trial_table[0]) == header.split(",") + ["no_split_tree"]
  ctx = ra.RowContext(str(tmp_path / "tree"), "x", 35, "abc")
  path = ra.write_trial_table(ctx, "x_M035_s00_k01", result.trial_table)
  with open(path, encoding="utf-8", newline="") as handle:
    assert handle.readline().rstrip("\n") == header + ",no_split_tree"


# --- hardening from the task reviews ---------------------------------------------------

def _all_infeasible_elimination(tmp_path, arm, max_blocks):
  """_run_elimination for one arm over a search that finds no feasible trial (the real
  trainer on 400 rows: per-tree forests that small are all stumps; any other arm is starved
  of blocks), with a row context so the run's files are written."""
  import optuna
  from src.training import feature_selection as fs
  from src.training.config import TrainConfig
  from src.training.row_artifacts import RowContext
  from tests.test_run_elimination import FEATURE_NAMES, _splits
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  app, ddos = _splits()
  cfg = TrainConfig(n_trials=6, min_feasible_before_stop=3, lookback=2)
  rows = fs._run_elimination(
      arm=arm, split_idx=0, app=app, ddos=ddos, feature_names=list(FEATURE_NAMES),
      max_blocks=max_blocks, cfg=cfg, row_context=RowContext(str(tmp_path), arm, 35, "abc"))
  return rows


def test_a_per_tree_search_with_no_feasible_trial_still_counts_its_stumps(tmp_path):
  import csv
  rows = _all_infeasible_elimination(tmp_path, "joint-tree", 200)
  assert [row["infeasible"] != "" for row in rows] == [True]    # nothing feasible
  path = tmp_path / "trials" / (rows[0]["row_id"] + ".csv")
  assert path.is_file()
  with open(path, encoding="utf-8", newline="") as handle:
    table = list(csv.DictReader(handle))
  assert len(table) == 6
  assert {row["no_split_tree"] for row in table} == {"True"}
  assert {row["feasible"] for row in table} == {"False"}


def test_the_failed_per_tree_search_carries_the_table_on_the_exception():
  from src.training.config import TrainConfig
  from src.training.errors import NoFeasibleSolution
  from tests.test_train_model_contract import _call
  cfg = TrainConfig(n_trials=4, min_feasible_before_stop=3, lookback=2)
  with pytest.raises(NoFeasibleSolution) as e:
    _call(encoding=PER_TREE_ENCODING, cfg=cfg, max_blocks=200, n=240)
  assert list(e.value.trial_table[0])[-1] == "no_split_tree"
  assert all(row["no_split_tree"] for row in e.value.trial_table)
  with pytest.raises(NoFeasibleSolution) as e:
    _call(encoding=DISJOINT_ENCODING, cfg=cfg, max_blocks=1)
  assert e.value.trial_table is None


@pytest.mark.parametrize("arm", ["independent", "joint", "joint-task"])
def test_another_arm_with_no_feasible_trial_still_writes_no_trials_file(tmp_path, arm):
  rows = _all_infeasible_elimination(tmp_path, arm, 1)
  assert all(row["infeasible"] != "" for row in rows)
  assert not (tmp_path / "trials").exists() or not list((tmp_path / "trials").iterdir())


@pytest.mark.parametrize("refusal", [ActionDataTooWide("too wide", ACTION_DATA_BUS_BYTES + 1),
                                     NoSplitTree("tree app_0 has no split", "app_0"),
                                     PhvContainersExhausted("no PHV container left", 1)])
def test_hardware_validation_reports_a_per_tree_refusal_as_unavailable(monkeypatch, refusal):
  """The per-tree generator refusals are RuntimeErrors, not ValueErrors: they must degrade
  to the 'unavailable' marker like a missing register, not crash the split."""
  from src.training import feature_selection as fs

  def refuse(*args, **kwargs):
    raise refusal

  monkeypatch.setattr(bps, "generate_P4_code", refuse)
  handle = fs._kickoff_hardware_validation(
      True, "unused/", 0, "joint-tree", 3, _forest(2, 1), _forest(2, 1), FEATURES, FEATURES,
      "per_tree")
  assert fs._is_unavailable(handle)
  assert handle[1] == str(refusal)
