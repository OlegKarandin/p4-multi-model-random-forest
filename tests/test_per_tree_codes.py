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
import hashlib
import json
import os
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4model.refusals import CONSTRAINTS, violations
from p4gen.p4_replay import model_breakdown, parse_program_text
from p4model import lanes, packed
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, ENCODINGS, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING, TaskSpec,
                             feature_of_per_tree_field)
from p4model.target import ACTION_DATA_BUS_BYTES
from tests.test_per_task_codes import FEATURES, _forest, _own, _rng_design, _three_tasks
from tests.test_per_task_codes import _synthetic_vote_exports  # noqa: F401 (autouse fixture)

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
# Re-frozen 2026-11-14 after decisions 1a / 7.1 option B, which rewrote every program's
# register bodies and apply block, and again for the per-chain IAT scratch fields (no shared
# current_iat, which also leaves the template), and again 2026-11-14 for the vote export (decision 1f:
# the vote action writes ig_tm_md, no classification_<task> field); on the 210 identity-harness designs
# (model8/impl) every changed text is exactly those rewrites of its old text
# (impl/scripts/t2_text_diff.py, t2b_text_diff.py). Re-frozen 2026-11-14 for key pruning
# (decision 2.3: a tree keys only the code fields of the features it splits on): seed 0 and
# "three" changed, each new text exactly its old text minus some classification-key lines, and
# the seed-0 entries (fixtures/entries_by_encoding.json) exactly their old records minus
# all-'*' key fields (impl/scripts/t5_fixture_compare.py); seeds 1-3 key every field on every tree.
# Re-frozen 2026-11-14 for decision 4.3 (code fields declared at their containers' full width,
# range action parameters as wide, tree keys the live-bit slices): every text is exactly its old
# text with that rewrite, and every entries record its old one with each tree key field renamed
# to its bf_rt slice names, values/masks unchanged (impl/scripts/t7_fixture_compare.py).
# Re-frozen 2026-11-15 (Task 17b) for the template move: the two forwarding assignments to
# ig_tm_md now open the ingress apply block instead of closing it (p4c ordered them after the
# votes, which write ig_tm_md too, and sometimes spent a stage on them); every new text with
# that move undone has its old sha (impl/scripts/t17b_refreeze.py).
# Re-frozen 2026-11-15 (Task 23, gate G3 pins): every text is exactly its old text with the
# fwd and class_tree_* @pa_no_overlay lines and the code_* @pa_solitary lines deleted and one
# @pa_container_size(<chain>_current_iat, 16) per scratch field added after the last PHV
# pragma (impl/scripts/t23_text_diff.py, 26/26); the shas are now of the whole text (the old
# ones left the fwd pin out).
_OLD_FLAG_SHAS = {
    ("joint", 0): "9ae190379d505bb40354dfe4ae1f2e58b5316d7e7d9eb38d7601e26ba957e1c8",
    ("disjoint", 0): "40e6a816e0d1a23a15c799e25847f1d350d64d2ee2f6c2c671d16186a9865212",
    ("per_task", 0): "8a3e66ff2cb14f26e1b0ce56667af7ea88d263e6285bcb79f82715a98cbbb7d0",
    ("joint", 1): "ee4888cc0b4a843c9bb53e3a29ce9d451c2200ae868fc41d794aa6870fc65592",
    ("disjoint", 1): "4d44ffe1223d137b9b297abbd209289720ebb00d6a743bc0a3fab289bb85635a",
    ("per_task", 1): "9d3b038d78547e47fd3cba4b9665cceb8f20e2d7a087401eb079f321873438d2",
    ("joint", 2): "6eadd2d23e2885acbf32836b86ff3b94f56767ead5af3a7275ba754f8da2a545",
    ("disjoint", 2): "e448243167a60e5157b2ea505370f32da5f84401046d35261f064ecbc98f5fad",
    ("per_task", 2): "bcfcb8b3ed18de9b75908ca6fef275fe69ebe43a8f15228610265c6698041386",
    ("joint", 3): "732b4d781096e74e76df1d915f3a597bfbbdad381a109b1f812244941d3a502c",
    ("disjoint", 3): "e2386eb98ad26fb3c59574b3eeaa6254f14d628bf605c2a021f8aee149ad5479",
    ("per_task", 3): "556bf6136fcac60cf1b02687500ee661f03f4e86ad615b43e9f6a2687fed9a49",
    ("disjoint", "three"): "f7f0a62567ab68faf5bd85609f3697c77ad24c2d1e59fd77ea7a93fa2311022d",
    ("per_task", "three"): "ff087ae54dff72b2f1be624e0e1585eaa4917680cf393c9f94647e1e1f9b82b1",
}


def _sha(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


_PIN_LINE = re.compile(r'^@(pa_\w+)\("ingress", "ig_md\.(\w+)"((?:, \d+)*)\)$', re.M)
_DECLARED = re.compile(r'^\tbit<(\d+)> (\w+);$', re.M)


def _g3_pin_set(text):
  """The PHV pin set gate G3 (2026-11-15) fixed, from the program's declarations:
  pa_solitary on every class_tree_*; pa_container_size 16 on every <f>_val and every IAT
  scratch field <chain>_current_iat; pa_no_overlay and pa_container_size on every code
  field (code_pk_* too). Nothing on meta.fwd."""
  pins = set()
  for bits, field in _DECLARED.findall(text):
    if field.startswith("class_tree_"):
      pins.add(("pa_solitary", field, ""))
    elif field.endswith("_current_iat") or (field.endswith("_val") and not field.startswith("code_")):
      pins.add(("pa_container_size", field, ", 16"))
    elif field.startswith("code_"):
      pins.add(("pa_no_overlay", field, ""))
  return pins


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING,
                                      PER_TREE_ENCODING])
def test_every_encoding_carries_exactly_the_g3_pin_set(tmp_path, encoding, seed):
  app, ddos, names = _rng_design(seed)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    intervals = (joint, joint)
  else:
    intervals = (_own(app, names), _own(ddos, names))
  text = _two_task(tmp_path, app, ddos, names, intervals, encoding=encoding)
  lines = _PIN_LINE.findall(text)
  assert len(lines) == len(set(lines)), "a duplicate pragma is a p4c error"
  code_sizes = {field: sizes for pin, field, sizes in lines
                if pin == "pa_container_size" and field.startswith("code_")}
  others = {line for line in lines
            if not (line[0] == "pa_container_size" and line[1].startswith("code_"))}
  assert others == _g3_pin_set(text)
  declared = dict((field, int(bits)) for bits, field in _DECLARED.findall(text))
  codes = {field for field in declared if field.startswith("code_")}
  assert set(code_sizes) == codes
  for field, sizes in code_sizes.items():
    assert sum(int(s) for s in sizes.split(",") if s.strip()) == declared[field]
  # the scratch pin is there (every design here has an IAT chain) ...
  scratch = [field for field in declared if field.endswith("_current_iat")]
  assert scratch
  for field in scratch:
    assert '@pa_container_size("ingress", "ig_md.%s", 16)\n' % field in text
  # ... and the three pins dropped at G3 are not (3.1, 3.3, 3.5)
  assert '"ig_md.fwd"' not in text
  assert not re.search(r'@pa_no_overlay\("ingress", "ig_md\.class_tree_', text)
  assert not re.search(r'@pa_solitary\("ingress", "ig_md\.code_', text)
  # and the program passes the replay's shape check, which requires exactly this set
  assert model_breakdown(parse_program_text(text))["refusal"] != "program_shape"


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
    assert not (set(range(a, a + bits_a)) & set(range(b, b + bits_b)))   # bit offsets, 4.4
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
  # range_levels are not compared: they come from the register schedule, whose tie
  # order counts the trees that read each feature (dependents), and per-tree counts
  # only the trees that split on it, per-task every tree of a selecting task. They had
  # agreed on this design by coincidence; under call-site ties (S5, 2026-11-14) they
  # differ ([6, 6, 5, 5, 5] vs [6, 5, 6, 5, 5]).
  for key in ("range_table_specs", "range_task", "range_fields",
              "emitted_features", "register_names", "range_entries", "range_blocks"):
    assert pool[key] == per_task[key], key


# sha256 of json.dumps(model_breakdown(...), sort_keys=True) for the program each other encoding
# writes for test_per_task_codes' random design of the seed, frozen 2026-11-08 at 64ab25c,
# before the replay learned per-tree codes.
_OTHER_REPLAY_SHAS = {
    ("joint", 0): "e9fd593ccdf73ea48e8d280d75b890d89f8161695071c1781f86d20ee59b7a19",  # 2026-11-14 key pruning (2.3): depth 10 -> 9, blocks 36 -> 31
    ("disjoint", 0): "84a71640e61d60e7e09e356f38fe6297bbc072c81afba97fa0c71c3ce93acf89",  # 2026-11-14 key pruning: same totals (9, 28)
    ("per_task", 0): "0df851c3acf7b61e1a8bd79f16ad0f9fd0f53807c5a7ef634b0c73925b1f3993",  # 2026-11-14 key pruning: same totals (9, 24)
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
    # Unchanged by decisions 1a / 7.1 option B (2026-11-14): these designs' register
    # schedules gain nothing from losing the block order.
    # The numbers only: the refusal keys (Task 12) are newer than the shas.
    replay = model_breakdown(program)
    refusal = replay.pop("refusal")
    assert replay.pop("refusal_reason") == refusal
    assert refusal == ("stages_over" if replay["stage_depth"] > 12 else None)
    breakdown = json.dumps(replay, sort_keys=True)
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


def test_the_primary_preset_is_independent_and_joint_off():
  from src.main import select_arms
  from src.training.config import arm_encoding
  assert [cfg.arm_slug(arm_encoding(arm)) for arm, cfg in select_arms("primary")] == [
      "independent", "joint-off"]


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


@pytest.mark.parametrize("error", ["RangeEntriesDoNotFit", "CrossbarKeyTooWide"])
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
  assert {n: first.user_attrs[n] for n in CONSTRAINTS} == violations("bus_bytes_over", 5)
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert not first.user_attrs["no_split_tree"]
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
  assert {n: first.user_attrs[n] for n in CONSTRAINTS} == violations("phv_containers_short", 4)
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
  assert {t.user_attrs["bus_bytes_over"] for t in trials} == {1}


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
  # The constraint of that name, magnitude 1 (decision 9.2), and the trial
  # table's column of that name.
  assert {n: first.user_attrs[n] for n in CONSTRAINTS} == violations("no_split_tree", 1)
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert result is not None
  rows = {row["number"]: row for row in result.trial_table}
  assert rows[0]["no_split_tree"] == 1 and rows[0]["feasible"] is False
  # the column is the trials' own attribute (the search also meets stumps by itself)
  assert {number for number, row in rows.items() if row["no_split_tree"]} == {
      t.number for t in trials if t.user_attrs.get("no_split_tree")}
  assert all(row["feasible"] is False for row in rows.values() if row["no_split_tree"])
  assert all(row[name] == 0 for row in rows.values() if row["feasible"] for name in CONSTRAINTS)


def test_every_arms_trial_table_has_one_column_per_constraint(tmp_path):
  # Decision 9.2 (controller ruling, Task 12 fix 1): the measured columns, then one
  # column per refusals.CONSTRAINTS name, in every arm.
  from src.training import row_artifacts as ra
  from tests.test_train_model_contract import _call
  header = ("number,params,feasible,acc_sel_app,acc_sel_ddos,blocks,stage_depth,"
            "b_app,c_app,b_ddos,c_ddos,tied," + ",".join(CONSTRAINTS))
  for encoding in (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING, PER_TREE_ENCODING):
    # 4000 rows: stumps are refused in every encoding (R6), and forests trained on
    # the default 300 are mostly stumps
    result = _call(encoding=encoding, n=4000)
    assert list(result.trial_table[0]) == header.split(",")
    ctx = ra.RowContext(str(tmp_path / encoding), "x", 35, "abc")
    path = ra.write_trial_table(ctx, "x_M035_s00_k01", result.trial_table)
    with open(path, encoding="utf-8", newline="") as handle:
      assert handle.readline().rstrip("\n") == header


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


def test_a_per_tree_search_with_no_feasible_trial_still_counts_its_stumps(tmp_path, monkeypatch):
  # Stumps need big leaves on 400 rows: restore a high leaf floor for this premise.
  import src.training.train_model as tm
  monkeypatch.setattr(tm, "MIN_SAMPLES_LEAF_MIN", 30)
  import csv
  rows = _all_infeasible_elimination(tmp_path, "joint-tree", 200)
  assert [row["infeasible"] != "" for row in rows] == [True]    # nothing feasible
  path = tmp_path / "trials" / (rows[0]["row_id"] + ".csv")
  assert path.is_file()
  with open(path, encoding="utf-8", newline="") as handle:
    table = list(csv.DictReader(handle))
  assert len(table) == 6
  assert {row["no_split_tree"] for row in table} == {"1"}
  assert {row["feasible"] for row in table} == {"False"}


def test_the_failed_per_tree_search_carries_the_table_on_the_exception():
  from src.training.config import TrainConfig
  from src.training.errors import NoFeasibleSolution
  from tests.test_train_model_contract import _call
  cfg = TrainConfig(n_trials=4, min_feasible_before_stop=3, lookback=2)
  with pytest.raises(NoFeasibleSolution) as e:
    # Seeded: whether all 4 trials draw a no-split tree depends on the draw
    # (unseeded, it failed 2 of 18 runs alone).
    _call(encoding=PER_TREE_ENCODING, cfg=cfg, max_blocks=200, n=240, optuna_seed=0)
  assert list(e.value.trial_table[0])[-len(CONSTRAINTS):] == list(CONSTRAINTS)
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


def test_a_placement_failure_is_an_infeasible_trial(monkeypatch):
  from p4model.errors import PlacementFailed
  from src.training import early_stopping

  def replacement(real, call):
    if call == 0:
      def refuse(*args, **kwargs):
        raise PlacementFailed("no placement")
      return refuse
    return real

  trials, result = _search(monkeypatch, replacement)
  first = trials[0]
  assert {n: first.user_attrs[n] for n in CONSTRAINTS} == violations("placement_failed", 1)
  assert first.values == [-1.0, -1.0, float("inf")]
  assert not early_stopping.is_feasible(first)
  assert len(trials) > 1                          # the search continued
  assert result is not None
