"""Per-tree codes (program.PER_TREE_ENCODING): the names, the two limits, the Spilled width
marker and the pin rule (p4model/per_tree.py) that generator and model both call; then the
generator's per-tree program, its one `encoding` argument, and rule A.

Spec docs/superpowers/specs/2026-11-06-per-tree-arm-design.md Sec 7, 9.2, 9.3 and its
amendments of 2026-11-08 (A1: rule A, A2: one encoding argument, A3: the pin rule is the
spill by range table). That the generator's per-tree text equals the investigations' harness
byte for byte on 477 stored designs is checked outside git
(results/encoding_evidence_2026_11/per_tree_arm/scripts/check_identity.py)."""
import hashlib
import json
import os
import pickle
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.p4_replay import model_breakdown, parse_program_text
from p4gen.switch_semantics import per_tree_class_indices, vote_winner
from p4model import lanes
from p4model.errors import ActionDataTooWide, CrossbarKeyTooWide, NoSplitTree
from p4model.lanes import Spilled
from p4model.per_tree import (action_bytes, base_pins, marked_key_bits, pins_for,
                              spill_table_fields, widths_of)
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, ENCODINGS, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING, TaskSpec,
                             feature_of_per_tree_field, per_tree_field)
from p4model.target import PER_TREE_MAX_ACTION_BYTES, PER_TREE_MAX_PIN8_FIELDS
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


def _widths(app=(), ddos=()):
  """{task: [{feature: bits}]} from per-tree lists of (feature, bits) pairs."""
  return {"app": [dict(t) for t in app], "ddos": [dict(t) for t in ddos]}


def _counted(counts, bits=3):
  """One app tree per row so that feature f has counts[f] narrow codes: tree i holds every
  feature with more than i codes."""
  trees = [{f: bits for f, n in counts.items() if n > i} for i in range(max(counts.values()))]
  return {"app": trees, "ddos": []}


def test_field_helpers_undo_each_other_with_digits_and_underscores():
  assert per_tree_field("app", 0, "flow_iat_max") == "tr_app_0_flow_iat_max"
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app_0") == "f"
  assert feature_of_per_tree_field("code_tk_app_f", "app") is None
  assert feature_of_per_tree_field("code_tr_ddos_2_f", "app") is None


def test_the_encoding_is_named_and_listed():
  assert PER_TREE_ENCODING == "per_tree" and PER_TREE_ENCODING in ENCODINGS
  assert PER_TREE_MAX_PIN8_FIELDS == 44 and PER_TREE_MAX_ACTION_BYTES == 112


def test_widths_are_interval_counts_minus_one_and_skip_featureless_entries():
  ivs = {"app": [{"f": [0, 5, 9], "g": [0]}], "ddos": [{"f": list(range(10))}]}
  assert widths_of(ivs) == {"app": [{"f": 2}], "ddos": [{"f": 9}]}


def test_44_narrow_fields_spill_nothing_and_45_spill_one_whole_feature_in_order():
  # 5 app + 4 ddos trees x features f0..f4 of 3 bits = 45 narrow fields; drop the last ddos
  # tree's f4 for 44
  full = _widths([{f"f{j}": 3 for j in range(5)}] * 5, [{f"f{j}": 3 for j in range(5)}] * 4)
  w45 = full
  w44 = _widths([{f"f{j}": 3 for j in range(5)}] * 5,
                [{f"f{j}": 3 for j in range(5)}] * 3 + [{f"f{j}": 3 for j in range(4)}])
  pins_44, spilled_44 = pins_for(w44)
  assert spilled_44 == []
  assert sum(p == [8] for p in pins_44.values()) == 44
  pins_45, spilled_45 = pins_for(w45)
  # every feature holds 9 codes, so the first alphabetically goes, all of its codes together,
  # tasks app then ddos, trees ascending
  assert spilled_45 == [per_tree_field("app", i, "f0") for i in range(5)] + \
                       [per_tree_field("ddos", i, "f0") for i in range(4)]
  assert all(pins_45[n] == [16] for n in spilled_45)
  assert sum(p == [8] for p in pins_45.values()) == 36
  assert len(pins_45) == 45


def test_a_spill_takes_the_smallest_code_count_at_or_above_the_need():
  widths = _counted({"a": 1, "b": 2, "c": 4})        # 7 narrow codes
  pins = base_pins(widths)
  # need 3: a+b is exactly 3, c alone would be 4
  assert spill_table_fields(widths, pins, 4) == [per_tree_field("app", 0, "a"),
                                                  per_tree_field("app", 0, "b"),
                                                  per_tree_field("app", 1, "b")]
  # need 4: c alone (4) ties a+b+... only at its own count, and the fewest features win
  assert spill_table_fields(widths, pins, 3) == [per_tree_field("app", i, "c") for i in range(4)]
  # need 1: a (1 code) is the smallest total at or above it
  assert spill_table_fields(widths, pins, 6) == [per_tree_field("app", 0, "a")]


def test_a_spill_prefers_fewer_features_then_the_alphabet_on_equal_counts():
  widths = _counted({"a": 1, "b": 2, "c": 3})        # 6 narrow codes
  pins = base_pins(widths)
  # need 3: {c} and {a,b} both total 3; one feature beats two
  assert spill_table_fields(widths, pins, 3) == [per_tree_field("app", i, "c") for i in range(3)]
  tie = _counted({"d": 2, "b": 2})                   # need 2: b and d tie, b is first
  assert spill_table_fields(tie, base_pins(tie), 2) == [per_tree_field("app", i, "b") for i in range(2)]
  nothing = spill_table_fields(widths, pins, 6)
  assert nothing == []


def test_the_spill_leaves_wider_fields_and_other_features_alone():
  widths = _widths([{"f": 3, "g": 12, "h": 3}] * 3)
  pins = base_pins(widths)
  # 6 narrow codes, 5 allowed: f (3 codes) is the smallest total at or above 1
  assert spill_table_fields(widths, pins, 5) == [per_tree_field("app", i, "f") for i in range(3)]
  assert pins[per_tree_field("app", 0, "g")] == [16]       # a 12-bit field is H already, never 'narrow'


def test_too_many_features_to_enumerate_raises():
  widths = _widths([{f"f{j:02d}": 3 for j in range(21)}] * 3)
  with pytest.raises(ValueError):
    spill_table_fields(widths, base_pins(widths), 10)


def test_112_action_bytes_pass_and_113_raise():
  # 28 trees with one 32-bit field on f: 28 x 4 = 112 bytes
  ok = _widths([{"f": 32}] * 28)
  pins, spilled = pins_for(ok)
  assert spilled == [] and action_bytes(ok, pins) == {"f": 112}
  bad = _widths([{"f": 32}] * 28 + [{"f": 8}])
  with pytest.raises(ActionDataTooWide) as e:
    pins_for(bad)
  assert e.value.args[1] == 113


def test_action_bytes_count_every_slice_of_a_multi_container_field():
  widths = _widths([{"f": 33, "g": 3}], [{"f": 49}])
  pins = base_pins(widths)
  assert pins["tr_app_0_f"] == [8, 32] and pins["tr_ddos_0_f"] == [32, 32]
  # f: B + W (1 + 4) in the app tree, W + W (8) in the ddos tree; g: one B
  assert action_bytes(widths, pins) == {"f": 13, "g": 1}


def test_spilled_fields_count_as_the_16_bit_containers_they_are_pinned_to():
  widths = _counted({"a": 1, "b": 2})
  pins = base_pins(widths)
  assert action_bytes(widths, pins) == {"a": 1, "b": 2}
  pins = {**pins, per_tree_field("app", 0, "b"): [16]}
  assert action_bytes(widths, pins) == {"a": 1, "b": 3}


def test_a_tree_with_no_splits_raises():
  with pytest.raises(NoSplitTree) as e:
    pins_for(_widths([{"f": 3}, {}]))
  assert e.value.args[1] == "app_1"
  with pytest.raises(NoSplitTree):
    pins_for(_widths([{"f": 3}], [{}]))


def test_marked_key_bits_mark_exactly_the_narrow_fields_pinned_wide():
  widths = _widths([{"a": 3, "b": 12, "c": 5}])
  pins = base_pins(widths)
  assert marked_key_bits("app", 0, widths["app"][0], pins) == (3, 5, 12)
  assert not any(isinstance(b, Spilled) for b in marked_key_bits("app", 0, widths["app"][0], pins))
  pins[per_tree_field("app", 0, "c")] = [16]
  bits = marked_key_bits("app", 0, widths["app"][0], pins)
  assert bits != (3, 5, 12)                    # the marked width is not equal to the plain one
  assert [isinstance(b, Spilled) for b in bits] == [False, True, False]
  assert [int(b) for b in bits] == [3, 5, 12]
  # a 12-bit field pinned to its natural H is not a spill
  assert not isinstance(bits[2], Spilled)


def test_spilled_prices_as_h_and_stays_apart_from_a_plain_width():
  assert lanes.key_bytes([Spilled(3)]) == lanes.bytes_from_layout([3], lambda w: [("H", w)])
  assert hash(Spilled(3)) != hash(3) and Spilled(3) != 3 and 3 != Spilled(3)
  assert Spilled(3) == Spilled(3) and not Spilled(3) != Spilled(3)
  assert {Spilled(3): 1}.get(3) is None


def test_a_mixed_key_prices_the_same_in_any_order_and_after_a_pickle():
  a, b = (Spilled(3), 3, 20), (20, 3, Spilled(3))
  assert lanes.table_blocks(a) == lanes.table_blocks(b)
  assert sorted(lanes.key_bytes(a)) == sorted(lanes.key_bytes(b))
  assert isinstance(pickle.loads(pickle.dumps(Spilled(3))), Spilled)
  assert pickle.loads(pickle.dumps(Spilled(3))) == Spilled(3)


def test_a_spilled_field_is_one_crossbar_byte_of_the_h_kind():
  # ceil(3/8) == 1 byte either way; only the lane kind differs
  assert [kind for kind, _, _ in lanes.key_bytes([Spilled(3)] * 3)] == ["H"] * 3
  assert [kind for kind, _, _ in lanes.key_bytes([3] * 3)] == ["B"] * 3


def test_bare_widths_price_as_before():
  for key in [(3,), (3, 3, 20), (1, 2, 3, 40, 52), (27, 52), (9, 9, 9, 9)]:
    assert lanes.table_blocks(key) == lanes.table_blocks(list(reversed(key)))
    assert lanes.key_bytes(key) == lanes.bytes_from_layout(list(key), lanes.key_layout(list(key)))
  assert lanes.table_blocks([]) == lanes.table_blocks(())


def test_a_spilled_key_too_wide_still_raises():
  # a 512-bit field is 64 crossbar bytes; one spilled byte more is past what a stage supplies
  with pytest.raises(CrossbarKeyTooWide) as e:
    lanes.table_blocks((Spilled(3), 512))
  assert e.value.args[1] == 65
  assert lanes.table_blocks((512,)) >= 1


# --- the generator's per-tree program ---------------------------------------------

def _specs(app, ddos, names_app=FEATURES, names_ddos=FEATURES, intervals_app=None,
           intervals_ddos=None):
  return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), names_ddos,
                   _own(ddos, names_ddos) if intervals_ddos is None else intervals_ddos),
          TaskSpec(APP_TASK, 3, list(app.estimators_), names_app,
                   _own(app, names_app) if intervals_app is None else intervals_app)]


def _deep_specs(n_app=2):
  """Deep app trees on one feature: codes wider than 64 bits, codes that fill their pins and
  narrow ones (the entries tests' fixture too)."""
  app, ddos = _deep_forest(3, 0, depth=9, n_trees=n_app), _deep_forest(2, 1, depth=3)
  return _specs(app, ddos, DEEP_FEATURES, DEEP_FEATURES)


def _program(tmp_path, tasks, name="tree.p4", encoding=PER_TREE_ENCODING, **kwargs):
  path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename=name,
                                 encoding=encoding, **kwargs)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


def _pinned(tasks):
  """{resolved field: (code width, pinned width)} for every per-tree code of the design."""
  _, fields, trees = bps.resolve_per_tree_plan(tasks)
  pins, _ = pins_for(widths_of(trees))
  return {name: (len(iv) - 1, sum(pins[name])) for name, (_, iv, _) in fields.items()}


def _declared(width, pinned):
  """Rule A: a code wider than 8 bits is declared at its pinned width."""
  return pinned if width > 8 else width


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
    # program order: app's trees, then ddos', each tree index ascending
    assert written == [("%s_%d" % (t, i), per_tree_field(t, i, feature), len(trees[t][i][feature]) - 1)
                       for t in (APP_TASK, DDOS_TASK) for i in range(len(trees[t]))
                       if feature in trees[t][i]]
  assert list(fields) == [field for _, written in tables.values() for _, field, _ in written]
  for t in (APP_TASK, DDOS_TASK):
    for i, tree in enumerate(trees[t]):
      for f, iv in tree.items():
        assert fields[per_tree_field(t, i, f)] == (f, iv, {t})


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
        w.rsplit("_", 1)[0] for w, _, _ in tables[feature][1]))


def test_a_feature_one_tree_splits_gets_one_parameter(tmp_path):
  # one tree per task, on features the other task never selects: every feature has one writer
  app = _forest(3, 2, FEATURES[:2], n_trees=1, depth=3)
  ddos = _forest(2, 1, FEATURES[2:4], n_trees=1, depth=3)
  tasks = _specs(app, ddos, FEATURES[:2], FEATURES[2:4])
  tables, _, _ = bps.resolve_per_tree_plan(tasks)
  pinned = _pinned(tasks)
  text = _program(tmp_path, tasks)
  assert len(tables) == 4
  for feature, (_, written) in tables.items():
    (writer, field, width), = written
    assert ("action set_code_%s (bit<%d> code_%s) {\n        meta.code_%s = code_%s;\n    }"
            % (feature, _declared(width, pinned[field][1]), writer, field, writer)) in text


def test_a_feature_only_one_task_splits_writes_only_its_trees(tmp_path):
  app = _forest(3, 2, FEATURES[:2], depth=3)
  ddos = _forest(2, 1, FEATURES[2:4], depth=3)
  tasks = _specs(app, ddos, FEATURES[:2], FEATURES[2:4])
  _, _, trees = bps.resolve_per_tree_plan(tasks)
  text = _program(tmp_path, tasks)
  for task, features in ((APP_TASK, FEATURES[:2]), (DDOS_TASK, FEATURES[2:4])):
    for feature in features:
      params = re.search(r"action set_code_%s \(([^)]*)\)" % feature, text)[1]
      assert re.findall(r"code_(\w+)", params) == [
          "%s_%d" % (task, i) for i, own in enumerate(trees[task]) if feature in own]


def test_a_task_with_no_trees_generates(tmp_path):
  app = _forest(3, 2)
  tasks = [TaskSpec(DDOS_TASK, 2, [], None, {}),
           TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, _own(app))]
  text = _program(tmp_path, tasks)
  assert "get_classification_tree_app_0" in text and "code_tr_app_0_" in text
  assert "code_tr_ddos_" not in text and "get_classification_tree_ddos" not in text
  assert "vote_ddos" not in text


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
          "code_" + per_tree_field(t, i, f) for f in sorted(own)]


def test_every_code_field_is_declared_and_pragmas_keep_the_harness_order(tmp_path):
  app, ddos = _forest(3, 2, n_trees=2), _forest(2, 1)
  tasks = _specs(app, ddos)
  _, _, trees = bps.resolve_per_tree_plan(tasks)
  pins, _ = pins_for(widths_of(trees))
  pinned = _pinned(tasks)
  text = _program(tmp_path, tasks)
  lines = text.split("\n")
  for name, (width, pin) in pinned.items():
    assert "\tbit<%d> code_%s;" % (_declared(width, pin), name) in lines
  assert not re.search(r"bit<\d+> code_(?!tr_)\w+;", text)
  # one (feature, task) with trees 0 and 1, in the order the harness rewrote the per-task
  # program's pragmas in: solitary 0, solitary 1, no_overlay 0, size 0, no_overlay 1, size 1
  feature = next(f for f in trees[APP_TASK][0] if f in trees[APP_TASK][1])
  a, b = ("ig_md.code_" + per_tree_field(APP_TASK, i, feature) for i in (0, 1))

  def size(field, i):
    return '@pa_container_size("ingress", "%s", %s)' % (
        field, ", ".join(map(str, pins[per_tree_field(APP_TASK, i, feature)])))
  expected = ['@pa_solitary("ingress", "%s")' % a, '@pa_solitary("ingress", "%s")' % b,
              '@pa_no_overlay("ingress", "%s")' % a, size(a, 0),
              '@pa_no_overlay("ingress", "%s")' % b, size(b, 1)]
  start = lines.index(expected[0])
  block = lines[start:start + len(expected)]
  assert block == expected


def test_spilled_fields_are_pinned_to_16(tmp_path):
  app, ddos = _forest(3, 2, n_trees=10, depth=3), _forest(2, 1, n_trees=10, depth=3)
  tasks = _specs(app, ddos)
  _, _, trees = bps.resolve_per_tree_plan(tasks)
  pins, spilled = pins_for(widths_of(trees))
  assert spilled
  text = _program(tmp_path, tasks)
  for name in spilled:
    assert '@pa_container_size("ingress", "ig_md.code_%s", 16)' % name in text
  emitted = re.findall(r'@pa_container_size\("ingress", "ig_md\.code_(tr_\w+)", ([\d, ]+)\)', text)
  assert {n: [int(s) for s in sizes.split(", ")] for n, sizes in emitted} == pins
  assert sum(sizes == "8" for _, sizes in emitted) == PER_TREE_MAX_PIN8_FIELDS


@pytest.mark.parametrize("deep", [False, True])
def test_everything_but_the_codes_matches_the_per_task_program(tmp_path, deep):
  """Registers, range tables, table order and sizes are the per-task design's, so a
  difference between the two encodings on p4c can only come from the code fields."""
  tasks = _deep_specs() if deep else _specs(_forest(3, 2), _forest(2, 1))
  tree = _program(tmp_path, tasks, "tree.p4")
  task = _program(tmp_path, tasks, "task.p4", encoding=PER_TASK_ENCODING)

  def without_codes(text):
    return [line for line in text.split("\n") if "code_tk_" not in line and "code_tr_" not in line
            and "set_code_" not in line and not re.match(r"^\s*\}\s*$", line)]
  assert without_codes(tree) == without_codes(task)
  assert parse_program_text(tree).sizes == parse_program_text(task).sizes


def test_the_generator_refuses_an_action_above_112_bytes(tmp_path):
  tasks = _deep_specs(n_app=20)
  _, _, trees = bps.resolve_per_tree_plan(tasks)
  widths = widths_of(trees)
  with pytest.raises(ActionDataTooWide) as e:
    _program(tmp_path, tasks)
  assert e.value.args[1] == max(action_bytes(widths, base_pins(widths)).values())
  assert e.value.args[1] > PER_TREE_MAX_ACTION_BYTES
  assert not (tmp_path / "tree.p4").exists()


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


def test_the_per_task_codes_argument_is_gone(tmp_path):
  tasks, _, _ = _three_tasks()
  with pytest.raises(TypeError):
    bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename="t.p4",
                            per_task_codes=True)
  app, ddos = _forest(3, 2), _forest(2, 1)
  with pytest.raises(TypeError):
    _two_task(tmp_path, app, ddos, FEATURES, (_own(app), _own(ddos)), per_task_codes=True)
  with pytest.raises(TypeError):
    bps.generate_table_entries(tasks, per_task_codes=True)


# --- rule A -------------------------------------------------------------------------

def _rule_a(tmp_path):
  tasks = _deep_specs()
  return tasks, _pinned(tasks), _program(tmp_path, tasks)


def test_a_code_wider_than_8_bits_is_declared_at_its_pinned_width(tmp_path):
  _, pinned, text = _rule_a(tmp_path)
  padded = {n: (w, pin) for n, (w, pin) in pinned.items() if w > 8 and pin != w}
  assert padded
  for name, (width, pin) in padded.items():
    assert "\tbit<%d> code_%s;\n" % (pin, name) in text
    assert "bit<%d> code_%s;" % (width, name) not in text


def test_its_action_parameter_is_widened_to_match(tmp_path):
  tasks, pinned, text = _rule_a(tmp_path)
  tables, _, _ = bps.resolve_per_tree_plan(tasks)
  assert any(w > 8 and pin != w for w, pin in pinned.values())
  for feature, (_, written) in tables.items():
    params = ", ".join("bit<%d> code_%s" % (_declared(*pinned[field]), writer)
                       for writer, field, _ in written)
    assert "action set_code_%s (%s) {" % (feature, params) in text


def test_its_key_reads_only_the_codes_own_bits(tmp_path):
  _, pinned, text = _rule_a(tmp_path)
  lines = text.split("\n")
  sliced = {n: w for n, (w, pin) in pinned.items() if 8 < w <= 64 and pin != w}
  assert sliced
  for name, width in sliced.items():
    assert "            meta.code_%s[%d:0] : ternary;" % (name, width - 1) in lines
    assert "meta.code_%s : ternary;" % name not in text


def test_a_code_wider_than_64_bits_is_keyed_as_slices_of_at_most_64(tmp_path):
  _, pinned, text = _rule_a(tmp_path)
  lines = text.split("\n")
  wide = {n: w for n, (w, pin) in pinned.items() if w > 64 and pin != w}
  assert wide
  for name, width in wide.items():
    want = ["            meta.code_%s[%d:%d] : ternary;" % (name, min(width, lo + 64) - 1, lo)
            for lo in range(0, width, 64)]
    assert len(want) >= 2
    start = lines.index(want[0])
    assert lines[start:start + len(want)] == want
    assert "meta.code_%s : ternary;" % name not in text


def test_a_code_of_8_bits_or_one_filling_its_containers_is_unchanged(tmp_path):
  _, pinned, text = _rule_a(tmp_path)
  lines = text.split("\n")
  whole = {n: w for n, (w, pin) in pinned.items() if w <= 8 or pin == w}
  assert any(w <= 8 for w in whole.values()) and any(w > 8 for w in whole.values())
  for name, width in whole.items():
    assert "\tbit<%d> code_%s;" % (width, name) in lines
    assert "            meta.code_%s : ternary;" % name in lines
    assert "meta.code_%s[" % name not in text


def test_replay_prices_the_rule_a_program_as_the_unpadded_one(tmp_path):
  _, pinned, text = _rule_a(tmp_path)
  unpadded = text
  for name, (width, pin) in pinned.items():
    if bps.per_tree_key_slices(width, pin):
      unpadded = unpadded.replace("bit<%d> code_%s;" % (pin, name),
                                  "bit<%d> code_%s;" % (width, name))
      unpadded = re.sub(r"( +meta\.code_%s)\[\d+:0\]( : ternary;\n)"
                        r"(?: +meta\.code_%s\[\d+:\d+\] : ternary;\n)*" % (name, name),
                        r"\1\2", unpadded)
  assert unpadded != text
  assert not any("[" in line for line in unpadded.split("\n") if line.endswith(": ternary;"))
  program = parse_program_text(text)
  for name, (width, _) in pinned.items():
    assert program.bits["code_" + name] == width
  assert program.tables == parse_program_text(unpadded).tables
  assert model_breakdown(program) == model_breakdown(parse_program_text(unpadded))


# --- the control-plane entries -------------------------------------------------------
#
# Asserted on MEANING, like test_per_task_codes' entry tests: the entries are played as the
# switch plays them (a range table's action writes each tree's code, every tree table matches
# value/mask on each key slice) and must give every tree's own class. The slicing is rule A of
# the 2026-11 action-data-bus investigation; that the compiled program loads and runs these
# entries was checked live on tofino-model (results/encoding_evidence_2026_11/
# adb_investigation/step7_live).

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


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_generate_p4_program_writes_the_entries_of_its_encoding(tmp_path, encoding):
  tasks = _deep_specs() if encoding == PER_TREE_ENCODING else _entry_tasks(encoding, 0)
  _program(tmp_path, tasks, encoding=encoding, entries_filename="entries.json",
           use_default_action_discount=True)
  with open(tmp_path / "entries.json", encoding="utf-8") as handle:
    written = json.load(handle)
  assert written == bps.generate_table_entries(tasks, use_default_action_discount=True,
                                               encoding=encoding)
  assert written
  # each encoding addresses its own code parameters: code_<task> per task, code_<task>_<i> per tree
  params = {name for e in written if e["table_name"].startswith("table_")
            for name in e["action_params"]}
  per_tree = {name for name in params if re.match(r"^code_\w+_\d+$", name)}
  assert bool(per_tree) == (encoding == PER_TREE_ENCODING)
  if encoding == PER_TREE_ENCODING:
    assert written == bps.generate_per_tree_table_entries(tasks, use_default_action_discount=True)
  if encoding == PER_TASK_ENCODING:
    assert all(re.match(r"^code_(app|ddos)$", name) for name in params)


def _program_tables(text):
  """{table: [(bf_rt key name, match kind)]} of a program text, in key order."""
  return {name: [(bps.bfrt_key_name(expression), kind)
                 for expression, kind in re.findall(r"(meta\.[\w\[\]:]+)\s*:\s*(\w+);", block)]
          for name, block in re.findall(r"table (\w+) \{\s*key = \{(.*?)\}", text, re.S)}


def test_entry_key_names_are_the_programs_keys_as_bf_rt_names_them(tmp_path):
  tasks = _deep_specs()
  declared = _program_tables(_program(tmp_path, tasks))
  entries = bps.generate_table_entries(tasks, encoding=PER_TREE_ENCODING)
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
  # a range table is addressed by its raw value field
  assert all(kind == "range" for t in ranges for _, kind in declared[t])
  assert all(kind == "ternary" for t in trees for _, kind in declared[t])
  names = {name for t in trees for name, _ in declared[t]}
  # a whole key, a single slice and a code wider than 64 bits (two slices) are all addressed
  assert any(re.match(r"^code_tr_\w+_\d+_0_$", n) and not n.endswith("_63_0_") for n in names)
  assert any(n.endswith("_63_0_") for n in names)
  assert any(re.match(r"^code_tr_\w+_\d+_\d+_$", n) and not n.endswith("_0_") for n in names)
  assert any(n.startswith("code_tr_") and not re.search(r"_\d+_\d+_$", n) for n in names)


def test_entry_params_match_the_action_params_in_the_program_text(tmp_path):
  for tasks in (_deep_specs(), _specs(_forest(3, 2), _forest(2, 1))):
    text = _program(tmp_path, tasks)
    actions = {name: [(p.split()[1], int(p.split()[0][4:-1])) for p in params.split(",") if p.strip()]
               for name, params in re.findall(r"action (\w+)\s*\(([^)]*)\)", text)}
    entries = bps.generate_table_entries(tasks, use_default_action_discount=True,
                                         encoding=PER_TREE_ENCODING)
    assert entries
    for entry in entries:
      declared = actions[entry["action"]]
      assert list(entry["action_params"]) == [name for name, _ in declared], entry["action"]
      for (name, width), value in zip(declared, entry["action_params"].values()):
        assert 0 <= int(value) < (1 << width), (entry["action"], name, width, value)


def test_entries_with_one_task_having_no_trees(tmp_path):
  app = _forest(3, 2)
  tasks = [TaskSpec(DDOS_TASK, 2, [], None, {}),
           TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, _own(app))]
  declared = _program_tables(_program(tmp_path, tasks))
  entries = bps.generate_table_entries(tasks, encoding=PER_TREE_ENCODING)
  assert entries and {e["table_name"] for e in entries} <= set(declared)
  assert not any("ddos" in e["table_name"] or "ddos" in " ".join(e["action_params"])
                 for e in entries)
  X = np.random.RandomState(1).randint(0, 2100, size=(40, len(FEATURES)))
  want = per_tree_class_indices(app, X)
  for s, values in enumerate(X):
    got = _play(entries, dict(zip(FEATURES, values)))
    assert got == {"get_classification_tree_app_%d" % i: int(c) for i, c in enumerate(want[:, s])}


_SLICE = re.compile(r"^(code_tr_\w+?)_(\d+)_(\d+)_$")


def _play(entries, row):
  """{tree table: class} for one input, the entries interpreted as the switch does."""
  codes, by_table = {}, {}
  for entry in entries:
    by_table.setdefault(entry["table_name"], []).append(entry)
  for table, rows in by_table.items():
    if not table.startswith("table_"):
      continue
    feature = rows[0]["action"][len("set_code_"):]
    hits = [e for e in rows if int(e["key_fields"][feature + "_val"]["start"]) <= row[feature]
            <= int(e["key_fields"][feature + "_val"]["end"])]
    assert len(hits) == 1, (table, row)
    for param, value in hits[0]["action_params"].items():
      task, tree = re.match(r"^code_(app|ddos)_(\d+)$", param).groups()
      codes["code_" + per_tree_field(task, int(tree), feature)] = int(value)
  classes = {}
  for table, rows in by_table.items():
    if not table.startswith("get_classification_tree_"):
      continue

    def matches(entry):
      for name, spec in entry["key_fields"].items():
        m = _SLICE.match(name)
        field, hi, lo = (m[1], int(m[2]), int(m[3])) if m else (name, None, 0)
        value = codes[field] >> lo
        if hi is not None:
          value &= (1 << (hi - lo + 1)) - 1
        if value & int(spec["mask"], 16) != int(spec["value"], 16):
          return False
      return True
    hits = [e for e in rows if not e["is_default_action"] and matches(e)]
    assert len(hits) <= 1, (table, row)
    hits = hits or [e for e in rows if e["is_default_action"]]
    assert len(hits) == 1, (table, row)
    classes[table] = int(hits[0]["action_params"]["class"])
  return classes


def test_bfrt_key_name_keeps_a_slices_bounds():
  assert bps.bfrt_key_name("meta.code_tr_app_0_bwd_packet_length_mean[63:0]") == "code_tr_app_0_bwd_packet_length_mean_63_0_"
  assert bps.bfrt_key_name("meta.code_tr_app_0_bwd_packet_length_mean[89:64]") == "code_tr_app_0_bwd_packet_length_mean_89_64_"
  assert bps.bfrt_key_name("meta.code_flow_iat_max") == "code_flow_iat_max"
  assert bps.bfrt_key_name("meta.flow_iat_max_val") == "flow_iat_max_val"


@pytest.mark.parametrize("width, pinned, slices", [
    (3, 8, None),               # a narrow code keeps its padding: two may share a byte
    (8, 8, None),
    (27, 32, [(26, 0)]),        # declared bit<32>, keyed on its own 27 bits
    (32, 32, None),             # fills its container: nothing to pad
    (36, 40, [(35, 0)]),
    (64, 64, None),
    (74, 80, [(63, 0), (73, 64)]),   # p4c asserts on one slice wider than 64 bits
    (90, 96, [(63, 0), (89, 64)]),
    (200, 208, [(63, 0), (127, 64), (191, 128), (199, 192)]),
])
def test_per_tree_key_slices_is_rule_a(width, pinned, slices):
  assert bps.per_tree_key_slices(width, pinned) == slices


def test_fixture_has_codes_that_are_split_sliced_and_whole():
  names = {n for e in bps.generate_table_entries(_deep_specs(), encoding=PER_TREE_ENCODING)
           if e["table_name"].startswith("get_") for n in e["key_fields"]}
  assert any(n.endswith("_63_0_") for n in names), "no code wider than 64 bits"
  assert any(_SLICE.match(n) and not n.endswith("_63_0_") and n.split("_")[-2] == "0" for n in names), "no single slice"
  assert any(not _SLICE.match(n) for n in names), "no whole key"


@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_per_tree_entries_classify_every_input_exactly_as_each_tree(seed, discount):
  app, ddos = _deep_forest(3, seed, depth=9), _deep_forest(2, seed + 10, depth=4)
  tasks = _specs(app, ddos, DEEP_FEATURES, DEEP_FEATURES)
  entries = bps.generate_table_entries(tasks, use_default_action_discount=discount,
                                       encoding=PER_TREE_ENCODING)
  rng = np.random.RandomState(seed)
  X = np.vstack([np.array([[v] * len(DEEP_FEATURES) for v in (0, 1, 2500, 4999, 5000, 65535)]),
                 rng.randint(0, 5100, size=(250, len(DEEP_FEATURES)))])
  trees = {APP_TASK: per_tree_class_indices(app, X), DDOS_TASK: per_tree_class_indices(ddos, X)}
  for s, values in enumerate(X):
    got = _play(entries, dict(zip(DEEP_FEATURES, values)))
    want = {"get_classification_tree_%s_%d" % (t, i): int(c) for t in trees
            for i, c in enumerate(trees[t][:, s])}
    assert got == want
    assert vote_winner([got["get_classification_tree_app_%d" % i] for i in range(2)], 3) == \
        vote_winner(list(trees[APP_TASK][:, s]), 3)


def test_each_tree_writes_its_own_thermometer_code():
  app, ddos = _deep_forest(3, 2, depth=5), _deep_forest(2, 3, depth=3)
  tasks = _specs(app, ddos, DEEP_FEATURES, DEEP_FEATURES)
  own = {t.name: bps.per_tree_intervals(t) for t in tasks}
  for entry in bps.generate_table_entries(tasks, encoding=PER_TREE_ENCODING):
    if not entry["table_name"].startswith("table_"):
      continue
    feature = entry["action"][len("set_code_"):]
    x = int(entry["key_fields"][feature + "_val"]["start"])
    for param, value in entry["action_params"].items():
      task, tree = re.match(r"^code_(app|ddos)_(\d+)$", param).groups()
      iv = own[task][int(tree)][feature]
      idx = next(j for j, (lo, hi) in enumerate(iv) if lo <= x <= hi)
      assert int(value) == int(bps.thermometer_code(len(iv), idx), 2)
