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

DEEP_FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_total"]


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
# Re-frozen 2026-11-15 (Task 31, review A1: 12-bit @symmetric flow hash): every text is exactly
# its old text with Task 28's probe edit a1ab (Hash<bit<12>>, bit<12> flow_hash and register
# indices, no `& 0xFFF`, @symmetric pairs, raw hash inputs) plus @pa_solitary(now_pseudo_us)
# before `struct metadata_t` (impl/scripts/t31_text_diff.py, 14/14). Re-frozen again (review A2:
# no vote table for a one-tree task): every text is its pre-A1 text with the A1 edit, then the
# A2 edit (a one-tree task's tree action writes the export, its vote action / table / apply, class
# field and class pin removed; the template comment above the forwarding assignments) (14/14).
# And for review B1: each text the previous one with @name("<bfrt_key_name>") on every sliced
# key (Task 28's probe edit b1, 14/14). And for the review Part C clean-ups: the Tofino-2
# include and the empty /* CLASSIFICATION */ marker's blank lines gone, the timestamp written
# `ig_prsr_md.global_tstamp[25:10]` (14/14; each generated text == every edit applied to its
# pre-A1 text). And for @pa_solitary(flow_hash) (Task 32 addendum F3): each text the previous one
# plus that one pragma line (14/14). Re-frozen 2026-11-15 (Task 34: V1o 32-bit clock and clamp,
# C1f first-packet rule, Task 33b orientation fix): each text is its previous one with Task 33's
# probe edit c1f (the compiled, live-checked register bodies) and the orientation body of
# live_prog_orient (impl/scripts/t34_text_diff.py, 14/14). Re-frozen 2026-11-15 (Task 38, P3h:
# the pool's bwd_iat_mean is bwd_iat_total): each text is Task 37's P3h probe edit of its previous
# one (p37_probe.p3c, the coarse clock from ingress_mac_tstamp; the Mean's register becomes the
# {first, dur} pair, its chain's last-arrival register dropped) with bwd_iat_mean renamed, the
# now_coarse declaration aside (impl/scripts/t38_refreeze.py, 14/14; seed 3 has no Mean and kept
# its sha).
_OLD_FLAG_SHAS = {
    ("joint", 0): "df665d89b4d77b2f0050c101987b4328023f3fce5ce76a34715b14e7c4db4dac",
    ("disjoint", 0): "cbf4aa67c007ca1b12c3c04dea70bb61ea4f736330933765f6734c366797dbeb",
    ("per_task", 0): "6b2f91480338cdd0ca4b9038152d19f90cdf2fe2e0c38bbe5a86b23e35ee0b70",
    ("joint", 1): "8fbcb5e7f55bf3d1671fc49370e6de521988e6eb0fe209a87b6b8b621ffd5843",
    ("disjoint", 1): "4b8f57fa46baebcd6215a8f0a9b88abf223ff4dcda1cc25f76eeacbb40ecb73f",
    ("per_task", 1): "d4d6f04b880b4d093027b693423a25cfa596458659ca8138a13bd289f73eea09",
    ("joint", 2): "ccdb30379aa428b4761e89762f9341d62e1e5158343139efe6545a8a41765988",
    ("disjoint", 2): "d5a28e07f58bbdd4eea1681fbd7791def0c69a0b9ed1dd630b233b98b0ff1bab",
    ("per_task", 2): "2e8bbb75102deec840199df1c4f9404396008bf5421593d82bcbac1e3c2969cc",
    ("joint", 3): "b0d97c21b928b2ea5176d7b9e5aee60f83e17bc0ac3918210ddf2156eca74586",
    ("disjoint", 3): "72b1cd384fb38005e7e1268b0e036c9cc4740f2cc3990f40f243a50b72fa1833",
    ("per_task", 3): "45c14308c0e36994d73cf08f45faed79bd94105b98ec1f6673d7ef8f9f2ce15d",
    ("disjoint", "three"): "ffb811b04b1ef94e8991173d12d8a061c14d19083c33b18d1cca7fe11cfeef48",
    ("per_task", "three"): "feafa1cc4aedf214bc31b5777dd378d8a71fc06ec86972119e24747f8d4c7a69",
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
  field (code_pk_* too). Nothing on meta.fwd. Since review A1b, pa_solitary on
  now_pseudo_us in a program that computes the timestamp (calc_timestamp); since the
  Task 31 miss, pa_solitary on flow_hash in every program; since Task 38, on each clock
  calc_timestamp writes (now_pseudo_us for a last-arrival register, now_coarse for a
  duration)."""
  pins = {("pa_solitary", "flow_hash", "")}
  for clock in ("now_pseudo_us", "now_coarse"):
    if "meta.%s = " % clock in text:
      assert "calc_timestamp();" in text
      pins.add(("pa_solitary", clock, ""))
  for bits, field in _DECLARED.findall(text):
    if field.startswith("class_tree_"):
      pins.add(("pa_solitary", field, ""))
    elif field.endswith("_current_iat") or (field.endswith("_val") and not field.startswith("code_")):
      pins.add(("pa_container_size", field, ", 16"))
    elif field.startswith("code_"):
      pins.add(("pa_no_overlay", field, ""))
  return pins


@pytest.mark.parametrize("seed", [0, 1, 7])
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
  # the scratch pin is there (seeds 0 and 7 have an IAT chain; seed 1 only the Task 38
  # duration bwd_iat_total, which needs none) ...
  scratch = [field for field in declared if field.endswith("_current_iat")]
  assert bool(scratch) == (seed != 1)
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
  `encoding` replaced it (joint and disjoint were both flag off; the intervals decide).
  Re-frozen 2026-11-15 (Task 34, IAT thresholds in 1.024 us ticks): every record is its
  old one with an IAT feature's range bounds through build_p4_script.switch_range, the
  10 intervals whose bounds collapse onto one tick dropped, nothing else changed
  (impl/scripts/t34_entries.py, 36/36). Re-frozen 2026-11-15 (Task 38: the pool's
  bwd_iat_mean, an IAT feature, is the duration bwd_iat_total, installed unchanged): every
  record outside the bwd_iat_total range tables is its old one renamed (36/36), those tables'
  bounds are the trees' own, and the intervals the tick rule had collapsed are installed
  (seed 2: +1 / +2 entries) (impl/scripts/t38_refreeze.py)."""
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
    assert not (set(range(a, a + bits_a)) & set(range(b, b + bits_b)))   # bit offsets
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
# before the replay learned per-tree codes. Re-frozen 2026-11-15 (Task 31, review A1: hash
# prologue 3 -> 1 stage, program.FLOW_HASH_LEVEL): every breakdown is exactly its old one with
# every table's stage and the stage depth 2 lower, blocks unchanged (impl/scripts/
# t31_replay_shas.py, 18/18). Re-frozen again for review A2 (no vote epilogue for a one-tree
# task): 15/18 identical to the A1 breakdowns; seed 3 (one tree per task, all three encodings)
# has every table on the same stage and the depth 1 lower (6 -> 5). Re-frozen 2026-11-15 (Task 38,
# P3h: bwd_iat_mean -> bwd_iat_total, one register instead of the chain's two): seeds 0-2, which
# have it, run its range table(s) one stage earlier (4 -> 3) and every tree after them with it,
# stage depth 1 lower except joint seed 0 (7, its app_0 stays in stage 5; its four app trees take
# their 31 blocks in another order); blocks unchanged; seeds 3-5 identical
# (impl/scripts/t38_refreeze.py).
_OTHER_REPLAY_SHAS = {
    ("joint", 0): "03b34fc5643982368ac8c8e036084784c05352eea789838272638dfba1ee39d4",  # Task 38: 7 / 31
    ("disjoint", 0): "70141230a6935f5741fd9c77eed0b074a82b62b48715e0b2e09058217217eb54",  # Task 38: 6 / 28
    ("per_task", 0): "cdb0f981325ac339e4bebd223ee56f153dfd7b03aa9adc47aa077817fe454714",  # Task 38: 6 / 24
    ("joint", 1): "cfd48594c1a38b86ca25a95b8e1ca561085fba9420b71c8b6ab5797c927c76bc",  # Task 38: 6 / 7
    ("disjoint", 1): "c8a34fd77388dadf3b9e563513dd89d6482f3750c420727b7d4a2a8371b343e7",  # Task 38: 6 / 8
    ("per_task", 1): "cfd48594c1a38b86ca25a95b8e1ca561085fba9420b71c8b6ab5797c927c76bc",  # Task 38: 6 / 7
    ("joint", 2): "0a8dcd8d2099da84d36059ff583404fd99683c1e64e29122c219eae78731d8d0",  # Task 38: 11 / 91
    ("disjoint", 2): "15b1c0f4baacfe5d3fc30b2fdbfbdb91a31288fbd07ce3ef625df45e8ffea91c",  # Task 38: 8 / 60
    ("per_task", 2): "2c01ca7a2948710ddb6ac3fef7d61b46d14a535e132e50a4b39c1c83cfcf3311",  # Task 38: 8 / 57
    ("joint", 3): "0878d7fc5268df8ea6657c6e16f95e0a9512fddee02fe8cb7152a45fbaef3237",
    ("disjoint", 3): "870f08838561dc1b721b3e4bd340f0aba131ace8ed53fd68d042c54fd013346f",
    ("per_task", 3): "0878d7fc5268df8ea6657c6e16f95e0a9512fddee02fe8cb7152a45fbaef3237",
    ("joint", 4): "a9f1ec751c1d9b0fa78d97658db6952b86fae8fe6b5d1dcccfa25eb1ec7c633f",
    ("disjoint", 4): "a892dc0b270778793dfb359595f8de29201041a97df2c02a30a63a6898a2d90f",
    ("per_task", 4): "a9f1ec751c1d9b0fa78d97658db6952b86fae8fe6b5d1dcccfa25eb1ec7c633f",
    ("joint", 5): "7db4a695d130b81c359c7f9e2d0a4a748b32cb41ab4f9a8d3d95c4a18afe200e",  # 2026-11-13: joint pool simulated, same totals
    ("disjoint", 5): "41890e4495c30f4de8f98629d6c88dfef4671a1399b2aae6a0490e03fdae2ca0",
    ("per_task", 5): "46737d4ea7610eb82b540632a322b0f87e7bf0e88add9b5fa7cf9fa3081bc7ff",
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
