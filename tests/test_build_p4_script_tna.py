"""
Tests for Task B2b: wiring Tier-3 codeword splitting + the TNA register
generator into `build_p4_script.py`'s P4-generation pipeline.

Covers the four rewritten functions:
  - generate_P4_actions        (Tier-3 rewrite, num_trees_*==0 guards)
  - get_table_entries           (per-feature ternary key slicing)
  - generate_P4_tables_and_apply (multi-key classification tables,
                                   _val-suffixed feature tables)
  - generate_P4_code            (full rewire: registers + Tier-3 metadata)

Uses small synthetic feature_intervals/codewords/model stand-ins throughout
-- no real trained model or compiled P4 is needed (that's Task B3's job).
"""

import json
import os
import re
from itertools import product
from statistics import mode

import pytest

from src.p4gen import build_p4_script as bps
from src.p4gen.build_p4_script import (
    generate_P4_actions,
    generate_P4_tables_and_apply,
    generate_P4_code,
    generate_voting_code,
    get_table_entries,
    get_ternary_value_and_mask,
    most_common_class_and_dropped_codewords,
    OUTPUT_PATH,
)


INFINITE = bps.INFINITE


class _StubTree:
  """Minimal stand-in for one fitted sklearn DecisionTreeClassifier inside a
  forest. Only `.get_n_leaves()` is ever read by the code under test (the
  table-sizing follow-up's structural fallback, used when no
  `selected_features_*` list is available to recompute real codewords)."""

  DEFAULT_N_LEAVES = 5

  def __init__(self, n_leaves=DEFAULT_N_LEAVES):
    self._n_leaves = n_leaves

  def get_n_leaves(self):
    return self._n_leaves


class _StubClassifier:
  """Minimal stand-in for a trained sklearn RandomForestClassifier: only
  `.estimators_`'s length -- and, since the table-sizing follow-up, each
  estimator's `.get_n_leaves()` -- is ever read by the code under test."""

  def __init__(self, num_trees, n_leaves=_StubTree.DEFAULT_N_LEAVES):
    self.estimators_ = [_StubTree(n_leaves) for _ in range(num_trees)]


# ---------------------------------------------------------------------------
# Task 3: _resolve_disjoint_feature_plan
# ---------------------------------------------------------------------------

def test_resolve_disjoint_feature_plan_shares_identical_intervals():
    # Both models select "flow_iat_max" with the SAME intervals -- must
    # resolve to ONE shared (non-namespaced) entry, exactly like today's
    # joint-encoding behavior.
    shared = {"flow_iat_max": [(0, 100), (101, 65535)]}
    plan = bps._resolve_disjoint_feature_plan(shared, shared)
    assert list(plan.keys()) == ["flow_iat_max"]
    raw_name, intervals, models = plan["flow_iat_max"]
    assert raw_name == "flow_iat_max"
    assert models == {"app", "ddos"}


def test_resolve_disjoint_feature_plan_namespaces_differing_intervals():
    # Both models select "flow_iat_max" but with DIFFERENT intervals
    # (genuine disjoint encoding) -- must resolve to two namespaced,
    # independent entries.
    app_intervals = {"flow_iat_max": [(0, 50), (51, 65535)]}
    ddos_intervals = {"flow_iat_max": [(0, 200), (201, 65535)]}
    plan = bps._resolve_disjoint_feature_plan(app_intervals, ddos_intervals)
    assert set(plan.keys()) == {"app_flow_iat_max", "ddos_flow_iat_max"}
    assert plan["app_flow_iat_max"][0] == "flow_iat_max"
    assert plan["app_flow_iat_max"][2] == {"app"}
    assert plan["ddos_flow_iat_max"][0] == "flow_iat_max"
    assert plan["ddos_flow_iat_max"][2] == {"ddos"}


def test_resolve_disjoint_feature_plan_handles_model_exclusive_features():
    # A feature selected by only one model needs no namespacing at all.
    app_intervals = {"fwd_packet_length_max": [(0, 1500), (1501, 65535)]}
    ddos_intervals = {}
    plan = bps._resolve_disjoint_feature_plan(app_intervals, ddos_intervals)
    assert list(plan.keys()) == ["fwd_packet_length_max"]
    assert plan["fwd_packet_length_max"][2] == {"app"}


# ---------------------------------------------------------------------------
# generate_P4_actions
# ---------------------------------------------------------------------------

FEATURE_INTERVALS_2F = {
    "Flow_IAT_Max": [(0, 5), (6, 10), (11, INFINITE)],          # width 2
    "Fwd_Packet_Length_Max": [(0, 3), (4, INFINITE)],           # width 1
}


def test_generate_P4_actions_zero_app_trees_does_not_raise_and_omits_app_action():
  # num_trees_app == 0 used to raise ValueError (math domain error) from
  # math.log2(0). Confirm it no longer raises and emits no app action text.
  action_templates = generate_P4_actions(
      FEATURE_INTERVALS_2F, num_trees_app=0, num_trees_ddos=1,
      bit_per_classes_app=0, bit_per_classes_ddos=1,
  )
  assert "classify_flow_codeword_app" not in action_templates
  assert "classify_flow_codeword_ddos" in action_templates


def test_generate_P4_actions_zero_ddos_trees_does_not_raise_and_omits_ddos_action():
  action_templates = generate_P4_actions(
      FEATURE_INTERVALS_2F, num_trees_app=1, num_trees_ddos=0,
      bit_per_classes_app=1, bit_per_classes_ddos=0,
  )
  assert "classify_flow_codeword_app" in action_templates
  assert "classify_flow_codeword_ddos" not in action_templates


def test_generate_P4_actions_both_tasks_present_and_tier3_action_bodies():
  action_templates = generate_P4_actions(
      FEATURE_INTERVALS_2F, num_trees_app=2, num_trees_ddos=1,
      bit_per_classes_app=2, bit_per_classes_ddos=1,
  )

  assert "classify_flow_codeword_app" in action_templates
  assert "classify_flow_codeword_ddos" in action_templates

  # No leftover Tier-2 bit-slice placeholders/output.
  assert "<END_BIT>" not in action_templates
  assert "<INIT_BIT>" not in action_templates
  assert "meta.codeword[" not in action_templates

  # Tier 3: each feature's action body is a plain field assignment, not a
  # bit-slice into a combined codeword.
  assert "action set_code_flow_iat_max (bit<2> code) {" in action_templates
  assert "meta.code_flow_iat_max = code;" in action_templates
  assert "action set_code_fwd_packet_length_max (bit<1> code) {" in action_templates
  assert "meta.code_fwd_packet_length_max = code;" in action_templates


# ---------------------------------------------------------------------------
# Task M2-B2: num_trees > 1 classify-action fix (per-tree dedicated actions)
# ---------------------------------------------------------------------------
#
# TNA rejects a runtime "if (tree == i) {...}" branch inside an action when
# the branch decides which of several DIFFERENT metadata fields gets written
# (a rejected IR::Mux over an action-data-parameter conditional -- see
# af64bc2 and reviews/t11_tofino_port_and_env.md Part G.2). M1 special-cased
# num_trees==1 to skip the branch; M2 (num_trees_app=3) generalizes the fix
# by giving every tree its own dedicated, unconditional-write action instead
# -- validated against the real TNA compiler in
# p4/tofino_spike/tna_m2_numtrees3_spike.p4.

def test_generate_P4_actions_multi_tree_app_emits_one_dedicated_action_per_tree():
  action_templates = generate_P4_actions(
      FEATURE_INTERVALS_2F, num_trees_app=3, num_trees_ddos=0,
      bit_per_classes_app=2, bit_per_classes_ddos=0,
  )

  # Exactly 3 distinct, unconditional per-tree actions -- no shared,
  # tree-parameterized action.
  for i in range(3):
    assert "action classify_flow_codeword_app_"+str(i)+"(bit<2> class){" in action_templates
    assert "meta.class_tree_app_"+str(i)+" = class;" in action_templates

  # No `tree` action-data parameter anywhere (the old shared-action
  # signature took two params, "tree" and "class"; the new per-tree actions
  # take only "class" -- no comma-separated second parameter at all, and no
  # feature action takes two params either, so this also holds across the
  # whole file's action set). Note: "tree" as a substring legitimately
  # appears in metadata field names (meta.class_tree_app_0), so check for
  # the removed *parameter*/*branch* forms specifically, not the bare word.
  assert ", bit<" not in action_templates
  assert "bit<2> tree" not in action_templates
  assert "> tree," not in action_templates

  # No conditional/branch of any kind in the classify actions -- the
  # TNA-rejected pattern this task removes.
  assert "if (tree ==" not in action_templates
  assert "if (" not in action_templates

  # The old shared, unsuffixed action name must be gone entirely.
  assert "action classify_flow_codeword_app(" not in action_templates


def test_generate_P4_actions_num_trees_app_1_now_named_with_tree_suffix():
  # Regenerelizing away the num_trees==1 special case is an intentional
  # naming change: the single-tree action is now named with an explicit
  # "_0" suffix (classify_flow_codeword_app_0), not the old unsuffixed
  # classify_flow_codeword_app used by M1's special-cased branch.
  action_templates = generate_P4_actions(
      FEATURE_INTERVALS_2F, num_trees_app=1, num_trees_ddos=1,
      bit_per_classes_app=1, bit_per_classes_ddos=1,
  )

  assert "action classify_flow_codeword_app_0(bit<1> class){" in action_templates
  assert "meta.class_tree_app_0 = class;" in action_templates
  assert "action classify_flow_codeword_app(" not in action_templates

  assert "action classify_flow_codeword_ddos_0(bit<1> class){" in action_templates
  assert "meta.class_tree_ddos_0 = class;" in action_templates
  assert "action classify_flow_codeword_ddos(" not in action_templates

  # Still unconditional -- no if/Mux, matching M1's validated shape.
  assert "if (" not in action_templates
  assert "bit<1> tree" not in action_templates


# ---------------------------------------------------------------------------
# get_table_entries
# ---------------------------------------------------------------------------

def _decode_ternary(spec, width):
  """Invert get_ternary_value_and_mask: recover the original bit-string
  ('0'/'1'/'*' per position) from a table_entries.json ternary key_fields
  spec ({"value": "0xV", "mask": "0xM"}), given the known bit width of the
  slice it was generated from."""
  value_bits = bin(int(spec["value"], 16))[2:].zfill(width)
  mask_bits = bin(int(spec["mask"], 16))[2:].zfill(width)
  return "".join(
      "*" if m == "0" else v
      for v, m in zip(value_bits, mask_bits)
  )


# 2 features, widths 3 and 1 (total codeword width 4) -- width 3 keeps the
# hex round-trip exact (a whole number of nibbles isn't required, but this
# keeps the decode arithmetic easy to eyeball).
GTE_FEATURE_INTERVALS = {
    "Feature_A": [(0, 10), (11, 20), (21, 30), (31, INFINITE)],  # width 3
    "Feature_B": [(0, 5), (6, INFINITE)],                        # width 1
}

# Minimal stub: get_table_entries computes `features_involved` from this
# but never actually uses it -- an empty per-tree dict is sufficient.
GTE_PATHS_LEAF_NODES = {0: {}}


def test_get_table_entries_slices_codeword_per_feature_single_model(tmp_path):
  codeword = "101*"  # Feature_A chunk "101" (width 3), Feature_B chunk "*" (width 1)
  codewords = {0: {codeword: "1.0"}}

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries.json",
  )

  with open(tmp_path / "table_entries.json") as f:
    entries = json.load(f)

  # Task M2-B2: per-tree dedicated action name (tree_idx 0 here), not the
  # old shared unsuffixed "classify_flow_codeword".
  classification_entries = [e for e in entries if e["action"] == "classify_flow_codeword_0"]
  assert len(classification_entries) == 1
  entry = classification_entries[0]

  # `tree` is no longer a runtime action parameter (it's implicit in which
  # per-tree action/table is used) -- action_params holds only the class,
  # under bf_rt's real action-data field name for classify_flow_codeword_*
  # (the literal `class` parameter generate_P4_actions emits).
  assert entry["action_params"] == {"class": "1"}
  assert entry["is_default_action"] is False
  assert entry["priority"] == 0

  # One NAMED ternary key field per feature (bf_rt's real
  # add_with_<action>(<field>=..., <field>_mask=...) convention), keyed on
  # the same meta.code_<feature> name generate_P4_tables_and_apply declares.
  assert set(entry["key_fields"]) == {"code_feature_a", "code_feature_b"}

  # Round-trip: decoding each ternary field against its feature's known
  # width must recover exactly that feature's slice of the original
  # codeword.
  decoded_a = _decode_ternary(entry["key_fields"]["code_feature_a"], width=3)
  decoded_b = _decode_ternary(entry["key_fields"]["code_feature_b"], width=1)
  assert decoded_a == "101"
  assert decoded_b == "*"
  assert decoded_a + decoded_b == codeword


def test_get_table_entries_multi_model_offset_still_slices_per_feature(tmp_path):
  # offset=1 -> tree_idx 0 is "app", tree_idx 1 is "ddos" (tree_idx - offset == 0).
  codeword_app = "010*"
  codeword_ddos = "*11*"
  codewords = {
      0: {codeword_app: "0.0"},
      1: {codeword_ddos: "1.0"},
  }

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=1, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_offset.json",
  )

  with open(tmp_path / "table_entries_offset.json") as f:
    entries = json.load(f)

  # Task M2-B2: per-tree dedicated action names (tree_idx 0 in both the app
  # and ddos groups here, since ddos's tree_idx 1 minus offset 1 == 0), not
  # the old shared unsuffixed "classify_flow_codeword_app"/"_ddos".
  app_entries = [e for e in entries if e["action"] == "classify_flow_codeword_app_0"]
  ddos_entries = [e for e in entries if e["action"] == "classify_flow_codeword_ddos_0"]
  assert len(app_entries) == 1
  assert len(ddos_entries) == 1

  assert app_entries[0]["table_name"] == "get_classification_tree_app_0"
  assert ddos_entries[0]["table_name"] == "get_classification_tree_ddos_0"

  # `tree` is no longer a runtime action parameter -- action_params holds
  # only the single class value.
  assert app_entries[0]["action_params"] == {"class": "0"}
  assert ddos_entries[0]["action_params"] == {"class": "1"}

  for entry, original in ((app_entries[0], codeword_app), (ddos_entries[0], codeword_ddos)):
    assert set(entry["key_fields"]) == {"code_feature_a", "code_feature_b"}
    decoded = (_decode_ternary(entry["key_fields"]["code_feature_a"], width=3)
               + _decode_ternary(entry["key_fields"]["code_feature_b"], width=1))
    assert decoded == original


def test_get_table_entries_feature_range_entries_use_named_start_end_fields(tmp_path):
  # Section 1 (feature-value -> codeword-bit tables): one NAMED range key
  # field per entry, which bf_rt's real add_with_<action> exposes as the two
  # kwargs <field>_start / <field>_end. The field name is the same
  # meta.<raw_feature>_val name generate_P4_tables_and_apply keys the range
  # table on (see the naming-consistency test further below).
  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, {0: {}},
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_ranges.json",
  )
  with open(tmp_path / "table_entries_ranges.json") as f:
    entries = json.load(f)

  range_entries = [e for e in entries if e["table_name"].startswith("table_")]
  assert len(range_entries) == 4 + 2  # Feature_A has 4 intervals, Feature_B has 2
  for entry in range_entries:
    assert entry["is_default_action"] is False
    assert len(entry["key_fields"]) == 1
    field_name, spec = next(iter(entry["key_fields"].items()))
    assert field_name in ("feature_a_val", "feature_b_val")
    assert set(spec) == {"start", "end"}
    assert int(spec["start"]) <= int(spec["end"])

  # Feature_A's four intervals, in the order this function writes them, with
  # their real boundaries and their real code values.
  feature_a = [e for e in range_entries if e["table_name"] == "table_0_feature_a"]
  assert [e["key_fields"]["feature_a_val"] for e in feature_a] == [
      {"start": "31", "end": str(INFINITE)},
      {"start": "21", "end": "30"},
      {"start": "11", "end": "20"},
      {"start": "0", "end": "10"},
  ]
  # 3 codeword bits for 4 intervals, highest interval first: 000, 001, 011,
  # 111 -> 0, 1, 3, 7.
  assert [e["action_params"] for e in feature_a] == [
      {"code": "0"}, {"code": "1"}, {"code": "3"}, {"code": "7"}]
  # Priority is that entry's own 0-indexed position within its own table.
  assert [e["priority"] for e in feature_a] == [0, 1, 2, 3]
  assert [e["priority"] for e in range_entries
          if e["table_name"] == "table_1_feature_b"] == [0, 1]


# ---------------------------------------------------------------------------
# Task 7: Planter-style default-action discount
# ---------------------------------------------------------------------------
#
# Planter RF_EB's table_generator.py drops the single most common leaf/class
# from each tree's classification table, relying on the table's
# default_action instead (default_vote = max(collect_votes,
# key=collect_votes.count), table_generator.py:408-431). use_default_action_discount
# ports the same discount into this project's own get_table_entries/
# generate_P4_tables_and_apply, opt-in (default False, byte-identical to
# the pre-Task-7 output confirmed by every test above still passing
# unmodified).

def test_most_common_class_and_dropped_codewords_drops_every_matching_leaf():
    # class 0 has THREE leaves ("000", "010", "100"); class 1 has ONE ("001").
    # Old (wrong) behavior would return just one of the three class-0 codewords;
    # the fix must return all three.
    tree_codewords = {"000": 0, "001": 1, "010": 0, "100": 0}
    class_value, dropped = bps.most_common_class_and_dropped_codewords(tree_codewords)
    assert class_value == 0
    assert set(dropped) == {"000", "010", "100"}


def test_most_common_class_and_dropped_codewords_single_leaf_class():
    # Degenerate case: only one leaf has the majority class -- must still work
    # (this is the case the OLD implementation already handled correctly).
    tree_codewords = {"000": 0, "001": 1, "010": 1}
    class_value, dropped = bps.most_common_class_and_dropped_codewords(tree_codewords)
    assert class_value == 1
    assert dropped == ["001", "010"] or set(dropped) == {"001", "010"}


def test_get_table_entries_default_action_discount_omits_every_majority_class_leaf(tmp_path):
  # 3 leaves for tree 0: class 0 wins (two leaves: "101*" and "111*"),
  # class 1 has one ("010*"). The corrected discount must omit BOTH
  # class-0 leaves (not just one), leaving 1 explicit classification
  # entry (down from 3) -- only the class-1 leaf remains explicit.
  codewords = {0: {"101*": 0, "010*": 1, "111*": 0}}
  _, dropped_codewords = most_common_class_and_dropped_codewords(codewords[0])

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_discount.json",
      use_default_action_discount=True,
  )

  with open(tmp_path / "table_entries_discount.json") as f:
    entries = json.load(f)

  classification_entries = [e for e in entries
                            if e["action"] == "classify_flow_codeword_0"
                            and not e["is_default_action"]]
  assert len(classification_entries) == 1   # 3 leaves minus both class-0 default-action leaves

  written_codewords = {
      _decode_ternary(entry["key_fields"]["code_feature_a"], width=3)
      + _decode_ternary(entry["key_fields"]["code_feature_b"], width=1)
      for entry in classification_entries
  }
  assert written_codewords.isdisjoint(dropped_codewords)


def test_get_table_entries_discount_writes_one_default_action_record_per_tree(tmp_path):
  # The load-bearing half of the discount under the real bf_rt schema: the
  # dropped leaves' class is not lost, it becomes ONE default-action record
  # per tree's table, which p4/deploy_table_entries.py installs via bf_rt's
  # real set_default_with_<action>. Without it, every flow whose codeword
  # was discounted away would hit no entry at all and go unclassified.
  codewords = {
      0: {"101*": 0, "010*": 1, "111*": 0},   # app tree 0 -> majority class 0
      1: {"000*": 1, "011*": 1, "110*": 0},   # ddos tree 0 -> majority class 1
  }
  expected_default_class = {}
  for tree_id in codewords:
    class_value, _ = most_common_class_and_dropped_codewords(codewords[tree_id])
    expected_default_class[tree_id] = str(int(float(class_value)))

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=1, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_defaults.json",
      use_default_action_discount=True,
  )
  with open(tmp_path / "table_entries_defaults.json") as f:
    entries = json.load(f)

  defaults = [e for e in entries if e["is_default_action"]]
  assert len(defaults) == 2
  by_table = {e["table_name"]: e for e in defaults}
  assert set(by_table) == {"get_classification_tree_app_0",
                           "get_classification_tree_ddos_0"}

  app_default = by_table["get_classification_tree_app_0"]
  ddos_default = by_table["get_classification_tree_ddos_0"]
  assert app_default["action"] == "classify_flow_codeword_app_0"
  assert ddos_default["action"] == "classify_flow_codeword_ddos_0"
  assert app_default["action_params"] == {"class": expected_default_class[0]}
  assert ddos_default["action_params"] == {"class": expected_default_class[1]}

  # A default-action record has no key concept at all: bf_rt's real
  # set_default_with_<action> takes only data fields (no key fields, no
  # $MATCH_PRIORITY).
  for entry in defaults:
    assert entry["key_fields"] == {}
    assert entry["priority"] is None

  # ...and the same table's explicit entries really did lose the dropped
  # codewords they now cover.
  for tree_id, table_name in ((0, "get_classification_tree_app_0"),
                              (1, "get_classification_tree_ddos_0")):
    _, dropped = most_common_class_and_dropped_codewords(codewords[tree_id])
    explicit = [e for e in entries
                if e["table_name"] == table_name and not e["is_default_action"]]
    assert len(explicit) == len(codewords[tree_id]) - len(dropped)
    written = {
        _decode_ternary(e["key_fields"]["code_feature_a"], width=3)
        + _decode_ternary(e["key_fields"]["code_feature_b"], width=1)
        for e in explicit
    }
    assert written.isdisjoint(dropped)


def test_get_table_entries_discount_off_by_default_writes_every_leaf(tmp_path):
  # Regression guard: omitting use_default_action_discount entirely must
  # still write all 3 leaves AND emit no default-action record anywhere --
  # the new capability is genuinely opt-in.
  codewords = {0: {"101*": 0, "010*": 1, "111*": 0}}

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_no_discount.json",
  )

  with open(tmp_path / "table_entries_no_discount.json") as f:
    entries = json.load(f)

  classification_entries = [e for e in entries if e["action"] == "classify_flow_codeword_0"]
  assert len(classification_entries) == 3
  assert [e for e in entries if e["is_default_action"]] == []
  # Every record carries the flag explicitly, so a consumer never has to
  # guess from its absence.
  assert all("is_default_action" in e for e in entries)


def test_get_table_entries_discount_all_leaves_same_class_leaves_only_the_default(tmp_path):
  # Degenerate but real: every leaf of a tree carries the same class, so the
  # discount drops all of them. The tree must still be fully represented --
  # by its single default-action record, which classifies every flow.
  codewords = {0: {"101*": 1, "010*": 1, "111*": 1}}

  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_all_dropped.json",
      use_default_action_discount=True)
  with open(tmp_path / "table_entries_all_dropped.json") as f:
    entries = json.load(f)

  tree_records = [e for e in entries
                  if e["table_name"] == "get_classification_tree_0"]
  assert len(tree_records) == 1
  assert tree_records[0]["is_default_action"] is True
  assert tree_records[0]["action_params"] == {"class": "1"}


def test_get_table_entries_empty_tree_writes_nothing_at_all(tmp_path):
  # A tree with no codewords at all must not produce a default-action record
  # carrying a made-up class (most_common_class_and_dropped_codewords would
  # raise on an empty dict, so this also guards the guard).
  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, {0: {}},
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_empty_tree.json",
      use_default_action_discount=True)
  with open(tmp_path / "table_entries_empty_tree.json") as f:
    entries = json.load(f)

  assert [e for e in entries if e["table_name"].startswith("get_classification")] == []


def test_get_ternary_value_and_mask_splits_value_and_mask():
  # '*' positions are wildcards: 0 in both value and mask; fixed bits are
  # themselves in the value and 1 in the mask. Hex strings, zero-padded to
  # the codeword's nibble count -- the same computation the old combined
  # "0xV&&&0xM" helper did, just returned separately so callers can feed
  # bf_rt's real <field>= / <field>_mask= kwargs directly.
  assert get_ternary_value_and_mask("101*") == ("0xA", "0xE")
  assert get_ternary_value_and_mask("101") == ("0x5", "0x7")
  assert get_ternary_value_and_mask("****") == ("0x0", "0x0")
  assert get_ternary_value_and_mask("11111111") == ("0xFF", "0xFF")


def test_get_table_entries_priority_restarts_per_tree(tmp_path):
  # $MATCH_PRIORITY is per TABLE, and each tree has its own table -- a global
  # running counter would hand tree 1's first entry a priority of len(tree 0),
  # which is wrong (harmless only because this project's entries provably
  # never overlap, but still not what the field means).
  codewords = {
      0: {"101*": 0, "010*": 1, "111*": 2},
      1: {"000*": 1, "011*": 0},
  }
  get_table_entries(
      GTE_PATHS_LEAF_NODES, GTE_FEATURE_INTERVALS, codewords,
      offset=1, path_to_output=str(tmp_path) + os.sep,
      output_filename="table_entries_priority.json",
  )
  with open(tmp_path / "table_entries_priority.json") as f:
    entries = json.load(f)

  for table_name, expected in (("get_classification_tree_app_0", [0, 1, 2]),
                               ("get_classification_tree_ddos_0", [0, 1])):
    assert [e["priority"] for e in entries
            if e["table_name"] == table_name] == expected


def test_range_table_key_field_name_matches_generated_p4_exactly(tmp_path):
  # The seam a naming-convention drift would silently break: nothing raises
  # if generate_P4_tables_and_apply declares `meta.flow_iat_max_val` while
  # get_table_entries names the same key field something else -- the two
  # would just talk past each other, and every real bf_rt insertion would
  # fail at deploy time with an unknown-field error. Generate both halves
  # from the SAME feature_intervals and compare by string equality.
  feature_intervals = {"Flow_IAT_Max": [(0, 50), (51, 900), (901, INFINITE)]}

  written_path = bps.generate_P4_code(
      0, 2, None, _tiny_ddos_forest(),
      feature_intervals_app={}, feature_intervals_ddos=feature_intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="key_name_seam.p4")
  with open(written_path) as f:
    p4_text = f.read()

  get_table_entries(
      GTE_PATHS_LEAF_NODES, feature_intervals, {0: {}},
      offset=None, path_to_output=str(tmp_path) + os.sep,
      output_filename="key_name_seam.json")
  with open(tmp_path / "key_name_seam.json") as f:
    entries = json.load(f)

  # Every range-matching key field the generated P4 actually declares...
  declared = re.findall(r"meta\.(\w+)\s*:\s*range;", p4_text)
  assert declared, "generated P4 declares no range-matching key at all"
  # ...and every range key field name table_entries.json actually writes.
  written = {name
             for e in entries if e["table_name"].startswith("table_")
             for name in e["key_fields"]}
  assert written == set(declared) == {"flow_iat_max_val"}

  # Same for the table the entries are addressed to: bf_rt looks the table
  # up by this exact name on the running program.
  declared_tables = set(re.findall(r"table\s+(table_\w+)\s*\{", p4_text))
  written_tables = {e["table_name"] for e in entries
                    if e["table_name"].startswith("table_")}
  assert written_tables == declared_tables == {"table_0_flow_iat_max"}


# ---------------------------------------------------------------------------
# generate_P4_tables_and_apply
# ---------------------------------------------------------------------------

def test_generate_P4_tables_and_apply_classification_table_uses_multikey_template():
  table_templates, apply_templates = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=1,
  )

  assert "get_classification_tree_ddos_0" in table_templates
  assert "meta.code_flow_iat_max : ternary;" in table_templates
  assert "meta.code_fwd_packet_length_max : ternary;" in table_templates

  # Not the old Tier-2 single combined-codeword key.
  assert "meta.codeword: ternary;" not in table_templates
  assert "meta.codeword : ternary;" not in table_templates


def test_generate_P4_tables_and_apply_feature_tables_key_on_val_suffix():
  table_templates, apply_templates = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=0,
  )

  assert "meta.flow_iat_max_val: range;" in table_templates
  assert "meta.fwd_packet_length_max_val: range;" in table_templates
  # Not keyed on the raw (non-tracked-value) field name.
  assert "meta.flow_iat_max: range;" not in table_templates
  assert "meta.fwd_packet_length_max: range;" not in table_templates


def test_generate_P4_tables_and_apply_zero_app_trees_produces_no_app_classification_table():
  table_templates, apply_templates = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=1,
  )
  assert "get_classification_tree_app" not in table_templates
  assert "classify_flow_codeword_app" not in table_templates
  assert "get_classification_tree_app" not in apply_templates


def test_generate_P4_tables_and_apply_zero_ddos_trees_produces_no_ddos_classification_table():
  table_templates, apply_templates = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=1, num_trees_ddos=0,
  )
  assert "get_classification_tree_ddos" not in table_templates
  assert "classify_flow_codeword_ddos" not in table_templates
  assert "get_classification_tree_ddos" not in apply_templates


def test_generate_P4_tables_and_apply_each_tree_table_references_its_own_action():
  # Task M2-B2: each per-tree table's <ACTIONS> substitution must reference
  # THAT tree's dedicated action, not one shared action name copy-pasted
  # into every table.
  table_templates, apply_templates = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=3, num_trees_ddos=2,
  )

  for i in range(3):
    assert "classify_flow_codeword_app_"+str(i)+";" in table_templates
  for i in range(2):
    assert "classify_flow_codeword_ddos_"+str(i)+";" in table_templates

  # The old shared, unsuffixed action names must not appear anywhere in the
  # generated table text.
  assert "classify_flow_codeword_app;" not in table_templates
  assert "classify_flow_codeword_ddos;" not in table_templates


def test_generate_P4_tables_and_apply_never_declares_a_default_action():
  # Real-compile A/B this session: the
  # `const default_action = classify_flow_codeword_<task>_<i>(<literal>);`
  # line Task 1 introduced cost +1 real pipeline stage for one modified
  # table and +2 for all four (9 -> 11), even though table_summary.log's own
  # "critical path length through the table dependency graph" stayed at 9 in
  # every variant -- a compiler placement artifact, not a resource shortage.
  # Deleting the line entirely restored 9 stages with identical
  # tcam/sram/gateway/map_ram. Planter's own shipped decision table
  # (P4/DT_performance_Iris_EB.p4) likewise never declares a
  # compile-time-constant default action; its real default class is installed
  # by the control plane at deploy time, which is what get_table_entries'
  # is_default_action records + p4/deploy_table_entries.py now do here.
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=2, num_trees_ddos=1)
  assert "default_action" not in table_templates
  # The marker line itself is gone from both classification templates, so it
  # can never leak into generated P4 as a literal string either.
  assert "<DEFAULT_ACTION>" not in table_templates


def test_generate_P4_tables_and_apply_exact_match_tables_declare_no_default_action_either():
  # The same removal must hold for resources/table_classification_exact.p4,
  # the other classification template.
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=1, num_trees_ddos=1,
      match_type='exact')
  assert "default_action" not in table_templates
  assert "<DEFAULT_ACTION>" not in table_templates


def test_generate_P4_tables_and_apply_rejects_removed_discount_parameters():
  # Regression guard: `codewords` / `use_default_action_discount` are really
  # GONE from this function's signature, not merely ignored -- a caller still
  # passing either must fail loudly rather than silently getting a table with
  # no default action when it believed it asked for one.
  with pytest.raises(TypeError):
    generate_P4_tables_and_apply(
        ["flow_iat_max"], 1, 0, codewords={0: {"101*": 0}})
  with pytest.raises(TypeError):
    generate_P4_tables_and_apply(
        ["flow_iat_max"], 1, 0, use_default_action_discount=True)


# ---------------------------------------------------------------------------
# generate_voting_code -- table-based rewrite (Task 2) + regression test
# ---------------------------------------------------------------------------
#
# generate_voting_code used to emit an if-cascade (one `if` block per
# combination of per-tree class predictions). Task 2 replaced it with a
# single exact-match P4 table: generate_voting_code now returns a 2-tuple
# (table_declaration_text, apply_call_text) instead of one if-cascade
# string. Both tests below lock in the table's *decisions* (not the string
# formatting), using statistics.mode() for tie-breaking, same as before --
# this is a mechanism change, not a behavior change.

_VOTING_ENTRY_RE = re.compile(
    r"\((?P<combo>[\d,\s]+)\) : set_classification_app\((?P<winner>\d+)\);"
)


def _parse_voting_table_decisions(table_decl):
  """Parse generate_voting_code's `const entries` table declaration back
  into a {(c0, c1, ..., c_{n-1}): winner} dict, keyed by per-tree class
  tuples in tree-index order (matching the key declaration order)."""
  decisions = {}
  for m in _VOTING_ENTRY_RE.finditer(table_decl):
    combo = tuple(int(c) for c in m.group("combo").split(","))
    decisions[combo] = int(m.group("winner"))
  return decisions


def test_generate_voting_code_emits_table_not_if_cascade():
  num_trees, num_classes = 3, 3
  result = bps.generate_voting_code(num_trees, num_classes, "app")

  assert isinstance(result, tuple) and len(result) == 2
  table_decl, apply_call = result

  assert "if (" not in table_decl
  assert "table vote_app {" in table_decl
  assert "meta.class_tree_app_0 : exact;" in table_decl
  assert "meta.class_tree_app_1 : exact;" in table_decl
  assert "meta.class_tree_app_2 : exact;" in table_decl
  assert "const entries" in table_decl

  decisions = _parse_voting_table_decisions(table_decl)
  expected = {
      combo: mode(combo)
      for combo in product(range(num_classes), repeat=num_trees)
  }
  assert len(expected) == num_classes ** num_trees == 27
  assert decisions == expected

  assert apply_call.strip() == "vote_app.apply();"


def test_generate_voting_code_table_size_is_the_exact_entry_count():
  # The table emits exactly num_classes ** num_trees const entries, so that
  # IS its size (reviews/p4_tofino_reference.md Sec 4.4: size must come from
  # the real logical entry count). A `max(32, ...)` floor used to be applied
  # as a conservative guard; it is not merely redundant but wrong -- see the
  # 1-bit key case below.
  table_decl_small, _ = bps.generate_voting_code(3, 3, "app")
  assert "size = 27;" in table_decl_small

  # 5 trees x 3 classes = 243 entries.
  table_decl_large, _ = bps.generate_voting_code(5, 3, "app")
  assert "size = 243;" in table_decl_large


def test_generate_voting_code_size_respects_a_one_bit_key_space():
  # The real M1/M2 DDoS config: 1 tree, 2 classes. The key is a single
  # bit<1> field, so the table can hold at most 2 entries -- and emits
  # exactly 2. Declaring 32 is unreachable by construction and the real
  # Tofino compiler rejects it outright:
  #   "warning: Shrinking table SwitchIngress.vote_ddos: with 1 match bits,
  #    can only have 2 entries"
  table_decl, _ = bps.generate_voting_code(1, 2, "ddos")

  assert "size = 2;" in table_decl
  assert "size = 32;" not in table_decl


def test_generate_voting_code_voting_decisions_match_statistics_mode_for_all_combos():
  num_trees, num_classes = 3, 3
  table_decl, _apply_call = bps.generate_voting_code(num_trees, num_classes, "app")

  decisions = _parse_voting_table_decisions(table_decl)

  expected = {
      combo: mode(combo)
      for combo in product(range(num_classes), repeat=num_trees)
  }

  assert len(expected) == num_classes ** num_trees == 27
  assert decisions == expected


# ---------------------------------------------------------------------------
# generate_P4_code -- end-to-end smoke test
# ---------------------------------------------------------------------------

# The real M1 (DDoS-only) feature set, hand-built small synthetic intervals
# (no real trained model needed for this unit test).
M1_FEATURE_INTERVALS = {
    "Flow_IAT_Max": [(0, 100), (101, 500), (501, INFINITE)],
    "Fwd_IAT_Max": [(0, 50), (51, INFINITE)],
    "Fwd_Packet_Length_Max": [(0, 64), (65, 1500), (1501, INFINITE)],
}

_OUTPUT_FILE = os.path.join(OUTPUT_PATH, "p4_code_RF_models.p4")

# Any remaining bare marker comment (splicing silently failed for that
# marker) or unresolved <PLACEHOLDER>-style token.
_MARKER_RE = re.compile(r"/\*\s*(METADATA|REGISTERS|REGISTER_ACTIONS|ACTIONS|TABLES|FEATURE_UPDATE_APPLY|APPLY|CLASSIFICATION)\s*\*/")
_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z_]*>")


@pytest.fixture(scope="module")
def m1_ddos_only_output():
  generate_P4_code(
      num_class_app=3, num_class_ddos=2,
      clf_app=None, clf_ddos=_StubClassifier(1),
      feature_intervals_app={}, feature_intervals_ddos=M1_FEATURE_INTERVALS,
  )
  with open(_OUTPUT_FILE, "r") as f:
    return f.read()


def test_generate_P4_code_no_markers_or_placeholders_remain(m1_ddos_only_output):
  assert not _MARKER_RE.search(m1_ddos_only_output), "a marker comment was left unspliced"
  assert not _PLACEHOLDER_RE.search(m1_ddos_only_output), "an unresolved <PLACEHOLDER> token remains"


def test_generate_P4_code_ddos_only_has_no_app_task_artifacts(m1_ddos_only_output):
  assert "classification_app" not in m1_ddos_only_output
  assert "class_tree_app_" not in m1_ddos_only_output
  assert "classify_flow_codeword_app" not in m1_ddos_only_output


def test_generate_P4_code_contains_expected_m1_feature_text(m1_ddos_only_output):
  for expected in (
      "flow_iat_max_val",
      "code_flow_iat_max",
      "flow_orientation_action",
      "set_code_flow_iat_max",
  ):
    assert expected in m1_ddos_only_output, "missing expected text: {0}".format(expected)


def test_generate_P4_code_ddos_task_artifacts_present(m1_ddos_only_output):
  assert "classification_ddos" in m1_ddos_only_output
  assert "class_tree_ddos_0" in m1_ddos_only_output
  assert "classify_flow_codeword_ddos" in m1_ddos_only_output


def test_generate_P4_code_ddos_single_tree_action_now_uses_tree_suffixed_name(m1_ddos_only_output):
  # Task M2-B2: the num_trees==1 special case is removed in favor of a
  # uniform per-tree-action loop, so the single DDoS tree's classify action
  # is now named with an explicit "_0" suffix -- a real, intentional naming
  # change from M1's already-compiled artifact, which used the old
  # unsuffixed "classify_flow_codeword_ddos" action name.
  assert "action classify_flow_codeword_ddos_0(bit<1> class){" in m1_ddos_only_output
  assert "action classify_flow_codeword_ddos(" not in m1_ddos_only_output


# ---------------------------------------------------------------------------
# Task 0: Injectable output path
# ---------------------------------------------------------------------------

def _tiny_ddos_forest():
  """Minimal stand-in for a trained DDoS classifier: single tree."""
  return _StubClassifier(1)


def _tiny_app_forest():
  """Minimal stand-in for a trained App classifier: two trees. Follows the
  same _StubClassifier convention as _tiny_ddos_forest() above -- neither
  generate_P4_code nor generate_P4_tables_and_apply ever reads tree
  internals, only len(clf.estimators_), so a real fitted RandomForestClassifier
  isn't needed for these tests."""
  return _StubClassifier(2)


# ---------------------------------------------------------------------------
# Task 3: single-pipeline generator for disjoint encoding
# ---------------------------------------------------------------------------

def test_generate_P4_code_disjoint_shares_when_intervals_match():
    # Both models pick the same intervals for a shared feature name --
    # must produce exactly ONE set_code_<feature> table, not two.
    clf_ddos = _tiny_ddos_forest()
    shared = {"Flow_IAT_Max": [(0, 100), (101, 65535)]}
    written_path = bps.generate_P4_code(
        0, 2, None, clf_ddos, feature_intervals_app=shared, feature_intervals_ddos=shared)
    with open(written_path) as f:
        text = f.read()
    # "set_code_flow_iat_max" legitimately appears twice for ANY single
    # working feature -- once as the action's own declaration, once as the
    # range table's <ACTIONS> reference to it (confirmed against
    # pre-Task-3 master: bare count is 2 there too, for exactly this
    # reason). The real "shared, not duplicated" invariant is that only
    # ONE such action is ever *declared*.
    assert text.count("action set_code_flow_iat_max") == 1
    assert "set_code_app_flow_iat_max" not in text
    assert "set_code_ddos_flow_iat_max" not in text


def test_generate_P4_code_disjoint_namespaces_when_intervals_differ():
    # Both models pick DIFFERENT intervals for the same feature name --
    # must produce two independent, namespaced set_code_ tables in the
    # SAME generated file (one combined pipeline, not two programs).
    clf_app = _tiny_app_forest()  # reuse this file's existing tiny-forest fixture helper
    clf_ddos = _tiny_ddos_forest()
    app_intervals = {"Flow_IAT_Max": [(0, 50), (51, 65535)]}
    ddos_intervals = {"Flow_IAT_Max": [(0, 200), (201, 65535)]}
    written_path = bps.generate_P4_code(
        3, 2, clf_app, clf_ddos,
        feature_intervals_app=app_intervals, feature_intervals_ddos=ddos_intervals)
    with open(written_path) as f:
        text = f.read()
    assert "set_code_app_flow_iat_max" in text
    assert "set_code_ddos_flow_iat_max" in text
    # exactly one raw-value register/table reading the underlying feature,
    # shared by both discretizations. A plain metadata-field count is NOT
    # enough to catch the regression this test guards against:
    # generate_P4_registers_and_apply silently SKIPS any feature name it
    # can't find in FEATURE_REGISTER_CATALOG (no error raised), so if a
    # future change fed it NAMESPACED names (e.g. "app_flow_iat_max") in
    # place of the deduplicated RAW name ("flow_iat_max"), the raw
    # register declaration and its populating .execute() call site would
    # both silently vanish from the generated text -- while the
    # `meta.flow_iat_max_val` FIELD declaration (emitted by a separate,
    # always-raw-keyed loop in generate_P4_code) would still be present,
    # keeping a bare substring-count check green. Assert both the field
    # declaration AND the register-populating call site appear exactly
    # once each, so either one vanishing (the real regression mode) or
    # either one being duplicated fails this test.
    assert text.count("bit<16> flow_iat_max_val;") == 1
    assert text.count(
        "meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);"
    ) == 1


def test_generate_P4_code_writes_to_injectable_path(tmp_path):
  clf_ddos = _tiny_ddos_forest()  # reuse this file's existing tiny-forest fixture helper
  feature_intervals = {"F": [(0, 100), (101, 65535)]}
  out_dir = str(tmp_path) + os.sep
  written_path = bps.generate_P4_code(
      num_class_app=0, num_class_ddos=2, clf_app=None, clf_ddos=clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=feature_intervals,
      output_dir=out_dir, output_filename="custom_name.p4",
  )
  assert written_path == out_dir + "custom_name.p4"
  assert os.path.exists(written_path)
  # default path must NOT have been touched by this call
  assert not os.path.exists(bps.OUTPUT_PATH + "custom_name.p4")


def test_generate_P4_code_default_path_unchanged():
  clf_ddos = _tiny_ddos_forest()
  feature_intervals = {"F": [(0, 100), (101, 65535)]}
  written_path = bps.generate_P4_code(
      num_class_app=0, num_class_ddos=2, clf_app=None, clf_ddos=clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=feature_intervals,
  )
  assert written_path == bps.OUTPUT_PATH + "p4_code_RF_models.p4"


# ---------------------------------------------------------------------------
# Task 8: Planter RF_EB-style exact-match code/decision tables
# ---------------------------------------------------------------------------
#
# match_type is a new, opt-in parameter on generate_P4_tables_and_apply (and
# generate_P4_code, which passes it straight through). Default 'ternary'
# must remain byte-identical to every test above (none of which pass
# match_type); 'exact' switches ONLY the classification tables'
# `: ternary;`/`: exact;` key text and template file -- feature-range
# tables stay `: range;` either way (Planter's RF_EB scheme keeps ternary
# feature tables, only code/decision tables move to exact).

def test_generate_P4_tables_and_apply_default_match_type_is_ternary():
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=1,
  )
  assert "meta.code_flow_iat_max : ternary;" in table_templates
  assert ": exact;" not in table_templates


def test_generate_P4_tables_and_apply_match_type_exact_uses_exact_keys():
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=1,
      match_type='exact',
  )
  assert "meta.code_flow_iat_max : exact;" in table_templates
  assert "meta.code_fwd_packet_length_max : exact;" in table_templates
  # No leftover ternary classification key text.
  assert ": ternary;" not in table_templates


def test_generate_P4_tables_and_apply_match_type_exact_leaves_feature_tables_range():
  # Feature-range tables are unaffected by match_type either way -- only
  # the classification tables' key kind changes.
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=0,
      match_type='exact',
  )
  assert "meta.flow_iat_max_val: range;" in table_templates
  assert "meta.fwd_packet_length_max_val: range;" in table_templates


def test_generate_P4_tables_and_apply_invalid_match_type_raises():
  with pytest.raises(ValueError):
    generate_P4_tables_and_apply(
        list(FEATURE_INTERVALS_2F.keys()), num_trees_app=0, num_trees_ddos=1,
        match_type='bogus',
    )


def test_generate_P4_code_match_type_exact_propagates_to_generated_p4(tmp_path):
  clf_ddos = _tiny_ddos_forest()
  feature_intervals = {"F": [(0, 100), (101, 65535)]}
  out_dir = str(tmp_path) + os.sep
  written_path = bps.generate_P4_code(
      num_class_app=0, num_class_ddos=2, clf_app=None, clf_ddos=clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=feature_intervals,
      output_dir=out_dir, output_filename="exact_match_test.p4",
      match_type='exact',
  )
  with open(written_path, "r") as f:
    generated = f.read()
  # The classification table's own field key must be exact...
  assert "meta.code_f : exact;" in generated
  # ...and no leftover ternary classification key text. (Note:
  # generate_voting_code's vote_ddos table also legitimately uses
  # ": exact;" on meta.class_tree_ddos_0 -- that table is unrelated to
  # match_type and always exact, so this test checks the classification
  # key specifically rather than asserting ": ternary;" is wholly absent.)
  assert "meta.code_f : ternary;" not in generated


def test_generate_P4_code_default_match_type_still_ternary(tmp_path):
  # Regression guard: generate_P4_code's own default (no match_type passed)
  # must remain byte-identical to every pre-Task-8 caller.
  clf_ddos = _tiny_ddos_forest()
  feature_intervals = {"F": [(0, 100), (101, 65535)]}
  out_dir = str(tmp_path) + os.sep
  written_path = bps.generate_P4_code(
      num_class_app=0, num_class_ddos=2, clf_app=None, clf_ddos=clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=feature_intervals,
      output_dir=out_dir, output_filename="ternary_default_test.p4",
  )
  with open(written_path, "r") as f:
    generated = f.read()
  assert "meta.code_f : ternary;" in generated
  # (generate_voting_code's vote_ddos table legitimately uses ": exact;" on
  # meta.class_tree_ddos_0 regardless of match_type -- see note above.)
  assert "meta.code_f : exact;" not in generated


# ---------------------------------------------------------------------------
# Follow-up (post-plan): use_default_action_discount wired into generate_P4_code
# ---------------------------------------------------------------------------
#
# Given each model's ORIGINAL ordered training-feature-name list,
# generate_P4_code recomputes per-model codewords. Those codewords drive the
# discount's two REAL effects: each classification table's declared `size`
# (the entries the control plane no longer installs) and get_table_entries'
# per-tree is_default_action record. What they must NOT drive any more is a
# `const default_action` line in the generated P4 -- see
# test_generate_P4_tables_and_apply_never_declares_a_default_action for the
# real-compile evidence behind that removal.
#
# Unlike _tiny_app_forest()/_tiny_ddos_forest() (bare _StubClassifier stand-ins,
# enough only because nothing read tree internals before), these tests need
# REAL fitted forests: sklearn's export_text is what turns a model into the
# tree text the codeword derivation parses.

_DISCOUNT_FEATURE_NAMES = ["Flow_IAT_Max", "Fwd_Packet_Length_Max"]


def _fit_real_forest(labels, seed, n_estimators, value_scale):
  """A really fitted, really splitting RandomForestClassifier over
  _DISCOUNT_FEATURE_NAMES, with integer thresholds (dt_thresholds_float_to_int,
  exactly as the production pipeline does before generating P4). value_scale
  controls the feature value range, so two forests fitted with different
  scales genuinely discretize the SAME feature name into different intervals
  -- which is what the disjoint-namespacing composition test below needs."""
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier

  rnd = np.random.RandomState(seed)
  X = rnd.randint(0, value_scale, size=(40, len(_DISCOUNT_FEATURE_NAMES)))
  y = np.array([labels[i % len(labels)] for i in range(40)])
  # make the labels learnable so the trees actually split (and so every tree
  # really has several leaves sharing a majority class)
  X[:, 0] += np.array([value_scale // 4 * (labels.index(v) + 1) for v in y])

  clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=3,
                               random_state=seed).fit(X, y)
  return bps.dt_thresholds_float_to_int(clf)


def _derive_intervals(clf):
  """Each model's OWN feature_intervals, via the same
  tree_nodes -> thresholds -> intervals path feature_selection and
  evaluation both use."""
  trees = bps.get_tree_textual_representation(clf, _DISCOUNT_FEATURE_NAMES)
  tree_nodes = {tree: bps.get_nodes(trees[tree]) for tree in trees}
  return bps.get_feature_intervals_from_thresholds(bps.get_feature_thresholds(tree_nodes))


def test_generate_P4_code_discount_emits_no_const_default_action_for_either_task(tmp_path):
  # Same fixture that previously produced one
  # `const default_action = classify_flow_codeword_<task>_<i>(<class>);` per
  # tree (2 app trees + 1 ddos tree, each with a real majority class, so the
  # discount genuinely triggers) -- the construct must now be absent
  # everywhere in the generated program.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)
  intervals_app = _derive_intervals(clf_app)
  intervals_ddos = _derive_intervals(clf_ddos)

  written_path = bps.generate_P4_code(
      3, 2, clf_app, clf_ddos,
      feature_intervals_app=intervals_app,
      feature_intervals_ddos=intervals_ddos,
      output_dir=str(tmp_path) + os.sep, output_filename="discount_on.p4",
      use_default_action_discount=True,
      selected_features_app=_DISCOUNT_FEATURE_NAMES,
      selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    text = f.read()

  # Precondition: this fixture really does exercise the discount (otherwise
  # the assertion below would pass vacuously).
  for clf, intervals in ((clf_app, intervals_app), (clf_ddos, intervals_ddos)):
    for tree_codewords in _codewords_of(clf, intervals).values():
      _, dropped = most_common_class_and_dropped_codewords(tree_codewords)
      assert len(dropped) >= 1

  assert "const default_action" not in text
  assert "default_action" not in text
  assert "<DEFAULT_ACTION>" not in text
  # The discount's real effects are unaffected: tables still exist and are
  # still sized from their discounted codeword counts (covered by the table
  # sizing tests below).
  assert "get_classification_tree_app_0" in text
  assert "get_classification_tree_ddos_0" in text


def test_generate_P4_code_discount_via_config_object(tmp_path):
  # config must take precedence over the individual keyword argument, the
  # same way it already does for match_type.
  from src.p4gen import p4_gen_config

  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)
  intervals_ddos = _derive_intervals(clf_ddos)
  codewords = _codewords_of(clf_ddos, intervals_ddos)
  _, dropped = most_common_class_and_dropped_codewords(codewords[0])
  assert len(dropped) >= 1, "fixture no longer exercises a real discount"

  def _generate(filename, **extra):
    path = bps.generate_P4_code(
        0, 2, None, clf_ddos,
        feature_intervals_app={}, feature_intervals_ddos=intervals_ddos,
        output_dir=str(tmp_path) + os.sep, output_filename=filename,
        selected_features_ddos=_DISCOUNT_FEATURE_NAMES, **extra)
    with open(path) as f:
      return f.read()

  text = _generate("discount_config.p4",
                   config=p4_gen_config.P4GenConfig(use_default_action_discount=True))

  # The discount's remaining generated-P4 effect is the declared table size:
  # the entries it folds into the control-plane-set default action come off
  # the table's reservation. (It no longer emits any default_action line --
  # see test_generate_P4_tables_and_apply_never_declares_a_default_action.)
  assert _table_sizes(text)["get_classification_tree_ddos_0"] == \
      max(1, len(codewords[0]) - len(dropped))
  assert _table_sizes(_generate("no_discount_config.p4"))[
      "get_classification_tree_ddos_0"] == len(codewords[0])
  assert "default_action" not in text


def test_generate_P4_code_discount_ddos_only_needs_no_app_features(tmp_path):
  # clf_app is None -- there is no App model to recompute codewords for, so
  # selected_features_app must NOT be required.
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)
  written_path = bps.generate_P4_code(
      0, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=_derive_intervals(clf_ddos),
      output_dir=str(tmp_path) + os.sep, output_filename="discount_ddos_only.p4",
      use_default_action_discount=True,
      selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    text = f.read()

  assert "get_classification_tree_ddos_0" in text
  assert "classify_flow_codeword_app" not in text
  assert "default_action" not in text


def test_generate_P4_code_discount_without_selected_features_app_raises(tmp_path):
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)

  with pytest.raises(ValueError) as excinfo:
    bps.generate_P4_code(
        3, 2, clf_app, clf_ddos,
        feature_intervals_app=_derive_intervals(clf_app),
        feature_intervals_ddos=_derive_intervals(clf_ddos),
        output_dir=str(tmp_path) + os.sep, output_filename="missing_app_features.p4",
        use_default_action_discount=True,
        selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  assert "selected_features_app" in str(excinfo.value)


def test_generate_P4_code_discount_without_selected_features_ddos_raises(tmp_path):
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)

  with pytest.raises(ValueError) as excinfo:
    bps.generate_P4_code(
        3, 2, clf_app, clf_ddos,
        feature_intervals_app=_derive_intervals(clf_app),
        feature_intervals_ddos=_derive_intervals(clf_ddos),
        output_dir=str(tmp_path) + os.sep, output_filename="missing_ddos_features.p4",
        use_default_action_discount=True,
        selected_features_app=_DISCOUNT_FEATURE_NAMES)
  assert "selected_features_ddos" in str(excinfo.value)


def test_generate_P4_code_discount_composes_with_disjoint_namespacing(tmp_path):
  # Both models select the SAME two feature names but -- fitted on different
  # value scales -- genuinely discretize them into DIFFERENT intervals, so
  # Task 3's namespacing kicks in. Per-model codeword computation must then
  # run against each model's own intervals; a single combined call would
  # silently produce wrong codewords for one side.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)
  intervals_app = _derive_intervals(clf_app)
  intervals_ddos = _derive_intervals(clf_ddos)
  # Precondition of this test: the intervals really do differ per model.
  assert intervals_app["Flow_IAT_Max"] != intervals_ddos["Flow_IAT_Max"]

  written_path = bps.generate_P4_code(
      3, 2, clf_app, clf_ddos,
      feature_intervals_app=intervals_app, feature_intervals_ddos=intervals_ddos,
      output_dir=str(tmp_path) + os.sep, output_filename="discount_namespaced.p4",
      use_default_action_discount=True,
      selected_features_app=_DISCOUNT_FEATURE_NAMES,
      selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    text = f.read()

  # Namespacing still in force...
  assert "set_code_app_flow_iat_max" in text
  assert "set_code_ddos_flow_iat_max" in text
  # ...and every classification table on BOTH sides still got its discount --
  # visible now as its declared size, computed from THAT model's own per-model
  # codewords (recomputed here against that model's own intervals, exactly as
  # the generator must), never as a default_action line.
  sizes = _table_sizes(text)
  for clf, intervals, task in ((clf_app, intervals_app, "app"),
                               (clf_ddos, intervals_ddos, "ddos")):
    codewords = _codewords_of(clf, intervals)
    for tree_id in codewords:
      _, dropped = most_common_class_and_dropped_codewords(codewords[tree_id])
      assert sizes["get_classification_tree_{}_{}".format(task, tree_id)] == \
          max(1, len(codewords[tree_id]) - len(dropped))
  assert "default_action" not in text


# Byte-identical regression guard for the default (discount off) path: this
# sha256 pins the exact call below. It must not change -- if a future task
# legitimately alters generated output, that task re-records it deliberately
# rather than this one drifting silently.
#
# Re-recorded ONCE, deliberately, by the table-sizing follow-up: that task
# replaced the fixed `size = 200` / `size = 400` table literals with real,
# per-table entry counts, which changes this default call's output. The value
# it replaced is kept below as _PRE_TABLE_SIZING_OUTPUT_SHA256, and
# test_generate_P4_code_default_call_changes_only_table_sizes proves the size
# numbers are the ONLY thing that changed between the two.
# Re-recorded a SECOND time, deliberately, by the PHV-pinning change: each
# feature value field now carries an @pa_container_size pragma (real-compile
# evidence in the pinning tests at the end of this file). The size literals
# and those pragma lines remain the only deltas from the pre-table-sizing
# baseline -- _reconstruct_pre_table_sizing_text undoes both, and
# test_generate_P4_code_default_call_changes_only_table_sizes proves it.
# Re-recorded a THIRD time by dropping generate_voting_code's max(32, ...)
# size floor: the vote tables now declare their exact const-entry count.
_DEFAULT_CALL_OUTPUT_SHA256 = (
    "3b6f2cedde35f98149224be9e7c346d1b39369093d370a17cda0a2ce826c5af5")

# The sha256 the same call produced BEFORE the table-sizing follow-up (when
# every feature table declared `size = 200;` and every classification table
# `size = 400;`), originally recorded against the code as it stood before the
# discount was wired into generate_P4_code.
_PRE_TABLE_SIZING_OUTPUT_SHA256 = (
    "839a2cbb9753f29546d16b97b5841f14cdd67d7c2b5ece991d891b59cf2820c5")


def _sha256_of_default_generate_P4_code_call(tmp_path, filename, **extra):
  import hashlib

  written_path = bps.generate_P4_code(
      3, 2, _tiny_app_forest(), _tiny_ddos_forest(),
      feature_intervals_app={"Flow_IAT_Max": [(0, 50), (51, INFINITE)]},
      feature_intervals_ddos={"Flow_IAT_Max": [(0, 200), (201, INFINITE)],
                              "Fwd_Packet_Length_Max": [(0, 64), (65, INFINITE)]},
      output_dir=str(tmp_path) + os.sep, output_filename=filename, **extra)
  with open(written_path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()


def test_generate_P4_code_default_output_is_byte_identical_to_pre_wiring(tmp_path):
  assert _sha256_of_default_generate_P4_code_call(
      tmp_path, "baseline.p4") == _DEFAULT_CALL_OUTPUT_SHA256


# ---------------------------------------------------------------------------
# Follow-up (post-plan): every generated table sized from its REAL entry count
# ---------------------------------------------------------------------------
#
# generate_P4_tables_and_apply used to stamp two fixed literals into every
# table it emitted -- `size = 200;` for the range-matching feature tables and
# `size = 400;` for the classification tables -- completely disconnected from
# how many entries those tables can actually hold. That made every
# P4-generation-time entry-count optimization (the Planter-style
# default-action discount above, in particular) invisible to the real Tofino
# compiler's TCAM/SRAM reservation, which is what
# reviews/t12_tcam_model_experiment_plan.md's real before/after compile
# comparison found.

def _table_sizes(p4_text):
  """table name -> its declared `size = N;`, for every table in generated P4."""
  sizes = {}
  current_table = None
  for line in p4_text.splitlines():
    stripped = line.strip()
    name_match = re.match(r"table\s+(\w+)\s*\{", stripped)
    if name_match:
      current_table = name_match.group(1)
      continue
    size_match = re.match(r"size\s*=\s*(\d+)\s*;", stripped)
    if size_match and current_table is not None:
      sizes[current_table] = int(size_match.group(1))
      current_table = None
  return sizes


def _codewords_of(clf, intervals):
  """That model's per-tree {codeword: class} dicts (0-indexed tree ids), via
  the exact path generate_P4_code recomputes them with."""
  trees = bps.get_tree_textual_representation(clf, _DISCOUNT_FEATURE_NAMES)
  tree_nodes = {tree: bps.get_nodes(trees[tree]) for tree in trees}
  return bps.generate_codewords(bps.get_root_to_leaf_paths(tree_nodes), intervals)


# --- Part A: range-matching feature tables ---------------------------------

def test_generate_P4_code_feature_table_size_is_real_interval_count(tmp_path):
  # Each range-matching table gets exactly one entry per interval
  # (get_table_entries' section 1), so its size must be that count -- not 200.
  clf_ddos = _tiny_ddos_forest()
  intervals = {"Flow_IAT_Max": [(0, 50), (51, 120), (121, 900), (901, INFINITE)]}
  written_path = bps.generate_P4_code(
      0, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="feature_table_size.p4")
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  assert sizes["table_0_flow_iat_max"] == len(intervals["Flow_IAT_Max"]) == 4


def test_generate_P4_code_feature_table_sizes_are_per_feature(tmp_path):
  # Two features with DIFFERENT interval counts must get different table
  # sizes -- proving the size really tracks each table's own entry count.
  clf_ddos = _tiny_ddos_forest()
  intervals = {
      "Flow_IAT_Max": [(0, 50), (51, INFINITE)],
      "Fwd_Packet_Length_Max": [(0, 8), (9, 64), (65, 512), (513, 1024), (1025, INFINITE)],
  }
  written_path = bps.generate_P4_code(
      0, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="feature_table_sizes_differ.p4")
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  by_feature = {name: size for name, size in sizes.items() if name.startswith("table_")}
  assert sorted(by_feature.values()) == [2, 5]


def test_generate_P4_tables_and_apply_without_feature_table_sizes_keeps_literal():
  # Regression guard for DIRECT callers that never pass the new optional
  # dict: their output must stay byte-identical, i.e. still `size = 200;`.
  table_templates, _ = generate_P4_tables_and_apply(
      ["flow_iat_max"], 0, 1)
  assert _table_sizes(table_templates)["table_0_flow_iat_max"] == 200


def test_generate_P4_tables_and_apply_feature_table_sizes_are_used_when_given():
  table_templates, _ = generate_P4_tables_and_apply(
      ["flow_iat_max"], 0, 1, feature_table_sizes={"flow_iat_max": 7})
  assert _table_sizes(table_templates)["table_0_flow_iat_max"] == 7


# --- Part B: classification tables -----------------------------------------

def test_generate_P4_code_classification_size_is_real_codeword_count(tmp_path):
  # Discount OFF but selected_features_app GIVEN: codewords are computable,
  # so each classification table is sized at its EXACT entry count.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  intervals_app = _derive_intervals(clf_app)
  written_path = bps.generate_P4_code(
      3, 2, clf_app, None,
      feature_intervals_app=intervals_app, feature_intervals_ddos={},
      output_dir=str(tmp_path) + os.sep, output_filename="clf_size_codewords.p4",
      selected_features_app=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  codewords = _codewords_of(clf_app, intervals_app)
  for i in range(len(clf_app.estimators_)):
    expected = len(codewords[i])
    assert sizes["get_classification_tree_app_" + str(i)] == expected
    assert sizes["get_classification_tree_app_" + str(i)] != 400
    # the structural fallback is a real upper bound on the exact count
    assert expected <= clf_app.estimators_[i].get_n_leaves()


def test_generate_P4_code_classification_size_shrinks_under_discount(tmp_path):
  # Discount ON: the entries the control plane no longer has to install
  # (every leaf carrying the tree's majority class) must actually come off
  # the declared table size -- the whole point of this follow-up.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  intervals_app = _derive_intervals(clf_app)
  written_path = bps.generate_P4_code(
      3, 2, clf_app, None,
      feature_intervals_app=intervals_app, feature_intervals_ddos={},
      output_dir=str(tmp_path) + os.sep, output_filename="clf_size_discounted.p4",
      use_default_action_discount=True,
      selected_features_app=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  codewords = _codewords_of(clf_app, intervals_app)
  shrank_somewhere = False
  for i in range(len(clf_app.estimators_)):
    _, dropped = most_common_class_and_dropped_codewords(codewords[i])
    expected = max(1, len(codewords[i]) - len(dropped))
    assert sizes["get_classification_tree_app_" + str(i)] == expected
    if expected < len(codewords[i]):
      shrank_somewhere = True
  # precondition of this fixture: at least one tree really has >1 leaf on its
  # majority class, so the discount genuinely reduces a table's size.
  assert shrank_somewhere, "fixture no longer exercises a real discount reduction"


def test_generate_P4_code_classification_size_never_drops_below_one(tmp_path):
  # Degenerate but real case: a tree that splits, yet whose leaves ALL carry
  # the same class -- the discount then drops every single entry, and a naive
  # count would declare `size = 0;`, which P4 rejects.
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier

  rnd = np.random.RandomState(3)
  X = rnd.randint(0, 4000, size=(40, len(_DISCOUNT_FEATURE_NAMES)))
  y = np.zeros(40, dtype=int)
  y[:2] = 1
  clf_ddos = bps.dt_thresholds_float_to_int(
      RandomForestClassifier(n_estimators=1, max_depth=1, random_state=3).fit(X, y))
  intervals_ddos = _derive_intervals(clf_ddos)
  codewords = _codewords_of(clf_ddos, intervals_ddos)
  # precondition: every leaf really does carry the same class
  _, dropped = most_common_class_and_dropped_codewords(codewords[0])
  assert len(dropped) == len(codewords[0]) > 0

  written_path = bps.generate_P4_code(
      3, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=intervals_ddos,
      output_dir=str(tmp_path) + os.sep, output_filename="clf_size_degenerate.p4",
      use_default_action_discount=True,
      selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  assert sizes["get_classification_tree_ddos_0"] == 1


def test_generate_P4_code_classification_size_falls_back_to_leaf_count(tmp_path):
  # Discount OFF and selected_features_app NOT given at all: codewords are
  # not computable, so sizing falls back to the fitted tree's REAL leaf count
  # (a safe, never-underestimating upper bound on the codeword count) --
  # still a real number, never the old 400 literal.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  intervals_app = _derive_intervals(clf_app)
  written_path = bps.generate_P4_code(
      3, 2, clf_app, None,
      feature_intervals_app=intervals_app, feature_intervals_ddos={},
      output_dir=str(tmp_path) + os.sep, output_filename="clf_size_fallback.p4")
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  for i in range(len(clf_app.estimators_)):
    assert sizes["get_classification_tree_app_" + str(i)] == clf_app.estimators_[i].get_n_leaves()
    assert sizes["get_classification_tree_app_" + str(i)] != 400


def test_generate_P4_code_classification_size_ddos_uses_offset_tree_ids(tmp_path):
  # The DDoS tables must read their sizes at the same tree_id offset
  # (num_trees_app + i) the codewords dict itself is keyed with -- an
  # off-by-num_trees_app bug here would silently size a DDoS table from an
  # App tree.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  clf_ddos = _fit_real_forest([0, 1], seed=7, n_estimators=1, value_scale=900)
  intervals_app = _derive_intervals(clf_app)
  intervals_ddos = _derive_intervals(clf_ddos)
  written_path = bps.generate_P4_code(
      3, 2, clf_app, clf_ddos,
      feature_intervals_app=intervals_app, feature_intervals_ddos=intervals_ddos,
      output_dir=str(tmp_path) + os.sep, output_filename="clf_size_both_tasks.p4",
      selected_features_app=_DISCOUNT_FEATURE_NAMES,
      selected_features_ddos=_DISCOUNT_FEATURE_NAMES)
  with open(written_path) as f:
    sizes = _table_sizes(f.read())

  codewords_ddos = _codewords_of(clf_ddos, intervals_ddos)
  assert sizes["get_classification_tree_ddos_0"] == len(codewords_ddos[0])
  codewords_app = _codewords_of(clf_app, intervals_app)
  for i in range(len(clf_app.estimators_)):
    assert sizes["get_classification_tree_app_" + str(i)] == len(codewords_app[i])


def test_generate_P4_tables_and_apply_without_classification_sizes_keeps_literal():
  # The other half of the direct-caller regression guard: no new dicts
  # passed -> both old literals, exactly as before this follow-up.
  table_templates, _ = generate_P4_tables_and_apply(
      ["flow_iat_max"], 1, 1)
  assert _table_sizes(table_templates) == {
      "get_classification_tree_app_0": 400,
      "get_classification_tree_ddos_0": 400,
      "table_0_flow_iat_max": 200,
  }


def test_generate_P4_tables_and_apply_classification_sizes_are_used_when_given():
  table_templates, _ = generate_P4_tables_and_apply(
      ["flow_iat_max"], 2, 1,
      classification_table_sizes={0: 11, 1: 12, 2: 13})
  # tree_id 2 is the DDoS tree: keyed at num_trees_app + i, the same
  # convention codewords.get(...) already uses.
  assert _table_sizes(table_templates) == {
      "get_classification_tree_app_0": 11,
      "get_classification_tree_app_1": 12,
      "get_classification_tree_ddos_0": 13,
      "table_0_flow_iat_max": 200,
  }


def _reconstruct_pre_table_sizing_text(p4_text):
  """Undo every delta applied to the generated text since the pre-table-sizing
  baseline, so what remains must be byte-identical to that baseline:

    - put the OLD fixed literals back into the table declarations -- 200 for
      every range-matching feature table, 400 for every classification table,
      and 32 for every vote_* table. vote_* used to be `max(32, num_classes **
      num_trees)`, and both configs in this fixture (3 trees x 3 classes = 27,
      1 tree x 2 classes = 2) sat under that floor, so 32 is what the baseline
      declared for both; the floor was later dropped because a 1-bit key
      cannot hold 32 entries at all;
    - drop the @pa_container_size PHV-pinning pragma lines, added later to
      pin each feature value field to a 16-bit container (see the pinning
      tests at the end of this file for the real-compile evidence). The
      pragmas occupy whole lines of their own between MAX_NUM_FLOWS and
      `struct metadata_t`, so removing those lines restores the original
      spacing exactly.

  Keeping both undos in one helper is what lets the byte-identity test stay
  meaningful: it proves these are the ONLY things that ever changed."""
  rebuilt = []
  current_table = None
  for line in p4_text.splitlines(keepends=True):
    stripped = line.strip()
    if stripped.startswith("@pa_container_size("):
      continue
    name_match = re.match(r"table\s+(\w+)\s*\{", stripped)
    if name_match:
      current_table = name_match.group(1)
      rebuilt.append(line)
      continue
    size_match = re.match(r"size\s*=\s*(\d+)\s*;", stripped)
    if size_match and current_table is not None:
      old_literal = None
      if current_table.startswith("get_classification_tree_"):
        old_literal = "400"
      elif current_table.startswith("table_"):
        old_literal = "200"
      elif current_table.startswith("vote_"):
        old_literal = "32"
      if old_literal is not None:
        line = line.replace("size = " + size_match.group(1) + ";",
                            "size = " + old_literal + ";")
      current_table = None
    rebuilt.append(line)
  return "".join(rebuilt)


def test_generate_P4_code_default_call_changes_only_table_sizes(tmp_path):
  # Full regenerate-and-diff regression check: for the exact default call
  # shape most existing tests use, the ONLY delta this follow-up introduces
  # is the `size = ` numbers themselves. Putting the old literals back must
  # reproduce the pre-follow-up output byte for byte.
  import hashlib

  written_path = bps.generate_P4_code(
      3, 2, _tiny_app_forest(), _tiny_ddos_forest(),
      feature_intervals_app={"Flow_IAT_Max": [(0, 50), (51, INFINITE)]},
      feature_intervals_ddos={"Flow_IAT_Max": [(0, 200), (201, INFINITE)],
                              "Fwd_Packet_Length_Max": [(0, 64), (65, INFINITE)]},
      output_dir=str(tmp_path) + os.sep, output_filename="only_sizes_changed.p4")
  with open(written_path, "rb") as f:
    raw = f.read()

  # sanity: the sizes really did change (otherwise this test is vacuous)
  assert hashlib.sha256(raw).hexdigest() != _PRE_TABLE_SIZING_OUTPUT_SHA256
  reconstructed = _reconstruct_pre_table_sizing_text(raw.decode("utf-8"))
  assert hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() == \
      _PRE_TABLE_SIZING_OUTPUT_SHA256


def test_generate_P4_code_selected_features_without_flag_only_refines_sizes(tmp_path):
  # Passing selected_features_* WITHOUT the discount flag is now meaningful:
  # codewords get computed for SIZING only. Nothing else about the generated
  # program may change, and the exact codeword count may never exceed the
  # leaf-count fallback the same call produces without those lists.
  clf_app = _fit_real_forest([0, 1, 2], seed=0, n_estimators=2, value_scale=4000)
  intervals_app = _derive_intervals(clf_app)

  def _generate(filename, **extra):
    path = bps.generate_P4_code(
        3, 2, clf_app, None,
        feature_intervals_app=intervals_app, feature_intervals_ddos={},
        output_dir=str(tmp_path) + os.sep, output_filename=filename, **extra)
    with open(path) as f:
      return f.read()

  without = _generate("no_selected_features.p4")
  with_lists = _generate("with_selected_features.p4",
                         selected_features_app=_DISCOUNT_FEATURE_NAMES)

  sizes_without = _table_sizes(without)
  sizes_with = _table_sizes(with_lists)
  for table, size in sizes_with.items():
    assert size <= sizes_without[table], (
        "codeword-exact sizing must never exceed the leaf-count fallback")
  # every non-size line is identical
  stripped_without = [l for l in without.splitlines()
                      if not re.match(r"size\s*=\s*\d+\s*;", l.strip())]
  stripped_with = [l for l in with_lists.splitlines()
                   if not re.match(r"size\s*=\s*\d+\s*;", l.strip())]
  assert stripped_without == stripped_with


# ---------------------------------------------------------------------------
# PHV container pinning for range-matched feature value fields.
#
# Measured against the real Tofino compiler (reviews/p4_tofino_reference.md
# Sec 4.2): a `bit<16>` range key the PHV allocator parks in a 32-bit W
# container costs TWO physical TCAM blocks per table ("1 in 2 (88)"), while
# the same key in a 16-bit H container costs one ("1 in 1 (44)"). The choice
# is invisible to the trained model, so the cost model cannot predict it --
# it has to be pinned. Pinning all four feature value fields of one real M2
# program took it from 14 TCAM blocks / 10 stages to 12 blocks / 9 stages,
# with 12 being exactly what evaluation.py predicts.
#
# The pragma names the field as the PHV logs do -- `ig_md.<raw>_val`, after
# SwitchIngressParser's `out metadata_t ig_md` parameter, NOT the `meta`
# name SwitchIngress binds it to.
# ---------------------------------------------------------------------------

_PA_PRAGMA_RE = re.compile(
    r'@pa_container_size\(\s*"ingress"\s*,\s*"ig_md\.([A-Za-z0-9_]+)"\s*,\s*(\d+)\s*\)')


def test_generate_P4_code_pins_every_feature_value_field_to_a_16_bit_container(tmp_path):
  clf_ddos = _tiny_ddos_forest()
  intervals = {
      "Flow_IAT_Max": [(0, 50), (51, INFINITE)],
      "Fwd_Packet_Length_Max": [(0, 8), (9, INFINITE)],
  }
  written_path = bps.generate_P4_code(
      0, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="pinned.p4")
  with open(written_path) as f:
    text = f.read()

  pinned = dict(_PA_PRAGMA_RE.findall(text))

  assert pinned == {"flow_iat_max_val": "16", "fwd_packet_length_max_val": "16"}


def test_generate_P4_code_pins_exactly_the_declared_value_fields(tmp_path):
  # The regression this guards: a pragma naming a field that does not exist
  # is silently ignored by p4c, so the pinning would stop working with no
  # error anywhere. Assert the pinned set is exactly the set of `<name>_val`
  # metadata fields the generator declared.
  clf_ddos = _tiny_ddos_forest()
  intervals = {
      "Flow_IAT_Max": [(0, 50), (51, INFINITE)],
      "Fwd_IAT_Max": [(0, 7), (8, 900), (901, INFINITE)],
      "Fwd_Packet_Length_Max": [(0, 8), (9, INFINITE)],
  }
  written_path = bps.generate_P4_code(
      0, 2, None, clf_ddos,
      feature_intervals_app={}, feature_intervals_ddos=intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="pinned_exact.p4")
  with open(written_path) as f:
    text = f.read()

  declared = set(re.findall(r"bit<16> ([A-Za-z0-9_]+_val);", text))
  pinned = set(_PA_PRAGMA_RE.findall(text))

  assert declared, "fixture produced no feature value fields at all"
  assert {name for name, _ in pinned} == declared


def test_generate_P4_code_pins_a_shared_raw_field_only_once(tmp_path):
  # Under disjoint namespacing two resolved entries (app_/ddos_) share ONE
  # raw value field, which is declared once -- so it must be pinned once.
  # A duplicate @pa_container_size for the same field is a compile error.
  clf_app = _tiny_app_forest()
  clf_ddos = _tiny_ddos_forest()
  app_intervals = {"Flow_IAT_Max": [(0, 50), (51, INFINITE)]}
  ddos_intervals = {"Flow_IAT_Max": [(0, 200), (201, INFINITE)]}
  written_path = bps.generate_P4_code(
      3, 2, clf_app, clf_ddos,
      feature_intervals_app=app_intervals, feature_intervals_ddos=ddos_intervals,
      output_dir=str(tmp_path) + os.sep, output_filename="pinned_shared.p4")
  with open(written_path) as f:
    text = f.read()

  assert text.count("bit<16> flow_iat_max_val;") == 1
  assert [name for name, _ in _PA_PRAGMA_RE.findall(text)] == ["flow_iat_max_val"]
