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

import build_p4_script as bps
from build_p4_script import (
    generate_P4_actions,
    generate_P4_tables_and_apply,
    generate_P4_code,
    generate_voting_code,
    get_table_entries,
    get_ternary_match,
    most_common_class_and_dropped_codewords,
    OUTPUT_PATH,
)


INFINITE = bps.INFINITE


class _StubClassifier:
  """Minimal stand-in for a trained sklearn RandomForestClassifier: only
  `.estimators_`'s length is ever read by the code under test."""

  def __init__(self, num_trees):
    self.estimators_ = [object()] * num_trees


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

def _decode_ternary(ternary, width):
  """Invert get_ternary_match: recover the original bit-string ('0'/'1'/'*'
  per position) from a "0xV&&&0xM" ternary string, given the known bit
  width of the slice it was generated from."""
  value_hex, mask_hex = ternary.split("&&&")
  value_bits = bin(int(value_hex, 16))[2:].zfill(width)
  mask_bits = bin(int(mask_hex, 16))[2:].zfill(width)
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
  # per-tree action/table is used) -- action_params holds only the class.
  assert entry["action_params"] == ["1"]

  # One ternary key component per feature (2 features -> 2-element list),
  # not a single combined-codeword ternary string.
  assert len(entry["key"]) == 2

  # Round-trip: decoding each ternary component against its feature's known
  # width must recover exactly that feature's slice of the original
  # codeword, in feature_intervals iteration order.
  decoded_a = _decode_ternary(entry["key"][0], width=3)
  decoded_b = _decode_ternary(entry["key"][1], width=1)
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
  assert app_entries[0]["action_params"] == ["0"]
  assert ddos_entries[0]["action_params"] == ["1"]

  for entry, original in ((app_entries[0], codeword_app), (ddos_entries[0], codeword_ddos)):
    assert len(entry["key"]) == 2
    decoded = _decode_ternary(entry["key"][0], width=3) + _decode_ternary(entry["key"][1], width=1)
    assert decoded == original


def test_get_table_entries_feature_range_entries_unaffected(tmp_path):
  # Sanity: section 1 (feature-value -> codeword-bit tables) is untouched by
  # the Tier-3 slicing change -- still a single-element ".." range key.
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
    assert len(entry["key"]) == 1
    assert ".." in entry["key"][0]


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

  classification_entries = [e for e in entries if e["action"] == "classify_flow_codeword_0"]
  assert len(classification_entries) == 1   # 3 leaves minus both class-0 default_action leaves

  written_codewords = {
      _decode_ternary(entry["key"][0], width=3) + _decode_ternary(entry["key"][1], width=1)
      for entry in classification_entries
  }
  assert written_codewords.isdisjoint(dropped_codewords)


def test_get_table_entries_discount_off_by_default_writes_every_leaf(tmp_path):
  # Regression guard: omitting use_default_action_discount entirely must
  # still write all 3 leaves, matching pre-Task-7 behavior exactly.
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


def test_generate_P4_tables_and_apply_default_action_discount_emits_const_default_action():
  codewords = {0: {"101*": 0, "010*": 1, "111*": 0}}   # app tree 0: class 0 wins
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=1, num_trees_ddos=0,
      codewords=codewords, use_default_action_discount=True,
  )
  assert "const default_action = classify_flow_codeword_app_0(0);" in table_templates
  assert "<DEFAULT_ACTION>" not in table_templates


def test_generate_P4_tables_and_apply_discount_off_by_default_marker_line_fully_stripped():
  # Task 7's central promise: with the parameter unset/False, generated
  # text is byte-identical to before this task -- the whole
  # "<DEFAULT_ACTION>" marker line (added to
  # resources/table_classification.p4) must vanish entirely, not just the
  # bare token.
  codewords = {0: {"101*": 0, "010*": 1, "111*": 0}}
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=1, num_trees_ddos=0,
      codewords=codewords, use_default_action_discount=False,
  )
  assert "default_action" not in table_templates
  assert "<DEFAULT_ACTION>" not in table_templates


def test_generate_P4_tables_and_apply_discount_ddos_tree_reads_codewords_at_app_offset():
  # ddos tree i must read codewords[num_trees_app + i], not codewords[i] --
  # confirms the ddos branch doesn't accidentally reuse the app tree's slot.
  codewords = {
      0: {"000": 2, "001": 2, "010": 1},   # app tree 0 -> majority class 2
      1: {"100": 1, "101": 0, "110": 1},   # ddos tree 0 (index 1 == num_trees_app + 0) -> majority class 1
  }
  table_templates, _ = generate_P4_tables_and_apply(
      ["Feature_A"], num_trees_app=1, num_trees_ddos=1,
      codewords=codewords, use_default_action_discount=True,
  )
  assert "const default_action = classify_flow_codeword_app_0(2);" in table_templates
  assert "const default_action = classify_flow_codeword_ddos_0(1);" in table_templates


def test_generate_P4_tables_and_apply_discount_true_without_codewords_still_byte_identical():
  # use_default_action_discount=True with no codewords supplied (e.g. a
  # caller that hasn't wired codewords through yet) must not crash and
  # must not emit a default_action line -- there is nothing to compute it
  # from.
  table_templates, _ = generate_P4_tables_and_apply(
      list(FEATURE_INTERVALS_2F.keys()), num_trees_app=1, num_trees_ddos=0,
      use_default_action_discount=True,
  )
  assert "default_action" not in table_templates
  assert "<DEFAULT_ACTION>" not in table_templates


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


def test_generate_voting_code_table_size_scales_with_entry_count():
  # Final-review fix: table size used to be hardcoded to 32, which would
  # silently undersize the table once num_classes ** num_trees exceeds 32
  # (e.g. 5 trees x 3 classes = 243 entries). size must now scale as
  # max(32, num_classes ** num_trees).

  # Already-validated 3-tree/3-class config: 27 < 32, floor still applies,
  # so this must NOT change output for this config.
  table_decl_small, _ = bps.generate_voting_code(3, 3, "app")
  assert "size = 32;" in table_decl_small

  # 5 trees x 3 classes = 243 > 32: size must scale up.
  table_decl_large, _ = bps.generate_voting_code(5, 3, "app")
  assert "size = 243;" in table_decl_large
  assert "size = 32;" not in table_decl_large


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
      feature_intervals=M1_FEATURE_INTERVALS,
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


def test_generate_P4_code_writes_to_injectable_path(tmp_path):
  clf_ddos = _tiny_ddos_forest()  # reuse this file's existing tiny-forest fixture helper
  feature_intervals = {"F": [(0, 100), (101, 65535)]}
  out_dir = str(tmp_path) + os.sep
  written_path = bps.generate_P4_code(
      num_class_app=0, num_class_ddos=2, clf_app=None, clf_ddos=clf_ddos,
      feature_intervals=feature_intervals,
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
      feature_intervals=feature_intervals,
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
      feature_intervals=feature_intervals,
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
      feature_intervals=feature_intervals,
      output_dir=out_dir, output_filename="ternary_default_test.p4",
  )
  with open(written_path, "r") as f:
    generated = f.read()
  assert "meta.code_f : ternary;" in generated
  # (generate_voting_code's vote_ddos table legitimately uses ": exact;" on
  # meta.class_tree_ddos_0 regardless of match_type -- see note above.)
  assert "meta.code_f : exact;" not in generated
