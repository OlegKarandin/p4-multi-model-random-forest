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

import pytest

import build_p4_script as bps
from build_p4_script import (
    generate_P4_actions,
    generate_P4_tables_and_apply,
    generate_P4_code,
    get_table_entries,
    get_ternary_match,
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
      "flows_test_other",
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
