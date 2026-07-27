"""
Tests for `feature_registers.FEATURE_REGISTER_CATALOG` and
`build_p4_script.generate_P4_registers_and_apply`.

These are validated against the ground-truth, compiler-validated Tofino
spike checked into this repo at
`p4/tofino_spike/compile_logs_m1_flows_iat/tna_m1_flows_iat_spike.p4pp`
(the `SwitchIngress` control block, lines 1659-1859), NOT against the
generator's own current behavior -- the point is to catch places where the
generator diverges from what actually compiled with the real Tofino p4c.

Action *names* are allowed to differ between the generator and the spike
(the generator derives action names from catalog register names, the spike
hand-wrote its own); only RegisterAction *bodies* are compared.
"""

import os
import re

import pytest

import build_p4_script as bps
from build_p4_script import generate_P4_registers_and_apply, MAX_REGISTER_TOUCHES
from feature_registers import FEATURE_REGISTER_CATALOG


SPIKE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "p4", "tofino_spike", "compile_logs_m1_flows_iat", "tna_m1_flows_iat_spike.p4pp",
)

# Brief-designated authoritative section: SwitchIngress control block,
# 1-indexed lines 1659-1859 (inclusive).
_SPIKE_CONTROL_START_LINE = 1659
_SPIKE_CONTROL_END_LINE = 1859


def _load_spike_control_block():
  with open(SPIKE_PATH, "r") as spike_file:
    lines = spike_file.readlines()
  return "".join(lines[_SPIKE_CONTROL_START_LINE - 1:_SPIKE_CONTROL_END_LINE])


SPIKE_CONTROL = _load_spike_control_block()

# The M1 feature set as generate_P4_registers_and_apply's real caller
# (build_p4_script.get_nodes()/get_feature_intervals()) actually produces
# it: Title_Case_With_Underscores keys, values unused.
M1_FEATURE_INTERVALS = {
    "Flow_IAT_Max": None,
    "Fwd_IAT_Max": None,
    "Fwd_Packet_Length_Max": None,
}

# Spike hand-wrote its own RegisterAction names; map each catalog register
# name to the spike's action name so bodies can be compared.
_SPIKE_ACTION_NAME_BY_REGISTER = {
    "flow_last_arrival_time": "flow_iat_action",
    "flow_iat_max": "flow_iat_max_action",
    "fwd_last_arrival_time": "fwd_iat_action",
    "fwd_iat_max": "fwd_iat_max_action",
    "fwd_packet_length_max": "fwd_packet_length_max_action",
}


def _normalize(text):
  """Strip line-level indentation/blank-line noise so the generator's
  tab-formatted output can be content-compared against the spike's
  space-formatted text."""
  return [line.strip() for line in text.strip("\n").splitlines() if line.strip()]


def _extract_action_body(source, action_name):
  """Pull the body out of `<action_name> = { ... };` (a RegisterAction
  block) in `source`."""
  pattern = re.compile(re.escape(action_name) + r"\s*=\s*\{(.*?)\n[ \t]*\};", re.DOTALL)
  match = pattern.search(source)
  assert match is not None, "could not find action '{0}' in source".format(action_name)
  return match.group(1)


def _extract_calc_timestamp_body(source):
  pattern = re.compile(r"action calc_timestamp\(\) \{(.*?)\n[ \t]*\}", re.DOTALL)
  match = pattern.search(source)
  assert match is not None, "could not find calc_timestamp action in source"
  return match.group(1)


def _extract_register_declarations(source):
  """Map register base name -> width from `Register<bit<W>,
  bit<32>>(MAX_NUM_FLOWS) <name>_reg;` declarations in `source`."""
  pattern = re.compile(r"Register<bit<(\d+)>,\s*bit<32>>\(MAX_NUM_FLOWS\)\s+(\w+)_reg;")
  return {name: int(width) for width, name in pattern.findall(source)}


@pytest.fixture(scope="module")
def m1_generated():
  return generate_P4_registers_and_apply(M1_FEATURE_INTERVALS)


# ---------------------------------------------------------------------------
# FEATURE_REGISTER_CATALOG shape
# ---------------------------------------------------------------------------

def test_catalog_has_exactly_the_three_m1_features():
  assert set(FEATURE_REGISTER_CATALOG.keys()) == {
      "flow_iat_max", "fwd_iat_max", "fwd_packet_length_max",
  }


def test_flow_iat_max_catalog_entry():
  entry = FEATURE_REGISTER_CATALOG["flow_iat_max"]
  assert entry["gated_by"] is None
  assert entry["registers"] == [
      {"name": "flow_last_arrival_time", "role": "dependency", "width": 16, "body": "iat_delta"},
      {"name": "flow_iat_max", "role": "value", "width": 16, "body": "running_max_iat"},
  ]


def test_fwd_iat_max_catalog_entry():
  entry = FEATURE_REGISTER_CATALOG["fwd_iat_max"]
  assert entry["gated_by"] == "fwd"
  assert entry["registers"] == [
      {"name": "fwd_last_arrival_time", "role": "dependency", "width": 16, "body": "iat_delta"},
      {"name": "fwd_iat_max", "role": "value", "width": 16, "body": "running_max_iat"},
  ]


def test_fwd_packet_length_max_catalog_entry():
  entry = FEATURE_REGISTER_CATALOG["fwd_packet_length_max"]
  assert entry["gated_by"] == "fwd"
  assert entry["registers"] == [
      {"name": "fwd_packet_length_max", "role": "value", "width": 16, "body": "running_max_packet_length"},
  ]


# ---------------------------------------------------------------------------
# generate_P4_registers_and_apply vs. spike ground truth
# ---------------------------------------------------------------------------

def test_m1_register_declarations_match_spike(m1_generated):
  registers_code, _, _ = m1_generated
  generated = _extract_register_declarations(registers_code)
  spike = _extract_register_declarations(SPIKE_CONTROL)

  expected_names = {
      "flows", "flow_last_arrival_time", "flow_iat_max",
      "fwd_last_arrival_time", "fwd_iat_max", "fwd_packet_length_max",
  }
  assert set(spike.keys()) == expected_names
  assert generated == spike


def test_register_action_bodies_match_spike():
  # Compare each catalog register's symbolic body kind, as expanded by
  # _REGISTER_ACTION_BODIES, against the spike's hand-written action body
  # for that same register -- content only, not the (allowed-to-differ)
  # action name.
  for feature, entry in FEATURE_REGISTER_CATALOG.items():
    for reg in entry["registers"]:
      spike_action_name = _SPIKE_ACTION_NAME_BY_REGISTER[reg["name"]]
      spike_body = _extract_action_body(SPIKE_CONTROL, spike_action_name)
      generated_body = bps._REGISTER_ACTION_BODIES[reg["body"]].format(width=reg["width"])
      assert _normalize(generated_body) == _normalize(spike_body), (
          "feature '{0}' register '{1}' (body kind '{2}') diverges from "
          "spike action '{3}'".format(feature, reg["name"], reg["body"], spike_action_name)
      )


def test_flows_reg_actions_match_spike(m1_generated):
  _, register_actions_code, _ = m1_generated
  for action_name in ("flows_test_other", "flows_set_self"):
    generated_body = _extract_action_body(register_actions_code, action_name)
    spike_body = _extract_action_body(SPIKE_CONTROL, action_name)
    assert _normalize(generated_body) == _normalize(spike_body)


def test_flows_reg_two_touch_pattern_order(m1_generated):
  _, _, apply_code = m1_generated

  assert "bit<1> other_seen = flows_test_other.execute(meta.flow_hash_other);" in apply_code
  other_seen_idx = apply_code.index("flows_test_other.execute(meta.flow_hash_other)")
  if_other_seen_idx = apply_code.index("if (other_seen == 1) {")
  set_self_idx = apply_code.index("flows_set_self.execute(meta.flow_hash_self)")

  # Matches the spike (lines 1830-1839): read-only test of the *other*
  # direction's slot always runs first; the *own*-slot test-and-set only
  # runs in the "not already seen" (else) branch.
  assert other_seen_idx < if_other_seen_idx < set_self_idx


def test_calc_timestamp_emitted_and_matches_spike(m1_generated):
  _, register_actions_code, _ = m1_generated
  assert "action calc_timestamp() {" in register_actions_code

  spike_body = _extract_calc_timestamp_body(SPIKE_CONTROL)
  generated_body = _extract_calc_timestamp_body(register_actions_code)
  assert _normalize(generated_body) == _normalize(spike_body)
  assert ">> 10" in generated_body


def test_fwd_gating_structure(m1_generated):
  _, _, apply_code = m1_generated

  fwd_block_idx = apply_code.index("if (meta.fwd == 1) {")

  # flow_iat_max is ungated ("gated_by": None) -- its execute call sites
  # must precede (be outside) the fwd-gated block.
  flow_dep_idx = apply_code.index("meta.current_iat = flow_last_arrival_time_action.execute(meta.flow_hash);")
  flow_val_idx = apply_code.index("meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);")
  assert flow_dep_idx < fwd_block_idx
  assert flow_val_idx < fwd_block_idx

  # fwd_iat_max and fwd_packet_length_max are "gated_by": "fwd" -- their
  # execute call sites must follow (be inside) the fwd-gated block.
  fwd_dep_idx = apply_code.index("meta.current_iat = fwd_last_arrival_time_action.execute(meta.flow_hash);")
  fwd_val_idx = apply_code.index("meta.fwd_iat_max_val = fwd_iat_max_action.execute(meta.flow_hash);")
  fwd_pktlen_idx = apply_code.index("meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);")
  assert fwd_dep_idx > fwd_block_idx
  assert fwd_val_idx > fwd_block_idx
  assert fwd_pktlen_idx > fwd_block_idx


def test_dependency_registers_assigned_to_current_iat(m1_generated):
  _, _, apply_code = m1_generated
  # Both IAT features' dependency registers (flow_last_arrival_time,
  # fwd_last_arrival_time) feed the shared meta.current_iat scratch field.
  assert apply_code.count("meta.current_iat = ") == 2


def test_value_registers_assigned_with_val_suffix(m1_generated):
  _, _, apply_code = m1_generated
  # Post-fix convention (matches the spike's metadata_t field naming):
  # every "value" register's .execute() result is assigned to
  # meta.<feature>_val, consistently for all three M1 features -- including
  # fwd_packet_length_max, whose spike-only shortcut (reading
  # hdr.ipv4.total_len directly in its classification table instead) is
  # deliberately not replicated here.
  for feature in ("flow_iat_max", "fwd_iat_max", "fwd_packet_length_max"):
    assert "meta.{0}_val = ".format(feature) in apply_code
    # Must not regress to the old, spike-metadata-struct-mismatched,
    # no-suffix naming.
    assert "meta.{0} = ".format(feature) not in apply_code


# ---------------------------------------------------------------------------
# Feature resolution behavior
# ---------------------------------------------------------------------------

def test_empty_feature_intervals_returns_empty_strings():
  assert generate_P4_registers_and_apply({}) == ("", "", "")


def test_all_unknown_features_returns_empty_strings():
  assert generate_P4_registers_and_apply({"Totally_Bogus_Feature": None}) == ("", "", "")


def test_unknown_feature_mixed_with_known_is_skipped(m1_generated):
  mixed_intervals = dict(M1_FEATURE_INTERVALS)
  mixed_intervals["Totally_Bogus_Feature"] = None
  assert generate_P4_registers_and_apply(mixed_intervals) == m1_generated


def test_case_insensitive_feature_lookup(m1_generated):
  # feature_intervals keys are Title_Case_With_Underscores in real callers;
  # the catalog's keys are lowercase. Confirm .lower()-based resolution
  # works and produces byte-identical output either way.
  lowercase_intervals = {name.lower(): value for name, value in M1_FEATURE_INTERVALS.items()}
  assert generate_P4_registers_and_apply(lowercase_intervals) == m1_generated


# ---------------------------------------------------------------------------
# Guardrails (synthetic catalogs only -- never mutate the real M1 catalog)
# ---------------------------------------------------------------------------

def test_register_touch_limit_raises():
  # Two synthetic features share one register name; together they touch it
  # more times than MAX_REGISTER_TOUCHES allows.
  shared_register = {
      "name": "shared_synthetic_reg",
      "role": "value",
      "width": 16,
      "body": "running_max_iat",
  }
  touches_needed = MAX_REGISTER_TOUCHES + 1
  split_a = touches_needed // 2
  split_b = touches_needed - split_a
  synthetic_catalog = {
      "synthetic_feature_a": {"registers": [shared_register] * split_a, "gated_by": None},
      "synthetic_feature_b": {"registers": [shared_register] * split_b, "gated_by": None},
  }
  feature_intervals = {"Synthetic_Feature_A": None, "Synthetic_Feature_B": None}

  with pytest.raises(RuntimeError):
    generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)


def test_unsupported_gated_by_raises():
  synthetic_catalog = {
      "synthetic_feature": {
          "registers": [{
              "name": "synthetic_reg",
              "role": "value",
              "width": 16,
              "body": "running_max_iat",
          }],
          "gated_by": "bwd",
      },
  }
  feature_intervals = {"Synthetic_Feature": None}

  with pytest.raises(RuntimeError):
    generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)
