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
from build_p4_script import generate_P4_registers_and_apply
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

# M2's mean/EWMA ground truth spike (Task M2-B1) -- a small, standalone,
# not-preprocessed .p4 file, so no line-range slicing is needed: the whole
# file is the relevant content.
M2_MEAN_SPIKE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "p4", "tofino_spike", "tna_m2_mean_spike.p4",
)


def _load_m2_mean_spike():
  with open(M2_MEAN_SPIKE_PATH, "r") as spike_file:
    return spike_file.read()


M2_MEAN_SPIKE = _load_m2_mean_spike()

# The M1 feature set as generate_P4_registers_and_apply's real caller
# (build_p4_script.get_nodes()/get_feature_intervals()) actually produces
# it: Title_Case_With_Underscores keys, values unused.
M1_FEATURE_INTERVALS = {
    "Flow_IAT_Max": None,
    "Fwd_IAT_Max": None,
    "Fwd_Packet_Length_Max": None,
}

# M2-B1: flow_iat_max + flow_iat_mean share one dependency register
# (flow_last_arrival_time) -- the shared-dependency dedup scenario this task
# fixes.
M2_MEAN_FEATURE_INTERVALS = {
    "Flow_IAT_Max": None,
    "Flow_IAT_Mean": None,
}

# M2-B1: all three of M1's flow_iat_max, fwd_iat_max, plus M2's flow_iat_mean
# together -- confirms flow_last_arrival_time (shared by flow_iat_max +
# flow_iat_mean) and fwd_last_arrival_time (independent, fwd_iat_max only)
# don't cross-contaminate each other's dedup accounting.
M2_MEAN_PLUS_FWD_FEATURE_INTERVALS = {
    "Flow_IAT_Max": None,
    "Fwd_IAT_Max": None,
    "Flow_IAT_Mean": None,
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


@pytest.fixture(scope="module")
def m2_mean_generated():
  return generate_P4_registers_and_apply(M2_MEAN_FEATURE_INTERVALS)


@pytest.fixture(scope="module")
def m2_mean_plus_fwd_generated():
  return generate_P4_registers_and_apply(M2_MEAN_PLUS_FWD_FEATURE_INTERVALS)


# ---------------------------------------------------------------------------
# FEATURE_REGISTER_CATALOG shape
# ---------------------------------------------------------------------------

def test_catalog_has_exactly_the_four_m1_m2_features():
  # NOTE: this test's assertion is intentionally updated by Task M2-B1 (was
  # "test_catalog_has_exactly_the_three_m1_features", asserting only M1's 3
  # keys) -- adding the "flow_iat_mean" catalog entry is this task's own
  # required deliverable, so the catalog's key set necessarily grows by one.
  # This is the one pre-existing test whose assertion must change; every
  # other M1 test/behavior is unaffected (see task report for details).
  assert set(FEATURE_REGISTER_CATALOG.keys()) == {
      "flow_iat_max", "fwd_iat_max", "fwd_packet_length_max", "flow_iat_mean",
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


def test_flow_iat_mean_catalog_entry():
  entry = FEATURE_REGISTER_CATALOG["flow_iat_mean"]
  assert entry["gated_by"] is None
  assert entry["registers"] == [
      {"name": "flow_last_arrival_time", "role": "dependency", "width": 16, "body": "iat_delta"},
      {"name": "flow_iat_mean", "role": "value", "width": 16, "body": "mathunit_ewma"},
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
  # Compare each M1 catalog register's symbolic body kind, as expanded by
  # _REGISTER_ACTION_BODIES, against the M1 spike's hand-written action body
  # for that same register -- content only, not the (allowed-to-differ)
  # action name. Scoped to _SPIKE_ACTION_NAME_BY_REGISTER's own M1 feature
  # set (not "every FEATURE_REGISTER_CATALOG entry") because M2's
  # flow_iat_mean is validated against a *different* spike file
  # (tna_m2_mean_spike.p4, exercised separately by
  # test_mathunit_ewma_body_matches_spike) -- it has no entry in this M1
  # spike's SPIKE_CONTROL text.
  for feature in ("flow_iat_max", "fwd_iat_max", "fwd_packet_length_max"):
    entry = FEATURE_REGISTER_CATALOG[feature]
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
# M2-B1: flow_iat_mean catalog entry, mathunit_ewma body kind, and the
# shared-dependency-register dedup fix.
# ---------------------------------------------------------------------------

def test_mathunit_ewma_body_matches_spike(m2_mean_generated):
  # Transcribed-verbatim check against the compiled ground-truth spike
  # (p4/tofino_spike/tna_m2_mean_spike.p4, flow_iat_mean_ewma_action). The
  # spike hand-wrote a single global `halve_unit` identifier; the generator
  # parametrizes it per-register as `<name>_halve_unit` (so a future second
  # mathunit_ewma register wouldn't collide on one shared MathUnit instance)
  # -- substitute that one identifier before comparing body content.
  spike_body = _extract_action_body(M2_MEAN_SPIKE, "flow_iat_mean_ewma_action")
  generated_body = bps._REGISTER_ACTION_BODIES["mathunit_ewma"].format(width=16, name="flow_iat_mean")
  normalized_spike = [line.replace("halve_unit", "flow_iat_mean_halve_unit") for line in _normalize(spike_body)]
  assert _normalize(generated_body) == normalized_spike


def test_shared_dependency_register_executed_exactly_once(m2_mean_generated):
  # The core dedup fix: flow_iat_max and flow_iat_mean both list
  # "flow_last_arrival_time" as a dependency register. Before the fix,
  # _execute_lines emitted one .execute() call per (feature, register)
  # pair -- two calls for one register. After the fix, exactly one call
  # site must appear, regardless of how many features reference it.
  _, _, apply_code = m2_mean_generated
  assert apply_code.count("flow_last_arrival_time_action.execute(meta.flow_hash)") == 1

  # Both features' own "value" registers must still each get their own
  # execute call site, both consuming the single shared meta.current_iat
  # the deduped dependency execution produced.
  assert "meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);" in apply_code
  assert "meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);" in apply_code
  assert apply_code.count("meta.current_iat = ") == 1


def test_mathunit_declaration_emitted_before_register_action(m2_mean_generated):
  _, register_actions_code, _ = m2_mean_generated
  mathunit_idx = register_actions_code.index(
      "MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;")
  action_idx = register_actions_code.index(
      "RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {")
  assert mathunit_idx < action_idx

  # The MathUnit<> declaration is specific to "mathunit_ewma"-bodied
  # registers -- must not leak in front of unrelated RegisterActions.
  assert register_actions_code.count("MathUnit<bit<16>>(MathOp_t.MUL, 1, 2)") == 1


def test_dedup_does_not_cross_contaminate_flow_and_fwd_namespaces(m2_mean_plus_fwd_generated):
  # Flow_IAT_Max + Fwd_IAT_Max + Flow_IAT_Mean together: flow_last_arrival_time
  # is shared (flow_iat_max + flow_iat_mean) and deduped to one execute call;
  # fwd_last_arrival_time is a wholly separate, independently-touched
  # register (only fwd_iat_max references it) -- confirming the dedup fix
  # tracks *register names*, not e.g. a blanket "first dependency only" rule
  # that would incorrectly also swallow fwd's independent dependency.
  _, _, apply_code = m2_mean_plus_fwd_generated
  assert apply_code.count("flow_last_arrival_time_action.execute(meta.flow_hash)") == 1
  assert apply_code.count("fwd_last_arrival_time_action.execute(meta.flow_hash)") == 1
  # Two distinct dependency registers (flow_*, fwd_*), each executed once =
  # two meta.current_iat assignments total (not three, which is what the
  # pre-fix double-execute bug would have produced).
  assert apply_code.count("meta.current_iat = ") == 2


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

def test_register_touch_limit_raises(monkeypatch):
  # Post-M2-B1 fix-round-1: register_touch_count now counts each register's
  # REAL, deduplicated .execute() call-site count (see _note_touch), the
  # same model _execute_lines uses to emit code. Under that model, a
  # catalog-driven register referenced by any number of features can only
  # ever accumulate 1 real touch (whichever feature reaches it first wins
  # the one call site; every later reference reuses that value instead of
  # adding another touch) -- so a synthetic catalog can no longer organically
  # manufacture a >MAX_REGISTER_TOUCHES scenario for a single catalog
  # register by repeating its name (that was the *old*, buggy, per-reference
  # accounting this task fixed; see
  # test_shared_dependency_does_not_trip_touch_guard_even_with_many_sharers
  # below for that "no longer organically possible" case, tested directly
  # against the real, unpatched MAX_REGISTER_TOUCHES=4).
  #
  # To still exercise the guard's comparison logic (`count > MAX_REGISTER_
  # TOUCHES`) meaningfully, shrink MAX_REGISTER_TOUCHES to 0 for this test
  # only: with a cap of 0, even the baseline `flows` bookkeeping register's
  # always-real, always-legitimate 2 touches (present the moment any feature
  # resolves at all -- not part of this task's bug, see the brief) exceeds
  # the cap, proving the guard's raise still fires on real, correctly
  # counted touches rather than having been silently disabled or broken by
  # this fix. This keeps the guard's actual protective purpose intact -- the
  # only thing that changed is *how a touch is counted*, not whether an
  # over-the-limit count still raises.
  monkeypatch.setattr(bps, "MAX_REGISTER_TOUCHES", 0)

  synthetic_catalog = {
      "synthetic_feature": {
          "registers": [{
              "name": "shared_synthetic_reg",
              "role": "value",
              "width": 16,
              "body": "running_max_iat",
          }],
          "gated_by": None,
      },
  }
  feature_intervals = {"Synthetic_Feature": None}

  with pytest.raises(RuntimeError):
    generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)


def test_shared_dependency_does_not_trip_touch_guard_even_with_many_sharers():
  # The direct, real-world-limit (no monkeypatching MAX_REGISTER_TOUCHES;
  # this runs against the actual 4-touch Tofino hardware cap) proof of
  # Finding 1's fix: 5 synthetic features all sharing ONE register name.
  # Under the OLD (buggy) per-(feature, register-list-entry) accounting,
  # this would have counted 5 touches for "shared_synthetic_reg" -- exceeding
  # MAX_REGISTER_TOUCHES=4 and incorrectly raising RuntimeError for a
  # register that is, in reality, only ever given ONE .execute() call site
  # (see _execute_lines' dedup). Under the fixed, deduplicated accounting,
  # this register is counted once no matter how many features reference it,
  # so resolving all 5 features together must NOT raise.
  shared_register = {
      "name": "shared_synthetic_reg",
      "role": "value",
      "width": 16,
      "body": "running_max_iat",
  }
  synthetic_catalog = {
      "synthetic_feature_{0}".format(i): {"registers": [shared_register], "gated_by": None}
      for i in range(5)
  }
  feature_intervals = {"Synthetic_Feature_{0}".format(i): None for i in range(5)}

  _, _, apply_code = generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)

  # And the emitted code backs up the count: exactly one real .execute()
  # call site for the shared register, regardless of 5 features referencing
  # it (mirrors test_shared_dependency_register_executed_exactly_once, here
  # against a purely synthetic catalog rather than the real flow_iat_mean
  # scenario).
  assert apply_code.count("shared_synthetic_reg_action.execute(meta.flow_hash)") == 1


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
