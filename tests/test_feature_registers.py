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

from src.p4gen import build_p4_script as bps
from src.p4gen.build_p4_script import generate_P4_registers_and_apply
from src.p4gen import feature_registers as fr
from src.p4gen.feature_registers import FEATURE_REGISTER_CATALOG


SPIKE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


# tna_m1_flows_iat_spike.p4pp was never git-tracked and was deleted along with
# the rest of p4/tofino_spike/compile_logs_m1_flows_iat/ as a (mistakenly
# assumed disposable) compiler-log artifact. Skip the tests that depend on it
# instead of erroring out at collection time; regenerate via the WSL2 p4c
# toolchain and drop it back at SPIKE_PATH to re-enable them.
try:
  SPIKE_CONTROL = _load_spike_control_block()
except FileNotFoundError:
  SPIKE_CONTROL = None

# M2's mean/EWMA ground truth spike (Task M2-B1) -- a small, standalone,
# not-preprocessed .p4 file, so no line-range slicing is needed: the whole
# file is the relevant content.
M2_MEAN_SPIKE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "p4", "tofino_spike", "tna_m2_mean_spike.p4",
)


def _load_m2_mean_spike():
  with open(M2_MEAN_SPIKE_PATH, "r") as spike_file:
    return spike_file.read()


M2_MEAN_SPIKE = _load_m2_mean_spike()

# Task 3: symmetric-hash flow-direction bookkeeping ground truth (compiled
# clean against the real Tofino compiler this session, 0 errors) -- the
# spike-comparison source for flow_orientation_action, replacing the M1
# spike's now-superseded flows_test_other/flows_set_self design.
M2_SYMMETRIC_HASH_SPIKE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "p4", "tofino_spike", "tna_m2_symmetric_hash_spike.p4",
)


def _load_m2_symmetric_hash_spike():
  with open(M2_SYMMETRIC_HASH_SPIKE_PATH, "r") as spike_file:
    return spike_file.read()


M2_SYMMETRIC_HASH_SPIKE = _load_m2_symmetric_hash_spike()

# The M1 feature set as generate_P4_registers_and_apply's real caller
# (build_p4_script.get_nodes()/get_feature_intervals()) actually produces
# it: lowercase, underscore-joined keys (normalise_feature_name()), values
# unused.
M1_FEATURE_INTERVALS = {
    "flow_iat_max": None,
    "fwd_iat_max": None,
    "fwd_packet_length_max": None,
}

# M2-B1: flow_iat_max + flow_iat_mean share one dependency register
# (flow_last_arrival_time) -- the shared-dependency dedup scenario this task
# fixes.
M2_MEAN_FEATURE_INTERVALS = {
    "flow_iat_max": None,
    "flow_iat_mean": None,
}

# M2-B1: all three of M1's flow_iat_max, fwd_iat_max, plus M2's flow_iat_mean
# together -- confirms flow_last_arrival_time (shared by flow_iat_max +
# flow_iat_mean) and fwd_last_arrival_time (independent, fwd_iat_max only)
# don't cross-contaminate each other's dedup accounting.
M2_MEAN_PLUS_FWD_FEATURE_INTERVALS = {
    "flow_iat_max": None,
    "fwd_iat_max": None,
    "flow_iat_mean": None,
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

def test_catalog_has_exactly_the_eighteen_pool_features():
  # NOTE: this test's assertion has been updated twice now -- Task M2-B1 grew
  # it from 3 to 4 keys (adding flow_iat_mean), and Task 9 grows it from 4 to
  # all 18 keys of main.py's feature pool (main.py:308-314), its own required
  # deliverable. This is the one pre-existing test whose assertion changes
  # each time the catalog's scope grows; every other M1/M2 test/behavior is
  # unaffected.
  assert set(FEATURE_REGISTER_CATALOG.keys()) == {
      "flow_iat_max", "flow_iat_mean", "flow_iat_min",
      "fwd_iat_max", "fwd_iat_mean", "fwd_iat_min",
      "bwd_iat_max", "bwd_iat_mean", "bwd_iat_min",
      "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean",
      "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean",
      "min_packet_length", "max_packet_length", "packet_length_mean",
  }


def test_every_catalog_entry_orders_dependency_registers_before_value_registers():
  # The primary risk Task 9 called out for its 14 new entries: _execute_lines
  # (build_p4_script.py) emits each feature's "registers" list IN ORDER, so a
  # "value" register that consumes meta.current_iat must be listed AFTER the
  # "dependency" register that produces it, never before.
  for feature, entry in FEATURE_REGISTER_CATALOG.items():
    roles = [reg["role"] for reg in entry["registers"]]
    dependency_positions = [i for i, role in enumerate(roles) if role == "dependency"]
    value_positions = [i for i, role in enumerate(roles) if role == "value"]
    if dependency_positions and value_positions:
      assert max(dependency_positions) < min(value_positions), (
          feature, entry["registers"])


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

@pytest.mark.skipif(SPIKE_CONTROL is None, reason="tna_m1_flows_iat_spike.p4pp fixture is missing, see SPIKE_PATH")
def test_m1_register_declarations_match_spike(m1_generated):
  # Task 3: the M1 spike (SPIKE_CONTROL) still reflects the OLD two-hash/
  # flows_reg design, so it is no longer a valid ground truth for the
  # bookkeeping register's declaration -- only the catalog-driven feature
  # registers are still compared against it. The bookkeeping register
  # itself (`flow_forward_srcaddr_reg`, width 32) is checked directly
  # against the new symmetric-hash design instead of the M1 spike.
  registers_code, _, _, _ = m1_generated
  generated = _extract_register_declarations(registers_code)
  spike = _extract_register_declarations(SPIKE_CONTROL)

  feature_expected_names = {
      "flow_last_arrival_time", "flow_iat_max",
      "fwd_last_arrival_time", "fwd_iat_max", "fwd_packet_length_max",
  }
  spike_expected_names = feature_expected_names | {"flows"}
  assert set(spike.keys()) == spike_expected_names

  # Catalog-driven feature registers must still match the spike exactly.
  assert {name: width for name, width in generated.items() if name in feature_expected_names} == {
      name: width for name, width in spike.items() if name in feature_expected_names
  }

  # The old "flows" register is gone; the new fixed bookkeeping register
  # (flow_forward_srcaddr_reg, width 32 -- captured as "flow_forward_srcaddr"
  # by _extract_register_declarations' `(\w+)_reg;` pattern) replaces it.
  assert "flows" not in generated
  assert generated["flow_forward_srcaddr"] == 32
  assert set(generated.keys()) == feature_expected_names | {"flow_forward_srcaddr"}


@pytest.mark.skipif(SPIKE_CONTROL is None, reason="tna_m1_flows_iat_spike.p4pp fixture is missing, see SPIKE_PATH")
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


def test_symmetric_hash_bookkeeping_design(m1_generated):
  # Task 3 (Step 2, TDD RED): the new single-hash/single-register design
  # replaces the old two-Hash<>/flows_reg/flows_test_other/flows_set_self
  # "test other, then set self" bookkeeping (see
  # p4/tofino_spike/tna_m2_symmetric_hash_spike.p4 for the validated ground
  # truth this codegen change transcribes).
  registers_code, register_actions_code, apply_code, _ = m1_generated

  # Exactly one Hash<> instance now (was two: flow_hash_calc_self/_other).
  assert registers_code.count("Hash<bit<32>>(HashAlgorithm_t.CRC32)") == 1
  assert "flow_hash_calc_self" not in registers_code
  assert "flow_hash_calc_other" not in registers_code

  # The symmetric XOR-based hash fields.
  assert "hdr.ipv4.src_addr ^ hdr.ipv4.dst_addr" in register_actions_code
  assert "hdr.tcp.src_port ^ hdr.tcp.dst_port" in register_actions_code

  # The old flows_reg/flows_test_other/flows_set_self design is gone.
  assert "flows_reg" not in registers_code
  assert "flows_test_other" not in register_actions_code
  assert "flows_set_self" not in register_actions_code

  # The new fixed register + RegisterAction are present.
  assert "flow_forward_srcaddr_reg" in registers_code
  assert "flow_orientation_action" in register_actions_code

  # Apply-block: calc_flow_hash() exactly once, the new single execute call
  # site, and no leftover "other_seen" direction-test bookkeeping anywhere.
  assert apply_code.count("calc_flow_hash()") == 1
  assert "flow_orientation_action.execute(meta.flow_hash)" in apply_code
  assert "other_seen" not in apply_code


def test_flow_orientation_action_matches_spike(m1_generated):
  # Task 3: the old flows_test_other/flows_set_self design is replaced by a
  # single flow_orientation_action RegisterAction on flow_forward_srcaddr_reg
  # -- compare against tna_m2_symmetric_hash_spike.p4 (the validated ground
  # truth for the new design), not the M1 spike (which still reflects the
  # old two-hash design).
  _, register_actions_code, _, _ = m1_generated
  generated_body = _extract_action_body(register_actions_code, "flow_orientation_action")
  spike_body = _extract_action_body(M2_SYMMETRIC_HASH_SPIKE, "flow_orientation_action")
  assert _normalize(generated_body) == _normalize(spike_body)


def test_flow_orientation_action_executed_exactly_once(m1_generated):
  # Task 3: the old design's two-touch ("test other, then set self") pattern
  # is gone -- there is now only one call site per packet, so there is no
  # more "which runs first" ordering to assert, only a single-touch count.
  _, _, apply_code, _ = m1_generated

  assert apply_code.count("flow_orientation_action.execute(meta.flow_hash)") == 1


@pytest.mark.skipif(SPIKE_CONTROL is None, reason="tna_m1_flows_iat_spike.p4pp fixture is missing, see SPIKE_PATH")
def test_calc_timestamp_emitted_and_matches_spike(m1_generated):
  _, register_actions_code, _, _ = m1_generated
  assert "action calc_timestamp() {" in register_actions_code

  spike_body = _extract_calc_timestamp_body(SPIKE_CONTROL)
  generated_body = _extract_calc_timestamp_body(register_actions_code)
  assert _normalize(generated_body) == _normalize(spike_body)
  assert ">> 10" in generated_body


def test_fwd_gating_structure(m1_generated):
  _, _, apply_code, _ = m1_generated

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
  _, _, apply_code, _ = m1_generated
  # Both IAT features' dependency registers (flow_last_arrival_time,
  # fwd_last_arrival_time) feed the shared meta.current_iat scratch field.
  assert apply_code.count("meta.current_iat = ") == 2


def test_value_registers_assigned_with_val_suffix(m1_generated):
  _, _, apply_code, _ = m1_generated
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
  _, _, apply_code, _ = m2_mean_generated
  assert apply_code.count("flow_last_arrival_time_action.execute(meta.flow_hash)") == 1

  # Both features' own "value" registers must still each get their own
  # execute call site, both consuming the single shared meta.current_iat
  # the deduped dependency execution produced.
  assert "meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);" in apply_code
  assert "meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);" in apply_code
  assert apply_code.count("meta.current_iat = ") == 1


def test_mathunit_declaration_emitted_before_register_action(m2_mean_generated):
  _, register_actions_code, _, _ = m2_mean_generated
  mathunit_idx = register_actions_code.index(
      "MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;")
  action_idx = register_actions_code.index(
      "RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {")
  assert mathunit_idx < action_idx

  # The MathUnit<> declaration is specific to "mathunit_ewma"-bodied
  # registers -- must not leak in front of unrelated RegisterActions.
  assert register_actions_code.count("MathUnit<bit<16>>(MathOp_t.MUL, 1, 2)") == 1


def test_dedup_does_not_cross_contaminate_flow_and_fwd_namespaces(m2_mean_plus_fwd_generated):
  # flow_iat_max + fwd_iat_max + flow_iat_mean together: flow_last_arrival_time
  # is shared (flow_iat_max + flow_iat_mean) and deduped to one execute call;
  # fwd_last_arrival_time is a wholly separate, independently-touched
  # register (only fwd_iat_max references it) -- confirming the dedup fix
  # tracks *register names*, not e.g. a blanket "first dependency only" rule
  # that would incorrectly also swallow fwd's independent dependency.
  _, _, apply_code, _ = m2_mean_plus_fwd_generated
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
  # Empty genuinely means "nothing to do": there is no missing/uncatalogued
  # feature to raise about, so this path is untouched by the F2 fix below
  # and still returns empty strings (plus an empty resolved set).
  assert generate_P4_registers_and_apply({}) == ("", "", "", set())


def test_all_unknown_features_raises():
  # F2: a feature entirely absent from the catalog used to be silently
  # skipped, leaving a <feature>_val field declared, keyed on by a range
  # table, and never written -- reading 0 for every packet forever. The
  # raise itself lives one layer up from generate_P4_registers_and_apply
  # (which still just silently resolves whatever it can -- see its own
  # docstring; that contract is unchanged): generate_P4_code diffs the
  # requested feature set against what actually resolved and raises the
  # moment anything is missing. clf_app/clf_ddos=None ("no task", the same
  # convention generate_P4_code's own docstring uses for a missing task) is
  # enough to reach that raise -- feature resolution runs before any
  # tree/model-dependent work.
  # generate_P4_code needs real (non-None) interval lists ahead of the F2
  # check (it computes each resolved entry's codeword width from
  # len(intervals) before reaching the missing-feature diff) -- unlike
  # M1_FEATURE_INTERVALS above, whose None values are fine only for calling
  # generate_P4_registers_and_apply directly (it ignores values entirely).
  with pytest.raises(ValueError, match="Totally_Bogus_Feature"):
    bps.generate_P4_code(
        num_class_app=0, num_class_ddos=0,
        clf_app=None, clf_ddos=None,
        feature_intervals_app={"Totally_Bogus_Feature": [(0, 10), (11, 20)]},
        feature_intervals_ddos={},
    )


def test_unknown_feature_mixed_with_known_raises():
  # Same F2 raise, but mixed with a feature that DOES resolve -- confirms
  # the raise fires on the presence of ANY uncatalogued feature, not just
  # an all-unknown set.
  mixed_intervals = {
      "flow_iat_max": [(0, 100), (101, 500), (501, 9999)],
      "Totally_Bogus_Feature": [(0, 10), (11, 20)],
  }
  with pytest.raises(ValueError, match="Totally_Bogus_Feature"):
    bps.generate_P4_code(
        num_class_app=0, num_class_ddos=0,
        clf_app=None, clf_ddos=None,
        feature_intervals_app=mixed_intervals,
        feature_intervals_ddos={},
    )


def test_generate_P4_code_does_not_false_positive_raise_on_mixed_case_known_feature():
  # Regression guard for a latent case-sensitivity gap in the F2 raise
  # (build_p4_script.py, the `missing = ...` line right after
  # generate_P4_registers_and_apply's call in generate_P4_code):
  # `resolved` (that function's 4th return value) holds catalog-matched
  # names in LOWERCASE -- it builds `matched_features` via `name.lower()`
  # before checking catalog membership -- while `raw_feature_intervals`'s
  # keys preserve whatever case the caller supplied. A bare
  # `set(raw_feature_intervals) - resolved` would therefore treat a
  # mixed-case spelling of a feature that resolves JUST FINE
  # case-insensitively (e.g. "Flow_Iat_Max") as "missing" and raise a
  # false-positive ValueError for it. Every real production caller's
  # names are already canonical by the time they reach generate_P4_code
  # (get_nodes()'s normalise_feature_name(), Task 4) -- but this test
  # proves the guarantee holds at generate_P4_code's own layer too,
  # rather than leaving it silently assumed from an upstream invariant
  # this function doesn't itself enforce.
  bps.generate_P4_code(
      num_class_app=0, num_class_ddos=0,
      clf_app=None, clf_ddos=None,
      feature_intervals_app={"Flow_Iat_Max": [(0, 100), (101, 500), (501, 9999)]},
      feature_intervals_ddos={},
  )  # must not raise


def test_case_insensitive_feature_lookup(m1_generated):
  # M1_FEATURE_INTERVALS is already canonical (lowercase, underscore-joined
  # -- what get_nodes()/get_feature_intervals() actually produce). This
  # function's own .lower() is a defensive fallback for callers that build
  # a feature_intervals dict by hand rather than via get_nodes; confirm a
  # Title_Case caller still resolves to byte-identical output.
  titlecase_intervals = {name.title(): value for name, value in M1_FEATURE_INTERVALS.items()}
  assert generate_P4_registers_and_apply(titlecase_intervals) == m1_generated


# ---------------------------------------------------------------------------
# Guardrails (synthetic catalogs only -- never mutate the real M1 catalog)
# ---------------------------------------------------------------------------

def test_shared_dependency_register_deduplicated_across_many_sharers():
  # Proof of the dedup property _note_touch/_execute_lines implement: 5
  # synthetic features all sharing ONE register name must still produce
  # exactly ONE real `.execute()` call site for that register, no matter
  # how many features reference it -- whichever feature is resolved first
  # emits the call site, and every later feature reuses the value it
  # produced instead of re-invoking `.execute()`. (This used to also be
  # framed as proof that a register-touch-count guard, `MAX_REGISTER_
  # TOUCHES`, couldn't be organically tripped by sharing; that guard was
  # deleted -- re-verified dead against Phase 2's full 18-feature catalog,
  # where every register, including the shared `flow_last_arrival_time`/
  # `fwd_last_arrival_time`/`bwd_last_arrival_time` dependency registers,
  # still resolves to exactly one real touch -- so this test now stands on
  # its own as a direct proof of the dedup behaviour, not as a guard's
  # counter-example.)
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

  _, _, apply_code, _ = generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)

  # And the emitted code backs up the count: exactly one real .execute()
  # call site for the shared register, regardless of 5 features referencing
  # it (mirrors test_shared_dependency_register_executed_exactly_once, here
  # against a purely synthetic catalog rather than the real flow_iat_mean
  # scenario).
  assert apply_code.count("shared_synthetic_reg_action.execute(meta.flow_hash)") == 1


def test_unsupported_gated_by_raises():
  # "bwd" is now a supported gate class (Task 6) -- use a value that is
  # still genuinely unsupported to keep exercising this guard.
  synthetic_catalog = {
      "synthetic_feature": {
          "registers": [{
              "name": "synthetic_reg",
              "role": "value",
              "width": 16,
              "body": "running_max_iat",
          }],
          "gated_by": "sideways",
      },
  }
  feature_intervals = {"Synthetic_Feature": None}

  with pytest.raises(RuntimeError):
    generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)


def test_bwd_gated_feature_wrapped_in_meta_fwd_equals_0_block():
  synthetic_catalog = {
      "synthetic_bwd_feature": {
          "registers": [{
              "name": "synthetic_bwd_reg",
              "role": "value",
              "width": 16,
              "body": "running_max_iat",
          }],
          "gated_by": "bwd",
      },
  }
  feature_intervals = {"Synthetic_Bwd_Feature": None}

  _, _, apply_code, _ = generate_P4_registers_and_apply(
      feature_intervals, catalog=synthetic_catalog)

  assert "if (meta.fwd == 0) {" in apply_code
  assert "synthetic_bwd_reg_action.execute(meta.flow_hash)" in apply_code
  # The execute call site must be textually inside the fwd==0 block, not
  # emitted unconditionally.
  gate_index = apply_code.index("if (meta.fwd == 0) {")
  execute_index = apply_code.index("synthetic_bwd_reg_action.execute(meta.flow_hash)")
  assert execute_index > gate_index


def test_register_shared_across_gate_classes_raises():
  # A register first .execute()'d inside one gated block (or ungated) and
  # then reused by a feature in a DIFFERENT gate class would read a
  # value that only some packets ever set -- garbage for the rest. This
  # must be a real error, not silently-wrong P4.
  shared_register = {
      "name": "shared_cross_gate_reg",
      "role": "value",
      "width": 16,
      "body": "running_max_iat",
  }
  synthetic_catalog = {
      "ungated_feature": {
          "registers": [shared_register],
          "gated_by": None,
      },
      "fwd_feature": {
          "registers": [shared_register],
          "gated_by": "fwd",
      },
  }
  feature_intervals = {"Ungated_Feature": None, "Fwd_Feature": None}

  with pytest.raises(ValueError, match="shared_cross_gate_reg"):
    generate_P4_registers_and_apply(feature_intervals, catalog=synthetic_catalog)


def test_real_catalog_has_no_cross_gate_register_sharing():
  # Ground truth: resolving every feature in the real catalog together
  # must NOT trip the cross-gate hazard guard above.
  feature_intervals = {name: None for name in FEATURE_REGISTER_CATALOG}

  # Should not raise.
  generate_P4_registers_and_apply(feature_intervals)


def test_resolving_flow_iat_mean_alone_auto_executes_its_dependency():
  """Proves it is safe that Task 9 deleted the old
  "flow_iat_mean requires flow_iat_max" guard: flow_iat_mean's own catalog
  entry lists "flow_last_arrival_time" as its OWN dependency register (see
  FEATURE_REGISTER_CATALOG["flow_iat_mean"]), so resolving flow_iat_mean by
  itself -- with no flow_iat_max in the selected set at all -- still emits
  the dependency's .execute() call site, and BEFORE the value assignment
  that consumes its output. No external guard is needed: the auto-execution
  falls out of _execute_lines walking each feature's own "registers" list.
  """
  _, _, apply_code, resolved = generate_P4_registers_and_apply({"flow_iat_mean": None})

  assert resolved == {"flow_iat_mean"}
  assert "flow_iat_max" not in apply_code

  dependency_line = "flow_last_arrival_time_action.execute(meta.flow_hash)"
  value_line = "meta.flow_iat_mean_val ="
  dependency_index = apply_code.index(dependency_line)
  value_index = apply_code.index(value_line)
  assert dependency_index < value_index, apply_code


# Tests for register_names_for and register_width_bits (Task 5)

def test_register_names_for_deduplicates_shared_dependency_registers():
  """flow_last_arrival_time serves flow_iat_max AND flow_iat_mean, so the
  union is strictly smaller than the naive per-feature sum (spec 4.1)."""
  names = fr.register_names_for(['flow_iat_max', 'flow_iat_mean'])
  assert names.count('flow_last_arrival_time') == 1
  assert len(names) == 3          # 1 shared dependency + 2 value registers


def test_register_names_for_normalises_training_feature_names():
  """The catalog is keyed on normalised names; the campaign passes
  'Flow.IAT.Max'. Missing this makes every register_count read low."""
  assert fr.register_names_for(['Flow.IAT.Max']) == \
      fr.register_names_for(['flow_iat_max'])


def test_the_whole_catalog_needs_twenty_one_distinct_registers():
  """Spec 4.2: at k=17 the true figure is 20, not the reviewer's 17.
  Over all 18 features it is 21 -- 18 value + 3 shared *_last_arrival_time."""
  assert len(fr.register_names_for(fr.FEATURE_REGISTER_CATALOG)) == 21


def test_register_width_bits_happy_path():
  """register_width_bits returns the correct width for a known register name."""
  # flow_last_arrival_time is a known register with width 16
  assert fr.register_width_bits("flow_last_arrival_time") == 16
  # flow_iat_max is also a known register with width 16
  assert fr.register_width_bits("flow_iat_max") == 16
  # All registers in the catalog have width 16
  assert fr.register_width_bits("fwd_packet_length_max") == 16


def test_register_width_bits_miss_path():
  """register_width_bits raises ValueError for an unknown register name."""
  with pytest.raises(ValueError, match="Register 'totally_unknown_register' not found"):
    fr.register_width_bits("totally_unknown_register")


def test_register_names_for_catalog_miss_is_silently_skipped():
  """When a feature name is not in the catalog after normalization,
  register_names_for silently skips it (consistent with feature_readiness_level
  precedent). This test ensures the skip behavior is preserved and any future
  refactor cannot accidentally change it to a KeyError without notice."""
  # Mix a known feature with a completely unknown feature
  features = ["flow_iat_max", "totally_unknown_feature"]
  result = fr.register_names_for(features)
  # Should only contain registers for flow_iat_max, not raise or include the unknown
  assert "flow_last_arrival_time" in result
  assert "flow_iat_max" in result
  assert len(result) == 2  # Only the two registers from flow_iat_max

  # All-unknown case should return empty tuple
  result_all_unknown = fr.register_names_for(["unknown_a", "unknown_b", "unknown_c"])
  assert result_all_unknown == ()
