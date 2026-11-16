"""
Feature -> TNA register dependency catalog.

This is a Python-side data catalog (no P4 text lives here) used by
`generate_P4_registers_and_apply()` in `build_p4_script.py` to figure out
which Tofino (TNA) `Register<>`s a selected feature set needs, and how to
wire the atomic read-modify-write logic for each of them. It targets the
TNA architecture validated in
`p4/tofino_spike/tna_m1_flows_iat_spike.p4` (compiled successfully this
session with the real `p4c -b tofino -a tna` compiler) -- read that file's
header comment before changing anything here.

This is the EXAMPLE catalog: the 18 features this project's generator
supports. A reader modelling their own P4 program replaces this catalog
(and its callers' inputs) with their own feature/register set -- nothing
about the model in p4model/ depends on these particular 18 entries.

Key casing
----------
Catalog keys are lowercase, underscore-joined feature names (e.g.
"flow_iat_max", "fwd_iat_max", "fwd_packet_length_max"). `feature_intervals`
(built by `get_nodes()`/`get_feature_intervals()` in build_p4_script.py) now
uses exactly this casing too: `get_nodes()`'s feature-name extraction runs
every parsed name through `normalise_feature_name()`, which lowercases it
and collapses every run of non-alphanumeric characters (spaces, dots,
underscores alike -- dataset.py ships dot-separated names like
"Flow.IAT.Max") to a single "_". So a `feature_intervals` key can be looked
up in this catalog directly, with no further transformation needed. Callers
that build a feature_intervals-shaped dict by hand rather than via
get_nodes should still route the key through
`build_p4_script.normalise_feature_name()` first, for the same reason.

Entry shape
-----------
FEATURE_REGISTER_CATALOG maps a lowercase feature name to:
{
    "registers": [
        {
            "name": <str>,           # register base name (P4 identifier
                                      # will be "<name>_reg" / "<name>_action")
            "role": "dependency" | "value",
                # "dependency": this register's value never itself becomes
                #   a codeword feature value (e.g. a *_last_arrival_time
                #   register) -- its `.execute()` result feeds its own
                #   scratch field (build_p4_script.iat_field_name, e.g.
                #   `meta.flow_current_iat`), consumed by the paired "value"
                #   register(s).
                # "value": this register's `.execute()` result IS the
                #   feature value later consumed by feature-encoding
                #   tables, assigned to `meta.<feature>_val` (matching the
                #   spike's `metadata_t` field naming, e.g.
                #   `meta.flow_iat_max_val`).
            "width": <int>,          # register bit width: 16 for every value
                                      # register (the spike's and the original
                                      # v1model's bit<16>); 32 for the
                                      # last-arrival ("iat_delta") registers
                                      # since Task 34 (V1o: they hold the
                                      # 32-bit arrival clock, see
                                      # build_p4_script._REGISTER_ACTION_BODIES)
                                      # and for the flow duration (Task 38: its
                                      # 32-bit first arrival); 64 for the Fwd /
                                      # Bwd duration pair registers {first, dur}
                                      # (two 32-bit halves, Task 38). Whatever the
                                      # width, the feature value <feature>_val is
                                      # 16 bits.
            "body": <str>,           # symbolic RegisterAction body kind;
                                      # see build_p4_script.py's
                                      # _REGISTER_ACTION_BODIES for the
                                      # exact P4 text each kind expands to.
        },
        ...
    ],
    "gated_by": None | "fwd" | "bwd",
        # None: this feature's registers update on every packet
        #   (whole-flow feature).
        # "fwd": this feature's registers update only when
        #   meta.fwd == 1 (forward-direction-only feature).
        # "bwd": this feature's registers update only when
        #   meta.fwd == 0 (backward-direction-only feature).
        # Since decisions 1a / 7.1 option B (2026-11-14) every register is
        #   executed on every packet; a fwd/bwd register tests meta.fwd in its
        #   own body and only reads in the other direction
        #   (build_p4_script._DIRECTION_READ_BODIES). Support for
        #   this gate class was added in Task 6 (see
        #   generate_P4_registers_and_apply in build_p4_script.py) and
        #   validated by compiling p4/tofino_spike/tna_m3_bwd_gate_spike.p4
        #   (0 errors). Task 9 populated the bwd_* entries themselves
        #   (bwd_iat_max, bwd_iat_mean, bwd_iat_min, bwd_packet_length_max,
        #   bwd_packet_length_min, bwd_packet_length_mean; the two Means are
        #   bwd_iat_total / bwd_packet_length_total since Task 38).
}

A feature's "registers" list is ordered: a "dependency" register (if any)
always precedes the "value" register(s) that consume its
scratch-field output (`meta.<chain>_current_iat`), matching the order the generator emits them
into the apply block.

Scope
-----
(History first: the Milestone / Task 9 paragraphs name features and body
kinds -- the Means, mathunit_ewma_packet_length -- that Task 38 removed; the
Task 38 paragraph below them is the current state.)

Milestone 1's 3 validated features (and their direct register
dependencies) are populated here: flow_iat_max, fwd_iat_max,
fwd_packet_length_max -- the M1 DDoS-only feature set traced against
p4/p4_code_RF_models.p4's apply block and validated by compiling
p4/tofino_spike/tna_m1_flows_iat_spike.p4. Milestone 2 adds one more
validated feature, flow_iat_mean (the App task's mean/EWMA feature),
traced against and validated by compiling
p4/tofino_spike/tna_m2_mean_spike.p4 (flow_duration since Task 38, below).

Task 9 (Phase 2's capstone) adds the remaining 14 features that make up
main.py's full 18-feature pool (main.py:308-314): flow_iat_min,
fwd_iat_mean, fwd_iat_min, bwd_iat_max, bwd_iat_mean, bwd_iat_min,
fwd_packet_length_min, fwd_packet_length_mean, bwd_packet_length_max,
bwd_packet_length_min, bwd_packet_length_mean, min_packet_length,
max_packet_length, and packet_length_mean (the five Means among them replaced
in Task 38, below) -- all wired from body kinds
already validated by real p4c compiles in Tasks 6-8 (bwd gating,
running_min_iat, running_min_packet_length, mathunit_ewma_packet_length);
no new body kinds or P4 syntax were introduced to add these entries. One
new dependency register, bwd_last_arrival_time, was added alongside them,
reusing the existing "iat_delta" body kind (the same body
flow_last_arrival_time and fwd_last_arrival_time already use) -- it is a
new register instance, not a new body kind.

Task 38 (P3h, user decision 2026-11-15) REPLACED the six Mean features. A
Tofino-1 stateful ALU has no divide and no shift, and the MathUnit "EWMA"
bodies the Means used were miscompiled by p4c into a running sum of halves
(Task 33b, live; probe_mean.md), so no Mean was ever what its name says. In
their place, each in the switch's own unit (dataset.py derives the training
columns in the same unit, so the control plane installs every threshold
unchanged):
  * flow_duration, fwd_iat_total, bwd_iat_total: DURATIONS, last minus first
    arrival of the flow / of the direction, on a coarse clock of 65.536 us
    ticks (calc_timestamp: meta.now_coarse = ig_intr_md.ingress_mac_tstamp
    [47:16]), clamped at 65535 (4.3 s). Body kinds "coarse_duration" (the
    flow's: a 32-bit register holding the first arrival) and
    "coarse_duration_pair" (Fwd / Bwd: a 64-bit {first, dur} pair, so the
    other direction's packets read the owner's duration too). No dependency
    register: a duration needs no gap, so a chain's last-arrival register is
    emitted only when some IAT Max / Min of that chain is selected.
  * fwd_packet_length_total, bwd_packet_length_total, packet_length_total:
    saturating sums of hdr.ipv4.total_len (body kind
    "saturating_total_packet_length", `value |+| total_len`), the flow's =
    fwd + bwd in training.
Evidence: Task 37 (results/encoding_evidence_2026_11/model8/impl/
probe_p3c.md): p4c compiled exactly these bodies at no stage or block cost,
and an exact emulation of the compiled program matched tofino-model live on
64/64 packets. Each body's text and its evidence are at its emission site,
build_p4_script._REGISTER_ACTION_BODIES / _DIRECTION_READ_BODIES. Before
Task 38 the catalog had flow_iat_mean, fwd_iat_mean, bwd_iat_mean,
fwd_packet_length_mean, bwd_packet_length_mean and packet_length_mean; a
program or forest that names them is an older generator's (program_checks
refuses such a text).

Do not add entries for features not yet validated by a real p4c compile --
resolving those is explicitly deferred to whichever later milestone needs
them, not guessed up front.

Note: two things the generator always emits are NOT catalog entries, and
neither goes through the register_order/_note_touch dedup machinery. Both are
fixed, generator-level requirements whenever the resolved feature set is
non-empty (see generate_P4_registers_and_apply's docstring in
build_p4_script.py):

  * the single `flow_hash_calc` Hash<> instance, whose `calc_flow_hash`
    action produces the canonical, direction-independent flow index
    meta.flow_hash that every per-flow register is indexed by. It is a Hash<>,
    not a register.
  * `flow_forward_srcaddr_reg` (build_p4_script.py:2048), the one register
    backing `flow_orientation_action` (build_p4_script.py:2087), which
    resolves meta.fwd. Its .execute() is emitted unconditionally, so it always
    claims a stateful ALU in the first register stage and every fwd-/bwd-gated
    register waits a stage on it -- which is why program.ORIENTATION_REGISTER
    exists and why the schedule has to add it by hand.

An earlier note here described a `flows` bookkeeping register (fwd/bwd/new-flow
tracking) as the thing producing meta.flow_hash. The generator no longer emits
any such register, and the description was wrong about the hash besides: the
symmetric-hash change (build_p4_script.py:1849-1866) replaced the old two-touch
direction logic with the single flow_orientation_action touch above, and
meta.flow_hash comes from the symmetric calc_flow_hash action (an @symmetric
12-bit CRC32 of the 5-tuple since review A1, an XOR of each address/port pair
before), not from a register.

register_width_bits has no reader inside the model itself -- its only
one, register_sram_bits, was dropped in Task 1 -- but it is kept here as
part of the catalog's public surface (build_p4_script.py and
tests/test_feature_registers.py still call it directly).
"""
from p4model.names import normalise_feature_name

FEATURE_REGISTER_CATALOG = {
    "flow_iat_max": {
        "registers": [
            {
                "name": "flow_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "flow_iat_max",
                "role": "value",
                "width": 16,
                "body": "running_max_iat",
            },
        ],
        "gated_by": None,
    },
    "fwd_iat_max": {
        "registers": [
            {
                "name": "fwd_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "fwd_iat_max",
                "role": "value",
                "width": 16,
                "body": "running_max_iat",
            },
        ],
        "gated_by": "fwd",
    },
    "fwd_packet_length_max": {
        "registers": [
            {
                "name": "fwd_packet_length_max",
                "role": "value",
                "width": 16,
                "body": "running_max_packet_length",
            },
        ],
        "gated_by": "fwd",
    },
    # Task 38 (replaces flow_iat_mean): the flow's duration in 65.536 us ticks,
    # clamped at 65535. ONE value register holding the flow's first arrival
    # (32 bits of the coarse clock), no dependency register: a duration needs no
    # gap, so flow_last_arrival_time is emitted only for flow_iat_max / _min.
    "flow_duration": {
        "registers": [
            {
                "name": "flow_duration",
                "role": "value",
                "width": 32,
                "body": "coarse_duration",
            },
        ],
        "gated_by": None,
    },

    # ------------------------------------------------------------------
    # Task 9: the remaining 14 features of main.py's 18-feature pool.
    # Every body kind used below was already validated by a real p4c
    # compile in Tasks 6-8 -- this section only wires catalog entries
    # against them, dependency-first, per feature -- see the module
    # docstring's "Task 9" paragraph above for the full list. Task 38
    # replaced the five Means among them (module docstring).
    # ------------------------------------------------------------------

    # Lists "flow_last_arrival_time" as its dependency -- the SAME dependency
    # register "flow_iat_max" (above) already declares.
    # generate_P4_registers_and_apply() dedupes shared dependency registers
    # by name, so this is safe: whichever selected feature is resolved first
    # .execute()s flow_last_arrival_time_reg once, and both features'
    # "value" registers consume that single meta.flow_current_iat result.
    # A catalog entry that lists its own dependency register can safely stand
    # alone -- its dependency will be executed when that feature is resolved.
    "flow_iat_min": {
        "registers": [
            {
                "name": "flow_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "flow_iat_min",
                "role": "value",
                "width": 16,
                "body": "running_min_iat",
            },
        ],
        "gated_by": None,
    },
    # Task 38 (replaces fwd_iat_mean): the forward direction's duration (CSV
    # Fwd.IAT.Total = last minus first forward arrival) in 65.536 us ticks,
    # clamped at 65535. ONE value register, a 64-bit {first, dur} pair: a register
    # holding only the first arrival could not give the duration on a backward
    # packet. No dependency register (see flow_duration).
    "fwd_iat_total": {
        "registers": [
            {
                "name": "fwd_iat_total",
                "role": "value",
                "width": 64,
                "body": "coarse_duration_pair",
            },
        ],
        "gated_by": "fwd",
    },
    # Shares "fwd_last_arrival_time" with fwd_iat_max above.
    "fwd_iat_min": {
        "registers": [
            {
                "name": "fwd_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "fwd_iat_min",
                "role": "value",
                "width": 16,
                "body": "running_min_iat",
            },
        ],
        "gated_by": "fwd",
    },

    # "bwd_last_arrival_time" is a NEW dependency register (Task 9) -- it
    # reuses the existing "iat_delta" body kind (the same body
    # flow_last_arrival_time and fwd_last_arrival_time already use above),
    # just a new register instance for the backward direction. bwd_iat_max and
    # bwd_iat_min both declare it and rely on
    # generate_P4_registers_and_apply()'s by-name dedup (same story as
    # flow_iat_min's comment above) to .execute() it exactly once.
    "bwd_iat_max": {
        "registers": [
            {
                "name": "bwd_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "bwd_iat_max",
                "role": "value",
                "width": 16,
                "body": "running_max_iat",
            },
        ],
        "gated_by": "bwd",
    },
    # Task 38 (replaces bwd_iat_mean): as fwd_iat_total, for the backward
    # direction.
    "bwd_iat_total": {
        "registers": [
            {
                "name": "bwd_iat_total",
                "role": "value",
                "width": 64,
                "body": "coarse_duration_pair",
            },
        ],
        "gated_by": "bwd",
    },
    "bwd_iat_min": {
        "registers": [
            {
                "name": "bwd_last_arrival_time",
                "role": "dependency",
                "width": 32,
                "body": "iat_delta",
            },
            {
                "name": "bwd_iat_min",
                "role": "value",
                "width": 16,
                "body": "running_min_iat",
            },
        ],
        "gated_by": "bwd",
    },

    # Packet-length features have no dependency register at all -- their
    # value bodies read hdr.ipv4.total_len directly, not a dependency
    # register's meta.<chain>_current_iat scratch field
    # (see running_max_packet_length/running_min_packet_length/
    # saturating_total_packet_length in build_p4_script.py's
    # _REGISTER_ACTION_BODIES). The three *_packet_length_total entries replace
    # the length Means (Task 38): a saturating 16-bit sum of the IP total lengths.
    "fwd_packet_length_min": {
        "registers": [
            {
                "name": "fwd_packet_length_min",
                "role": "value",
                "width": 16,
                "body": "running_min_packet_length",
            },
        ],
        "gated_by": "fwd",
    },
    "fwd_packet_length_total": {
        "registers": [
            {
                "name": "fwd_packet_length_total",
                "role": "value",
                "width": 16,
                "body": "saturating_total_packet_length",
            },
        ],
        "gated_by": "fwd",
    },
    "bwd_packet_length_max": {
        "registers": [
            {
                "name": "bwd_packet_length_max",
                "role": "value",
                "width": 16,
                "body": "running_max_packet_length",
            },
        ],
        "gated_by": "bwd",
    },
    "bwd_packet_length_min": {
        "registers": [
            {
                "name": "bwd_packet_length_min",
                "role": "value",
                "width": 16,
                "body": "running_min_packet_length",
            },
        ],
        "gated_by": "bwd",
    },
    "bwd_packet_length_total": {
        "registers": [
            {
                "name": "bwd_packet_length_total",
                "role": "value",
                "width": 16,
                "body": "saturating_total_packet_length",
            },
        ],
        "gated_by": "bwd",
    },
    "min_packet_length": {
        "registers": [
            {
                "name": "min_packet_length",
                "role": "value",
                "width": 16,
                "body": "running_min_packet_length",
            },
        ],
        "gated_by": None,
    },
    "max_packet_length": {
        "registers": [
            {
                "name": "max_packet_length",
                "role": "value",
                "width": 16,
                "body": "running_max_packet_length",
            },
        ],
        "gated_by": None,
    },
    "packet_length_total": {
        "registers": [
            {
                "name": "packet_length_total",
                "role": "value",
                "width": 16,
                "body": "saturating_total_packet_length",
            },
        ],
        "gated_by": None,
    },
}


def register_names_for(features, catalog=None):
  """Return an ordered, deduplicated tuple of all register names needed by features.

  features: Iterable of feature names (strings) or a dict-like object with keys
    (e.g., FEATURE_REGISTER_CATALOG itself). Feature names are normalized via
    normalise_feature_name() to match catalog keys.

  catalog: Dict mapping normalized feature names to catalog entries (each with a
    "registers" list). Defaults to FEATURE_REGISTER_CATALOG.

  Returns: tuple[str, ...] of register names (e.g., "flow_last_arrival_time",
    "flow_iat_max"), ordered dependency-first (as stored in catalog entries)
    and deduplicated by first-seen (the ordering matches the first catalog
    entry that declares each register, across all selected features).

  Example:
    >>> register_names_for(['flow_iat_max', 'flow_iat_min', 'flow_duration'])
    ('flow_last_arrival_time', 'flow_iat_max', 'flow_iat_min', 'flow_duration')

  Note: This helper does NOT replicate the cross-gate sharing guard from
  generate_P4_registers_and_apply's _note_touch closure (build_p4_script.py
  lines 2002-2012). That guard prevents a register from being shared between
  two features with different gate classes (e.g., one ungated, one fwd-gated).
  This function only tracks register names as they appear, in order; validation
  of the cross-gate constraint is deferred to generate_P4_registers_and_apply.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  register_order = []
  seen = set()

  for feature_name in features:
    # Normalize the feature name to match catalog keys
    normalized_name = normalise_feature_name(feature_name)
    entry = catalog.get(normalized_name)
    if entry is None:
      # Skip features not in the catalog
      continue

    # Walk through the registers in this feature's entry (dependency-first)
    for register in entry["registers"]:
      reg_name = register["name"]
      if reg_name not in seen:
        register_order.append(reg_name)
        seen.add(reg_name)

  return tuple(register_order)


def register_width_bits(name, catalog=None):
  """Return the bit width of a register by name.

  name: Register name (e.g., "flow_last_arrival_time", "flow_iat_max").

  catalog: Dict mapping normalized feature names to catalog entries. Defaults to
    FEATURE_REGISTER_CATALOG.

  Returns: int, the bit width of the named register (e.g., 16).

  Raises: ValueError if the register name is not found in any catalog entry.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # Search through all features for the register
  for entry in catalog.values():
    for register in entry["registers"]:
      if register["name"] == name:
        return register["width"]

  raise ValueError(f"Register '{name}' not found in catalog")
