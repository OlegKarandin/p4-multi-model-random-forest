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
                #   register) -- its `.execute()` result feeds
                #   `meta.current_iat`, consumed by a paired "value"
                #   register.
                # "value": this register's `.execute()` result IS the
                #   feature value later consumed by feature-encoding
                #   tables, assigned to `meta.<feature>_val` (matching the
                #   spike's `metadata_t` field naming, e.g.
                #   `meta.flow_iat_max_val`).
            "width": <int>,          # register bit width (16 for all M1
                                      # registers, matching the spike and
                                      # the original v1model bit<16>).
            "body": <str>,           # symbolic RegisterAction body kind;
                                      # see build_p4_script.py's
                                      # _REGISTER_ACTION_BODIES for the
                                      # exact P4 text each kind expands to.
        },
        ...
    ],
    "gated_by": None | "fwd" | "bwd",
        # None: this feature's registers are touched unconditionally
        #   (whole-flow feature).
        # "fwd": this feature's registers are only touched when
        #   meta.fwd == 1 (forward-direction-only feature).
        # "bwd": this feature's registers are only touched when
        #   meta.fwd == 0 (backward-direction-only feature). Support for
        #   this gate class was added in Task 6 (see
        #   generate_P4_registers_and_apply in build_p4_script.py) and
        #   validated by compiling p4/tofino_spike/tna_m3_bwd_gate_spike.p4
        #   (0 errors). Task 9 populated the bwd_* entries themselves
        #   (bwd_iat_max, bwd_iat_mean, bwd_iat_min, bwd_packet_length_max,
        #   bwd_packet_length_min, bwd_packet_length_mean).
}

A feature's "registers" list is ordered: a "dependency" register (if any)
always precedes the "value" register(s) that consume its
`meta.current_iat` output, matching the order the generator emits them
into the apply block.

Scope
-----
Milestone 1's 3 validated features (and their direct register
dependencies) are populated here: flow_iat_max, fwd_iat_max,
fwd_packet_length_max -- the M1 DDoS-only feature set traced against
p4/p4_code_RF_models.p4's apply block and validated by compiling
p4/tofino_spike/tna_m1_flows_iat_spike.p4. Milestone 2 adds one more
validated feature, flow_iat_mean (the App task's mean/EWMA feature),
traced against and validated by compiling
p4/tofino_spike/tna_m2_mean_spike.p4 -- see that entry's own comment below
for the dependency-sharing design.

Task 9 (Phase 2's capstone) adds the remaining 14 features that make up
main.py's full 18-feature pool (main.py:308-314): flow_iat_min,
fwd_iat_mean, fwd_iat_min, bwd_iat_max, bwd_iat_mean, bwd_iat_min,
fwd_packet_length_min, fwd_packet_length_mean, bwd_packet_length_max,
bwd_packet_length_min, bwd_packet_length_mean, min_packet_length,
max_packet_length, and packet_length_mean -- all wired from body kinds
already validated by real p4c compiles in Tasks 6-8 (bwd gating,
running_min_iat, running_min_packet_length, mathunit_ewma_packet_length);
no new body kinds or P4 syntax were introduced to add these entries. One
new dependency register, bwd_last_arrival_time, was added alongside them,
reusing the existing "iat_delta" body kind (the same body
flow_last_arrival_time and fwd_last_arrival_time already use) -- it is a
new register instance, not a new body kind.

ACCURACY CAVEAT for every "*_mean" feature in this catalog: the
"mathunit_ewma"/"mathunit_ewma_packet_length" body kinds compute an
alpha=0.5 exponentially-weighted moving average (new = (old + current) / 2
via Tofino's MathUnit<> hardware primitive), NOT the dataset column's
faithful arithmetic mean. This was a known, deliberate redesign already
accepted for flow_iat_mean (see p4/tofino_spike/tna_m2_mean_spike.p4's
header comment for why the two simpler, truer-to-arithmetic-mean designs
both failed against the real Tofino compiler) and Task 9 extends the same
approximation to five more "*_mean" features: fwd_iat_mean, bwd_iat_mean,
fwd_packet_length_mean, bwd_packet_length_mean, and packet_length_mean.
Every downstream accuracy number for any of these six "*_mean" features is
therefore the switch's EWMA approximation, not the dataset's column value
-- keep that in mind when comparing switch-side accuracy against
sklearn-side accuracy for a model trained on the true column mean.

Do not add entries for features not yet validated by a real p4c compile --
resolving those is explicitly deferred to whichever later milestone needs
them, not guessed up front.

Note: the `flows` bookkeeping register (fwd/bwd/new-flow tracking) is NOT
a catalog entry. It is a fixed, generator-level requirement whenever the
resolved feature set is non-empty -- see generate_P4_registers_and_apply's
docstring in build_p4_script.py -- because it is the only thing that
produces the canonical, direction-independent flow index (meta.flow_hash)
every other per-flow register (in this catalog or not) relies on.
"""

FEATURE_REGISTER_CATALOG = {
    "flow_iat_max": {
        "registers": [
            {
                "name": "flow_last_arrival_time",
                "role": "dependency",
                "width": 16,
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
                "width": 16,
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
    # Deliberately lists "flow_last_arrival_time" as its dependency -- the
    # SAME dependency register "flow_iat_max" (above) already declares.
    # generate_P4_registers_and_apply() dedupes shared dependency registers
    # by name, so this is safe: whichever selected feature is resolved first
    # .execute()s flow_last_arrival_time_reg once, and both features'
    # "value" registers consume that single meta.current_iat result.
    # A catalog entry that lists its own dependency register can safely stand
    # alone -- its dependency will be executed when that feature is resolved.
    "flow_iat_mean": {
        "registers": [
            {
                "name": "flow_last_arrival_time",
                "role": "dependency",
                "width": 16,
                "body": "iat_delta",
            },
            {
                "name": "flow_iat_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma",
            },
        ],
        "gated_by": None,
    },

    # ------------------------------------------------------------------
    # Task 9: the remaining 14 features of main.py's 18-feature pool.
    # Every body kind used below was already validated by a real p4c
    # compile in Tasks 6-8 -- this section only wires catalog entries
    # against them, dependency-first, per feature -- see the module
    # docstring's "Task 9" paragraph above for the full list and the
    # accuracy caveat that applies to every "*_mean" entry here.
    # ------------------------------------------------------------------

    # Shares "flow_last_arrival_time" with flow_iat_max/flow_iat_mean above
    # (same dedup story as flow_iat_mean's comment describes).
    "flow_iat_min": {
        "registers": [
            {
                "name": "flow_last_arrival_time",
                "role": "dependency",
                "width": 16,
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
    # Shares "fwd_last_arrival_time" with fwd_iat_max above.
    "fwd_iat_mean": {
        "registers": [
            {
                "name": "fwd_last_arrival_time",
                "role": "dependency",
                "width": 16,
                "body": "iat_delta",
            },
            {
                "name": "fwd_iat_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma",
            },
        ],
        "gated_by": "fwd",
    },
    # Shares "fwd_last_arrival_time" with fwd_iat_max/fwd_iat_mean above.
    "fwd_iat_min": {
        "registers": [
            {
                "name": "fwd_last_arrival_time",
                "role": "dependency",
                "width": 16,
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
    # just a new register instance for the backward direction. The three
    # bwd_iat_* entries below all declare it and rely on
    # generate_P4_registers_and_apply()'s by-name dedup (same story as
    # flow_iat_mean's comment above) to .execute() it exactly once.
    "bwd_iat_max": {
        "registers": [
            {
                "name": "bwd_last_arrival_time",
                "role": "dependency",
                "width": 16,
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
    "bwd_iat_mean": {
        "registers": [
            {
                "name": "bwd_last_arrival_time",
                "role": "dependency",
                "width": 16,
                "body": "iat_delta",
            },
            {
                "name": "bwd_iat_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma",
            },
        ],
        "gated_by": "bwd",
    },
    "bwd_iat_min": {
        "registers": [
            {
                "name": "bwd_last_arrival_time",
                "role": "dependency",
                "width": 16,
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
    # value bodies read hdr.ipv4.total_len directly, not meta.current_iat
    # (see running_max_packet_length/running_min_packet_length/
    # mathunit_ewma_packet_length in build_p4_script.py's
    # _REGISTER_ACTION_BODIES).
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
    "fwd_packet_length_mean": {
        "registers": [
            {
                "name": "fwd_packet_length_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma_packet_length",
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
    "bwd_packet_length_mean": {
        "registers": [
            {
                "name": "bwd_packet_length_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma_packet_length",
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
    "packet_length_mean": {
        "registers": [
            {
                "name": "packet_length_mean",
                "role": "value",
                "width": 16,
                "body": "mathunit_ewma_packet_length",
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
    >>> register_names_for(['flow_iat_max', 'flow_iat_mean'])
    ('flow_last_arrival_time', 'flow_iat_max', 'flow_iat_mean')

  Note: This helper does NOT replicate the cross-gate sharing guard from
  generate_P4_registers_and_apply's _note_touch closure (build_p4_script.py
  lines 2002-2012). That guard prevents a register from being shared between
  two features with different gate classes (e.g., one ungated, one fwd-gated).
  This function only tracks register names as they appear, in order; validation
  of the cross-gate constraint is deferred to generate_P4_registers_and_apply.
  """
  # Import here to avoid circular dependency
  from src.p4gen.build_p4_script import normalise_feature_name

  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # If features is a dict-like object (e.g., FEATURE_REGISTER_CATALOG), use keys()
  if hasattr(features, 'keys'):
    feature_names = features.keys()
  else:
    feature_names = features

  register_order = []
  seen = set()

  for feature_name in feature_names:
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
