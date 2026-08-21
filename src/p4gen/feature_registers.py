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
    "gated_by": None | "fwd",
        # None: this feature's registers are touched unconditionally
        #   (whole-flow feature).
        # "fwd": this feature's registers are only touched when
        #   meta.fwd == 1 (forward-direction-only feature). "bwd" gating
        #   is intentionally not modeled yet -- out of scope until a
        #   milestone that actually needs it validates the design.
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
for the dependency-sharing design. Do not add entries
for features not yet validated by a real p4c compile (bwd_*, other
candidates) -- resolving those is explicitly deferred to whichever later
milestone needs them, not guessed up front.

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
}
