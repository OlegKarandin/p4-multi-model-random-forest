"""Tofino 1 facts: measured off the real chip or read straight out of the
compiler's own sizing logic (mau_spec.h, sizing/placement logs). Retargeting
this model to a different ASIC means changing this file and nothing else --
program.py describes choices this generator makes, not the hardware under
it."""

TCAM_BLOCKS_PER_STAGE = 24
# ...and those 24 blocks are not one undifferentiated pool. mau_spec.h:88-90 gives
# Tofino_tcam_rows=12, Tofino_tcam_columns=2, with an explicit source comment that the
# figure is correct for Tofino 1, 2 and 3. A table needing several blocks chains them
# down ONE column, so which tables can share a stage depends on their widths and not
# only on their total -- three 8-block tables total exactly 24 and still need two
# stages, while four 6-block tables (also 24) fit in one. Measured directly against
# real p4c over synthetic tables of 5..12 blocks: scripts/tcam_column_sweep.py, and
# reviews/p4_tofino_reference.md Appendix B "Mechanism C". packing.fits_two_columns is
# the packing test; TCAM_BLOCKS_PER_STAGE remains the (implied) total.
TCAM_ROWS_PER_STAGE = 12
TCAM_COLUMNS_PER_STAGE = 2
TCAM_BLOCK_KEY_LENGTH = 44
# ...and those 44 bits are 5 PRIVATE bytes plus one nibble of a MIDBYTE (5 x 8 + 4
# = 44). The ternary input crossbar is 12 groups x 5 private bytes + 6 midbytes =
# 66 bytes total. The split matters because the mandatory 2-bit --version-- field
# may live ONLY in a midbyte nibble: a key that leaves no nibble free costs an
# extra TCAM block to hold two bits. See tables.crossbar_capacity /
# tables.codeword_to_blocks and reviews/github_issue_tcam_version_bit_packing.md
# Sec 1.2-1.3.
#
# A note this comment used to carry and that is now RETRACTED (2026-09-21): that
# a midbyte is owned exclusively by its "pair partner" groups 2i / 2i+1, laid out
# private-x5 | midbyte | private-x5. p4c's own assembly says otherwise -- a block
# may pair with ANY of a stage's 6 midbytes, and a key's groups need not even be
# consecutive (measured runs {0,1,3,4}, {0,3}). Only the resulting COUNT survives,
# as tables.crossbar_capacity; the geometry, and with it the whole start_group /
# version_block_penalty apparatus it justified, is gone. reviews/
# p4_tofino_reference.md Sec 4.1.1 and Appendix B "Mechanism G".
CROSSBAR_PRIVATE_BYTES_PER_GROUP = 5
# The stage's crossbar, in the two units the groups above come in. Documentation
# and provenance ONLY -- deliberately consumed by nothing, because no group-BUDGET
# term exists in this model. The near-cap effect a budget would price (spec F5)
# is handled by the lane simulation (lanes.py) alone; the placement refusal that once sat beside it was retired 2026-09-29 (see below).
#
# Provenance, and a retraction. scripts/tcam_group_cap_probe.py's point
# `groups_13_bytes_64` was read for months as "a per-stage group cap was probed
# and NOT found -- two keys needing 7 + 6 = 13 groups landed in one stage that has
# 12". That reading was wrong on its own evidence: 7 and 6 are BLOCK counts, the
# two tables share group 5, and the probe's assembly uses groups 0-11 -- exactly
# 12 of 12, landing ON the cap rather than past it. The cap is real and is this
# constant.
# Provenance: ir/tofino.def:38, MAX_TERNARY_GROUPS (model_audit_2026-09-27.md
# Appendix C).
TERNARY_CROSSBAR_GROUPS_PER_STAGE = 12      # 5 private bytes each
TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE = 6  # midbytes, 2 nibbles each
TERNARY_MATCHING_ENTRIES_PER_BLOCK = 512
# Provenance: mau/tofino/memories.h:54, TERNARY_TABLES_MAX (model_audit_2026-
# 09-27.md Appendix C).
TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8    # hard cap, binds for narrow keys (<=64 bits)
TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE = 64    # byte budget, binds for wider keys
# What one stage's crossbar holds across ALL its keys, as opposed to the limit
# above, which MAX_CODEWORD_LENGTH and CrossbarKeyTooWide read for a lone key and
# which has only been validated at 64. 12 groups x 5 private bytes + 6 midbytes
# (p4c's input_xbar.cpp:1319 says 66), and real stages of 65 and 66 distinct bytes
# were built. Replay of 691 designs: 5 stage depths corrected, 0 regressions.
# Bytes alone do not admit a stage at 65-66: the lane simulation still has to fit
# every key (lanes.price_with_supply returning None is what makes a stage rise).
TERNARY_CROSSBAR_STAGE_BYTES = 66
# RETIRED 2026-09-29: TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62, the
# "greedy-give-up safety net" that refused a stage holding two different keys
# past 62 combined crossbar bytes. Its one real-design case, M150_k7_s11
# (22 + 41 = 63 bytes, p4c moved the app key a stage), was a PHV LAYOUT
# artifact: with the generator's @pa_container_size pins on every tree-key
# code_* field (build_p4_script.code_field_container_sizes), p4c places both
# keys in one stage, as the lane simulation does. Every probe point at 56-64
# bytes (tcam_mixed_key_cap_sweep, tcam_discount_scan dsp01-42) stayed in one
# stage and _stage_key_prices reproduces all 42 charges; dsp43-45 at 65-67 bytes
# stay refused, by the lane simulation at 65-66 and TERNARY_CROSSBAR_STAGE_BYTES
# at 67. Evidence: spec
# docs/superpowers/specs/2026-09-29-overlay-and-layout-pragmas-design.md
# Sec 5.2. THE RETIREMENT HOLDS ONLY WHILE THOSE PINS ARE EMITTED -- a
# generator change that drops them re-exposes M150_k7_s11.
# MAX_CODEWORD_LENGTH is a CONSEQUENCE of TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE, not an
# independent limit: a single-stage ternary table can occupy at most the 64-byte-per-stage
# crossbar budget above, so a codeword wider than 64 bytes (512 bits) could never fit in one
# stage's crossbar regardless of TCAM row/block capacity. That 64-byte figure is a measured
# hardware constant -- validated by a sweep of key widths from 8 to 512 bits with two exact
# saturations observed (including one table at exactly 64 bytes); see
# reviews/p4_tofino_reference.md §4.3. (No repo history was found recording an earlier,
# independent provenance for the 512-bit figure beyond this derivation.)
MAX_CODEWORD_LENGTH = TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE * 8

# §5's constants table (device.h:186); hard, per §7's real tofino2h compile
# failure -- a 9-stage program is rejected outright by the backend assembler.
TOFINO_PIPELINE_STAGES = 12

# p4c's compile-time sizing rule for a range table (§4.2, Appendix B "Mechanism
# E"): one entry in every RANGE_WORST_CASE_ENTRY_FRACTION is assumed to need
# the worst-case row count for the key's nibble geometry, capped at
# RANGE_WORST_CASE_ROWS_CAP; the rest are priced at one row. See
# compiler_range_rows. Provenance: bf-p4c mau/resource_estimate.cpp:1628-1693
# / .h:213-214, RangeEntries::preorder/postorder (model_audit_2026-09-27.md
# Appendix C).
RANGE_WORST_CASE_ENTRY_FRACTION = 4
RANGE_WORST_CASE_ROWS_CAP = 8


MAX_RANGE_KEY_BITS = 19   # §4.2: a 20-bit range key does not compile at all

# The non-codeword key bits every classification-table row carries alongside
# the codeword itself. Factored out of the inline `codeword_length + 4` this
# replaces so the band arithmetic lives in exactly one place. Physical
# origin: the mandatory 2-bit --version-- field, which p4c
# always parks in a whole midbyte NIBBLE, never fewer bits
# (model_audit_2026-09-27.md Appendix A.2; tofino/input_xbar.cpp). 4 is that
# nibble, not a fitted or undocumented number. `codeword_bits_to_blocks`
# carries this constant only for the empty-key floor -- everywhere else
# `crossbar_capacity`'s own `floor((g-1)/2)` term already withholds one
# half-byte for the version field, so this constant is not re-added there.
CODEWORD_KEY_OVERHEAD_BITS = 4

# A Tofino stage has four stateful ("meter") ALUs, and every RegisterAction
# this generator emits occupies one for a whole stage. Read straight off the
# compiler's own arithmetic rather than fitted: mau.resources.log's percentage
# table reports a Meter ALU count of 4 as 100.00% (joint_high_sd7 stages 3-6,
# among others). Swept over 2/3/4/5/6/8 against the 18 committed calibration
# placements, only 4 reproduces the compiler's last-register stage on every
# row -- its neighbours manage 13, 11, 10 and 8 of 18. Provenance:
# mau/memories.h:67, mau/resource_estimate.h:32 (model_audit_2026-09-27.md
# Appendix C).
METER_ALUS_PER_STAGE = 4


# The action data bus: what a stage's match tables hand their actions. 128 bytes
# (ADB_BYTES), fed from 16-byte action RAM words (BYTES_PER_RAM); p4c's
# mau/tofino/action_data_bus.h. Where p4c puts each table's words on the bus
# is its allocator's choice, so the model counts no positions: each range table
# takes whole RAM words and the tables sharing a stage take at most
# ACTION_DATA_BUS_WORDS of them (packing.crossbar_stages_needed). Evidence:
# spec 2026-11-11-packed-codes-stage-pin Sec 6 and the model4 probes
# bus_packed.py / bus_bytes.py.
ACTION_DATA_BUS_BYTES = 128
ACTION_RAM_WORD_BYTES = 16

# Ingress PHV containers per size (bits): 4 / 6 / 4 groups of 16 containers of 8 / 16 / 32
# bits, one group of each size egress-only (phv/phv_spec.cpp:383-429), so 48 / 80 / 48.
# Spec docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 7 (the 32-bit
# entry is new there; the per-tree pilot never came near it).
PHV_INGRESS_CONTAINERS = {8: 48, 16: 80, 32: 48}
# The program's own fields that take a whole container of that size, in every compile
# (spec Sec 3.3, MEASURED on the 68 packed compiles of model4: 32-bit used 3 or 5 by the
# program, never more; 16-bit 6 to 22 = k + 5 at most; 8-bit 6 to 15). The 16-bit lists
# stand for the 5 fixed fields; the packed layout adds one more per selected feature
# (the range table's value field) and one 8-bit class field per tree (packed.budgets).
# $hash_field_argument1 is p4c's name for the 16-bit operand of the flow hash.
PHV_TEMPLATE_FIELDS = {
    8: ('ig_intr_md_for_dprsr.mirror_type', 'bypass_egress', 'hdr.ipv4.protocol',
        'header-valid bits'),
    16: ('ig_intr_md.ingress_port', 'ig_intr_md_for_tm.ucast_egress_port',
         'hdr.tcp.src_port', 'hdr.tcp.dst_port', '$hash_field_argument1'),
    32: ('hdr.ipv4.src_addr', 'hdr.ipv4.dst_addr', 'hash operand', 'ig_md.flow_hash',
         'a temporary'),
}
# Words the action data bus carries: 128 bytes of 16-byte action RAM words. A packed code
# field's range-table action may write at most this many (spec Sec 6 rule 3).
ACTION_DATA_BUS_WORDS = ACTION_DATA_BUS_BYTES // ACTION_RAM_WORD_BYTES
# p4c asserts on a ternary key slice wider than 64 bits (lib/bitvec.h getrange, `sz <= 64`):
# a wider code is keyed as several slices of at most this many bits (moved here from
# p4gen/build_p4_script.py).
MAX_KEY_SLICE_BITS = 64
