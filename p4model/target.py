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
# Used only by tables.codeword_bits_to_blocks, the empty-key floor (a key with no
# bytes still takes one block for the version field); nothing else prices by it.
# ...and those 44 bits are 5 PRIVATE bytes plus one nibble of a MIDBYTE (5 x 8 + 4
# = 44). The ternary input crossbar is 12 groups x 5 private bytes + 6 midbytes =
# 66 bytes total. The split matters because the mandatory 2-bit --version-- field
# may live ONLY in a midbyte nibble: a key that leaves no nibble free costs an
# extra TCAM block to hold two bits. The ported crossbar (xbar.py,
# xbar.blocks_per_row) prices this; reviews/github_issue_tcam_version_bit_packing.md
# Sec 1.2-1.3. (The byte-count "ladder" that restated it, tables.crossbar_capacity /
# codeword_to_blocks, was deleted 2026-11-15, Task 32.)
#
# A note this comment used to carry and that is now RETRACTED (2026-09-21): that
# a midbyte is owned exclusively by its "pair partner" groups 2i / 2i+1, laid out
# private-x5 | midbyte | private-x5. p4c's own assembly says otherwise -- a block
# may pair with ANY of a stage's 6 midbytes, and a key's groups need not even be
# consecutive (measured runs {0,1,3,4}, {0,3}). reviews/p4_tofino_reference.md
# Sec 4.1.1 and Appendix B "Mechanism G".
CROSSBAR_PRIVATE_BYTES_PER_GROUP = 5
# The stage's crossbar group count. No group-BUDGET term exists in this model:
# the near-cap effect a budget would price (spec F5) is handled by the ternary
# crossbar port's stage simulation (xbar.py, packing._crossbar_stage, since
# 2026-11-13; the lane simulation before), which allocates a stage's groups as
# p4c does; the placement refusal that once sat beside it was retired
# 2026-09-29 (see below). The midbyte count (6 per stage) lives in
# xbar.BYTE_GROUPS.
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
TERNARY_MATCHING_ENTRIES_PER_BLOCK = 512
# Provenance: mau/tofino/memories.h:54, TERNARY_TABLES_MAX (model_audit_2026-
# 09-27.md Appendix C).
TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8    # hard cap, binds for narrow keys (<=64 bits)
# The key-size rule on a LONE ternary table key is p4c's own two checks (decision 6a, Task 32),
# read by every site that refuses a key -- training, the replay's check_trees, the packer and
# threshold alignment (tables.check_key_bits and the crossbar port's fit):
#   1. LIVE key bits <= MAX_TERNARY_KEY_BITS. p4c sums bits.size() of every key slice of the
#      table and errors above 524 ("uses 528b as ternary match key. Maximum number of bits
#      allowed is 524b"): open-p4studio bf-p4c phv/table_phv_constraints.cpp:88. Bits, not
#      bytes: a 66-byte key of 524 live bits passes it. 524 = the stage's 66 crossbar bytes
#      less the 4-bit version nibble.
#   2. The ported crossbar (xbar.py) places the key in one stage's 12 groups AND one 512-row
#      word takes at most TCAM_ROWS_PER_STAGE blocks (a word chains its blocks down one
#      column; tables.word_blocks). It is this second half, not a byte count, that refuses a
#      66-byte key of 524 bits with no one-nibble byte: 66 whole bytes = 12 groups x 5 + 6
#      whole midbytes, so the version bits need a 13th block per word.
# Evidence (Task 26 B, p4c 9.13.4, model8/impl/probe_6a_precise.md, 5 compiles): 65 x 8 + one
# 4-bit (66 bytes, 524 bits) compiles at 12 blocks; 64 x 8 + two 6-bit (524 bits) is refused at
# placement ("could not fit in stage 0 with 512 entries"; port: 13 blocks per word); 66 x 8
# (528 bits) is refused by the bit check; 16 x 32 + one 12-bit compiles at 12; 15 x 32 + two
# 22-bit is refused (port: no fit in 12 groups). The rule predicts all 5. It replaced
# MAX_KEY_BYTES = 65 (Task 17: 64 and 65 bytes compile, 66 x 8 refused), which falsely refused
# the two compiling 66-byte keys.
MAX_TERNARY_KEY_BITS = 524
# What one stage's crossbar holds across ALL its keys, as opposed to the limit
# above, which is for one table's key. 12 groups x 5 private bytes + 6 midbytes
# (p4c's input_xbar.cpp:1319 says 66), and real stages of 65 and 66 distinct bytes
# were built. Replay of 691 designs: 5 stage depths corrected, 0 regressions.
# Bytes alone do not admit a stage at 65-66: the ternary crossbar port
# (packing._crossbar_stage, xbar.py, since 2026-11-13) still has to find every
# key an allocation, and refuses the stage when it cannot. Since then it is read
# by the range pool's packer (crossbar_stages_needed's fits), packing._stage_shards'
# guard and lanes' shortcut for a key of more bytes than the crossbar has slots.
TERNARY_CROSSBAR_STAGE_BYTES = 66
# RETIRED 2026-09-29: TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62, the
# "greedy-give-up safety net" that refused a stage holding two different keys
# past 62 combined crossbar bytes. Its one real-design case, M150_k7_s11
# (22 + 41 = 63 bytes, p4c moved the app key a stage), was a PHV LAYOUT
# artifact: with the generator's @pa_container_size pins on every tree-key
# code_* field (build_p4_script.code_field_container_sizes), p4c places both
# keys in one stage, as the model's stage simulation does. History: every probe
# point at 56-64 bytes (tcam_mixed_key_cap_sweep, tcam_discount_scan dsp01-42)
# stayed in one stage, and the lane simulation of the time reproduced all 42
# charges and refused dsp43-45 (65-67 bytes). Since 2026-11-13 that stage
# simulation is the ternary crossbar port (packing._crossbar_stage, xbar.py),
# which relies on the same pins. Evidence: spec
# docs/superpowers/specs/2026-09-29-overlay-and-layout-pragmas-design.md
# Sec 5.2. THE RETIREMENT HOLDS ONLY WHILE THOSE PINS ARE EMITTED -- a
# generator change that drops them re-exposes M150_k7_s11.

# Tofino-1 has 12 ingress MAU stages (device.h:186, Tofino-1's device description); a
# program that needs more is rejected outright by the backend assembler ("supports up to
# 12 stages, using 13", p4_compile.read_compilation_succeeded).
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


# Usable rows of a TCAM block when entries are installed by the control plane: 512 minus
# up to 6 stranded by insertion-order fragmentation (multi-row range entries need
# contiguous free rows, pipe_mgr_tcam_find_next_free; CLAUDE.md "Real usable capacity is
# 506-512 rows/block"). tables.range_deployment_overflow checks real range entries against
# the blocks p4c committed at THIS capacity (decision 6d). Measured to cost nothing at the
# scale seen so far (largest range table 375 rows), so the margin is a guard, not a fit.
TCAM_USABLE_ROWS_PER_BLOCK = 506

# A range key is matched on both nibbles of every byte of its value field, so each byte
# takes two crossbar byte slots, RANGE_LO and RANGE_HI (xbar.range_key); the range pool
# counts a 16-bit key as 4 slots (decision 6f).
RANGE_KEY_SLOTS_PER_BYTE = 2

MAX_RANGE_KEY_BITS = 19   # §4.2: a 20-bit range key does not compile at all

# The non-codeword key bits every classification-table row carries alongside
# the codeword itself. Factored out of the inline `codeword_length + 4` this
# replaces so the band arithmetic lives in exactly one place. Physical
# origin: the mandatory 2-bit --version-- field, which p4c
# always parks in a whole midbyte NIBBLE, never fewer bits
# (model_audit_2026-09-27.md Appendix A.2; tofino/input_xbar.cpp). 4 is that
# nibble, not a fitted or undocumented number. `codeword_bits_to_blocks`
# carries this constant only for the empty-key floor -- everywhere else the
# crossbar port (xbar.blocks_per_row's `+ 1`) already reserves the version
# nibble, so this constant is not re-added there.
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
# The containers the program's own fields take, per size: everything p4model.phv.count
# does not count itself (it adds one 16-bit container per _val, per IAT scratch field and
# per call-site constant, one 8-bit per class_tree field, the two clocks' containers when
# the program computes them -- phv.FINE_CLOCK_CONTAINERS, phv.COARSE_CLOCK_CONTAINERS --
# and each code field's pinned containers). Only the tuple LENGTHS are used.
# Decision 6c, re-measured in Task 39 (2026-11-15) on p4c's allocation of the 160 Task 17b
# base designs compiled with the FINAL generator (596b45d: G3 pins, A1/A2/B1, the flow_hash
# pin, the key rule, tree cap 7, V1o / C1f, P3h): results/encoding_evidence_2026_11/model8/
# impl/remeasure.md "Task 39: final generator", scripts impl/scripts/t18_phv_budgets.py and
# t39_phv_fields.py. Re-run them and reset the lengths whenever a generator change moves PHV.
# Each length is the MEDIAN of p4c's containers of that size minus the model's count of the
# rest (the clocks included), over the 86 designs with no class field for the 8-bit row --
# p4c overlaid class fields into containers dead by then, which the 8-bit count does not
# net, so over all 160 the 8-bit median (2) would only be the overlay -- and over all 160 for
# the 16- and 32-bit rows. (Since 2026-11-17 every class_tree_* field carries @pa_no_overlay
# and no code field does; the lengths were RE-CHECKED on the 160 variant-q compiles and kept:
# model minus p4c summed over sizes is never negative (0..16, exact on 20 of 160), and any
# one length lowered by 1 makes the sum under-price those 20. The model now over-counts the
# 16-bit row by a median of 4, the overlay of code fields it does not net;
# results/encoding_evidence_2026_11/model8/fairness/step8_phv_template.md.) As measured: 8-bit 5 (no class field: 5 on 51 of 86, 2..4 on the rest),
# 16-bit 4 (160: 4 on 103, 2..5), 32-bit 3 (160: 3 on 113, 4..6 on 47). Model minus p4c
# after, k<=8 | k>=9: 8-bit 0..+8 (exact 36/80, under 0) | 0..+9 (15/80, under 0); 16-bit
# -1..+1 (54, under 21) | -1..+2 (49, under 3); 32-bit -2..0 (77, under 3) | -3..0 (36,
# under 44); all sizes summed 0..+8 (35, under 0) | 0..+8 (31, under 0). The 32-bit unders:
# p4c moves qid and the shared 8-bit byte below into W containers (qid 32-bit alone on 24
# designs, the byte on 24), all in disjoint / per-task k>=9 designs with W to spare.
# Where the fields really sit (p4c, 160 designs):
#   8-bit: mirror_type alone 160; ipv4.protocol alone 160; ig_tm_md.qid alone 127 (else
#     32-bit, above); bypass_egress, ingress_cos, fwd and the three header valid bits share
#     two 8-bit containers (136; else W). ig_md.flow_hash left this row (16-bit row).
#   16-bit: ucast_egress_port alone 160, ipv4.total_len alone 160, a p4c temporary ($tmp<n>,
#     alone 133); ig_md.flow_hash in one 16-bit container alone on 150 (its @pa_solitary pin,
#     Task 32; 2 shared 8-bit on 10); ingress_port shares a _val's container 160/160.
#   32-bit: ipv4.src_addr and ipv4.dst_addr alone 160; tcp.src_port and tcp.dst_port share
#     one 32-bit container 160/160.
#   The fine clock (now_pseudo_us + global_tstamp[41:10]) is no longer here: it takes
#     2 B + 4 H exactly where the program computes it (105 / 105) and nothing elsewhere,
#     counted by p4model.phv (FINE_CLOCK_CONTAINERS, review minor 4 of Task 38). The coarse
#     clock takes 2 W on 114 / 114 designs that compute it (COARSE_CLOCK_CONTAINERS).
#   IAT scratch (<chain>_current_iat): one 16-bit container each (G3 pin), as counted.
# History: 4 / 5 / 5 until 2026-11-15 (model4's 68 packed compiles); 6 / 5 / 5 from Task 18;
# 3 / 6 / 5 from Task 23 (G3 pins); 5 / 4 / 3 from Task 31 (hash rewrite A1, one-tree tasks
# A2; the fine clock's 8- and 16-bit pieces inside the template); 5 / 4 / 3 from Task 39 with
# different fields (flow_hash 8 -> 16 bits, the fine clock moved out into its own count).
PHV_TEMPLATE_FIELDS = {
    8: ('ig_intr_md_for_dprsr.mirror_type', 'hdr.ipv4.protocol', 'ig_tm_md.qid',
        'bypass_egress / ingress_cos / fwd / header valid bits, first container',
        'bypass_egress / ingress_cos / fwd / header valid bits, second container'),
    16: ('ig_intr_md_for_tm.ucast_egress_port', 'hdr.ipv4.total_len', 'a temporary ($tmp<n>)',
         'ig_md.flow_hash'),
    32: ('hdr.ipv4.src_addr', 'hdr.ipv4.dst_addr', 'hdr.tcp.src_port + hdr.tcp.dst_port'),
}
# Words the action data bus carries: 128 bytes of 16-byte action RAM words. A packed code
# field's range-table action may write at most this many (spec Sec 6 rule 3).
ACTION_DATA_BUS_WORDS = ACTION_DATA_BUS_BYTES // ACTION_RAM_WORD_BYTES
# p4c asserts on a ternary key slice wider than 64 bits (lib/bitvec.h getrange, `sz <= 64`):
# a wider code is keyed as several slices of at most this many bits (moved here from
# p4gen/build_p4_script.py).
MAX_KEY_SLICE_BITS = 64
