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
# is handled as a placement refusal instead: TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE
# below.
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
# Two tighter limits for a stage holding two or more DIFFERENT keys (spec "F5",
# the "crowded stage"). A later key must take whatever crossbar groups the first
# left; once the stage is nearly full, p4c routes its bytes through midbyte
# nibbles at extra TCAM blocks. packing.crossbar_stages_needed applies both:
#
#   * above ..._FREE_BYTES (58) and up to ..._BYTES (62): every table of the
#     NON-first key pays +1 block, whatever the key's own slack. This is the
#     model's ONLY sharing charge since the per-key saturation margin was
#     retired on the same evidence (packing.charged());
#   * above ..._BYTES (62): the co-location is refused outright, because the
#     measured extra reaches +2 there and a +1 margin cannot cover it.
#
# Evidence (2026-09-25), per combined distinct-key bytes in one stage:
#   * <= 58: six probe shapes behind a solid spacer never pay
#     (scripts/tcam_mixed_key_cap_sweep.py, results/tcam_mixed_key_cap_onset.csv);
#   * 59-62: (54, 56) -- independent_low_sd5's real app key, not saturated --
#     pays +1; (46, 46, 48, 64) pays +1 at 61; the real design M150_k5_s12's
#     11-byte ddos trees pay +1 at 61 (results/tcam_margin_screen_compiled.csv);
#   * 63-64: (84, 84), (54, 56) and (27, 52) pay +2 (tcam_discount_scan,
#     tcam_mixed_key_cap_sweep).
# Scored on every source together (19 fitted designs, 8 held-out, 16 real
# campaign designs compiled for this, 405 per-table observations), this is the
# only rule tried with 0 under-predictions on all of them, and dropping the
# older saturation margin in its favour made every gate MORE exact. Its cost,
# the safe direction: some real stages at 59-62 bytes pay nothing (three campaign
# designs at 59, independent_low_sd12 at 60), so a few designs at the 12-stage
# ceiling are priced one stage deeper than p4c places them. Which key pays is
# the allocator's greedy choice; byte totals cannot separate the two cases.
TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE = 58
TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62
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
# replaces so the band arithmetic lives in exactly one place -- src/training/
# align_budget.py gates C1's accuracy spending on it and must not re-declare
# it. Physical origin: the mandatory 2-bit --version-- field, which p4c
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
