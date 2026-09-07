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
# ...and those 44 bits are 5 PRIVATE bytes plus one nibble of a MIDBYTE the group
# shares with its pair partner (5 x 8 + 4 = 44). The ternary input crossbar is
# 12 groups x 5 private bytes + 6 midbytes = 66 bytes total, so group 2i and
# group 2i+1 are fed from one 11-byte span laid out private-x5, midbyte,
# private-x5. The split matters because the mandatory 2-bit --version-- field
# may live ONLY in a midbyte nibble: a key that consumes every midbyte its
# groups reach costs an extra TCAM block to hold two bits. See
# tables.version_block_penalty and reviews/github_issue_tcam_version_bit_packing.md
# Sec 1.2-1.3.
CROSSBAR_PRIVATE_BYTES_PER_GROUP = 5
TERNARY_MATCHING_ENTRIES_PER_BLOCK = 512
TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8    # hard cap, binds for narrow keys (<=64 bits)
TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE = 64    # byte budget, binds for wider keys
# MAX_CODEWORD_LENGTH is a CONSEQUENCE of TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE, not an
# independent limit: a single-stage ternary table can occupy at most the 64-byte-per-stage
# crossbar budget above, so a codeword wider than 64 bytes (512 bits) could never fit in one
# stage's crossbar regardless of TCAM row/block capacity. That 64-byte figure is a measured
# hardware constant -- validated by a sweep of key widths from 8 to 512 bits with two exact
# saturations observed (including one table at exactly 64 bytes); see
# reviews/p4_tofino_reference.md §2.3. (No repo history was found recording an earlier,
# independent provenance for the 512-bit figure beyond this derivation.)
MAX_CODEWORD_LENGTH = TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE * 8

TOFINO_PIPELINE_STAGES = 12   # §3; hard, per Appendix B's tofino2h failure

# p4c's compile-time sizing rule for a range table (§2.2, Appendix B "Mechanism
# E"): one entry in every RANGE_WORST_CASE_ENTRY_FRACTION is assumed to need
# the worst-case row count for the key's nibble geometry, capped at
# RANGE_WORST_CASE_ROWS_CAP; the rest are priced at one row. See
# compiler_range_rows.
RANGE_WORST_CASE_ENTRY_FRACTION = 4
RANGE_WORST_CASE_ROWS_CAP = 8


MAX_RANGE_KEY_BITS = 19   # §2.2: a 20-bit range key does not compile at all

# The non-codeword key bits every classification-table row carries alongside
# the codeword itself. Factored out of the inline `codeword_length + 4` this
# replaces so the band arithmetic lives in exactly one place -- src/training/
# align_budget.py gates C1's accuracy spending on it and must not re-declare
# it. Its physical origin is not documented in this repo; the value is
# pre-existing behaviour and is NOT changed here.
CODEWORD_KEY_OVERHEAD_BITS = 4

# A Tofino stage has four stateful ("meter") ALUs, and every RegisterAction
# this generator emits occupies one for a whole stage. Read straight off the
# compiler's own arithmetic rather than fitted: mau.resources.log's percentage
# table reports a Meter ALU count of 4 as 100.00% (joint_high_sd7 stages 3-6,
# among others). Swept over 2/3/4/5/6/8 against the 18 committed calibration
# placements, only 4 reproduces the compiler's last-register stage on every
# row -- its neighbours manage 13, 11, 10 and 8 of 18.
METER_ALUS_PER_STAGE = 4
