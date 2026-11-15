"""The per-table TCAM block price: how many blocks does one classification
table's crossbar key cost, version field included?

Every number asserted here was READ OFF a real p4c compile, not derived:

  * `results/compiler_calibration_v6/compiles/*/pipe/logs/resources.json`
    records both the TCAM units each table got (`tcams.tcams`) and the exact
    crossbar bytes its key was given (`xbar_bytes`, `byte_type: ternary`).
  * `results/tcam_version_sweep.csv` supplies 12 fresh out-of-sample compiles
    chosen to sit on the old byte-count rule's decision boundary.

2026-11-15 (Task 32, G2): these facts used to pin the byte-count "ladder"
(`tables.codeword_to_blocks`, `crossbar_capacity`, `tail_is_isolatable`),
which threshold alignment priced with. Alignment now prices with the ternary
crossbar port like everything else, the ladder is deleted, and every p4c
number below is asserted against the production price `lanes.table_blocks`
(xbar.alone_blocks of the key's pinned layout) instead. The ladder-only tests
(the isolatability table, the ladder's fixed points as a formula, the bit-band
lower bound on the ladder) went with it. History of the file: renamed from
`test_version_block.py` 2026-09-27; offset-based API retired 2026-09-21.
"""
import pytest

from p4model.lanes import table_blocks


# --------------------------------------------------------------------------
# results/tcam_version_sweep.csv -- fresh p4c compiles on geometries that sit
# on a block boundary (scripts/tcam_version_sweep.py). Every number below is
# p4c's own committed TCAM count.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field_bits,real_blocks,why", [
    ((80,),  2, "10 bytes, one spare slot"),
    ((88,),  3, "11 bytes, saturated -> the version nibble needs a block"),
    ((168,), 4, "21 bytes, one spare slot"),
    ((176,), 5, "22 bytes, saturated"),
    ((256,), 6, "32 bytes, one spare slot"),
    ((264,), 7, "33 bytes, saturated"),
    ((44, 40, 88), 4, "the 44-bit field's 4-bit tail is a free nibble"),
    ((128,), 3, "ends with a spare byte slot"),
    ((124,), 3, "same, ragged"),
])
def test_the_port_matches_the_fresh_compile_sweep(field_bits, real_blocks, why):
    assert table_blocks(field_bits) == real_blocks, why


def test_the_84_84_key_on_p4cs_layout_and_on_the_generators_pins():
    # (84, 84): every p4c compile of this key pinned NO containers. p4c charged
    # it 5 alone (results/tcam_version_sweep.csv d_unreachable_B22) and 5 at 43
    # of the 45 spacer widths of results/tcam_discount_scan.csv (7 at dsp41 /
    # dsp42, where it shares a stage), laying each 84-bit field out W24 | W32 |
    # W28 (prog.bfa; tests/test_packing.py). On that layout the port gives 5,
    # p4c's number. The 4 below is the port on the layout the GENERATOR pins
    # (lanes.layout, fill-low W32 | W32 | W20), which has never been compiled:
    # an untested prediction on the under-price side (Task 32 review Important
    # 1; a pinned compile is queued for the final recompile).
    from p4model import xbar
    p4c_layout = tuple((i, (("W", 24), ("W", 32), ("W", 28))) for i in range(2))
    assert xbar.alone_blocks(p4c_layout) == 5
    assert table_blocks((84, 84)) == 4


def test_sd5_ddos_key_needs_a_third_block():
    # independent_low_sd5's ddos trees key 27 + 52 bits = 11 crossbar bytes;
    # resources.json stage 6 shows them at 3 TCAM blocks (p4c's midbyte choice
    # leaves no nibble for the version bits).
    assert table_blocks((27, 52)) == 3


def test_sd5_app_key():
    # The same stage's app tree keys 54 + 56 bits = 14 bytes: 3 blocks.
    assert table_blocks((54, 56)) == 3


def test_a_ragged_49_byte_key_costs_9_blocks_alone():
    # ragged_ax1_bx5 (scripts/tcam_stretch_sweep.py), alone in its stage:
    # 23 + 26 bytes = 49 crossbar bytes, 9 blocks. The SAME key measured 10
    # when a DIFFERENT key shared its stage -- a stage-placement effect, which
    # packing's stage simulation (xbar.py) prices.
    assert table_blocks((179, 204)) == 9


def test_a_solid_49_byte_key_costs_9_blocks():
    # bit<392>: 49 fully used bytes, measured 9 blocks (a49x1_b12x5 / _b12x4).
    assert table_blocks((392,)) == 9


def test_a_key_with_a_spare_crossbar_byte():
    # independent_low_sd6's app trees: one bit<146> field = 19 bytes, 4 blocks.
    assert table_blocks((146,)) == 4


def test_independent_low_sd7_ddos_key():
    # 46 + 46 + 48 + 64 bits = 26 bytes, 5 blocks.
    assert table_blocks((46, 46, 48, 64)) == 5


@pytest.mark.parametrize("field_bits,expected", [
    ([321], 8),                        # joint_low_sd12, 41 bytes
    ([93, 156], 6),                    # joint_low_sd7,  32 bytes
    ([46], 2),                         # independent_low_sd10 ddos, 6 bytes
    ([64, 69], 4),                     # independent_low_sd8 app,  17 bytes
    ([26, 30, 31, 32, 32], 4),         # independent_low_sd12 ddos, 20 bytes
    ([47, 56, 66, 69, 72], 8),         # independent_low_sd12 app,  40 bytes
])
def test_real_calibration_keys(field_bits, expected):
    assert table_blocks(field_bits) == expected


def test_an_empty_key_costs_one_block():
    # A version field still needs a physical block even when the key itself
    # claims none -- reachable whenever every tree in a forest is a single leaf.
    assert table_blocks(()) == 1


def test_the_entry_to_block_arithmetic_has_a_name():
    """Audit §8.1's multiplier. A block is MEMORY, so it is charged once per
    TREE -- unlike the crossbar's byte slots, which a stage charges once
    however many tables read them. Alignment needs this quantity by name to
    weigh a range step (1 block) against a ternary step
    (align_budget.tree_keys). The ceil is what stops "one block-row per tree"
    from being baked in."""
    from p4model.tables import (entries_across_trees_to_blocks,
                                    tree_entries_to_blocks)

    assert tree_entries_to_blocks(1) == 1
    assert tree_entries_to_blocks(512) == 1
    assert tree_entries_to_blocks(513) == 2
    assert entries_across_trees_to_blocks([100, 200, 300]) == 3
    assert entries_across_trees_to_blocks([600, 100]) == 3
    assert entries_across_trees_to_blocks([]) == 0


def test_a_nibble_clean_field_saves_a_block():
    # Eleven 5-bit fields: 11 crossbar bytes, every byte a whole one -- 3
    # blocks. Swap one 5-bit field for a 4-bit one: still 11 bytes, but its
    # byte is a single nibble, which a half midbyte holds beside the version
    # bits -- 2 blocks.
    assert table_blocks((5,) * 11) == 3
    assert table_blocks((5,) * 10 + (4,)) == 2


@pytest.mark.parametrize("byte_width,blocks", [
    (10, 2), (16, 3), (21, 4), (27, 5), (32, 6), (38, 7), (43, 8), (49, 9), (54, 10), (60, 11)])
def test_solid_keys_at_each_block_boundary(byte_width, blocks):
    # One solid field of byte_width bytes is the largest such key `blocks`
    # blocks hold; one byte more costs a block more.
    assert table_blocks((byte_width * 8,)) == blocks
    assert table_blocks((byte_width * 8 + 8,)) == blocks + 1
