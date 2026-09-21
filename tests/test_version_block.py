"""The per-table TCAM block price: how many blocks does one classification
table's crossbar key cost, version field included?

Every number asserted here was READ OFF a real p4c compile, not derived:

  * `results/compiler_calibration_v6/compiles/*/pipe/logs/resources.json`
    records both the TCAM units each table got (`tcams.tcams`) and the exact
    crossbar bytes its key was given (`xbar_bytes`, `byte_type: ternary`).
  * `results/tcam_version_sweep.csv` supplies 12 fresh out-of-sample compiles
    chosen to sit on the pricing rule's decision boundary.

2026-09-21 (TCAM block model rewrite Task 2): this file used to be built
entirely around an offset-based API (`crossbar_groups_needed`,
`version_block_penalty`, `codeword_to_blocks(bits, start_group)`) modelling a
key's price as a function of WHERE its crossbar run started. Reading p4c's
own assembly showed that premise false -- a block may pair with any of a
stage's midbytes, not a fixed partner, and groups need not be consecutive
(`docs/superpowers/specs/2026-09-20-tcam-block-model-rewrite-design.md` Sec 2).
The corrected, simpler model in `src/p4model/tables.py` has no `start_group`
parameter: a key's own price no longer depends on where it starts. Every test
below that pinned a per-table fact (a key's price alone) is kept, rewritten
against the new no-offset signature. Every test that specifically pinned the
OFFSET/SHARING effect itself (a key costing more because a DIFFERENT key
shares its stage) is removed from here with a pointer to where that effect
now lives -- `src/p4model/packing.py`'s stage-sharing margin, spec Sec 13.2 --
since there is no offset parameter left on this module's functions to test it
with. See this task's report for the full accounting of what moved versus
what was retired outright.
"""
import math

import numpy as np
import pytest

from src.p4model.tables import (codeword_bits_to_blocks, codeword_bytes_to_blocks,
                                codeword_to_blocks, codeword_to_blocks_headline,
                                crossbar_capacity, tail_is_isolatable)


# --------------------------------------------------------------------------
# results/tcam_version_sweep.csv -- 12 FRESH p4c compiles run specifically to
# test this rule out of sample, on geometries chosen to sit on its decision
# boundary rather than comfortably inside it (scripts/tcam_version_sweep.py).
# Every number below is p4c's own committed TCAM count. These already call
# codeword_to_blocks(field_bits) with no offset, so they need no change
# beyond confirming they still pass against the rewritten implementation.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field_bits,real_blocks,why", [
    ((80,),  2, "g=2, 10 bytes, one spare slot"),
    ((88,),  3, "g=2, 11 bytes, saturated -> pays"),
    ((168,), 4, "g=4, 21 bytes, one spare slot"),
    ((176,), 5, "g=4, 22 bytes, saturated -> pays"),
    ((256,), 6, "g=6, 32 bytes, one spare slot"),
    ((264,), 7, "g=6, 33 bytes, saturated -> pays"),
    ((84, 84), 5, "isolation credit fails: neither 84-bit field is isolatable"),
    ((44, 40, 88), 4, "isolation credit holds: the 44-bit field is isolatable"),
    ((128,), 3, "g=3 ends with a spare byte slot"),
    ((124,), 3, "same, ragged"),
])
def test_block_factor_matches_the_fresh_compile_sweep(field_bits, real_blocks, why):
    assert codeword_to_blocks(field_bits) == real_blocks, why


# --------------------------------------------------------------------------
# The one real penalty in the whole calibration set.
# --------------------------------------------------------------------------
def test_sd5_ddos_key_pays_a_version_block_because_it_saturates_its_groups():
    # independent_low_sd5's ddos trees key code_bwd_packet_length_max (27 bits,
    # 4 bytes) + code_ddos_packet_length_mean (52 bits, 7 bytes) = 11 crossbar
    # bytes, which is EXACTLY two groups' 2 x 5.5 bytes. resources.json stage 6
    # shows all three of them at 3 TCAM blocks where the byte width alone buys
    # 2 (codeword_bytes_to_blocks), because neither field's tail is isolatable
    # (27 % 32 = 27 and 52 % 32 = 20 both land in the Sec 2.3 "NOT" cells).
    assert codeword_bytes_to_blocks(11) == 2
    assert codeword_to_blocks((27, 52)) == 3


def test_sd5_app_key_pays_nothing():
    # The same stage's app tree keys 54 + 56 bits = 14 bytes in 3 groups.
    # resources.json shows this table at 3 blocks, no penalty -- p4c placed it
    # at group offset 3 with its midbyte left empty, but that placement detail
    # is no longer part of this table's OWN price (it moved to packing.py's
    # stage-sharing margin, spec Sec 13.2): this key never saturates its
    # groups regardless of where the crossbar starts it.
    assert codeword_to_blocks((54, 56)) == 3


def test_a_ragged_49_byte_key_costs_9_blocks_alone():
    # ragged_ax1_bx5 (scripts/tcam_stretch_sweep.py), the table alone in its
    # stage: 23 + 26 bytes = 49 crossbar bytes, one isolatable tail (the
    # 204-bit field), 9 blocks. The SAME key measured 10 blocks when a
    # DIFFERENT key shared its stage (ragged_ax1_bx4/ragged_ax2_bx2) -- that
    # is a stage-PLACEMENT effect, not a property of this key alone, and it
    # now belongs entirely to src/p4model/packing.py's stage-sharing margin
    # (spec Sec 13.2, "Effect 4"); there is no start_group parameter left on
    # codeword_to_blocks to reproduce it here. See this task's report.
    assert codeword_to_blocks((179, 204)) == 9


def test_a_solid_key_has_no_isolatable_tail_to_lose():
    # The control the stretch sweep measured: one bit<392> field is 49 fully-
    # used bytes with no nibble-clean byte anywhere (392 % 8 == 0), so the
    # isolation credit never applies. Measured 9 blocks (probe points
    # a49x1_b12x5 and a49x1_b12x4 -- both 9, confirming the old "priced the
    # same at every offset" fact was really "this key has nothing an offset
    # could take away", which the new model states directly: no offset term
    # at all).
    assert codeword_to_blocks((392,)) == 9


def test_a_key_with_a_spare_crossbar_byte_never_pays():
    # independent_low_sd6's app trees key one bit<146> field = 19 bytes in 4
    # groups (20 private byte slots), one spare. 4 blocks, measured.
    assert codeword_to_blocks((146,)) == 4


def test_an_odd_group_count_leaves_a_trailing_half_midbyte_free():
    # independent_low_sd7's ddos key: 46+46+48+64 bits = 26 bytes, 5 groups,
    # one spare slot at g=5 (25 private bytes + no isolatable tail). 5 blocks.
    assert codeword_to_blocks((46, 46, 48, 64)) == 5


@pytest.mark.parametrize("field_bits,expected", [
    ([321], 8),                        # joint_low_sd12, 41 bytes
    ([93, 156], 6),                    # joint_low_sd7,  32 bytes
    ([46], 2),                         # independent_low_sd10 ddos, 6 bytes
    ([64, 69], 4),                     # independent_low_sd8 app,  17 bytes
    ([26, 30, 31, 32, 32], 4),         # independent_low_sd12 ddos, 20 bytes
    ([47, 56, 66, 69, 72], 8),         # independent_low_sd12 app,  40 bytes
])
def test_real_calibration_keys_cost_exactly_their_crossbar_width(field_bits,
                                                                 expected):
    assert codeword_to_blocks(field_bits) == expected


def test_an_empty_key_costs_one_block():
    # A version field still needs a physical block even when the key itself
    # claims none -- reachable whenever every tree in a forest is a single
    # leaf (see tests/test_align_budget.py's
    # test_factor_of_an_empty_width_dict_is_the_empty_key_factor, which pins
    # the same fact against src/training/align_budget.py's own call site).
    assert codeword_to_blocks(()) == 1
    assert codeword_to_blocks_headline(()) == 1


def test_the_entry_to_block_arithmetic_has_a_name():
    """Audit §8.1's multiplier. A block is MEMORY, so it is charged once per
    TREE -- unlike the crossbar's byte slots, which a stage charges once
    however many tables read them. Alignment needs this quantity by name to
    weigh a range step (1 block) against a ternary step (the multiplier, 8-80
    blocks in the golden fixture).

    At this project's tree sizes every tree costs exactly one block-row, so the
    multiplier is simply the table count -- true on all 19 golden rows. The
    ceil is what stops that coincidence from being baked in.
    """
    from src.p4model.tables import (entries_across_trees_to_blocks,
                                    tree_entries_to_blocks)

    assert tree_entries_to_blocks(1) == 1
    assert tree_entries_to_blocks(512) == 1
    assert tree_entries_to_blocks(513) == 2
    assert entries_across_trees_to_blocks([100, 200, 300]) == 3
    assert entries_across_trees_to_blocks([600, 100]) == 3
    assert entries_across_trees_to_blocks([]) == 0


# --------------------------------------------------------------------------
# The isolation credit's per-table effect (Sec 2.3), distinguished from the
# retired offset mechanism it used to be entangled with. Both keys below are
# 11 crossbar bytes; they differ only in whether one field's nibble-clean
# tail is isolatable, and that alone changes the block count -- no group
# offset is involved on either side.
# --------------------------------------------------------------------------
def test_an_isolatable_nibble_clean_tail_saves_a_block():
    # Eleven 5-bit fields: 11 crossbar bytes, none nibble-clean (5 % 8 == 5,
    # outside 1..4), so nothing can be isolated -- 3 blocks (this is also one
    # of the fresh-compile sweep points above, (5,) * 11 is not itself a
    # measured point but the arithmetic matches solid_g2_B11's B=11 saturation
    # exactly). Swap one 5-bit field for a 4-bit one: still 11 bytes, but the
    # 4-bit field is nibble-clean AND isolatable (tail 4, index 0), so the
    # ledger's nibble test passes one group earlier -- 2 blocks, not 3.
    assert codeword_to_blocks((5,) * 11) == 3
    assert codeword_to_blocks((5,) * 10 + (4,)) == 2


def test_tail_is_isolatable_matches_the_sec_2_3_table():
    # One worked example per row of the table tail_is_isolatable implements,
    # chosen so index and container count are both unambiguous.
    assert tail_is_isolatable(4) is True          # index 0, single container
    assert tail_is_isolatable(12) is True         # index 1, single container
    assert tail_is_isolatable(20) is True         # index 2, single container (20 <= 32)
    assert tail_is_isolatable(52) is False        # index 2, two containers (52 > 32)
    assert tail_is_isolatable(28) is False        # index 3, either way


# --------------------------------------------------------------------------
# Properties pinned by the 2026-09-20 rewrite design's own validation
# (300 000 random non-empty keys, 0 violations for both properties below).
# Reusing this file's existing hand-rolled randomised-loop convention rather
# than the spec's own scratch-script trial count, to keep this a fast
# regression pin rather than a from-scratch re-validation.
# --------------------------------------------------------------------------
def _random_field_widths(rng):
    return tuple(int(w) for w in
                rng.integers(1, 81, size=int(rng.integers(1, 16))))


@pytest.mark.parametrize("seed", range(3000))
def test_the_bit_width_bound_never_exceeds_the_composed_cost(seed):
    """`codeword_bits_to_blocks` (the bit-width band) is kept only as a
    lower-bound assertion target now -- never a search floor or a max() arm
    inside codeword_to_blocks's general path (spec F6). This is that
    assertion, as a test: the bound must still hold, by inspection of every
    trial rather than by construction, since nothing in the new implementation
    forces it structurally the way the old max() composition did."""
    rng = np.random.default_rng(seed)
    widths = _random_field_widths(rng)
    assert codeword_bits_to_blocks(sum(widths)) <= codeword_to_blocks(widths), widths


@pytest.mark.parametrize("seed", range(3000))
def test_the_isolation_refinement_never_raises_the_headline_price(seed):
    """The Sec 2.3 isolation credit is a REFINEMENT: it can only lower a key's
    price relative to the Sec 13.1 headline (`S = 0`) ladder, never raise it,
    because the credit only ever subtracts from the nibble test's left-hand
    side. codeword_to_blocks_headline is codeword_to_blocks with the credit
    switched off, so this is the direct comparison."""
    rng = np.random.default_rng(seed + 1_000_000)   # disjoint stream from the test above
    widths = _random_field_widths(rng)
    assert codeword_to_blocks(widths) <= codeword_to_blocks_headline(widths), widths


# --------------------------------------------------------------------------
# The ladder's fixed points: B = 10, 16, 21, 27, 32, 38, 43, 49, 54, 60 --
# exactly where crossbar_capacity(g) == B for each g -- map to consecutive
# block counts 2..11. Checked against crossbar_capacity directly rather than
# hardcoding the block numbers a second time.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("byte_width", [10, 16, 21, 27, 32, 38, 43, 49, 54, 60])
def test_the_ladders_fixed_points_match_crossbar_capacity(byte_width):
    blocks = next(g for g in range(1, 20) if crossbar_capacity(g) == byte_width)
    # One solid field of byte_width bytes (bit width a clean multiple of 8,
    # so codeword_fields_to_bytes_from_bits's ceil is exact) exercises the
    # headline ladder with no isolation credit in play.
    assert codeword_to_blocks_headline((byte_width * 8,)) == blocks
    assert codeword_to_blocks((byte_width * 8,)) == blocks
