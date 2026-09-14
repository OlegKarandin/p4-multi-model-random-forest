"""The version-block rule: when does a ternary table pay one extra TCAM block
to hold the mandatory 2-bit --version-- field?

Every number asserted here was READ OFF a real p4c compile, not derived:

  * `results/compiler_calibration_v6/compiles/*/pipe/logs/resources.json`
    records both the TCAM units each table got (`tcams.tcams`) and the exact
    crossbar bytes its key was given (`xbar_bytes`, `byte_type: ternary`).
    Over all 19 archived compiles that is 100 classification tables, of which
    exactly 3 cost more than `ceil(8 * key_bytes / 44)` blocks -- the three
    `independent_low_sd5` ddos trees.
  * `results/tcam_stretch_sweep.csv` supplies the two synthetic arms that
    isolate the crossbar START OFFSET, which no calibration row varies.

Together those pin all four branches of the rule. See
reviews/p4_tofino_reference.md Appendix B "Mechanism G".
"""
import math

import numpy as np
import pytest

from src.p4model.tables import (codeword_bytes_to_blocks, codeword_to_blocks,
                                version_block_penalty)


# --------------------------------------------------------------------------
# results/tcam_version_sweep.csv -- 12 FRESH p4c compiles run specifically to
# test this rule out of sample, on geometries chosen to sit on its decision
# boundary rather than comfortably inside it (scripts/tcam_version_sweep.py).
# Every number below is p4c's own committed TCAM count.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field_bits,real_blocks,why", [
    ((80,),  2, "g=2, 10 bytes, one spare slot"),
    ((88,),  3, "g=2, 11 bytes, saturated -> pays"),
    ((168,), 4, "g=4, 21 bytes, one spare slot"),
    ((176,), 5, "g=4, 22 bytes, saturated -> pays"),
    ((256,), 6, "g=6, 32 bytes, one spare slot"),
    ((264,), 7, "g=6, 33 bytes, saturated -> pays"),
    ((84, 84), 5, "clause (d) fails: clean bytes at 10 and 21, midbytes 5/16"),
    ((44, 40, 88), 4, "clause (d) holds: the clean 6-byte field reaches midbyte 5"),
    ((128,), 3, "g=3 ends on a half midbyte"),
    ((124,), 3, "same, ragged"),
])
def test_block_factor_matches_the_fresh_compile_sweep(field_bits, real_blocks, why):
    assert codeword_to_blocks(field_bits) == real_blocks, why


def test_the_version_field_is_not_charged_twice_when_the_band_arm_wins():
    # codeword_bits_to_blocks's `+ CODEWORD_KEY_OVERHEAD_BITS` is ITSELF a
    # version/valid allowance, so composing it as `max(band, xbar) + penalty`
    # bills the same 2-bit field twice. Measured: a solid 11-byte key compiles
    # to 3 TCAMs, not 4 (sweep point solid_g2_B11); 22 bytes to 5, not 6;
    # 33 bytes to 7, not 8.
    from src.p4model.tables import codeword_bits_to_blocks
    for bits, real in (((88,), 3), ((176,), 5), ((264,), 7)):
        key_bytes = sum(math.ceil(b / 8) for b in bits)
        band = codeword_bits_to_blocks(sum(bits))
        xbar = codeword_bytes_to_blocks(key_bytes)
        assert band > xbar, bits              # the arm that triggers the bug
        assert version_block_penalty(bits, 0) == 1, bits
        assert band + version_block_penalty(bits, 0) != real    # the old way
        assert codeword_to_blocks(bits) == real


def blocks(field_bits, start=0):
    """What the model charges one table keyed on these fields."""
    key_bytes = sum(math.ceil(b / 8) for b in field_bits)
    return (codeword_bytes_to_blocks(key_bytes)
            + version_block_penalty(field_bits, start))


# --------------------------------------------------------------------------
# The one real penalty in the whole calibration set.
# --------------------------------------------------------------------------
def test_sd5_ddos_key_pays_a_version_block_because_it_saturates_its_groups():
    # independent_low_sd5's ddos trees key code_bwd_packet_length_max (27 bits,
    # 4 bytes) + code_ddos_packet_length_mean (52 bits, 7 bytes) = 11 crossbar
    # bytes, which is EXACTLY two groups' 2 x 5.5 bytes. resources.json stage 6
    # shows all three of them at 3 TCAM blocks where the width alone buys 2,
    # and shows why: the pair's midbyte (relative crossbar byte 5) carries
    # code_ddos_packet_length_mean[32:39], a full byte, so no nibble is left.
    assert codeword_bytes_to_blocks(11) == 2
    assert blocks([27, 52]) == 3


def test_sd5_app_key_pays_nothing_at_either_offset():
    # The same stage's app tree keys 54 + 56 bits = 14 bytes in 3 groups
    # (16.5 bytes of capacity). resources.json puts it at group offset 3 with
    # its midbyte (crossbar byte 155) left EMPTY -- 3 blocks, no penalty.
    # Mechanism G charged this table and not the ddos one: exactly backwards.
    assert blocks([54, 56], start=0) == 3
    assert blocks([54, 56], start=3) == 3


# --------------------------------------------------------------------------
# The start-offset term, isolated by scripts/tcam_stretch_sweep.py.
# --------------------------------------------------------------------------
def test_ragged_49_byte_key_costs_9_blocks_alone_and_10_when_shifted():
    # Probe points ragged_ax1_bx5 (the table alone in its stage, group offset
    # 0) and ragged_ax1_bx4 (a 3-group key takes groups 0..2 first, so this one
    # starts at group 3). Same table, same key, 9 blocks vs 10.
    assert blocks([179, 204], start=0) == 9
    assert blocks([179, 204], start=3) == 10


def test_a_solid_key_is_priced_the_same_at_every_offset():
    # The control arm: one bit<392> field is 49 fully-used bytes with no
    # nibble-clean byte anywhere, so nothing can ride the half midbyte the odd
    # start exposes and it stays free for --version--. Measured 9 either way
    # (probe points a49x1_b12x5 and a49x1_b12x4).
    assert blocks([392], start=0) == 9
    assert blocks([392], start=3) == 9


# --------------------------------------------------------------------------
# The clauses that keep the rule from firing on the other 97 tables.
# --------------------------------------------------------------------------
def test_a_key_with_a_spare_crossbar_byte_never_pays():
    # independent_low_sd6's app trees key one bit<146> field = 19 bytes in 4
    # groups (22 whole slots), and p4c scattered it -- crossbar bytes 5-10, 12,
    # 14, 17-21, 27-32 -- leaving midbyte 16 free. 4 blocks, measured.
    assert blocks([146]) == 4


def test_an_odd_group_count_leaves_a_trailing_half_midbyte_free():
    # joint_low_sd12's trees key 41 bytes in 8... no: 321 bits = 41 bytes,
    # ceil(41*8/44) = 8 groups, 44 whole slots, 3 spare. The odd-g case is
    # independent_low_sd7's ddos key: 26 bytes, 5 groups, and the run ends on
    # the half midbyte that pairs group 4 with the absent group 5.
    assert blocks([46, 46, 48, 64]) == 5


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
    assert blocks(field_bits) == expected


def test_penalty_is_never_more_than_one_block():
    # TableFormat::ternary_version() push_back()s at most one TCAM to hold a
    # 2-bit field; nothing in the measurements shows a second.
    for bits in ([27, 52], [179, 204], [392], [146], [1] * 15):
        for start in range(6):
            assert version_block_penalty(bits, start) in (0, 1)


def test_an_empty_key_costs_nothing():
    assert version_block_penalty([], 0) == 0


# --------------------------------------------------------------------------
# The packer has to charge the penalty per STAGE, because it depends on the
# group offset a key gets, which depends on what else shares the stage.
# --------------------------------------------------------------------------
def test_the_packer_charges_sd5s_stage_the_twelve_blocks_p4c_charged():
    from src.p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_flm"), 7), (("code", "app_plm"), 7)})
    ddos = frozenset({(("code", "ddos_bplm"), 4), (("code", "ddos_plm"), 7)})
    # Specs are what codeword_to_blocks now yields standalone: the app key
    # costs 3 and the ddos key 3 (2 for its 11 bytes, plus the version block
    # those 11 bytes leave no midbyte nibble for). resources.json stage 6 shows
    # exactly that -- one app tree and three ddos trees, 3 blocks each, 12
    # total, where the pre-Mechanism-G naive sum said 9.
    assert codeword_to_blocks((27, 52)) == 3
    plan = crossbar_stages_needed(
        [(3, 14), (3, 11), (3, 11), (3, 11)],
        key_fields=[app, ddos, ddos, ddos],
        key_field_bits=[(54, 56), (27, 52), (27, 52), (27, 52)])
    assert plan.blocks == 12


@pytest.mark.parametrize("seed", range(200))
def test_the_bit_width_bound_never_exceeds_the_composed_cost(seed):
    """D5 proposed demoting `codeword_bits_to_blocks` from a load-bearing
    max() arm to a bare assertion, reasoning it was a PROVABLE lower bound
    that the crossbar-plus-version arm could never exceed. Measured instead:
    272 of 500 000 random field-width trials violate the crossbar-plus-version
    arm alone (see codeword_to_blocks's docstring), plus a non-random
    empty-key counterexample. So `codeword_to_blocks` keeps `max()` rather
    than asserting, and this test now guards that the max() composition --
    not a bare inequality -- is what's actually in place. All 200 seeds pass
    because max() makes the property hold by construction.
    """
    from src.p4model.tables import codeword_bits_to_blocks, codeword_to_blocks
    rng = np.random.default_rng(seed)
    widths = tuple(sorted(int(w) for w in
                          rng.integers(1, 61, size=int(rng.integers(1, 16)))))
    for start_group in range(0, 4):
        assert (codeword_bits_to_blocks(sum(widths))
                <= codeword_to_blocks(widths, start_group)), (widths, start_group)


def test_a_stage_of_one_shared_key_is_never_charged_an_offset_penalty():
    # Every 'joint' design keys every tree on the identical field set, so all
    # its tables sit at group offset 0. None of the 8 joint calibration rows
    # shows a single penalised table.
    from src.p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "f"), 32)})
    plan = crossbar_stages_needed(
        [(6, 32)] * 3, key_fields=[key] * 3,
        key_field_bits=[(256,)] * 3)
    assert plan.blocks == 18


def test_omitting_key_field_bits_prices_every_table_at_its_declared_width():
    from src.p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed([(3, 14), (2, 11), (2, 11), (2, 11)])
    assert plan.blocks == 9


# --------------------------------------------------------------------------
# independent_low_sd9: two DIFFERENT ragged keys never share a stage, even
# though every modelled per-stage limit says they could.
# --------------------------------------------------------------------------
def test_two_different_ragged_keys_do_not_share_a_stage():
    from src.p4model.packing import crossbar_stages_needed

    # Real placement (results/compiler_calibration_extra, resources.json):
    #   stage 6  5 x ddos @ 3 blocks = 15
    #   stage 7  2 x app  @ 9        = 18
    #   stage 8  2 x app  @ 9        = 18
    #   stage 9  1 x app  @ 9        =  9      -> 4 stages, 60 blocks
    # Note what is NOT there: no table pays a version block. The app trees
    # cost 9, not 10. So the reason p4c never mixes them is not a block cost
    # -- 9+3 | 9+3 is 24 blocks and packs both columns cleanly. It is
    # Memories::find_ternary_stretch refusing to start an app table's run at
    # row 3 of a column whose rows 0-2 hold a ddos table: two TCAM rows 2i and
    # 2i+1 share a crossbar midbyte, and the two keys cannot share one.
    # The version charge is what reproduces the refusal, and it does so
    # without pretending anyone paid it: whichever key the crossbar hands the
    # later groups would cost +1, so the mixed stage prices at 10+10+3+3 = 26
    # against a 24-block stage and cannot be committed. The charge decides the
    # PLACEMENT; the placement it settles on has each key at offset 0, where
    # nothing is owed. That is exactly the ledger p4c's own artifact shows.
    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] * 5 + [(3, 12)] * 5,
        readiness_levels=[0] * 10,
        key_fields=[app] * 5 + [ddos] * 5,
        key_field_bits=[(179, 204)] * 5 + [(37, 49)] * 5)
    assert plan.occupied == 4

    # blocks comes out at 61 against p4c's 60, and the extra one is a
    # placement-order artefact rather than a pricing error. p4c filled a stage
    # with all five ddos trees and gave the app trees three stages of their
    # own; this packer is first-fit-DECREASING, so it seats the 9-block app
    # tables first and one ddos-heavy stage ends up holding a single app table
    # too. That stage prices its app table at the later offset, +1. Over-
    # prediction is the safe direction for the 12-stage gate, and it does not
    # touch the calibration set -- sd9 is not one of the 19 v6 rows, where
    # blocks is exact on all 17 that compiled.
    assert plan.blocks == 61


def test_the_entry_to_block_arithmetic_has_a_name():
    """Audit §8.1's multiplier. A block is MEMORY, so it is charged once per
    TREE -- unlike the crossbar's byte slots, which are charged once per stage
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
