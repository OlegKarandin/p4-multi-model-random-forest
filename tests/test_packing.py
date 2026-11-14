"""Behaviour of p4model/packing.py: the column geometry and the stage packer
(`crossbar_stages_needed`)."""
import os

import pytest

from p4model import packing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_table_wider_than_one_column_never_fits():
    # A table chains its blocks down ONE 12-row column (scripts/
    # tcam_column_sweep.py, Mechanism C), so a 13-block shard fits no stage;
    # _stage_shards splits such tables before they reach fits_two_columns.
    assert not packing.fits_two_columns([13])
    assert packing.fits_two_columns([12])


def test_a_deep_table_pays_a_later_keys_price_once_per_row_word():
    """A table deeper than one 512-row word stores more rows through the SAME
    key: the key's crossbar slots are laid out once, and every word repeats
    that layout. So a key that pays a leftover price behind a different key
    pays it once per word -- the ordered stage simulation's `price x rows`
    (audit C5; reviews/model_audit_scratch/proto_model.py charges the same).

    (This replaces finding 1.4's test, which pinned `offsets_for`'s running
    sum of key widths. offsets_for, key_width and the crowded-stage margin
    they served are deleted: the ordered simulation reads the ORDER keys were
    placed in directly, and no offset exists any more.)

    The spacer key (360,) is placed first (placement priority 2) and pays its
    standalone 9 blocks; the probe key (54, 56) -- independent_low_sd5's real
    app key, 3 blocks alone -- costs 4 per word in the slots the spacer left
    (results/tcam_mixed_key_cap_sweep.csv measured exactly 4 behind such a
    spacer). One word: 9 + 4 = 13. Two words (declared 6 blocks): 9 + 8 = 17.
    """
    from p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 45)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[spacer, probe],
                  key_field_bits=[(360,), (54, 56)], placement_priority=[2, 1])

    shallow = crossbar_stages_needed([(9, 45), (3, 14)], **kwargs)
    deep = crossbar_stages_needed([(9, 45), (6, 14)], **kwargs)

    assert (shallow.occupied, deep.occupied) == (1, 1), (shallow, deep)
    assert (shallow.blocks, deep.blocks) == (13, 17), (shallow, deep)


# --------------------------------------------------------------------------
# The sharing charge (p4model/packing.py's ordered stage simulation): tables
# are placed in p4c's order (placement priority, then the table listed LAST
# first), and each stage's distinct keys are priced in that order -- the first
# key its declared blocks, every later key the price p4c's crossbar allocation
# (ported in p4model/xbar.py, 2026-11-13) gives it in the crossbar slots the
# keys before it left. Two different keys share a stage whenever that
# allocation fits them. Some comments below still record the price the
# earlier lane count (p4model/lanes.py, 2026-10-04) gave, for history.
# --------------------------------------------------------------------------
def test_the_packer_charges_sd5s_stage_twelve_blocks_as_p4c_does():
    # Modelled on independent_low_sd5's stage 6, where p4c charged its app and
    # ddos trees 3 TCAM blocks each (resources.json): 12 in all. Under the lane
    # price (2026-10-04) the ddos key (27, 52) cost 2 per tree -- 9 in all, the
    # accepted greedy-miss UNDER of 3. p4c's crossbar allocation, ported
    # (xbar.py, 2026-11-13), gives the key 3 and the stage 12.
    from p4model.lanes import table_blocks
    from p4model.packing import crossbar_stages_needed
    from p4model.tables import codeword_to_blocks

    app = frozenset({(("code", "app_flm"), 7), (("code", "app_plm"), 7)})
    ddos = frozenset({(("code", "ddos_bplm"), 4), (("code", "ddos_plm"), 7)})
    assert codeword_to_blocks((27, 52)) == 3
    assert table_blocks((27, 52)) == 3
    assert table_blocks((54, 56)) == 3
    plan = crossbar_stages_needed(
        [(3, 14), (3, 11), (3, 11), (3, 11)],
        key_fields=[app, ddos, ddos, ddos],
        key_field_bits=[(54, 56), (27, 52), (27, 52), (27, 52)],
        placement_priority=[1, 2, 2, 2])
    assert plan.blocks == 12


def test_a_stage_of_one_shared_key_is_charged_exactly_its_declared_blocks():
    # Every 'joint' design keys every tree on the identical field set, so a
    # joint stage has exactly ONE distinct key, which is its first key and is
    # charged its declared blocks (plan invariant 1 -- what keeps joint
    # designs identical to threshold alignment's total_blocks). The key below
    # is 60 bytes, specifically so this test cannot pass merely because the
    # key happened to be small: one key may use the whole crossbar.
    from p4model.lanes import table_blocks
    from p4model.packing import crossbar_stages_needed
    from p4model.tables import codeword_to_blocks

    key = frozenset({(("code", "f"), 60)})
    assert codeword_to_blocks((480,)) == 11
    assert table_blocks((480,)) == 11
    plan = crossbar_stages_needed(
        [(11, 60)] * 3, key_fields=[key] * 3,
        key_field_bits=[(480,)] * 3)
    assert plan.blocks == 33


def test_omitting_key_field_bits_prices_every_table_at_its_declared_width():
    # Without field widths there is no crossbar simulation: every table costs what
    # it declares. That is what keeps the range pool -- the one caller that
    # passes no bits -- on exactly its pre-existing pricing.
    from p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed([(3, 14), (2, 11), (2, 11), (2, 11)])
    assert plan.blocks == 9


def test_placement_priority_needs_key_field_bits_and_one_entry_per_table():
    from p4model.packing import crossbar_stages_needed

    with pytest.raises(ValueError, match="key_field_bits"):
        crossbar_stages_needed([(1, 2)], placement_priority=[1])
    with pytest.raises(ValueError, match="placement_priority"):
        crossbar_stages_needed([(1, 2), (1, 2)], key_field_bits=[(8,), (8,)],
                               placement_priority=[1])


def test_the_generator_emits_the_placement_priority_the_packer_replays():
    # The simulation's order is only right if it is the order the generator
    # pins: every classification tree table in the committed golden program
    # (p4/p4_code_RF_models_shared.p4, regenerated and diffed by
    # test_build_p4_script_tna.py) carries @placement_priority equal to
    # program.PLACEMENT_PRIORITY of its task, and the trees are listed app
    # first, then ddos, each by index -- evaluation._pool_inputs' order.
    import re

    from p4model.program import PLACEMENT_PRIORITY

    path = os.path.join(REPO_ROOT, "p4", "p4_code_RF_models_shared.p4")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = re.findall(
        r"@placement_priority\((\d+)\)\s*\n\s*table get_classification_tree_"
        r"(app|ddos)_(\d+)\s*\{", text)
    assert found
    for priority, task, _index in found:
        assert int(priority) == PLACEMENT_PRIORITY[task], (task, priority)
    listed = [(task, int(index)) for _p, task, index in found]
    assert listed == sorted(listed, key=lambda t: (t[0] != "app", t[1]))
    assert len(re.findall(r"table get_classification_tree_", text)) == len(found)


def test_two_different_ragged_keys_do_not_share_a_stage():
    # independent_low_sd9, the row a sharing price exists for. Its app trees
    # key 179 + 204 bits = 49 crossbar bytes at 9 blocks; its ddos trees key
    # 37 + 49 bits = 12 bytes at 3 blocks. 2 app + 2 ddos is exactly 24 blocks
    # and packs both columns as 9+3 | 9+3, so every BLOCK/TABLE-COUNT limit
    # would let them share a stage -- and p4c refuses. The simulation places
    # the five ddos trees first (priority 2), and an app tree behind them
    # costs 10, not 9, in the slots they left (15 + 10 > 24): the app trees go
    # 2 | 2 | 1 in three more stages. p4c's committed placement exactly
    # (resources.json: 5 ddos | 2 app | 2 app | 1 app, 5 x 9 + 5 x 3 = 60
    # blocks). The retired crowded margin co-located one app tree with the
    # ddos trees and charged 61. Measured on the same key by
    # scripts/tcam_stretch_sweep.py: 9 blocks alone, 10 beside a 12-byte key.
    from p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] * 5 + [(3, 12)] * 5,
        readiness_levels=[0] * 10,
        key_fields=[app] * 5 + [ddos] * 5,
        key_field_bits=[(179, 204)] * 5 + [(37, 49)] * 5,
        placement_priority=[1] * 5 + [2] * 5)
    assert (plan.occupied, plan.blocks) == (4, 60)
    assert [load.blocks for load in plan.stage_loads] == [
        (3, 3, 3, 3, 3), (9, 9), (9, 9), (9,)]


def test_the_ragged_key_pays_one_block_behind_four_narrow_tables():
    # scripts/tcam_stretch_sweep.py's ragged_ax1_bx4: a 49-byte app key (9
    # blocks) beside four 12-byte ddos keys (3 blocks each). p4c places all
    # five in ONE stage at 22 TCAM blocks (results/tcam_stretch_sweep.csv):
    # the ddos key first (4 x 3 = 12, one column) and the app key second at
    # 10, the other column. The simulation, ddos first by priority, agrees.
    from p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4,
        readiness_levels=[0] * 5,
        key_fields=[app] + [ddos] * 4,
        key_field_bits=[(179, 204)] + [(37, 49)] * 4,
        placement_priority=[1] + [2] * 4)
    assert (plan.occupied, plan.blocks) == (1, 22)


_SPACER = frozenset({(("code", "spacer"), 41)})
_PROBE = frozenset({(("code", "probe_a"), 11), (("code", "probe_b"), 11)})


def test_a_probe_behind_a_crowded_spacer_pays_p4cs_seven_blocks():
    # results/tcam_discount_scan.csv dsp41: a (84, 84) probe key (22 crossbar
    # bytes) behind a 41-byte spacer, both in ONE stage; p4c charged the probe 7
    # blocks. That compile pinned no containers: p4c laid each 84-bit field out
    # as W24 | W32 | W28 (its prog.bfa), and with that layout the crossbar port
    # (xbar.py) charges exactly p4c's 8 + 7. Under the layout the generator pins
    # (lanes.layout, fill-low W32 | W32 | W20) the port's price is 6, never
    # compiled.
    from p4model.packing import crossbar_stages_needed

    def probe(slices):
        return tuple((("probe", i), slices) for i in range(2))

    spacer = ((("spacer", 0), (("B", 8),) + (("W", 32),) * 10),)
    measured = crossbar_stages_needed(
        [(8, 41), (4, 22)], readiness_levels=[0] * 2,
        key_fields=[_SPACER, _PROBE], key_field_bits=[(328,), (84, 84)],
        key_slices=[spacer, probe((("W", 24), ("W", 32), ("W", 28)))],
        placement_priority=[2, 1])
    assert (measured.occupied, measured.table_blocks, measured.table_stages) == (
        1, (8, 7), (0, 0))

    pinned = crossbar_stages_needed(
        [(8, 41), (4, 22)], readiness_levels=[0] * 2,
        key_fields=[_SPACER, _PROBE], key_field_bits=[(328,), (84, 84)],
        placement_priority=[2, 1])
    assert (pinned.occupied, pinned.table_blocks, pinned.table_stages) == (
        1, (8, 6), (0, 0))


def test_a_41_plus_22_byte_mixed_stage_now_shares_one_stage():
    # The dsp41 shape (not M150_k7_s11's real keys): 41 + 22 = 63 bytes of
    # two different keys, refused by the old 62-byte
    # net; with the generator's layout pins p4c places both in one stage
    # (spec 2026-09-29 Sec 5.2), and so does the packer. Probe listed last,
    # so placed first at its own 4; the spacer behind it pays its crossbar
    # leftover 8.
    from p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed(
        [(8, 41), (4, 22)], readiness_levels=[0] * 2,
        key_fields=[_SPACER, _PROBE], key_field_bits=[(328,), (84, 84)])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 12, (0, 0))


def test_a_later_key_pays_its_crossbar_leftover_price():
    # results/tcam_mixed_key_cap_sweep.csv: independent_low_sd5's real app key
    # (54, 56) -- 14 bytes, 3 blocks alone -- costs 4 behind a solid spacer at
    # 59-62 combined bytes. Placed second (the spacer's priority is higher),
    # the probe is priced in the crossbar slots a 45-byte spacer left: 4, as measured
    # -- 9 + 4 = 13 in one stage. Placed FIRST, the same probe pays its own
    # 3 and the spacer, behind it, still finds its 9: 12. The order is the
    # generator's to pin, not the packer's to guess.
    from p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 45)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[spacer, probe],
                  key_field_bits=[(360,), (54, 56)])
    spacer_first = crossbar_stages_needed(
        [(9, 45), (3, 14)], placement_priority=[2, 1], **kwargs)
    probe_first = crossbar_stages_needed(
        [(9, 45), (3, 14)], placement_priority=[1, 2], **kwargs)
    assert (spacer_first.occupied, spacer_first.blocks) == (1, 13)
    assert spacer_first.stage_loads[0].blocks == (9, 4)
    assert (probe_first.occupied, probe_first.blocks) == (1, 12)


def test_a_later_key_with_crossbar_slots_to_spare_pays_nothing():
    # Across six probe shapes no probe paid anything at 58 combined bytes or
    # below (tcam_mixed_key_cap_sweep/onset). Behind a 44-byte spacer the
    # (54, 56) probe still finds slots for its own 3 blocks: 12. (The retired
    # margin read this as its "free edge"; the simulation needs no threshold.)
    from p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 44), (3, 14)], readiness_levels=[0] * 2,
        key_fields=[spacer, probe], key_field_bits=[(352,), (54, 56)],
        placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks) == (1, 12)
    assert plan.stage_loads[0].blocks == (9, 3)


def test_one_key_may_use_the_whole_crossbar_in_one_stage():
    # Crowding is about a SECOND key finding the groups the first left; one key
    # alone may use the whole 64-byte budget (measured: one table at exactly
    # 64 bytes, reviews/p4_tofino_reference.md Sec 4.3). Two trees of one
    # 64-byte key share their field, so the stage still holds 64 bytes.
    from p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "wide"), 64)})
    plan = crossbar_stages_needed(
        [(12, 64), (12, 64)], readiness_levels=[0] * 2,
        key_fields=[key, key], key_field_bits=[(512,), (512,)])
    assert plan.occupied == 1


def test_a_table_is_sharded_as_full_columns_plus_a_remainder():
    """Finding 1.5a. The equal split rounded each piece up, so a 13-block table
    was charged 14. A column holds TCAM_ROWS_PER_STAGE blocks; fill columns and
    leave the remainder, and the shards sum to the table."""
    from p4model.packing import _stage_shards

    assert _stage_shards(13, 30) == [(12, 30), (1, 30)]
    assert _stage_shards(23, 30) == [(12, 30), (11, 30)]
    assert _stage_shards(25, 30) == [(12, 30), (12, 30), (1, 30)]


def test_shards_always_sum_to_the_table_and_fit_a_column():
    """The two invariants the split must never break, over the whole reachable
    range: nothing is lost, and no shard is wider than one column."""
    from p4model.packing import _stage_shards
    from p4model.target import TCAM_ROWS_PER_STAGE

    for blocks in range(0, 60):
        shards = _stage_shards(blocks, 30)

        assert sum(b for b, _w in shards) == blocks, blocks
        assert all(b <= TCAM_ROWS_PER_STAGE for b, _w in shards), blocks
        assert shards, blocks


def test_a_table_inside_one_column_is_not_sharded():
    """The common case, and the one the whole 19-row archive lives in: no
    archived table exceeds 12 blocks, so this path is the only one the
    calibration ever exercises."""
    from p4model.packing import _stage_shards

    assert _stage_shards(12, 30) == [(12, 30)]
    assert _stage_shards(1, 30) == [(1, 30)]
    assert _stage_shards(0, 30) == [(0, 30)]


def _parity_feasible_under_some_order(heights, rows, columns):
    """Is there ANY order and column assignment placing all of `heights` under
    p4c's row-parity rule -- a run of EVEN height may only start on an EVEN
    row, a run of odd height may start anywhere
    (Memories::find_ternary_stretch, rewrite design Sec 6.2/13.2)?

    Exhaustive with memoisation on (tables left, column loads); a column's
    load IS the next free row in it, since runs are placed bottom-up with no
    deliberate holes."""
    seen = set()

    def search(remaining, loads):
        if not remaining:
            return True
        if (remaining, loads) in seen:
            return False
        seen.add((remaining, loads))
        for i, height in enumerate(remaining):
            if i and remaining[i - 1] == height:
                continue                       # same multiset branch already tried
            rest = remaining[:i] + remaining[i + 1:]
            for column in range(columns):
                if loads[column] + height > rows:
                    continue
                if height % 2 == 0 and loads[column] % 2:
                    continue                   # even run may not start on an odd row
                bumped = list(loads)
                bumped[column] += height
                if search(rest, tuple(bumped)):
                    return True
        return False

    return search(tuple(sorted(heights)), (0,) * columns)


def test_row_parity_never_rejects_a_stage_that_fits_by_size():
    """Step 3a. p4c's row-parity rule costs this packer nothing, so
    fits_two_columns is right to model column LOADS and ignore rows.

    Proof by exhaustion over every multiset of block heights 1..12 that is
    size-feasible in two 12-row columns: parity-feasible-under-some-order is
    EQUIVALENT to fits_two_columns on all 8 618 of them, with no disagreement
    in either direction. The reason it must be so: place every EVEN height
    first, and the prefix sums of even numbers stay even, so every even run
    lands on an even row; odd runs then go anywhere. A multiset that fits by
    size therefore always fits under parity for SOME order.

    A superset of a size-infeasible multiset is size-infeasible and
    parity-infeasible alike (both tests are monotone in the multiset), so
    pruning there loses no case.

    Confirmed against the archive, not just by construction: over the 20
    compiles the rewrite design cites, 317 committed runs include 103 that
    start on an ODD row and ZERO even-height odd starts -- the rule itself
    holds without exception -- and exactly 2 of 183 committed columns contain
    a parity gap at all (independent_high_sd6 and _sd7, stage 9: a 3-high ddos
    run at rows 0-2, then a 2-high app run forced off row 3 to row 4). Both
    gaps sit in columns using 6 of 12 rows, so neither spilled anything.
    """
    from p4model.packing import fits_two_columns
    from p4model.target import TCAM_COLUMNS_PER_STAGE, TCAM_ROWS_PER_STAGE

    rows, columns = TCAM_ROWS_PER_STAGE, TCAM_COLUMNS_PER_STAGE
    checked = 0

    def walk(prefix, largest):
        nonlocal checked
        if prefix:
            checked += 1
            by_size = fits_two_columns(prefix)
            assert by_size == _parity_feasible_under_some_order(
                prefix, rows, columns), prefix
            if not by_size:
                return                        # monotone: every superset fails both
        for height in range(min(largest, rows), 0, -1):
            walk(prefix + [height], height)   # non-increasing, so each multiset once

    walk([], rows)
    assert checked == 8618, checked


def test_a_saturated_key_in_an_uncrowded_shared_stage_pays_nothing():
    # independent_high_sd7's real stage 9: a SATURATED 10-byte app key
    # (crossbar_capacity(2) == 10) beside a 12-byte ddos key, 22 bytes in all,
    # and p4c charged nothing (tcam_offset_harvest.csv). The retired per-key
    # saturation margin charged it +1; the crossbar port finds the later
    # key's slots free and charges nothing, whichever key goes first.
    from p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_%d" % i), 1) for i in range(10)})
    ddos = frozenset({(("code", "ddos_%d" % i), 1) for i in range(12)})
    plan = crossbar_stages_needed(
        [(2, 10), (3, 12)], readiness_levels=[0] * 2,
        key_fields=[app, ddos],
        key_field_bits=[(1, 1, 1, 2, 3, 3, 3, 3, 4, 8),
                        (1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 5)])
    assert (plan.occupied, plan.blocks) == (1, 5)


# --- seed_stages: a pool packed into stages another pool already partly fills
# (audit C1). Seeds count against the table cap, the byte limit and the column
# packing, and are never charged or reported by the seeded pool.

def _seed(index, blocks, byte_widths, tables=None):
    from p4model.packing import StageLoad

    return StageLoad(index=index, blocks=tuple(blocks),
                     fields=frozenset((("val", index, i), w)
                                      for i, w in enumerate(byte_widths)),
                     tables=len(blocks) if tables is None else tables)


def test_a_seed_counts_against_the_eight_table_cap():
    from p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "k"), 2)})
    plan = crossbar_stages_needed(
        [(1, 2), (1, 2)], readiness_levels=[5, 5], key_fields=[key, key],
        seed_stages=[_seed(5, [1] * 7, [2] * 7)])
    # 7 seeded + 1 = 8 at stage 5; the second table spills to 6.
    assert plan.table_stages in ((5, 6), (6, 5))
    assert plan.indices == {5, 6}


def test_a_seed_counts_against_the_64_byte_limit():
    from p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "k"), 8)})
    plan = crossbar_stages_needed(
        [(2, 8)], readiness_levels=[5], key_fields=[key],
        seed_stages=[_seed(5, [1], [60])])
    assert plan.table_stages == (6,)


def test_a_seed_counts_against_the_column_packing():
    from p4model.packing import crossbar_stages_needed

    seed = _seed(5, [12, 8], [2, 2])
    # 12 | 8 + 8 = 16 overflows the second column; 12 | 8 + 4 fits.
    assert crossbar_stages_needed(
        [(8, 5)], readiness_levels=[5],
        seed_stages=[seed]).table_stages == (6,)
    assert crossbar_stages_needed(
        [(4, 5)], readiness_levels=[5],
        seed_stages=[seed]).table_stages == (5,)


def test_a_seed_is_never_charged_or_reported_by_the_seeded_pool():
    from p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed(
        [(3, 5)], readiness_levels=[5],
        seed_stages=[_seed(5, [2, 2], [2, 2]), _seed(9, [4], [2])])
    assert (plan.occupied, plan.depth, plan.indices, plan.blocks) == (
        1, 6, frozenset({5}), 3)
    assert [(load.index, load.blocks, load.tables)
            for load in plan.stage_loads] == [(5, (3,), 1)]


def test_seeds_leave_an_unseeded_placement_unchanged():
    # Seeds at stages the pool never reaches change nothing at all.
    from p4model.packing import crossbar_stages_needed

    specs = [(9, 45), (3, 14), (6, 10), (6, 10)]
    fields = [frozenset({(("code", "s"), 45)}),
              frozenset({(("code", "a"), 7), (("code", "b"), 7)}),
              frozenset({(("code", "c"), 10)}), frozenset({(("code", "c"), 10)})]
    bits = [(360,), (54, 56), (80,), (80,)]
    kwargs = dict(readiness_levels=[4] * 4, key_fields=fields,
                  key_field_bits=bits)
    assert crossbar_stages_needed(specs, **kwargs) == crossbar_stages_needed(
        specs, seed_stages=[_seed(0, [12, 12], [2, 2]), _seed(3, [1], [2])],
        **kwargs)


def test_seed_range_keys_are_placed_ahead_of_every_tree_key():
    # A seed's range fields sit on the same ternary crossbar, and the range
    # tables were placed before any tree reached the stage, so the stage
    # simulation places them FIRST: every tree key there is a later key.
    # Nothing was ever measured with a range key beside a tree key (before
    # audit C1 the two pools never shared a stage), so they are not exempted.
    # Without a seed the spacer (listed last, so placed first) and the probe
    # share stage 0 at 9 + 3 = 12. With two 2-byte range keys already there,
    # 4 + 44 + 14 = 62 bytes are within the 64-byte crossbar, but the lanes
    # are not: the spacer still fits behind the range keys, and the probe,
    # placed third, paid its lane leftover price of 4 instead of 3 for the
    # slots the range keys took. The seed itself is never charged.
    # 2026-11-13, p4c's crossbar allocation (xbar.py): a range byte takes a
    # slot for each of its two nibble ranges, so the two 2-byte range keys
    # hold 8 slots; behind them and the 44-byte spacer the 14-byte probe no
    # longer fits the stage, and waits for stage 1 at its stand-alone 3.
    # (Until 2026-11-06 the second range key consumed a whole crossbar group,
    # which left the probe no legal fit at all and moved it to stage 1; a
    # later key now occupies only the slots its fit takes, lanes.
    # later_key_occupancy.)
    from p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[probe, spacer],
                  key_field_bits=[(54, 56), (352,)])
    alone = crossbar_stages_needed([(3, 14), (9, 44)], **kwargs)
    seeded = crossbar_stages_needed([(3, 14), (9, 44)],
                                    seed_stages=[_seed(0, [1, 1], [2, 2])],
                                    **kwargs)
    assert (alone.occupied, alone.blocks, alone.table_stages) == (1, 12, (0, 0))
    assert (seeded.occupied, seeded.blocks, seeded.table_stages) == (2, 12, (1, 0))


def test_a_tree_key_behind_a_seed_pays_its_leftover_price_never_less():
    # The 60-byte (480,) key costs 11 blocks alone. Behind one 2-byte range key
    # its crossbar slots are short and it costs 12; the first tree key behind a
    # seed is charged max(codeword_to_blocks, leftover price), so never below 11.
    from p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "wide"), 60)})
    plan = crossbar_stages_needed(
        [(11, 60)], readiness_levels=[0], key_fields=[key],
        key_field_bits=[(480,)], seed_stages=[_seed(0, [1], [2])])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 12, (0,))
    small = frozenset({(("code", "small"), 2)})
    plan = crossbar_stages_needed(
        [(1, 2)], readiness_levels=[0], key_fields=[small],
        key_field_bits=[(16,)], seed_stages=[_seed(0, [1], [2])])
    assert plan.blocks == 1


def test_a_seed_and_tree_keys_share_a_stage_only_if_the_crossbar_fits_them():
    # 44 + 14 + 6 = 64 bytes, three range keys ahead of two tree keys. The old
    # 62-byte net refused this outright; now only the crossbar port decides.
    # It still cannot fit the spacer behind the seed and the probe, so the
    # spacer moves on -- same placement, new reason.
    from p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 44), (3, 14)], readiness_levels=[0] * 2,
        key_fields=[spacer, probe], key_field_bits=[(352,), (54, 56)],
        seed_stages=[_seed(0, [1, 1, 1], [2, 2, 2])])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (2, 12, (1, 0))


def test_seed_stages_need_readiness_levels_and_one_load_per_stage():
    from p4model.packing import crossbar_stages_needed

    with pytest.raises(ValueError, match="readiness_levels"):
        crossbar_stages_needed([(1, 2)], seed_stages=[_seed(0, [1], [2])])
    with pytest.raises(ValueError, match="two seed_stages entries"):
        crossbar_stages_needed([(1, 2)], readiness_levels=[0],
                               seed_stages=[_seed(0, [1], [2]),
                                            _seed(0, [1], [2])])


def test_table_stages_is_each_tables_latest_shard_in_both_branches():
    from p4model.packing import crossbar_stages_needed

    # Pure packer: a 30-block table is 12 | 12 | 6 shards; the two full ones
    # fill stage 0, the 6 and the 1-block table share stage 1.
    plan = crossbar_stages_needed([(30, 5), (1, 2)])
    assert plan.table_stages == (1, 1)
    assert [load.blocks for load in plan.stage_loads] == [(12, 12), (6, 1)]
    # Dependency-aware: the same table from level 3 ends at stage 4.
    plan = crossbar_stages_needed([(30, 5), (1, 2)], readiness_levels=[3, 7])
    assert plan.table_stages == (4, 7)
    assert sum(sum(load.blocks) for load in plan.stage_loads) == plan.blocks


def test_stage_load_fits_checks_the_combined_limits():
    from p4model.packing import stage_load_fits

    assert stage_load_fits([_seed(0, [12], [2]), _seed(0, [6, 6], [30])])
    assert not stage_load_fits([_seed(0, [12], [2]), _seed(0, [8, 8], [2])])
    assert not stage_load_fits([_seed(0, [1] * 5, [2]), _seed(0, [1] * 4, [2])])
    # No byte clause (2026-11-13): the crossbar simulation placed the keys.
    assert stage_load_fits([_seed(0, [1], [40]), _seed(0, [1], [30])])


def test_a_classification_table_wider_than_a_stage_is_split_by_whole_rows():
    # A 30-block table of a 5-block key (6 row words) cannot sit in one
    # stage's 24 blocks. The ordered simulation (two keys here, so it runs)
    # splits it into chunks of whole words that each can -- 4 words (20
    # blocks, 12 | 8) then 2 (10) -- every chunk keeping the full key; the
    # table is done at the later stage. The 1-block second key, placed after
    # the first chunk, finds crossbar slots to spare.
    from p4model.packing import crossbar_stages_needed

    wide = frozenset({(("code", "w"), 25)})
    small = frozenset({(("code", "s"), 2)})
    plan = crossbar_stages_needed([(30, 25), (1, 2)], key_fields=[wide, small],
                                  key_field_bits=[(200,), (16,)],
                                  placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (2, 31, (1, 0))
    assert [load.blocks for load in plan.stage_loads] == [(12, 8, 1), (10,)]


def test_a_classification_table_wider_than_a_column_stays_in_one_stage():
    # 15 blocks spans both columns of ONE stage (12 | 3, measured at 14/16/24
    # blocks, _stage_shards) and counts as ONE table against the 8-table cap.
    from p4model.packing import crossbar_stages_needed

    wide = frozenset({(("code", "w"), 25)})
    small = frozenset({(("code", "s"), 2)})
    plan = crossbar_stages_needed([(15, 25), (1, 2)], key_fields=[wide, small],
                                  key_field_bits=[(200,), (16,)],
                                  readiness_levels=[2, 2],
                                  placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 16, (2, 2))
    assert plan.stage_loads[0].tables == 2


def _single_key_config(rng, dist):
    from p4model.errors import CrossbarKeyTooWide
    from p4model.lanes import table_blocks

    while True:
        if dist == "small":
            bits = tuple(rng.randint(1, 12) for _ in range(rng.randint(1, 4)))
        else:
            bits = tuple(rng.randint(1, 60) for _ in range(rng.randint(1, 16)))
        width = sum(-(-b // 8) for b in bits)
        if width > 64:
            continue
        try:
            per_row = table_blocks(bits)
        except CrossbarKeyTooWide:
            continue
        break
    trees = rng.randint(1, 30)

    def rows():
        if dist == "mix":           # ~85% one-word trees, the realistic shape
            return 1 if rng.random() < 0.85 else rng.choice([2, 2, 3, 4])
        return rng.choice([1, 2])   # "small"

    specs = [(per_row * rows(), width) for _ in range(trees)]
    key = frozenset({(("code", i), -(-b // 8)) for i, b in enumerate(bits)})
    level = rng.randint(3, 8)
    priority = [rng.choice([1, 2]) for _ in range(trees)]
    return specs, key, bits, [level] * trees, priority


@pytest.mark.parametrize("dist", ["mix", "small"])
def test_a_single_key_pool_is_never_charged_above_its_stand_alone_prices(dist):
    # Since 2026-11-13 a single-key pool (every 'joint' design) is placed by the
    # crossbar simulation in p4c's order, not by the declared-price packer
    # it was until then: on campaign_2026_11's 4,800 joint programs
    # that puts 15,116 of 15,144 trees in p4c's own stage (the old packer
    # 12,546). A later table of the stage's one key reuses its slots, so no
    # table is ever charged above its stand-alone blocks; it can be charged
    # less (test_a_later_table_of_one_key_can_cost_a_block_less).
    import random

    from p4model.packing import StageLoad, crossbar_stages_needed

    rng = random.Random("single-key-" + dist)
    for case in range(40):
        specs, key, bits, levels, priority = _single_key_config(rng, dist)
        kwargs = dict(key_fields=[key] * len(specs), key_field_bits=[bits] * len(specs),
                      placement_priority=priority)
        if case % 4 == 0:
            plan = crossbar_stages_needed(specs, **kwargs)
        else:
            seed = StageLoad(index=levels[0] - 1, blocks=(1, 1),
                             fields=frozenset({(("val", 0), 2), (("val", 1), 2)}),
                             tables=2)
            plan = crossbar_stages_needed(specs, readiness_levels=levels,
                                          seed_stages=[seed], **kwargs)
        assert all(charged <= declared for charged, (declared, _)
                   in zip(plan.table_blocks, specs)), (dist, case)
        assert plan.blocks == sum(plan.table_blocks)


def test_a_later_table_of_one_key_can_cost_a_block_less():
    # joint-off_M035_s22_k06 (campaign_2026_11), stage 7: four ddos trees of one
    # (17, 21, 21, 25, 27, 31) key, charged 5, 5, 4, 5 by p4c -- the third
    # reuses the earlier tables' slots in a shape one block cheaper than alone.
    from p4model.lanes import table_blocks
    from p4model.packing import crossbar_stages_needed

    bits = (17, 21, 21, 25, 27, 31)
    key = frozenset({(("code", i), -(-b // 8)) for i, b in enumerate(bits)})
    assert table_blocks(bits) == 5
    plan = crossbar_stages_needed(
        [(5, 21)] * 4, readiness_levels=[0] * 4, key_fields=[key] * 4,
        key_field_bits=[bits] * 4, placement_priority=[2] * 4)
    assert (plan.table_blocks, plan.table_stages) == ((5, 5, 4, 5), (0, 0, 0, 0))


def test_a_second_key_or_a_reachable_seed_switches_the_simulation_on():
    # The single-key shortcut must not swallow a real pricing question: a
    # 12-byte key ahead of the ragged 49-byte one still makes it pay 10, and a
    # range seed IN the trees' stage still counts as a key placed first.
    from p4model.packing import StageLoad, crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    two = crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4, readiness_levels=[0] * 5,
        key_fields=[app] + [ddos] * 4,
        key_field_bits=[(179, 204)] + [(37, 49)] * 4,
        placement_priority=[1] + [2] * 4)
    assert two.blocks == 22
    wide = frozenset({(("code", "wide"), 60)})
    seed = StageLoad(index=0, blocks=(1,), fields=frozenset({(("val", 0), 2)}),
                     tables=1)
    seeded = crossbar_stages_needed(
        [(11, 60)], readiness_levels=[0], key_fields=[wide],
        key_field_bits=[(480,)], seed_stages=[seed])
    assert seeded.blocks == 12


# --------------------------------------------------------------------------
# Decision 7.4 (S12, S22, S29, S30): no load score, seeds in placement order,
# placement failures are refusals.
# --------------------------------------------------------------------------

def test_range_pool_orders_a_level_by_readers_then_later_listed_not_by_size():
    # S12: the `load` score is gone; a level's shards go by readers (more
    # first), then the table listed LATER first. The 8-table cap leaves the
    # LAST placed one behind, whatever its size.
    from p4model.packing import crossbar_stages_needed

    specs = [(5, 4), (1, 4), (1, 4)] + [(1, 4)] * 6
    fields = [frozenset({(("val", i), 4)}) for i in range(len(specs))]
    plan = crossbar_stages_needed(specs, readiness_levels=[0] * 9, key_fields=fields,
                                  table_dependents=[0] * 9)
    # later-listed first: tables 8..1 fill stage 0 (8 tables), table 0 -- the
    # big one, which `load` used to put first -- spills to stage 1
    assert plan.table_stages == (1,) + (0,) * 8


def test_range_pool_records_the_placement_order_of_a_stages_fields():
    # S22: StageLoad.field_order is the order tables were placed in the stage.
    from p4model.packing import crossbar_stages_needed

    fields = [frozenset({(("val", name), 4)}) for name in ("b", "a", "c")]
    plan = crossbar_stages_needed([(1, 4)] * 3, readiness_levels=[0] * 3,
                                  key_fields=fields, table_dependents=[0, 2, 1])
    (load,) = plan.stage_loads
    # most readers first: "a" (2), "c" (1), then "b" (0) -- not sorted, not listed
    assert load.field_order == ((("val", "a"), 4), (("val", "c"), 4), (("val", "b"), 4))
    assert set(load.field_order) == load.fields


def test_seeded_range_keys_enter_the_stage_crossbar_in_placement_order(monkeypatch):
    # S22: the seed's order, not repr order, reaches the crossbar simulation; a
    # hand-built seed without an order falls back to sorted.
    from p4model import packing
    from p4model.packing import StageLoad, crossbar_stages_needed

    seen = []
    real = packing._crossbar_stage

    def spy(seed_fields, units, mode='packed'):
        seen.append(seed_fields)
        return real(seed_fields, units, mode)

    monkeypatch.setattr(packing, '_crossbar_stage', spy)
    z, a = (("val", "z"), 4), (("val", "a"), 4)
    spec = dict(readiness_levels=[0], key_fields=[frozenset({(("code", "t"), 5)})],
                key_field_bits=[(40,)])
    ordered = StageLoad(index=0, blocks=(1, 1), fields=frozenset({z, a}), tables=2,
                        field_order=(z, a))
    crossbar_stages_needed([(1, 5)], seed_stages=[ordered], **spec)
    assert seen[-1] == (z, a)
    plain = StageLoad(index=0, blocks=(1, 1), fields=frozenset({z, a}), tables=2)
    crossbar_stages_needed([(1, 5)], seed_stages=[plain], **spec)
    assert seen[-1] == tuple(sorted((z, a), key=repr))


def test_a_classification_pool_that_never_places_is_a_placement_failure():
    # S30: the horizon guard is a refusal, not a RuntimeError crash.
    from p4model.errors import PlacementFailed
    from p4model.packing import StageLoad, _Unit, _place_in_order

    unit = _Unit(0, 0, 1, frozenset(), (), 0, 0)
    never = lambda here, unit, seed: None
    with pytest.raises(PlacementFailed, match="could not be placed"):
        _place_in_order([unit], never,
                        lambda i: StageLoad(index=i, blocks=(), fields=frozenset(), tables=0), 0)
