"""Task 3 (plan "Step 2"): the per-table regression gate, scored over every
CSV this project has archived from real p4c compiles (2026-09-20 rewrite
design Sec 6.4). See scripts/tcam_table_scoreboard.py's module docstring for
what each of the three predicted quantities is and how a table sharing its
stage with a different key is charged per source -- the packer's own ordered
stage simulation (audit C5), since the fitted crowded-stage margin was
retired on 2026-09-28.
"""
import os

import pytest

from scripts import tcam_table_scoreboard as scoreboard

pytestmark = pytest.mark.skipif(
    not scoreboard.sources_present(),
    reason="needs results/*.csv (gitignored, real p4c compile output) -- "
           "see scripts/tcam_table_scoreboard.py's SOURCE_FILES")

# The packer refused no archived placement once the 62-byte mixed-key net
# was retired (spec 2026-09-29): every observation is scored, and the lane
# price reproduces p4c's charge on every later key except the four
# a32x2_b32x2 tables (two solid 32-byte keys at exactly 64 bytes), charged
# 7 against p4c's 6 -- an over-prediction, the safe side.
OVER_PREDICTED_BESIDE_ANOTHER_KEY = frozenset(
    "a32x2_b32x2/tern_%s%d" % (tag, i) for tag in "ab" for i in (0, 1))


def test_total_observation_count_is_405():
    # 308 fitted-corpus observations + 50 held-out tables + 47 mixed-key cap
    # probes (tcam_mixed_key_cap_sweep 30, tcam_mixed_key_cap_onset 17).
    rows = scoreboard.score_all()

    assert len(rows) == 405


def test_the_held_out_tables_are_priced_exactly():
    """tcam_heldout_harvest.csv: 50 classification tables from the 8
    compiles in results/compiler_calibration_extra/, which no part of the
    model was fitted on. Every one priced exactly -- refined and charged."""
    rows = [r for r in scoreboard.score_all()
            if r["source"] == "tcam_heldout_harvest"]

    assert len(rows) == 50
    assert all(r["diff_refined"] == 0 for r in rows)
    assert all(r["diff_charged"] == 0 for r in rows)


# Unders that are expected under the lane price (2026-10-04). Pre-pin probes:
# compiled WITHOUT @pa_container_size, so p4c splits each 24-remainder field
# W24-low (results/tcam_phv_slice_sweep/compiles/w051 phv_allocation_summary:
# key_a0[50:24] | [23:0]) and charges the ladder's price, one block above the
# lane price of the PINNED fill-low layout every generated design now uses.
# independent_low_sd5: the accepted greedy miss (pinned, lane 2 vs p4c 3).
_PRE_PIN_FILL_LOW_PROBES = {
    ("tcam_discount_scan", i) for i in (
        "dsp01", "dsp02", "dsp03", "dsp04", "dsp05", "dsp06", "dsp07",
        "dsp08", "dsp09", "dsp10", "dsp43", "dsp44", "dsp45")} | {
    ("tcam_phv_slice_sweep", "w049"), ("tcam_phv_slice_sweep", "w050"),
    ("tcam_phv_slice_sweep", "w051"), ("tcam_phv_slice_sweep", "w052"),
    ("tcam_ledger_divergence_sweep", "div00_n1_g3"),
    ("tcam_version_sweep", "d_unreachable_B22"),
    ("tcam_lane_sweep", "lane_a6"),
    ("tcam_field_count_sweep", "t51_n2"), ("tcam_field_count_sweep", "t51_n3_big")}
_ACCEPTED_GREEDY_MISSES = {
    ("tcam_offset_harvest", "independent_low_sd5/get_classification_tree_ddos_%d" % i)
    for i in range(3)}


def test_blocks_charged_never_under_predicts_a_placement_the_model_emits():
    """The gate the whole script exists for: on every observation whose
    placement the packer would also produce, blocks_charged is never below
    what p4c charged -- except the listed pre-pin probes and the accepted
    greedy miss, each exactly one block (three trees for sd5)."""
    rows = scoreboard.score_all()
    unders = scoreboard.under_predictions(rows, "diff_charged")
    assert {(r["source"], r["identifier"]) for r in unders} == (
        _PRE_PIN_FILL_LOW_PROBES | _ACCEPTED_GREEDY_MISSES)
    assert all(r["diff_charged"] == -1 for r in unders)


def test_no_archived_placement_is_refused():
    """Without the 62-byte net the packer can emit every archived shared
    stage: none is refused, and none for want of a lane fit either."""
    rows = scoreboard.score_all()

    assert scoreboard.refused_placements(rows) == []


def test_every_charge_above_the_refined_price_is_what_p4c_charged():
    """A table is charged above its standalone (refined) price only when a
    different key was placed ahead of it in its stage -- and on every such
    emitted row the lane leftover price is exactly p4c's count: the ragged
    (179, 204) key at 10 behind a 12-byte key (tcam_stretch_sweep), the
    (54, 56) probe at 4 behind a spacer at 59-62 bytes, the
    (46, 46, 48, 64) probe at 6 at 61 bytes (tcam_mixed_key_cap_sweep), the
    dsp41/dsp42 (84, 84) probe at 7 at 63-64 bytes, rag11 at 5 at 64, rag14
    at 5 at 63-64. The one exception is the a32x2_b32x2 over-prediction (7
    against p4c's 6, two solid 32-byte keys at exactly 64 bytes).
    The retired crowded-stage margin charged +1 by byte total alone."""
    rows = scoreboard.score_all()

    extra = [r for r in rows if r["blocks_charged"] > r["blocks_refined"]
             and not r["placement_refused"]]

    assert len(extra) == 16
    assert all(r["not_first"] and r["behind_other_key"] for r in extra)
    over = {r["identifier"] for r in extra if r["diff_charged"] != 0}
    assert over == OVER_PREDICTED_BESIDE_ANOTHER_KEY
    assert all(r["diff_charged"] == 1 for r in extra
               if r["identifier"] in OVER_PREDICTED_BESIDE_ANOTHER_KEY)


def test_a_later_key_is_charged_its_lane_price_at_59_to_64_bytes():
    """rag14 = (54, 56), not saturated: 3 blocks alone, 4 behind a solid
    spacer at 59-62 bytes -- the lanes the spacer leaves hold its bytes only
    at 4 blocks. Charged 4 there, as p4c did."""
    rows = {r["identifier"]: r for r in scoreboard.score_all()
            if r["source"] == "tcam_mixed_key_cap_sweep"}
    for t, blocks in ((59, 4), (60, 4), (61, 4), (62, 4), (63, 5), (64, 5)):
        row = rows["rag14_t%d" % t]
        assert not row["saturated"]
        assert (row["blocks_charged"], row["observed_blocks"]) == (blocks, blocks)


def test_the_solid_wide_key_is_charged_nothing_behind_a_narrow_key():
    """a49x1_b12x4/a49x1_b12x5: the SOLID 49-byte control shares a 61-byte
    stage with 12-byte keys and p4c charged it 9, its standalone price. The
    retired crowded margin charged 10; the lane simulation, like p4c, 9."""
    rows = {r["identifier"]: r for r in scoreboard.score_all()
            if r["source"] == "tcam_stretch_sweep"}
    for point in ("a49x1_b12x4", "a49x1_b12x5"):
        row = rows["%s/tern_a0" % point]
        assert (row["blocks_charged"], row["observed_blocks"]) == (9, 9)


def test_phv_slice_sweep_scores_two_fields_not_one_merged_field():
    """C2: the probe (scripts/tcam_phv_slice_sweep.py:193) declares TWO
    fields, `fields = (clean_bits, 8 * solid_bytes)`, and calls
    codeword_to_blocks(fields) with that pair -- so the scoreboard must score
    the same pair, not a single field merged to clean_bits + solid_bits. The
    merge silently drops the isolation credit (tail_is_isolatable only fires
    on the true nibble-clean field, clean_bits) and over-predicts 16 of the
    24 rows; scored correctly every row is exact."""
    rows = scoreboard.score_phv_slice_sweep(
        os.path.join(scoreboard.RESULTS_DIR, "tcam_phv_slice_sweep.csv"))

    assert len(rows) == 24
    exact = [r for r in rows if r["diff_refined"] == 0]
    assert len(exact) == 24, [(r["identifier"], r["diff_refined"]) for r in rows]


def test_offset_harvest_headline_over_predicts_exactly_8_of_100():
    """Spec Sec 4.2/13.1's own headline gate: the isolation refinement is a
    no-op on every archived design's totals, but visible per table -- 8 of
    the 100 archived classification tables show blocks_headline strictly
    above the real count (the two 33-byte, 15-feature keys, once per tree),
    and blocks_refined (the production quantity, isolation credit included)
    is exact -- 0 over, 0 under -- on all 100."""
    rows = scoreboard.score_offset_harvest(
        os.path.join(scoreboard.RESULTS_DIR, "tcam_offset_harvest.csv"))

    assert len(rows) == 100

    headline_over = [r for r in rows if r["diff_headline"] > 0]
    headline_under = [r for r in rows if r["diff_headline"] < 0]
    assert len(headline_over) == 8
    assert len(headline_under) == 0

    refined_exact = [r for r in rows if r["diff_refined"] == 0]
    assert len(refined_exact) == 100
