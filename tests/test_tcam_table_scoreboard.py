"""Task 3 (plan "Step 2"): the per-table regression gate, scored over every
CSV this project has archived from real p4c compiles (2026-09-20 rewrite
design Sec 6.4). See scripts/tcam_table_scoreboard.py's module docstring for
what each of the three predicted quantities is and how the Sec 13.2
stage-sharing margin is applied per source.
"""
import os

import pytest

from scripts import tcam_table_scoreboard as scoreboard

pytestmark = pytest.mark.skipif(
    not scoreboard.sources_present(),
    reason="needs results/*.csv (gitignored, real p4c compile output) -- "
           "see scripts/tcam_table_scoreboard.py's SOURCE_FILES")

# Observations at a stage the packer refuses: two DIFFERENT keys past
# target.TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE (62 bytes), where p4c's
# extra charge reaches +2 (dsp41/dsp42: the (84, 84) probe at 63/64 bytes, 7
# blocks instead of 5). They are reported as REFUSED, not scored: the gate is
# 0 under on every placement the model can actually emit. (59-62-byte rows are
# scored, with the crowded-stage +1.)
KNOWN_REFUSED_PLACEMENTS = frozenset({
    ("tcam_discount_scan", "dsp41"),
    ("tcam_discount_scan", "dsp42"),
})


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


def test_blocks_charged_never_under_predicts_a_placement_the_model_emits():
    """The gate the whole script exists for: on every observation whose
    placement the packer would also produce, blocks_charged is never below
    what p4c charged."""
    rows = scoreboard.score_all()

    unders = scoreboard.under_predictions(rows, "diff_charged")

    assert unders == [], [(r["source"], r["identifier"]) for r in unders]


def test_the_refused_placements_are_exactly_the_rows_past_the_cap():
    """Rows are excluded from the gate only because the packer refuses their
    stage: every one sits above 62 combined bytes."""
    rows = scoreboard.score_all()

    refused = scoreboard.refused_placements(rows)
    by_source = {}
    for r in refused:
        by_source.setdefault(r["source"], set()).add(r["identifier"])

    assert by_source.pop("tcam_discount_scan") == {i for _, i in KNOWN_REFUSED_PLACEMENTS}
    assert by_source.pop("tcam_mixed_key_cap_sweep") == {
        "%s_t%d" % (tag, t) for tag in ("solid22", "rag11", "rag14", "four26", "hik16")
        for t in (63, 64)}
    # Two solid 32-byte keys sharing one stage at 64 bytes; p4c charged
    # nothing extra, the packer refuses the stage (an over-prediction).
    assert by_source.pop("tcam_stretch_sweep") == {
        "a32x2_b32x2/tern_%s%d" % (tag, i) for tag in "ab" for i in (0, 1)}
    assert by_source == {}


def test_no_saturated_key_is_charged_outside_a_crowded_stage():
    """The per-key saturation margin is retired: a table is charged beyond
    its refined price only when its stage is crowded (> 58 bytes)."""
    rows = scoreboard.score_all()

    extra = [r for r in rows if r["blocks_charged"] > r["blocks_refined"]]

    assert extra and all(r["crowded"] and r["not_first"] for r in extra)


def test_the_crowded_stage_margin_is_applied_to_59_to_62_byte_rows():
    """rag14 = (54, 56), not saturated: 3 blocks alone, 4 at 59-62 bytes.
    Charged 4 there only because the stage is crowded."""
    rows = {r["identifier"]: r for r in scoreboard.score_all()
            if r["source"] == "tcam_mixed_key_cap_sweep"}
    for t in (59, 60, 61, 62):
        row = rows["rag14_t%d" % t]
        assert not row["saturated"]
        assert (row["blocks_charged"], row["observed_blocks"]) == (4, 4)


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
