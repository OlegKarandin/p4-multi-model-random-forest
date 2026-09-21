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

# KNOWN, DOCUMENTED gap -- not silently swallowed. Both rows are the same
# (84, 84)-bit probe (22 crossbar bytes, blocks_refined = 5, not saturated)
# squeezed behind a 41/42-byte spacer that leaves it only 4 free crossbar
# groups instead of 5; p4c's real cost jumps to 7 blocks. This is F5, the
# "near-cap crossbar group budget" effect the 2026-09-20 rewrite design's own
# Sec 8 (E1) and Sec 13.2 explicitly say is NOT modelled by the Sec 13.2
# stage-sharing margin -- that margin is capped at exactly +1 block by
# design (Sec 13.2: "Bound: +1 block per affected table, exactly, never
# more"), and these two rows need +2 (5 -> 7). No reading of "not_first" can
# close a 2-block gap with a margin that only ever adds 1, so this is a
# genuine, structural residual of the model as specified, reproducing Sec 8's
# own recorded (spacer, groups-left, blocks) table point-for-point -- not a
# bug in how this script applies the margin. See task-3-report.md for the
# full account. If E1 (or a successor) ever models F5, this set should
# shrink; a new, DIFFERENT under-prediction appearing here is a regression
# and should not be added to this list without the same level of scrutiny.
KNOWN_UNDER_PREDICTIONS = frozenset({
    ("tcam_discount_scan", "dsp41"),
    ("tcam_discount_scan", "dsp42"),
})


def test_total_observation_count_is_308():
    rows = scoreboard.score_all()

    assert len(rows) == 308


def test_blocks_charged_under_predictions_are_exactly_the_known_f5_gap():
    """The gate the whole script exists for: blocks_charged must never
    under-predict, except for the one documented, structural gap (F5) the
    Sec 13.2 margin was never meant to cover. Any OTHER under-prediction --
    or the disappearance of these two without the set being updated -- fails
    this test rather than passing silently."""
    rows = scoreboard.score_all()

    unders = scoreboard.under_predictions(rows, "diff_charged")
    observed = frozenset((r["source"], r["identifier"]) for r in unders)

    assert observed == KNOWN_UNDER_PREDICTIONS, (
        "blocks_charged under-predictions changed shape: %r -- either a new "
        "regression (investigate before touching this test) or the F5 gap "
        "was modelled away (shrink KNOWN_UNDER_PREDICTIONS with a reason)"
        % (observed.symmetric_difference(KNOWN_UNDER_PREDICTIONS),))


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
