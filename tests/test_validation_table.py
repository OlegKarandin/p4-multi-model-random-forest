"""Aggregation and formatting on synthetic pairs -- no fixture, no CSV.

The 'never under-predicts' property is COMPUTED from the data, not asserted, so
these feed aggregate() deliberate under-predictions and check it says so."""
import pytest

from scripts import validation_table as vt


def _pairs(*triples):
    return [{"row_id": rid, "predicted": p, "real": r} for rid, p, r in triples]


def test_aggregate_counts_exact_matches_and_ignores_missing_ground_truth():
    stats = vt.aggregate(_pairs(("a", 8, 8), ("b", 9, 9), ("c", 14, None)))
    assert stats["n"] == 3
    assert stats["n_compared"] == 2
    assert stats["exact"] == 2
    assert stats["mean_error"] == 0.0
    assert stats["max_error"] == 0


def test_aggregate_reports_direction_separately():
    # over = predicted above real (the safe direction); under = predicted below.
    stats = vt.aggregate(_pairs(("a", 14, 13), ("b", 8, 8), ("c", 10, 12)))
    assert stats["over"] == 1
    assert stats["under"] == 1
    assert stats["exact"] == 1
    assert stats["max_error"] == 2


def test_aggregate_does_not_let_opposite_errors_cancel():
    # A signed mean would report 0.0 here and hide both misses.
    stats = vt.aggregate(_pairs(("a", 12, 10), ("b", 8, 10)))
    assert stats["mean_error"] == 2.0


def test_aggregate_on_no_comparable_rows_does_not_divide_by_zero():
    stats = vt.aggregate(_pairs(("a", 5, None)))
    assert stats["n_compared"] == 0
    assert stats["mean_error"] is None
    assert stats["max_error"] is None


def test_format_table_marks_under_predictions_visibly():
    text = vt.format_table(_pairs(("under_row", 5, 9)), "stage_depth")
    assert "under_row" in text
    assert "UNDER" in text


def test_format_table_renders_missing_ground_truth_without_crashing():
    text = vt.format_table(_pairs(("blank_row", 14, None)), "stage_depth")
    assert "blank_row" in text


def test_heldout_pairs_replay_every_extra_row_against_p4c_ground_truth():
    # The 8 held-out compiles (results/compiler_calibration_extra/), replayed
    # end to end by the model. Ground truth is p4c's own *_real columns; the
    # CSV's stale blocks/stage_depth columns must never be read as truth.
    import os
    if not os.path.isdir(os.path.join(vt.ROOT, "results",
                                      "compiler_calibration_extra", "compiles")):
        pytest.skip("needs results/compiler_calibration_extra (gitignored)")
    stage_pairs, blocks_pairs = vt.heldout_pairs()
    assert len(stage_pairs) == len(blocks_pairs) == 8
    by_row = {p["row_id"]: p for p in blocks_pairs}
    assert by_row["joint_high_sd9"]["real"] == 42       # tcam_real, not blocks=38
    assert by_row["independent_high_sd9"]["real"] is None  # never allocated
    assert vt.aggregate(stage_pairs)["under"] == 0
    assert vt.aggregate(blocks_pairs)["under"] == 0


def test_pinned_pairs_are_informational_since_the_code_pins():
    # The pinned archive predates the code_* pins, so with the 62-byte net
    # retired M150_k7_s11 reads 11 against its unpinned 12 -- the layout
    # artifact the pins remove (its pragmas compile is 11).
    import os
    if not (os.path.isfile(vt.PINNED_CSV)
            and os.path.isdir(os.path.join(vt.PINNED_ROOT, "compiles"))):
        pytest.skip("needs results/compiler_calibration_pinned (gitignored)")
    stage_pairs, blocks_pairs = vt.pinned_pairs()
    stages = vt.aggregate(stage_pairs)
    assert (stages["n_compared"], stages["exact"]) == (43, 41)
    misses = {p["row_id"]: p["predicted"] - p["real"] for p in stage_pairs
              if p["predicted"] != p["real"]}
    assert misses == {"independent_high_sd12": -1,
                      "margin_independent_M150_k7_s11": -1}
    assert set(misses) == set(vt.KNOWN_PINNED_MISSES)
    blocks = vt.aggregate(blocks_pairs)
    assert (blocks["n_compared"], blocks["exact"], blocks["under"],
            blocks["over"]) == (38, 38, 0, 0)


def test_pragmas_pairs_are_the_primary_gate():
    # results/compiler_calibration_pragmas_2026_09_29/ (spec 2026-09-29 Sec 5):
    # the 43 pinned + 30 held-out designs recompiled with the generator's
    # code_* pins. Every FEASIBLE design (<= 12 stages, 60 of 73) is exact on
    # stage_depth; the three misses are all infeasible under both model and
    # p4c. Blocks 60/60 (13 designs never got a TCAM allocation).
    import os
    if not (os.path.isfile(vt.PRAGMAS_CSV)
            and os.path.isdir(os.path.join(vt.PRAGMAS_ROOT, "compiles"))):
        pytest.skip("needs results/compiler_calibration_pragmas_2026_09_29 (gitignored)")
    stage_pairs, blocks_pairs = vt.pragmas_pairs()
    stages = vt.aggregate(stage_pairs)
    blocks = vt.aggregate(blocks_pairs)
    assert (stages["n_compared"], stages["exact"]) == (73, 70)
    misses = {p["row_id"]: p["predicted"] - p["real"] for p in stage_pairs
              if p["predicted"] != p["real"]}
    assert misses == {"independent_high_sd12": -1,
                      "heldout_independent_M150_k15_s16": 1,
                      "heldout_independent_M250_k14_s12": -1}
    assert set(misses) == set(vt.KNOWN_PRAGMAS_MISSES)
    feasible = [p for p in stage_pairs if p["real"] <= 12]
    assert len(feasible) == 60
    assert all(p["predicted"] == p["real"] for p in feasible)
    assert (blocks["n_compared"], blocks["exact"], blocks["under"],
            blocks["over"]) == (60, 60, 0, 0)
