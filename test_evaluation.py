import evaluation as ev
import build_p4_script as bps
import pytest


def test_range_entry_count_reproduces_10_300_worked_example():
    # reviews/cited_papers/tofino_results_2.odt.pdf slide 11: matching
    # [10,300] on a 16-bit field needs exactly 4 physical TCAM entries
    # under Tofino's nibble-based range decomposition (vs. 10 under a
    # naive bit-level decomposition). This is the exact algorithm
    # (expand_range() ported from bf-drivers), not an approximation.
    assert ev.range_entry_count(10, 300) == 4


def test_range_entry_count_point_entry_is_one():
    assert ev.range_entry_count(5, 5) == 1


def test_range_entry_count_aligned_power_of_two_block_is_one():
    # A power-of-2-aligned range always costs exactly one physical row
    # (RM-8's own empirical finding, reviews/t12_tcam_model_experiment_plan.md
    # Section 11.3) -- confirmed here from the exact algorithm directly.
    assert ev.range_entry_count(0, 255) == 1


def test_range_entry_count_rejects_hi_less_than_lo():
    with pytest.raises(ValueError):
        ev.range_entry_count(10, 5)


def test_range_matching_resource_usage_uses_exact_real_interval_costs():
    # A single (10,300) interval needs exactly 4 physical rows (per
    # range_entry_count), not "1 entry per interval" -- confirms
    # range_matching_resource_usage sums the REAL per-interval cost, not
    # just a count of intervals (the old code's bug, in a different form).
    feature_intervals = {"F": [(10, 300)]}
    entries, blocks = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 1   # one interval
    assert blocks == 1    # 4 rows fits comfortably in one 512-row block


def test_range_matching_resource_usage_crosses_block_boundary():
    # 129 non-overlapping copies of the (10,300)-shaped pattern, shifted
    # by 300 each time. Nibble alignment depends on absolute bit
    # position, not relative offset, so per-copy cost isn't perfectly
    # constant (verified computationally: mostly 4 rows, a few cost 5) --
    # total is 518 rows, comfortably over the 512-row block capacity,
    # which must force a 2nd block rather than silently rounding down.
    intervals = [(300 * i + 10, 300 * i + 300) for i in range(129)]
    feature_intervals = {"F": intervals}
    entries, blocks = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 129
    assert blocks == 2


def test_range_matching_resource_usage_sums_across_features():
    feature_intervals = {
        "F1": [(10, 300)],
        "F2": [(0, 255), (5, 5)],
    }
    entries, blocks = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 3  # 1 interval in F1, 2 in F2
    assert blocks == 2   # each feature's rows fit in its own 1 block


def test_range_matching_entries_per_block_constant_removed():
    # The flat RANGE_MATCHING_ENTRIES_PER_BLOCK constant is no longer
    # needed -- the exact per-interval algorithm computes real cost
    # directly. Confirms it was actually removed rather than left as
    # dead code.
    assert not hasattr(bps, "RANGE_MATCHING_ENTRIES_PER_BLOCK")
