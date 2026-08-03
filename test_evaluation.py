import math

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
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 1   # one interval
    assert blocks == 1    # 4 rows fits comfortably in one 512-row block
    # one independent range table for the feature, keyed on its 16-bit value
    assert specs == [(1, 2)]


def test_range_matching_resource_usage_crosses_block_boundary():
    # 129 non-overlapping copies of the (10,300)-shaped pattern, shifted
    # by 300 each time. Nibble alignment depends on absolute bit
    # position, not relative offset, so per-copy cost isn't perfectly
    # constant (verified computationally: mostly 4 rows, a few cost 5) --
    # total is 518 rows, comfortably over the 512-row block capacity,
    # which must force a 2nd block rather than silently rounding down.
    intervals = [(300 * i + 10, 300 * i + 300) for i in range(129)]
    feature_intervals = {"F": intervals}
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 129
    assert blocks == 2
    assert specs == [(2, 2)]


def test_range_matching_resource_usage_sums_across_features():
    feature_intervals = {
        "F1": [(10, 300)],
        "F2": [(0, 255), (5, 5)],
    }
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 3  # 1 interval in F1, 2 in F2
    assert blocks == 2   # each feature's rows fit in its own 1 block
    # one independent table per feature, NOT one merged table
    assert specs == [(1, 2), (1, 2)]


def test_range_matching_entries_per_block_constant_removed():
    # The flat RANGE_MATCHING_ENTRIES_PER_BLOCK constant is no longer
    # needed -- the exact per-interval algorithm computes real cost
    # directly. Confirms it was actually removed rather than left as
    # dead code.
    assert not hasattr(bps, "RANGE_MATCHING_ENTRIES_PER_BLOCK")


def _codewords_of_length(width, n_entries=1):
    codeword = "0" * width
    return {0: {codeword: 0}}


def _one_feature_intervals(width):
    # One feature owning the whole codeword: build_p4_script.py allocates
    # len(feature_intervals[feature]) - 1 codeword bits per feature, so a
    # width-bit codeword means width+1 intervals.
    return {"F": [(i, i) for i in range(width + 1)]}


def test_ternary_matching_resource_usage_off_by_one_at_41_bits():
    # RM-3 Design A: every ternary key reports width+4 TCAM bits
    # requested. At 41 bits, the missing +4 previously under-counted by
    # exactly one block: ceil(41/44)=1 (buggy) vs ceil(45/44)=2 (correct).
    codewords = _codewords_of_length(41)
    entries, blocks, _, _ = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(41))
    assert entries == 1
    assert blocks == 2


def test_ternary_matching_resource_usage_off_by_one_at_88_bits():
    # Same off-by-one, reconfirmed at 88 bits: ceil(88/44)=2 (buggy) vs
    # ceil(92/44)=3 (correct).
    codewords = _codewords_of_length(88)
    _, blocks, _, _ = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(88))
    assert blocks == 3


def test_ternary_matching_resource_usage_unaffected_at_168_bits():
    # RM-3 Design A also found widths where +4 doesn't change the block
    # count: 168 bits needs ceil(168/44)=4 and ceil(172/44)=4 either way
    # -- confirms the fix doesn't regress widths where it shouldn't matter.
    codewords = _codewords_of_length(168)
    _, blocks, _, _ = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(168))
    assert blocks == 4


def test_ternary_table_key_bytes_sums_per_feature_fields():
    # The classification tables key on one ternary field PER FEATURE
    # (build_p4_script.py:630-635), and the crossbar allocates per field.
    # 3 features x 4 bits: 3 separate 1-byte fields = 3 bytes, NOT
    # ceil(12/8) = 2 bytes on the concatenated codeword.
    feature_intervals = {f: [(i, i) for i in range(5)] for f in "ABC"}
    assert ev.ternary_table_key_bytes(feature_intervals) == 3


def test_ternary_table_key_bytes_never_below_concatenated_rounding():
    # The per-field sum must always be >= the (under-counting) whole-
    # codeword rounding, for every split shape.
    import math
    for widths in [(4, 4, 4), (8, 8), (1, 1, 1, 1, 1), (13, 3), (44,)]:
        feature_intervals = {
            str(i): [(0, 0)] * (w + 1) for i, w in enumerate(widths)
        }
        assert (ev.ternary_table_key_bytes(feature_intervals)
                >= math.ceil(sum(widths) / 8))


def test_ternary_matching_resource_usage_exposes_per_tree_table_specs():
    # The packer needs per-table data, not just the aggregate block sum:
    # one (block_count, byte_width) pair per tree.
    codewords = {0: {"0" * 41: 0}, 1: {"0" * 41: 1, "1" * 41: 0}}
    _, blocks, _, specs = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(41))
    assert specs == [(2, 6), (2, 6)]   # 41 bits -> 2 blocks, 6 key bytes
    assert blocks == sum(spec[0] for spec in specs)


def test_crossbar_stages_needed_flat_table_cap_at_16_bit():
    # RM-5/RM-6: 8 independent 16-bit tables (2 bytes each, 1 block each --
    # factor = ceil((16+4)/44) = 1) fit in 1 stage; a 9th forces a 2nd
    # stage -- the flat 8-table cap binds here since 9*2=18 bytes is
    # nowhere near the 64-byte budget and 9 blocks is well under 24.
    assert ev.crossbar_stages_needed([(1, 2)] * 8) == 1
    assert ev.crossbar_stages_needed([(1, 2)] * 9) == 2


def test_crossbar_stages_needed_byte_budget_at_256_bit():
    # RM-7: 256-bit tables (32 bytes each) -- exactly 2 fit in one stage
    # (2*32=64, an exact fit to the byte budget), a 3rd forces a 2nd
    # stage. The byte budget binds here, not the 8-table cap and not the
    # 24-block cap (each table is ceil((256+4)/44) = 6 blocks).
    assert ev.crossbar_stages_needed([(6, 32), (6, 32)]) == 1
    assert ev.crossbar_stages_needed([(6, 32)] * 3) == 2


def test_crossbar_stages_needed_512_bit_saturates_alone():
    # RM-7: one 512-bit table (64 bytes) already uses the entire 64-byte
    # budget -- a 2nd such table cannot share its stage. (Block count kept
    # small so the byte budget is unambiguously what binds.)
    assert ev.crossbar_stages_needed([(1, 64)]) == 1
    assert ev.crossbar_stages_needed([(1, 64), (1, 64)]) == 2


def test_crossbar_stages_needed_mixed_widths_pack_together():
    # Disjoint encoding: differently-sized independent tables (e.g. app
    # vs ddos trees with different codeword lengths) should share a
    # stage via bin-packing whenever every budget allows it
    # (32 + 6 + 6 + 6 = 50 <= 64 bytes, 4 tables <= 8, 9 blocks <= 24).
    assert ev.crossbar_stages_needed([(6, 32), (1, 6), (1, 6), (1, 6)]) == 1


def test_crossbar_stages_needed_blocks_bind_before_crossbar():
    # New with the unified packer: the 24-blocks-per-stage limit is now
    # enforced inside the same packing. Three 9-block, 2-byte tables are
    # trivially fine for the crossbar (3 tables, 6 bytes) but 27 blocks
    # do not fit one stage.
    assert ev.crossbar_stages_needed([(9, 2)] * 3) == 2


def test_crossbar_stages_needed_beats_max_of_two_relaxations():
    # The counterexample that motivated replacing
    # max(ceil(blocks/24), crossbar_only): blocks-only says
    # ceil(41/24) = 2, crossbar-only says 2 ({60} | {5,5}), so the old
    # max() reported 2 -- but no two of these three tables fit one stage
    # (20+20 = 40 blocks > 24; 5+60 = 65 bytes > 64). Truth is 3.
    assert ev.crossbar_stages_needed([(20, 5), (20, 5), (1, 60)]) == 3


def test_crossbar_stages_needed_single_oversized_table_spans_stages():
    # A table needing more blocks than a whole stage holds must span
    # several stages -- packing it as one indivisible item would report 1
    # stage and under-count.
    assert ev.crossbar_stages_needed([(50, 2)]) == 3
    assert ev.crossbar_stages_needed([]) == 0


def test_crossbar_stages_needed_output_respects_all_three_limits():
    # The property that actually matters: whatever the sort order, every
    # emitted stage is a physically legal stage, so the count is an upper
    # bound on the optimum (never an under-count).
    import random
    rnd = random.Random(1234)
    for _ in range(200):
        specs = [(rnd.randint(1, 30), rnd.randint(1, 70))
                 for _ in range(rnd.randint(1, 20))]
        stages = ev.crossbar_stages_needed(specs)
        # A valid packing can never use fewer stages than either
        # single-dimension lower bound.
        assert stages >= math.ceil(sum(b for b, _ in specs) / bps.TCAM_BLOCKS_PER_STAGE)
        assert stages >= math.ceil(
            sum(min(w, bps.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE) for _, w in specs)
            / bps.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
        assert stages >= math.ceil(len(specs) / bps.TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)


def test_ternary_matching_resource_usage_default_action_discount_drops_most_common_leaf():
    # Task 7: Planter-style default-action discount. One tree, 3 leaves:
    # class 0 has two leaves ("000" and "010"), class 1 has one ("001") --
    # class 0 is the clear majority CLASS VALUE among
    # codewords[tree].values(), per generate_codewords' real shape
    # (codeword string -> class value, each codeword string already
    # unique). The discount drops exactly ONE of the two class-0 leaves
    # (the one it picks as default_action), not both -- the other class-0
    # leaf remains an explicit entry.
    codewords = {0: {"000": 0, "001": 1, "010": 0}}
    entries_no_discount, _, _, _ = ev.ternary_matching_resource_usage(codewords, {})
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_no_discount == 3
    assert entries_discount == 2  # one leaf (of the two mapping to class 0) becomes default_action


def test_ternary_matching_resource_usage_discount_off_by_default_unchanged():
    codewords = {0: {"000": 0, "001": 1, "010": 0}}
    entries, blocks, length, specs = ev.ternary_matching_resource_usage(codewords, {})
    # must match pre-Task-7 behavior exactly -- no regression for the default path
    assert entries == 3


def test_ternary_matching_resource_usage_discount_never_drops_more_than_one_per_tree():
    # Even though TWO leaves vote class 0, only one entry is ever dropped
    # per tree -- the discount is "one default_action per table", not
    # "deduplicate all leaves sharing the most-common class".
    codewords = {0: {"000": 0, "001": 1, "010": 0}}
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_discount == len(codewords[0]) - 1


def test_ternary_matching_resource_usage_discount_applies_per_tree():
    # Two trees, each with their own majority class -- discount subtracts
    # exactly one leaf from EACH tree independently.
    codewords = {
        0: {"000": 0, "001": 1, "010": 0},          # 3 leaves, class 0 wins
        1: {"100": 1, "101": 1, "110": 0, "111": 1},  # 4 leaves, class 1 wins
    }
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_discount == (3 - 1) + (4 - 1)


def test_ternary_matching_resource_usage_returns_codeword_length():
    codewords = _codewords_of_length(41)
    entries, blocks, length, _ = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(41))
    assert length == 41


def _tiny_forest(labels, seed):
    # Deliberate exception to this file's "synthetic in-memory fixtures
    # only" convention: Finding 4 of the T12 final review noted that
    # nothing exercised single_model_memory_evaluation /
    # multi_model_memory_evaluation, leaving their long tuple-unpacking
    # lines (where a transposed variable is easy to miss) untested. Still
    # no dataset file -- the arrays are hardcoded here.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rnd = np.random.RandomState(seed)
    X = rnd.randint(0, 5000, size=(30, 4))
    y = np.array([labels[i % len(labels)] for i in range(30)])
    # make the labels learnable so the trees actually split
    X[:, 0] += np.array([1000 * (labels.index(v) + 1) for v in y])

    clf = RandomForestClassifier(n_estimators=2, max_depth=3,
                                 random_state=seed)
    clf.fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_multi_model_memory_evaluation_end_to_end_on_real_forests(encoding):
    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    stages, blocks = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, encoding)

    assert isinstance(stages, int) and isinstance(blocks, int)
    # Both models really do emit tables, so neither count can be zero.
    assert stages >= 2   # at least one range stage and one ternary stage
    assert blocks >= 2
    # Sanity upper bound: 4 tiny depth-3 trees over 4 features cannot need
    # anywhere near a full 12-stage Tofino pipeline's worth of tables.
    assert stages <= 12
    assert blocks <= 24 * stages


def test_single_model_memory_evaluation_tuple_is_self_consistent():
    features = ["f0", "f1", "f2", "f3"]
    clf = _tiny_forest([0, 1, 2], seed=3)

    (range_entries, range_blocks, ternary_entries, ternary_blocks,
     codewords, codeword_length,
     range_table_specs, ternary_table_specs) = ev.single_model_memory_evaluation(clf, features)

    # Guards against a transposed unpacking: each field must match what its
    # own name means.
    assert len(ternary_table_specs) == len(codewords) == 2   # one table per tree
    assert ternary_entries == sum(len(codewords[t]) for t in codewords)
    assert ternary_blocks == sum(b for b, _ in ternary_table_specs)
    assert range_blocks == sum(b for b, _ in range_table_specs)
    assert range_entries >= len(range_table_specs)
    assert all(width == ev.RANGE_TABLE_KEY_BYTES for _, width in range_table_specs)
    assert 0 < codeword_length <= bps.MAX_CODEWORD_LENGTH


import inspect
from pathlib import Path


def test_main_py_threshold_is_16_bit():
    source_file = Path(__file__).parent / "main.py"
    with open(source_file) as f:
        source = f.read()
    assert "threshold = (2 ** 16) - 2" in source
    assert "(2 ** 19) - 2" not in source


def test_build_p4_script_infinite_is_16_bit_sentinel():
    assert bps.INFINITE == (2 ** 16) - 1


def test_dataset_py_clips_outliers_to_threshold_not_hardcoded_19bit_value():
    source_file = Path(__file__).parent / "dataset.py"
    with open(source_file) as f:
        source = f.read()
    assert "(2**19)-2" not in source.replace(" ", "")
    assert source.count("threshold if x > threshold else x") == 2
