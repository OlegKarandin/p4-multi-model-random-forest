import collections
import math
import dataclasses
import pathlib

from src.p4gen import evaluation as ev
from src.p4gen import build_p4_script as bps
from src.p4model.ranges import range_entry_count
import pytest


def test_accuracy_metrics_rejects_unknown_task():
    # Before this guard, an unrecognized `task` fell through the if/elif with
    # `lab` never assigned, and blew up with an UnboundLocalError deep inside
    # f1_score(...) instead of a clear error at the call site.
    with pytest.raises(ValueError):
        ev.accuracy_metrics([0, 1], [0, 1], 'not-a-task')


def test_accuracy_metrics_app_and_ddos_still_work():
    accuracy, f1score = ev.accuracy_metrics([0, 1, 2], [0, 1, 2], 'app')
    assert accuracy == 1.0
    assert f1score == 1.0

    accuracy, f1score = ev.accuracy_metrics([-1, 1], [-1, 1], 'ddos')
    assert accuracy == 1.0
    assert f1score == 1.0


# range_entry_count's own worked-example tests, and the identity check
# proving evaluation.py shares (not duplicates) the implementation, now
# live in tests/test_ranges.py alongside src/p4model/ranges.py.


def test_range_matching_resource_usage_uses_exact_real_interval_costs():
    # A single (10,300) interval needs exactly 4 physical rows (per
    # range_entry_count), not "1 entry per interval" -- confirms
    # range_matching_resource_usage sums the REAL per-interval cost, not
    # just a count of intervals (the old code's bug, in a different form).
    # A1 (Task 4): entries is the expanded row count, so it's 4 here, not 1.
    feature_intervals = {"F": [(10, 300)]}
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 4   # 4 physical rows, not 1 interval
    assert blocks == 1    # 4 rows fits comfortably in one 512-row block
    # one independent range table for the feature, keyed on its 16-bit value
    assert specs == [(1, 2)]


def test_expanded_rows_no_longer_drive_the_range_block_count():
    # SEMANTICS CHANGED 2026-09-06 (Sec 7 "Mechanism E"): this fixture used
    # to assert blocks == 2, on the theory that 518 expanded rows overflow a
    # 512-row block. That conflated two quantities. p4c allocates blocks at
    # COMPILE time from the declared entry count -- 129 intervals price at
    # 321 rows, so it commits 1 block -- and never sees the expansion at all.
    # The 518 rows are real, but they are an INSERTION-time fact, and this is
    # precisely the case where the two disagree: see
    # test_range_deployment_overflow_flags_expansion_beyond_the_allocation,
    # which keeps this fixture honest by catching the same 518-vs-512 problem
    # in the constraint that actually owns it.
    intervals = [(300 * i + 10, 300 * i + 300) for i in range(129)]
    feature_intervals = {"F": intervals}
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    assert entries == 518   # still the exact expanded row count
    assert blocks == 1      # what the compiler commits
    assert specs == [(1, 2)]


def test_range_matching_resource_usage_sums_across_features():
    feature_intervals = {
        "F1": [(10, 300)],
        "F2": [(0, 255), (5, 5)],
    }
    entries, blocks, specs = ev.range_matching_resource_usage(feature_intervals)
    # A1 (Task 4): expanded rows, not intervals -- F1's (10,300) costs 4
    # rows, F2's (0,255) and (5,5) each cost 1 row apiece: 4 + 1 + 1 = 6.
    assert entries == 6
    assert blocks == 2   # each feature's rows fit in its own 1 block
    # one independent table per feature, NOT one merged table
    assert specs == [(1, 2), (1, 2)]


def test_range_entries_counts_expanded_tcam_rows_not_intervals():
    """A1: deliverable 8 compares entries against blocks, and blocks are
    ceil(rows/512). Counting intervals would compare two unrelated
    quantities. A single [1, 20] interval crosses a 4-bit nibble boundary
    (task brief's own [1, 14] example turns out to be degenerate: it fits
    entirely inside one nibble and expands to exactly 1 row, the same as
    the interval count, so it can't distinguish the two definitions --
    [1, 20] genuinely expands to 2 physical TCAM rows)."""
    intervals = {'f': [(1, 20)]}
    entries, blocks, specs = ev.range_matching_resource_usage(intervals)
    assert entries == sum(range_entry_count(lo, hi, ev.nibble_widths_for(16))
                          for lo, hi in intervals['f'])
    assert entries > len(intervals['f'])


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


def test_codeword_fields_to_bytes_sums_per_feature_fields():
    # The classification tables key on one ternary field PER FEATURE
    # (build_p4_script.py:630-635), and the crossbar allocates per field.
    # 3 features x 4 bits: 3 separate 1-byte fields = 3 bytes, NOT
    # ceil(12/8) = 2 bytes on the concatenated codeword.
    feature_intervals = {f: [(i, i) for i in range(5)] for f in "ABC"}
    assert ev.codeword_fields_to_bytes(feature_intervals) == 3


def test_codeword_fields_to_bytes_never_below_concatenated_rounding():
    # The per-field sum must always be >= the (under-counting) whole-
    # codeword rounding, for every split shape.
    import math
    for widths in [(4, 4, 4), (8, 8), (1, 1, 1, 1, 1), (13, 3), (44,)]:
        feature_intervals = {
            str(i): [(0, 0)] * (w + 1) for i, w in enumerate(widths)
        }
        assert (ev.codeword_fields_to_bytes(feature_intervals)
                >= math.ceil(sum(widths) / 8))


def test_ternary_matching_resource_usage_exposes_per_tree_table_specs():
    # The packer needs per-table data, not just the aggregate block sum:
    # one (block_count, byte_width) pair per tree.
    codewords = {0: {"0" * 41: 0}, 1: {"0" * 41: 1, "1" * 41: 0}}
    _, blocks, _, specs = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(41))
    assert specs == [(2, 6), (2, 6)]   # 41 bits -> 2 blocks, 6 key bytes
    assert blocks == sum(spec[0] for spec in specs)


def test_codeword_bytes_to_blocks_charges_44_bits_per_5_and_a_half_bytes():
    # Ref 4.1 / Sec 7 "Mechanism D": one TCAM block is fed by ONE ternary
    # crossbar group, and a group delivers 5 private bytes + 1 midbyte
    # nibble = 44 bits = 5.5 bytes. Measured ladder from 144 real compiled
    # tables -- 33 bytes is the largest key that still fits 6 blocks, 34
    # needs 7.
    assert ev.codeword_bytes_to_blocks(5) == 1
    assert ev.codeword_bytes_to_blocks(6) == 2
    assert ev.codeword_bytes_to_blocks(33) == 6
    assert ev.codeword_bytes_to_blocks(34) == 7


def test_ternary_blocks_charge_byte_rounded_key_fields_not_raw_bits():
    # Sec 7 "Mechanism D". Six features of 5 codeword bits each: 30 bits
    # total, so codeword_bits_to_blocks says ceil((30+4)/44) = 1 block. But
    # the crossbar allocates per FIELD, byte-rounded, so the key really costs
    # 6 bytes = 48 bits and needs 2 blocks. Mirrors the real
    # independent_high_sd7 (10 fields, 29 bits, 6 crossbar bytes, 2 committed
    # blocks).
    feature_intervals = {f: [(i, i) for i in range(6)] for f in "ABCDEF"}
    codewords = {0: {"0" * 30: 0}}
    _, blocks, codeword_length, specs = ev.ternary_matching_resource_usage(
        codewords, feature_intervals)
    assert codeword_length == 30
    assert ev.codeword_bits_to_blocks(codeword_length) == 1     # what the old model charged
    assert blocks == 2
    assert specs == [(2, 6)]


def test_range_blocks_come_from_the_compilers_compile_time_sizing():
    # Sec 4.2 / Sec 7 "Mechanism E". p4c sizes a range table at COMPILE time
    # from the declared `size` (= the interval count, build_p4_script.py:1611),
    # pricing a quarter of the entries at the worst case min(8, 2*nibbles-1)
    # = 7 rows for a 16-bit key and the rest at 1. That puts the per-block
    # capacity at exactly 206 declared intervals, the figure Sec 4.2 measured
    # independently. Point intervals cost 1 physical row each, so the old
    # rows/512 model would say 1 block for both of these.
    fits = {"F": [(i, i) for i in range(206)]}
    over = {"F": [(i, i) for i in range(207)]}
    assert ev.range_matching_resource_usage(fits)[1] == 1
    assert ev.range_matching_resource_usage(over)[1] == 2


def test_range_deployment_overflow_flags_expansion_beyond_the_allocation():
    # The second, independent constraint: the compiler's block allocation is
    # fixed in the binary, so real control-plane insertion must fit INSIDE it
    # (Sec 4.2 -- the control plane gets "[Not enough space]", it never grows
    # the table). 129 intervals of this shape are priced at 1 block by the
    # compiler, but expand to 518 physical rows at insertion time, 6 over the
    # 512 that block holds.
    intervals = [(300 * i + 10, 300 * i + 300) for i in range(129)]
    entries, blocks, _ = ev.range_matching_resource_usage({"F": intervals})
    assert (entries, blocks) == (518, 1)
    assert ev.range_deployment_overflow({"F": intervals}) == {"F": (518, 512)}


def test_range_deployment_overflow_is_empty_when_the_real_rows_fit():
    # The normal case, and why this is a guard rather than a cost term.
    feature_intervals = {"F1": [(10, 300)], "F2": [(0, 255), (5, 5)]}
    assert ev.range_deployment_overflow(feature_intervals) == {}


def test_stage_shards_rejects_a_key_wider_than_one_stage_crossbar():
    # F3: splitting a table's ROWS across stages is real; splitting its KEY
    # is not -- a stage's crossbar cannot deliver more than
    # TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes, and the compiler rejects
    # such a table outright rather than spreading it over stages.
    with pytest.raises(ev.CrossbarKeyTooWide) as excinfo:
        ev._stage_shards(1, 100)
    assert excinfo.value.args[1] == 100


def test_crossbar_stages_needed_propagates_key_too_wide():
    # The packer must not silently price an impossible table as 2 stages --
    # it has to propagate the same raise _stage_shards produces.
    with pytest.raises(ev.CrossbarKeyTooWide) as excinfo:
        ev.crossbar_stages_needed([(1, 100)])
    assert excinfo.value.args[1] == 100


def test_ternary_matching_resource_usage_rejects_many_narrow_features():
    # Reachable shape: the crossbar allocates PER FIELD, so 100 features x 2
    # codeword bits each = 100 crossbar bytes, well under the separate
    # MAX_CODEWORD_LENGTH (512-bit) guard (this codeword is only 200 bits) --
    # so the codeword-length guard alone would not have caught this table.
    feature_intervals = {"F{}".format(i): [(0, 0), (0, 0), (0, 0)]
                          for i in range(100)}  # 3 intervals -> 2 bits each
    codewords = {0: {"0" * 200: 0}}
    with pytest.raises(ev.CrossbarKeyTooWide):
        ev.ternary_matching_resource_usage(codewords, feature_intervals)


def test_crossbar_stages_needed_flat_table_cap_at_16_bit():
    # RM-5/RM-6: 8 independent 16-bit tables (2 bytes each, 1 block each --
    # factor = ceil((16+4)/44) = 1) fit in 1 stage; a 9th forces a 2nd
    # stage -- the flat 8-table cap binds here since 9*2=18 bytes is
    # nowhere near the 64-byte budget and 9 blocks is well under 24.
    assert ev.crossbar_stages_needed([(1, 2)] * 8).occupied == 1
    assert ev.crossbar_stages_needed([(1, 2)] * 9).occupied == 2


def test_crossbar_stages_needed_byte_budget_at_256_bit():
    # RM-7: 256-bit tables (32 bytes each) -- exactly 2 fit in one stage
    # (2*32=64, an exact fit to the byte budget), a 3rd forces a 2nd
    # stage. The byte budget binds here, not the 8-table cap and not the
    # 24-block cap (each table is ceil((256+4)/44) = 6 blocks).
    assert ev.crossbar_stages_needed([(6, 32), (6, 32)]).occupied == 1
    assert ev.crossbar_stages_needed([(6, 32)] * 3).occupied == 2


def test_crossbar_stages_needed_512_bit_saturates_alone():
    # RM-7: one 512-bit table (64 bytes) already uses the entire 64-byte
    # budget -- a 2nd such table cannot share its stage. (Block count kept
    # small so the byte budget is unambiguously what binds.)
    assert ev.crossbar_stages_needed([(1, 64)]).occupied == 1
    assert ev.crossbar_stages_needed([(1, 64), (1, 64)]).occupied == 2


def test_crossbar_stages_needed_mixed_widths_pack_together():
    # Disjoint encoding: differently-sized independent tables (e.g. app
    # vs ddos trees with different codeword lengths) should share a
    # stage via bin-packing whenever every budget allows it
    # (32 + 6 + 6 + 6 = 50 <= 64 bytes, 4 tables <= 8, 9 blocks <= 24).
    assert ev.crossbar_stages_needed([(6, 32), (1, 6), (1, 6), (1, 6)]).occupied == 1


def test_crossbar_stages_needed_blocks_bind_before_crossbar():
    # New with the unified packer: the 24-blocks-per-stage limit is now
    # enforced inside the same packing. Three 9-block, 2-byte tables are
    # trivially fine for the crossbar (3 tables, 6 bytes) but 27 blocks
    # do not fit one stage.
    assert ev.crossbar_stages_needed([(9, 2)] * 3).occupied == 2


def test_crossbar_stages_needed_beats_max_of_two_relaxations():
    # The counterexample that motivated replacing
    # max(ceil(blocks/24), crossbar_only): blocks-only says
    # ceil(41/24) = 2, crossbar-only says 2 ({60} | {5,5}), so the old
    # max() reported 2 -- but no two of these three tables fit one stage
    # (20+20 = 40 blocks > 24; 5+60 = 65 bytes > 64). Truth is 3.
    assert ev.crossbar_stages_needed([(20, 5), (20, 5), (1, 60)]).occupied == 3


def test_crossbar_stages_needed_single_oversized_table_spans_stages():
    # A table needing more blocks than a whole stage holds must span
    # several stages -- packing it as one indivisible item would report 1
    # stage and under-count.
    assert ev.crossbar_stages_needed([(50, 2)]).occupied == 3
    assert ev.crossbar_stages_needed([]).occupied == 0


# ---------------------------------------------------------------------------
# Mechanism C: a stage's 24 TCAM blocks are 2 columns of 12 rows, and a table
# needing several blocks chains them down ONE column. Measured against real
# p4c over synthetic tables of 5..12 blocks -- scripts/tcam_column_sweep.py,
# reviews/p4_tofino_reference.md Sec 7.
# ---------------------------------------------------------------------------

def test_fits_two_columns_packs_by_width_not_by_total():
    # The whole content of Mechanism C, in the two measurements that decide
    # it. Three 8-block tables total exactly TCAM_BLOCKS_PER_STAGE and still
    # need two stages (8+8 = 16 overflows a 12-row column); four 6-block
    # tables total the same 24 and fit one stage (6+6 | 6+6). A flat block cap
    # cannot tell these apart, which is why it under-counted.
    assert not ev.fits_two_columns([8, 8, 8])
    assert ev.fits_two_columns([6, 6, 6, 6])
    assert sum([8, 8, 8]) == sum([6, 6, 6, 6]) == bps.TCAM_BLOCKS_PER_STAGE


def test_fits_two_columns_at_the_discriminating_width():
    # w=7 is where the two rules disagree most cleanly: 3 tables are 21 of 24
    # blocks, comfortably inside a flat cap, and the compiler still needs two
    # stages. joint_low_sd10's real placement is exactly this -- 2 tables of 7
    # per stage, 14 of 24 blocks used.
    assert ev.fits_two_columns([7, 7])
    assert not ev.fits_two_columns([7, 7, 7])


def test_fits_two_columns_is_exact_not_greedy():
    # First-fit-decreasing puts 7 then 5 in one column and 6 in the other,
    # then fails on the last 6; the true packing is (7+5 | 6+6). A greedy test
    # would invent violations, and inventing one is how the column rule got
    # written off as refuted the first time.
    assert ev.fits_two_columns([7, 6, 6, 5])


def test_a_single_table_may_span_both_columns():
    # Measured, and it is the reason _stage_shards splits at the column height
    # rather than treating an over-wide table as unplaceable: single tables of
    # 14, 16 and even 24 blocks each compile into ONE stage. Chaining within a
    # column constrains which tables SHARE a stage; it does not confine a
    # table that needs more than a column to begin with.
    assert ev.crossbar_stages_needed([(14, 4)]).occupied == 1
    assert ev.crossbar_stages_needed([(16, 4)]).occupied == 1
    assert ev.crossbar_stages_needed([(24, 4)]).occupied == 1


def test_stage_shards_splits_at_the_column_height():
    # Not cosmetic: a 24-block table is one shard the column packing can never
    # place, so without this the eager packer would advance its stage index
    # forever looking for room. Splitting at TCAM_ROWS_PER_STAGE makes every
    # shard column-sized by construction.
    assert all(blocks <= bps.TCAM_ROWS_PER_STAGE
               for blocks, _ in ev._stage_shards(24, 4))
    assert ev._stage_shards(11, 4) == [(11, 4)]      # unchanged below a column


def test_crossbar_stages_needed_enforces_the_column_geometry():
    # The packer, not just the predicate: three 8-block tables that a flat
    # 24-block cap would have put in one stage now correctly take two.
    assert ev.crossbar_stages_needed([(8, 4)] * 3).occupied == 2
    assert ev.crossbar_stages_needed([(6, 4)] * 4).occupied == 1
    assert ev.crossbar_stages_needed([(7, 4)] * 3).occupied == 2


def test_column_geometry_never_reports_fewer_stages_than_the_flat_cap_did():
    # The direction that matters for a cost model that must not under-count:
    # the column constraint is strictly tighter than the flat block cap, so it
    # can only ever move a design's stage count UP.
    import random
    rnd = random.Random(99)
    for _ in range(200):
        specs = [(rnd.randint(1, 24), rnd.randint(1, 8))
                 for _ in range(rnd.randint(1, 12))]
        stages = ev.crossbar_stages_needed(specs).occupied
        assert stages >= math.ceil(sum(b for b, _ in specs)
                                   / bps.TCAM_BLOCKS_PER_STAGE)


def test_crossbar_stages_needed_output_respects_all_three_limits():
    # The property that actually matters: whatever the sort order, every
    # emitted stage is a physically legal stage, so the count is an upper
    # bound on the optimum (never an under-count).
    #
    # Byte width is capped at TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE (Task 10,
    # F3): a table whose key is wider than one stage's crossbar budget is not
    # a splittable-across-stages shape at all -- crossbar_stages_needed now
    # raises CrossbarKeyTooWide for it instead of pricing a design the
    # compiler would reject outright, so this generator must not produce one.
    import random
    rnd = random.Random(1234)
    for _ in range(200):
        specs = [(rnd.randint(1, 30), rnd.randint(1, bps.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE))
                 for _ in range(rnd.randint(1, 20))]
        stages = ev.crossbar_stages_needed(specs).occupied
        # A valid packing can never use fewer stages than either
        # single-dimension lower bound.
        assert stages >= math.ceil(sum(b for b, _ in specs) / bps.TCAM_BLOCKS_PER_STAGE)
        assert stages >= math.ceil(
            sum(min(w, bps.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE) for _, w in specs)
            / bps.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
        assert stages >= math.ceil(len(specs) / bps.TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)


def test_ternary_matching_resource_usage_discount_drops_every_majority_leaf():
    # 4 leaves for one tree: 3 vote class 0, 1 votes class 1. Old (wrong)
    # behavior subtracted exactly 1 regardless; corrected behavior must
    # subtract all 3 majority-class leaves.
    codewords = {0: {"000": 0, "001": 1, "010": 0, "100": 0}}
    entries_no_discount, _, _, _ = ev.ternary_matching_resource_usage(codewords, {})
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_no_discount == 4
    assert entries_discount == 1  # only the single class-1 leaf remains explicit


def test_ternary_matching_resource_usage_discount_off_by_default_unchanged():
    codewords = {0: {"000": 0, "001": 1, "010": 0}}
    entries, blocks, length, specs = ev.ternary_matching_resource_usage(codewords, {})
    # must match pre-Task-7 behavior exactly -- no regression for the default path
    assert entries == 3


def test_ternary_matching_resource_usage_discount_drops_all_leaves_sharing_majority_class():
    # TWO leaves vote class 0 -- the corrected discount drops BOTH, not
    # just one; the discount is "every leaf sharing the majority class",
    # not "one default_action per table" (that was the bug this task fixed).
    codewords = {0: {"000": 0, "001": 1, "010": 0}}
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_discount == len(codewords[0]) - 2


def test_ternary_matching_resource_usage_discount_applies_per_tree():
    # Two trees, each with their own majority class -- discount subtracts
    # every leaf sharing that tree's majority class, independently per tree.
    codewords = {
        0: {"000": 0, "001": 1, "010": 0},          # 3 leaves, class 0 wins (2 leaves)
        1: {"100": 1, "101": 1, "110": 0, "111": 1},  # 4 leaves, class 1 wins (3 leaves)
    }
    entries_discount, _, _, _ = ev.ternary_matching_resource_usage(
        codewords, {}, use_default_action_discount=True)
    assert entries_discount == (3 - 2) + (4 - 3)


def test_ternary_matching_resource_usage_returns_codeword_length():
    codewords = _codewords_of_length(41)
    entries, blocks, length, _ = ev.ternary_matching_resource_usage(
        codewords, _one_feature_intervals(41))
    assert length == 41


def _tiny_forest(labels, seed):
    # Deliberate exception to this file's "synthetic in-memory fixtures
    # only" convention: Finding 4 of the T12 final review noted that
    # nothing exercised single_model_memory_evaluation /
    # multi_model_memory_evaluation, leaving single_model_memory_evaluation's
    # long tuple-unpacking lines (where a transposed variable is easy to miss)
    # untested -- multi_model_memory_evaluation now returns a ResourceUsage
    # object instead. Still no dataset file -- the arrays are hardcoded here.
    #
    # The golden tuples below (_PRE_TASK_SINGLE_APP etc.) were computed
    # against plain sklearn's tree-building. sklearnex.patch_sklearn() is a
    # process-global monkeypatch: once any other imported module (e.g.
    # train_model.py) has triggered it, scikit-learn's accelerated backend
    # produces a genuinely different (though equally valid) tree for the
    # same data/seed, which would make these exact-value assertions flaky
    # depending on unrelated test-collection order. Unpatch explicitly so
    # this fixture's tree shape stays deterministic regardless of global
    # process state.
    try:
        from sklearnex import unpatch_sklearn
        unpatch_sklearn()
    except ImportError:
        pass

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

    usage = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, encoding)

    assert isinstance(usage.stages, int) and isinstance(usage.blocks, int)
    # Both models really do emit tables, so neither count can be zero.
    assert usage.stages >= 2   # at least one range stage and one ternary stage
    assert usage.blocks >= 2
    # Sanity upper bound: 4 tiny depth-3 trees over 4 features cannot need
    # anywhere near a full 12-stage Tofino pipeline's worth of tables.
    assert usage.stages <= 12
    assert usage.blocks <= 24 * usage.stages
    # stage_depth (F5/F6) is a DIFFERENT quantity from stages -- pipeline
    # depth, not occupied-stage count -- so it can exceed stages, but never
    # be less than it (depth is at least the count of occupied indices).
    assert isinstance(usage.stage_depth, int)
    assert usage.stage_depth >= usage.stages
    assert usage.stage_depth <= 12


def test_multi_model_memory_evaluation_returns_a_frozen_resource_usage():
  """D1: a 5-tuple of same-typed ints containing three confusable stage
  quantities is the hazard F5/F6 documents. Every caller names its field."""
  FEATURES_APP = ["f0", "f1", "f2", "f3"]
  FEATURES_DDOS = ["f0", "f1", "f2", "f3"]
  usage = ev.multi_model_memory_evaluation(
      _tiny_forest([0, 1, 2], seed=0),
      _tiny_forest([-1, 1], seed=7),
      FEATURES_APP, FEATURES_DDOS, 'joint')
  assert isinstance(usage, ev.ResourceUsage)
  assert dataclasses.is_dataclass(usage)
  with pytest.raises(dataclasses.FrozenInstanceError):
    usage.blocks = 0
  # Deliberately absent: unlike StagePlan, no __int__ shim, so a caller
  # cannot silently use the whole object where a count is meant.
  assert not hasattr(usage, '__int__') or type(usage).__int__ is object.__int__
  assert usage.range_entries > 0 and usage.ternary_entries > 0


def test_multi_model_memory_evaluation_raises_on_unknown_encoding():
    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    with pytest.raises(ValueError, match="unknown encoding"):
        ev.multi_model_memory_evaluation(
            clf_app, clf_ddos, features, features, encoding='shared')


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


def test_single_model_memory_evaluation_raises_on_colliding_feature_names():
    # Final-review finding #1: Task 16 swapped this function's internal
    # get_feature_intervals(...) call for a direct tree_nodes_for(...) +
    # feature_intervals_from_nodes(...) pair, which skips
    # _reject_colliding_feature_names -- the guard lives inside
    # get_feature_intervals, not inside tree_nodes_for/
    # feature_intervals_from_nodes. Two differently-spelled feature names
    # that normalise to the same canonical key ('Flow.IAT.Max' and
    # 'Flow IAT Max' both -> 'flow_iat_max') used to silently merge their
    # intervals with no raise when reached THROUGH this function -- even
    # though get_feature_intervals itself already raised for the exact same
    # input. This test exercises the gap that let that regression through:
    # calling get_feature_intervals directly would not have caught it.
    features = ["Flow.IAT.Max", "Flow IAT Max", "f2", "f3"]
    clf = _tiny_forest([0, 1, 2], seed=3)

    with pytest.raises(ValueError, match="collide"):
        ev.single_model_memory_evaluation(clf, features)


def test_multi_model_memory_evaluation_joint_raises_on_cross_model_colliding_feature_names():
    # Same regression as test_single_model_memory_evaluation_raises_on_
    # colliding_feature_names above, but for multi_model_memory_evaluation's
    # 'joint' branch, which also swapped get_joint_feature_intervals(...)
    # for a direct merge_tree_nodes(tree_nodes_for(...), tree_nodes_for(...))
    # pair -- skipping the SAME guard. The collision here is CROSS-model:
    # neither features_app nor features_ddos collides within itself, only
    # their union does ('Flow.IAT.Max' in features_app, 'Flow IAT Max' in
    # features_ddos). get_joint_feature_intervals checks exactly this union
    # already; this test proves multi_model_memory_evaluation's joint branch
    # does too, reached through the real entry point, not the helper
    # directly. (The 'disjoint' branch already re-calls get_feature_intervals
    # for range_levels, so it incidentally still raises -- not tested here,
    # only the joint branch actually had the gap.)
    features_app = ["Flow.IAT.Max", "a1", "a2", "a3"]
    features_ddos = ["d0", "Flow IAT Max", "d2", "d3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    with pytest.raises(ValueError, match="collide"):
        ev.multi_model_memory_evaluation(
            clf_app, clf_ddos, features_app, features_ddos, encoding='joint')


from pathlib import Path


def test_build_p4_script_infinite_is_16_bit_sentinel():
    # The invariant guarded here is "feature values are clipped to a 16-bit
    # bound, never a 19-bit one". main.py's compare_independent_joint_mapping
    # spells its clipping threshold as `threshold = INFINITE`, deriving it
    # from this shared constant rather than a duplicated literal -- so
    # pinning INFINITE's own width is what actually protects that invariant.
    assert bps.INFINITE == (2 ** 16) - 1


def test_dataset_py_clips_outliers_to_threshold_not_hardcoded_19bit_value():
    source_file = Path(__file__).parent.parent / "src" / "training" / "dataset.py"
    with open(source_file) as f:
        source = f.read()
    assert "(2**19)-2" not in source.replace(" ", "")


def test_read_app_dataset_clips_outliers_to_threshold(monkeypatch):
    # Behavioural check for the live clipping path (dataset.py's
    # `df.clip(upper=threshold)` in read_app_dataset), replacing a
    # source-text grep that used to assert on `load_dataset`'s clipping --
    # dead code with no caller, since deleted.
    import pandas as pd
    from src.training import dataset as dataset_mod

    threshold = 100
    fake_df = pd.DataFrame({
        'ProtocolName': ['SKYPE', 'DROPBOX', 'GOOGLE'],
        'Feat1': [50, 150, 30],
    })
    monkeypatch.setattr(dataset_mod.pd, 'read_csv', lambda *a, **k: fake_df.copy())

    result = dataset_mod.read_app_dataset(['Feat1'], threshold)

    assert result['Feat1'].max() <= threshold
    # The 150 -> 100 outlier survived filtering (it's a real SKYPE/Dropbox/
    # Google row, not negative or NaN), so its presence at exactly
    # `threshold` proves clip() actually fired rather than the input simply
    # never exceeding the bound.
    assert (result['Feat1'] == threshold).any()


def test_read_DDOS_dataset_clips_outliers_to_threshold(monkeypatch):
    # Behavioural check for the live clipping path (dataset.py's
    # `df.clip(upper=threshold)` in read_DDOS_dataset). This function
    # hard-samples exactly 10000 rows per class, so the synthetic frame
    # needs >= 10000 rows in each of the two classes it keeps (BENIGN,
    # DoS Hulk). `idx` makes every row unique and is never itself clipped
    # (it stays far below `threshold`), so drop_duplicates() never removes
    # a row and the class-balance sampling always has enough to draw from.
    import numpy as np
    import pandas as pd
    from src.training import dataset as dataset_mod

    threshold = 10 ** 6
    n = 10000
    idx = np.arange(2 * n)
    feat1 = np.full(2 * n, 5.0)
    feat1[0] = threshold + 12345  # the one outlier clip() must catch
    labels = ['BENIGN'] * n + ['DoS Hulk'] * n
    fake_df = pd.DataFrame({'Feat1': feat1, 'idx': idx, 'Label': labels})
    monkeypatch.setattr(dataset_mod.pd, 'read_csv', lambda *a, **k: fake_df.copy())

    result = dataset_mod.read_DDOS_dataset(['Feat1', 'idx'], threshold)

    assert result['Feat1'].max() <= threshold
    assert (result['Feat1'] == threshold).any()


# ---------------------------------------------------------------------------
# Task 8: Planter RF_EB-style exact-match resource accounting
# ---------------------------------------------------------------------------
#
# Confirmed directly against build_p4_script.generate_codewords: each feature
# gets its own fixed-width, thermometer/unary-coded segment (width =
# len(feature_intervals[feature]) - 1), concatenated in feature_intervals
# iteration order (generate_codewords appends '*' at build_p4_script.py:
# 368/375, and narrows a leaf-tested feature's segment to a run of '0's
# followed by a run of '1's). A feature the leaf's path never tests gets its
# *entire* segment wildcarded, and that segment has exactly
# len(feature_intervals[feature]) reachable values (one per position of the
# 0/1 boundary) -- NOT 2**width independent bit choices. The two only
# coincide when width == 1 (a 2-interval feature), which is why a fixture
# using only width-1 features can't catch a width > 1 miscount; the fixtures
# below deliberately include a width-2 (3-interval) feature so an entirely
# wildcarded segment on it distinguishes the correct len(intervals)==3
# multiplier from the wrong 2**2==4 one.

def test_exact_match_resource_usage_enumerates_wildcarded_features():
    # Real generate_codewords-shaped fixture: feature A has 3 intervals (2
    # codeword bits, thermometer-coded: "11"/"01"/"00"), feature B has 2
    # intervals (1 codeword bit: "1"/"0"). Segments concatenate in
    # feature_intervals iteration order (A then B), so total codeword width
    # is 2 + 1 = 3 for every leaf, matching what generate_codewords itself
    # would emit.
    feature_intervals = {"A": [(0, 10), (11, 20), (21, 65535)], "B": [(0, 100), (101, 65535)]}
    codewords = {
        0: {
            "01*": 0,  # A tested (segment "01", one concrete value), B untested (all-'*')
            "**1": 1,  # A untested (all-'*', width 2 -> 3 reachable values), B tested
            "110": 1,  # both tested, no wildcards at all
        }
    }
    sram_entries, sram_blocks = ev.exact_match_resource_usage(codewords, feature_intervals)
    # "01*": A concrete (x1) * B all-wildcard width-1 segment -> len(B intervals)==2 => 2
    # "**1": A all-wildcard width-2 segment -> len(A intervals)==3 * B concrete (x1) => 3
    # "110": no wildcards -> 1
    # total = 2 + 3 + 1 == 6. The old `2 ** codeword.count('*')` formula
    # would instead give 2 + 4 + 1 == 7 for this same fixture (miscounting
    # the width-2 all-wildcard "**" segment as 2**2==4 instead of the
    # correct len(A intervals)==3), which is exactly the over-count this
    # fix corrects.
    assert sram_entries == 6


def test_exact_match_resource_usage_no_wildcards_matches_ternary_entry_count():
    # a tree where every leaf tests every feature has nothing to expand --
    # exact and ternary entry counts must be identical in this case
    feature_intervals = {"A": [(0, 10), (11, 65535)]}
    codewords = {0: {"0": 0, "1": 1}}
    sram_entries, _ = ev.exact_match_resource_usage(codewords, feature_intervals)
    ternary_entries, _, _, _ = ev.ternary_matching_resource_usage(codewords, feature_intervals)
    assert sram_entries == ternary_entries


def test_exact_match_resource_usage_multiple_wildcards_multiply():
    # Two independent, fully-wildcarded width-1 features (2 intervals each)
    # in the same codeword -> len(A intervals) * len(B intervals) = 2 * 2 =
    # 4 concrete entries for that one leaf, confirming the per-feature
    # factors multiply rather than a flat "+1 wildcard present" bump. Both
    # features are width 1 here, so len(intervals) == 2**width for each --
    # this test is about the multiplicative combination across features,
    # not the width > 1 miscount (that's covered above).
    feature_intervals = {"A": [(0, 10), (11, 65535)], "B": [(0, 50), (51, 65535)]}
    codewords = {0: {"**": 0}}
    sram_entries, sram_blocks = ev.exact_match_resource_usage(codewords, feature_intervals)
    assert sram_entries == 4
    assert sram_blocks is None  # SRAM per-block capacity not yet established (see evaluation.py)


def test_exact_match_resource_usage_sums_across_multiple_trees():
    feature_intervals = {"A": [(0, 10), (11, 65535)]}
    codewords = {
        0: {"0": 0, "1": 1},
        1: {"*": 0},
    }
    sram_entries, _ = ev.exact_match_resource_usage(codewords, feature_intervals)
    assert sram_entries == 1 + 1 + 2  # tree 0: 2 concrete leaves; tree 1: 1 leaf x len(A intervals)==2


# ---------------------------------------------------------------------------
# Follow-up (post-plan): use_default_action_discount threaded through the
# TCAM-entry estimators.
#
# `ternary_matching_resource_usage` has supported the flag since Task 1, but
# neither `single_model_memory_evaluation` nor `multi_model_memory_evaluation`
# ever passed it through -- so no caller of the estimator API could actually
# obtain discounted numbers. These tests pin both the new opt-in behavior and
# the unchanged default.
# ---------------------------------------------------------------------------

# Pre-change values, recorded by running the estimators on these exact
# fixtures BEFORE this task's edits. Hardcoded on purpose: comparing the new
# default path against a derived expression would only prove self-consistency,
# not that nothing moved.
# range_entries' leading value changed from 13 to 52 under Task 4 (A1):
# range_matching_resource_usage's range_entries now counts EXPANDED TCAM
# rows (what range_blocks quantizes), not raw interval counts. This
# fixture's 4 features (f0..f3) have 3, 3, 3, 4 intervals respectively
# (13 total, the old value); their expanded row costs are 12, 12, 12, 16
# (52 total, hand-verified via range_entry_count against feature_intervals
# printed from this exact fixture). Only this first value moved -- nothing
# else in this tuple depends on range_entries.
_PRE_TASK_SINGLE_APP = (52, 4, 11, 2, 9)          # range_entries, range_blocks, ternary_entries, ternary_blocks, codeword_length
_PRE_TASK_SINGLE_APP_RANGE_SPECS = [(1, 2)] * 4
_PRE_TASK_SINGLE_APP_TERNARY_SPECS = [(1, 4), (1, 4)]
# stage_depth moved 3 -> 6 on both when the readiness origin was corrected to
# FLOW_HASH_LEVEL = 3 and stage_depth started counting VOTE_EPILOGUE_STAGES
# (2026-09-05, calibrated against 19 real compiles). `stages` and `blocks` are
# unchanged, which is the point of keeping this pin: the crossbar key-sharing
# fix landed in the same change and must not have moved either of them on a
# fixture whose tables all key on their own fields.
_PRE_TASK_MULTI_JOINT = (2, 8, 6)                  # stages, blocks, stage_depth
# disjoint's blocks went 11 -> 13 on 2026-09-06, when StagePlan.blocks began
# reporting the charged total rather than the naive per-table sum, and back to
# 11 on 2026-09-07, when the charge itself was replaced. The old rule charged
# +1 to any "ragged" key that landed on an ODD crossbar group offset; measured
# against the 100 classification tables of the 19 archived compiles it fired on
# 5 stages where p4c charged nothing, and on the one stage where p4c DID charge
# it picked the wrong table. tables.version_block_penalty replaces it: the
# block is spent only when the mandatory 2-bit --version-- field has no midbyte
# nibble left to live in. This fixture's two keys both keep one, so 11 -- the
# naive sum -- is now the right answer, and it is what the pre-2026-09-06 model
# reported too. See tests/test_version_block.py for the rule's measurement set.
# `stages` and `stage_depth` were unaffected throughout: the charge moves a
# total, not a placement.
_PRE_TASK_MULTI_DISJOINT = (2, 11, 6)


def test_single_model_memory_evaluation_default_is_unchanged_by_discount_wiring():
    features = ["f0", "f1", "f2", "f3"]
    clf = _tiny_forest([0, 1, 2], seed=0)

    (range_entries, range_blocks, ternary_entries, ternary_blocks,
     _codewords, codeword_length,
     range_table_specs, ternary_table_specs) = ev.single_model_memory_evaluation(clf, features)

    assert (range_entries, range_blocks, ternary_entries, ternary_blocks,
            codeword_length) == _PRE_TASK_SINGLE_APP
    assert range_table_specs == _PRE_TASK_SINGLE_APP_RANGE_SPECS
    assert ternary_table_specs == _PRE_TASK_SINGLE_APP_TERNARY_SPECS


def test_single_model_memory_evaluation_discount_drops_every_majority_leaf():
    features = ["f0", "f1", "f2", "f3"]
    clf = _tiny_forest([0, 1, 2], seed=0)

    (_, _, entries_off, _, codewords, _, _, _) = ev.single_model_memory_evaluation(clf, features)
    (_, _, entries_on, _, _, _, _, _) = ev.single_model_memory_evaluation(
        clf, features, use_default_action_discount=True)

    expected_dropped = sum(
        len(bps.most_common_class_and_dropped_codewords(codewords[tree])[1])
        for tree in codewords)
    assert expected_dropped > 0                    # fixture really exercises the discount
    assert entries_on == entries_off - expected_dropped
    assert entries_on < entries_off


def test_multi_model_memory_evaluation_default_is_unchanged_by_discount_wiring():
    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    usage_joint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, 'joint')
    assert (usage_joint.stages, usage_joint.blocks, usage_joint.stage_depth) == _PRE_TASK_MULTI_JOINT
    usage_disjoint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, 'disjoint')
    assert (usage_disjoint.stages, usage_disjoint.blocks,
            usage_disjoint.stage_depth) == _PRE_TASK_MULTI_DISJOINT


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_multi_model_memory_evaluation_discount_lowers_blocks(monkeypatch, encoding):
    # multi_model_memory_evaluation returns a ResourceUsage(stages, blocks,
    # stage_depth, range_entries, ternary_entries) -- the discounted ENTRY
    # count it feeds into the block formula IS NOW returned in the ResourceUsage
    # (range_entries and ternary_entries fields). With these tiny forests every
    # tree still fits in one 512-entry TCAM block either way, so the discount
    # would be invisible at the real per-block capacity. Shrinking that one
    # hardware constant for the duration of the test makes the entry reduction
    # observable in the returned blocks, so this asserts on REAL returned
    # numbers rather than on a spy. Both calls run under the same shrunken
    # constant, so the difference is attributable to the discount alone.
    #
    # The 'disjoint' case is the load-bearing one: its ternary accounting
    # happens entirely inside the two NESTED single_model_memory_evaluation
    # calls, so it only shrinks if the flag is threaded into those too.
    # TERNARY_MATCHING_ENTRIES_PER_BLOCK is read by range_/ternary_matching_
    # resource_usage, which now live in src/p4model/tables.py -- so the patch
    # has to land in THAT module's namespace. Patching evaluation's re-export
    # would be a silent no-op: the functions never look there.
    from src.p4model import tables

    monkeypatch.setattr(tables, "TERNARY_MATCHING_ENTRIES_PER_BLOCK", 2)

    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    usage_off = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, encoding)
    usage_on = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, encoding,
        use_default_action_discount=True)

    assert usage_on.blocks < usage_off.blocks


# ---------------------------------------------------------------------------
# Range-table key WIDTH: nibble geometry and the 19-bit SDE ceiling.
#
# Unlike ternary_matching_resource_usage's ceil((codeword+4)/44) width term
# (words-per-entry genuinely grows with ternary key width), a range key's
# words-per-entry is fixed by PHV container width, which generate_P4_code
# pins to 16 bits via @pa_container_size -- so range_matching_resource_usage
# no longer charges any width-based block inflation. What DOES vary with
# key_bit_width is (a) the nibble geometry range_entry_count() decomposes
# against, and (b) the crossbar byte width reported per table. Above
# MAX_RANGE_KEY_BITS (19) the real SDE refuses to compile the table at all
# (Sec 4.2), so nibble_widths_for() raises instead of pricing it.
# ---------------------------------------------------------------------------


def test_nibble_widths_for_16_bits_is_four_nibbles():
    assert ev.nibble_widths_for(16) == (4, 4, 4, 4)


def test_nibble_widths_for_12_bits_is_three_nibbles():
    assert ev.nibble_widths_for(12) == (4, 4, 4)


def test_nibble_widths_for_18_bits_has_a_two_bit_remainder_nibble():
    assert ev.nibble_widths_for(18) == (4, 4, 4, 4, 2)


def test_nibble_widths_for_rejects_widths_above_the_sde_ceiling():
    with pytest.raises(ValueError):
        ev.nibble_widths_for(20)


def test_range_matching_resource_usage_rejects_a_key_wider_than_the_ceiling():
    # This is the worked example that motivated the fix: before threading
    # key_bit_width into the nibble geometry, this call silently returned
    # (2, 1, [(1, 4)]) -- rows priced as if the key were 16-bit, even though
    # a 32-bit range key does not compile on real hardware. It must now
    # raise instead of returning a bogus price.
    feature_intervals = {"f": [(0, 100000), (100001, 200000)]}
    with pytest.raises(ValueError):
        ev.range_matching_resource_usage(feature_intervals, key_bit_width=32)


def test_range_table_specs_report_the_real_key_byte_width():
    # The crossbar byte cost per table must follow the declared key width,
    # even though (post-fix) the block count no longer inflates with it --
    # blocks are decided by the actual row count, not a width fudge.
    feature_intervals = {"F": [(0, 255)]}

    _, blocks_8, specs_8 = ev.range_matching_resource_usage(feature_intervals, key_bit_width=8)
    _, blocks_16, specs_16 = ev.range_matching_resource_usage(feature_intervals, key_bit_width=16)

    assert blocks_8 == 1
    assert blocks_16 == 1
    assert specs_8 == [(1, 1)]                 # 1 block, 1 crossbar byte
    assert specs_16 == [(1, 2)]                # 1 block, 2 crossbar bytes


def test_range_matching_resource_usage_default_width_is_the_project_16_bit():
    # Regression guard: the project's decided feature precision is 16-bit
    # (reviews/p4_tofino_reference.md Sec 5), so leaving key_bit_width
    # unspecified must not change any existing caller's numbers.
    feature_intervals = {"F": [(0, 255)], "G": [(10, 300)]}

    assert (ev.range_matching_resource_usage(feature_intervals) ==
            ev.range_matching_resource_usage(feature_intervals, key_bit_width=16))


# ---------------------------------------------------------------------------
# Dependency-aware stage placement.
#
# crossbar_stages_needed alone is a pure bin-packer: it answers "how few
# stages could these tables fit in", which is a LOWER bound. The real
# compiler also obeys data dependencies -- a feature's range table cannot be
# placed before the register chain that produces its key value has run --
# and it places eagerly, at the earliest legal stage rather than the latest.
#
# The chain depth is fully derivable from FEATURE_REGISTER_CATALOG:
#     level = 1 (flow hash)
#           + 1 if the feature is fwd-gated (flow_orientation must resolve)
#           + one per RegisterAction in the feature's chain
#
# Measured against a real compile of the M2 program (3 app trees + 1 ddos
# tree over these 4 features): levels 3/3/3/4 reproduce the observed stage
# offsets 0/0/0/1 exactly -- the range tables really do occupy 2 stages, not
# the 1 the pure packer predicted, and the classification tables 1 more.
# ---------------------------------------------------------------------------


def test_feature_readiness_level_counts_hash_gating_and_chain_depth():
    # FLOW_HASH_LEVEL is 3, not 1: p4c spends three real stages before any
    # register can run (metadata init, hash $precompute, hash). These four
    # levels are M2's, and 5/5/5/6 is now literally where the compiler put
    # M2's four range tables -- see the section comment above.
    # ungated, 2-deep chain (last_arrival_time -> max): 3 + 0 + 2
    assert ev.feature_readiness_level("flow_iat_max") == 5
    # ungated, 2-deep chain (shared last_arrival_time -> mean): 3 + 0 + 2
    assert ev.feature_readiness_level("flow_iat_mean") == 5
    # fwd-gated, 1-deep chain: 3 + 1 + 1
    assert ev.feature_readiness_level("fwd_packet_length_max") == 5
    # fwd-gated, 2-deep chain: 3 + 1 + 2 -- the deepest, and the one the real
    # compiler pushed into a stage of its own
    assert ev.feature_readiness_level("fwd_iat_max") == 6


def test_feature_readiness_level_bwd_gated_costs_same_as_fwd_gated():
    # A bwd-gated feature waits on the same meta.fwd signal as a fwd-gated
    # one (flow_orientation_action resolves it unconditionally either way),
    # so it must cost the same +1 gate stage.
    synthetic_catalog = {
        "bwd_synthetic_feature": {
            "registers": [{
                "name": "bwd_synthetic_reg",
                "role": "value",
                "width": 16,
                "body": "running_max_iat",
            }],
            "gated_by": "bwd",
        },
    }
    # hash + 1 gate + 1-deep chain: 3 + 1 + 1
    assert ev.feature_readiness_level("bwd_synthetic_feature", catalog=synthetic_catalog) == 5


def test_feature_readiness_level_unknown_feature_is_ready_after_the_hash():
    # A feature with no catalog entry gets no registers emitted at all, so
    # nothing gates its table beyond the flow hash itself -- which is three
    # real stages (FLOW_HASH_LEVEL), not one.
    assert ev.feature_readiness_level("Not_A_Catalog_Feature") == 3


def test_readiness_levels_follow_feature_intervals_order():
    # Must align positionally with range_matching_resource_usage's specs,
    # which follow feature_intervals iteration order.
    feature_intervals = {
        "fwd_iat_max": [(0, 5)],
        "flow_iat_max": [(0, 5)],
        "Not_A_Catalog_Feature": [(0, 5)],
    }

    assert ev.readiness_levels_for(feature_intervals) == [6, 5, 3]


def test_feature_readiness_level_resolves_dotted_dataset_names():
    """F5: the catalog is keyed 'flow_iat_max' but read_app_dataset /
    read_DDOS_dataset ship 'Flow.IAT.Max'. Only spaces were normalised, so
    every real feature name missed the catalog and fell back to
    FLOW_HASH_LEVEL -- which left the register-dependency model in
    crossbar_stages_needed inert while `stages` was still being reported."""
    assert ev.feature_readiness_level("Flow.IAT.Max") == 5
    assert ev.feature_readiness_level("Flow.IAT.Mean") == 5
    assert ev.feature_readiness_level("Fwd.Packet.Length.Max") == 5
    assert ev.feature_readiness_level("Fwd.IAT.Max") == 6


def test_feature_readiness_level_unknown_dotted_feature_still_falls_back():
    """Task 9 grew FEATURE_REGISTER_CATALOG to cover all 18 of main.py's
    selected features (previously only 4 had entries, and these two dotted
    names fell back to FLOW_HASH_LEVEL). Both now resolve to a real,
    normalisation-surviving catalog entry instead of a KeyError or a
    fallback -- bwd_packet_length_min is bwd-gated with one value register
    (no dependency register: packet-length bodies read hdr.ipv4.total_len
    directly), and packet_length_mean is ungated with one value register."""
    assert ev.feature_readiness_level("Bwd.Packet.Length.Min") == ev.FLOW_HASH_LEVEL + 1 + 1
    assert ev.feature_readiness_level("Packet.Length.Mean") == ev.FLOW_HASH_LEVEL + 0 + 1


def test_readiness_levels_for_real_dataset_feature_names():
    """readiness_levels_for is positionally aligned with feature_intervals, so
    the levels must follow the dict's key order exactly. Bwd.IAT.Min now has
    a real catalog entry too (Task 9): bwd-gated (+1) with a dependency
    register (bwd_last_arrival_time) plus its own value register (+2).

    Bwd.IAT.Min is a level LATER than the symmetric Fwd.IAT.Max, and the
    asymmetry is real: generate_P4_registers_and_apply emits
    `if (meta.fwd == 1) { ... }` before `if (meta.fwd == 0) { ... }`, and
    p4c's placer cannot reach the second block until the first is fully
    placed (see register_stage_schedule's per-block floor). Chain depth alone
    would report 6 for both."""
    feature_intervals = {
        "Fwd.IAT.Max": [(0, 10), (11, 65535)],
        "Flow.IAT.Max": [(0, 20), (21, 65535)],
        "Bwd.IAT.Min": [(0, 30), (31, 65535)],
    }
    assert ev.readiness_levels_for(feature_intervals) == [6, 5, 7]


def test_crossbar_stages_needed_separates_tables_by_readiness_level():
    # Four trivially small tables that the pure packer puts in one stage.
    specs = [(1, 2)] * 4
    assert ev.crossbar_stages_needed(specs).occupied == 1

    # The same four, with one not ready until a later level, must occupy two
    # distinct stages -- exactly the M2 range-pool case.
    assert ev.crossbar_stages_needed(specs, readiness_levels=[3, 3, 3, 4]).occupied == 2


def test_crossbar_stages_needed_counts_occupied_stages_not_the_span():
    # A single late table occupies ONE stage, however deep its level is --
    # the earlier stage indices belong to other work (registers), not to this
    # table pool.
    assert ev.crossbar_stages_needed([(1, 2)], readiness_levels=[7]).occupied == 1


def test_crossbar_stages_needed_spills_past_its_level_when_full():
    # Nine same-level tables cannot share one stage (8-table crossbar cap),
    # so one spills into the next stage even though its level allows earlier.
    assert ev.crossbar_stages_needed([(1, 2)] * 9, readiness_levels=[3] * 9).occupied == 2


def test_classification_tables_are_placed_after_the_last_occupied_range_stage():
    """Nine features at one readiness level: the ninth range table spills past the
    8-table crossbar cap into the next stage, so classification must start one
    stage later than max(levels)+1 would say."""
    range_specs = [(1, 2)] * 9
    levels = [3] * 9

    plan = ev.crossbar_stages_needed(range_specs, readiness_levels=levels)

    assert plan.indices == frozenset({3, 4})
    assert plan.depth == 5              # max index + 1 -- NOT max(levels) + 1 == 4
    assert plan.occupied == 2


def test_range_and_ternary_pools_reproduce_the_measured_m2_stage_count():
    # The real M2 feature set. Real compile: range tables in 2 stages,
    # classification tables in 1 -> 3 stages of match tables.
    feature_intervals = {
        "flow_iat_max": [(0, 100), (101, 65535)],
        "flow_iat_mean": [(0, 100), (101, 65535)],
        "fwd_iat_max": [(0, 100), (101, 65535)],
        "fwd_packet_length_max": [(0, 100), (101, 65535)],
    }
    _, _, range_specs = ev.range_matching_resource_usage(feature_intervals)
    range_levels = ev.readiness_levels_for(feature_intervals)

    range_plan = ev.crossbar_stages_needed(range_specs, readiness_levels=range_levels)
    # Classification tables read every feature's codeword, so they cannot be
    # placed until one stage after the last range table actually landed
    # (F10: range_plan.depth, not max(range_levels) + 1).
    ternary_level = range_plan.depth
    ternary_plan = ev.crossbar_stages_needed([(2, 11)] * 4,
                                             readiness_levels=[ternary_level] * 4)

    # With FLOW_HASH_LEVEL corrected to 3 these are no longer just internally
    # consistent indices -- they are the stage numbers the compiler really
    # used for M2: range tables at 5/5/5/6, classification at 7 (see
    # reviews/p4_tofino_reference.md Sec 4.6's measured table).
    assert range_plan.indices == frozenset({5, 6})
    assert ternary_plan.indices == frozenset({7})
    assert not (range_plan.indices & ternary_plan.indices)
    assert range_plan.occupied == 2
    assert ternary_plan.occupied == 1
    assert range_plan.occupied + ternary_plan.occupied == 3


def _forest_using_all_four_catalog_features(labels, seed):
    """A forest that really splits on all four M2 catalog features, so the
    readiness levels under test are all actually present."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rnd = np.random.RandomState(seed)
    X = rnd.randint(0, 60000, size=(400, 4))
    y = np.array([labels[(a // 30000 + b // 30000 + c // 30000 + d // 30000)
                         % len(labels)]
                  for a, b, c, d in X])
    # Deliberately SHALLOW: few intervals per feature, so the pure packer
    # needs only one range stage and any second stage can only come from
    # dependency depth (asserted explicitly in the tests below).
    clf = RandomForestClassifier(n_estimators=1, max_depth=4,
                                 random_state=seed, bootstrap=False).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


_M2_CATALOG_FEATURES = ["flow_iat_max", "flow_iat_mean",
                        "fwd_iat_max", "fwd_packet_length_max"]


def test_multi_model_memory_evaluation_accounts_for_register_dependency_depth():
    # End-to-end: the reported stage count must include the extra stage that
    # fwd_iat_max's deeper register chain forces, matching the real compile
    # (range tables over 2 stages + classification tables in 1 = 3), not the
    # pure packer's 2.
    clf_app = _forest_using_all_four_catalog_features([0, 1, 2], seed=0)
    clf_ddos = _forest_using_all_four_catalog_features([-1, 1], seed=7)

    intervals = bps.get_feature_intervals(clf_app, _M2_CATALOG_FEATURES)
    assert set(intervals) == set(_M2_CATALOG_FEATURES), (
        "fixture did not split on every catalog feature: {}".format(sorted(intervals)))
    # Pin the baseline: without dependency levels these range tables all pack
    # into ONE stage, so a result of 3 below can only come from the extra
    # stage fwd_iat_max's deeper chain forces -- this test cannot pass for
    # the wrong reason.
    _, _, range_specs = ev.range_matching_resource_usage(intervals)
    assert ev.crossbar_stages_needed(range_specs).occupied == 1

    usage = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, _M2_CATALOG_FEATURES, _M2_CATALOG_FEATURES, "joint")

    assert usage.stages == 3
    # This IS the M2 fixture the brief's own worked example cites. It used to
    # report depth 6 against the real compiler's 9; with the readiness origin
    # and the vote epilogue corrected it reports 9 -- the same number, on the
    # same program, that p4_compile.parse_compile_logs measured.
    assert usage.stage_depth == 9


def test_multi_model_memory_evaluation_uncatalogued_features_have_no_extra_depth():
    # The same shaped models over feature names with no catalog entry have no
    # register chains at all, so nothing forces a second range stage.
    clf_app = _forest_using_all_four_catalog_features([0, 1, 2], seed=0)
    clf_ddos = _forest_using_all_four_catalog_features([-1, 1], seed=7)

    usage = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, ["g0", "g1", "g2", "g3"], ["g0", "g1", "g2", "g3"], "joint")

    assert usage.stages == 2
    # 3 hash stages + 1 range + 1 classification + 1 vote.
    assert usage.stage_depth == 6


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_stage_depth_equals_max_of_range_and_ternary_depth(encoding):
    clf_app, clf_ddos, names = _joint_pair_fixture()
    usage = ev.multi_model_memory_evaluation(clf_app, clf_ddos, names, names, encoding)
    assert usage.stage_depth == (max(usage.range_depth, usage.ternary_depth)
                                 + ev.VOTE_EPILOGUE_STAGES)


def test_ternary_tables_equals_total_tree_count_regardless_of_encoding():
    # ternary_matching_resource_usage builds one table PER TREE
    # (evaluation.py:299-302's own docstring), so the count does not depend
    # on whether the two models share a codeword -- only the total tree
    # count across both forests does.
    clf_app, clf_ddos, names = _joint_pair_fixture()
    total_trees = clf_app.n_estimators + clf_ddos.n_estimators
    for encoding in ('joint', 'disjoint'):
        usage = ev.multi_model_memory_evaluation(clf_app, clf_ddos, names, names, encoding)
        assert usage.ternary_tables == total_trees


def test_joint_range_tables_are_fewer_than_disjoint_on_the_same_feature_union():
    # D2's fixture requirement (spec Sec 4.1/Sec 7): under 'joint', every
    # selected feature gets ONE range table shared by both models; under
    # 'disjoint', each model gets its OWN table for the same feature name --
    # so with identical feature lists on both sides, joint must have fewer
    # range_tables than disjoint, since joint shares across models.
    clf_app = _forest_using_all_four_catalog_features([0, 1, 2], seed=0)
    clf_ddos = _forest_using_all_four_catalog_features([-1, 1], seed=7)

    joint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, _M2_CATALOG_FEATURES, _M2_CATALOG_FEATURES, 'joint')
    disjoint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, _M2_CATALOG_FEATURES, _M2_CATALOG_FEATURES, 'disjoint')

    assert joint.range_tables == 4      # one table per feature in the union
    assert disjoint.range_tables == 7   # app splits on all 4, ddos on 3 (no sharing)
    assert joint.range_tables < disjoint.range_tables


def test_register_depth_is_identical_under_joint_and_disjoint_encoding():
    # D1 as a unit test (spec Sec 5.2, Sec 7): the premise the whole
    # attribution rests on. register_depth is a function of the SELECTED
    # FEATURE SET only (Sec 2.1), which 'joint' vs 'disjoint' never changes,
    # so this must hold exactly, not approximately.
    clf_app = _forest_using_all_four_catalog_features([0, 1, 2], seed=0)
    clf_ddos = _forest_using_all_four_catalog_features([-1, 1], seed=7)

    joint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, _M2_CATALOG_FEATURES, _M2_CATALOG_FEATURES, 'joint')
    disjoint = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, _M2_CATALOG_FEATURES, _M2_CATALOG_FEATURES, 'disjoint')

    assert joint.register_depth == disjoint.register_depth
    assert joint.register_depth > 0   # a real, nontrivial readiness level


# ---------------------------------------------------------------------------
# register_depth / register_count (Task 6).
# ---------------------------------------------------------------------------


def _single_catalog_feature_forest(labels, seed):
    """A one-column forest, so selecting that single column under its own
    catalog feature name always yields a real (nonempty) feature_intervals
    entry -- deliberately shallow, like _forest_using_all_four_catalog_features,
    so only the register-dependency chain (not the range-table packing) is
    under test here."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rnd = np.random.RandomState(seed)
    X = rnd.randint(0, 60000, size=(400, 1))
    y = np.array([labels[(a // 30000) % len(labels)] for (a,) in X])
    clf = RandomForestClassifier(n_estimators=1, max_depth=4,
                                 random_state=seed, bootstrap=False).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


def _app_forest():
    return _single_catalog_feature_forest([0, 1, 2], seed=11)


def _ddos_forest():
    return _single_catalog_feature_forest([-1, 1], seed=13)


def test_register_depth_and_count_over_the_selected_features():
    """Spec 4.1/4.2. Depth is max readiness level -- how many stages elapse
    before ANY classification table can run. It is NOT a capacity guarantee:
    whether the registers FIT in those stages has never been measured here."""
    usage = ev.multi_model_memory_evaluation(
        _app_forest(), _ddos_forest(), ['flow_iat_max'], ['fwd_iat_max'], 'disjoint')
    # fwd_iat_max: FLOW_HASH_LEVEL(3) + gated(1) + 2 registers
    assert usage.register_depth == 6
    assert usage.register_count == 4          # 2 chains x (dependency + value)


def test_max_num_flows_matches_the_p4_template_it_claims_to_mirror():
    """build_p4_script.MAX_NUM_FLOWS is never read as a Python value today --
    it only appears as literal text inside emitted P4, so pin it against the
    authoritative source."""
    template = pathlib.Path('resources/p4_template.p4').read_text()
    assert 'const bit<32> MAX_NUM_FLOWS = {};'.format(bps.MAX_NUM_FLOWS) in template


def test_resource_usage_does_not_report_register_sram_bits():
    # Dropped 2026-09-06. It was reported to the campaign CSV and read by
    # nothing: no figure, no claim, no statistical test, no premise check, no
    # constraint, no tie-break. register_depth stays (stage_attribution.py:50-64
    # uses it as a Sec 5.2 premise check) and register_count stays (it is what
    # makes the 4-stateful-ALUs-per-stage argument legible), but total register
    # bits across all flows answers no question this project asks -- and SRAM
    # block prediction, the one thing it might have grown into, is a permanent
    # non-goal.
    import dataclasses

    fields = {f.name for f in dataclasses.fields(ev.ResourceUsage)}
    assert "register_sram_bits" not in fields
    assert "register_depth" in fields
    assert "register_count" in fields


def _joint_pair_fixture():
    """A joint-encoded pair of classifiers (app and ddos) trained on the same
    feature space, used for multi_model_memory_evaluation tests."""
    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)
    return clf_app, clf_ddos, features


def test_codeword_bits_to_blocks_matches_the_inline_expression_it_replaces():
    """The 44-bit step structure C1 gates on. Boundaries are at L+4 == 44k,
    i.e. L in {40, 84, 128, ...}, NOT at multiples of 44."""
    assert ev.codeword_bits_to_blocks(0) == 1
    assert ev.codeword_bits_to_blocks(40) == 1      # 44 key bits exactly
    assert ev.codeword_bits_to_blocks(41) == 2      # first bit of the second block
    assert ev.codeword_bits_to_blocks(84) == 2      # 88 key bits exactly
    assert ev.codeword_bits_to_blocks(85) == 3


def test_codeword_length_is_the_pooled_threshold_count():
    """C1's central identity. generate_codewords emits exactly
    len(intervals_f) - 1 bits per feature (build_p4_script.py:514/520), so the
    codeword length is the pooled split-threshold count -- which is what
    joint_interval_count measures, less one interval per feature.

    If this fails, BlockBudget's arithmetic is measuring the wrong quantity and
    every task after this one is built on sand.
    """
    from src.training.threshold_alignment import (
        joint_interval_count, extract_feature_intervals)

    clf_app, clf_ddos, names = _joint_pair_fixture()
    usage = ev.multi_model_memory_evaluation(clf_app, clf_ddos, names, names, 'joint')

    iv1 = extract_feature_intervals(clf_app)
    iv2 = extract_feature_intervals(clf_ddos)
    n_features = len(set(iv1) | set(iv2))

    assert usage.codeword_length == joint_interval_count(iv1, iv2) - n_features


def test_resource_usage_carries_the_codeword_length_on_both_encodings():
    clf_app, clf_ddos, names = _joint_pair_fixture()
    for encoding in ('joint', 'disjoint'):
        usage = ev.multi_model_memory_evaluation(clf_app, clf_ddos, names, names, encoding)
        assert usage.codeword_length > 0
        assert usage.codeword_length <= bps.MAX_CODEWORD_LENGTH


# ---------------------------------------------------------------------------
# Compiler-calibration corrections (2026-09-05).
#
# scripts/compiler_calibration.py compiled 19 real campaign-scale programs
# through p4c and their committed placements (results/compiler_calibration/
# compiles/*/pipe/logs/) contradict two things this estimator assumed:
#
#   * the Ternary Match Input crossbar charges the UNION of the distinct key
#     FIELDS present in a stage, not the sum of each table's key width -- and
#     every tree of one task keys on the identical meta.code_<feature> field;
#   * feature_readiness_level's origin was 2 stages early, and stage_depth
#     never counted the vote tables' trailing stage.
#
# See reviews/p4_tofino_reference.md Sec 7 for the full measurement.
# ---------------------------------------------------------------------------


def test_crossbar_charges_one_shared_key_field_once_not_once_per_table():
    # Measured, joint_low_sd7 stage 7 (mau.resources.log): four classification
    # tables all keyed on the SAME bit<256> codeword field report 32 crossbar
    # bytes for that stage, not 4 x 32 = 128. The field occupies its byte
    # slots once and all four tables read from those same slots.
    shared = frozenset({("code_fwd_packet_length_max", 32)})
    assert ev.crossbar_stages_needed([(1, 32)] * 4,
                                     key_fields=[shared] * 4).occupied == 1


def test_crossbar_charges_distinct_key_fields_separately():
    # Measured, independent_low_sd6 stage 7: two 19-byte app tables plus two
    # 4-byte ddos tables report 23 bytes -- 19 + 4, each distinct field once.
    # Scaled to 3 + 3 here so the naive per-table sum (3*19 + 3*4 = 69) would
    # exceed the 64-byte budget and force a second stage, while the real
    # union (23) does not: the two accountings give different answers.
    app = frozenset({("code_app", 19)})
    ddos = frozenset({("code_ddos", 4)})
    assert ev.crossbar_stages_needed(
        [(1, 19)] * 3 + [(1, 4)] * 3,
        key_fields=[app] * 3 + [ddos] * 3).occupied == 1


def test_crossbar_key_fields_must_account_for_the_declared_byte_width():
    # key_fields and the spec's byte_width describe the same table. A caller
    # that lets the two drift would silently mis-price every stage the table
    # appears in, so the mismatch is rejected at the boundary.
    with pytest.raises(ValueError):
        ev.crossbar_stages_needed([(1, 32)],
                                  key_fields=[frozenset({("code", 8)})])


def test_crossbar_without_key_fields_charges_every_table_its_own_width():
    # Backwards compatibility: callers that pass no key_fields keep the
    # conservative "every table has its own private key" accounting, so four
    # 32-byte tables still need two stages under the 64-byte budget.
    assert ev.crossbar_stages_needed([(1, 32)] * 4).occupied == 2


def test_feature_readiness_level_lands_on_the_stage_the_compiler_uses():
    # FLOW_HASH_LEVEL has to cover THREE real stages, not one: p4c emits a
    # metadata-init table (stage 0) and splits the hash into
    # tbl_calc_flow_hash$precompute (stage 1) and tbl_calc_flow_hash (stage
    # 2), so the first register in any chain can only run at stage 3.
    # Measured against the committed placements:
    #   independent_low_sd6  table_1_bwd_packet_length_max   -> real stage 5
    #   independent_high_sd6 table_6_app_packet_length_mean  -> real stage 4
    assert ev.feature_readiness_level("bwd_packet_length_max") == 5
    assert ev.feature_readiness_level("packet_length_mean") == 4


# ---------------------------------------------------------------------------
# Stateful-ALU width (2026-09-05, second calibration pass).
#
# feature_readiness_level models a feature's register chain DEPTH. It does not
# model the pipeline's register WIDTH: a Tofino stage has only 4 stateful
# ("meter") ALUs, and every RegisterAction this generator emits burns one. At
# k >= 13 features the design needs 16-20 registers, so they cannot all issue
# in the two or three stages their dependency chains allow -- the compiler
# serialises them and every downstream range table slides with them.
#
# Confirmed against all 18 committed calibration placements: the list schedule
# below reproduces the compiler's own last-register stage EXACTLY on every row
# (7,7,7,7,4,5,4,4,4,4,7,7,7,4,4,4,4,4), where chain depth alone gives 5 on
# each of the six high-k rows. See reviews/p4_tofino_reference.md Sec 7.
# ---------------------------------------------------------------------------


def test_meter_alus_per_stage_matches_the_compilers_own_saturation():
    # mau.resources.log's percentage table reports "Meter ALU 4" as 100.00%
    # in joint_high_sd7 stages 3-6 -- the ceiling read off the compiler's own
    # arithmetic, not fitted. Sweeping the constant over 2/3/4/5/6/8 against
    # the 18 committed placements, only 4 reproduces every row (18 vs 13, 11,
    # 10, 8 for its neighbours).
    assert ev.METER_ALUS_PER_STAGE == 4


def test_register_schedule_serialises_registers_past_the_alu_cap():
    # Six independent, ungated one-register features. Their chains are all
    # depth 1, so chain depth alone would run every register at FLOW_HASH_LEVEL.
    # With flow_forward_srcaddr (emitted unconditionally -- the apply block
    # always calls flow_orientation_action.execute, build_p4_script.py:2092)
    # that is 7 RegisterActions competing for 4 ALUs, so they need two stages.
    catalog = {
        "f%d" % i: {"registers": [{"name": "f%d" % i, "role": "value",
                                   "width": 16, "body": "running_max_iat"}],
                    "gated_by": None}
        for i in range(6)
    }
    placed = ev.register_stage_schedule(list(catalog), catalog=catalog)

    by_stage = collections.Counter(placed.values())
    assert by_stage[ev.FLOW_HASH_LEVEL] == ev.METER_ALUS_PER_STAGE
    assert by_stage[ev.FLOW_HASH_LEVEL + 1] == 3
    assert max(placed.values()) == ev.FLOW_HASH_LEVEL + 1


def test_register_schedule_never_delays_the_orientation_register():
    # flow_forward_srcaddr resolves meta.fwd, so EVERY gated feature's chain
    # hangs off it. Filling the first register stage with leaf registers and
    # spilling it forward would push that whole subtree back a stage. Real
    # placements agree: it sits in the first register stage on all 18 rows.
    # Four ungated leaves would take the entire ALU budget of stage 3 on a
    # naive earliest-first schedule; critical-path priority takes the
    # orientation register first instead.
    catalog = {
        "leaf%d" % i: {"registers": [{"name": "leaf%d" % i, "role": "value",
                                      "width": 16, "body": "running_max_iat"}],
                       "gated_by": None}
        for i in range(4)
    }
    catalog["gated"] = {"registers": [
        {"name": "gated_dep", "role": "dependency", "width": 16, "body": "iat"},
        {"name": "gated_val", "role": "value", "width": 16, "body": "running_max_iat"},
    ], "gated_by": "fwd"}

    placed = ev.register_stage_schedule(list(catalog), catalog=catalog)
    assert placed[ev.ORIENTATION_REGISTER] == ev.FLOW_HASH_LEVEL
    assert placed["gated_dep"] == ev.FLOW_HASH_LEVEL + 1
    assert placed["gated_val"] == ev.FLOW_HASH_LEVEL + 2


def test_register_schedule_keeps_a_chain_sequential_even_with_alus_free():
    # The ALU cap is an extra constraint, never a relaxation: a dependency
    # register still has to run a whole stage before the value register that
    # consumes its meta.current_iat, however idle the stage's other ALUs are.
    catalog = {
        "chained": {"registers": [
            {"name": "dep", "role": "dependency", "width": 16, "body": "iat"},
            {"name": "val", "role": "value", "width": 16, "body": "running_max_iat"},
        ], "gated_by": None},
    }
    placed = ev.register_stage_schedule(["chained"], catalog=catalog)
    assert placed["val"] == placed["dep"] + 1


def test_readiness_levels_push_past_the_chain_depth_at_campaign_feature_counts():
    # independent_high_sd10's real 16-feature set. Its 20 registers schedule
    # 4,4,4,4,4 across stages 3-7 -- exactly where the compiler put them --
    # so its latest feature is ready at stage 8, not the 6 that chain depth
    # alone reports. Two whole stages of pipeline depth the model never saw.
    features = [
        "bwd_iat_max", "bwd_iat_mean", "bwd_iat_min", "bwd_packet_length_max",
        "bwd_packet_length_mean", "bwd_packet_length_min", "flow_iat_max",
        "flow_iat_mean", "flow_iat_min", "fwd_iat_mean", "fwd_iat_min",
        "fwd_packet_length_max", "fwd_packet_length_mean", "min_packet_length",
        "packet_length_mean", "fwd_iat_max",
    ]
    feature_intervals = {name: [(0, 5)] for name in features}

    placed = ev.register_stage_schedule(features)
    assert len(placed) == 20
    assert collections.Counter(placed.values()) == {3: 4, 4: 4, 5: 4, 6: 4, 7: 4}

    levels = ev.readiness_levels_for(feature_intervals)
    assert max(levels) == 8
    assert max(ev.feature_readiness_level(f) for f in features) == 6


def test_readiness_levels_are_independent_of_feature_order():
    # The schedule's makespan must be a property of the design, not of dict
    # iteration order. Verified over 300 shuffles of every calibration row's
    # feature list: the last-register stage never moved.
    features = ["fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "bwd_iat_max",
                "bwd_iat_min", "bwd_iat_mean", "flow_iat_max", "flow_iat_min"]
    forward = ev.register_stage_schedule(features)
    backward = ev.register_stage_schedule(list(reversed(features)))
    assert max(forward.values()) == max(backward.values())


def test_readiness_levels_schedule_both_models_registers_on_one_pipeline():
    # Under 'disjoint' each model keeps its own intervals, but there is only
    # ONE register block and one set of stateful ALUs: a register a feature
    # needs is emitted once however many models select that feature
    # (register_names_for dedupes by name). So the levels for one model's
    # features have to be read off a schedule built from the UNION -- pricing
    # each model's registers against a private pipeline would understate the
    # pressure whenever the two models select different features.
    #
    # The app features here are bwd-gated and the ddos ones fwd-gated on
    # purpose: `if (meta.fwd == 0)` is emitted after `if (meta.fwd == 1)`, so
    # the other model's registers push this one's whole block later, which is
    # ALU pressure and control-flow order compounding rather than either
    # alone.
    app = {"bwd_iat_max": [(0, 5)], "bwd_iat_mean": [(0, 5)]}
    ddos = {"fwd_iat_max": [(0, 5)], "fwd_iat_min": [(0, 5)],
            "fwd_iat_mean": [(0, 5)], "fwd_packet_length_max": [(0, 5)],
            "fwd_packet_length_min": [(0, 5)], "fwd_packet_length_mean": [(0, 5)]}
    union = list(app) + list(ddos)

    alone = ev.readiness_levels_for(app)
    together = ev.readiness_levels_for(app, emitted_features=union)

    assert len(together) == len(app)
    assert max(together) > max(alone)


# ---------------------------------------------------------------------------
# Mechanism B: a gated register sub-block starves the table pools that follow
# it. Tofino has no program counter -- each table hands the next stage a
# next-table pointer -- so p4c's placer walks the control block with a work-
# list CURSOR, and a table is a placement candidate only once the cursor
# reaches it. generate_P4_registers_and_apply emits the unconditional
# registers, then `if (meta.fwd == 1) {...}`, then `if (meta.fwd == 0) {...}`,
# and every match table AFTER both blocks. While the cursor is inside a gated
# block, those tables are out of reach. See reviews/p4_tofino_reference.md
# Sec 7, "Residual root-caused -- Mechanisms A/B/C".
# ---------------------------------------------------------------------------

_BLOCK_ORDER_CATALOG = {
    # One fwd-gated feature whose three-register chain necessarily spans three
    # stages, and one bwd-gated leaf that could run in the very first of them
    # if the placer were free to reorder across the two `if` blocks.
    "fwd_chain": {"registers": [
        {"name": "fwd_a", "role": "dependency", "width": 16, "body": "iat"},
        {"name": "fwd_b", "role": "dependency", "width": 16, "body": "iat"},
        {"name": "fwd_c", "role": "value", "width": 16, "body": "running_max_iat"},
    ], "gated_by": "fwd"},
    "bwd_leaf": {"registers": [
        {"name": "bwd_a", "role": "value", "width": 16, "body": "running_max_iat"},
    ], "gated_by": "bwd"},
}


def test_register_schedule_orders_the_gated_blocks_against_each_other():
    # The cursor is a SEQUENCE position: it only reaches `if (meta.fwd == 0)`
    # once the preceding `if (meta.fwd == 1)` block is fully placed. bwd_a's
    # own dependency chain is one register deep and stages 4 and 5 have three
    # free ALUs each, so nothing but that ordering keeps it out of them.
    placed = ev.register_stage_schedule(list(_BLOCK_ORDER_CATALOG),
                                        catalog=_BLOCK_ORDER_CATALOG)

    assert [placed["fwd_a"], placed["fwd_b"], placed["fwd_c"]] == [4, 5, 6]
    assert placed["bwd_a"] == 6


def test_gated_block_spanning_three_stages_has_one_interior_stage():
    # fwd_a/fwd_b/fwd_c occupy stages 4, 5 and 6. Stage 4 is where the placer
    # DESCENDS into the block (outer tables can still be back-filled there)
    # and stage 6 is where it POPS OUT (free again). Only stage 5 is fully
    # interior, and nothing after the block can enter it.
    interior = ev.gated_block_interior_stages(list(_BLOCK_ORDER_CATALOG),
                                               catalog=_BLOCK_ORDER_CATALOG)

    assert interior == frozenset({5})


def test_gated_block_spanning_two_stages_costs_nothing():
    # Descend and pop out with no stage in between. This is what keeps the
    # penalty from being a flat per-gated-block constant: on the calibration
    # sample the `if (meta.fwd == 0)` block usually spans exactly two stages.
    catalog = {"fwd_pair": {"registers": [
        {"name": "fwd_a", "role": "dependency", "width": 16, "body": "iat"},
        {"name": "fwd_b", "role": "value", "width": 16, "body": "running_max_iat"},
    ], "gated_by": "fwd"}}

    placed = ev.register_stage_schedule(list(catalog), catalog=catalog)
    assert [placed["fwd_a"], placed["fwd_b"]] == [4, 5]
    assert ev.gated_block_interior_stages(list(catalog), catalog=catalog) == frozenset()


def test_ungated_registers_never_make_a_stage_interior():
    # The unconditional registers sit in the OUTER sequence, so the cursor is
    # never "inside" anything while placing them and a following table can
    # share their stage. Only a gated `if` block hides the rest of the program.
    catalog = {
        "chain": {"registers": [
            {"name": "a", "role": "dependency", "width": 16, "body": "iat"},
            {"name": "b", "role": "dependency", "width": 16, "body": "iat"},
            {"name": "c", "role": "value", "width": 16, "body": "running_max_iat"},
        ], "gated_by": None},
    }
    assert ev.gated_block_interior_stages(list(catalog), catalog=catalog) == frozenset()


def test_packer_refuses_to_place_a_table_in_an_interior_stage():
    # The stage is not full -- it is unreachable. In independent_high_sd10's
    # empty stage 5 the TCAM is 0/24, the ternary crossbar 0/66 and the
    # logical table IDs 4/16; the table simply is not a candidate yet.
    specs, levels = [(1, 4)], [3]

    assert ev.crossbar_stages_needed(specs, readiness_levels=levels).indices == frozenset({3})
    pushed = ev.crossbar_stages_needed(specs, readiness_levels=levels,
                                        unavailable_stages=frozenset({3}))
    assert pushed.indices == frozenset({4})
    assert pushed.depth == 5


def test_interior_stage_below_a_tables_readiness_level_costs_nothing():
    # A gated block that finishes before the pool's first table could have
    # been placed anyway changes nothing -- which is why two of the five
    # calibration rows carrying an interior stage show a gap of +0. The
    # mechanism has to produce that for free; a flat penalty cannot.
    specs, levels = [(1, 4)], [7]

    blocked = ev.crossbar_stages_needed(specs, readiness_levels=levels,
                                         unavailable_stages=frozenset({5}))
    assert blocked.indices == frozenset({7})


def test_stage_depth_counts_the_vote_epilogue_stage():
    # Measured on all 19 compiles: SwitchIngress.vote_app/vote_ddos always
    # occupy exactly one stage after the last classification table, and
    # stage_depth -- the quantity checked against TOFINO_PIPELINE_STAGES --
    # never counted it.
    features = ["f0", "f1", "f2", "f3"]
    usage = ev.multi_model_memory_evaluation(
        _tiny_forest([0, 1, 2], seed=0), _tiny_forest([-1, 1], seed=7),
        features, features, 'joint')
    assert ev.VOTE_EPILOGUE_STAGES == 1
    assert usage.stage_depth == (max(usage.range_depth, usage.ternary_depth)
                                 + ev.VOTE_EPILOGUE_STAGES)


# --- Mechanism G: the version-block penalty (tables.version_block_penalty)
#
# SUPERSEDED RULE. This section used to assert "a ragged key at an ODD crossbar
# group offset costs +1 block". That predicate over-fired on 5 of the 6
# calibration stages where it was live, and on the one row it appeared to fix it
# charged the WRONG table: resources.json for independent_low_sd5 shows p4c
# penalising the ddos key at group offset 0 and leaving the app key at offset 3
# alone. What actually costs the block is the mandatory 2-bit --version-- field
# having no free midbyte nibble to live in. See tests/test_version_block.py for
# the rule's own measurement set.

APP_49 = (179, 204)      # 23 + 26 = 49 crossbar bytes, 9 groups, ragged
DDOS_12 = (37, 49)       # 5 + 7 = 12 crossbar bytes, 3 groups
SOLID_49 = (392,)        # one dense field: 49 bytes, no part-used byte at all


def test_a_key_that_loses_its_last_midbyte_nibble_costs_an_extra_block():
    # Measured, scripts/tcam_stretch_sweep.py: a table keying 179+204 bits
    # (49 crossbar bytes, 9 groups) costs 9 TCAMs alone and 10 when a second
    # 12-byte key holds groups 0..2 ahead of it. At group 3 the run no longer
    # ends on a half midbyte, its 49 bytes fill every whole slot its groups
    # supply, and no nibble-clean byte can reach an interior midbyte -- so the
    # version field gets a TCAM block to itself (the waste
    # reviews/github_issue_tcam_version_bit_packing.md documents).
    solid = ev.crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 5,
        key_fields=[frozenset({(('a',), 49)})] +
                   [frozenset({(('b',), 12)})] * 5,
        key_field_bits=[SOLID_49] + [DDOS_12] * 5)
    assert solid.occupied == 1

    ragged = ev.crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 5,
        key_fields=[frozenset({(('a',), 49)})] +
                   [frozenset({(('b',), 12)})] * 5,
        key_field_bits=[APP_49] + [DDOS_12] * 5)
    assert ragged.occupied == 2


def test_the_same_key_pays_nothing_when_it_starts_at_group_zero():
    # 21 blocks, same two keys: measured to fit one stage (probe point
    # ragged_ax1_bx4), with the wide table charged 10 and the four narrow
    # ones 3 each -- 10 | 12 across the two columns.
    plan = ev.crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4,
        key_fields=[frozenset({(('a',), 49)})] +
                   [frozenset({(('b',), 12)})] * 4,
        key_field_bits=[APP_49] + [DDOS_12] * 4)
    assert plan.occupied == 1


def test_stage_plan_blocks_reflects_the_version_charge_not_the_naive_sum():
    # Same ragged_ax1_bx4 ground truth as the test above -- real p4c charges
    # the wide table 10 blocks and each narrow table its declared 3, for 22
    # total. independent_low_sd5 is the same mechanism at production scale:
    # multi_model_memory_evaluation reported 13 blocks (the naive per-table
    # sum) where p4c used 16, because the charge was only ever wired into
    # stage PLACEMENT, never into a total a caller could read.
    plan = ev.crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4,
        key_fields=[frozenset({(('a',), 49)})] +
                   [frozenset({(('b',), 12)})] * 4,
        key_field_bits=[APP_49] + [DDOS_12] * 4)
    assert plan.blocks == 22            # not 21, the naive sum


def test_stage_plan_blocks_matches_the_naive_sum_for_a_solid_key():
    # A key of whole-byte fields presents no nibble-clean byte, so nothing can
    # ride the half midbyte an odd start exposes and it stays free for version
    # at every offset -- the sweep's solid control arm, 9 blocks either way.
    plan = ev.crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4,
        key_fields=[frozenset({(('a',), 49)})] +
                   [frozenset({(('b',), 12)})] * 4,
        key_field_bits=[SOLID_49] + [DDOS_12] * 4)
    assert plan.blocks == 9 + 4 * 3


def test_stage_plan_blocks_is_inert_on_a_single_key_stage():
    # Every joint-encoding stage and most independent stages key one shared
    # field -- the commonest case in this generator -- and it sits at offset 0.
    # None of the 8 joint calibration rows has a single penalised table.
    fields = [frozenset({(('shared',), 12)})] * 4
    specs = [(3, 12)] * 4
    plan = ev.crossbar_stages_needed(specs, key_fields=fields,
                                     key_field_bits=[DDOS_12] * 4)
    assert plan.blocks == sum(blocks for blocks, _ in specs)


def test_the_version_penalty_never_fires_on_a_single_key_stage():
    # Every tree of one task keys the identical code_<feature> set, so the
    # commonest stage in this generator holds ONE key set at offset 0.
    fields = [frozenset({(('shared',), 12)})] * 4
    specs = [(3, 12)] * 4
    assert (ev.crossbar_stages_needed(specs, key_fields=fields,
                                      key_field_bits=[DDOS_12] * 4).occupied ==
            ev.crossbar_stages_needed(specs, key_fields=fields).occupied)


def test_key_field_bits_defaults_to_the_pre_existing_pricing():
    specs = [(9, 49)] + [(3, 12)] * 5
    fields = [frozenset({(('a',), 49)})] + [frozenset({(('b',), 12)})] * 5
    assert (ev.crossbar_stages_needed(specs, key_fields=fields).occupied ==
            ev.crossbar_stages_needed(specs, key_fields=fields,
                                      key_field_bits=[SOLID_49] +
                                                     [DDOS_12] * 5).occupied)


def test_key_field_bits_must_be_positionally_aligned_with_table_specs():
    with pytest.raises(ValueError, match="key_field_bits"):
        ev.crossbar_stages_needed([(1, 4), (1, 4)], key_field_bits=[(30,)])


def test_ternary_key_field_bits_reports_every_code_field_width():
    # A field of len(intervals) - 1 bits is what build_p4_script declares, and
    # the penalty depends on the multiset of those widths, so they come back
    # sorted rather than in emission order.
    assert ev.ternary_key_field_bits({'f': list(range(180))}) == (179,)
    assert ev.ternary_key_field_bits({'a': list(range(180)),
                                      'b': list(range(193))}) == (179, 192)
    assert ev.ternary_key_field_bits({}) == ()


# --- Task 9: the _pool_inputs / assemble_usage seam

POOL_KEYS = {
    "range_table_specs", "ternary_table_specs", "range_levels", "range_fields",
    "ternary_fields", "ternary_key_bits", "interior_stages", "emitted_features",
    "register_names", "range_entries", "range_blocks", "ternary_entries",
    "codeword_length",
}


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_pool_inputs_then_assemble_reproduces_multi_model_exactly(encoding):
    # The seam is only worth having if it is transparent: whatever
    # multi_model_memory_evaluation returns, _pool_inputs -> assemble_usage must
    # return the identical ResourceUsage. This is the property the golden
    # fixture relies on -- it is dumped at the seam and replayed through
    # assemble_usage, so a seam that drifted would silently invalidate it.
    from src.p4model.usage import assemble_usage

    features = ["f0", "f1", "f2", "f3"]
    clf_app = _tiny_forest([0, 1, 2], seed=0)
    clf_ddos = _tiny_forest([-1, 1], seed=7)

    direct = ev.multi_model_memory_evaluation(
        clf_app, clf_ddos, features, features, encoding)

    pool = ev._pool_inputs(clf_app, clf_ddos, features, features, encoding)
    assert set(pool) == POOL_KEYS
    replayed, _range_plan, _ternary_plan = assemble_usage(pool)

    assert replayed == direct


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_pool_inputs_key_field_widths_agree_with_table_spec_byte_widths(encoding):
    # The invariant crossbar_stages_needed checks internally, asserted at the
    # seam so the fixture can record BITS and reconstruct bytes safely: a
    # ternary table's byte width IS the byte-rounded sum of its key fields.
    features = ["f0", "f1", "f2", "f3"]
    pool = ev._pool_inputs(
        _tiny_forest([0, 1, 2], seed=0), _tiny_forest([-1, 1], seed=7),
        features, features, encoding)

    for (_, byte_width), fields in zip(pool["ternary_table_specs"],
                                       pool["ternary_fields"]):
        assert sum(width for _, width in fields) == byte_width
