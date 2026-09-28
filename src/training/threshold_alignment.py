from src.p4gen.build_p4_script import INFINITE, get_feature_intervals_from_thresholds
from src.training.align_budget import (_factor,
                                       _own_floor_widths,
                                       _pooled_widths,
                                       bits_to_reach,
                                       blocks_bought_by,
                                       pooled_interval_count,
                                       pooled_key_bytes, total_blocks,
                                       tree_multiplier)
from src.training.align_targets import (LOWER_EDGE, boundary_moves,
                                        boundary_moves_to, boundary_pairs,
                                        candidate_targets,
                                        hypothetical_ranges, lower_boundary,
                                        neighbour_writes)
from src.training.errors import AlignmentInvariantError
from src.training.incremental_metrics import IncrementalMetrics
from src.training.trial_selection import rel_deg
import copy
import sklearn
import numpy as np


# Hard cap on the per-feature candidate-recompute loop (C3).
#
# A CYCLE GUARD, not a tuning parameter. The loop's real stopping rule is
# `progressed`, which is an EXACT fixpoint test: a round that accepts nothing
# changed no interval tuple, so the recomputed sweep would return the
# identical candidate list with every member already retired in `seen`, and
# the next round would do zero work. Every converging run therefore exits on
# `progressed` BEFORE the cap is consulted, which is what makes a generous cap
# close to free -- an unused round costs nothing at all, since it never runs.
#
# The cap exists because termination cannot be PROVED: list length is not
# strictly decreasing per round, and even joint_interval_count -- which,
# under the corrected pooled-threshold definition (see its own docstring),
# is provably non-increasing move-by-move, since every write relocates a
# threshold to a value already present in the pooled set rather than
# introducing a new one -- can go an entire round without decreasing at all
# (see test_a_single_accepted_move_can_leave_the_joint_interval_count_flat),
# so a genuinely cycling value-pair sequence has to be caught rather than
# ruled out.
#
# Why 32 and not 8 (P3b T3 measurement, ruling P3b-6, superseded by the P3b
# Task 5 measurement below). Measured with the cap lifted to 64 so the
# numbers are true fixpoint depths rather than truncations: on a realistic
# probe (4000x17 and 3000x17 samples, 7 trees, max_depth=10,
# min_samples_leaf=5) run over 18 seed x arm configurations
# (`scripts/measure_alignment_fixpoint_depth.py`), the deepest feature
# converges in at most 10 rounds -- the first Task 3 measurement, over only
# 12 configurations, had seen a maximum of 8 and was overtaken by the wider
# probe. At a cap of 10 (or below) a run reaching that depth would abort a
# campaign split with AlignmentInvariantError for no reason at all. 32 is
# 3.2x the observed maximum of 10. This is a SAMPLE maximum over the seeds
# probed, not a proven bound -- a 19th configuration could exceed it, which
# is exactly why the cap is kept several times larger rather than pinned to
# the observed value. Smaller fixtures are far below it: 2 to 4 rounds on
# the test suite's forests.
MAX_RECOMPUTE_ROUNDS = 32


# The alignment objective. Single-valued since the 2026-09-07 cost-model
# repair: 'stages' and 'both' are retired because classification-pool stages
# are DERIVABLE from the block factor (design §4.3, pinned by
# tests/test_resource_model_golden.py::
# test_classification_stages_are_derivable_from_the_block_factor). The two
# objectives existed because blocks stepped on a plain bit sum while stages
# stepped on byte-quantised key width; the crossbar finding moved blocks into
# the byte domain too, the domains merged, and with them the justification.
#
# Kept as a tuple, and TrainConfig.align_objective kept as a field, so every
# config and manifest that recorded 'blocks' still loads unchanged.
ALIGN_OBJECTIVES = ('blocks',)


def _legacy_combined_order(intervals1, intervals2):
    """The pre-2026-09-07 order: combined interval count descending.

    Retained for the characterisation tests ONLY, so the behavioural delta of
    the switch to byte-completion ordering is measurable rather than silent.
    Not used by alignment. Do not reintroduce it as a live option -- it orders
    for a cost model the hardware does not have (design §4.2).
    """
    common = set(intervals1) & set(intervals2)
    return sorted(common,
                  key=lambda f: (-(len(intervals1.get(f, []))
                                   + len(intervals2.get(f, []))), f))


def feature_order(intervals1, intervals2, *, multiplier, widths=None,
                  floors=None, features=None):
    """The order features are offered to the alignment loop, cheapest first.

    Ranked by the BLOCKS a shed on that feature would actually buy and what it
    would cost, against the CURRENT widths:

        key = (-blocks_bought, bits_spent, -combined, feature)

    This is one edit fixing two audit gaps, and they are inseparable. Gap 2:
    the old key was `bits_to_next_byte`, a byte-domain proxy blind both to
    version_block_penalty (which depends on the width MULTISET, so no
    per-feature scalar can see it) and to range steps. Gap 1: range blocks are
    not alignment-invariant, so a feature four intervals from a free range
    block outranks a feature one bit from a byte boundary that buys nothing.
    A ranking by achieved cost is meaningless on stale widths, so the staleness
    fix is the same edit as the ranking fix.

    RECOMPUTED PER FEATURE. This used to be computed once at entry, which was
    justified by features being structurally independent in the mutation loop
    -- true of the loop, false of the COST: the bits still needed for a ternary
    step are a joint quantity across features, so after one feature sheds, the
    ranking of the rest is out of date (audit §8.2 item 9). The caller takes
    `[0]` from a fresh call per feature and passes the budget's live widths.

    A feature that can buy nothing is NOT dropped: it only loses priority.
    Among such features the pre-existing combined-count key still decides, so
    the fallback is the order the archive was produced under.

    NOT A STRICT IMPROVEMENT, and no write-up may claim it is (design D6).
    Acceptance is greedy and the accuracy guard is ONE GLOBAL anchor (the
    pre-alignment scores, spec 2026-09-28 T3), not a per-feature budget: the
    accidental gains above the start are a slack the whole run shares, so
    which feature is visited first changes which moves are still affordable
    when a later feature's turn comes.
    Ranking by blocks bought is a better HEURISTIC; individual rows may do
    worse than the byte-domain order. Audit §8.4's
    joint-dinf/M100/k17/split12 row is the precedent that this shape really
    occurs -- per-instance optimality was never claimed for a greedy heuristic
    and is not claimed here.

    The trailing feature index makes this a TOTAL order, keeping the run
    deterministic -- which train_model.py's refit assertion depends on
    (invariant 5).

    multiplier : what one step of the block factor is worth, in blocks
        (align_budget.tree_multiplier). Required: without it a range step and a
        ternary step are incomparable, which is the whole defect.
    widths, floors : optional precomputed _pooled_widths / _own_floor_widths.
        align_rf_thresholds passes the BUDGET's live widths, not its entry
        copy.
    features : which features to rank. Defaults to the common set.
    """
    if widths is None:
        widths = _pooled_widths(intervals1, intervals2)
    if floors is None:
        floors = _own_floor_widths(intervals1, intervals2)
    if features is None:
        features = set(intervals1) & set(intervals2)

    def key(feature):
        bought, spent = blocks_bought_by(widths, floors, feature, multiplier)
        combined = len(intervals1[feature]) + len(intervals2[feature])
        return (-bought, spent, -combined, feature)

    return sorted(features, key=key)


def accept_alignment(before, after, delta_rel):
    """Whether an alignment may stand, judged PER TASK (spec B.4).

    before, after : 4-tuples (acc_app, f1_app, acc_ddos, f1_ddos) -- the
    order every 4-tuple in this module uses.
    delta_rel : permitted relative-error degradation per metric, or None to
        accept unconditionally.

    Four independent guards, not an average. Averaging let a move costing DDoS
    0.009 while gaining App 0.001 through: the mean drops 0.0040, inside the old
    0.005 tolerance, while per task it gives away 22.5% of DDoS's error. That is
    the mechanism behind the measured DDoS-specific alignment tax of ~0.004 that
    is invariant to budget -- DDoS lost even when the joint arm was handed MORE
    capacity and App gained.

    No amount of gain on one metric can offset a loss on another: `all`, over
    per-metric tests, never a sum.
    """
    if delta_rel is None:
        return True
    return all(rel_deg(b, a) <= delta_rel for b, a in zip(before, after))


# DELETED 2026-09-28: ratchet(before, after), the element-wise high-water
# marks of spec B.4. Its rationale was drift under delta > 0: judged against
# the running best, each task's total drift was bounded at delta_rel. delta is
# fixed at 0 now, and the pre-alignment anchor that replaced it (spec
# 2026-09-28 T3, see align_rf_thresholds' `started_at`) bounds total drift
# anyway -- per metric, from the START -- without turning an accidental
# val_align gain into the floor for every later move.


def joint_interval_count(intervals1, intervals2):
    """Total TCAM-relevant interval count under JOINT encoding: for a feature
    both models split on, the two models' TCAM entries are pooled into one
    table keyed on that feature, so the count is the size of the COMMON
    REFINEMENT of both models' thresholds for that feature -- exactly what
    evaluation.py's multi_model_memory_evaluation builds (via
    get_feature_intervals_from_thresholds) from the pooled thresholds of the
    merged tree set. (ResourceUsage.range_entries is NOT this interval
    count -- it is the expanded PHYSICAL TCAM ROW count computed from these
    intervals; see evaluation.range_matching_resource_usage. No
    ResourceUsage field holds a bare interval count.) For a feature only one
    model splits on, there is nothing to pool, so its own interval count is
    added directly.

    This is NOT the union of the two models' interval TUPLES: pooling
    thresholds {10} and {5} on the same feature partitions it into 3 ranges
    -- (0,5),(6,10),(11,INF) -- but as tuples (0,10) != (0,5) and
    (11,INF) != (6,INF), so a tuple union overcounts to 4. The two answers
    coincide only when both models already split the feature at exactly the
    same points.

    This -- not a flat sum of each model's own interval count, and not the
    tuple union either -- is the quantity that actually shrinks when
    alignment succeeds: a successful move relocates one model's threshold to
    coincide with the other's, shrinking the pooled threshold SET for that
    feature. Alignment only ever relocates a threshold, never deletes one, so
    each model's OWN interval count never changes; a stat built from
    per-model sums alone is structurally constant and cannot reflect any TCAM
    savings at all.

    Delegates to align_budget._pooled_widths -- the per-feature decomposition
    of this exact total, already needed by the block budget -- instead
    of re-deriving the same pooled threshold set a second way. A feature's
    width is its pooled interval count minus one, so summing widths and
    adding back one per feature recovers the interval count exactly; provably
    equal to the direct computation this replaced, not merely observed equal
    -- see the two functions' docstrings for the one-to-one correspondence.
    """
    widths = _pooled_widths(intervals1, intervals2)
    return sum(widths.values()) + len(widths)


def _rank_targets(range1, range2, ranges1, ranges2, idx1, idx2, feature_idx,
                  sorted_cols1, sorted_cols2):
    """Admissible corner targets, best first (C2).

    Sorted by (gain descending, damage ascending, generation order): a target
    that sheds two bits beats one that sheds one whatever the damage, and the
    accuracy guard is what bounds the damage anyway. In the common case where
    s1 != s2, e1 != e2 and no boundary touches an edge, all four
    corners shed the SAME two bits -- each of the two boundary gaps is crossed
    exactly once whichever corner wins, and only WHICH MODEL pays for which
    gap changes -- so damage is the effective discriminator and gain only
    separates the degenerate cases.

    Damage is a max over the two models, never a sum or a mean: the same
    principle accept_alignment and rel_shortfall already enforce, that a gain
    on one task may not offset a loss on the other. Generation order is the
    final tiebreak so the ranking is a total order and the run stays
    deterministic -- which the refit assertion at train_model.py:373-377
    depends on.

    Returns (before, ranked): `before` is pooled_interval_count(ranges1,
    ranges2) at entry, unaffected by which target (if any) the caller later
    accepts, and `ranked` is [(target, after), ...] best first, where `after`
    is pooled_interval_count(hypo1, hypo2) for that target -- exactly what
    the count becomes once update_neighboring_ranges_and_index applies it,
    since both call the same neighbour_writes on the same (ranges, idx,
    old_range, new_range) whenever nothing has mutated `ranges1`/`ranges2` in
    between (true here: only an ACCEPTED move mutates them, and the caller's
    loop over `ranked` stops at the first accept). Handing both numbers back
    lets the caller's post-acceptance shed bookkeeping (the `live_widths`
    decrement) read counts already paid for while ranking, instead of
    recomputing pooled_interval_count a second time around the mutation it
    predicted.
    """
    before = pooled_interval_count(ranges1, ranges2)
    scored = []
    for order, target in enumerate(candidate_targets(range1, range2)):
        hypo1 = hypothetical_ranges(ranges1, idx1, range1, target)
        hypo2 = hypothetical_ranges(ranges2, idx2, range2, target)
        if hypo1 is None or hypo2 is None:
            continue

        moves1 = boundary_moves(range1, target)
        moves2 = boundary_moves(range2, target)
        if not moves1 and not moves2:
            continue

        after = pooled_interval_count(hypo1, hypo2)
        gain = before - after
        if gain <= 0:
            continue

        damage = 0.0
        for sorted_cols, moves in ((sorted_cols1, moves1), (sorted_cols2, moves2)):
            for old, new in moves:
                damage = max(damage,
                             shift_mass(sorted_cols[:, feature_idx], old, new))

        scored.append((-gain, damage, order, target, after))

    scored.sort()
    return before, [(target, after) for _, _, _, target, after in scored]


def align_rf_thresholds(rf1, rf2, X_val1, y_val1, X_val2, y_val2,
                        delta_rel=0.0, align_stats=None,
                        candidate_log=None):
    """
    Aligns feature ranges by adjusting boundary thresholds of pure overlapping regions.

    Parameters:
    -----------
    rf1, rf2 : RandomForestClassifier or RandomForestRegressor
        The two pretrained RandomForest models to align
    delta_rel : float or None
        Permitted relative-error degradation. None accepts every move and
        skips the accuracy evaluation entirely (the "inf" anchor).

    Returns:
    --------
    rf1_aligned, rf2_aligned : Deep copies of rf1/rf2 with aligned thresholds.
        rf1/rf2 themselves are left untouched (C8) -- the return value is the
        only way to get the aligned models; discarding it discards the
        alignment.
    """
    # C8: deepcopy before anything below reads or mutates rf1/rf2, and
    # specifically before build_prediction_cache -- its tree_predictions feed
    # IncrementalMetrics' vote matrix, so if the copy happened after that
    # call, the cache would describe the caller's forests while every
    # mutation below landed on the copies, and the accept/reject loop would
    # silently score the wrong models. ~401 KB per pair against a ~550 ms
    # fit (measured) -- negligible next to what it protects.
    rf1 = copy.deepcopy(rf1)
    rf2 = copy.deepcopy(rf2)

    # Cast ONCE. estimator.predict / decision_path each run
    # check_array(X, dtype=np.float32) internally, and the arrays arriving from
    # feature_selection are float64 -- so without this every one of the
    # thousands of calls below re-casts and re-copies.
    #
    # Exactly value-preserving for this project's data: after
    # dt_thresholds_float_to_int every threshold is an integer, and every
    # feature value is an integer clipped at INFINITE = 65535 -- both far below
    # float32's 2**24 exact-integer limit. Local copies, so the caller's arrays
    # are untouched.
    X_val1 = np.ascontiguousarray(X_val1, dtype=np.float32)
    X_val2 = np.ascontiguousarray(X_val2, dtype=np.float32)

    # One sort per model, for shift_mass. Under C2 this is no longer
    # diagnostic: it is how a candidate's predicted damage is priced, so it
    # must be available whenever the policy ranks targets. One np.sort per
    # model against a ~550 ms fit -- negligible. Per-model is correct: damage
    # to rf1 depends on X_val1's distribution, not X_val2's. Feature indices
    # line up -- trees are fit on X_*_train[:, remaining] and validated on
    # X_*_val[:, remaining], the same column space.
    sorted_cols1 = np.sort(X_val1, axis=0)
    sorted_cols2 = np.sort(X_val2, axis=0)

    threshold_index1 = build_threshold_index(rf1)

    threshold_index2 = build_threshold_index(rf2)

    intervals1 = extract_feature_intervals(rf1)
    intervals2 = extract_feature_intervals(rf2)

    with sklearn.config_context(assume_finite=True):
        tree_predictions1, node_to_samples1 = build_prediction_cache(rf1, X_val1)
        tree_predictions2, node_to_samples2 = build_prediction_cache(rf2, X_val2)

    # The per-model metric state -- vote matrix, per-sample winner, confusion
    # matrix -- seeded from the initial predictions. Only needed for the
    # accept/reject comparison below, which is skipped entirely when delta_rel
    # is None (the inf anchor); not building it there is what makes that arm
    # the cheapest, and is also why build_prediction_cache does NOT return the
    # vote matrix itself.
    #
    # Each candidate then costs O(#changed samples) instead of two full
    # validation-set passes twice over: the from-scratch
    # compute_ensemble_prediction re-counted every (tree, sample) vote and
    # re-argmaxed every sample, and accuracy_metrics paid sklearn's fixed
    # per-call validation overhead four times -- measured at 4022us per
    # accuracy_metrics call at n=4000 against 256us for the prediction it was
    # measuring. Every number produced here is bit-identical to what those
    # calls produced; see incremental_metrics' module docstring.
    metrics1 = IncrementalMetrics(tree_predictions1, rf1, y_val1, task="app")
    metrics2 = IncrementalMetrics(tree_predictions2, rf2, y_val2, task="ddos")

    # The run's starting point: the PRE-ALIGNMENT scores, in (acc_app,
    # f1_app, acc_ddos, f1_ddos) order. Every candidate is judged against
    # these four, per metric, for the WHOLE run (spec 2026-09-28 T3). They
    # used to be ratcheted up to max(marks, after) after every accepted move
    # (spec B.4's high-water marks), which at delta = 0 made each accidental
    # gain on val_align the floor for every later move. A fixed anchor still
    # bounds each metric's total drift -- at zero, at delta = 0 -- so the
    # ratchet bought nothing the anchor does not. Consequence, accepted: the
    # run now shares one accuracy slack (the accidental gains above the
    # start), so which feature is visited first can matter again -- the
    # reason feature_order's ranking is kept. §2.4's accuracy_spent is
    # measured from here to the final `current`.
    started_at = metrics1.metrics() + metrics2.metrics()
    # Last-ACCEPTED state -- the model's actual current metrics. Before any
    # candidate it coincides with started_at.
    current = started_at

    stats = align_stats if align_stats is not None else {}
    stats['attempted'] = 0
    stats['accepted'] = 0
    stats['intervals_before'] = joint_interval_count(intervals1, intervals2)

    # L -- the pooled split-threshold count, which IS the classification
    # table's codeword length (see align_budget's module docstring). An
    # interval list holds one more entry than it has thresholds, so the
    # feature count is exactly what separates the two quantities. Recorded
    # rather than derived downstream because the block cost is a step function
    # of L and nothing else, and until now L appeared in no artifact at all.
    n_features = len(set(intervals1) | set(intervals2))
    stats['codeword_before'] = stats['intervals_before'] - n_features

    # The byte domain, recorded unconditionally.
    stats['key_bytes_before'] = pooled_key_bytes(intervals1, intervals2)
    # §4.1: computed unconditionally at entry. The BLOCK factor needs them on
    # every path, because the version-block charge is a function of the width
    # MULTISET and cannot be recovered from any scalar (design §1.3). Cheap
    # either way -- one pass over at most ~15 features.
    pooled_widths = _pooled_widths(intervals1, intervals2)
    own_floor_widths = _own_floor_widths(intervals1, intervals2)

    # §4.6. The per-table block factor at entry, at exit (below), and at the
    # floor -- the best any alignment of this pair could reach, since a common
    # feature's pooled width can never drop below max(own1, own2).
    #
    # audit §8.2 item 7. `factor_*` is the per-TABLE block factor -- the ternary
    # key's own width in blocks. `total_blocks_*` is what ResourceUsage.blocks
    # charges: that factor times the tree multiplier, plus every feature's range
    # table. Two names because they are two quantities; archived CSVs carry the
    # factor under the old `blocks_*` name and must never be read as totals.
    multiplier = tree_multiplier(rf1, rf2)
    stats['factor_before'] = _factor(pooled_widths)
    stats['factor_floor'] = _factor(own_floor_widths)
    stats['total_blocks_before'] = total_blocks(pooled_widths, multiplier)
    stats['total_blocks_floor'] = total_blocks(own_floor_widths, multiplier)

    # §4.6. bits_to_reach survives, aimed at the next cheaper BLOCK factor
    # instead of the retired stage step. A factor of f - 1 is fed by f - 1
    # crossbar groups of 5.5 bytes each, so it needs key_bytes <=
    # (11 * (f - 1)) // 2. Still a documented LOWER BOUND, and now doubly so:
    # version_block_penalty can hold the factor up past that width, and the
    # cheapest bits are not necessarily the least damaging ones. Reported,
    # never enforced -- no run is skipped on it.
    block_target = (11 * (stats['factor_before'] - 1)) // 2
    stats['bits_to_reach'] = (
        bits_to_reach(pooled_widths, own_floor_widths, block_target)
        if stats['factor_before'] > 1 else None)

    # The width dict alignment actually works on, decremented per accepted move
    # so feature_order sees the CURRENT cost.
    #
    # Copied, not aliased: `pooled_widths` above is the entry snapshot that
    # `stats` and the floor comparison are computed from, and mutating it in
    # place would silently rewrite both.
    #
    # Until 2026-09-15 this dict lived inside a BlockBudget, which also decided
    # whether a candidate could be judged at a non-zero delta. Track 5 returned
    # delta_helps = FALSE and the whole tolerance axis went; every candidate is
    # now judged at the caller's delta, which the campaign always leaves at 0.
    live_widths = dict(pooled_widths)

    # Recomputed per feature against the LIVE widths (audit §8.2 item 9).
    # Computing it once at entry was justified by features being structurally
    # independent in the loop below -- true of the loop, false of the cost.
    # `remaining` and the total order inside feature_order keep this
    # deterministic, which the refit assertion depends on.
    remaining = set(intervals1) & set(intervals2)

    while remaining:
        feature_idx = feature_order(intervals1, intervals2,
                                    multiplier=multiplier,
                                    widths=live_widths,
                                    floors=own_floor_widths,
                                    features=remaining)[0]
        remaining.discard(feature_idx)

        current_ranges1 = intervals1[feature_idx]
        current_ranges2 = intervals2[feature_idx]

        # C3. Candidate ORDER, stated once because nothing documented it
        # before:
        #   features, ranked by feature_order (blocks a shed would buy,
        #     descending; bits it would cost, ascending; combined interval
        #     count, descending; feature index -- the last two only as
        #     tiebreaks now that the ranking is cost-aware, not the primary
        #     key);
        #   then ROUNDS, each recomputing the overlap list from the CURRENT,
        #     already-mutated interval lists -- this is what makes an overlap
        #     CREATED by an earlier accepted move reachable at all. Aligning
        #     range i widens its neighbours (the target is
        #     (max(s1,s2), min(e1,e2)), so whatever the aligned range gives up
        #     its neighbours take), and a widened neighbour can overlap a
        #     range in the other model that nothing overlapped before. With a
        #     single fixed overlap list those pairs were unreachable, however
        #     many times the list was re-read;
        #   then, within a round, the sweep's (i ascending, j ascending)
        #     order -- which is the old nested scan's order exactly (T1).
        #
        # Affordable only because T1 made the sweep O(n+m): a round costs one
        # linear pass over two interval lists, not a quadratic rescan.
        #
        # `seen` keys on VALUE pairs, not index pairs: an accepted move
        # rewrites tuples in place (it never inserts or deletes one), so the
        # same index pair names a different candidate in a later round, and
        # the same candidate can turn up at a different index. It is reset per
        # feature -- features are structurally independent, each owning its
        # own interval lists and its own threshold-index keys.
        #
        # `progressed` is the real stopping rule and it is EXACT, not a
        # heuristic: a round that accepts nothing changed no tuple, so the
        # recomputed sweep returns the identical list, every member of which
        # is already in `seen`, so the next round would do zero work.
        # MAX_RECOMPUTE_ROUNDS is only the backstop for genuine cycling.
        #
        # Every feature runs to that fixpoint (spec 2026-09-28 T4). A
        # per-feature early exit used to retire a feature at the end of the
        # round in which it first bought a block (audit §8.2 item 4); it left
        # later purchases on the same feature unbought (measured +16 blocks
        # on 3/935 replay rows, never worse) and was deleted.

        seen = set()
        progressed = True
        rounds = 0

        while progressed and rounds < MAX_RECOMPUTE_ROUNDS:
            progressed = False
            rounds += 1

            overlaps = find_partially_overlapping_ranges(current_ranges1,
                                                         current_ranges2)

            # Apply alignment for each overlap
            for (idx1, idx2) in overlaps:
                # Re-read: an accepted move earlier in THIS round may have
                # rewritten either tuple.
                range1 = current_ranges1[idx1]
                range2 = current_ranges2[idx2]

                if range1 == range2:
                    continue

                if (range1, range2) in seen:
                    continue
                seen.add((range1, range2))

                # Three named, unconditional correctness checks. The ratio test
                # that used to gate admission below it is gone (D4, Task 7) --
                # it was a separate, heuristic concern; these are not, and must
                # never again be disableable by the same knob.
                if not still_overlaps(range1, range2):
                    continue
                if not structurally_alignable(range1, range2):
                    continue

                # The three checks above are what admission actually is
                # (design §6.1); calculate_range_overlap and endpoint_ratio,
                # which used to be computed here purely for candidate_log, were
                # pruned 2026-09-15 (task 14) once nothing compared them to a
                # threshold any more.
                pooled_before, ranked_targets = _rank_targets(
                    range1, range2, current_ranges1, current_ranges2,
                    idx1, idx2, feature_idx, sorted_cols1, sorted_cols2)

                for target, pooled_after in ranked_targets:
                    if not target_is_well_formed(target):
                        continue

                    # Purely diagnostic -- only computed when a candidate_log
                    # is actually requested.
                    mass1 = mass2 = None
                    if candidate_log is not None:
                        mass1 = max(shift_mass(sorted_cols1[:, feature_idx], old, new)
                                    for old, new in ((range1[0], target[0]),
                                                     (range1[1], target[1])))
                        mass2 = max(shift_mass(sorted_cols2[:, feature_idx], old, new)
                                    for old, new in ((range2[0], target[0]),
                                                     (range2[1], target[1])))

                    modifications1 = adjust_range_boundaries(
                        rf1, feature_idx, range1, target, threshold_index1)
                    modifications2 = adjust_range_boundaries(
                        rf2, feature_idx, range2, target, threshold_index2)

                    if not modifications1 and not modifications2:
                        # P5: adjust_range_boundaries declined every move.
                        # Under c1c2 _rank_targets has already dropped these,
                        # but the legacy single-target path still reaches here
                        # and there is nothing to evaluate, restore or undo.
                        continue

                    undo_info1 = update_cache_for_modifications(
                        rf1, X_val1, tree_predictions1, node_to_samples1, modifications1)
                    undo_info2 = update_cache_for_modifications(
                        rf2, X_val2, tree_predictions2, node_to_samples2, modifications2)

                    stats['attempted'] += 1

                    # IncrementalMetrics' ordering contract: apply reads the NEW
                    # per-tree predictions out of tree_predictions and the OLD
                    # ones out of undo_info, so it must run AFTER
                    # update_cache_for_modifications and BEFORE any
                    # undo_cache_update.
                    mtoken1 = metrics1.apply(tree_predictions1, undo_info1)
                    mtoken2 = metrics2.apply(tree_predictions2, undo_info2)
                    after = metrics1.metrics() + metrics2.metrics()
                    accepted = accept_alignment(started_at, after, delta_rel)

                    if candidate_log is not None:
                        candidate_log.append({
                            'feature_idx': int(feature_idx),
                            'round': rounds,
                            'range1': tuple(range1),
                            'range2': tuple(range2),
                            'target': tuple(target),
                            'error_app': 1.0 - current[0],
                            'error_ddos': 1.0 - current[2],
                            'shift_mass_1': mass1,
                            'shift_mass_2': mass2,
                            # Local, immediate-effect degradation: current is the
                            # actual model state right before THIS candidate, as
                            # opposed to the fixed pre-alignment anchor
                            # started_at (which accept_alignment above correctly
                            # uses instead -- spec 2026-09-28 T3 -- and which
                            # this diagnostic does not affect). Comparing a local
                            # physical bound (shift_mass) against a cumulative
                            # quantity would be apples-to-oranges.
                            'rel_deg': tuple(rel_deg(b, a)
                                             for b, a in zip(current, after)),
                            'accepted': bool(accepted),
                        })

                    if not accepted:
                        restore_thresholds(rf1, modifications1)
                        restore_thresholds(rf2, modifications2)
                        undo_cache_update(tree_predictions1, node_to_samples1, undo_info1)
                        undo_cache_update(tree_predictions2, node_to_samples2, undo_info2)
                        # The metric state is the fifth structure a rejected
                        # candidate has to restore. revert is independent of
                        # undo_cache_update (it restores from its own stored copy,
                        # not from tree_predictions), so the order here is free --
                        # but it must happen on EVERY reject, or the next
                        # candidate's `after` is measured from a model state
                        # that no longer exists.
                        metrics1.revert(mtoken1)
                        metrics2.revert(mtoken2)
                        # C2: the pair is not dead yet -- try the next-ranked
                        # corner. With a single target this falls straight out
                        # of the loop, exactly as the old `continue` did.
                        continue

                    # Only an ACCEPTED move can change the candidate set: a
                    # reject restores thresholds, both caches, the metric
                    # state and the interval lists, so a rescan after one
                    # would return exactly the list already being iterated.
                    progressed = True
                    stats['accepted'] += 1
                    current = after

                    # Realised shed for THIS move, measured on the one feature
                    # that moved. `pooled_before`/`pooled_after` come straight
                    # from `_rank_targets`, which already computed them while
                    # ranking this exact target (see its own docstring for why
                    # that is exact, not approximate) -- recomputing
                    # joint_interval_count here would be O(total thresholds)
                    # paid thousands of times per trial; these two lists hold
                    # 25-50 entries. Pinned against the whole-run total by
                    # test_the_per_move_sheds_sum_to_the_whole_runs_shed.
                    update_neighboring_ranges_and_index(
                        current_ranges1, idx1, range1, target,
                        feature_idx, threshold_index1)
                    update_neighboring_ranges_and_index(
                        current_ranges2, idx2, range2, target,
                        feature_idx, threshold_index2)

                    live_widths[feature_idx] -= pooled_before - pooled_after

                    # First acceptance wins: the ranking already put the
                    # cheapest admissible corner first, and the tuples this
                    # pair was named by no longer exist.
                    break

        if progressed and rounds > 1:
            # Truncated while still accepting moves: the loop never reached a
            # fixpoint, so the result depends on where it was cut off. That is
            # an invariant violation, not a slower run.
            #
            # `rounds > 1` is the "recomputation was actually running" test:
            # at MAX_RECOMPUTE_ROUNDS == 1 the loop is DELIBERATELY reduced to
            # the single pre-C3 pass (that is the configuration the regression
            # gate in test_threshold_alignment.py pins against pre-C3 golden
            # values), and truncation there is the point rather than an
            # anomaly.
            raise AlignmentInvariantError(
                'feature {} did not reach an alignment fixpoint within '
                'MAX_RECOMPUTE_ROUNDS={} rounds'.format(
                    feature_idx, MAX_RECOMPUTE_ROUNDS))

    intervals1_after = extract_feature_intervals(rf1)
    intervals2_after = extract_feature_intervals(rf2)
    stats['intervals_after'] = joint_interval_count(intervals1_after,
                                                    intervals2_after)
    stats['codeword_after'] = stats['intervals_after'] - n_features
    stats['key_bytes_after'] = pooled_key_bytes(intervals1_after, intervals2_after)
    widths_after = _pooled_widths(intervals1_after, intervals2_after)
    stats['factor_after'] = _factor(widths_after)
    stats['total_blocks_after'] = total_blocks(widths_after, multiplier)

    # §2.4: what this run gave away, in the same units accept_alignment uses,
    # priced as a MAX across the four metrics rather than a sum or a mean --
    # the standard this module already applies in accept_alignment's all(),
    # and in _rank_targets' damage. Recorded unconditionally
    # (design spec: "Unchanged, still written with exactly today's values")
    # so a campaign always has this stat to compare runs against. With the
    # delta_align axis deleted (2026-09-15) alignment only ever accepts free
    # moves, so this is 0.0 on every campaign run; it is kept as the check
    # that this really is so, rather than as a swept quantity's record.
    stats['accuracy_spent'] = max(0.0, max(rel_deg(b, a)
                                           for b, a in zip(started_at, current)))

    return rf1, rf2 #, alignment_stats


def extract_feature_intervals(rf):
    """Feature intervals for `rf`, keyed by feature INDEX.

    Delegates to the generator's own get_feature_intervals_from_thresholds so
    the two cannot diverge again (C1). That function is key-agnostic -- it needs
    only (key, threshold) tuples sorted by key then threshold -- so feature
    indices work exactly as feature names do.

    Why delegation rather than a patch: this module used to skip splits at
    threshold 0 while the generator (deliberately, see build_p4_script.py's own
    comment) does not. Alignment therefore optimised a partition that was not
    the partition the TCAM cost was computed from, and its block savings were
    mis-targeted wherever a zero split existed. The dedup rules also differed
    -- a set() here, skip-if-equal-to-previous there -- equivalent then, free to
    drift later.
    """
    feature_thresholds = []

    for estimator in rf.estimators_:
        tree = estimator.tree_
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:  # Not a leaf node
                feature_thresholds.append((int(tree.feature[node_idx]),
                                           int(round(tree.threshold[node_idx]))))

    # get_feature_intervals_from_thresholds relies on the list being sorted by
    # (key, threshold) -- that is how it dedups and how it chains intervals.
    feature_thresholds.sort()

    return get_feature_intervals_from_thresholds(feature_thresholds)


def build_threshold_index(rf):
    """
    Build a dictionary mapping (feature_idx, threshold) -> [(tree_idx, node_idx), ...]
    """
    threshold_index = {}
    
    for tree_idx, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_
        
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:  # Not a leaf node
                feature_idx = tree.feature[node_idx]
                threshold = int(round(tree.threshold[node_idx]))
                
                key = (feature_idx, threshold)
                if key not in threshold_index:
                    threshold_index[key] = []
                threshold_index[key].append((tree_idx, node_idx))

    return threshold_index


def build_prediction_cache(rf, X_val):
    """
    Build cache of per-tree predictions and decision paths.
    Returns:
        - tree_predictions: (n_trees, n_samples) array of per-tree class predictions
        - node_to_samples: dict mapping (tree_idx, node_idx) -> array of sample indices
    """
    n_samples = X_val.shape[0]
    n_trees = len(rf.estimators_)

    tree_predictions = np.zeros((n_trees, n_samples), dtype=np.intp)
    node_to_samples = {}

    for tree_idx, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_

        # Class INDICES, not labels. A RandomForest's sub-estimators are fit on
        # encoded y, so estimator.predict already returns indices -- the
        # rf.classes_[...] round-trip here existed only to be undone by a
        # per-element dict lookup in compute_ensemble_prediction.
        tree_predictions[tree_idx] = estimator.predict(X_val).astype(np.intp)

        # decision_path returns CSR, and slicing ONE column of a CSR matrix is
        # O(nnz) -- doing it per node made this O(n_nodes x nnz). One tocsc()
        # makes the whole node -> samples inversion a single O(nnz) pass, since
        # each node is then one contiguous CSC column.
        decision_path = estimator.decision_path(X_val).tocsc()
        decision_path.sort_indices()

        # Convert to node -> samples mapping for non-leaf nodes only
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:  # Not a leaf
                start, end = decision_path.indptr[node_idx], decision_path.indptr[node_idx + 1]
                node_to_samples[(tree_idx, node_idx)] = decision_path.indices[start:end].copy()

    return tree_predictions, node_to_samples


def compute_ensemble_prediction(tree_predictions, rf):
    """Hard majority vote over per-tree class indices, returning class labels.

    Deliberately NOT rf.predict, which averages predict_proba (a SOFT vote):
    the switch votes hard, via generate_voting_code's exact-match table whose
    const entries are mode() over the per-tree class indices. Ties break toward
    the smallest class index in both -- np.argmax here, mode() there.

    Vectorised as one bincount over a sample-major offset array. The previous
    pure-Python double loop ran ~n_trees x n_samples interpreted iterations
    (~28k at n_trees=7, 4000 samples) twice per alignment candidate.

    THIS FUNCTION IS THE TEST ORACLE, AND THAT IS WHY IT IS STILL HERE.
    P3b T2b moved the alignment loop onto IncrementalMetrics, which maintains
    the same hard vote incrementally, so this has no production caller left --
    but it is deliberately kept as the from-scratch reference that the
    incremental path is checked against, in
    test_incremental_metrics.py's equivalence property tests and in
    test_threshold_alignment.py's switch_predict / vote_winner agreement
    tests. Deleting it as dead code deletes the only independent statement of
    what the incremental state is supposed to compute, and takes those tests
    with it. If it ever regains a production caller, say so here; do not
    remove the oracle role.
    """
    n_trees, n_samples = tree_predictions.shape
    n_classes = rf.n_classes_

    # Offset each sample into its own length-n_classes slot, then count the
    # whole (n_trees, n_samples) block in a single pass.
    offsets = np.arange(n_samples, dtype=np.intp) * n_classes
    flat = (offsets[None, :] + tree_predictions).ravel()
    votes = np.bincount(flat, minlength=n_samples * n_classes).reshape(n_samples, n_classes)

    return rf.classes_[np.argmax(votes, axis=1)]


def find_partially_overlapping_ranges(ranges1, ranges2):
    """Two-pointer merge sweep, O(n+m), replacing a nested O(n*m) scan.

    Both inputs must be sorted and internally non-overlapping -- exactly what
    extract_feature_intervals / get_feature_intervals_from_thresholds
    produce: a gap-free tiling (0,t1),(t1+1,t2),...,(tk+1,INFINITE).

    Intervals are INCLUSIVE integer ranges, so two overlap iff
    max(s1, s2) <= min(e1, e2) (spec 2026-09-28 T1). A one-value interval
    (t, t) is an ordinary interval under that test and forms pairs like any
    other: (6, 6) against (4, 9) offers target (6, 6), which moves the second
    model's cuts 3 -> 5 and 9 -> 6. Until 2026-09-28 the sweep skipped every
    `end <= start` interval and used the strict test `s1 < e2 and s2 < e1`,
    so no such pair was ever tried. An interval with end < start cannot exist
    in a valid tiling, so one is an AlignmentInvariantError, not a skip.
    Identical tuples are still not a pair: there is nothing to align.

    Checked against the nested O(n*m) inclusive scan on random tilings that
    include one-value intervals, order included
    (test_the_sweep_matches_the_nested_scan_on_random_gap_free_tilings).

    Retirement invariant: at the top of each iteration, every reportable pair
    (a,b) with a < i or b < j has already been emitted.
      - end1 < end2 (retire i): for any j' > j, the tiling gives
        start_j' > end2 > end1, so ranges1[i] can reach nothing past j.
      - end2 < end1: symmetric.
      - end1 == end2: both retirements are independently justified (for
        j' > j, start_j' > end2 == end1 kills any pair with ranges1[i]; for
        i' > i, start_i' > end1 == end2 kills any pair with ranges2[j]).
        Retiring only i (as below) merely re-tests ranges2[j] against
        ranges1[i+1], which cannot overlap it; it skips nothing.
      - Order: both pointers are monotone and every iteration advances
        exactly one, so emission is lexicographic in (i, j) -- exactly the
        nested loop's order, which align_stats and candidate_log rely on.
    """
    overlaps = []
    i = j = 0
    while i < len(ranges1) and j < len(ranges2):
        s1, e1 = ranges1[i]
        s2, e2 = ranges2[j]
        if e1 < s1 or e2 < s2:
            raise AlignmentInvariantError(
                'inverted interval {} in a tiling'.format(
                    (s1, e1) if e1 < s1 else (s2, e2)))
        if max(s1, s2) <= min(e1, e2) and not (s1 == s2 and e1 == e2):
            overlaps.append((i, j))
        if e1 <= e2:      # retire whichever ends first -- it cannot meet anything later
            i += 1
        else:
            j += 1
    return overlaps


def shift_mass(sorted_col, old_thr, new_thr):
    """Fraction of validation rows that change side when a split moves.

    sklearn sends x <= threshold left, so the affected set is (lo, hi]. This is
    the quantity the endpoint ratio (a pure diagnostic, pruned 2026-09-15
    once its `endpoint_ratio_cap` admission role was gone) used to be a proxy
    for -- and the proxy was exact only when the feature is log-distributed.
    It is O(log n) per candidate against the O(n_trees x n_samples) oracle.
    """
    lo, hi = (old_thr, new_thr) if old_thr <= new_thr else (new_thr, old_thr)
    return float(np.searchsorted(sorted_col, hi, 'right')
                 - np.searchsorted(sorted_col, lo, 'right')) / len(sorted_col)


def still_overlaps(range1, range2):
    """Do these two intervals, as they read RIGHT NOW, actually overlap?

    find_partially_overlapping_ranges is computed once per ROUND and its pairs
    are re-read -- not recomputed -- inside the round, so an accepted move
    earlier in the same round can have rewritten either tuple. A pair that no
    longer overlaps yields an empty-intersection target such as (701, 700),
    which inverts the tiling when committed: traced live as
    `AlignmentInvariantError: (0, 700) missing from threshold_index` (audit
    §8.3).

    Until 2026-09-14 this job was done only INCIDENTALLY, by the since-pruned
    `calculate_range_overlap(...) < overlap_threshold` returning 0.0 for a
    non-overlapping pair -- which is why setting that threshold to 0.0 disabled
    a correctness check along with the similarity heuristic. Unconditional now,
    and named, so the two can never be disabled together again.

    INCLUSIVE since 2026-09-28 (spec T1), like the sweep: intervals are
    inclusive integer ranges, so (6, 6) overlaps (4, 9) and (0, 100) overlaps
    (100, 200). The stale (581, 700) & (701, 1005) pair is still rejected --
    it shares no value -- and target_is_well_formed remains the last guard.
    """
    (start1, end1), (start2, end2) = range1, range2
    return max(start1, start2) <= min(end1, end2)


def structurally_alignable(range1, range2):
    """Is this pair admissible at all, before its corners are priced?

    A boundary whose source or target is an EDGE -- LOWER_EDGE at the bottom,
    INFINITE at the top -- is never moved (adjust_range_boundaries' own
    guard). Where exactly one side sat on one, nothing used to veto the PAIR,
    so update_neighboring_ranges_and_index wrote the shrunk boundary into
    `ranges` while the model kept splitting where it was and the index kept
    the true key: the C5 bug. neighbour_writes now mirrors every per-boundary
    refusal into the effective range, which is what keeps the three
    structures consistent; this veto is a coarser second line.

    NARROWED 2026-09-28 (spec T2) to what invariant 2 needs: a corner is
    admissible iff its moves touch no edge, and the per-boundary refusal
    already guarantees that for the lower edge. So a pair where exactly one
    side is a first interval (0, c) is no longer vetoed -- e.g. (0, 10) vs
    (6, 15), corner (0, 10): the second model's 15 -> 10 moves and its lower
    boundary, which would have gone to the edge, stays. Before T2 the lower
    edge and a real cut at 0 were the same value, so the old clause also
    froze every pair touching a real cut at 0.

    The INFINITE-side clause is kept as it was: the spec narrows only the
    first-interval veto. dataset.py clips every feature at INFINITE, so a
    (m, INFINITE) interval is common, not exotic.
    """
    (_, max1), (_, max2) = range1, range2
    return (max1 == INFINITE) == (max2 == INFINITE)


def target_is_well_formed(target):
    """Is this target a non-empty, non-inverted interval?

    NEW as of 2026-09-14. neighbour_writes validates that NEIGHBOURING
    intervals do not invert; nothing validated the target itself, which is
    exactly what the §8.3 crash committed. Cheap, unconditional, and the last
    line of defence if a future candidate generator offers a corner
    still_overlaps did not already rule out.
    """
    low, high = target
    return low <= high


def adjust_range_boundaries(rf, feature_idx, source_range, target_range, threshold_index):
    """Move `source_range`'s boundaries to `target_range`'s in the forest.

    Each boundary is the cut it stands for -- `start - 1` below, `end` above
    (align_targets.boundary_pairs) -- and moves only if it changes and
    neither its source nor its target is an EDGE (LOWER_EDGE below, INFINITE
    above; align_targets.boundary_moves_to). Since spec 2026-09-28 T2 a cut
    at 0 is an ordinary cut here, and a lower target of LOWER_EDGE is
    refused rather than written as a threshold of 0.
    """
    modifications = []

    # Min side (edge LOWER_EDGE) and max side (edge INFINITE): identical
    # guard, identical AlignmentInvariantError, identical mutation loop.
    for threshold_source, threshold_target, edge in boundary_pairs(
            source_range, target_range):
        if boundary_moves_to(threshold_source, threshold_target, edge):

            if (feature_idx, threshold_source) not in threshold_index:
                raise AlignmentInvariantError(
                    '{} missing from threshold_index'.format((feature_idx, threshold_source)))

            for tree_idx, node_idx in threshold_index[(feature_idx, threshold_source)]:
                tree = rf.estimators_[tree_idx].tree_
                modifications.append((tree_idx, node_idx, threshold_source))
                tree.threshold[node_idx] = threshold_target

    return modifications


def _get_descendant_nodes(tree, node_idx):
    """Get all descendant node indices (including the node itself)."""
    descendants = []
    stack = [node_idx]
    while stack:
        n = stack.pop()
        descendants.append(n)
        left = tree.children_left[n]
        right = tree.children_right[n]
        if left >= 0:
            stack.append(left)
        if right >= 0:
            stack.append(right)
    return descendants


def update_cache_for_modifications(rf, X_val, tree_predictions, node_to_samples, modifications):
    """
    Update cache after threshold modifications.

    Updates node_to_samples for modified nodes and their descendants,
    and properly merges affected samples with unaffected samples.

    Returns:
        undo_info: dict with 'predictions' and 'node_samples' to pass to undo function
    """
    # Arrays, not Python sets of NumPy scalars: np.unique on a concatenation is
    # one C-level pass, where set.update was boxing every index.
    per_tree_sample_arrays = {}
    for tree_idx, node_idx, _ in modifications:
        if (tree_idx, node_idx) in node_to_samples:
            per_tree_sample_arrays.setdefault(tree_idx, []).append(
                node_to_samples[(tree_idx, node_idx)])

    trees_to_repredict = {
        tree_idx: np.unique(np.concatenate(arrays))
        for tree_idx, arrays in per_tree_sample_arrays.items()
    }

    # Capture old state for undo
    undo_info = {
        'predictions': {},  # tree_idx -> (sample_indices, old_predictions)
        'node_samples': {}  # (tree_idx, node_idx) -> old_samples
    }

    for tree_idx, sample_indices in trees_to_repredict.items():
        if sample_indices.size == 0:
            continue

        # Save old predictions
        undo_info['predictions'][tree_idx] = (
            sample_indices.copy(),
            tree_predictions[tree_idx, sample_indices].copy()
        )

        # tree_.decision_path bypasses estimator.predict/decision_path's
        # sklearn-API input validation (check_array, tag lookups) -- dead
        # weight here since X_val is already float32 C-contiguous (cast once
        # at entry, see align_rf_thresholds). One CSR traversal instead of
        # two: the leaf each sample lands on is the last node on its path,
        # and the tree's own per-leaf class distribution gives the hard-vote
        # prediction from that -- bit-identical to estimator.predict on this
        # array (RandomForest sub-estimators have n_outputs_ == 1, so
        # tree.value's middle axis is a length-1 dummy). Verified equal to
        # estimator.predict/.decision_path across all trees of the
        # regression fixture; ~3.4x faster per tree-call.
        X_subset = X_val[sample_indices]
        tree = rf.estimators_[tree_idx].tree_
        raw_path = tree.decision_path(X_subset)
        leaves = raw_path.indices[raw_path.indptr[1:] - 1]
        new_predictions = np.argmax(tree.value[leaves, 0, :], axis=1).astype(np.intp)
        tree_predictions[tree_idx, sample_indices] = new_predictions

        decision_path = raw_path.tocsc()
        decision_path.sort_indices()

        # Find all nodes that need updating: modified nodes and their descendants
        modified_nodes_in_tree = {node_idx for t_idx, node_idx, _ in modifications if t_idx == tree_idx}
        nodes_to_update = set()
        for mod_node in modified_nodes_in_tree:
            nodes_to_update.update(_get_descendant_nodes(tree, mod_node))

        # Update node_to_samples for modified nodes and descendants only
        for node_idx in nodes_to_update:
            if tree.feature[node_idx] < 0:  # Skip leaf nodes
                continue

            key = (tree_idx, node_idx)

            # Save old state for undo (only once per key)
            if key not in undo_info['node_samples']:
                undo_info['node_samples'][key] = node_to_samples[key].copy()

            # Get which affected samples now pass through this node
            start, end = decision_path.indptr[node_idx], decision_path.indptr[node_idx + 1]
            local_indices = decision_path.indices[start:end]
            node_to_samples[key] = sample_indices[local_indices]

    return undo_info


def undo_cache_update(tree_predictions, node_to_samples, undo_info):
    """Reverse the effects of update_cache_for_modifications."""
    # Restore predictions
    for tree_idx, (sample_indices, old_predictions) in undo_info['predictions'].items():
        tree_predictions[tree_idx, sample_indices] = old_predictions
    
    # Restore node_to_samples
    for key, old_samples in undo_info['node_samples'].items():
        node_to_samples[key] = old_samples


def restore_thresholds(rf, modifications):
    """
    Restore the exact thresholds that were modified.
    
    Parameters:
    -----------
    rf : RandomForest model
        The model to restore thresholds to
    modifications : list of tuples
        List of (tree_idx, node_idx, original_threshold) to restore
    """
    for tree_idx, node_idx, original_threshold in modifications:
        rf.estimators_[tree_idx].tree_.threshold[node_idx] = original_threshold


def update_neighboring_ranges_and_index(ranges, target_idx, old_range, new_range,
                                        feature_idx, threshold_index):
    """Apply a boundary move to `ranges` and the threshold index.

    The arithmetic itself lives in align_targets.neighbour_writes, which C2's
    admissibility filter also calls -- the predicate and the mutator cannot
    drift because there is only one of them. This is now all-or-nothing: the
    inversion is detected before any write lands, where the previous version
    raised from the middle of the neighbour loop leaving earlier neighbours
    already rewritten. The raise itself is unchanged.
    """
    effective_range, writes, inverted = neighbour_writes(
        ranges, target_idx, old_range, new_range)

    if inverted is not None:
        (range_min, range_max), (bad_min, bad_max) = inverted
        raise AlignmentInvariantError(
            'neighboring range {} would invert to ({}, {}) while '
            'absorbing the boundary move of target range {} -> {} '
            'for feature {}'.format(
                (range_min, range_max), bad_min, bad_max,
                old_range, new_range, feature_idx))

    if effective_range == old_range:
        return

    ranges[target_idx] = effective_range

    # The index follows the EFFECTIVE range, which neighbour_writes built
    # with the same per-boundary edge guards adjust_range_boundaries applies
    # (align_targets.boundary_moves_to) -- so a boundary the model refused to
    # move is not re-keyed either, whether the refusal was its source or its
    # target being an edge.
    old_min, old_max = old_range
    effective_min, effective_max = effective_range
    if effective_min != old_min:
        update_threshold_index(threshold_index, feature_idx,
                               lower_boundary(old_min),
                               lower_boundary(effective_min))
    if effective_max != old_max:
        update_threshold_index(threshold_index, feature_idx, old_max,
                               effective_max)

    for i, tup in writes:
        ranges[i] = tup


def update_threshold_index(threshold_index, feature_idx, old_threshold, new_threshold):
    """
    Update a single threshold in the index.
    """

    if (feature_idx, old_threshold) not in threshold_index:
        raise AlignmentInvariantError(
            '{} missing from threshold_index'.format((feature_idx, old_threshold)))

    nodes = threshold_index.pop((feature_idx, old_threshold))
    if (feature_idx, new_threshold) in threshold_index:
        existing = set(threshold_index[(feature_idx, new_threshold)])
        existing.update(nodes)
        threshold_index[(feature_idx, new_threshold)] = list(existing)
    else:
        threshold_index[(feature_idx, new_threshold)] = nodes


# DELETED 2026-09-15: crossed_a_boundary(stats), and with it
# align_with_policy's commit-or-rollback.
#
# crossed_a_boundary asked `stats['factor_after'] < stats['factor_before']` --
# did this run buy a cheaper block FACTOR? -- and align_with_policy used it to
# undo a speculative run: BlockBudget's reachability test proved a cheaper
# factor was REACHABLE, not that it would be REACHED, so a run could pay
# accuracy and buy nothing, and the rollback re-ran the same pair at delta = 0
# and kept that instead (recorded as stats['rolled_back']).
#
# Both were unreachable without a non-zero delta -- align_with_policy returned
# the speculative result untouched whenever spent_budget was False -- and
# Track 5's pre-registered live-Optuna trial returned delta_helps = FALSE
# (mean_d000 0.7956173344395895 vs mean_d020 0.7861922400433382,
# cells_favouring_d020 14/24). Review finding 2.1 -- that this test priced the
# factor where feature_order and the per-feature early exit price total blocks
# -- is closed by deleting the test, not by repricing it.


def align_with_policy(rf1, rf2, X_val1, y_val1, X_val2, y_val2, *,
                      delta_rel=0.0,
                      align_stats=None, candidate_log=None):
    """align_rf_thresholds, with `align_stats` cleared first.

    Kept under its own name because train_model.py and
    scripts/replay_alignment.py both call it, and because a caller passing a
    reused stats dict must not be able to read a previous run's keys back out
    of it.

    It used to be C1's commit-or-rollback wrapper: run at the configured delta,
    and if budget was genuinely spent while the run bought no block, discard
    that result and re-run at delta = 0 keeping only the free moves. There is
    no configured delta any more (2026-09-15, Track 5's delta_helps = FALSE),
    so the speculation the rollback protected against cannot happen and there
    is nothing left to undo -- which is why this is a single call and no longer
    costs 2x alignment runtime on the runs where the speculation failed.
    """
    stats = align_stats if align_stats is not None else {}
    stats.clear()
    return align_rf_thresholds(
        rf1, rf2, X_val1, y_val1, X_val2, y_val2,
        delta_rel=delta_rel,
        align_stats=stats, candidate_log=candidate_log)