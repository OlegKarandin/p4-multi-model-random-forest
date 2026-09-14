from src.p4gen.build_p4_script import INFINITE, get_feature_intervals_from_thresholds
from src.training.align_budget import (BlockBudget, _factor,
                                       _own_floor_widths,
                                       _pooled_widths,
                                       bits_to_next_byte, bits_to_reach,
                                       codeword_floor,
                                       key_bytes_floor, pooled_interval_count,
                                       pooled_key_bytes, total_blocks,
                                       tree_multiplier)
from src.training.align_targets import (boundary_moves, candidate_targets,
                                        hypothetical_ranges, neighbour_writes)
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


def feature_order(intervals1, intervals2, *, widths=None, floors=None):
    """The order features are offered to the alignment loop.

    Byte completion, cheapest first: features that can actually complete a
    crossbar byte come first, the rest follow by combined interval count
    descending. This is the BLOCK-CORRECT order and, since 2026-09-07, the
    only one -- codeword_bytes_to_blocks depends on sum(ceil(w_f / 8)), so a shed
    bit can only change the factor by completing a byte on some feature.
    Shedding bits that complete no byte is the waste the repair exists to stop.

    A feature that cannot complete a byte is NOT dropped: it only loses
    priority. Byte distance participates in the key only for reachable
    features, so among unreachable ones the combined-count key still decides.

    Key: (0 if reachable else 1, to_next_byte if reachable else 0, -combined, f)
    reachable := bits_to_next_byte(w_f) <= w_f - max(own1_f, own2_f)

    The trailing feature index makes it a TOTAL order, keeping the run
    deterministic -- which train_model.py:373-377's refit assertion depends on
    (invariant 5).

    Computed ONCE at entry, which is correct rather than a shortcut: features
    are structurally independent in the loop below -- each owns its interval
    lists, `seen` resets per feature, and no accepted move on one feature
    changes another's widths.

    widths, floors : optional precomputed _pooled_widths / _own_floor_widths.
        align_rf_thresholds passes its own copies, already built for the block
        factor, instead of paying for the identical O(n_features) pass twice.
    """
    common = set(intervals1) & set(intervals2)
    if widths is None:
        widths = _pooled_widths(intervals1, intervals2)
    if floors is None:
        floors = _own_floor_widths(intervals1, intervals2)

    def key(feature):
        step = bits_to_next_byte(widths[feature])
        reachable = step <= widths[feature] - floors[feature]
        combined = len(intervals1[feature]) + len(intervals2[feature])
        return (0 if reachable else 1, step if reachable else 0, -combined, feature)

    return sorted(common, key=key)


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


def ratchet(before, after):
    """Element-wise high-water marks (spec B.4).

    Per task, not on the mean. With only the mean ratcheted, a sequence where
    App improves while DDoS degrades keeps the mean flat, no single move trips
    the guard, and DDoS drifts arbitrarily far. Independent marks bound each
    task's total drift from ITS OWN best at delta_rel, independently of the
    other task -- strictly stronger than the per-move test alone.
    """
    return tuple(max(b, a) for b, a in zip(before, after))


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
    s1 != s2, e1 != e2 and neither boundary sits on a sentinel, all four
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
    lets the caller's post-acceptance shed bookkeeping (BlockBudget.note_shed)
    read counts already paid for while ranking, instead of recomputing
    pooled_interval_count a second time around the mutation it predicted.
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

    # Four independent high-water marks, in (acc_app, f1_app, acc_ddos,
    # f1_ddos) order.
    marks = metrics1.metrics() + metrics2.metrics()
    # Last-ACCEPTED state -- the model's actual current metrics, as opposed
    # to marks' running per-task max. Before any candidate, both coincide.
    current = marks
    # The run's starting point, kept separate from `marks` because `marks`
    # ratchets upward and would understate what a run gave away. §2.4's
    # accuracy_spent is measured from HERE to the final `current`.
    started_at = list(marks)

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
    stats['codeword_floor'] = codeword_floor(intervals1, intervals2)
    stats['rolled_back'] = False

    # The byte domain, recorded unconditionally.
    stats['key_bytes_before'] = pooled_key_bytes(intervals1, intervals2)
    stats['key_bytes_floor'] = key_bytes_floor(intervals1, intervals2)
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

    # §4.1. The budget prices what ResourceUsage charges: the per-table block
    # factor over the pooled width dict, gated against the floor width vector
    # -- the best case any alignment of this pair could reach.
    budget = BlockBudget(pooled_widths, own_floor_widths, delta_rel)

    sorted_features = feature_order(intervals1, intervals2,
                                    widths=pooled_widths, floors=own_floor_widths)

    for feature_idx in sorted_features:
        current_ranges1 = intervals1[feature_idx]
        current_ranges2 = intervals2[feature_idx]

        # C3. Candidate ORDER, stated once because nothing documented it
        # before:
        #   features, descending by combined interval count (unchanged);
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

                # Computed for candidate_log only: nothing compares it to a
                # threshold any more. The three checks above are what admission
                # actually is (design §6.1).
                overlap_ratio = calculate_range_overlap(range1, range2)

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

                    effective_delta = budget.delta_for_candidate()

                    # IncrementalMetrics' ordering contract: apply reads the NEW
                    # per-tree predictions out of tree_predictions and the OLD
                    # ones out of undo_info, so it must run AFTER
                    # update_cache_for_modifications and BEFORE any
                    # undo_cache_update.
                    mtoken1 = metrics1.apply(tree_predictions1, undo_info1)
                    mtoken2 = metrics2.apply(tree_predictions2, undo_info2)
                    after = metrics1.metrics() + metrics2.metrics()
                    accepted = accept_alignment(marks, after, effective_delta)

                    if candidate_log is not None:
                        candidate_log.append({
                            'feature_idx': int(feature_idx),
                            'round': rounds,
                            'range1': tuple(range1),
                            'range2': tuple(range2),
                            'target': tuple(target),
                            'overlap_ratio': float(overlap_ratio),
                            'endpoint_ratio': float(endpoint_ratio(range1, range2)),
                            'error_app': 1.0 - current[0],
                            'error_ddos': 1.0 - current[2],
                            'shift_mass_1': mass1,
                            'shift_mass_2': mass2,
                            # Local, immediate-effect degradation: current is the
                            # actual model state right before THIS candidate, as
                            # opposed to marks' cumulative per-task high-water mark
                            # (which accept_alignment above correctly uses instead --
                            # that ratchet is deliberate, spec B.4, and unaffected
                            # by this diagnostic). Comparing a local physical bound
                            # (shift_mass) against a cumulative quantity would be
                            # apples-to-oranges.
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
                        # but it must happen on EVERY reject, or the ratchet starts
                        # comparing against a model state that no longer exists.
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
                    marks = ratchet(marks, after)
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

                    budget.note_shed(feature_idx, pooled_before - pooled_after)

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
    stats['spent_budget'] = budget.spent_budget

    # §2.4: what this run gave away, in the same units accept_alignment uses,
    # priced as a MAX across the four metrics rather than a sum or a mean --
    # the standard this module already applies in accept_alignment's all(),
    # in ratchet, and in _rank_targets' damage. Recorded unconditionally
    # (design spec: "Unchanged, still written with exactly today's values")
    # so a campaign always has this stat to compare runs against, regardless
    # of which objective or delta_align produced them.
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

    Verified against the nested scan over 200 000 random tilings (including
    ones containing a (0,0) interval): 0 mismatches, order included.

    Retirement invariant: at the top of each iteration, every reportable pair
    (a,b) with a < i or b < j has already been emitted.
      - end1 < end2 (retire i): for any j' > j, disjointness gives
        start_j' > end2 > end1, so ranges1[i] can reach nothing past j.
      - end2 < end1: symmetric.
      - end1 == end2: both retirements are independently justified (for
        j' > j, start_j' > end2 == end1 kills any pair with ranges1[i]; for
        i' > i, start_i' > end1 == end2 kills any pair with ranges2[j]).
        Retiring only i (as below) merely re-tests an already-emitted pair
        next iteration; it cannot skip anything.
      - Degenerate skip: advancing i past an end1 <= start1 interval without
        advancing j loses nothing -- that interval participates in no pair,
        and ranges2[j] is re-tested against ranges1[i+1] next iteration.
      - Order: both pointers are monotone and every iteration advances at
        least one, so emission is lexicographic in (i, j) -- exactly the
        nested loop's order, which align_stats and candidate_log rely on.

    The end <= start filter also excludes (0,0) intervals -- consistent, not
    a bug: calculate_range_overlap already vetoes any pair where exactly one
    side starts at 0, and adjust_range_boundaries refuses to move a boundary
    at 0, so a (0,0) interval could never be aligned anyway.

    KNOWN FUTURE WORK, deliberately preserved here rather than fixed: the same
    filter also excludes (t,t) intervals for t > 0, and those are NOT always
    no-ops -- e.g. range1=(6,6), range2=(4,9) has target (6,6): side 1 doesn't
    move, but side 2's (4,9) -> (6,6) is a real move never attempted today.
    Pre-existing behaviour; this task is a pure refactor, not a fix.
    """
    overlaps = []
    i = j = 0
    while i < len(ranges1) and j < len(ranges2):
        s1, e1 = ranges1[i]
        s2, e2 = ranges2[j]
        if e1 <= s1:
            i += 1; continue
        if e2 <= s2:
            j += 1; continue
        if s1 < e2 and s2 < e1 and not (s1 == s2 and e1 == e2):
            overlaps.append((i, j))
        if e1 <= e2:      # retire whichever ends first -- it cannot meet anything later
            i += 1
        else:
            j += 1
    return overlaps


def endpoint_ratio(range1, range2):
    """The larger of the two endpoint ratios -- the quantity the historic
    `endpoint_ratio_cap = 5` thresholds. A pure diagnostic after Task 7; kept
    so the instrumented run can quantify how often it disagreed with the oracle.
    """
    min1, max1 = range1
    min2, max2 = range2

    ratios = [1.0]
    if min1 and min2:
        ratios.append(max(min1, min2) / min(min1, min2))
    if max1 and max2:
        ratios.append(max(max1, max2) / min(max1, max2))
    return max(ratios)


def shift_mass(sorted_col, old_thr, new_thr):
    """Fraction of validation rows that change side when a split moves.

    sklearn sends x <= threshold left, so the affected set is (lo, hi]. This is
    the quantity the endpoint ratio was a proxy for -- and the proxy is exact
    only when the feature is log-distributed. It is O(log n) per candidate
    against the O(n_trees x n_samples) oracle.
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

    Until 2026-09-14 this job was done only INCIDENTALLY, by
    `calculate_range_overlap(...) < overlap_threshold` returning 0.0 for a
    non-overlapping pair -- which is why setting that threshold to 0.0 disabled
    a correctness check along with the similarity heuristic. Unconditional now,
    and named, so the two can never be disabled together again.
    """
    (start1, end1), (start2, end2) = range1, range2
    return max(start1, start2) < min(end1, end2)


def structurally_alignable(range1, range2):
    """Can adjust_range_boundaries move these boundaries at all?

    A boundary sitting ON a sentinel -- 0 at the bottom, INFINITE at the top --
    is never moved (adjust_range_boundaries' own guard). Where exactly one side
    sits on one, nothing vetoed the PAIR, so
    update_neighboring_ranges_and_index wrote the shrunk boundary into `ranges`
    while the model kept splitting at the sentinel and the index kept the true
    key: the C5 bug. dataset.py clips every feature at INFINITE, so a
    (m, INFINITE) interval is common, not exotic.

    Extracted verbatim from calculate_range_overlap's two early returns, whose
    0.0 made them indistinguishable from "no overlap".
    """
    (min1, max1), (min2, max2) = range1, range2
    return ((min1 == 0) == (min2 == 0)
            and (max1 == INFINITE) == (max2 == INFINITE))


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


def calculate_range_overlap(range1, range2):
    """Overlap ratio between two ranges; 0.0 also means 'vetoed'.

    NOTE this function's 0.0 return is overloaded: it means both "no overlap"
    and "vetoed". The zero-side and INFINITE-side vetoes below are structural
    -- as of Task 7 they are duplicated (not delegated) by
    `structurally_alignable`, which is what align_rf_thresholds actually
    consults; this function's own vetoes are now dead code from admission's
    point of view, kept only because the ratio itself is still computed and
    returned for candidate_log. The old endpoint-ratio-cap heuristic
    pre-filter that used to live here is gone as of Task 7; align_rf_thresholds
    does not veto candidates on shift_mass either (removed in P3 Task 8), and
    since Task 7 it does not compare this ratio to a threshold at all -- the
    returned value is uninterpreted, a diagnostic only.
    """
    min1, max1 = range1
    min2, max2 = range2

    # Early exit if either range starts at 0 but not both
    if (min1 == 0) != (min2 == 0):
        return 0.0

    # C5: the mirror of the above at the top end. adjust_range_boundaries
    # refuses to move a threshold at INFINITE (its max-side guard) exactly as
    # it refuses to move one at 0 -- but nothing vetoed the PAIR, so
    # update_neighboring_ranges_and_index wrote the shrunk boundary into
    # `ranges` while the model kept splitting at INFINITE and the index kept
    # the true key. Every later decision on that feature was then wrong, and
    # nothing covered the tail. dataset.py clips every feature at INFINITE, so
    # a (m, INFINITE) interval is common, not exotic.
    if (max1 == INFINITE) != (max2 == INFINITE):
        return 0.0

    # Calculate intersection
    intersection_start = max(min1, min2)
    intersection_end = min(max1, max2)
    
    # No overlap if intersection is invalid
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection_length = intersection_end - intersection_start
    
    # Calculate lengths and return ratio
    range1_length = max1 - min1
    range2_length = max2 - min2
    
    return intersection_length / max(range1_length, range2_length)


def calculate_target_range(range1, range2):
    """Calculate the target range for alignment"""
    return (max(range1[0], range2[0]), min(range1[1], range2[1]))


def adjust_range_boundaries(rf, feature_idx, source_range, target_range, threshold_index):
    """
    Adjust thresholds using the pre-built index
    """
    source_min, source_max = source_range
    target_min, target_max = target_range
    
    threshold_source_min = source_min - 1 if source_min > 0 else source_min
    threshold_target_min = target_min - 1 if target_min > 0 else target_min

    threshold_source_max = source_max
    threshold_target_max = target_max
        
    modifications = []

    # Min side (sentinel 0) and max side (sentinel INFINITE): identical guard
    # shape, identical AlignmentInvariantError, identical mutation loop --
    # differing only in which sentinel refuses the move and which
    # source/target pair is used.
    for threshold_source, threshold_target, sentinel in (
        (threshold_source_min, threshold_target_min, 0),
        (threshold_source_max, threshold_target_max, INFINITE),
    ):
        if threshold_source != threshold_target and threshold_source != sentinel:

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

    old_min, old_max = old_range
    new_min, new_max = new_range
    threshold_old_min = old_min - 1 if old_min > 0 else old_min
    threshold_new_min = new_min - 1 if new_min > 0 else new_min
    if threshold_old_min != threshold_new_min and threshold_old_min != 0:
        update_threshold_index(threshold_index, feature_idx,
                               threshold_old_min, threshold_new_min)
    if old_max != new_max and old_max != INFINITE:
        update_threshold_index(threshold_index, feature_idx, old_max, new_max)

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


def crossed_a_boundary(stats):
    """Did this run buy a cheaper block FACTOR?

    Still the factor, not total blocks, and deliberately so: audit §8.2 item 2
    shows this test is insufficient (a run can pass it while landing on
    strictly more total blocks than the free alternative for the same pair),
    but the fix is only reachable when spent_budget is True, which requires
    delta > 0. Whether this becomes `total_blocks_after < total_blocks_before`
    or is DELETED along with the whole rollback is exactly what the
    live-Optuna delta trial decides (design §3, §8.2).
    """
    return stats['factor_after'] < stats['factor_before']


def align_with_policy(rf1, rf2, X_val1, y_val1, X_val2, y_val2, *,
                      delta_rel=0.0,
                      align_stats=None, candidate_log=None):
    """align_rf_thresholds with C1's commit-or-rollback guarantee.

    `factor(current) > factor(floor)` proves a cheaper block factor is
    REACHABLE, not that it will be REACHED: the candidate generator can run dry
    mid-flight, leaving a run that paid accuracy and bought nothing -- exactly
    the waste C1 exists to remove. So: run at the configured delta; if budget
    was genuinely spent and the run bought no block (`crossed_a_boundary`),
    discard that result and re-run the same pair at delta = 0, keeping only the
    free moves.

    A whole-function retry rather than in-loop state surgery, because
    align_rf_thresholds is already a pure function of (models, validation data,
    params) and already deep-copies its inputs (C8) -- so re-running it from
    the caller's untouched forests IS the rollback. Cost is 2x alignment
    runtime on exactly the runs where the speculation failed, 1x everywhere
    else.

    Keeping a "best intermediate state" instead was considered and rejected
    (design §4.5): a run that bought no block bought nothing by definition, so
    there is nothing to keep, and the accuracy is better returned.
    """
    stats = align_stats if align_stats is not None else {}
    stats.clear()
    speculative = align_rf_thresholds(
        rf1, rf2, X_val1, y_val1, X_val2, y_val2,
        delta_rel=delta_rel,
        align_stats=stats, candidate_log=candidate_log)

    if not stats['spent_budget'] or crossed_a_boundary(stats):
        return speculative

    # Spent and bought nothing. Redo at delta = 0 and keep THAT.
    if candidate_log is not None:
        # The speculative run's candidates never happened as far as the
        # returned models are concerned, so its log must not be reported
        # alongside them.
        del candidate_log[:]
    stats.clear()
    result = align_rf_thresholds(
        rf1, rf2, X_val1, y_val1, X_val2, y_val2,
        delta_rel=0.0,
        align_stats=stats, candidate_log=candidate_log)
    stats['rolled_back'] = True
    return result