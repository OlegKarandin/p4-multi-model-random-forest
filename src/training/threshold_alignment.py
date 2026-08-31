from src.p4gen.build_p4_script import INFINITE, get_feature_intervals_from_thresholds
from src.p4gen.evaluation import band_factor
from src.training.align_budget import (BandBudget, StageBudget, _own_floor_widths,
                                       _pooled_widths,
                                       bits_to_next_byte, bits_to_reach,
                                       byte_width, codeword_floor,
                                       key_bytes_floor, pooled_interval_count,
                                       pooled_key_bytes, stage_step_target,
                                       ternary_stages)
from src.training.align_targets import (boundary_moves, candidate_targets,
                                        hypothetical_ranges, neighbour_writes)
from src.training.errors import AlignmentInvariantError
from src.training.incremental_metrics import IncrementalMetrics
from src.training.trial_selection import rel_deg
import collections
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


# What the shed bits are AIMED at -- an objective axis, not another policy
# name. The two concerns were always independent: 'blocks' is today's
# behaviour and the default everywhere, 'stages' aims at the step that
# reduces ceil(T / floor(64/B)), and 'both' runs BOTH orderings, rolls each
# back independently, and keeps whichever ranks better (_rank_key). Only
# align_with_policy understands 'both'; align_rf_thresholds validates against
# _SINGLE_PASS_OBJECTIVES.
ALIGN_OBJECTIVES = ('blocks', 'stages', 'both')

# The objectives a SINGLE alignment pass can run. 'both' is not one of them:
# one pass uses one feature order, so 'both' only ever made sense one layer
# up, in align_with_policy, which expands it into these two.
_SINGLE_PASS_OBJECTIVES = ('blocks', 'stages')


def feature_order(intervals1, intervals2, objective):
    """The order features are offered to the alignment loop.

    objective='blocks' reproduces the pre-existing order exactly: combined
    interval count descending, over the set of common features.

    Otherwise: features that can actually complete a byte come first,
    cheapest first; the rest follow in the pre-existing order. A feature that
    cannot complete a byte is NOT dropped -- its bits still shrink L and buy
    blocks -- it only loses priority. Byte distance participates in the key
    only for reachable features, so among unreachable ones the pre-existing
    key still decides.

    Key: (0 if reachable else 1, to_next_byte if reachable else 0, -combined, f)
    reachable := bits_to_next_byte(w_f) <= w_f - max(own1_f, own2_f)

    The trailing feature index makes it a total order, keeping the run
    deterministic -- which train_model.py:373-377's refit assertion depends on.

    Computed ONCE at entry, which is correct rather than a shortcut: features
    are structurally independent in the loop below -- each owns its interval
    lists, `seen` resets per feature, and no accepted move on one feature
    changes another's widths.
    """
    common = set(intervals1) & set(intervals2)
    if objective == 'blocks':
        return sorted(common,
                      key=lambda f: len(intervals1.get(f, [])) + len(intervals2.get(f, [])),
                      reverse=True)

    widths = _pooled_widths(intervals1, intervals2)
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
    """
    common = set(intervals1) & set(intervals2)

    pooled = []
    for f in common:
        thresholds = set()
        for intervals in (intervals1[f], intervals2[f]):
            thresholds.update(hi for _, hi in intervals if hi != INFINITE)
        pooled.extend((f, t) for t in thresholds)
    pooled.sort()

    total = sum(len(v) for v in
                get_feature_intervals_from_thresholds(pooled).values())
    total += sum(len(v) for f, v in intervals1.items() if f not in common)
    total += sum(len(v) for f, v in intervals2.items() if f not in common)
    return total


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

        gain = before - pooled_interval_count(hypo1, hypo2)
        if gain <= 0:
            continue

        damage = 0.0
        for sorted_cols, moves in ((sorted_cols1, moves1), (sorted_cols2, moves2)):
            for old, new in moves:
                damage = max(damage,
                             shift_mass(sorted_cols[:, feature_idx], old, new))

        scored.append((-gain, damage, order, target))

    scored.sort()
    return [target for _, _, _, target in scored]


# §2.2: the read-heavy setup align_rf_thresholds does before its first
# candidate. It depends only on the ORIGINAL, unaligned forests and the
# validation data -- never on feature order or delta_rel -- so two runs over
# the same pair can share it instead of each paying for it.
#
# Safe to build from the caller's forests and use inside a deep copy of them:
# build_threshold_index returns (feature_idx, threshold) -> [(tree_idx,
# node_idx)] and build_prediction_cache returns an array plus (tree_idx,
# node_idx) -> sample indices. Every one of those is pure index/value data
# holding NO reference to an estimator object, so it stays valid for any
# structurally-identical copy. Had any of them held estimator references this
# sharing would be unsound.
_SharedSetup = collections.namedtuple('_SharedSetup', [
    'X_val1', 'X_val2', 'sorted_cols1', 'sorted_cols2',
    'threshold_index1', 'threshold_index2', 'intervals1', 'intervals2',
    'tree_predictions1', 'tree_predictions2',
    'node_to_samples1', 'node_to_samples2'])


def _build_shared_setup(rf1, rf2, X_val1, X_val2):
    """The objective-independent half of align_rf_thresholds' setup.

    Read-only with respect to rf1/rf2 -- it never mutates the caller's
    forests, so C8's guarantee is unaffected.
    """
    X_val1 = np.ascontiguousarray(X_val1, dtype=np.float32)
    X_val2 = np.ascontiguousarray(X_val2, dtype=np.float32)
    with sklearn.config_context(assume_finite=True):
        tree_predictions1, node_to_samples1 = build_prediction_cache(rf1, X_val1)
        tree_predictions2, node_to_samples2 = build_prediction_cache(rf2, X_val2)
    return _SharedSetup(
        X_val1=X_val1, X_val2=X_val2,
        sorted_cols1=np.sort(X_val1, axis=0), sorted_cols2=np.sort(X_val2, axis=0),
        threshold_index1=build_threshold_index(rf1),
        threshold_index2=build_threshold_index(rf2),
        intervals1=extract_feature_intervals(rf1),
        intervals2=extract_feature_intervals(rf2),
        tree_predictions1=tree_predictions1, tree_predictions2=tree_predictions2,
        node_to_samples1=node_to_samples1, node_to_samples2=node_to_samples2)


def _thaw(state):
    """Per-arm mutable copies of `state`'s mutable members.

    tree_predictions is mutated IN PLACE by update_cache_for_modifications, so
    it needs a real array copy. node_to_samples' values are REPLACED rather
    than mutated (see update_cache_for_modifications and undo_cache_update),
    so a shallow dict copy isolates an arm correctly. threshold_index's and
    intervals' list values are both mutated, so each list is copied. X_val and
    sorted_cols are read-only and are shared as they are.
    """
    copy_of_lists = lambda d: {k: list(v) for k, v in d.items()}
    return (copy_of_lists(state.threshold_index1),
            copy_of_lists(state.threshold_index2),
            copy_of_lists(state.intervals1), copy_of_lists(state.intervals2),
            state.tree_predictions1.copy(), state.tree_predictions2.copy(),
            dict(state.node_to_samples1), dict(state.node_to_samples2))


def align_rf_thresholds(rf1, rf2, X_val1, y_val1, X_val2, y_val2,
                        overlap_threshold=0.5, delta_rel=0.0, align_stats=None,
                        candidate_log=None, *, align_objective='blocks',
                        _state=None):
    """
    Aligns feature ranges by adjusting boundary thresholds of pure overlapping regions.

    Parameters:
    -----------
    rf1, rf2 : RandomForestClassifier or RandomForestRegressor
        The two pretrained RandomForest models to align
    overlap_threshold : float, default=0.5
        Minimum overlap ratio required to consider ranges similar enough to align
    delta_rel : float or None
        Permitted relative-error degradation. None accepts every move and
        skips the accuracy evaluation entirely (the "inf" anchor).
    align_objective : one of _SINGLE_PASS_OBJECTIVES, default 'blocks'
        What the shed bits are aimed at. 'blocks' is the pre-existing
        behaviour exactly. 'stages' orders features so that bits complete
        whole crossbar bytes, and spends the tolerance while a paying stage
        step is still reachable. 'both' is not accepted here -- only
        align_with_policy understands it, expanding it into one run of each
        single-pass objective and keeping whichever ranks better.
    _state : INTERNAL. A _SharedSetup from _build_shared_setup, or None.
        None -- the default and what every public caller passes -- builds the
        setup from scratch, exactly as before. Given, the objective-independent
        setup is taken from it (cheap per-arm copies) instead of rebuilt. Exists
        only so align_with_policy can share one build across the two arms of
        align_objective='both'; it is a performance parameter and must never be
        observable in the result (pinned by
        test_the_shared_state_path_is_observationally_identical).

    Returns:
    --------
    rf1_aligned, rf2_aligned : Deep copies of rf1/rf2 with aligned thresholds.
        rf1/rf2 themselves are left untouched (C8) -- the return value is the
        only way to get the aligned models; discarding it discards the
        alignment.
    """
    if align_objective not in _SINGLE_PASS_OBJECTIVES:
        raise ValueError('align_objective must be one of {}, got {!r}'.format(
            _SINGLE_PASS_OBJECTIVES, align_objective))
    # C8: deepcopy before anything below reads or mutates rf1/rf2, and
    # specifically before build_prediction_cache -- its tree_predictions feed
    # IncrementalMetrics' vote matrix, so if the copy happened after that
    # call, the cache would describe the caller's forests while every
    # mutation below landed on the copies, and the accept/reject loop would
    # silently score the wrong models. ~401 KB per pair against a ~550 ms
    # fit (measured) -- negligible next to what it protects.
    rf1 = copy.deepcopy(rf1)
    rf2 = copy.deepcopy(rf2)

    if _state is None:
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
    else:
        # Shared with the other arm: read-only members are used as they are,
        # mutable ones are copied so this pass cannot corrupt the other's.
        X_val1, X_val2 = _state.X_val1, _state.X_val2
        sorted_cols1, sorted_cols2 = _state.sorted_cols1, _state.sorted_cols2
        (threshold_index1, threshold_index2, intervals1, intervals2,
         tree_predictions1, tree_predictions2,
         node_to_samples1, node_to_samples2) = _thaw(_state)

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

    # The byte domain, recorded unconditionally. T is the number of
    # classification tables -- one per tree, both models
    # (build_p4_script.py:636-659) -- and is constant for the run, since
    # alignment relocates thresholds and never changes the forests.
    n_tables = len(rf1.estimators_) + len(rf2.estimators_)
    stats['key_bytes_before'] = pooled_key_bytes(intervals1, intervals2)
    stats['key_bytes_floor'] = key_bytes_floor(intervals1, intervals2)
    stats['ternary_stages_before'] = ternary_stages(stats['key_bytes_before'],
                                                    n_tables)
    stats['stage_target'] = stage_step_target(stats['key_bytes_before'], n_tables)
    stats['bits_to_reach'] = (
        bits_to_reach(_pooled_widths(intervals1, intervals2),
                      _own_floor_widths(intervals1, intervals2),
                      stats['stage_target'])
        if stats['stage_target'] is not None else None)

    budget = BandBudget(stats['codeword_before'], stats['codeword_floor'],
                        delta_rel)

    stage_budget = None
    if align_objective != 'blocks':
        stage_budget = StageBudget(stats['key_bytes_before'],
                                   stats['key_bytes_floor'], delta_rel, n_tables)

    # 'stages' with no target has nothing to chase, so it falls back to the
    # block order rather than reordering for a step that cannot be bought.
    # 'both' is not reachable here -- align_with_policy expands it into two
    # runs of this function, one per single-pass objective (§2.1).
    order_objective = align_objective
    if align_objective == 'stages' and stats['stage_target'] is None:
        order_objective = 'blocks'

    sorted_features = feature_order(intervals1, intervals2, order_objective)

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

                overlap_ratio = calculate_range_overlap(range1, range2)

                if overlap_ratio < overlap_threshold:
                    continue

                targets = _rank_targets(
                    range1, range2, current_ranges1, current_ranges2,
                    idx1, idx2, feature_idx, sorted_cols1, sorted_cols2)

                for target in targets:
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

                    # Spending is the OR of the two budgets. BandBudget closes
                    # EXACTLY when a band is crossed -- i.e. just after
                    # shedding ~44 bits, which is when a byte is most likely
                    # to be one or two bits away. Measured on real runs: the
                    # band gate is closed while a byte is still reachable in
                    # 1 of 24 cells at entry but 6 of 24 at exit, and one of
                    # those six carries ~19% of the whole grid's stage
                    # opportunity. Evaluating this at entry would have cut the
                    # gate as dead weight.
                    effective_delta = budget.delta_for_candidate()
                    if (stage_budget is not None and effective_delta == 0.0
                            and delta_rel != 0.0):
                        effective_delta = stage_budget.delta_for_candidate()

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
                    # that moved. Recomputing joint_interval_count here would be
                    # O(total thresholds) paid thousands of times per trial;
                    # these two lists hold 25-50 entries. Pinned against the
                    # whole-run total by
                    # test_the_per_move_sheds_sum_to_the_whole_runs_shed.
                    pooled_before = pooled_interval_count(current_ranges1,
                                                          current_ranges2)

                    update_neighboring_ranges_and_index(
                        current_ranges1, idx1, range1, target,
                        feature_idx, threshold_index1)
                    update_neighboring_ranges_and_index(
                        current_ranges2, idx2, range2, target,
                        feature_idx, threshold_index2)

                    pooled_after = pooled_interval_count(current_ranges1,
                                                         current_ranges2)
                    budget.note_shed(pooled_before - pooled_after)
                    if stage_budget is not None:
                        # An interval list holds one more entry than it has
                        # thresholds, so the width is the count minus one.
                        # Only this feature moved, so only its byte-rounded
                        # width can have changed.
                        stage_budget.note_shed_bytes(
                            byte_width(pooled_before - 1)
                            - byte_width(pooled_after - 1))

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
    stats['ternary_stages_after'] = ternary_stages(stats['key_bytes_after'],
                                                   n_tables)
    stats['spent_budget'] = budget.spent_budget or (
        stage_budget is not None and stage_budget.spent_budget)

    # §2.4: what this run gave away, in the same units accept_alignment uses,
    # priced as a MAX across the four metrics rather than a sum or a mean --
    # the standard this module already applies in accept_alignment's all(),
    # in ratchet, and in _rank_targets' damage. Recorded on every objective:
    # 'both' ranks on it, and single-objective runs need it so a campaign has
    # something to compare a 'both' run against.
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


def calculate_range_overlap(range1, range2):
    """Overlap ratio between two ranges; 0.0 also means 'vetoed'.

    NOTE this function's 0.0 return is overloaded: it means both "no overlap"
    and "vetoed". The zero-side and INFINITE-side vetoes below are structural
    (adjust_range_boundaries cannot move those boundaries at all). The old
    endpoint-ratio-cap heuristic pre-filter that used to live here is gone as
    of Task 7; align_rf_thresholds does not veto candidates on shift_mass
    either (removed in P3 Task 8), so no heuristic pre-filter remains --
    only the two structural vetoes below.
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

        X_subset = X_val[sample_indices]
        new_predictions = rf.estimators_[tree_idx].predict(X_subset).astype(np.intp)
        tree_predictions[tree_idx, sample_indices] = new_predictions

        tree = rf.estimators_[tree_idx].tree_
        decision_path = rf.estimators_[tree_idx].decision_path(X_subset).tocsc()
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


def crossed_a_boundary(stats, objective, n_tables):
    """Did this run buy anything the objective was aiming at?

    Under 'blocks' this is the pre-existing test exactly. Under the 'stages'
    objective it also accepts a stage step -- compared on STAGES, never on
    tables-per-stage: fit rising from 2 to 3 at T=4 leaves stage_depth
    unchanged, and keeping such a run would reproduce in the byte domain
    precisely the failure this rollback exists to prevent.
    """
    if band_factor(stats['codeword_after']) < band_factor(stats['codeword_before']):
        return True
    if objective == 'blocks':
        return False
    return (ternary_stages(stats['key_bytes_after'], n_tables)
            < ternary_stages(stats['key_bytes_before'], n_tables))


def _rank_key(stats, objective):
    """§2.4: how 'both' picks between two rollback-corrected arms.

    Stages, then blocks, then accuracy, then a fixed tiebreak. That priority
    follows the recorded cost model rather than taste: block headroom is
    large and rarely binds, while the 64-byte ternary crossbar cap typically
    does, so a stage saved is worth preferring over a block saved where the
    two trade off. accuracy_spent lands last because it separates two arms
    that reached the SAME place -- which is exactly the gap measured at
    (M=25, k=9), where both objectives crossed the identical block boundary
    and one paid 1.64pp more app accuracy for it.

    The blocks tier compares band_factor(codeword_after), never raw
    codeword_after -- band_factor is the step function blocks actually costs
    against (crossed_a_boundary, one function above, already compares it this
    way). Two arms can differ in raw codeword_after while sitting in the
    identical band, in which case neither spent bits on anything a real
    joint_blocks count would ever see; ranking on the raw value there would
    let 'both' spend real accuracy_spent to win a purely cosmetic codeword
    difference. Measured directly in the 2026-08-31 replay validation: all 9
    of its codeword-differing cells shared one band_factor on both sides, so
    joint_blocks never moved in any of them.

    The trailing constant is not decoration: on an exact tie the run must
    still be deterministic, because train_model.py:373-377 refits the winning
    trial rather than caching it. Same reason feature_order carries a trailing
    feature index and _rank_targets carries generation order.
    """
    return (stats['ternary_stages_after'],
            band_factor(stats['codeword_after']),
            stats['accuracy_spent'],
            0 if objective == 'blocks' else 1)


def _run_one_arm(rf1, rf2, X_val1, y_val1, X_val2, y_val2, *, objective,
                 overlap_threshold, delta_rel, state, candidate_log):
    """One objective's complete commit-or-rollback cycle, on a fresh stats dict.

    This is align_with_policy's original body verbatim, parameterised by which
    single-pass objective to run and which shared setup to run it against.
    Extracted so align_objective='both' can invoke it twice and get two
    independently rollback-corrected results -- §2.3's ordering constraint:
    correct each arm FIRST, rank SECOND. Ranking speculative results would let
    an arm that spent accuracy and crossed nothing win on paper, which is
    precisely the failure C1's rollback exists to prevent, moved one level up.
    """
    stats = {}
    speculative = align_rf_thresholds(
        rf1, rf2, X_val1, y_val1, X_val2, y_val2,
        overlap_threshold=overlap_threshold, delta_rel=delta_rel,
        align_stats=stats, candidate_log=candidate_log,
        align_objective=objective, _state=state)

    if not stats['spent_budget']:
        return speculative, stats

    n_tables = len(rf1.estimators_) + len(rf2.estimators_)
    if crossed_a_boundary(stats, objective, n_tables):
        return speculative, stats

    # Spent and crossed nothing. Redo at delta = 0 and keep THAT.
    if candidate_log is not None:
        # The speculative run's candidates never happened as far as the
        # returned models are concerned, so its log must not be reported
        # alongside them.
        del candidate_log[:]
    stats.clear()
    result = align_rf_thresholds(
        rf1, rf2, X_val1, y_val1, X_val2, y_val2,
        overlap_threshold=overlap_threshold, delta_rel=0.0,
        align_stats=stats, candidate_log=candidate_log,
        align_objective=objective, _state=state)
    stats['rolled_back'] = True
    return result, stats


def align_with_policy(rf1, rf2, X_val1, y_val1, X_val2, y_val2, *,
                      overlap_threshold=0.5, delta_rel=0.0,
                      align_stats=None, candidate_log=None,
                      align_objective='blocks'):
    """align_rf_thresholds with C1's commit-or-rollback guarantee.

    `band_target(L) >= floor` (and its stage-domain counterpart) proves the
    next boundary is REACHABLE, not that it will be REACHED: the candidate
    generator can run dry mid-flight, leaving a run that paid accuracy and
    bought nothing -- exactly the waste C1 exists to remove, just narrower.
    So: run at the configured delta; if budget was genuinely spent and the
    run crossed no boundary the objective was aiming at
    (`crossed_a_boundary`), discard that result and re-run the same pair at
    delta = 0, keeping only the free moves.

    align_objective : one of ALIGN_OBJECTIVES, default 'blocks'. For 'blocks'
        and 'stages' it is forwarded unchanged to align_rf_thresholds (via
        _run_one_arm) on both the speculative and (if needed) the delta=0
        rerun, and it also decides what "crossed a boundary" means here:
        under 'blocks' only a block-band drop keeps the speculative run;
        under 'stages' a stage step ALSO keeps it, even without a band drop
        (crossed_a_boundary does the check). For 'both', align_with_policy
        never forwards 'both' itself down to align_rf_thresholds -- it
        expands it into two single-pass runs, one 'blocks' and one 'stages',
        each independently taken through the same commit-or-rollback cycle
        described below; see the dual-run paragraph.

    This is a whole-function retry rather than in-loop state surgery because
    align_rf_thresholds is already a pure function of (models, validation data,
    params) and already deep-copies its inputs (C8), so re-running it from the
    caller's untouched forests IS the rollback. For a single-pass objective
    ('blocks' or 'stages') cost is 2x alignment runtime on exactly the runs
    where the speculation failed, and 1x everywhere else. For 'both', cost is
    roughly 2x baseline -- one shared setup, then two arms each run once --
    and can climb higher still if either arm also needs its own delta=0
    retry; the shared setup is what keeps this near 2x rather than 3-4x, and
    a Task 8 validation replay measured it at ~1.82-1.84x in practice.

    Under 'both', align_with_policy builds one shared setup
    (_build_shared_setup) from the caller's untouched forests and runs BOTH
    'blocks' and 'stages' against it as independently rollback-corrected arms
    via _run_one_arm -- each arm gets its own commit-or-rollback cycle, so
    neither can win by comparison alone against a speculative result that
    spent accuracy and crossed nothing (see _run_one_arm's docstring). The two
    corrected arms are then ranked by _rank_key (stages, then block cost,
    then accuracy spent, then a fixed tiebreak) and the winner is returned.

    Makes "accuracy is never spent for nothing" a property of the code rather
    than a measured hope: a run that paid the configured tolerance but crossed
    none of the boundaries its objective was aiming at is discarded and
    replaced by the free-moves-only result, regardless of which policy,
    objective, or criterion is doing the measuring. The winning stats also
    always carry 'objective_used' (which single-pass objective produced the
    returned result -- for 'blocks'/'stages' this is just align_objective
    itself; for 'both' it is whichever arm won) and 'arms_differed' (whether
    the two arms actually reached different end states, or a 'both' run just
    paid twice to re-derive the answer a single objective would have given
    for free) -- set on every objective, not just 'both', so a campaign
    comparing runs across objectives always has something to read.
    """
    stats = align_stats if align_stats is not None else {}

    if align_objective != 'both':
        result, arm_stats = _run_one_arm(
            rf1, rf2, X_val1, y_val1, X_val2, y_val2,
            objective=align_objective, overlap_threshold=overlap_threshold,
            delta_rel=delta_rel, state=None, candidate_log=candidate_log)
        stats.clear()
        stats.update(arm_stats)
        stats['objective_used'] = align_objective
        stats['arms_differed'] = False
        return result

    # §2.2: one setup build, shared by both arms. Built from the caller's
    # ORIGINAL, untouched forests -- each arm still deep-copies them itself.
    state = _build_shared_setup(rf1, rf2, X_val1, X_val2)

    arms = {}
    for objective in _SINGLE_PASS_OBJECTIVES:
        # Each arm needs its own log: only the winner's candidates actually
        # happened as far as the returned models are concerned.
        arm_log = [] if candidate_log is not None else None
        models, arm_stats = _run_one_arm(
            rf1, rf2, X_val1, y_val1, X_val2, y_val2,
            objective=objective, overlap_threshold=overlap_threshold,
            delta_rel=delta_rel, state=state, candidate_log=arm_log)
        arms[objective] = (models, arm_stats, arm_log)

    # Recorded before ranking, and it is the measurement §5 turns on: if the
    # two arms never reach different end states, the dual run is paying twice
    # to re-derive one answer and 'both' should not enter a campaign at all.
    compared = ('codeword_after', 'key_bytes_after', 'ternary_stages_after',
                'accuracy_spent')
    blocks_stats, stages_stats = arms['blocks'][1], arms['stages'][1]
    differed = any(blocks_stats[k] != stages_stats[k] for k in compared)

    winner = min(_SINGLE_PASS_OBJECTIVES,
                 key=lambda o: _rank_key(arms[o][1], o))
    models, winning_stats, winning_log = arms[winner]

    stats.clear()
    stats.update(winning_stats)
    stats['objective_used'] = winner
    stats['arms_differed'] = differed
    if candidate_log is not None:
        del candidate_log[:]
        candidate_log.extend(winning_log)
    return models