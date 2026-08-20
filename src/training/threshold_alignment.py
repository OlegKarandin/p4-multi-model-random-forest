from src.p4gen.build_p4_script import INFINITE, get_feature_intervals_from_thresholds
from src.training.errors import AlignmentInvariantError
from src.training.incremental_metrics import IncrementalMetrics
import sklearn
import numpy as np


def rel_deg(before, after):
    """Degradation as a fraction of `before`'s error. Same definition as
    trial_selection.rel_deg -- duplicated rather than imported to keep this
    module free of a training-package dependency it otherwise does not need."""
    return (before - after) / max(1e-9, 1.0 - before)


# (acc_app, f1_app, acc_ddos, f1_ddos) -- the order every 4-tuple in this
# module uses. Named so a reader of accept_alignment knows what position 2 is.
METRIC_NAMES = ('acc_app', 'f1_app', 'acc_ddos', 'f1_ddos')


def accept_alignment(before, after, delta_rel):
    """Whether an alignment may stand, judged PER TASK (spec B.4).

    before, after : 4-tuples (acc_app, f1_app, acc_ddos, f1_ddos).
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
    both models split on, a shared interval list needs only ONE set of TCAM
    entries -- so the count is the size of the UNION of the two models'
    interval tuples, not their sum. For a feature only one model splits on,
    there is nothing to share, so its own interval count is added directly.

    This -- not a flat sum of each model's own interval count -- is the
    quantity that actually shrinks when alignment succeeds: a successful move
    makes two previously-different interval lists for the same feature
    IDENTICAL, collapsing their union. Alignment only ever relocates a
    threshold, never deletes one, so each model's OWN interval count never
    changes; a stat built from per-model sums alone is structurally constant
    and cannot reflect any TCAM savings at all.
    """
    common = set(intervals1) & set(intervals2)
    total = sum(len(set(intervals1[f]) | set(intervals2[f])) for f in common)
    total += sum(len(v) for f, v in intervals1.items() if f not in common)
    total += sum(len(v) for f, v in intervals2.items() if f not in common)
    return total


def align_rf_thresholds(rf1, rf2, X_val1, y_val1, X_val2, y_val2,
                        overlap_threshold=0.5, delta_rel=0.0, align_stats=None,
                        candidate_log=None):
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

    Returns:
    --------
    rf1_aligned, rf2_aligned : Modified RandomForest models with aligned thresholds
    alignment_stats : dict
        Statistics about the alignment process
    """

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

    # One sort per model, for shift_mass -- a purely diagnostic quantity now
    # (P3 Task 8: the shift_mass-derived pre-filter was removed; every
    # overlap-eligible candidate reaches the real oracle check below). Only
    # computed when candidate_log asks for it, so it costs nothing on the
    # default (no-logging) hot path. Per-model is correct: damage to rf1
    # depends on X_val1's distribution, not X_val2's. Feature indices line up
    # -- trees are fit on X_*_train[:, remaining] and validated on
    # X_*_val[:, remaining], the same column space.
    sorted_cols1 = sorted_cols2 = None
    if candidate_log is not None:
        sorted_cols1 = np.sort(X_val1, axis=0)
        sorted_cols2 = np.sort(X_val2, axis=0)

    #print('Threshold index 1')
    threshold_index1 = build_threshold_index(rf1)

    #print('Threshold index 2')
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
    marks = None
    current = None
    metrics1 = metrics2 = None
    if delta_rel is not None:
        metrics1 = IncrementalMetrics(tree_predictions1, rf1, y_val1, task="app")
        metrics2 = IncrementalMetrics(tree_predictions2, rf2, y_val2, task="ddos")

        # Four independent high-water marks, in METRIC_NAMES order.
        marks = metrics1.metrics() + metrics2.metrics()
        # Last-ACCEPTED state -- the model's actual current metrics, as opposed
        # to marks' running per-task max. Before any candidate, both coincide.
        current = marks

    stats = align_stats if align_stats is not None else {}
    stats['attempted'] = 0
    stats['accepted'] = 0
    stats['intervals_before'] = joint_interval_count(intervals1, intervals2)

    # Find common features
    common_features = set(intervals1.keys()) & set(intervals2.keys())
 
    sorted_features = sorted(common_features, 
                        key=lambda f: len(intervals1.get(f, [])) + len(intervals2.get(f, [])), 
                        reverse=True)
    
    # Initialize statistics
    '''alignment_stats = {
        'common_features': len(common_features),
        'total_aligned_ranges': 0,
        'feature_details': {}
    }'''
    
    for feature_idx in sorted_features:
        current_ranges1 = intervals1[feature_idx]
        current_ranges2 = intervals2[feature_idx]

        #print('Ranges A: {}'.format(current_ranges1))
        #print('Ranges B: {}'.format(current_ranges2))
        
        overlaps = find_partially_overlapping_ranges(current_ranges1, current_ranges2)

        '''alignment_stats['feature_details'][feature_idx] = {
            'ranges_rf1': intervals1[feature_idx],  # Store original ranges
            'ranges_rf2': intervals2[feature_idx],  # Store original ranges
            'pure_overlaps': len(overlaps),
            'alignments': []
        }'''
        
        # Apply alignment for each overlap
        for (idx1, idx2) in overlaps:
            range1 = current_ranges1[idx1]
            range2 = current_ranges2[idx2]

            if range1 == range2:
                continue
            
            overlap_ratio = calculate_range_overlap(range1, range2)

            #print(range1, range2, overlap_ratio, overlap_ratio >= overlap_threshold)

            if overlap_ratio >= overlap_threshold:
                target = calculate_target_range(range1, range2)

                # Purely diagnostic now -- only computed when a candidate_log
                # is actually requested (see the sorted_cols1/2 gating above).
                mass1 = mass2 = None
                if candidate_log is not None:
                    mass1 = max(shift_mass(sorted_cols1[:, feature_idx], old, new)
                                for old, new in ((range1[0], target[0]), (range1[1], target[1])))
                    mass2 = max(shift_mass(sorted_cols2[:, feature_idx], old, new)
                                for old, new in ((range2[0], target[0]), (range2[1], target[1])))

                # Adjust boundaries to make ranges identical
                modifications1 = adjust_range_boundaries(
                    rf1, feature_idx, range1, target, threshold_index1
                )
                #print('Mods1', modifications1)

                modifications2 = adjust_range_boundaries(
                    rf2, feature_idx, range2, target, threshold_index2
                )
                #print('Mods2', modifications2)

                if not modifications1 and not modifications2:
                    # P5: adjust_range_boundaries declined every move (source
                    # min is 0, source max is INFINITE, or source already
                    # equals target). Every feature's interval list starts at 0
                    # and ends at INFINITE, so this is a common path -- and
                    # there is nothing to evaluate, restore or undo.
                    continue

                undo_info1 = update_cache_for_modifications(rf1, X_val1, tree_predictions1, node_to_samples1, modifications1)
                undo_info2 = update_cache_for_modifications(rf2, X_val2, tree_predictions2, node_to_samples2, modifications2)

                stats['attempted'] += 1

                # Rollback tokens for the metric state, paired 1:1 with
                # undo_info1/2. Bound to None here rather than only inside the
                # else branch so the reject path below reads the same on every
                # arm, including the inf one where no apply ever ran.
                mtoken1 = mtoken2 = None

                if delta_rel is None:
                    # The inf anchor: accept unconditionally. Skipping the
                    # predict/metric machinery is why this arm is cheapest.
                    accepted = True
                    after = None
                else:
                    # IncrementalMetrics' ordering contract: apply reads the NEW
                    # per-tree predictions out of tree_predictions and the OLD
                    # ones out of undo_info, so it must run AFTER
                    # update_cache_for_modifications and BEFORE any
                    # undo_cache_update.
                    mtoken1 = metrics1.apply(tree_predictions1, undo_info1)
                    mtoken2 = metrics2.apply(tree_predictions2, undo_info2)

                    after = metrics1.metrics() + metrics2.metrics()
                    accepted = accept_alignment(marks, after, delta_rel)

                if candidate_log is not None:
                    candidate_log.append({
                        'feature_idx': int(feature_idx),
                        'range1': tuple(range1),
                        'range2': tuple(range2),
                        'overlap_ratio': float(overlap_ratio),
                        'endpoint_ratio': float(endpoint_ratio(range1, range2)),
                        # current is None only on the delta_rel=None (inf) arm,
                        # where accept/reject -- and therefore any notion of
                        # "current error" -- is skipped entirely; 0.0 mirrors
                        # rel_deg's own placeholder for that arm below.
                        'error_app': 1.0 - current[0] if current is not None else 0.0,
                        'error_ddos': 1.0 - current[2] if current is not None else 0.0,
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
                        'rel_deg': tuple(rel_deg(b, a) for b, a in zip(current, after))
                                   if after is not None else (0.0, 0.0, 0.0, 0.0),
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
                    if delta_rel is not None:
                        metrics1.revert(mtoken1)
                        metrics2.revert(mtoken2)
                else:
                    stats['accepted'] += 1
                    if after is not None:
                        marks = ratchet(marks, after)
                        current = after

                    update_neighboring_ranges_and_index(
                        current_ranges1, idx1, range1, target, 
                        feature_idx, threshold_index1
                    )

                    update_neighboring_ranges_and_index(
                        current_ranges2, idx2, range2, target,
                        feature_idx, threshold_index2
                    )

                    #print('Resulting Ranges A: {}'.format(current_ranges1))
                    #print('Resulting Ranges B: {}'.format(current_ranges2))
                    
                    # Record alignment
                    '''alignment_stats['feature_details'][feature_idx]['alignments'].append({
                        'range1': range1,
                        'range2': range2,
                        'target': target,
                        'overlap_ratio': overlap_ratio,
                    })
                    alignment_stats['total_aligned_ranges'] += 1'''

    stats['intervals_after'] = joint_interval_count(
        extract_feature_intervals(rf1), extract_feature_intervals(rf2))

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
                #print('{}: {}'.format(key, (tree_idx, node_idx)))
    
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


def shift_mass_cap(delta_rel, before_acc, slack=2.0):
    """The largest shift_mass that can be permitted at tolerance `delta_rel`.

    Derived, not tuned -- for the CORE bound. Only samples inside the moved
    window can change ANY prediction -- however many tree nodes share that
    threshold, the affected sample set is the same set. So a move relocating
    fraction p satisfies delta_accuracy <= p, hence rel_deg <= p / error.
    Requiring p <= delta_rel * error (i.e. slack = 1) therefore guarantees:
    any candidate that PASSES that cap is itself guaranteed -- by the physical
    fact above, not by assumption -- to satisfy the accuracy bound at EXACTLY
    delta_rel. At the shipped default slack = 2.0, the same argument only
    guarantees the bound at slack * delta_rel = 2 * delta_rel, not delta_rel
    itself -- every test of this specific claim in the test suite therefore
    uses slack=1.0 explicitly; the shipped default is a deliberately looser,
    UNtested-for-exactness setting.

    This does NOT by itself prove the reverse (that every VETOED candidate
    would truly have been rejected by the full per-task guard in
    align_rf_thresholds): shift_mass is an upper bound on possible damage, not
    a lower bound, so a large-mass move could in principle have had small real
    damage. Treat this as a well-motivated, empirically-supported heuristic
    (P3 Task 6's instrumentation found zero violations of the local
    delta_accuracy <= p bound, though only a weak Pearson correlation between
    shift_mass and actual damage -- r~+0.14 for app, r~+0.01 for ddos), not a
    proof that a filter built on this cap never discards a move the full
    guard would have kept.

    NOTE (P3 Task 8): this function is currently UNUSED for filtering --
    align_rf_thresholds no longer vetoes candidates on shift_mass at all (a
    final whole-branch review found the cap is identically 0 whenever
    delta_rel = 0, silently discarding confirmed-harmless moves; the project
    owner chose removal over patching the edge case). It remains here, tested,
    in case a future phase wants to reintroduce a pre-filter.

    `slack` widens the cap beyond the exact-accuracy bound above; it is NOT a
    proven safety margin for weighted F1. F1 can move further than accuracy
    per flip when flips concentrate in one class -- which argues, if anything,
    for a TIGHTER cap to protect F1, not a looser one. The honest
    justification for slack > 1 is simply "lean permissive and let the real
    oracle decide" rather than any F1-specific soundness claim.
    """
    if delta_rel is None:
        return float('inf')
    return slack * delta_rel * (1.0 - before_acc)


def calculate_range_overlap(range1, range2):
    """Overlap ratio between two ranges; 0.0 also means 'vetoed'.

    NOTE this function's 0.0 return is overloaded: it means both "no overlap"
    and "vetoed". The zero-side and INFINITE-side vetoes below are structural
    (adjust_range_boundaries cannot move those boundaries at all). The old
    endpoint-ratio-cap heuristic pre-filter that used to live here is gone as
    of Task 7, replaced by the delta-derived shift_mass_cap veto in the
    candidate loop of align_rf_thresholds.
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
    
    if threshold_source_min != threshold_target_min and threshold_source_min != 0:

        if (feature_idx, threshold_source_min) not in threshold_index:
            raise AlignmentInvariantError(
                '{} missing from threshold_index'.format((feature_idx, threshold_source_min)))

        for tree_idx, node_idx in threshold_index[(feature_idx, threshold_source_min)]:
            
            #print('Modifying threshold {} of feature {} in tree {} node {} to {}'.format(threshold_source_min, feature_idx, node_idx, tree_idx, threshold_target_min))

            tree = rf.estimators_[tree_idx].tree_
            modifications.append((tree_idx, node_idx, threshold_source_min))
            tree.threshold[node_idx] = threshold_target_min
    
    if threshold_source_max != threshold_target_max and threshold_source_max != INFINITE:

        if (feature_idx, threshold_source_max) not in threshold_index:
            raise AlignmentInvariantError(
                '{} missing from threshold_index'.format((feature_idx, threshold_source_max)))

        for tree_idx, node_idx in threshold_index[(feature_idx, threshold_source_max)]:

            #print('Modifying threshold {} of feature {} in tree {} node {} to {}'.format(threshold_source_max, feature_idx, node_idx, tree_idx, threshold_target_max))

            tree = rf.estimators_[tree_idx].tree_
            modifications.append((tree_idx, node_idx, threshold_source_max))
            tree.threshold[node_idx] = threshold_target_max
    
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
        List of (tree_idx, node_idx, original_threshold, new_threshold) to restore
    """
    for tree_idx, node_idx, original_threshold in modifications:

        #print('Restoring threshold {} in tree {} node {} to the original value {}'.format(rf.estimators_[tree_idx].tree_.threshold[node_idx], tree_idx, node_idx, original_threshold))

        rf.estimators_[tree_idx].tree_.threshold[node_idx] = original_threshold


def update_neighboring_ranges_and_index(ranges, target_idx, old_range, new_range, feature_idx, threshold_index):
    old_min, old_max = old_range
    new_min, new_max = new_range

    # Apply the -1 adjustment for the threshold index
    threshold_old_min = old_min - 1 if old_min > 0 else old_min
    threshold_new_min = new_min - 1 if new_min > 0 else new_min

    threshold_old_max = old_max
    threshold_new_max = new_max

    # C5: mirror adjust_range_boundaries' own guards. It declines to move a
    # threshold at 0 (min side) or at INFINITE (max side), so `ranges` must not
    # claim those boundaries moved -- that disagreement between `ranges`,
    # tree_.threshold and threshold_index IS C5.
    effective_min = new_min if threshold_old_min != 0 else old_min
    effective_max = new_max if threshold_old_max != INFINITE else old_max
    effective_range = (effective_min, effective_max)

    # Update the target range
    if effective_range != old_range:
        ranges[target_idx] = effective_range

        # Update threshold index
        if threshold_old_min != threshold_new_min and threshold_old_min != 0:
            update_threshold_index(threshold_index, feature_idx, threshold_old_min, threshold_new_min)
        if threshold_old_max != threshold_new_max and threshold_old_max != INFINITE:
            update_threshold_index(threshold_index, feature_idx, threshold_old_max, threshold_new_max)

        # Update neighboring ranges
        for i, (range_min, range_max) in enumerate(ranges):
            if i == target_idx:
                continue

            new_range_min = range_min
            new_range_max = range_max

            # Check if this range's max boundary matches the old min boundary
            if range_max + 1 == old_min:
                new_range_max = effective_min - 1

            # Check if this range's min boundary matches the old max boundary
            if range_min - 1 == old_max:
                new_range_min = effective_max + 1

            if new_range_min > new_range_max:
                raise RuntimeError("Smth is very-very wrong")

            # Update the range tuple if needed
            if new_range_min != range_min or new_range_max != range_max:
                ranges[i] = (new_range_min, new_range_max)


def update_threshold_index(threshold_index, feature_idx, old_threshold, new_threshold):
    """
    Update a single threshold in the index.
    """

    if (feature_idx, old_threshold) not in threshold_index:
        raise AlignmentInvariantError(
            '{} missing from threshold_index'.format((feature_idx, old_threshold)))

    nodes = threshold_index.pop((feature_idx, old_threshold))
    ##print('Updating threshold {} to {} for nodes {}'.format(old_threshold, new_threshold, nodes))
    if (feature_idx, new_threshold) in threshold_index:
        existing = set(threshold_index[(feature_idx, new_threshold)])
        existing.update(nodes)
        threshold_index[(feature_idx, new_threshold)] = list(existing)
    else:
        threshold_index[(feature_idx, new_threshold)] = nodes