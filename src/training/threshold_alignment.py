from src.p4gen.evaluation import accuracy_metrics
from src.p4gen.build_p4_script import INFINITE
from src.training.errors import AlignmentInvariantError
import sklearn
import numpy as np


def rel_deg(before, after):
    """Degradation as a fraction of `before`'s error. Same definition as
    trial_selection.rel_deg -- duplicated rather than imported to keep this
    module free of a training-package dependency it otherwise does not need."""
    return (before - after) / max(1e-9, 1.0 - before)


def align_rf_thresholds(rf1, rf2, X_val1, y_val1, X_val2, y_val2,
                        overlap_threshold=0.5, delta_rel=0.0):
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

        NOTE: in P2 this still guards the AVERAGE of the two tasks -- the
        minimal faithful step, replacing a hardcoded 0.005 absolute tolerance
        with a relative one. P3 replaces it with four independent per-task
        guards and a per-task high-water ratchet. Joint-arm accuracy numbers
        produced between P2 and P3 are not meaningful.

    Returns:
    --------
    rf1_aligned, rf2_aligned : Modified RandomForest models with aligned thresholds
    alignment_stats : dict
        Statistics about the alignment process
    """

    #print('Threshold index 1')
    threshold_index1 = build_threshold_index(rf1)

    #print('Threshold index 2')
    threshold_index2 = build_threshold_index(rf2)

    intervals1 = extract_feature_intervals(rf1)
    intervals2 = extract_feature_intervals(rf2)

    with sklearn.config_context(assume_finite=True):
        tree_predictions1, node_to_samples1 = build_prediction_cache(rf1, X_val1)
        tree_predictions2, node_to_samples2 = build_prediction_cache(rf2, X_val2)

    # Initial predictions -- only needed to seed the accept/reject comparison
    # below, which itself is skipped entirely when delta_rel is None (the inf
    # anchor). Not computing them here is what makes that arm the cheapest.
    before_acc_av = before_fscore_av = None
    if delta_rel is not None:
        initial_pred1 = compute_ensemble_prediction(tree_predictions1, rf1)
        initial_pred2 = compute_ensemble_prediction(tree_predictions2, rf2)

        before_acc1, before_fscore1 = accuracy_metrics(y_val1, initial_pred1, task="app")
        before_acc2, before_fscore2 = accuracy_metrics(y_val2, initial_pred2, task="ddos")

        before_acc_av = (before_acc1 + before_acc2) / 2
        before_fscore_av = (before_fscore1 + before_fscore2) / 2

        #print(before_acc_av, before_fscore_av)

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
                
                # Adjust boundaries to make ranges identical
                modifications1 = adjust_range_boundaries(
                    rf1, feature_idx, range1, target, threshold_index1
                )
                #print('Mods1', modifications1)

                modifications2 = adjust_range_boundaries(
                    rf2, feature_idx, range2, target, threshold_index2
                )
                #print('Mods2', modifications2)

                undo_info1 = update_cache_for_modifications(rf1, X_val1, tree_predictions1, node_to_samples1, modifications1)
                undo_info2 = update_cache_for_modifications(rf2, X_val2, tree_predictions2, node_to_samples2, modifications2)

                if delta_rel is None:
                    # The inf anchor: accept unconditionally. Skipping the
                    # predict/restore/undo machinery is why this arm is the
                    # cheapest to run.
                    accepted = True
                else:
                    new_pred1 = compute_ensemble_prediction(tree_predictions1, rf1)
                    new_pred2 = compute_ensemble_prediction(tree_predictions2, rf2)

                    with sklearn.config_context(assume_finite=True):
                        after_acc1, after_fscore1 = accuracy_metrics(y_val1, new_pred1, task="app")
                        after_acc2, after_fscore2 = accuracy_metrics(y_val2, new_pred2, task="ddos")

                    after_acc_av = (after_acc1 + after_acc2) / 2
                    after_fscore_av = (after_fscore1 + after_fscore2) / 2

                    accepted = not (rel_deg(before_acc_av, after_acc_av) > delta_rel
                                    or rel_deg(before_fscore_av, after_fscore_av) > delta_rel)

                if not accepted:
                    restore_thresholds(rf1, modifications1)
                    restore_thresholds(rf2, modifications2)
                    undo_cache_update(tree_predictions1, node_to_samples1, undo_info1)
                    undo_cache_update(tree_predictions2, node_to_samples2, undo_info2)
                else:
                    if delta_rel is not None:
                        if after_acc_av > before_acc_av:
                            before_acc_av = after_acc_av
                        if after_fscore_av > before_fscore_av:
                            before_fscore_av = after_fscore_av

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

    return rf1, rf2 #, alignment_stats


def extract_feature_intervals(rf):
    """
    Extract feature intervals from a scikit-learn random forest model.
    
    Args:
        rf: A trained RandomForestClassifier or RandomForestRegressor
    
    Returns:
        dict: Dictionary with feature index as key and list of intervals as values
    """
    # Step 1: Extract all thresholds for each feature
    feature_thresholds = {}
    
    for estimator in rf.estimators_:
        tree = estimator.tree_
        
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:  # Not a leaf node
                feature_idx = tree.feature[node_idx]
                threshold = int(round(tree.threshold[node_idx]))

                if threshold == 0:
                    continue
                
                if feature_idx not in feature_thresholds:
                    feature_thresholds[feature_idx] = set()
                feature_thresholds[feature_idx].add(threshold)
    
    # Step 2: Build intervals for each feature
    feature_intervals = {}
    
    for feature_idx, thresholds in feature_thresholds.items():
        if not thresholds:
            continue
        
        # Sort thresholds in ascending order
        sorted_thresholds = sorted(list(thresholds))
        intervals = []
        
        # First interval: (0, smallest_threshold)
        intervals.append((0, sorted_thresholds[0]))
        
        # Middle intervals: (threshold_i + 1, threshold_i+1)
        for i in range(len(sorted_thresholds) - 1):
            intervals.append((sorted_thresholds[i] + 1, sorted_thresholds[i + 1]))
        
        # Last interval: (largest_threshold + 1, INFINITE)
        intervals.append((sorted_thresholds[-1] + 1, INFINITE))
        
        feature_intervals[feature_idx] = intervals
    
    return feature_intervals


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
        
        # Get predictions for this tree
        tree_predictions[tree_idx] = rf.classes_[estimator.predict(X_val).astype(int)]
        
        # Get decision path (sparse matrix: n_samples x n_nodes)
        decision_path = estimator.decision_path(X_val)
        
        # Convert to node -> samples mapping for non-leaf nodes only
        for node_idx in range(tree.node_count):
            if tree.feature[node_idx] >= 0:  # Not a leaf
                # Get samples that pass through this node
                samples = decision_path[:, node_idx].nonzero()[0]
                node_to_samples[(tree_idx, node_idx)] = samples
    
    return tree_predictions, node_to_samples


def compute_ensemble_prediction(tree_predictions, rf):
    """Compute ensemble prediction from per-tree predictions via majority vote."""
    n_trees, n_samples = tree_predictions.shape
    n_classes = rf.n_classes_

    # Build mapping from class label to index (handles non-0-indexed classes)
    class_to_idx = {c: i for i, c in enumerate(rf.classes_)}

    # Count votes per class
    votes = np.zeros((n_samples, n_classes), dtype=np.intp)
    for tree_idx in range(n_trees):
        for sample_idx in range(n_samples):
            class_label = int(tree_predictions[tree_idx, sample_idx])
            votes[sample_idx, class_to_idx[class_label]] += 1

    return rf.classes_[np.argmax(votes, axis=1)]


def find_partially_overlapping_ranges(ranges1, ranges2):
    """
    Find partially overlapping regions
    """
    overlaps = []
    
    for i, (start1, end1) in enumerate(ranges1):
        if end1 <= start1:
            continue
        for j, (start2, end2) in enumerate(ranges2):
            if end2 <= start2:
                continue
            if start1 == start2 and end1 == end2:
                continue
            if start1 < end2 and start2 < end1:
                overlaps.append((i, j))
    
    return overlaps


def calculate_range_overlap(range1, range2):
    """Calculate the overlap ratio between two ranges"""
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

    # Early exit for large ratio differences
    if min1 and min2:  # Both non-zero
        min_ratio = max(min1, min2) / min(min1, min2)
        if min_ratio > 5:
            return 0.0
    
    if max1 and max2:  # Both non-zero
        max_ratio = max(max1, max2) / min(max1, max2)
        if max_ratio > 5:
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
    trees_to_repredict = {}

    for tree_idx, node_idx, _ in modifications:
        if (tree_idx, node_idx) in node_to_samples:
            samples = node_to_samples[(tree_idx, node_idx)]
            if tree_idx not in trees_to_repredict:
                trees_to_repredict[tree_idx] = set()
            trees_to_repredict[tree_idx].update(samples)

    # Capture old state for undo
    undo_info = {
        'predictions': {},  # tree_idx -> (sample_indices, old_predictions)
        'node_samples': {}  # (tree_idx, node_idx) -> old_samples
    }

    for tree_idx, sample_indices_set in trees_to_repredict.items():
        
        if not sample_indices_set:  # Skip if no samples need reprediction
            continue
        sample_indices = np.array(list(sample_indices_set), dtype=np.intp)

        # Save old predictions
        undo_info['predictions'][tree_idx] = (
            sample_indices.copy(),
            tree_predictions[tree_idx, sample_indices].copy()
        )

        X_subset = X_val[sample_indices]
        new_predictions = rf.classes_[rf.estimators_[tree_idx].predict(X_subset).astype(int)]
        tree_predictions[tree_idx, sample_indices] = new_predictions

        tree = rf.estimators_[tree_idx].tree_
        decision_path = rf.estimators_[tree_idx].decision_path(X_subset)

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
            local_indices = decision_path[:, node_idx].nonzero()[0]
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