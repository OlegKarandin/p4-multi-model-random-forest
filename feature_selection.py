import numpy as np
from sklearn.inspection import permutation_importance

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from train_model import train_multi_RF_Optuna_multi_constrained
from evaluation import accuracy_metrics


def compare_feature_selection_approaches(X_app, X_ddos, y_app, y_ddos, n_trees, max_depth, max_blocks,
                                       feature_names,
                                       n_splits, random_state=42, verbose=False):
    """
    Compare single-task (L1 Logistic) vs multi-task (MTL) feature selection using regularization paths.

    Parameters:
    -----------
    X_app, X_ddos : array-like
        Feature matrices for App and DDoS datasets
    y_app, y_ddos : array-like
        Target vectors
    n_trees, max_depth, max_blocks : int
        Model training constraints
    feature_names : list
        Feature names
    n_splits : int
        Number of train/test splits
    random_state : int
        Random seed (default: 42)
    verbose : bool
        Print progress (default: False)

    Returns:
    --------
    results_df : pd.DataFrame
        Results with columns for each (method, regularization_value, k)
    """

    # Ensure both datasets have same number of features
    if X_app.shape[1] != X_ddos.shape[1]:
        raise ValueError("Both datasets must have the same number of features")

    print(f"Starting comparison with {n_splits} splits")
    print(f"App dataset shape: {X_app.shape}, DDoS dataset shape: {X_ddos.shape}")
    print("-" * 70)

    # Initialize results storage
    results = []

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Progress bar for all experiments
    pbar = tqdm(total=n_splits, desc="Running experiments")

    for split_idx in range(n_splits):

        if verbose:
            print(f"\n=== Split {split_idx} ===")

        # Create train-test splits for both datasets
        X_app_temp, X_app_test, y_app_temp, y_app_test = train_test_split(
            X_app, y_app, test_size=0.15,
            random_state=random_state + split_idx,
            stratify=y_app
        )

        X_app_train, X_app_val, y_app_train, y_app_val = train_test_split(
            X_app_temp, y_app_temp, test_size=0.176,
            random_state=random_state + split_idx,
            stratify=y_app_temp
        )

        X_ddos_temp, X_ddos_test, y_ddos_temp, y_ddos_test = train_test_split(
            X_ddos, y_ddos, test_size=0.15,
            random_state=random_state + split_idx,
            stratify=y_ddos
        )

        X_ddos_train, X_ddos_val, y_ddos_train, y_ddos_val = train_test_split(
            X_ddos_temp, y_ddos_temp, test_size=0.176,
            random_state=random_state + split_idx,
            stratify=y_ddos_temp
        )
        
        try:
            
            remaining_features_app = list(range(X_app_train.shape[1]))
            remaining_features_ddos = list(range(X_ddos_train.shape[1]))
            feature_names_app = list(feature_names)
            feature_names_ddos = list(feature_names)

            while True:
                k_app = len(remaining_features_app)
                k_ddos = len(remaining_features_ddos)
                
                # Train models with current feature sets
                model_app, model_ddos, stages, blocks, _ = train_multi_RF_Optuna_multi_constrained(
                    X_app_train[:, remaining_features_app],
                    y_app_train,
                    X_ddos_train[:, remaining_features_ddos],
                    y_ddos_train,
                    X_app_val[:, remaining_features_app],
                    y_app_val,
                    X_ddos_val[:, remaining_features_ddos],
                    y_ddos_val,
                    feature_names_app,
                    feature_names_ddos,
                    n_trees,
                    max_depth,
                    max_blocks,
                    'disjoint'
                )

                # Calculate accuracy metrics
                with sklearn.config_context(assume_finite=True):
                    acc_app, f1_app = accuracy_metrics(
                        y_app_test,
                        model_app.predict(X_app_test[:, remaining_features_app]),
                        task="app"
                    )
                    acc_ddos, f1_ddos = accuracy_metrics(
                        y_ddos_test,
                        model_ddos.predict(X_ddos_test[:, remaining_features_ddos]),
                        task="ddos"
                    )

                if verbose:
                    print(f"Single-task k_app={k_app}, k_ddos={k_ddos}: blocks={blocks}, acc_app={acc_app:.4f}, acc_ddos={acc_ddos:.4f}")

                results.append({
                    'method': 'single',
                    'split': split_idx,
                    'k_app': k_app,
                    'k_ddos': k_ddos,
                    'features_app': list(feature_names_app),
                    'features_ddos': list(feature_names_ddos),
                    'acc_app': acc_app,
                    'f1_app': f1_app,
                    'acc_ddos': acc_ddos,
                    'f1_ddos': f1_ddos,
                    'stages': stages,
                    'blocks': blocks,
                })

                if len(remaining_features_app) == 1 and len(remaining_features_ddos) == 1:
                    break

                # Calculate permutation importance for each problem independently
                importance_results_app = permutation_importance(
                    model_app, X_app_val[:, remaining_features_app], y_app_val,
                    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1
                )
                lowest_importance_idx_app = importance_results_app.importances_mean.argmin()
                del remaining_features_app[lowest_importance_idx_app]
                del feature_names_app[lowest_importance_idx_app]

                importance_results_ddos = permutation_importance(
                    model_ddos, X_ddos_val[:, remaining_features_ddos], y_ddos_val,
                    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1
                )
                lowest_importance_idx_ddos = importance_results_ddos.importances_mean.argmin()
                del remaining_features_ddos[lowest_importance_idx_ddos]
                del feature_names_ddos[lowest_importance_idx_ddos]


            remaining_features_shared = list(range(X_app_train.shape[1]))
            feature_names_shared = list(feature_names)

            while True:
                k = len(remaining_features_shared)
                
                # Train models with current feature set
                model_app, model_ddos, stages, blocks, _ = train_multi_RF_Optuna_multi_constrained(
                    X_app_train[:, remaining_features_shared],
                    y_app_train,
                    X_ddos_train[:, remaining_features_shared],
                    y_ddos_train,
                    X_app_val[:, remaining_features_shared],
                    y_app_val,
                    X_ddos_val[:, remaining_features_shared],
                    y_ddos_val,
                    feature_names_shared,
                    feature_names_shared,
                    n_trees,
                    max_depth,
                    max_blocks,
                    'joint'
                )

                # Calculate accuracy metrics
                with sklearn.config_context(assume_finite=True):
                    acc_app, f1_app = accuracy_metrics(
                        y_app_test,
                        model_app.predict(X_app_test[:, remaining_features_shared]),
                        task="app"
                    )
                    acc_ddos, f1_ddos = accuracy_metrics(
                        y_ddos_test,
                        model_ddos.predict(X_ddos_test[:, remaining_features_shared]),
                        task="ddos"
                    )
                    
                results.append({
                    'method': 'multi',
                    'split': split_idx,
                    'k': k,
                    'features_app': list(feature_names_shared),
                    'features_ddos': list(feature_names_shared),
                    'acc_app': acc_app,
                    'f1_app': f1_app,
                    'acc_ddos': acc_ddos,
                    'f1_ddos': f1_ddos,
                    'stages': stages,
                    'blocks': blocks,
                })

                if len(remaining_features_shared) == 1:
                    break

                # Calculate permutation importance and remove least important feature
                importance_results_app = permutation_importance(
                    model_app, X_app_val[:, remaining_features_shared], y_app_val,
                    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1
                )
                importance_results_ddos = permutation_importance(
                    model_ddos, X_ddos_val[:, remaining_features_shared], y_ddos_val,
                    scoring='accuracy', n_repeats=10, random_state=42, n_jobs=-1
                )
                
                # Combine importances
                combined_importance = importance_results_app.importances_mean + importance_results_ddos.importances_mean
                lowest_importance_idx = combined_importance.argmin()
                
                # Remove least important feature
                del remaining_features_shared[lowest_importance_idx]
                del feature_names_shared[lowest_importance_idx]

        except Exception as e:
            print(f"Error in split {split_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        pbar.update(1)

    pbar.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    print(f"\nCompleted {len(results_df)} successful experiments")
    print(f"Results shape: {results_df.shape}")
    print(f"Methods: {results_df['method'].value_counts().to_dict()}")

    return results_df


# =============================================================================
# PARALLEL VERSION
# =============================================================================

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class SplitResult:
    """Container for results from a single split - immutable and picklable."""
    split_idx: int
    results: List[Dict[str, Any]]
    error: Optional[str] = None


def _derive_feature_intervals(clf, feature_names):
    """Derives a `feature_intervals` dict for one model, via the exact same
    tree_nodes -> thresholds -> intervals code path
    `evaluation.single_model_memory_evaluation` uses internally (mirrored
    here rather than imported, since that function also computes range/
    ternary usage we don't need -- `generate_P4_code` recomputes codewords
    from `clf`/`feature_intervals` on its own).
    """
    from build_p4_script import get_tree_textual_representation, get_nodes, \
        get_feature_thresholds, get_feature_intervals_from_thresholds

    trees = get_tree_textual_representation(clf, feature_names)
    tree_nodes = {tree: get_nodes(trees[tree]) for tree in trees}
    feature_thresholds = get_feature_thresholds(tree_nodes)
    return get_feature_intervals_from_thresholds(feature_thresholds)


def _derive_joint_feature_intervals(model_app, model_ddos, feature_names_app, feature_names_ddos):
    """Derives ONE shared `feature_intervals` dict for both models, mirroring
    `evaluation.multi_model_memory_evaluation`'s 'joint' branch exactly (same
    offset trick merging both models' trees into one `tree_nodes` dict
    before deriving intervals) -- this is the analytical model the real
    compiled numbers are meant to validate against, so the real program must
    be built from the same interval derivation, not a re-invented one.
    """
    from build_p4_script import get_tree_textual_representation, get_nodes, \
        get_feature_thresholds, get_feature_intervals_from_thresholds

    trees_app = get_tree_textual_representation(model_app, feature_names_app)
    trees_ddos = get_tree_textual_representation(model_ddos, feature_names_ddos)

    tree_nodes = {}
    for tree_app in trees_app:
        tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes)
    for tree_ddos in trees_ddos:
        tree_nodes[tree_ddos + offset] = get_nodes(trees_ddos[tree_ddos])

    feature_thresholds = get_feature_thresholds(tree_nodes)
    return get_feature_intervals_from_thresholds(feature_thresholds)


class _MergedCompileHandle:
    """Wraps two `compile_p4_async` Futures (the disjoint loop's independent
    app-only and ddos-only programs) behind the same `.result(timeout=...)`
    interface a single Future exposes, so the loop-splicing code
    (`pending_previous`/`pending_next`) can treat "this iteration's hardware
    validation" as one handle regardless of how many real p4c invocations it
    took.

    Both futures are submitted (via `compile_p4_async`) immediately and run
    concurrently on `p4_compile`'s shared 2-worker pool, rather than as one
    task that calls the blocking `compile_p4` twice in sequence -- this
    keeps both loops' real-compiler-facing calls behind the same
    `compile_p4_async` seam (the interface this task's brief names as what
    `_process_single_split` consumes), so a test can mock exactly one
    function to cover both the joint and disjoint paths, and it never
    silently attempts a real, unmocked WSL2 invocation in the default test
    suite. As a side effect it also finishes in roughly one compile's wall
    time instead of two, which only helps.
    """

    def __init__(self, future_app, future_ddos):
        self._future_app = future_app
        self._future_ddos = future_ddos

    def result(self, timeout=None):
        from p4_compile import CompileResult

        result_app = self._future_app.result(timeout=timeout)
        result_ddos = self._future_ddos.result(timeout=timeout)

        def _sum_or_none(a, b):
            # "0 resources used" and "one program's compile failed" must
            # stay distinguishable (CompileResult's own None-means-unknown
            # convention) -- so a single None input makes the merged field
            # None too, never silently treated as 0.
            return None if (a is None or b is None) else a + b

        return CompileResult(
            errors=_sum_or_none(result_app.errors, result_ddos.errors),
            warnings=_sum_or_none(result_app.warnings, result_ddos.warnings),
            stages=_sum_or_none(result_app.stages, result_ddos.stages),
            tables=_sum_or_none(result_app.tables, result_ddos.tables),
            gateway=_sum_or_none(result_app.gateway, result_ddos.gateway),
            sram=_sum_or_none(result_app.sram, result_ddos.sram),
            map_ram=_sum_or_none(result_app.map_ram, result_ddos.map_ram),
            tcam=_sum_or_none(result_app.tcam, result_ddos.tcam),
        )


def _kickoff_hardware_validation(validate_on_hardware, hardware_output_dir, split_idx, method, k,
                                  model_app, model_ddos, feature_names_app, feature_names_ddos, encoding):
    """Kicks off (non-blocking) real-compiler validation for one iteration's
    trained model(s). Returns None (never a handle) when validate_on_hardware
    is False, preserving today's zero-cost behavior exactly. Otherwise
    returns an object exposing `.result(timeout=...)` -- either the raw
    Future from `compile_p4_async` (encoding == 'joint') or a
    `_MergedCompileHandle` wrapping two of them (encoding == 'disjoint').

    encoding == 'joint': both models share ONE feature_intervals dict
    (`_derive_joint_feature_intervals`) and are compiled together into ONE
    P4 program -- one real compile produces this iteration's
    stages_real/tcam_real/compile_errors directly.

    encoding == 'disjoint': the two models are NOT required to share
    interval boundaries under disjoint encoding -- merging them into one
    feature_intervals dict the way 'joint' does would silently impose a
    shared discretization that `evaluation.multi_model_memory_evaluation`'s
    'disjoint' branch does not describe (it evaluates each model against
    its OWN independently-derived feature_intervals). So two INDEPENDENT P4
    programs are generated and compiled here -- one containing only
    model_app (clf_ddos=None, num_class_ddos=0), one containing only
    model_ddos (mirrored) -- and their results are merged into one
    CompileResult by `_MergedCompileHandle`.
    """
    if not validate_on_hardware:
        return None

    from p4_compile import compile_p4_async
    from build_p4_script import generate_P4_code

    if encoding == 'joint':
        feature_intervals = _derive_joint_feature_intervals(
            model_app, model_ddos, feature_names_app, feature_names_ddos)

        filename = f"split{split_idx}_{method}_k{k}.p4"
        written_path = generate_P4_code(
            3, 2, model_app, model_ddos, feature_intervals,
            output_dir=hardware_output_dir, output_filename=filename)
        log_dir = hardware_output_dir + f"logs_split{split_idx}_{method}_k{k}/"
        return compile_p4_async(written_path, log_dir)

    elif encoding == 'disjoint':
        feature_intervals_app = _derive_feature_intervals(model_app, feature_names_app)
        feature_intervals_ddos = _derive_feature_intervals(model_ddos, feature_names_ddos)

        filename_app = f"split{split_idx}_{method}_k{k}_app.p4"
        filename_ddos = f"split{split_idx}_{method}_k{k}_ddos.p4"
        written_path_app = generate_P4_code(
            3, 0, model_app, None, feature_intervals_app,
            output_dir=hardware_output_dir, output_filename=filename_app)
        written_path_ddos = generate_P4_code(
            0, 2, None, model_ddos, feature_intervals_ddos,
            output_dir=hardware_output_dir, output_filename=filename_ddos)

        log_dir_app = hardware_output_dir + f"logs_split{split_idx}_{method}_k{k}_app/"
        log_dir_ddos = hardware_output_dir + f"logs_split{split_idx}_{method}_k{k}_ddos/"

        future_app = compile_p4_async(written_path_app, log_dir_app)
        future_ddos = compile_p4_async(written_path_ddos, log_dir_ddos)
        return _MergedCompileHandle(future_app, future_ddos)

    else:
        raise ValueError(f"Unknown encoding for hardware validation: {encoding!r}")


def _advance_pending_compile(results, pending_previous, pending_next):
    """Joins the PREVIOUS iteration's still-pending hardware-validation handle
    -- which has now had one full training step's wall time to finish in the
    background -- attaching its numbers to that earlier row (`results[-2]`).
    The just-appended row (`results[-1]`) gets its own numbers only once
    `pending_next` is itself joined on a later call; on the very first
    iteration (`pending_previous` is None, nothing to join yet) that row's
    three fields are marked None directly instead.

    Shared by both the disjoint ('single') and joint ('multi') loops in
    `_process_single_split`, which differ only in how `pending_next` itself
    was produced (each calls `_kickoff_hardware_validation` with its own
    args) -- the splicing logic that follows is identical.

    Returns the new `pending_previous` value (i.e. `pending_next`) for the
    loop to carry into its next iteration.
    """
    if pending_previous is not None:
        compile_result = pending_previous.result(timeout=600)
        results[-2]['stages_real'] = compile_result.stages
        results[-2]['tcam_real'] = compile_result.tcam
        results[-2]['compile_errors'] = compile_result.errors
    else:
        results[-1]['stages_real'] = None
        results[-1]['tcam_real'] = None
        results[-1]['compile_errors'] = None
    return pending_next


def _join_final_pending_compile(results, pending_previous):
    """Post-loop counterpart to `_advance_pending_compile`: the final
    iteration has no "next" iteration to overlap with, so whatever compile is
    still outstanding is joined directly here and attached to the last row
    appended (`results[-1]`). No-op when `pending_previous` is None (either
    validate_on_hardware was False throughout, or -- impossible in practice,
    since every iteration kicks off a new pending compile -- there simply was
    none left outstanding).
    """
    if pending_previous is not None:
        compile_result = pending_previous.result(timeout=600)
        results[-1]['stages_real'] = compile_result.stages
        results[-1]['tcam_real'] = compile_result.tcam
        results[-1]['compile_errors'] = compile_result.errors


def _process_single_split(
    split_idx: int,
    X_app: np.ndarray,
    X_ddos: np.ndarray,
    y_app: np.ndarray,
    y_ddos: np.ndarray,
    n_trees: int,
    max_depth: int,
    max_blocks: int,
    feature_names: List[str],
    random_state: int,
    verbose: bool,
    validate_on_hardware: bool = False,
    hardware_output_dir: Optional[str] = None,
) -> SplitResult:
    """
    Process a single train/test split.

    This function is designed to be called in a separate process.
    It returns all results as a SplitResult object - no shared mutable state.

    validate_on_hardware : bool
        When True, each iteration's freshly trained model(s) are also
        compiled with the real Tofino toolchain (`p4_compile.compile_p4_async`),
        kicked off right after training and joined one iteration later so the
        added wall time overlaps with the next iteration's training instead of
        blocking on it (see `_kickoff_hardware_validation`). Every result row
        gains three keys either way: `stages_real`, `tcam_real`,
        `compile_errors` -- all None when validate_on_hardware is False
        (default, preserving today's behavior and cost exactly).
    hardware_output_dir : str, optional
        Directory .p4 files and compile logs are written under when
        validate_on_hardware is True. Must be provided in that case, and
        (matching `generate_P4_code`'s own plain string-concatenation
        convention for output_dir + output_filename) should end with a path
        separator -- one is appended automatically if missing.
    """
    # Import inside function to ensure proper pickling in subprocess
    from train_model import train_multi_RF_Optuna_multi_constrained
    from evaluation import accuracy_metrics

    if validate_on_hardware and hardware_output_dir and not hardware_output_dir.endswith(('/', '\\')):
        hardware_output_dir = hardware_output_dir + "/"

    try:
        results = []
        split_random_state = random_state + split_idx

        # Create train-test splits
        X_app_temp, X_app_test, y_app_temp, y_app_test = train_test_split(
            X_app, y_app, test_size=0.15,
            random_state=split_random_state,
            stratify=y_app
        )
        X_app_train, X_app_val, y_app_train, y_app_val = train_test_split(
            X_app_temp, y_app_temp, test_size=0.176,
            random_state=split_random_state,
            stratify=y_app_temp
        )

        X_ddos_temp, X_ddos_test, y_ddos_temp, y_ddos_test = train_test_split(
            X_ddos, y_ddos, test_size=0.15,
            random_state=split_random_state,
            stratify=y_ddos
        )
        X_ddos_train, X_ddos_val, y_ddos_train, y_ddos_val = train_test_split(
            X_ddos_temp, y_ddos_temp, test_size=0.176,
            random_state=split_random_state,
            stratify=y_ddos_temp
        )

        
        # Track best params for warm-starting
        warm_start_params_single = None
        # Handle to the previous iteration's still-running (or already
        # finished) real-compile validation -- joined one iteration behind
        # so the compile's wall time overlaps with this iteration's training
        # instead of blocking on it. None whenever validate_on_hardware is
        # False, or on the very first iteration (nothing to join yet).
        pending_previous_single = None

        remaining_features_app = list(range(X_app_train.shape[1]))
        remaining_features_ddos = list(range(X_ddos_train.shape[1]))
        feature_names_app = list(feature_names)
        feature_names_ddos = list(feature_names)

        while True:
            k_app = len(remaining_features_app)
            k_ddos = len(remaining_features_ddos)
            
            # Train models with current feature sets
            model_app, model_ddos, stages, blocks, best_params = train_multi_RF_Optuna_multi_constrained(
                X_app_train[:, remaining_features_app],
                y_app_train,
                X_ddos_train[:, remaining_features_ddos],
                y_ddos_train,
                X_app_val[:, remaining_features_app],
                y_app_val,
                X_ddos_val[:, remaining_features_ddos],
                y_ddos_val,
                feature_names_app,
                feature_names_ddos,
                n_trees,
                max_depth,
                max_blocks,
                'disjoint',
                warm_start_params_single
            )
            warm_start_params_single = best_params  # Use for next k

            # Calculate accuracy metrics
            with sklearn.config_context(assume_finite=True):
                acc_app, f1_app = accuracy_metrics(
                    y_app_test,
                    model_app.predict(X_app_test[:, remaining_features_app]),
                    task="app"
                )
                acc_ddos, f1_ddos = accuracy_metrics(
                    y_ddos_test,
                    model_ddos.predict(X_ddos_test[:, remaining_features_ddos]),
                    task="ddos"
                )

            results.append({
                'method': 'single',
                'split': split_idx,
                'k': k_app,
                #'features_app': list(feature_names_app),
                #'features_ddos': list(feature_names_ddos),
                'acc_app': acc_app,
                'f1_app': f1_app,
                'acc_ddos': acc_ddos,
                'f1_ddos': f1_ddos,
                'stages': stages,
                'blocks': blocks,
            })

            # Kick off this iteration's hardware validation (non-blocking),
            # then join the PREVIOUS iteration's handle -- which has now had
            # one full training step's wall time to finish in the
            # background -- attaching its numbers to that earlier row. The
            # just-appended row's own numbers land one iteration from now.
            pending_next_single = _kickoff_hardware_validation(
                validate_on_hardware, hardware_output_dir, split_idx, 'single', k_app,
                model_app, model_ddos, feature_names_app, feature_names_ddos, 'disjoint')
            pending_previous_single = _advance_pending_compile(
                results, pending_previous_single, pending_next_single)

            if len(remaining_features_app) == 1 and len(remaining_features_ddos) == 1:
                break

            # Calculate permutation importance for each problem independently
            importance_results_app = permutation_importance(
                model_app, X_app_val[:, remaining_features_app], y_app_val,
                scoring='accuracy', n_repeats=10, random_state=42, #n_jobs=-1
            )
            lowest_importance_idx_app = importance_results_app.importances_mean.argmin()
            del remaining_features_app[lowest_importance_idx_app]
            del feature_names_app[lowest_importance_idx_app]

            importance_results_ddos = permutation_importance(
                model_ddos, X_ddos_val[:, remaining_features_ddos], y_ddos_val,
                scoring='accuracy', n_repeats=10, random_state=42, #n_jobs=-1
            )
            lowest_importance_idx_ddos = importance_results_ddos.importances_mean.argmin()
            del remaining_features_ddos[lowest_importance_idx_ddos]
            del feature_names_ddos[lowest_importance_idx_ddos]

        _join_final_pending_compile(results, pending_previous_single)

        warm_start_params_multi = None
        # Same overlap handle as the disjoint loop above, tracked separately
        # since the two loops' iterations are otherwise independent.
        pending_previous_multi = None
        remaining_features_shared = list(range(X_app_train.shape[1]))
        feature_names_shared = list(feature_names)

        while True:
            k = len(remaining_features_shared)
            
            # Train models with current feature set
            model_app, model_ddos, stages, blocks, best_params = train_multi_RF_Optuna_multi_constrained(
                X_app_train[:, remaining_features_shared],
                y_app_train,
                X_ddos_train[:, remaining_features_shared],
                y_ddos_train,
                X_app_val[:, remaining_features_shared],
                y_app_val,
                X_ddos_val[:, remaining_features_shared],
                y_ddos_val,
                feature_names_shared,
                feature_names_shared,
                n_trees,
                max_depth,
                max_blocks,
                'joint',
                warm_start_params_multi
            )
            warm_start_params_multi = best_params

            # Calculate accuracy metrics
            with sklearn.config_context(assume_finite=True):
                acc_app, f1_app = accuracy_metrics(
                    y_app_test,
                    model_app.predict(X_app_test[:, remaining_features_shared]),
                    task="app"
                )
                acc_ddos, f1_ddos = accuracy_metrics(
                    y_ddos_test,
                    model_ddos.predict(X_ddos_test[:, remaining_features_shared]),
                    task="ddos"
                )

            results.append({
                'method': 'multi',
                'split': split_idx,
                'k': k,
                #'features_app': list(feature_names_shared),
                #'features_ddos': list(feature_names_shared),
                'acc_app': acc_app,
                'f1_app': f1_app,
                'acc_ddos': acc_ddos,
                'f1_ddos': f1_ddos,
                'stages': stages,
                'blocks': blocks,
            })

            pending_next_multi = _kickoff_hardware_validation(
                validate_on_hardware, hardware_output_dir, split_idx, 'multi', k,
                model_app, model_ddos, feature_names_shared, feature_names_shared, 'joint')
            pending_previous_multi = _advance_pending_compile(
                results, pending_previous_multi, pending_next_multi)

            if len(remaining_features_shared) == 1:
                break

            # Calculate permutation importance and remove least important feature
            importance_results_app = permutation_importance(
                model_app, X_app_val[:, remaining_features_shared], y_app_val,
                scoring='accuracy', n_repeats=10, random_state=42, #n_jobs=-1
            )
            importance_results_ddos = permutation_importance(
                model_ddos, X_ddos_val[:, remaining_features_shared], y_ddos_val,
                scoring='accuracy', n_repeats=10, random_state=42, #n_jobs=-1
            )

            # Combine importances
            combined_importance = importance_results_app.importances_mean + importance_results_ddos.importances_mean
            lowest_importance_idx = combined_importance.argmin()

            # Remove least important feature
            del remaining_features_shared[lowest_importance_idx]
            del feature_names_shared[lowest_importance_idx]

        _join_final_pending_compile(results, pending_previous_multi)

        return SplitResult(split_idx=split_idx, results=results)

    except Exception as e:
        import traceback
        return SplitResult(
            split_idx=split_idx,
            results=[],
            error=f"{str(e)}\n{traceback.format_exc()}"
        )


def compare_feature_selection_approaches_parallel(
    X_app, X_ddos, y_app, y_ddos,
    n_trees, max_depth, max_blocks,
    feature_names,
    n_splits, random_state=42, verbose=False,
    max_workers=None
):
    """
    Compare single-task vs multi-task feature selection using parallel processing.

    Each split is processed in a separate process to avoid race conditions.
    Results are collected safely after all workers complete.

    Parameters
    ----------
    X_app, X_ddos : array-like
        Feature matrices for App and DDoS datasets
    y_app, y_ddos : array-like
        Target vectors
    n_trees, max_depth, max_blocks : int
        Model training constraints
    feature_names : list
        Feature names
    n_splits : int
        Number of train/test splits
    random_state : int
        Random seed (default: 42)
    verbose : bool
        Print progress (default: False)
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(n_splits, cpu_count - 1).

    Returns
    -------
    results_df : pd.DataFrame
        Results with columns for each (method, regularization_value, k)
    """
    import os

    if X_app.shape[1] != X_ddos.shape[1]:
        raise ValueError("Both datasets must have the same number of features")

    print(f"Starting PARALLEL comparison with {n_splits} splits")
    print(f"App dataset shape: {X_app.shape}, DDoS dataset shape: {X_ddos.shape}")
    print("-" * 70)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(n_splits, max(1, os.cpu_count() - 1))

    print(f"Using {max_workers} parallel workers")

    all_results = []
    completed = 0
    failed = 0

    # Use ProcessPoolExecutor for true parallelism (avoids GIL)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all splits
        futures = {
            executor.submit(
                _process_single_split,
                split_idx,
                X_app, X_ddos, y_app, y_ddos,
                n_trees, max_depth, max_blocks,
                feature_names,
                random_state, verbose
            ): split_idx
            for split_idx in range(10, 10 + n_splits)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            split_idx = futures[future]

            try:
                result: SplitResult = future.result()

                if result.error:
                    print(f"Split {result.split_idx} failed: {result.error}")
                    failed += 1
                else:
                    all_results.extend(result.results)
                    completed += 1
                    print(f"Completed split {result.split_idx} ({completed}/{n_splits})")

            except Exception as e:
                print(f"Split {split_idx} raised exception: {e}")
                failed += 1

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    print(f"\nCompleted {completed} splits, {failed} failed")
    print(f"Total experiments: {len(results_df)}")

    if len(results_df) > 0:
        print(f"Methods: {results_df['method'].value_counts().to_dict()}")
        print(f"k values: {sorted(results_df['k'].unique())}")

    return results_df