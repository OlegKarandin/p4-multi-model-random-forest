import json

import numpy as np
from sklearn.inspection import permutation_importance

import sklearn
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.training.errors import NoFeasibleSolution
from src.training.config import TrainConfig
from src.training.splits import make_task_splits
from src.p4gen import p4_gen_config


# =============================================================================
# PARALLEL VERSION
#
# The formerly-sequential compare_feature_selection_approaches (single
# process, one arm hardcoded per loop body) was deleted here: after Task 6's
# rewrite of main.compare_independent_joint_mapping to call ONLY the parallel
# driver below, it had no remaining caller anywhere in the repo and was never
# covered by a test, so keeping it around would have left a second, silently
# broken code path still built on _process_single_split's old signature.
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
    """Derives a `feature_intervals` dict for one model, via
    `build_p4_script.get_feature_intervals` -- the exact same tree_nodes ->
    thresholds -> intervals code path `evaluation.single_model_memory_evaluation`
    uses internally (`generate_P4_code` recomputes codewords from
    `clf`/`feature_intervals` on its own, so only the intervals are needed
    here).
    """
    from src.p4gen.build_p4_script import get_feature_intervals

    return get_feature_intervals(clf, feature_names)


def _derive_joint_feature_intervals(model_app, model_ddos, feature_names_app, feature_names_ddos):
    """Derives ONE shared `feature_intervals` dict for both models, via
    `build_p4_script.get_joint_feature_intervals` -- the same offset trick
    `evaluation.multi_model_memory_evaluation`'s 'joint' branch uses
    internally, merging both models' trees into one keyed structure before
    deriving intervals -- this is the analytical model the real compiled
    numbers are meant to validate against, so the real program must be built
    from the same interval derivation, not a re-invented one.
    """
    from src.p4gen.build_p4_script import get_joint_feature_intervals

    return get_joint_feature_intervals(
        model_app, feature_names_app, model_ddos, feature_names_ddos)


def _kickoff_hardware_validation(validate_on_hardware, hardware_output_dir, split_idx, method, k,
                                  model_app, model_ddos, feature_names_app, feature_names_ddos, encoding,
                                  config: Optional[p4_gen_config.P4GenConfig] = None):
    """Kicks off (non-blocking) real-compiler validation for one iteration's
    trained model(s). Returns None (never a handle) when validate_on_hardware
    is False, preserving today's zero-cost behavior exactly. Otherwise
    returns the raw Future from `compile_p4_async`, exposing
    `.result(timeout=...)`.

    Task 3: `generate_P4_code` now represents genuine disjoint encoding
    (two independently-derived, possibly differently-discretized
    feature_intervals for a feature both models select) inside ONE combined
    P4 program via `_resolve_disjoint_feature_plan` -- so 'joint' and
    'disjoint' no longer need different real-compile strategies. Both
    branches generate ONE P4 program and issue ONE `compile_p4_async` call,
    matching real production deployment (both tasks always ship together in
    one combined pipeline -- confirmed by the project owner, not optional).
    This removes the need for the prior plan's two-separate-programs
    approximation (`_MergedCompileHandle`, since removed): that
    approximation over-counted by paying for two full parser/deparser
    pipelines instead of one shared one.

    encoding == 'joint': both models share ONE feature_intervals dict
    (`_derive_joint_feature_intervals`), which _resolve_disjoint_feature_plan
    resolves to entirely shared (non-namespaced) entries.

    encoding == 'disjoint': each model's OWN, independently-derived
    feature_intervals (`_derive_feature_intervals`) is passed straight
    through as feature_intervals_app/feature_intervals_ddos --
    `generate_P4_code` resolves per-feature sharing vs namespacing itself.

    config: forwarded verbatim to `generate_P4_code` in both branches, so
    `P4GenConfig.match_type` / `use_default_action_discount` actually take
    effect on this real-compiler-validation path (they used to be silently
    dropped here). `feature_names_app`/`feature_names_ddos` are additionally
    passed as `generate_P4_code`'s `selected_features_app`/
    `selected_features_ddos`: they ARE the ordered training-feature-name
    lists it needs to recompute codewords for the default-action discount.
    None (the default) leaves `generate_P4_code` on its own defaults, exactly
    as before.
    """
    if not validate_on_hardware:
        return None

    from src.p4gen.p4_compile import compile_p4_async
    from src.p4gen.build_p4_script import generate_P4_code

    if encoding == 'joint':
        feature_intervals = _derive_joint_feature_intervals(
            model_app, model_ddos, feature_names_app, feature_names_ddos)

        filename = f"split{split_idx}_{method}_k{k}.p4"
        written_path = generate_P4_code(
            3, 2, model_app, model_ddos,
            feature_intervals_app=feature_intervals, feature_intervals_ddos=feature_intervals,
            output_dir=hardware_output_dir, output_filename=filename,
            selected_features_app=feature_names_app,
            selected_features_ddos=feature_names_ddos,
            config=config)
        log_dir = hardware_output_dir + f"logs_split{split_idx}_{method}_k{k}/"
        return compile_p4_async(written_path, log_dir)

    elif encoding == 'disjoint':
        feature_intervals_app = _derive_feature_intervals(model_app, feature_names_app)
        feature_intervals_ddos = _derive_feature_intervals(model_ddos, feature_names_ddos)

        filename = f"split{split_idx}_{method}_k{k}.p4"
        written_path = generate_P4_code(
            3, 2, model_app, model_ddos,
            feature_intervals_app=feature_intervals_app,
            feature_intervals_ddos=feature_intervals_ddos,
            output_dir=hardware_output_dir, output_filename=filename,
            selected_features_app=feature_names_app,
            selected_features_ddos=feature_names_ddos,
            config=config)
        log_dir = hardware_output_dir + f"logs_split{split_idx}_{method}_k{k}/"
        return compile_p4_async(written_path, log_dir)

    else:
        raise ValueError(f"Unknown encoding for hardware validation: {encoding!r}")


def _advance_pending_compile(current_row, pending_previous, pending_next):
    """Joins the PREVIOUS iteration's still-pending hardware-validation handle
    -- which has now had one full training step's wall time to finish in the
    background -- attaching its numbers directly to the row it belongs to.

    `pending_previous` and `pending_next` are each either None or a
    `(handle, row_dict)` pair: the handle is the raw Future returned by
    `_kickoff_hardware_validation`, and `row_dict` is the specific results
    row that Future's numbers must land on. Writing into `row_dict` by
    reference -- rather than indexing `results` by position (`results[-2]`)
    -- means this no longer depends on exactly one row having been appended
    per call. A feasible iteration's infeasible-k row(s) can be interleaved
    in between (F3b's infeasible branch appends a row and `continue`s
    without ever producing a pending compile) and the attribution still
    lands on the correct row, because that row's dict was captured at
    kickoff time, not looked up again later by position.

    `current_row` is the just-appended row for THIS iteration; when nothing
    is pending yet (`pending_previous` is None, i.e. the very first
    iteration), its three fields are marked None directly instead.

    Shared by both the disjoint ('single') and joint ('multi') loops in
    `_process_single_split`, which differ only in how `pending_next` itself
    was produced (each calls `_kickoff_hardware_validation` with its own
    args) -- the splicing logic that follows is identical.

    Returns the new `pending_previous` value (i.e. `pending_next`) for the
    loop to carry into its next iteration.
    """
    if pending_previous is not None:
        handle, row = pending_previous
        compile_result = handle.result(timeout=600)
        row['stages_real'] = compile_result.stages
        row['tcam_real'] = compile_result.tcam
        row['compile_errors'] = compile_result.errors
    else:
        current_row['stages_real'] = None
        current_row['tcam_real'] = None
        current_row['compile_errors'] = None
    return pending_next


def _join_final_pending_compile(pending_previous):
    """Post-loop counterpart to `_advance_pending_compile`: the final
    iteration has no "next" iteration to overlap with, so whatever compile is
    still outstanding is joined directly here and attached to the row it
    belongs to. `pending_previous` is either None or a `(handle, row_dict)`
    pair (see `_advance_pending_compile`); writing into `row_dict` by
    reference means this is unaffected by any infeasible rows appended after
    the handle was captured. No-op when `pending_previous` is None (either
    validate_on_hardware was False throughout, or -- impossible in practice,
    since every iteration kicks off a new pending compile -- there simply was
    none left outstanding).
    """
    if pending_previous is not None:
        handle, row = pending_previous
        compile_result = handle.result(timeout=600)
        row['stages_real'] = compile_result.stages
        row['tcam_real'] = compile_result.tcam
        row['compile_errors'] = compile_result.errors


def _run_elimination(arm, split_idx, app, ddos, feature_names, max_blocks, cfg,
                     validate_on_hardware=False, hardware_output_dir=None,
                     config=None, rows=None):
    """Recursive feature elimination for ONE arm.

    Replaces two ~90-line near-duplicate loops that differed in exactly three
    ways: the encoding, whether the two tasks share one feature list, and
    whether the importance ranking is per-task or summed. Everything else was
    verbatim duplication, so every schema or signature change had to be made
    twice and could silently diverge.

    arm : 'independent' (disjoint encoding, each task eliminates its own
        features) or 'joint' (joint encoding, ONE shared feature set --
        required, because under joint encoding every tree's codeword spans the
        merged interval pool, so a feature used by only one task widens BOTH
        tasks' codewords).
    app, ddos : TaskSplits.

    Both arms drop exactly one feature per task per iteration, so the two
    feature lists always have equal length and one `k` describes the row.

    rows : list, optional
        Caller-owned accumulator. When given, each completed row is appended
        directly into THIS list (by reference) as soon as it's produced,
        instead of a local list only handed back on a clean return. That
        matters because `train_multi_RF_Optuna_multi_constrained` can raise
        something other than `NoFeasibleSolution` (e.g. an
        `AlignmentInvariantError` from alignment, or a determinism
        `AssertionError`) mid-loop -- with a purely local accumulator, that
        exception would propagate out of this function with every row
        already completed for this split silently discarded. Passing the
        caller's own list (`_process_single_split` passes its `results`)
        means those rows survive the raise. Defaults to a fresh list when
        omitted, preserving the previous return-only-on-clean-exit behavior
        for callers that don't need mid-raise durability (this function
        still returns `rows` either way).
    """
    from src.training.train_model import train_multi_RF_Optuna_multi_constrained
    from src.p4gen.evaluation import accuracy_metrics
    from src.p4gen.switch_semantics import switch_accuracy_scorer, switch_predict

    if arm not in ('independent', 'joint'):
        raise ValueError("arm must be 'independent' or 'joint', got {!r}".format(arm))

    shared = (arm == 'joint')
    encoding = 'joint' if shared else 'disjoint'
    method = 'multi' if shared else 'single'

    remaining_app = list(range(app.X_train.shape[1]))
    remaining_ddos = list(remaining_app)
    names_app = list(feature_names)
    names_ddos = list(feature_names)

    if rows is None:
        rows = []
    warm_start_params = None
    pending_previous = None
    # F3b: importance vectors positionally aligned with remaining_*, carried so
    # an infeasible k can still drop a feature along the last feasible model's
    # ranking. None until the first feasible iteration.
    carried_app = None
    carried_ddos = None

    while True:
        k = len(remaining_app)

        try:
            train_result = train_multi_RF_Optuna_multi_constrained(
                app.X_train[:, remaining_app], app.y_train,
                ddos.X_train[:, remaining_ddos], ddos.y_train,
                (app.X_val_align[:, remaining_app], app.y_val_align),
                (ddos.X_val_align[:, remaining_ddos], ddos.y_val_align),
                (app.X_val_select[:, remaining_app], app.y_val_select),
                (ddos.X_val_select[:, remaining_ddos], ddos.y_val_select),
                names_app, names_ddos,
                max_blocks, encoding, cfg,
                warm_start_params)
        except NoFeasibleSolution as exc:
            rows.append({
                'arm': arm, 'method': method, 'split': split_idx, 'k': k,
                'acc_app': None, 'f1_app': None, 'acc_ddos': None, 'f1_ddos': None,
                'acc_sel_app': None, 'acc_sel_ddos': None,
                'stages': None, 'blocks': None,
                'infeasible': str(exc),
                'stages_real': None, 'tcam_real': None, 'compile_errors': None,
                'features_app': ';'.join(names_app), 'features_ddos': ';'.join(names_ddos),
                # best_params is deliberately '', NOT json.dumps(warm_start_params):
                # warm_start_params is still bound from the previous (feasible) k
                # here, so writing it would silently attribute the previous k's
                # parameters to this infeasible one.
                'best_params': '',
                'rel_shortfall': '', 'n_trials_run': '', 'n_feasible': '',
                'align_attempted': '', 'align_accepted': '',
                'intervals_before': '', 'intervals_after': '',
            })
            if carried_app is None or k == 1:
                break  # No ranking to continue from, or nothing left to drop.
            remaining_app, names_app, carried_app = _drop_least_important(
                remaining_app, names_app, carried_app)
            remaining_ddos, names_ddos, carried_ddos = _drop_least_important(
                remaining_ddos, names_ddos, carried_ddos)
            continue

        model_app, model_ddos = train_result.model_A, train_result.model_B
        stages, blocks = train_result.stages, train_result.blocks
        acc_sel_app, acc_sel_ddos = train_result.acc_sel_A, train_result.acc_sel_B
        best_params = train_result.best_params

        warm_start_params = best_params

        # switch_predict, NOT model.predict: these are the numbers the thesis
        # reports, so they must be the numbers the deployed switch produces.
        # rf.predict's soft vote is up to 1.7 points optimistic (P1 Task 7).
        with sklearn.config_context(assume_finite=True):
            acc_app, f1_app = accuracy_metrics(
                app.y_test, switch_predict(model_app, app.X_test[:, remaining_app]), task="app")
            acc_ddos, f1_ddos = accuracy_metrics(
                ddos.y_test, switch_predict(model_ddos, ddos.X_test[:, remaining_ddos]), task="ddos")

        rows.append({
            'arm': arm, 'method': method, 'split': split_idx, 'k': k,
            'acc_app': acc_app, 'f1_app': f1_app,
            'acc_ddos': acc_ddos, 'f1_ddos': f1_ddos,
            'acc_sel_app': acc_sel_app, 'acc_sel_ddos': acc_sel_ddos,
            'stages': stages, 'blocks': blocks,
            'infeasible': '',
            'features_app': ';'.join(names_app), 'features_ddos': ';'.join(names_ddos),
            'best_params': json.dumps(best_params),
            'rel_shortfall': train_result.rel_shortfall,
            'n_trials_run': train_result.n_trials_run,
            'n_feasible': train_result.n_feasible,
            # None (not 0) means alignment never ran for this arm/config --
            # preserve that distinction as '' rather than erasing it with a
            # falsy test (a real align_accepted of 0 must stay 0, not '').
            'align_attempted': train_result.align_attempted if train_result.align_attempted is not None else '',
            'align_accepted': train_result.align_accepted if train_result.align_accepted is not None else '',
            'intervals_before': train_result.intervals_before if train_result.intervals_before is not None else '',
            'intervals_after': train_result.intervals_after if train_result.intervals_after is not None else '',
        })

        pending_next = _kickoff_hardware_validation(
            validate_on_hardware, hardware_output_dir, split_idx, method, k,
            model_app, model_ddos, names_app, names_ddos, encoding, config=config)
        if pending_next is not None:
            pending_next = (pending_next, rows[-1])
        pending_previous = _advance_pending_compile(rows[-1], pending_previous, pending_next)

        if k == 1:
            break

        # Importance on val_select -- disjoint from val_align, so an aligned
        # feature's importance is no longer measured on the set alignment was
        # fitted to.
        #
        # scoring=switch_accuracy_scorer, not 'accuracy': the built-in scorer
        # calls estimator.predict, i.e. the soft vote, so feature importances
        # would rank features by their effect on semantics the switch does not
        # have -- and those rankings decide the entire elimination order.
        importance_app = permutation_importance(
            model_app, app.X_val_select[:, remaining_app], app.y_val_select,
            scoring=switch_accuracy_scorer, n_repeats=10, random_state=42,
        ).importances_mean
        importance_ddos = permutation_importance(
            model_ddos, ddos.X_val_select[:, remaining_ddos], ddos.y_val_select,
            scoring=switch_accuracy_scorer, n_repeats=10, random_state=42,
        ).importances_mean

        if shared:
            # One shared feature set, so one ranking: the sum over both tasks,
            # which are both scored on accuracy over the same column space.
            combined = importance_app + importance_ddos
            carried_app = combined
            carried_ddos = combined
        else:
            carried_app = importance_app
            carried_ddos = importance_ddos

        remaining_app, names_app, carried_app = _drop_least_important(
            remaining_app, names_app, carried_app)
        remaining_ddos, names_ddos, carried_ddos = _drop_least_important(
            remaining_ddos, names_ddos, carried_ddos)

    _join_final_pending_compile(pending_previous)
    return rows


def _drop_least_important(remaining, names, importance):
    """Drop the argmin of `importance` from all three lists, keeping them
    positionally aligned. Returns new (remaining, names, importance)."""
    idx = int(np.asarray(importance).argmin())
    remaining = list(remaining)
    names = list(names)
    del remaining[idx]
    del names[idx]
    return remaining, names, np.delete(np.asarray(importance), idx)


def _process_single_split(
    split_idx: int,
    X_app: np.ndarray,
    X_ddos: np.ndarray,
    y_app: np.ndarray,
    y_ddos: np.ndarray,
    max_blocks: int,
    feature_names: List[str],
    random_state: int,
    arm: str = 'independent',
    cfg: Optional[TrainConfig] = None,
    validate_on_hardware: bool = False,
    hardware_output_dir: Optional[str] = None,
    config: Optional[p4_gen_config.P4GenConfig] = None,
) -> SplitResult:
    """
    Process a single train/test split.

    This function is designed to be called in a separate process.
    It returns all results as a SplitResult object - no shared mutable state.

    Thin wrapper: builds each task's TaskSplits (Task 3) and delegates the
    actual elimination loop to `_run_elimination` (Task 5) for the single
    `arm` requested. `n_trees`/`max_depth` are gone from the parameter list --
    they live on `cfg` (F10i: no more `-1` sentinel); `arm` selects
    'independent' (disjoint encoding) or 'joint' encoding.

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
    config : p4_gen_config.P4GenConfig, optional
        Additive convenience: when given, `config.validate_on_hardware` /
        `config.hardware_output_dir` take precedence over the individual
        `validate_on_hardware` / `hardware_output_dir` keyword arguments
        above (which remain the source of truth when `config` is None, so
        every existing caller is unaffected). The object itself is also
        forwarded to `_kickoff_hardware_validation` (and from there to
        `generate_P4_code`), so its `match_type` /
        `use_default_action_discount` fields reach the real-compiler
        validation path too instead of being dropped here.
    """
    if cfg is None:
        cfg = TrainConfig()

    if config is not None:
        validate_on_hardware = config.validate_on_hardware
        hardware_output_dir = config.hardware_output_dir

    if validate_on_hardware and hardware_output_dir and not hardware_output_dir.endswith(('/', '\\')):
        hardware_output_dir = hardware_output_dir + "/"

    results = []
    try:
        split_random_state = random_state + split_idx
        app = make_task_splits(X_app, y_app, split_random_state)
        ddos = make_task_splits(X_ddos, y_ddos, split_random_state)

        # `results` is passed in as `_run_elimination`'s caller-owned `rows`
        # accumulator (not built from its return value via `.extend`): if
        # elimination raises mid-loop, every row completed before the raise
        # is already IN `results` by reference, so the `except` branch below
        # still returns them instead of silently discarding a split's worth
        # of completed work (F3a).
        _run_elimination(
            arm=arm, split_idx=split_idx, app=app, ddos=ddos,
            feature_names=feature_names, max_blocks=max_blocks, cfg=cfg,
            validate_on_hardware=validate_on_hardware,
            hardware_output_dir=hardware_output_dir, config=config,
            rows=results)

        return SplitResult(split_idx=split_idx, results=results)

    except Exception as e:
        import traceback
        return SplitResult(
            split_idx=split_idx,
            results=results,
            error=f"{str(e)}\n{traceback.format_exc()}"
        )


def _collect_split_results(split_results):
    """Fold SplitResults into (rows, n_completed, n_failed, n_partial).

    F3a: a split that raised partway carries BOTH `results` and `error`. The
    previous collector treated them as mutually exclusive, so one infeasible k
    at the end of a 17-step elimination discarded the 16 rows that had already
    succeeded. Rows are kept whenever present; the error is still counted and
    still printed by the caller.
    """
    rows = []
    n_completed = 0
    n_failed = 0
    n_partial = 0

    for result in split_results:
        if result.results:
            rows.extend(result.results)
        if result.error:
            n_failed += 1
            if result.results:
                n_partial += 1
        else:
            n_completed += 1

    return rows, n_completed, n_failed, n_partial


def compare_feature_selection_approaches_parallel(
    X_app, X_ddos, y_app, y_ddos,
    max_blocks,
    feature_names,
    n_splits,
    arm,
    cfg,
    random_state=42,
    max_workers=None,
    config: Optional[p4_gen_config.P4GenConfig] = None,
):
    """Run ONE arm across n_splits, each split in its own process.

    arm : 'independent' or 'joint'.
    cfg : TrainConfig -- the arm definition. Frozen, so it is safe to ship to
        every worker.

    One arm per call: the delta sweep has seven arms, and recomputing the
    independent baseline for each joint arm would spend about half the
    campaign's compute producing six identical copies of it.

    Each split is processed in a separate process to avoid race conditions.
    Results are collected safely after all workers complete.

    Parameters
    ----------
    X_app, X_ddos : array-like
        Feature matrices for App and DDoS datasets
    y_app, y_ddos : array-like
        Target vectors
    max_blocks : int
        Model training constraint (TCAM block budget)
    feature_names : list
        Feature names
    n_splits : int
        Number of train/test splits
    random_state : int
        Random seed (default: 42)
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(n_splits, cpu_count - 1).
    config : p4_gen_config.P4GenConfig, optional
        Forwarded as-is to each worker's `_process_single_split` call (its
        `validate_on_hardware` / `hardware_output_dir` fields take
        precedence there when given). Defaults to None, so every existing
        caller's behavior (no hardware validation ever kicked off through
        this parallel path) is unchanged.

    Returns
    -------
    results_df : pd.DataFrame
        Results with columns for each (method, regularization_value, k)
    """
    import os

    if X_app.shape[1] != X_ddos.shape[1]:
        raise ValueError("Both datasets must have the same number of features")

    encoding = 'joint' if arm == 'joint' else 'disjoint'
    print(f"Starting arm {cfg.arm_slug(encoding)} at M={max_blocks} over {n_splits} splits")
    print(f"App dataset shape: {X_app.shape}, DDoS dataset shape: {X_ddos.shape}")
    print("-" * 70)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(n_splits, max(1, os.cpu_count() - 1))

    print(f"Using {max_workers} parallel workers")

    collected = []

    # Use ProcessPoolExecutor for true parallelism (avoids GIL)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_split,
                split_idx,
                X_app, X_ddos, y_app, y_ddos,
                max_blocks,
                feature_names,
                random_state,
                arm, cfg,
                config=config,
            ): split_idx
            for split_idx in range(10, 10 + n_splits)
        }

        for future in as_completed(futures):
            split_idx = futures[future]
            try:
                result: SplitResult = future.result()
            except Exception as e:
                print(f"Split {split_idx} raised exception: {e}")
                collected.append(SplitResult(split_idx=split_idx, results=[], error=str(e)))
                continue

            if result.error:
                print(f"Split {result.split_idx} failed after "
                      f"{len(result.results)} rows: {result.error}")
            else:
                print(f"Completed split {result.split_idx}")
            collected.append(result)

    all_results, completed, failed, partial = _collect_split_results(collected)
    results_df = pd.DataFrame(all_results)

    print(f"\nCompleted {completed} splits, {failed} failed ({partial} of them partial)")
    print(f"Total experiments: {len(results_df)}")

    if len(results_df) > 0:
        print(f"Methods: {results_df['method'].value_counts().to_dict()}")
        print(f"k values: {sorted(results_df['k'].unique())}")

    return results_df