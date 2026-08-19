import os

# Intel's oneDAL acceleration is OPT-IN and off by default. Two reasons, both
# fatal to this pipeline when it is on:
#   1. In the PolimiML env its RandomForestClassifier cannot round-trip trees
#      through scikit-learn 1.6.1's Tree.__setstate__, so every .fit() raises
#      "node array from the pickle has an incompatible dtype".
#   2. This project mutates tree_.threshold IN PLACE after fitting
#      (dt_thresholds_float_to_int, align_rf_thresholds). oneDAL keeps its own
#      internal model representation, so those mutations may not be observed --
#      which would silently invalidate every result rather than crash.
# Set THESIS_USE_SKLEARNEX=1 only after verifying both of the above.
if os.environ.get('THESIS_USE_SKLEARNEX') == '1':
  try:
    from sklearnex import patch_sklearn
    patch_sklearn()
  except ImportError:
    pass

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

from src.p4gen.build_p4_script import dt_thresholds_float_to_int, MAX_CODEWORD_LENGTH
from src.p4gen.evaluation import multi_model_memory_evaluation
from src.p4gen.switch_semantics import switch_predict
from src.training.threshold_alignment import align_rf_thresholds
from src.training import early_stopping
from src.training import trial_selection

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def _vary_hyperparams(params: dict, n_trees: int, max_depth: int) -> dict:
    """
    Create a variation of hyperparameters for warm-start diversity.

    Randomly varies some parameters by small amounts to explore
    the neighborhood of the previous best solution.
    """
    varied = params.copy()

    for key in varied:
        if 'n_estimators' in key:
            delta = np.random.choice([-2, 0, 2])
            varied[key] = max(1, min(n_trees, varied[key] + delta))
        elif 'max_depth' in key:
            delta = np.random.choice([-1, 0, 1])
            varied[key] = max(2, min(max_depth, varied[key] + delta))
        elif 'min_samples_leaf' in key:
            delta = np.random.choice([-20, -10, 0, 10, 20])
            varied[key] = max(5, min(200, varied[key] + delta))
        elif 'min_samples_split' in key:
            delta = np.random.choice([-20, -10, 0, 10, 20])
            varied[key] = max(10, min(400, varied[key] + delta))

    return varied


def train_multi_RF_Optuna_multi_constrained(
        X_A, y_A, X_B, y_B,
        val_align_A, val_align_B,
        val_select_A, val_select_B,
        features_A, features_B,
        max_blocks, encoding, cfg,
        warm_start_params=None):
    """Search hyperparameters for both tasks under a shared block budget.

    Spec B.1/B.2. One fit per trial on the FULL training set: that single model
    is both what the constraint is measured on and what is returned, so
    `blocks <= max_blocks` holds for the shipped artifact rather than for a
    CV-fold proxy (F7). Both the objective and the final selection are
    per-task (F11 sites 1-3).

    val_align_*, val_select_* : (X, y) tuples. val_align serves alignment and
        nothing else; val_select serves this objective and, one level up,
        permutation_importance. Disjoint by construction (splits.py).

    Returns (model_A, model_B, stages, blocks, acc_sel_A, acc_sel_B, best_params).

    Raises NoFeasibleSolution when no trial satisfies both the block budget and
    the codeword limit -- an expected outcome at tight max_blocks, handled per-k
    by the caller.
    """

    def rf_params(source, suffix):
        """Both the objective and the final refit build params the same way, so
        the two cannot drift. `source` is a trial (during the search) or a plain
        params dict (when refitting the winner)."""
        if hasattr(source, 'suggest_int'):
            return {
                'n_estimators': source.suggest_int('n_estimators_' + suffix, 1, cfg.n_trees, step=2),
                'min_samples_leaf': source.suggest_int('min_samples_leaf_' + suffix, 5, 200, step=10),
                'min_samples_split': source.suggest_int('min_samples_split_' + suffix, 10, 400, step=10),
                'max_depth': source.suggest_int('max_depth_' + suffix, 2, cfg.max_depth),
                'random_state': 42,
            }
        return {
            'n_estimators': source['n_estimators_' + suffix],
            'min_samples_leaf': source['min_samples_leaf_' + suffix],
            'min_samples_split': source['min_samples_split_' + suffix],
            'max_depth': source['max_depth_' + suffix],
            'random_state': 42,
        }

    def fit_pair(params_A, params_B, align_stats=None):
        """ONE fit per task on the FULL training set -- this IS the deployment
        model, not a CV fold (F7).

        n_jobs=1: the parallelism is across splits (11 workers on 12 logical
        cores), so -1 here oversubscribes.

        Fully deterministic given (params, data): random_state is fixed above,
        and align_rf_thresholds is a deterministic function of the two models
        and val_align. That determinism is what lets the winner be REFIT below
        instead of cached (F8) -- a measured 401 KB per model pair, ~100 pairs
        per search, per worker.
        """
        with sklearn.config_context(assume_finite=True):
            model_A = RandomForestClassifier(**params_A, n_jobs=1).fit(X_A, y_A)
            model_B = RandomForestClassifier(**params_B, n_jobs=1).fit(X_B, y_B)

        model_A = dt_thresholds_float_to_int(model_A)
        model_B = dt_thresholds_float_to_int(model_B)

        if encoding == 'joint' and cfg.alignment_enabled:
            model_A, model_B = align_rf_thresholds(
                model_A, model_B,
                val_align_A[0], val_align_A[1],
                val_align_B[0], val_align_B[1],
                overlap_threshold=cfg.overlap_threshold,
                delta_rel=cfg.delta_align,
                align_stats=align_stats,
                endpoint_ratio_cap=cfg.endpoint_ratio_cap)

        return model_A, model_B

    def objective(trial):
        # (a) The single fit.
        align_stats = {}
        model_A, model_B = fit_pair(rf_params(trial, 'A'), rf_params(trial, 'B'),
                                   align_stats=align_stats)
        for name, value in align_stats.items():
            trial.set_user_attr('align_' + name, value)

        # (b) Constraint, on the shipped artifact -- exact, no proxy.
        try:
            stages, blocks = multi_model_memory_evaluation(
                model_A, model_B, features_A, features_B, encoding)
        except RuntimeError as e:
            trial.set_user_attr('codeword_violation', e.args[1] - MAX_CODEWORD_LENGTH)
            trial.set_user_attr('blocks_violation', 0.0)
            return -1.0, -1.0, float('inf')

        if blocks > max_blocks:
            trial.set_user_attr('codeword_violation', 0.0)
            trial.set_user_attr('blocks_violation', blocks - max_blocks)
            return -1.0, -1.0, float('inf')

        # (c) Only FEASIBLE trials pay for scoring. At tight max_blocks most of
        # the search is infeasible, so this ordering preserves the early bail.
        #
        # switch_predict, NOT model.score: score() averages predict_proba (a
        # SOFT vote) while the generated switch votes hard over per-tree
        # classes. Optimising the soft number ranks trials by an accuracy the
        # switch never reaches -- measured at up to 1.7 points, and up to +55%
        # relative error on DDoS, wider than the entire delta_align grid. Worse,
        # the gap widens with min_samples_leaf, so the soft objective pushes the
        # search toward exactly the configurations where its own number is least
        # honest. See P1 Task 7.
        with sklearn.config_context(assume_finite=True):
            acc_A = accuracy_score(val_select_A[1], switch_predict(model_A, val_select_A[0]))
            acc_B = accuracy_score(val_select_B[1], switch_predict(model_B, val_select_B[0]))

        trial.set_user_attr('acc_app', acc_A)
        trial.set_user_attr('acc_ddos', acc_B)
        trial.set_user_attr('blocks', int(blocks))
        trial.set_user_attr('stages', int(stages))
        trial.set_user_attr('codeword_violation', 0.0)
        trial.set_user_attr('blocks_violation', 0.0)

        # Blocks stays a third OBJECTIVE rather than a pure constraint because
        # plots use REALIZED blocks as an axis: without minimize-blocks pressure
        # the realized values pile up near max_blocks and that axis loses all
        # resolution.
        return acc_A, acc_B, float(blocks)

    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
        warn_independent_sampling=True,
        constant_liar=True,
        constraints_func=early_stopping.constraint_values,
    )

    study = optuna.create_study(
        directions=['maximize', 'maximize', 'minimize'],
        sampler=sampler)

    if warm_start_params is not None:
        try:
            study.enqueue_trial(warm_start_params)
            for _ in range(2):
                study.enqueue_trial(
                    _vary_hyperparams(warm_start_params, cfg.n_trees, cfg.max_depth))
        except Exception:
            pass  # Ignore if enqueue fails (e.g. bounds changed between k)

    # catch=() deliberately: the ONLY expected RuntimeError is the codeword
    # violation, and the objective already handles that itself. The previous
    # catch=(RuntimeError,) also swallowed align_rf_thresholds' own
    # RuntimeError("Smth is very-very wrong"), hiding corrupted interval
    # bookkeeping for a whole campaign. F3a/F3b make a dead split survivable,
    # so surfacing the error costs one split's tail rather than a silent lie.
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        callbacks=[early_stopping.ParetoStagnationStopper(
            min_feasible=cfg.min_feasible_before_stop, lookback=cfg.lookback)],
        catch=())

    feasible_trials = [t for t in study.trials if early_stopping.is_feasible(t)]

    best_trial, shortfall = trial_selection.select_best_trial(
        feasible_trials, cfg.delta_select, k=len(features_A), max_blocks=max_blocks)

    # Refit the winner rather than caching every feasible trial's model pair
    # (F8: a measured 401 KB per pair, ~100 pairs per search, 11 workers). One
    # extra fit-pair, ~550 ms, against ~40 MB of live cache per worker.
    #
    # Re-measuring here is not redundant: it is the B.1 invariant checked on the
    # artifact that is actually returned, and it fails loudly if anything in the
    # fit -> align -> measure path is non-deterministic.
    model_A, model_B = fit_pair(rf_params(best_trial.params, 'A'),
                                rf_params(best_trial.params, 'B'))
    stages, blocks = multi_model_memory_evaluation(
        model_A, model_B, features_A, features_B, encoding)

    if blocks != best_trial.user_attrs['blocks'] or blocks > max_blocks:
        raise AssertionError(
            'refit of trial {} gave {} blocks, the search recorded {} '
            '(max {}) -- the pipeline is not deterministic'.format(
                best_trial.number, blocks, best_trial.user_attrs['blocks'], max_blocks))

    return (model_A, model_B, int(stages), int(blocks),
            best_trial.user_attrs['acc_app'], best_trial.user_attrs['acc_ddos'],
            dict(best_trial.params))