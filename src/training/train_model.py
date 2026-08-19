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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from src.p4gen.build_p4_script import dt_thresholds_float_to_int, MAX_CODEWORD_LENGTH
from src.p4gen.evaluation import multi_model_memory_evaluation
from src.training.threshold_alignment import align_rf_thresholds

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


def train_multi_RF_Optuna_multi_constrained(X_A, y_A, X_B, y_B, x_val_A, y_val_A, x_val_B, y_val_B, features_A, features_B, n_trees, max_depth, max_blocks, encoding, warm_start_params=None):
    """
    Train multi-model Random Forest with Optuna optimization.

    Parameters
    ----------
    warm_start_params : dict, optional
        Previously found good hyperparameters to enqueue as starting point.
        Should contain keys like 'n_estimators_A', 'max_depth_A', etc.
    """

    num_cross_validation_splits = 3

    if n_trees == -1:
        n_trees = 7

    if max_depth == -1:
        max_depth = 10

    # Cache models from feasible trials: trial_number -> (model_A, model_B, stages, blocks)
    models_cache = {}

    def objective(trial):
        params_A = {
            'n_estimators': trial.suggest_int('n_estimators_A', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_A', 5, 200, step=10),
            'min_samples_split': trial.suggest_int('min_samples_split_A', 10, 400, step=10),
            'max_depth': trial.suggest_int('max_depth_A', 2, max_depth),
            'random_state': 42
        }
        
        params_B = {
            'n_estimators': trial.suggest_int('n_estimators_B', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_B', 5, 200, step=10),
            'min_samples_split': trial.suggest_int('min_samples_split_B', 10, 400, step=10),
            'max_depth': trial.suggest_int('max_depth_B', 2, max_depth),
            'random_state': 42
        }

        skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
        skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

        score_vect_A = []
        score_vect_B = []
        memory_blocks_list = []
        
        # Track best fold's models (by blocks, then accuracy)
        best_fold_model_A = None
        best_fold_model_B = None
        best_fold_stages = None
        best_fold_blocks = float('inf')
        best_fold_accuracy = -1.0

        for (train_index_A, test_index_A), (train_index_B, test_index_B) in zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B)):
            x_train_A, x_test_A = X_A[train_index_A], X_A[test_index_A]
            y_train_A, y_test_A = y_A[train_index_A], y_A[test_index_A]
            x_train_B, x_test_B = X_B[train_index_B], X_B[test_index_B]
            y_train_B, y_test_B = y_B[train_index_B], y_B[test_index_B]

            model_A = RandomForestClassifier(**params_A, n_jobs=-1)
            model_B = RandomForestClassifier(**params_B, n_jobs=-1)

            with sklearn.config_context(assume_finite=True):
                model_A.fit(x_train_A, y_train_A)
                model_B.fit(x_train_B, y_train_B)

            model_A = dt_thresholds_float_to_int(model_A)
            model_B = dt_thresholds_float_to_int(model_B)

            if encoding == 'joint':
                model_A, model_B = align_rf_thresholds(
                    model_A, model_B,
                    x_val_A, y_val_A,
                    x_val_B, y_val_B,
                    overlap_threshold=0.5)

            try:
                stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
                
                if blocks > max_blocks:
                    trial.set_user_attr('codeword_violation', 0.0)
                    trial.set_user_attr('blocks_violation', blocks - max_blocks)
                    return -1.0, float('inf')
                    
            except RuntimeError as e:
                codeword_length = e.args[1]
                trial.set_user_attr('codeword_violation', codeword_length - MAX_CODEWORD_LENGTH)
                trial.set_user_attr('blocks_violation', 0.0)
                return -1.0, float('inf')

            memory_blocks_list.append(blocks)

            with sklearn.config_context(assume_finite=True):
                acc_A = model_A.score(x_test_A, y_test_A)
                acc_B = model_B.score(x_test_B, y_test_B)
                score_vect_A.append(acc_A)
                score_vect_B.append(acc_B)
            
            # Keep the fold with best accuracy (tie-break by lowest blocks)
            fold_accuracy = (acc_A + acc_B) / 2
            if fold_accuracy > best_fold_accuracy or (fold_accuracy == best_fold_accuracy and blocks < best_fold_blocks):
                best_fold_model_A = model_A
                best_fold_model_B = model_B
                best_fold_stages = stages
                best_fold_blocks = blocks
                best_fold_accuracy = fold_accuracy

        # Only reached if all folds pass constraints
        avg_accuracy = (np.mean(score_vect_A) + np.mean(score_vect_B)) / 2
        max_blocks_used = np.max(memory_blocks_list)
        
        trial.set_user_attr('memory_blocks_list', memory_blocks_list)
        trial.set_user_attr('avg_accuracy', avg_accuracy)
        trial.set_user_attr('codeword_violation', 0.0)
        trial.set_user_attr('blocks_violation', 0.0)
        
        # Cache the best fold's models for this trial
        models_cache[trial.number] = (best_fold_model_A, best_fold_model_B, best_fold_stages, best_fold_blocks)

        return avg_accuracy, max_blocks_used

    def constraints_func(trial):
        codeword = trial.user_attrs.get('codeword_violation', float('inf'))
        blocks = trial.user_attrs.get('blocks_violation', float('inf'))
        return [codeword, blocks]

    def early_stopping_callback(study, trial):
        
        feasible = [t for t in study.trials 
                    if t.state == optuna.trial.TrialState.COMPLETE 
                    and all(c <= 0 for c in constraints_func(t))]
        
        if len(feasible) < 25:
            return
        
        # Check Pareto front stability
        feasible_pareto = [t for t in study.best_trials if all(c <= 0 for c in constraints_func(t))]
        
        if len(feasible_pareto) == 0:
            return  # No feasible Pareto solutions yet, keep searching
        
        lookback = 20
        recent_threshold = len(study.trials) - lookback
        new_pareto = [t for t in feasible_pareto if t.number >= recent_threshold]

        if len(new_pareto) == 0:
            study.stop()

    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
        warn_independent_sampling=True,
        constant_liar=True,
        constraints_func=constraints_func
    )

    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=sampler
    )

    # Warm-start: enqueue previous good parameters if provided
    if warm_start_params is not None:
        try:
            # Enqueue the exact previous best
            study.enqueue_trial(warm_start_params)

            # Enqueue variations around the previous best for diversity
            for _ in range(2):
                varied_params = _vary_hyperparams(warm_start_params, n_trees, max_depth)
                study.enqueue_trial(varied_params)
        except Exception:
            pass  # Ignore if enqueue fails (e.g., invalid params)

    study.optimize(
        objective,
        n_trials=1000,
        callbacks=[early_stopping_callback],
        catch=(RuntimeError,)
    )

    # Filter feasible trials
    feasible_trials = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE
                       and all(c <= 0 for c in constraints_func(t))]

    if not feasible_trials:
        raise RuntimeError('No feasible solutions found')

    # Find best accuracy among feasible
    best_accuracy = max(t.user_attrs['avg_accuracy'] for t in feasible_trials)
    accuracy_threshold = 0.0025

    # Filter solutions within accuracy threshold, pick minimal memory
    close_to_best = [t for t in feasible_trials 
                     if t.user_attrs['avg_accuracy'] >= best_accuracy - accuracy_threshold]
    
    best_trial = min(close_to_best, 
                     key=lambda t: np.max(t.user_attrs['memory_blocks_list']))

    print(f"\n=== BEST SOLUTION ===")
    print(f"Parameters: {best_trial.params}")
    print(f"Accuracy: {best_trial.user_attrs['avg_accuracy']:.4f}")
    print(f"Memory usage per fold: {best_trial.user_attrs['memory_blocks_list']}")

    # Return cached models from the best trial, plus best params for warm-starting
    model_A, model_B, stages, blocks = models_cache[best_trial.number]

    return model_A, model_B, stages, blocks, best_trial.params