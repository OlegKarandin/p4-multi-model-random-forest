import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import hyperopt

from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np

from src.p4gen.build_p4_script import dt_thresholds_float_to_int, MAX_CODEWORD_LENGTH
from src.p4gen.evaluation import single_model_memory_evaluation, multi_model_memory_evaluation, accuracy_metrics
from src.training.threshold_alignment import align_rf_thresholds

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

INTERMEDIATE = "temp/"

def train_single_RF(X, y, features, n_trees, max_depth):

    space4RF = {
        'n_estimators': hp.choice('n_estimators', np.arange(1, n_trees + 1, 2)),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(50, 201, 25)),
        #'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'min_samples_split': hp.choice('min_samples_split', np.arange(50, 1001, 50)),
        #'max_features': hp.choice('max_features', ['log2', 'sqrt']),
        'max_depth': hp.choice('max_depth', np.arange(2, max_depth + 1, 1))}

    def f(params):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        score_vect = []

        for train_index, val_index in skf.split(X, y):
            x_train, x_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

            model = RandomForestClassifier(**params, n_jobs = -1)
            model.fit(x_train, y_train)

            model = dt_thresholds_float_to_int(model)
            score_vect.append(model.score(x_val, y_val))

            try:
                single_model_memory_evaluation(model, features)
            except RuntimeError as e:
                print('Codeword length constraint failed', e.args[1])
                return {'status': STATUS_OK,
                        'loss': e.args[1] / 512,
                        'loss_variance': 0}
        
        return {'status': STATUS_OK,
                'loss': -np.mean(score_vect),
                'loss_variance': np.var(score_vect, ddof=1)}

    trials = Trials()
    best_params = fmin(f, space4RF, algo=tpe.suggest, max_evals=50, early_stop_fn=no_progress_loss(10), trials=trials)
    best_params = hyperopt.space_eval(space4RF, best_params)
    print(best_params)

    clf = RandomForestClassifier(**best_params, n_jobs = -1)
    clf.fit(X, y)

    return clf


def train_multi_RF(X_A, y_A, X_B, y_B, features_A, features_B, n_trees, max_depth, max_blocks, encoding):

    num_cross_validation_splits = 3

    if n_trees == -1:
        n_trees = 7

    if max_depth == -1:
        max_depth = 7

    space4RF = {
        'n_estimators_A': hp.choice('n_estimators_A', np.arange(1, n_trees + 1, 2)),
        'min_samples_leaf_A': hp.choice('min_samples_leaf_A', np.arange(10, 201, 20)),
        #'criterion_A': hp.choice('criterion_A', ['gini', 'entropy']),
        'min_samples_split_A': hp.choice('min_samples_split_A', np.arange(50, 1001, 50)),
        #'max_features_A': hp.choice('max_features_A', ['log2', 'sqrt']),
        'max_depth_A': hp.choice('max_depth_A', np.arange(2, max_depth + 1, 1)),
        
        'n_estimators_B': hp.choice('n_estimators_B', np.arange(1, n_trees + 1, 2)),
        'min_samples_leaf_B': hp.choice('min_samples_leaf_B', np.arange(10, 201, 20)),
        #'criterion_B': hp.choice('criterion_B', ['gini', 'entropy']),
        'min_samples_split_B': hp.choice('min_samples_split_B', np.arange(50, 1001, 50)),
        #'max_features_B': hp.choice('max_features_B', ['log2', 'sqrt']),
        'max_depth_B': hp.choice('max_depth_B', np.arange(2, max_depth + 1, 1)),
        }

    def f(params):

        params_A = {k.replace('_A', ''): v for k, v in params.items() if k.endswith('_A')}
        params_B = {k.replace('_B', ''): v for k, v in params.items() if k.endswith('_B')}

        skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
        skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

        score_vect_A = []
        score_vect_B = []

        for (train_index_A, val_index_A), (train_index_B, val_index_B) in zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B)):
            x_train_A, x_val_A, y_train_A, y_val_A = X_A[train_index_A], X_A[val_index_A], y_A[train_index_A], y_A[val_index_A]
            x_train_B, x_val_B, y_train_B, y_val_B = X_B[train_index_B], X_B[val_index_B], y_B[train_index_B], y_B[val_index_B]

            model_A = RandomForestClassifier(**params_A, n_jobs = -1)
            model_B = RandomForestClassifier(**params_B, n_jobs = -1)

            model_A.fit(x_train_A, y_train_A)
            model_B.fit(x_train_B, y_train_B)

            model_A = dt_thresholds_float_to_int(model_A)
            model_B = dt_thresholds_float_to_int(model_B)

            score_vect_A.append(model_A.score(x_val_A, y_val_A))
            score_vect_B.append(model_B.score(x_val_B, y_val_B))

            try:
                stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
            except RuntimeError as e:
                codeword_excess_ratio = np.exp((e.args[1] - MAX_CODEWORD_LENGTH) / MAX_CODEWORD_LENGTH)
                score_vect_A[-1] -= codeword_excess_ratio
                score_vect_B[-1] -= codeword_excess_ratio
                blocks = max_blocks

                #print('With {} codeword bits the score turns from {} to {}'.format(e.args[1], score_vect_A[-1] + codeword_excess_ratio, score_vect_A[-1]))
                '''return {'status': STATUS_OK,
                        'loss': e.args[1] / 512,
                        'loss_variance': 0}'''

            if blocks > max_blocks:
                '''return {'status': STATUS_OK,
                        'loss': blocks / max_blocks,
                        'loss_variance': 0}'''
                blocks_excess_ratio = np.exp((blocks - max_blocks) / max_blocks) #(blocks - max_blocks) / max_blocks
                score_vect_A[-1] -= blocks_excess_ratio
                score_vect_B[-1] -= blocks_excess_ratio

                #print('With {} blocks the score turns from {} to {}'.format(blocks, score_vect_A[-1] + blocks_excess_ratio, score_vect_A[-1]))

        return {'status': STATUS_OK,
                'loss': -(np.mean(score_vect_A) + np.mean(score_vect_B)) / 2,
                'loss_variance': np.var([score_vect_A, score_vect_B], ddof=1)}

    trials = Trials()
    best_params = fmin(f, space4RF, algo=tpe.suggest, max_evals=50, early_stop_fn=no_progress_loss(20), trials=trials)
    best_params = hyperopt.space_eval(space4RF, best_params)
    print(best_params)

    best_params_A = {k.replace('_A', ''): v for k, v in best_params.items() if k.endswith('_A')}
    best_params_B = {k.replace('_B', ''): v for k, v in best_params.items() if k.endswith('_B')}


    skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
    skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

    loss_vect = []
    model_vect = []

    for (train_index_A, val_index_A), (train_index_B, val_index_B) in zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B)):
        x_train_A, x_val_A, y_train_A, y_val_A = X_A[train_index_A], X_A[val_index_A], y_A[train_index_A], y_A[val_index_A]
        x_train_B, x_val_B, y_train_B, y_val_B = X_B[train_index_B], X_B[val_index_B], y_B[train_index_B], y_B[val_index_B]

        model_A = RandomForestClassifier(**best_params_A, n_jobs = -1)
        model_B = RandomForestClassifier(**best_params_B, n_jobs = -1)

        model_A.fit(x_train_A, y_train_A)
        model_B.fit(x_train_B, y_train_B)

        model_A = dt_thresholds_float_to_int(model_A)
        model_B = dt_thresholds_float_to_int(model_B)

        score_A = model_A.score(x_val_A, y_val_A)
        score_B = model_B.score(x_val_B, y_val_B)

        try:
            stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
        except RuntimeError as e:
            continue
        
        if blocks > max_blocks:
            continue
        
        model_vect.append((model_A, model_B))
        loss_vect.append(-(score_A + score_B) / 2)

    if len(model_vect) == 0:
        raise RuntimeError('No feasible split')

    return model_vect[np.argmin(loss_vect)]

    '''clf_A = RandomForestClassifier(**best_params_A, n_jobs = -1)
    clf_A.fit(X_A, y_A)

    clf_B = RandomForestClassifier(**best_params_B, n_jobs = -1)
    clf_B.fit(X_B, y_B)

    return clf_A, clf_B'''


def train_multi_RF_Optuna(X_A, y_A, X_B, y_B, features_A, features_B, n_trees, max_depth, max_blocks, encoding):
    num_cross_validation_splits = 2

    if n_trees == -1:
        n_trees = 7

    if max_depth == -1:
        max_depth = 6

    def objective(trial):
        # Suggest hyperparameters for both models
        params_A = {
            'n_estimators': trial.suggest_int('n_estimators_A', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_A', 10, 200, step=20),
            'min_samples_split': trial.suggest_int('min_samples_split_A', 50, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth_A', 2, max_depth)
        }
        
        params_B = {
            'n_estimators': trial.suggest_int('n_estimators_B', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_B', 10, 200, step=20),
            'min_samples_split': trial.suggest_int('min_samples_split_B', 50, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth_B', 2, max_depth)
        }

        skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
        skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

        score_vect_A = []
        score_vect_B = []
        memory_blocks_list = []
        codeword_length_check_failed = False
        codeword_length = 0

        for fold_idx, ((train_index_A, val_index_A), (train_index_B, val_index_B)) in enumerate(zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B))):
            x_train_A, x_val_A, y_train_A, y_val_A = X_A[train_index_A], X_A[val_index_A], y_A[train_index_A], y_A[val_index_A]
            x_train_B, x_val_B, y_train_B, y_val_B = X_B[train_index_B], X_B[val_index_B], y_B[train_index_B], y_B[val_index_B]

            model_A = RandomForestClassifier(**params_A, n_jobs=-1)
            model_B = RandomForestClassifier(**params_B, n_jobs=-1)

            model_A.fit(x_train_A, y_train_A)
            model_B.fit(x_train_B, y_train_B)

            model_A = dt_thresholds_float_to_int(model_A)
            model_B = dt_thresholds_float_to_int(model_B)

            score_vect_A.append(model_A.score(x_val_A, y_val_A))
            score_vect_B.append(model_B.score(x_val_B, y_val_B))

            try:
                stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
                memory_blocks_list.append(blocks)
            except RuntimeError as e:
                codeword_length = e.args[1]
                break

        # Store detailed memory information for analysis
        trial.set_user_attr('memory_blocks_list', memory_blocks_list)
        trial.set_user_attr('codeword_length', codeword_length)
        
        # Calculate base objectives
        avg_accuracy = (np.mean(score_vect_A) + np.mean(score_vect_B)) / 2
    
        
        # PENALTY CALCULATIONS
        total_penalty = 0.0
        
        # codeword length
        codeword_length_penalty = 0.0
        if codeword_length > MAX_CODEWORD_LENGTH:
            codeword_length_penalty = np.exp((e.args[1] - MAX_CODEWORD_LENGTH) / MAX_CODEWORD_LENGTH)
            total_penalty += codeword_length_penalty
        
        # Exceeding max memory blocks (any fold)
        max_memory_penalty = 0.0
        if memory_blocks_list:
            max_blocks_used = max(memory_blocks_list)
            if max_blocks_used > max_blocks:
                max_memory_penalty = np.exp((max_blocks_used - max_blocks) / max_blocks)
                total_penalty += max_memory_penalty
        
        # Normalize accuracy to [0, 1] range (assuming accuracy is between 0 and 1)
        normalized_accuracy_loss = 1.0 - avg_accuracy  # Convert to loss (higher is worse)
        
        # Combined single objective (to minimize)
        combined_objective = normalized_accuracy_loss + total_penalty
        
        # Store components for analysis
        trial.set_user_attr('base_accuracy', avg_accuracy)
        trial.set_user_attr('normalized_accuracy_loss', normalized_accuracy_loss)
        trial.set_user_attr('total_penalty', total_penalty)
        trial.set_user_attr('max_memory_penalty', max_memory_penalty)  
        
        return combined_objective

    # Create study with single objective optimization
    study = optuna.create_study(
        direction='minimize',  # Minimize the combined objective
        sampler=TPESampler(
            n_startup_trials=20,
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            warn_independent_sampling=True,
            constant_liar=True
        )
    )

    # Add early stopping based on no improvement
    def early_stopping_callback(study, trial):
        if len(study.trials) >= 20:
            # Get best value from last 10 trials
            recent_trials = study.trials[-10:]
            recent_best = min(t.value for t in recent_trials if t.state == optuna.trial.TrialState.COMPLETE)
            
            # Get overall best
            overall_best = study.best_value
            
            # Stop if no improvement in recent trials
            improvement = (overall_best - recent_best) / abs(overall_best) if overall_best != 0 else 0
            if improvement < 0.001:  # Less than 0.1% improvement
                study.stop()

    # Optimize the single objective
    study.optimize(
        objective, 
        n_trials=60,
        callbacks=[early_stopping_callback],
        catch=(RuntimeError,)
    )

    # Analyze results
    best_trial = study.best_trial
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    feasible_trials = [t for t in completed_trials if t.user_attrs.get('total_penalty', float('inf')) == 0.0]
    
    print(f"=== OPTIMIZATION RESULTS ===")
    print(f"Total completed trials: {len(completed_trials)}")
    print(f"Feasible trials (no penalties): {len(feasible_trials)}")
    print(f"Best trial found at trial #{best_trial.number}")
    
    # Detailed results for best trial
    print(f"\n=== BEST SOLUTION ===")
    print(f"Parameters: {best_trial.params}")
    print(f"Combined objective value: {best_trial.value:.6f}")
    
    # Component breakdown
    base_accuracy = best_trial.user_attrs.get('base_accuracy', 0)
    normalized_acc_loss = best_trial.user_attrs.get('normalized_accuracy_loss', 0)
    total_penalty = best_trial.user_attrs.get('total_penalty', 0)
    
    
    print(f"\nObjective breakdown:")
    print(f"  Base accuracy: {base_accuracy:.4f}")
    print(f"  Accuracy component: {normalized_acc_loss:.6f}")
    print(f"  Penalty component: {total_penalty:.6f}")
    
    memory_blocks = best_trial.user_attrs.get('memory_blocks_list', [])
    if memory_blocks:
        print(f"\nMemory usage per fold: {memory_blocks}")
        print(f"Memory stats - Min: {min(memory_blocks):.2f}, Max: {max(memory_blocks):.2f}, "
              f"Avg: {np.mean(memory_blocks):.2f}, Std: {np.std(memory_blocks):.2f}")

    # Show top 5 trials for comparison
    print(f"\n=== TOP 5 TRIALS ===")
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
    for i, trial in enumerate(sorted_trials):
        acc = trial.user_attrs.get('base_accuracy', 0)
        penalty = trial.user_attrs.get('total_penalty', 0)
        print(f"  {i+1}. Objective: {trial.value:.6f}, Accuracy: {acc:.4f}, Penalty: {penalty:.4f}")

    # Extract best parameters
    best_params_A = {
        'n_estimators': best_trial.params['n_estimators_A'],
        'min_samples_leaf': best_trial.params['min_samples_leaf_A'],
        'min_samples_split': best_trial.params['min_samples_split_A'],
        'max_depth': best_trial.params['max_depth_A']
    }
    
    best_params_B = {
        'n_estimators': best_trial.params['n_estimators_B'],
        'min_samples_leaf': best_trial.params['min_samples_leaf_B'],
        'min_samples_split': best_trial.params['min_samples_split_B'],
        'max_depth': best_trial.params['max_depth_B']
    }

    # Train final models with best parameters
    skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
    skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

    loss_vect = []
    model_vect = []

    for (train_index_A, val_index_A), (train_index_B, val_index_B) in zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B)):
        x_train_A, x_val_A, y_train_A, y_val_A = X_A[train_index_A], X_A[val_index_A], y_A[train_index_A], y_A[val_index_A]
        x_train_B, x_val_B, y_train_B, y_val_B = X_B[train_index_B], X_B[val_index_B], y_B[train_index_B], y_B[val_index_B]

        model_A = RandomForestClassifier(**best_params_A, n_jobs=-1)
        model_B = RandomForestClassifier(**best_params_B, n_jobs=-1)

        model_A.fit(x_train_A, y_train_A)
        model_B.fit(x_train_B, y_train_B)

        model_A = dt_thresholds_float_to_int(model_A)
        model_B = dt_thresholds_float_to_int(model_B)

        score_A = model_A.score(x_val_A, y_val_A)
        score_B = model_B.score(x_val_B, y_val_B)

        try:
            stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
        except RuntimeError as e:
            continue
        
        if blocks > max_blocks:
            continue
        
        model_vect.append((model_A, model_B))
        loss_vect.append(-(score_A + score_B) / 2)

    if len(model_vect) == 0:
        raise RuntimeError('No feasible split found even with relaxed constraints')

    return model_vect[np.argmin(loss_vect)]


def train_multi_RF_Optuna_multi(X_A, y_A, X_B, y_B, x_val_A, y_val_A, x_val_B, y_val_B, features_A, features_B, n_trees, max_depth, max_blocks, encoding):
    num_cross_validation_splits = 3

    if n_trees == -1:
        n_trees = 7

    if max_depth == -1:
        max_depth = 10

    def objective(trial):
        # Suggest hyperparameters for both models
        params_A = {
            'n_estimators': trial.suggest_int('n_estimators_A', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_A', 5, 200, step=10),    #25, 251, step=25), #5, 21, step = 5),
            'min_samples_split': trial.suggest_int('min_samples_split_A', 10, 400, step=10),   #10, 511, step=50), #10, 51, step=5), 
            'max_depth': trial.suggest_int('max_depth_A', 2, max_depth),
            'random_state': 42
        }
        
        params_B = {
            'n_estimators': trial.suggest_int('n_estimators_B', 1, n_trees, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf_B', 5, 200, step=10),    #25, 251, step=25), #5, 21, step = 5),
            'min_samples_split': trial.suggest_int('min_samples_split_B', 10, 400, step=10),   #10, 511, step=50), #10, 51, step=5), 
            'max_depth': trial.suggest_int('max_depth_B', 2, max_depth),
            'random_state': 42
        }

        skf_A = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)
        skf_B = StratifiedKFold(n_splits=num_cross_validation_splits, shuffle=True, random_state=42)

        score_vect_A = []
        score_vect_B = []
        penalty_vect = []
        memory_blocks_list = []

        for (train_index_A, test_index_A), (train_index_B, test_index_B) in zip(skf_A.split(X_A, y_A), skf_B.split(X_B, y_B)):
            x_train_A, x_test_A, y_train_A, y_test_A = X_A[train_index_A], X_A[test_index_A], y_A[train_index_A], y_A[test_index_A]
            x_train_B, x_test_B, y_train_B, y_test_B = X_B[train_index_B], X_B[test_index_B], y_B[train_index_B], y_B[test_index_B]

            model_A = RandomForestClassifier(**params_A, n_jobs=-1)
            model_B = RandomForestClassifier(**params_B, n_jobs=-1)

            with sklearn.config_context(assume_finite=True):
                model_A.fit(x_train_A, y_train_A)
                model_B.fit(x_train_B, y_train_B)

            model_A = dt_thresholds_float_to_int(model_A)
            model_B = dt_thresholds_float_to_int(model_B)

            if encoding == 'joint':

                #_, blocks_before = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
                #acc_before = (model_A.score(x_test_A, y_test_A) + model_B.score(x_test_B, y_test_B))/2

                # Align the forests
                model_A, model_B = align_rf_thresholds(
                    model_A, model_B,
                    x_val_A, y_val_A,
                    x_val_B, y_val_B,
                    overlap_threshold=0.5)

            with sklearn.config_context(assume_finite=True):
                score_vect_A.append(model_A.score(x_test_A, y_test_A))
                score_vect_B.append(model_B.score(x_test_B, y_test_B))

            penalty_vect.append(0)

            try:
                stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
                memory_blocks_list.append(blocks)
            except RuntimeError as e:
                codeword_length = e.args[1]
                
                violation_ratio = (codeword_length - MAX_CODEWORD_LENGTH) / MAX_CODEWORD_LENGTH
                codeword_length_penalty = violation_ratio ** 2
                #print('Codeword length is {} and penalty: {}'.format(codeword_length, codeword_length_penalty))
                score_vect_A[-1] -= codeword_length_penalty
                score_vect_B[-1] -= codeword_length_penalty

                penalty_vect[-1] += codeword_length_penalty

                blocks = max_blocks
                memory_blocks_list.append(blocks)
                break

            if blocks > max_blocks:
                violation_ratio = (blocks - max_blocks) / max_blocks
                max_memory_penalty = violation_ratio ** 2

                if max_memory_penalty < 0.1:
                    max_memory_penalty = 0.1

                #print('Number of blocks is {} and penalty: {}'.format(blocks, max_memory_penalty))
                score_vect_A[-1] -= max_memory_penalty
                score_vect_B[-1] -= max_memory_penalty

                penalty_vect[-1] += max_memory_penalty
                break

            '''print('Blocks before {}'.format(blocks_before))
            print('Accuracy before {}'.format(acc_before))

            print('Blocks after {}'.format(blocks))
            print('Accuracy after {}'.format((model_A.score(x_test_A, y_test_A) + model_B.score(x_test_B, y_test_B))/2))'''

        # Store detailed memory information for analysis
        trial.set_user_attr('memory_blocks_list', memory_blocks_list)
        
        # Calculate base objectives
        avg_accuracy = (np.mean(score_vect_A) + np.mean(score_vect_B)) / 2
        
        # Store components for analysis
        trial.set_user_attr('avg_accuracy', avg_accuracy)
        trial.set_user_attr('max_penalty', np.max(penalty_vect))

        return avg_accuracy, np.max(memory_blocks_list)

    def early_stopping_callback(study, trial):

        feasible_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get('max_penalty', float('inf')) == 0.0]
        if len(feasible_trials) == 25:
            study.stop()
        
        # Get recent trials
        recent_window = 25
        recent_trials = study.trials[-recent_window:]
        completed_recent = [t for t in recent_trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get('max_penalty', float('inf')) == 0.0]
        
        if len(completed_recent) < recent_window // 2:
            return
        
        # Check if any new Pareto optimal solutions were found recently
        current_pareto = study.best_trials
        recent_pareto = [t for t in current_pareto if t.number >= len(study.trials) - recent_window]
        
        # Stop if no new Pareto optimal solutions in recent trials
        if len(recent_pareto) == 0:
            study.stop()

    # Create study with multi-objective optimization
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            warn_independent_sampling=True,
            constant_liar=True
        )
    )

    # Optimize the objectives
    study.optimize(
        objective, 
        n_trials=1000,
        callbacks=[early_stopping_callback],
        catch=(RuntimeError,)
    )

    # Get solutions with no penalties
    feasible_trials = [t for t in study.trials if t.user_attrs.get('max_penalty', float('inf')) == 0.0]

    if not feasible_trials:
        raise RuntimeError('No feasible solutions found')

    # Find the best accuracy among feasible solutions
    best_accuracy = max(t.user_attrs.get('avg_accuracy', 0) for t in feasible_trials)
    accuracy_threshold = 0.0025

    # Filter solutions within accuracy threshold
    close_to_best = [t for t in feasible_trials 
                    if (t.user_attrs.get('avg_accuracy', 0) >= best_accuracy - accuracy_threshold)]
    
    # Among those, select the one with minimal memory blocks
    best_trial = min(close_to_best, key=lambda t: np.max(t.user_attrs.get('memory_blocks_list', [float('inf')])))

    print(f"\n=== BEST SOLUTION ===")
    print(f"Parameters: {best_trial.params}")
    print(f"Accuracy: {best_trial.user_attrs.get('avg_accuracy', 0):.4f}")
    
    memory_blocks = best_trial.user_attrs.get('memory_blocks_list', [])
    print(f"Memory usage per fold: {memory_blocks}")

    # Extract best parameters
    best_params_A = {
        'n_estimators': best_trial.params['n_estimators_A'],
        'min_samples_leaf': best_trial.params['min_samples_leaf_A'],
        'min_samples_split': best_trial.params['min_samples_split_A'],
        'max_depth': best_trial.params['max_depth_A'],
        'random_state': 42
    }
    
    best_params_B = {
        'n_estimators': best_trial.params['n_estimators_B'],
        'min_samples_leaf': best_trial.params['min_samples_leaf_B'],
        'min_samples_split': best_trial.params['min_samples_split_B'],
        'max_depth': best_trial.params['max_depth_B'],
        'random_state': 42
    }

    # Train final models with best parameters
    max_retries = 5
    for seed in range(42, 42 + max_retries):
        best_params_A['random_state'] = seed
        best_params_B['random_state'] = seed
        
        model_A = RandomForestClassifier(**best_params_A, n_jobs=-1)
        model_B = RandomForestClassifier(**best_params_B, n_jobs=-1)
        
        with sklearn.config_context(assume_finite=True):
            model_A.fit(X_A, y_A)
            model_B.fit(X_B, y_B)
        
        model_A = dt_thresholds_float_to_int(model_A)
        model_B = dt_thresholds_float_to_int(model_B)
        
        if encoding == 'joint':
        # Align the forests
            model_A, model_B = align_rf_thresholds(
                model_A, model_B,
                x_val_A, y_val_A,
                x_val_B, y_val_B,
                overlap_threshold=0.5)
        
        try:
            stages, blocks = multi_model_memory_evaluation(model_A, model_B, features_A, features_B, encoding)
            if blocks <= max_blocks:
                return model_A, model_B, stages, blocks
        except RuntimeError:
            continue

    raise RuntimeError('Could not produce feasible final model after retries')


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


def training_and_feature_selection(df_app, df_ddos, required_num_features, num_trees_app, num_trees_ddos):

    # split the data into training, validation (used for feature selection) and testing datasets
    X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(df_app.drop(columns=['Label']), df_app['Label'], test_size=0.2, random_state=42)
    X_train_app, X_val_app, y_train_app, y_val_app = train_test_split(X_train_app, y_train_app, test_size=0.25, random_state=42)

    X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(df_ddos.drop(columns=['Label']), df_ddos['Label'], test_size=0.2, random_state=42)
    X_train_ddos, X_val_ddos, y_train_ddos, y_val_ddos = train_test_split(X_train_ddos, y_train_ddos, test_size=0.25, random_state=42)

    current_num_features = len(list(X_train_ddos.columns))
    with open(INTERMEDIATE + 'training_and_feature_selection_results.txt', 'w') as log_file:
        log_file.write('Number of features in the dataset: {}\n'.format(current_num_features))
        log_file.write('Required number of features: {}\n'.format(required_num_features))

    while current_num_features > required_num_features:
        
        clf_app = train_classifier_RF(X_train_app, y_train_app, num_trees_app)
        clf_ddos = train_classifier_RF(X_train_ddos, y_train_ddos, num_trees_ddos)

        importances_app = permutation_importance(clf_app, X_val_app, y_val_app, scoring = 'accuracy', n_repeats=10, random_state=42, n_jobs=-1)
        importances_ddos = permutation_importance(clf_ddos, X_val_ddos, y_val_ddos, scoring = 'accuracy', n_repeats=10, random_state=42, n_jobs=-1)

        combined_importance = np.add(importances_app.importances_mean, importances_ddos.importances_mean)

        lowest_importance_ind = combined_importance.argmin()
        feature_name = X_val_app.columns[lowest_importance_ind]

        # remove the feature that contributed less to the accuracy of both models
        X_train_app.drop(columns=[feature_name], inplace = True)
        X_test_app.drop(columns=[feature_name], inplace = True)
        X_val_app.drop(columns=[feature_name], inplace = True)

        X_train_ddos.drop(columns=[feature_name], inplace = True)
        X_test_ddos.drop(columns=[feature_name], inplace = True)
        X_val_ddos.drop(columns=[feature_name], inplace = True)

        current_num_features = len(list(X_train_ddos.columns))

    selected_features = list(X_train_ddos.columns)
    with open(INTERMEDIATE + 'training_and_feature_selection_results.txt', 'a') as log_file:
        log_file.write('Selected features are: {}\n'.format(selected_features))

    clf_app = train_classifier_RF(X_train_app, y_train_app, num_trees_app)
    clf_ddos = train_classifier_RF(X_train_ddos, y_train_ddos, num_trees_ddos)

    # predict the label for the test datasets
    y_pred_test_app = clf_app.predict(X_test_app)
    y_pred_test_ddos = clf_ddos.predict(X_test_ddos)

    # calculate the accuracy metrics
    accuracy_metrics(y_test_app, y_pred_test_app, 'app')
    accuracy_metrics(y_test_ddos, y_pred_test_ddos, 'ddos')

    return clf_app, clf_ddos, selected_features