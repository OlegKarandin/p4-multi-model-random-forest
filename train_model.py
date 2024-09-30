from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import sklearn.metrics as mt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import hyperopt

import pandas as pd
import numpy as np

INTERMEDIATE = "intermediate/"

def num_stages_feature_based(num_models, num_trees, num_features, sharing_flag):

    if sharing_flag:
        num_feature_mapping_stages = num_features

    else:
        num_feature_mapping_stages = num_features * num_models

    return (num_models * num_trees + num_feature_mapping_stages)

def train_classifier_RF(X_train, y_train, n_trees):

    space4xgb = {
        #'n_estimators': hp.choice('n_estimators', np.arange(1, 5, 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 10, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, 1)),
        'max_features': hp.choice('max_features', ['log2', 'sqrt']),
        'max_depth': hp.choice('max_depth', np.arange(5, 20, 1))}

    def hyperopt_train_test(params):
        params['n_estimators'] = n_trees
        model = RandomForestClassifier(**params, n_jobs = -1)
        return cross_val_score(model, X_train, y_train, cv = 3, n_jobs = -1)

    def f(params):
        score_vect = hyperopt_train_test(params)
        return {'status': STATUS_OK,
                'loss': -np.mean(score_vect),
                'loss_variance': np.var(score_vect, ddof=1)}

    trials = Trials()
    best_params = fmin(f, space4xgb, algo=tpe.suggest, max_evals=25, early_stop_fn=no_progress_loss(5), trials=trials)
    best_params = hyperopt.space_eval(space4xgb, best_params)

    best_params['n_estimators'] = n_trees
    print(best_params)

    clf = RandomForestClassifier(**best_params, n_jobs = -1)
    clf.fit(X_train, y_train)

    return clf

def accuracy_metrics(y_test, y_pred, task):

    y_true = y_test

    if task == 'app':
        lab = [0, 1, 2]
        l_names = ['Realtime', 'Non-realtime', 'Websites']
        av = 'weighted'

    elif task == 'ddos':
        lab = [1]
        l_names = ['Benign', 'Attack']
        av = None

    accuracy = mt.accuracy_score(y_true, y_pred)
    precision = mt.precision_score(y_true, y_pred, labels=lab, average=av) #F: average=None gives per-class results
    recall = mt.recall_score(y_true, y_pred, labels=lab, average=av)
    f1score = mt.f1_score(y_true, y_pred, labels=lab, average=av)

    with open(INTERMEDIATE + 'feature_selection_results.txt', 'a') as log_file:
        log_file.write('{}\n'.format(task))
        log_file.write('Accuracy: {}\n'.format(accuracy))
        log_file.write('Precision: {}\n'.format(precision))
        log_file.write('Recall: {}\n'.format(recall))
        log_file.write('F1score: {}\n\n'.format(f1score))


    if task == 'app':
        return (accuracy, precision, recall, f1score)

    elif task == 'ddos':
        return (accuracy, precision[0], recall[0], f1score[0])

def joint_feature_selection(df_app, df_ddos, num_final_features, num_trees_app, num_trees_ddos, num_cross_validations):

    num_features_to_eliminate = len(list(df_app.columns)) - num_final_features
    result_columns = ['Features', 'M/A stages', 'Accuracy', 'Precision', 'Recall', 'F1-score']

    results_app_array = np.zeros(shape=(num_cross_validations, num_features_to_eliminate, len(result_columns)))
    results_ddos_array = np.zeros(shape=(num_cross_validations, num_features_to_eliminate, len(result_columns)))

    for i in range(num_cross_validations):
        print("num_cross_validations: " + str(i))
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(df_app.drop(columns=['Label']), df_app['Label'], test_size=0.2, random_state=i)
        X_train_app, X_val_app, y_train_app, y_val_app = train_test_split(X_train_app, y_train_app, test_size=0.25, random_state=i)

        X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(df_ddos.drop(columns=['Label']), df_ddos['Label'], test_size=0.2, random_state=i)
        X_train_ddos, X_val_ddos, y_train_ddos, y_val_ddos = train_test_split(X_train_ddos, y_train_ddos, test_size=0.25, random_state=i)

        counter = 0
        num_features = len(list(X_train_ddos.columns))

        while num_features > num_final_features:
            print("Num features: " + str(num_features) + ", num final features: " + str(num_final_features))
            selected_features = list(X_train_ddos.columns)
            print("Selected features are:")
            print(selected_features)
            clf_app = train_classifier_RF(X_train_app, y_train_app, num_trees_app)
            acc_tuple = accuracy_metrics(y_test_app, clf_app.predict(X_test_app), 'app')
            results_app_array[i, counter, :] = [num_features, num_stages_feature_based(2, num_trees_app, num_features, True), acc_tuple[0], acc_tuple[1], acc_tuple[2], acc_tuple[3]]

            clf_ddos = train_classifier_RF(X_train_ddos, y_train_ddos, num_trees_ddos)
            acc_tuple = accuracy_metrics(y_test_ddos, clf_ddos.predict(X_test_ddos), 'ddos')
            results_ddos_array[i, counter, :] = [num_features, num_stages_feature_based(2, num_trees_app, num_features, True), acc_tuple[0], acc_tuple[1], acc_tuple[2], acc_tuple[3]]

            importances_app = permutation_importance(clf_app, X_val_app, y_val_app, scoring = 'accuracy', n_repeats=10, random_state=42, n_jobs=-1)
            importances_ddos = permutation_importance(clf_ddos, X_val_ddos, y_val_ddos, scoring = 'accuracy', n_repeats=10, random_state=42, n_jobs=-1)

            combined_importance = np.add(importances_app.importances_mean, importances_ddos.importances_mean)

            lowest_importance_ind = combined_importance.argmin()
            feature_name = X_val_app.columns[lowest_importance_ind]

            X_train_app.drop(columns=[feature_name], inplace = True)
            X_test_app.drop(columns=[feature_name], inplace = True)
            X_val_app.drop(columns=[feature_name], inplace = True)

            X_train_ddos.drop(columns=[feature_name], inplace = True)
            X_test_ddos.drop(columns=[feature_name], inplace = True)
            X_val_ddos.drop(columns=[feature_name], inplace = True)

            num_features = len(list(X_train_ddos.columns))
            counter += 1

    results_app_df = pd.DataFrame(data = results_app_array.mean(axis=0), columns=result_columns)
    results_ddos_df = pd.DataFrame(data = results_ddos_array.mean(axis=0), columns=result_columns)

    return results_app_df, results_ddos_df, selected_features