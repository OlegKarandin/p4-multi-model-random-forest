from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import sklearn.metrics as mt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import hyperopt

import numpy as np

INTERMEDIATE = "intermediate/"


def train_classifier_RF(X_train, y_train, n_trees):

    space4xgb = {
        #'n_estimators': hp.choice('n_estimators', np.arange(1, 5, 1)),
        'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 10, 1)),
        #'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'min_samples_split': hp.choice('min_samples_split', np.arange(2, 10, 1)),
        #'max_depth': hp.choice('max_depth', np.arange(5, 20, 1)),
        'max_features': hp.choice('max_features', ['log2', 'sqrt'])}

    def hyperopt_train_test(params):
        params['n_estimators'] = n_trees
        params['max_depth'] = 5
        params['criterion'] = 'entropy'
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
    best_params['max_depth'] = 5
    best_params['criterion'] = 'entropy'
    print(best_params)

    clf = RandomForestClassifier(**best_params, n_jobs = -1)
    clf.fit(X_train, y_train)

    return clf


def accuracy_metrics(y_true, y_pred, task):

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

    with open(INTERMEDIATE + 'training_and_feature_selection_results.txt', 'a') as log_file:
        log_file.write('{}\n'.format(task))
        log_file.write('Accuracy: {}\n'.format(accuracy))
        log_file.write('Precision: {}\n'.format(precision))
        log_file.write('Recall: {}\n'.format(recall))
        log_file.write('F1score: {}\n\n'.format(f1score))


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