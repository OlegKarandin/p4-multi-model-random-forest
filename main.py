import statistics
import sklearn.ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from train_model import joint_feature_selection
from build_p4_script import *


if __name__ == '__main__':
    # In this case study, both the traffic flows classifier and the DDOS detector will be trained with only 1 tree
    """
    num_class_app = 3
    num_class_ddos = 2
    num_trees_app = 1
    num_trees_ddos = 1
    num_final_features = 4 - 1

    df_app = load_dataset('apps_flow_features.csv')
    df_ddos = load_dataset('Wednesday-workingHours.pcap_ISCX.csv')

    _, _, selected_features = joint_feature_selection(df_app, df_ddos, num_final_features, num_trees_app, num_trees_ddos, 1)

    results_app = []
    results_ddos = []

    for i in range(100):
        X_app = df_app[selected_features]
        y_app = df_app.Label
        X_ddos = df_ddos[selected_features]
        y_ddos = df_ddos.Label

        # Split dataset into training set and test set
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app,
                                                                            test_size=0.3)  # 70% training and 30% test
        X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(X_ddos, y_ddos,
                                                                                test_size=0.3)  # 70% training and 30% test

        clf_app = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_app, criterion='entropy', max_depth=5)
        clf_ddos = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_ddos, criterion='entropy', max_depth=5)

        # Train Random Forest Classifer
        clf_app.fit(X_train_app, y_train_app)
        clf_ddos.fit(X_train_ddos, y_train_ddos)

        # Predict the response for test dataset
        y_pred_test_app = clf_app.predict(X_test_app)
        y_pred_test_ddos = clf_ddos.predict(X_test_ddos)

        results_app.append(accuracy_score(y_test_app, y_pred_test_app))
        results_ddos.append(accuracy_score(y_test_ddos, y_pred_test_ddos))

    mean_app = statistics.mean(results_app)
    std_dev_app = statistics.stdev(results_app)
    mean_ddos = statistics.mean(results_ddos)
    std_dev_ddos = statistics.stdev(results_ddos)
    print("Mean Accuracy Traffic Flows Classification:", mean_app)
    print("Standard Deviation Accuracy Traffic Flows Classification:", std_dev_app)
    print("Mean Accuracy DDOS Detection Classification:", mean_ddos)
    print("Standard Deviation DDOS Detection Traffic Flows Classification:", std_dev_ddos)

    clf_app = dt_thresholds_float_to_int(clf_app)
    clf_ddos = dt_thresholds_float_to_int(clf_ddos)

    trees_app = get_tree_textual_representation(clf_app, selected_features)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features)

    tree_nodes = {}
    tree_nodes_app = {}
    tree_nodes_ddos = {}

    for tree_app in trees_app:
        tree_nodes_app[tree_app] = get_nodes(trees_app[tree_app])
        tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes_app)

    for tree_ddos in trees_ddos:
        tree_nodes_ddos[tree_ddos] = get_nodes(trees_ddos[tree_ddos])
        tree_nodes[tree_ddos + offset] = get_nodes(trees_ddos[tree_ddos])

    feature_splits = get_feature_splits(tree_nodes)
    feature_intervals = get_feature_intervals(feature_splits)
    feature_intervals_to_csv(feature_intervals)
    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    #No more the same as notebook
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    codeword_length = len(next(iter(codewords[0].items()))[0])
    #No more the same as notebook
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset)

    generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos, codeword_length, feature_intervals)
    """



    """
    #In this case study, the traffic flows classifier will be trained with 1 tree
    #while the DDOS detector will be trained with 3 trees
    num_class_app = 3
    num_class_ddos = 2
    num_trees_app = 1
    num_trees_ddos = 3
    num_final_features = 4 - 1

    df_app = load_dataset('apps_flow_features.csv')
    df_ddos = load_dataset('Wednesday-workingHours.pcap_ISCX.csv')

    _, _, selected_features = joint_feature_selection(df_app, df_ddos, num_final_features, num_trees_ddos, 1)

    results_app = []
    results_ddos = []

    for i in range(100):
        X_app = df_app[selected_features]
        y_app = df_app.Label
        X_ddos = df_ddos[selected_features]
        y_ddos = df_ddos.Label

        # Split dataset into training set and test set
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app,
                                                                            test_size=0.3)  # 70% training and 30% test
        X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(X_ddos, y_ddos,
                                                                                test_size=0.3)  # 70% training and 30% test

        clf_app = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_app, criterion='entropy', max_depth=5)
        clf_ddos = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_ddos, criterion='entropy', max_depth=5)

        # Train Random Forest Classifer
        clf_app.fit(X_train_app, y_train_app)
        clf_ddos.fit(X_train_ddos, y_train_ddos)

        # Predict the response for test dataset
        y_pred_test_app = clf_app.predict(X_test_app)
        y_pred_test_ddos = clf_ddos.predict(X_test_ddos)

        results_app.append(accuracy_score(y_test_app, y_pred_test_app))
        results_ddos.append(accuracy_score(y_test_ddos, y_pred_test_ddos))

    mean_app = statistics.mean(results_app)
    std_dev_app = statistics.stdev(results_app)
    mean_ddos = statistics.mean(results_ddos)
    std_dev_ddos = statistics.stdev(results_ddos)
    print("Mean Accuracy Traffic Flows Classification:", mean_app)
    print("Standard Deviation Accuracy Traffic Flows Classification:", std_dev_app)
    print("Mean Accuracy DDOS Detection Classification:", mean_ddos)
    print("Standard Deviation DDOS Detection Traffic Flows Classification:", std_dev_ddos)

    clf_app = dt_thresholds_float_to_int(clf_app)
    clf_ddos = dt_thresholds_float_to_int(clf_ddos)

    trees_app = get_tree_textual_representation(clf_app, selected_features)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features)

    tree_nodes = {}
    tree_nodes_app = {}
    tree_nodes_ddos = {}

    for tree_app in trees_app:
        tree_nodes_app[tree_app] = get_nodes(trees_app[tree_app])
        tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes_app)

    for tree_ddos in trees_ddos:
        tree_nodes_ddos[tree_ddos] = get_nodes(trees_ddos[tree_ddos])
        tree_nodes[tree_ddos + offset] = get_nodes(trees_ddos[tree_ddos])

    feature_splits = get_feature_splits(tree_nodes)
    feature_intervals = get_feature_intervals(feature_splits)
    feature_intervals_to_csv(feature_intervals)
    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    #No more the same as notebook
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    codeword_length = len(next(iter(codewords[0].items()))[0])
    #No more the same as notebook
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset)

    generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos, codeword_length, feature_intervals)
    """


    #In this case study, the traffic flows classifier will be trained with 3 trees
    #while the DDOS detector will be trained with 1 tree
    num_class_app = 3
    num_class_ddos = 2
    num_trees_app = 3
    num_trees_ddos = 1
    num_final_features = 4 - 1

    df_app = load_dataset('apps_flow_features.csv')
    df_ddos = load_dataset('Wednesday-workingHours.pcap_ISCX.csv')

    _, _, selected_features = joint_feature_selection(df_app, df_ddos, num_final_features, num_trees_app, num_trees_ddos, 1)

    results_app = []
    results_ddos = []

    for i in range(100):
        X_app = df_app[selected_features]
        y_app = df_app.Label
        X_ddos = df_ddos[selected_features]
        y_ddos = df_ddos.Label

        # Split dataset into training set and test set
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app,
                                                                            test_size=0.3)  # 70% training and 30% test
        X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(X_ddos, y_ddos,
                                                                                test_size=0.3)  # 70% training and 30% test

        clf_app = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_app, criterion='entropy', max_depth=5)
        clf_ddos = sklearn.ensemble.RandomForestClassifier(n_estimators=num_trees_ddos, criterion='entropy', max_depth=5)

        # Train Random Forest Classifer
        clf_app.fit(X_train_app, y_train_app)
        clf_ddos.fit(X_train_ddos, y_train_ddos)

        # Predict the response for test dataset
        y_pred_test_app = clf_app.predict(X_test_app)
        y_pred_test_ddos = clf_ddos.predict(X_test_ddos)

        results_app.append(accuracy_score(y_test_app, y_pred_test_app))
        results_ddos.append(accuracy_score(y_test_ddos, y_pred_test_ddos))

    mean_app = statistics.mean(results_app)
    std_dev_app = statistics.stdev(results_app)
    mean_ddos = statistics.mean(results_ddos)
    std_dev_ddos = statistics.stdev(results_ddos)
    print("Mean Accuracy Traffic Flows Classification:", mean_app)
    print("Standard Deviation Accuracy Traffic Flows Classification:", std_dev_app)
    print("Mean Accuracy DDOS Detection Classification:", mean_ddos)
    print("Standard Deviation DDOS Detection Traffic Flows Classification:", std_dev_ddos)

    clf_app = dt_thresholds_float_to_int(clf_app)
    clf_ddos = dt_thresholds_float_to_int(clf_ddos)

    trees_app = get_tree_textual_representation(clf_app, selected_features)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features)

    tree_nodes = {}
    tree_nodes_app = {}
    tree_nodes_ddos = {}

    for tree_app in trees_app:
        tree_nodes_app[tree_app] = get_nodes(trees_app[tree_app])
        tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes_app)

    for tree_ddos in trees_ddos:
        tree_nodes_ddos[tree_ddos] = get_nodes(trees_ddos[tree_ddos])
        tree_nodes[tree_ddos + offset] = get_nodes(trees_ddos[tree_ddos])

    feature_splits = get_feature_splits(tree_nodes)
    feature_intervals = get_feature_intervals(feature_splits)
    feature_intervals_to_csv(feature_intervals)
    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    #No more the same as notebook
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    codeword_length = len(next(iter(codewords[0].items()))[0])
    #No more the same as notebook
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset)

    generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos, codeword_length, feature_intervals)
