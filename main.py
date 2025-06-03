from dataset import load_dataset
from train_model import training_and_feature_selection
from build_p4_script import *


if __name__ == '__main__':

    # we classify traffic of 3 applications and detect if traffic is an attack or is benign
    num_classes_app = 3
    num_classes_ddos = 2

    # traffic flows classifier is trained with 3 trees, while the DDOS detector is trained with 1 tree
    num_trees_app = 3 # 1
    num_trees_ddos = 1 # 3

    # number of features we want to use in both models
    num_features = 3

    datasets_path = "resources/"
    threshold = (2 ** 19) - 2
    df_app = load_dataset(datasets_path, 'apps_flow_features.csv', threshold)
    df_ddos = load_dataset(datasets_path, 'Wednesday-workingHours.pcap_ISCX.csv', threshold)

    clf_app, clf_ddos, selected_features = training_and_feature_selection(df_app, df_ddos, num_features, num_trees_app, num_trees_ddos)

    clf_app = dt_thresholds_float_to_int(clf_app)
    clf_ddos = dt_thresholds_float_to_int(clf_ddos)

    # export the trees into textual format 
    trees_app = get_tree_textual_representation(clf_app, selected_features)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features)

    # extract node features (leaf or internal)
    tree_nodes = {}

    for tree_app in trees_app:
        tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes)

    for tree_ddos in trees_ddos:
        tree_nodes[tree_ddos + offset] = get_nodes(trees_ddos[tree_ddos])

    feature_splits = get_feature_splits(tree_nodes)
    feature_intervals = get_feature_intervals(feature_splits)
    feature_intervals_to_csv(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    codeword_length = len(next(iter(codewords[0].items()))[0])
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset)

    generate_P4_code(num_classes_app, num_classes_ddos, clf_app, clf_ddos, codeword_length, feature_intervals)
