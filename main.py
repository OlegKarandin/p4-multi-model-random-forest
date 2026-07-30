from dataset import load_dataset, read_app_dataset, read_DDOS_dataset
from train_model import training_and_feature_selection
from build_p4_script import *
from feature_selection import compare_feature_selection_approaches, compare_feature_selection_approaches_parallel

from analysis import analyze_multi_objective_results

import pandas as pd
import numpy as np

from pathlib import Path

def implement_tree_models_in_P4():
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

    feature_thresholds = get_feature_thresholds(tree_nodes)
    feature_intervals = get_feature_intervals_from_thresholds(feature_thresholds)
    feature_intervals_to_csv(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    codeword_length = len(next(iter(codewords[0].items()))[0])
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset)

    generate_P4_code(num_classes_app, num_classes_ddos, clf_app, clf_ddos, codeword_length, feature_intervals)


def remove_correlated_features_both_datasets(df_app, df_ddos, threshold=0.95):
    """
    Remove features that are highly correlated in BOTH datasets.
    
    Parameters:
    -----------
    df_app : pd.DataFrame
        Application dataset with features and 'Label' column
    df_ddos : pd.DataFrame
        DDoS dataset with features and 'Label' column
    threshold : float
        Correlation threshold (default 0.95)
        
    Returns:
    --------
    X_app : np.array
        App feature matrix with correlated features removed
    X_ddos : np.array
        DDoS feature matrix with correlated features removed
    feature_names : list
        Names of remaining features
    """
    
    # Get feature names (excluding Label)
    feature_names = [col for col in df_app.columns if col != 'Label']
    
    # Extract feature matrices
    X_app_full = df_app.drop(columns=["Label"]).to_numpy()
    X_ddos_full = df_ddos.drop(columns=["Label"]).to_numpy()
    
    # Calculate correlation matrices
    corr_app = np.corrcoef(X_app_full.T)
    corr_ddos = np.corrcoef(X_ddos_full.T)
    
    # Find features to remove
    n_features = len(feature_names)
    features_to_remove = set()
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Check if correlation is high in BOTH datasets
            if (abs(corr_app[i, j]) > threshold and 
                abs(corr_ddos[i, j]) > threshold):
                
                # Remove the feature with lower average absolute correlation
                # with all other features (less informative overall)
                avg_corr_i_app = np.mean(np.abs(corr_app[i, :]))
                avg_corr_j_app = np.mean(np.abs(corr_app[j, :]))
                avg_corr_i_ddos = np.mean(np.abs(corr_ddos[i, :]))
                avg_corr_j_ddos = np.mean(np.abs(corr_ddos[j, :]))
                
                avg_corr_i = (avg_corr_i_app + avg_corr_i_ddos) / 2
                avg_corr_j = (avg_corr_j_app + avg_corr_j_ddos) / 2
                
                # Remove the feature that's more correlated with others on average
                if avg_corr_i > avg_corr_j:
                    features_to_remove.add(i)
                else:
                    features_to_remove.add(j)
    
    # Create mask for features to keep
    features_to_keep = [i for i in range(n_features) if i not in features_to_remove]
    
    # Filter datasets
    X_app = X_app_full[:, features_to_keep]
    X_ddos = X_ddos_full[:, features_to_keep]
    
    # Get remaining feature names
    remaining_features = [feature_names[i] for i in features_to_keep]
    removed_features = [feature_names[i] for i in features_to_remove]
    
    print(f"Original number of features: {n_features}")
    print(f"Features removed: {len(removed_features)}")
    print(f"Features remaining: {len(remaining_features)}")
    
    if removed_features:
        print(f"\nRemoved features: {removed_features[:10]}..." 
              if len(removed_features) > 10 else f"\nRemoved features: {removed_features}")
    
    # Print correlation analysis
    print(f"\nCorrelation analysis (threshold={threshold}):")
    for i in features_to_remove:
        # Find which features this one was correlated with
        correlated_with = []
        for j in range(n_features):
            if i != j and abs(corr_app[i, j]) > threshold and abs(corr_ddos[i, j]) > threshold:
                correlated_with.append((feature_names[j], 
                                       f"app: {corr_app[i,j]:.3f}, ddos: {corr_ddos[i,j]:.3f}"))
        if correlated_with and len(features_to_remove) <= 10:  # Only show details for small number
            print(f"  '{feature_names[i]}' correlated with: {correlated_with[:3]}")
    
    return X_app, X_ddos, remaining_features


def compare_independent_joint_mapping(M_values, eps, n_Cs, n_gammas, selection_threshold, n_splits, parallel=True, max_workers=None):
    """
    Run feature selection comparison experiments.

    Parameters
    ----------
    M_values : list
        List of max_blocks values to test
    eps, n_Cs, n_gammas, selection_threshold, n_splits : various
        Experiment parameters
    parallel : bool
        If True, use parallel processing across splits (default: True)
    max_workers : int, optional
        Number of parallel workers (only used if parallel=True)
    """
    threshold = INFINITE

    selected_features = [
    'Fwd.Packet.Length.Max', 'Fwd.Packet.Length.Min', 'Fwd.Packet.Length.Mean',
    'Bwd.Packet.Length.Max', 'Bwd.Packet.Length.Min', 'Bwd.Packet.Length.Mean',
    'Flow.IAT.Mean', 'Flow.IAT.Max', 'Flow.IAT.Min',
    'Fwd.IAT.Mean',  'Fwd.IAT.Max',  'Fwd.IAT.Min',
    'Bwd.IAT.Mean',  'Bwd.IAT.Max',  'Bwd.IAT.Min',
    'Min.Packet.Length', 'Max.Packet.Length', 'Packet.Length.Mean']

    df_app = read_app_dataset(selected_features, threshold)
    df_ddos = read_DDOS_dataset(selected_features, threshold)

    X_app, X_ddos, selected_features = remove_correlated_features_both_datasets(df_app, df_ddos)

    y_app = df_app.Label.to_numpy()
    y_ddos = df_ddos.Label.to_numpy()

    print("Starting Multi-Task vs Single-Task Feature Selection Comparison")
    print("=" * 70)

    n_features = X_app.shape[1]
    print(f"Total number of features: {n_features}")
    print(f"Lasso path: eps={eps}, n_Cs={n_Cs}")
    print(f"MTL path: n_gammas={n_gammas}, selection_threshold={selection_threshold}")
    print(f"Parallel processing: {parallel}" + (f" (max_workers={max_workers})" if max_workers else ""))

    for t in [-1]: #np.arange(1, 8, 2):
        for d in [-1]: #np.arange(2, 7, 2):
            for max_blocks in M_values:

                print('Max {} blocks'.format(max_blocks))

                # Run the comparison using regularization paths
                if parallel:
                    results_df = compare_feature_selection_approaches_parallel(
                        X_app, X_ddos, y_app, y_ddos, t, d, max_blocks,
                        feature_names=selected_features,
                        n_splits=n_splits,
                        random_state=42,
                        max_workers=max_workers
                    )
                else:
                    results_df = compare_feature_selection_approaches(
                        X_app, X_ddos, y_app, y_ddos, t, d, max_blocks,
                        feature_names=selected_features,
                        n_splits=n_splits,
                        random_state=42
                    )

                name = os.path.join('results', 'feature_selection_comparison_results_by_k_{}_{}_{}'.format(
                    t, d, max_blocks
                ))
                file_exists = os.path.exists(name + '.csv')
                results_df.to_csv(name + '.csv', mode='a', index=False, header=not file_exists)

                #summary_stats = plot_comparison_results_by_k(results_df, name + '.png', save_plots=True)


def load_and_combine_data(folder_path, M_values):
    """
    Load dataframes for different M values and combine them
    """
    all_data = []

    for M in M_values:
        filename = f'feature_selection_comparison_results_by_k_{-1}_{-1}_{M}.csv'
        filepath = Path(folder_path) / filename

        try:
            df = pd.read_csv(filepath)
            df['M'] = M  # Add M column
            all_data.append(df)
            print(f"Loaded data for M={M}: {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: File not found for M={M}: {filename}")

    if not all_data:
        raise ValueError("No data files found!")

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


if __name__ == '__main__':

    new_results = False
    #M = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    M = [25, 40, 50, 60, 75, 90, 100]

    # Regularization path parameters
    eps = 1e-5
    n_Cs = 100
    n_gammas = 100
    selection_threshold = 1e-2
    n_splits = 15

    # Parallelization settings
    parallel = True      # Set to False to use sequential processing
    max_workers = None   # None = auto (cpu_count - 1), or set to specific number

    if new_results:
        compare_independent_joint_mapping(
            M_values=M,
            eps=eps,
            n_Cs=n_Cs,
            n_gammas=n_gammas,
            selection_threshold=selection_threshold,
            n_splits=n_splits,
            parallel=parallel,
            max_workers=max_workers
        )

    else:
        df = load_and_combine_data(folder_path='results', M_values=M)

        #create_comparison_plots(df)

        analysis = analyze_multi_objective_results(df, list(range(17, 0, -2)))
        print(f"\nMulti approach covers {analysis['all_k']['coverage_ratio']['multi_covers_single']:.1%} of Single approach solutions")