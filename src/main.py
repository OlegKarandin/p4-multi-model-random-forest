from src.training.dataset import read_app_dataset, read_DDOS_dataset
from src.p4gen.build_p4_script import *
from src.training.feature_selection import compare_feature_selection_approaches_parallel
from src.training.config import TrainConfig

from src.reporting.analysis import analyze_multi_objective_results

import argparse
import os
import pandas as pd
import numpy as np

from pathlib import Path


# Spec A.2's arm grid. Two anchors bracket the frontier: `joint-off` is a
# genuine SKIP of the align_rf_thresholds call -- not delta = 0 -- so the arm is
# provably prediction-identical to the unaligned models and doubles as the
# requested ablation; `joint-dinf` accepts every alignment unconditionally and
# bounds the maximum achievable sharing.
PRIMARY_ARMS = [
    ('independent', TrainConfig()),
    ('joint', TrainConfig(alignment_enabled=False)),
    ('joint', TrainConfig(delta_align=0.0)),
]

# The swept variable. delta = 0.01 is deliberately excluded: at val_align ~3000
# and DDoS error ~0.04, one flipped sample is 0.83% relative error, so 1%
# permits at most one flip and is operationally identical to 0. The grid reaches
# 20% and inf on purpose -- today's effective behaviour sits around 10-20%
# relative on DDoS, so the sweep must bracket it, and the frontier should show
# saturation rather than only its steep part.
SENSITIVITY_ARMS = [
    ('joint', TrainConfig(delta_align=0.02)),
    ('joint', TrainConfig(delta_align=0.05)),
    ('joint', TrainConfig(delta_align=0.10)),
    ('joint', TrainConfig(delta_align=0.20)),
    ('joint', TrainConfig(delta_align=None)),
]


def select_arms(which):
    if which == 'primary':
        return list(PRIMARY_ARMS)
    if which == 'sensitivity':
        return list(SENSITIVITY_ARMS)
    if which == 'all':
        return PRIMARY_ARMS + SENSITIVITY_ARMS
    raise ValueError("arms must be 'primary', 'sensitivity' or 'all', got {!r}".format(which))


def arm_result_path(arm, cfg, max_blocks):
    """One file per (arm, M): self-describing, globbable, and resumable.

    Replaces feature_selection_comparison_results_by_k_{t}_{d}_{M}.csv, whose
    -1_-1 sentinel recorded neither the effective n_trees nor max_depth (F10i).
    """
    encoding = 'joint' if arm == 'joint' else 'disjoint'
    return os.path.join('results', 'rf_t{}_d{}_M{}_{}.csv'.format(
        cfg.n_trees, cfg.max_depth, max_blocks, cfg.arm_slug(encoding)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Feature selection experiment runner")
    parser.add_argument(
        "--mode", choices=["compute", "plot"], default="plot",
        help="'compute' runs new feature-selection experiments via "
             "compare_independent_joint_mapping (expensive, real Optuna searches); "
             "'plot' loads and analyzes already-computed results (default, matches "
             "today's checked-in new_results=False behavior)")
    parser.add_argument(
        "--arms", choices=["primary", "sensitivity", "all"], default="primary",
        help="which arm grid to run in compute mode. 'primary' is the three "
             "arms the headline comparison needs (independent, joint@off, "
             "joint@delta=0); 'sensitivity' is the five swept tolerances")
    parser.add_argument(
        "--redo", action="store_true",
        help="recompute (arm, M) cells whose result file already exists. The "
             "default skips them, so re-running the same command resumes a "
             "partially finished campaign instead of redoing it")
    parser.add_argument(
        "--M", dest="M", type=_parse_M_grid, default=None,
        help="comma-separated TCAM block budgets to sweep in compute mode, "
             "e.g. '--M 25' for a single pilot cell or '--M 25,40,60' for a "
             "partial sweep. Defaults to today's full grid "
             "[25,40,50,60,75,90,100] when omitted, so a pilot run is a "
             "command-line flag rather than an edit to this file")
    parser.add_argument(
        "--n-splits", dest="n_splits", type=int, default=None,
        help="number of CV splits per (arm, M) cell in compute mode. "
             "Defaults to today's value (15) when omitted")
    return parser.parse_args(argv)


def _parse_M_grid(value):
    """--M's argparse type: comma-separated TCAM block budgets, e.g. '25' or
    '25,40,60'. Comma-separated (rather than a repeated flag) keeps a single
    pilot cell as short as --M 25 while a partial sweep stays one greppable
    token."""
    return [int(v) for v in value.split(',')]


def implement_tree_models_in_P4(clf_app, clf_ddos, selected_features,
                                num_classes_app=3, num_classes_ddos=2,
                                output_dir=OUTPUT_PATH,
                                output_filename='p4_code_RF_models.p4',
                                use_default_action_discount=False):
    """Compile two ALREADY-TRAINED Random Forests into one combined TNA
    program plus its control-plane entries, under joint encoding (both tasks
    share one discretization derived from the union of every tree's splits).

    Returns the path of the written .p4 file; `table_entries.json` is written
    alongside it in the same directory.

    Takes trained models rather than training them itself. The previous
    zero-argument version orchestrated its own dataset loading and training
    via training_and_feature_selection(), which could not run at all: that
    function calls train_classifier_RF, a name defined only in
    legacy/feature_sharing_script.py and never imported, so every invocation
    raised NameError. Only the training half was broken -- the P4 generation
    below is unchanged -- so the fix is to let callers own training and keep
    this as the model -> P4 interface.

    selected_features must be the model's ORDERED training-feature-name list
    (feature_names[i] is training column i), which is what export_text needs;
    it is not interchangeable with the alphabetically-sorted key order that
    get_feature_thresholds produces.
    """
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

    ensure_directory_exists(output_dir)
    feature_intervals_to_csv(feature_intervals, path_to_output=output_dir)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)

    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset,
                      path_to_output=output_dir,
                      use_default_action_discount=use_default_action_discount)

    return generate_P4_code(
        num_classes_app, num_classes_ddos, clf_app, clf_ddos,
        feature_intervals_app=feature_intervals, feature_intervals_ddos=feature_intervals,
        output_dir=output_dir, output_filename=output_filename,
        use_default_action_discount=use_default_action_discount,
        selected_features_app=selected_features,
        selected_features_ddos=selected_features)


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


def compare_independent_joint_mapping(M_values, n_splits, arms=None,
                                      max_workers=None,
                                      skip_existing=True):
    """Run one (arm, M) cell per output file.

    arms : list of (arm, TrainConfig). Defaults to PRIMARY_ARMS.
    skip_existing : skip any cell whose output file already exists. The
        campaign is ~40 h at a +/-2x estimate and the seven M values are
        independent runs, so it is meant to be resumed and chunked; the default
        makes re-invoking the same command continue rather than redo. Pass
        False (CLI: --redo) to force recomputation.
    """
    if arms is None:
        arms = PRIMARY_ARMS

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

    print("Starting per-task objective campaign")
    print("=" * 70)
    print(f"Total number of features: {X_app.shape[1]}")

    for max_blocks in M_values:
        for arm, cfg in arms:
            encoding = 'joint' if arm == 'joint' else 'disjoint'
            path = arm_result_path(arm, cfg, max_blocks)

            if skip_existing and os.path.exists(path):
                # Resumability: one file per (arm, M) IS the unit of work, and a
                # complete file means that cell is done. Cells are written
                # atomically below, so a file's existence is a reliable
                # completion marker rather than a maybe-partial artifact.
                print(f"\n=== M={max_blocks}  arm={cfg.arm_slug(encoding)} -- already complete, skipping ===")
                continue

            print(f"\n=== M={max_blocks}  arm={cfg.arm_slug(encoding)} ===")

            results_df = compare_feature_selection_approaches_parallel(
                X_app, X_ddos, y_app, y_ddos,
                max_blocks,
                feature_names=selected_features,
                n_splits=n_splits,
                arm=arm,
                cfg=cfg,
                random_state=42,
                max_workers=max_workers,
            )

            if len(results_df) == 0:
                # Every split raised (compare_feature_selection_approaches_parallel
                # swallows per-split exceptions into SplitResult.error), so there
                # is nothing to write. Writing an empty "complete" file here would
                # make skip_existing treat this cell as permanently done -- silent
                # data loss for the life of the campaign. Leave no file behind so
                # the next invocation retries the cell instead.
                print(f"=== M={max_blocks}  arm={cfg.arm_slug(encoding)} -- "
                      f"ALL SPLITS FAILED, not writing (cell will retry on next invocation) ===")
                continue

            # Identity columns, so an arm is recoverable from the row as well as
            # from the filename. P5 extends this to the full C.1 schema.
            # 'arm' itself is not stamped here: _run_elimination already sets it
            # per row, and re-stamping it at the frame level would silently
            # flatten any future heterogeneity in that column instead of
            # surfacing it.
            results_df['alignment_enabled'] = cfg.alignment_enabled and arm == 'joint'
            results_df['delta_align'] = cfg.delta_align_label(encoding)
            results_df['delta_select'] = cfg.delta_select
            results_df['M'] = max_blocks
            results_df['n_trees'] = cfg.n_trees
            results_df['max_depth'] = cfg.max_depth
            # Suppressed for the disjoint arm the same way delta_align is
            # (TrainConfig.overlap_threshold_label): alignment runs on the
            # joint arm only, so an independent-arm row carrying it would
            # misrepresent the baseline as having used a joint-arm setting.
            results_df['overlap_threshold'] = cfg.overlap_threshold_label(encoding)

            # Overwrite, NOT append -- and write atomically.
            #
            # The old code appended (mode='a', header=not file_exists). With one
            # file per (arm, M) as the resumability unit and a cost estimate
            # stated at +/-2x, re-running individual cells is expected, not
            # exceptional -- and appending would silently DOUBLE a re-run cell's
            # rows. Every claim in section C.3 is a paired test on (M, split, k);
            # duplicated rows corrupt those with no error and no visible symptom.
            #
            # Temp-then-rename so an interrupted cell never leaves a partial CSV
            # that the skip-if-exists guard above would read as complete.
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            tmp_path = path + '.partial'
            results_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)


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


def run_main():
    args = parse_args()

    #M = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # --M / --n-splits (both default to None) let a pilot cell run as a
    # command -- e.g. --M 25 --n-splits 2 -- instead of an edit to this file.
    # Omitting both must reproduce today's grid exactly.
    M = args.M if args.M is not None else [25, 40, 50, 60, 75, 90, 100]

    n_splits = args.n_splits if args.n_splits is not None else 15

    # Parallelization settings
    max_workers = None   # None = auto (cpu_count - 1), or set to specific number

    if args.mode == "compute":
        compare_independent_joint_mapping(
            M_values=M,
            n_splits=n_splits,
            arms=select_arms(args.arms),
            max_workers=max_workers,
            skip_existing=not args.redo,
        )

    else:
        df = load_and_combine_data(folder_path='results', M_values=M)

        #create_comparison_plots(df)

        analysis = analyze_multi_objective_results(df, list(range(17, 0, -2)))
        print(f"\nMulti approach covers {analysis['all_k']['coverage_ratio']['multi_covers_single']:.1%} of Single approach solutions")


if __name__ == '__main__':
    run_main()