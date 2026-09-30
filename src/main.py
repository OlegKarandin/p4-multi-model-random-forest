from src.training.dataset import read_app_dataset, read_DDOS_dataset
from src.p4gen.build_p4_script import (
    INFINITE,
    OUTPUT_PATH,
    dt_thresholds_float_to_int,
    ensure_directory_exists,
    feature_intervals_from_nodes,
    feature_intervals_to_csv,
    generate_P4_code,
    generate_codewords,
    get_root_to_leaf_paths,
    get_table_entries,
    merge_tree_nodes,
    tree_nodes_for,
)
from src.training.config import TrainConfig
from src.training import campaign_run
from src.training.campaign_runner import plan_jobs, run_jobs

from src.reporting.campaign_data import load_campaign
from src.reporting import claims
from src.reporting import figures
from src.reporting.manifest import git_provenance, write_campaign_manifest

import argparse
import os
import sys
import numpy as np


# Design 2026-09-03 §2.1(c): ccp_alpha is implemented, validated and was
# switched off for the whole archive. It carries the larger of the two measured
# feasibility effects (+15.4pp vs alignment's +5.3pp at T >= 4), never costs
# feasibility, and uses its range rather than a degenerate corner. The cap does
# not bind -- 0.2% of winners land within 2x of it -- so raising it would only
# add dead space at the degenerate-pruning end.
CAMPAIGN_CCP_ALPHA_MAX = 0.05

# The compiler-verified campaign's grid (campaign_run.DEFAULT_M_GRID:
# 15, 25, 35, 50, 75 and the unbudgeted inf cell), replacing the archive grid
# [25, 50, 100, 150, 250]: the verified campaign re-runs every arm, so it no
# longer needs cells matched to the archive.
DEFAULT_M_GRID = list(campaign_run.DEFAULT_M_GRID)
DEFAULT_SPLITS = '0-9'


# `joint-off` is a genuine SKIP of the align_rf_thresholds call, so that arm is
# provably prediction-identical to the unaligned models and doubles as the
# requested ablation; `joint` runs alignment and keeps only the FREE moves.
#
# There is no swept tolerance axis any more, and therefore only one grid.
# Until 2026-09-15 this module also defined DELTA_ALIGNS = (0.0, 0.02, 0.05,
# 0.10, 0.20, None) and a SENSITIVITY_ARMS grid built from it; Track 5's
# pre-registered live-Optuna trial returned delta_helps = FALSE (mean_d000
# 0.7956173344395895 vs mean_d020 0.7861922400433382, cells_favouring_d020
# 14/24), so the axis and every arm that existed only to sweep it are gone --
# as is the `joint-dinf` accept-everything anchor, which only bounded the
# maximum sharing that axis could buy. Until 2026-09-14 the same arms were
# additionally crossed with an `overlap_threshold` axis, retired by design D4.
PRIMARY_ARMS = [
    ('independent', TrainConfig(ccp_alpha_max=CAMPAIGN_CCP_ALPHA_MAX)),
    ('joint', TrainConfig(alignment_enabled=False,
                          ccp_alpha_max=CAMPAIGN_CCP_ALPHA_MAX)),
    ('joint', TrainConfig(ccp_alpha_max=CAMPAIGN_CCP_ALPHA_MAX)),
]


def select_arms(which):
    if which == 'primary':
        return list(PRIMARY_ARMS)
    raise ValueError("arms must be 'primary', got {!r}".format(which))


def select_arm_slugs(slugs):
    """The (arm, cfg) pairs named by these arm slugs, in the order asked.

    `--arms`' presets are the campaign's own groupings; this is the escape
    hatch for naming a single cell, or an arm SET that crosses them. Track 5
    (spec 2026-09-15 §2.2) needed exactly {joint-d000, joint-d020, joint-dinf}
    -- neither preset -- run in a pre-registered ORDER that the M-outer /
    arm-inner loop in the job planner cannot emit, so each
    cell was launched as its own invocation and this is how one was named.
    Those three slugs no longer exist (Track 5 returned delta_helps = FALSE and
    the tolerance axis was deleted on 2026-09-15); the flag itself stays,
    because naming one cell is independently useful.

    Order is the CALLER's, not the catalogue's, because that order is the
    wall-clock hedge: spec §2.4 ranks the cells so a timeout costs context
    rather than the answer.
    """
    catalogue = {}
    for arm, cfg in PRIMARY_ARMS:
        encoding = 'disjoint' if arm == 'independent' else 'joint'
        catalogue[cfg.arm_slug(encoding)] = (arm, cfg)

    chosen = []
    for slug in slugs:
        if slug not in catalogue:
            raise ValueError(
                "unknown arm slug {!r}; known slugs are {}".format(
                    slug, ', '.join(sorted(catalogue))))
        chosen.append(catalogue[slug])
    return chosen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Feature selection experiment runner")
    parser.add_argument(
        "--mode", choices=["compute", "plot"], default="plot",
        help="'compute' runs new feature-selection experiments via "
             "run_compute_mode (expensive, real Optuna searches); "
             "'plot' loads and analyzes already-computed results (default, matches "
             "today's checked-in new_results=False behavior)")
    parser.add_argument(
        "--arms", choices=["primary"], default="primary",
        help="which arm grid to run in compute mode. 'primary' is the three "
             "arms the headline comparison needs (independent, joint@off and "
             "joint). The 'sensitivity'/'all' grids are gone with the "
             "delta_align tolerance axis they swept (2026-09-15)")
    parser.add_argument(
        "--arm-slugs", dest="arm_slugs", type=_parse_arm_slugs, default=None,
        help="run exactly these arms, named by slug (e.g. "
             "'--arm-slugs independent,joint'), in the order given. "
             "Overrides --arms. Exists because an arm SET can cross the "
             "--arms presets and because cell ORDER is sometimes "
             "pre-registered; pass an unknown slug to see the known ones in "
             "the error message")
    parser.add_argument(
        "--redo", action="store_true",
        help="recompute (arm, M, split) jobs whose split file already exists "
             "under --run. The default skips them, so re-running the same "
             "command resumes a partially finished campaign instead of "
             "redoing it")
    parser.add_argument(
        "--run", dest="run", default=None,
        help="run directory (required in compute mode): split CSVs land in "
             "<run>/rows/, designs in <run>/designs/, and <run>/run_manifest"
             ".json records the grid and every batch of splits run into it")
    parser.add_argument(
        "--splits", dest="splits", type=_parse_splits,
        default=campaign_run.parse_splits(DEFAULT_SPLITS),
        help="which CV splits to run, e.g. '0-9', '0,3,5' or '0-2,7'. "
             "Defaults to 0-9. Each split's data seed is 42 + split")
    parser.add_argument(
        "--M", dest="M", type=_parse_M_grid, default=list(DEFAULT_M_GRID),
        help="comma-separated TCAM block budgets to sweep in compute mode, "
             "e.g. '--M 25' for a single pilot cell or '--M 15,inf'; 'inf' "
             "is the unbudgeted cell. Defaults to the full grid "
             "[15,25,35,50,75,inf] when omitted")
    parser.add_argument(
        "--n-splits", action=_RemovedNSplits, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-workers", dest="max_workers", type=_parse_max_workers, default=None,
        help="number of parallel worker processes in compute mode, shared by "
             "all jobs. Defaults to min(n_jobs, cpu_count - 1) when omitted, which "
             "reserves one core for the orchestrator process; pass this to use "
             "every core on a small machine (e.g. --max-workers 4 on a 4-core "
             "Codespace) at the cost of the orchestrator competing with workers")
    parser.add_argument(
        "--allow-partial-family", dest="allow_partial_family",
        action="store_true",
        help="in plot mode, render even when the campaign under results/ "
             "does not yet cover the full 7-arm sweep (e.g. a single-M "
             "pilot). The default requires the complete pre-registered "
             "35-comparison Holm family (7 joint arms x 5 tests) and raises "
             "otherwise, so a partial campaign never silently applies a "
             "weaker multiplicity correction than the one pre-registered "
             "in spec C.3")
    args = parser.parse_args(argv)
    if args.mode == "compute" and args.run is None:
        parser.error("--run <dir> is required in compute mode")
    return args


class _RemovedNSplits(argparse.Action):
    """--n-splits is gone: a count could not say WHICH splits, so a run
    could neither resume nor shard. Fail loudly, naming the replacement."""
    def __init__(self, option_strings, dest, **kwargs):
        kwargs['nargs'] = '?'
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error("--n-splits was removed; use --splits (e.g. --splits 0-9)")


def _parse_M_grid(value):
    """--M's argparse type: comma-separated TCAM block budgets, e.g. '25' or
    '15,inf'. Every value is validated through campaign_run.m_token here, so a
    budget that cannot be named in a row_id fails at parse time, not hours
    into a run."""
    try:
        return campaign_run.parse_M_grid(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--M: {}".format(exc))


def _parse_splits(value):
    """--splits' argparse type: '0-9', '0,3,5' or a mix such as '0-2,7'."""
    try:
        return campaign_run.parse_splits(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def _parse_arm_slugs(value):
    """--arm-slugs' argparse type: comma-separated arm slugs, e.g.
    'joint' or 'independent,joint'. Same comma convention as --M, so a
    single-cell launch stays one short token."""
    slugs = [slug.strip() for slug in value.split(',') if slug.strip()]
    if not slugs:
        raise argparse.ArgumentTypeError(
            "--arm-slugs needs at least one slug, got {!r}".format(value))
    return slugs


def _parse_max_workers(value):
    """--max-workers' argparse type: positive integer worker-process count.
    Rejects zero or negative values with an error message that names the flag."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--max-workers must be positive, got {!r}".format(value))
    return n


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

    # extract node features (leaf or internal) straight off each estimator's
    # tree_ arrays. tree_nodes is needed below by get_root_to_leaf_paths,
    # get_table_entries, and (via feature_intervals_from_nodes) the joint
    # interval derivation -- computed once and shared across all three.
    tree_nodes_app = tree_nodes_for(clf_app, selected_features)
    tree_nodes = merge_tree_nodes(
        tree_nodes_app, tree_nodes_for(clf_ddos, selected_features))

    # Same "app tree count" offset get_table_entries needs below to re-key
    # the DDoS trees -- merge_tree_nodes applies the identical offset
    # internally when it builds tree_nodes above.
    offset = len(tree_nodes_app)

    feature_intervals = feature_intervals_from_nodes(tree_nodes)

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


def load_campaign_data():
    """The campaign's dataset, exactly as run_compute_mode sees it: both CSVs read, clipped at INFINITE, then correlation-pruned to
    the shared feature set.

    Extracted so scripts/replay_alignment.py cannot drift from the pipeline it
    is replaying -- a replay on a differently-pruned feature set is not a
    replay of anything.

    Returns (X_app, X_ddos, y_app, y_ddos, selected_features), where
    selected_features indexes the columns of both matrices.
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

    X_app, X_ddos, selected_features = remove_correlated_features_both_datasets(
        df_app, df_ddos)

    return (X_app, X_ddos, df_app.Label.to_numpy(), df_ddos.Label.to_numpy(),
            selected_features)


def run_compute_mode(args):
    """`--mode compute`'s entire body: one job per (arm, M, split).

    Loads the data, records the grid in <run>/run_manifest.json (a later
    invocation into the same run must use the same grid; it appends a batch),
    plans every job whose split file does not yet exist (all of them with
    --redo), runs them through one process pool and exits non-zero if any job
    failed, so the campaign log shows it. A failed job writes no file, so the
    next invocation retries it.
    """
    arms = (select_arm_slugs(args.arm_slugs) if args.arm_slugs is not None
            else select_arms(args.arms))
    M_values = args.M
    splits = args.splits

    X_app, X_ddos, y_app, y_ddos, selected_features = load_campaign_data()
    print("Starting compiler-verified campaign run in {}".format(args.run))
    print("=" * 70)
    print(f"Total number of features: {X_app.shape[1]}")

    write_campaign_manifest(args.run, arms, M_values, splits,
                            n_rows_app=X_app.shape[0], n_rows_ddos=X_ddos.shape[0])

    jobs = plan_jobs(arms, M_values, splits, args.run,
                     skip_existing=not args.redo)
    print(f"{len(jobs)} job(s) to run")
    summary = run_jobs(jobs, (X_app, X_ddos, y_app, y_ddos, selected_features),
                       args.run, git_provenance()['sha'],
                       max_workers=args.max_workers)

    print(f"\n{len(summary.done)} job(s) done, {len(summary.failed)} failed")
    for key, error in sorted(summary.failed.items()):
        print(f"FAILED {key}: {error.splitlines()[0] if error else ''}")
    if summary.failed:
        sys.exit(1)
    return summary


def run_plot_mode(results_dir='results', output_dir=None,
                  allow_partial_family=False):
    """`--mode plot`'s entire body: load the campaign, render every §C.5
    deliverable, and print where each one landed.

    Replaces `load_and_combine_data` + `analyze_multi_objective_results`,
    which built dead `..._-1_-1_{M}.csv` filenames the current pipeline
    never writes and fused analysis with plotting
    (`analyze_multi_objective_results` called `create_multidim_
    visualizations` unconditionally at `analysis.py:36`). `campaign_data.py`
    (loading/pairing), `claims.py` (every statistic) and `figures.py`
    (rendering only) keep those as separate layers; this function is the
    thin glue between them, not a third place either concern lives.

    `output_dir=None` defaults to `figures.DEFAULT_FIGURE_DIR`
    ('results/figures') -- resolved here rather than in the signature so
    `results_dir` and `output_dir` can be varied independently by a caller
    (e.g. a pilot run pointed at a scratch directory) without the two
    silently tracking each other.

    The capacity-ceiling appendix (deliverable 6) replays a measurement
    `scripts/capacity_ceiling.py` writes separately
    (`results_dir/capacity_ceiling.csv`) and has nothing in the campaign
    frame to reconstruct it from. `figures.appendix_6_capacity_ceiling`
    raises FileNotFoundError rather than rendering nothing when that file is
    absent, so this checks for it first and passes `ceiling_csv=None` --
    `figures.render_all`'s documented way to omit deliverable 6 -- instead
    of letting a routine pilot run (which has no ceiling measurement yet)
    fail outright.

    `allow_partial_family` controls the thing carried forward from Task 13,
    now gating deliverable 4's two independent Holm families PLUS
    deliverable 3's separate substitution family:
    `claims.paired_tests` (the 35-comparison superiority family),
    `claims.noninferiority_tests` (D13's 14-comparison non-inferiority
    family, Task 19), and `claims.substitution_test_all_arms` (the
    7-comparison substitution family, Task 23) all default their
    `expected_family_size` to None, which lets Holm-Bonferroni quietly
    correct over however many contrasts happen to be present -- a weaker
    correction than the pre-registered size on any campaign that has not
    yet run all seven joint arms, silent apart from a line in the rendered
    markdown. The default here (False) instead passes
    `claims.PRE_REGISTERED_FAMILY_SIZE`, `claims.NONINFERIORITY_FAMILY_SIZE`
    and `claims.SUBSTITUTION_FAMILY_SIZE` explicitly, so a partial campaign
    RAISES rather than silently weakening any of the three corrections. Pass
    `--allow-partial-family` (allow_partial_family=True) to render anyway --
    e.g. a single-M pilot, which by construction can never assemble the
    full 7-arm family and is not trying to support any corrected claim yet.
    """
    if output_dir is None:
        output_dir = figures.DEFAULT_FIGURE_DIR

    df = load_campaign(results_dir=results_dir)

    ceiling_csv = os.path.join(results_dir, 'capacity_ceiling.csv')
    if not os.path.exists(ceiling_csv):
        print(f"No capacity-ceiling measurement at {ceiling_csv!r} -- "
              f"skipping deliverable 6 (run scripts/capacity_ceiling.py "
              f"once, ~10 minutes, to produce it)")
        ceiling_csv = None

    expected_family_size = (
        None if allow_partial_family else claims.PRE_REGISTERED_FAMILY_SIZE)
    expected_noninferiority_family_size = (
        None if allow_partial_family else claims.NONINFERIORITY_FAMILY_SIZE)
    expected_substitution_family_size = (
        None if allow_partial_family else claims.SUBSTITUTION_FAMILY_SIZE)

    deliverables = figures.render_all(
        df, output_dir=output_dir, ceiling_csv=ceiling_csv,
        expected_family_size=expected_family_size,
        expected_noninferiority_family_size=expected_noninferiority_family_size,
        expected_substitution_family_size=expected_substitution_family_size)

    for deliverable in deliverables:
        print(f"\n=== {deliverable.number}. {deliverable.title} ===")
        for path in deliverable.paths:
            print(f"  wrote {path}")

    return deliverables


def run_main():
    args = parse_args()

    if args.mode == "compute":
        run_compute_mode(args)

    else:
        run_plot_mode(allow_partial_family=args.allow_partial_family)


if __name__ == '__main__':
    run_main()