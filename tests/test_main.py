import os
import sys
from unittest.mock import patch
from src import main as m


def test_parse_args_defaults_to_plot_mode():
    """Today's checked-in default is new_results = False (plotting/analysis
    of existing results) -- --mode must default to 'plot' so nobody's
    existing invocation habit changes silently."""
    args = m.parse_args([])
    assert args.mode == "plot"


def test_parse_args_accepts_compute_mode():
    args = m.parse_args(["--mode", "compute"])
    assert args.mode == "compute"


def test_parse_args_rejects_unknown_mode():
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--mode", "bogus"])


def test_main_block_dispatches_to_compute_path_when_mode_is_compute():
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch("src.main.run_plot_mode") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute"]):
        m.run_main()
        assert mock_compute.called
        assert not mock_plot.called


def test_main_block_dispatches_to_plot_path_when_mode_is_plot():
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch("src.main.run_plot_mode") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "plot"]):
        mock_plot.return_value = []
        m.run_main()
        assert mock_plot.called
        assert not mock_compute.called


def test_run_main_plot_mode_passes_the_allow_partial_family_flag_through():
    """--allow-partial-family must actually reach run_plot_mode, not just
    parse -- see test_parse_args_accepts_allow_partial_family_flag for the
    parsing half."""
    with patch("src.main.run_plot_mode") as mock_plot, \
         patch.object(sys, "argv",
                      ["main.py", "--mode", "plot", "--allow-partial-family"]):
        mock_plot.return_value = []
        m.run_main()
    assert mock_plot.call_args.kwargs['allow_partial_family'] is True


def test_run_main_plot_mode_defaults_allow_partial_family_to_false():
    with patch("src.main.run_plot_mode") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "plot"]):
        mock_plot.return_value = []
        m.run_main()
    assert mock_plot.call_args.kwargs['allow_partial_family'] is False


# ---------------------------------------------------------------------------
# --allow-partial-family
# ---------------------------------------------------------------------------

def test_parse_args_allow_partial_family_flag_defaults_to_false():
    assert m.parse_args([]).allow_partial_family is False


def test_parse_args_accepts_allow_partial_family_flag():
    assert m.parse_args(["--allow-partial-family"]).allow_partial_family is True


# ---------------------------------------------------------------------------
# run_plot_mode: the P7d rewire onto campaign_data.load_campaign +
# figures.render_all, replacing the old load_and_combine_data +
# analyze_multi_objective_results path (dead filenames, and fused analysis
# with plotting -- analyze_multi_objective_results called
# create_multidim_visualizations unconditionally).
# ---------------------------------------------------------------------------

def test_run_plot_mode_loads_the_campaign_from_the_given_results_dir():
    with patch("src.main.load_campaign") as mock_load, \
         patch("src.main.figures.render_all") as mock_render:
        mock_load.return_value = "the-df"
        mock_render.return_value = []
        m.run_plot_mode(results_dir="somewhere", output_dir="out")
    mock_load.assert_called_once_with(results_dir="somewhere")


def test_run_plot_mode_renders_the_loaded_frame_not_a_copy_or_a_summary():
    with patch("src.main.load_campaign") as mock_load, \
         patch("src.main.figures.render_all") as mock_render:
        mock_load.return_value = "the-df"
        mock_render.return_value = []
        m.run_plot_mode(output_dir="out")
    assert mock_render.call_args.args[0] == "the-df"


def test_run_plot_mode_defaults_to_the_pre_registered_holm_family_size():
    """Carried forward from Task 13: the figures path itself defaults
    expected_family_size to None, which lets Holm quietly correct over a
    smaller, weaker family on a partial campaign. main.py must wire the
    pre-registered 35-comparison family explicitly so a partial campaign
    raises instead of silently weakening the correction.

    Task 19 added a second, independent Holm family (D13's non-inferiority
    tests) gated the same way -- expected_noninferiority_family_size must
    default to claims.NONINFERIORITY_FAMILY_SIZE alongside
    expected_family_size, not be silently dropped on the way to
    figures.render_all. Task 23 added a third, the substitution family --
    expected_substitution_family_size must default to
    claims.SUBSTITUTION_FAMILY_SIZE the same way."""
    from src.reporting import claims
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(output_dir="out")
    assert mock_render.call_args.kwargs['expected_family_size'] == \
        claims.PRE_REGISTERED_FAMILY_SIZE
    assert mock_render.call_args.kwargs['expected_noninferiority_family_size'] == \
        claims.NONINFERIORITY_FAMILY_SIZE
    assert mock_render.call_args.kwargs['expected_substitution_family_size'] == \
        claims.SUBSTITUTION_FAMILY_SIZE


def test_run_plot_mode_allow_partial_family_disables_the_family_size_check():
    """--allow-partial-family must disable ALL THREE of the Holm family
    gates (deliverable 4's two plus deliverable 3's substitution family,
    Task 23), not just the superiority one (Task 19)."""
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(output_dir="out", allow_partial_family=True)
    assert mock_render.call_args.kwargs['expected_family_size'] is None
    assert mock_render.call_args.kwargs['expected_noninferiority_family_size'] is None
    assert mock_render.call_args.kwargs['expected_substitution_family_size'] is None


def test_run_plot_mode_omits_the_capacity_ceiling_deliverable_when_its_csv_is_absent(tmp_path):
    """scripts/capacity_ceiling.py has not necessarily been run against a
    given results_dir (e.g. a fresh pilot). appendix_6_capacity_ceiling
    raises FileNotFoundError rather than rendering nothing, so run_plot_mode
    must check for the file itself and pass ceiling_csv=None -- render_all's
    documented way to omit deliverable 6 -- instead of letting that
    exception propagate out of an otherwise-successful plot run."""
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(results_dir=str(tmp_path), output_dir="out")
    assert mock_render.call_args.kwargs['ceiling_csv'] is None


def test_run_plot_mode_prints_the_missing_ceiling_notice_loudly(tmp_path, capsys):
    """Ruling P7-6: a silently-omitted deliverable is the same class of
    failure as a silently truncated grid, so passing ceiling_csv=None must
    not be the only observable effect -- the notice that fires along the way
    has to actually print, and has to say which deliverable it is skipping
    and how to produce the missing file (same standard the manifest
    warning -- test_manifest.py's capsys tests -- was held to)."""
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(results_dir=str(tmp_path), output_dir="out")
    captured = capsys.readouterr()
    assert 'deliverable 6' in captured.out
    assert 'scripts/capacity_ceiling.py' in captured.out


def test_run_plot_mode_does_not_print_the_ceiling_notice_when_the_csv_exists(tmp_path, capsys):
    ceiling_csv = tmp_path / 'capacity_ceiling.csv'
    ceiling_csv.write_text("a,b\n1,2\n")
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(results_dir=str(tmp_path), output_dir="out")
    captured = capsys.readouterr()
    assert 'deliverable 6' not in captured.out


def test_run_plot_mode_passes_the_existing_capacity_ceiling_csv_through(tmp_path):
    ceiling_csv = tmp_path / 'capacity_ceiling.csv'
    ceiling_csv.write_text("a,b\n1,2\n")
    with patch("src.main.load_campaign", return_value="df"), \
         patch("src.main.figures.render_all") as mock_render:
        mock_render.return_value = []
        m.run_plot_mode(results_dir=str(tmp_path), output_dir="out")
    assert mock_render.call_args.kwargs['ceiling_csv'] == str(ceiling_csv)


# ---------------------------------------------------------------------------
# implement_tree_models_in_P4
#
# The old zero-argument version could not run at all: it called
# training_and_feature_selection, which calls train_classifier_RF -- a name
# that exists only in legacy/feature_sharing_script.py and is not imported,
# so any invocation raised NameError. The P4-generation half was fine; only
# the training orchestration was broken. It is now an interface that takes
# ALREADY-TRAINED models, so callers own their own training.
# ---------------------------------------------------------------------------

_P4_FEATURES = ["flow_iat_max", "flow_iat_mean",
                "fwd_iat_max", "fwd_packet_length_max"]


def _trained_pair():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(11)
    X = rng.randint(0, 60000, size=(600, 4))
    y_app = ((X[:, 0] // 20000) + (X[:, 2] // 25000)) % 3
    y_ddos = (X[:, 3] > 30000).astype(int)
    clf_app = RandomForestClassifier(n_estimators=2, max_depth=4,
                                     random_state=11, bootstrap=False).fit(X, y_app)
    clf_ddos = RandomForestClassifier(n_estimators=1, max_depth=4,
                                      random_state=11, bootstrap=False).fit(X, y_ddos)
    return clf_app, clf_ddos


def test_implement_tree_models_in_P4_generates_from_already_trained_models(tmp_path):
    import os

    clf_app, clf_ddos = _trained_pair()

    written = m.implement_tree_models_in_P4(
        clf_app, clf_ddos, _P4_FEATURES,
        output_dir=str(tmp_path) + os.sep)

    with open(written) as f:
        text = f.read()

    # both tasks' classification tables, the PHV pins, and no leftover markers
    assert "get_classification_tree_app_0" in text
    assert "get_classification_tree_ddos_0" in text
    assert "@pa_container_size" in text
    for marker in ("/* METADATA */", "/* TABLES */", "/* APPLY */", "/* PHV_PRAGMAS */"):
        assert marker not in text

    # the control-plane artifact lands beside it
    assert os.path.isfile(os.path.join(str(tmp_path), "table_entries.json"))


def test_implement_tree_models_in_P4_requires_trained_models():
    # Guards the interface change itself: the old no-argument form is gone,
    # so nobody can call the (previously NameError-ing) training path.
    import pytest
    with pytest.raises(TypeError):
        m.implement_tree_models_in_P4()


def test_compute_mode_runs_one_arm_per_cell_and_writes_one_file_each(tmp_path, monkeypatch):
    """A run is one (arm, M) cell. If a run produced both arms, the
    independent baseline would be recomputed once per joint arm -- six
    identical copies, about half the campaign's compute."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    frame = pd.DataFrame([{'arm': 'independent', 'split': 10, 'k': 3}])
    X = np.zeros((10, 4))

    # A real to_csv either raises or leaves a file at the path it's called
    # with; a side-effect-free mock is not an accurate stand-in for that, and
    # lets an unconditional os.replace(tmp_path, path) go completely
    # unexercised. Run in an isolated cwd with a real results/ dir so the
    # side effect actually creates the file os.replace needs.
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=frame) as mock_run, \
         patch("src.main.read_app_dataset"), \
         patch("src.main.read_DDOS_dataset"), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("pandas.DataFrame.to_csv",
               side_effect=lambda p, **kw: open(p, 'w').close()) as mock_csv:
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS)

    assert mock_run.call_count == 3       # one call per primary arm
    # Each call actually carried ITS OWN (arm, cfg) pair, not e.g. the same
    # cfg reused three times or arm/cfg transposed between calls.
    assert [c.kwargs['arm'] for c in mock_run.call_args_list] == \
        [arm for arm, _ in m.PRIMARY_ARMS]
    assert [c.kwargs['cfg'] for c in mock_run.call_args_list] == \
        [cfg for _, cfg in m.PRIMARY_ARMS]
    assert mock_csv.call_count == 3       # one file per (arm, M)
    # Overwrite, never append: a re-run cell must replace its rows, not double
    # them. Every C.3 claim is a paired test on (M, split, k).
    for call in mock_csv.call_args_list:
        assert call.kwargs.get('mode', 'w') == 'w'
    # os.replace actually ran (not short-circuited): the three real files
    # exist under results/, with no leftover .partial temp files.
    written = sorted(p.name for p in (tmp_path / 'results').iterdir())
    assert len(written) == 3
    assert all(not name.endswith('.partial') for name in written)


def test_independent_arm_rows_do_not_carry_the_joint_arms_alignment_settings(tmp_path, monkeypatch):
    """Regression: TrainConfig() defaults to alignment_enabled=True,
    delta_align=0.0 -- the SAME values joint-d000 uses -- so writing them
    unconditionally for every arm made the independent baseline's rows
    byte-identical to joint-d000's on these two columns, even though
    alignment never runs for the independent arm (spec A.2/C.1:
    delta_align='' and alignment_enabled should read as "off" there)."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    written = {}

    def fake_to_csv(self, path, **kw):
        written[path] = self.copy()
        open(path, 'w').close()

    frame = pd.DataFrame([{'arm': 'x', 'split': 10, 'k': 3}])
    X = np.zeros((10, 4))

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=frame), \
         patch("src.main.read_app_dataset"), \
         patch("src.main.read_DDOS_dataset"), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("pandas.DataFrame.to_csv", new=fake_to_csv):
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS)

    independent_df = next(df for p, df in written.items() if 'independent' in p)
    joint_d000_df = next(df for p, df in written.items() if 'joint-d000' in p)

    assert (~independent_df['alignment_enabled']).all()
    assert (independent_df['delta_align'] == '').all()
    assert joint_d000_df['alignment_enabled'].all()
    assert (joint_d000_df['delta_align'] == '0').all()

    # overlap_threshold is a joint-arm-only setting too (it only governs
    # candidate selection inside align_rf_thresholds, which the independent
    # arm never calls) -- same suppression as delta_align, same regression.
    assert (independent_df['overlap_threshold'] == '').all()
    assert (joint_d000_df['overlap_threshold'] == '0.5').all()

    # The two arms must actually differ -- guards against a fix that makes
    # both columns constant across arms instead of correctly arm-dependent.
    assert not independent_df['alignment_enabled'].equals(joint_d000_df['alignment_enabled'])
    assert not independent_df['delta_align'].equals(joint_d000_df['delta_align'])
    assert not independent_df['overlap_threshold'].equals(joint_d000_df['overlap_threshold'])


def test_a_cell_whose_file_already_exists_is_skipped():
    """Resumability: the campaign is ~40 h at a +/-2x estimate over seven
    independent M values, so re-invoking the same command must continue rather
    than redo -- and must not append to what is already there."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    frame = pd.DataFrame([{'arm': 'independent', 'split': 10, 'k': 3}])
    X = np.zeros((10, 4))

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=frame) as mock_run, \
         patch("src.main.read_app_dataset"), \
         patch("src.main.read_DDOS_dataset"), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("src.main.os.path.exists", return_value=True), \
         patch("pandas.DataFrame.to_csv"):
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS)

    assert mock_run.call_count == 0


def test_redo_forces_recomputation():
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    frame = pd.DataFrame([{'arm': 'independent', 'split': 10, 'k': 3}])
    X = np.zeros((10, 4))

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=frame) as mock_run, \
         patch("src.main.read_app_dataset"), \
         patch("src.main.read_DDOS_dataset"), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("src.main.os.path.exists", return_value=True), \
         patch("src.main.os.replace"), \
         patch("pandas.DataFrame.to_csv"):
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS, skip_existing=False)

    assert mock_run.call_count == 3


def test_redo_flag_defaults_to_off():
    assert m.parse_args([]).redo is False
    assert m.parse_args(['--redo']).redo is True


def test_a_cell_where_every_split_failed_is_not_written():
    """compare_feature_selection_approaches_parallel swallows per-split
    exceptions into SplitResult.error and returns a 0-row frame when every
    split failed. Writing that as a "complete" file would make skip_existing
    treat the cell as permanently done -- silent data loss for the life of
    the campaign -- so it must be skipped instead, leaving the cell to retry
    on the next invocation."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    empty_frame = pd.DataFrame([])
    X = np.zeros((10, 4))

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=empty_frame) as mock_run, \
         patch("src.main.read_app_dataset"), \
         patch("src.main.read_DDOS_dataset"), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("src.main.os.path.exists", return_value=False), \
         patch("src.main.os.replace") as mock_replace, \
         patch("pandas.DataFrame.to_csv") as mock_csv:
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS)

    assert mock_run.call_count == 3
    assert mock_csv.call_count == 0
    assert mock_replace.call_count == 0


# ---------------------------------------------------------------------------
# --M / --n-splits
#
# M and n_splits were hardcoded in run_main(), so running one small pilot
# cell meant editing main.py -- exactly the kind of edit that gets committed
# by accident and silently truncates a later full run. --M is comma-separated
# (a single flag reads better than a repeated one for a short list of
# integers, and keeps "--M 25" trivial for a one-cell pilot while "--M
# 25,40,60" stays a single, greppable token for a partial sweep).
# ---------------------------------------------------------------------------

def test_M_flag_parses_a_comma_separated_list():
    assert m.parse_args(["--M", "25,40,60"]).M == [25, 40, 60]


def test_M_flag_accepts_a_single_value():
    assert m.parse_args(["--M", "25"]).M == [25]


def test_M_and_n_splits_flags_default_to_none_so_run_main_can_supply_todays_values():
    args = m.parse_args([])
    assert args.M is None
    assert args.n_splits is None


def test_n_splits_flag_parses_as_an_int():
    assert m.parse_args(["--n-splits", "3"]).n_splits == 3


def test_n_splits_flag_rejects_zero_with_error_mentioning_flag_name(capsys):
    """--n-splits 0 must fail with an error message that names the flag,
    not fail incidentally inside ProcessPoolExecutor with a cryptic message."""
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--n-splits", "0"])
    captured = capsys.readouterr()
    assert 'n_splits' in captured.err or '--n-splits' in captured.err


def test_n_splits_flag_rejects_negative_with_error_mentioning_flag_name(capsys):
    """--n-splits with a negative value must fail with an error message that
    names the flag, not fail incidentally inside ProcessPoolExecutor."""
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--n-splits", "-1"])
    captured = capsys.readouterr()
    assert 'n_splits' in captured.err or '--n-splits' in captured.err


def test_omitting_M_and_n_splits_reproduces_todays_grid_exactly():
    """The property that matters most: a campaign invocation with no --M or
    --n-splits must run the exact same grid it runs today. A test asserting
    only that the flags parse would not catch a default that quietly drifted
    from [25, 40, 50, 60, 75, 90, 100] / 15 -- the failure mode this guards
    against is a full ~40h campaign that silently runs a truncated grid and
    looks like it succeeded."""
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute"]):
        m.run_main()

    assert mock_compute.call_args.kwargs['M_values'] == [25, 40, 50, 60, 75, 90, 100]
    assert mock_compute.call_args.kwargs['n_splits'] == 15


def test_M_and_n_splits_flags_actually_take_effect():
    """This is what makes a pilot cell a command rather than a patch: --M 25
    --n-splits 2 must reach compare_independent_joint_mapping unchanged, not
    just parse into args.M/args.n_splits."""
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch.object(sys, "argv",
                      ["main.py", "--mode", "compute", "--M", "25,40", "--n-splits", "2"]):
        m.run_main()

    assert mock_compute.call_args.kwargs['M_values'] == [25, 40]
    assert mock_compute.call_args.kwargs['n_splits'] == 2


# ---------------------------------------------------------------------------
# --max-workers
#
# max_workers was hardcoded to None (auto: min(n_splits, cpu_count - 1)) in
# run_main(), reserving one core for the orchestrator process. A small
# Codespace (e.g. a 4-core account ceiling) wants every core instead --
# --max-workers makes that a command-line flag rather than an edit to this
# file, same pattern as --M/--n-splits.
# ---------------------------------------------------------------------------

def test_max_workers_flag_defaults_to_none_so_run_main_can_supply_auto():
    assert m.parse_args([]).max_workers is None


def test_max_workers_flag_parses_as_an_int():
    assert m.parse_args(["--max-workers", "4"]).max_workers == 4


def test_max_workers_flag_rejects_zero_with_error_mentioning_flag_name(capsys):
    """--max-workers 0 must fail with an error message that names the flag,
    not fail incidentally inside ProcessPoolExecutor with a cryptic message."""
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--max-workers", "0"])
    captured = capsys.readouterr()
    assert 'max_workers' in captured.err or '--max-workers' in captured.err


def test_max_workers_flag_rejects_negative_with_error_mentioning_flag_name(capsys):
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--max-workers", "-1"])
    captured = capsys.readouterr()
    assert 'max_workers' in captured.err or '--max-workers' in captured.err


def test_omitting_max_workers_reproduces_todays_auto_behavior():
    """The property that matters most: a campaign invocation with no
    --max-workers must still let compare_independent_joint_mapping apply its
    own min(n_splits, cpu_count - 1) auto-detection, not silently pin a
    worker count."""
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute"]):
        m.run_main()

    assert mock_compute.call_args.kwargs['max_workers'] is None


def test_max_workers_flag_actually_takes_effect():
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch.object(sys, "argv",
                      ["main.py", "--mode", "compute", "--max-workers", "4"]):
        m.run_main()

    assert mock_compute.call_args.kwargs['max_workers'] == 4


# ---------------------------------------------------------------------------
# Gap 6 (P5): the run manifest, exercised end to end through
# compare_independent_joint_mapping rather than only at the module level
# (tests/test_manifest.py covers that). This is what proves the hook itself
# is wired to the grid actually passed in, lands in results/manifests/ (not
# results/, where it would trip skip_existing and the protected file-listing
# assertion above), and round-trips.
# ---------------------------------------------------------------------------

def test_a_run_manifest_lands_in_results_manifests_with_the_grid_actually_used(
        tmp_path, monkeypatch):
    import json
    import os
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    frame = pd.DataFrame([{'arm': 'independent', 'split': 10, 'k': 3}])
    X = np.zeros((10, 4))
    # Real DataFrames (unlike the bare-Mock read_*_dataset used by the other
    # compute-mode tests above) so df_app.shape[0] is a genuine, JSON-able
    # int -- the manifest write only actually lands when its inputs really
    # are picklable/serialisable, by design (see write_run_manifest's
    # docstring: build-then-serialise before any I/O).
    df_app = pd.DataFrame({'f': range(37), 'Label': [0] * 37})
    df_ddos = pd.DataFrame({'f': range(53), 'Label': [0] * 53})

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()

    with patch("src.main.compare_feature_selection_approaches_parallel",
               return_value=frame), \
         patch("src.main.read_app_dataset", return_value=df_app), \
         patch("src.main.read_DDOS_dataset", return_value=df_ddos), \
         patch("src.main.remove_correlated_features_both_datasets",
               return_value=(X, X, ['Flow.IAT.Max'])), \
         patch("pandas.DataFrame.to_csv",
               side_effect=lambda p, **kw: open(p, 'w').close()):
        m.compare_independent_joint_mapping(
            M_values=[25, 40], n_splits=2, arms=m.PRIMARY_ARMS)

    manifests_dir = tmp_path / 'results' / 'manifests'
    assert manifests_dir.is_dir()
    written = list(manifests_dir.glob('manifest_*.json'))
    assert len(written) == 1

    with open(written[0]) as f:
        loaded = json.load(f)

    assert loaded['M_values'] == [25, 40]
    assert loaded['n_splits'] == 2
    assert loaded['dataset_rows'] == {'app': 37, 'ddos': 53}
    assert len(loaded['arms']) == 3

    # And the protected assertion elsewhere in this file (the exact file
    # listing of results/) is exactly why this lives one level down: the top
    # of results/ itself carries only the two (arm, M) CSV files' worth of
    # per-cell output plus the manifests/ subdirectory, never a bare
    # manifest file competing with skip_existing's completion marker.
    top_level = {p.name for p in (tmp_path / 'results').iterdir()}
    assert 'manifests' in top_level
    assert all(name == 'manifests' or name.endswith('.csv') for name in top_level)


# ---------------------------------------------------------------------------
# run_plot_mode end-to-end: real load_campaign + real figures.render_all,
# nothing mocked. This is the test that would actually notice an averaged
# quantity reappearing on the path from a fitted model to a figure -- the
# defect this whole rerun (and P7d specifically) exists to eliminate.
# ---------------------------------------------------------------------------

def _plot_mode_row(arm, method, split, k, delta_align='', alignment_enabled=False,
                   overlap_threshold='', acc_app=0.9, acc_ddos=0.85, blocks=40):
    import json
    return {
        'arm': arm, 'method': method, 'split': split, 'k': k,
        'acc_app': acc_app, 'f1_app': acc_app - 0.02,
        'acc_ddos': acc_ddos, 'f1_ddos': acc_ddos - 0.02,
        'acc_sel_app': acc_app, 'acc_sel_ddos': acc_ddos,
        'stages': 3, 'blocks': blocks,
        'infeasible': '',
        'stages_real': '', 'tcam_real': '', 'compile_errors': '',
        'features_app': 'F1;F2', 'features_ddos': 'F1;F2',
        'best_params': json.dumps({'n_estimators': 11}),
        'rel_shortfall': 0.01, 'n_trials_run': 50, 'n_feasible': 10,
        'align_attempted': 2, 'align_accepted': 1,
        'intervals_before': 8, 'intervals_after': 7,
        'alignment_enabled': alignment_enabled, 'delta_align': delta_align,
        'delta_select': 0.02, 'overlap_threshold': overlap_threshold,
    }


def _write_plot_mode_campaign_file(results_dir, n_trees, max_depth, M,
                                   arm_slug, rows):
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame['M'] = M
    frame['n_trees'] = n_trees
    frame['max_depth'] = max_depth
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        results_dir / f'rf_t{n_trees}_d{max_depth}_M{M}_{arm_slug}.csv',
        index=False)


def _write_small_two_arm_campaign(results_dir):
    independent_rows = [
        _plot_mode_row('independent', 'single', split=s, k=k,
                       acc_app=0.90 + 0.001 * s, acc_ddos=0.80 + 0.001 * k,
                       blocks=30 + s)
        for s in range(3) for k in (5, 9)
    ]
    joint_rows = [
        _plot_mode_row('joint', 'multi', split=s, k=k,
                       alignment_enabled=True, delta_align='0.05',
                       overlap_threshold='0.5',
                       acc_app=0.91 + 0.001 * s, acc_ddos=0.82 + 0.001 * k,
                       blocks=25 + s)
        for s in range(3) for k in (5, 9)
    ]
    _write_plot_mode_campaign_file(results_dir, 11, 14, 25, 'independent',
                                   independent_rows)
    _write_plot_mode_campaign_file(results_dir, 11, 14, 25, 'joint-d005',
                                   joint_rows)


def test_plot_mode_end_to_end_never_averages_the_two_tasks_accuracy(tmp_path):
    """The defect the whole rerun exists to fix: the old analysis.py
    averaged acc_app and acc_ddos into one 'accuracy' number, which could
    hide a model excellent on one task and useless on the other. Drives
    --mode plot's real path (load_campaign -> figures.render_all, nothing
    mocked) over a small synthetic campaign and checks the rendered
    deliverable 1 data keeps the two tasks as separate columns."""
    results_dir = tmp_path / 'results'
    _write_small_two_arm_campaign(results_dir)

    deliverables = m.run_plot_mode(
        results_dir=str(results_dir), output_dir=str(tmp_path / 'figures'),
        allow_partial_family=True)

    front_table = next(d for d in deliverables
                       if d.slug == 'accuracy_vs_blocks_per_task')
    assert 'acc_app' in front_table.data.columns
    assert 'acc_ddos' in front_table.data.columns
    assert 'accuracy' not in front_table.data.columns
    assert not any('avg' in column.lower() for column in front_table.data.columns)


def test_plot_mode_end_to_end_writes_all_deliverables_that_apply_to_a_campaign_with_no_ceiling_csv(tmp_path):
    results_dir = tmp_path / 'results'
    _write_small_two_arm_campaign(results_dir)
    figures_dir = tmp_path / 'figures'

    deliverables = m.run_plot_mode(
        results_dir=str(results_dir), output_dir=str(figures_dir),
        allow_partial_family=True)

    # Seven, not eight: deliverable 6 (capacity ceiling) is correctly
    # omitted because no capacity_ceiling.csv exists under results_dir.
    assert len(deliverables) == 7
    assert sorted(d.number for d in deliverables) == [1, 2, 3, 4, 5, 7, 8]
    for deliverable in deliverables:
        assert len(deliverable.paths) > 0
        for path in deliverable.paths:
            assert os.path.isfile(path)


def test_plot_mode_end_to_end_raises_on_a_partial_campaign_when_allow_partial_family_is_not_set(tmp_path):
    """A two-arm synthetic campaign can never assemble the pre-registered
    7-arm family, so the default (allow_partial_family=False) must raise
    rather than silently Holm-correcting over the 3 comparisons this
    campaign actually has."""
    import pytest

    results_dir = tmp_path / 'results'
    _write_small_two_arm_campaign(results_dir)

    with pytest.raises(ValueError):
        m.run_plot_mode(results_dir=str(results_dir),
                        output_dir=str(tmp_path / 'figures'))
