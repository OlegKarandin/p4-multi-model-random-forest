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
         patch("src.main.load_and_combine_data") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute"]):
        m.run_main()
        assert mock_compute.called
        assert not mock_plot.called


def test_main_block_dispatches_to_plot_path_when_mode_is_plot():
    with patch("src.main.compare_independent_joint_mapping") as mock_compute, \
         patch("src.main.load_and_combine_data") as mock_plot, \
         patch("src.main.analyze_multi_objective_results") as mock_analyze, \
         patch.object(sys, "argv", ["main.py", "--mode", "plot"]):
        mock_plot.return_value = None
        mock_analyze.return_value = {"all_k": {"coverage_ratio": {"multi_covers_single": 0.5}}}
        m.run_main()
        assert mock_plot.called
        assert not mock_compute.called


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

_P4_FEATURES = ["Flow_IAT_Max", "Flow_IAT_Mean",
                "Fwd_IAT_Max", "Fwd_Packet_Length_Max"]


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


def test_compute_mode_runs_one_arm_per_cell_and_writes_one_file_each():
    """A run is one (arm, M) cell. If a run produced both arms, the
    independent baseline would be recomputed once per joint arm -- six
    identical copies, about half the campaign's compute."""
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
         patch("pandas.DataFrame.to_csv") as mock_csv:
        m.compare_independent_joint_mapping(
            M_values=[25], n_splits=2, arms=m.PRIMARY_ARMS)

    assert mock_run.call_count == 3       # one call per primary arm
    assert mock_csv.call_count == 3       # one file per (arm, M)
    # Overwrite, never append: a re-run cell must replace its rows, not double
    # them. Every C.3 claim is a paired test on (M, split, k).
    for call in mock_csv.call_args_list:
        assert call.kwargs.get('mode', 'w') == 'w'


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
