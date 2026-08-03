import sys
from unittest.mock import patch
import main as m


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
    with patch("main.compare_independent_joint_mapping") as mock_compute, \
         patch("main.load_and_combine_data") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute"]):
        m.run_main()
        assert mock_compute.called
        assert not mock_plot.called


def test_main_block_dispatches_to_plot_path_when_mode_is_plot():
    with patch("main.compare_independent_joint_mapping") as mock_compute, \
         patch("main.load_and_combine_data") as mock_plot, \
         patch("main.analyze_multi_objective_results") as mock_analyze, \
         patch.object(sys, "argv", ["main.py", "--mode", "plot"]):
        mock_plot.return_value = None
        mock_analyze.return_value = {"all_k": {"coverage_ratio": {"multi_covers_single": 0.5}}}
        m.run_main()
        assert mock_plot.called
        assert not mock_compute.called
