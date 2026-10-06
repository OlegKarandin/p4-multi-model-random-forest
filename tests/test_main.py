import os
import sys
from unittest.mock import patch

import pytest

from src import main as m


def test_parse_args_defaults_to_plot_mode():
    """Today's checked-in default is new_results = False (plotting/analysis
    of existing results) -- --mode must default to 'plot' so nobody's
    existing invocation habit changes silently."""
    args = m.parse_args([])
    assert args.mode == "plot"


def test_parse_args_accepts_compute_mode():
    args = m.parse_args(["--mode", "compute", "--run", "results/r"])
    assert args.mode == "compute"


def test_parse_args_rejects_unknown_mode():
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--mode", "bogus"])


def test_main_block_dispatches_to_compute_path_when_mode_is_compute():
    with patch("src.main.run_compute_mode") as mock_compute, \
         patch("src.main.run_plot_mode") as mock_plot, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute", "--run", "r"]):
        m.run_main()
        assert mock_compute.called
        assert not mock_plot.called


def test_main_block_dispatches_to_plot_path_when_mode_is_plot():
    with patch("src.main.run_compute_mode") as mock_compute, \
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


# ---------------------------------------------------------------------------
# run_compute_mode: one CSV per (arm, M, split) under --run <dir>/rows/.
# Replaces compare_independent_joint_mapping's one-file-per-(arm, M) loop; the
# intents below (one arm per job, per-arm alignment column, resume, --redo,
# never write a failed unit of work, the manifest records the grid actually
# used) carry over to the per-split unit.
# ---------------------------------------------------------------------------

def _compute_args(tmp_path, *extra):
    return m.parse_args(["--mode", "compute", "--run", str(tmp_path / "run"),
                         "--M", "25", "--splits", "0-1", *extra])


def _stub_split_worker(calls, error=None):
    from src.training.feature_selection import SplitResult

    def fake(split_idx, X_app, X_ddos, y_app, y_ddos, max_blocks, feature_names,
             random_state, arm='independent', cfg=None, row_context=None, **kw):
        calls.append((arm, cfg, max_blocks, split_idx))
        rows = [{'row_id': 'x', 'arm': arm, 'split': split_idx, 'k': 3}]
        return SplitResult(split_idx=split_idx, results=rows, error=error)
    return fake


def _run_compute(args, calls, error=None, rows_app=10, rows_ddos=10):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    data = (np.zeros((rows_app, 4)), np.zeros((rows_ddos, 4)),
            np.zeros(rows_app), np.zeros(rows_ddos), ['Flow.IAT.Max'])
    with patch("src.main.load_campaign_data", return_value=data), \
         patch("src.training.campaign_runner._process_single_split",
               new=_stub_split_worker(calls, error)), \
         patch("src.training.campaign_runner.ProcessPoolExecutor",
               new=ThreadPoolExecutor):
        return m.run_compute_mode(args)


def test_compute_mode_runs_one_arm_per_job_and_writes_one_file_each(tmp_path):
    """A job is one (arm, M, split). If a job produced both arms, the
    independent baseline would be recomputed once per joint arm."""
    calls = []
    _run_compute(_compute_args(tmp_path), calls)

    assert len(calls) == 3 * 2              # 3 primary arms x 2 splits
    # Each job carried ITS OWN (arm, cfg) pair -- not the same cfg reused, and
    # not arm/cfg transposed between jobs.
    assert sorted((m.PRIMARY_ARMS.index((arm, cfg)), s)
                  for arm, cfg, _M, s in calls) == \
        [(i, s) for i in range(3) for s in (0, 1)]
    rows_dir = tmp_path / "run" / "rows"
    written = sorted(p.name for p in rows_dir.iterdir())
    assert len(written) == 6
    assert all(name.endswith('.csv') for name in written)   # no .partial left
    assert 'rf_t7_d14_M025_joint_s01.csv' in written


def test_independent_arm_rows_do_not_carry_the_joint_arms_alignment_settings(tmp_path):
    """Regression: TrainConfig() defaults to alignment_enabled=True, the SAME
    value the aligned joint arm uses, so it must be stamped per arm (spec
    A.2/C.1). overlap_threshold and delta_align are no longer written at all."""
    import pandas as pd
    _run_compute(_compute_args(tmp_path), [])
    rows_dir = tmp_path / "run" / "rows"
    independent_df = pd.read_csv(rows_dir / 'rf_t7_d14_M025_independent_s00.csv')
    joint_df = pd.read_csv(rows_dir / 'rf_t7_d14_M025_joint_s00.csv')

    assert (~independent_df['alignment_enabled']).all()
    assert joint_df['alignment_enabled'].all()
    for column in ('overlap_threshold', 'delta_align'):
        assert column not in independent_df.columns
        assert column not in joint_df.columns
    assert not independent_df['alignment_enabled'].equals(joint_df['alignment_enabled'])


def test_a_split_whose_file_already_exists_is_skipped(tmp_path):
    """Resumability: re-invoking the same command continues rather than
    redoes -- and never appends to what is already there."""
    first, second = [], []
    _run_compute(_compute_args(tmp_path), first)
    _run_compute(_compute_args(tmp_path), second)
    assert len(first) == 6
    assert second == []


def test_redo_forces_recomputation(tmp_path):
    first, second = [], []
    _run_compute(_compute_args(tmp_path), first)
    _run_compute(_compute_args(tmp_path, "--redo"), second)
    assert len(second) == 6


def test_redo_is_refused_once_the_run_has_verify_results(tmp_path, capsys):
    """A redo rewrites designs/<row_id>.p4 under the same row_id while the old
    verify/<row_id>.json still counts as done: stale p4c numbers for a
    different program. Refused before any data load or training."""
    import pytest
    verify = tmp_path / "run" / "verify"
    verify.mkdir(parents=True)
    (verify / "x.json").write_text("{}")
    calls = []
    with pytest.raises(SystemExit) as exc:
        _run_compute(_compute_args(tmp_path, "--redo"), calls)
    assert exc.value.code not in (0, None)
    assert calls == []
    err = capsys.readouterr().err
    assert "--redo" in err and "new --run" in err


def test_without_redo_a_verified_run_still_resumes(tmp_path):
    verify = tmp_path / "run" / "verify"
    verify.mkdir(parents=True)
    (verify / "x.json").write_text("{}")
    calls = []
    _run_compute(_compute_args(tmp_path), calls)
    assert len(calls) == 6


def test_redo_flag_defaults_to_off():
    assert m.parse_args([]).redo is False
    assert m.parse_args(['--redo']).redo is True


def test_a_split_that_failed_is_not_written_and_the_run_exits_non_zero(tmp_path):
    """A SplitResult with an error writes NO file (the next invocation
    retries it), and run_compute_mode exits non-zero so the campaign log
    shows it."""
    import pytest
    with pytest.raises(SystemExit) as exc:
        _run_compute(_compute_args(tmp_path), [], error='boom')
    assert exc.value.code != 0
    rows_dir = tmp_path / "run" / "rows"
    assert not rows_dir.exists() or list(rows_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# --run / --splits / --M
#
# The grid is a command, not an edit to main.py. --splits replaces the
# removed --n-splits (a count could not say WHICH splits, so a run could not
# be resumed or sharded); --M accepts 'inf' for the unbudgeted cell.
# ---------------------------------------------------------------------------

def test_compute_mode_requires_run(capsys):
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--mode", "compute"])
    assert '--run' in capsys.readouterr().err


def test_splits_and_M_parse_ranges_and_inf():
    args = m.parse_args(["--mode", "compute", "--run", "r",
                         "--splits", "0-2", "--M", "15,inf"])
    assert args.splits == [0, 1, 2]
    assert args.M == [15, float('inf')]


def test_n_splits_is_removed_and_the_error_names_splits(capsys):
    import pytest
    with pytest.raises(SystemExit):
        m.parse_args(["--mode", "compute", "--run", "r", "--n-splits", "3"])
    assert '--splits' in capsys.readouterr().err


def test_splits_defaults_to_0_to_9():
    assert m.parse_args([]).splits == list(range(10))


def test_M_flag_parses_a_comma_separated_list():
    assert m.parse_args(["--M", "25,40,60"]).M == [25, 40, 60]


def test_M_flag_accepts_a_single_value():
    assert m.parse_args(["--M", "25"]).M == [25]


def test_M_flag_defaults_to_the_campaign_grid():
    assert m.parse_args([]).M == [15, 25, 35, 50, 75, float('inf')]


@pytest.mark.parametrize("bad", ["15.5", "2000", "-inf", "INF", "0"])
def test_M_flag_rejects_an_unnameable_budget_at_parse_time(bad, capsys):
    """Fail fast: a budget m_token cannot name would otherwise crash inside
    row_id hours into a run."""
    with pytest.raises(SystemExit):
        m.parse_args(["--M", bad])
    assert '--M' in capsys.readouterr().err


def test_omitting_M_and_splits_runs_the_campaign_grid_exactly():
    """A campaign invocation with no --M or --splits must run the pre-
    registered grid; a default that quietly drifted would be a full campaign
    that silently runs a truncated grid and looks like it succeeded."""
    with patch("src.main.run_compute_mode") as mock_compute, \
         patch.object(sys, "argv", ["main.py", "--mode", "compute", "--run", "r"]):
        m.run_main()
    args = mock_compute.call_args.args[0]
    assert args.M == [15, 25, 35, 50, 75, float('inf')]
    assert args.splits == list(range(10))


def test_M_and_splits_flags_actually_take_effect(tmp_path):
    """--M 25 --splits 0-1 must reach the jobs unchanged, not just parse."""
    calls = []
    _run_compute(_compute_args(tmp_path), calls)
    assert {M for _a, _c, M, _s in calls} == {25}
    assert {s for _a, _c, _M, s in calls} == {0, 1}


# ---------------------------------------------------------------------------
# --max-workers
#
# max_workers was hardcoded to None (auto: min(n_jobs, cpu_count - 1)) in
# run_main(), reserving one core for the orchestrator process. A small
# Codespace (e.g. a 4-core account ceiling) wants every core instead --
# --max-workers makes that a command-line flag rather than an edit to this
# file, same pattern as --M/--splits.
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


def test_omitting_max_workers_reproduces_todays_auto_behavior(tmp_path):
    """No --max-workers must still let run_jobs apply its own
    min(n_jobs, cpu_count - 1) auto-detection, not silently pin a count."""
    import numpy as np
    from src.training.campaign_runner import RunSummary
    data = (np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(3), np.zeros(3), ['f'])
    with patch("src.main.load_campaign_data", return_value=data), \
         patch("src.main.run_jobs", return_value=RunSummary([], {})) as mock_run:
        m.run_compute_mode(_compute_args(tmp_path))
    assert mock_run.call_args.kwargs['max_workers'] is None


def test_max_workers_flag_actually_takes_effect(tmp_path):
    import numpy as np
    from src.training.campaign_runner import RunSummary
    data = (np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(3), np.zeros(3), ['f'])
    with patch("src.main.load_campaign_data", return_value=data), \
         patch("src.main.run_jobs", return_value=RunSummary([], {})) as mock_run:
        m.run_compute_mode(_compute_args(tmp_path, "--max-workers", "4"))
    assert mock_run.call_args.kwargs['max_workers'] == 4


# ---------------------------------------------------------------------------
# The run manifest, exercised end to end through run_compute_mode
# (tests/test_campaign_runner.py covers write_campaign_manifest directly):
# the hook is wired to the grid actually passed in and lands at
# <run>/run_manifest.json, beside rows/, never inside it.
# ---------------------------------------------------------------------------

def test_a_run_manifest_lands_in_the_run_dir_with_the_grid_actually_used(tmp_path):
    import json
    args = m.parse_args(["--mode", "compute", "--run", str(tmp_path / "run"),
                         "--M", "25,inf", "--splits", "0"])
    _run_compute(args, [], rows_app=37, rows_ddos=53)

    with open(tmp_path / "run" / "run_manifest.json") as f:
        loaded = json.load(f)
    assert loaded['M_values'] == [25, 'inf']
    assert loaded['splits'] == [0]
    assert loaded['dataset_rows'] == {'app': 37, 'ddos': 53}
    assert len(loaded['arms']) == 3
    assert len(loaded['batches']) == 1
    # rows/ holds only split CSVs: the manifest never competes with them.
    assert all(p.name.endswith('.csv') for p in (tmp_path / "run" / "rows").iterdir())


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
    # The fresh aligned arm (no delta_align tolerance since 2026-09-15), so
    # this legacy flat-directory campaign pairs one of claims.JOINT_ARM_SLUGS
    # against the baseline -- the archived 'joint-d005' slug is no longer a
    # contrast any claims.py family runs.
    joint_rows = [
        _plot_mode_row('joint', 'multi', split=s, k=k,
                       alignment_enabled=False, delta_align='',
                       overlap_threshold='',
                       acc_app=0.91 + 0.001 * s, acc_ddos=0.82 + 0.001 * k,
                       blocks=25 + s)
        for s in range(3) for k in (5, 9)
    ]
    _write_plot_mode_campaign_file(results_dir, 11, 14, 25, 'independent',
                                   independent_rows)
    _write_plot_mode_campaign_file(results_dir, 11, 14, 25, 'joint-off',
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
    3-arm family (independent + both joint arms), so the default
    (allow_partial_family=False) must raise rather than silently
    Holm-correcting over the comparisons this campaign actually has."""
    import pytest

    results_dir = tmp_path / 'results'
    _write_small_two_arm_campaign(results_dir)

    with pytest.raises(ValueError):
        m.run_plot_mode(results_dir=str(results_dir),
                        output_dir=str(tmp_path / 'figures'))


def test_load_campaign_data_returns_aligned_columns_and_names():
    """The replay harness maps a CSV row's ';'-joined feature names back to
    column indices, so the name list and the matrix must agree in width and
    order."""
    X_app, X_ddos, y_app, y_ddos, names = m.load_campaign_data()
    assert X_app.shape[1] == len(names)
    assert X_ddos.shape[1] == len(names)
    assert len(y_app) == X_app.shape[0]
    assert len(y_ddos) == X_ddos.shape[0]


def test_arm_slugs_flag_parses_a_comma_separated_list():
    args = m.parse_args(['--mode', 'compute', '--run', 'r',
                         '--arm-slugs', 'joint-d000,joint-d020'])

    assert args.arm_slugs == ['joint-d000', 'joint-d020']


def test_arm_slugs_defaults_to_none_so_arms_presets_still_apply():
    args = m.parse_args(['--mode', 'compute', '--run', 'r'])

    assert args.arm_slugs is None
    assert args.arms == 'primary'


# ---------------------------------------------------------------------------
# Task 14: a compiler-verified RUN through --mode plot -- the render gate
# (every design row must carry a verdict) and deliverables 9 and 10.
# ---------------------------------------------------------------------------

def _verified_run_rows(splits=(0, 1, 2, 3), ks=(5, 6), Ms=(35, 'inf')):
    """Rows and verification lines for a small three-arm run, built on Task
    12's `_run_row` / `_ver` builders."""
    from tests.test_campaign_data import _run_row, _ver

    arms = (('independent', 'independent', 'single', False, 0.000, 0),
            ('joint-off', 'joint', 'multi', False, 0.004, -2),
            ('joint-off-al', 'joint', 'multi', False, 0.008, -4))
    rows, lines = [], []
    for slug, arm, method, aligned, gain, saving in arms:
        for M in Ms:
            for split in splits:
                for k in ks:
                    blocks = 30 + split + k + saving
                    row = _run_row(split=split, k=k, M=M, blocks=blocks)
                    token = 'inf' if M == 'inf' else '{:03d}'.format(M)
                    row_id = '{}_M{}_s{:02d}_k{:02d}'.format(slug, token, split, k)
                    row.update({
                        'arm': arm, 'method': method,
                        'alignment_enabled': aligned, 'row_id': row_id,
                        'alignment_postprocess': slug == 'joint-off-al',
                        'acc_app': 0.80 + 0.01 * split + 0.002 * k + gain,
                        'f1_app': 0.78 + 0.01 * split + 0.002 * k + gain,
                        'acc_ddos': 0.90 + 0.005 * split - 0.001 * k + gain,
                        'f1_ddos': 0.88 + 0.005 * split - 0.001 * k + gain,
                    })
                    rows.append(row)
                    lines.append(_ver(row_id, M='' if M == 'inf' else M,
                                      p4c_stage_depth=9, p4c_blocks=blocks))
    return rows, lines


def test_plot_mode_end_to_end_over_a_verified_run_renders_deliverables_9_and_10(tmp_path):
    from tests.test_campaign_data import _write_run

    rows, lines = _verified_run_rows()
    run = _write_run(tmp_path, rows, lines)

    deliverables = m.run_plot_mode(results_dir=run,
                                   output_dir=str(tmp_path / 'figures'))

    assert [d.number for d in deliverables] == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    for deliverable in deliverables:
        assert deliverable.paths
        for path in deliverable.paths:
            assert os.path.isfile(path)


def test_plot_mode_refuses_to_render_a_run_with_a_missing_verdict(tmp_path, capsys):
    from src.reporting.campaign_data import UnverifiedRowsError
    from tests.test_campaign_data import _write_run

    rows, lines = _verified_run_rows(splits=(0,), ks=(5,), Ms=(35,))
    missing = lines.pop()['row_id']
    run = _write_run(tmp_path, rows, lines)

    with pytest.raises(UnverifiedRowsError):
        m.run_plot_mode(results_dir=run, output_dir=str(tmp_path / 'figures'))
    out = capsys.readouterr().out
    assert 'refusing to render: 1 rows have no verdict:' in out
    assert missing in out
    assert not (tmp_path / 'figures').exists()


def test_parse_args_results_dir_defaults_to_results():
    assert m.parse_args([]).results_dir == 'results'
    assert m.parse_args(['--results-dir', 'runs/r1']).results_dir == 'runs/r1'


def test_run_main_plot_mode_passes_the_results_dir_through():
    with patch("src.main.run_plot_mode") as mock_plot,          patch.object(sys, "argv",
                      ["main.py", "--mode", "plot", "--results-dir", "runs/r1"]):
        mock_plot.return_value = []
        m.run_main()
    assert mock_plot.call_args.kwargs['results_dir'] == 'runs/r1'


def test_arm_slugs_flag_drops_duplicate_slugs_keeping_first_order():
    args = m.parse_args(['--mode', 'compute', '--run', 'r',
                         '--arm-slugs', 'joint,independent,joint'])

    assert args.arm_slugs == ['joint', 'independent']
