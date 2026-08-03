"""Tests for feature_selection.py's real-compiler hardware-validation wiring:
`_process_single_split`'s two new opt-in parameters (`validate_on_hardware`,
`hardware_output_dir`) and the `_kickoff_hardware_validation` helper.

Task 3 of the 2026-08-03 P4-generator-fixes-and-config plan replaced the
disjoint loop's prior two-independent-programs-merged-via-`_MergedCompileHandle`
approximation with a single combined compile (via
`build_p4_script._resolve_disjoint_feature_plan`), matching the 'joint'
branch's own shape -- `_MergedCompileHandle` has since been removed as dead
code (no remaining callers).

The three full end-to-end tests (`test_process_single_split_*`) exercise the
real training pipeline (`train_multi_RF_Optuna_multi_constrained`) on a tiny
synthetic dataset. Only the *compiler* is mocked (`p4_compile.compile_p4_async`),
never WSL2/p4c itself, so this file never attempts a real toolchain invocation
-- but the real Optuna search itself (hardcoded `n_trials=1000` with an
early-stopping callback requiring 25 feasible + 20 pareto-stable trials,
train_model.py:871-876/827-841) is measured to take several minutes per
`_process_single_split` call even on this tiny dataset (n_jobs=-1's process-
pool overhead dominates each trial's otherwise-trivial fit; not something
this task's scope covers fixing). These three are therefore marked
`pytest.mark.slow`, matching this repo's existing convention for expensive
real-dependency tests (see test_p4_compile.py's own real-toolchain test) --
not run by a bare `pytest` invocation (pytest.ini: `addopts = -m "not slow"`),
run explicitly with `pytest test_feature_selection.py -m slow -v`. The direct
unit tests below them exercise the exact same new code
(`_kickoff_hardware_validation`) against models fitted directly (bypassing
Optuna) and run in seconds.
"""

import numpy as np
import pytest
from unittest.mock import patch

import feature_selection as fs
import p4_compile as pc

# `feature_selection` imports `train_model`, which calls sklearnex's
# `patch_sklearn()` at import time (train_model.py:9-10) to globally swap in
# Intel's oneDAL-accelerated RandomForestClassifier. In this environment that
# accelerated implementation's tree (de)serialization is incompatible with
# the installed sklearn's `Tree.__setstate__` (a pre-existing version-skew
# bug in this conda env, unrelated to this task: "node array from the pickle
# has an incompatible dtype"), so ANY `RandomForestClassifier(...).fit(...)`
# call -- including every one inside `train_multi_RF_Optuna_multi_constrained`
# -- raises ValueError as soon as sklearnex's patch is active. Undoing the
# patch here (test-only, after the module import that applied it) restores
# plain sklearn for the rest of this test session so the real training
# pipeline these tests exercise actually runs, without touching train_model.py
# itself or any other production file.
from sklearnex import unpatch_sklearn
unpatch_sklearn()

# train_model.py's objective() hardcodes `RandomForestClassifier(**params,
# n_jobs=-1)` for every one of its up to 1000 Optuna trials
# (train_model.py:754-755). Measured directly (not guessed): on this tiny
# (27-40 row) synthetic data, n_jobs=-1 costs ~20x more wall time per .fit()
# than n_jobs=1 (0.023s vs 0.0005s/fit) -- Windows loky/joblib worker-pool
# overhead dominates a workload this trivial. That measured 20x turns a
# few-second training call into several minutes, times 6 calls per
# _process_single_split (3 iterations x 2 loops), which is impractical to
# run repeatedly as a test. This test-only monkeypatch forces n_jobs=1 for
# every RandomForestClassifier train_model.py constructs, without touching
# train_model.py itself -- it only changes how fast these tests run, not
# what train_model.py does in production.
import train_model as _train_model_module
from sklearn.ensemble import RandomForestClassifier as _RealRandomForestClassifier


def _forced_single_job_rf(*args, **kwargs):
    kwargs['n_jobs'] = 1
    return _RealRandomForestClassifier(*args, **kwargs)


_train_model_module.RandomForestClassifier = _forced_single_job_rf


def _tiny_dataset(n=40, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 65535, size=(n, n_features))
    y_app = rng.randint(0, 3, size=n)
    y_ddos = rng.choice([-1, 1], size=n)
    return X, X.copy(), y_app, y_ddos


def _fit_tiny_rf(X, y, n_estimators=1, max_depth=2, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    import build_p4_script as bps

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=seed).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


# ---------------------------------------------------------------------------
# Step 1/2: full _process_single_split behavior (adapted from the brief to
# the real signature/shape at feature_selection.py:439-453).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_process_single_split_without_hardware_validation_has_none_fields():
    X_app, X_ddos, y_app, y_ddos = _tiny_dataset()
    result = fs._process_single_split(
        split_idx=0, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
        n_trees=1, max_depth=3, max_blocks=50,
        feature_names=["f0", "f1", "f2"], random_state=42, verbose=False,
    )
    assert result.error is None
    assert len(result.results) > 0
    for row in result.results:
        assert row["stages_real"] is None
        assert row["tcam_real"] is None
        assert row["compile_errors"] is None


@pytest.mark.slow
def test_process_single_split_with_hardware_validation_calls_compile_async(tmp_path):
    X_app, X_ddos, y_app, y_ddos = _tiny_dataset()
    with patch("p4_compile.compile_p4_async") as mock_compile:
        fake_future = type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(
            errors=0, warnings=0, stages=3, tables=5, tcam=2)})()
        mock_compile.return_value = fake_future

        result = fs._process_single_split(
            split_idx=0, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            n_trees=1, max_depth=3, max_blocks=50,
            feature_names=["f0", "f1", "f2"], random_state=42, verbose=False,
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )

    assert result.error is None
    assert mock_compile.called
    assert len(result.results) > 0
    for row in result.results:
        assert "stages_real" in row and "tcam_real" in row and "compile_errors" in row
    # Every row must eventually be filled in (validate_on_hardware=True,
    # matching the "always present, only None if the compile really wasn't
    # collected" contract) -- both the 'single' (disjoint, merged-handle)
    # and 'multi' (joint) rows.
    assert all(row["stages_real"] is not None for row in result.results)
    assert any(row["stages_real"] == 3 for row in result.results)


@pytest.mark.slow
def test_process_single_split_disjoint_and_joint_rows_each_get_their_own_compile(tmp_path):
    """Task 3: the disjoint ('single') loop now compiles ONE combined
    program per iteration too (no more app-leg/ddos-leg merge) -- this
    proves each loop's rows get their OWN loop's real compile result (not a
    neighboring loop's, and not a merged sum), by giving the 'single' and
    'multi' loops different fake stage counts and checking each row's
    stages_real matches its own loop's value directly.
    """
    X_app, X_ddos, y_app, y_ddos = _tiny_dataset()

    import os
    import re

    call_count = {"n": 0}

    def _fake_compile_async(p4_path, output_dir, **kwargs):
        call_count["n"] += 1
        # Both loops now write ONE combined file per iteration, named
        # split{split_idx}_{method}_k{k}.p4 (see
        # _kickoff_hardware_validation) -- distinguish by the embedded
        # method segment ('single' for disjoint, 'multi' for joint). Match
        # against os.path.basename, NOT the full path: pytest's `tmp_path`
        # fixture derives its directory name from the TEST FUNCTION's own
        # name, which contains "single_split" as a substring (this very
        # test's name does), so a full-path substring check is a
        # false-positive trap.
        basename = os.path.basename(p4_path)
        m = re.match(r'split\d+_(single|multi)_k\d+\.p4$', basename)
        stages = 3 if m.group(1) == 'single' else 10
        return type("F", (), {"result": lambda self, timeout=None, s=stages: pc.CompileResult(
            errors=0, warnings=0, stages=s, tables=s, tcam=s)})()

    with patch("p4_compile.compile_p4_async", side_effect=_fake_compile_async) as mock_compile:
        result = fs._process_single_split(
            split_idx=7, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            n_trees=1, max_depth=3, max_blocks=50,
            feature_names=["f0", "f1", "f2"], random_state=42, verbose=False,
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )

    assert result.error is None
    assert call_count["n"] > 0
    single_rows = [row for row in result.results if row["method"] == "single"]
    multi_rows = [row for row in result.results if row["method"] == "multi"]
    assert len(single_rows) > 0 and len(multi_rows) > 0
    # disjoint rows: one combined program -> 3, never merged with joint's 10.
    assert all(row["stages_real"] == 3 for row in single_rows)
    # joint rows: one combined program -> 10.
    assert all(row["stages_real"] == 10 for row in multi_rows)


# ---------------------------------------------------------------------------
# Fast exercise of _process_single_split's real overlap/splicing mechanics
# (pending_previous / results[-2] indexing across iterations), without
# paying Optuna's >15-minutes-per-call cost like the three `slow` tests
# above. `train_multi_RF_Optuna_multi_constrained` is mocked with a fake
# that fits real (instant) tiny RandomForestClassifiers instead of running
# the real hyperparameter search; `p4_compile.compile_p4_async` is mocked
# the same way the four direct unit tests below already do.
# ---------------------------------------------------------------------------

def test_process_single_split_splices_compile_results_onto_correct_iteration(tmp_path):
    """Verifies each result row's `stages_real` is attributed to ITS OWN
    iteration's compile result -- not a neighboring iteration's -- across
    the full range of splicing cases in one run: the first iteration (no
    `pending_previous` yet, so the row is initially marked None and only
    filled in when joined during the NEXT iteration), a middle iteration,
    and the last iteration (filled in by the post-loop final join, since
    there is no next iteration to overlap with).

    Uses 4 starting features so both the disjoint ('single') and joint
    ('multi') loops each run exactly 4 iterations (k=4,3,2,1), touching all
    three cases. `p4_compile.compile_p4_async` is faked to encode the k it
    was called for (parsed back out of the .p4 filename
    `_kickoff_hardware_validation` builds) into its returned CompileResult,
    so each row can be checked against the specific compile result that
    belongs to it.
    """
    import os
    import re
    from sklearn.ensemble import RandomForestClassifier
    import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset(n=40, n_features=4)

    def _fake_train(X_A, y_A, X_B, y_B, x_val_A, y_val_A, x_val_B, y_val_B,
                     features_A, features_B, n_trees, max_depth, max_blocks,
                     encoding, warm_start_params=None):
        # Real (tiny, instantly-fit) models -- not mocked -- so the
        # downstream real code (accuracy_metrics, permutation_importance,
        # and _kickoff_hardware_validation's real generate_P4_code /
        # feature-interval derivation) all run for real on something
        # shaped like an actual trained model. Only the expensive Optuna
        # search itself is skipped.
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return model_A, model_B, 1, 1, {}

    def _fake_compile_async(p4_path, log_dir, **kwargs):
        # Task 3: both loops now write ONE combined file per iteration
        # (split{split_idx}_{method}_k{k}.p4, see
        # _kickoff_hardware_validation) -- distinguish by the embedded
        # method segment ('single' for disjoint, 'multi' for joint). Match
        # against os.path.basename, NOT the full path: pytest's `tmp_path`
        # fixture derives its directory name from the TEST FUNCTION's own
        # name, which itself may contain "single"/"multi" as a substring
        # (e.g. this very test's name contains "single_split"), so a
        # full-path substring check is a false-positive trap -- only the
        # filename's own embedded method segment is meaningful here.
        basename = os.path.basename(p4_path)
        m = re.match(r'split\d+_(single|multi)_k(\d+)\.p4$', basename)
        method, k = m.group(1), int(m.group(2))
        if method == 'single':
            stages = 100 + k  # disjoint loop's single combined program
        else:
            stages = 1000 + k  # joint loop's single combined program
        return type("F", (), {"result": lambda self, timeout=None, s=stages: pc.CompileResult(
            errors=0, warnings=0, stages=s, tables=s, tcam=s)})()

    with patch("train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result = fs._process_single_split(
            split_idx=3, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            n_trees=1, max_depth=3, max_blocks=50,
            feature_names=["f0", "f1", "f2", "f3"], random_state=42, verbose=False,
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )

    assert result.error is None

    single_rows = sorted((r for r in result.results if r['method'] == 'single'),
                          key=lambda r: -r['k'])
    multi_rows = sorted((r for r in result.results if r['method'] == 'multi'),
                         key=lambda r: -r['k'])

    # 4 starting features -> k=4,3,2,1 for each loop: first iteration,
    # (at least one) middle iteration, and the last (post-loop-join)
    # iteration are all exercised.
    assert [r['k'] for r in single_rows] == [4, 3, 2, 1]
    assert [r['k'] for r in multi_rows] == [4, 3, 2, 1]

    # Disjoint ('single') rows: one combined program per k -- 100+k.
    for row in single_rows:
        assert row['stages_real'] == 100 + row['k']
        assert row['tcam_real'] == 100 + row['k']
        assert row['compile_errors'] == 0

    # Joint ('multi') rows come from one combined program per k: 1000+k.
    for row in multi_rows:
        assert row['stages_real'] == 1000 + row['k']
        assert row['tcam_real'] == 1000 + row['k']
        assert row['compile_errors'] == 0


def test_process_single_split_config_matches_equivalent_individual_kwargs(tmp_path):
    """Task 4: a P4GenConfig passed via `config=` must produce the exact
    same result as passing its `validate_on_hardware`/`hardware_output_dir`
    values as the individual keyword arguments directly -- `config` is an
    additive convenience, not a different code path. Reuses the same fast
    mocked-training pattern as
    `test_process_single_split_splices_compile_results_onto_correct_iteration`
    above (real, instantly-fit tiny RandomForestClassifiers standing in for
    the expensive Optuna search; `p4_compile.compile_p4_async` faked the
    same way) rather than a new slow real-Optuna test.
    """
    import p4_gen_config
    from sklearn.ensemble import RandomForestClassifier
    import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset(n=30, n_features=2)

    def _fake_train(X_A, y_A, X_B, y_B, x_val_A, y_val_A, x_val_B, y_val_B,
                     features_A, features_B, n_trees, max_depth, max_blocks,
                     encoding, warm_start_params=None):
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return model_A, model_B, 1, 1, {}

    def _fake_compile_async(p4_path, log_dir, **kwargs):
        return type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(
            errors=0, warnings=0, stages=7, tables=7, tcam=7)})()

    kwargs_dir = str(tmp_path / "kwargs") + "/"
    config_dir = str(tmp_path / "config") + "/"

    with patch("train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result_kwargs = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            n_trees=1, max_depth=3, max_blocks=50,
            feature_names=["f0", "f1"], random_state=42, verbose=False,
            validate_on_hardware=True, hardware_output_dir=kwargs_dir,
        )

    cfg = p4_gen_config.P4GenConfig(validate_on_hardware=True, hardware_output_dir=config_dir)
    with patch("train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result_config = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            n_trees=1, max_depth=3, max_blocks=50,
            feature_names=["f0", "f1"], random_state=42, verbose=False,
            config=cfg,
        )

    assert result_kwargs.error is None
    assert result_config.error is None
    # Result rows never record the output directory itself, only the
    # compile numbers the (identically-faked) compiler returned -- so a
    # direct equality check confirms `config=` and the equivalent individual
    # kwargs produced byte-identical results.
    assert result_kwargs.results == result_config.results


# ---------------------------------------------------------------------------
# Direct unit tests for _kickoff_hardware_validation, isolated from the
# (slow) real Optuna training pipeline above.
# ---------------------------------------------------------------------------

def test_kickoff_hardware_validation_returns_none_when_disabled():
    assert fs._kickoff_hardware_validation(
        False, None, 0, 'single', 3, None, None, None, None, 'disjoint'
    ) is None


def test_kickoff_hardware_validation_joint_makes_one_compile_call(tmp_path):
    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([-1, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    with patch("p4_compile.compile_p4_async") as mock_compile:
        fake_future = type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(stages=9, tcam=1)})()
        mock_compile.return_value = fake_future

        handle = fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'multi', 2,
            clf_app, clf_ddos, ["f0", "f1"], ["f0", "f1"], 'joint')

    assert mock_compile.call_count == 1
    result = handle.result(timeout=1)
    assert result.stages == 9


def test_kickoff_hardware_validation_disjoint_makes_one_compile_call_and_writes_one_file(tmp_path):
    # Task 3: disjoint's real-compile validation now generates and compiles
    # ONE combined program (via _resolve_disjoint_feature_plan inside
    # generate_P4_code), matching real production deployment and the
    # 'joint' branch's own shape -- not two separate app-only/ddos-only
    # programs merged after the fact.
    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([-1, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    with patch("p4_compile.compile_p4_async") as mock_compile:
        fake_future = type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(stages=5, tcam=1)})()
        mock_compile.return_value = fake_future

        handle = fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'single', 2,
            clf_app, clf_ddos, ["f0", "f1"], ["f0", "f1"], 'disjoint')

    assert mock_compile.call_count == 1
    # One combined .p4 file was actually written to disk (not two).
    written = sorted(p.name for p in tmp_path.glob("*.p4"))
    assert written == ["split0_single_k2.p4"]

    result = handle.result(timeout=1)
    assert result.stages == 5
