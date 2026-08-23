"""Tests for feature_selection.py's real-compiler hardware-validation wiring:
`_process_single_split`'s two new opt-in parameters (`validate_on_hardware`,
`hardware_output_dir`) and the `_kickoff_hardware_validation` helper.

Task 3 of the 2026-08-03 P4-generator-fixes-and-config plan replaced the
disjoint loop's prior two-independent-programs-merged-via-`_MergedCompileHandle`
approximation with a single combined compile (via
`build_p4_script._resolve_disjoint_feature_plan`), matching the 'joint'
branch's own shape -- `_MergedCompileHandle` has since been removed as dead
code (no remaining callers).

Post-P2 cleanup: this file used to carry three `pytest.mark.slow` tests that
ran the real Optuna search (`train_multi_RF_Optuna_multi_constrained`)
end-to-end on a tiny synthetic dataset -- several minutes per
`_process_single_split` call even at this scale, since n_jobs=-1's
process-pool overhead dominates each trial's otherwise-trivial fit. Two of
them (hardware validation ON with a basic "compile was called" check, and
"disjoint/joint each get their own compile result, not a merged one")
became exact duplicates of
`test_process_single_split_splices_compile_results_onto_correct_iteration`
below once that test was restructured for the one-arm-per-call
`_process_single_split` (Task 5) -- both check disjoint's rows get one
stage count and joint's rows get another, with real hardware-validation
kickoff and real P4 codegen, just fast (mocked training) instead of slow
(real Optuna). They were deleted rather than fixed, to stop paying real
Optuna's wall time for coverage that already exists elsewhere. The third
(hardware validation OFF, confirming every row carries the three
compile-result keys as None) covered a genuinely distinct code path with no
fast equivalent, so it was converted to the same fast mocked-training
pattern the rest of this file already uses, instead of being deleted --
there is no `pytest.mark.slow` test left in this file.
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.training import feature_selection as fs
from src.training.train_model import TrainResult
from src.p4gen import p4_compile as pc


def _tiny_dataset(n=40, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 65535, size=(n, n_features))
    y_app = rng.randint(0, 3, size=n)
    y_ddos = rng.choice([-1, 1], size=n)
    return X, X.copy(), y_app, y_ddos


def _fit_tiny_rf(X, y, n_estimators=1, max_depth=2, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=seed).fit(X, y)
    return bps.dt_thresholds_float_to_int(clf)


def _stub_train_result(model_A, model_B, **overrides):
    """The 7-line TrainResult(...) boilerplate pasted at every fake trainer in
    this file (and in test_run_elimination.py, which keeps its own copy with
    different defaults -- the two files don't share a conftest.py). Every
    call site only cares about a handful of fields differing from the common
    "everything succeeded, nothing special" shape; **overrides substitutes
    just those."""
    fields = dict(
        model_A=model_A, model_B=model_B, stages=1, blocks=1, stage_depth=1,
        acc_sel_A=0.7, acc_sel_B=0.9, best_params={},
        rel_shortfall=0.0, n_trials_run=1, n_feasible=1,
        align_attempted=None, align_accepted=None,
        intervals_before=None, intervals_after=None,
        range_entries=1, ternary_entries=1,
        register_depth=1, register_count=1, register_sram_bits=1)
    fields.update(overrides)
    return TrainResult(**fields)


# ---------------------------------------------------------------------------
# Step 1/2: full _process_single_split behavior (adapted from the brief to
# the real signature/shape at feature_selection.py:439-453).
# ---------------------------------------------------------------------------

def test_process_single_split_without_hardware_validation_has_none_fields(monkeypatch):
    """Default path (validate_on_hardware=False): every row must carry the
    three compile-result keys as None -- `_process_single_split`'s docstring
    promises this unconditionally, not just when hardware validation ran.
    Fast/mocked (see the module docstring for why the real-Optuna version of
    this test was replaced): `train_multi_RF_Optuna_multi_constrained` is
    replaced with a fake that fits real (instant) tiny RandomForestClassifiers,
    same pattern as the tests below.
    """
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset()

    def _fake_train(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                     val_select_A, val_select_B, features_A, features_B,
                     max_blocks, encoding, cfg, warm_start_params=None):
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return _stub_train_result(model_A, model_B)

    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained', _fake_train)

    result = fs._process_single_split(
        split_idx=0, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
        max_blocks=50, feature_names=["f0", "f1", "f2"],
        random_state=42,
    )
    assert result.error is None
    assert len(result.results) > 0
    for row in result.results:
        assert row["stages_real"] is None
        assert row["tcam_real"] is None
        assert row["compile_errors"] is None


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

    Uses 3 starting features -- FEATURE_REGISTER_CATALOG's three
    dependency-free entries (flow_iat_max, fwd_iat_max,
    fwd_packet_length_max; the catalog's fourth entry, flow_iat_mean,
    requires flow_iat_max to also survive elimination alongside it -- see
    generate_P4_registers_and_apply's guard -- so it's avoided here to keep
    this test's real-compile path independent of elimination order) -- so
    both the disjoint ('single') and joint ('multi') loops each run exactly
    3 iterations (k=3,2,1), touching all three splice cases: first (no
    pending yet), middle (fills the previous iteration's pending AND kicks
    its own), and last (filled by the post-loop final join).
    `p4_compile.compile_p4_async` is faked to encode the k it
    was called for (parsed back out of the .p4 filename
    `_kickoff_hardware_validation` builds) into its returned CompileResult,
    so each row can be checked against the specific compile result that
    belongs to it.

    Task 5: `_process_single_split` now runs exactly one arm per call, so the
    disjoint ('single') and joint ('multi') loops are exercised by two
    separate calls (arm='independent' and arm='joint') sharing one
    `hardware_output_dir` -- the compile filenames already disambiguate by
    method, so nothing collides.
    """
    import os
    import re
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset(n=40, n_features=3)
    _CATALOG_FEATURES = ["flow_iat_max", "fwd_iat_max", "fwd_packet_length_max"]

    def _fake_train(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                     val_select_A, val_select_B, features_A, features_B,
                     max_blocks, encoding, cfg, warm_start_params=None):
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
        return _stub_train_result(model_A, model_B)

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

    with patch("src.training.train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("src.p4gen.p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result_single = fs._process_single_split(
            split_idx=3, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, feature_names=_CATALOG_FEATURES,
            random_state=42, arm='independent',
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )
        result_multi = fs._process_single_split(
            split_idx=3, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, feature_names=_CATALOG_FEATURES,
            random_state=42, arm='joint',
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )

    assert result_single.error is None
    assert result_multi.error is None

    single_rows = sorted((r for r in result_single.results if r['method'] == 'single'),
                          key=lambda r: -r['k'])
    multi_rows = sorted((r for r in result_multi.results if r['method'] == 'multi'),
                         key=lambda r: -r['k'])

    # 3 starting features -> k=3,2,1 for each loop: first iteration, the
    # middle iteration, and the last (post-loop-join) iteration are all
    # exercised.
    assert [r['k'] for r in single_rows] == [3, 2, 1]
    assert [r['k'] for r in multi_rows] == [3, 2, 1]

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


@pytest.mark.parametrize("arm", ["independent", "joint"])
def test_process_single_split_config_matches_equivalent_individual_kwargs(tmp_path, arm):
    """Task 4: a P4GenConfig passed via `config=` must produce the exact
    same result as passing its `validate_on_hardware`/`hardware_output_dir`
    values as the individual keyword arguments directly -- `config` is an
    additive convenience, not a different code path. Parametrized over both
    arms: the config-forwarding logic in `_process_single_split` is
    arm-independent, but a change that only broke one arm's plumbing would
    otherwise slip past single-arm coverage. Reuses the same fast
    mocked-training pattern as
    `test_process_single_split_splices_compile_results_onto_correct_iteration`
    above (real, instantly-fit tiny RandomForestClassifiers standing in for
    the expensive Optuna search; `p4_compile.compile_p4_async` faked the
    same way) rather than a new slow real-Optuna test.
    """
    from src.p4gen import p4_gen_config
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset(n=30, n_features=2)

    def _fake_train(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                     val_select_A, val_select_B, features_A, features_B,
                     max_blocks, encoding, cfg, warm_start_params=None):
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return _stub_train_result(model_A, model_B)

    def _fake_compile_async(p4_path, log_dir, **kwargs):
        return type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(
            errors=0, warnings=0, stages=7, tables=7, tcam=7)})()

    kwargs_dir = str(tmp_path / "kwargs") + "/"
    config_dir = str(tmp_path / "config") + "/"

    with patch("src.training.train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("src.p4gen.p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result_kwargs = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, arm=arm,
            feature_names=["flow_iat_max", "fwd_iat_max"], random_state=42,
            validate_on_hardware=True, hardware_output_dir=kwargs_dir,
        )

    cfg = p4_gen_config.P4GenConfig(validate_on_hardware=True, hardware_output_dir=config_dir)
    with patch("src.training.train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("src.p4gen.p4_compile.compile_p4_async", side_effect=_fake_compile_async):
        result_config = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, arm=arm,
            feature_names=["flow_iat_max", "fwd_iat_max"], random_state=42,
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


def test_kickoff_hardware_validation_uncatalogued_feature_returns_unavailable(tmp_path):
    # F2: generate_P4_code raises ValueError when a selected feature has no
    # FEATURE_REGISTER_CATALOG entry. _kickoff_hardware_validation must
    # catch that and degrade to ('unavailable', reason) instead of letting
    # the split die -- "f0"/"f1" are not catalogued (the real catalog only
    # has flow_iat_max, fwd_iat_max, fwd_packet_length_max, flow_iat_mean).
    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([-1, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    with patch("src.p4gen.p4_compile.compile_p4_async") as mock_compile:
        handle = fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'single', 2,
            clf_app, clf_ddos, ["f0", "f1"], ["f0", "f1"], 'disjoint')

    assert mock_compile.call_count == 0  # never reached: caught before compiling
    assert handle[0] == 'unavailable'
    assert isinstance(handle[1], str) and handle[1]  # non-empty reason
    assert "f0" in handle[1] or "f1" in handle[1]


def test_join_pending_compile_unavailable_leaves_stages_and_tcam_none():
    # Direct unit test of _join_pending_compile's F2 branch: an
    # ('unavailable', reason) handle must not be treated as a real Future
    # (calling .result() on the tuple would AttributeError) -- stages_real/
    # tcam_real stay None and the reason lands verbatim in compile_errors.
    row = {}
    fs._join_pending_compile(('unavailable', 'no catalog entry for X'), row)
    assert row['stages_real'] is None
    assert row['tcam_real'] is None
    assert row['compile_errors'] == 'no catalog entry for X'


def test_process_single_split_uncatalogued_feature_degrades_not_aborts(tmp_path):
    # Step 4's end-to-end guarantee: a split whose feature set includes an
    # uncatalogued feature must still COMPLETE (validate_on_hardware=True)
    # -- every row's hardware-validation fields record 'unavailable'
    # (stages_real/tcam_real None, a non-empty reason in compile_errors)
    # instead of the whole split raising.
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    X_app, X_ddos, y_app, y_ddos = _tiny_dataset(n=40, n_features=2)

    def _fake_train(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                     val_select_A, val_select_B, features_A, features_B,
                     max_blocks, encoding, cfg, warm_start_params=None):
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return _stub_train_result(model_A, model_B)

    with patch("src.training.train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train):
        result = fs._process_single_split(
            split_idx=0, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, feature_names=["f0", "f1"],
            random_state=42, arm='independent',
            validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        )

    assert result.error is None
    assert len(result.results) > 0
    for row in result.results:
        assert row['stages_real'] is None
        assert row['tcam_real'] is None
        assert isinstance(row['compile_errors'], str) and row['compile_errors']


def test_kickoff_hardware_validation_joint_makes_one_compile_call(tmp_path):
    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([-1, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    with patch("src.p4gen.p4_compile.compile_p4_async") as mock_compile:
        fake_future = type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(stages=9, tcam=1)})()
        mock_compile.return_value = fake_future

        handle = fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'multi', 2,
            clf_app, clf_ddos, ["flow_iat_max", "fwd_iat_max"], ["flow_iat_max", "fwd_iat_max"], 'joint')

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

    with patch("src.p4gen.p4_compile.compile_p4_async") as mock_compile:
        fake_future = type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(stages=5, tcam=1)})()
        mock_compile.return_value = fake_future

        handle = fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'single', 2,
            clf_app, clf_ddos, ["flow_iat_max", "fwd_iat_max"], ["flow_iat_max", "fwd_iat_max"], 'disjoint')

    assert mock_compile.call_count == 1
    # One combined .p4 file was actually written to disk (not two).
    written = sorted(p.name for p in tmp_path.glob("*.p4"))
    assert written == ["split0_single_k2.p4"]

    result = handle.result(timeout=1)
    assert result.stages == 5


# ---------------------------------------------------------------------------
# Follow-up (post-plan): _kickoff_hardware_validation must stop dropping the
# P4GenConfig object.
#
# Before this, neither of its two `generate_P4_code(...)` calls received
# `config` at all, so `P4GenConfig.match_type` / `use_default_action_discount`
# were silently ignored on the entire real-compiler-validation path -- the one
# path whose numbers the paper quotes. The `selected_features_*` lists
# `generate_P4_code` needs to recompute codewords are the same
# `feature_names_app`/`feature_names_ddos` this function already receives.
# ---------------------------------------------------------------------------

def _discount_and_exact_config(tmp_path):
    from src.p4gen import p4_gen_config
    return p4_gen_config.P4GenConfig(
        validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        use_default_action_discount=True, match_type='exact')


@pytest.mark.parametrize("encoding,method", [('disjoint', 'single'), ('joint', 'multi')])
def test_kickoff_hardware_validation_threads_config_into_generate_P4_code(
        tmp_path, encoding, method):
    """Both branches must forward `config` AND the ordered training-feature-name
    lists. `generate_P4_code` is spied on but still really runs (side_effect is
    the real function), so this checks the generated program too:
    match_type='exact' must actually reach the classification tables' key kind,
    and no table may declare a default action (a config carrying
    use_default_action_discount=True used to emit
    `const default_action = <action>(<literal>);`; that construct is gone --
    the discount now only shrinks table sizes / explicit table_entries.json
    entries, and the real default class is installed by the control plane, see
    build_p4_script.generate_P4_tables_and_apply's docstring).
    """
    import re
    from src.p4gen import build_p4_script as bps

    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([0, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    cfg = _discount_and_exact_config(tmp_path)
    real_generate_P4_code = bps.generate_P4_code

    with patch("src.p4gen.build_p4_script.generate_P4_code",
               side_effect=real_generate_P4_code) as spy_generate, \
         patch("src.p4gen.p4_compile.compile_p4_async") as mock_compile:
        mock_compile.return_value = type("F", (), {
            "result": lambda self, timeout=None: pc.CompileResult(stages=5, tcam=1)})()

        fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, method, 2,
            clf_app, clf_ddos, ["flow_iat_max", "fwd_iat_max"], ["flow_iat_max", "fwd_iat_max"], encoding, config=cfg)

    assert spy_generate.call_count == 1
    kwargs = spy_generate.call_args.kwargs
    assert kwargs["config"] is cfg
    assert kwargs["selected_features_app"] == ["flow_iat_max", "fwd_iat_max"]
    assert kwargs["selected_features_ddos"] == ["flow_iat_max", "fwd_iat_max"]

    text = (tmp_path / "split0_{}_k2.p4".format(method)).read_text()
    # The generated program declares no default action at all, discount or not.
    assert "default_action" not in text
    # match_type really landed in the generated program.
    assert re.search(r"meta\.code_\S+ : exact;", text)
    assert not re.search(r"meta\.code_\S+ : ternary;", text)


def test_kickoff_hardware_validation_without_config_is_unchanged(tmp_path):
    """Regression guard: omitting `config` must leave generate_P4_code on its
    own defaults -- no discount, ternary keys -- exactly as before this change.
    """
    import re
    X = np.random.RandomState(0).randint(0, 65535, size=(60, 2))
    y_app = np.random.RandomState(0).randint(0, 3, size=60)
    y_ddos = np.random.RandomState(1).choice([0, 1], size=60)
    clf_app = _fit_tiny_rf(X, y_app, seed=0)
    clf_ddos = _fit_tiny_rf(X, y_ddos, seed=1)

    with patch("src.p4gen.p4_compile.compile_p4_async") as mock_compile:
        mock_compile.return_value = type("F", (), {
            "result": lambda self, timeout=None: pc.CompileResult(stages=5, tcam=1)})()

        fs._kickoff_hardware_validation(
            True, str(tmp_path) + "/", 0, 'single', 2,
            clf_app, clf_ddos, ["flow_iat_max", "fwd_iat_max"], ["flow_iat_max", "fwd_iat_max"], 'disjoint')

    text = (tmp_path / "split0_single_k2.p4").read_text()
    assert "const default_action" not in text
    assert re.search(r"meta\.code_\S+ : ternary;", text)


def test_process_single_split_forwards_config_to_kickoff_hardware_validation(tmp_path):
    """`_process_single_split` already takes a `config`; it must now forward it
    one level further, to BOTH of its `_kickoff_hardware_validation` call sites
    (the disjoint/'single' loop and the joint/'multi' loop). Uses the same fast
    mocked-training pattern as the tests above; `_kickoff_hardware_validation`
    itself is spied on but still really runs, so the real generate_P4_code
    (with the config's discount actually in force) executes for every
    iteration.

    Task 5: `_process_single_split` now runs exactly one arm per call, so
    each of `_kickoff_hardware_validation`'s two call sites (disjoint/
    'single', joint/'multi') is exercised by its own call
    (arm='independent'/arm='joint'); the two calls' `spy_kickoff` histories
    are combined below to check both methods appear across them.
    """
    from src.p4gen import p4_gen_config
    from sklearn.ensemble import RandomForestClassifier
    from src.p4gen import build_p4_script as bps

    # Same size/shape/labels as every other _process_single_split test in this
    # file. In particular y_ddos keeps this codebase's {-1, 1} DDoS label
    # convention, which evaluation.accuracy_metrics hardcodes (`lab = [-1, 1]`)
    # -- remapping it to {0, 1} here would leave label -1 with no true and no
    # predicted samples and emit UndefinedMetricWarning noise.
    X_app, X_ddos, y_app, y_ddos = _tiny_dataset()

    def _fake_train(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                     val_select_A, val_select_B, features_A, features_B,
                     max_blocks, encoding, cfg, warm_start_params=None):
        model_A = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A))
        model_B = bps.dt_thresholds_float_to_int(
            RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1).fit(X_B, y_B))
        return _stub_train_result(model_A, model_B)

    def _fake_compile_async(p4_path, log_dir, **kwargs):
        return type("F", (), {"result": lambda self, timeout=None: pc.CompileResult(
            errors=0, warnings=0, stages=7, tables=7, tcam=7)})()

    cfg = p4_gen_config.P4GenConfig(
        validate_on_hardware=True, hardware_output_dir=str(tmp_path) + "/",
        use_default_action_discount=True)

    real_kickoff = fs._kickoff_hardware_validation

    with patch("src.training.train_model.train_multi_RF_Optuna_multi_constrained", side_effect=_fake_train), \
         patch("src.p4gen.p4_compile.compile_p4_async", side_effect=_fake_compile_async), \
         patch.object(fs, "_kickoff_hardware_validation",
                      side_effect=real_kickoff) as spy_kickoff:
        # FEATURE_REGISTER_CATALOG's three dependency-free entries (see
        # test_process_single_split_splices_compile_results_onto_correct_iteration
        # above for why flow_iat_mean is avoided here).
        catalog_features = ["flow_iat_max", "fwd_iat_max", "fwd_packet_length_max"]
        result_single = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, feature_names=catalog_features,
            random_state=42, arm='independent', config=cfg)
        result_multi = fs._process_single_split(
            split_idx=1, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
            max_blocks=50, feature_names=catalog_features,
            random_state=42, arm='joint', config=cfg)

    assert result_single.error is None
    assert result_multi.error is None
    # Each call runs exactly one arm to k=1, so together both call sites are
    # exercised.
    methods = {call.args[3] for call in spy_kickoff.call_args_list}
    assert methods == {'single', 'multi'}
    assert spy_kickoff.call_count > 0
    for call in spy_kickoff.call_args_list:
        assert call.kwargs.get("config") is cfg

    # And the config really reached the generated programs -- which now
    # declare no default action at all (the discount's effect moved to table
    # sizing + table_entries.json's is_default_action records).
    generated = sorted(tmp_path.glob("*.p4"))
    assert generated
    for p4_file in generated:
        text = p4_file.read_text()
        assert "classify_flow_codeword_" in text
        assert "default_action" not in text
