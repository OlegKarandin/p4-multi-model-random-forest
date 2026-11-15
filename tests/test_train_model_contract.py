"""Spec B.1/B.2: one fit, per-task objectives, constraint on what is returned."""
from p4model import target
from p4model.errors import CrossbarKeyTooWide
from p4model.refusals import CONSTRAINTS, violations
import dataclasses

import numpy as np
import pytest

from src.training.config import TrainConfig
from src.training.errors import NoFeasibleSolution
from src.training.train_model import TrainResult, train_multi_RF_Optuna_multi_constrained

FEATURE_NAMES = ['Flow.IAT.Max', 'Fwd.IAT.Max', 'Fwd.Packet.Length.Max']


def _tiny_problem(n=4000, seed=0):
    """Small enough for a real Optuna search in a test, structured enough that
    a shallow forest beats chance. 4000 rows: a stump (a tree with no split) is
    refused in every encoding (R6), and forests trained on 300 were mostly stumps."""
    rng = np.random.default_rng(seed)
    X_app = rng.integers(0, 400, size=(n, 3)).astype(float)
    y_app = (X_app[:, 0] // 134).astype(int)
    X_ddos = rng.integers(0, 400, size=(n, 3)).astype(float)
    y_ddos = np.where(X_ddos[:, 1] > 200, 1, -1)
    return X_app, y_app, X_ddos, y_ddos


def _call(encoding='disjoint', cfg=None, max_blocks=60, n=4000, optuna_seed=0):
    """One small search. Seeded by default: an unseeded Optuna search makes
    any assertion about which trials it draws random."""
    X_app, y_app, X_ddos, y_ddos = _tiny_problem(n=n)
    cfg = cfg or TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3)
    return train_multi_RF_Optuna_multi_constrained(
        X_app, y_app, X_ddos, y_ddos,
        (X_app, y_app), (X_ddos, y_ddos),
        (X_app, y_app), (X_ddos, y_ddos),
        FEATURE_NAMES, FEATURE_NAMES,
        max_blocks, encoding, cfg, optuna_seed=optuna_seed)


def test_the_contract_returns_a_frozen_train_result_with_all_thirty_fields():
    out = _call()

    assert isinstance(out, TrainResult)
    assert len(dataclasses.fields(out)) == 30
    assert hasattr(out.model_A, 'predict') and hasattr(out.model_B, 'predict')
    assert isinstance(out.stages, (int, np.integer))
    assert isinstance(out.blocks, (int, np.integer))
    assert isinstance(out.stage_depth, (int, np.integer))
    assert 0.0 <= out.acc_sel_A <= 1.0
    assert 0.0 <= out.acc_sel_B <= 1.0
    assert isinstance(out.best_params, dict)
    assert isinstance(out.rel_shortfall, float)
    assert isinstance(out.n_trials_run, int) and out.n_trials_run > 0
    assert isinstance(out.n_feasible, int) and out.n_feasible > 0

    with pytest.raises(dataclasses.FrozenInstanceError):
        out.blocks = 999


def test_blocks_le_max_blocks_holds_for_the_RETURNED_models():
    """THE invariant, and the entire point of F7: the constraint is evaluated
    on the shipped artifact, not on a CV-fold proxy. Re-measure independently
    rather than trusting the returned number."""
    from p4gen.evaluation import multi_model_memory_evaluation

    for max_blocks in (40, 60, 90):
        out = _call(max_blocks=max_blocks)

        remeasured = multi_model_memory_evaluation(
            out.model_A, out.model_B, FEATURE_NAMES, FEATURE_NAMES, 'disjoint')

        assert remeasured.blocks == out.blocks, max_blocks
        assert out.blocks <= max_blocks, max_blocks


def test_the_returned_model_is_fit_on_the_full_training_set_not_a_fold():
    """F7: a 3-fold CV model sees ~2/3 of the rows. weighted_n_node_samples at
    the root of every tree must equal the full training set size -- plain
    n_node_samples undercounts under sklearn's default bootstrap=True, since
    the Splitter drops zero-weight (unsampled) rows from that unweighted
    count while still weighting the full row set into the fit."""
    X_app, y_app, X_ddos, y_ddos = _tiny_problem()
    cfg = TrainConfig(n_trials=8, min_feasible_before_stop=3, lookback=2)

    out = train_multi_RF_Optuna_multi_constrained(
        X_app, y_app, X_ddos, y_ddos,
        (X_app, y_app), (X_ddos, y_ddos),
        (X_app, y_app), (X_ddos, y_ddos),
        FEATURE_NAMES, FEATURE_NAMES, 60, 'disjoint', cfg)
    model_A, model_B = out.model_A, out.model_B

    for tree in model_A.estimators_:
        assert tree.tree_.weighted_n_node_samples[0] == len(y_app)
    for tree in model_B.estimators_:
        assert tree.tree_.weighted_n_node_samples[0] == len(y_ddos)


def test_an_impossible_block_budget_raises_no_feasible_solution():
    with pytest.raises(NoFeasibleSolution) as excinfo:
        _call(max_blocks=0)

    assert excinfo.value.max_blocks == 0
    assert excinfo.value.k == 3


def test_alignment_enabled_false_never_calls_align_with_policy(monkeypatch):
    """Spec A.2: the ablation arm is a genuine SKIP of the call, not an
    alignment run that happens to accept nothing, so the arm is provably
    prediction-identical to the unaligned models."""
    import src.training.train_model as tm

    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return args[0], args[1]

    monkeypatch.setattr(tm, 'align_with_policy', spy)

    _call(encoding='joint',
          cfg=TrainConfig(alignment_enabled=False, n_trials=6,
                          min_feasible_before_stop=2, lookback=2))

    assert calls == []


def test_the_independent_arm_never_aligns(monkeypatch):
    """Alignment is a joint-arm treatment. Under disjoint encoding each model
    keeps its own intervals, so there is nothing to share."""
    import src.training.train_model as tm

    calls = []
    monkeypatch.setattr(tm, 'align_with_policy',
                        lambda rf1, rf2, *a, **k: (calls.append(1), (rf1, rf2))[1])

    _call(encoding='disjoint')

    assert calls == []


def test_every_feasible_trial_records_the_attrs_the_selection_rule_reads(monkeypatch):
    """select_best_trial and constraint_values both read user_attrs, so a
    feasible trial missing one would crash the search at the very end."""
    import optuna
    import src.training.train_model as tm
    from src.training import early_stopping

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    _call()

    feasible = [t for t in captured['study'].trials if early_stopping.is_feasible(t)]
    assert feasible, 'the tiny problem should admit at least one feasible trial'
    for trial in feasible:
        for attr in ('acc_app', 'acc_ddos', 'blocks', 'stages') + CONSTRAINTS:
            assert attr in trial.user_attrs, (trial.number, attr)


def _only(trial, name, magnitude):
    """The trial records `magnitude` under `name` and 0 under every other
    constraint name (decision 9.2)."""
    assert {n: trial.user_attrs[n] for n in CONSTRAINTS} == violations(name, magnitude)


def test_crossbar_key_too_wide_records_key_too_wide(monkeypatch):
    """Task 10 (F3): a table whose key p4c refuses is a distinct failure from
    an over-budget model. The objective must record `key_too_wide` (live bits
    over target.MAX_TERNARY_KEY_BITS), leave every other
    constraint at 0, and return the infeasible triple rather than crash the
    search."""
    import optuna
    import src.training.train_model as tm
    from src.training.errors import NoFeasibleSolution

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    reported_bits = 600
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: (_ for _ in ()).throw(
            CrossbarKeyTooWide('table key too wide', reported_bits)))

    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        _only(trial, 'key_too_wide', reported_bits - target.MAX_TERNARY_KEY_BITS)
        assert trial.values == [-1.0, -1.0, float('inf')]


def test_range_entries_that_do_not_fit_record_range_rows_over(monkeypatch):
    """Decision 6d: a design whose real range entries need more rows than the compiled
    blocks hold is an infeasible trial, `range_rows_over` = rows over."""
    import optuna
    import src.training.train_model as tm
    from p4model.errors import RangeEntriesDoNotFit
    from src.training.errors import NoFeasibleSolution

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: (_ for _ in ()).throw(RangeEntriesDoNotFit('no room', 12)))

    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials
    for trial in trials:
        _only(trial, 'range_rows_over', 12)
        assert trial.values == [-1.0, -1.0, float('inf')]


def test_stage_depth_over_the_tofino_ceiling_records_stages_over_not_the_others(monkeypatch):
    """F5: a trial whose predicted pipeline depth exceeds the hard 12-stage
    Tofino-1 ceiling (TOFINO_PIPELINE_STAGES) must be rejected the same way
    the codeword/blocks/crossbar constraints are -- infeasible triple
    returned, only `stages_over` carrying a nonzero magnitude -- even
    though this trial is well within every OTHER constraint (small blocks,
    short codeword, narrow crossbar keys)."""
    import optuna
    import src.training.train_model as tm

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    import p4gen.evaluation as ev

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    over_ceiling_depth = tm.TOFINO_PIPELINE_STAGES + 3
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: ev.ResourceUsage(
            stages=1, blocks=1, stage_depth=over_ceiling_depth,
            range_entries=1, ternary_entries=1, codeword_length=40,
            register_depth=1, register_count=0,
            range_depth=1, ternary_depth=over_ceiling_depth,
            range_tables=1, ternary_tables=1))

    # Every trial is over-ceiling, so nothing is feasible -- same shape as
    # test_crossbar_key_too_wide_records_crossbar_violation.
    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        _only(trial, 'stages_over', over_ceiling_depth - tm.TOFINO_PIPELINE_STAGES)
        assert trial.values == [-1.0, -1.0, float('inf')]

    from src.training import early_stopping
    assert not any(early_stopping.is_feasible(t) for t in trials)


def test_the_winner_is_refit_deterministically_not_cached(monkeypatch):
    """F8: caching every feasible trial's model pair costs a measured 401 KB
    per pair, ~100 pairs per search, 11 workers. Because random_state is fixed
    in the params and alignment is deterministic given the same inputs, the
    winner can be refit from best_trial.params instead -- one fit-pair (~550 ms)
    rather than 40 MB of live cache per worker.

    The refit must reproduce the trial's own recorded numbers exactly. If it
    does not, something in the pipeline is non-deterministic and every reported
    result is a different model from the one that was selected."""
    import optuna
    import src.training.train_model as tm
    from src.training import early_stopping, trial_selection

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    # selection_rule='balanced': this test compares the shipped trial with
    # select_best_trial's pick, which coincide only under the balanced rule
    # (the tied_cheapest default may ship a different, cheaper trial).
    out = _call(cfg=TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3,
                                selection_rule='balanced'))

    feasible = [t for t in captured['study'].trials if early_stopping.is_feasible(t)]
    winner, winner_shortfall = trial_selection.select_best_trial(
        feasible, 0.02, k=3, max_blocks=60)

    assert out.best_params == dict(winner.params)
    assert out.blocks == winner.user_attrs['blocks']
    assert out.stages == winner.user_attrs['stages']
    assert out.acc_sel_A == winner.user_attrs['acc_app']
    assert out.acc_sel_B == winner.user_attrs['acc_ddos']
    assert out.rel_shortfall == winner_shortfall
    assert out.n_trials_run == len(captured['study'].trials)
    assert out.n_feasible == len(feasible)


def test_the_joint_arm_always_builds_the_metric_oracle(monkeypatch):
    """Spec A.2 once claimed an inf anchor -- `delta_rel=None` skipping the
    predict/restore/undo machinery entirely, making it the cheapest arm. That
    stopped holding at the 2026-08-30 policy-ladder deletion, and the arm
    itself is gone as of 2026-09-15 (Track 5's delta_helps = FALSE verdict).
    Alignment now judges every candidate at delta = 0, which needs the oracle,
    so the joint arm always pays for it -- pinned here at the train_model level
    and in test_threshold_alignment.py at the alignment level."""
    from src.training import threshold_alignment as ta

    scored = []
    real_init = ta.IncrementalMetrics.__init__

    def spy(self, tree_predictions, rf, y_true, task):
        scored.append(task)
        return real_init(self, tree_predictions, rf, y_true, task)

    monkeypatch.setattr(ta.IncrementalMetrics, '__init__', spy)

    _call(encoding='joint',
          cfg=TrainConfig(n_trials=6,
                          min_feasible_before_stop=2, lookback=2))

    assert scored, 'the joint arm must build the oracle now that gating is unconditional'
    assert set(scored) == {'app', 'ddos'}


def test_align_stats_on_the_result_describe_the_refit_not_an_earlier_trial(monkeypatch):
    """P5 gap 1: the post-search refit must pass its OWN align_stats dict into
    fit_pair, so the four align_* fields on the returned TrainResult describe
    the artifact actually returned -- not some earlier trial's user_attrs.

    Force every call's stats to be visibly distinct (a monotonically
    increasing counter) and assert the RETURNED result carries the counter
    value from the LAST call, which is the refit -- one more than the number
    of trials that called align_with_policy during the search. Before gap
    1 is fixed, the refit calls fit_pair without an align_stats kwarg, so the
    dict this test reads from stays empty and the fields come back None
    instead of matching the counter."""
    import src.training.train_model as tm

    calls = {'n': 0}

    def spy(rf1, rf2, *args, **kwargs):
        calls['n'] += 1
        n = calls['n']
        stats = kwargs.get('align_stats')
        if stats is not None:
            stats['attempted'] = n
            stats['accepted'] = n
            stats['intervals_before'] = n
            stats['intervals_after'] = n
        return rf1, rf2

    monkeypatch.setattr(tm, 'align_with_policy', spy)

    # selection_rule='balanced': the tied_cheapest default may add a SECOND
    # (reference) refit after the shipped one, which would make the last
    # counter value belong to the reference, not the shipped artifact.
    out = _call(encoding='joint',
                cfg=TrainConfig(n_trials=6, min_feasible_before_stop=2, lookback=2,
                                selection_rule='balanced'))

    assert calls['n'] >= 2, 'expected at least one trial call plus the refit call'
    assert out.align_attempted == calls['n']
    assert out.align_accepted == calls['n']
    assert out.intervals_before == calls['n']
    assert out.intervals_after == calls['n']


def test_alignment_fields_are_real_ints_when_alignment_runs():
    """The joint arm with alignment enabled is where align_rf_thresholds
    actually runs, so the four fields must be real (non-negative) ints, not
    the None sentinel reserved for arms where alignment never ran."""
    out = _call(encoding='joint',
                cfg=TrainConfig(n_trials=6, min_feasible_before_stop=2, lookback=2))

    for value in (out.align_attempted, out.align_accepted,
                  out.intervals_before, out.intervals_after):
        assert isinstance(value, int)
        assert value >= 0


def test_alignment_fields_are_none_not_zero_on_the_disjoint_arm():
    """Alignment is a joint-arm-only treatment (fit_pair never calls
    align_rf_thresholds under 'disjoint'), so the four fields have no value
    to report there. A silent 0 would be indistinguishable from 'alignment
    ran and accepted nothing', a real and different outcome -- so the chosen
    representation is None, which Task 8 writes into the CSV as ''."""
    out = _call(encoding='disjoint')

    assert out.align_attempted is None
    assert out.align_accepted is None
    assert out.intervals_before is None
    assert out.intervals_after is None


def test_alignment_fields_are_none_not_zero_when_the_joint_arm_disables_alignment():
    """Same None sentinel applies to the joint arm's own ablation: A.2's
    alignment_enabled=False is a genuine skip of align_rf_thresholds, not a
    run that accepted nothing, so it must be just as distinguishable from a
    real zero."""
    out = _call(encoding='joint',
                cfg=TrainConfig(alignment_enabled=False, n_trials=6,
                                min_feasible_before_stop=2, lookback=2))

    assert out.align_attempted is None
    assert out.align_accepted is None
    assert out.intervals_before is None
    assert out.intervals_after is None


def test_a_feasible_trial_records_stage_depth_separately_from_stages(monkeypatch):
    """§2.6. `stages` is OCCUPIED match-table count; `stage_depth` is PIPELINE
    DEPTH. Conflating them already reversed one stated conclusion (2026-08-27
    findings §9), so this asserts they are sourced from the two DIFFERENT
    ResourceUsage fields -- not merely that both keys exist, which would pass
    even if both were wired to usage.stages."""
    import optuna
    import src.training.train_model as tm
    import p4gen.evaluation as ev

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    # Deliberately different values, both feasible: blocks under max_blocks,
    # stage_depth under TOFINO_PIPELINE_STAGES, codeword short.
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: ev.ResourceUsage(
            stages=3, blocks=8, stage_depth=7,
            range_entries=1, ternary_entries=1, codeword_length=40,
            register_depth=1, register_count=0,
            range_depth=4, ternary_depth=7,
            range_tables=1, ternary_tables=1))

    _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    from src.training import early_stopping
    feasible = [t for t in captured['study'].trials
                if early_stopping.is_feasible(t)]
    assert feasible, 'the search should have produced at least one feasible trial'
    for trial in feasible:
        assert trial.user_attrs['stages'] == 3
        assert trial.user_attrs['stage_depth'] == 7


def test_rf_params_from_params_reproduces_the_search_space_mapping():
    """The refit path and the replay harness must build identical estimators
    from a recorded best_params dict, or a replayed model is not the campaign's
    model. random_state is fixed at 42, which is what makes the refit
    assertion at train_model.py:373-377 -- and the whole replay -- possible."""
    import src.training.train_model as tm

    params = {'n_estimators_A': 7, 'min_samples_leaf_A': 25,
              'min_samples_split_A': 50, 'max_depth_A': 9,
              'n_estimators_B': 3, 'min_samples_leaf_B': 15,
              'min_samples_split_B': 30, 'max_depth_B': 4}
    assert tm.rf_params_from_params(params, 'A') == {
        'n_estimators': 7, 'min_samples_leaf': 25, 'min_samples_split': 50,
        'max_depth': 9, 'random_state': 42, 'ccp_alpha': 0.0}
    assert tm.rf_params_from_params(params, 'B')['n_estimators'] == 3


def test_ccp_alpha_min_floor_nests_the_unpruned_baseline():
    """CCP_ALPHA_MIN must be small enough that fitting at ccp_alpha=CCP_ALPHA_MIN
    is bit-identical to fitting at ccp_alpha=0 (today's baseline)."""
    import src.training.train_model as tm
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(5)
    X = rng.integers(0, 90000, size=(300, 4)).astype(float)
    y = np.array([c % 3 for c in range(300)])
    kwargs = dict(n_estimators=7, max_depth=5, min_samples_leaf=20, random_state=0)

    baseline = RandomForestClassifier(ccp_alpha=0.0, **kwargs).fit(X, y)
    floored = RandomForestClassifier(ccp_alpha=tm.CCP_ALPHA_MIN, **kwargs).fit(X, y)

    assert len(baseline.estimators_) == len(floored.estimators_)
    for t1, t2 in zip(baseline.estimators_, floored.estimators_):
        assert np.array_equal(t1.tree_.threshold, t2.tree_.threshold)
        assert np.array_equal(t1.tree_.feature, t2.tree_.feature)
        assert np.array_equal(t1.tree_.children_left, t2.tree_.children_left)
        assert np.array_equal(t1.tree_.children_right, t2.tree_.children_right)


def test_n_trees_min_pins_the_n_estimators_dimension(monkeypatch):
    """cfg.n_trees_min == cfg.n_trees must collapse the n_estimators search
    dimension to a single value -- the T-pinning mechanism config.py's
    n_trees_min docstring describes."""
    import optuna
    import src.training.train_model as tm

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    _call(cfg=TrainConfig(n_trees_min=3, n_trees=3, n_trials=6,
                          min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        assert trial.params['n_estimators_A'] == 3
        assert trial.params['n_estimators_B'] == 3


def test_default_cfg_search_space_has_no_ccp_alpha_dimension_and_n_estimators_low_of_1(monkeypatch):
    """Structural contract at default cfg (n_trees_min=1, ccp_alpha_max=0.0):
    the n_estimators dimension keeps today's low=1 bound, and the ccp_alpha
    dimension must be entirely ABSENT -- not present-but-zero -- since a
    present dimension would change the search space Optuna's sampler sees
    even if every sampled value happened to collapse to 0."""
    import optuna
    import src.training.train_model as tm

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        study = real_create(*args, **kwargs)
        captured['study'] = study
        return study

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    _call(cfg=TrainConfig(n_trials=6, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        assert trial.distributions['n_estimators_A'].low == 1
        assert trial.distributions['n_estimators_B'].low == 1
        assert 'ccp_alpha_A' not in trial.distributions
        assert 'ccp_alpha_B' not in trial.distributions
        assert 'ccp_alpha_A' not in trial.params
        assert 'ccp_alpha_B' not in trial.params


def test_rf_params_from_params_ccp_alpha_get_default_round_trip():
    """Archived best_params dicts (all of campaign_backup_20260825) have no
    ccp_alpha_* key -- rf_params_from_params must default to 0.0 for those,
    and honor the key when a future archive does carry it."""
    import src.training.train_model as tm

    archive_shaped = {
        'n_estimators_A': 7, 'min_samples_leaf_A': 25,
        'min_samples_split_A': 50, 'max_depth_A': 9,
    }
    assert tm.rf_params_from_params(archive_shaped, 'A')['ccp_alpha'] == 0.0

    with_ccp_alpha = dict(archive_shaped, ccp_alpha_A=0.00042)
    assert tm.rf_params_from_params(with_ccp_alpha, 'A')['ccp_alpha'] == 0.00042


def test_rf_params_from_params_still_reads_archived_raw_min_samples_split():
    """Every one of campaign_backup_20260825's 7055 rows records a raw
    min_samples_split_<suffix>. The multiplier parameterisation must not make
    those rows unrefittable, or every replay study over the archive breaks
    silently -- design 2026-09-03 §2.1(a), the highest-risk item."""
    import src.training.train_model as tm

    archived = {'n_estimators_A': 3, 'min_samples_leaf_A': 75,
                'min_samples_split_A': 270, 'max_depth_A': 11}
    assert tm.rf_params_from_params(archived, 'A') == {
        'n_estimators': 3, 'min_samples_leaf': 75, 'min_samples_split': 270,
        'max_depth': 11, 'random_state': 42, 'ccp_alpha': 0.0}


def test_rf_params_from_params_reads_the_new_multiplier_key():
    """Rows written after the reparameterisation carry a multiplier instead."""
    import src.training.train_model as tm

    new_shape = {'n_estimators_A': 3, 'min_samples_leaf_A': 25,
                 'min_samples_split_mult_A': 4, 'max_depth_A': 11}
    built = tm.rf_params_from_params(new_shape, 'A')
    assert built['min_samples_split'] == 100
    assert built['min_samples_leaf'] == 25


def test_archived_best_params_refit_to_a_byte_identical_forest():
    """Stronger than the kwargs check: the actual estimator is unchanged.
    Uses the archived fixture's own recorded best_params."""
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import src.training.train_model as tm

    frame = pd.read_csv('tests/fixtures/rf_t11_d14_M25_historical.csv')
    params = json.loads(frame['best_params'].dropna().iloc[0])

    rng = np.random.default_rng(0)
    X = rng.integers(0, 400, size=(600, 3)).astype(float)
    y = (X[:, 0] // 134).astype(int)

    via_helper = RandomForestClassifier(
        **tm.rf_params_from_params(params, 'A'), n_jobs=1).fit(X, y)
    literal = RandomForestClassifier(
        n_estimators=params['n_estimators_A'],
        min_samples_leaf=params['min_samples_leaf_A'],
        min_samples_split=params['min_samples_split_A'],
        max_depth=params['max_depth_A'],
        random_state=42, ccp_alpha=0.0, n_jobs=1).fit(X, y)

    assert len(via_helper.estimators_) == len(literal.estimators_)
    for a, b in zip(via_helper.estimators_, literal.estimators_):
        assert np.array_equal(a.tree_.threshold, b.tree_.threshold)
        assert np.array_equal(a.tree_.feature, b.tree_.feature)


@pytest.mark.parametrize('leaf', [5, 15, 25, 35, 45, 55, 65, 105, 195])
def test_only_the_lowest_multiplier_is_inert(leaf):
    """min_samples_split <= 2*min_samples_leaf cannot bind: a node needs
    min_samples_split samples AND both children need min_samples_leaf. That
    inert region covered ~400 of the old grid's 800 combinations. Expressing
    split as a multiplier shrinks it to ONE value by construction rather than
    clamping, which would preserve the many-to-one mapping that wastes the TPE
    budget: since decision 9.1 the lowest multiplier, 2, sits on the boundary
    (split == 2*leaf, inert), and every higher one binds (split > 2*leaf)."""
    import src.training.train_model as tm

    assert tm.MIN_SAMPLES_SPLIT_MULT_MIN == 2
    for mult in range(tm.MIN_SAMPLES_SPLIT_MULT_MIN,
                      tm.MIN_SAMPLES_SPLIT_MULT_MAX + 1):
        split = tm.min_samples_split_from_mult(leaf, mult)
        if mult == 2:
            assert split == 2 * leaf
        else:
            assert split > 2 * leaf


def test_multiplier_bounds_are_two_to_sixty_six():
    """Decision 9.1: the multiplier starts at 2 (split == 2*leaf is the
    boundary, reachable); mult=3 is the first that always binds."""
    import src.training.train_model as tm

    assert (tm.MIN_SAMPLES_SPLIT_MULT_MIN, tm.MIN_SAMPLES_SPLIT_MULT_MAX) == (2, 66)
    assert tm.min_samples_split_from_mult(25, 2) == 50
    assert tm.min_samples_split_from_mult(25, 3) == 75


def test_search_declares_the_multiplier_range():
    dist = _search_distributions()['min_samples_split_mult_A']
    assert (dist.low, dist.high) == (2, 66)


def test_search_declares_odd_trees_to_7_and_depth_to_20():
    # decision 9.1: one shared search space for every arm (trees 1..7 since Task 32)
    trees = _search_distributions()['n_estimators_A']
    assert (trees.low, trees.high, trees.step) == (1, 7, 2)
    depth = _search_distributions()['max_depth_A']
    assert (depth.low, depth.high) == (2, 20)


def test_warm_start_jitter_keeps_trees_odd_within_1_15_and_depth_in_range():
    import numpy as np
    import src.training.train_model as tm
    rng = np.random.default_rng(1)
    for n0 in (1, 3, 13, 15):
        for d0 in (2, 10, 20):
            for _ in range(100):
                out = tm._vary_hyperparams(
                    {'min_samples_leaf_A': 5, 'min_samples_split_mult_A': 2,
                     'n_estimators_A': n0, 'max_depth_A': d0},
                    tm.SEARCH_MAX_TREES, tm.SEARCH_MAX_DEPTH, rng)
                assert out['n_estimators_A'] % 2 == 1 and 1 <= out['n_estimators_A'] <= 15
                assert 2 <= out['max_depth_A'] <= 20


def test_leaf_jitter_stays_inside_the_bounds_and_moves_small_values():
    import numpy as np
    import src.training.train_model as tm

    lo, hi = tm.MIN_SAMPLES_LEAF_MIN, tm.MIN_SAMPLES_LEAF_MAX
    seen = set()
    for start in (1, 2, 5, 30, 55):
        rng = np.random.default_rng(0)
        for _ in range(200):
            out = tm._vary_hyperparams(
                {'min_samples_leaf_A': start, 'min_samples_split_mult_A': 2,
                 'n_estimators_A': 3, 'max_depth_A': 5}, 15, 20, rng)
            assert 1 <= out['n_estimators_A'] <= 15
            assert 2 <= out['max_depth_A'] <= 20
            assert lo <= out['min_samples_leaf_A'] <= hi
            assert out['n_estimators_A'] % 2 == 1
            assert (tm.MIN_SAMPLES_SPLIT_MULT_MIN <= out['min_samples_split_mult_A']
                    <= tm.MIN_SAMPLES_SPLIT_MULT_MAX)
            seen.add((start, out['min_samples_leaf_A']))
    assert (30, 30) in seen and any(s == 30 and v != 30 for s, v in seen)
    assert (1, 1) in seen and all(v >= 1 for _, v in seen)
    assert any(s == 55 and v == 55 for s, v in seen)


def _search_distributions(cfg=None):
    """Every Optuna distribution the search declares, keyed by param name.

    rf_params is a closure inside train_multi_RF_Optuna_multi_constrained, so
    the search space is not directly callable. Capturing the Study the way
    scripts/feasibility_frontier.py:285-287 does is how a test reads it.
    """
    import optuna
    cfg = cfg or TrainConfig(n_trials=6, min_feasible_before_stop=2, lookback=2)
    captured = {}
    real_create_study = optuna.create_study
    optuna.create_study = (
        lambda *a, **kw: captured.setdefault('study', real_create_study(*a, **kw)))
    try:
        _call(cfg=cfg)
    except Exception:
        pass
    finally:
        optuna.create_study = real_create_study
    return captured['study'].trials[0].distributions


def test_min_samples_leaf_is_log_uniform_integer_one_to_fifty_five():
    """Decision 9.1: was 5..55 step 10."""
    for name in ('min_samples_leaf_A', 'min_samples_leaf_B'):
        dist = _search_distributions()[name]
        assert (dist.low, dist.high, dist.step, dist.log) == (1, 55, 1, True)


def test_the_search_declares_a_multiplier_and_no_raw_split():
    """The reparameterisation must actually replace the dimension, not add to
    it -- a search carrying both would be strictly worse than before."""
    names = set(_search_distributions())

    assert 'min_samples_split_mult_A' in names
    assert 'min_samples_split_mult_B' in names
    assert 'min_samples_split_A' not in names
    assert 'min_samples_split_B' not in names


def test_default_config_still_declares_no_ccp_alpha_dimension():
    """Non-campaign callers are unaffected by the campaign's arm settings:
    TrainConfig() defaults ccp_alpha_max to 0.0, so the dimension is absent."""
    names = set(_search_distributions())

    assert 'ccp_alpha_A' not in names
    assert 'ccp_alpha_B' not in names


def test_a_key_with_no_crossbar_fit_inside_the_bit_limit_is_still_a_violation(monkeypatch):
    """Review focus 1: table_blocks raises CrossbarKeyTooWide with live bits
    <= 524 when the crossbar cannot place the key. bits - 524 would be <= 0, which
    early_stopping.is_feasible reads as FEASIBLE. The violation is floored at 1."""
    import optuna
    import src.training.train_model as tm
    from src.training import early_stopping
    from src.training.errors import NoFeasibleSolution

    captured = {}
    real_create = optuna.create_study

    def capture(*args, **kwargs):
        captured['study'] = real_create(*args, **kwargs)
        return captured['study']

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: (_ for _ in ()).throw(CrossbarKeyTooWide('no lane fit', 40)))
    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=3, min_feasible_before_stop=2, lookback=2))
    for trial in captured['study'].trials:
        _only(trial, 'key_too_wide', 1)
        assert not early_stopping.is_feasible(trial)


def test_key_too_wide_is_the_bit_overflow_floored_at_one():
    assert CrossbarKeyTooWide('', 600).magnitude == 600 - target.MAX_TERNARY_KEY_BITS
    assert CrossbarKeyTooWide('', 524).magnitude == 1
    assert CrossbarKeyTooWide('', 11).magnitude == 1
    assert CrossbarKeyTooWide('', 600).constraint == 'key_too_wide'


def test_the_result_carries_the_reference_and_the_trial_table():
    out = _call()
    assert isinstance(out.chosen_trial, int) and isinstance(out.ref_trial, int)
    assert isinstance(out.n_tied, int) and out.n_tied >= 1
    assert out.blocks <= out.ref_blocks
    numbers = [row['number'] for row in out.trial_table]
    assert numbers == sorted(numbers) and len(numbers) == out.n_trials_run
    shipped = next(r for r in out.trial_table if r['number'] == out.chosen_trial)
    assert shipped['tied'] is True and shipped['blocks'] == out.blocks
    if out.ref_trial == out.chosen_trial:
        assert out.ref_model_A is None and out.ref_model_B is None
        assert (out.ref_blocks, out.ref_acc_sel_A) == (out.blocks, out.acc_sel_A)
    else:
        assert hasattr(out.ref_model_A, 'predict')


def test_balanced_rule_ships_the_reference():
    out = _call(cfg=TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3,
                                selection_rule='balanced'))
    assert out.chosen_trial == out.ref_trial
    assert out.ref_model_A is None


def test_alpha_zero_ships_the_cheapest_feasible_trial():
    out = _call(cfg=TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3,
                                select_alpha=0.0))
    feasible = [r for r in out.trial_table if r['feasible']]
    assert out.blocks == min(r['blocks'] for r in feasible)


def test_correctness_capture_does_not_change_the_selection_accuracies(monkeypatch):
    """Review focus 4: the objective's acc_app/acc_ddos must be the same floats
    accuracy_score gives on the same predictions -- balanced must reproduce the
    pre-2026-10-04 search exactly."""
    from sklearn.metrics import accuracy_score
    from p4gen.switch_semantics import switch_predict
    out = _call(cfg=TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3,
                                selection_rule='balanced'))
    X_app, y_app, X_ddos, y_ddos = _tiny_problem()
    assert out.acc_sel_A == accuracy_score(y_app, switch_predict(out.model_A, X_app))
    assert out.acc_sel_B == accuracy_score(y_ddos, switch_predict(out.model_B, X_ddos))


def test_the_reference_refit_adds_exactly_one_evaluation(monkeypatch):
    """Spec II.6: the reference refit is held to the same determinism assert as
    the shipped one, so it costs one extra evaluation when it differs."""
    import src.training.train_model as tm
    real = tm.multi_model_memory_evaluation
    calls = {'n': 0}

    def counting(*a, **k):
        # counted before the call: a refused trial (a stump, R6) raises from it
        calls['n'] += 1
        return real(*a, **k)

    monkeypatch.setattr(tm, 'multi_model_memory_evaluation', counting)
    out = _call(cfg=TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3,
                                select_alpha=0.0))
    # one evaluation per COMPLETED trial plus one refit, plus one more when the
    # reference differs from the shipped trial
    extra = 2 if out.ref_trial != out.chosen_trial else 1
    assert calls['n'] == out.n_trials_run + extra


# Prerequisites for the per-task objective rerun (spec risk 5). (merged from test_train_model_prereqs.py)

def test_importing_train_model_does_not_patch_sklearn_by_default():
    """sklearnex.patch_sklearn() replaces sklearn.ensemble.RandomForestClassifier
    with Intel's oneDAL implementation process-wide, at import time. In the
    PolimiML env that implementation cannot round-trip trees through
    scikit-learn 1.6.1's Tree.__setstate__, so every .fit() raises. It also may
    not observe the in-place tree_.threshold mutations that
    dt_thresholds_float_to_int and align_with_policy depend on. Default off."""
    import src.training.train_model  # noqa: F401  -- the import IS the action under test
    from sklearn.ensemble import RandomForestClassifier

    assert 'sklearnex' not in RandomForestClassifier.__module__
    assert 'daal' not in RandomForestClassifier.__module__


def test_random_forest_fit_works_after_importing_train_model():
    """The end the guard exists for: a plain fit must survive the import."""
    import src.training.train_model  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.integers(0, 100, size=(60, 3)).astype(float)
    y = np.array([0, 1, 2] * 20)

    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(X, y)

    assert model.predict(X).shape == (60,)
