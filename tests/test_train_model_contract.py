"""Spec B.1/B.2: one fit, per-task objectives, constraint on what is returned."""
import dataclasses

import numpy as np
import pytest

from src.training.config import TrainConfig
from src.training.errors import NoFeasibleSolution
from src.training.train_model import TrainResult, train_multi_RF_Optuna_multi_constrained

FEATURE_NAMES = ['Flow.IAT.Max', 'Fwd.IAT.Max', 'Fwd.Packet.Length.Max']


def _tiny_problem(n=300, seed=0):
    """Small enough for a real Optuna search in a test, structured enough that
    a shallow forest beats chance."""
    rng = np.random.default_rng(seed)
    X_app = rng.integers(0, 400, size=(n, 3)).astype(float)
    y_app = (X_app[:, 0] // 134).astype(int)
    X_ddos = rng.integers(0, 400, size=(n, 3)).astype(float)
    y_ddos = np.where(X_ddos[:, 1] > 200, 1, -1)
    return X_app, y_app, X_ddos, y_ddos


def _call(encoding='disjoint', cfg=None, max_blocks=60, n=300):
    X_app, y_app, X_ddos, y_ddos = _tiny_problem(n=n)
    cfg = cfg or TrainConfig(n_trials=12, min_feasible_before_stop=4, lookback=3)
    return train_multi_RF_Optuna_multi_constrained(
        X_app, y_app, X_ddos, y_ddos,
        (X_app, y_app), (X_ddos, y_ddos),
        (X_app, y_app), (X_ddos, y_ddos),
        FEATURE_NAMES, FEATURE_NAMES,
        max_blocks, encoding, cfg)


def test_the_contract_returns_a_frozen_train_result_with_all_twenty_fields():
    out = _call()

    assert isinstance(out, TrainResult)
    assert len(dataclasses.fields(out)) == 20
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
    from src.p4gen.evaluation import multi_model_memory_evaluation

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
    """Spec A.2: the ablation arm is a genuine SKIP of the call, not
    delta_align = 0, so the arm is provably prediction-identical to the
    unaligned models."""
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


def test_the_joint_arm_aligns_through_the_rollback_wrapper(monkeypatch):
    """Design 2026-08-30 §2.7: the campaign called align_rf_thresholds
    directly, so no campaign ever had C1's commit-or-rollback. Routing through
    align_with_policy is what makes align_objective expressible at campaign
    quality at all -- without it a rerun bakes 'blocks' into the new archive
    and needs a second rerun to compare objectives."""
    import src.training.train_model as tm

    calls = []

    def spy(rf1, rf2, *args, **kwargs):
        calls.append(kwargs)
        return rf1, rf2

    monkeypatch.setattr(tm, 'align_with_policy', spy)

    _call(encoding='joint',
          cfg=TrainConfig(delta_align=0.05, overlap_threshold=0.4,
                          align_objective='stages', n_trials=6,
                          min_feasible_before_stop=2, lookback=2))

    assert calls, 'alignment should have run in the joint arm'
    assert all(c['align_objective'] == 'stages' for c in calls)
    assert all(c['delta_rel'] == 0.05 for c in calls)
    assert all(c['overlap_threshold'] == 0.4 for c in calls)


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
        for attr in ('acc_app', 'acc_ddos', 'blocks', 'stages',
                     'codeword_violation', 'blocks_violation', 'crossbar_violation',
                     'stages_violation'):
            assert attr in trial.user_attrs, (trial.number, attr)


def test_crossbar_key_too_wide_records_crossbar_violation_not_codeword(monkeypatch):
    """Task 10 (F3): a table whose key exceeds the stage crossbar is a
    distinct failure from a too-long codeword -- computing
    `e.args[1] - MAX_CODEWORD_LENGTH` on it would be a nonsense violation
    number. The objective must instead record `crossbar_violation`, leave
    the other two violation attrs at 0.0, and return the infeasible triple
    rather than crash the search."""
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
    reported_width = 100
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: (_ for _ in ()).throw(
            tm.CrossbarKeyTooWide('table key too wide', reported_width)))

    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        assert trial.user_attrs['crossbar_violation'] == (
            reported_width - tm.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
        assert trial.user_attrs['codeword_violation'] == 0.0
        assert trial.user_attrs['blocks_violation'] == 0.0
        assert trial.user_attrs['stages_violation'] == 0.0
        assert trial.values == [-1.0, -1.0, float('inf')]


def test_stage_depth_over_the_tofino_ceiling_records_stages_violation_not_the_others(monkeypatch):
    """F5: a trial whose predicted pipeline depth exceeds the hard 12-stage
    Tofino-1 ceiling (TOFINO_PIPELINE_STAGES) must be rejected the same way
    the codeword/blocks/crossbar constraints are -- infeasible triple
    returned, only `stages_violation` carrying a nonzero magnitude -- even
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

    import src.p4gen.evaluation as ev

    monkeypatch.setattr(tm.optuna, 'create_study', capture)
    over_ceiling_depth = tm.TOFINO_PIPELINE_STAGES + 3
    monkeypatch.setattr(
        tm, 'multi_model_memory_evaluation',
        lambda *a, **k: ev.ResourceUsage(
            stages=1, blocks=1, stage_depth=over_ceiling_depth,
            range_entries=1, ternary_entries=1, codeword_length=40,
            register_depth=1, register_count=0, register_sram_bits=0,
            range_depth=1, ternary_depth=over_ceiling_depth,
            range_tables=1, ternary_tables=1))

    # Every trial is over-ceiling, so nothing is feasible -- same shape as
    # test_crossbar_key_too_wide_records_crossbar_violation_not_codeword.
    with pytest.raises(NoFeasibleSolution):
        _call(cfg=TrainConfig(n_trials=5, min_feasible_before_stop=2, lookback=2))

    trials = captured['study'].trials
    assert trials, 'the search should have run at least one trial'
    for trial in trials:
        assert trial.user_attrs['stages_violation'] == (
            over_ceiling_depth - tm.TOFINO_PIPELINE_STAGES)
        assert trial.user_attrs['codeword_violation'] == 0.0
        assert trial.user_attrs['blocks_violation'] == 0.0
        assert trial.user_attrs['crossbar_violation'] == 0.0
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
    out = _call()

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


def test_delta_align_none_still_builds_the_oracle_now_that_gating_is_unconditional(monkeypatch):
    """Spec A.2's inf-anchor claim -- that delta_rel=None skips the
    predict/restore/undo machinery entirely, making it the cheapest arm --
    no longer holds after the 2026-08-30 policy-ladder deletion: C1's
    non-spending state judges candidates at delta=0, which needs the oracle
    even when the caller asked for delta_rel=None, and that gating is now
    unconditional (see threshold_alignment.align_rf_thresholds and
    test_the_oracle_is_built_even_at_an_unbounded_delta in
    test_threshold_alignment.py). So the inf arm no longer skips scoring at
    this level either -- the number moved because target ranking and budget
    gating are now unconditional."""
    from src.training import threshold_alignment as ta

    scored = []
    real_init = ta.IncrementalMetrics.__init__

    def spy(self, tree_predictions, rf, y_true, task):
        scored.append(task)
        return real_init(self, tree_predictions, rf, y_true, task)

    monkeypatch.setattr(ta.IncrementalMetrics, '__init__', spy)

    _call(encoding='joint',
          cfg=TrainConfig(delta_align=None, n_trials=6,
                          min_feasible_before_stop=2, lookback=2))

    assert scored, 'delta_align=None must still build the oracle now that gating is unconditional'
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

    out = _call(encoding='joint',
                cfg=TrainConfig(n_trials=6, min_feasible_before_stop=2, lookback=2))

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
    alignment_enabled=False is a genuine skip of align_rf_thresholds, not
    delta_align=0, so it must be just as distinguishable from a real zero."""
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
    import src.p4gen.evaluation as ev

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
            register_depth=1, register_count=0, register_sram_bits=0,
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
        'max_depth': 9, 'random_state': 42}
    assert tm.rf_params_from_params(params, 'B')['n_estimators'] == 3
