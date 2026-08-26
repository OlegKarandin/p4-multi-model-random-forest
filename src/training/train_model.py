import os

# Intel's oneDAL acceleration is OPT-IN and off by default. Two reasons, both
# fatal to this pipeline when it is on:
#   1. In the PolimiML env its RandomForestClassifier cannot round-trip trees
#      through scikit-learn 1.6.1's Tree.__setstate__, so every .fit() raises
#      "node array from the pickle has an incompatible dtype".
#   2. This project mutates tree_.threshold IN PLACE after fitting
#      (dt_thresholds_float_to_int, align_rf_thresholds). oneDAL keeps its own
#      internal model representation, so those mutations may not be observed --
#      which would silently invalidate every result rather than crash.
# Set THESIS_USE_SKLEARNEX=1 only after verifying both of the above.
if os.environ.get('THESIS_USE_SKLEARNEX') == '1':
  try:
    from sklearnex import patch_sklearn
    patch_sklearn()
  except ImportError:
    pass

from dataclasses import dataclass
from typing import Any, Optional

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

from src.p4gen.build_p4_script import (
    dt_thresholds_float_to_int, MAX_CODEWORD_LENGTH,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
from src.p4gen.evaluation import (
    CodewordTooLong, CrossbarKeyTooWide, TOFINO_PIPELINE_STAGES,
    multi_model_memory_evaluation)
from src.p4gen.switch_semantics import switch_predict
from src.training.threshold_alignment import align_rf_thresholds
from src.training import early_stopping
from src.training import trial_selection

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def _vary_hyperparams(params: dict, n_trees: int, max_depth: int) -> dict:
    """
    Create a variation of hyperparameters for warm-start diversity.

    Randomly varies some parameters by small amounts to explore
    the neighborhood of the previous best solution.
    """
    varied = params.copy()

    for key in varied:
        if 'n_estimators' in key:
            delta = np.random.choice([-2, 0, 2])
            varied[key] = max(1, min(n_trees, varied[key] + delta))
        elif 'max_depth' in key:
            delta = np.random.choice([-1, 0, 1])
            varied[key] = max(2, min(max_depth, varied[key] + delta))
        elif 'min_samples_leaf' in key:
            delta = np.random.choice([-20, -10, 0, 10, 20])
            varied[key] = max(5, min(200, varied[key] + delta))
        elif 'min_samples_split' in key:
            delta = np.random.choice([-20, -10, 0, 10, 20])
            varied[key] = max(10, min(400, varied[key] + delta))

    return varied


@dataclass(frozen=True)
class TrainResult:
    """Return contract for train_multi_RF_Optuna_multi_constrained (P5 gap 2).

    Replaces a 7-tuple rather than growing it to twelve-plus positional
    values: a dataclass makes every caller name the field it reads instead of
    counting positions.

    model_A, model_B, stages, blocks, acc_sel_A, acc_sel_B, best_params are
    the original seven. The rest:

    stage_depth : StagePlan.depth (F5/F6) -- the pipeline-DEPTH quantity the
        hard TOFINO_PIPELINE_STAGES ceiling is compared against, from the
        refit's own `ResourceUsage` result from `multi_model_memory_evaluation`.
        NOT the same quantity as `stages` (occupied match-table stage COUNT) --
        see evaluation.multi_model_memory_evaluation's docstring for the
        full three-quantity disambiguation.
    rel_shortfall : the winning trial's balance shortfall from
        trial_selection.select_best_trial (spec B.3) -- how far the
        worse-served task fell below its own achievable best.
    n_trials_run : len(study.trials) -- how many trials the search actually
        ran before stopping (early stopping may cut this short of cfg.n_trials).
    n_feasible : how many of those trials satisfied both the codeword and
        block-budget constraints.
    align_attempted, align_accepted, intervals_before, intervals_after :
        the four align_stats keys from threshold_alignment.align_rf_thresholds,
        measured on the REFIT winner (gap 1), not an earlier trial. Alignment
        is a joint-arm-only treatment (fit_pair calls align_rf_thresholds only
        when encoding == 'joint' and cfg.alignment_enabled) -- on every other
        arm these four fields are None, deliberately NOT 0: a silent 0 would
        be indistinguishable from "alignment ran and accepted nothing", which
        is a real and different outcome. Task 8 writes '' into the CSV for
        the None case.
    range_entries, ternary_entries : the same-named fields off the refit's
        own `ResourceUsage` (evaluation.multi_model_memory_evaluation) --
        physical TCAM row counts for the range-matching and ternary-matching
        tables respectively. Distinct from `blocks` (the block count those
        rows pack into) and from `stages`/`stage_depth` (pipeline stage
        quantities, not row counts) -- see ResourceUsage's own docstring for
        the full disambiguation.
    register_depth, register_count, register_sram_bits : the same-named
        fields off that same `ResourceUsage`, reporting the Tofino
        `Register<>` state the design needs (Task 6, Spec 4.1/4.2/4.3).
        register_depth is a STAGE count (readiness-level depth of the
        register dependency chain) and must not be confused with
        `stage_depth` above, which is the whole pipeline's depth including
        the match tables that read those registers. See ResourceUsage's own
        docstring for the capacity caveat: these report depth/count, not
        whether the registers fit.

    Frozen so a ProcessPoolExecutor worker cannot mutate a result after it
    crosses the pickle boundary. Every field is plain data (numbers, a dict,
    or the two fitted models already returned pre-P5) so the whole thing
    pickles the same as the tuple it replaces.
    """
    model_A: Any
    model_B: Any
    stages: int
    blocks: int
    stage_depth: int
    acc_sel_A: float
    acc_sel_B: float
    best_params: dict
    rel_shortfall: float
    n_trials_run: int
    n_feasible: int
    align_attempted: Optional[int]
    align_accepted: Optional[int]
    intervals_before: Optional[int]
    intervals_after: Optional[int]
    range_entries: int
    ternary_entries: int
    register_depth: int
    register_count: int
    register_sram_bits: int


def rf_params_from_params(params, suffix):
    """RandomForestClassifier kwargs from a recorded best_params dict.

    Module level, not a closure, because scripts/replay_alignment.py refits
    campaign winners from their CSV-recorded best_params and must build exactly
    the estimator the campaign built. random_state is fixed at 42 here: that
    determinism is what the refit assertion below and the whole replay design
    depend on.
    """
    return {
        'n_estimators': params['n_estimators_' + suffix],
        'min_samples_leaf': params['min_samples_leaf_' + suffix],
        'min_samples_split': params['min_samples_split_' + suffix],
        'max_depth': params['max_depth_' + suffix],
        'random_state': 42,
    }


def train_multi_RF_Optuna_multi_constrained(
        X_A, y_A, X_B, y_B,
        val_align_A, val_align_B,
        val_select_A, val_select_B,
        features_A, features_B,
        max_blocks, encoding, cfg,
        warm_start_params=None):
    """Search hyperparameters for both tasks under a shared block budget.

    Spec B.1/B.2. One fit per trial on the FULL training set: that single model
    is both what the constraint is measured on and what is returned, so
    `blocks <= max_blocks` holds for the shipped artifact rather than for a
    CV-fold proxy (F7). Both the objective and the final selection are
    per-task (F11 sites 1-3).

    val_align_*, val_select_* : (X, y) tuples. val_align serves alignment and
        nothing else; val_select serves this objective and, one level up,
        permutation_importance. Disjoint by construction (splits.py).

    Returns a TrainResult (see its docstring for the full field list).

    Raises NoFeasibleSolution when no trial satisfies both the block budget and
    the codeword limit -- an expected outcome at tight max_blocks, handled per-k
    by the caller.
    """

    def rf_params(source, suffix):
        """Both the objective and the final refit build params the same way, so
        the two cannot drift. `source` is a trial (during the search) or a plain
        params dict (when refitting the winner)."""
        if hasattr(source, 'suggest_int'):
            return {
                'n_estimators': source.suggest_int('n_estimators_' + suffix, 1, cfg.n_trees, step=2),
                'min_samples_leaf': source.suggest_int('min_samples_leaf_' + suffix, 5, 200, step=10),
                'min_samples_split': source.suggest_int('min_samples_split_' + suffix, 10, 400, step=10),
                'max_depth': source.suggest_int('max_depth_' + suffix, 2, cfg.max_depth),
                'random_state': 42,
            }
        return rf_params_from_params(source, suffix)

    def fit_pair(params_A, params_B, align_stats=None):
        """ONE fit per task on the FULL training set -- this IS the deployment
        model, not a CV fold (F7).

        n_jobs=1: the parallelism is across splits (11 workers on 12 logical
        cores), so -1 here oversubscribes.

        Fully deterministic given (params, data): random_state is fixed above,
        and align_rf_thresholds is a deterministic function of the two models
        and val_align. That determinism is what lets the winner be REFIT below
        instead of cached (F8) -- a measured 401 KB per model pair, ~100 pairs
        per search, per worker.
        """
        with sklearn.config_context(assume_finite=True):
            model_A = RandomForestClassifier(**params_A, n_jobs=1).fit(X_A, y_A)
            model_B = RandomForestClassifier(**params_B, n_jobs=1).fit(X_B, y_B)

        model_A = dt_thresholds_float_to_int(model_A)
        model_B = dt_thresholds_float_to_int(model_B)

        if encoding == 'joint' and cfg.alignment_enabled:
            model_A, model_B = align_rf_thresholds(
                model_A, model_B,
                val_align_A[0], val_align_A[1],
                val_align_B[0], val_align_B[1],
                overlap_threshold=cfg.overlap_threshold,
                delta_rel=cfg.delta_align,
                align_stats=align_stats)

        return model_A, model_B

    def objective(trial):
        # (a) The single fit.
        align_stats = {}
        model_A, model_B = fit_pair(rf_params(trial, 'A'), rf_params(trial, 'B'),
                                   align_stats=align_stats)

        # (b) Constraint, on the shipped artifact -- exact, no proxy.
        try:
            usage = multi_model_memory_evaluation(
                model_A, model_B, features_A, features_B, encoding)
        except CodewordTooLong as e:
            trial.set_user_attr('codeword_violation', e.args[1] - MAX_CODEWORD_LENGTH)
            trial.set_user_attr('blocks_violation', 0.0)
            trial.set_user_attr('crossbar_violation', 0.0)
            trial.set_user_attr('stages_violation', 0.0)
            return -1.0, -1.0, float('inf')
        except CrossbarKeyTooWide as e:
            trial.set_user_attr('crossbar_violation',
                                e.args[1] - TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
            trial.set_user_attr('codeword_violation', 0.0)
            trial.set_user_attr('blocks_violation', 0.0)
            trial.set_user_attr('stages_violation', 0.0)
            return -1.0, -1.0, float('inf')

        if usage.blocks > max_blocks:
            trial.set_user_attr('codeword_violation', 0.0)
            trial.set_user_attr('blocks_violation', usage.blocks - max_blocks)
            trial.set_user_attr('crossbar_violation', 0.0)
            trial.set_user_attr('stages_violation', 0.0)
            return -1.0, -1.0, float('inf')

        # F5: hard 12-ingress-stage Tofino-1 pipeline ceiling
        # (TOFINO_PIPELINE_STAGES), compared against stage_depth -- NOT
        # `stages` (see evaluation.multi_model_memory_evaluation's docstring
        # for why those are different quantities).
        #
        # NECESSARY BUT NOT SUFFICIENT: stage_depth is this model's predicted
        # pipeline depth from placed match tables alone (StagePlan.depth);
        # the real compiler also spends stages on parsing/bookkeeping
        # overhead this model does not capture -- measured on M2, this model
        # predicts depth 6 where the real compiler needs 9 stages
        # (`stages_real`, p4_compile.parse_compile_logs). A violation here is
        # therefore a DEFINITE reject (the real pipeline can only need MORE
        # stages than predicted, never fewer); passing this check is NOT a
        # guarantee the real compile fits within TOFINO_PIPELINE_STAGES.
        if usage.stage_depth > TOFINO_PIPELINE_STAGES:
            trial.set_user_attr('codeword_violation', 0.0)
            trial.set_user_attr('blocks_violation', 0.0)
            trial.set_user_attr('crossbar_violation', 0.0)
            trial.set_user_attr('stages_violation', usage.stage_depth - TOFINO_PIPELINE_STAGES)
            return -1.0, -1.0, float('inf')

        # (c) Only FEASIBLE trials pay for scoring. At tight max_blocks most of
        # the search is infeasible, so this ordering preserves the early bail.
        #
        # switch_predict, NOT model.score: score() averages predict_proba (a
        # SOFT vote) while the generated switch votes hard over per-tree
        # classes. Optimising the soft number ranks trials by an accuracy the
        # switch never reaches -- measured at up to 1.7 points, and up to +55%
        # relative error on DDoS, wider than the entire delta_align grid. Worse,
        # the gap widens with min_samples_leaf, so the soft objective pushes the
        # search toward exactly the configurations where its own number is least
        # honest. See P1 Task 7.
        with sklearn.config_context(assume_finite=True):
            acc_A = accuracy_score(val_select_A[1], switch_predict(model_A, val_select_A[0]))
            acc_B = accuracy_score(val_select_B[1], switch_predict(model_B, val_select_B[0]))

        trial.set_user_attr('acc_app', acc_A)
        trial.set_user_attr('acc_ddos', acc_B)
        trial.set_user_attr('blocks', int(usage.blocks))
        trial.set_user_attr('stages', int(usage.stages))
        trial.set_user_attr('codeword_violation', 0.0)
        trial.set_user_attr('blocks_violation', 0.0)
        trial.set_user_attr('crossbar_violation', 0.0)
        trial.set_user_attr('stages_violation', 0.0)

        # Blocks stays a third OBJECTIVE rather than a pure constraint because
        # plots use REALIZED blocks as an axis: without minimize-blocks pressure
        # the realized values pile up near max_blocks and that axis loses all
        # resolution.
        return acc_A, acc_B, float(usage.blocks)

    sampler = TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
        warn_independent_sampling=True,
        constant_liar=True,
        constraints_func=early_stopping.constraint_values,
    )

    study = optuna.create_study(
        directions=['maximize', 'maximize', 'minimize'],
        sampler=sampler)

    if warm_start_params is not None:
        try:
            study.enqueue_trial(warm_start_params)
            for _ in range(2):
                study.enqueue_trial(
                    _vary_hyperparams(warm_start_params, cfg.n_trees, cfg.max_depth))
        except Exception:
            pass  # Ignore if enqueue fails (e.g. bounds changed between k)

    # catch=() deliberately: the ONLY expected RuntimeErrors are the codeword
    # (CodewordTooLong) and crossbar-width (CrossbarKeyTooWide) violations,
    # and the objective already handles both itself. The previous
    # catch=(RuntimeError,) also swallowed align_rf_thresholds' own
    # AlignmentInvariantError (raised as a bare RuntimeError("Smth is
    # very-very wrong") before it was given its own type), hiding corrupted
    # interval bookkeeping for a whole campaign. AlignmentInvariantError does
    # not subclass RuntimeError, so catch=(RuntimeError,) would no longer
    # swallow it either way -- but catch=() stays the deliberate choice, since
    # F3a/F3b make a dead split survivable, so surfacing the error costs one
    # split's tail rather than a silent lie.
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        callbacks=[early_stopping.ParetoStagnationStopper(
            min_feasible=cfg.min_feasible_before_stop, lookback=cfg.lookback)],
        catch=())

    feasible_trials = [t for t in study.trials if early_stopping.is_feasible(t)]

    best_trial, shortfall = trial_selection.select_best_trial(
        feasible_trials, cfg.delta_select, k=len(features_A), max_blocks=max_blocks)

    # Refit the winner rather than caching every feasible trial's model pair
    # (F8: a measured 401 KB per pair, ~100 pairs per search, 11 workers). One
    # extra fit-pair, ~550 ms, against ~40 MB of live cache per worker.
    #
    # Re-measuring here is not redundant: it is the B.1 invariant checked on the
    # artifact that is actually returned, and it fails loudly if anything in the
    # fit -> align -> measure path is non-deterministic.
    #
    # align_stats (P5 gap 1): a fresh dict per refit, same as the objective
    # passes per trial (a), so the four align_* fields on TrainResult describe
    # THIS artifact rather than whichever earlier trial happened to be the
    # winner. Stays empty (all four .get()s below fall through to None) on any
    # arm where fit_pair does not call align_rf_thresholds at all.
    align_stats = {}
    model_A, model_B = fit_pair(rf_params(best_trial.params, 'A'),
                                rf_params(best_trial.params, 'B'),
                                align_stats=align_stats)
    usage = multi_model_memory_evaluation(
        model_A, model_B, features_A, features_B, encoding)

    # C4 cross-check: align_rf_thresholds counts intervals in the models'
    # COLUMN-INDEX space; multi_model_memory_evaluation counts over the union
    # of the two models' selected feature NAMES. They are the same number only
    # while both models are fit on the same column space -- true here (both
    # tasks eliminate features together on the joint arm) and required by
    # align_rf_thresholds' own contract, but a silent divergence would make
    # every band decision wrong. Guarded on the key's presence so the arms
    # where alignment never runs are unaffected.
    if 'codeword_after' in align_stats:
        assert align_stats['codeword_after'] == usage.codeword_length, (
            'alignment measured codeword {} but the generator measured {} -- '
            'the two feature spaces have diverged'.format(
                align_stats['codeword_after'], usage.codeword_length))

    if usage.blocks != best_trial.user_attrs['blocks'] or usage.blocks > max_blocks:
        raise AssertionError(
            'refit of trial {} gave {} blocks, the search recorded {} '
            '(max {}) -- the pipeline is not deterministic'.format(
                best_trial.number, usage.blocks, best_trial.user_attrs['blocks'], max_blocks))

    # F5: the winning trial already passed the stage_depth <= TOFINO_PIPELINE_STAGES
    # ceiling check inside objective() (a violation there returns the
    # infeasible triple, which trial_selection.select_best_trial never
    # selects), so the refit -- deterministic given the same params -- must
    # reproduce that too.
    if usage.stage_depth > TOFINO_PIPELINE_STAGES:
        raise AssertionError(
            'refit of trial {} gave stage_depth {} exceeding the {}-stage '
            'ceiling -- the pipeline is not deterministic'.format(
                best_trial.number, usage.stage_depth, TOFINO_PIPELINE_STAGES))

    return TrainResult(
        model_A=model_A,
        model_B=model_B,
        stages=int(usage.stages),
        blocks=int(usage.blocks),
        stage_depth=int(usage.stage_depth),
        acc_sel_A=best_trial.user_attrs['acc_app'],
        acc_sel_B=best_trial.user_attrs['acc_ddos'],
        best_params=dict(best_trial.params),
        rel_shortfall=shortfall,
        n_trials_run=len(study.trials),
        n_feasible=len(feasible_trials),
        align_attempted=align_stats.get('attempted'),
        align_accepted=align_stats.get('accepted'),
        intervals_before=align_stats.get('intervals_before'),
        intervals_after=align_stats.get('intervals_after'),
        range_entries=int(usage.range_entries),
        ternary_entries=int(usage.ternary_entries),
        register_depth=int(usage.register_depth),
        register_count=int(usage.register_count),
        register_sram_bits=int(usage.register_sram_bits),
    )
