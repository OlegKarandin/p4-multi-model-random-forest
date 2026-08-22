"""Optuna early stopping that actually terminates (finding F6).

The original callback (train_model.py, pre-P1) asked whether ANY trial on the
feasible Pareto front had a trial number within the last `lookback` trials.
`study.best_trials` returns every non-dominated trial, and trials that tie on
the objective vector are all non-dominated -- a measured run had 103
best_trials covering 13 distinct Pareto points. With ~26% of trials on the
front, the probability that none of the last 20 is on it is 0.74**20 < 0.3%,
so the callback effectively never fired and every search ran all 1000 trials.

This version keys on the set of DISTINCT objective vectors on the feasible
front, which is what "the front stopped moving" means, and recomputes it only
when a feasible trial has just completed -- the front cannot change otherwise.
That also removes the measured callback overhead (200 trials: 9.9 s with the
old callback, 3.3 s without), which came from rescanning study.trials and
re-deriving constraints on every trial.
"""
import optuna

# The objective records violation MAGNITUDES rather than booleans so Optuna's
# constrained TPE sampler can order infeasible trials by how badly they miss.
#
# crossbar_violation (Task 10, F3): a table whose key exceeds the per-stage
# crossbar's byte budget is a third, distinct failure from a too-long
# codeword or an over-block-budget model. Without it here, a trial that hit
# CrossbarKeyTooWide -- which sets codeword_violation and blocks_violation to
# 0.0 precisely because it isn't either of those -- would read as feasible
# (both tracked magnitudes <= 0) despite never having set acc_app/acc_ddos/
# blocks/stages, crashing trial_selection.select_best_trial with a KeyError
# instead of correctly excluding the trial.
#
# stages_violation (Task 13, F5): the hard 12-stage Tofino-1 pipeline-depth
# ceiling is a fourth, distinct failure -- a trial can be within both the
# codeword and block budgets and still predict a pipeline deeper than
# TOFINO_PIPELINE_STAGES. Without it here, such a trial would read as
# feasible despite objective() having returned the infeasible triple
# (-1.0, -1.0, inf) for it, letting trial_selection.select_best_trial pick a
# trial whose acc_app/acc_ddos are the -1.0 sentinel rather than a real score.
_VIOLATION_ATTRS = ('codeword_violation', 'blocks_violation', 'crossbar_violation',
                    'stages_violation')


def constraint_values(trial):
    """The violation magnitudes for `trial`; every element <= 0 means feasible.

    A missing attribute means the objective never reached the point where it
    sets them -- the trial failed, was pruned, or is still running. That must
    read as infeasible (inf), never as feasible-by-default.
    """
    return [trial.user_attrs.get(name, float('inf')) for name in _VIOLATION_ATTRS]


def is_feasible(trial):
    """True when `trial` completed and satisfies every constraint."""
    return (trial.state == optuna.trial.TrialState.COMPLETE
            and all(value <= 0 for value in constraint_values(trial)))


class ParetoStagnationStopper:
    """Stop once the feasible Pareto front has neither gained nor lost a
    distinct objective vector for `lookback` trials, and at least
    `min_feasible` feasible trials exist.

    Stateful across calls by design: the feasible count is accumulated
    incrementally instead of rescanning the study, and the front is re-derived
    only on trials that could have changed it. One instance per study.
    """

    def __init__(self, min_feasible=25, lookback=20):
        self.min_feasible = min_feasible
        self.lookback = lookback
        self._n_feasible = 0
        self._front = frozenset()
        self._last_change = None

    def __call__(self, study, trial):
        if not is_feasible(trial):
            return
        self._n_feasible += 1
        if self._n_feasible < self.min_feasible:
            return

        front = frozenset(tuple(t.values) for t in study.best_trials if is_feasible(t))
        if not front:
            return

        if front != self._front:
            self._front = front
            self._last_change = trial.number
            return

        if trial.number - self._last_change >= self.lookback:
            study.stop()
