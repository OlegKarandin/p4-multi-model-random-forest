"""F6: the early-stopping callback never fired, so every search ran 1000 trials."""
import optuna

from src.training import early_stopping

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def _study():
    return optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=optuna.samplers.TPESampler(seed=0))


def _converged_objective(trial):
    """Every trial lands on the SAME single Pareto point: a converged search.
    The front is stable from trial 0, so a correct stopper must stop."""
    trial.suggest_int('x', 0, 100)
    trial.set_user_attr('codeword_violation', 0.0)
    trial.set_user_attr('blocks_violation', 0.0)
    trial.set_user_attr('crossbar_violation', 0.0)
    return 0.9, 10.0


def test_stopper_terminates_on_a_converged_front():
    stopper = early_stopping.ParetoStagnationStopper(min_feasible=25, lookback=20)
    study = _study()

    study.optimize(_converged_objective, n_trials=500, callbacks=[stopper])

    # 25 feasible trials are needed before checking begins (the 25th is trial
    # number 24), then 20 trials of no movement -> stop during trial 44.
    assert 45 <= len(study.trials) <= 46


def test_stopper_keeps_searching_while_the_front_still_moves():
    """Each trial strictly dominates all previous ones, so the front is a
    single point that MOVES every trial. Stopping here would be wrong."""
    stopper = early_stopping.ParetoStagnationStopper(min_feasible=5, lookback=10)
    study = _study()

    def objective(trial):
        trial.suggest_int('x', 0, 100)
        trial.set_user_attr('codeword_violation', 0.0)
        trial.set_user_attr('blocks_violation', 0.0)
        trial.set_user_attr('crossbar_violation', 0.0)
        return 0.5 + trial.number * 1e-4, 100.0 - trial.number

    study.optimize(objective, n_trials=40, callbacks=[stopper])

    assert len(study.trials) == 40


def test_infeasible_trials_never_satisfy_the_feasible_minimum():
    """At tight max_blocks most of the search is infeasible. Those trials must
    not count toward min_feasible, or the search stops before finding anything
    it is allowed to return."""
    stopper = early_stopping.ParetoStagnationStopper(min_feasible=3, lookback=1)
    study = _study()

    def objective(trial):
        trial.suggest_int('x', 0, 100)
        trial.set_user_attr('codeword_violation', 0.0)
        trial.set_user_attr('blocks_violation', 5.0)
        trial.set_user_attr('crossbar_violation', 0.0)
        return -1.0, 1e9

    study.optimize(objective, n_trials=20, callbacks=[stopper])

    assert len(study.trials) == 20


def test_a_trial_missing_its_violation_attrs_reads_as_infeasible():
    """The objective sets the attrs only once it reaches that point. A trial
    that died earlier must not be treated as feasible-by-default."""
    study = _study()
    study.add_trial(optuna.trial.create_trial(
        params={'x': 1},
        distributions={'x': optuna.distributions.IntDistribution(0, 100)},
        values=[0.9, 10.0]))
    trial = study.trials[0]

    assert early_stopping.constraint_values(trial) == [float('inf')] * 3
    assert early_stopping.is_feasible(trial) is False
