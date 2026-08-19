"""Per-task final trial selection (F11 site 3, spec B.3).

The old rule averaged the two tasks' accuracies and took the cheapest trial
within 0.0025 of the best average. Because 0.0025 of an average is up to
0.005 on a single task -- and 0.005 is 2.3% of App's error but 12.5% of
DDoS's -- the rule systematically bought blocks out of DDoS and the average
hid it. Spec B.3's worked example: today's rule picks a trial giving away
22.5% of DDoS's error to save 6 blocks.

The replacement: find the most balanced trial available, widen the band by
delta_select, take the cheapest trial in that band. Balance is the WORSE
task's shortfall from its own best, as a fraction of that task's own error.
"""
from src.training.errors import NoFeasibleSolution


def rel_deg(before, after):
    """Degradation from `before` to `after` as a fraction of `before`'s error.

    Scale-matched across tasks: a 0.005 drop is 2.3% of App's 0.22 error and
    12.5% of DDoS's 0.04 error, which is exactly the asymmetry averaging hides.
    The max() guards a perfect `before` score, where the error is 0.
    """
    return (before - after) / max(1e-9, 1.0 - before)


def rel_shortfall(acc_app, acc_ddos, best_app, best_ddos):
    """How far the WORSE-served task falls below its own achievable best.

    max, not mean: the whole point is that a loss on one task cannot be
    masked by a gain on the other.
    """
    return max(rel_deg(best_app, acc_app), rel_deg(best_ddos, acc_ddos))


def select_best_trial(feasible_trials, delta_select, k, max_blocks):
    """The cheapest trial whose imbalance is within delta_select of the floor.

    Returns (trial, its rel_shortfall). Raises NoFeasibleSolution when there is
    nothing to choose from.

    `best_app` and `best_ddos` usually come from DIFFERENT trials, so they
    define an ideal corner that is typically not on the Pareto front at all.
    `floor +` is therefore required rather than a bare delta_select band: the
    trial attaining `floor` is always a member, so the band is never empty --
    including at delta_select = 0.

    Ties on blocks are broken by trial number so two workers on the same cell
    return the same model.
    """
    if not feasible_trials:
        raise NoFeasibleSolution(k=k, max_blocks=max_blocks)

    best_app = max(t.user_attrs['acc_app'] for t in feasible_trials)
    best_ddos = max(t.user_attrs['acc_ddos'] for t in feasible_trials)

    shortfalls = {
        t.number: rel_shortfall(
            t.user_attrs['acc_app'], t.user_attrs['acc_ddos'], best_app, best_ddos)
        for t in feasible_trials
    }
    floor = min(shortfalls.values())

    close = [t for t in feasible_trials
             if shortfalls[t.number] <= floor + delta_select]

    chosen = min(close, key=lambda t: (t.user_attrs['blocks'], t.number))
    return chosen, shortfalls[chosen.number]
