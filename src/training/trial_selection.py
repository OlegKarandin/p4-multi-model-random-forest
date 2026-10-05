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

TIE-AWARE SELECTION (2026-10-04, spec Part II): select_best_trial's band
holds ~1.2 trials -- narrower than val_select's own noise -- so in effect it
ships the single best-balanced trial, and the search reinvests freed blocks in
larger forests whose extra validation accuracy is noise. select_tied_cheapest
keeps that pick as the reference R and ships the cheapest feasible trial that
is not significantly worse than R on EITHER task (one-sided exact McNemar on
per-flow right/wrong marks).
"""
import json

import numpy as np
from scipy.stats import binomtest

from src.training.errors import NoFeasibleSolution

TASKS = ('app', 'ddos')


def _shortfalls(feasible_trials):
    best_app = max(t.user_attrs['acc_app'] for t in feasible_trials)
    best_ddos = max(t.user_attrs['acc_ddos'] for t in feasible_trials)
    return {t.number: rel_shortfall(t.user_attrs['acc_app'], t.user_attrs['acc_ddos'],
                                    best_app, best_ddos)
            for t in feasible_trials}


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

    shortfalls = _shortfalls(feasible_trials)
    floor = min(shortfalls.values())

    close = [t for t in feasible_trials
             if shortfalls[t.number] <= floor + delta_select]

    chosen = min(close, key=lambda t: (t.user_attrs['blocks'], t.number))
    return chosen, shortfalls[chosen.number]


def discordant(ref_correct, cand_correct):
    """(b, c) of a paired comparison: b = flows the reference gets right and
    the candidate wrong, c = the reverse."""
    ref = np.asarray(ref_correct, dtype=bool)
    cand = np.asarray(cand_correct, dtype=bool)
    if ref.shape != cand.shape:
        raise ValueError('correctness marks differ in length: {} vs {}'.format(
            ref.shape, cand.shape))
    return int(np.count_nonzero(ref & ~cand)), int(np.count_nonzero(cand & ~ref))


def significantly_worse(b, c, alpha):
    """One-sided exact McNemar: is the candidate significantly WORSE than the
    reference? n = b + c = 0 (no discordant flow) counts as tied."""
    n = b + c
    if n == 0:
        return False
    return bool(binomtest(b, n=n, p=0.5, alternative='greater').pvalue < alpha)


def select_tied_cheapest(feasible_trials, correctness, alpha, k, max_blocks,
                         delta_select=0.02):
    """The cheapest feasible trial statistically tied with the balanced pick.

    R = select_best_trial(feasible_trials, delta_select, ...). A trial is TIED
    when it is not significantly worse than R on App AND on DDoS. Ship the tied
    trial with the fewest blocks, then the shallowest stage_depth, then the
    lowest trial number. R is always tied (b = c = 0), so the choice never
    costs more blocks than R.

    correctness : {trial number: (app marks, ddos marks)}, boolean per-flow
        right/wrong on val_select. Every feasible trial must have an entry.

    Returns (chosen, reference, table); table is one dict per feasible trial,
    sorted by number, carrying b/c per task against R and `tied`."""
    reference, _ = select_best_trial(feasible_trials, delta_select, k, max_blocks)
    missing = [t.number for t in feasible_trials if t.number not in correctness]
    if missing:
        raise ValueError('no correctness marks for feasible trial(s) {}'.format(missing))
    shortfalls = _shortfalls(feasible_trials)
    ref_marks = correctness[reference.number]
    table = []
    for trial in sorted(feasible_trials, key=lambda t: t.number):
        row = {'number': trial.number,
               'blocks': trial.user_attrs['blocks'],
               'stage_depth': trial.user_attrs['stage_depth'],
               'acc_sel_app': trial.user_attrs['acc_app'],
               'acc_sel_ddos': trial.user_attrs['acc_ddos'],
               'rel_shortfall': shortfalls[trial.number]}
        worse = False
        for index, task in enumerate(TASKS):
            b, c = discordant(ref_marks[index], correctness[trial.number][index])
            row['b_' + task], row['c_' + task] = b, c
            worse = worse or significantly_worse(b, c, alpha)
        row['tied'] = not worse
        table.append(row)
    by_number = {t.number: t for t in feasible_trials}
    best = min((r for r in table if r['tied']),
               key=lambda r: (r['blocks'], r['stage_depth'], r['number']))
    return by_number[best['number']], reference, table


_TRIAL_COLUMNS = ('acc_sel_app', 'acc_sel_ddos', 'blocks', 'stage_depth',
                  'b_app', 'c_app', 'b_ddos', 'c_ddos', 'tied')


def trial_rows(all_trials, table):
    """trials/<row_id>.csv rows: one per study trial, in number order. A trial
    not in `table` (infeasible, failed) carries '' in every measured column, so
    the rule can be re-run offline at any select_alpha from b/c alone."""
    by_number = {row['number']: row for row in table}
    rows = []
    for trial in sorted(all_trials, key=lambda t: t.number):
        measured = by_number.get(trial.number)
        row = {'number': trial.number,
               'params': json.dumps(trial.params, sort_keys=True),
               'feasible': measured is not None}
        for column in _TRIAL_COLUMNS:
            row[column] = measured[column] if measured is not None else ''
        rows.append(row)
    return rows
