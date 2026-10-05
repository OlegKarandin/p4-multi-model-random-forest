"""F11 site 3: which trial wins was decided on the two tasks' AVERAGE."""
import pytest

from src.training.errors import NoFeasibleSolution
from src.training import trial_selection as ts


class FakeTrial:
    """Duck-types the three user_attrs select_best_trial reads."""

    def __init__(self, name, acc_app, acc_ddos, blocks):
        self.name = name
        self.number = int(name[1:])
        self.user_attrs = {'acc_app': acc_app, 'acc_ddos': acc_ddos, 'blocks': blocks}


# Spec B.3's worked example, verbatim. best_app = 0.780 (error 0.220),
# best_ddos = 0.960 (error 0.040); no trial attains both, so floor = 2.3%.
WORKED_EXAMPLE = [
    FakeTrial('T1', 0.780, 0.941, 25),
    FakeTrial('T2', 0.759, 0.960, 23),
    FakeTrial('T3', 0.775, 0.960, 24),
    FakeTrial('T4', 0.778, 0.958, 22),
    FakeTrial('T5', 0.770, 0.957, 21),
    FakeTrial('T6', 0.762, 0.952, 18),
    FakeTrial('T8', 0.780, 0.951, 19),
]


def test_rel_deg_is_relative_to_the_error_not_the_accuracy():
    assert ts.rel_deg(0.96, 0.951) == pytest.approx(0.225, abs=1e-4)
    assert ts.rel_deg(0.78, 0.775) == pytest.approx(0.022727, abs=1e-5)
    assert ts.rel_deg(0.96, 0.96) == 0.0


def test_rel_deg_handles_a_perfect_before_score_without_dividing_by_zero():
    assert ts.rel_deg(1.0, 0.999) > 0
    assert ts.rel_deg(1.0, 1.0) == 0.0


def test_rel_shortfall_is_scale_matched_across_the_two_tasks():
    """The same 0.005 absolute drop is 2.3% of App's error but 12.5% of
    DDoS's. Averaging accuracy treats them as equal; this must not."""
    app_only = ts.rel_shortfall(0.775, 0.960, best_app=0.780, best_ddos=0.960)
    ddos_only = ts.rel_shortfall(0.780, 0.955, best_app=0.780, best_ddos=0.960)

    assert app_only == pytest.approx(0.022727, abs=1e-5)
    assert ddos_only == pytest.approx(0.125, abs=1e-6)
    assert ddos_only > app_only


def test_rel_shortfall_is_the_worse_served_task():
    """max, not mean: the point is that a loss on one task cannot be masked."""
    assert ts.rel_shortfall(0.775, 0.955, 0.780, 0.960) == pytest.approx(0.125, abs=1e-6)


def test_worked_example_floor_is_not_zero():
    """best_app and best_ddos come from DIFFERENT trials (T1/T8 and T2/T3), so
    the ideal corner is unattainable and a bare delta_select band would be
    empty at delta_select = 0. floor + delta is required."""
    trial, shortfall = ts.select_best_trial(
        WORKED_EXAMPLE, delta_select=0.0, k=17, max_blocks=25)

    assert trial.name == 'T3'
    assert shortfall == pytest.approx(0.022727, abs=1e-5)


def test_worked_example_at_each_delta_select_matches_the_spec_table():
    for delta, expected_name, expected_blocks in ((0.00, 'T3', 24),
                                                  (0.02, 'T3', 24),
                                                  (0.05, 'T4', 22),
                                                  (0.10, 'T5', 21)):
        trial, _ = ts.select_best_trial(
            WORKED_EXAMPLE, delta_select=delta, k=17, max_blocks=25)
        assert (trial.name, trial.user_attrs['blocks']) == (expected_name, expected_blocks), delta


def test_the_new_rule_never_picks_the_trial_the_average_rule_picked():
    """T8 buys its 19 blocks entirely out of DDoS -- 22.5% of that task's
    error -- and today's average band selects it. No delta_select in the grid
    may reproduce that."""
    for delta in (0.00, 0.02, 0.05, 0.10, 0.20):
        trial, _ = ts.select_best_trial(
            WORKED_EXAMPLE, delta_select=delta, k=17, max_blocks=25)
        assert trial.name != 'T8', delta


def test_the_band_is_never_empty():
    """floor is attained by construction, so the trial defining it is always a
    member -- even at delta_select = 0."""
    for delta in (0.0, 0.001, 0.5):
        trial, _ = ts.select_best_trial(
            WORKED_EXAMPLE, delta_select=delta, k=17, max_blocks=25)
        assert trial is not None


def test_ties_on_blocks_are_broken_deterministically():
    """Two workers on the same cell must return the same model."""
    tied = [FakeTrial('T1', 0.78, 0.96, 20), FakeTrial('T2', 0.78, 0.96, 20)]

    first, _ = ts.select_best_trial(tied, delta_select=0.02, k=3, max_blocks=25)
    second, _ = ts.select_best_trial(list(reversed(tied)), delta_select=0.02, k=3, max_blocks=25)

    assert first.name == second.name == 'T1'


def test_no_feasible_trials_raises_no_feasible_solution():
    with pytest.raises(NoFeasibleSolution) as excinfo:
        ts.select_best_trial([], delta_select=0.02, k=7, max_blocks=25)

    assert excinfo.value.k == 7
    assert excinfo.value.max_blocks == 25


# ---------------------------------------------------------------------------
# Tie-aware selection (spec 2026-10-04 Part II)
# ---------------------------------------------------------------------------
import json

import numpy as np


class TieTrial:
    def __init__(self, number, acc_app, acc_ddos, blocks, stage_depth=6, params=None,
                 feasible=True):
        self.number = number
        self.params = params or {'n_estimators_A': number}
        self.feasible = feasible
        self.user_attrs = ({'acc_app': acc_app, 'acc_ddos': acc_ddos, 'blocks': blocks,
                            'stage_depth': stage_depth} if feasible else {})


N = 200


def _marks(wrong):
    """Correctness marks with exactly the listed flow indices wrong."""
    out = np.ones(N, dtype=bool)
    out[list(wrong)] = False
    return out


def _case():
    """R (trial 0) is the balanced pick: best on both tasks.
    1: cheaper, 3 extra app errors (b=3, c=0: p=0.125 -> tied at 0.05).
    2: cheapest, 12 extra ddos errors (b=12, c=0: p=0.000244 -> worse).
    3: as cheap as 1, discordant both ways (b=5, c=5 -> tied), deeper; its acc_app is
    below R's band so that R (trial 0), not 3, is select_best_trial's pick."""
    trials = [TieTrial(0, 0.90, 0.95, 20, 6), TieTrial(1, 0.885, 0.95, 15, 6),
              TieTrial(2, 0.90, 0.89, 10, 6), TieTrial(3, 0.89, 0.95, 15, 7)]
    base = range(0)
    corr = {
        0: (_marks(base), _marks(base)),
        1: (_marks(range(3)), _marks(base)),
        2: (_marks(base), _marks(range(12))),
        3: (_marks(range(5)), _marks(base)),
    }
    # trial 3 also gets 5 flows right that R gets wrong on app:
    r_app = _marks(range(100, 105))
    corr[0] = (r_app, corr[0][1])
    corr[1] = (_marks(list(range(3)) + list(range(100, 105))), corr[1][1])
    corr[2] = (_marks(range(100, 105)), corr[2][1])
    return trials, corr


def test_discordant_counts_both_directions():
    r = np.array([True, True, False, False])
    c = np.array([True, False, True, False])
    assert ts.discordant(r, c) == (1, 1)
    with pytest.raises(ValueError):
        ts.discordant(r, c[:3])


def test_n_zero_counts_as_tied():
    assert ts.significantly_worse(0, 0, 0.05) is False


def test_significance_is_one_sided_exact_binomial():
    assert ts.significantly_worse(12, 0, 0.05) is True      # p = 2**-12
    assert ts.significantly_worse(3, 0, 0.05) is False      # p = 0.125
    assert ts.significantly_worse(0, 12, 0.05) is False     # C BETTER: never worse


def test_a_tied_cheaper_trial_is_chosen_and_a_significantly_worse_one_is_not():
    trials, corr = _case()
    chosen, ref, table = ts.select_tied_cheapest(trials, corr, 0.05, k=5, max_blocks=50)
    assert ref.number == 0
    tied = {row['number'] for row in table if row['tied']}
    assert tied == {0, 1, 3}                # 2 is cheapest but worse on DDoS
    assert chosen.number == 1               # 15 blocks; ties 3 on blocks, shallower


def test_tie_break_is_blocks_then_stage_depth_then_number():
    trials = [TieTrial(0, 0.9, 0.9, 20, 6), TieTrial(5, 0.9, 0.9, 15, 7),
              TieTrial(4, 0.9, 0.9, 15, 7), TieTrial(3, 0.9, 0.9, 15, 8)]
    corr = {t.number: (_marks(()), _marks(())) for t in trials}
    chosen, _, _ = ts.select_tied_cheapest(trials, corr, 0.05, k=5, max_blocks=50)
    assert chosen.number == 4


def test_the_reference_is_always_tied_and_never_cheaper_than_the_choice():
    import random
    rng = random.Random('tie-aware')
    for _ in range(200):
        trials, corr = [], {}
        for number in range(rng.randint(1, 8)):
            trials.append(TieTrial(number, rng.uniform(0.7, 0.9), rng.uniform(0.85, 0.99),
                                   rng.randint(5, 40), rng.randint(4, 12)))
            corr[number] = (np.array([rng.random() < 0.8 for _ in range(60)]),
                            np.array([rng.random() < 0.9 for _ in range(60)]))
        chosen, ref, table = ts.select_tied_cheapest(
            trials, corr, rng.choice([0.0, 0.05, 0.5, 1.0]), k=5, max_blocks=50)
        assert next(r for r in table if r['number'] == ref.number)['tied']
        assert chosen.user_attrs['blocks'] <= ref.user_attrs['blocks']
        if chosen.user_attrs['blocks'] == ref.user_attrs['blocks']:
            assert chosen.user_attrs['stage_depth'] <= ref.user_attrs['stage_depth']


def test_alpha_zero_ships_the_globally_cheapest_feasible_trial():
    trials, corr = _case()
    chosen, _, _ = ts.select_tied_cheapest(trials, corr, 0.0, k=5, max_blocks=50)
    assert chosen.number == 2


def test_alpha_one_ties_only_trials_with_b_zero_on_both_tasks():
    trials, corr = _case()
    _, _, table = ts.select_tied_cheapest(trials, corr, 1.0, k=5, max_blocks=50)
    for row in table:
        assert row['tied'] == (row['b_app'] == 0 and row['b_ddos'] == 0)


def test_the_reference_is_select_best_trials_pick():
    trials, corr = _case()
    _, ref, _ = ts.select_tied_cheapest(trials, corr, 0.05, k=5, max_blocks=50,
                                        delta_select=0.02)
    best, _ = ts.select_best_trial(trials, 0.02, k=5, max_blocks=50)
    assert ref is best


def test_no_feasible_trial_raises_no_feasible_solution():
    with pytest.raises(NoFeasibleSolution):
        ts.select_tied_cheapest([], {}, 0.05, k=5, max_blocks=50)


def test_missing_correctness_for_a_feasible_trial_raises():
    trials, corr = _case()
    del corr[3]
    with pytest.raises(ValueError, match='3'):
        ts.select_tied_cheapest(trials, corr, 0.05, k=5, max_blocks=50)


def test_trial_rows_cover_every_trial_and_blank_the_infeasible_ones():
    trials, corr = _case()
    _, _, table = ts.select_tied_cheapest(trials, corr, 0.05, k=5, max_blocks=50)
    infeasible = TieTrial(9, None, None, None, feasible=False)
    rows = ts.trial_rows(trials + [infeasible], table)
    assert [r['number'] for r in rows] == [0, 1, 2, 3, 9]
    assert rows[-1]['feasible'] is False and rows[-1]['blocks'] == '' and rows[-1]['tied'] == ''
    assert json.loads(rows[0]['params']) == trials[0].params
    assert rows[2]['b_ddos'] == 12 and rows[2]['tied'] is False
