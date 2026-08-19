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
