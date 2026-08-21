"""F3a: rows produced before a split failed must survive collection."""
import numpy as np

from src.training import feature_selection as fs
from src.training.errors import NoFeasibleSolution


def _row(k, method='single', split=0):
    return {'method': method, 'split': split, 'k': k, 'acc_app': 0.7, 'acc_ddos': 0.9,
            'f1_app': 0.7, 'f1_ddos': 0.9, 'stages': 2, 'blocks': 10}


def test_collector_keeps_rows_from_a_split_that_also_reported_an_error():
    """The whole point of F3a: 16 good rows plus one failure at the last k
    used to be recorded as zero rows. `total_loss` (an error with zero rows,
    e.g. a split that failed before ever reaching a completed k) stays
    hand-built here -- SplitResult itself is a plain, always-legal container;
    only the *partial* case (rows AND an error together) needed to be proven
    reachable from real production code, which is done below via
    `_process_single_split`."""
    clean = fs.SplitResult(split_idx=0, results=[_row(3), _row(2)])
    total_loss = fs.SplitResult(split_idx=2, results=[], error='died before any row')

    rows, completed, failed, partial_count = fs._collect_split_results(
        [clean, total_loss])

    assert len(rows) == 2
    assert sorted(r['split'] for r in rows) == [0, 0]
    assert completed == 1
    assert failed == 1
    assert partial_count == 0


def test_process_single_split_keeps_rows_completed_before_an_unhandled_raise(monkeypatch):
    """F3a's real production path: `_run_elimination` appends into a
    caller-owned list, so rows completed before a raise reach
    `SplitResult(results=..., error=...)` in `_process_single_split`'s
    `except` branch -- this state (rows AND an error together) used to be
    unreachable from production code; it's hand-built no longer.

    The fake trainer succeeds at k=3 and k=2 (two rows), then raises a plain
    AssertionError at k=1 -- standing in for an unhandled failure such as
    AlignmentInvariantError or train_model.py's own determinism assertion
    (NOT NoFeasibleSolution, which `_run_elimination` already catches and
    handles per-k without propagating)."""
    from sklearn.ensemble import RandomForestClassifier
    from src.training.train_model import TrainResult

    def trainer(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                val_select_A, val_select_B, features_A, features_B,
                max_blocks, encoding, cfg, warm_start_params=None):
        k = X_A.shape[1]
        if k == 1:
            raise AssertionError('simulated determinism failure')
        model_A = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A)
        model_B = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_B, y_B)
        return TrainResult(
            model_A=model_A, model_B=model_B, stages=1, blocks=1,
            acc_sel_A=0.7, acc_sel_B=0.9, best_params={},
            rel_shortfall=0.0, n_trials_run=1, n_feasible=1,
            align_attempted=None, align_accepted=None,
            intervals_before=None, intervals_after=None)

    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained', trainer)

    rng = np.random.RandomState(0)
    n = 40
    X_app = rng.randint(0, 65535, size=(n, 3))
    X_ddos = X_app.copy()
    y_app = rng.randint(0, 3, size=n)
    y_ddos = rng.choice([-1, 1], size=n)

    result = fs._process_single_split(
        split_idx=0, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
        max_blocks=50, feature_names=['f0', 'f1', 'f2'],
        random_state=42, verbose=False, arm='independent')

    assert len(result.results) == 2
    assert sorted(r['k'] for r in result.results) == [2, 3]
    assert result.error is not None
    assert 'simulated determinism failure' in result.error

    # n_partial (rows AND an error on the SAME split) is reachable for the
    # first time -- exercise the collector against this real SplitResult too.
    rows, completed, failed, partial_count = fs._collect_split_results([result])
    assert len(rows) == 2
    assert completed == 0
    assert failed == 1
    assert partial_count == 1


def test_collector_on_an_empty_input_is_not_an_error():
    rows, completed, failed, partial_count = fs._collect_split_results([])

    assert rows == []
    assert (completed, failed, partial_count) == (0, 0, 0)


def test_no_feasible_solution_carries_the_cell_it_failed_on():
    """Task 5 catches this per-k and records the row as infeasible, so the
    message must name the cell without the caller reconstructing it."""
    exc = NoFeasibleSolution(k=3, max_blocks=25)

    assert exc.k == 3
    assert exc.max_blocks == 25
    assert str(exc) == 'no feasible solution at k=3 under max_blocks=25'
