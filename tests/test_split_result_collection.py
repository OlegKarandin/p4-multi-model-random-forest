"""F3a: rows produced before a split failed must survive collection."""
from src.training import feature_selection as fs
from src.training.errors import NoFeasibleSolution


def _row(k, method='single', split=0):
    return {'method': method, 'split': split, 'k': k, 'acc_app': 0.7, 'acc_ddos': 0.9,
            'f1_app': 0.7, 'f1_ddos': 0.9, 'stages': 2, 'blocks': 10}


def test_collector_keeps_rows_from_a_split_that_also_reported_an_error():
    """The whole point of F3a: 16 good rows plus one failure at the last k
    used to be recorded as zero rows."""
    clean = fs.SplitResult(split_idx=0, results=[_row(3), _row(2)])
    partial = fs.SplitResult(split_idx=1, results=[_row(3, split=1)], error='boom\ntraceback')
    total_loss = fs.SplitResult(split_idx=2, results=[], error='died before any row')

    rows, completed, failed, partial_count = fs._collect_split_results(
        [clean, partial, total_loss])

    assert len(rows) == 3
    assert sorted(r['split'] for r in rows) == [0, 0, 1]
    assert completed == 1
    assert failed == 2
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
