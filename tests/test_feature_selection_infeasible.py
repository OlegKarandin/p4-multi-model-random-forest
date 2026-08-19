"""F3b: an infeasible k records a row and elimination continues."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.training import feature_selection as fs
from src.training.errors import NoFeasibleSolution

FEATURE_NAMES = ['Flow.IAT.Max', 'Fwd.IAT.Max', 'Fwd.Packet.Length.Max']


def _synthetic_data(n=180, seed=0):
    """Small but real: permutation_importance and accuracy_metrics both run for
    real in these tests. Only the Optuna search is replaced."""
    rng = np.random.default_rng(seed)
    X_app = rng.integers(0, 500, size=(n, 3)).astype(float)
    y_app = np.array([0, 1, 2] * (n // 3))
    X_ddos = rng.integers(0, 500, size=(n, 3)).astype(float)
    y_ddos = np.array([-1, 1] * (n // 2))
    return X_app, X_ddos, y_app, y_ddos


def _fake_trainer(raise_at_k=()):
    """Stands in for train_multi_RF_Optuna_multi_constrained.

    Returns the 5-tuple that function returns TODAY: (model_A, model_B,
    stages, blocks, best_params). P2 widens it to 7 -- update this helper when
    P2's Task 4 lands.
    """
    def trainer(X_A, y_A, X_B, y_B, *args, **kwargs):
        k = X_A.shape[1]
        if k in raise_at_k:
            raise NoFeasibleSolution(k=k, max_blocks=1)
        model_A = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A)
        model_B = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_B, y_B)
        return model_A, model_B, 1, 1, {}
    return trainer


def _run(monkeypatch, raise_at_k=()):
    # _process_single_split imports the trainer INSIDE the function (to stay
    # picklable for ProcessPoolExecutor), so patching the module attribute
    # takes effect at call time.
    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained',
        _fake_trainer(raise_at_k))
    X_app, X_ddos, y_app, y_ddos = _synthetic_data()
    return fs._process_single_split(
        split_idx=10, X_app=X_app, X_ddos=X_ddos, y_app=y_app, y_ddos=y_ddos,
        n_trees=-1, max_depth=-1, max_blocks=25,
        feature_names=FEATURE_NAMES, random_state=42, verbose=False)


def test_all_k_feasible_produces_a_row_per_k_per_method(monkeypatch):
    result = _run(monkeypatch)

    assert result.error is None
    single = sorted(r['k'] for r in result.results if r['method'] == 'single')
    multi = sorted(r['k'] for r in result.results if r['method'] == 'multi')
    assert single == [1, 2, 3]
    assert multi == [1, 2, 3]
    assert all(r['infeasible'] == '' for r in result.results)


def test_an_infeasible_middle_k_is_recorded_and_elimination_continues(monkeypatch):
    """k=2 has no feasible model; k=1 must still be attempted and recorded."""
    result = _run(monkeypatch, raise_at_k=(2,))

    assert result.error is None
    single = {r['k']: r for r in result.results if r['method'] == 'single'}
    assert sorted(single) == [1, 2, 3]
    assert single[2]['infeasible'] == 'no feasible solution at k=2 under max_blocks=1'
    assert single[2]['acc_app'] is None
    assert single[2]['blocks'] is None
    assert single[3]['infeasible'] == ''
    assert single[1]['infeasible'] == ''


def test_an_infeasible_first_k_breaks_the_loop_with_one_row(monkeypatch):
    """No previous ranking exists, so there is no defensible next feature to
    drop. Record the row, stop that loop -- do not guess."""
    result = _run(monkeypatch, raise_at_k=(3,))

    assert result.error is None
    single = [r for r in result.results if r['method'] == 'single']
    assert len(single) == 1
    assert single[0]['k'] == 3
    assert single[0]['infeasible'].startswith('no feasible solution at k=3')


def test_every_k_infeasible_yields_exactly_one_row_per_method(monkeypatch):
    result = _run(monkeypatch, raise_at_k=(1, 2, 3))

    assert result.error is None
    assert len([r for r in result.results if r['method'] == 'single']) == 1
    assert len([r for r in result.results if r['method'] == 'multi']) == 1
