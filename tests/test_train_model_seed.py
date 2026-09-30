import numpy as np
from sklearn.datasets import make_classification

from src.training.config import TrainConfig
from src.training.train_model import train_multi_RF_Optuna_multi_constrained

_NAMES = ["Fwd.Packet.Length.Max", "Flow.IAT.Max", "Bwd.IAT.Min"]


def _task(n_classes, seed):
    X, y = make_classification(n_samples=240, n_features=3, n_informative=3,
                               n_redundant=0, n_classes=n_classes, random_state=seed)
    X = np.abs(X * 1000).astype(int)
    return X[:120], y[:120], (X[120:180], y[120:180]), (X[180:], y[180:])


def _search(seed, warm=None):
    XA, yA, alA, selA = _task(3, 0)
    XB, yB, alB, selB = _task(2, 1)
    cfg = TrainConfig(n_trials=12, min_feasible_before_stop=5, lookback=4)
    return train_multi_RF_Optuna_multi_constrained(
        XA, yA, XB, yB, alA, alB, selA, selB, _NAMES, _NAMES,
        float("inf"), "disjoint", cfg, warm_start_params=warm, optuna_seed=seed)


def test_same_seed_same_search():
    first, second = _search(7), _search(7)
    assert first.best_params == second.best_params
    assert (first.blocks, first.stage_depth, first.n_trials_run) == (
        second.blocks, second.stage_depth, second.n_trials_run)


def test_same_seed_same_search_with_warm_start_variants():
    warm = _search(7).best_params
    assert _search(8, warm).best_params == _search(8, warm).best_params
