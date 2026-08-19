"""Prerequisites for the per-task objective rerun (spec risk 5)."""
import numpy as np


def test_importing_train_model_does_not_patch_sklearn_by_default():
    """sklearnex.patch_sklearn() replaces sklearn.ensemble.RandomForestClassifier
    with Intel's oneDAL implementation process-wide, at import time. In the
    PolimiML env that implementation cannot round-trip trees through
    scikit-learn 1.6.1's Tree.__setstate__, so every .fit() raises. It also may
    not observe the in-place tree_.threshold mutations that
    dt_thresholds_float_to_int and align_rf_thresholds depend on. Default off."""
    import src.training.train_model  # noqa: F401  -- the import IS the action under test
    from sklearn.ensemble import RandomForestClassifier

    assert 'sklearnex' not in RandomForestClassifier.__module__
    assert 'daal' not in RandomForestClassifier.__module__


def test_random_forest_fit_works_after_importing_train_model():
    """The end the guard exists for: a plain fit must survive the import."""
    import src.training.train_model  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.integers(0, 100, size=(60, 3)).astype(float)
    y = np.array([0, 1, 2] * 20)

    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(X, y)

    assert model.predict(X).shape == (60,)
