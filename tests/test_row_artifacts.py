import json
import os

import joblib
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src.p4gen.build_p4_script import dt_thresholds_float_to_int
from src.p4gen.evaluation import multi_model_memory_evaluation
from src.training import row_artifacts as ra
from tests.test_train_model_seed import _NAMES, _task


def _forests():
    XA, yA, _, _ = _task(3, 0)
    XB, yB, _, _ = _task(2, 1)
    fa = dt_thresholds_float_to_int(RandomForestClassifier(3, max_depth=4, random_state=42).fit(XA, yA))
    fd = dt_thresholds_float_to_int(RandomForestClassifier(3, max_depth=4, random_state=42).fit(XB, yB))
    return fa, fd


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_writes_program_model_json_and_forests(tmp_path, encoding):
    fa, fd = _forests()
    usage = multi_model_memory_evaluation(fa, fd, _NAMES, _NAMES, encoding)
    ctx = ra.RowContext(str(tmp_path), "joint" if encoding == "joint" else "independent", 35, "abc123")
    model = ra.write_row_artifacts(ctx, "r_M035_s00_k03", fa, fd, _NAMES, _NAMES, encoding, usage)
    designs = tmp_path / "designs"
    assert (designs / "r_M035_s00_k03.p4").is_file()
    on_disk = json.loads((designs / "r_M035_s00_k03.model.json").read_text())
    assert on_disk == model
    assert model["training_blocks"] == usage.blocks
    assert model["model_paths_differ"] == ((model["stage_depth"], model["blocks"])
                                           != (usage.stage_depth, usage.blocks))
    assert model["budget_feasible"] == (model["blocks"] <= 35)
    assert sum(t["blocks"] for t in model["tables"]) == model["blocks"]
    stored = joblib.load(tmp_path / "forests" / "r_M035_s00_k03.joblib")
    assert stored["sklearn"] == sklearn.__version__ and stored["features_app"] == _NAMES
    assert not list(tmp_path.rglob("*.partial"))


def test_unbudgeted_row_is_budget_feasible_and_M_null(tmp_path):
    fa, fd = _forests()
    usage = multi_model_memory_evaluation(fa, fd, _NAMES, _NAMES, "joint")
    model = ra.write_row_artifacts(ra.RowContext(str(tmp_path), "joint", float("inf"), "c"),
                                   "x", fa, fd, _NAMES, _NAMES, "joint", usage)
    assert model["M"] is None and model["budgeted"] is False and model["budget_feasible"] is True


def test_generator_value_error_is_recorded_not_raised(tmp_path, monkeypatch):
    fa, fd = _forests()
    usage = multi_model_memory_evaluation(fa, fd, _NAMES, _NAMES, "joint")

    def refuse(*args, **kwargs):
        raise ValueError("shared-field layout conflict on code_x")

    monkeypatch.setattr(ra, "generate_P4_code", refuse)
    model = ra.write_row_artifacts(ra.RowContext(str(tmp_path), "joint", 35, "c"),
                                   "x", fa, fd, _NAMES, _NAMES, "joint", usage)
    assert model["generator_error"] == "shared-field layout conflict on code_x"
    assert model["stage_depth"] is None and model["tables"] is None
    assert (tmp_path / "designs" / "x.generator_error.txt").read_text() == \
        "shared-field layout conflict on code_x"
    assert (tmp_path / "forests" / "x.joblib").is_file()


def test_load_forests_refuses_another_sklearn(tmp_path):
    path = tmp_path / "f.joblib"
    joblib.dump({"sklearn": "0.0.1"}, path)
    with pytest.raises(RuntimeError, match="0.0.1"):
        ra.load_forests(str(path))


def test_write_trial_table_writes_one_atomic_csv_per_row(tmp_path):
    import pandas as pd
    from src.training.row_artifacts import RowContext, write_trial_table
    ctx = RowContext(str(tmp_path), 'joint', 35.0, 'deadbeef')
    rows = [{'number': 0, 'params': '{}', 'feasible': True, 'acc_sel_app': 0.9,
             'acc_sel_ddos': 0.95, 'blocks': 20, 'stage_depth': 6, 'b_app': 0,
             'c_app': 0, 'b_ddos': 0, 'c_ddos': 0, 'tied': True}]
    path = write_trial_table(ctx, 'joint_M035_s00_k05', rows)
    assert path == str(tmp_path / 'trials' / 'joint_M035_s00_k05.csv')
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(rows[0])
    assert not (tmp_path / 'trials' / 'joint_M035_s00_k05.csv.partial').exists()
