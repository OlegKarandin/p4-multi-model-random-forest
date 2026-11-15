import pytest

from p4gen.p4_ground_truth import P4cNumbers
from src.verify.verdicts import Verdict, classify

INF = float("inf")


def _model(depth=10, blocks=30, feasible=True, tables=None):
    return {"stage_depth": depth, "blocks": blocks, "hw_feasible": feasible,
            "tables": tables or [{"table": "t", "kind": "ternary", "blocks": blocks, "stage": 8}]}


def _p4c(depth=10, blocks=30, table_blocks=None):
    tb = table_blocks if table_blocks is not None else ({"t": blocks} if blocks is not None else None)
    return P4cNumbers(depth, blocks, tb, 40, 10, 60)


CASES = [
    ("exact", _model(), _p4c(), 35, None, Verdict("EXACT", False, False, False, [])),
    ("false feasible", _model(), _p4c(13, None), 35, None, Verdict("FALSE_FEASIBLE", False, True, False, [])),
    ("under on depth", _model(10), _p4c(11), 35, None, Verdict("UNDER", False, False, False, [])),
    ("under on blocks", _model(blocks=29), _p4c(blocks=30), 35, None,
     Verdict("UNDER", False, False, False, [{"table": "t", "model": 29, "p4c": 30}])),
    ("over", _model(11), _p4c(10), 35, None, Verdict("OVER", False, False, False, [])),
    ("false infeasible", _model(13, feasible=False), _p4c(12), 35, None,
     Verdict("FALSE_INFEASIBLE", False, False, False, [])),
    ("over budget still exact", _model(blocks=36), _p4c(blocks=36), 35, None,
     Verdict("EXACT", True, False, False, [])),
    ("unbudgeted never over budget", _model(blocks=300), _p4c(blocks=300), INF, None,
     Verdict("EXACT", False, False, False, [])),
    ("compile error", _model(), None, 35, "COMPILE_ERROR", Verdict("COMPILE_ERROR", False, False, True, [])),
    ("timeout", _model(), None, 35, "TIMEOUT", Verdict("TIMEOUT", False, False, True, [])),
    ("never placed", _model(), _p4c(None, None), 35, None, Verdict("COMPILE_ERROR", False, False, True, [])),
    # cases the brief leaves undefined
    ("both infeasible, model deeper-than-none", _model(13, feasible=False), _p4c(14, None), 35, None,
     Verdict("UNDER", False, True, False, [])),
    ("both infeasible, same depth", _model(14, feasible=False), _p4c(14, None), 35, None,
     Verdict("EXACT", False, True, False, [])),
    ("both infeasible, model depth None", _model(None, None, feasible=False), _p4c(14, None), 35, None,
     Verdict("UNDER", False, True, False, [])),
    ("feasible model, depth None, p4c fits", _model(None, None), _p4c(10), 35, None,
     Verdict("UNDER", False, False, False, [{"table": "t", "model": 0, "p4c": 30}])),
    ("model blocks None, p4c allocated", _model(10, None), _p4c(blocks=30), 35, None,
     Verdict("UNDER", False, False, False, [{"table": "t", "model": 0, "p4c": 30}])),
    ("table only on one side", _model(blocks=30, tables=[{"table": "a", "kind": "t", "blocks": 30, "stage": 1}]),
     _p4c(blocks=31, table_blocks={"a": 30, "b": 1}), 35, None,
     Verdict("UNDER", False, False, False, [{"table": "b", "model": 0, "p4c": 1}])),
]


def _bff(model, p4c, M=35):
    return classify(model, p4c, M).verdict


def test_budget_false_feasible_is_model_within_m_and_p4c_over_m():
    got = classify(_model(blocks=30), _p4c(blocks=40), 35)
    assert got.verdict == "BUDGET_FALSE_FEASIBLE" and got.p4c_over_budget
    assert got.tables_differing == [{"table": "t", "model": 30, "p4c": 40}]


def test_budget_false_feasible_boundaries():
    assert _bff(_model(blocks=35), _p4c(blocks=36)) == "BUDGET_FALSE_FEASIBLE"
    # model already over M: an ordinary UNDER (the model knew it was over)
    assert _bff(_model(blocks=36), _p4c(blocks=40)) == "UNDER"
    # p4c exactly at M is within budget
    assert _bff(_model(blocks=30), _p4c(blocks=35)) == "UNDER"
    # M = inf never triggers
    assert classify(_model(blocks=30), _p4c(blocks=500), INF).verdict == "UNDER"
    # no model block number: cannot claim it was within budget
    assert _bff(_model(10, None), _p4c(blocks=40)) == "UNDER"
    # model over-predicts blocks but p4c still busts M: OVER stays OVER
    # only when the model was over M too
    assert _bff(_model(blocks=50), _p4c(blocks=40)) == "OVER"


def test_budget_false_feasible_precedence_pairings():
    # stage FALSE_FEASIBLE beats it (p4c over stages, blocks over M too)
    ff = classify(_model(blocks=30), _p4c(13, 40), 35)
    assert ff.verdict == "FALSE_FEASIBLE" and ff.p4c_over_budget
    # FALSE_INFEASIBLE beats it
    fi = classify(_model(13, blocks=30, feasible=False), _p4c(12, 40), 35)
    assert fi.verdict == "FALSE_INFEASIBLE"
    # beats UNDER on depth (model depth below p4c, blocks bust M)
    assert _bff(_model(9, 30), _p4c(10, 40)) == "BUDGET_FALSE_FEASIBLE"
    # beats OVER on depth (model deeper, but its blocks fit and p4c's do not)
    assert _bff(_model(11, 30), _p4c(10, 40)) == "BUDGET_FALSE_FEASIBLE"
    # EXACT blocks over M (model also over M) is still EXACT
    assert _bff(_model(blocks=40), _p4c(blocks=40)) == "EXACT"
    # compile error and model refusal take precedence (no p4c numbers / model)
    assert classify(_model(blocks=30), None, 35, "COMPILE_ERROR").verdict == "COMPILE_ERROR"
    refused = dict(_model(blocks=30), model_refusal="stage_pin")
    assert classify(refused, _p4c(blocks=40), 35).verdict == "MODEL_REFUSED:stage_pin"


@pytest.mark.parametrize("name,model,p4c,M,failure,want", CASES, ids=[c[0] for c in CASES])
def test_classify(name, model, p4c, M, failure, want):
    assert classify(model, p4c, M, failure) == want
