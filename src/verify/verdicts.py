"""Classify a model prediction against the p4c ground truth, worst verdict first."""
from dataclasses import dataclass

from src.p4gen.p4_ground_truth import P4cNumbers
from src.p4model.target import TOFINO_PIPELINE_STAGES

VERDICTS = ('FALSE_FEASIBLE', 'UNDER', 'FALSE_INFEASIBLE', 'OVER', 'EXACT',
            'COMPILE_ERROR', 'TIMEOUT')


@dataclass(frozen=True)
class Verdict:
    verdict: str
    p4c_over_budget: bool
    p4c_over_stages: bool
    unverified: bool
    tables_differing: list


def _model_table_blocks(model):
    out = {}
    for t in model.get('tables') or []:
        out[t['table']] = out.get(t['table'], 0) + (t.get('blocks') or 0)
    return out


def _tables_differing(model, p4c):
    mine = _model_table_blocks(model)
    theirs = p4c.table_blocks or {}
    rows = []
    for name in sorted(set(mine) | set(theirs)):
        m, p = mine.get(name, 0), theirs.get(name, 0)
        if m != p:
            rows.append({'table': name, 'model': m, 'p4c': p})
    return rows


def _cmp(model_value, p4c_value):
    """-1 model below p4c, +1 above, 0 equal. An unknown (None) model value
    counts as below: never hide a prediction we cannot stand behind."""
    if model_value is None:
        return -1
    return (model_value > p4c_value) - (model_value < p4c_value)


def classify(model, p4c, M, failure=None):
    if failure or p4c is None or p4c.stage_depth is None:
        return Verdict(failure or 'COMPILE_ERROR', False, False, True, [])

    over_stages = p4c.stage_depth > TOFINO_PIPELINE_STAGES
    over_budget = (p4c.blocks is not None and M != float('inf')
                   and p4c.blocks > M)
    feasible = bool(model['hw_feasible'])

    def out(name, differing=None):
        return Verdict(name, over_budget, over_stages, False, differing or [])

    if feasible and over_stages:
        return out('FALSE_FEASIBLE')
    if not feasible and not over_stages:
        return out('FALSE_INFEASIBLE')

    cmps = [_cmp(model.get('stage_depth'), p4c.stage_depth)]
    differing = []
    if p4c.allocated:
        cmps.append(_cmp(model.get('blocks'), p4c.blocks))
        if cmps[-1] != 0:
            differing = _tables_differing(model, p4c)
    if any(c < 0 for c in cmps):
        return out('UNDER', differing)
    if any(c > 0 for c in cmps):
        return out('OVER', differing)
    return out('EXACT')
