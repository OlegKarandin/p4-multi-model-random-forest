"""Classify a model prediction against the p4c ground truth, worst verdict first."""
from dataclasses import dataclass

from p4model.target import TOFINO_PIPELINE_STAGES

VERDICTS = ('FALSE_FEASIBLE', 'BUDGET_FALSE_FEASIBLE', 'UNDER', 'FALSE_INFEASIBLE',
            'OVER', 'EXACT', 'COMPILE_ERROR', 'TIMEOUT')

# A program the model's replay refuses outright (model.json's model_refusal,
# p4_replay.model_breakdown) has no prediction to classify: its verdict is
# MODEL_REFUSED_PREFIX + the refusal's reason, e.g. MODEL_REFUSED:stage_pin.
# Those verdicts are NOT in VERDICTS (the reason varies): a filter such as
# `isin(VERDICTS)` would drop them silently -- test with is_verdict instead.
MODEL_REFUSED_PREFIX = 'MODEL_REFUSED:'


def is_verdict(name):
    """True for one of VERDICTS or a MODEL_REFUSED:<reason> verdict."""
    return name in VERDICTS or (isinstance(name, str)
                                and name.startswith(MODEL_REFUSED_PREFIX))


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
    """Verdict of one design. Precedence, first match wins (decision 8.4):

    1. MODEL_REFUSED:<reason>  the model refused the program (no numbers).
    2. COMPILE_ERROR / TIMEOUT  no p4c stage count (unverified). Deliverable 9
       counts one on a model-feasible design as its own miss class.
    3. FALSE_FEASIBLE          model feasible, p4c over the stage limit.
    4. FALSE_INFEASIBLE        model infeasible, p4c within the stage limit.
    5. BUDGET_FALSE_FEASIBLE   model feasible, p4c within the stage limit, the
       model's blocks <= M but p4c's blocks > M (never for M = inf, or when the
       model has no block number). Ranked above UNDER because it is the
       consequential under-prediction: the design was selected as within budget
       and is not. A stage-level false verdict (3, 4) keeps priority, so a
       stage FALSE_FEASIBLE stays FALSE_FEASIBLE even when it also busts M.
    6. UNDER, 7. OVER, 8. EXACT  as before.
    """
    if model.get('model_refusal'):
        # Checked first: with no model numbers there is nothing to compare. The
        # p4c side is kept and flagged as usual; the row is unverified only when
        # p4c gave no stage count (a failed compile).
        compiled = (not failure and p4c is not None and p4c.stage_depth is not None)
        over_stages = compiled and p4c.stage_depth > TOFINO_PIPELINE_STAGES
        over_budget = (compiled and p4c.blocks is not None and M != float('inf')
                       and p4c.blocks > M)
        return Verdict(MODEL_REFUSED_PREFIX + model['model_refusal'], bool(over_budget),
                       bool(over_stages), not compiled, [])
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

    model_blocks = model.get('blocks')
    if (feasible and over_budget and model_blocks is not None
            and M != float('inf') and model_blocks <= M):
        differing = _tables_differing(model, p4c) if p4c.allocated else []
        return out('BUDGET_FALSE_FEASIBLE', differing)

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
