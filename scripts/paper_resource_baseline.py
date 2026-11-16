"""Paper Table II: the resource model and a public-specification baseline
estimator against p4c, per arm.

The baseline knows the generated program's dependency chain and the public
Tofino-1 figures (512 x 44-bit TCAM blocks, 24 blocks and 4 stateful ALUs per
stage) but none of the rules derived from the compiler:

  tree table   ceil(L / 512) * ceil(live key bits / 44), the live key bits being
               the bits the tree's key reads (p4_replay.tree_key_bits, the widths
               p4c's 524-bit key check sums): the code fields' live bits, or under
               per-tree codes the tree's own slices of the code_pk_ fields (not
               their declared container bits, which every tree would otherwise be
               charged in full)
  range table  ceil(declared intervals / 512)
  stage depth  program.FLOW_HASH_LEVEL (the flow-hash prologue) + ceil(registers / 4)
               + ceil(range blocks / 24) + ceil(tree blocks / 24)
               + the vote epilogue of usage.pipeline_depth (one stage, for a task
                 with a vote_<task> table; a one-tree task has none, review A2)

i.e. the phases run in sequence and a stage is 24 interchangeable blocks. The
prologue and the vote rule are the current generator's, read from p4model (never
copied here): run on texts of an older generator (3-stage prologue, a vote table
for every task), the baseline depth is that of the current program shape.

Usage:
  python scripts/paper_resource_baseline.py --run results/campaign_2026_11
      (needs <run>/designs/<row_id>.p4 and <run>/verification.csv)
  python scripts/paper_resource_baseline.py --calibration \
      results/compiler_calibration_pragmas_2026_09_29
      (needs <dir>/p4_src/<row_id>.p4 and <dir>.csv; the model side is
      replayed from the program)
Writes <out>/resource_baseline_rows.csv and prints the summary table, per arm
slug (independent, joint-off, joint-off-al, joint-task, joint-tree, any other
found) and over all arms, each for k <= 8, k >= 9 and all k (k from the row id).
A run's rows without p4c numbers (compile error, timeout) are skipped and
counted, and a twin whose verification record was copied from its
byte-identical source (`copied_from`) is the source's compile, counted once.
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4gen.p4_replay import (_TREE_TASK, model_breakdown_text,  # noqa: E402
                             parse_program_text, tree_key_bits)
from p4model import program as program_model  # noqa: E402
from p4model.target import (METER_ALUS_PER_STAGE as ALUS_PER_STAGE,  # noqa: E402
                                TCAM_BLOCK_KEY_LENGTH as TCAM_BITS,
                                TCAM_BLOCKS_PER_STAGE as BLOCKS_PER_STAGE,
                                TERNARY_MATCHING_ENTRIES_PER_BLOCK as TCAM_ROWS,
                                TOFINO_PIPELINE_STAGES as PIPELINE_STAGES)
from p4model.usage import pipeline_depth  # noqa: E402


def baseline(text):
    """(stage_depth, blocks) of the public-specification estimator."""
    program = parse_program_text(text)
    tree_blocks = range_blocks = 0
    tree_tasks = []
    for name, keys in program.tables.items():
        if not keys or name not in program.sizes:
            continue
        rows = math.ceil(program.sizes[name] / TCAM_ROWS)
        if name.startswith('get_classification_tree'):
            # The bits this tree's key reads (user decision 2026-11-16, final
            # review src I3): for a packed per-tree program its own slices, not
            # the code_pk_ fields' declared container bits.
            tree_blocks += rows * math.ceil(tree_key_bits(program, name) / TCAM_BITS)
            tree_tasks.append(_TREE_TASK.match(name).group(1))
        elif name.startswith('table_'):
            range_blocks += rows
    registers = len(re.findall(r'\bRegisterAction<', text))
    # Every phase in sequence, the trees in the phase's last stage(s), then the
    # vote epilogue for the tasks that have a vote table.
    end = (program_model.FLOW_HASH_LEVEL + math.ceil(registers / ALUS_PER_STAGE)
           + math.ceil(range_blocks / BLOCKS_PER_STAGE)
           + math.ceil(tree_blocks / BLOCKS_PER_STAGE))
    voting = {task for task in tree_tasks if 'vote_' + task in program.tables}
    stages = pipeline_depth(end, [end - 1] * len(tree_tasks), tree_tasks, voting)
    return stages, range_blocks + tree_blocks


# A campaign row id: {arm_slug}_M{token}_s{NN}_k{NN} (campaign_run.row_id).
_ROW_ID = re.compile(r'^(?P<arm>.+)_M(?:\d{3}|inf)_s\d+_k(?P<k>\d+)$')
_ANY_K = re.compile(r'_k(\d+)(?:_|$)')
ARM_ORDER = ('independent', 'joint-off', 'joint-off-al', 'joint-task', 'joint-tree')
K_GROUPS = ('k<=8', 'k>=9', 'all')


def arm_of(row_id):
    """The row's arm slug. A calibration-archive name (not a campaign row id)
    falls back to 'independent' / 'joint' by prefix."""
    m = _ROW_ID.match(row_id)
    if m:
        return m.group('arm')
    if row_id.startswith(('independent', 'heldout_independent',
                          'margin_independent')):
        return 'independent'
    return 'joint'


def k_group(row_id):
    """'k<=8' or 'k>=9' from the row id's k token; None when it has none."""
    m = _ROW_ID.match(row_id) or _ANY_K.search(row_id)
    if m is None:
        return None
    k = int(m.group('k') if m.re is _ROW_ID else m.group(1))
    return 'k<=8' if k <= 8 else 'k>=9'


def load_run(run):
    """One row per distinct compiled design of a run. Skipped and counted (the
    second value): rows without p4c numbers ('no_p4c': compile error, timeout)
    and twins whose record was copied from their byte-identical source
    ('copied', scored once, as the source)."""
    out, skipped = [], defaultdict(list)
    with open(os.path.join(run, 'verification.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if (r.get('copied_from') or '').strip():
                skipped['copied'].append(r['row_id'])
                continue
            if not (r.get('p4c_stage_depth') or '').strip():
                skipped['no_p4c'].append(r['row_id'])
                continue
            p4 = os.path.join(run, 'designs', r['row_id'] + '.p4')
            with open(p4, encoding='utf-8', errors='replace') as fh2:
                b_s, b_b = baseline(fh2.read())
            out.append(dict(row_id=r['row_id'], arm=arm_of(r['row_id']),
                            k_group=k_group(r['row_id']),
                            M=float(r['M']),
                            p4c_s=int(r['p4c_stage_depth']),
                            p4c_b=int(r['p4c_blocks']) if r['p4c_blocks'] else None,
                            # '' on a MODEL_REFUSED:<reason> row (decision 8.2)
                            model_s=(int(r['model_stage_depth'])
                                     if r['model_stage_depth'] else None),
                            model_b=int(r['model_blocks']) if r['model_blocks'] else None,
                            base_s=b_s, base_b=b_b))
    return out, dict(skipped)


def load_calibration(directory):
    out = []
    with open(directory.rstrip('/\\') + '.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            with open(os.path.join(directory, 'p4_src', r['row_id'] + '.p4'),
                      encoding='utf-8', errors='replace') as fh2:
                text = fh2.read()
            model = model_breakdown_text(text, r['row_id'])
            if model['stage_depth'] is None:
                # The replay refuses the archived program (decision 8.2, e.g. a
                # gated register block of an older generator): no model numbers.
                print('model refuses %s: %s' % (r['row_id'], model['refusal_reason']))
            b_s, b_b = baseline(text)
            out.append(dict(row_id=r['row_id'], arm=arm_of(r['row_id']),
                            k_group=k_group(r['row_id']),
                            M=float('inf'),
                            p4c_s=int(r['stages_real']),
                            p4c_b=(int(float(r['tcam_real']))
                                   if r['allocated'] == 'True' else None),
                            model_s=model['stage_depth'], model_b=model['blocks'],
                            base_s=b_s, base_b=b_b))
    return out, {}


def grouped(rows):
    """{(arm, k group): rows}: every arm present (ARM_ORDER first) and 'all',
    each for 'k<=8', 'k>=9' and 'all' (a row with no k token is in 'all' only)."""
    groups = defaultdict(list)
    for r in rows:
        for arm in (r['arm'], 'all'):
            for kg in ({r.get('k_group'), 'all'} - {None}):
                groups[(arm, kg)].append(r)
    arms = [a for a in ARM_ORDER if any(r['arm'] == a for r in rows)]
    arms += sorted({r['arm'] for r in rows} - set(arms)) + ['all']
    return [((arm, kg), groups[(arm, kg)]) for arm in arms for kg in K_GROUPS
            if groups[(arm, kg)]]


def feasible(stages, blocks, M):
    return stages <= PIPELINE_STAGES and blocks <= M


def summarise(rows, est):
    # A design the estimator refuses (the model's MODEL_REFUSED rows, decision
    # 8.2) has no numbers to compare: counted, then left out.
    refused = sum(r[est + '_s'] is None for r in rows)
    rows = [r for r in rows if r[est + '_s'] is not None]
    n = len(rows)
    with_b = [r for r in rows if r['p4c_b'] is not None]
    ds = [r[est + '_s'] - r['p4c_s'] for r in rows]
    db = [r[est + '_b'] - r['p4c_b'] for r in with_b]
    exact = under = 0
    for r in rows:
        cs = r[est + '_s'] - r['p4c_s']
        cb = 0 if r['p4c_b'] is None else r[est + '_b'] - r['p4c_b']
        if cs == 0 and cb == 0:
            exact += 1
        elif cs < 0 or cb < 0:
            under += 1
    false_feas = sum(feasible(r[est + '_s'], r[est + '_b'], r['M'])
                     and not feasible(r['p4c_s'], r['p4c_b'] or 0, r['M'])
                     for r in rows)
    false_infeas = sum(not feasible(r[est + '_s'], r[est + '_b'], r['M'])
                       and feasible(r['p4c_s'], r['p4c_b'] or 0, r['M'])
                       for r in rows)
    pct = lambda k, d=n: 100.0 * k / d if d else float('nan')
    return {
        'designs': n,
        'refused (not compared)': refused,
        'stage exact': pct(sum(d == 0 for d in ds)),
        'stage within 1': pct(sum(abs(d) <= 1 for d in ds)),
        'blocks exact': pct(sum(d == 0 for d in db), len(db)),
        'blocks within 1': pct(sum(abs(d) <= 1 for d in db), len(db)),
        'both exact': pct(exact),
        'under-predicts': pct(under),
        'over-predicts': pct(n - exact - under),
        'feasible, not by p4c': pct(false_feas),
        'infeasible, feasible by p4c': pct(false_infeas),
        'stage error range': (min(ds), max(ds)) if ds else None,
        'block error range': (min(db), max(db)) if db else None,
        'block MAE': sum(abs(d) for d in db) / len(db) if db else None,
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--run')
    g.add_argument('--calibration')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows, skipped = (load_run(args.run) if args.run
                     else load_calibration(args.calibration))
    for reason, ids in sorted(skipped.items()):
        print('skipped %d rows (%s): %s' % (len(ids), reason, ', '.join(ids)))
    out = args.out or (args.run or args.calibration.rstrip('/\\') + '_baseline')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'resource_baseline_rows.csv'), 'w',
              newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for est in ('model', 'base'):
        print('==', est)
        for (arm, kg), cell in grouped(rows):
            s = summarise(cell, est)
            print(' ', arm, kg, {k: (round(v, 1) if isinstance(v, float) else v)
                                 for k, v in s.items()})


if __name__ == '__main__':
    main()
