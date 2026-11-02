"""Paper Table II: the resource model and a public-specification baseline
estimator against p4c, per arm.

The baseline knows the generated program's dependency chain and the public
Tofino-1 figures (512 x 44-bit TCAM blocks, 24 blocks and 4 stateful ALUs per
stage) but none of the rules derived from the compiler:

  tree table   ceil(L / 512) * ceil(sum of code-field bits / 44)
  range table  ceil(declared intervals / 512)
  stage depth  3 (prologue) + ceil(registers / 4) + ceil(range blocks / 24)
               + ceil(tree blocks / 24) + 1 (vote)

i.e. the phases run in sequence and a stage is 24 interchangeable blocks.

Usage:
  python scripts/paper_resource_baseline.py --run results/campaign_2026_11
      (needs <run>/designs/<row_id>.p4 and <run>/verification.csv)
  python scripts/paper_resource_baseline.py --calibration \
      results/compiler_calibration_pragmas_2026_09_29
      (needs <dir>/p4_src/<row_id>.p4 and <dir>.csv; the model side is
      replayed from the program)
Writes <out>/resource_baseline_rows.csv and prints the summary table.
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4gen.p4_replay import model_breakdown, parse_program_text  # noqa: E402

TCAM_ROWS, TCAM_BITS, BLOCKS_PER_STAGE, ALUS_PER_STAGE = 512, 44, 24, 4
PROLOGUE_STAGES, VOTE_STAGES, PIPELINE_STAGES = 3, 1, 12


def baseline(text):
    """(stage_depth, blocks) of the public-specification estimator."""
    program = parse_program_text(text)
    tree_blocks = range_blocks = 0
    for name, keys in program.tables.items():
        if not keys or name not in program.sizes:
            continue
        rows = math.ceil(program.sizes[name] / TCAM_ROWS)
        if name.startswith('get_classification_tree'):
            key_bits = sum(program.bits[key] for key in keys)
            tree_blocks += rows * math.ceil(key_bits / TCAM_BITS)
        elif name.startswith('table_'):
            range_blocks += rows
    registers = len(re.findall(r'\bRegisterAction<', text))
    stages = (PROLOGUE_STAGES + math.ceil(registers / ALUS_PER_STAGE)
              + math.ceil(range_blocks / BLOCKS_PER_STAGE)
              + math.ceil(tree_blocks / BLOCKS_PER_STAGE) + VOTE_STAGES)
    return stages, range_blocks + tree_blocks


def arm_of(row_id):
    if row_id.startswith(('independent', 'heldout_independent',
                          'margin_independent')):
        return 'independent'
    return 'joint'


def load_run(run):
    out = []
    with open(os.path.join(run, 'verification.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            p4 = os.path.join(run, 'designs', r['row_id'] + '.p4')
            with open(p4, encoding='utf-8', errors='replace') as fh2:
                b_s, b_b = baseline(fh2.read())
            out.append(dict(row_id=r['row_id'], arm=arm_of(r['row_id']),
                            M=float(r['M']),
                            p4c_s=int(r['p4c_stage_depth']),
                            p4c_b=int(r['p4c_blocks']) if r['p4c_blocks'] else None,
                            model_s=int(r['model_stage_depth']),
                            model_b=int(r['model_blocks']),
                            base_s=b_s, base_b=b_b))
    return out


def load_calibration(directory):
    out = []
    with open(directory.rstrip('/\\') + '.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            with open(os.path.join(directory, 'p4_src', r['row_id'] + '.p4'),
                      encoding='utf-8', errors='replace') as fh2:
                text = fh2.read()
            model = model_breakdown(parse_program_text(text), r['row_id'])
            b_s, b_b = baseline(text)
            out.append(dict(row_id=r['row_id'], arm=arm_of(r['row_id']),
                            M=float('inf'),
                            p4c_s=int(r['stages_real']),
                            p4c_b=(int(float(r['tcam_real']))
                                   if r['allocated'] == 'True' else None),
                            model_s=model['stage_depth'], model_b=model['blocks'],
                            base_s=b_s, base_b=b_b))
    return out


def feasible(stages, blocks, M):
    return stages <= PIPELINE_STAGES and blocks <= M


def summarise(rows, est):
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
        'stage exact': pct(sum(d == 0 for d in ds)),
        'stage within 1': pct(sum(abs(d) <= 1 for d in ds)),
        'blocks exact': pct(sum(d == 0 for d in db), len(db)),
        'blocks within 1': pct(sum(abs(d) <= 1 for d in db), len(db)),
        'both exact': pct(exact),
        'under-predicts': pct(under),
        'over-predicts': pct(n - exact - under),
        'feasible, not by p4c': pct(false_feas),
        'infeasible, feasible by p4c': pct(false_infeas),
        'stage error range': (min(ds), max(ds)),
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
    rows = load_run(args.run) if args.run else load_calibration(args.calibration)
    out = args.out or (args.run or args.calibration.rstrip('/\\') + '_baseline')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'resource_baseline_rows.csv'), 'w',
              newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r['arm']].append(r)
        groups['all'].append(r)
    for est in ('model', 'base'):
        print('==', est)
        for arm in ('independent', 'joint', 'all'):
            s = summarise(groups[arm], est)
            print(' ', arm, {k: (round(v, 1) if isinstance(v, float) else v)
                             for k, v in s.items()})


if __name__ == '__main__':
    main()
