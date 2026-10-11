"""N = 3 compile smoke test (spec 2026-10-11, Testing item 3). Information only,
not a gate.

Each synthetic design takes one campaign row's two forests (ddos, app) and adds
a third task, `app2`: the app forest of the SAME arm/M/k on a different split,
so it has different trees (and, under 'disjoint', different intervals) while
every code path of the three-task generator is exercised. Task order is
(ddos, app, app2): priorities 3, 2, 1. 'disjoint' designs come from the
independent arm with per-task intervals; 'joint' ones from joint-off with one
interval set over all three forests' trees.

Each program is compiled with p4c (src.p4gen.p4_compile.compile_p4, WSL) and the
model (p4_replay.model_breakdown) is compared with p4c's committed stage depth
and blocks, exactly as src.verify.model_archive does for the calibration
archive.

  python -m scripts.ntask_smoke --campaign <run dir> --out <scratch dir>
"""
import argparse
import os
import sys

from src.p4gen.build_p4_script import (feature_intervals_from_nodes, generate_p4_program,
                                       get_feature_intervals, merge_tree_nodes,
                                       tree_nodes_for)
from src.p4gen.p4_compile import compile_p4
from src.p4gen.p4_ground_truth import committed_blocks, committed_stages_real
from src.p4gen.p4_replay import model_breakdown, parse_program
from src.p4model.program import TaskSpec
from src.p4model.target import TOFINO_PIPELINE_STAGES
from src.training.row_artifacts import load_forests

# (encoding, M token, split, k, split of the third task's forest)
DESIGNS = (
    ('disjoint', '035', 0, 3, 12), ('disjoint', '035', 1, 5, 13),
    ('disjoint', '050', 2, 4, 14), ('disjoint', '025', 3, 6, 15),
    ('disjoint', '075', 4, 3, 16),
    ('joint', '035', 0, 3, 12), ('joint', '035', 1, 5, 13),
    ('joint', '050', 2, 4, 14), ('joint', '025', 3, 6, 15),
    ('joint', '075', 4, 3, 16),
)
ARM = {'disjoint': 'independent', 'joint': 'joint-off'}


def _row(arm, m, split, k):
    return f'{arm}_M{m}_s{split:02d}_k{k:02d}'


def build(run_dir, out_dir, encoding, m, split, k, third_split):
    arm = ARM[encoding]
    base = load_forests(os.path.join(run_dir, 'forests', _row(arm, m, split, k) + '.joblib'))
    extra = load_forests(os.path.join(run_dir, 'forests',
                                      _row(arm, m, third_split, k) + '.joblib'))
    members = [('ddos', 2, base['ddos'], base['features_ddos']),
               ('app', 3, base['app'], base['features_app']),
               ('app2', 3, extra['app'], extra['features_app'])]
    if encoding == 'joint':
        joint = feature_intervals_from_nodes(merge_tree_nodes(
            *(tree_nodes_for(forest, names) for _, _, forest, names in members)))
        intervals = [joint] * 3
    else:
        intervals = [get_feature_intervals(forest, names) for _, _, forest, names in members]
    tasks = [TaskSpec(name, n_classes, list(forest.estimators_), list(names), iv)
             for (name, n_classes, forest, names), iv in zip(members, intervals)]
    design = f'ntask3_{encoding}_M{m}_s{split:02d}_k{k:02d}_x{third_split:02d}'
    path = generate_p4_program(tasks, output_dir=out_dir + os.sep,
                               output_filename=design + '.p4')
    return design, path


def compare(design, path, out_dir):
    model = model_breakdown(parse_program(path), design)
    compile_dir = os.path.join(out_dir, design)
    result = compile_p4(path, compile_dir, timeout_seconds=1800)
    logs = os.path.join(compile_dir, 'pipe', 'logs')
    try:
        p4c_depth = committed_stages_real(logs)
        p4c_tables = committed_blocks(logs) or {}
    except Exception as exc:  # noqa: BLE001 -- reported, not a gate
        return {'design': design, 'error': f'{type(exc).__name__}: {exc}',
                'errors': result.errors}
    priced = [t['table'] for t in model['tables']]
    per_table = {t['table']: t['blocks'] for t in model['tables']}
    return {
        'design': design, 'errors': result.errors,
        'model_stage_depth': model['stage_depth'], 'p4c_stage_depth': p4c_depth,
        'model_blocks': model['blocks'],
        'p4c_blocks': sum(p4c_tables.get(t, 0) for t in priced),
        'tables_exact': sum(per_table[t] == p4c_tables.get(t) for t in priced),
        'tables': len(priced),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for spec in DESIGNS:
        design, path = build(args.campaign, args.out, *spec)
        row = compare(design, path, args.out)
        rows.append(row)
        print(row, flush=True)
    print()
    print('| design | model depth | p4c depth | model blocks | p4c blocks | tables exact |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        if 'error' in r:
            print(f"| {r['design']} | compile failed: {r['error']} |||||")
            continue
        print(f"| {r['design']} | {r['model_stage_depth']} | {r['p4c_stage_depth']} | "
              f"{r['model_blocks']} | {r['p4c_blocks']} | {r['tables_exact']}/{r['tables']} |")
    ok = [r for r in rows if 'error' not in r]
    feasible = [r for r in ok if r['p4c_stage_depth'] <= TOFINO_PIPELINE_STAGES]
    print(f"compiled {len(ok)}/{len(rows)}; stage_depth exact "
          f"{sum(r['model_stage_depth'] == r['p4c_stage_depth'] for r in ok)}/{len(ok)}; "
          f"feasible {len(feasible)}, blocks exact on feasible "
          f"{sum(r['model_blocks'] == r['p4c_blocks'] for r in feasible)}/{len(feasible)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
