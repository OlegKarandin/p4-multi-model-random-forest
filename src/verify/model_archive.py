"""python -m src.verify --model-archive DIR: the MODEL-side gate on a
calibration archive (spec 2026-10-04 Sec I.5).

`--archive` recompiles every design and compares p4c with the archived p4c
logs -- it tests the ENVIRONMENT and never runs the model. This module is the
model-side counterpart: model_breakdown(parse_program(p4_src/<row>.p4)) against
the archived committed logs (compiles/<row>/pipe/logs). Port of
scratchpad/campaign_2026_10_analysis/oct03/a3_archive.py.

stage_depth is compared on every design; blocks only on feasible ones (p4c
stage_depth <= TOFINO_PIPELINE_STAGES), as the model's total against the sum of
p4c's committed blocks over the tables the model prices. Exit 1 on any miss not
in KNOWN_MISSES with the same numbers, and on any KNOWN_MISSES entry that no
longer occurs. The archive is not in git, so this is a command, not a test."""
import os

from src.p4gen.p4_ground_truth import committed_blocks, committed_stages_real
from src.p4gen.p4_replay import model_breakdown, parse_program
from src.p4model.target import TOFINO_PIPELINE_STAGES

# results/compiler_calibration_pragmas_2026_09_29 under the lane price
# (2026-10-04): stage_depth 70/73 (feasible 60/60), blocks 59/60.
# row -> {metric: (model, p4c)}. See CLAUDE.md "Accuracy -- current gates".
KNOWN_MISSES = {
    # Infeasible either way (> 12 stages); listed in CLAUDE.md.
    'heldout_independent_M150_k15_s16': {'stage_depth': (14, 13)},
    'heldout_independent_M250_k14_s12': {'stage_depth': (15, 16)},
    'independent_high_sd12': {'stage_depth': (13, 14)},
    # The lane price's accepted greedy miss: ddos key (27, 52), 2 per tree vs
    # p4c's 3, three trees (golden fixture known_findings 'lane_price_2026_10').
    'independent_low_sd5': {'blocks': (13, 16)},
}


def _design(archive_dir, row, known):
    model = model_breakdown(parse_program(os.path.join(archive_dir, 'p4_src', row + '.p4')), row)
    logs = os.path.join(archive_dir, 'compiles', row, 'pipe', 'logs')
    p4c_depth = committed_stages_real(logs)
    feasible = p4c_depth <= TOFINO_PIPELINE_STAGES
    priced = [t['table'] for t in model['tables']]
    p4c_tables = committed_blocks(logs) or {}
    p4c_blocks = sum(p4c_tables.get(t, 0) for t in priced)
    misses = {}
    if model['stage_depth'] != p4c_depth:
        misses['stage_depth'] = (model['stage_depth'], p4c_depth)
    if feasible and model['blocks'] != p4c_blocks:
        misses['blocks'] = (model['blocks'], p4c_blocks)
    listed = known.get(row, {})
    unexpected = {m: v for m, v in misses.items() if listed.get(m) != v}
    return {'row': row, 'feasible': feasible,
            'model_stage_depth': model['stage_depth'], 'p4c_stage_depth': p4c_depth,
            'model_blocks': model['blocks'], 'p4c_blocks': p4c_blocks,
            'misses': misses, 'unexpected': unexpected}


def check(archive_dir, known_misses=None):
    """One result dict per p4_src/<row>.p4; prints a per-design line for every
    miss and the totals."""
    known = KNOWN_MISSES if known_misses is None else known_misses
    rows = sorted(n[:-3] for n in os.listdir(os.path.join(archive_dir, 'p4_src'))
                  if n.endswith('.p4'))
    results = [_design(archive_dir, row, known) for row in rows]
    for r in results:
        for metric, (model, p4c) in sorted(r['misses'].items()):
            tag = 'UNEXPECTED' if metric in r['unexpected'] else 'known'
            kind = 'over' if model > p4c else 'under'
            print(f"{r['row']}: {metric} model {model} p4c {p4c} ({kind}, {tag})")
    feasible = [r for r in results if r['feasible']]
    print(f"designs {len(results)} feasible {len(feasible)}; stage_depth exact all "
          f"{sum('stage_depth' not in r['misses'] for r in results)}/{len(results)}, "
          f"feasible {sum('stage_depth' not in r['misses'] for r in feasible)}/{len(feasible)}; "
          f"blocks exact feasible {sum('blocks' not in r['misses'] for r in feasible)}/{len(feasible)}")
    return results


def main_exit_code(results, known_misses=None):
    """1 on any unexpected miss or stale KNOWN_MISSES entry, else 0."""
    known = KNOWN_MISSES if known_misses is None else known_misses
    seen = {r['row']: r['misses'] for r in results}
    stale = [(row, metric) for row, metrics in sorted(known.items())
             for metric, value in sorted(metrics.items())
             if seen.get(row, {}).get(metric) != value
             and not any(r['row'] == row and metric in r['unexpected'] for r in results)]
    for row, metric in stale:
        print(f"{row}: stale known miss ({metric} {known[row][metric]} no longer occurs)")
    return 1 if stale or any(r['unexpected'] for r in results) else 0
