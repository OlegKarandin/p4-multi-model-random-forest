"""python -m src.verify --model-archive DIR: the MODEL-side gate on a
calibration archive (spec 2026-10-04 Sec I.5).

`--archive` recompiles every design and compares p4c with the archived p4c
logs -- it tests the ENVIRONMENT and never runs the model. This module is the
model-side counterpart: model_breakdown(parse_program(p4_src/<row>.p4)) against
the archived committed logs (compiles/<row>/pipe/logs). Port of
scratchpad/campaign_2026_10_analysis/oct03/a3_archive.py.

stage_depth is compared on every design; blocks only on feasible ones (p4c
stage_depth <= TOFINO_PIPELINE_STAGES), as the model's total against the sum of
p4c's committed blocks over the tables the model prices. A program the replay
refuses (an older generator's: @stage pins, gated register blocks, missing pins)
is reported as a miss MODEL_REFUSED:<reason> instead of stopping the check. Exit 1 on any miss not
in KNOWN_MISSES with the same numbers, and on any KNOWN_MISSES entry that no
longer occurs. The archive is not in git, so this is a command, not a test."""
import os

from p4gen.p4_ground_truth import committed_blocks, committed_stages_real
from p4gen.p4_replay import model_breakdown, parse_program
from p4model.target import TOFINO_PIPELINE_STAGES
from src.verify.verdicts import MODEL_REFUSED_PREFIX

# row -> {metric: (model, p4c)}: misses the archive is expected to show.
# EMPTY since Task 18 (2026-11-15). The one calibration archive this gate was
# run on, results/compiler_calibration_pragmas_2026_09_29 (73 designs), holds
# programs generated before decisions 1a / 7.1 option B: the replay refuses
# every one by shape (MODEL_REFUSED:gated_register_block), so it no longer
# scores the model and its old misses (stage_depth 72/73 under the crossbar
# port) are history, not expectations. The whole-program check on programs of
# the CURRENT generator is tests/test_p4c_base_17b.py (12 Task 17b base designs
# recompiled with the G3 pins in Task 23; 11/12 exact, one named miss); a new archive for this command
# needs new compiles.
KNOWN_MISSES = {}


def _design(archive_dir, row, known):
    model = model_breakdown(parse_program(os.path.join(archive_dir, 'p4_src', row + '.p4')), row)
    logs = os.path.join(archive_dir, 'compiles', row, 'pipe', 'logs')
    p4c_depth = committed_stages_real(logs)
    feasible = p4c_depth <= TOFINO_PIPELINE_STAGES
    priced = [t['table'] for t in model['tables']]
    p4c_tables = committed_blocks(logs) or {}
    p4c_blocks = sum(p4c_tables.get(t, 0) for t in priced)
    misses = {}
    if model['stage_depth'] is None:
        # The replay refused the program (p4_replay.model_breakdown): no
        # prediction, one miss named by its reason (MODEL_REFUSED:<reason>).
        misses['refusal'] = MODEL_REFUSED_PREFIX + model['refusal_reason']
    elif model['stage_depth'] != p4c_depth:
        misses['stage_depth'] = (model['stage_depth'], p4c_depth)
    if feasible and model['stage_depth'] is not None and model['blocks'] != p4c_blocks:
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
        for metric, value in sorted(r['misses'].items()):
            tag = 'UNEXPECTED' if metric in r['unexpected'] else 'known'
            if metric == 'refusal':
                print(f"{r['row']}: {value} ({tag})")
                continue
            model, p4c = value
            kind = 'over' if model > p4c else 'under'
            print(f"{r['row']}: {metric} model {model} p4c {p4c} ({kind}, {tag})")
    feasible = [r for r in results if r['feasible']]
    print(f"designs {len(results)} feasible {len(feasible)}; stage_depth exact all "
          f"{sum('stage_depth' not in r['misses'] for r in results)}/{len(results)}, "
          f"feasible {sum('stage_depth' not in r['misses'] for r in feasible)}/{len(feasible)}; "
          f"blocks exact feasible {sum('blocks' not in r['misses'] for r in feasible)}/{len(feasible)}; "
          f"refused by the model {sum('refusal' in r['misses'] for r in results)}")
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
