"""Score an alignment-objective replay against the pre-registered criteria
E2-E7 (design 2026-08-30 §5).

Separate from src/reporting/replay_scoring.py on purpose: S2-S5 there are
pre-registered against the BLOCK domain, and re-pointing "best" at a run
optimising a different resource would invalidate them (§2.8). This module
scores the objective axis instead and leaves that one objective-blind.

EVERY criterion is PAIRED per model pair via an inner merge on PAIR_KEYS, for
the same reason the replay exists at all: pooling across pairs would
reintroduce the search-noise confound the harness removes. A pair missing one
objective is DROPPED, never averaged around, and every verdict reports its
surviving cell count first.

E1/E1b/E1c are NOT scored here. They are premises about arithmetic, pinned as
unit tests (tests/test_align_budget.py, tests/test_threshold_alignment.py);
a replay cannot fail them without the test suite failing first.

Run: PYTHONPATH=. python scripts/score_objective_replay.py results/replay_objective_20260830.csv
"""
import argparse
import sys

import pandas as pd

# One model pair. overlap_threshold is part of the identity because the same
# refit is replayed once per threshold.
PAIR_KEYS = ['source_arm', 'M', 'split', 'k', 'overlap_threshold']

# One flipped validation sample at n ~ 3000 is ~0.0003, so -0.001 reads as
# "no worse than noise" rather than "no worse at all". Same constant, same
# reasoning, as replay_scoring.ACCURACY_TOLERANCE.
ACCURACY_TOLERANCE = -0.001

# §5's ceiling on campaign_backup_20260825's models: 16 stages across 25 cells
# with perfect targeting at delta=inf. "Anything at or above half of that is a
# strong result for this mechanism." NOTE (§5.1): this expires with the
# campaign rerun -- it is a property of the OLD archive's fitted models, and
# must be recomputed before it can score anything on new ones.
CEILING_STAGES_PER_CELL = 0.64


def _paired(frame, objective_a, objective_b):
    """Rows of two objectives merged on PAIR_KEYS, suffixed _a / _b."""
    left = frame[frame['objective'] == objective_a]
    right = frame[frame['objective'] == objective_b]
    return left.merge(right, on=PAIR_KEYS, suffixes=('_a', '_b'))


def _verdict(n, passed, value, detail):
    return {'passed': bool(passed), 'value': float(value), 'n_cells': int(n),
            'detail': '{} cells paired: {}'.format(n, detail)}


def score_objectives(frame, n_tables=None):
    """E2-E7, per cell, with n. Never replaced by a pooled verdict.

    n_tables, when given, is used by E7 to recompute what a T-blind
    stage_step_target would have proposed. When absent E7 falls back to the
    recorded align_ternary_stages_before, which is enough to identify the
    zero-payoff cells (stages already 1) but not the general case.
    """
    from src.training.align_budget import (stage_step_target, tables_per_stage,
                                           TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)

    aligned = frame[frame['policy'] == 'aligned'].copy()
    out = {}

    # E2 -- the headline: cells where 'stages' lowers stage_depth vs 'blocks'.
    pair = _paired(aligned, 'stages', 'blocks')
    if len(pair):
        improved = pair['joint_stage_depth_b'] - pair['joint_stage_depth_a']
        won = int((improved > 0).sum())
        detail = '{} of {} cells save a stage, {} total, {:.2f}/cell ' \
                 '(ceiling {:.2f}/cell)'.format(
                     won, len(pair), int(improved.clip(lower=0).sum()),
                     improved.clip(lower=0).mean(), CEILING_STAGES_PER_CELL)
        if won == 0:
            detail += ' -- FALSIFIED: the mechanism is unreachable by ' \
                      'threshold relocation at this data\'s capacities. Not ' \
                      'repairable by loosening delta; the floors bound it ' \
                      'regardless of delta.'
        out['E2'] = _verdict(
            len(pair),
            improved.clip(lower=0).mean() >= CEILING_STAGES_PER_CELL / 2,
            won, detail)
    else:
        out['E2'] = _verdict(0, False, 0, 'no stages/blocks pairs')

    # E3 -- what the reordering cost in accuracy, per task, per cell.
    if len(pair):
        d_app = (pair['acc_app_a'] - pair['acc_app_b'])
        d_ddos = (pair['acc_ddos_a'] - pair['acc_ddos_b'])
        worst = float(min(d_app.mean(), d_ddos.mean()))
        out['E3'] = _verdict(len(pair), worst >= ACCURACY_TOLERANCE, worst,
                             'mean d_acc app {:+.4f} ddos {:+.4f}; worst cell '
                             'app {:+.4f} ddos {:+.4f}'.format(
                                 d_app.mean(), d_ddos.mean(),
                                 d_app.min(), d_ddos.min()))
    else:
        out['E3'] = _verdict(0, False, 0.0, 'no stages/blocks pairs')

    # E4 -- THE REAL RISK. Byte-first ordering spends the budget on different
    # features and could starve the moves that were crossing block bands. A
    # result that trades 1 stage for 3 blocks is a trade-off to REPORT.
    if len(pair):
        cost = pair['joint_blocks_a'] - pair['joint_blocks_b']
        out['E4'] = _verdict(len(pair), cost.max() <= 0, float(cost.max()),
                             'blocks under stages minus under blocks: mean '
                             '{:+.2f}, worst cell {:+.0f}'.format(
                                 cost.mean(), cost.max()))
    else:
        out['E4'] = _verdict(0, False, 0.0, 'no stages/blocks pairs')

    # E5 -- calibrates the bits_to_reach lower bound against what was actually
    # shed, where the stage route was attempted. Decides whether a later design
    # may GATE on the bound. Reported, never enforced.
    attempted = aligned[(aligned['objective'] == 'stages')
                        & aligned['align_bits_to_reach'].notna()]
    if len(attempted):
        shed = attempted['align_codeword_before'] - attempted['align_codeword_after']
        bound = attempted['align_bits_to_reach']
        sufficient = (shed >= bound)
        reached = (attempted['align_ternary_stages_after']
                   < attempted['align_ternary_stages_before'])
        out['E5'] = _verdict(
            len(attempted), True, float((sufficient & reached).sum()),
            'shed >= bound in {}/{} cells; of those, {} actually crossed a '
            'step (mean shed {:.1f} bits vs mean bound {:.1f})'.format(
                int(sufficient.sum()), len(attempted),
                int((sufficient & reached).sum()), shed.mean(), bound.mean()))
    else:
        out['E5'] = _verdict(0, False, 0.0, 'the stage route was never live')

    # E6 -- do the two orders reach different end states? Gates whether the
    # deferred multi-arm design (§6) is ever worth writing.
    pair6 = _paired(aligned, 'both', 'stages')
    if len(pair6):
        differ = ((pair6['align_key_bytes_after_a'] != pair6['align_key_bytes_after_b'])
                  | (pair6['align_codeword_after_a'] != pair6['align_codeword_after_b']))
        out['E6'] = _verdict(len(pair6), True, int(differ.sum()),
                             '{} of {} cells reach different end states under '
                             "'both' vs 'stages'".format(int(differ.sum()), len(pair6)))
    else:
        out['E6'] = _verdict(0, False, 0, 'no both/stages pairs')

    # E7 -- how many cells would a T-BLIND stage_step_target have spent
    # accuracy in for a target that provably cannot change depth? Scored
    # against §1.5's own population: 18 of 25 cells save nothing at the floor.
    rows = aligned[aligned['objective'] == 'stages']
    if len(rows) and n_tables is not None:
        wasteful = 0
        for _, row in rows.iterrows():
            key_bytes = int(row['align_key_bytes_before'])
            floor = int(row['align_key_bytes_floor'])
            blind = TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE // (
                tables_per_stage(key_bytes) + 1)
            aware = stage_step_target(key_bytes, n_tables)
            if blind >= floor and (aware is None or aware < floor):
                wasteful += 1
        out['E7'] = _verdict(len(rows), True, wasteful,
                             '{} of {} cells where a T-blind target would '
                             'spend and the T-aware one declines'.format(
                                 wasteful, len(rows)))
    elif len(rows):
        already_one = int((rows['align_ternary_stages_before'] <= 1).sum())
        out['E7'] = _verdict(len(rows), True, already_one,
                             '{} of {} cells already at one stage (pass '
                             '--n-tables for the full T-blind comparison)'.format(
                                 already_one, len(rows)))
    else:
        out['E7'] = _verdict(0, False, 0, 'no stages rows')

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--n-tables', type=int, default=None,
                        help='classification tables (trees, both models) for '
                             "E7's T-blind comparison")
    args = parser.parse_args(argv)

    verdicts = score_objectives(pd.read_csv(args.csv), n_tables=args.n_tables)
    for name in sorted(verdicts):
        v = verdicts[name]
        print('{}  {}  value={:g}  {}'.format(
            name, 'PASS' if v['passed'] else 'FAIL', v['value'], v['detail']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
