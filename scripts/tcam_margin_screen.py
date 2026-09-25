"""Find and compile REAL campaign designs where the stage-sharing margin or the
mixed-key byte cap can act, and score the model on them.

WHY. Both one-sided placement rules in src/p4model/packing.py -- the +1 margin
on a saturated, not-first key and the 62-byte mixed-key cap -- are supported
by one archived design (independent_low_sd9) plus synthetic sweeps. Neither
the fitted archive nor the held-out one (results/compiler_calibration_extra)
contains a real stage where they fire. This script looks for campaign designs
that CAN trigger them and compiles them.

Stage 1 (screen, no compiles): refit independent-arm campaign rows (the only
arm with two distinct keys), price each task's classification key with the
current model, and keep rows where the two keys could share a stage
(B_app + B_ddos <= 64) AND either key is saturated
(crossbar_capacity(g) == B) or the pair sits near the cap (>= 56 bytes).

Stage 2 (--compile N): compile the first N screened rows that the model calls
feasible, through compiler_calibration.run_one_row, into
results/tcam_margin_screen/. Stage 3 runs automatically after: the
end-to-end replay (p4_artifact_replay.replay_design) against p4c's real
stages/blocks, and a per-stage account of which keys p4c actually co-located.

Run (from the repository root):
  PYTHONPATH=. python scripts/tcam_margin_screen.py --limit 80
  PYTHONPATH=. python scripts/tcam_margin_screen.py --compile 12
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from scripts.compiler_calibration import (  # noqa: E402
    CAMPAIGN_BACKUP_DIR, add_group_and_band, run_one_row)
from scripts.p4_artifact_replay import _p4_table_keys, replay_design  # noqa: E402
from scripts.replay_alignment import load_backup, refit_pair  # noqa: E402
from scripts.tcam_version_sweep import read_committed  # noqa: E402
from src.main import load_campaign_data  # noqa: E402
from src.p4gen.build_p4_script import get_feature_intervals  # noqa: E402
from src.p4gen.evaluation import multi_model_memory_evaluation  # noqa: E402
from src.p4model.tables import (  # noqa: E402
    codeword_fields_to_bytes_from_bits, codeword_to_blocks, crossbar_capacity,
    ternary_key_field_bits)
from src.p4model.target import (  # noqa: E402
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE, TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE)

SCREEN_OUT = 'results/tcam_margin_screen.csv'
COMPILE_OUT = 'results/tcam_margin_screen_compiled.csv'
OUTPUT_ROOT = 'results/tcam_margin_screen'
NEAR_CAP_BYTES = 56


def candidate_rows(frame):
    """Independent-arm, feasible rows, deepest designs first (they hold the
    most tables, so sharing is likeliest), deterministic order."""
    frame = add_group_and_band(frame)
    out = frame[frame['group'] == 'independent']
    if 'infeasible' in out.columns:
        out = out[out['infeasible'].isna() | (out['infeasible'] == '')]
    out = out[out['best_params'].notna()]
    return out.sort_values(['stage_depth', 'k', 'M', 'split'],
                           ascending=[False, False, True, True])


def key_facts(bits):
    B = codeword_fields_to_bytes_from_bits(bits)
    g = codeword_to_blocks(bits)
    return B, g, crossbar_capacity(g) == B


def screen(limit):
    data = load_campaign_data()
    frame = candidate_rows(load_backup(CAMPAIGN_BACKUP_DIR))
    records = []
    for idx, (_, row) in enumerate(frame.head(limit).iterrows()):
        model_app, model_ddos, *_ = refit_pair(row, data)
        names_app = row['features_app'].split(';')
        names_ddos = row['features_ddos'].split(';')
        bits_app = ternary_key_field_bits(get_feature_intervals(model_app, names_app))
        bits_ddos = ternary_key_field_bits(get_feature_intervals(model_ddos, names_ddos))
        B_a, g_a, sat_a = key_facts(bits_app)
        B_d, g_d, sat_d = key_facts(bits_ddos)
        usage = multi_model_memory_evaluation(model_app, model_ddos, names_app,
                                              names_ddos, 'disjoint')
        total = B_a + B_d
        can_share = total <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
        record = {
            'row_id': 'margin_%s_M%d_k%d_s%d' % (row['arm_slug'], row['M'], row['k'],
                                                 row['split']),
            'source_index': row.name, 'M': row['M'], 'k': row['k'],
            'split': row['split'],
            'B_app': B_a, 'g_app': g_a, 'sat_app': sat_a,
            'B_ddos': B_d, 'g_ddos': g_d, 'sat_ddos': sat_d,
            'B_total': total, 'can_share': can_share,
            'margin_eligible': can_share and (sat_a or sat_d),
            'near_cap': can_share and total >= NEAR_CAP_BYTES,
            'pred_stage_depth': usage.stage_depth, 'pred_blocks': usage.blocks,
            'n_trees_app': len(model_app.estimators_),
            'n_trees_ddos': len(model_ddos.estimators_),
        }
        records.append(record)
        if len(records) % 25 == 0:          # a long screen must not lose work
            pd.DataFrame(records).to_csv(SCREEN_OUT, index=False)
        print('%3d %-34s B=%2d+%2d sat=%d/%d depth=%2d %s' % (
            idx, record['row_id'], B_a, B_d, sat_a, sat_d, usage.stage_depth,
            'MARGIN' if record['margin_eligible'] else
            ('NEARCAP' if record['near_cap'] else '')), flush=True)
    out = pd.DataFrame(records)
    out.to_csv(SCREEN_OUT, index=False)
    print('\nwrote %s: %d rows, %d margin-eligible, %d near-cap'
          % (SCREEN_OUT, len(out), out['margin_eligible'].sum(), out['near_cap'].sum()))
    return out


def select_for_compile(screened, n):
    """A deterministic, stratified pick of up to n feasible rows:
    half from the contested 59-64-byte band (where the mixed-key cap decides),
    a third with a saturated key at <= 58 bytes (where only the margin acts),
    the rest from the 56-58 edge. Within a stratum, one row per distinct
    (B_app, B_ddos) so the batch spans shapes rather than repeating one."""
    pool = screened[(screened['pred_stage_depth'] <= 12) & screened['can_share']]
    pool = pool.drop_duplicates(['B_app', 'B_ddos'])
    band = pool[(pool['B_total'] >= 59) & (pool['B_total'] <= 64)]
    band = band.sort_values(['B_total', 'row_id'])
    margin = pool[pool['margin_eligible'] & (pool['B_total'] <= 58)]
    margin = margin.sort_values(['B_total', 'row_id'], ascending=[False, True])
    edge = pool[(pool['B_total'] >= 56) & (pool['B_total'] <= 58)
                & ~pool['margin_eligible']].sort_values(['B_total', 'row_id'])
    n_band, n_margin = n // 2, n // 3

    def spread(frame, k):                      # evenly across the sorted list
        if len(frame) <= k:
            return frame
        step = len(frame) / k
        return frame.iloc[[int(i * step) for i in range(k)]]
    picked = pd.concat([spread(band, n_band), spread(margin, n_margin),
                        spread(edge, n - n_band - n_margin)])
    return picked.drop_duplicates('row_id')


def compile_selected(n):
    chosen = select_for_compile(pd.read_csv(SCREEN_OUT), n)
    print(chosen[['row_id', 'B_app', 'B_ddos', 'B_total', 'sat_app', 'sat_ddos',
                  'pred_stage_depth']].to_string(index=False), flush=True)
    data = load_campaign_data()
    source = load_backup(CAMPAIGN_BACKUP_DIR)
    done = set(pd.read_csv(COMPILE_OUT)['row_id']) if os.path.exists(COMPILE_OUT) else set()
    rows = pd.read_csv(COMPILE_OUT).to_dict('records') if done else []
    for _, pick in chosen.iterrows():
        if pick['row_id'] in done:
            continue
        print('=== compiling %s' % pick['row_id'], flush=True)
        try:
            result = run_one_row(pick['row_id'], 'independent',
                                 source.loc[pick['source_index']], data, OUTPUT_ROOT)
        except Exception:
            traceback.print_exc()
            continue
        rows.append(result)
        pd.DataFrame(rows).to_csv(COMPILE_OUT, index=False)


def stage_account(row_id):
    """Per stage: which distinct keys p4c co-located, and per table the
    model's standalone price vs what p4c charged."""
    logs = os.path.join(OUTPUT_ROOT, 'compiles', row_id, 'pipe', 'logs')
    committed = read_committed(logs)
    tables, _, bits = _p4_table_keys(os.path.join(OUTPUT_ROOT, 'p4_src', row_id + '.p4'))
    lines = []
    stages = {}
    for name, rec in committed.items():
        if name.startswith('get_classification_tree_'):
            stages.setdefault(rec['stage'], []).append(name)
    for stage in sorted(stages):
        keys = {tuple(sorted(bits[f] for f in tables[n])) for n in stages[stage]}
        total = sum(codeword_fields_to_bytes_from_bits(k) for k in keys)
        cells = []
        for name in sorted(stages[stage]):
            key = tuple(sorted(bits[f] for f in tables[name]))
            B, g, sat = key_facts(key)
            cells.append('%s:B%d g%d%s real%d' % (name.replace('get_classification_tree_', ''),
                                                  B, g, '*' if sat else '',
                                                  committed[name]['blocks']))
        lines.append('   stage %2d  keys=%d bytes=%2d%s  %s' % (
            stage, len(keys), total,
            ' >CAP' if len(keys) > 1 and total > TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE else '',
            '  '.join(cells)))
    return lines


def score():
    if not os.path.exists(COMPILE_OUT):
        print('nothing compiled yet')
        return
    compiled = pd.read_csv(COMPILE_OUT)
    print('\n### end-to-end: model vs p4c on real designs chosen to exercise the '
          'margin / cap  (* = saturated key)')
    for _, r in compiled.iterrows():
        try:
            depth, blocks = replay_design(r['row_id'], OUTPUT_ROOT)
        except Exception as exc:                    # no program on disk
            print('%s: %s' % (r['row_id'], exc))
            continue
        print('%-34s depth model=%2d p4c=%2s | blocks model=%3d p4c=%s'
              % (r['row_id'], depth, r['stages_real'], blocks, r['tcam_real']))
        for line in stage_account(r['row_id']):
            print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None,
                        help='stage 1: screen this many candidate rows')
    parser.add_argument('--compile', type=int, default=None,
                        help='stage 2: compile this many screened rows')
    args = parser.parse_args(argv)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if args.limit:
        screen(args.limit)
    if args.compile:
        compile_selected(args.compile)
    score()


if __name__ == '__main__':
    main()
