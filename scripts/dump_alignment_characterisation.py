"""Dumps tests/fixtures/alignment_characterisation.json: what threshold
alignment currently DOES, so the 2026-09-07 cost-model repair's behavioural
delta is visible step by step instead of silent.

WHY THIS EXISTS. Design §6: nothing currently fails -- the alignment code is
self-consistent, it is simply consistent with a superseded cost model. So the
repair cannot be driven by a red test, and without a record of current output
each step's effect on real forests would be invisible.

THIS FIXTURE IS A RECORD, NOT A REQUIREMENT. Unlike
tests/fixtures/resource_model_golden.json (a refactor anchor that must never
change), this one is EXPECTED to change: the gate repair and the feature-order
change both alter what alignment accepts. Regenerate it in the same commit as
the change and explain the diff in the commit message. A diff that cannot be
explained is the bug this fixture exists to catch.

The capture function lives here and is imported by the replay test, so the
dumper and the checker cannot drift apart.

Run (from the repository root; ~1 minute -- 6 alignment runs over a 7-tree
pair, no Optuna search and no p4c):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/dump_alignment_characterisation.py
"""
import json
import os

# The configurations. delta_rel spans the three regimes that behave
# differently: 0.0 accepts only free moves, 0.05 spends a real tolerance, and
# None is the accept-everything anchor that skips accuracy evaluation entirely.
# Both overlap thresholds are swept because the 2026-08-31 spike measured
# overlap_threshold, not delta, as the binding gate on candidate supply.
CONFIGS = [
    {'delta_rel': 0.0, 'overlap_threshold': 0.5},
    {'delta_rel': 0.05, 'overlap_threshold': 0.5},
    {'delta_rel': None, 'overlap_threshold': 0.5},
    {'delta_rel': 0.0, 'overlap_threshold': 0.25},
    {'delta_rel': 0.05, 'overlap_threshold': 0.25},
    {'delta_rel': None, 'overlap_threshold': 0.25},
]

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, 'tests', 'fixtures',
                       'alignment_characterisation.json')


def capture():
    """Every configuration's full result, as JSON-safe plain data."""
    from src.training import threshold_alignment as ta
    from tests.test_threshold_alignment import _golden_alignment_pair

    rows = []
    for config in CONFIGS:
        rf1, X1, y1, rf2, X2, y2 = _golden_alignment_pair()
        stats, log = {}, []
        a1, a2 = ta.align_rf_thresholds(
            rf1, rf2, X1, y1, X2, y2,
            overlap_threshold=config['overlap_threshold'],
            delta_rel=config['delta_rel'], align_stats=stats, candidate_log=log)
        rows.append({
            'config': config,
            # dt_thresholds_float_to_int ran inside _golden_alignment_pair, so
            # every threshold is an integer and the arrays serialise exactly --
            # no float round-tripping to argue about.
            'thresholds': [[int(t) for t in est.tree_.threshold]
                           for est in list(a1.estimators_) + list(a2.estimators_)],
            'stats': {k: (None if v is None else
                          (bool(v) if isinstance(v, bool) else
                           (float(v) if isinstance(v, float) else v)))
                      for k, v in sorted(stats.items())},
            # Only the ACCEPTED candidates: a rejected one restores every
            # structure it touched, so it cannot influence the result, and the
            # full log is thousands of rows of diagnostic noise.
            'accepted': [{'feature_idx': e['feature_idx'], 'round': e['round'],
                          'range1': list(e['range1']), 'range2': list(e['range2']),
                          'target': list(e['target'])}
                         for e in log if e['accepted']],
        })
    return {'rows': rows}


def main():
    data = capture()
    with open(FIXTURE, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.write('\n')
    for row in data['rows']:
        print('delta={!r:>5} overlap={:g}  accepted={:<4} blocks {}->{} (floor {})'.format(
            row['config']['delta_rel'], row['config']['overlap_threshold'],
            len(row['accepted']), row['stats']['blocks_before'],
            row['stats']['blocks_after'], row['stats']['blocks_floor']))
    print('wrote', os.path.normpath(FIXTURE))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
