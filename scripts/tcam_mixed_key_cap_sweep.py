"""HISTORICAL: the cap this sweep tested was retired 2026-09-29 (spec
2026-09-29-overlay-and-layout-pragmas-design.md Sec 5.2); the summary now checks
the lane price instead.

E1-lite: does TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62 hold for probe
keys other than the one it was read off?

The cap (src/p4model/target.py) was set from ONE probe, results/
tcam_discount_scan.csv's (84, 84) key: 5 blocks at 60-62 combined crossbar
bytes, 7 at 63-64. This sweep puts five differently-shaped probe keys -- solid,
two-field ragged, four-field, and a 13-field high-k key taken from a real
archived design -- behind a single-field spacer sized so the two keys together
fill 59..64 bytes, and records what p4c charges the probe. The question it
answers is one-sided: does any probe pay MORE than its standalone price at a
combined load the cap still admits (<= 62)? That would be an under-prediction
the cap misses. A probe paying nothing at 63-64 is expected for some shapes and
is the cap's known over-prediction.

Reuses scripts/tcam_offset_scan.py's run_one (spacer = key 'b', probe = key
'a', one table each, one compile per point).

Run (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_mixed_key_cap_sweep.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from scripts.tcam_offset_scan import collect  # noqa: E402
from scripts.tcam_stretch_sweep import key_bytes_for  # noqa: E402
from src.p4model.tables import codeword_to_blocks  # noqa: E402
from src.p4model.packing import _stage_key_prices  # noqa: E402

DEFAULT_OUT = 'results/tcam_mixed_key_cap_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_mixed_key_cap_sweep'
TOTALS = range(59, 65)

# tag -> probe field bit widths
PROBES = {
    'solid22': (176,),                        # 22 B, one solid field
    'rag11': (27, 52),                        # 11 B, independent_low_sd5's ddos key
    'rag14': (54, 56),                        # 14 B, independent_low_sd5's app key
    'four26': (46, 46, 48, 64),               # 26 B, independent_low_sd7's app key
    'hik16': (1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 14),  # 16 B, sd8's ddos key
}


def points_for(tag, fields):
    probe_bytes = key_bytes_for(fields)
    return [('%s_t%d' % (tag, total), total - probe_bytes) for total in TOTALS]


def summarize(out):
    frame = pd.read_csv(out)
    frame['total_bytes'] = frame['spacer_bytes'] + frame['probe_key_bytes']
    frame['extra'] = frame['probe_real_blocks'] - frame['probe_predicted_blocks_at_0']
    cols = ['point_id', 'probe_fields', 'total_bytes', 'both_in_one_stage',
            'probe_predicted_blocks_at_0', 'probe_real_blocks', 'extra']
    print(frame[cols].to_string(index=False))
    shared = frame[frame['both_in_one_stage'] == True]  # noqa: E712
    def lane_price(row):
        # The spacer is one solid field (tcam_offset_scan's key 'b'); the
        # CSV stores probe_fields space-separated (e.g. "54 56").
        spacer = (8 * int(row['spacer_bytes']),)
        probe = tuple(int(x) for x in str(row['probe_fields'])
                      .replace(',', ' ').replace('(', ' ').replace(')', ' ').split())
        prices = _stage_key_prices((), (spacer, probe))
        return None if prices is None else prices[1]
    shared = shared.assign(lane=shared.apply(lane_price, axis=1))
    missed = shared[shared['lane'].isna() | (shared['lane'] < shared['probe_real_blocks'])]
    print('\nshared-stage points the lane price would MISS (under p4c): %d' % len(missed))
    if len(missed):
        print(missed[cols + ['lane']].to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args(argv)
    os.makedirs(args.output_root, exist_ok=True)
    for tag, fields in PROBES.items():
        print('### probe %s %s: %d bytes, %d blocks alone'
              % (tag, fields, key_bytes_for(fields), codeword_to_blocks(fields)),
              flush=True)
        collect(args.out, args.output_root, points_for(tag, fields), fields,
                args.size)
    summarize(args.out)


if __name__ == '__main__':
    main()
