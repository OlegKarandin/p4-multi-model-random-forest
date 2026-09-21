"""Which nibble-clean bytes can actually reach a midbyte? The PHV-lane test.

WHY THIS EXISTS. M3 (scripts/tcam_ledger_divergence_sweep.py) showed clause
(d) of `tables.version_block_penalty` over-predicts on 11 of 13 fresh
compiles, while the compiler's own Sec 4 ledger -- which has no positional
term at all -- is wrong on only 4 keys out of the 56 distinct keys measured
so far. So neither "any nibble-clean byte can reach any midbyte" (the
ledger) nor "only a contiguous field ordering can place one" (clause d) is
the real constraint. This sweep isolates the constraint.

THE HYPOTHESIS UNDER TEST (the issue draft's own Sec 5.2 footnote, listed as
"mod-4 lane modelling" in the 2026-09-20 findings doc's step 5). A crossbar
byte inherits its PHV container lane: a byte from lane L of a 32-bit
container can only be delivered to crossbar positions congruent to L mod 4;
from a 16-bit container, L mod 2; from an 8-bit container, anywhere.
Midbytes sit at crossbar positions 11i + 5 -> 5, 16, 27, 38, 49, 60, whose
residues mod 4 are 1, 0, 3, 2, 1, 0. So whether a key's nibble-clean byte can
reach a midbyte depends on WHERE IN ITS FIELD that byte falls, not on how the
fields could be ordered.

THE DESIGN. Every point is the same size and shape -- B = 11 crossbar bytes,
g = 2 groups, exactly saturating, exactly one nibble-clean field (S = 1), one
midbyte (crossbar position 5, residue 1). The ONLY thing that varies is where
the clean byte sits inside its own field:

    field A = 8a + 4 bits  (a + 1 bytes, its last byte nibble-clean)
    field B = 8(10 - a) bits (10 - a bytes, solid, no clean byte)

for a = 0..9. `a` walks the clean byte through every container lane:

    a = 0  -> 4 bits          -> 8-bit container,  unrestricted
    a = 1  -> 12 bits         -> 16-bit container, lane 1 -> odd positions
    a = 2  -> 20 bits         -> 32-bit container, lane 2 -> pos = 2 mod 4
    a = 3  -> 28 bits         -> 32-bit container, lane 3 -> pos = 3 mod 4
    a = 4  -> 36 bits = 32+4  -> 8-bit tail,       unrestricted
    ... and so on with period 4.

PREDICTIONS, which differ on 5 of the 10 points:
  * plain ledger  -> no penalty anywhere (S = 1 everywhere).
  * clause (d)    -> penalty everywhere EXCEPT a = 5 (the only ordering that
                     lands the clean byte on slot 5).
  * lane model    -> penalty exactly where the lane cannot reach position 5,
                     i.e. a = 2, 3, 6, 7 (residue 2 and 3), and free at
                     a = 0, 1, 4, 5, 8, 9.

Two archived keys already sit on this axis and agree with the lane model:
`(44, 40, 88)` is a = 5 (free, measured 4 blocks) and
`independent_low_sd5`'s `(52, 27)` is a = 6 (pays, measured 3 blocks).

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_lane_sweep.py

Ten real p4c compiles, a few seconds each. Resumable: points already present
in --out are skipped.
"""
import argparse
import math
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_ledger_divergence_sweep import ledger_blocks
from scripts.tcam_stretch_sweep import key_bytes_for, synthetic_program
from scripts.tcam_version_sweep import measured_start_group, read_committed
from src.p4gen.p4_compile import compile_p4
from src.p4model.tables import codeword_to_blocks

DEFAULT_OUT = 'results/tcam_lane_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_lane_sweep'

TOTAL_BYTES = 11          # exactly saturating at g = 2 (5 + midbyte + 5)
MIDBYTE_POSITION = 5      # the only full midbyte a 2-group run from 0 owns


def container_and_lane(bits):
    """(container_bits, lane) the nibble-clean TAIL byte of a `bits`-wide
    field lands in, under the simplest defensible PHV model: 32-bit
    containers are filled first and the remainder goes in the smallest
    container that holds it.

    This is the model under test, not an established fact -- the sweep exists
    to accept or reject it."""
    tail = bits % 32 or 32
    container = 8 if tail <= 8 else (16 if tail <= 16 else 32)
    lane = math.ceil(tail / 8) - 1
    return container, lane


def lane_can_reach(bits, position):
    container, lane = container_and_lane(bits)
    if container == 8:
        return True                      # an 8-bit container byte goes anywhere
    if container == 16:
        return position % 2 == lane % 2
    return position % 4 == lane % 4


def points():
    out = []
    for a in range(0, 10):
        clean_bits = 8 * a + 4                    # a + 1 bytes, tail nibble
        solid_bits = 8 * (TOTAL_BYTES - (a + 1))  # the rest, no clean byte
        out.append(('lane_a%d' % a, (clean_bits, solid_bits), a))
    return out


def run_point(point_id, fields, a, output_root, size):
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    # A lone table: its crossbar run starts at group 0 by construction, so
    # the single full midbyte is at position MIDBYTE_POSITION.
    source = synthetic_program(fields, 1, (8,), 0, size=size)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(source)
    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        shutil.rmtree(compile_dir)
    started = time.time()
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')
    committed = read_committed(os.path.join(compile_dir, 'pipe', 'logs'))
    rec = committed.get('tern_a0')
    start, ambiguous = (measured_start_group(rec['xbar_bytes'])
                        if rec else (None, None))

    clean_bits = fields[0]
    container, lane = container_and_lane(clean_bits)
    lane_free = lane_can_reach(clean_bits, MIDBYTE_POSITION)
    # Two groups with no version charge is 2 blocks; a charge makes it 3.
    lane_pred = 2 if lane_free else 3

    return {
        'point_id': point_id,
        'a': a,
        'fields': ','.join(str(b) for b in fields),
        'key_bytes': key_bytes_for(fields),
        'clean_field_bits': clean_bits,
        'container_bits': container,
        'lane': lane,
        'lane_reaches_midbyte': lane_free,
        'pred_lane_model': lane_pred,
        'pred_ledger': ledger_blocks(fields),
        'pred_current': codeword_to_blocks(fields, 0),
        'real_blocks': rec['blocks'] if rec else None,
        'measured_start_group': start,
        'start_group_ambiguous': ambiguous,
        'xbar_bytes': ' '.join(str(x) for x in rec['xbar_bytes']) if rec else '',
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        print('\n(no rows)')
        return
    print('\n### Lane sweep -- B = 11 bytes, g = 2, S = 1, midbyte at position 5\n')
    cols = ['point_id', 'a', 'fields', 'clean_field_bits', 'container_bits',
           'lane', 'lane_reaches_midbyte', 'pred_ledger', 'pred_current',
           'pred_lane_model', 'real_blocks']
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))

    done = frame[frame['real_blocks'].notna()]
    if done.empty:
        return
    for name, col in (('ledger', 'pred_ledger'), ('current rule', 'pred_current'),
                      ('lane model', 'pred_lane_model')):
        exact = (done[col] == done['real_blocks']).sum()
        print('  %-14s %d/%d exact' % (name, exact, len(done)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)
    done, rows = set(), []
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        rows = existing.to_dict('records')
        done = set(existing['point_id'])

    for point_id, fields, a in points():
        if point_id in done:
            continue
        print('=== %s -- fields=%s' % (point_id, fields), flush=True)
        try:
            row = run_point(point_id, fields, a, args.output_root, args.size)
        except Exception:
            traceback.print_exc()
            row = {'point_id': point_id, 'a': a, 'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    ledger=%s current=%s lane=%s  REAL=%s' % (
            row.get('pred_ledger'), row.get('pred_current'),
            row.get('pred_lane_model'), row.get('real_blocks')), flush=True)

    print('\nwrote %s (%d points)' % (args.out, len(rows)))
    summarize(rows)


if __name__ == '__main__':
    main()
