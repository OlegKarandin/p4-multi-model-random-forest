"""Does FIELD COUNT really decide whether a non-isolatable nibble tail pays?

WHY THIS EXISTS. `scripts/tcam_phv_slice_sweep.py` measured, over 24 compiles
at B = 11 bytes / g = 2 / S = 1, exactly when p4c splits a field's 1-4
leftover bits off onto a byte group (free) rather than parking a whole byte
there (pays). Every one of those points had TWO fields. Two 3+-field keys
measured elsewhere contradict that table:

  (47, 51, 71)  22 bytes, only clean field 51 bits -> NOT isolatable by the
                measured table, yet p4c charged nothing (4 blocks, not 5).
  (4, 6, 67)    11 bytes, same shape, also free.

The candidate rule therefore carries a FITTED discontinuity: apply the
measured isolation table for keys of 1-2 fields, assume always-isolatable for
3+. That is the weakest part of the model and this sweep exists to break it.

THE DESIGN. Hold EVERYTHING fixed except the number of fields:

  * B = 11 crossbar bytes, g = 2 groups, exactly saturating (one byte must
    ride a byte group, so the charge is decided by whether that byte is a
    nibble or a whole byte -- the cleanest possible setting);
  * exactly ONE nibble-clean field (S = 1), at a width the isolation table
    calls NOT isolatable -- either 28 bits (leftover in container byte 3) or
    51 bits (leftover in container byte 2, two containers);
  * every partner field a whole number of BYTES (bits % 8 == 0), so no
    partner can contribute a second nibble candidate and S stays 1;
  * the partners' 7 or 4 bytes split into 1, 2, 3 or 4 fields.

PREDICTIONS, which differ on every point but the n = 2 controls:
  * fitted branch (3+ fields always isolatable) -> n = 2 pays, n >= 3 free.
  * unified isolation table (no field-count branch) -> ALL points pay.

If the n >= 3 points pay, the field-count branch is refuted and the isolation
table generalises -- which would then make `(47, 51, 71)` and `(4, 6, 67)`
the anomalies needing a different explanation (they differ from these points
in B/g, not in field count). If they come back free, field count really is
the discriminator at fixed B and g, and the branch stands on 8 more compiles
than the 2 it currently rests on.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_field_count_sweep.py

Ten real p4c compiles. Resumable: points already present in --out are skipped.
"""
import argparse
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_phv_slice_sweep import read_bfa_table, read_phv_container
from scripts.tcam_stretch_sweep import key_bytes_for, synthetic_program
from scripts.tcam_version_sweep import read_committed
from src.p4gen.p4_compile import compile_p4
from src.p4model.tables import codeword_to_blocks

DEFAULT_OUT = 'results/tcam_field_count_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_field_count_sweep'

# (point_id, fields, note). Every key is 11 crossbar bytes, S = 1, and its
# one clean field is NOT isolatable by the phv-slice table.
POINTS = [
    # --- clean field 28 bits (4 bytes, leftover in container byte 3) -------
    ('t28_n2', (28, 56), 'control, already measured as w028: must PAY'),
    ('t28_n3', (28, 32, 24), '3 fields, partners 4 + 3 bytes'),
    ('t28_n4', (28, 24, 16, 16), '4 fields, partners 3 + 2 + 2 bytes'),
    ('t28_n5', (28, 16, 16, 16, 8), '5 fields, partners 2 + 2 + 2 + 1 bytes'),
    # --- clean field 51 bits (7 bytes, leftover in container byte 2, multi) -
    ('t51_n2', (51, 32), 'control, already measured as w051: must PAY'),
    ('t51_n3', (51, 16, 16), '3 fields, partners 2 + 2 bytes'),
    ('t51_n4', (51, 16, 8, 8), '4 fields, partners 2 + 1 + 1 bytes'),
    ('t51_n5', (51, 8, 8, 8, 8), '5 fields, partners 1 + 1 + 1 + 1 bytes'),
    # --- does a 3-field key pay when its partners are LARGE? ---------------
    # (47, 51, 71) is free at 22 bytes; this is its 11-byte analogue, to
    # separate "3 fields" from "22 bytes / 4 groups".
    ('t51_n3_big', (51, 24, 8), '3 fields, partners 3 + 1 bytes'),
    ('t28_n3_alt', (28, 40, 16), '3 fields, partners 5 + 2 bytes'),
]


def run_point(point_id, fields, note, output_root, size):
    field_widths = {'key_a%d' % i: b for i, b in enumerate(fields)}
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
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

    logs = os.path.join(compile_dir, 'pipe', 'logs')
    committed = read_committed(logs)
    rec = committed.get('tern_a0')
    bfa = read_bfa_table(os.path.join(compile_dir, 'pipe', 'prog.bfa'),
                         field_widths)

    row = {
        'point_id': point_id,
        'note': note,
        'fields': ','.join(str(b) for b in fields),
        'n_fields': len(fields),
        'clean_field_bits': fields[0],
        'key_bytes': key_bytes_for(fields),
        'pred_fitted_branch': 2 if len(fields) > 2 else 3,
        'pred_unified_isolation': 3,
        'pred_current': codeword_to_blocks(fields),
        'real_blocks': rec['blocks'] if rec else None,
        'phv_containers': read_phv_container(logs),
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }
    row.update(bfa)
    return row


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        print('\n(no rows)')
        return
    print('\n### Field-count sweep -- B = 11 bytes, g = 2, S = 1, '
          'clean field NOT isolatable\n')
    cols = ['point_id', 'fields', 'n_fields', 'byte_group_min_bits',
           'byte_group_holds_whole_byte', 'match_lines', 'version_only_blocks',
           'real_blocks', 'pred_fitted_branch', 'pred_unified_isolation']
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))

    done = frame[frame['real_blocks'].notna()]
    if done.empty:
        return
    print('\n### Verdict\n')
    for name, col in (('fitted branch (field count matters)', 'pred_fitted_branch'),
                      ('unified isolation (no branch)', 'pred_unified_isolation')):
        exact = (done[col] == done['real_blocks']).sum()
        print('  %-38s %d/%d exact' % (name, exact, len(done)))
    by_n = done.groupby('n_fields')['real_blocks'].apply(
        lambda v: '/'.join(str(int(x)) for x in v))
    print('\n  blocks by field count:')
    for n, v in by_n.items():
        print('    n=%d -> %s' % (n, v))


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

    for point_id, fields, note in POINTS:
        if point_id in done:
            continue
        print('=== %s -- %s %s' % (point_id, fields, note), flush=True)
        try:
            row = run_point(point_id, fields, note, args.output_root, args.size)
        except Exception:
            traceback.print_exc()
            row = {'point_id': point_id, 'fields': ','.join(str(b) for b in fields),
                   'n_fields': len(fields), 'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    byte_group=%s  whole_byte=%s  REAL=%s  (fitted says %s, '
              'unified says 3)' % (
                  row.get('byte_group_occupants'),
                  row.get('byte_group_holds_whole_byte'),
                  row.get('real_blocks'), row.get('pred_fitted_branch')),
              flush=True)

    print('\nwrote %s (%d points)' % (args.out, len(rows)))
    summarize(rows)


if __name__ == '__main__':
    main()
