"""B3 -- can a spacer put a saturating ternary key at an ODD crossbar offset?
(finding 1.2's discriminating case)

WHY THIS EXISTS. `scripts/tcam_version_sweep.py` tried to test clause (c) of
`version_block_penalty` by MOVING the key itself (`c_solid_shifted`,
`d_ragged_shifted`) and both came back at group 0 -- p4c re-sorts ternary
tables by key width before placing them (`memories.cpp:1800`), and in those
points the two-key program's wider table always went first regardless of
program order, so the key under test was never actually displaced.

This sweep does not try to move the key. It puts a WIDER spacer table in
front of it instead, so THAT wins the width sort and takes group 0, pushing
the key under test to whatever offset the spacer's own width lands on. Only
two of the twelve points are designed to land an ODD offset this way:

    sat11_sp16: (5,)*11 saturating (2 groups) behind a 16-byte spacer
                (3 groups) -- key under test starts at group 3.
    sat22_sp27: (5,)*22 saturating (4 groups) behind a 27-byte spacer
                (5 groups) -- key under test starts at group 5.

The rest of the grid is NOT wasted: every point still compiles a real table
and contributes a (key_bytes, measured_start_group, observed_blocks) triple,
scored observationally rather than assumed, because `start_group_ambiguous`
(a byte sitting exactly on a shared midbyte) can turn even a "designed" point
into a non-reading. Two solid controls (`sat11_solid_sp16`, `sat22_solid_sp27`)
share the exact same byte totals and spacer as their ragged twins and differ
ONLY in whether the key has a nibble-clean byte -- exactly the variable
`version_block_penalty` clause (d)/(a) turns on.

Modelled closely on `scripts/tcam_version_sweep.py`'s `run_point`: same
`compile_p4` call, same `include_path='p4/tofino_spike/common'`, same stale
compile-dir cleanup before each compile, same `size=512` per table. Reuses
`scripts.tcam_stretch_sweep.synthetic_program` (one table per key, `n_a =
n_b = 1`, `@pa_solitary` on every field so a PHV container conflict cannot
masquerade as crossbar geometry) and
`scripts.tcam_version_sweep.read_committed` / `measured_start_group` to read
back the compiler's own committed crossbar byte map.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_spacer_sweep.py

Twelve real p4c compiles; expect tens of minutes. Resumable: points already
present in --out are skipped. A point that fails to compile is recorded with
its error and the sweep continues -- a refused program is a gap in the
sample, not a reason to abort.
"""
import argparse
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_stretch_sweep import as_fields, key_bytes_for, synthetic_program
from scripts.tcam_version_sweep import measured_start_group, read_committed
from src.p4gen.p4_compile import compile_p4
from src.p4model.tables import codeword_bytes_to_blocks, codeword_to_blocks

DEFAULT_OUT = 'results/tcam_spacer_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_spacer_sweep'

# (point_id, saturating key_a, spacer key_b (int bytes), purpose)
#
# key_a is always a tuple of 5-bit fields (ragged, `@pa_solitary`'d one per
# field) except the two solid controls, which are ONE field wide enough to
# match their ragged twin's byte total exactly -- (88,) = 11 bytes like
# (5,)*11, (176,) = 22 bytes like (5,)*22. key_b is always an int: one solid
# spacer field of that many bytes. n_a = n_b = 1 throughout (run_point below).
POINTS = [
    ('sat11_sp5', (5,) * 11, 5,
     'key_a (2 groups) wider than spacer (1 group): key_a sorts first, '
     'expect group 0'),
    ('sat11_sp10', (5,) * 11, 10,
     'key_a (2 groups) and spacer (2 groups) tie in groups; observational'),
    ('sat11_sp16', (5,) * 11, 16,
     'DESIGNED ODD OFFSET: spacer (3 groups) wider than key_a (2 groups) -- '
     'spacer sorts first, key_a expected to start at group 3'),
    ('sat22_sp5', (5,) * 22, 5,
     'key_a (4 groups) wider than spacer (1 group): expect group 0'),
    ('sat22_sp10', (5,) * 22, 10,
     'key_a (4 groups) wider than spacer (2 groups): expect group 0'),
    ('sat22_sp16', (5,) * 22, 16,
     'key_a (4 groups) wider than spacer (3 groups): expect group 0'),
    ('sat22_sp27', (5,) * 22, 27,
     'DESIGNED ODD OFFSET: spacer (5 groups) wider than key_a (4 groups) -- '
     'spacer sorts first, key_a expected to start at group 5'),
    ('sat33_sp5', (5,) * 33, 5,
     'key_a (6 groups) wider than spacer (1 group): expect group 0'),
    ('sat33_sp10', (5,) * 33, 10,
     'key_a (6 groups) wider than spacer (2 groups): expect group 0'),
    ('sat33_sp16', (5,) * 33, 16,
     'key_a (6 groups) wider than spacer (3 groups): expect group 0'),
    ('sat11_solid_sp16', (88,), 16,
     'solid control for sat11_sp16 -- same 11+16 bytes, no ragged field'),
    ('sat22_solid_sp27', (176,), 27,
     'solid control for sat22_sp27 -- same 22+27 bytes, no ragged field'),
]


def run_point(point, output_root, size):
    point_id, key_a, spacer_bytes, purpose = point
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    source = synthetic_program(key_a, 1, spacer_bytes, 1, size=size)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(source)
    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an existing dir; a stale one means a
        # prior attempt never finished.
        shutil.rmtree(compile_dir)
    started = time.time()
    # Same includes the other TCAM probes use -- headers.p4 / util.p4 live
    # here, not in resources/, and compile_p4 copies the include dir into its
    # ASCII WSL scratch alongside the source.
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')
    committed = read_committed(os.path.join(compile_dir, 'pipe', 'logs'))

    fields_a = as_fields(key_a)
    declared_key_bytes_a = key_bytes_for(key_a)
    a_rec = committed.get('tern_a0')
    b_rec = committed.get('tern_b0')

    start, ambiguous = (measured_start_group(a_rec['xbar_bytes'])
                        if a_rec else (None, None))

    # codeword_to_blocks no longer takes an offset (2026-09-20 rewrite design
    # Sec 13.1 retired the mechanism it priced), so both columns are now
    # identical regardless of the measured start group. Kept as two columns
    # rather than restructuring the CSV schema.
    predicted_blocks_at_0 = codeword_to_blocks(fields_a)
    predicted_blocks_at_measured = predicted_blocks_at_0

    return {
        'point_id': point_id,
        'purpose': purpose,
        'fields_a': ','.join(str(b) for b in fields_a),
        'spacer_bytes': key_bytes_for(spacer_bytes),
        'declared_key_bytes_a': declared_key_bytes_a,
        'measured_key_bytes_a': len(a_rec['xbar_bytes']) if a_rec else None,
        'groups_a': codeword_bytes_to_blocks(declared_key_bytes_a),
        'measured_start_group': start,
        'start_group_ambiguous': ambiguous,
        'predicted_blocks_at_0': predicted_blocks_at_0,
        'predicted_blocks_at_measured': predicted_blocks_at_measured,
        'real_blocks_a': a_rec['blocks'] if a_rec else None,
        'real_blocks_b': b_rec['blocks'] if b_rec else None,
        'both_in_one_stage': (a_rec is not None and b_rec is not None
                              and a_rec['stage'] == b_rec['stage']),
        'xbar_bytes_a': ' '.join(str(x) for x in a_rec['xbar_bytes'])
                        if a_rec else '',
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }


def summarize(rows):
    frame = pd.DataFrame(rows)
    print('\n### Non-ambiguous ODD measured_start_group on the saturating key\n')
    if frame.empty:
        print('  (no rows)')
        return
    odd = frame[(frame['measured_start_group'].notna())
               & (frame['start_group_ambiguous'] == False)  # noqa: E712
               & (frame['measured_start_group'] % 2 == 1)]
    print('  %d of %d point(s) landed a non-ambiguous odd offset' %
          (len(odd), len(frame)))
    if len(odd):
        print('\n  point_id, fields_a, key_bytes, measured_start_group, '
              'predicted_blocks_at_measured, real_blocks_a')
        for _, row in odd.iterrows():
            print('    %s: (%s) bytes=%s start=%s predicted=%s real=%s' % (
                row['point_id'], row['fields_a'], row['declared_key_bytes_a'],
                row['measured_start_group'], row['predicted_blocks_at_measured'],
                row['real_blocks_a']))

    print('\n### Solid / ragged control pairs\n')
    pairs = [('sat11_sp16', 'sat11_solid_sp16'),
            ('sat22_sp27', 'sat22_solid_sp27')]
    by_id = {row['point_id']: row for _, row in frame.iterrows()}
    cols = ['fields_a', 'measured_start_group', 'start_group_ambiguous',
           'predicted_blocks_at_measured', 'real_blocks_a']
    for ragged_id, solid_id in pairs:
        print('  %s vs %s:' % (ragged_id, solid_id))
        for pid in (ragged_id, solid_id):
            if pid not in by_id:
                print('    %s: (not yet compiled)' % pid)
                continue
            row = by_id[pid]
            print('    %s: %s' % (pid, {c: row.get(c) for c in cols}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--only', default=None,
                        help='comma-separated point_ids to run')
    args = parser.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)
    done = set()
    rows = []
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        rows = existing.to_dict('records')
        done = set(existing['point_id'])

    wanted = set(args.only.split(',')) if args.only else None
    for point in POINTS:
        if point[0] in done or (wanted and point[0] not in wanted):
            continue
        print('=== %s -- %s' % (point[0], point[3]), flush=True)
        try:
            row = run_point(point, args.output_root, args.size)
        except Exception:                       # a probe must not lose the run
            traceback.print_exc()
            row = {'point_id': point[0], 'purpose': point[3],
                   'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    real_a=%s  real_b=%s  start_group=%s  ambiguous=%s  '
              'predicted_at_measured=%s' % (
                  row.get('real_blocks_a'), row.get('real_blocks_b'),
                  row.get('measured_start_group'),
                  row.get('start_group_ambiguous'),
                  row.get('predicted_blocks_at_measured')),
              flush=True)

    print('\nwrote %s (%d points)' % (args.out, len(rows)))
    summarize(rows)


if __name__ == '__main__':
    main()
