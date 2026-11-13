"""M3 -- the divergence-class sweep (docs/2026-09-20-version-block-findings.md).

WHY THIS EXISTS. N4 of that session found that our rule
(`tables.codeword_to_blocks`) never prices a key below the compiler's own
arithmetic ledger (the issue draft's Sec 4:
`overflow = max(0, B - 5g); nibbles = 2*overflow - min(overflow, S);
feasible(g) <=> nibbles + 1 <= g`), but is STRICTER than that ledger on 7.9%
of 200,000 random field-width tuples. On the one REAL example of that class
in the calibration archive -- `(27, 52)` -- our rule is right and the ledger
is wrong (ledger says 2 blocks, p4c emits 3). But that is a sample size of
one. This sweep manufactures 10-15 more divergence-class keys (saturated,
>=1 nibble-clean field, no nibble-clean byte reaches a full midbyte at
offset 0) and compiles each ALONE (so its crossbar run starts at group 0 by
construction -- no spacer, no offset dependence, no reliance on M1) to see
whether the real compiler keeps agreeing with us or was just being generous
on `(27, 52)`.

Reuses `scripts.tcam_stretch_sweep.synthetic_program`/`key_bytes_for`, and
`tcam_version_sweep.read_committed`/`measured_start_group` to read back the
compiler's own crossbar byte map (so a compile that does NOT actually land at
group 0 -- p4c is free to choose -- is reported rather than assumed).

The point generator (`is_divergence_point` / `find_divergence_points`)
called `tables.version_block_penalty`, deleted on 2026-09-21 with the
offset-taking `codeword_to_blocks` it interrogated, so `main()`'s default path
raised NameError. It was removed on 2026-11-13 (see git history); the 13
collected points live in the gitignored results/tcam_ledger_divergence_sweep.csv,
which scripts/tcam_table_scoreboard.py scores directly. `ledger_blocks` stays:
scripts/tcam_lane_sweep.py imports it. `main()` now summarises --out and
compiles only the points given with --fields.

Run it (from the repository root; compiling needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_ledger_divergence_sweep.py [--fields "47,51,71;23,27,48,69"]

Resumable: points already present in --out are skipped.
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

from scripts.tcam_stretch_sweep import key_bytes_for, synthetic_program
from scripts.tcam_version_sweep import measured_start_group, read_committed
from p4gen.p4_compile import compile_p4
from p4model.tables import codeword_to_blocks

DEFAULT_OUT = 'results/tcam_ledger_divergence_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_ledger_divergence_sweep'


def ledger_blocks(field_bit_widths):
    """The issue draft's Sec 4 ledger: min g such that a B-byte key with S
    nibble-clean bytes, plus the mandatory version nibble, fits g groups.

    A direct port of `IXBar::increase_ternary_ixbar_space` +
    `TableFormat::ternary_version`'s combined effect, with NO start-offset
    term (every point here is a solo table, so the run starts at group 0 and
    the offset term this ledger admits it is missing does not apply)."""
    byte_widths = [math.ceil(b / 8) for b in field_bit_widths]
    total_bytes = sum(byte_widths)
    if total_bytes == 0:
        return 0
    nibble_clean = sum(1 for b in field_bit_widths if 1 <= b % 8 <= 4)
    g = 1
    while True:
        overflow = max(0, total_bytes - 5 * g)
        if overflow <= math.ceil(g / 2):
            nibbles = 2 * overflow - min(overflow, nibble_clean)
            if nibbles + 1 <= g:
                return g
        g += 1


def run_point(point_id, fields, output_root, size):
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    # n_b = 0: a lone table, so the crossbar run starts at group 0 by
    # construction (there is nothing else in the program to place first).
    source = synthetic_program(fields, 1, (8,), 0, size=size)
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
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')
    committed = read_committed(os.path.join(compile_dir, 'pipe', 'logs'))
    a_rec = committed.get('tern_a0')
    start, ambiguous = (measured_start_group(a_rec['xbar_bytes'])
                        if a_rec else (None, None))

    return {
        'point_id': point_id,
        'fields': ','.join(str(b) for b in fields),
        'key_bytes': key_bytes_for(fields),
        'our_blocks_at_0': codeword_to_blocks(fields),
        'ledger_blocks': ledger_blocks(fields),
        'measured_start_group': start,
        'start_group_ambiguous': ambiguous,
        'real_blocks': a_rec['blocks'] if a_rec else None,
        'xbar_bytes': ' '.join(str(x) for x in a_rec['xbar_bytes'])
                      if a_rec else '',
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        print('\n(no rows)')
        return
    print('\n### M3 -- divergence-class sweep\n')
    cols = ['point_id', 'fields', 'key_bytes', 'ledger_blocks',
           'our_blocks_at_0', 'real_blocks', 'measured_start_group',
           'start_group_ambiguous']
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))

    readable = frame[frame['real_blocks'].notna()]
    if readable.empty:
        return
    ours_right = (readable['real_blocks'] == readable['our_blocks_at_0']).sum()
    ledger_right = (readable['real_blocks'] == readable['ledger_blocks']).sum()
    over = (readable['our_blocks_at_0'] > readable['real_blocks']).sum()
    under = (readable['our_blocks_at_0'] < readable['real_blocks']).sum()
    print('\n%d/%d points: our rule matches the real compiler exactly, '
          '%d/%d: the ledger does. Our rule over-predicts on %d, '
          'under-predicts on %d.' % (
              ours_right, len(readable), ledger_right, len(readable),
              over, under))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--fields', default='',
                        help='points to compile, as semicolon-separated '
                             'comma-separated field bit widths, e.g. "47,51,71;23,27"')
    args = parser.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)
    done = set()
    rows = []
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        rows = existing.to_dict('records')
        done = set(existing['point_id'])

    points = [tuple(sorted(int(b) for b in point.split(',')))
              for point in args.fields.split(';') if point.strip()]
    for i, fields in enumerate(points, start=len(rows)):
        point_id = 'div%02d_n%d_g%d' % (
            i, len(fields), codeword_to_blocks(fields))
        if point_id in done:
            continue
        print('=== %s -- fields=%s' % (point_id, fields), flush=True)
        try:
            row = run_point(point_id, fields, args.output_root, args.size)
        except Exception:                       # a probe must not lose the run
            traceback.print_exc()
            row = {'point_id': point_id, 'fields': ','.join(str(b) for b in fields),
                   'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    ledger=%s  ours=%s  real=%s  start_group=%s' % (
            row.get('ledger_blocks'), row.get('our_blocks_at_0'),
            row.get('real_blocks'), row.get('measured_start_group')),
            flush=True)

    print('\nwrote %s (%d points)' % (args.out, len(rows)))
    summarize(rows)


if __name__ == '__main__':
    main()
