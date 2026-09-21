"""M1/M2 -- can a spacer ever land a probe key at a genuine, UNAMBIGUOUS ODD
crossbar group offset? (docs/2026-09-20-version-block-findings.md)

WHY THIS EXISTS. Both `scripts/tcam_version_sweep.py` (moving the key itself)
and `scripts/tcam_spacer_sweep.py` (a wider RAGGED spacer, key built from many
small fields) got ZERO usable points: every measured start group came back
even, or the reading was ambiguous because the probe key's lowest crossbar
byte landed EXACTLY on a shared midbyte position (byte index 5 of the 11-byte
pair cycle), which `measured_start_group` correctly refuses to call even or
odd. This script tries the shape the findings doc proposes instead: a SOLID,
single-field spacer (so its own placement cannot scatter) at a plain LINEAR
SCAN of byte widths, in front of a fixed single-field probe key -- rather
than guessing one "designed" width from first-principles run-capacity
arithmetic (which the findings doc's own N7 shows is unreliable), this just
tries every width from 1 to `--max-spacer-bytes` and reads back what p4c
actually does. If any width lands the probe unambiguously at an odd group,
stage 2 (`--probe`) re-runs that exact width against the other probe keys
this doc's M1/M2 care about (a plain saturating solid key, and the M2
"discount" key `2 x bit<84>`).

Reuses `scripts.tcam_stretch_sweep.synthetic_program` (`@pa_solitary` per
field, one action per table, `size=512`) and
`scripts.tcam_version_sweep.read_committed`/`measured_start_group`, exactly
as the other TCAM probes in this project do.

Run the scan (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_offset_scan.py --max-spacer-bytes 40

Then, if the scan finds a usable width W:
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_offset_scan.py --probe W

Resumable: points already present in --out are skipped.
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
from src.p4model.tables import codeword_to_blocks

DEFAULT_OUT = 'results/tcam_offset_scan.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_offset_scan'

# The general-purpose M1 probe: a single solid field, saturating (11 bytes,
# 2 groups) -- the plainest possible "does an odd offset cost more" test.
PROBE_SOLID11 = (88,)

# M2's exact probe: `2 x bit<84>` -- 22 crossbar bytes, 4 groups, saturated,
# measured at 5 blocks standalone (the "legitimate +1" row in N10/N9).
# Two fields, not one, but only two -- far less scatter risk than the
# 11/22-field ragged keys tcam_spacer_sweep used.
PROBE_DISCOUNT = (84, 84)


def run_one(point_id, spacer_bytes, probe_fields, output_root, size):
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    # key_b = spacer, key_a = probe under test. NOT a width-sort call (see
    # scan_points' docstring): two independent prior scans with the probe as
    # 'b' put the PROBE at group 0 on every single point regardless of which
    # key was wider, both here and in tcam_spacer_sweep/tcam_version_sweep's
    # own archived data once re-read the same way -- 'b' always lands first,
    # 'a' always gets pushed behind it. So the probe must be 'a' to be
    # shiftable at all.
    source = synthetic_program(probe_fields, 1, spacer_bytes, 1, size=size)
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
    a_rec = committed.get('tern_a0')          # the probe under test
    b_rec = committed.get('tern_b0')          # the spacer

    start, ambiguous = (measured_start_group(a_rec['xbar_bytes'])
                        if a_rec else (None, None))
    predicted_at_0 = codeword_to_blocks(probe_fields)
    # Offset-dependent pricing was retired (2026-09-20 rewrite design Sec
    # 13.1): a key's price no longer depends on start_group at all, so the
    # delta between "at 0" and "at measured" is always 0 by construction.
    # Column kept only for CSV-schema stability with the archived
    # results/tcam_offset_scan.csv.
    delta_at_measured = 0 if start is not None else None

    return {
        'point_id': point_id,
        'spacer_bytes': spacer_bytes,
        'probe_fields': ','.join(str(b) for b in as_fields(probe_fields)),
        'probe_key_bytes': key_bytes_for(probe_fields),
        'spacer_real_blocks': b_rec['blocks'] if b_rec else None,
        'probe_real_blocks': a_rec['blocks'] if a_rec else None,
        'probe_predicted_blocks_at_0': predicted_at_0,
        'measured_start_group': start,
        'start_group_ambiguous': ambiguous,
        'predicted_delta_at_measured': delta_at_measured,
        'both_in_one_stage': (a_rec is not None and b_rec is not None
                              and a_rec['stage'] == b_rec['stage']),
        'spacer_xbar_bytes': ' '.join(str(x) for x in b_rec['xbar_bytes'])
                             if b_rec else '',
        'probe_xbar_bytes': ' '.join(str(x) for x in a_rec['xbar_bytes'])
                            if a_rec else '',
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }


def scan_points(max_spacer_bytes, probe_fields):
    """Spacer widths to try, in the 'b' slot.

    CORRECTED TWICE. tcam_spacer_sweep.py's own comment claimed p4c always
    places the WIDER of two tables first (memories.cpp:1800). A first scan
    here put the probe in 'b' behind a WIDER 'a' spacer (12-40 bytes) and saw
    the probe land at group 0 on all 29 points; a second scan tried a
    NARROWER 'a' spacer (1-10 bytes) and saw the SAME thing -- group 0 on
    all 10, regardless of width. The common factor was not width at all: in
    both runs the probe was table 'b'. Re-reading tcam_spacer_sweep.py's own
    archived data (sat11_sp16: key_a is the probe, key_b the spacer, and key_a
    is the one that moved) and tcam_version_sweep.py's d_ragged_shifted (same
    convention, key_a moved) shows the same pattern: table 'b' lands at group
    0 and 'a' is what gets pushed behind it, independent of which one is
    wider. So the probe must be 'a' and the spacer 'b' -- this function no
    longer needs to reason about relative width at all, and just tries a
    plain range of spacer sizes."""
    points = []
    for spacer_bytes in range(1, max_spacer_bytes + 1):
        points.append(('sp%02d' % spacer_bytes, spacer_bytes))
    return points


def collect(out, output_root, points, probe_fields, size, only=None):
    done = set()
    rows = []
    if os.path.exists(out):
        existing = pd.read_csv(out)
        rows = existing.to_dict('records')
        done = set(existing['point_id'])
    wanted = set(only.split(',')) if only else None
    for point_id, spacer_bytes in points:
        if point_id in done or (wanted and point_id not in wanted):
            continue
        print('=== %s -- spacer=%d bytes, probe=%s' %
              (point_id, spacer_bytes, as_fields(probe_fields)), flush=True)
        try:
            row = run_one(point_id, spacer_bytes, probe_fields, output_root, size)
        except Exception:                       # a probe must not lose the run
            traceback.print_exc()
            row = {'point_id': point_id, 'spacer_bytes': spacer_bytes,
                   'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print('    start_group=%s  ambiguous=%s  probe_real_blocks=%s  '
              'predicted_at_0=%s' % (
                  row.get('measured_start_group'), row.get('start_group_ambiguous'),
                  row.get('probe_real_blocks'), row.get('probe_predicted_blocks_at_0')),
              flush=True)
    return rows


def summarize(rows, label):
    frame = pd.DataFrame(rows)
    print('\n### %s\n' % label)
    if frame.empty:
        print('  (no rows)')
        return frame
    cols = ['point_id', 'spacer_bytes', 'probe_fields', 'probe_key_bytes',
           'measured_start_group', 'start_group_ambiguous',
           'probe_predicted_blocks_at_0', 'probe_real_blocks',
           'predicted_delta_at_measured']
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))

    clean_odd = frame[(frame['measured_start_group'].notna())
                      & (frame['start_group_ambiguous'] == False)  # noqa: E712
                      & (frame['measured_start_group'] % 2 == 1)]
    print('\n%d of %d point(s) landed the probe at a non-ambiguous ODD group.'
          % (len(clean_odd), len(frame)))
    if len(clean_odd):
        print(clean_odd[[c for c in cols if c in clean_odd.columns]]
              .to_string(index=False))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--max-spacer-bytes', type=int, default=40)
    parser.add_argument('--only', default=None,
                        help='comma-separated point_ids to run (stage 1)')
    parser.add_argument('--probe', type=int, default=None,
                        help='stage 2: re-run this exact spacer width against '
                             'the M1 solid-11 probe and the M2 discount probe '
                             '(2 x bit<84>), writing to a separate --probe-out')
    parser.add_argument('--probe-out', default='results/tcam_offset_probe.csv')
    parser.add_argument('--discount-scan', action='store_true',
                        help='M2: scan spacer widths 1..--max-spacer-bytes '
                             'against the discount probe (2 x bit<84>) '
                             'directly, writing to --discount-out. This is '
                             'the scan that found the discount is FALSE: '
                             'real p4c costs 5 blocks at the unambiguous odd '
                             'offsets 5 and 7 (spacer widths 23-29, 36-39), '
                             'not the model-predicted 4.')
    parser.add_argument('--discount-out', default='results/tcam_discount_scan.csv')
    args = parser.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)

    if args.discount_scan:
        points = scan_points(args.max_spacer_bytes, PROBE_DISCOUNT)
        print('scanning %d spacer widths against the M2 discount probe %s'
              % (len(points), PROBE_DISCOUNT))
        rows = collect(args.discount_out, args.output_root, points,
                      PROBE_DISCOUNT, args.size, only=args.only)
        summarize(rows, 'M2 scan -- 2 x bit<84> probe behind a linear spacer scan')
        return

    if args.probe is not None:
        spacer_bytes = args.probe
        stage2_points = []
        for tag, fields in (('solid11', PROBE_SOLID11),
                            ('discount22', PROBE_DISCOUNT)):
            point_id = 'probe_%s_sp%02d' % (tag, spacer_bytes)
            rows = collect(args.probe_out, args.output_root,
                           [(point_id, spacer_bytes)], fields, args.size)
            stage2_points.extend(rows)
        summarize(stage2_points, 'M1/M2 -- targeted probes at spacer=%d' % spacer_bytes)
        return

    points = scan_points(args.max_spacer_bytes, PROBE_SOLID11)
    print('scanning %d spacer widths against probe %s' % (len(points), PROBE_SOLID11))
    rows = collect(args.out, args.output_root, points, PROBE_SOLID11, args.size,
                  only=args.only)
    summarize(rows, 'M1 scan -- solid 11-byte probe behind a linear spacer scan')


if __name__ == '__main__':
    main()
