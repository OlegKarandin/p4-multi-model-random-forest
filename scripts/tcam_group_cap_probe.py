"""Is there a 12-crossbar-group-per-stage cap, separate from the modelled
64-byte-per-stage cap?

WHY THIS EXISTS. Finding 1.5b: `packing.crossbar_stages_needed` checks
`TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` (64 bytes) but no per-stage GROUP
cap. Verified live against current code:

    codeword_bytes_to_blocks(34) = 7 groups,  codeword_bytes_to_blocks(30) = 6 groups
    34 + 30 = 64 bytes  ->  passes TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE (64)
    crossbar_stages_needed([(7, 34), (6, 30)])
        -> StagePlan(occupied=1, depth=1, indices=frozenset({0}), blocks=13)

One stage, 13 groups, in a stage the model treats as having 12 (2 columns x
6? -- no: a stage physically supplies 24 TCAM blocks over 12 rows x 2
columns, but the TERNARY MATCH INPUT CROSSBAR is a separate resource from
the TCAM array, and the open question is whether IT additionally caps out
at 12 groups regardless of the 64-byte figure). Two outcomes, both decisive:

  * p4c puts both tables in ONE stage -> the 64-byte cap is the real limit,
    the model is right, NO CHANGE (a follow-up task adding a 12-group cap
    would be a no-op).
  * p4c puts them in TWO -> a 12-group-per-stage cap exists and the model
    under-counts stages for any design that reaches 13+ groups while still
    under 64 bytes.

A 59-byte / 11-group control point (`groups_12_bytes_59`) accompanies the
64-byte / 13-group test point: it sits under BOTH caps and must compile to
one shared stage, or the test point's result cannot be read as evidence of
a group cap rather than of two wide ternary tables simply refusing to share
a stage in this probe's program shape (the same confound tcam_stretch_sweep
and tcam_version_sweep exist to rule out for their own probes).

Modelled closely on scripts/tcam_version_sweep.py's `run_point`: same
`compile_p4` call, same `include_path='p4/tofino_spike/common'`, same stale
compile-dir cleanup before each compile, same `size=512` per table. Consumes
`scripts.tcam_stretch_sweep.synthetic_program`/`measure` exactly as that
module defines them -- one SOLID key per table (an int key means one solid
field of that many bytes), one table per key, so the group count a key needs
is unambiguous.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_group_cap_probe.py

This runs two real p4c compiles and takes minutes each. Resumable: points
already present in --out are skipped.
"""
import argparse
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_stretch_sweep import measure, synthetic_program
from src.p4gen.p4_compile import compile_p4
from src.p4model.packing import crossbar_stages_needed
from src.p4model.tables import codeword_bytes_to_blocks

DEFAULT_OUT = 'results/tcam_group_cap_probe.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_group_cap_probe'

# (point_id, key_a_bytes, key_b_bytes, purpose). Both points are n_a = n_b = 1
# solid keys, so groups_a/groups_b are exactly codeword_bytes_to_blocks of
# each byte width -- no field-splitting ambiguity.
#
# point_id 'groups_12_bytes_59' names the control's INTENT (comfortably under
# a putative 12-group cap), not its exact group total -- 32 + 27 bytes need
# 6 + 5 = 11 groups, one short of 12; kept verbatim from the task brief's
# point table rather than renamed to 'groups_11_bytes_59', since the id is a
# label, not a computed field (groups_total below carries the real number).
POINTS = [
    ('groups_13_bytes_64', 34, 30,
     '13 groups at exactly the 64-byte crossbar cap: does a separate '
     '12-group cap force a second stage?'),
    ('groups_12_bytes_59', 32, 27,
     'control: 11 groups and 59 bytes, under both caps, must share one '
     'stage or the test point above is not readable as a group-cap effect'),
]


def run_point(point, output_root, size):
    point_id, key_a_bytes, key_b_bytes, purpose = point
    groups_a = codeword_bytes_to_blocks(key_a_bytes)
    groups_b = codeword_bytes_to_blocks(key_b_bytes)
    model_stages = crossbar_stages_needed(
        [(groups_a, key_a_bytes), (groups_b, key_b_bytes)]).occupied

    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    source = synthetic_program(key_a_bytes, 1, key_b_bytes, 1, size=size)
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
    measured = measure(os.path.join(compile_dir, 'pipe', 'logs'))

    row = {
        'point_id': point_id,
        'key_a_bytes': key_a_bytes,
        'key_b_bytes': key_b_bytes,
        'groups_a': groups_a,
        'groups_b': groups_b,
        'groups_total': groups_a + groups_b,
        'bytes_total': key_a_bytes + key_b_bytes,
        'model_stages': model_stages,
        'real_stages': measured['occupied_stages'] if measured else None,
        'both_keys_in_one_stage':
            measured['both_keys_in_one_stage'] if measured else None,
        'layout': measured['layout'] if measured else None,
        # CompileResult.errors is already an error COUNT (not a list), unlike
        # what scripts/tcam_version_sweep.py's `len(result.errors)` assumes --
        # that expression only survives there because a successful compile's
        # errors == 0 is falsy and takes its `else 0` branch; a genuine
        # nonzero count would raise TypeError inside the very code path meant
        # to report it. Recorded directly here instead, so a real compile
        # failure is reported rather than swallowed.
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }
    return row, purpose


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
            row, purpose = run_point(point, args.output_root, args.size)
        except Exception:
            # A toolchain failure (compile_p4 raises RuntimeError on a p4c
            # crash / timeout) carries the real error text in its message --
            # print it in full rather than swallowing it, per the brief: a
            # refused program is a finding, not something to route around.
            traceback.print_exc()
            print('\n%s FAILED -- stopping (a failed compile is a genuine '
                  'finding, not something to invent a result for).'
                  % point[0], flush=True)
            sys.exit(1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    model_stages=%s  real_stages=%s  '
              'both_keys_in_one_stage=%s  compile_errors=%s' % (
                  row['model_stages'], row['real_stages'],
                  row['both_keys_in_one_stage'], row['compile_errors']),
              flush=True)
        if row['compile_errors']:
            print('\n%s compiled with %d error(s) -- stopping (see the '
                  'compile output above for the exact p4c error).'
                  % (point[0], row['compile_errors']), flush=True)
            sys.exit(1)
    print('\nwrote %s (%d points)' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
