"""Out-of-sample test of `tables.version_block_penalty` against real p4c.

WHY THIS EXISTS. The rule was scored exact on all 100 classification tables of
the 19 archived compiles -- but 89 of those are trivial negatives (a key with
slack, or a group run that ends on a half midbyte), every one of them sits at
crossbar group offset 0, and clauses (b) and (d) were shaped against those same
artifacts. Clause (c) has ZERO support from real data. So "100/100" is an
in-sample fit over a sample that barely exercises the rule.

This sweep compiles designs chosen to sit ON the decision boundary rather than
comfortably inside it, and it is deliberately adversarial: several points are
places where the CURRENT code and a proposed correction disagree, so the
compiler decides rather than the author.

WHAT EACH ARM ASKS.

  A. The saturation edge, solid keys, ONE table (so the group offset is
     unambiguously 0). B and B+1 crossbar bytes across the predicted step: a
     key with one spare byte slot must not pay, a key with none must. Three of
     these are also DOUBLE-COUNT discriminators -- `codeword_bits_to_blocks`'s
     `+4` overhead bits are themselves a version/valid allowance, so
     `max(band, xbar) + penalty` (what ships today) charges the version field
     twice when the band arm wins, where `max(band, xbar + penalty)` charges it
     once. The two differ by a whole block on these points.

  B. Clause (d), isolated. Two keys of the SAME 22 bytes and the same 4 groups,
     both saturating, differing only in whether any nibble-clean byte can be
     ordered onto a midbyte slot. If the rule is right these compile to
     5 and 4 blocks respectively; if clause (d) is fiction they both give 5.

  C. The start offset, including the first real test of clause (c). A SOLID key
     shifted to an odd start should stay free (nothing can occupy the half
     midbyte it exposes); the same geometry built from a RAGGED field should
     pay. Two-key points, so the offset is whatever p4c chooses -- this script
     MEASURES it from the crossbar byte map rather than assuming it, and says
     so when the placement did not produce the shift the point was after.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_version_sweep.py

Resumable: points already present in --out are skipped.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_stretch_sweep import as_fields, key_bytes_for, synthetic_program
from src.p4gen.p4_compile import compile_p4
from src.p4model.tables import codeword_bits_to_blocks, codeword_bytes_to_blocks

DEFAULT_OUT = 'results/tcam_version_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_version_sweep'

TERNARY_XBAR_BASE = 128        # ternary crossbar bytes are numbered from here
PAIR_BYTES = 11                # 5 private + 1 midbyte + 5 private


# (point_id, key_a, n_a, key_b, n_b, what the point is for)
POINTS = [
    # --- Arm A: the saturation edge, one solid key, offset 0 ---------------
    ('solid_g2_B10', (80,), 1, None, 0,
     'g=2, 10 bytes: one spare slot, must not pay'),
    ('solid_g2_B11', (88,), 1, None, 0,
     'g=2, 11 bytes: saturated, must pay -- DOUBLE-COUNT DISCRIMINATOR'),
    ('solid_g4_B21', (168,), 1, None, 0,
     'g=4, 21 bytes: one spare slot, must not pay'),
    ('solid_g4_B22', (176,), 1, None, 0,
     'g=4, 22 bytes: saturated, must pay -- DOUBLE-COUNT DISCRIMINATOR'),
    ('solid_g6_B32', (256,), 1, None, 0,
     'g=6, 32 bytes: one spare slot, must not pay'),
    ('solid_g6_B33', (264,), 1, None, 0,
     'g=6, 33 bytes: saturated, must pay -- DOUBLE-COUNT DISCRIMINATOR'),

    # --- Arm B: clause (d), same B and g, only reachability differs --------
    ('d_unreachable_B22', (84, 84), 1, None, 0,
     'clause (d) FAILS: clean bytes land at 10 and 21, midbytes are 5 and 16'),
    ('d_reachable_B22', (44, 40, 88), 1, None, 0,
     'clause (d) HOLDS: the 6-byte clean field can be ordered onto midbyte 5'),

    # --- Arm C: the start offset, and the first real test of clause (c) ----
    ('c_solid_alone', (128,), 1, None, 0,
     'solid 16-byte key alone: g=3 ends on a half midbyte, free'),
    ('c_solid_shifted', (128,), 1, (40,), 1,
     'clause (c): same solid key pushed off group 0 -- no clean byte exists '
     'to occupy the half midbyte, so it must STAY free'),
    ('d_ragged_alone', (124,), 1, None, 0,
     'ragged 16-byte key alone: also free, by the same clause (a)'),
    ('d_ragged_shifted', (124,), 1, (40,), 1,
     'the same ragged key pushed off group 0: now it must pay'),
]


def predict(field_bits, start_group):
    """Both compositions, so the compiler can choose between them.

    `shipped`   = max(codeword_bits_to_blocks, codeword_bytes_to_blocks) + version penalty
                  -- what the code did before the 2026-09-20 tables.py rewrite.
    `corrected` = max(codeword_bits_to_blocks, codeword_bytes_to_blocks + version penalty)
                  -- codeword_bits_to_blocks's own +4 version/valid bits stop
                  being charged a second time.
    They differ only where the band arm wins AND the penalty fires.

    STALE (2026-09-21, Task 2): `version_block_penalty` was retired along with
    the offset-taking `codeword_to_blocks` this sweep was scoring -- the
    mechanism it priced (a version block's home depending on `start_group`)
    is no longer part of the per-table price at all (2026-09-20 rewrite
    design Sec 13.1). This is a one-shot instrument script already run to
    produce the gitignored results/tcam_version_sweep.csv (see Task 2's
    report); `main()`/`run_point()` -> `predict()` is dead code left
    unexecuted, not fixed, the same way scripts/tcam_offset_scan.py was left
    -- nothing imports this function, only the module-level `measured_start_group`
    and `read_committed` it sits beside, which scripts/tcam_offset_harvest.py
    (and its test suite) still use."""
    bits = list(field_bits)
    key_bytes = sum(math.ceil(b / 8) for b in bits)
    band = codeword_bits_to_blocks(sum(bits))
    xbar = codeword_bytes_to_blocks(key_bytes)
    penalty = version_block_penalty(bits, start_group)  # noqa: F821 -- see docstring
    return {
        'key_bytes': key_bytes,
        'groups': xbar,
        # Column name kept as 'band_factor' deliberately: results/tcam_version_sweep.csv
        # is archived data, and renaming a column silently rewrites what an
        # archived row means. The identifier is codeword_bits_to_blocks.
        'band_factor': band,
        'penalty': penalty,
        'shipped': max(band, xbar) + penalty,
        'corrected': max(band, xbar + penalty),
    }


def read_committed(logs_dir):
    """{table -> {'blocks': n, 'xbar_bytes': [relative byte numbers]}} from the
    compiler's own resources.json -- the same file the archived compiles were
    re-scored from, so this sweep and that scoring read the identical quantity."""
    path = os.path.join(logs_dir, 'resources.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as handle:
        doc = json.load(handle)
    out = {}
    for stage in doc['resources']['mau']['mau_stages']:
        for unit in stage['tcams']['tcams']:
            for use in unit['usages']:
                rec = out.setdefault(use['used_by'].split('.')[-1],
                                     {'blocks': 0, 'xbar_bytes': set(),
                                      'stage': stage['stage_number']})
                rec['blocks'] += 1
        for byte in stage['xbar_bytes']['bytes']:
            if byte.get('byte_type') != 'ternary':
                continue
            for use in byte['usages']:
                name = use['used_by'].split('.')[-1]
                if name in out:
                    out[name]['xbar_bytes'].add(
                        byte['byte_number'] - TERNARY_XBAR_BASE)
    for rec in out.values():
        rec['xbar_bytes'] = sorted(rec['xbar_bytes'])
    return out


def measured_start_group(xbar_bytes):
    """The crossbar group a key's run begins at, from its lowest byte.

    A pair of groups spans PAIR_BYTES bytes as 5 private, 1 midbyte, 5 private,
    so relative byte r sits in group 2*(r//11) when r%11 < 5 and 2*(r//11)+1
    when r%11 > 5. A byte at r%11 == 5 IS the midbyte and belongs to both, so
    it is reported as ambiguous rather than guessed."""
    if not xbar_bytes:
        return None, None
    low = min(xbar_bytes)
    pair, within = divmod(low, PAIR_BYTES)
    if within < 5:
        return 2 * pair, False
    if within == 5:
        return 2 * pair, True          # shared midbyte: could be either group
    return 2 * pair + 1, False


def run_point(point, output_root, size):
    point_id, key_a, n_a, key_b, n_b, purpose = point
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    # n_b == 0 still needs a syntactically valid key_b; its fields are declared
    # but no table reads them, so p4c drops them and the crossbar never sees
    # them. Verified by the measured byte map, which is reported per table.
    source = synthetic_program(key_a, n_a, key_b if key_b else (8,), n_b,
                               size=size)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(source)
    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an existing dir; a stale one means a prior
        # attempt never finished.
        shutil.rmtree(compile_dir)
    started = time.time()
    # Same includes the other TCAM probes use -- headers.p4 / util.p4 live
    # here, not in resources/, and compile_p4 copies the include dir into its
    # ASCII WSL scratch alongside the source.
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')
    committed = read_committed(os.path.join(compile_dir, 'pipe', 'logs'))

    a_tables = {k: v for k, v in committed.items() if k.startswith('tern_a')}
    b_tables = {k: v for k, v in committed.items() if k.startswith('tern_b')}
    a_blocks = sorted({v['blocks'] for v in a_tables.values()})
    a_bytes = sorted({len(v['xbar_bytes']) for v in a_tables.values()})
    start, ambiguous = (measured_start_group(
        next(iter(a_tables.values()))['xbar_bytes']) if a_tables else (None, None))

    pred0 = predict(as_fields(key_a), 0)
    predm = predict(as_fields(key_a), start) if start is not None else None
    return {
        'point_id': point_id,
        'purpose': purpose,
        'fields_a': ','.join(str(b) for b in as_fields(key_a)),
        'n_a': n_a,
        'fields_b': ','.join(str(b) for b in as_fields(key_b)) if key_b else '',
        'n_b': n_b,
        'declared_key_bytes': pred0['key_bytes'],
        'measured_key_bytes': a_bytes[0] if len(a_bytes) == 1 else str(a_bytes),
        'groups': pred0['groups'],
        'band_factor': pred0['band_factor'],
        'measured_start_group': start,
        'start_group_ambiguous': ambiguous,
        'penalty_at_0': pred0['penalty'],
        'penalty_at_measured': predm['penalty'] if predm else None,
        'pred_shipped': pred0['shipped'],
        'pred_corrected': pred0['corrected'],
        'pred_corrected_at_measured': predm['corrected'] if predm else None,
        'real_blocks_a': a_blocks[0] if len(a_blocks) == 1 else str(a_blocks),
        'real_blocks_b': (sorted({v['blocks'] for v in b_tables.values()})[0]
                          if b_tables else None),
        'shared_stage': (len({v['stage'] for v in committed.values()}) == 1
                         and bool(b_tables)),
        'xbar_bytes_a': ' '.join(
            str(x) for x in next(iter(a_tables.values()))['xbar_bytes']
        ) if a_tables else '',
        'compile_errors': len(result.errors) if result.errors else 0,
        'seconds': round(time.time() - started, 1),
    }


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
        print('=== %s -- %s' % (point[0], point[5]), flush=True)
        try:
            row = run_point(point, args.output_root, args.size)
        except Exception:                       # a probe must not lose the run
            traceback.print_exc()
            row = {'point_id': point[0], 'purpose': point[5],
                   'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    real=%s  shipped=%s  corrected=%s  start_group=%s' % (
            row.get('real_blocks_a'), row.get('pred_shipped'),
            row.get('pred_corrected'), row.get('measured_start_group')),
            flush=True)
    print('\nwrote %s (%d points)' % (args.out, len(rows)))


if __name__ == '__main__':
    main()
