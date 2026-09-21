"""M4 -- does a single 13-block ternary table really split 12|1 across one
stage's two TCAM columns, or 7|6 as the old equal-split rule charged?

WHY THIS EXISTS (docs/2026-09-20-version-block-findings.md, and finding
1.5a in packing._stage_shards). A table needing more TCAM blocks than one
column (`TCAM_ROWS_PER_STAGE` = 12) holds spreads its rows further within a
stage. Before finding 1.5a, `_stage_shards` split such a table into N EQUAL
pieces, each rounded up -- a 13-block table became two 7-block shards (14
charged, not 13). The repaired version fills one column-sized (12-block)
shard and leaves the remainder (1 block) -- 13 charged, matching the table's
own declared block count exactly.

No compile has ever run at a size where the two formulas actually disagree:
the three existing wide compiles (`scripts/tcam_column_sweep.py`) sit at 14,
16 and 24 blocks, where equal-split and column-fill happen to agree. 13, 23
and 25 are the smallest sizes where they diverge; this probe compiles a
single narrow-keyed table sized to exactly 13 TCAM blocks DEEP (`size =
13 * TERNARY_MATCHING_ENTRIES_PER_BLOCK`), not wide -- key width is pinned
at 1 block (a 5-byte solid field, well under a block's 5.5-byte capacity) so
the block count is driven purely by entry depth, the same isolation
`tcam_stretch_sweep`'s SOLID arm uses for width.

Two things this probe reads back, both decisive:
  * the table's own committed TCAM block count (should be 13 either way --
    it is a floor from `size`, not from the packer under test);
  * WHICH row/column layout p4c gave it. 12 rows in column 0 plus 1 row at
    the top of column 1 confirms column-fill; some other split (13 split
    some other way, or 2 stages) would refute it.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_stage_shard_probe.py
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
from src.p4model.packing import _stage_shards
from src.p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK

DEFAULT_OUT = 'results/tcam_stage_shard_probe.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_stage_shard_probe'

KEY_BYTES = 5                     # 1 crossbar group, 1 TCAM block wide
TARGET_BLOCKS = 13                # the smallest size where the two formulas
                                   # under test (equal-split vs column-fill)
                                   # disagree: equal-split charges 14, real
                                   # blocks (and column-fill) charge 13.


def run_point(output_root, size):
    old_shards = [b for b, _ in _stage_shards(TARGET_BLOCKS, KEY_BYTES)]
    equal_split_pred = 2 * -(-TARGET_BLOCKS // 2)   # the retired n-equal-pieces rule

    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    source = synthetic_program(KEY_BYTES, 1, 8, 0, size=size)
    p4_path = os.path.join(p4_dir, 'shard13.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(source)

    compile_dir = os.path.join(output_root, 'compiles', 'shard13')
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an existing dir; a stale one means a
        # prior attempt never finished.
        shutil.rmtree(compile_dir)
    started = time.time()
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')
    measured = measure(os.path.join(compile_dir, 'pipe', 'logs'))

    row = {
        'target_blocks': TARGET_BLOCKS,
        'key_bytes': KEY_BYTES,
        'declared_size': size,
        'column_fill_shards': str(old_shards),
        'column_fill_total': sum(old_shards),
        'equal_split_would_charge': equal_split_pred,
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }
    if measured is None:
        row.update({'real_blocks': None, 'occupied_stages': None, 'layout': None})
    else:
        layout = measured['layout']
        row.update({'occupied_stages': measured['occupied_stages'], 'layout': layout})
    return row


def report(row):
    print('\n### M4 -- stage-shard probe (13-block table)\n')
    for key in ('target_blocks', 'key_bytes', 'declared_size',
               'column_fill_shards', 'column_fill_total',
               'equal_split_would_charge', 'occupied_stages', 'layout',
               'compile_errors'):
        print('  %s: %s' % (key, row.get(key)))
    print(
        '\nColumn-fill predicts %d blocks total; the retired equal-split '
        'rule would have charged %d. The real committed layout above (per '
        'stage, per table: column/first-row/block-count) says which one the '
        'compiler agrees with.'
        % (row.get('column_fill_total'), row.get('equal_split_would_charge')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    if os.path.exists(args.out):
        print('%s already exists -- skipping (delete it to re-run)' % args.out)
        report(pd.read_csv(args.out).iloc[0].to_dict())
        return

    size = TARGET_BLOCKS * TERNARY_MATCHING_ENTRIES_PER_BLOCK
    os.makedirs(args.output_root, exist_ok=True)
    print('=== shard13 -- one %d-byte-keyed table, size=%d (target %d blocks)'
          % (KEY_BYTES, size, TARGET_BLOCKS), flush=True)
    try:
        row = run_point(args.output_root, size)
    except Exception:
        traceback.print_exc()
        row = {'compile_errors': -1}
    pd.DataFrame([row]).to_csv(args.out, index=False)
    report(row)


if __name__ == '__main__':
    main()
