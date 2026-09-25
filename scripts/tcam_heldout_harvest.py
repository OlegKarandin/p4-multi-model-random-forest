"""Harvest per-table TCAM observations from the HELD-OUT archive,
results/compiler_calibration_extra/ -- 8 real p4c compiles that were never
used to fit or re-tune the block model (tcam_offset_harvest.py and the golden
fixture read compiler_calibration_v6 only). No new compiles are run.

Unlike tcam_offset_harvest.py there is no fixture row to recover field widths
from, so they are read from the generated program's own metadata declarations
(compiler_calibration._p4_table_keys), which is what the model would see
before compiling anyway. Observed blocks and crossbar bytes come from the
compiler's committed resources.json (tcam_version_sweep.read_committed).

Run it (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_heldout_harvest.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from scripts.p4_artifact_replay import _p4_table_keys  # noqa: E402
from scripts.tcam_version_sweep import measured_start_group, read_committed  # noqa: E402

ARCHIVE = os.path.join('results', 'compiler_calibration_extra')
DEFAULT_OUT = os.path.join('results', 'tcam_heldout_harvest.csv')


def harvest_row(row_id, archive=ARCHIVE):
    """One dict per classification table p4c placed for this row."""
    committed = read_committed(os.path.join(archive, 'compiles', row_id,
                                            'pipe', 'logs'))
    tables, _, bits = _p4_table_keys(os.path.join(archive, 'p4_src',
                                                  row_id + '.p4'))
    stage_keys = {}
    for name, rec in committed.items():
        if name.startswith('get_classification_tree_'):
            stage_keys.setdefault(rec['stage'], set()).add(
                tuple(sorted(tables[name])))
    rows = []
    for name, rec in sorted(committed.items()):
        if not name.startswith('get_classification_tree_'):
            continue
        start_group, ambiguous = measured_start_group(rec['xbar_bytes'])
        rows.append({
            'row_id': row_id,
            'table': name,
            'stage': rec['stage'],
            'field_bits': ','.join(str(bits[f]) for f in sorted(tables[name])),
            'measured_start_group': start_group,
            'start_group_ambiguous': ambiguous,
            'distinct_keys_in_stage': len(stage_keys[rec['stage']]),
            'observed_blocks': rec['blocks'],
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    row_ids = sorted(os.listdir(os.path.join(ARCHIVE, 'compiles')))
    rows = [r for row_id in row_ids for r in harvest_row(row_id)]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('wrote %s (%d tables from %d held-out designs)'
          % (args.out, len(rows), len(row_ids)))


if __name__ == '__main__':
    main()
