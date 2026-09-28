"""Dumps tests/fixtures/c1_replay_designs.json: the two archived compiles whose
predicted stage_depth audit C1 (per-task tree readiness under 'disjoint') moves,
captured so the move is checkable without the gitignored results/ tree.

For each design it stores what p4_artifact_replay.replay_program consumes --
the parsed program (table -> key fields, field -> crossbar bytes and declared
bits, table -> declared entries) -- plus p4c's own stages_real/tcam_real from
the compile CSV and the model's prediction before C1 (the commit before
per-task readiness, 56e5869). tests/test_compiler_calibration.py replays them.

  margin_independent_M250_k4_s15 (results/tcam_margin_screen): 12 -> 11, real 11
  independent_high_sd12 (results/compiler_calibration_v6):     14 -> 13, real 13

Run (from the repository root):
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/dump_c1_replay_fixture.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.p4_artifact_replay import _p4_table_keys, _p4_table_sizes  # noqa: E402

OUT = os.path.join(ROOT, 'tests', 'fixtures', 'c1_replay_designs.json')

# row_id -> (artifacts root, ground-truth CSV, stage_depth predicted before C1)
DESIGNS = {
    'margin_independent_M250_k4_s15': (
        os.path.join('results', 'tcam_margin_screen'),
        os.path.join('results', 'tcam_margin_screen_compiled.csv'), 12),
    'independent_high_sd12': (
        os.path.join('results', 'compiler_calibration_v6'),
        os.path.join('results', 'compiler_calibration_v6.csv'), 14),
}


def _truth(csv_path, row_id):
    with open(os.path.join(ROOT, csv_path), encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['row_id'] == row_id:
                tcam = row.get('tcam_real') or None
                return int(row['stages_real']), (int(float(tcam)) if tcam else None)
    raise KeyError('%s has no row %s' % (csv_path, row_id))


def main():
    designs = {}
    for row_id, (root, csv_path, before) in DESIGNS.items():
        p4_path = os.path.join(ROOT, root, 'p4_src', row_id + '.p4')
        tables, widths, bits = _p4_table_keys(p4_path)
        stages_real, tcam_real = _truth(csv_path, row_id)
        designs[row_id] = {
            'source': os.path.join(root, 'p4_src', row_id + '.p4').replace(os.sep, '/'),
            'tables': tables,
            'widths': widths,
            'bits': bits,
            'sizes': _p4_table_sizes(p4_path),
            'stages_real': stages_real,
            'tcam_real': tcam_real,
            'stage_depth_before_c1': before,
        }
    with open(OUT, 'w', encoding='utf-8') as handle:
        # NOT sort_keys: `tables` must keep PROGRAM order. The packer breaks
        # placement ties by table order, and independent_high_sd12 replays
        # at 14, not 13, with its tables sorted alphabetically.
        json.dump({'designs': designs}, handle, indent=1)
        handle.write('\n')
    print('wrote %d designs to %s' % (len(designs), OUT))


if __name__ == '__main__':
    main()
