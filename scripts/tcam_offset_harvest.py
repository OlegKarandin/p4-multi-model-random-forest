"""B1: harvest per-table crossbar offsets from the 19 already-archived p4c
compiles -- no new compiles are run.

WHY THIS EXISTS. The 19 archived compiles already record, per table, both the
TCAM units it got (`tcams.tcams`) and the exact crossbar bytes its key was
given (`xbar_bytes`). The calibration only ever scored TOTALS, so per-table
offsets were observed and discarded. This is free ground truth that has never
been used this way, and it is the only evidence that can discriminate finding
1.4.

This is an OBSERVATIONAL harvest: it scores whatever placement p4c chose and
never asks for a particular offset, which is why it cannot come back empty the
way results/tcam_version_sweep.csv's `c_solid_shifted` / `d_ragged_shifted`
points did.

Run it (from the repository root):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_offset_harvest.py
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_version_sweep import measured_start_group, read_committed
from src.p4model.tables import codeword_bytes_to_blocks, codeword_to_blocks

ARCHIVE = 'results/compiler_calibration_v6/compiles'
FIXTURE = 'tests/fixtures/resource_model_golden.json'


def _load_fixture_rows():
    with open(FIXTURE, encoding='utf-8') as handle:
        doc = json.load(handle)
    return {row['row_id']: row for row in doc['rows']}


def _archive_row_ids():
    return {name for name in os.listdir(ARCHIVE)
            if os.path.isdir(os.path.join(ARCHIVE, name))}


_FIXTURE_ROWS = _load_fixture_rows()

_archive_ids = _archive_row_ids()
_fixture_ids = set(_FIXTURE_ROWS)
if _archive_ids != _fixture_ids:
    only_archive = sorted(_archive_ids - _fixture_ids)
    only_fixture = sorted(_fixture_ids - _archive_ids)
    raise RuntimeError(
        'archive and fixture row ids disagree -- a silent partial harvest '
        'would understate the evidence: only in archive=%s only in '
        'fixture=%s' % (only_archive, only_fixture))

ROW_IDS = sorted(_archive_ids)


def field_bits_by_key_bytes(row_id):
    """{key_bytes: tuple_of_field_bits} for every distinct ternary key field
    set this row's tables key on.

    Raises if two distinct field-width tuples in one row share a byte width --
    the join in harvest_row would then be ambiguous."""
    row = _FIXTURE_ROWS[row_id]
    key_field_sets = row['key_field_sets']
    set_ids = set(row['inputs']['ternary_key_field_set_ids'])
    out = {}
    for set_id in set_ids:
        bits = tuple(b for _, b in key_field_sets[set_id])
        key_bytes = sum(math.ceil(b / 8) for b in bits)
        if key_bytes in out and out[key_bytes] != bits:
            raise ValueError(
                'row %r: field width tuples %r and %r both key_bytes=%d -- '
                'the key_bytes join is ambiguous' %
                (row_id, out[key_bytes], bits, key_bytes))
        out[key_bytes] = bits
    return out


def harvest_row(row_id):
    """list[dict], one per classification table in this row's archived
    compile, scored against the model at p4c's own MEASURED offset."""
    logs_dir = os.path.join(ARCHIVE, row_id, 'pipe', 'logs')
    committed = read_committed(logs_dir)
    field_bits_by_bytes = field_bits_by_key_bytes(row_id)

    classification = {name: rec for name, rec in committed.items()
                       if name.startswith('get_classification_tree_')}

    stage_key_bytes = {}
    for name, rec in classification.items():
        stage_key_bytes.setdefault(rec['stage'], set()).add(len(rec['xbar_bytes']))

    rows = []
    for table, rec in classification.items():
        key_bytes = len(rec['xbar_bytes'])
        field_bits = field_bits_by_bytes[key_bytes]
        groups = codeword_bytes_to_blocks(key_bytes)
        start_group, ambiguous = measured_start_group(rec['xbar_bytes'])
        rows.append({
            'row_id': row_id,
            'table': table,
            'stage': rec['stage'],
            'key_bytes': key_bytes,
            'field_bits': field_bits,
            'groups': groups,
            'measured_start_group': start_group,
            'start_group_ambiguous': ambiguous,
            'observed_blocks': rec['blocks'],
            'predicted_blocks': codeword_to_blocks(field_bits, start_group),
            'predicted_blocks_at_0': codeword_to_blocks(field_bits, 0),
            'distinct_keys_in_stage': len(stage_key_bytes[rec['stage']]),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/tcam_offset_harvest.csv')
    args = parser.parse_args(argv)

    all_rows = []
    for row_id in ROW_IDS:
        all_rows.extend(harvest_row(row_id))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out, index=False)

    total = len(all_rows)
    ambiguous = sum(1 for r in all_rows if r['start_group_ambiguous'])
    mispredicted = sum(
        1 for r in all_rows
        if not r['start_group_ambiguous']
        and r['predicted_blocks'] != r['observed_blocks'])
    discriminating = [
        r for r in all_rows
        if r['measured_start_group'] is not None
        and r['measured_start_group'] % 2 == 1
        and not r['start_group_ambiguous']
    ]

    print('wrote %s (%d rows)' % (args.out, total))
    print('total tables: %d' % total)
    print('ambiguous start group: %d' % ambiguous)
    print('mispredicted (non-ambiguous only): %d' % mispredicted)
    print('discriminating rows for finding 1.2 (odd measured_start_group, '
          'non-ambiguous): %d' % len(discriminating))
    for r in discriminating:
        print('  %s / %s: field_bits=%s groups=%d predicted_blocks=%d '
              'observed_blocks=%d' % (
                  r['row_id'], r['table'], r['field_bits'], r['groups'],
                  r['predicted_blocks'], r['observed_blocks']))


if __name__ == '__main__':
    main()
