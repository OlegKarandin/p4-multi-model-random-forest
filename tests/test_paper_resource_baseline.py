"""scripts/paper_resource_baseline.py (paper Table II): arm / k grouping, rows it
skips, and the per-tree baseline's live key bits (final review src I2, I3)."""
import csv
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'scripts'))
from paper_resource_baseline import (arm_of, baseline, grouped, k_group,  # noqa: E402
                                     load_run)
from p4gen.p4_replay import parse_program_text, tree_key_bits  # noqa: E402
from p4model.target import (TCAM_BLOCK_KEY_LENGTH,  # noqa: E402
                            TERNARY_MATCHING_ENTRIES_PER_BLOCK)

BASE_17B = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'p4c_base_17b')


def test_arm_and_k_come_from_the_row_id():
    assert arm_of('joint-off-al_M035_s07_k09') == 'joint-off-al'
    assert arm_of('joint-tree_Minf_s00_k12') == 'joint-tree'
    assert arm_of('heldout_independent_M150_k14_s13') == 'independent'   # archive name
    assert k_group('joint-task_M075_s13_k04') == 'k<=8'
    assert k_group('joint-task_M075_s13_k09') == 'k>=9'
    assert k_group('heldout_independent_M150_k14_s13') == 'k>=9'
    assert k_group('no_k_here') is None


def test_per_tree_trees_are_charged_their_own_slices():
    with open(os.path.join(BASE_17B, 'joint-tree_M075_s02_k07.p4'), encoding='utf-8') as fh:
        text = fh.read()
    program = parse_program_text(text)
    trees = [n for n in program.tables if n.startswith('get_classification_tree')]
    ranges = [n for n in program.tables if n.startswith('table_') and program.tables[n]]
    rows = lambda n: math.ceil(program.sizes[n] / TERNARY_MATCHING_ENTRIES_PER_BLOCK)
    expected = (sum(rows(n) for n in ranges)
                + sum(rows(n) * math.ceil(tree_key_bits(program, n) / TCAM_BLOCK_KEY_LENGTH)
                      for n in trees))
    declared = sum(program.bits[k] for n in trees for k in program.tables[n])
    assert sum(tree_key_bits(program, n) for n in trees) < declared
    assert baseline(text)[1] == expected


def _line(row_id, **kw):
    line = {'row_id': row_id, 'M': '75', 'p4c_stage_depth': '9', 'p4c_blocks': '30',
            'model_stage_depth': '9', 'model_blocks': '30', 'copied_from': ''}
    line.update(kw)
    return line


def test_load_run_skips_unverified_rows_and_counts_a_copied_twin_once(tmp_path):
    designs = tmp_path / 'designs'
    designs.mkdir()
    lines = [_line('joint-off_M075_s00_k07'),
             _line('joint-off-al_M075_s00_k07', copied_from='joint-off_M075_s00_k07'),
             _line('joint-off_M075_s01_k12', p4c_stage_depth='', p4c_blocks=''),
             _line('independent_M075_s00_k07')]
    for line in lines:
        shutil.copy(os.path.join(BASE_17B, 'joint-off_M075_s19_k07.p4'),
                    designs / (line['row_id'] + '.p4'))
    with open(tmp_path / 'verification.csv', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    rows, skipped = load_run(str(tmp_path))
    assert sorted(r['row_id'] for r in rows) == ['independent_M075_s00_k07',
                                                 'joint-off_M075_s00_k07']
    assert skipped == {'copied': ['joint-off-al_M075_s00_k07'],
                       'no_p4c': ['joint-off_M075_s01_k12']}
    cells = dict(grouped(rows))
    assert [key for key in cells] == [
        ('independent', 'k<=8'), ('independent', 'all'), ('joint-off', 'k<=8'),
        ('joint-off', 'all'), ('all', 'k<=8'), ('all', 'all')]
    assert len(cells[('all', 'all')]) == 2
