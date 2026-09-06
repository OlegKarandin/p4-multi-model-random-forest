"""Unit tests for scripts/tcam_stretch_sweep.py -- the synthetic sweep that
settles Mechanism G (reviews/p4_tofino_reference.md Sec 7): when two
DIFFERENT ternary keys share a stage, does the second one cost an extra TCAM
block?

Pure-function tests only. The real p4c compiles are the point of the script
and are driven by its CLI, not by this suite.
"""
import json
import os
import re

import pytest

import scripts.tcam_stretch_sweep as sweep


# ---------------------------------------------------------------------------
# Key specs: the solid/ragged distinction the whole sweep turns on.
# ---------------------------------------------------------------------------

def test_an_int_key_is_one_solid_field_of_that_many_bytes():
    assert sweep.as_fields(49) == (392,)
    assert sweep.key_bytes_for(49) == 49


def test_a_tuple_key_is_a_list_of_field_bit_widths():
    assert sweep.as_fields(sweep.SD9_APP_KEY) == (179, 204)


def test_the_two_arms_present_the_same_crossbar_bytes_and_blocks():
    # The arms have to be comparable or the sweep measures raggedness
    # confounded with size. 179 bits is 23 bytes and 204 is 26 -- byte-rounded
    # per FIELD, exactly as evaluation.ternary_table_key_bytes does it -- so
    # the ragged app key costs the same 49 bytes and 9 blocks as the solid
    # 392-bit one, and the ddos key the same 12 bytes and 3 blocks.
    assert sweep.key_bytes_for(sweep.SD9_APP_KEY) == sweep.key_bytes_for(49) == 49
    assert sweep.blocks_for_key(sweep.SD9_APP_KEY) == sweep.blocks_for_key(49) == 9
    assert sweep.key_bytes_for(sweep.SD9_DDOS_KEY) == sweep.key_bytes_for(12) == 12
    assert sweep.blocks_for_key(sweep.SD9_DDOS_KEY) == sweep.blocks_for_key(12) == 3


def test_the_ragged_key_really_is_ragged_and_the_solid_one_is_not():
    # The property under test: a field whose width is not a multiple of 8
    # hands the crossbar a part-used byte, and only such a byte can ride a
    # midbyte nibble.
    assert all(bits % 8 for bits in sweep.as_fields(sweep.SD9_APP_KEY))
    assert all(bits % 8 for bits in sweep.as_fields(sweep.SD9_DDOS_KEY))
    assert not any(bits % 8 for bits in sweep.as_fields(49))


def test_sd9_keys_match_the_row_they_are_taken_from():
    # Read off results/compiler_calibration_extra/p4_src/independent_low_sd9.p4:
    # code_app_fwd_packet_length_max is bit<179>, code_packet_length_mean
    # bit<204>, code_bwd_packet_length_mean bit<37>,
    # code_ddos_fwd_packet_length_max bit<49>. Pinned so a probe cannot drift
    # into measuring an analogy of that row instead of the row.
    assert sweep.SD9_APP_KEY == (179, 204)
    assert sweep.SD9_DDOS_KEY == (37, 49)


# ---------------------------------------------------------------------------
# The generated program.
# ---------------------------------------------------------------------------

def test_the_two_keys_are_separate_fields_so_a_midbyte_can_be_split():
    # A shared key is charged once by the crossbar and occupies one group
    # run, so it can never put a second key at a non-zero group offset. Two
    # distinct fields is the entire premise of this probe.
    source = sweep.synthetic_program(49, 1, 12, 1)
    assert 'bit<392> key_a0;' in source
    assert 'bit<96> key_b0;' in source
    assert 'meta.key_a0 : ternary;' in source
    assert 'meta.key_b0 : ternary;' in source


def test_a_ragged_key_emits_one_key_line_per_field():
    # build_p4_script emits one `meta.code_<feature> : ternary` line per
    # selected feature; the crossbar allocates per field, so the split is
    # part of what is under test rather than a formatting choice.
    source = sweep.synthetic_program(sweep.SD9_APP_KEY, 1, sweep.SD9_DDOS_KEY, 1)
    for declared in ('bit<179> key_a0;', 'bit<204> key_a1;',
                     'bit<37> key_b0;', 'bit<49> key_b1;'):
        assert declared in source
    table_a = source.split('table tern_a0 {')[1].split('}')[0]
    assert 'meta.key_a0 : ternary;' in table_a
    assert 'meta.key_a1 : ternary;' in table_a


def test_every_table_writes_its_own_solitary_field():
    # Otherwise Mechanism A (two tables whose actions write one PHV
    # container cannot share a stage) would masquerade as a block-cost limit.
    source = sweep.synthetic_program(49, 2, 12, 2)
    for name in ('result_a0', 'result_a1', 'result_b0', 'result_b1'):
        assert '@pa_solitary("ingress", "ig_md.%s")' % name in source
        assert 'meta.%s = r;' % name in source


def test_key_fields_are_pinned_solitary_too():
    # The generator pins its own code_* fields (Mechanism A's fix), and the
    # crossbar layout a key gets depends on how PHV packed it, so an unpinned
    # probe key would not be measuring the same thing the generator emits.
    source = sweep.synthetic_program(sweep.SD9_APP_KEY, 1, 12, 1)
    assert '@pa_solitary("ingress", "ig_md.key_a0")' in source
    assert '@pa_solitary("ingress", "ig_md.key_a1")' in source


def test_tables_are_applied_unconditionally_and_independently():
    source = sweep.synthetic_program(49, 1, 12, 2)
    applies = re.findall(r'tern_(\w+)\.apply\(\);', source)
    assert applies == ['a0', 'b0', 'b1']
    assert 'if (' not in source.split('apply {')[-1]


def test_table_size_is_one_block_of_entries():
    # size drives a table's row count; fixing it at one block's worth leaves
    # KEY WIDTH as the only thing setting the block count.
    source = sweep.synthetic_program(49, 1, 12, 1)
    assert source.count('size = 512;') == 2


def test_rejects_a_program_with_no_wide_table():
    with pytest.raises(ValueError, match='n_a >= 1'):
        sweep.synthetic_program(49, 0, 12, 1)


# ---------------------------------------------------------------------------
# Reading the compiler back.
# ---------------------------------------------------------------------------

def _write_resources(tmp_path, cells_by_stage):
    logs = tmp_path / 'logs'
    logs.mkdir(exist_ok=True)
    stages = []
    for stage, cells in sorted(cells_by_stage.items()):
        stages.append({
            'stage_number': stage,
            'tcams': {'nRows': 12, 'nColumns': 2, 'tcams': [
                {'column': column, 'row': row,
                 'usages': [{'used_by': 'SwitchIngress.' + table,
                             'used_for': 'ternary_match'}]}
                for table, column, row in cells]},
        })
    (logs / 'resources.json').write_text(json.dumps(
        {'resources': {'mau': {'nStages': len(stages), 'mau_stages': stages}}}),
        encoding='utf-8')
    return str(logs)


def test_committed_tcam_grid_reads_the_column_and_row_of_every_block(tmp_path):
    logs = _write_resources(tmp_path, {
        3: [('tern_a0', 0, 0), ('tern_a0', 0, 1), ('tern_b0', 1, 4)]})
    grid = sweep.committed_tcam_grid(logs)
    assert grid == {3: {(0, 0): 'tern_a0', (0, 1): 'tern_a0',
                        (1, 4): 'tern_b0'}}


def test_committed_tcam_grid_is_empty_when_the_backend_allocated_nothing(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    assert sweep.committed_tcam_grid(str(logs)) == {}


def test_spans_report_the_column_and_first_row_of_a_contiguous_table():
    grid = {(0, 3): 't', (0, 4): 't', (0, 5): 't'}
    assert sweep.spans(grid) == {'t': (0, 3, 3)}


def test_spans_flag_a_table_that_is_not_one_run_in_one_column():
    # A table wider than a column really does span both (measured at 14, 16
    # and 24 blocks), so this is a fact to surface, not an error -- but it
    # must never be reported as a start row in one column.
    grid = {(0, 0): 't', (1, 0): 't'}
    assert sweep.spans(grid)['t'][0] is None


# ---------------------------------------------------------------------------
# The sweep's own shape.
# ---------------------------------------------------------------------------

def test_the_sweep_runs_both_arms_over_the_same_geometry():
    points = {point_id: (key_a, n_a, key_b, n_b)
              for point_id, key_a, n_a, key_b, n_b in sweep.sweep_points()}
    solid = points['a49x1_b12x5']
    ragged = points['ragged_ax1_bx5']
    assert (solid[1], solid[3]) == (ragged[1], ragged[3])
    assert (sweep.blocks_for_key(solid[0]), sweep.blocks_for_key(solid[2])) == \
           (sweep.blocks_for_key(ragged[0]), sweep.blocks_for_key(ragged[2]))


def test_the_discriminating_points_are_exactly_one_stage_worth_of_blocks():
    # 24 blocks is the whole question: at 21 the stage has slack for the
    # extra block and at 25 nothing could fit anyway, so only a point that
    # sums to exactly TCAM_BLOCKS_PER_STAGE under the OLD pricing separates
    # the two arms.
    totals = {}
    for point_id, key_a, n_a, key_b, n_b in sweep.sweep_points():
        totals[point_id] = (n_a * sweep.blocks_for_key(key_a)
                            + n_b * sweep.blocks_for_key(key_b))
    assert totals['a49x1_b12x5'] == totals['ragged_ax1_bx5'] == 24
    assert totals['a49x2_b12x2'] == totals['ragged_ax2_bx2'] == 24
    # ... and the 21-block control, which must fit in BOTH arms.
    assert totals['a49x1_b12x4'] == totals['ragged_ax1_bx4'] == 21


def test_point_ids_are_unique():
    ids = [point[0] for point in sweep.sweep_points()]
    assert len(ids) == len(set(ids))


def test_already_done_skips_recorded_points(tmp_path):
    out = tmp_path / 'sweep.csv'
    out.write_text('point_id,blocks_a\na49x1_b12x5,9\n', encoding='utf-8')
    assert sweep.already_done(str(out)) == {'a49x1_b12x5'}
    assert sweep.already_done(str(tmp_path / 'missing.csv')) == set()
