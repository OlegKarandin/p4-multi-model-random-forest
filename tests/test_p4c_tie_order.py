"""p4c's tie order among ready tables, and per-tree readiness (2026-11-08).

Investigation: results/encoding_evidence_2026_11/per_tree_model2 (notes.md, report.md). When more tables are ready for a
stage than it can hold, p4c's placer (table_placement.cpp DecidePlacement::is_better, read in its own placement log) takes
the longer dependence chain first, then the table more classification trees depend on, then the table listed LATER.
The model now follows that order for registers (register_stage_schedule, a stage-by-stage list schedule) and for the
range pool (crossbar_stages_needed's table_dependents), and a tree waits only for the range tables its own key reads
(usage.tree_readiness_levels, rule E).

The three fixture programs are campaign_2026_11d designs, byte-identical to what p4c compiled (sha256 prefixes in
FIXTURES). Each was mispriced by one of the partial fixes on the way, so together they pinned the whole rule:
  independent_M075_s20_k10  register tie-break alone (no range-pool order): 11 stages, p4c 12
  independent_M075_s20_k09  range-pool order with a level-first register greedy: 13 stages, p4c 12
  joint-task_M075_s20_k12   the old model: 53 blocks, p4c 51
They carry gated register blocks (`if (meta.fwd == ...)`), so since decisions 1a / 7.1 option B (2026-11-14) the
replay refuses them; the register tie order is now pinned by synthetic schedules below.
"""
import hashlib
import os

import pytest

from p4gen import p4_replay
from p4gen.p4_replay import parse_program, tree_range_reads
from p4model import registers
from p4model.packing import crossbar_stages_needed
from p4model.program import APP_TASK, DDOS_TASK, SHARED_TASK
from p4model.usage import tree_readiness_levels

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'p4c_tie_order')
# row_id: (p4c stage_depth, p4c blocks, sha256 prefix of the compiled program)
FIXTURES = {
    'independent_M075_s20_k09': (12, 71, '0cabfe396b14'),
    'independent_M075_s20_k10': (12, 53, 'de2dd4365bcf'),
    'joint-task_M075_s20_k12': (12, 51, 'ed9fb2ad74ea'),
}


def _program(row_id):
    path = os.path.join(HERE, row_id + '.p4')
    with open(path, 'rb') as handle:
        # Hashed with LF line ends: git may check the fixture out with CRLF (core.autocrlf).
        text = handle.read().replace(b'\r\n', b'\n')
    assert hashlib.sha256(text).hexdigest().startswith(FIXTURES[row_id][2])
    return parse_program(path)


@pytest.mark.parametrize('row_id', sorted(FIXTURES))
def test_the_designs_that_pinned_the_tie_order_are_refused_as_gated_programs(row_id):
    with pytest.raises(ValueError, match='gated register block'):
        _program(row_id)


def test_a_newly_ready_register_with_more_dependents_beats_one_spilled_from_earlier():
    # The rule independent_M075_s20_k09 showed in p4c's log ("fewer total dependencies in dominance frontier"): the
    # rank ignores how long a register has been ready. Seven candidates at the first register stage spill four
    # (the orientation register and the three earliest-listed); at the next stage c_val, ready only now but read by
    # seven trees, goes first, and the orientation register (no feature lists it, so it is "earliest") waits again.
    from p4model.program import FLOW_HASH_LEVEL, ORIENTATION_REGISTER
    catalog = {f'a{i}': {'registers': [{'name': f'a{i}_reg'}]} for i in range(6)}
    catalog['c'] = {'registers': [{'name': 'c_dep'}, {'name': 'c_val'}]}
    features = ['c'] + [f'a{i}' for i in range(6)]
    placed = registers.register_stage_schedule(features, catalog=catalog, dependents={'c': 7})
    first = FLOW_HASH_LEVEL
    assert {name for name, stage in placed.items() if stage == first} == {'c_dep', 'a5_reg', 'a4_reg', 'a3_reg'}
    assert placed['c_val'] == first + 1
    assert {name for name, stage in placed.items() if stage == first + 2} == {ORIENTATION_REGISTER}


def test_without_dependents_the_later_listed_register_goes_first():
    # Five one-register, ungated features, all ready at the first register stage, plus the orientation register the
    # generator always emits: six candidates for four stateful ALUs. With no dependents p4c's "default choice" keeps
    # the later-listed ones; a feature more trees read goes first whatever its position.
    from p4model.program import FLOW_HASH_LEVEL, ORIENTATION_REGISTER
    catalog = {f'f{i}': {'registers': [{'name': f'f{i}_reg'}]} for i in range(5)}
    features = [f'f{i}' for i in range(5)]
    placed = registers.register_stage_schedule(features, catalog=catalog)
    first = FLOW_HASH_LEVEL
    assert {name for name, stage in placed.items() if stage == first} == {'f1_reg', 'f2_reg', 'f3_reg', 'f4_reg'}
    assert placed['f0_reg'] == placed[ORIENTATION_REGISTER] == first + 1
    favoured = registers.register_stage_schedule(features, catalog=catalog, dependents={'f0': 5})
    assert favoured['f0_reg'] == first and favoured['f1_reg'] == first + 1


def test_range_pool_leaves_over_the_table_fewest_trees_read_then_the_first_listed():
    # Nine one-block range tables ready at stage 5; a stage holds eight tables. Old order: the LAST listed spills to 6.
    specs = [(1, 2)] * 9
    fields = [frozenset({(('val', i), 2)}) for i in range(9)]
    levels = [5] * 9
    old = crossbar_stages_needed(specs, readiness_levels=levels, key_fields=fields)
    assert old.table_stages.index(6) == 8
    equal = crossbar_stages_needed(specs, readiness_levels=levels, key_fields=fields, table_dependents=[4] * 9)
    assert equal.table_stages.index(6) == 0
    fewest = crossbar_stages_needed(specs, readiness_levels=levels, key_fields=fields,
                                    table_dependents=[4, 4, 4, 1, 4, 4, 4, 4, 4])
    assert fewest.table_stages.index(6) == 3
    with pytest.raises(ValueError, match='positionally aligned'):
        crossbar_stages_needed(specs, readiness_levels=levels, key_fields=fields, table_dependents=[1])
    with pytest.raises(ValueError, match='placement_priority'):
        crossbar_stages_needed(specs, readiness_levels=levels, key_fields=fields, table_dependents=[1] * 9,
                               key_field_bits=[(16,)] * 9)


def test_a_tree_waits_for_the_tables_it_reads_and_never_precedes_the_last():
    stages = (4, 6, 8)
    labels = [SHARED_TASK] * 3
    # By label every tree reads all three: one stage past the last.
    assert tree_readiness_levels(stages, labels, [APP_TASK, DDOS_TASK]) == [9, 9]
    # By key: the app tree reads only the stage-4 table, so it may share stage 8 with the last range table but not
    # precede it; the ddos tree reads the stage-8 table and waits for 9.
    assert tree_readiness_levels(stages, labels, [APP_TASK, DDOS_TASK], tree_reads=[[0], [1, 2]]) == [8, 9]
    with pytest.raises(ValueError, match='positionally aligned'):
        tree_readiness_levels(stages, labels, [APP_TASK, DDOS_TASK], tree_reads=[[0]])
    with pytest.raises(ValueError, match='outside'):
        tree_readiness_levels(stages, labels, [APP_TASK], tree_reads=[[3]])


def test_tree_range_reads_maps_every_key_or_gives_up():
    tables = {'table_0_a': ['a_val'], 'table_1_app_b': ['b_val'], 'table_2_tk': ['tk_val'],
              'get_classification_tree_app_0': ['code_a', 'code_app_b'],
              'get_classification_tree_ddos_0': ['code_a', 'code_tk_ddos_tk']}
    assert tree_range_reads(tables) == {'get_classification_tree_app_0': ['table_0_a', 'table_1_app_b'],
                                        'get_classification_tree_ddos_0': ['table_0_a', 'table_2_tk']}
    tables['get_classification_tree_ddos_0'].append('code_nowhere')
    assert tree_range_reads(tables) is None
    # The replay then falls back to task labels for readiness, and prices as before.
    assert p4_replay.table_tasks(tables)['table_0_a'] == SHARED_TASK
