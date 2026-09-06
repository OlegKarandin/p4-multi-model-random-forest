"""Unit tests for scripts/tcam_column_sweep.py -- the synthetic sweep that
settles Mechanism C (reviews/p4_tofino_reference.md Sec 7): does a Tofino
stage hold a flat 24 TCAM blocks, or 2 columns x 12 rows with a wide table's
blocks chained inside one column?

Pure-function tests only. The real p4c compiles are the point of the script
and are driven by its CLI, not by this suite.
"""
import os
import re
import textwrap

import pytest

import scripts.tcam_column_sweep as sweep


# ---------------------------------------------------------------------------
# The two competing predictions, and where they actually disagree.
# ---------------------------------------------------------------------------

def test_flat_24_and_two_column_predictions_differ_only_at_widths_7_and_8():
    # This is why the sweep runs 5..12 but only two of those widths carry any
    # information: everywhere else the two rules coincide, so a compile there
    # confirms both or neither.
    disagree = [w for w in sweep.BLOCK_WIDTHS
                if sweep.tables_per_stage_flat_24(w)
                != sweep.tables_per_stage_two_columns(w)]
    assert disagree == [7, 8]
    assert (sweep.tables_per_stage_flat_24(7),
            sweep.tables_per_stage_two_columns(7)) == (3, 2)
    assert (sweep.tables_per_stage_flat_24(8),
            sweep.tables_per_stage_two_columns(8)) == (3, 2)


def test_two_column_rule_never_predicts_more_than_the_flat_cap():
    # 2*floor(12/w) <= floor(24/w) for every w, so the column rule can only
    # ever be the TIGHTER of the two. A measurement above the flat cap would
    # refute both rules rather than pick between them.
    for w in sweep.BLOCK_WIDTHS:
        assert (sweep.tables_per_stage_two_columns(w)
                <= sweep.tables_per_stage_flat_24(w))


# ---------------------------------------------------------------------------
# Key width -> block count. The probe is only meaningful if each table really
# costs the W blocks it is supposed to.
# ---------------------------------------------------------------------------

def test_key_bits_produce_exactly_the_requested_block_count():
    # Sec 4.1's own formula: a ternary table costs
    # ceil((key_bits + 4) / TCAM_BLOCK_KEY_LENGTH) blocks per 512 entries.
    for w in sweep.BLOCK_WIDTHS:
        bits = sweep.key_bits_for(w)
        assert -(-(bits + sweep.TERNARY_ENTRY_OVERHEAD_BITS)
                 // sweep.TCAM_BLOCK_KEY_LENGTH) == w


def test_key_widths_are_byte_aligned_so_the_extra_block_anomaly_cannot_fire():
    # The sweep's own control caught this: the obvious 44*w - 4 is congruent
    # to 4 mod 8 for every EVEN w, and Sec 4.1.2's extra-block anomaly then
    # charges w+1 blocks (measured 7, 9, 11 at intended 6, 8, 10). An even-w
    # probe therefore measured the wrong table -- fatal at w=8, the width
    # where the two rules disagree. Byte-aligned keys avoid it.
    for w in sweep.BLOCK_WIDTHS:
        assert sweep.key_bits_for(w) % 8 == 0
    # ...and they stay in the same block bucket rather than dropping a block.
    assert sweep.key_bits_for(8) == 344
    assert sweep.key_bits_for(7) == 304


def test_key_never_exceeds_the_single_stage_crossbar_budget():
    # A key wider than TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE cannot be matched
    # in one stage at all, so a table built from one would measure the
    # crossbar rather than the TCAM geometry the sweep is after. At w=12 the
    # naive 44*12-4 bits would be 66 bytes and does exceed it, so that width
    # is capped rather than skipped.
    for w in sweep.BLOCK_WIDTHS:
        key_bytes = -(-sweep.key_bits_for(w) // 8)
        assert key_bytes <= sweep.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
    assert sweep.key_bits_for(12) == 8 * sweep.TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE


# ---------------------------------------------------------------------------
# Program generation.
# ---------------------------------------------------------------------------

def test_every_table_keys_the_same_field():
    # The load-bearing property of the whole probe. The Ternary Match Input
    # crossbar charges per distinct FIELD, so N tables on N private wide keys
    # would exhaust its 64 bytes long before 24 blocks bound -- two 6-block
    # tables are already 66 bytes. Sharing one field reproduces the real
    # generator (every tree of a task keys the identical code_* set) and
    # leaves blocks as the binding dimension.
    source = sweep.synthetic_program(block_width=7, n_tables=4)
    keys = re.findall(r'key\s*=\s*\{\s*meta\.(\w+)\s*:\s*ternary;', source)
    assert len(keys) == 4
    assert set(keys) == {sweep.KEY_FIELD}


def test_each_table_gets_its_own_solitary_result_field():
    # Mechanism A -- two tables whose actions write one PHV container cannot
    # share a stage -- would masquerade as a geometry limit here. @pa_solitary
    # on every written field removes it by construction, the same fix
    # generate_P4_code applies to class_tree_*/code_*.
    source = sweep.synthetic_program(block_width=7, n_tables=3)
    assert source.count('@pa_solitary("ingress", "ig_md.result_') == 3
    for i in range(3):
        assert '@pa_solitary("ingress", "ig_md.result_%d")' % i in source
        assert 'meta.result_%d = r;' % i in source


def test_tables_are_applied_unconditionally_and_independently():
    # No gateway, no shared written field, no data flow between them: any
    # stage split the compiler makes is a resource decision, not a dependency.
    source = sweep.synthetic_program(block_width=5, n_tables=3)
    for i in range(3):
        assert 'tern_table_%d.apply();' % i in source
    assert 'if (' not in source


def test_key_field_is_declared_at_the_requested_width():
    source = sweep.synthetic_program(block_width=9, n_tables=2)
    assert 'bit<%d> %s;' % (sweep.key_bits_for(9), sweep.KEY_FIELD) in source


def test_table_size_is_one_block_of_entries():
    # 512 entries = exactly one block's worth of rows, so the block count is
    # driven purely by key WIDTH. A larger size would multiply blocks by
    # ceil(size/512) and confound the width sweep.
    source = sweep.synthetic_program(block_width=6, n_tables=2)
    assert source.count('size = 512;') == 2


def test_rejects_a_width_the_sweep_cannot_build():
    with pytest.raises(ValueError):
        sweep.synthetic_program(block_width=0, n_tables=1)
    with pytest.raises(ValueError):
        sweep.synthetic_program(block_width=5, n_tables=0)


# ---------------------------------------------------------------------------
# Reading the committed placement back.
# ---------------------------------------------------------------------------

_FAKE_LOG = textwrap.dedent("""\
    Allocated Resource Usage
    ------------------------------------------------------------------
    |    Table    | Stage | Crossbar | Hash | Gateways | RAMs | TCAMs |
    ------------------------------------------------------------------
    | SwitchIngress.tern_table_0 |   3    |    28    |  0   |    0     |  0   |   7   |
    | SwitchIngress.tern_table_0$action |   3    |    0    |  0   |    0     |  1   |   0   |
    | SwitchIngress.tern_table_1 |   3    |    0    |  0   |    0     |  0   |   7   |
    | SwitchIngress.tern_table_2 |   3    |    0    |  0   |    0     |  0   |   7   |
    | SwitchIngress.tern_table_3 |   4    |    28    |  0   |    0     |  0   |   7   |
    """)


def test_measure_reports_tables_per_stage_and_blocks_per_table(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'mau.resources.log').write_text(_FAKE_LOG, encoding='utf-8')

    measured = sweep.measure(str(logs))

    # 3 tables in stage 3, 1 in stage 4 -- the stage the compiler filled is
    # the capacity, so max, not mean.
    assert measured['max_tables_per_stage'] == 3
    assert measured['blocks_per_table'] == 7
    assert measured['n_tables_placed'] == 4
    assert measured['occupied_stages'] == 2
    assert measured['max_blocks_per_stage'] == 21


def test_measure_ignores_action_rows(tmp_path):
    # $action rows carry the table's SRAM, not its TCAM, and counting them
    # would double every table.
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'mau.resources.log').write_text(_FAKE_LOG, encoding='utf-8')
    assert sweep.measure(str(logs))['n_tables_placed'] == 4


def test_measure_returns_none_when_the_backend_allocated_nothing(tmp_path):
    # Same contract as compiler_calibration._committed_blocks: a program the
    # backend refused has no committed placement, and inventing one would
    # record a failed compile as a measurement.
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'mau.resources.log').write_text('nothing here\n', encoding='utf-8')
    assert sweep.measure(str(logs)) is None


# ---------------------------------------------------------------------------
# The verdict, given measurements.
# ---------------------------------------------------------------------------

def test_verdict_names_flat_24_when_a_stage_holds_the_flat_cap():
    rows = [{'block_width': 7, 'max_tables_per_stage': 3},
            {'block_width': 8, 'max_tables_per_stage': 3}]
    assert sweep.verdict(rows) == 'flat_24'


def test_verdict_names_two_columns_when_every_stage_stops_at_the_column_cap():
    rows = [{'block_width': 7, 'max_tables_per_stage': 2},
            {'block_width': 8, 'max_tables_per_stage': 2}]
    assert sweep.verdict(rows) == 'two_columns'


def test_verdict_is_inconclusive_without_a_discriminating_width():
    # w=5,6,9..12 agree under both rules, so no number of them decides
    # anything.
    rows = [{'block_width': 5, 'max_tables_per_stage': 4},
            {'block_width': 10, 'max_tables_per_stage': 2}]
    assert sweep.verdict(rows) == 'inconclusive'


def test_verdict_reports_a_split_rather_than_picking_a_side():
    # The honest outcome if w=7 and w=8 disagree: neither rule as stated
    # survives, and saying "flat_24" because one width matched would repeat
    # exactly the mistake that made Mechanism C "suggestive, not established".
    rows = [{'block_width': 7, 'max_tables_per_stage': 3},
            {'block_width': 8, 'max_tables_per_stage': 2}]
    assert sweep.verdict(rows) == 'split'


def test_verdict_flags_a_measurement_above_the_flat_cap():
    rows = [{'block_width': 7, 'max_tables_per_stage': 4}]
    assert sweep.verdict(rows) == 'refutes_both'


def test_verdict_ignores_tables_wider_than_a_column():
    # The wide arm's points (14, 16, 24 blocks) answer whether one table can
    # SPAN both columns, not how many share a stage -- the column rule
    # predicts 0 tables per stage there, which is not a competing prediction.
    # Scoring them would have this report 'flat_24' off measurements that say
    # nothing about the choice.
    rows = [{'block_width': 14, 'max_tables_per_stage': 1},
            {'block_width': 24, 'max_tables_per_stage': 1}]
    assert sweep.verdict(rows) == 'inconclusive'


# ---------------------------------------------------------------------------
# The column hypothesis proper: pack into 2 columns of 12, each table's blocks
# inside ONE column. tables_per_stage_two_columns is only its uniform-width
# special case, and conflating the two is what made Mechanism C look refuted.
# ---------------------------------------------------------------------------

def test_two_columns_is_a_bin_packing_not_a_per_width_quotient():
    # Three tables of 5, 4 and 3 blocks pack as (5+4 | 3) and really do fit,
    # while the uniform shortcut reasons about a width none of them has.
    assert sweep.fits_two_columns([5, 4, 3])
    # Three 7-block tables cannot: any column holding two needs 14 of 12.
    assert not sweep.fits_two_columns([7, 7, 7])
    # ...even though their 21 blocks pass a flat 24-block cap, which is
    # exactly what makes width 7 discriminating.
    assert sum([7, 7, 7]) <= sweep.TCAM_BLOCKS_PER_STAGE


def test_a_table_wider_than_one_column_never_fits():
    assert not sweep.fits_two_columns([13])
    assert sweep.fits_two_columns([12])


def test_two_columns_packing_is_exact_not_greedy():
    # A first-fit-decreasing pass puts 7 then 5 in one column and 6 in the
    # other, then fails on the last 6; the true packing is (7+5 | 6+6). A
    # greedy check would report a violation the hardware does not have, and
    # such false violations are what a refutation gets built out of.
    assert sweep.fits_two_columns([7, 6, 6, 5])


def test_two_columns_agrees_with_the_shortcut_on_uniform_widths():
    for w in sweep.BLOCK_WIDTHS:
        cap = sweep.tables_per_stage_two_columns(w)
        assert sweep.fits_two_columns([w] * cap)
        assert not sweep.fits_two_columns([w] * (cap + 1))


# ---------------------------------------------------------------------------
# Sweep points.
# ---------------------------------------------------------------------------

def test_point_ids_pin_the_key_width_they_were_built_from():
    # A change to key_bits_for must not silently resume onto artifacts built
    # from the old one -- the key width is what determines the block cost, so
    # it belongs in the identity of the measurement.
    ids = [p[0] for p in sweep.sweep_points((7, 8))]
    assert all('k%d' % sweep.key_bits_for(7) in i for i in ids if i.startswith('w7'))
    assert all('k%d' % sweep.key_bits_for(8) in i for i in ids if i.startswith('w8'))


def test_every_width_gets_the_flat_cap_and_a_saturating_n():
    points = {p[0]: (p[1], p[2]) for p in sweep.sweep_points((7,))}
    assert sorted(n for _, n in points.values()) == [3, 7]
