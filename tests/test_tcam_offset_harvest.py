"""Per-table crossbar placement facts harvested from the 19 archived p4c
compiles (results/compiler_calibration_v6), scored against the model.

Originally written for Task 7's offset-based mechanism (`start_group`,
`version_block_penalty`), which the 2026-09-20 rewrite retired: no per-table
price in `src.p4model.tables` depends on where a key's crossbar run starts
any more (see `packing.key_width`'s "HONEST LIMIT" note). The harvest module
this file exercises (`scripts/tcam_offset_harvest.py`) already says so in its
own comments. What these tests still pin: the raw archived measurements
(crossbar offsets, block counts) as historical ground truth, and a per-table
exactness check that duplicates `scripts/tcam_table_scoreboard.py`'s 100/100
gate on this one archive rather than adding new coverage.
"""
import pytest

from scripts import tcam_offset_harvest as harvest


def test_harvest_reads_independent_low_sd5s_four_classification_tables():
    rows = harvest.harvest_row('independent_low_sd5')

    assert len(rows) == 4
    assert {r['table'] for r in rows} == {
        'get_classification_tree_app_0',
        'get_classification_tree_ddos_0',
        'get_classification_tree_ddos_1',
        'get_classification_tree_ddos_2',
    }


def test_the_archived_app_key_started_at_the_ddos_keys_block_count_not_its_group_count():
    """Historical ground truth, not a live discrimination.

    The ddos key is 11 crossbar bytes = 2 groups and costs 3 blocks (2 groups +
    a version-nibble block). p4c started the UNRELATED app key at crossbar
    group 3 in this archived compile -- the ddos key's BLOCK count, not its
    GROUP count. Back when a key's own price depended on where its crossbar
    run started, this was the fact `packing.offsets_for`'s width-based
    (rather than group-based) sum was built to match. The 2026-09-20 rewrite
    deleted `start_group` from every per-table price, so this measurement no
    longer discriminates anything the current model reads -- it is kept as
    the archived fact itself, not as a live test of finding 1.4.
    """
    rows = {r['table']: r for r in harvest.harvest_row('independent_low_sd5')}

    ddos = rows['get_classification_tree_ddos_0']
    app = rows['get_classification_tree_app_0']

    assert (ddos['key_bytes'], ddos['groups'], ddos['observed_blocks']) == (11, 2, 3)
    assert ddos['measured_start_group'] == 0
    assert app['measured_start_group'] == 3


@pytest.mark.parametrize('row_id', harvest.ROW_IDS)
def test_every_archived_classification_table_is_predicted_exactly(row_id):
    """Per-table exactness on this one archive -- duplicates the 100/100
    archived-table result `scripts/tcam_table_scoreboard.py` reports over all
    14 result CSVs; kept here because it replays directly off this harvest's
    own fixture with no extra harness."""
    misses = [r for r in harvest.harvest_row(row_id)
              if not r['start_group_ambiguous']
              and r['predicted_blocks'] != r['observed_blocks']]

    assert misses == [], misses
