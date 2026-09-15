"""B1: the archived compiles' per-table offsets, scored against the model.

These tests pin the two facts Task 7 adjudicates on, so a later change to
either the harvest or the cost rule cannot quietly move them.
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


def test_the_app_key_starts_at_the_ddos_tables_block_count_not_its_group_count():
    """The one measurement in the whole archive that discriminates finding 1.4.

    The ddos key is 11 crossbar bytes = 2 groups and costs 3 blocks (2 groups +
    the version penalty). p4c starts the app key at group 3, i.e. it advanced
    the offset by the BLOCK count. packing.offsets_for summing blocks is
    therefore what the hardware does; summing group counts would predict 2.
    """
    rows = {r['table']: r for r in harvest.harvest_row('independent_low_sd5')}

    ddos = rows['get_classification_tree_ddos_0']
    app = rows['get_classification_tree_app_0']

    assert (ddos['key_bytes'], ddos['groups'], ddos['observed_blocks']) == (11, 2, 3)
    assert ddos['measured_start_group'] == 0
    assert app['measured_start_group'] == 3


@pytest.mark.parametrize('row_id', harvest.ROW_IDS)
def test_every_archived_classification_table_is_predicted_exactly(row_id):
    """The whole point of the ledger: the model, evaluated at each table's
    MEASURED offset, reproduces p4c's own per-table block count."""
    misses = [r for r in harvest.harvest_row(row_id)
              if not r['start_group_ambiguous']
              and r['predicted_blocks'] != r['observed_blocks']]

    assert misses == [], misses
