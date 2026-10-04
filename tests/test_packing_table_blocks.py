"""StagePlan.table_blocks (spec 2026-09-29 §5.3): what each table is CHARGED,
so model.json can be joined to p4c's mau.resources.log by table name."""
import pytest

from src.p4model.packing import crossbar_stages_needed
from src.p4model.usage import assemble_usage
from tests.test_resource_model_golden import load_fixture, rebuild_pool

APP_49 = (179, 204)      # 49 crossbar bytes, 9 blocks standalone, ragged
DDOS_12 = (37, 49)       # 12 crossbar bytes, 3 blocks


def test_declared_pool_charges_each_table_its_declared_blocks():
  specs = [(1, 2), (3, 2), (13, 4)]
  plan = crossbar_stages_needed(specs)
  assert plan.table_blocks == (1, 3, 13)
  assert sum(plan.table_blocks) == plan.blocks


def test_a_later_key_is_charged_its_lane_price_not_its_declared_blocks():
  # Same call as tests/test_evaluation.py::_stretch_probe(APP_49, 4): behind the
  # 12-byte key the ragged 49-byte key needs 10 blocks, not its declared 9.
  plan = crossbar_stages_needed(
      [(9, 49)] + [(3, 12)] * 4,
      key_fields=[frozenset({(('a',), 49)})] +
                 [frozenset({(('b',), 12)})] * 4,
      key_field_bits=[APP_49] + [DDOS_12] * 4)
  assert plan.table_blocks[0] == 10
  assert plan.table_blocks[1:] == (3, 3, 3, 3)
  assert sum(plan.table_blocks) == plan.blocks == 22


@pytest.mark.parametrize("row", load_fixture()["rows"],
                         ids=lambda row: row["row_id"])
def test_table_blocks_sum_to_the_plan_total_on_every_golden_row(row):
  _usage, range_plan, ternary_plan = assemble_usage(rebuild_pool(row))
  for plan in (range_plan, ternary_plan):
    assert len(plan.table_blocks) == len(plan.table_stages)
    assert sum(plan.table_blocks) == plan.blocks


def test_ternary_matching_prices_each_tree_with_table_blocks():
    """independent_low_sd5's ddos key (27, 52): ladder 3, lane 2 per word."""
    from src.p4model.tables import ternary_matching_resource_usage
    intervals = {'a': list(range(28)), 'b': list(range(53))}   # widths 27, 52
    codewords = {0: {'0' * 79: 0}, 1: {'0' * 79: 0}}
    entries, blocks, length, specs = ternary_matching_resource_usage(codewords, intervals)
    assert specs == [(2, 11), (2, 11)]
    assert blocks == 4
