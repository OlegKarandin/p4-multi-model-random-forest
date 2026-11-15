"""Decision 6a: ONE limit on a lone table key, target.MAX_KEY_BYTES, read by every
site that refuses a key. Each site is exercised at the limit and one byte past it,
so a site that kept its own copy of the old 64 would fail here. 65 is p4c's limit
(Task 17 probe: 64 and 65 bytes compile at 12 blocks, 66 is refused, 528b > 524b)."""
import pytest

from p4gen import evaluation as ev
from p4model import lanes, packing, tables, target
from p4model.errors import CrossbarKeyTooWide


def test_the_limit_is_the_probed_65_bytes():
  assert target.MAX_KEY_BYTES == 65


def _byte_wide_intervals(n_fields):
  """n_fields features of 8 code bits (9 intervals) each: n_fields crossbar bytes."""
  return {"F%d" % i: [(j, j) for j in range(9)] for i in range(n_fields)}


def _training(n_bytes):
  """tables.ternary_matching_resource_usage, the joint / disjoint / per-task training refusal."""
  return tables.ternary_matching_resource_usage(
      {0: {"0" * 8 * n_bytes: 0}}, _byte_wide_intervals(n_bytes))


def _lane_price(n_bytes):
  lanes._table_blocks.cache_clear()
  return lanes.table_blocks((8,) * n_bytes)


def _stage_test(n_bytes):
  """packing's stage test, which the replay of a program text and the range pool reach."""
  return packing.crossbar_stages_needed([(1, n_bytes)])


def _packed_training(n_bytes):
  """evaluation._packed_tree_entries, the per-tree training refusal."""
  key = tuple(("B", 0, False, index) for index in range(n_bytes))
  return ev._packed_tree_entries({0: {"0": 0}}, key, False)


SITES = [_training, _lane_price, _stage_test, _packed_training]


@pytest.mark.parametrize("site", SITES, ids=lambda site: site.__name__)
def test_one_key_limit(site):
  site(65)                                        # at the limit: accepted
  with pytest.raises(CrossbarKeyTooWide) as excinfo:
    site(66)                                      # one past it: refused
  assert excinfo.value.args[1] == 66


def test_a_range_key_counts_four_crossbar_byte_slots():
  """Decision 6f: RANGE_LO + RANGE_HI per byte of the 16-bit key, the unit
  xbar.range_key uses; the specs and the stage fields carry the same 4."""
  intervals = {"Flow IAT Max": [(0, 10), (11, 20)]}
  (fields,) = tables.range_key_fields_for(intervals)
  assert {size for _, size in fields} == {4}
  _, _, specs = tables.range_matching_resource_usage(intervals)
  assert [width for _, width in specs] == [4]
