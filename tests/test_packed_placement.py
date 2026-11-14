"""crossbar_stages_needed's explicit-key-bytes mode (packed codes): the stage's crossbar is simulated as p4c
allocates it (p4model/xbar.py, spec 2026-11-13): range keys first, then tree keys in placement order, each on the
slots the earlier ones left. A tree that no longer fits waits for a later stage; a tree that fits is charged its
rows x the blocks of its actual allocation. No @stage pin is written."""
import pytest
from p4model import lanes, packing, xbar
from p4model.errors import CrossbarKeyTooWide
from p4model.packing import StageLoad, crossbar_stages_needed


def _key(n):                      # n bytes, each in an 8-bit container of its own (any lane)
  return tuple(("B", 0, False) for _ in range(n))


def _plan(byte_counts, rows=None, seeds=(), levels=None, priorities=None):
  rows = rows or [1] * len(byte_counts)
  keys = [_key(n) for n in byte_counts]
  specs = [(packing.packed_alone_blocks(tuple(k)) * r, n) for k, r, n in zip(keys, rows, byte_counts)]
  return crossbar_stages_needed(specs, readiness_levels=levels or [5] * len(specs), key_bytes=keys,
                                placement_priority=priorities or [1] * len(specs), seed_stages=seeds)


def _range_seed(tables):          # `tables` range tables in stage 5, a 2-byte value key = 4 byte slots each (6f)
  return StageLoad(index=5, blocks=(1,) * tables, fields=frozenset((("r", i), 4) for i in range(tables)),
                   tables=tables)


def test_keys_filling_12_groups_share_a_stage_at_their_own_price():
  plan = _plan([20, 20, 20])                     # 4 + 4 + 4 groups
  assert plan.table_stages == (5, 5, 5) and plan.table_blocks == (4, 4, 4)


def test_three_keys_can_fill_the_whole_crossbar():
  plan = _plan([22, 22, 22])                     # 66 bytes: 12 groups and all 6 midbytes
  assert plan.table_stages == (5, 5, 5) and plan.table_blocks == (5, 5, 5)


def test_a_key_that_no_longer_fits_waits_for_the_next_stage_and_is_not_pinned():
  # placement order: equal priority, the table listed last first -- 3, 2, 1 fill stage 5; 0 has no room
  plan = _plan([22, 22, 22, 5])
  assert plan.table_stages == (6, 5, 5, 5) and plan.table_blocks == (5, 5, 5, 1)


def test_a_later_key_is_charged_for_the_leftover_slots_it_gets():
  # five range tables take 20 slots first; index 1 then takes 4 groups; index 0 gets what is left and
  # needs 6 blocks, not its stand-alone 4 (the squeeze the 12-group rule used to prevent with a pin)
  plan = _plan([20, 20], seeds=(_range_seed(5),))
  assert plan.table_stages == (5, 5) and plan.table_blocks == (6, 4)


def test_range_keys_take_crossbar_slots_before_the_trees():
  # 20 range bytes + 25 + 25 tree bytes = 70 > 66: the second tree goes to the next stage
  plan = _plan([25, 25], seeds=(_range_seed(5),))
  assert plan.table_stages == (6, 5)


def test_a_table_split_over_stages_pays_each_part_where_it_lands():
  # index 2: 15 bytes x 9 rows. The two 25-byte keys go first (higher priority) and leave 16 slots of stage
  # 5, where one row of index 2 costs 4 blocks; the other 8 rows sit alone in stage 6 at 3 blocks a row.
  plan = _plan([25, 25, 15], rows=[1, 1, 9], priorities=[2, 2, 1])
  assert plan.table_stages == (5, 5, 6) and plan.table_blocks == (5, 5, 8 * 3 + 4)


def test_seed_range_tables_count_toward_the_8_table_cap():
  plan = _plan([5, 5], seeds=(_range_seed(7),))
  assert plan.table_stages == (6, 5)


def test_the_8_table_cap():
  plan = _plan([5] * 9)                          # 9 one-group keys
  assert sorted(plan.table_stages) == [5] * 8 + [6]


def test_readiness_and_priority_order_with_a_full_stage():
  plan = _plan([22] * 5, levels=[5, 5, 5, 6, 6], priorities=[1, 1, 1, 2, 2])
  assert plan.table_stages == (5, 5, 5, 6, 6)


def test_a_key_with_no_allocation_in_12_groups_raises():
  key = tuple(("W", 0, False) for _ in range(30))   # 30 bytes that all need lane 0
  assert packing.packed_alone_blocks(key) is None
  with pytest.raises(CrossbarKeyTooWide):
    crossbar_stages_needed([(12, 30)], readiness_levels=[5], key_bytes=[key])


def test_key_bytes_and_key_field_bits_together_raise():
  with pytest.raises(ValueError):
    crossbar_stages_needed([(1, 5)], readiness_levels=[5], key_bytes=[_key(5)], key_field_bits=[(8,) * 5])


def test_key_bytes_needs_readiness_levels():
  with pytest.raises(ValueError):
    crossbar_stages_needed([(1, 5)], key_bytes=[_key(5)])


def test_a_plan_records_no_pins():
  # The per-tree @stage pin fields of spec 2026-11-11 are gone with the pins.
  plan = crossbar_stages_needed([(5, 25), (5, 25), (5, 25)], readiness_levels=[5] * 3,
                                key_field_bits=[(200,), (200,), (200,)])
  for name in ('moved', 'unpinnable', 'first_refused'):
    assert not hasattr(plan, name)
