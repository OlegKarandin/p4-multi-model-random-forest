"""crossbar_stages_needed's explicit-key-bytes mode (packed codes, spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 4.2-4.3,
Sec 5 rules 1-2): every tree table at its stand-alone price, and the tree keys of
a stage needing at most 12 crossbar groups between them."""
import pytest
from p4model import lanes
from p4model.packing import StageLoad, crossbar_stages_needed

def _key(n):                      # n lane-free bytes
  return tuple(("B", 0, False) for _ in range(n))

def _plan(byte_counts, rows=None, seeds=(), levels=None, priorities=None):
  rows = rows or [1] * len(byte_counts)
  keys = [_key(n) for n in byte_counts]
  specs = [(lanes.standalone(list(k)) * r, n) for k, r, n in zip(keys, rows, byte_counts)]
  return crossbar_stages_needed(specs, readiness_levels=levels or [5] * len(specs), key_bytes=keys,
                                placement_priority=priorities or [1] * len(specs), seed_stages=seeds)

def test_keys_needing_exactly_12_groups_share_a_stage():
  plan = _plan([20, 20, 20])                     # 4 + 4 + 4 groups
  assert plan.table_stages == (5, 5, 5) and plan.moved == () and plan.table_blocks == (4, 4, 4)

def test_the_key_that_would_pass_12_groups_moves_to_the_next_stage_and_is_reported():
  # 22 lane-free bytes are 5 groups; three of them are 66 bytes, inside the 66-byte limit, so only the
  # group rule (15 > 12) keeps the third out (three 25-byte keys would be 75 bytes, kept out by bytes).
  plan = _plan([22, 22, 22])                     # 5 + 5 + 5: the tie order places index 2 first
  assert sorted(plan.table_stages) == [5, 5, 6]
  assert plan.moved == (plan.table_stages.index(6),)
  assert plan.table_blocks == (5, 5, 5)          # stand-alone price whatever the position

def test_a_key_kept_out_by_the_byte_limit_is_not_moved():
  plan = _plan([25, 25, 25])                     # 75 bytes > 66: bytes, not groups, keep the third out
  assert sorted(plan.table_stages) == [5, 5, 6] and plan.moved == ()

def test_range_tables_in_the_stage_count_two_fifths_of_a_group_each():
  seed = StageLoad(index=5, blocks=(1,) * 5, fields=frozenset((("r", i), 2) for i in range(5)), tables=5)
  assert _plan([25, 25], seeds=(seed,)).moved == ()            # 2 + 5 + 5 = 12
  assert len(_plan([25, 25, 5], seeds=(seed,)).moved) == 1     # + 1 = 13

def test_a_table_kept_out_by_the_table_cap_is_not_moved():
  plan = _plan([5] * 9)                          # 9 one-group keys: the 8-table cap splits them
  assert sorted(plan.table_stages) == [5] * 8 + [6] and plan.moved == ()

def test_a_moved_table_too_big_for_one_stage_is_unpinnable_not_moved():        # Review Focus 3
  # index 2: a 3-group key of 9 rows = 27 blocks, placed as units of 24 and 3 blocks. The two 5-group keys
  # go first (higher priority); the 3-block unit then fits stage 5 in everything but groups (5 + 5 + 3).
  plan = _plan([25, 25, 15], rows=[1, 1, 9], priorities=[2, 2, 1])
  assert plan.unpinnable == (2,) and plan.moved == ()
  assert plan.table_blocks == (5, 5, 27) and plan.table_stages[2] > 5

def test_an_empty_stage_admits_any_single_table(monkeypatch):
  monkeypatch.setattr("p4model.packing.TERNARY_CROSSBAR_GROUPS_PER_STAGE", 2)
  plan = _plan([25])                             # 5 groups, over the patched limit, alone
  assert plan.table_stages == (5,) and plan.moved == ()

def test_the_group_limit_is_read_at_call_time(monkeypatch):
  monkeypatch.setattr("p4model.packing.TERNARY_CROSSBAR_GROUPS_PER_STAGE", 9)
  plan = _plan([20, 20, 20])                     # 12 groups: legal at 12, over the patched 9
  assert sorted(plan.table_stages) == [5, 5, 6] and len(plan.moved) == 1

def test_key_bytes_and_key_field_bits_together_raise():
  with pytest.raises(ValueError):
    crossbar_stages_needed([(1, 5)], readiness_levels=[5], key_bytes=[_key(5)], key_field_bits=[(8,) * 5])

def test_key_bytes_needs_readiness_levels():
  with pytest.raises(ValueError):
    crossbar_stages_needed([(1, 5)], key_bytes=[_key(5)])

def test_the_other_modes_report_nothing_moved():
  plan = crossbar_stages_needed([(5, 25), (5, 25), (5, 25)], readiness_levels=[5] * 3,
                                key_field_bits=[(200,), (200,), (200,)])
  assert plan.moved == () and plan.unpinnable == ()
