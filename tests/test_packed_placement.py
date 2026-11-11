"""crossbar_stages_needed's explicit-key-bytes mode (packed codes, spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 4.2-4.3,
Sec 5 rules 1-2): every tree table at its stand-alone price, and the tree keys of
a stage needing at most 12 crossbar groups between them."""
import pytest
from p4model import lanes
from p4model.errors import CrossbarKeyTooWide
from p4model.packing import StageLoad, crossbar_stages_needed

def _key(n):                      # n lane-free bytes
  return tuple(("B", 0, False) for _ in range(n))

def _plan(byte_counts, rows=None, seeds=(), levels=None, priorities=None, unavailable=frozenset()):
  rows = rows or [1] * len(byte_counts)
  keys = [_key(n) for n in byte_counts]
  specs = [(lanes.standalone(list(k)) * r, n) for k, r, n in zip(keys, rows, byte_counts)]
  return crossbar_stages_needed(specs, readiness_levels=levels or [5] * len(specs), key_bytes=keys,
                                placement_priority=priorities or [1] * len(specs), seed_stages=seeds,
                                unavailable_stages=unavailable)

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
  assert plan.first_refused == ((plan.moved[0], 5),)   # refused at 5, pinned at 6: gap 1


def test_the_first_refusal_is_recorded_when_the_pin_lands_two_stages_later():
  # Spec Sec 4.3 assumes a moved table is pinned one stage after its first refusal; it can be more.
  # Index 0 (lowest priority) is refused at stage 5 by indices 1, 2; at stage 6 indices 3, 4 (ready at 6,
  # higher priority) take it first and the group rule refuses index 0 again; it lands at stage 7.
  plan = _plan([22] * 5, levels=[5, 5, 5, 6, 6], priorities=[1, 1, 1, 2, 2])
  assert plan.table_stages == (7, 5, 5, 6, 6)
  assert plan.moved == (0,) and plan.first_refused == ((0, 5),)

def test_a_key_kept_out_by_the_byte_limit_is_not_moved():
  plan = _plan([25, 25, 25])                     # 75 bytes > 66: bytes, not groups, keep the third out
  assert sorted(plan.table_stages) == [5, 5, 6] and plan.moved == ()

def test_range_tables_in_the_stage_count_two_fifths_of_a_group_each():
  seed = StageLoad(index=5, blocks=(1,) * 5, fields=frozenset((("r", i), 2) for i in range(5)), tables=5)
  assert _plan([25, 25], seeds=(seed,)).moved == ()            # 2 + 5 + 5 = 12
  # the tie order tries index 2 (1 group), then 1 (5), then 0 (5): 2 + 1 + 5 + 5 = 13, so index 0 moves
  plan = _plan([25, 25, 5], seeds=(seed,))
  assert plan.moved == (0,) and plan.table_stages == (6, 5, 5)


def test_seed_range_tables_count_toward_the_8_table_cap():
  # 7 seeded range tables leave one table slot: the second one-group key (2 + 1 + 1 = 4 groups, far
  # inside 12) is kept out by the cap, so it is not moved.
  seed = StageLoad(index=5, blocks=(1,) * 7, fields=frozenset((("r", i), 2) for i in range(7)), tables=7)
  plan = _plan([5, 5], seeds=(seed,))
  assert plan.table_stages == (6, 5) and plan.moved == () and plan.first_refused == ()


def test_unavailable_stages_are_skipped_in_key_bytes_mode():
  plan = _plan([20, 20, 20], unavailable=frozenset({5, 6}))
  assert plan.table_stages == (7, 7, 7) and plan.moved == ()


def test_a_key_with_no_lane_fit_in_12_groups_raises():
  # 30 bytes that all need lane 0: 12 groups hold at most 24 of them (one slot plus the extra per
  # group), so no stage can place the key, though its 30 bytes are inside the 64-byte budget.
  key = tuple(("W", 0, False) for _ in range(30))
  assert lanes.standalone(list(key)) is None
  with pytest.raises(CrossbarKeyTooWide):
    crossbar_stages_needed([(12, 30)], readiness_levels=[5], key_bytes=[key])

def test_a_table_kept_out_by_the_table_cap_is_not_moved():
  plan = _plan([5] * 9)                          # 9 one-group keys: the 8-table cap splits them
  assert sorted(plan.table_stages) == [5] * 8 + [6] and plan.moved == ()

def test_a_moved_table_too_big_for_one_stage_is_unpinnable_not_moved():        # Review Focus 3
  # index 2: a 3-group key of 9 rows = 27 blocks, placed as units of 24 and 3 blocks. The two 5-group keys
  # go first (higher priority); the 3-block unit then fits stage 5 in everything but groups (5 + 5 + 3).
  plan = _plan([25, 25, 15], rows=[1, 1, 9], priorities=[2, 2, 1])
  assert plan.unpinnable == (2,) and plan.moved == () and plan.first_refused == ()
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
  assert plan.moved == () and plan.unpinnable == () and plan.first_refused == ()
