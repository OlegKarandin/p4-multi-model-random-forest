"""p4model/xbar.py: the port of p4c's ternary crossbar allocation.

The main test replays real compiles: tests/fixtures/xbar_p4c_stages.json holds every ternary stage of six programs
compiled by p4c (three with k <= 8, three with k >= 9, each with a stage of three or more tree tables and a midbyte),
with each key byte's PHV container and p4c's own slot, read from prog.bfa. Allocated in p4c's placement order with
p4c's own container numbering (tie='index'), every table must land slot for slot where p4c put it."""
import json
import os

import pytest

from p4model import xbar

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'xbar_p4c_stages.json')


def _programs():
  with open(FIXTURE, encoding='utf-8') as fh:
    return json.load(fh)['programs']


def _signature(rows, locs):
  return sorted(((c, lo, spec), tuple(loc)) for (c, lo, _, spec, _, _), loc in zip(rows, locs))


@pytest.mark.parametrize('program', _programs(), ids=lambda p: p['row_id'])
def test_reproduces_p4c_slot_for_slot(program):
  tables = 0
  for stage in program['stages']:
    crossbar = xbar.StageXbar(tie='index')
    for table in stage['tables']:
      rows = table['bytes']
      key = [xbar.Byte(c, lo, bits, spec, range_index) for c, lo, bits, spec, range_index, _ in rows]
      assert crossbar.find_alloc(key) is not None, (stage['stage'], table['name'])
      assert _signature(rows, [b.loc for b in key]) == _signature(rows, [r[5] for r in rows]), (
          stage['stage'], table['name'])
      for b, r in zip(key, rows):          # the stage as p4c left it, for the next table
        b.loc = tuple(r[5])
      crossbar.commit(key)
      tables += 1
  assert tables >= 10


def test_fixture_exercises_squeezed_trees_and_midbytes():
  crowded = midbytes = 0
  for program in _programs():
    for stage in program['stages']:
      trees = [t for t in stage['tables'] if t['name'].startswith('get_classification_tree_')]
      crowded += len(trees) >= 3
      midbytes += sum(r[5][1] == xbar.BYTES_PER_GROUP for t in trees for r in t['bytes'])
  assert crowded >= 6 and midbytes >= 6


def test_a_32_bit_container_byte_takes_only_its_own_lane():
  crossbar = xbar.StageXbar()
  key = [xbar.Byte('W0', 8 * k, 0xFF) for k in range(4)]
  assert crossbar.allocate(key)
  assert [xbar.ternary_ord(*b.loc) % 4 for b in key] == [0, 1, 2, 3]


def test_blocks_per_row_of_simple_keys():
  five = [xbar.Byte('B%d' % i, 0, 0xFF) for i in range(5)]
  assert xbar.StageXbar().allocate(five)
  assert xbar.blocks_per_row(five) == 1
  ten = [xbar.Byte('B%d' % i, 0, 0xFF) for i in range(10)]
  assert xbar.StageXbar().allocate(ten)
  assert xbar.blocks_per_row(ten) == 2


def test_a_full_stage_refuses_the_next_key():
  crossbar = xbar.StageXbar()
  assert crossbar.allocate([xbar.Byte('B%d' % i, 0, 0xFF) for i in range(66)])
  assert crossbar.find_alloc([xbar.Byte('B999', 0, 0xFF)]) is None


def test_model_keys_never_share_a_slot_between_tables():
  key = (('W', 0, False), ('W', 1, False), ('W', 2, False), ('W', 3, False), ('B', 0, True))
  a, b = xbar.tree_key(key, 0), xbar.tree_key(key, 1)
  assert {x.container for x in a}.isdisjoint({x.container for x in b})
  assert [x.container for x in a] == ['W1000', 'W1000', 'W1000', 'W1000', 'B1000']
  assert a[-1].only_one_nibble()
  crossbar = xbar.StageXbar()
  assert crossbar.allocate(a) and crossbar.allocate(b)
  assert {x.loc for x in a}.isdisjoint({x.loc for x in b})


def test_range_key_is_lo_and_hi_per_byte():
  key = xbar.range_key(3, 2)
  assert [(b.container, b.lo, b.spec) for b in key] == [
      ('H900003', 0, xbar.RANGE_LO), ('H900003', 0, xbar.RANGE_HI),
      ('H900003', 8, xbar.RANGE_LO), ('H900003', 8, xbar.RANGE_HI)]


def test_tie_must_be_known():
  with pytest.raises(ValueError):
    xbar.StageXbar(tie='alphabetical')
