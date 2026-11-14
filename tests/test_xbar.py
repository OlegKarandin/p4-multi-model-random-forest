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


# joint-task_M055_s20_k12 (Task 5, key pruning), stage 8: five ddos trees whose keys overlap but differ, in
# p4c's placement order, with the code fields' widths (p4c put the 1-4-bit ones in B, the 9-16-bit ones in H
# containers, as lanes.layout says). p4c charged [4, 4, 3, 4, 4] blocks per 512-row word (prog.bfa, all slot for
# slot reproduced by the port on p4c's own containers); container numbers handed out by first use gave
# get_classification_tree_ddos_0 3.
_M055_WIDTHS = {'bwd_iat_max': 4, 'bwd_iat_mean': 4, 'bwd_packet_length_max': 14,
                'bwd_packet_length_mean': 14, 'bwd_packet_length_min': 1, 'flow_iat_mean': 11,
                'flow_iat_min': 16, 'fwd_iat_min': 13, 'fwd_packet_length_max': 9,
                'fwd_packet_length_mean': 14, 'min_packet_length': 1, 'packet_length_mean': 10}
_M055_STAGE8 = [
    ('ddos_4', ['bwd_iat_max', 'bwd_iat_mean', 'bwd_packet_length_max', 'bwd_packet_length_mean',
                'flow_iat_min', 'fwd_iat_min', 'fwd_packet_length_max', 'fwd_packet_length_mean',
                'min_packet_length', 'packet_length_mean']),
    ('ddos_3', ['bwd_iat_max', 'bwd_packet_length_max', 'bwd_packet_length_mean', 'flow_iat_mean',
                'flow_iat_min', 'fwd_iat_min', 'fwd_packet_length_max', 'fwd_packet_length_mean',
                'packet_length_mean']),
    ('ddos_2', ['bwd_packet_length_max', 'bwd_packet_length_mean', 'bwd_packet_length_min',
                'flow_iat_mean', 'flow_iat_min', 'fwd_packet_length_max', 'fwd_packet_length_mean',
                'min_packet_length', 'packet_length_mean']),
    ('ddos_1', ['bwd_iat_max', 'bwd_iat_mean', 'bwd_packet_length_max', 'bwd_packet_length_mean',
                'flow_iat_mean', 'flow_iat_min', 'fwd_iat_min', 'fwd_packet_length_max',
                'fwd_packet_length_mean', 'min_packet_length', 'packet_length_mean']),
    ('ddos_0', ['bwd_iat_mean', 'bwd_packet_length_max', 'bwd_packet_length_mean', 'flow_iat_mean',
                'flow_iat_min', 'fwd_packet_length_max', 'fwd_packet_length_mean', 'min_packet_length',
                'packet_length_mean']),
]


def test_widest_first_numbering_reproduces_p4c_on_overlapping_keys():
  from p4model import lanes
  crossbar, numbering, blocks = xbar.StageXbar(tie='whole_first'), {}, []
  for _, fields in _M055_STAGE8:
    key = xbar.field_key([('code_tk_ddos_' + f, tuple(lanes.layout(_M055_WIDTHS[f]))) for f in fields],
                         numbering)
    assert crossbar.allocate(key)
    blocks.append(xbar.blocks_per_row(key))
  assert blocks == [4, 4, 3, 4, 4]


def test_code_containers_are_numbered_widest_first_then_name_then_slice():
  key = xbar.field_key([('code_b', (('B', 4),)), ('code_a', (('H', 16), ('B', 4))), ('code_c', (('B', 4),))],
                       {})
  order = sorted({b.ckey for b in key})
  by_ckey = {b.ckey: b.container for b in key}
  # code_a (20 bits) first, its low slice before its high one, then code_b before code_c (4 bits each)
  # bytes: code_b's B, code_a's two H bytes and its B, code_c's B
  assert [by_ckey[k] for k in order if k[1] == 8] == [key[3].container, key[0].container, key[4].container]
  assert xbar.code_container_order('code_a', 20, 0) < xbar.code_container_order('code_a', 20, 1) < \
      xbar.code_container_order('code_b', 4, 0) < xbar.code_container_order('code_c', 4, 0)
