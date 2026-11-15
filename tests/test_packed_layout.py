import itertools
import random

import pytest
from p4model import packed, packing
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted

def _w(app=(), ddos=()):
  return {"app": [dict(t) for t in app], "ddos": [dict(t) for t in ddos]}

def test_budgets_are_ingress_containers_minus_the_programs_own_fields():
  assert packed.budgets(n_features=7, n_trees=6) == {32: 48 - 3, 16: 80 - 7 - 4, 8: 48 - 6 - 5}

def test_budgets_count_the_scratch_fields():
  assert packed.budgets(n_features=7, n_trees=6, n_scratch=3)[16] == 80 - 7 - 3 - 4

def test_no_fall_through(monkeypatch):
  """A wanted size with no container left is refused, never replaced by another size
  (decision 4.2): a 32-bit field used to fall to 16 then 8 bits and now raises."""
  from p4model import phv
  tight = dict(phv.PHV_INGRESS_CONTAINERS)
  tight[32] = len(phv.PHV_TEMPLATE_FIELDS[32])                          # no 32-bit container left
  monkeypatch.setattr(phv, "PHV_INGRESS_CONTAINERS", tight)
  with pytest.raises(PhvContainersExhausted) as refused:
    packed.layout_for(_w(app=[{"f": 12}, {"f": 12}, {"f": 12}]), ["f"], ["app", "ddos"])   # 6 bytes: wants 32 + 16
  assert refused.value.args[1] == 1                                      # one container short
  monkeypatch.undo()
  assert packed.layout_for(_w(app=[{"f": 12}, {"f": 12}, {"f": 12}]), ["f"], ["app", "ddos"]).containers["f"] == (32, 16)

@pytest.mark.parametrize("bits, containers", [
    ([6], (8,)), ([12], (16,)), ([12, 6], (16, 8)),
    ([12, 9, 6], (32,)),                           # 4 bytes: the 6-bit code sits in the 9-bit one's last byte
    ([12, 12, 5], (32, 8)),                        # 5 bytes: 5 bits fit neither 12-bit code's 4 free bits
    ([32, 32, 32, 32], (32, 32, 32, 32)),          # 16 bytes: not over one RAM word
    ([32, 32, 32, 32, 6], (32,) * 5),              # 17 bytes: 32-bit only, last one padded
])
def test_containers_of_a_field(bits, containers):
  widths = _w(app=[{"f": b} for b in bits])
  assert packed.layout_for(widths, ["f"], ["app", "ddos"]).containers["f"] == containers

def test_small_codes_share_a_byte():                                             # decision 4.4
  layout = packed.layout_for(_w(app=[{"f": 3}, {"f": 3}], ddos=[{"f": 2}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (8,)
  assert layout.place == {("app", 0, "f"): (0, 3), ("app", 1, "f"): (3, 3), ("ddos", 0, "f"): (6, 2)}

def test_long_code_starts_on_fresh_byte():
  # the 3-bit code makes byte 0 first (program order), but codes go widest first: the 10-bit
  # code takes two fresh bytes and the 3-bit code its last byte's free bits
  layout = packed.layout_for(_w(app=[{"f": 3}, {"f": 10}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (16,)
  assert layout.place[("app", 1, "f")] == (0, 10) and layout.place[("app", 0, "f")] == (10, 3)
  # a fresh byte holding small codes never takes a long code's start
  layout = packed.layout_for(_w(app=[{"f": 4}, {"f": 4}, {"f": 9}, {"f": 12}]), ["f"], ["app", "ddos"])
  for (_, _, _), (offset, bits) in layout.place.items():
    if bits > 8:
      assert offset % 8 == 0

def test_no_code_straddles_a_byte():
  rng = random.Random(4)
  for _ in range(300):
    trees = [{"f": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 17, 30, 70])}
             for _ in range(rng.randint(1, 9))]
    layout = packed.layout_for(_w(app=trees), ["f"], ["app", "ddos"])
    spans = sorted(layout.place.values())
    used = set()
    for offset, bits in spans:
      assert not packed.straddles(offset, bits), (trees, offset, bits)
      cover = set(range(offset, offset + bits))
      assert not cover & used                                      # codes never overlap
      used |= cover
    assert max(o + b for o, b in spans) <= packed.field_bits(layout, "f")

def test_codes_are_ordered_to_spread_each_trees_bytes_over_the_lanes():
  # tree 0 already has a byte counted in lane 0 (g: a 1-byte field in an 8-bit container);
  # 5-bit codes cannot share a byte, so each is a unit of its own, as before 4.4
  widths = _w(app=[{"g": 5, "f": 5}, {"f": 5}, {"f": 5}, {"f": 5}])
  layout = packed.layout_for(widths, ["g", "f"], ["app", "ddos"])
  assert layout.containers == {"g": (8,), "f": (32,)}
  assert [layout.place[("app", t, "f")][0] for t in range(4)] == [8, 0, 16, 24]

def test_key_bytes_name_container_kind_lane_nibble_and_identity():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 9}], ddos=[{"f": 6}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (32,)
  assert packed.key_bytes(layout, "app", 0) == (("W", 0, False, ("f", 0)), ("W", 1, True, ("f", 0)))   # 8 + 4
  assert packed.key_bytes(layout, "app", 1) == (("W", 2, False, ("f", 0)), ("W", 3, True, ("f", 0)))   # 8 + 1
  # the 6-bit code shares byte 3 (its bits 1-6: both nibbles, so not a one-nibble byte, P13)
  assert layout.place[("ddos", 0, "f")] == (25, 6)
  assert packed.key_bytes(layout, "ddos", 0) == (("W", 3, False, ("f", 0)),)

def test_one_nibble_is_either_nibble_alone():
  assert packed.one_nibble(0x0F) and packed.one_nibble(0xF0) and packed.one_nibble(0x30)
  assert not packed.one_nibble(0x18) and not packed.one_nibble(0xFF)

def test_a_lone_narrow_code_is_one_lane_free_byte_at_its_standalone_price():     # Review Focus 5
  layout = packed.layout_for(_w(app=[{"f": 3}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (8,)
  assert packing.packed_alone_blocks(packed.key_bytes(layout, "app", 0)) == 1

def test_slices_are_field_positions_of_at_most_64_bits():                        # Review Focus 2
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 70}]), ["f"], ["app", "ddos"])
  offset, bits = layout.place[("app", 1, "f")]
  assert bits == 70 and offset > 0 and offset % 8 == 0
  assert packed.key_slices(layout, "app", 1, "f") == [(offset + 63, offset), (offset + 69, offset + 64)]
  base = layout.place[("app", 0, "f")][0]
  assert packed.key_slices(layout, "app", 0, "f") == [(base + 11, base)]
  shared = packed.layout_for(_w(app=[{"f": 3}, {"f": 4}]), ["f"], ["app", "ddos"])
  assert packed.key_slices(shared, "app", 0, "f") == [(6, 4)]                    # the 4-bit code is first

def test_range_value_or_of_shifted_codes():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 3}, {"f": 2}], ddos=[{"f": 3}]), ["f"], ["app", "ddos"])
  trees = [("app", 0), ("app", 1), ("app", 2), ("ddos", 0)]
  for codes in itertools.product(range(4), repeat=4):
    values = {tree: [0xABC, 0x5, 0x3, 0x6][i] >> codes[i] for i, tree in enumerate(trees)}
    value = packed.range_value(layout, "f", values)
    for tree, code in values.items():                              # each tree reads its own code back
      offset, bits = layout.place[tree + ("f",)]
      assert (value >> offset) & ((1 << bits) - 1) == code
  with pytest.raises(ValueError):
    packed.range_value(layout, "f", {("app", 1): 0x8})            # a 4-bit value for a 3-bit code

def test_bus_bytes_are_the_fields_container_bytes():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 12}], ddos=[{"f": 5}]), ["f"], ["app", "ddos"])
  assert packed.field_bits(layout, "f") == 40 and packed.bus_bytes(layout, "f") == 5

def test_a_budget_that_runs_out_mid_field_is_refused_not_filled_with_smaller_ones(monkeypatch):
  # decision 4.2 (was: falls to the next size). 20 bytes want five 32-bit containers, 3 left
  monkeypatch.setattr(packed, "budgets", lambda n_features, n_trees, n_scratch=0: {32: 3, 16: 80, 8: 40})
  with pytest.raises(PhvContainersExhausted) as refused:
    packed.layout_for(_w(app=[{"f": 32}] * 5), ["f"], ["app", "ddos"])
  assert refused.value.args[1] == 2

def test_two_trees_keying_one_byte_share_its_crossbar_slot():
  # the crossbar moves whole bytes: in one stage the second tree finds the first one's slot
  layout = packed.layout_for(_w(app=[{"f": 3}, {"f": 3}]), ["f"], ["app", "ddos"])
  keys = [packed.key_bytes(layout, "app", t) for t in range(2)]
  assert keys[0][0][3] == keys[1][0][3]
  units = tuple((t, keys[t], 1) for t in range(2))
  crossbar, _, per_row = packing._stage_state((), units, "packed")
  used = sum(slot is not None for row in crossbar.use for slot in row) + sum(
      mid is not None for mid in crossbar.bgu)
  assert used == 1 and per_row == {0: 1, 1: 1}

def test_refusals():
  with pytest.raises(NoSplitTree):
    packed.layout_for(_w(app=[{}]), [], ["app", "ddos"])
  with pytest.raises(ActionDataTooWide) as wide:                                  # a field over 128 bytes
    packed.layout_for(_w(app=[{"f": 520}, {"f": 520}]), ["f"], ["app", "ddos"])
  assert wide.value.args[1] == 130                                               # 65 + 65 code bytes
  with pytest.raises(PhvContainersExhausted):
    packed.layout_for(_w(app=[{f"f{i}": 120 for i in range(40)}]), [f"f{i}" for i in range(40)], ["app", "ddos"])
