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
    ([12, 9, 6], (32, 8)),                         # 5 bytes: every code has its own (2 + 2 + 1)
    ([3, 3, 2], (16, 8)),                          # 3 bytes: three small codes, one byte each
    ([32, 32, 32, 32], (32, 32, 32, 32)),          # 16 bytes: not over one RAM word
    ([32, 32, 32, 32, 6], (32,) * 5),              # 17 bytes: 32-bit only, last one padded
])
def test_containers_of_a_field(bits, containers):
  widths = _w(app=[{"f": b} for b in bits])
  assert packed.layout_for(widths, ["f"], ["app", "ddos"]).containers["f"] == containers

def test_small_codes_do_not_share_a_byte():
  # three codes of 3, 3 and 2 bits would fit one byte; each gets a byte of its own (two trees
  # keying one byte make p4c's block count depend on its PHV container numbers)
  layout = packed.layout_for(_w(app=[{"f": 3}, {"f": 3}], ddos=[{"f": 2}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (16, 8)
  assert layout.place == {("app", 0, "f"): (0, 3), ("app", 1, "f"): (8, 3), ("ddos", 0, "f"): (16, 2)}

def test_a_small_code_does_not_sit_in_a_long_codes_last_byte():
  layout = packed.layout_for(_w(app=[{"f": 3}, {"f": 10}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (16, 8)                         # 1 + 2 bytes, not the 10-bit code's 6 free bits
  assert layout.place == {("app", 0, "f"): (0, 3), ("app", 1, "f"): (8, 10)}

def test_every_code_starts_a_byte_no_other_code_touches():
  rng = random.Random(4)
  for _ in range(300):
    trees = [{"f": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 17, 30, 70])}
             for _ in range(rng.randint(1, 9))]
    layout = packed.layout_for(_w(app=trees), ["f"], ["app", "ddos"])
    spans = sorted(layout.place.values())
    used = set()
    for offset, bits in spans:
      assert offset % 8 == 0, (trees, offset, bits)
      cover = set(range(offset // 8, (offset + bits - 1) // 8 + 1))
      assert not cover & used, (trees, offset, bits)               # no byte holds two codes
      used |= cover
    assert max(o + b for o, b in spans) <= packed.field_bits(layout, "f")
    assert len(used) == sum(-(-bits // 8) for _, bits in spans)     # and no byte is left out between codes

def test_codes_are_ordered_to_spread_each_trees_bytes_over_the_lanes():
  # tree 0 already has a byte counted in lane 0 (g: a 1-byte field in an 8-bit container);
  # each 5-bit code has a byte of its own
  widths = _w(app=[{"g": 5, "f": 5}, {"f": 5}, {"f": 5}, {"f": 5}])
  layout = packed.layout_for(widths, ["g", "f"], ["app", "ddos"])
  assert layout.containers == {"g": (8,), "f": (32,)}
  assert [layout.place[("app", t, "f")][0] for t in range(4)] == [8, 0, 16, 24]

def test_key_bytes_name_container_kind_lane_nibble_and_identity():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 9}], ddos=[{"f": 6}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (32, 8)
  assert packed.key_bytes(layout, "app", 0) == (("W", 0, False, ("f", 0)), ("W", 1, True, ("f", 0)))   # 8 + 4
  assert packed.key_bytes(layout, "app", 1) == (("W", 2, False, ("f", 0)), ("W", 3, True, ("f", 0)))   # 8 + 1
  # the 6-bit code has the fifth byte, in the field's second container; bits in both nibbles (P13)
  assert layout.place[("ddos", 0, "f")] == (32, 6)
  assert packed.key_bytes(layout, "ddos", 0) == (("B", 0, False, ("f", 1)),)

def test_a_half_byte_beside_a_whole_byte_of_its_class_is_keyed_whole():
  # one tree: f 12 bits in a 16-bit container (byte 0 whole, byte 1 four bits), g 4 bits and h
  # 8 bits each in an 8-bit container. g's half byte and h's whole byte need the same kind of
  # crossbar slot, so p4c's container numbers would pick which one a midbyte gets: g is keyed
  # as a whole byte. f's byte 1 is the only 16-bit byte-1 of the key and stays a half byte.
  layout = packed.layout_for(_w(app=[{"f": 12, "g": 4, "h": 8}]), ["f", "g", "h"], ["app", "ddos"])
  assert layout.containers == {"f": (16,), "g": (8,), "h": (8,)}
  assert packed.key_bytes(layout, "app", 0) == (
      ("H", 0, False, ("f", 0)), ("H", 1, True, ("f", 0)), ("B", 0, False, ("g", 0)), ("B", 0, False, ("h", 0)))
  assert packed.key_slices(layout, "app", 0, "g") == [(7, 0)]
  assert packed.key_slices(layout, "app", 0, "f") == [(11, 0)]
  assert layout.place[("app", 0, "g")] == (0, 4)                   # the code itself is still 4 bits
  assert packed.key_bits(layout, "app", 0) == 12 + 8 + 8

def test_half_bytes_alone_in_their_class_stay_half():
  layout = packed.layout_for(_w(app=[{"g": 3, "h": 4}]), ["g", "h"], ["app", "ddos"])
  assert packed.key_bytes(layout, "app", 0) == (("B", 0, True, ("g", 0)), ("B", 0, True, ("h", 0)))
  assert packed.key_slices(layout, "app", 0, "g") == [(2, 0)]
  assert packed.key_slices(layout, "app", 0, "h") == [(3, 0)]
  assert packed.key_bits(layout, "app", 0) == 7

def test_a_long_codes_half_top_byte_is_padded_too():
  # f 12 bits and g 16 bits, each in a 16-bit container: byte 1 is half for f, whole for g
  layout = packed.layout_for(_w(app=[{"f": 12, "g": 16}]), ["f", "g"], ["app", "ddos"])
  assert packed.key_bytes(layout, "app", 0) == (
      ("H", 0, False, ("f", 0)), ("H", 1, False, ("f", 0)), ("H", 0, False, ("g", 0)), ("H", 1, False, ("g", 0)))
  assert packed.key_slices(layout, "app", 0, "f") == [(15, 0)]

def test_the_class_is_the_byte_position_in_a_32_bit_container():
  # f 28 bits: byte 3 of its 32-bit container holds 4 bits. g 32 bits has a whole byte 3: padded.
  layout = packed.layout_for(_w(app=[{"f": 28, "g": 32}]), ["f", "g"], ["app", "ddos"])
  assert layout.containers == {"f": (32,), "g": (32,)}
  assert packed.key_slices(layout, "app", 0, "f") == [(31, 0)]
  # g 24 bits sits in a 16-bit and an 8-bit container: no 32-bit byte 3 beside f's, f stays half
  layout = packed.layout_for(_w(app=[{"f": 28, "g": 24}]), ["f", "g"], ["app", "ddos"])
  assert layout.containers == {"f": (32,), "g": (16, 8)}
  assert packed.key_slices(layout, "app", 0, "f") == [(27, 0)]
  assert packed.key_bytes(layout, "app", 0)[3] == ("W", 3, True, ("f", 0))

def test_only_the_trees_own_key_decides_its_padding():
  # field g holds tree 0's 4-bit code and tree 1's 8-bit code; tree 0 keys only its own byte
  layout = packed.layout_for(_w(app=[{"g": 4}, {"g": 8}]), ["g"], ["app", "ddos"])
  assert layout.containers["g"] == (16,)
  assert packed.key_bytes(layout, "app", 0) == (("H", 0, True, ("g", 0)),)
  assert packed.key_bytes(layout, "app", 1) == (("H", 1, False, ("g", 0)),)

def test_a_padded_code_over_64_bits_is_cut_into_slices_of_its_padded_width():
  # f 68 bits: 9 bytes, the ninth with 4 bits, in an 8-bit container; g 72 bits has a whole
  # byte in its own 8-bit container, so f is keyed as 72 bits
  layout = packed.layout_for(_w(app=[{"f": 68, "g": 72}]), ["f", "g"], ["app", "ddos"])
  assert layout.containers == {"f": (32, 32, 8), "g": (32, 32, 8)}
  assert packed.key_slices(layout, "app", 0, "f") == [(63, 0), (71, 64)]

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
  small = packed.layout_for(_w(app=[{"f": 3}, {"f": 4}]), ["f"], ["app", "ddos"])
  assert packed.key_slices(small, "app", 0, "f") == [(2, 0)]
  assert packed.key_slices(small, "app", 1, "f") == [(11, 8)]                    # its own byte, the second

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
  monkeypatch.setattr(packed, "budgets", lambda n_features, n_trees, n_scratch=0, n_constants=0,
                      coarse_clock=False, fine_clock=False: {32: 3, 16: 80, 8: 40})
  with pytest.raises(PhvContainersExhausted) as refused:
    packed.layout_for(_w(app=[{"f": 32}] * 5), ["f"], ["app", "ddos"])
  assert refused.value.args[1] == 2

def test_two_trees_keying_one_byte_share_its_crossbar_slot():
  # the crossbar moves whole bytes: in one stage the second tree finds the first one's slot.
  # The generator no longer lays codes out like this; the port still prices such keys.
  keys = [(("B", 0, True, ("f", 0)),), (("B", 0, True, ("f", 0)),)]
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
