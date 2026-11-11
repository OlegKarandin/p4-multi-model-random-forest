import pytest
from p4model import lanes, packed
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted

def _w(app=(), ddos=()):
  return {"app": [dict(t) for t in app], "ddos": [dict(t) for t in ddos]}

def test_budgets_are_ingress_containers_minus_the_programs_own_fields():
  assert packed.budgets(n_features=7, n_trees=6) == {32: 43, 16: 80 - 7 - 5, 8: 48 - 6 - 4}

@pytest.mark.parametrize("bits, containers", [
    ([6], (8,)), ([12], (16,)), ([12, 6], (16, 8)), ([12, 9, 6], (32, 8)),
    ([32, 32, 32, 32], (32, 32, 32, 32)),          # 16 bytes: not over one RAM word
    ([32, 32, 32, 32, 6], (32,) * 5),              # 17 bytes: 32-bit only, last one padded
])
def test_containers_of_a_field(bits, containers):
  widths = _w(app=[{"f": b} for b in bits])
  assert packed.layout_for(widths, ["f"], ["app", "ddos"]).containers["f"] == containers

def test_codes_are_ordered_to_spread_each_trees_bytes_over_the_lanes():
  # tree 0 already has a byte counted in lane 0 (g: a 1-byte field in an 8-bit container)
  widths = _w(app=[{"g": 3, "f": 3}, {"f": 3}, {"f": 3}, {"f": 3}])
  layout = packed.layout_for(widths, ["g", "f"], ["app", "ddos"])
  assert layout.containers == {"g": (8,), "f": (32,)}
  assert [layout.place[("app", t, "f")][0] for t in range(4)] == [1, 0, 2, 3]

def test_key_bytes_name_container_kind_lane_and_nibble():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 9}], ddos=[{"f": 6}]), ["f"], ["app", "ddos"])
  assert packed.key_bytes(layout, "app", 0) == (("W", 0, False), ("W", 1, True))   # 12 bits: 8 + 4
  assert packed.key_bytes(layout, "app", 1) == (("W", 2, False), ("W", 3, True))   # 9 bits: 8 + 1
  assert packed.key_bytes(layout, "ddos", 0) == (("B", 0, False),)                # 6 bits, 8-bit container

def test_a_lone_narrow_code_is_one_lane_free_byte_at_its_standalone_price():     # Review Focus 5
  layout = packed.layout_for(_w(app=[{"f": 3}]), ["f"], ["app", "ddos"])
  assert layout.containers["f"] == (8,)
  assert lanes.standalone(list(packed.key_bytes(layout, "app", 0))) == 1

def test_slices_are_field_positions_of_at_most_64_bits():                        # Review Focus 2
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 70}]), ["f"], ["app", "ddos"])
  offset, bits = layout.place[("app", 1, "f")]
  assert bits == 70 and offset > 0
  assert packed.key_slices(layout, "app", 1, "f") == [(8 * offset + 63, 8 * offset), (8 * offset + 69, 8 * offset + 64)]
  assert packed.key_slices(layout, "app", 0, "f") == [(8 * layout.place[("app", 0, "f")][0] + 11, 8 * layout.place[("app", 0, "f")][0])]

def test_range_value_puts_each_code_at_its_byte_offset():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 9}]), ["f"], ["app", "ddos"])
  o0, o1 = layout.place[("app", 0, "f")][0], layout.place[("app", 1, "f")][0]
  assert packed.range_value(layout, "f", {("app", 0): 0xABC, ("app", 1): 0x1FF}) == (0xABC << 8 * o0) | (0x1FF << 8 * o1)

def test_bus_bytes_are_the_fields_container_bytes():
  layout = packed.layout_for(_w(app=[{"f": 12}, {"f": 9}], ddos=[{"f": 6}]), ["f"], ["app", "ddos"])
  assert packed.field_bits(layout, "f") == 40 and packed.bus_bytes(layout, "f") == 5

def test_a_budget_that_runs_out_mid_field_falls_to_the_next_size(monkeypatch):   # Review Focus 1
  monkeypatch.setattr(packed, "budgets", lambda n_features, n_trees: {32: 3, 16: 80, 8: 40})
  layout = packed.layout_for(_w(app=[{"f": 32}] * 5), ["f"], ["app", "ddos"])     # 20 bytes
  assert layout.containers["f"] == (32, 32, 32, 16, 16, 16, 16)
  kinds = [kind for kind, _, _ in packed.key_bytes(layout, "app", 4)]
  assert set(kinds) <= {"W", "H"} and len(kinds) == 4

def test_refusals():
  with pytest.raises(NoSplitTree):
    packed.layout_for(_w(app=[{}]), [], ["app", "ddos"])
  with pytest.raises(ActionDataTooWide) as wide:                                  # a field over 128 bytes
    packed.layout_for(_w(app=[{"f": 520}, {"f": 520}]), ["f"], ["app", "ddos"])
  assert wide.value.args[1] == 130                                               # 65 + 65 code bytes
  with pytest.raises(PhvContainersExhausted):
    packed.layout_for(_w(app=[{f"f{i}": 120 for i in range(40)}]), [f"f{i}" for i in range(40)], ["app", "ddos"])
