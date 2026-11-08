"""Per-tree codes (program.PER_TREE_ENCODING): the names, the two limits, the Spilled width
marker and the pin rule (p4model/per_tree.py) that generator and model both call.

Spec docs/superpowers/specs/2026-11-06-per-tree-arm-design.md Sec 7, 9.2, 9.3 and its
amendments of 2026-11-08 (A3: the pin rule is the spill by range table)."""
import pickle

import pytest

from p4model import lanes
from p4model.errors import ActionDataTooWide, CrossbarKeyTooWide, NoSplitTree
from p4model.lanes import Spilled
from p4model.per_tree import (action_bytes, base_pins, marked_key_bits, pins_for,
                              spill_table_fields, widths_of)
from p4model.program import (ENCODINGS, PER_TREE_ENCODING, feature_of_per_tree_field,
                             per_tree_field)
from p4model.target import PER_TREE_MAX_ACTION_BYTES, PER_TREE_MAX_PIN8_FIELDS


def _widths(app=(), ddos=()):
  """{task: [{feature: bits}]} from per-tree lists of (feature, bits) pairs."""
  return {"app": [dict(t) for t in app], "ddos": [dict(t) for t in ddos]}


def _counted(counts, bits=3):
  """One app tree per row so that feature f has counts[f] narrow codes: tree i holds every
  feature with more than i codes."""
  trees = [{f: bits for f, n in counts.items() if n > i} for i in range(max(counts.values()))]
  return {"app": trees, "ddos": []}


def test_field_helpers_undo_each_other_with_digits_and_underscores():
  assert per_tree_field("app", 0, "flow_iat_max") == "tr_app_0_flow_iat_max"
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("tr_app_0_1_f", "app") == "1_f"
  assert feature_of_per_tree_field("code_tr_app_0_1_f", "app_0") == "f"
  assert feature_of_per_tree_field("code_tk_app_f", "app") is None
  assert feature_of_per_tree_field("code_tr_ddos_2_f", "app") is None


def test_the_encoding_is_named_and_listed():
  assert PER_TREE_ENCODING == "per_tree" and PER_TREE_ENCODING in ENCODINGS
  assert PER_TREE_MAX_PIN8_FIELDS == 44 and PER_TREE_MAX_ACTION_BYTES == 112


def test_widths_are_interval_counts_minus_one_and_skip_featureless_entries():
  ivs = {"app": [{"f": [0, 5, 9], "g": [0]}], "ddos": [{"f": list(range(10))}]}
  assert widths_of(ivs) == {"app": [{"f": 2}], "ddos": [{"f": 9}]}


def test_44_narrow_fields_spill_nothing_and_45_spill_one_whole_feature_in_order():
  # 5 app + 4 ddos trees x features f0..f4 of 3 bits = 45 narrow fields; drop the last ddos
  # tree's f4 for 44
  full = _widths([{f"f{j}": 3 for j in range(5)}] * 5, [{f"f{j}": 3 for j in range(5)}] * 4)
  w45 = full
  w44 = _widths([{f"f{j}": 3 for j in range(5)}] * 5,
                [{f"f{j}": 3 for j in range(5)}] * 3 + [{f"f{j}": 3 for j in range(4)}])
  pins_44, spilled_44 = pins_for(w44)
  assert spilled_44 == []
  assert sum(p == [8] for p in pins_44.values()) == 44
  pins_45, spilled_45 = pins_for(w45)
  # every feature holds 9 codes, so the first alphabetically goes, all of its codes together,
  # tasks app then ddos, trees ascending
  assert spilled_45 == [per_tree_field("app", i, "f0") for i in range(5)] + \
                       [per_tree_field("ddos", i, "f0") for i in range(4)]
  assert all(pins_45[n] == [16] for n in spilled_45)
  assert sum(p == [8] for p in pins_45.values()) == 36
  assert len(pins_45) == 45


def test_a_spill_takes_the_smallest_code_count_at_or_above_the_need():
  widths = _counted({"a": 1, "b": 2, "c": 4})        # 7 narrow codes
  pins = base_pins(widths)
  # need 3: a+b is exactly 3, c alone would be 4
  assert spill_table_fields(widths, pins, 4) == [per_tree_field("app", 0, "a"),
                                                  per_tree_field("app", 0, "b"),
                                                  per_tree_field("app", 1, "b")]
  # need 4: c alone (4) ties a+b+... only at its own count, and the fewest features win
  assert spill_table_fields(widths, pins, 3) == [per_tree_field("app", i, "c") for i in range(4)]
  # need 1: a (1 code) is the smallest total at or above it
  assert spill_table_fields(widths, pins, 6) == [per_tree_field("app", 0, "a")]


def test_a_spill_prefers_fewer_features_then_the_alphabet_on_equal_counts():
  widths = _counted({"a": 1, "b": 2, "c": 3})        # 6 narrow codes
  pins = base_pins(widths)
  # need 3: {c} and {a,b} both total 3; one feature beats two
  assert spill_table_fields(widths, pins, 3) == [per_tree_field("app", i, "c") for i in range(3)]
  tie = _counted({"d": 2, "b": 2})                   # need 2: b and d tie, b is first
  assert spill_table_fields(tie, base_pins(tie), 2) == [per_tree_field("app", i, "b") for i in range(2)]
  nothing = spill_table_fields(widths, pins, 6)
  assert nothing == []


def test_the_spill_leaves_wider_fields_and_other_features_alone():
  widths = _widths([{"f": 3, "g": 12, "h": 3}] * 3)
  pins = base_pins(widths)
  # 6 narrow codes, 5 allowed: f (3 codes) is the smallest total at or above 1
  assert spill_table_fields(widths, pins, 5) == [per_tree_field("app", i, "f") for i in range(3)]
  assert pins[per_tree_field("app", 0, "g")] == [16]       # a 12-bit field is H already, never 'narrow'


def test_too_many_features_to_enumerate_raises():
  widths = _widths([{f"f{j:02d}": 3 for j in range(21)}] * 3)
  with pytest.raises(ValueError):
    spill_table_fields(widths, base_pins(widths), 10)


def test_112_action_bytes_pass_and_113_raise():
  # 28 trees with one 32-bit field on f: 28 x 4 = 112 bytes
  ok = _widths([{"f": 32}] * 28)
  pins, spilled = pins_for(ok)
  assert spilled == [] and action_bytes(ok, pins) == {"f": 112}
  bad = _widths([{"f": 32}] * 28 + [{"f": 8}])
  with pytest.raises(ActionDataTooWide) as e:
    pins_for(bad)
  assert e.value.args[1] == 113


def test_action_bytes_count_every_slice_of_a_multi_container_field():
  widths = _widths([{"f": 33, "g": 3}], [{"f": 49}])
  pins = base_pins(widths)
  assert pins["tr_app_0_f"] == [8, 32] and pins["tr_ddos_0_f"] == [32, 32]
  # f: B + W (1 + 4) in the app tree, W + W (8) in the ddos tree; g: one B
  assert action_bytes(widths, pins) == {"f": 13, "g": 1}


def test_spilled_fields_count_as_the_16_bit_containers_they_are_pinned_to():
  widths = _counted({"a": 1, "b": 2})
  pins = base_pins(widths)
  assert action_bytes(widths, pins) == {"a": 1, "b": 2}
  pins = {**pins, per_tree_field("app", 0, "b"): [16]}
  assert action_bytes(widths, pins) == {"a": 1, "b": 3}


def test_a_tree_with_no_splits_raises():
  with pytest.raises(NoSplitTree) as e:
    pins_for(_widths([{"f": 3}, {}]))
  assert e.value.args[1] == "app_1"
  with pytest.raises(NoSplitTree):
    pins_for(_widths([{"f": 3}], [{}]))


def test_marked_key_bits_mark_exactly_the_narrow_fields_pinned_wide():
  widths = _widths([{"a": 3, "b": 12, "c": 5}])
  pins = base_pins(widths)
  assert marked_key_bits("app", 0, widths["app"][0], pins) == (3, 5, 12)
  assert not any(isinstance(b, Spilled) for b in marked_key_bits("app", 0, widths["app"][0], pins))
  pins[per_tree_field("app", 0, "c")] = [16]
  bits = marked_key_bits("app", 0, widths["app"][0], pins)
  assert bits != (3, 5, 12)                    # the marked width is not equal to the plain one
  assert [isinstance(b, Spilled) for b in bits] == [False, True, False]
  assert [int(b) for b in bits] == [3, 5, 12]
  # a 12-bit field pinned to its natural H is not a spill
  assert not isinstance(bits[2], Spilled)


def test_spilled_prices_as_h_and_stays_apart_from_a_plain_width():
  assert lanes.key_bytes([Spilled(3)]) == lanes.bytes_from_layout([3], lambda w: [("H", w)])
  assert hash(Spilled(3)) != hash(3) and Spilled(3) != 3 and 3 != Spilled(3)
  assert Spilled(3) == Spilled(3) and not Spilled(3) != Spilled(3)
  assert {Spilled(3): 1}.get(3) is None


def test_a_mixed_key_prices_the_same_in_any_order_and_after_a_pickle():
  a, b = (Spilled(3), 3, 20), (20, 3, Spilled(3))
  assert lanes.table_blocks(a) == lanes.table_blocks(b)
  assert sorted(lanes.key_bytes(a)) == sorted(lanes.key_bytes(b))
  assert isinstance(pickle.loads(pickle.dumps(Spilled(3))), Spilled)
  assert pickle.loads(pickle.dumps(Spilled(3))) == Spilled(3)


def test_a_spilled_field_is_one_crossbar_byte_of_the_h_kind():
  # ceil(3/8) == 1 byte either way; only the lane kind differs
  assert [kind for kind, _, _ in lanes.key_bytes([Spilled(3)] * 3)] == ["H"] * 3
  assert [kind for kind, _, _ in lanes.key_bytes([3] * 3)] == ["B"] * 3


def test_bare_widths_price_as_before():
  for key in [(3,), (3, 3, 20), (1, 2, 3, 40, 52), (27, 52), (9, 9, 9, 9)]:
    assert lanes.table_blocks(key) == lanes.table_blocks(list(reversed(key)))
    assert lanes.key_bytes(key) == lanes.bytes_from_layout(list(key), lanes.key_layout(list(key)))
  assert lanes.table_blocks([]) == lanes.table_blocks(())


def test_a_spilled_key_too_wide_still_raises():
  # a 512-bit field is 64 crossbar bytes; one spilled byte more is past what a stage supplies
  with pytest.raises(CrossbarKeyTooWide) as e:
    lanes.table_blocks((Spilled(3), 512))
  assert e.value.args[1] == 65
  assert lanes.table_blocks((512,)) >= 1
