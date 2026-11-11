"""The per-tree encoding's pin rule and the sizes it implies: which container each
per-tree code field is pinned to, which of them spill from 8 to 16 bits, and how
many action-data bytes a range table's action writes (program.PER_TREE_ENCODING).

ONE function for both sides: the generator emits the pins it returns as
@pa_container_size, and the model prices keys from the same pins
(marked_key_bits), so training, the generated program and its replay cannot
disagree about a layout. Spec docs/superpowers/specs/2026-11-06-per-tree-arm-design.md
Sec 7 (spill by range table), 9.2 (pin rule), 9.3 (action data bus), 9.4 (a tree
with no splits) and the amendments of 2026-11-08 (A3).

`widths` throughout is {task: [{feature: code bits}, one dict per tree]}; a
resolved field name is program.per_tree_field(task, tree, feature), the metadata
field being code_<resolved>."""
import itertools

from p4model import lanes
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted
from p4model.program import per_tree_field
from p4model.target import (PER_TREE_MAX_ACTION_BYTES, PER_TREE_MAX_PIN8_FIELDS,
                            PHV_INGRESS_CONTAINERS, PHV_TEMPLATE_FIELDS)

# spill_table_fields enumerates feature subsets; the arm's designs have at most
# 17 selected features.
_MAX_SPILL_FEATURES = 20


def widths_of(tree_intervals):
  """widths from {task: [{feature: interval list}, one dict per tree]}: a code's
  bits are the interval count minus one (the generator's own rounding, from
  get_feature_intervals on a one-tree forest). A feature with a single interval
  has no threshold in that tree and no code."""
  return {task: [{feature: len(intervals) - 1
                  for feature, intervals in tree.items() if len(intervals) >= 2}
                 for tree in trees]
          for task, trees in tree_intervals.items()}


def base_pins(widths):
  """{resolved field: container sizes in bits, low slice first}, before any
  spill: each tree's key is laid out by lanes.key_layout, the choice pricing
  makes, and each field pinned to lanes.container_sizes of it (the generator's
  code_field_container_sizes rule)."""
  pins = {}
  for task, trees in widths.items():
    for index, tree in enumerate(trees):
      if not tree:
        continue
      layout = lanes.key_layout(list(tree.values()))
      for feature, bits in tree.items():
        pins[per_tree_field(task, index, feature)] = lanes.container_sizes(bits, layout)
  return pins


def code_slices8(widths, pins):
  """8-bit PHV containers the code pins take: every 8 in a code's container
  sizes, the narrow codes ([8]) and the 8-bit slices of multi-container codes
  ([8, 32], [16, 8], ...) alike. @pa_container_size is a hard requirement in
  p4c, so each such slice needs an 8-bit container of its own."""
  return sum(pins[per_tree_field(task, index, feature)].count(8)
             for task, trees in widths.items()
             for index, tree in enumerate(trees) for feature in tree)


def spill_table_fields(widths, pins, max_pin8):
  """The resolved fields that move from an 8-bit to a 16-bit pin (spec Sec 7):
  at a spill, ALL narrow codes (those pinned [8]) of a feature move together,
  so one range table's action never writes through both 8-bit and 16-bit
  action-data slots. need = 8-bit code slices (code_slices8) beyond
  `max_pin8`; only narrow codes can move, so the 8-bit slices of
  multi-container codes count toward the need but never spill. The features
  moved are the set with the smallest total of narrow codes at or above need,
  then the fewest features, then alphabetical; when the narrow codes cannot
  cover the need, all of them move. The list is in a fixed order: chosen
  features alphabetical, within one tasks alphabetical, trees ascending.

  The subsets are enumerated, but not blindly: the target total is found first
  by a subset-sum over the counts, and the subsets are then walked by size, in
  alphabetical order, until one reaches it."""
  narrow = {}
  for task in sorted(widths):
    for index, tree in enumerate(widths[task]):
      for feature in tree:
        name = per_tree_field(task, index, feature)
        if pins[name] == [8]:
          narrow.setdefault(feature, []).append(name)
  available = sum(len(names) for names in narrow.values())
  need = min(code_slices8(widths, pins) - max_pin8, available)
  if need <= 0:
    return []
  features = sorted(narrow)
  if need == available:
    return [name for f in features for name in narrow[f]]
  if len(features) > _MAX_SPILL_FEATURES:
    raise ValueError("too many features to enumerate a spill: %d" % len(features))
  totals = {0}
  for feature in features:
    totals |= {t + len(narrow[feature]) for t in totals}
  target = min(t for t in totals if t >= need)
  for size in range(1, len(features) + 1):
    for chosen in itertools.combinations(features, size):
      if sum(len(narrow[f]) for f in chosen) == target:
        return [name for f in chosen for name in narrow[f]]
  raise AssertionError("unreachable: the target total is a subset sum")


def pins_for(widths):
  """(pins, spilled): {resolved field: container sizes in bits} for every code
  field of the design, and the resolved fields spilled from [8] to [16], in
  spill_table_fields' order. The code pins keep at most
  target.PER_TREE_MAX_PIN8_FIELDS 8-bit slices (code_slices8) whenever the
  narrow codes can absorb the spill.

  Raises NoSplitTree(message, "<task>_<tree>") for a tree with no code field
  (spec 9.4), then ActionDataTooWide(message, bytes) when a feature's range
  table would write more than target.PER_TREE_MAX_ACTION_BYTES (spec 9.3; the
  largest such feature, counted after the spill), then
  PhvContainersExhausted(message, containers over) when the 8-bit code slices
  left after the spill pass the ingress 8-bit share (only multi-container codes
  remain then), or phv_demand's 16-bit count passes the 16-bit share."""
  for task, trees in widths.items():
    for index, tree in enumerate(trees):
      if not tree:
        raise NoSplitTree("tree %s_%d has no split, so its table would have no key"
                          % (task, index), "%s_%d" % (task, index))
  pins = base_pins(widths)
  spilled = spill_table_fields(widths, pins, PER_TREE_MAX_PIN8_FIELDS)
  for name in spilled:
    pins[name] = [16]
  per_feature = action_bytes(widths, pins)
  feature = max(per_feature, key=lambda f: (per_feature[f], f), default=None)
  if feature is not None and per_feature[feature] > PER_TREE_MAX_ACTION_BYTES:
    raise ActionDataTooWide(
        "the range table of %s writes %d container bytes; the action data bus "
        "carries at most %d" % (feature, per_feature[feature], PER_TREE_MAX_ACTION_BYTES),
        per_feature[feature])
  slices8 = code_slices8(widths, pins)
  if slices8 > PHV_INGRESS_CONTAINERS[8]:
    raise PhvContainersExhausted(
        "the code pins need %d 8-bit PHV containers after the spill; ingress has %d"
        % (slices8, PHV_INGRESS_CONTAINERS[8]), slices8 - PHV_INGRESS_CONTAINERS[8])
  need16 = phv_demand(widths, pins)[16]
  if need16 > PHV_INGRESS_CONTAINERS[16]:
    raise PhvContainersExhausted(
        "the design needs %d 16-bit PHV containers; ingress has %d"
        % (need16, PHV_INGRESS_CONTAINERS[16]), need16 - PHV_INGRESS_CONTAINERS[16])
  return pins, spilled


def phv_demand(widths, pins):
  """{container bits: ingress containers the program needs} for 8 and 16, every
  field counted from the program the generator writes for these pins:

  - each per-tree code takes one container per pinned slice (its
    @pa_container_size sizes, which are also @pa_solitary);
  - each feature with a code has one value field, pinned to 16 bits;
  - each tree has one class field (class_tree_<task>_<i>, @pa_solitary), a 1-
    or 2-bit field that takes a whole 8-bit container;
  - the template's own fields of target.PHV_TEMPLATE_FIELDS.

  8-bit fields beyond the 8-bit share are counted at 16 bits: when the 8-bit
  containers are full, p4c moves them to 16-bit ones (class fields: 151 of the
  163 that left an 8-bit container in the pilot went to a 16-bit one)."""
  need = {8: len(PHV_TEMPLATE_FIELDS[8]), 16: len(PHV_TEMPLATE_FIELDS[16])}
  for task, trees in widths.items():
    for index, tree in enumerate(trees):
      need[8] += 1                                    # class_tree_<task>_<index>
      for feature in tree:
        for bits in pins[per_tree_field(task, index, feature)]:
          if bits in need:
            need[bits] += 1
  need[16] += len({feature for trees in widths.values() for tree in trees for feature in tree})
  need[16] += max(0, need[8] - PHV_INGRESS_CONTAINERS[8])
  return need


def action_bytes(widths, pins):
  """{feature: container bytes its range table's action writes}: for every
  per-tree code on the feature, the sum of 4W + 2H + B over the containers the
  code is pinned to (spilled fields count as the 16-bit containers they are
  pinned to; a multi-container field counts every slice)."""
  total = {}
  for task, trees in widths.items():
    for index, tree in enumerate(trees):
      for feature in tree:
        size = sum(bits // 8 for bits in pins[per_tree_field(task, index, feature)])
        total[feature] = total.get(feature, 0) + size
  return total


def marked_key_bits(task, tree, tree_widths, pins):
  """The key bits lanes.table_blocks prices for tree `tree` of `task`: its code
  widths sorted, a field of 8 bits or fewer that the pins put in a 16-bit
  container marked lanes.Spilled. This is THE spilled test; the replay reads a
  program's @pa_container_size with the same one, so training and replay price
  the same layout."""
  bits = []
  for feature, width in tree_widths.items():
    spilled = width <= 8 and pins[per_tree_field(task, tree, feature)] == [16]
    bits.append(lanes.Spilled(width) if spilled else width)
  return tuple(sorted(bits, key=lambda b: (int(b), isinstance(b, lanes.Spilled))))
