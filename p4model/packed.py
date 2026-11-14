"""The packed-codes layout: ONE PHV field `code_pk_<feature>` per feature holds every
tree's code on that feature, each at its own byte offset (spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 3.1-3.3).

ONE function for every consumer: the generator emits the containers as
@pa_container_size and the tree keys as slices of the field, the model prices each
tree's key from the byte list, and the entry writer concatenates the codes, all from
the `Layout` that `layout_for` returns, so they cannot disagree.

`widths` throughout is {task: [{feature: code bits}, one dict per tree]} (widths_of).
Reference implementation of the layout: results/encoding_evidence_2026_11/model4/
scripts/pklib.py `plan(..., exact='big_one_size')`, which the 68 probe designs were
compiled with."""
from dataclasses import dataclass

from p4model import lanes
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted
from p4model.target import (ACTION_DATA_BUS_BYTES, ACTION_RAM_WORD_BYTES,
                            MAX_KEY_SLICE_BITS, PHV_INGRESS_CONTAINERS,
                            PHV_TEMPLATE_FIELDS)

_KIND = {32: 'W', 16: 'H', 8: 'B'}
# The size to take when the wanted one has no container left (spec Sec 3.2 rule 1).
_FALL = {32: (32, 16, 8), 16: (16, 8, 32), 8: (8, 16, 32)}


def widths_of(tree_intervals):
  """widths from {task: [{feature: interval list}, one dict per tree]}: a code's
  bits are the interval count minus one (the generator's own rounding, from
  get_feature_intervals on a one-tree forest). A feature with a single interval
  has no threshold in that tree and no code."""
  return {task: [{feature: len(intervals) - 1
                  for feature, intervals in tree.items() if len(intervals) >= 2}
                 for tree in trees]
          for task, trees in tree_intervals.items()}


def budgets(n_features, n_trees):
  """Containers left for code fields, per size: the ingress containers minus the
  program's own fields (spec Sec 3.3, measured on 68 packed compiles): the 32-bit
  template fields, one 16-bit value field per selected feature plus the 16-bit
  template fields, one 8-bit class field per tree plus the 8-bit template fields."""
  return {32: PHV_INGRESS_CONTAINERS[32] - len(PHV_TEMPLATE_FIELDS[32]),
          16: PHV_INGRESS_CONTAINERS[16] - n_features - len(PHV_TEMPLATE_FIELDS[16]),
          8: PHV_INGRESS_CONTAINERS[8] - n_trees - len(PHV_TEMPLATE_FIELDS[8])}


@dataclass(frozen=True)
class Layout:
  """containers: feature -> container bits of its field, low first;
  place: (task, tree, feature) -> (byte offset in the field, live bits of the code)."""
  containers: dict
  place: dict


def _bytes(bits):
  return -(-bits // 8)


def _byte_kinds(containers):
  """[(kind, byte index in its container)] for each byte of a field, low first."""
  return [(_KIND[bits], k) for bits in containers for k in range(bits // 8)]


def layout_for(widths, features, task_order):
  """The packed layout (spec Sec 3.2). Features in `features` order that have a
  code; within a feature the codes in task_order, then tree index.
  Raises NoSplitTree (a tree without a code), ActionDataTooWide (a field over
  the bus's 128 bytes) or PhvContainersExhausted (no container of any size left;
  its second arg is 1, a flag that the layout ran out, not how many containers
  it was short -- the layout stops at the first field it cannot place)."""
  for task in task_order:
    for index, tree in enumerate(widths.get(task, [])):
      if not tree:
        raise NoSplitTree("tree %s_%d has no threshold, so its table would have no key"
                          % (task, index), "%s_%d" % (task, index))
  codes = {feature: [(task, index, tree[feature])
                     for task in task_order
                     for index, tree in enumerate(widths.get(task, []))
                     if feature in tree]
           for feature in features}
  codes = {feature: found for feature, found in codes.items() if found}
  left = budgets(len(codes), sum(len(widths.get(task, [])) for task in task_order))
  lane_count = {}                                   # (task, tree) -> key bytes per lane
  containers, place = {}, {}
  for feature, found in codes.items():
    n = sum(_bytes(bits) for _, _, bits in found)
    if n > ACTION_DATA_BUS_BYTES:
      raise ActionDataTooWide(
          "field %s needs %d bus bytes, the action data bus carries %d"
          % (feature, n, ACTION_DATA_BUS_BYTES), n)
    chosen, covered = [], 0
    while covered < n:
      rest = n - covered
      want = 32 if (rest >= 4 or n > ACTION_RAM_WORD_BYTES) else (16 if rest >= 2 else 8)
      for size in _FALL[want]:
        if left[size] > 0:
          left[size] -= 1
          break
      else:
        raise PhvContainersExhausted(
            "no PHV container left for the code field of %s" % feature, 1)
      chosen.append(size)
      covered += size // 8
    containers[feature] = tuple(chosen)
    kinds = _byte_kinds(chosen)
    remaining, offset = list(found), 0
    while remaining:
      def cost(code):
        task, tree, bits = code
        counts = list(lane_count.get((task, tree), (0, 0, 0, 0)))
        for j in range(_bytes(bits)):
          kind, k = kinds[offset + j]
          if kind == 'W':
            counts[k] += 1
          elif kind == 'H':
            counts[k if counts[k] <= counts[k + 2] else k + 2] += 1
          else:
            counts[counts.index(min(counts))] += 1
        return max(counts) - min(counts), counts
      pick = min(remaining, key=lambda code: cost(code)[0])
      task, tree, bits = pick
      lane_count[(task, tree)] = cost(pick)[1]
      place[(task, tree, feature)] = (offset, bits)
      offset += _bytes(bits)
      remaining.remove(pick)
  return Layout(containers, place)


def key_bytes(layout, task, tree):
  """The tree's key as lanes.py byte tuples (kind, k, nibble_only): features in
  layout order, each code's bytes ascending; k is the byte's index in its
  container, nibble_only when the byte holds at most 4 live bits."""
  out = []
  for feature, containers in layout.containers.items():
    if (task, tree, feature) not in layout.place:
      continue
    offset, bits = layout.place[(task, tree, feature)]
    kinds = _byte_kinds(containers)
    for j in range(_bytes(bits)):
      kind, k = kinds[offset + j]
      out.append((kind, k, min(8, bits - 8 * j) <= 4))
  return tuple(out)


def key_slices(layout, task, tree, feature):
  """[(hi, lo)] in the field's bit positions of the tree's code on `feature`, cut
  into slices of at most MAX_KEY_SLICE_BITS from the code's bit 0, low first."""
  offset, bits = layout.place[(task, tree, feature)]
  return bit_slices(bits, 8 * offset)


def bit_slices(bits, base=0):
  """[(hi, lo)] field bit positions of a `bits`-bit code starting at field bit
  `base`, cut into slices of at most MAX_KEY_SLICE_BITS (p4c's ternary key-slice
  limit) from the code's bit 0, low first. Every encoding's tree keys read their
  codes this way: a packed tree its code at its byte offset (key_slices), any
  other tree the live bits of a code field declared at its containers' full width
  (base 0; decision 4.3)."""
  return [(base + min(bits, lo + MAX_KEY_SLICE_BITS) - 1, base + lo)
          for lo in range(0, bits, MAX_KEY_SLICE_BITS)]


def field_bits(layout, feature):
  """Container bits of the feature's field."""
  return sum(layout.containers[feature])


def bus_bytes(layout, feature):
  """Action data bus bytes the range table writing the field needs."""
  return field_bits(layout, feature) // 8


def range_value(layout, feature, codes):
  """The one value a range entry writes: each tree's code at its byte offset.
  codes = {(task, tree): int}."""
  value = 0
  for (task, tree), code in codes.items():
    value |= code << (8 * layout.place[(task, tree, feature)][0])
  return value
