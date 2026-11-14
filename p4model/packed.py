"""The packed-codes layout: ONE PHV field `code_pk_<feature>` per feature holds every
tree's code on that feature, each at its own bit offset (spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 3.1-3.3;
design review decision 4.4, 2026-11-14: codes of <= 8 bits share bytes, no code
straddles a byte boundary: a code of > 8 bits starts on a fresh byte).
A byte holding codes of several trees is a byte of each of their keys (each keys
its own bit slice of it; the crossbar moves whole bytes).

ONE function for every consumer: the generator emits the containers as
@pa_container_size and the tree keys as slices of the field, the model prices each
tree's key from the byte list, and the entry writer ORs the shifted codes, all from
the `Layout` that `layout_for` returns, so they cannot disagree.

`widths` throughout is {task: [{feature: code bits}, one dict per tree]} (widths_of).
Reference implementation of the layout: results/encoding_evidence_2026_11/model4/
scripts/pklib.py `plan(..., exact='big_one_size')`, which the 68 probe designs were
compiled with; the byte packing of decision 4.4 is model8/design_review/scripts/
nostraddle_scan.py's (pack_bytes reproduces its byte counts on the 272 campaign_2026_11e
designs)."""
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
  place: (task, tree, feature) -> (BIT offset of the code in the field, live bits
  of the code). A code of more than 8 bits starts on a byte boundary (offset % 8
  == 0); a code of <= 8 bits lies inside one byte (decision 4.4)."""
  containers: dict
  place: dict


def _bytes(bits):
  return -(-bits // 8)


def _byte_kinds(containers):
  """[(kind, byte index in its container)] for each byte of a field, low first."""
  return [(_KIND[bits], k) for bits in containers for k in range(bits // 8)]


def _container_of_byte(containers):
  """[index of the container holding field byte j] for each byte of a field."""
  return [index for index, bits in enumerate(containers) for _ in range(bits // 8)]


def straddles(offset, bits):
  """Whether a `bits`-bit code at field bit `offset` breaks decision 4.4: a code of
  <= 8 bits must lie inside one byte, a longer one must start on a byte boundary."""
  if bits <= 8:
    return offset // 8 != (offset + bits - 1) // 8
  return offset % 8 != 0


def pack_bytes(found):
  """Decision 4.4: the field's codes packed into bytes, first-fit decreasing
  (results/encoding_evidence_2026_11/model8/design_review/scripts/nostraddle_scan.py
  packed_bytes, whose byte counts this reproduces). `found` is [(task, tree, bits)]
  in program order. A code of > 8 bits takes ceil(bits/8) fresh bytes, and the free
  bits of its last byte can take small codes; a code of <= 8 bits goes into the
  first byte made so far (in making order) with room for it whole, else a fresh
  byte. Codes are taken widest first, ties in program order.

  Returns UNITS, [(n bytes, [(position in found, bit offset in the unit)])]: a wide
  code with the small codes in its last byte, or one byte of small codes. Units
  come in program order of their first code (the order the lane balancing breaks
  ties in), codes in a unit low bit first."""
  units = []              # [n bytes, [(i, bit offset)], next free bit]
  for i in sorted(range(len(found)), key=lambda i: -found[i][2]):
    bits = found[i][2]
    if bits > 8:
      units.append([_bytes(bits), [(i, 0)], bits])
      continue
    for unit in units:
      if 8 * unit[0] - unit[2] >= bits:
        unit[1].append((i, unit[2]))
        unit[2] += bits
        break
    else:
      units.append([1, [(i, 0)], bits])
  units.sort(key=lambda unit: min(i for i, _ in unit[1]))
  return [(n, codes) for n, codes, _ in units]


def layout_for(widths, features, task_order):
  """The packed layout (spec Sec 3.2, decision 4.4). Features in `features` order
  that have a code; within a feature the codes in task_order, then tree index
  (program order), packed into bytes by pack_bytes; the units are then ordered by
  the lane balancing below. Raises NoSplitTree (a tree without a code),
  ActionDataTooWide (a field over the bus's 128 bytes) or PhvContainersExhausted
  (no container of any size left; its second arg is 1, a flag that the layout ran
  out, not how many containers it was short -- the layout stops at the first field
  it cannot place).

  Lane balancing (spec Sec 3.2 rule 2, design review 3.13, unchanged per byte): at
  each byte offset take the remaining unit whose trees' per-lane key-byte counts
  stay most even -- W byte k counts in lane k, H byte k in the less loaded of lanes
  k and k+2, B in the least loaded lane, once for every tree with live bits in the
  byte -- scored as the LARGEST max-min over the unit's trees; ties to program
  order of the units' first codes. A unit of one code is today's rule exactly, so a
  field where no codes share a byte keeps its pre-4.4 order."""
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
    units = pack_bytes(found)
    n = sum(nbytes for nbytes, _ in units)
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
    remaining, offset = list(units), 0

    def cost(unit, offset):
      """(score, {(task, tree): lane counts after the unit}) at byte `offset`."""
      after = {}
      for i, bit in unit[1]:
        task, tree, bits = found[i]
        counts = list(lane_count.get((task, tree), (0, 0, 0, 0)))
        for j in range(bit // 8, (bit + bits - 1) // 8 + 1):
          kind, k = kinds[offset + j]
          if kind == 'W':
            counts[k] += 1
          elif kind == 'H':
            counts[k if counts[k] <= counts[k + 2] else k + 2] += 1
          else:
            counts[counts.index(min(counts))] += 1
        after[(task, tree)] = counts
      return max(max(c) - min(c) for c in after.values()), after

    while remaining:
      pick = min(remaining, key=lambda unit: cost(unit, offset)[0])
      lane_count.update(cost(pick, offset)[1])
      for i, bit in pick[1]:
        task, tree, bits = found[i]
        place[(task, tree, feature)] = (8 * offset + bit, bits)
      offset += pick[0]
      remaining.remove(pick)
  return Layout(containers, place)


def _live_mask(offset, bits, byte):
  """The bits of field byte `byte` (as an 8-bit mask) a `bits`-bit code at field
  bit `offset` covers."""
  lo, hi = max(offset, 8 * byte), min(offset + bits, 8 * byte + 8)
  return ((1 << (hi - lo)) - 1) << (lo - 8 * byte) if hi > lo else 0


def one_nibble(mask):
  """p4c's Use::Byte::only_one_nibble on a byte's live-bit mask (xbar.Byte): all
  live bits in the low nibble or all in the high one. A byte with live bits in
  BOTH nibbles is not a one-nibble byte, however few they are (design review P13,
  decision 4.4)."""
  return (mask & 0x0F) == 0 or (mask & 0xF0) == 0


def key_bytes(layout, task, tree):
  """The tree's key as byte tuples (kind, k, nibble_only, container): features in
  layout order, each code's bytes ascending; k is the byte's index in its
  container, nibble_only when the TREE's own live bits in the byte lie in one
  nibble (one_nibble), container = (feature, index of the container in its field):
  the byte's identity, which two trees keying codes in one byte share (decision
  4.4), so that a stage's crossbar holds the byte once (xbar.tree_key). The first
  three elements are lanes.py's byte-list tuple."""
  out = []
  for feature, containers in layout.containers.items():
    if (task, tree, feature) not in layout.place:
      continue
    offset, bits = layout.place[(task, tree, feature)]
    kinds, owner = _byte_kinds(containers), _container_of_byte(containers)
    for j in range(offset // 8, (offset + bits - 1) // 8 + 1):
      kind, k = kinds[j]
      out.append((kind, k, one_nibble(_live_mask(offset, bits, j)), (feature, owner[j])))
  return tuple(out)


def key_slices(layout, task, tree, feature):
  """[(hi, lo)] in the field's bit positions of the tree's code on `feature`, cut
  into slices of at most MAX_KEY_SLICE_BITS from the code's bit 0, low first."""
  offset, bits = layout.place[(task, tree, feature)]
  return bit_slices(bits, offset)


def bit_slices(bits, base=0):
  """[(hi, lo)] field bit positions of a `bits`-bit code starting at field bit
  `base`, cut into slices of at most MAX_KEY_SLICE_BITS (p4c's ternary key-slice
  limit) from the code's bit 0, low first. Every encoding's tree keys read their
  codes this way: a packed tree its code at its bit offset (key_slices), any
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
  """The one value a range entry writes: the OR of each tree's code shifted to its
  bit offset (codes sharing a byte occupy disjoint bits of it).
  codes = {(task, tree): int}."""
  value = 0
  for (task, tree), code in codes.items():
    offset, bits = layout.place[(task, tree, feature)]
    if code >> bits:
      raise ValueError('code %#x of tree %s_%d on %s is wider than its %d bits'
                       % (code, task, tree, feature, bits))
    value |= code << offset
  return value
