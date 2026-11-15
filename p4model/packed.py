"""The packed-codes layout: ONE PHV field `code_pk_<feature>` per feature holds every
tree's code on that feature, each in BYTES OF ITS OWN (spec
docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 3.1-3.3).

Two rules keep p4c's TCAM block count independent of its PHV container numbers,
which no model can know (p4c's ternary crossbar orders key bytes of equal alignment
by container number, xbar.py; results/encoding_evidence_2026_11/model8/impl/
per_tree_accuracy.md, 2026-11-15: blocks exact on 160 of 160 compiled designs):
  * no byte holds two trees' codes. Design-review decision 4.4 (codes of <= 8 bits
    share bytes) is taken back: a later tree reuses the crossbar slot an earlier
    tree chose for the shared byte, and that slot follows the container numbers;
  * inside one tree's key, bytes of one alignment class (container size, byte
    position in the container) are all one-nibble bytes or all whole bytes. A
    one-nibble byte beside a whole byte of its class is KEYED as a whole byte
    (`Layout.keyed`): the tree's key slice is padded up to the byte boundary, and
    the padding bits, which no code owns, are don't-care in every entry.

ONE function for every consumer: the generator emits the containers as
@pa_container_size and the tree keys as slices of the field, the model prices each
tree's key from the byte list, and the entry writer ORs the shifted codes, all from
the `Layout` that `layout_for` returns, so they cannot disagree.

`widths` throughout is {task: [{feature: code bits}, one dict per tree]} (widths_of).
Reference implementation of the container choice: results/encoding_evidence_2026_11/
model4/scripts/pklib.py `plan(..., exact='big_one_size')`, which the 68 probe designs
were compiled with."""
from dataclasses import dataclass, field

from p4model import lanes, phv
from p4model.errors import ActionDataTooWide, NoSplitTree, PhvContainersExhausted
from p4model.target import (ACTION_DATA_BUS_BYTES, ACTION_RAM_WORD_BYTES,
                            MAX_KEY_SLICE_BITS)

_KIND = {32: 'W', 16: 'H', 8: 'B'}


def check_bus_bytes(feature, n):
  """ActionDataTooWide when feature's packed code field needs n > the action data
  bus's ACTION_DATA_BUS_BYTES (R4). layout_for (training, generator) and the replay
  (p4_replay, from the field's container pin) both call it."""
  if n > ACTION_DATA_BUS_BYTES:
    raise ActionDataTooWide(
        "field %s needs %d bus bytes, the action data bus carries %d"
        % (feature, n, ACTION_DATA_BUS_BYTES), n)


def widths_of(tree_intervals):
  """widths from {task: [{feature: interval list}, one dict per tree]}: a code's
  bits are the interval count minus one (the generator's own rounding, from
  get_feature_intervals on a one-tree forest). A feature with a single interval
  has no threshold in that tree and no code."""
  return {task: [{feature: len(intervals) - 1
                  for feature, intervals in tree.items() if len(intervals) >= 2}
                 for tree in trees]
          for task, trees in tree_intervals.items()}


def budgets(n_features, n_trees, n_scratch=0, n_constants=0, coarse_clock=False, fine_clock=False):
  """Containers left for code fields, per size: the ingress containers minus the
  program's own fields (phv.budgets; target.PHV_TEMPLATE_FIELDS, measured
  in Task 39 on 160 compiles of the final generator): the 32-bit template fields plus
  the coarse clock's two (phv.COARSE_CLOCK_CONTAINERS, when a Task 38 duration is
  selected), the fine clock's two 8-bit and four 16-bit ones (phv.FINE_CLOCK_CONTAINERS,
  when an IAT Max / Min keeps a last-arrival register), one 16-bit value field per selected
  feature plus one per IAT scratch field plus one per call-site constant (C1f's Min
  read) plus the 16-bit template fields, one
  8-bit container per class field (n_trees of them: phv.class_fields, none for a
  one-tree task since review A2) plus the 8-bit template fields."""
  return phv.budgets(n_features, n_scratch, n_trees, n_constants, coarse_clock, fine_clock)


@dataclass(frozen=True)
class Layout:
  """containers: feature -> container bits of its field, low first;
  place: (task, tree, feature) -> (BIT offset of the code in the field, live bits
  of the code); layout_for starts every code on a byte boundary, in bytes no other
  code touches;
  keyed: (task, tree, feature) -> the bits the tree KEYS of that code, for the
  codes whose key slice is padded up to the byte boundary (the module docstring's
  second rule); a code not listed is keyed at its live bits. A Layout rebuilt
  from a program text (p4_replay.tree_key_bytes) carries the keyed bits as its
  place and lists none."""
  containers: dict
  place: dict
  keyed: dict = field(default_factory=dict)


def _bytes(bits):
  return -(-bits // 8)


def _byte_kinds(containers):
  """[(kind, byte index in its container)] for each byte of a field, low first."""
  return [(_KIND[bits], k) for bits in containers for k in range(bits // 8)]


def _container_of_byte(containers):
  """[index of the container holding field byte j] for each byte of a field."""
  return [index for index, bits in enumerate(containers) for _ in range(bits // 8)]


def _padded_codes(containers, place):
  """Layout.keyed: the codes whose top byte is a one-nibble byte (at most 4 live
  bits) while the SAME tree's key holds a whole byte of the same alignment class
  -- (container kind, byte index in the container): the bytes p4c's crossbar sort
  leaves tied (xbar.Byte.flags) -- each mapped to its bits rounded up to whole
  bytes. Only a code's top byte can be a one-nibble byte, since every code starts
  on a byte boundary. A class whose bytes are all one-nibble bytes is left alone,
  so the half-midbyte saving survives there; padding never mixes another class."""
  kinds = {feature: _byte_kinds(sizes) for feature, sizes in containers.items()}
  whole, half = set(), {}                 # (task, tree, class); code -> its half byte's class
  for (task, tree, feature), (offset, bits) in place.items():
    first, last = offset // 8, (offset + bits - 1) // 8
    for j in range(first, last + 1):
      if j == last and bits - 8 * (last - first) <= 4:
        half[(task, tree, feature)] = kinds[feature][j]
      else:
        whole.add((task, tree, kinds[feature][j]))
  return {code: 8 * _bytes(place[code][1]) for code, cls in half.items()
          if (code[0], code[1], cls) in whole}


def layout_for(widths, features, task_order, n_scratch=0, n_constants=0, coarse_clock=False,
               fine_clock=False):
  """The packed layout (spec Sec 3.2). Features in `features` order that have a
  code; within a feature the codes in task_order, then tree index (program
  order), each in ceil(bits / 8) bytes of its own, ordered by the lane balancing
  below; then the padded key slices (_padded_codes). Raises NoSplitTree (a tree without a code),
  ActionDataTooWide (a field over the bus's 128 bytes) or PhvContainersExhausted.
  A field's containers are the canonical ones for its bytes (spec Sec 3.2 rule 1:
  32-bit for 4 bytes or more, or whenever the field is over one RAM word; else 16
  for 2-3, else 8), and a size with none left is REFUSED, never replaced by
  another size (decision 4.2): its second arg is the containers short, counted over
  the field that cannot be placed (the layout stops at the first such field).
  `n_scratch` is the number of IAT scratch fields the program declares and
  `n_constants` the number of its call-site constants (16-bit containers of its
  own, phv.budgets); `coarse_clock`: the program computes the Task 38 coarse clock
  (two 32-bit containers of its own); `fine_clock`: it computes the fine IAT clock
  (phv.FINE_CLOCK_CONTAINERS).

  Lane balancing (spec Sec 3.2 rule 2, design review 3.13): at each byte offset
  take the remaining code whose tree's per-lane key-byte counts stay most even --
  W byte k counts in lane k, H byte k in the less loaded of lanes k and k+2, B in
  the least loaded lane -- scored as that tree's max-min; ties to program order."""
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
  left = budgets(len(codes), phv.class_fields(len(widths.get(task, [])) for task in task_order),
                 n_scratch, n_constants, coarse_clock, fine_clock)
  lane_count = {}                                   # (task, tree) -> key bytes per lane
  containers, place = {}, {}
  for feature, found in codes.items():
    n = sum(_bytes(bits) for _, _, bits in found)
    check_bus_bytes(feature, n)
    chosen, covered, short = [], 0, 0
    while covered < n:
      rest = n - covered
      size = 32 if (rest >= 4 or n > ACTION_RAM_WORD_BYTES) else (16 if rest >= 2 else 8)
      left[size] -= 1
      short += left[size] < 0
      chosen.append(size)
      covered += size // 8
    if short:
      raise PhvContainersExhausted(
          "no PHV container of the canonical size left for the code field of %s "
          "(%s bits wanted, %d short)" % (feature, chosen, short), short)
    containers[feature] = tuple(chosen)
    kinds = _byte_kinds(chosen)
    remaining, offset = list(found), 0

    def cost(code, offset):
      """(score, the tree's lane counts after the code) at byte `offset`."""
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

    while remaining:
      pick = min(remaining, key=lambda code: cost(code, offset)[0])
      task, tree, bits = pick
      lane_count[(task, tree)] = cost(pick, offset)[1]
      place[(task, tree, feature)] = (8 * offset, bits)
      offset += _bytes(bits)
      remaining.remove(pick)
  return Layout(containers, place, _padded_codes(containers, place))


def _keyed(layout, task, tree, feature):
  """(bit offset, bits the tree keys) of one code: its live bits, or its padded
  width (Layout.keyed)."""
  offset, bits = layout.place[(task, tree, feature)]
  return offset, layout.keyed.get((task, tree, feature), bits)


def _live_mask(offset, bits, byte):
  """The bits of field byte `byte` (as an 8-bit mask) a `bits`-bit code at field
  bit `offset` covers."""
  lo, hi = max(offset, 8 * byte), min(offset + bits, 8 * byte + 8)
  return ((1 << (hi - lo)) - 1) << (lo - 8 * byte) if hi > lo else 0


def one_nibble(mask):
  """p4c's Use::Byte::only_one_nibble on a byte's live-bit mask (xbar.Byte): all
  live bits in the low nibble or all in the high one. A byte with live bits in
  BOTH nibbles is not a one-nibble byte, however few they are (design review
  P13)."""
  return (mask & 0x0F) == 0 or (mask & 0xF0) == 0


def key_bytes(layout, task, tree):
  """The tree's key as byte tuples (kind, k, nibble_only, container): features in
  layout order, each code's bytes ascending; k is the byte's index in its
  container, nibble_only when the bits the TREE KEYS in the byte lie in one
  nibble (one_nibble; never for a padded code, Layout.keyed), container =
  (feature, index of the container in its field): the byte's identity, which names
  one container byte to a stage's crossbar (xbar.tree_key). The first three
  elements are lanes.py's byte-list tuple."""
  out = []
  for feature, containers in layout.containers.items():
    if (task, tree, feature) not in layout.place:
      continue
    offset, bits = _keyed(layout, task, tree, feature)
    kinds, owner = _byte_kinds(containers), _container_of_byte(containers)
    for j in range(offset // 8, (offset + bits - 1) // 8 + 1):
      kind, k = kinds[j]
      out.append((kind, k, one_nibble(_live_mask(offset, bits, j)), (feature, owner[j])))
  return tuple(out)


def key_slices(layout, task, tree, feature):
  """[(hi, lo)] in the field's bit positions of the bits the tree keys of its code
  on `feature` (its live bits, or its padded width: Layout.keyed), cut into slices
  of at most MAX_KEY_SLICE_BITS from the code's bit 0, low first."""
  offset, bits = _keyed(layout, task, tree, feature)
  return bit_slices(bits, offset)


def key_bits(layout, task, tree):
  """The bits of the tree's key: its key slices' widths summed, padding included
  -- what p4c's key-width check counts (tables.check_key_bits) and what the
  replay sums from the program's slices (p4_replay.tree_key_bits)."""
  return sum(_keyed(layout, task, tree, feature)[1]
             for (t, i, feature) in layout.place if (t, i) == (task, tree))


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
