# SPDX-License-Identifier: Apache-2.0
# Portions derived from open-p4studio (https://github.com/p4lang/open-p4studio), commit 0e81a46,
# p4c/backends/tofino/bf-p4c/mau/tofino/input_xbar.cpp and mau/input_xbar.cpp:
# Copyright (C) 2024 Intel Corporation. Licensed under the Apache License, Version 2.0.
# Ported to Python, ternary input-crossbar path only, and modified for this project.
"""p4c's Tofino ternary input-crossbar allocation, ported (spec 2026-11-13, crossbar port).

Which crossbar slots a stage's tree keys take, and so how many TCAM blocks each costs, depends on the order p4c
allocates them in and on what the earlier tables left. The per-tree model used to price every key alone and keep
crowded stages out by a 12-group rule plus generator @stage pins; this module replaces both by doing what p4c does.

Source: open-p4studio, p4c/backends/tofino/bf-p4c/mau/tofino/input_xbar.cpp -- calculate_sizes /
increase_ternary_ixbar_space (494, 485), setup_byte_vectors (986), calculate_found (671), calculate_free (714),
found_bytes (758), found_mid_bytes (795), free_bytes (831), free_mid_bytes (879), fill_out_use (955),
allocate_mid_bytes (1039), grp_use::is_better_group (1116), allocate_groups (1258), find_alloc (1347); the byte
alignment flags come from mau/input_xbar.cpp add_use (need_align_flags). Only the ternary path is ported, with
byte_mask == ~0 and no @ixbar_group_num pragma (this project uses neither). p4c allocates a stage's tables one at a
time in placement order, each on top of the earlier ones (table_placement.cpp try_alloc_ixbar).

Validation (results/encoding_evidence_2026_11/model7/findings.md): given p4c's own stage contents, placement order
and PHV containers, the allocation is slot-for-slot identical to p4c's on every ternary table of 2,481 compiles
(12,786 tree tables, 21,593 range tables). tests/test_xbar.py replays six of those programs.

A stage's ternary crossbar is 12 groups x 5 byte slots plus 6 midbytes (one per pair of groups). A key byte is a
(PHV container, byte offset) pair; its container's size fixes which slots it may use (slot ordinal mod 4): any slot
for an 8-bit container, two of four for a 16-bit one, one of four for a 32-bit one.

The one input the model does not know is p4c's PHV container NUMBERING. It only orders key bytes of equal alignment,
and so decides whether a midbyte takes a whole byte or a half byte. The model settles it with tie='whole_first' (a
midbyte takes a whole byte when one is eligible). tie='index' is p4c's own order. Code-field containers are numbered
widest field first (code_container_order, an ordering assumption, P12).
whole_first is NOT a one-sided bound. Evidence (Task 39 fix round 1, impl/scripts/t39_ties.py, p4c's own stage
contents and placement order, the model's key bytes; impl/t39/ties/report_all.txt):
  * 3,199 per-tree DEV compiles of the model6 / model7 investigations (generators before the design review): 0 of
    15,120 tree tables in shared stages priced under p4c; over on 31 / 104 tables (k<=8 / k>=9), i.e. 37 of 1,549 /
    136 of 1,650 programs over by one block or more;
  * the 38 per-tree programs of the Task 39 recompile (final generator): 4 of 170 shared-stage tables UNDER by one
    block (joint-tree_M075_s02_k07 app_1, M055_s00_k11 app_0, M055_s03_k12 app_0, M075_s01_k09 app_1), 3 of them
    still under when the bytes carry p4c's ACTUAL container numbers (only p4c's numbers with tie='index' reproduce
    p4c on all of them, and on every table of every archive).
nibble_first and index-on-the-model's-numbering under-price far more often (219 and 69 of the 3,237 programs).
"""
from collections import defaultdict
import re

TERNARY_GROUPS, BYTE_GROUPS, BYTES_PER_GROUP, _PAIR_ORDINALS = 12, 6, 5, 11
NONE_SPEC, RANGE_LO, RANGE_HI = 0, 3, 4                     # IXBar::byte_speciality_t values
A16LO, A16HI, A32LO, A32HI = 4, 8, 16, 32                   # IXBar::Use::flags_t
ALIGN_FAIL = (A16HI | A32HI, A16LO | A32HI, A16HI | A32LO, A16LO | A32LO)   # align_flags[]: flags that FAIL at ord%4
NEED_ALIGN = {8: (0, 0, 0, 0), 16: (A16LO, A16HI, A16LO, A16HI),
              32: (A16LO | A32LO, A16HI | A32LO, A16LO | A32HI, A16HI | A32HI)}
_KIND = {'T': 0, 'D': 1, 'M': 2, '': 3}                     # PHV::Kind order: tagalong < dark < mocha < normal
_SIZE = {'B': 8, 'H': 16, 'W': 32}
_CONTAINER = re.compile(r'^(T|D|M|)([BHW])(\d+)$')
TIES = ('index', 'whole_first', 'nibble_first')
BPG = BYTES_PER_GROUP


def container_info(name):
  """'W12' -> (sort key, size). The sort key follows PHV::Container::operator< (kind, size, index)."""
  m = _CONTAINER.match(name)
  if not m:
    raise ValueError('not a PHV container name: %r' % (name,))
  size = _SIZE[m[2]]
  return (_KIND[m[1]], size, int(m[3])), size


def ternary_ord(group, byte):
  """Loc::getOrd(true) for Tofino: 128 + (g/2)*11 + (g%2)*6 + byte. byte 5 is the midbyte (group even)."""
  return 128 + (group // 2) * _PAIR_ORDINALS + (group % 2) * (BPG + 1) + byte


class Byte:
  """IXBar::Use::Byte, reduced to what the ternary allocator reads. `loc` is (group, slot) once allocated, slot
  BYTES_PER_GROUP meaning the midbyte of the group pair."""
  __slots__ = ('container', 'lo', 'bits', 'spec', 'range_index', 'loc', 'ckey', 'size', 'tag')

  def __init__(self, container, lo, bits, spec=NONE_SPEC, range_index=-1, tag=None, order=None):
    """order: None, or the container's place in the stage's numbering when the name's index does not give it
    (code_container_order, for the code fields of field_key). The container sort key is (kind, size, (0, index))
    for a named container and (kind, size, (1,) + order) for an ordered one."""
    self.container, self.lo, self.bits, self.spec, self.range_index = container, lo, bits, spec, range_index
    (kind, size, index), self.size = container_info(container)
    self.ckey = (kind, size, (0, index) if order is None else (1,) + tuple(order))
    self.loc = None
    self.tag = tag                         # free for callers (the tests keep p4c's slot here)

  @property
  def flags(self):
    return NEED_ALIGN[self.size][(self.lo // 8) & 3]

  @property
  def is_range(self):
    return self.spec in (RANGE_LO, RANGE_HI)

  def only_one_nibble(self):
    if self.is_range:
      return False
    return (self.bits & 0x0F) == 0 or (self.bits & 0xF0) == 0

  def lt_key(self):
    """Use::Byte::operator< (container, lo, match_index=0, specialities, range_index)."""
    return (self.ckey, self.lo, 0, (1 << self.spec) if self.spec else 0, self.range_index)

  def ident(self):
    return (self.container, self.lo)

  def __repr__(self):
    s = {0: '', RANGE_LO: ' rlo', RANGE_HI: ' rhi'}[self.spec]
    return '%s[%d]%s@%s' % (self.container, self.lo, s, self.loc)


def can_use(flags):
  """can_use_from_flags: bitmask of the ord%4 positions a byte with these flags may take."""
  return sum(1 << i for i in range(4) if not (flags & ALIGN_FAIL[i]))


_BITS = tuple(tuple(i for i in range(4) if mask >> i & 1) for mask in range(16))


class _Grp:
  """IXBar::grp_use (a ternary group, or a midbyte as mid_byte_use with the single position 0)."""
  __slots__ = ('group', 'found', 'free', 'attempted', 'range_index')

  def __init__(self, g):
    self.group, self.found, self.free = g, set(), [set(), set(), set(), set()]
    self.attempted, self.range_index = False, -1

  def free_of(self, cu):
    out = set()
    for i in _BITS[cu]:
      out |= self.free[i]
    return out

  def metrics(self, full, half, mx):
    """The numbers is_better_group compares, for this group alone (they never depend on the other group): computed
    once per selection instead of once per comparison. full / half are the (can-use mask, count) requirements with
    one / two usable positions, mx the largest full count."""
    total = len(self.found) + len(self.max_free())
    a32 = sum(min(len(self.free_of(k)), n) for k, n in full)
    lk = self.clone()
    for k, n in full:
      idx = _BITS[k][0]
      for i in range(len(self.free_of(k))):
        if not lk.free[idx]:
          break
        if i == n:
          break
        lk.free[idx].discard(min(lk.free[idx]))
    a16 = sum(min(len(self.free_of(k)), n) for k, n in half)
    a16l = sum(min(len(lk.free_of(k)), n) for k, n in half)
    a32m = sum(min(len(self.free_of(k)), n) for k, n in full if n == mx)
    return (self.attempted, total, len(self.found), a32, a16l, a32m, a16, self.group)

  def max_free(self):
    return self.free[0] | self.free[1] | self.free[2] | self.free[3]

  def range_set(self):
    return self.range_index != -1

  def clone(self):
    c = _Grp(self.group)
    c.found, c.free = set(self.found), [set(s) for s in self.free]
    c.attempted, c.range_index = self.attempted, self.range_index
    return c

  def is_better(self, b, prefer_found, required, reqs):
    """grp_use::is_better_group: True if self (a) beats the currently selected b."""
    full, half, mx = _split_reqs(reqs)
    return _is_better(self.metrics(full, half, mx), b.metrics(full, half, mx), prefer_found, required)


def _split_reqs(reqs):
  """(full, half, mx) of is_better_group: the requirements whose bytes have one usable position, two, and the largest
  count among the first."""
  full = [(k, n) for k, n in sorted(reqs.items()) if bin(k).count('1') == 1]
  half = [(k, n) for k, n in sorted(reqs.items()) if bin(k).count('1') == 2]
  return full, half, max([n for _, n in full], default=0)


def _is_better(a, b, prefer_found, required):
  """grp_use::is_better_group on two groups' metrics (_Grp.metrics): attempted, total available, found, 32-bit
  aligned fit, 16-bit fit after the 32-bit bytes are locked, fit of the largest 32-bit requirement, 16-bit fit,
  total available, then the lower group number."""
  a_att, a_tot, a_found, a32, a16l, a32m, a16, a_grp = a
  b_att, b_tot, b_found, b32, b16l, b32m, b16, b_grp = b
  if not a_att and b_att:
    return True
  if a_att and not b_att:
    return False
  ac, bc = a_tot > required, b_tot > required
  if ac and not bc:
    return True
  if bc and not ac:
    return False
  if prefer_found:
    t = a_found - b_found
    if t:
      return t > 0
    t = a_tot - b_tot
    if t:
      return t < 0
  for t in (a32 - b32, a16l - b16l, a32m - b32m, a16 - b16, a_tot - b_tot):
    if t:
      return t > 0
  return a_grp < b_grp


def _sort_key(tie):
  """setup_byte_vectors: range bytes first, then range_index, then alignment flags DESCENDING, then Byte::operator<.
  tie != 'index' inserts a whole-vs-half-byte order before operator< (see the module docstring)."""
  if tie == 'whole_first':
    return lambda b: (0 if b.is_range else 1, b.range_index, -b.flags, 1 if b.only_one_nibble() else 0, b.lt_key())
  if tie == 'nibble_first':
    return lambda b: (0 if b.is_range else 1, b.range_index, -b.flags, 0 if b.only_one_nibble() else 1, b.lt_key())
  if tie == 'index':
    return lambda b: (0 if b.is_range else 1, b.range_index, -b.flags, b.lt_key())
  raise ValueError('tie must be one of %s, got %r' % (TIES, tie))


class StageXbar:
  """The ternary crossbar of one stage: ternary_use, byte_group_use and the ternary field multimap."""

  def __init__(self, tie='whole_first'):
    self.sort_key = _sort_key(tie)
    self.use = [[None] * BPG for _ in range(TERNARY_GROUPS)]
    self.bgu = [None] * BYTE_GROUPS
    self.fields = defaultdict(list)          # container -> [(group, slot)], slot 5 = the midbyte at (2m, 5)

  def clone(self):
    """An independent copy of the stage as it stands (find_alloc never changes it; commit does)."""
    other = StageXbar.__new__(StageXbar)
    other.sort_key = self.sort_key
    other.use = [list(row) for row in self.use]
    other.bgu = list(self.bgu)
    other.fields = defaultdict(list, {k: list(v) for k, v in self.fields.items()})
    return other

  def commit(self, placed):
    """Record a table's newly placed bytes (IXBar::update / fill_out_use)."""
    for b in placed:
      g, s = b.loc
      if s == BPG:
        if self.bgu[g // 2] == b.ident():
          continue
        assert self.bgu[g // 2] is None, ('midbyte taken', b)
        self.bgu[g // 2] = b.ident()
      else:
        if self.use[g][s] == b.ident():
          continue
        assert self.use[g][s] is None, ('slot taken', b)
        self.use[g][s] = b.ident()
      self.fields[b.container].append((g, s))

  def _calc_found(self, unalloced, order, mids):
    for g in order:
      g.found.clear()
    for m in mids:
      m.found.clear()
    for need in unalloced:
      for (pg, pb) in self.fields.get(need.container, ()):
        if pb == BPG:
          if need.is_range:
            continue
          u = self.bgu[pg // 2]
          if u is not None and u[1] == need.lo:
            mids[pg // 2].found.add(0)
          continue
        u = self.use[pg][pb]
        if u is not None and u[1] == need.lo:
          if need.is_range and order[pg].range_set() and order[pg].range_index != need.range_index:
            continue
          order[pg].found.add(pb)

  def _calc_free(self, order, mids):
    for g in range(TERNARY_GROUPS):
      for s in range(BPG):
        if self.use[g][s] is None:
          order[g].free[ternary_ord(g, s) % 4].add(s)
    for m in range(BYTE_GROUPS):
      if self.bgu[m] is None:
        mids[m].free[ternary_ord(2 * m, BPG) % 4].add(0)

  @staticmethod
  def _reqs(unalloced, skip):
    reqs = {}
    for b in unalloced:
      if skip(b):
        continue
      k = can_use(b.flags)
      reqs[k] = reqs.get(k, 0) + 1
    return reqs

  @staticmethod
  def _select(cands, prefer_found, required, reqs):
    full, half, mx = _split_reqs(reqs)
    sel = sel_m = None
    for g in cands:
      m = g.metrics(full, half, mx)
      if sel is None or _is_better(m, sel_m, prefer_found, required):
        sel, sel_m = g, m
    return sel

  def _found_bytes(self, grp, unalloced, st):
    nfound = len(grp.found)
    i = 0
    while i < len(unalloced):
      need = unalloced[i]
      if nfound == 0 or st['placed'] >= BPG:
        break
      if need.is_range and grp.range_set() and need.range_index != grp.range_index:
        i += 1
        continue
      took = False
      for (pg, pb) in self.fields.get(need.container, ()):
        if pb == BPG:
          continue
        u = self.use[pg][pb]
        if grp.group == pg and u is not None and u[1] == need.lo:
          if pb not in grp.found:
            continue
          grp.found.discard(pb)
          need.loc = (pg, pb)
          if need.is_range:
            grp.range_index = need.range_index
          unalloced.pop(i)
          nfound -= 1
          st['placed'] += 1
          took = True
          break
      if not took:
        i += 1

  def _found_mid(self, mb, unalloced, st, only_nibble):
    if 0 not in mb.found:
      return
    nfound = 1
    i = 0
    while i < len(unalloced):
      need = unalloced[i]
      if nfound == 0 or st['placed'] >= 1:
        break
      took = False
      for (pg, pb) in self.fields.get(need.container, ()):
        if pb != BPG:
          continue
        if only_nibble and not need.only_one_nibble():
          continue
        u = self.bgu[pg // 2]
        if mb.group == pg // 2 and u is not None and u[1] == need.lo:
          need.loc = (pg, pb)
          unalloced.pop(i)
          nfound -= 1
          st['placed'] += 1
          mb.found.discard(0)
          if need.only_one_nibble():
            st['prefer_nibble'] = False
          took = True
          break
      if not took:
        i += 1

  def _free_bytes(self, grp, unalloced, alloced, st):
    nfree = len(grp.max_free())
    align = ((grp.group // 2) * _PAIR_ORDINALS + (grp.group % 2) * (BPG + 1)) & 3
    i = 0
    while i < len(unalloced):
      if nfree == 0 or st['placed'] >= BPG:
        break
      need = unalloced[i]
      if need.is_range and grp.range_set() and need.range_index != grp.range_index:
        i += 1
        continue
      took = False
      for s in sorted(grp.max_free()):
        if ALIGN_FAIL[(s + align) & 3] & need.flags:
          continue
        need.loc = (grp.group, s)
        alloced.append(need)
        unalloced.pop(i)
        nfree -= 1
        st['placed'] += 1
        if need.is_range:
          grp.range_index = need.range_index
        grp.free[ternary_ord(grp.group, s) % 4].discard(s)
        took = True
        break
      if not took:
        i += 1

  def _free_mid(self, mb, unalloced, alloced, st, only_nibble):
    if 0 not in mb.max_free():
      return
    nfree = 1
    align = (mb.group * _PAIR_ORDINALS + BPG) & 3
    o4 = ternary_ord(2 * mb.group, BPG) % 4

    def place(i, need):
      need.loc = (2 * mb.group, BPG)
      alloced.append(need)
      unalloced.pop(i)
      st['placed'] += 1
      mb.free[o4].discard(0)

    if st['prefer_nibble']:
      i = 0
      while i < len(unalloced):
        if nfree == 0 or st['placed'] >= 1:
          break
        need = unalloced[i]
        if need.is_range or not need.only_one_nibble() or (ALIGN_FAIL[align] & need.flags):
          i += 1
          continue
        place(i, need)
        nfree -= 1
        st['prefer_nibble'] = False
    if only_nibble:
      return
    i = 0
    while i < len(unalloced):
      if nfree == 0 or st['placed'] >= 1:
        break
      need = unalloced[i]
      if need.is_range or (ALIGN_FAIL[align] & need.flags):
        i += 1
        continue
      place(i, need)
      nfree -= 1
      if need.only_one_nibble():
        st['prefer_nibble'] = False

  def find_alloc(self, table_bytes):
    """IXBar::find_alloc, ternary: allocate one table's key bytes on top of the stage as it stands, without
    recording them. Returns the bytes that take a new slot (each byte of the table gets .loc, including those that
    reuse a slot an earlier table holds), or None when the key does not fit in 12 groups."""
    total = len(table_bytes)
    if total == 0:
      return []
    groups, nibbles = 0, 0
    while True:
      if groups == 0:                        # calculate_sizes
        groups, nibbles = 1, 0
        while groups * BPG + (nibbles + 1) // 2 < total:
          if groups > nibbles:
            nibbles += 1
          else:
            groups += 1
      elif groups > nibbles:                 # increase_ternary_ixbar_space
        nibbles += 1
      else:
        groups += 1
      if groups > TERNARY_GROUPS:
        return None
      for prefer_found in (True, False):
        order = [_Grp(g) for g in range(TERNARY_GROUPS)]
        mids = [_Grp(m) for m in range(BYTE_GROUPS)]
        for b in table_bytes:
          b.loc = None
        unalloced = sorted(table_bytes, key=self.sort_key)
        alloced = []
        self._calc_found(unalloced, order, mids)
        self._calc_free(order, mids)
        st = {'placed': 0, 'prefer_nibble': nibbles % 2 == 1}
        to_alloc = total
        need_mid = nibbles // 2 + nibbles % 2         # allocate_mid_bytes
        for bus in range(need_mid):
          if not unalloced:
            break
          only_nibble = st['prefer_nibble'] and bus == need_mid - 1
          reqs = self._reqs(unalloced, lambda b: b.is_range or (only_nibble and not b.only_one_nibble()))
          sel = self._select(mids, prefer_found, 1, reqs)
          st['placed'] = 0
          self._found_mid(sel, unalloced, st, only_nibble)
          self._free_mid(sel, unalloced, alloced, st, only_nibble)
          sel.attempted = True
          to_alloc -= st['placed']
          self._calc_found(unalloced, order, mids)
        for bus in range(groups):                    # allocate_groups
          if not unalloced:
            break
          leeway = max((groups - bus) * BPG - to_alloc, 0)
          required = max(BPG - leeway, 0)
          reqs = self._reqs(unalloced, lambda b: False)
          sel = self._select(order, prefer_found, required, reqs)
          st['placed'] = 0
          self._found_bytes(sel, unalloced, st)
          self._free_bytes(sel, unalloced, alloced, st)
          sel.attempted = True
          to_alloc -= st['placed']
          self._calc_found(unalloced, order, mids)
        if not unalloced:
          return alloced

  def allocate(self, table_bytes):
    """find_alloc, then record it. True if the table fit."""
    alloced = self.find_alloc(table_bytes)
    if alloced is None:
      return False
    self.commit(alloced)
    return True


def table_shape(table_bytes):
  """(sorted groups touched, {midbyte: 'whole' | 'half'}) of an allocated table."""
  groups, mids = set(), {}
  for b in table_bytes:
    g, s = b.loc
    if s == BPG:
      mids[g // 2] = 'half' if b.only_one_nibble() else 'whole'
    else:
      groups.add(g)
  return tuple(sorted(groups)), mids


def blocks_per_row(table_bytes):
  """TCAM blocks of one 512-row slice of an allocated table: max(groups touched, 2 * whole midbytes + half
  midbytes + 1). Held on 998 of 998 compiled tree tables (spec 2026-11-12 F3)."""
  groups, mids = table_shape(table_bytes)
  whole = sum(1 for v in mids.values() if v == 'whole')
  return max(len(groups), 2 * whole + (len(mids) - whole) + 1)


# ---- the model's keys -----------------------------------------------------------------------------------------

def tree_key(key, table_idx, numbering=None):
  """A packed tree key, as packed.key_bytes tuples (kind 'W' | 'H' | 'B', byte index k in its container,
  nibble_only, container identity) in key order, -> Bytes.

  Container identity (design review P14, decision 4.4). Codes of <= 8 bits share bytes, so two trees can key the
  same container byte (each its own bits of it); the crossbar moves whole bytes, so both keys hold the byte, and in
  a stage the later table finds the earlier one's slot (found path) as p4c does. With `numbering` (one dict shared by
  a stage's keys, as field_key's) a container is named by its identity, so equal identities name one container
  byte; without it (a key priced alone) by the table's own numbers 1000 * (table_idx + 1) + i. The container SORT
  order is the container's first appearance in this key either way (Byte `order`), which is the key's own order,
  so a key that shares nothing allocates exactly as before 4.4.

  A 3-tuple (no identity: a hand-made key) gets the pre-4.4 inference: a new container starts whenever the kind
  changes, k does not follow the previous byte's k + 1, or the kind is 'B'; such a key shares no byte."""
  out, count, prev, seen = [], {'W': 0, 'H': 0, 'B': 0}, None, {}
  base = 1000 * (table_idx + 1)
  for item in key:
    kind, k, nibble_only = item[:3]
    shared = len(item) > 3
    if shared:
      ident = item[3]
    elif prev is None or prev[0] != kind or k != prev[1] + 1 or kind == 'B':
      ident = ('<own>', len(seen))
    else:
      ident = prev[2]
    if ident not in seen:
      if numbering is not None and shared:
        name = '%s%d' % (kind, 500000 + numbering.setdefault(ident, len(numbering)))
      else:
        name = '%s%d' % (kind, base + count[kind])
        count[kind] += 1
      seen[ident] = (len(seen), name)
    order, name = seen[ident]
    out.append(Byte(name, 8 * k, 0x0F if nibble_only else 0xFF, order=(order,)))
    prev = (kind, k, ident)
  return out


def range_key(index, nbytes):
  """A range table's key: one value field in a pinned 16-bit container (build_p4_script's @pa_container_size on
  every *_val field), every byte matched on both nibbles, so RANGE_LO and RANGE_HI bytes, range index 0. Container
  number 900000 + index: never one of tree_key's."""
  return [Byte('H%d' % (900000 + index), 8 * j, 0xFF, spec=spec, range_index=0)
          for j in range(nbytes) for spec in (RANGE_LO, RANGE_HI)]


_KINDS = {8: 'B', 16: 'H', 32: 'W'}


def fill_low(bits, sizes):
  """A `bits`-bit field pinned to containers of `sizes` bits (low container first, @pa_container_size's order) as
  (kind, used bits) slices: p4c fills the low container first (lanes.layout's fill-low rule, 7,479 of 7,479 fields),
  so every container is full except the last one used. A container left with no bits is dropped."""
  out, left = [], int(bits)
  for size in sizes:
    if left <= 0:
      break
    used = min(int(size), left)
    out.append((_KINDS[int(size)], used))
    left -= used
  if left > 0:
    raise ValueError('a %d-bit field does not fit containers %r' % (bits, tuple(sizes)))
  return tuple(out)


def code_container_order(field, width, slice_index):
  """The place of one code-field container in a stage's container numbering: WIDEST field first, ties by field
  name, then slice index (low slice first). Training and the replay order every code-field container by this one
  function (field_key), with the field named as the program names it (code_<resolved name>; a key's position for
  a key priced alone, lanes.key_fields, where equal widths are interchangeable).

  An ordering ASSUMPTION about p4c's PHV container numbers (design-review inventory P12), not a port of p4c's PHV
  allocator: the numbers only order key bytes p4c's crossbar sort leaves tied. Evidence: with p4c's own stage
  contents, widest-first numbering prices 3,598 / 3,598 tree tables of the model6 compiles as p4c did, first-use
  numbering 3,580 / 3,598 (results/encoding_evidence_2026_11/model7/findings.md Part 4, "Container numbering does
  not matter for blocks"); and on joint-task_M055_s20_k12 (decision 2.3 key pruning, stage 8: five ddos trees with
  overlapping but different keys) first use priced get_classification_tree_ddos_0 at 3 blocks, p4c and widest
  first at 4 (model8/impl/scripts/t5_xbar_diag.py; p4c numbered those fields widest first, H2..H12)."""
  return (-int(width), str(field), int(slice_index))


def field_key(fields, numbering):
  """A key of whole code FIELDS -- every encoding but per-tree -- as Bytes. `fields` is a sequence of
  (field_id, slices), slices being (kind, used bits) per container, low first (fill_low, lanes.layout). Each field
  starts at bit 0 of its own containers (declared at their full width, decision 4.3), so a container's byte k carries bits 8k.. and only the top
  byte can be part-used: at most 4 live bits make it a one-nibble byte. `numbering` maps (field_id, slice index) to
  a container name and is shared by every key of a stage, so two tables keying the same field name the same
  container bytes and p4c's found-byte path shares their slots. The containers are ORDERED by
  code_container_order (widest field first), which only orders bytes p4c's sort leaves tied (the module
  docstring's tie rule)."""
  out = []
  for field_id, slices in fields:
    width = sum(int(used) for _, used in slices)
    for index, (kind, used) in enumerate(slices):
      ident = (field_id, index)
      if ident not in numbering:
        numbering[ident] = len(numbering)
      name = '%s%d' % (kind, 2000 + numbering[ident])
      order = code_container_order(field_id, width, index)
      for k in range(-(-used // 8)):
        live = min(8, used - 8 * k)
        out.append(Byte(name, 8 * k, 0x0F if live <= 4 else 0xFF, order=order))
  return out


def alone_blocks(fields):
  """TCAM blocks per 512-row word of a key of whole fields (field_key's `fields`) alone in an empty stage, as p4c
  allocates it, or None when it does not fit 12 groups."""
  key = field_key(fields, {})
  if not StageXbar().allocate(key):
    return None
  return blocks_per_row(key)
