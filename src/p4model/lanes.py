"""Crossbar LANES: where a ternary key's bytes can physically sit, and what a key
costs when it shares a stage with a key placed before it.

tables.codeword_to_blocks (the LADDER) prices a key by its byte COUNT alone, as if it had the
stage's whole ternary crossbar to itself. That is exact for the first key p4c
places in a stage (it agrees with standalone() below on 74/74 real design
keys), but a LATER key in the same stage gets only what the earlier keys left,
and whether those leftovers can hold it depends on WHICH slots are free, not how
many. This module models that, in three layers:

  1. Layout (`layout`, `relaxed_layout`, `key_layout`, `key_bytes`, `container_sizes`): each `code_*` field's
     PHV container layout, predicted from its bit width alone -- the model runs
     before any compile, so it has no PHV log. Audit Sec 7.4.
  2. Lane price (`standalone`, `price_with_supply`): the smallest block count
     whose crossbar slots can legally hold the key's bytes, by counting.
     Audit Sec 7.2 and Appendix A.
  3. Stage simulation (`stage_prices`): keys priced one at a time in placement
     order, each later key into the leftovers of the ones before. Audit Sec
     7.2 / 7.5, the "fullest-lane midbytes, lane-aware fill" row.

THE LANE RULE (audit Appendix A.3). Every crossbar byte slot has a lane
(`slot index mod 4`). Byte k of a 32-bit (W) container may sit only in a
lane-k slot; a 16-bit (H) container's byte k only in a slot of lane parity
`k mod 2`; an 8-bit (B) container's byte anywhere. A stage's crossbar has 12
groups of 5 private slots (`GROUP_START` gives each group's first slot, hence
its lanes) and 6 midbytes (`MID_LANES` gives each one's lane). A key of `g`
blocks takes up to `g` groups, and the midbyte nibbles pair blocks: a key may
use `w` whole midbytes and at most one extra nibble-only "tail" midbyte `t`
when `2w + t + 1 <= g` -- the `+1` is the mandatory 2-bit version field's
nibble. A later key's blocks are `max(groups used, 2w + t + 1)`: data nibbles
beyond its groups become version-only blocks (dsp41: 4 groups + 3 whole
midbytes -> 7, as p4c charged).

A BYTE LIST is this module's key representation: one `(kind, k, nibble_only)`
tuple per crossbar byte, `kind` in {'W', 'H', 'B'} (the container size), `k`
the byte's index within its container slice, `nibble_only` True when the byte
carries at most 4 live bits (it may then sit in a single midbyte nibble).
`key_bytes(field_bit_widths)` builds one.

"No price" is `None` throughout (the audit's scratch code used 99): a key with
no legal fit in the 12 groups/6 midbytes available.

Ported from the audit's scratch scripts, `reviews/model_audit_scratch/`:
`width_layout.py` (layer 1), `fastlane.py` (layer 2), `stage_sim_fast.py`
(layer 3, with its module-global modes fixed to FIRST_MID='fullest',
FILL='lane' -- the mode the audit's Sec 7.5 selected and `proto_model.py` runs
end to end, 42/43 stage_depth and 38/38 blocks on the pragma'd arm-D compiles).
The width layout rule is p4c-version-specific: re-run the audit's
`phv_layout_survey.py` after a compiler upgrade."""
import collections
import functools
import itertools
import math

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.tables import (
    codeword_bits_to_blocks,
    codeword_fields_to_bytes_from_bits,
    codeword_to_blocks,
)
from src.p4model.target import (
    TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE,
    TERNARY_CROSSBAR_GROUPS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
)

_GROUPS = TERNARY_CROSSBAR_GROUPS_PER_STAGE       # 12
_MIDBYTES = TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE  # 6
_SLOTS_PER_GROUP = 5
_LANES = 4

# Lane of each of the stage's 6 midbytes (p4c's slot numbering: pair i of
# groups is private-x5 | private-x5 | midbyte, 11 slots per pair).
MID_LANES = [1, 0, 3, 2, 1, 0]
MID_PER_LANE = collections.Counter(MID_LANES)

# First crossbar slot of each of the 12 groups: pair g // 2 starts at slot
# 11 * (g // 2), the odd group of the pair 6 slots in. Slot j of group g sits
# in lane (GROUP_START[g] + j) % 4.
GROUP_START = [11 * (g // 2) + (0 if g % 2 == 0 else 6) for g in range(_GROUPS)]


# --------------------------------------------------------------------------
# Layer 1: PHV layout from widths (width_layout.py; audit Sec 7.4)
# --------------------------------------------------------------------------

def layout(w):
  """Predicted PHV container slices of one `w`-bit code field, as a list of
  `(kind, bits)` pairs, kind in {'B', 'H', 'W'} (8/16/32-bit container). Read
  off 1130 real `code_*` fields (90.8% exact):

    w <= 8 -> B, 9..16 -> H, 17..32 -> W (the whole field, from container
    bit 0); w > 32 -> a top W slice of 25..32 bits
    (top = 24 + ((w - 1) % 8) + 1), and the low `w - top` bits (a multiple of
    8) as W32 slices plus one B8 / H16 / W24 for the remainder.

  PINS FIX SIZES, p4c FILLS LOW FIRST (2026-10-04, measured 7,479/7,479 real
  fields over 32 bits): when the remainder is 24 bits the low slice is NOT a
  W24 -- the pinned containers are all W32, p4c fills them from the low one,
  and only the top container is partly used (51 -> W32 | W19). Container
  sizes, and so @pa_container_size, are the same either way.

  Every slice starts at container bit 0. A non-positive width has no slices."""
  if w <= 0:
    return []
  if w <= 8:
    return [('B', w)]
  if w <= 16:
    return [('H', w)]
  if w <= 32:
    return [('W', w)]
  top = 24 + ((w - 1) % 8) + 1
  low = w - top
  rem = low % 32
  if rem == 24:
    # The 24-remainder class (49-56, 81-88, 113-120 bits): its containers are
    # all W32 and pinned by @pa_container_size, and p4c fills the LOW
    # container first, so every slice is a full 32 bits except the top one,
    # which takes the remainder (51 bits -> W32 | W19, not W24 | W27).
    n = low // 32 + 2
    return [('W', 32)] * (n - 1) + [('W', w - 32 * (n - 1))]
  out = []
  if rem == 8:
    out.append(('B', 8))
  elif rem == 16:
    out.append(('H', 16))
  out += [('W', 32)] * (low // 32)
  out.append(('W', top))
  return out


def relaxed_layout(w):
  """The layout p4c's PHV allocator falls back to when whole-W fields would
  pile onto the same lanes (seen in M50_k8_s13 / M150_k7_s11): a 17..32-bit
  field goes into an H16 plus a B (<= 8 bits left) or H remainder. Every other
  width is laid out as `layout` does."""
  if 17 <= w <= 32:
    r = w - 16
    return [('H', 16), ('B', r) if r <= 8 else ('H', r)]
  return layout(w)


def bytes_from_layout(field_bit_widths, lay=layout):
  """The byte list (see module docstring) of a key whose fields are laid out
  by `lay` -- one entry per started byte of every slice."""
  out = []
  for w in field_bit_widths:
    for kind, s in lay(w):
      for k in range(math.ceil(s / 8)):
        used = min(8, s - 8 * k)
        out.append((kind, k, used <= 4))
  return out


def bytes_from_widths(field_bit_widths):
  """The byte list under the plain width rule (`layout`), no fallback."""
  return bytes_from_layout(field_bit_widths, layout)


def _price_le(a, b):
  """`a <= b` for lane prices, `None` (no fit) ranking above every number."""
  if b is None:
    return True
  if a is None:
    return False
  return a <= b


def key_layout(field_bit_widths):
  """The layout function -- `layout` or `relaxed_layout` -- this model prices
  a key with: THE LAYOUT FALLBACK RULE of audit Sec 7.4.

  Use the width-rule layout (`layout`), unless its standalone lane price is
  ABOVE the LADDER `tables.codeword_to_blocks` (deliberately not `table_blocks`:
  comparing against the ladder keeps the pins identical to campaign_2026_10's
  and avoids recursion) -- then
  p4c's PHV allocator is known to relax the layout (H containers instead of
  whole W ones), so use the relaxed layout (`relaxed_layout`) if it prices no
  worse than the width-rule one. With this rule the lane price from widths
  equals the lane price from the REAL PHV layout on 148/148 real keys
  (audit `layout_rule_check.py`).

  Returned as a function, not a byte list, because the generator needs it
  too: it pins every tree-key code_* field to exactly this layout with
  @pa_container_size (build_p4_script.code_field_container_sizes), so p4c
  can no longer pick a different split under PHV pressure (spec 2026-09-29
  Sec 1.2). Pricing (key_bytes) and pinning share this one decision so the
  two cannot drift."""
  field_bit_widths = list(field_bit_widths)
  prod = codeword_to_blocks(tuple(sorted(field_bit_widths)))
  plain_price = standalone(bytes_from_widths(field_bit_widths))
  if _price_le(plain_price, prod):
    return layout
  relaxed = bytes_from_layout(field_bit_widths, relaxed_layout)
  return relaxed_layout if _price_le(standalone(relaxed), plain_price) else layout


def key_bytes(field_bit_widths):
  """The byte list this model uses for a key: its fields laid out by
  `key_layout`'s choice."""
  field_bit_widths = list(field_bit_widths)
  return bytes_from_layout(field_bit_widths, key_layout(field_bit_widths))


def table_blocks(field_bit_widths):
  """THE production per-table price (2026-10-04): TCAM blocks one 512-row
  word of a classification table with this key costs, as the lane model
  prices the key's pinned PHV layout alone in a stage --
  `standalone(key_bytes(w))`.

  Replaces the byte-count ladder `tables.codeword_to_blocks` as the price
  every table is CHARGED (packing, p4_replay, ternary_matching_resource_usage).
  The ladder is kept for threshold alignment (align_budget) and as
  key_layout's reference, which keeps the pins identical and avoids recursion
  (table_blocks -> key_layout -> table_blocks).

  Evidence (reviews/campaign_2026_10_lane_findings.md Sec 3-5): on
  campaign_2026_10 the lane price is below the ladder on 133 tables and p4c
  agrees on all 133. One accepted miss: independent_low_sd5's (27, 52) key,
  lane 2 vs p4c 3 (p4c's greedy midbyte choice leaves no nibble for the
  version bits).

  Order-insensitive. The empty key returns the ladder's floor. Raises
  CrossbarKeyTooWide(message, byte_width) when the key exceeds the stage's
  TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes (checked before enumerating --
  66-72-byte keys would otherwise pay the full 1..12-block search) or when no
  lane-legal fit exists within the crossbar."""
  return _table_blocks(tuple(sorted(int(w) for w in field_bit_widths)))


@functools.lru_cache(maxsize=65536)
def _table_blocks(widths):
  byte_width = codeword_fields_to_bytes_from_bits(widths)
  if byte_width == 0:
    return codeword_bits_to_blocks(0)
  if byte_width > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), byte_width)
  price = standalone(key_bytes(widths))
  if price is None:
    raise CrossbarKeyTooWide(
        "table key of %d crossbar bytes has no lane-legal layout in one stage's "
        "crossbar" % byte_width, byte_width)
  return price


# The PHV container size each slice kind names, in the unit
# @pa_container_size takes.
CONTAINER_BITS = {'B': 8, 'H': 16, 'W': 32}


def container_sizes(w, lay=layout):
  """The @pa_container_size argument list for one `w`-bit field laid out by
  `lay`: one container size per slice, LOW SLICE FIRST (p4c honours `16, 8`
  as bits [15:0] in an H and the rest in a B -- code_bwd_iat_min[15:0] ->
  H142, [17:16] -> B77, spec 2026-09-29 Sec 3.1). Empty for a field with no
  bits, which must then get no pragma at all."""
  return [CONTAINER_BITS[kind] for kind, _ in lay(w)]


# --------------------------------------------------------------------------
# Layer 2: lane feasibility by counting (fastlane.py)
# --------------------------------------------------------------------------
#
# Exact for this structure, no bipartite matching needed: a byte multiset fits
# a per-lane slot supply s[0..3] iff
#     W[l] <= s[l] for every lane l,
#     H_even <= s[0] + s[2] - W[0] - W[2],  H_odd <= s[1] + s[3] - W[1] - W[3],
#     total <= sum(s)
# (Hall's condition on a laminar family: lanes nest inside parities inside
# "any").

def _counts(byte_list):
  """(W per lane, H per parity, B) byte counts of a byte list."""
  W = [0] * _LANES
  H = [0, 0]
  B = 0
  for kind, k, _ in byte_list:
    if kind == 'W':
      W[k] += 1
    elif kind == 'H':
      H[k % 2] += 1
    else:
      B += 1
  return tuple(W), tuple(H), B


def _fits(c, s):
  """Whether counts `c` fit the per-lane slot supply `s` (Hall, above)."""
  W, H, B = c
  if any(W[l] > s[l] for l in range(_LANES)):
    return False
  if H[0] > s[0] + s[2] - W[0] - W[2] or H[1] > s[1] + s[3] - W[1] - W[3]:
    return False
  return sum(W) + sum(H) + B <= sum(s)


def _remove_one(c, kind, lane_or_par):
  """Counts `c` minus one byte of `kind` at lane (W) / parity (H)."""
  W, H, B = list(c[0]), list(c[1]), c[2]
  if kind == 'W':
    W[lane_or_par] -= 1
  elif kind == 'H':
    H[lane_or_par] -= 1
  else:
    B -= 1
  return tuple(W), tuple(H), B


def _nibble_types(byte_list):
  """The (kind, lane-or-parity) classes of the key's nibble-only bytes -- the
  candidates for a tail nibble. B bytes are lane-free, recorded as ('B', 0)."""
  return frozenset(
      (kind, (k if kind == 'W' else (k % 2 if kind == 'H' else 0)))
      for kind, k, nibble_only in byte_list if nibble_only)


def _tail_allowed(kind, lane_or_par, lane):
  """Whether a nibble-only byte of this class may sit in a lane-`lane` slot."""
  if kind == 'W':
    return lane_or_par == lane
  if kind == 'H':
    return lane_or_par == lane % 2
  return True


@functools.lru_cache(maxsize=None)
def _standalone(W, H, B, nib_types):
  c = (W, H, B)
  n = sum(W) + sum(H) + B
  for g in range(1, _GROUPS + 1):
    if _SLOTS_PER_GROUP * g + (g - 1) // 2 + 1 < n:
      continue
    # Each group supplies one slot per lane plus one extra in a lane of its
    # choosing (the lane it starts on).
    for extras in itertools.combinations_with_replacement(range(_LANES), g):
      s0 = [g + extras.count(l) for l in range(_LANES)]
      for w in range(0, (g - 1) // 2 + 1):
        for whole in itertools.combinations_with_replacement(range(_LANES), w):
          wc = collections.Counter(whole)
          if any(wc[l] > MID_PER_LANE[l] for l in wc):
            continue
          s = [s0[l] + wc[l] for l in range(_LANES)]
          if _fits(c, s):
            return g
          if 2 * w + 2 <= g:  # try one tail nibble too
            for lane in range(_LANES):
              if wc[lane] + 1 > MID_PER_LANE[lane]:
                continue
              for kind, lp in nib_types:
                if not _tail_allowed(kind, lp, lane):
                  continue
                c2 = _remove_one(c, kind, lp)
                if min(c2[0]) < 0 or min(c2[1]) < 0 or c2[2] < 0:
                  continue
                if _fits(c2, s):
                  return g
  return None


def standalone(byte_list):
  """A key's lane price ALONE in a stage: the smallest block count `g` whose
  `g` groups (each one slot per lane plus one extra, lane chosen), `w` whole
  midbytes and at most one nibble-only tail midbyte `t`, with
  `2w + t + 1 <= g`, legally hold every byte. `None` if no `g <= 12` does.

  Agrees with p4c on 57/57 probe keys (from the real PHV layout) and with
  `tables.codeword_to_blocks` on 74/74 real design keys (from `key_bytes`)."""
  W, H, B = _counts(byte_list)
  return _standalone(W, H, B, _nibble_types(byte_list))


def price_with_supply(byte_list, free_groups, free_mids):
  """A key's LEFTOVER price: the cheapest lane-legal fit into a partly used
  stage. (The scratch `fastlane.py` reserved this name for a generic
  supply-driven price but left it unimplemented; the one supply-driven price
  the audit validated is `stage_sim_fast._later_price`, ported here.)

  free_groups : {group index: set of free slot positions 0..4} -- a partly
                used group may be shared; groups absent or with an empty set
                are unavailable.
  free_mids   : set of free midbyte indices 0..5.

  Any subset of groups with free slots may be used; whole bytes go in the
  LOWEST-index free midbytes, at most one tail nibble in the next one;
  blocks = max(groups used, 2w + t + 1). Returns the minimum such block count,
  or `None` if the key cannot fit. "Lowest free midbyte" reproduces p4c on
  133/133 keys in crowded stages given where the other keys actually landed
  (audit Sec 7.2)."""
  c = _counts(byte_list)
  n = len(byte_list)
  nib_types = _nibble_types(byte_list)
  cands = [g for g in range(_GROUPS) if free_groups.get(g)]
  lanes = {g: collections.Counter((GROUP_START[g] + j) % _LANES for j in free_groups[g])
           for g in cands}
  mids = sorted(free_mids)
  best = None
  for gcount in range(1, len(cands) + 1):
    if best is not None and gcount >= best:
      break
    seen = set()
    for S in itertools.combinations(cands, gcount):
      sig = tuple(sorted(tuple(sorted(lanes[x].items())) for x in S))
      if sig in seen:
        continue
      seen.add(sig)
      s0 = [sum(lanes[x][l] for x in S) for l in range(_LANES)]
      if sum(s0) + len(mids) < n:
        continue
      for w in range(0, len(mids) + 1):
        s = list(s0)
        for m in mids[:w]:
          s[MID_LANES[m]] += 1
        blocks = max(gcount, 2 * w + 1)
        if (best is None or blocks < best) and _fits(c, s):
          best = blocks
        if w < len(mids):  # plus one tail nibble
          tl = MID_LANES[mids[w]]
          for kind, lp in nib_types:
            if not _tail_allowed(kind, lp, tl):
              continue
            c2 = _remove_one(c, kind, lp)
            blocks = max(gcount, 2 * w + 2)
            if (best is None or blocks < best) and _fits(c2, s):
              best = blocks
  return best


# --------------------------------------------------------------------------
# Layer 3: the stage simulation (stage_sim_fast.py, FIRST_MID='fullest',
# FILL='lane' -- audit Sec 7.5's selected row: probes 334 exact / 0 under /
# 2 over of 336 keys, 0 under on every real design set)
# --------------------------------------------------------------------------

def first_key_occupancy(byte_list, g):
  """What the FIRST key in a stage, priced at `g` blocks, leaves free.

  Midbytes first (p4c allocates them before groups): `(g - 1) // 2` whole
  midbytes, each the free one whose lane has the most unplaced key bytes, ties
  to the lowest index ("fullest-lane", p4c's `allocate_mid_bytes`). Then
  groups 0..g-1, filled lane by lane: each slot takes an unplaced byte that
  may sit in its lane (W of that lane, else H of its parity, else B), and a
  slot no remaining byte can use stays free ("lane-aware fill").

  Returns `(free_groups, free_mids)` in `price_with_supply`'s format."""
  free = {grp: set(range(_SLOTS_PER_GROUP)) for grp in range(_GROUPS)}
  W, H, B = _counts(byte_list)
  W, H, Bc = list(W), list(H), [B]

  def take(lane):
    if W[lane] > 0:
      W[lane] -= 1
      return True
    if H[lane % 2] > 0:
      H[lane % 2] -= 1
      return True
    if Bc[0] > 0:
      Bc[0] -= 1
      return True
    return False

  mids = []
  avail = list(range(_MIDBYTES))
  for _ in range((g - 1) // 2):
    m = max(avail, key=lambda m: (W[MID_LANES[m]] + H[MID_LANES[m] % 2] + Bc[0], -m))
    avail.remove(m)
    mids.append(m)
    take(MID_LANES[m])
  for grp in range(min(g, _GROUPS)):
    for j in range(_SLOTS_PER_GROUP):
      if take((GROUP_START[grp] + j) % _LANES):
        free[grp].discard(j)
  return free, set(range(_MIDBYTES)) - set(mids)


def stage_prices(keys_bytes, order=None):
  """Price every DISTINCT key sharing one stage, p4c-style, one at a time.

  keys_bytes : list of byte lists, one per distinct key (`key_bytes`).
  order      : placement order, a sequence of indices into keys_bytes
               (default: list order). The first placed key pays its
               `standalone` price and occupies its midbytes and groups as
               `first_key_occupancy` says; each later key pays
               `price_with_supply` into what is left, then consumes the first
               `price` groups that still have a free slot (wholly) and the
               lowest `(price - 1) // 2` free midbytes.

  Returns {key index: blocks per 512-row table word, or None}. `None` means
  the key has no legal fit in that stage; once one key has no price, every
  key after it has none either (the scratch simulation's 99 sentinel consumed
  the whole crossbar, with the same effect). A caller should treat any `None`
  as "this stage does not fit"."""
  if order is None:
    order = range(len(keys_bytes))
  free = {g: set(range(_SLOTS_PER_GROUP)) for g in range(_GROUPS)}
  free_mids = set(range(_MIDBYTES))
  prices = {}
  exhausted = False
  for pos, k in enumerate(order):
    kb = keys_bytes[k]
    if exhausted:
      prices[k] = None
      continue
    if pos == 0:
      g = standalone(kb)
      prices[k] = g
      if g is None:
        exhausted = True
        continue
      free, free_mids = first_key_occupancy(kb, g)
      continue
    price = price_with_supply(kb, free, free_mids)
    prices[k] = price
    if price is None:
      exhausted = True
      continue
    free, free_mids = later_key_occupancy(free, free_mids, price)
  return prices


def later_key_occupancy(free_groups, free_mids, price):
  """What a LATER key priced at `price` blocks leaves free: it consumes, whole,
  the first `price` groups that still have a free slot, and the lowest
  `(price - 1) // 2` free midbytes (stage_prices' bookkeeping, split out so
  packing.crossbar_stages_needed can run the same simulation with the first
  key charged its production price). Returns new `(free_groups, free_mids)`;
  the arguments are not modified."""
  free = {grp: set(slots) for grp, slots in free_groups.items()}
  need = price
  for grp in range(_GROUPS):
    if need <= 0:
      break
    if free.get(grp):
      free[grp] = set()
      need -= 1
  return free, set(free_mids) - set(sorted(free_mids)[:max(0, (price - 1) // 2)])
