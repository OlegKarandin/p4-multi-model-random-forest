"""Crossbar LANES and key LAYOUTS: how a ternary key's code fields sit in PHV
containers, and the lane count that chooses that layout.

Since 2026-11-13 the price of a key -- alone (`table_blocks`) and in a shared
stage (packing's stage simulation) -- is p4c's ternary crossbar allocation,
ported (xbar.py). This module keeps what feeds it:

  1. Layout (`layout`, `key_fields`, `container_sizes`): each `code_*` field's
     PHV container layout, predicted from its bit width alone -- the model runs
     before any compile, so it has no PHV log. Audit Sec 7.4.
  2. Lane count (`standalone`): the smallest block count whose crossbar slots
     can legally hold the key's bytes, by counting. Audit Sec 7.2 and Appendix
     A. It is no longer a price, and it no longer chooses the layout: a code
     field's pin depends on its width alone (decision 4.1).
  3. `table_blocks` / `key_fields`: the key's fields in that layout, priced by
     xbar.alone_blocks. (The lane stage simulation -- `price_with_supply`,
     `fit_with_supply`, `first_key_occupancy`, `later_key_occupancy`,
     `stage_prices` -- was deleted 2026-11-13: the port replaced it.)

THE LANE RULE (audit Appendix A.3). Every crossbar byte slot has a lane
(`slot index mod 4`). Byte k of a 32-bit (W) container may sit only in a
lane-k slot; a 16-bit (H) container's byte k only in a slot of lane parity
`k mod 2`; an 8-bit (B) container's byte anywhere. A stage's crossbar has 12
groups of 5 private slots (`GROUP_START` gives each group's first slot, hence
its lanes) and 6 midbytes (`MID_LANES` gives each one's lane). A key of `g`
blocks takes up to `g` groups, and the midbyte nibbles pair blocks: a key may
use `w` whole midbytes and at most one extra nibble-only "tail" midbyte `t`
when `2w + t + 1 <= g` -- the `+1` is the mandatory 2-bit version field's
nibble.

A BYTE LIST is this module's key representation: one `(kind, k, nibble_only)`
tuple per crossbar byte, `kind` in {'W', 'H', 'B'} (the container size), `k`
the byte's index within its container slice, `nibble_only` True when the byte
carries at most 4 live bits (it may then sit in a single midbyte nibble).
`bytes_from_layout(field_bit_widths, layout)` builds the
one this model lays a key out with.

"No price" is `None` throughout (the audit's scratch code used 99): a key with
no legal fit in the 12 groups/6 midbytes available.

Ported from the audit's scratch scripts, `reviews/model_audit_scratch/`:
`width_layout.py` (layer 1) and `fastlane.py` (layer 2).
The width layout rule is p4c-version-specific: re-run the audit's
`phv_layout_survey.py` after a compiler upgrade."""
import collections
import functools
import itertools
import math

from p4model import xbar
from p4model.errors import CrossbarKeyTooWide
from p4model.tables import (
    check_key_bits,
    codeword_bits_to_blocks,
    codeword_fields_to_bytes_from_bits,
    word_blocks,
)
from p4model.target import TERNARY_CROSSBAR_GROUPS_PER_STAGE, TERNARY_CROSSBAR_STAGE_BYTES

_GROUPS = TERNARY_CROSSBAR_GROUPS_PER_STAGE       # 12
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
# How many groups start on each lane: 3 on every lane. A group's start lane is
# the one lane it offers TWO slots in, so a key can have at most this many
# doubled slots in a lane (heldout7 joint-tree_M035_s08_k16 ddos_2, 2026-11-11).
GROUPS_PER_START_LANE = collections.Counter(start % _LANES for start in GROUP_START)


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
    8) as W32 slices plus one B8 / H16 for the remainder (a 24-bit remainder
    is the fill-low case below).

  PINS FIX SIZES, p4c FILLS LOW FIRST (2026-10-04, measured 7,479/7,479 real
  fields over 32 bits): when the remainder is 24 bits the low slice is NOT a
  W24 -- the pinned containers are all W32, p4c fills them from the low one,
  and only the top container is partly used (51 -> W32 | W19). Container
  sizes, and so @pa_container_size, are the same either way PER FIELD, and
  a field's pin is this function's sizes whatever the rest of its key (decision
  4.1: the model once fell back to an H16 + remainder split for some keys, and
  that fallback is gone).

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


def table_blocks(field_bit_widths):
  """THE production per-table price: TCAM blocks one 512-row word of a
  classification table with this key costs alone in a stage, as p4c's ternary
  crossbar allocator (ported, xbar.py) lays out the key's pinned PHV
  containers -- `xbar.alone_blocks(key_fields(w))` (2026-11-13; the lane
  count `standalone(bytes_from_layout(w, layout))` until then).

  2026-11-13, campaign_2026_11 (7,763 distinct tree keys of 15,300 designs):
  the port and the lane count differ on 61 keys; on every one p4c's own
  allocation agrees with the port (the lane count was below p4c on 49 of
  them), model7/out/truth_alone.jsonl. The history below is the lane count's.

  Replaced the byte-count ladder `tables.codeword_to_blocks` as the price
  every table is CHARGED (packing, p4_replay, ternary_matching_resource_usage)
  in 2026-10; threshold alignment prices with it too since Task 32
  (align_budget.port_price), and the ladder is deleted.

  Evidence (reviews/campaign_2026_10_lane_findings.md Sec 3-5): on
  campaign_2026_10 the lane price is below the ladder on 133 tables and p4c
  agrees on all 133. Its one accepted miss, independent_low_sd5's (27, 52)
  key (lane 2 vs p4c 3: p4c's greedy midbyte choice leaves no nibble for the
  version bits), is priced 3 by the port, matching p4c.

  Order-insensitive. The empty key returns the band floor
  (tables.codeword_bits_to_blocks(0)). Raises CrossbarKeyTooWide(message,
  live bits) where p4c refuses the key -- its two checks (decision 6a, Task
  32): more than target.MAX_TERNARY_KEY_BITS live bits (tables.check_key_bits),
  or no placement in one stage (crossbar_blocks)."""
  widths = tuple(sorted(int(w) for w in field_bit_widths))
  check_key_bits(sum(widths))
  return _table_blocks(widths)


def crossbar_blocks(field_bit_widths):
  """table_blocks WITHOUT the live-bit check: the crossbar port's blocks per
  512-row word of the key, CrossbarKeyTooWide when it cannot be placed in one
  stage (tables.word_blocks: no allocation in 12 groups, or a word over 12
  blocks). For threshold alignment, which prices canonical stand-in widths
  (align_budget.port_width: same bytes, same nibbles, MORE bits) and applies
  the bit check to the real widths itself."""
  return _table_blocks(tuple(sorted(int(w) for w in field_bit_widths)))


@functools.lru_cache(maxsize=65536)
def _table_blocks(widths):
  byte_width = codeword_fields_to_bytes_from_bits(widths)
  if byte_width == 0:
    return codeword_bits_to_blocks(0)
  # A shortcut, not a limit: a key of more bytes than the stage's 66 crossbar
  # byte slots (12 groups x 5 + 6 midbytes) has no allocation, and the port
  # would only find that after trying every group count.
  price = (None if byte_width > TERNARY_CROSSBAR_STAGE_BYTES
           else word_blocks(xbar.alone_blocks(key_fields(widths))))
  if price is None:
    raise CrossbarKeyTooWide(
        "table key of %d crossbar bytes (%d live bits) cannot be placed in one "
        "stage's ternary crossbar" % (byte_width, sum(widths)), sum(widths))
  return price


def key_fields(field_bit_widths):
  """A key's fields as xbar.field_key takes them, (field id, slices) per field
  with any bits, laid out by `layout` -- the containers the
  generator pins (build_p4_script.code_field_container_sizes). Field ids are
  positions in `field_bit_widths`: a key never repeats a field."""
  field_bit_widths = [int(w) for w in field_bit_widths]
  return tuple((index, tuple(layout(w))) for index, w in enumerate(field_bit_widths)
               if w > 0)


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


def container_bytes(w):
  """Bytes of the containers a `w`-bit joint / disjoint / per-task code field is
  pinned to (container_sizes with this module's layout, the generator's
  @pa_container_size), which is also the field's declared width in bytes
  (decision 4.3). The action data bus carries containers, not bits, so this is
  what a range action writing the field puts on the bus (design review S14): a
  17-bit code in a W is 4 bytes, a 3-bit code in a B is 1."""
  return sum(container_sizes(w)) // 8


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
    # Each group supplies one slot per lane plus one extra in the lane it starts
    # on. The key picks its groups, but only GROUPS_PER_START_LANE start on a lane.
    for extras in itertools.combinations_with_replacement(range(_LANES), g):
      if any(extras.count(l) > GROUPS_PER_START_LANE[l] for l in range(_LANES)):
        continue
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
  `g` groups (each one slot per lane plus one extra on its start lane; at most
  GROUPS_PER_START_LANE groups start on a lane), `w` whole
  midbytes and at most one nibble-only tail midbyte `t`, with
  `2w + t + 1 <= g`, legally hold every byte. `None` if no `g <= 12` does.

  Agrees with p4c on 57/57 probe keys (from the real PHV layout) and with
  the (since deleted) byte-count ladder `tables.codeword_to_blocks` on 74/74
  real design keys (laid out by `layout`)."""
  W, H, B = _counts(byte_list)
  return _standalone(W, H, B, _nibble_types(byte_list))
