"""Canonical home for range_entry_count, nibble_widths_for and compiler_range_rows.

This module stays import-free apart from `math` and the two `src.p4model`
constant modules it needs, so it can be imported both here (via
src/p4gen/evaluation.py, as part of that module's full sklearn-heavy
dependency stack) and from p4/deploy_table_entries.py running inside
bfshell's embedded Python, which has no sklearn and thus cannot import
evaluation.py's own dependency stack. range_entry_count is pure integer
arithmetic with no imports of its own, so both sides can share this one
copy instead of duplicating it -- it used to live at p4/range_expansion.py
for exactly this reason; that separate module is now gone and this is the
single canonical copy."""
import math

from src.p4model.program import FEATURE_VALUE_BIT_WIDTH
from src.p4model.target import (
    MAX_RANGE_KEY_BITS,
    RANGE_WORST_CASE_ENTRY_FRACTION,
    RANGE_WORST_CASE_ROWS_CAP,
)


def range_entry_count(lo, hi, nibble_widths=(4, 4, 4, 4)):
  """Exact port of expand_range() (bf-drivers/src/pipe_mgr/pipe_mgr_entry_format.c,
  the real Tofino P4 driver source) -- computes the true number of physical
  TCAM rows the control plane needs to install a single range key [lo, hi],
  decomposed into consecutive 4-bit nibble segments (LSB-first). Verified by
  hand-trace against reviews/cited_papers/tofino_results_2.odt.pdf slide 11's
  worked example ([10,300] on 16 bits -> exactly 4 entries, matching the
  slide's exact sub-range boundaries, not just the count)."""
  n = len(nibble_widths)
  start_vals, end_vals = [], []
  shift = 0
  for w in nibble_widths:
    start_vals.append(1 << shift)
    end_vals.append((1 << (w + shift)) - 1)
    shift += w

  if hi < lo:
    raise ValueError("hi < lo")

  range_start, end, count = lo, hi, 0
  while True:
    if range_start == 0:
      start_nibble = n - 1
    else:
      zeroes = (range_start & -range_start).bit_length() - 1
      cum, start_nibble = 0, n - 1
      for j in range(n):
        cum += nibble_widths[j]
        if cum > zeroes:
          start_nibble = j
          break

    range_end = None
    for i in range(start_nibble + 1, 0, -1):
      candidate = range_start | end_vals[i - 1]
      while (candidate >= range_start and candidate > end and
             candidate >= start_vals[i - 1]):
        candidate -= start_vals[i - 1]
      if candidate >= range_start and candidate <= end:
        range_end = candidate
        break

    count += 1
    range_start = range_end + 1
    if range_end >= end:
      break

  return count


def nibble_widths_for(bits):
  """Nibble geometry expand_range() walks for a key of `bits` bits.

  Above MAX_RANGE_KEY_BITS the SDE refuses the table outright, so this raises
  rather than returning a geometry -- the case the old width_factor was
  insuring against does not need pricing, it needs rejecting."""
  if bits > MAX_RANGE_KEY_BITS:
    raise ValueError(
        "range key of %d bits does not compile (SDE ceiling is %d bits)"
        % (bits, MAX_RANGE_KEY_BITS))
  full, rem = divmod(bits, 4)
  return tuple([4] * full + ([rem] if rem else []))


def compiler_range_rows(entry_count, key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Physical TCAM rows p4c reserves for a range table of `entry_count`
  declared entries -- the COMPILE-TIME sizing, which is what decides how many
  blocks end up in the binary.

  The compiler never sees the interval bounds (this project's range tables are
  populated at runtime via the control plane, never `const entries` -- §4.4),
  so it cannot cost them exactly. It applies a fixed distributional
  guess instead: a quarter of the declared entries are priced at the
  worst-case row count for the key's nibble geometry, the rest at one row
  each.

  This is NOT interchangeable with range_entry_count. That one models
  expand_range(), the driver's exact per-value decomposition at INSERTION
  time; this one models the compiler's pessimistic pre-allocation. Blocks are
  the compiler's question -- using the driver's number to answer it
  under-counts (measured: a 478-entry table priced at 1 block against p4c's
  committed 3). reviews/p4_tofino_reference.md §4.2 and Appendix B
  "Mechanism E".

  Reproduces all five of §4.2's independently measured per-block
  capacities as the largest entry_count whose rows still fit 512: 512 (4-bit
  key), 342 (8-bit), 256 (12-bit), 206 (16-bit), 187 (19-bit)."""
  worst = min(RANGE_WORST_CASE_ROWS_CAP,
              2 * len(nibble_widths_for(key_bit_width)) - 1)
  quarter = entry_count // RANGE_WORST_CASE_ENTRY_FRACTION
  return quarter * worst + (entry_count - quarter)
