"""Zero-import module holding range_entry_count so it can be shared by both
src/p4gen/evaluation.py (imports it as p4.range_expansion) and
p4/deploy_table_entries.py (imports it as a same-directory module, since that
script runs inside bfshell's embedded Python, which has no sklearn and thus
cannot import evaluation.py's own dependency stack). range_entry_count is
pure integer arithmetic with no imports of its own, so both sides can share
this one copy instead of duplicating it."""


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
