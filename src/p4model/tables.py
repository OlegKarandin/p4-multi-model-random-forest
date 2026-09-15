"""What one table costs: rows, blocks, and the crossbar key fields it claims.

The two accounting rules that matter, both measured rather than derived: blocks
charge crossbar BYTES per key FIELD (codeword_bytes_to_blocks), not raw codeword
bits; and a range table's blocks come from the DECLARED interval count via
compiler_range_rows, never from the expanded physical row count that
range_entry_count gives -- those answer different questions, and
range_deployment_overflow exists to keep them apart."""
import math

from src.p4model.errors import CodewordTooLong, CrossbarKeyTooWide
from src.p4model.names import normalise_feature_name
from src.p4model.program import FEATURE_VALUE_BIT_WIDTH
from src.p4model.ranges import compiler_range_rows, nibble_widths_for, range_entry_count
from src.p4model.target import (
    CODEWORD_KEY_OVERHEAD_BITS,
    CROSSBAR_PRIVATE_BYTES_PER_GROUP,
    MAX_CODEWORD_LENGTH,
    TCAM_BLOCK_KEY_LENGTH,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
)


def range_deployment_overflow(feature_intervals,
                              key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Features whose REAL intervals will not fit the blocks the compiler
  allocated for them: {feature: (rows_needed, rows_available)}, empty when
  every table fits.

  The second of two independent constraints, and the one blocks does not
  subsume. A committed block count is fixed in the binary -- the control
  plane cannot grow a table, it just gets "[Not enough space]" partway
  through insertion (§4.2; that failure is literally how range_entry_count
  was validated, since bf_rt exposes no per-entry row visibility). So a design
  can be perfectly feasible on blocks and still be undeployable.

  It can genuinely happen: compiler_range_rows budgets 2.5 rows per entry at
  this project's 16-bit key width, while a single maximally-misaligned range
  costs up to 7. Measured reality averages ~1.96 rows/entry and every row of
  the calibration study clears its allocation by at least 1.66x, so this is a
  guard against a tail, not a routine constraint -- which is exactly why it
  belongs here as an assertion rather than inside the block cost."""
  nibble_widths = nibble_widths_for(key_bit_width)
  overflow = {}
  for feature, intervals in feature_intervals.items():
    needed = sum(range_entry_count(lo, hi, nibble_widths) for lo, hi in intervals)
    available = (math.ceil(compiler_range_rows(len(intervals), key_bit_width)
                           / TERNARY_MATCHING_ENTRIES_PER_BLOCK)
                 * TERNARY_MATCHING_ENTRIES_PER_BLOCK)
    if needed > available:
      overflow[feature] = (needed, available)
  return overflow


def range_matching_resource_usage(feature_intervals, key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Returns (range_entries, range_blocks, range_table_specs).

  range_entries is the EXPANDED PHYSICAL TCAM ROW COUNT (the same quantity
  range_blocks quantizes via ceil(total_rows / TERNARY_MATCHING_ENTRIES_PER_BLOCK)),
  NOT a count of distinct [lo, hi] intervals -- one range interval typically
  expands to several physical rows (see range_entry_count / expand_range()),
  so range_entries >= the interval count, often strictly greater. (A1: prior
  to this fix, range_entries counted intervals, an unrelated quantity that
  could not be meaningfully compared against range_blocks.)

  Every selected feature gets its OWN independent range-matching P4 table
  (build_p4_script.py:663-674, keyed on "meta.<feature>_val : range"), so
  range_table_specs is one (block_count, byte_width) pair per feature --
  the per-table data crossbar_stages_needed() needs. The aggregate
  range_blocks is still returned for the blocks half of the cost model.

  A physical TCAM block is TERNARY_MATCHING_ENTRIES_PER_BLOCK rows x
  TCAM_BLOCK_KEY_LENGTH key bits. For RANGE keys, unlike TERNARY keys (where
  ceil((bits + 4) / 44) genuinely applies), words-per-entry is not a function
  of key width at all -- it is decided by PHV container width, and
  generate_P4_code already pins every feature value field to a 16-bit
  container via an @pa_container_size pragma (build_p4_script.py). So this
  function's depth-only formula (ceil(total_rows / 512)) is correct BECAUSE
  of that pragma, not by coincidence: with the container width fixed at 16
  bits, one row always costs exactly one TCAM word, regardless of
  key_bit_width. Keys wider than MAX_RANGE_KEY_BITS never reach this
  computation -- nibble_widths_for() raises first, since the SDE would
  refuse such a table outright and pricing it is meaningless.

  IMPORTANT (measured, reviews/p4_tofino_reference.md §4.2): this
  correctness depends on the @pa_container_size pragma. A bit<16> range key
  that the compiler parks in a 32-bit W container really costs TWO TCAM
  words per entry ("1 in 2 (88)"), not one; without those pragmas this
  function would under-count by up to a factor of 2 per table."""
  range_entries, range_blocks = 0, 0
  range_table_specs = []

  key_bytes = math.ceil(key_bit_width / 8)
  nibble_widths = nibble_widths_for(key_bit_width)

  for feature in feature_intervals:
    total_rows = 0
    for lo, hi in feature_intervals[feature]:
      total_rows += range_entry_count(lo, hi, nibble_widths)

    # Blocks come from the compiler's own compile-time sizing of the DECLARED
    # entry count (build_p4_script writes size = len(intervals)), not from
    # total_rows. total_rows is an insertion-time quantity and answers a
    # different question -- range_deployment_overflow is where it belongs.
    feature_blocks = math.ceil(
        compiler_range_rows(len(feature_intervals[feature]), key_bit_width)
        / TERNARY_MATCHING_ENTRIES_PER_BLOCK)

    range_entries += total_rows
    range_blocks += feature_blocks
    range_table_specs.append((feature_blocks, key_bytes))

  return range_entries, range_blocks, range_table_specs


def codeword_fields_to_bytes(feature_intervals):
  """Crossbar byte width of ONE classification table.

  (Was `ternary_table_key_bytes` until 2026-09-14. Renamed, not changed: the
  name now states the decomposition step it performs -- feature interval
  FIELDS in, crossbar BYTES out.)

  The classification tables do not key on a single concatenated codeword
  field: build_p4_script.py:630-635 emits one separate ternary key field
  per selected feature ("meta.code_<feature> : ternary"), each declared
  bit<len(feature_intervals[feature]) - 1> at build_p4_script.py:773-776.
  The match input crossbar allocates per FIELD, so the real byte cost is
  the sum of each field's own byte-rounded width, which is always >=
  ceil(total_bits / 8) on the concatenation (e.g. 3 features x 4 bits:
  3 bytes, not 2). Rounding the concatenation would under-count.

  Note: the "+4" ternary overhead used by ternary_matching_resource_usage
  is a TCAM *block capacity* fact (RM-3 Design A), not a crossbar-byte
  fact, and is deliberately NOT applied here."""
  return codeword_fields_to_bytes_from_bits(
      [max(len(intervals) - 1, 0) for intervals in feature_intervals.values()])


def codeword_fields_to_bytes_from_bits(field_bit_widths):
  """Crossbar byte width of a key given its field BIT widths.

  The same rule as codeword_fields_to_bytes, which takes the interval dict the
  generator hands out; this one takes the bit tuple ternary_key_field_bits
  returns. Two entry points, ONE rounding rule -- the inline
  `sum(math.ceil(bits / 8) for bits in ...)` this replaces was the third live
  copy (design §5.1)."""
  return sum(math.ceil(bits / 8) for bits in field_bit_widths)


def codeword_bits_to_blocks(codeword_length):
  """How many TCAM_BLOCK_KEY_LENGTH-wide key blocks one classification-table
  row spans. THE step function alignment is optimising against: a shed bit is
  worth nothing unless it carries codeword_length across a band boundary, and
  then it is worth n_trees blocks at once.

  (Was `band_factor` until 2026-09-14. Renamed, not changed: the name now
  states what it takes -- raw codeword BITS -- so it cannot be mistaken for
  the byte-wise arm below.)"""
  return math.ceil(
      (codeword_length + CODEWORD_KEY_OVERHEAD_BITS) / TCAM_BLOCK_KEY_LENGTH)


def ternary_key_field_bits(feature_intervals):
  """The bit width of every `meta.code_<feature>` field in one classification
  table's key, which is what version_block_penalty prices.

  Each field is declared `bit<len(intervals) - 1>` (build_p4_script.py:773-776).
  Widths are returned SORTED because the penalty depends only on the multiset
  of field widths, never on the order the generator happens to emit them in --
  the crossbar allocator is free to place fields where it likes, and measurably
  does (independent_low_sd6's 19-byte key lands on crossbar bytes 5-10, 12, 14,
  17-21 and 27-32, not on a contiguous run)."""
  return tuple(sorted(max(len(intervals) - 1, 0)
                      for intervals in feature_intervals.values()))


def _full_midbytes(start_group, groups):
  """Midbytes both of whose groups lie inside the run [start, start+groups-1].

  Crossbar groups pair up: group 2i and 2i+1 are fed from one 11-byte span
  laid out as 5 private bytes, the shared MIDBYTE, 5 private bytes. A pair
  straddling the end of the run yields only a HALF midbyte -- one nibble --
  which can never carry a whole key byte."""
  last = start_group + groups - 1
  return sum(1 for i in range(start_group // 2, last // 2 + 1)
             if 2 * i >= start_group and 2 * i + 1 <= last)


def _midbyte_slot_indices(start_group, groups):
  """Where the fully-owned midbytes sit among the run's WHOLE byte slots.

  Walking the run group by group: every group contributes its 5 private byte
  slots, and an even group additionally contributes the midbyte it shares with
  its partner whenever that partner is also in the run."""
  slots, index = [], 0
  last = start_group + groups - 1
  for group in range(start_group, last + 1):
    index += CROSSBAR_PRIVATE_BYTES_PER_GROUP
    if group % 2 == 0 and group + 1 <= last:
      slots.append(index)
      index += 1
  return slots


def _clean_byte_can_reach_a_midbyte(byte_widths, nibble_clean, slots):
  """Could some ordering of the key's fields put a nibble-clean byte on a
  fully-owned midbyte? Such a byte claims one nibble and leaves the other
  free, which is a legal home for --version--
  (`IXBar::Use::Byte::only_one_nibble_in_use`, input_xbar.h:298).

  A field's nibble-clean byte is always its LAST one, so the slots it can
  reach are `(sum of the fields placed before it) + width - 1`. Those sums are
  every subset sum of the other fields' widths, computed here as a bitset --
  exact, and far cheaper than enumerating the 15! orderings a wide key admits.
  """
  if not slots or not any(nibble_clean):
    return False
  total = sum(byte_widths)
  for i, is_clean in enumerate(nibble_clean):
    if not is_clean:
      continue
    reachable = 1                       # bit j set == a preceding sum of j
    for j, width in enumerate(byte_widths):
      if j != i:
        reachable |= reachable << width
    for slot in slots:
      preceding = slot - byte_widths[i] + 1
      if 0 <= preceding <= total and (reachable >> preceding) & 1:
        return True
  return False


def _run_capacity(start_group, groups, nibble_clean_count):
  """Key BYTES the crossbar run [start_group, start_group + groups - 1] can hold.

  Three kinds of slot, and they are not interchangeable:
    * CROSSBAR_PRIVATE_BYTES_PER_GROUP private bytes per group, always whole;
    * the midbytes both of whose groups lie inside the run (_full_midbytes),
      also whole;
    * the HALF midbyte the run exposes at its low end when it starts on an odd
      group, and at its high end when it ends on an even one. Half a midbyte is
      one nibble: a whole byte can never ride it, and only a field whose last
      byte uses a single nibble (`1 <= bits % 8 <= 4`) can -- so at most
      nibble_clean_count of them are usable, however many are exposed.
  """
  last = start_group + groups - 1
  whole = (CROSSBAR_PRIVATE_BYTES_PER_GROUP * groups
           + _full_midbytes(start_group, groups))
  halves = ((1 if start_group % 2 == 1 else 0)
            + (1 if last % 2 == 0 else 0))
  return whole + min(halves, nibble_clean_count)


def crossbar_groups_needed(field_bit_widths, start_group=0):
  """Consecutive crossbar groups this key's bytes occupy starting at start_group.

  codeword_bytes_to_blocks answers this at offset 0 and ONLY at offset 0. A run
  of g groups from group 0 owns 5g + floor(g/2) = floor(5.5g) whole byte slots,
  which is exactly the widest key ceil(key_bytes / 5.5) assigns to g groups --
  so the two agree there by arithmetic, not by coincidence, and this function
  is a strict generalisation rather than a replacement. Shifted runs are
  poorer: a 2-group run starting ODD owns no full midbyte at all, so it holds
  10 bytes where the same run at 0 holds 11.

  EVIDENCE STATE. The offset-0 agreement is proved above and property-tested
  (test_a_run_at_offset_zero_takes_exactly_the_byte_derived_group_count). The
  SHIFTED arm is first-principles geometry, NOT hardware-confirmed: Task 5's
  spacer sweep (scripts/tcam_spacer_sweep.py) ran 12 real p4c compiles looking
  for a non-ambiguous odd-offset triple and got zero -- every point came back
  start_group_ambiguous=True (see that task's report). So this arm ships on
  geometry and property tests alone. The change is MONOTONE by construction --
  the loop only ever grows the run -- which is the safe direction under spec
  3.6: it can over-predict and reject a feasible design, never under-predict
  and admit an infeasible one.
  """
  byte_widths = [math.ceil(bits / 8) for bits in field_bit_widths]
  key_bytes = sum(byte_widths)
  if key_bytes == 0:
    return 0
  nibble_clean_count = sum(1 for bits in field_bit_widths
                           if 1 <= bits % 8 <= 4)
  groups = codeword_bytes_to_blocks(key_bytes)
  while _run_capacity(start_group, groups, nibble_clean_count) < key_bytes:
    groups += 1
  return groups


def codeword_to_blocks(field_bit_widths, start_group=0):
  """TCAM blocks ONE table word of this key spans, version field included.

  (Was `ternary_block_factor` until 2026-09-14.)

  THE composition, and the only place the version charge may be added: it is a
  function of the width MULTISET and of `start_group`, so neither leaf function
  below could carry it without lying about what it depends on.

  Keeps a max() between the bit-width lower bound and the crossbar-plus-version
  arm, unchanged from the retired `ternary_block_factor`. The plan this function
  was written under (design D5, 2026-09-14) intended to demote the bit-width arm
  to a bare assertion, reasoning it was a PROVABLE lower bound that could never
  exceed the crossbar arm. That reasoning does not hold against this codebase's
  actual `version_block_penalty`: its clause (a) only checks the HIGH end of a
  crossbar run for a free half-midbyte, missing (1) the LOW end when
  `start_group` is odd, and (2) the empty-key case, where a version field still
  needs a physical block even though the byte-domain arm claims 0 blocks
  suffice. PRE-FIX FIGURE, SUPERSEDED (kept only for the historical shape of
  the bug, not as a fact about the code below): measured against the
  now-replaced clause-(a) logic, 272 of 500 000 random field-width trials
  violated the "provable" bound this way (e.g. widths (38, 48) at
  start_group=3: bit-bound 3 vs crossbar-plus-version 2). The empty tuple --
  reachable whenever every tree in a forest is a single leaf, see
  test_factor_of_an_empty_width_dict_is_the_empty_key_factor -- violates it
  without any randomness at all (bit-bound 1 vs crossbar-plus-version 0), and
  this part is NOT superseded; see the current numbers a few paragraphs below.

  D5's demotion is therefore DECLINED here, not adopted -- but only half of the
  original reason still holds. The clause-(a) hole IS now repaired
  (`crossbar_groups_needed` grows the run until the key's own bytes actually
  fit before asking whether a nibble is left for --version--), so start_group=3
  odd-offset counterexamples like widths (38, 48) no longer violate the bound.
  What remains is the empty-key case: `codeword_to_blocks((), s) == 1` from the
  bit arm vs 0 from the crossbar arm, reachable whenever every tree in a forest
  is a single leaf (test_factor_of_an_empty_width_dict_is_the_empty_key_factor).
  A hard assertion would crash on that reachable, already-tested input, so the
  max() stays.

  Re-measured against the repaired code, same 500 000-trial generator as
  before (`random.seed(0)`, field counts 0-15, widths 1-80 bits, offsets 0-7):
  31 524 trials still violate the naive bound, but every single one is the
  empty-key case above (0 fields, so the crossbar arm is 0 and the bit arm is
  still >= 1) -- confirmed by re-running with the violations partitioned on
  `len(widths) == 0`. Non-empty violations, the shape this task fixes, are
  0/500 000, down from 328/500 000 measured against the pre-fix code with the
  identical generator (the closest comparable figure to this docstring's
  previous stale "272"). So the fix closes the odd-offset hole completely on
  this generator; only the structural empty-key case remains, which max()
  already covers and no test here claims otherwise."""
  blocks = (crossbar_groups_needed(field_bit_widths, start_group)
            + version_block_penalty(field_bit_widths, start_group))
  return max(codeword_bits_to_blocks(sum(field_bit_widths)), blocks)


def version_block_delta(field_bit_widths, start_group):
  """How many blocks this key costs at `start_group` OVER its cost at 0.

  In {-1, 0, +1}. The stage packer needs this rather than the penalty itself:
  a table's declared block count already prices the key standalone, i.e. at
  offset 0, so what a placement adds is only the difference the offset makes.
  Charging `version_block_penalty` directly there would re-bill a key that
  already paid in its own spec."""
  return (codeword_to_blocks(field_bit_widths, start_group)
          - codeword_to_blocks(field_bit_widths, 0))


def version_block_penalty(field_bit_widths, start_group=0):
  """Extra TCAM blocks (0 or 1) this key costs to house the --version-- field.

  Every ternary entry carries a mandatory 2-bit version/valid field, and it
  can live only in a crossbar MIDBYTE nibble (§1.3 of
  reviews/github_issue_tcam_version_bit_packing.md). p4c's crossbar sizing
  never reserves that nibble, so when the key's own bytes consume every
  midbyte the format falls through and `TableFormat::ternary_version()`
  push_back()s a whole extra TCAM to hold two bits.

  A key of `key_bytes` crossbar bytes takes `g = crossbar_groups_needed(
  field_bit_widths, start_group)` groups starting at `start_group` -- NOT a
  fixed `codeword_bytes_to_blocks(key_bytes)` count, because that ignores the
  offset. `crossbar_groups_needed` starts from that byte-derived count and
  grows the run one group at a time until the key's own bytes actually fit
  the slots the run owns at THIS start_group (finding 1.2: a run starting on
  an odd group owns no full midbyte, so it holds fewer whole byte slots per
  group than the same-size run at offset 0). Those `g` groups supply `5g`
  private byte slots plus `_full_midbytes` fully-owned midbytes; the run
  additionally exposes a HALF midbyte at its low end when it starts on an odd
  group, and at its high end when it ends on an even one. Version has a home
  when any of:

    (a) the run ENDS on a half midbyte -- a whole byte can never ride it, so
        it survives whatever the key does;
    (b) a whole byte slot is spare -- p4c scatters the key's bytes rather than
        packing them contiguously, so any slack anywhere lets it keep a
        midbyte open (measured: independent_low_sd6);
    (c) the run BEGINS on a half midbyte and the key has no nibble-clean byte
        to put there (this is why a SOLID key is priced the same at every
        offset, and a ragged one is not);
    (d) a nibble-clean byte can land on a fully-owned midbyte.

  MEASUREMENT. Over the 100 classification tables of the 19 archived compiles
  in `results/compiler_calibration_v6/`, block counts read straight out of
  `resources.json`, this is exact on every table: 3 penalties predicted, 3
  observed, no false alarms and no misses. The three are independent_low_sd5's
  ddos trees, whose 11-byte key exactly saturates two groups' 11 byte slots.
  The start-offset term is pinned separately by scripts/tcam_stretch_sweep.py,
  whose ragged arm costs the same table 9 blocks at offset 0 and 10 at
  offset 3 while its solid control costs 9 at both.

  THIS SUPERSEDES the earlier "Mechanism G" rule, which charged +1 whenever a
  ragged key sat at an ODD group offset. That predicate over-fired on 5 of the
  6 calibration stages where it was live, and on the one row it appeared to
  fix it charged the WRONG table -- resources.json shows p4c penalising the
  even-offset ddos key, not the odd-offset app key. See
  reviews/p4_tofino_reference.md Appendix B "Mechanism G"."""
  byte_widths = [math.ceil(bits / 8) for bits in field_bit_widths]
  key_bytes = sum(byte_widths)
  if key_bytes == 0:
    return 0
  nibble_clean = [1 <= (bits % 8) <= 4 for bits in field_bit_widths]
  # The run the key REALLY occupies at this offset, which is the run whose
  # midbytes version has to find a nibble in. Asking codeword_bytes_to_blocks
  # here was finding 1.2: clause (a) below would let a run off the hook for
  # ending on a half midbyte without ever checking the key fit inside it.
  groups = crossbar_groups_needed(field_bit_widths, start_group)
  last = start_group + groups - 1

  if last % 2 == 0:
    return 0                                                        # (a)
  whole_slots = (CROSSBAR_PRIVATE_BYTES_PER_GROUP * groups
                 + _full_midbytes(start_group, groups))
  if whole_slots - key_bytes >= 1:
    return 0                                                        # (b)
  if start_group % 2 == 1 and not any(nibble_clean):
    return 0                                                        # (c)
  return 0 if _clean_byte_can_reach_a_midbyte(                      # (d)
      byte_widths, nibble_clean,
      _midbyte_slot_indices(start_group, groups)) else 1


def codeword_bytes_to_blocks(key_bytes):
  """TCAM blocks one classification-table row spans because of the ternary
  input CROSSBAR, as opposed to because of its bit width.

  (Was `crossbar_block_width` until 2026-09-14. Renamed, not changed: the name
  now states what it takes -- crossbar key BYTES -- so the decomposition
  fields -> bytes -> blocks reads in one direction. It carries NO version
  charge; see version_block_penalty for why that charge cannot live here.)

  One block is fed by exactly one crossbar group, and a group delivers 5
  private bytes + 1 midbyte nibble = TCAM_BLOCK_KEY_LENGTH bits = 5.5 BYTES
  (reviews/p4_tofino_reference.md §4.1.1). The crossbar allocates per FIELD
  and byte-rounds each one, so
  what it charges is key_bytes (codeword_fields_to_bytes), not the raw
  codeword length -- a table keying 15 separate code_<feature> fields
  totalling 205 bits really presents 33 bytes = 264 bits and needs 6 blocks,
  not 5.

  This is the term codeword_bits_to_blocks misses, and it is why
  codeword_bits_to_blocks was accidentally right for years: on ONE dense wide
  codeword field byte-rounding is a no-op and the two agree exactly. They
  diverge as soon as the key is split per feature, which is what
  build_p4_script actually emits.

  Measured exact on 144 real compiled classification tables spanning three
  compile eras -- the whole observed key_bytes -> blocks ladder (4 -> 1,
  11 -> 2, 16 -> 3, 20 -> 4, 26 -> 5, 33 -> 6, 37 -> 7, 41 -> 8, 49 -> 9,
  52 -> 10, 60 -> 11) is single-valued and lands on this function. Appendix B
  "Mechanism D"."""
  return math.ceil(key_bytes * 8 / TCAM_BLOCK_KEY_LENGTH)


def tree_entries_to_blocks(entries):
  """Block-rows ONE tree's classification table needs for its own leaves.

  A block is TERNARY_MATCHING_ENTRIES_PER_BLOCK rows deep, and each leaf is one
  row. Multiply by the key's own block width (codeword_to_blocks) for the
  table's real block count: depth and width are independent and both cost.
  """
  return math.ceil(entries / TERNARY_MATCHING_ENTRIES_PER_BLOCK)


def entries_across_trees_to_blocks(entry_counts):
  """What ONE step of the key's block width is worth, in blocks.

  Blocks are charged once per TREE, because a block is memory and different
  tables store different rows -- the exact opposite of the crossbar's byte
  slots, which a stage charges once however many tables read them
  (crossbar_stages_needed's key_fields). Confusing the two is a mistake this
  project has already made once in the other direction.

  This is the multiplier src/training/align_budget.py needs to weigh a range
  step (worth 1 block) against a key-width step (worth this, 8-80 blocks across
  the golden fixture).
  """
  return sum(tree_entries_to_blocks(count) for count in entry_counts)


def ternary_matching_resource_usage(codewords, feature_intervals,
                                     dropped_per_tree=None):
  """Returns (ternary_entries, ternary_blocks, codeword_length,
  ternary_table_specs).

  Each tree gets its own independent classification table
  (build_p4_script.py:636-659), so ternary_table_specs is one
  (block_count, byte_width) pair per tree. All of those tables key on the
  same set of per-feature fields, so they share one byte width.

  dropped_per_tree: optional sequence of ints, positionally aligned with
  `codewords`, giving how many of each tree's entries the generator folds into
  the table's default_action instead of installing explicitly. None (the
  default) charges every codeword.

  The COUNT is taken, not the rule that produced it. Which leaves get folded is
  an encoding convention -- this project uses Planter RF_EB's majority-class
  rule (build_p4_script.most_common_class_and_dropped_codewords, and see
  reviews/p4_tofino_reference.md §4.5, which also records that at the scales
  tested this reduces control-plane load without crossing a physical
  block-packing boundary). Keeping the rule on the caller's side is what lets
  this module model a table's cost without knowing anything about decision
  trees."""

  ternary_entries, ternary_blocks = 0, 0
  ternary_table_specs = []
  codeword_length = len(next(iter(codewords[0].items()))[0])

  if dropped_per_tree is not None and len(dropped_per_tree) != len(codewords):
    raise ValueError(
        "ternary_matching_resource_usage: got %d dropped_per_tree for %d "
        "codewords; the two must be positionally aligned, one count per "
        "tree, or a tree would be discounted against another tree's count"
        % (len(dropped_per_tree), len(codewords)))

  if codeword_length > MAX_CODEWORD_LENGTH:
    raise CodewordTooLong("Codewords are too long", codeword_length)

  table_bytes = codeword_fields_to_bytes(feature_intervals)

  if table_bytes > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    # Checked here, where the key width is already known, rather than deep
    # inside crossbar_stages_needed/_stage_shards: a trial should be rejected
    # with a clear reason at the point that has the clearest context, not
    # crash mid-estimate several calls later. _stage_shards keeps its own
    # copy of this check too (defense in depth for any other caller that
    # reaches it directly).
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (table_bytes, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), table_bytes)

  # Two independent lower bounds on how many blocks one row spans: its bit
  # width (codeword_bits_to_blocks, which carries the +4 version/valid nibble)
  # and its crossbar byte width plus the version block that width may not
  # leave room for. codeword_to_blocks composes them with max().
  factor = codeword_to_blocks(ternary_key_field_bits(feature_intervals))
  for index, tree in enumerate(codewords):
    tree_entry_count = len(codewords[tree])
    if dropped_per_tree is not None:
      tree_entry_count -= dropped_per_tree[index]

    tree_blocks = tree_entries_to_blocks(tree_entry_count) * factor

    ternary_entries += tree_entry_count
    ternary_blocks += tree_blocks
    ternary_table_specs.append((tree_blocks, table_bytes))

  return ternary_entries, ternary_blocks, codeword_length, ternary_table_specs


def exact_match_resource_usage(codewords, feature_intervals):
  """Planter RF_EB-style exact-match/SRAM entry-count accounting for the
  match_type='exact' code/decision tables (build_p4_script.generate_codewords).
  Exact match cannot express a leaf's wildcarded ('*') codeword bits, so each
  wildcarded per-feature segment must be enumerated into concrete entries: a
  segment where the feature is untested on the leaf's path is a thermometer/
  unary code with exactly len(feature_intervals[feature]) reachable values
  (not 2**width independent bit combinations), while a segment where the
  feature IS on the path keeps a safe 2**(wildcards-in-segment) over-
  approximation.

  Deliberately caller-less: there is no production caller and none is
  planned. Per reviews/todo.md:343-349 (2026-08-03), building a working
  entry-generator for match_type='exact' was deferred, not pursued further
  -- at this project's real feature scale the entry count this function
  computes comes out to ~1.3x10**34, which no real switch's SRAM could hold.
  The multiplier this function reports IS the finding: the analytical
  accounting stays as documented output even though a generator for the
  approach it accounts for does not exist and is not being built.

  Returns (sram_entries, sram_blocks). sram_blocks is None by design: the
  per-block SRAM entry capacity for a plain exact-match key table is not a
  documented closed-form constant in this project -- it depends on the
  compiler's LayoutOption/"ways" search (packing entries against RAM row
  width, overhead/version bits, and hash-way constraints jointly), which is
  out of scope while the entry-generator itself remains deferred."""
  sram_entries = 0
  for tree in codewords:
    for codeword in codewords[tree]:
      entry_factor = 1
      position = 0
      for feature, intervals in feature_intervals.items():
        width = len(intervals) - 1
        if width <= 0:
          continue
        segment = codeword[position:position + width]
        position += width
        if segment == '*' * width:
          # Feature entirely untested on this leaf's path: thermometer
          # code has exactly len(intervals) reachable values, not 2**width.
          entry_factor *= len(intervals)
        else:
          # Feature IS on the path -- any remaining '*' in this segment is
          # not a full free choice among len(intervals) values. Keep the
          # old 2**(wildcards-in-segment) as a safe over-approximation.
          entry_factor *= 2 ** segment.count('*')
      sram_entries += entry_factor

  sram_blocks = None
  return sram_entries, sram_blocks


def range_key_fields_for(feature_intervals, key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """One frozenset of (field_id, field_bytes) per range table, positionally
  aligned with range_matching_resource_usage's range_table_specs.

  A range table keys on exactly one field, `meta.<raw_feature_name>_val`
  (build_p4_script.py:1170), and that raw value field is shared across models
  by construction -- _resolve_disjoint_feature_plan's own docstring: "the RAW
  VALUE register/computation is always shared and keyed by raw_feature_name,
  never by resolved_name". So the field id is the normalised raw feature
  name, and under 'disjoint' both models' tables for the same feature
  correctly resolve to the SAME crossbar field. Measured on
  independent_high_sd6 stage 5: four range tables, three distinct value
  fields (app and ddos both read flow_iat_min_val), 12 crossbar bytes
  reported rather than 16."""
  key_bytes = math.ceil(key_bit_width / 8)
  return [frozenset({(normalise_feature_name(feature), key_bytes)})
          for feature in feature_intervals]


def ternary_key_fields(feature_intervals):
  """The ONE frozenset of (field_id, field_bytes) that EVERY classification
  table of a model keys on -- one field per selected feature
  (`meta.code_<resolved_name> : ternary`, build_p4_script.py:1129), each
  ceil((len(intervals) - 1) / 8) bytes wide. Sums to
  codeword_fields_to_bytes(feature_intervals) by construction, which is the
  invariant crossbar_stages_needed checks.

  The field id carries the interval list, not just the name, because that is
  exactly what decides whether the generator namespaces the field:
  _resolve_disjoint_feature_plan gives the two models one shared
  code_<feature> only when BOTH select it AND their interval lists are
  identical, and separate code_app_<feature>/code_ddos_<feature> otherwise.
  Keying the id on (name, intervals) reproduces those equivalence classes
  exactly, so a disjoint pair that happens to agree on a feature shares its
  crossbar bytes and one that disagrees does not."""
  return frozenset(
      ((normalise_feature_name(feature), tuple(intervals)),
       math.ceil(max(len(intervals) - 1, 0) / 8))
      for feature, intervals in feature_intervals.items())
