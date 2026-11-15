"""What one table costs: rows, blocks, and the crossbar key fields it claims.

The two accounting rules that matter, both measured rather than derived: blocks
charge crossbar BYTES per key FIELD (the ternary crossbar port, lanes.table_blocks),
not raw codeword bits; and a range table's blocks come from the DECLARED interval count via
compiler_range_rows, never from the expanded physical row count that
range_entry_count gives -- those answer different questions, and
range_deployment_overflow exists to keep them apart."""
import math

from p4model import target
from p4model.errors import CrossbarKeyTooWide, RangeEntriesDoNotFit
from p4model.names import normalise_feature_name
from p4model.program import FEATURE_VALUE_BIT_WIDTH, per_task_field
from p4model.ranges import compiler_range_rows, nibble_widths_for, range_entry_count
from p4model.units import installed_intervals
from p4model.target import (
    ACTION_RAM_WORD_BYTES,
    CODEWORD_KEY_OVERHEAD_BITS,
    RANGE_KEY_SLOTS_PER_BYTE,
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_USABLE_ROWS_PER_BLOCK,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
)


def _usable_range_rows(interval_count, key_bit_width):
  """Rows a range table of `interval_count` declared intervals can actually hold:
  the blocks p4c commits for it (compiler_range_rows) x usable rows per block."""
  return (math.ceil(compiler_range_rows(interval_count, key_bit_width)
                    / TERNARY_MATCHING_ENTRIES_PER_BLOCK)
          * TCAM_USABLE_ROWS_PER_BLOCK)


def installed_rows(feature, intervals, nibble_widths):
  """Physical TCAM rows of one range table's REAL entries: range_entry_count over
  the bounds the control plane installs (units.installed_intervals -- an IAT
  feature's bounds in 1.024 us ticks, an interval collapsed onto one tick not
  installed; any other feature's bounds as they are). `feature` is the table's
  feature name (its catalog key; the pricing's interval dicts are keyed by it)."""
  return sum(range_entry_count(lo, hi, nibble_widths)
             for lo, hi in installed_intervals(feature, intervals))


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
  guard against a tail, not a routine constraint.

  A committed block holds TCAM_USABLE_ROWS_PER_BLOCK usable rows (506, not
  512: insertion-order fragmentation can strand a few). range_matching_
  resource_usage enforces this (decision 6d) and raises RangeEntriesDoNotFit
  when any table overflows; this function is the oracle it shares with the
  tests and the review scripts.

  The rows are those of the bounds the control plane INSTALLS (installed_rows):
  an IAT feature's in 1.024 us ticks, collapsed intervals left out (Task 34)."""
  nibble_widths = nibble_widths_for(key_bit_width)
  overflow = {}
  for feature, intervals in feature_intervals.items():
    needed = installed_rows(feature, intervals, nibble_widths)
    available = _usable_range_rows(len(intervals), key_bit_width)
    if needed > available:
      overflow[feature] = (needed, available)
  return overflow


def range_matching_resource_usage(feature_intervals, key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Returns (range_entries, range_blocks, range_table_specs).

  range_entries is the EXPANDED PHYSICAL TCAM ROW COUNT range_entry_count
  gives for actually INSERTING every interval (installed_rows: the bounds the
  control plane writes, an IAT feature's in ticks since Task 34) -- NOT a count of distinct
  [lo, hi] intervals, and NOT the quantity range_blocks is derived from. One
  range interval typically expands to several physical rows (see
  range_entry_count / expand_range()), so range_entries >= the interval
  count, often strictly greater. (A1: prior to this fix, range_entries
  counted intervals, an unrelated quantity that could not be meaningfully
  compared against range_blocks.)

  range_blocks, by contrast, is sized at COMPILE time from the DECLARED
  interval count via compiler_range_rows(len(intervals)) -- never from
  range_entries/total_rows, which is an insertion-time quantity answering a
  different question (range_deployment_overflow is where that question
  belongs; see its own docstring).

  Every selected feature gets its OWN independent range-matching P4 table
  (build_p4_script.py:663-674, keyed on "meta.<feature>_val : range"), so
  range_table_specs is one (block_count, byte_width) pair per feature --
  the per-table data crossbar_stages_needed() needs. The aggregate
  range_blocks is still returned for the blocks half of the cost model.

  A physical TCAM block is TERNARY_MATCHING_ENTRIES_PER_BLOCK rows x
  TCAM_BLOCK_KEY_LENGTH key bits. For RANGE keys, words-per-entry is not a
  function of key width at all -- it is decided by PHV container width, and
  generate_P4_code already pins every feature value field to a 16-bit
  container via an @pa_container_size pragma (build_p4_script.py). So this
  function's depth-only formula (ceil(compiler_range_rows(len(intervals)) /
  512)) is correct BECAUSE of that pragma, not by coincidence: with the
  container width fixed at 16 bits, one row always costs exactly one TCAM
  word, regardless of
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

  # A range key's crossbar byte SLOTS: RANGE_LO + RANGE_HI per key byte, the unit
  # the port's range_key uses (decision 6f).
  key_bytes = math.ceil(key_bit_width / 8) * RANGE_KEY_SLOTS_PER_BYTE
  nibble_widths = nibble_widths_for(key_bit_width)
  rows_over, overflowing = 0, []

  for feature in feature_intervals:
    # the rows of the entries the control plane installs (IAT bounds in ticks, Task 34)
    total_rows = installed_rows(feature, feature_intervals[feature], nibble_widths)

    # Blocks come from the compiler's own compile-time sizing of the DECLARED
    # entry count (build_p4_script writes size = len(intervals)), not from
    # total_rows. total_rows is an insertion-time quantity and answers a
    # different question -- range_deployment_overflow is where it belongs.
    feature_blocks = math.ceil(
        compiler_range_rows(len(feature_intervals[feature]), key_bit_width)
        / TERNARY_MATCHING_ENTRIES_PER_BLOCK)

    usable = _usable_range_rows(len(feature_intervals[feature]), key_bit_width)
    if total_rows > usable:
      rows_over += total_rows - usable
      overflowing.append(feature)

    range_entries += total_rows
    range_blocks += feature_blocks
    range_table_specs.append((feature_blocks, key_bytes))

  if rows_over:
    raise RangeEntriesDoNotFit(
        "the real range entries of %s need %d more TCAM rows than the blocks p4c "
        "commits for them hold (%d usable rows per block); the control plane "
        "could not install them" % (', '.join(overflowing), rows_over,
                                    TCAM_USABLE_ROWS_PER_BLOCK), rows_over)

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

  Note: ternary_matching_resource_usage no longer applies a "+4" bit
  overhead anywhere -- that band-domain rule (RM-3 Design A) was retired
  along with codeword_bits_to_blocks's role in per-table pricing. The
  mandatory version field is priced by the crossbar port itself
  (lanes.table_blocks, xbar.blocks_per_row), which has nothing to add here."""
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
  """How many TCAM_BLOCK_KEY_LENGTH-wide key blocks a codeword_length-bit key
  would span under the retired band model (raw bits, version overhead
  included) -- NOT what any price uses any more; the crossbar port
  (lanes.table_blocks) prices a key's fields instead. This function survives
  for two reasons only: it is the empty-key floor (lanes.table_blocks
  special-cases the empty key to codeword_bits_to_blocks(0), since a version
  field still needs a
  physical block even when the key claims none, and this is the only place
  target.TCAM_BLOCK_KEY_LENGTH is read); and src/reporting/replay_scoring.py
  still derives its (unread) band_factor_before/after columns from it.

  (Was `band_factor` until 2026-09-14. Renamed, not changed: the name now
  states what it takes -- raw codeword BITS -- so it cannot be mistaken for
  the byte-wise arm below.)"""
  return math.ceil(
      (codeword_length + CODEWORD_KEY_OVERHEAD_BITS) / TCAM_BLOCK_KEY_LENGTH)


def ternary_key_field_bits(feature_intervals):
  """The bit width of every `meta.code_<feature>` field in one classification
  table's key, which is what lanes.table_blocks prices.

  Each field is declared `bit<len(intervals) - 1>` (build_p4_script.py:773-776).
  Widths are returned SORTED because the price depends only on the multiset
  of field widths, never on the order the generator happens to emit them in --
  the crossbar allocator is free to place fields where it likes, and measurably
  does (independent_low_sd6's 19-byte key lands on crossbar bytes 5-10, 12, 14,
  17-21 and 27-32, not on a contiguous run)."""
  return tuple(sorted(max(len(intervals) - 1, 0)
                      for intervals in feature_intervals.values()))


def range_action_bytes_for(feature_intervals):
  """Action-data bus bytes per range table, one per feature in
  feature_intervals' order, for a table writing the one `code_<feature>`
  field of `len(intervals) - 1` live bits (ternary_key_field_bits' width): the
  bytes of the containers the field is pinned to and declared at
  (lanes.container_bytes; decision 4.3, design review S14), since the bus
  carries whole containers. packing.crossbar_stages_needed turns them into
  whole 16-byte bus words."""
  # lanes imports this module, so it is imported here, at call time.
  from p4model.lanes import container_bytes
  return [container_bytes(len(intervals) - 1) for intervals in feature_intervals.values()]


def check_key_bits(live_bits):
  """CrossbarKeyTooWide when one ternary table's key has more LIVE bits (the
  widths of its key slices, summed) than target.MAX_TERNARY_KEY_BITS -- the first
  of p4c's two key-size checks (bf-p4c phv/table_phv_constraints.cpp:88; decision
  6a, Task 32). Read through the module so a test can move the limit. The second
  check is the crossbar port's fit (word_blocks)."""
  if live_bits > target.MAX_TERNARY_KEY_BITS:
    raise CrossbarKeyTooWide(
        "table key has %d live bits; p4c allows at most %d bits in a ternary match "
        "key (phv/table_phv_constraints.cpp:88)"
        % (live_bits, target.MAX_TERNARY_KEY_BITS), live_bits)


def word_blocks(per_row):
  """The second key-size check: the crossbar port's blocks per 512-row word of
  a key (xbar.alone_blocks / blocks_per_row), or None when the key cannot be
  placed in one stage -- no allocation in its 12 groups (`per_row` None), or a
  word wider than target.TCAM_ROWS_PER_STAGE blocks, which no stage holds: a
  word chains its blocks down ONE column of 12 (mau_spec.h:88-90). Probe 6a
  (ii): 64 x 8 + two 6-bit fields, 524 bits, fit 12 groups at 13 blocks per word
  (6 whole midbytes leave no nibble for the version bits), and p4c refused the
  table ("could not fit in stage 0 with 512 entries")."""
  if per_row is None or per_row > target.TCAM_ROWS_PER_STAGE:
    return None
  return per_row


def tree_entries_to_blocks(entries):
  """Block-rows ONE tree's classification table needs for its own leaves.

  A block is TERNARY_MATCHING_ENTRIES_PER_BLOCK rows deep, and each leaf is one
  row. Multiply by the key's own block width (lanes.table_blocks) for the
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

  src/training/align_budget.tree_keys sums it per tree key, to weigh a range
  step (worth 1 block) against a key-width step (worth this).
  """
  return sum(tree_entries_to_blocks(count) for count in entry_counts)


def ternary_matching_resource_usage(codewords, feature_intervals,
                                     dropped_per_tree=None):
  """Returns (ternary_entries, ternary_blocks, codeword_length,
  ternary_table_specs).

  Each tree gets its own independent classification table
  (build_p4_script.py:636-659), so ternary_table_specs is one
  (block_count, byte_width) pair per tree, every tree priced on the key of
  feature_intervals. Since decision 2.3 each tree keys only the fields of the
  features it splits on, so evaluation._pruned_trees calls this once per tree
  with that tree's own intervals; a call with several trees prices them all on
  one key.

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

  table_bytes = codeword_fields_to_bytes(feature_intervals)

  # Checked here, where the key is already known, rather than deep inside
  # crossbar_stages_needed: a trial should be rejected with a clear reason at
  # the point that has the clearest context. The bit check first (p4c's own
  # order: it refuses the key before any placement), then the crossbar fit,
  # inside lanes.table_blocks.
  check_key_bits(sum(ternary_key_field_bits(feature_intervals)))

  # The per-tree block width: lanes.table_blocks, xbar.alone_blocks of the
  # table's pinned crossbar key layout (2026-11-13; the lane count from
  # 2026-10-04 until then). Imported here, not at module level: lanes imports
  # this module at load time.
  from p4model.lanes import table_blocks
  factor = table_blocks(ternary_key_field_bits(feature_intervals))
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
  key_bytes = math.ceil(key_bit_width / 8) * RANGE_KEY_SLOTS_PER_BYTE
  return [frozenset({(normalise_feature_name(feature), key_bytes)})
          for feature in feature_intervals]


def per_task_key_fields(task, feature_intervals):
  """ternary_key_fields for the per-task encoding (program.PER_TASK_ENCODING):
  feature_intervals is `task`'s OWN intervals and the field id is the generated
  field's own name, tagged with the task. Two tasks never share a code field
  here, even on identical interval lists, so the (name, intervals) id
  ternary_key_fields uses would wrongly merge them."""
  return frozenset(
      (per_task_field(task, normalise_feature_name(feature)),
       math.ceil(max(len(intervals) - 1, 0) / 8))
      for feature, intervals in feature_intervals.items())


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
