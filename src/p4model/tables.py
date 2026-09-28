"""What one table costs: rows, blocks, and the crossbar key fields it claims.

The two accounting rules that matter, both measured rather than derived: blocks
charge crossbar BYTES per key FIELD (codeword_to_blocks), not raw codeword
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
  guard against a tail, not a routine constraint.

  Deliberately caller-less, the same precedent exact_match_resource_usage
  sets: nothing in the Optuna loop or the P4 generator calls this as a
  build-time assertion -- its only callers are the evaluation.py re-export
  and tests/test_evaluation.py. It answers the deployability question
  analytically and is kept for review and tests, not wired into the commit
  path where a design is actually accepted. If the tail above is ever
  observed to bind in practice, this is where the guard would be added;
  until then it is a documented oracle, not an enforced one."""
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

  range_entries is the EXPANDED PHYSICAL TCAM ROW COUNT range_entry_count
  gives for actually INSERTING every interval -- NOT a count of distinct
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

  Note: ternary_matching_resource_usage no longer applies a "+4" bit
  overhead anywhere -- that band-domain rule (RM-3 Design A) was retired
  along with codeword_bits_to_blocks's role in per-table pricing. The
  mandatory version field is priced inside codeword_to_blocks's crossbar-byte
  ledger instead (see tables.crossbar_capacity), which has nothing to add
  here."""
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
  included) -- NOT what alignment optimises against any more;
  codeword_to_blocks prices per-feature crossbar bytes instead (see its own
  docstring). This function survives for two reasons only: codeword_to_blocks
  and codeword_to_blocks_headline both special-case the empty key to
  codeword_bits_to_blocks(0), since a version field still needs a physical
  block even when the key claims none; and src/reporting/replay_scoring.py
  still derives its (unread) band_factor_before/after columns from it.

  (Was `band_factor` until 2026-09-14. Renamed, not changed: the name now
  states what it takes -- raw codeword BITS -- so it cannot be mistaken for
  the byte-wise arm below.)"""
  return math.ceil(
      (codeword_length + CODEWORD_KEY_OVERHEAD_BITS) / TCAM_BLOCK_KEY_LENGTH)


def ternary_key_field_bits(feature_intervals):
  """The bit width of every `meta.code_<feature>` field in one classification
  table's key, which is what codeword_to_blocks prices.

  Each field is declared `bit<len(intervals) - 1>` (build_p4_script.py:773-776).
  Widths are returned SORTED because the price depends only on the multiset
  of field widths, never on the order the generator happens to emit them in --
  the crossbar allocator is free to place fields where it likes, and measurably
  does (independent_low_sd6's 19-byte key lands on crossbar bytes 5-10, 12, 14,
  17-21 and 27-32, not on a contiguous run)."""
  return tuple(sorted(max(len(intervals) - 1, 0)
                      for intervals in feature_intervals.values()))


def crossbar_capacity(g):
  """The whole-byte crossbar slots `g` TCAM blocks supply: `5g + (g - 1) // 2`.

  Sec 13.1's model of a TCAM block, read from p4c's own assembly (`prog.bfa`,
  2026-09-20 rewrite design Sec 2): a block is one crossbar group (5 private
  bytes) plus, every other block, one shared nibble -- so g blocks own 5g
  private bytes and `(g - 1) // 2` extra whole bytes from those shared
  nibbles pairing up two at a time. This is the SIMPLER, corrected mechanism
  that replaced the retired `crossbar_groups_needed`/`_full_midbytes` pairing
  (a consecutive group RUN starting at an offset, with midbytes owned
  exclusively by group pairs) -- that premise was read out of the assembly as
  false: a block may pair with ANY of a stage's midbytes, not a fixed
  partner, and groups need not be consecutive. Sec 13.1 keeps only the
  resulting COUNT, not the retired geometry.

  Ten calibration keys pin the ladder this function drives across its first
  ten steps exactly: B = 10, 16, 21, 27, 32, 38, 43, 49, 54, 60 crossbar
  bytes need g = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 blocks respectively -- each B
  is exactly `crossbar_capacity(g)` for its g, i.e. the largest key that g
  blocks still hold (test_the_ladders_fixed_points_match_crossbar_capacity).

  g = 0 deliberately returns -1, not 0: `codeword_to_blocks_headline`'s
  empty-key note depends on the g = 0 candidate failing outright (a B = 0 key
  still needs a physical block for --version--, so "0 blocks satisfy B = 0"
  must never look true here)."""
  return CROSSBAR_PRIVATE_BYTES_PER_GROUP * g + (g - 1) // 2


def codeword_to_blocks_headline(field_bit_widths):
  """The Sec 13.1 headline model, `S = 0` form -- the one sentence the paper
  states: `blocks(B) = min g : 5g + floor((g - 1) / 2) >= B`, i.e. the
  smallest `g` for which `crossbar_capacity(g) >= B`.

  This is `codeword_to_blocks` with the Sec 2.3 isolation credit switched
  off -- the conservative arm of the refinement layer (Sec 13.1's "Off" row):
  exact on 92 of the 100 archived classification tables, over by exactly one
  block per tree on the other 8 (the two 33-byte, 15-feature keys), and NEVER
  observed to under-predict across 405 table observations (389 scorable, 16
  refused at > 62 crossbar bytes -- scripts/tcam_table_scoreboard.py).
  `codeword_to_blocks`
  never returns MORE than this function does -- see
  test_the_isolation_refinement_never_raises_the_headline_price.

  The empty-key case (`field_bit_widths == ()`, or any width multiset whose
  bytes sum to 0) is special-cased rather than left to the ladder: `g = 0` is
  the only candidate the ladder would try before `g = 1`, and
  `crossbar_capacity(0) == -1 < 0` is false however B = 0 is compared against
  it, so the ladder's own domain has no g = 0 answer to B = 0 -- a version
  field still needs a physical block even when the key itself claims none.
  `codeword_bits_to_blocks(0)` is that answer (1), kept in exactly one place
  so this function and codeword_to_blocks agree on it by construction."""
  key_bytes = codeword_fields_to_bytes_from_bits(field_bit_widths)
  if key_bytes == 0:
    return codeword_bits_to_blocks(0)
  g = 1
  while crossbar_capacity(g) < key_bytes:
    g += 1
  return g


def tail_is_isolatable(field_bits):
  """Whether p4c can split ONE nibble-clean field's leftover 1-4 bits off as
  a standalone free-nibble entry (Sec 2.3), rather than parking a whole extra
  byte on a shared midbyte to hold it.

  Only meaningful for a field whose last crossbar byte is a nibble
  (`1 <= field_bits % 8 <= 4`) -- callers filter to that set before asking.

  THE MECHANISM (model_audit_2026-09-27.md Appendix A; p4c
  `tofino/input_xbar.cpp`, `align_flags` l.461, `allocate_mid_bytes`
  l.1039-1090, `free_mid_bytes` l.879, the `allocTable` loop l.1361-1390 --
  midbytes are allocated BEFORE groups). Every crossbar slot has a lane
  (`slot number mod 4`); byte k of a 32-bit container may only sit in a
  lane-k slot, a 16-bit container's bytes only at matching parity, an 8-bit
  container's anywhere (Appendix A.3). A group supplies one slot of each
  lane plus one extra in the lane it starts on, so g groups cover 2 lanes at
  full capacity and the rest at g-1 -- an 11-byte key (3 W-container bytes
  each in 3 lanes, 2 in the 4th, Appendix A.5) fits 2 blocks only if its
  3-4-bit tail takes the spare midbyte nibble AND the other 10 bytes still
  fit those two groups' lanes; the tail helps only when removing it relieves
  an over-full lane (Appendix A.4). Two probes at IDENTICAL lane counts
  (w027 = 27+56 bits, w019 = 19+64 bits, both (3,3,3,2)) diverge purely on
  which lane loses its tail: w019's removal leaves (3,3,2,2), which two
  groups cover exactly -> isolatable, 2 blocks; w027's leaves (3,3,3,1),
  needing 3 extras where only 2 exist -> p4c falls back to parking a whole
  byte on a midbyte, filling both its nibbles and leaving none for the
  mandatory 2-bit --version-- field -> a version-only block -> 3, not
  isolatable (Appendix A.6).

  This function is a WIDTH-ONLY PROXY for that lane rule: it infers the
  tail's lane from `field_bits mod 32` alone, assuming the field starts at a
  container boundary (27 -> byte 3, not isolatable; 19 -> byte 2, single
  container, isolatable) rather than walking the real PHV layout. A lane
  checker built from the compile's actual layout agrees with this proxy on
  every real key measured (74/74) and is exact (57/57) where the proxy is
  conservative (48/57, always erring toward NOT isolatable, never the other
  way) on synthetic multi-field probe families the proxy was not built to
  cover. The unknown this function approximates is the PHV layout, not
  whether the outcome is predictable -- it is lane arithmetic, not (as an
  older draft of the P4 reference's Sec 4.1.1 "residual" claimed) the
  allocator's unpredictable greedy choice.

  Measured over 24 compiles (`scripts/tcam_phv_slice_sweep.py`; the CSV has
  24 rows, matching that script's own docstring -- 34 was a stale count
  quoted elsewhere and is not this probe's real size) against the direct
  observable `byte_group_holds_whole_byte`: 12 pays / 12 free / 0
  disagreements with the width table below.

    tail  = field_bits mod 32 (32 when the remainder is 0)
    index = ceil(tail / 8) - 1        # which container byte holds the tail

  | index (tail range)      | field fits ONE 32-bit container | spans 2+ |
  |--------------------------|----------------------------------|----------|
  | 0  (1-8 bits)            | isolatable                       | isolatable |
  | 1  (9-16 bits)           | isolatable                       | isolatable |
  | 2  (17-24 bits)          | isolatable                       | NOT        |
  | 3  (25-32 bits)          | NOT                               | NOT        |

  `scripts/tcam_field_count_sweep.py` separately confirmed this table is NOT
  extendable to predict which byte the allocator picks among several
  candidates once a key has 3+ fields (44 compiles, no feature tried
  separates the pays/free outcomes there) -- this function answers only the
  single-field isolatability question the table above states, never a
  multi-field placement choice."""
  tail = field_bits % 32 or 32
  index = math.ceil(tail / 8) - 1
  if index in (0, 1):
    return True
  if index == 2:
    return field_bits <= 32          # single 32-bit container
  return False                        # index == 3


def codeword_to_blocks(field_bit_widths):
  """TCAM blocks ONE table word of this key spans, version field included.
  Sec 2.2/13.1's ledger, the production price -- headline ladder
  (codeword_to_blocks_headline) plus the Sec 2.3 isolation credit, capped at
  one nibble-clean field per Sec 6.1 amendment 2.

  (This replaces the retired offset-taking `codeword_to_blocks` of
  2026-09-14: reading p4c's own assembly (2026-09-20 rewrite design Sec 2)
  showed the run-from-`start_group`/paired-midbyte premise that function and
  `crossbar_groups_needed`/`version_block_penalty` were built on is false --
  a block may pair with ANY of a stage's midbytes, groups need not be
  consecutive, and groups are shared between tables. There is no
  `start_group` parameter any more because the corrected mechanism has
  nothing for one to modify: a key's OWN price no longer depends on where it
  starts. The real, measured effect the old offset term was chasing --
  `(179, 204)` costing 9 blocks alone and 10 when a different key shares its
  stage -- has not vanished; it belongs to STAGE PLACEMENT, not to this
  per-table price, and moves to `src/p4model/packing.py`'s stage-sharing
  margin, Sec 13.2.)

  `B = sum(ceil(w / 8) for w in field_bit_widths)` crossbar bytes. `g` blocks
  supply `5g` private byte slots and (from `crossbar_capacity`) roughly one
  shared nibble per block pair. A whole overflow byte -- one that does not
  fit the private slots -- needs TWO nibbles; a nibble-clean overflow byte
  (Sec 2.3, `tail_is_isolatable`) needs only one; the mandatory 2-bit
  `--version--` field needs one more. So, with `overflow(g) = max(0, B - 5g)`
  and `S_usable` counting nibble-clean, isolatable fields:

    blocks = min g : 2*overflow(g) - min(overflow(g), S_usable, 1) + 1 <= g

  One amendment on top of the bare ledger, compiler behaviour rather than
  block structure -- the nibble-clean credit is capped at ONE
  (`min(overflow, S_usable, 1)`, not `min(overflow, S_usable)`) because p4c's
  allocator (`IXBar::allocate_mid_bytes`/`free_mid_bytes`) guarantees at most
  one nibble-only midbyte per table by construction (Sec 12.4) -- a second
  credit is untested by any corpus point and would under-predict 3.7% of
  random keys at B in {17, 28, 39, 50, 61}.

  p4c's own sizing loop (`IXBar::calculate_sizes`, input_xbar.cpp:507-511)
  separately never plans more than `ceil(g / 2)` midbytes for `g` groups
  (`overflow(g) <= ceil(g / 2)`), which the crossbar's bare structure would
  not by itself imply (a block can pair with any of a stage's 6 midbytes, so
  structure alone would allow up to 6 overflow bytes per stage). That clause
  is NOT in the formula above because it never binds: with `credit <= 1` the
  ledger clause already forces `overflow <= floor(g / 2) <= ceil(g / 2)`, so
  the sizing-loop limit is implied rather than a second, independent test --
  confirmed by brute force over 200 000 random keys, 0 differences with the
  clause present or absent.

  The empty-key case is a real early return, not something the ladder
  happens to produce -- see codeword_to_blocks_headline's docstring for why
  `g = 0` cannot answer `B = 0` within the ladder's own domain."""
  key_bytes = codeword_fields_to_bytes_from_bits(field_bit_widths)
  if key_bytes == 0:
    return codeword_bits_to_blocks(0)

  s_usable = sum(1 for bits in field_bit_widths
                 if 1 <= bits % 8 <= 4 and tail_is_isolatable(bits))

  g = 1
  while True:
    overflow = max(0, key_bytes - CROSSBAR_PRIVATE_BYTES_PER_GROUP * g)
    credit = min(overflow, s_usable, 1)
    if 2 * overflow - credit + 1 <= g:
      return g
    g += 1


def codeword_bytes_to_blocks(key_bytes):
  """TCAM blocks a SINGLE key_bytes-byte-wide crossbar field would span,
  under the pre-2026-09-21 model: ceil(key_bytes * 8 / TCAM_BLOCK_KEY_LENGTH).

  SUPERSEDED for classification tables, which key several per-feature fields
  at once and are priced by codeword_to_blocks's crossbar-byte ledger
  instead: doc §4.1's measured ladder disagrees with this function from 11
  bytes up (11 -> 3 blocks there, 2 here; 33 needs the Sec 2.3 isolation
  credit to reach 6, which this function reaches only because its own
  rounding is cruder). No production code calls it since audit C5
  (2026-09-28) deleted its last caller, packing.key_width's range-pool
  branch; the probe scripts (tcam_offset_harvest, tcam_spacer_sweep,
  tcam_version_sweep, tcam_group_cap_probe) still use it to size SOLID
  single-field keys, the one shape both rules price identically, since there
  is no second field for the crossbar-byte ledger's per-field rounding to
  diverge on.

  (Was `crossbar_block_width` until 2026-09-14. Renamed, not changed: the name
  now states what it takes -- crossbar key BYTES -- so the decomposition
  fields -> bytes -> blocks reads in one direction. It carries NO version
  charge: this function prices a key standalone, and what a key pays when it
  is placed behind a different key in its stage is a different quantity
  computed by a different module -- see packing.crossbar_stages_needed's
  ordered stage simulation and lanes.price_with_supply.)

  One block is fed by exactly one crossbar group, and a group delivers 5
  private bytes + 1 midbyte nibble = TCAM_BLOCK_KEY_LENGTH bits = 5.5 BYTES
  (reviews/p4_tofino_reference.md §4.1.1)."""
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

  # The per-tree block width: codeword_to_blocks prices the table's crossbar
  # key (one meta.code_<feature> field per selected feature) via the Sec 2.2/
  # 13.1 ledger -- see its own docstring for the formula and evidence.
  # codeword_bits_to_blocks (the bit-width band) no longer feeds this at all;
  # it survives only as the empty-key value codeword_to_blocks special-cases.
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
