import collections
import itertools
import math
from dataclasses import dataclass
import sklearn.metrics as mt
from src.p4gen.build_p4_script import (
    MAX_CODEWORD_LENGTH,
    MAX_NUM_FLOWS,
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
    _reject_colliding_feature_names,
    feature_intervals_from_nodes,
    generate_codewords,
    get_feature_intervals,
    get_root_to_leaf_paths,
    merge_tree_nodes,
    most_common_class_and_dropped_codewords,
    normalise_feature_name,
    tree_nodes_for,
)
from src.p4gen.feature_registers import (
    FEATURE_REGISTER_CATALOG,
    register_names_for,
    register_width_bits,
)
from p4.range_expansion import range_entry_count

TOFINO_PIPELINE_STAGES = 12   # Ref 5; hard, per Ref 7's tofino2h failure


class CodewordTooLong(RuntimeError):
  """Codeword exceeds MAX_CODEWORD_LENGTH. args = (message, codeword_length)."""


class CrossbarKeyTooWide(RuntimeError):
  """One table's match key exceeds the per-stage ternary crossbar byte budget;
  the compiler rejects such a table outright rather than splitting it.
  args = (message, byte_width)."""


@dataclass(frozen=True)
class StagePlan:
  """crossbar_stages_needed's placement, not just its size -- F10: the stage
  a pool is DONE at (depth) is not the same quantity as how many stages it
  OCCUPIES (occupied): a stage can fill at the 8-table crossbar cap and spill
  a table forward past every level actually requested, so depth must be read
  from where tables landed, not from max(readiness_levels) + 1."""
  occupied: int          # how many stage indices hold a table from this pool
  depth: int             # max(occupied index) + 1 -- the quantity a 12-stage ceiling reads
  indices: frozenset     # for assertions and debugging
  blocks: int            # total TCAM blocks actually CHARGED across every stage --
                         # not the naive per-table sum passed in as table_specs, since
                         # a ragged key's group-offset penalty (see crossbar_stages_needed's
                         # ragged_keys) is a per-STAGE placement fact, not a per-table one.
                         # Measured to matter: independent_low_sd5 (compiler_calibration_v6,
                         # 2026-09-06) reported 13 from the naive sum where p4c used 16.

  def __int__(self):     # transitional: `stages` is still the occupancy count
    return self.occupied


@dataclass(frozen=True)
class ResourceUsage:
  """multi_model_memory_evaluation's return (D1).

  Replaces a 3-tuple rather than growing it to eight positional values, for
  the same reason TrainResult replaced a 7-tuple: three of these fields are
  same-typed ints naming DIFFERENT stage quantities, and positional access
  is how they get confused. See this module's
  multi_model_memory_evaluation docstring for the four-quantity
  disambiguation.

  No __int__ is provided. StagePlan has one as a transitional shim;
  repeating it here would let a caller silently use the whole object where
  a count is meant.

  codeword_length : classification-table key width in bits, before the
                  CODEWORD_KEY_OVERHEAD_BITS the block factor adds. Under
                  'joint' this is THE pooled split-threshold count of the
                  merged tree set -- the quantity threshold alignment
                  actually shrinks, and the one src/training/align_budget.py
                  gates on. Under 'disjoint' the two models keep independent
                  codewords and no single value is "the" codeword: the MAX of
                  the two is reported, because that is the one the 512-bit
                  MAX_CODEWORD_LENGTH limit binds on. Do NOT derive disjoint
                  block counts from it -- each model's own factor is applied
                  to its own trees (see the disjoint branch below).

  register_depth, register_count and register_sram_bits (Task 6, Spec
  4.1/4.2/4.3) report the Tofino `Register<>` state this design needs, on
  top of the match-table quantities above:

    register_depth     : max readiness level (feature_readiness_level) over
                          the selected feature(s) -- how many pipeline
                          stages elapse before the LAST register in any
                          feature's dependency chain has run, i.e. before
                          ANY classification table is even allowed to read
                          a value. Reuses whatever readiness-level list the
                          stage-placement code already computed; no new
                          traversal.
    register_count      : count of distinct Register<> instances
                          (register_names_for) the selected feature(s)
                          resolve to, deduplicated by name -- shared
                          dependency registers (e.g. flow_last_arrival_time,
                          reused by flow_iat_max and flow_iat_mean) are
                          counted once, matching
                          generate_P4_registers_and_apply's own by-name
                          dedup, and deduplicated ACROSS both models too:
                          registers are a single physical resource shared
                          by the whole generated program regardless of
                          'joint' vs 'disjoint' ENCODING (that choice only
                          affects codeword/interval sharing, never register
                          generation -- see build_p4_script.py's
                          raw_feature_intervals, keyed on the union of both
                          models' raw feature names).
    register_sram_bits  : sum(register_width_bits(name) for each of the
                          register_count registers) * MAX_NUM_FLOWS -- the
                          total per-flow SRAM the catalog's registers
                          occupy (every register is a MAX_NUM_FLOWS-deep
                          array, one slot per tracked flow).
                          CATALOG-ONLY and a KNOWN UNDER-COUNT: the P4
                          generator always emits one more register,
                          flow_forward_srcaddr_reg (bit<32>,
                          build_p4_script.py:2018), unconditionally and
                          OUTSIDE FEATURE_REGISTER_CATALOG by design --
                          "neither is catalog-driven or routed through the
                          register_order/_note_touch dedup machinery"
                          (generate_P4_registers_and_apply's own docstring,
                          build_p4_script.py:1801-1808; feature_registers.py's
                          module docstring notes the same fact for the
                          "flows" bookkeeping register it wires) -- this
                          field misses that register's
                          32 * MAX_NUM_FLOWS bits every time.

  CAVEAT (Spec 4.3, applies to all three fields above): this reports
  register DEPTH and COUNT (how many stages, how many registers), NOT
  register CAPACITY. Tofino has a limited number of stateful ALUs per
  stage, and whether these registers actually FIT has never been measured
  in this repo -- do not read register_depth/register_count/
  register_sram_bits as a feasibility guarantee.

  range_depth  : StagePlan.depth for the range-matching pool ALONE (Task 6
                extended by the stage-depth-attribution design's Phase 1).
                Component of stage_depth, not itself a ceiling-checked
                quantity -- only stage_depth is.
  ternary_depth: StagePlan.depth for the ternary classification pool ALONE.
                Equal to stage_depth in every non-degenerate design (the
                classification pool is placed last, see stage_depth's own
                docstring); the max() stage_depth takes exists only for the
                degenerate zero-ternary-table case.
  range_tables : count of range-matching tables (len(range_table_specs)) --
                one per selected feature under 'joint' (the union of both
                models' features), one per (model, feature) pair under
                'disjoint' -- so on identical feature lists, disjoint's
                range_tables is at most twice joint's, with equality only
                when both models actually split on every listed feature.
  ternary_tables: count of classification tables (len(ternary_table_specs))
                -- one per tree in the merged tree set. Invariant across
                encoding for the same two forests (every tree gets its own
                table regardless of whether the codeword is shared).
  """
  stages: int           # occupied match-table stage COUNT
  blocks: int
  stage_depth: int      # pipeline DEPTH (StagePlan.depth), what the 12-stage ceiling reads
  range_entries: int
  ternary_entries: int
  codeword_length: int  # pooled split-threshold count under 'joint'; see docstring
  register_depth: int   # max readiness level over the selected feature(s); see class docstring
  register_count: int   # distinct Register<> instances, deduplicated by name across both models
  register_sram_bits: int  # catalog-only per-flow SRAM bits; under-counts flow_forward_srcaddr_reg
  range_depth: int       # StagePlan.depth, range pool alone; see class docstring
  ternary_depth: int     # StagePlan.depth, ternary pool alone; see class docstring
  range_tables: int      # len(range_table_specs); see class docstring
  ternary_tables: int    # len(ternary_table_specs); see class docstring


def accuracy_metrics(y_true, y_pred, task):
    """Return (accuracy, weighted_f1) for a given task.

    Label sets are duplicated from `src.training.incremental_metrics.TASK_LABELS` to avoid
    a p4gen→training dependency; equivalence is pinned by `test_the_incremental_metrics_equal_the_from_scratch_oracle_after_every_step`
    and `test_both_label_spaces_work` in `tests/test_incremental_metrics.py`.
    """

    if task == 'app':
        lab = [0, 1, 2]

    elif task == 'ddos':
        lab = [-1, 1]

    else:
        raise ValueError(
            "accuracy_metrics: unknown task {!r}; expected 'app' or 'ddos'".format(task))

    accuracy = mt.accuracy_score(y_true, y_pred)
    f1score = mt.f1_score(y_true, y_pred, labels=lab, average='weighted')

    return accuracy, f1score


# range_entry_count now lives in p4/range_expansion.py (imported above) so
# bfshell's embedded Python (no sklearn, hence no import of this module) can
# share the exact same implementation instead of carrying its own copy.


# Width of every per-feature value field the range-matching tables key on
# (build_p4_script.py:775 emits "bit<16> <feature>_val" for each selected
# feature). 16 bits is the project's decided feature precision; one range
# table keys on exactly one such field, hence 2 crossbar bytes per table.
FEATURE_VALUE_BIT_WIDTH = 16
RANGE_TABLE_KEY_BYTES = math.ceil(FEATURE_VALUE_BIT_WIDTH / 8)

# p4c's compile-time sizing rule for a range table (Ref 4.2, Ref 7 "Mechanism
# E"): one entry in every RANGE_WORST_CASE_ENTRY_FRACTION is assumed to need
# the worst-case row count for the key's nibble geometry, capped at
# RANGE_WORST_CASE_ROWS_CAP; the rest are priced at one row. See
# compiler_range_rows.
RANGE_WORST_CASE_ENTRY_FRACTION = 4
RANGE_WORST_CASE_ROWS_CAP = 8


MAX_RANGE_KEY_BITS = 19   # Ref 4.2: a 20-bit range key does not compile at all


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
  populated at runtime via the control plane, never `const entries` -- Ref
  4.4), so it cannot cost them exactly. It applies a fixed distributional
  guess instead: a quarter of the declared entries are priced at the
  worst-case row count for the key's nibble geometry, the rest at one row
  each.

  This is NOT interchangeable with range_entry_count. That one models
  expand_range(), the driver's exact per-value decomposition at INSERTION
  time; this one models the compiler's pessimistic pre-allocation. Blocks are
  the compiler's question -- using the driver's number to answer it
  under-counts (measured: a 478-entry table priced at 1 block against p4c's
  committed 3). Ref 4.2 and Ref 7 "Mechanism E".

  Reproduces all five of Ref 4.2's independently measured per-block
  capacities as the largest entry_count whose rows still fit 512: 512 (4-bit
  key), 342 (8-bit), 256 (12-bit), 206 (16-bit), 187 (19-bit)."""
  worst = min(RANGE_WORST_CASE_ROWS_CAP,
              2 * len(nibble_widths_for(key_bit_width)) - 1)
  quarter = entry_count // RANGE_WORST_CASE_ENTRY_FRACTION
  return quarter * worst + (entry_count - quarter)


def range_deployment_overflow(feature_intervals,
                              key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Features whose REAL intervals will not fit the blocks the compiler
  allocated for them: {feature: (rows_needed, rows_available)}, empty when
  every table fits.

  The second of two independent constraints, and the one blocks does not
  subsume. A committed block count is fixed in the binary -- the control
  plane cannot grow a table, it just gets "[Not enough space]" partway
  through insertion (Ref 4.2; that failure is literally how range_entry_count
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

  IMPORTANT (measured, reviews/p4_tofino_reference.md Sec 4.2): this
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


def ternary_table_key_bytes(feature_intervals):
  """Crossbar byte width of ONE classification table.

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
  return sum(math.ceil(max(len(intervals) - 1, 0) / 8)
             for intervals in feature_intervals.values())


# The non-codeword key bits every classification-table row carries alongside
# the codeword itself. Factored out of the inline `codeword_length + 4` this
# replaces so the band arithmetic lives in exactly one place -- src/training/
# align_budget.py gates C1's accuracy spending on it and must not re-declare
# it. Its physical origin is not documented in this repo; the value is
# pre-existing behaviour and is NOT changed here.
CODEWORD_KEY_OVERHEAD_BITS = 4


def band_factor(codeword_length):
  """How many TCAM_BLOCK_KEY_LENGTH-wide key blocks one classification-table
  row spans. THE step function alignment is optimising against: a shed bit is
  worth nothing unless it carries codeword_length across a band boundary, and
  then it is worth n_trees blocks at once."""
  return math.ceil(
      (codeword_length + CODEWORD_KEY_OVERHEAD_BITS) / TCAM_BLOCK_KEY_LENGTH)


def ternary_key_is_ragged(feature_intervals):
  """Does this classification table's key leave part-used crossbar bytes?

  Each `meta.code_<feature>` field is declared `bit<len(intervals) - 1>`
  (build_p4_script.py:773-776) and the crossbar byte-rounds every field
  separately, so a field whose width is not a multiple of 8 hands the
  crossbar one byte that is only partly used. Such a byte can ride a MIDBYTE
  -- half a crossbar byte, shared between two neighbouring groups (Ref 4.1.1)
  -- and that is what makes the table's cost depend on WHICH group it starts
  at: see crossbar_stages_needed's group-offset penalty.

  A key of solid, byte-multiple fields presents only fully-used bytes, needs
  only whole midbytes, and is priced the same at every offset -- measured,
  scripts/tcam_stretch_sweep.py's solid arm."""
  return any((max(len(intervals) - 1, 0)) % 8
             for intervals in feature_intervals.values())


def crossbar_block_width(key_bytes):
  """TCAM blocks one classification-table row spans because of the ternary
  input CROSSBAR, as opposed to because of its bit width.

  One block is fed by exactly one crossbar group, and a group delivers 5
  private bytes + 1 midbyte nibble = TCAM_BLOCK_KEY_LENGTH bits = 5.5 BYTES
  (Ref 4.1.1). The crossbar allocates per FIELD and byte-rounds each one, so
  what it charges is key_bytes (ternary_table_key_bytes), not the raw codeword
  length -- a table keying 15 separate code_<feature> fields totalling 205
  bits really presents 33 bytes = 264 bits and needs 6 blocks, not 5.

  This is the term band_factor misses, and it is why band_factor was
  accidentally right for years: on ONE dense wide codeword field byte-rounding
  is a no-op and the two agree exactly. They diverge as soon as the key is
  split per feature, which is what build_p4_script actually emits.

  Measured exact on 144 real compiled classification tables spanning three
  compile eras -- the whole observed key_bytes -> blocks ladder (4 -> 1,
  11 -> 2, 16 -> 3, 20 -> 4, 26 -> 5, 33 -> 6, 37 -> 7, 41 -> 8, 49 -> 9,
  52 -> 10, 60 -> 11) is single-valued and lands on this function. Ref 7
  "Mechanism D"."""
  return math.ceil(key_bytes * 8 / TCAM_BLOCK_KEY_LENGTH)


def ternary_matching_resource_usage(codewords, feature_intervals,
                                     use_default_action_discount=False):
  """Returns (ternary_entries, ternary_blocks, codeword_length,
  ternary_table_specs).

  Each tree gets its own independent classification table
  (build_p4_script.py:636-659), so ternary_table_specs is one
  (block_count, byte_width) pair per tree. All of those tables key on the
  same set of per-feature fields, so they share one byte width.

  Task 7: when use_default_action_discount is True, this ports Planter
  RF_EB's own discount (table_generator.py:408-431's default_vote =
  max(collect_votes, key=collect_votes.count)) into this accounting: for
  each tree, EVERY leaf whose class value is the most common among that
  tree's codewords[tree].values() (see
  build_p4_script.most_common_class_and_dropped_codewords) becomes the
  table's default_action instead of an explicit entry, so that one tree's
  entry count drops by however many leaves share the most-common class
  (not capped at 1) before it feeds into the block-count formula below.
  False (the default) is byte-identical to pre-Task-7 behavior -- every
  existing caller/test is unaffected."""

  ternary_entries, ternary_blocks = 0, 0
  ternary_table_specs = []
  codeword_length = len(next(iter(codewords[0].items()))[0])

  if codeword_length > MAX_CODEWORD_LENGTH:
    raise CodewordTooLong("Codewords are too long", codeword_length)

  table_bytes = ternary_table_key_bytes(feature_intervals)

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
  # width (band_factor, which carries the +4 version/valid nibble) and its
  # crossbar byte width. The byte arm is the one that binds on every table
  # this generator emits; the bit arm is kept because Ref 4.1's
  # single-field 41-bit and 88-bit measurements need it.
  factor = max(band_factor(codeword_length), crossbar_block_width(table_bytes))
  for tree in codewords:
    tree_entry_count = len(codewords[tree])
    if use_default_action_discount and tree_entry_count > 0:
      _, dropped_codewords = most_common_class_and_dropped_codewords(codewords[tree])
      tree_entry_count -= len(dropped_codewords)

    tree_blocks = math.ceil(tree_entry_count / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor

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


# Every per-flow register in this design is indexed by meta.flow_hash, so the
# hash occupies whole stages ahead of any register touch -- THREE of them, not
# the one this constant used to claim. Measured over all 121 range tables in
# the 19 real compiles of results/compiler_calibration/ (see
# reviews/p4_tofino_reference.md Sec 7): every committed placement opens with
# a metadata-init table at stage 0, tbl_calc_flow_hash$precompute at stage 1
# and tbl_calc_flow_hash at stage 2, so the first RegisterAction in any
# feature's chain lands at stage 3. `real_stage - level` had a floor of
# exactly +2 in every one of the 19 rows under the old value of 1; at 3 the
# floor is 0, i.e. levels now name the earliest stage the compiler really
# uses. (Values above the floor are tables the packer legitimately pushed
# later, which is placement, not origin.)
FLOW_HASH_LEVEL = 3

# The vote_app/vote_ddos tables read every tree's class and so always sit one
# stage past the last classification table. Measured: exactly 1 in all 19
# compiles, with no exceptions and no scaling. stage_depth is the quantity
# checked against TOFINO_PIPELINE_STAGES, so leaving this out understated the
# depth of every design by one whole stage.
VOTE_EPILOGUE_STAGES = 1


def feature_readiness_level(feature_name, catalog=None):
  """Earliest pipeline stage at which this feature's `_val` field -- and so
  its range-matching table -- can possibly be placed.

    level = FLOW_HASH_LEVEL
          + 1 if the feature is fwd- or bwd-gated (flow_orientation_action has
            to resolve meta.fwd before the gated block can run, regardless of
            which direction the gate checks)
          + one per RegisterAction in the feature's chain

  Each chain entry is a genuinely sequential stage: a "dependency" register
  produces meta.current_iat, which the paired "value" register consumes.
  A register shared between two features (flow_last_arrival_time, executed
  once for both flow_iat_max and flow_iat_mean) still sits on both features'
  critical paths, so it counts for both.

  A feature absent from the catalog gets no registers emitted at all
  (generate_P4_registers_and_apply silently skips it), so nothing gates its
  table beyond the hash itself -- true of THIS estimator's own placement
  logic, which still happily prices such a feature set. Task 5's F2 check
  closes the gap one level up, at generate_P4_code: it raises a hard error
  for exactly this case (an uncatalogued feature would otherwise compile to
  a declared-but-never-written field), so a design this function readily
  estimates a stage/block count for may be one the generator now refuses to
  emit at all.

  Validated against a real compile: this yields 3/3/3/4 for M2's feature set,
  matching the compiler's observed stage offsets 0/0/0/1 exactly."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  entry = catalog.get(normalise_feature_name(feature_name))
  if entry is None:
    return FLOW_HASH_LEVEL

  gate_cost = 1 if entry.get("gated_by") in ("fwd", "bwd") else 0
  return FLOW_HASH_LEVEL + gate_cost + len(entry["registers"])


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
  ternary_table_key_bytes(feature_intervals) by construction, which is the
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


# A Tofino stage has four stateful ("meter") ALUs, and every RegisterAction
# this generator emits occupies one for a whole stage. Read straight off the
# compiler's own arithmetic rather than fitted: mau.resources.log's percentage
# table reports a Meter ALU count of 4 as 100.00% (joint_high_sd7 stages 3-6,
# among others). Swept over 2/3/4/5/6/8 against the 18 committed calibration
# placements, only 4 reproduces the compiler's last-register stage on every
# row -- its neighbours manage 13, 11, 10 and 8 of 18.
METER_ALUS_PER_STAGE = 4

# flow_forward_srcaddr_reg backs flow_orientation_action, which resolves
# meta.fwd. generate_P4_registers_and_apply emits its .execute() call
# UNCONDITIONALLY into the apply block (build_p4_script.py:2092), not just
# when a gated feature is selected, so it always claims one stateful ALU in
# the first register stage and every fwd-/bwd-gated register waits a stage on
# it. It lives outside FEATURE_REGISTER_CATALOG (no feature owns it), so the
# schedule has to add it by hand.
ORIENTATION_REGISTER = "flow_forward_srcaddr"

# The order generate_P4_registers_and_apply lays the RegisterAction .execute()
# call sites down in: the unconditional ones straight into the apply block,
# then `if (meta.fwd == 1) { ... }`, then `if (meta.fwd == 0) { ... }`
# (build_p4_script.py's three _execute_lines calls, in exactly this order).
# It is a real ordering, not a presentation choice: p4c's table placer walks
# the control block with a work-list CURSOR, so it cannot begin the second
# gated block before the first one is fully placed.
_REGISTER_BLOCK_ORDER = (None, "fwd", "bwd")


def _register_blocks(features, catalog):
  """(register name -> which of _REGISTER_BLOCK_ORDER's blocks emits it,
  register name -> the register it waits on).

  Walks the features once per block, which is what
  generate_P4_registers_and_apply does -- it filters matched_features into
  ungated/fwd-gated/bwd-gated lists and calls _execute_lines on each in turn,
  sharing one already_executed_registers set. Grouping cannot change which
  block claims a shared register: build_p4_script raises ValueError outright
  for a register named by features in two different gate classes ("it would
  be .execute()d inside one gated block and read as garbage from the other"),
  so every register has exactly one well-defined block."""
  gate = {ORIENTATION_REGISTER: None}
  predecessor = {ORIENTATION_REGISTER: None}
  for block in _REGISTER_BLOCK_ORDER:
    for feature in features:
      entry = catalog.get(normalise_feature_name(feature))
      if entry is None:
        continue
      # An unrecognised gated_by is treated as ungated here rather than
      # dropped -- build_p4_script raises RuntimeError for it, and this
      # function must not silently price such a design as register-free.
      gated_by = entry.get("gated_by")
      if (gated_by if gated_by in ("fwd", "bwd") else None) != block:
        continue
      # First register of a gated chain waits on the orientation register:
      # meta.fwd has to resolve before the block it guards can run.
      previous = ORIENTATION_REGISTER if block in ("fwd", "bwd") else None
      for register in entry["registers"]:
        name = register["name"]
        # First writer wins, mirroring already_executed_registers: a register
        # shared between two features is emitted at its first call site.
        if name not in gate:
          gate[name] = block
          predecessor[name] = previous
        previous = name
  return gate, predecessor


def register_stage_schedule(features, catalog=None,
                            alus_per_stage=METER_ALUS_PER_STAGE):
  """Register base name -> the pipeline stage its RegisterAction runs in, for
  the whole set of features whose registers the generator will emit.

  feature_readiness_level models a register chain's DEPTH -- how many stages
  a single feature's registers must run back to back. This models the
  pipeline's register WIDTH, which is the constraint that actually binds at
  campaign scale: only `alus_per_stage` RegisterActions can issue per stage,
  so a design needing 16-20 registers cannot run them in the two or three
  stages its dependency chains alone would allow. The compiler spills the
  surplus forward and every range table keyed on a spilled register's output
  slides with it.

  Registers are emitted once however many features share them
  (register_names_for dedupes by name, matching
  generate_P4_registers_and_apply's `already_executed_registers` set), so a
  shared register is scheduled once and both its features read the same
  stage.

  A greedy earliest-level-first list schedule, run once per _REGISTER_BLOCK_
  ORDER block with each block floored at the last stage the previous one
  used. That floor is the control-flow half of the story: p4c's placer holds
  a single work-list cursor, so it only reaches `if (meta.fwd == 0)` once
  `if (meta.fwd == 1)` is fully placed, and a bwd register cannot run in a
  stage the fwd block has already moved past however many ALUs sit idle
  there. Inside one block the placer IS free -- every register in it is a
  candidate at once -- so the greedy runs unconstrained there.

  Ties inside a level do move individual registers around, but never the
  makespan -- the only quantity readiness levels take from this: verified
  over 300 random feature-order shuffles of every calibration row, where the
  last-register stage never moved. Measured against the 18 committed
  placements, the makespan matches the compiler's own on EVERY row
  (7,7,7,7,4,5,4,4,4,4,7,7,7,4,4,4,4,4), where chain depth alone reports 5 on
  each of the six k>=13 rows. Adding the per-block floor left all 18 of those
  intact while raising per-register agreement from 88 to 148 of 170, and it
  is what makes gated_block_interior_stages exact -- see
  reviews/p4_tofino_reference.md Sec 7.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # Pass 1: collect every register the generator will emit, which block emits
  # it, and the register each one waits on -- its predecessor inside its own
  # feature's chain, or the orientation register for a gated feature's first.
  gate, predecessor = _register_blocks(features, catalog)

  # Pass 2: dependency level (what feature_readiness_level already charges).
  def level_of(name, seen=()):
    parent = predecessor.get(name)
    # `seen` guards a catalog that lists a register as its own ancestor;
    # generate_P4_registers_and_apply would emit unreachable P4 for that, but
    # this function must not hang while a caller is merely pricing it.
    if parent is None or name in seen:
      return FLOW_HASH_LEVEL
    return level_of(parent, seen + (name,)) + 1

  levels = {name: level_of(name) for name in predecessor}

  # Pass 3: how many stages of work still hang off each register. Standard
  # critical-path list scheduling: among registers ready in the same stage,
  # the one with the longest chain behind it goes first. It matters here for
  # one register in particular -- ORIENTATION_REGISTER carries every gated
  # feature's whole chain, so delaying it to make room for some leaf register
  # would push that entire subtree back a stage. The compiler agrees: it sits
  # in the first register stage in all 18 committed placements.
  # A predecessor always has a strictly lower level than the register that
  # waits on it, so walking levels downward visits every child before its
  # parent and one pass suffices.
  height = {name: 0 for name in levels}
  for name in sorted(levels, key=levels.get, reverse=True):
    parent = predecessor.get(name)
    if parent is not None:
      height[parent] = max(height[parent], height[name] + 1)

  # Pass 4: list schedule, one block at a time in emission order. Within a
  # block, sorted by level so a predecessor is always placed before anything
  # that waits on it; then by height, then by name so the result never
  # depends on feature iteration order. `floor` carries the cursor forward:
  # no register of a later block may precede the last stage an earlier one
  # used. ALU load is shared across blocks -- there is one set of four
  # stateful ALUs per stage, whichever block's register claims it.
  placed, load, floor = {}, collections.Counter(), 0
  for block in _REGISTER_BLOCK_ORDER:
    names = [name for name, owner in gate.items() if owner == block]
    if not names:
      continue
    for name in sorted(names, key=lambda n: (levels[n], -height[n], n)):
      stage = max(levels[name], floor)
      parent = predecessor.get(name)
      if parent is not None:
        stage = max(stage, placed[parent] + 1)
      while load[stage] >= alus_per_stage:
        stage += 1
      placed[name] = stage
      load[stage] += 1
    floor = max(placed[name] for name in names)
  return placed


def gated_block_interior_stages(features, catalog=None,
                                alus_per_stage=METER_ALUS_PER_STAGE):
  """The stage indices no match table can be placed in, because the placer
  spends the whole of each of them inside a gated register block.

  Tofino has no program counter: every table hands the next stage a
  next-table pointer, so p4c's placer walks the control block with a work-
  list cursor and a table is a placement candidate only once the cursor
  reaches it. generate_P4_registers_and_apply emits the gated register blocks
  BEFORE every match table, so while the cursor is inside `if (meta.fwd ==
  1) { ... }` the entire range and classification pools are out of reach --
  regardless of how empty the stage is. In independent_high_sd10's stage 5
  the TCAM is 0/24, the ternary crossbar 0/66 and the logical table IDs 4/16.
  This is a control-flow constraint, not a capacity one; there is no
  range-pool fill limit.

  A block costs a stage only where the cursor is fully INTERIOR to it. In the
  stage the cursor descends into the block, outer tables can still be
  back-filled (p4c logs them verbatim as "potential backfill ... before
  tbl_prog951"), and in the stage it pops back out they are candidates
  again -- so a block spanning stages [first, last] blocks exactly
  range(first + 1, last), and a two-stage block costs nothing at all.

  Measured against all 18 calibration rows: the range pool's committed
  occupancy has a hole on exactly 5 of them, and this returns precisely those
  holes -- right rows, right indices, nothing on the other 13. Two of the
  five (independent_high_sd6, joint_high_sd8) still cost +0 stages overall
  because their range tables were not going to occupy that stage anyway,
  which is why the caller must apply this as a placement constraint and never
  as a per-row penalty. See reviews/p4_tofino_reference.md Sec 7,
  "Mechanism B"."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  gate, _ = _register_blocks(features, catalog)
  placed = register_stage_schedule(features, catalog, alus_per_stage)

  interior = set()
  for block in _REGISTER_BLOCK_ORDER:
    # The unconditional registers sit in the OUTER sequence; the cursor is
    # never "inside" anything while placing them, so they hide nothing.
    if block is None:
      continue
    stages = [placed[name] for name, owner in gate.items() if owner == block]
    if len(stages) > 1:
      interior.update(range(min(stages) + 1, max(stages)))
  return frozenset(interior)


def readiness_levels_for(feature_intervals, catalog=None, emitted_features=None):
  """One readiness level per feature, positionally aligned with
  range_matching_resource_usage's range_table_specs (both follow
  feature_intervals iteration order).

  A feature's level is one stage past its LAST register in
  register_stage_schedule -- so it carries both the chain depth
  feature_readiness_level charges and the stateful-ALU serialisation that
  function cannot see (it prices one feature at a time; the ALU cap is a
  property of the whole selected set).

  emitted_features names the features whose registers the generator will
  actually emit, when that is a SUPERSET of feature_intervals. Under
  'disjoint' encoding each model keeps its own intervals but both share one
  register block and one set of stateful ALUs, so each model's levels must be
  read off a schedule built from the union; pricing a model against a private
  pipeline would understate the pressure. Defaults to feature_intervals
  itself, which is already the merged set on the 'joint' branch."""
  placed = register_stage_schedule(
      list(emitted_features if emitted_features is not None else feature_intervals),
      catalog)
  entries = FEATURE_REGISTER_CATALOG if catalog is None else catalog

  levels = []
  for feature in feature_intervals:
    entry = entries.get(normalise_feature_name(feature))
    if entry is None or not entry["registers"]:
      # No registers emitted at all, so nothing gates this feature's table
      # beyond the hash -- feature_readiness_level's own fallback.
      levels.append(feature_readiness_level(feature, catalog))
      continue
    levels.append(max(placed[r["name"]] for r in entry["registers"]) + 1)
  return levels


def fits_two_columns(block_widths, rows=TCAM_ROWS_PER_STAGE,
                     columns=TCAM_COLUMNS_PER_STAGE):
  """Can these tables share one stage's TCAM?

  A stage's TCAM_BLOCKS_PER_STAGE blocks are not one pool: mau_spec.h:88-90
  gives 12 rows x 2 columns, and a table needing several blocks chains them
  down ONE column. So what decides a stage is the WIDTHS, not the total --
  three 8-block tables total exactly 24 and need two stages (8+8 overflows a
  12-row column) while four 6-block tables, also 24, fit in one (6+6 | 6+6).
  Measured against real p4c over synthetic tables of 5..12 blocks all keyed on
  one shared field, so TCAM blocks rather than the crossbar bound the result:
  scripts/tcam_column_sweep.py, reviews/p4_tofino_reference.md Sec 7
  "Mechanism C". Consistent with every one of the 18 calibration rows'
  committed placements, in both pools, with no exceptions.

  Exact subset-sum over achievable column loads, not a greedy fit: with two
  columns and a handful of tables the state space is trivial, and a greedy
  would report false violations. That matters historically -- the column rule
  was once written off as "refuted" on the strength of a per-width shortcut
  (`2*floor(12/w)`) applied to stages holding mixed widths, where the real
  packing is far more permissive: (7+5 | 6+6) fits four tables the shortcut
  rejects.

  Callers must pass shards no wider than a column (`_stage_shards` guarantees
  it). A single table needing MORE than a column really does span both --
  measured at 14, 16 and 24 blocks, each compiling into one stage -- and
  _stage_shards models that by splitting it into column-sized pieces."""
  widths = [int(w) for w in block_widths]
  if any(w > rows for w in widths):
    return False
  loads = {(0,) * columns}
  for width in sorted(widths, reverse=True):
    nxt = set()
    for load in loads:
      for i in range(columns):
        if load[i] + width <= rows:
          bumped = list(load)
          bumped[i] += width
          nxt.add(tuple(sorted(bumped)))
    if not nxt:
      return False
    loads = nxt
  return True


def _stage_shards(block_count, byte_width):
  """Splits one logical table into pieces no wider than a single TCAM column,
  so the packer never reports a stage count below what the table alone already
  forces.

  Splitting a table's ROWS is real: a table needing more blocks than one
  column holds genuinely spreads them further, each shard still carrying the
  table's full key width. Splitting a table's KEY across stages is not real: a
  key is one indivisible match, TCAM compares a whole row in one clock, and a
  stage's crossbar physically cannot deliver more than
  TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes -- a table whose key exceeds that
  budget is rejected by the compiler outright, not spread across stages. Raise
  rather than silently pricing that impossible design as
  `ceil(byte_width / budget)` stages.

  The split is at TCAM_ROWS_PER_STAGE (one column), not TCAM_BLOCKS_PER_STAGE
  (the whole stage), because fits_two_columns places each shard inside one
  column and could never place a wider one -- under eager placement an
  unplaceable shard would advance its stage index without end. Column-sized
  shards also reproduce the measurement: a 24-block table becomes 12 | 12,
  which fills both columns of ONE stage, exactly as real p4c does with single
  tables of 14, 16 and 24 blocks."""
  if byte_width > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), byte_width)
  n = max(1, math.ceil(block_count / TCAM_ROWS_PER_STAGE)) if block_count > 0 else 1
  return [(math.ceil(block_count / n), byte_width)] * n


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           unavailable_stages=frozenset(), ragged_keys=None):
  """Packs independent match tables into pipeline stages under ALL three
  per-stage hardware limits simultaneously, and returns a StagePlan
  describing where the tables landed (not just how many stages that took).

  table_specs is one (block_count, byte_width) pair per independent P4
  table -- one per tree for the ternary classification tables
  (build_p4_script.py:636-659), one per feature for the range-matching
  tables (build_p4_script.py:663-674).

  key_fields (optional) is one frozenset of (field_id, field_bytes) per
  table, positionally aligned with table_specs, naming WHICH crossbar
  fields that table's key is made of. It exists because the Ternary Match
  Input crossbar charges per distinct FIELD placed on its byte slots, not
  per (table, field) pair: two tables in the same stage that match on the
  same field read the same slots and the field is charged ONCE. That is not
  a corner case in this generator -- every classification table of one task
  keys on the identical meta.code_<feature> field set (see
  ternary_matching_resource_usage), and under 'disjoint' both models' range
  tables for a shared feature key the identical meta.<feature>_val field
  (build_p4_script._resolve_disjoint_feature_plan: the raw value field is
  always shared, and a code_<feature> field is shared whenever both models
  select the feature with identical intervals).

  Measured against 19 real p4c compiles (results/compiler_calibration/, and
  reviews/p4_tofino_reference.md Sec 7): joint_low_sd7's stage 7 holds four
  tables on one 32-byte codeword and the compiler reports 32 crossbar bytes,
  not 128; independent_low_sd6's stage 7 holds two 19-byte and two 4-byte
  tables and reports 23, not 46. Summing per table over-counted stages by up
  to 6 on that sample -- for a 41-byte codeword the sum-based rule computes
  floor(64/41) = 1 table per stage where the hardware takes 2 or more.

  Passing key_fields=None keeps the old, conservative per-table accounting
  exactly: each table is given a private synthetic field, so the union
  arithmetic below reduces to the sum it replaces. Every pre-existing caller
  is therefore unaffected.

  unavailable_stages (optional) names stage indices no table may occupy,
  whatever their capacity -- gated_block_interior_stages produces them.
  It is only meaningful alongside readiness_levels: without those the
  returned plan has no absolute stage indices to exclude, so the pure packer
  below ignores it.

  ragged_keys (optional) is one bool per table, positionally aligned with
  table_specs, saying whether that table's key leaves part-used crossbar
  bytes (ternary_key_is_ragged). It exists because a table's BLOCK COUNT is
  not a property of the table alone: the crossbar hands out groups in one
  consecutive run per key, so the second distinct key in a stage starts at
  the group offset the first one ended at, and a ragged key starting on an
  ODD group reaches one midbyte fewer than it does at offset 0 -- the midbyte
  at its low end is shared with the previous key's last group. p4c then pads
  the table with a whole extra TCAM. Confirmed from the pack format, not
  inferred: sharing, the table's memory unit 0 holds the 2-bit --version--
  field and NOTHING else (bits [41:0] empty); alone, unit 0 holds version
  plus 40 bits of match data. That is TableFormat::ternary_version()
  push_back()ing a block because no midbyte nibble was left for version --
  the same waste reviews/github_issue_tcam_version_bit_packing.md documents,
  reached by a new trigger.

  Measured directly, scripts/tcam_stretch_sweep.py, one artifact set:
  a table keying 179 + 204 bits (49 crossbar bytes, 9 groups) costs 9 TCAMs
  ALONE in a stage and 10 when a 12-byte key holds groups 0..2 ahead of it;
  the same block-and-byte geometry built from SOLID single fields costs 9
  either way. That +1 is the whole of independent_low_sd9's stage divergence
  (10 blocks + 5 x 3 = 25 > TCAM_BLOCKS_PER_STAGE), the last one the
  compiler-calibration study had open. Ref 7 "Mechanism G".

  Passing ragged_keys=None -- the default -- charges every table its declared
  block count at every offset, i.e. exactly the pre-existing pricing.

  RM-5/RM-6/RM-7 measured these limits
  on the Ternary Match Input crossbar specifically. A follow-up compile
  sweep (reviews/open_issues.md item 3, results_rmx_crossbar.csv) confirmed
  the 8-tables/stage cap generalizes to range tables at 16-bit width, but
  found range tables cost ~2x the crossbar xbar-units per byte that ternary
  tables do at the same width -- so reusing TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
  verbatim for the range pool is NOT known-conservative; once a design's
  range-table byte-budget (rather than the 8-table cap) becomes the binding
  constraint, this function likely UNDER-counts range_stages instead of
  over-counting it. The exact byte-width crossover for range tables was not
  pinned down (needs a >64-bit combined-width multi-field range sweep).
  Both pools are packed separately with this one function, since they are
  physically distinct table pools.

  Every stage must satisfy at once:
    * <= TCAM_BLOCKS_PER_STAGE                 TCAM blocks
    * <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE independent tables
    * <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE  bytes of DISTINCT key fields
                                               present in the stage (see
                                               key_fields above; without it,
                                               each table's key is its own)
  (the table-count and byte caps are RM-5/RM-6/RM-7,
  reviews/t12_required_changes.md Section 1.3, confirmed across key widths
  8-512 bits.)

  These constraints are NOT separable: solving each relaxation alone and
  taking the max can under-count. Counterexample -- tables
  (20 blocks, 5 B), (20, 5), (1, 60): the blocks-only bound is
  ceil(41/24) = 2 and the crossbar-only bound is 2, but no two of the three
  fit in one stage (20+20 = 40 blocks > 24; 5+60 = 65 bytes > 64), so the
  true answer is 3.

  Uses first-fit-decreasing, sorting by each table's most-loaded dimension
  (its largest fraction of a per-stage limit) descending: the hardest
  tables to place go first, which is what makes FFD behave well when the
  binding dimension differs from table to table. Sort order only affects
  tightness, never validity -- FFD only ever emits a packing in which every
  stage respects all three limits, so its stage count is always an upper
  bound on the true optimum. That is the safe direction for an estimator
  that must never under-count real hardware usage."""

  if key_fields is not None:
    if len(key_fields) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d key_fields for %d table_specs; the "
          "two must be positionally aligned, one entry per table"
          % (len(key_fields), len(table_specs)))
    for idx, ((_, byte_width), fields) in enumerate(zip(table_specs, key_fields)):
      declared = sum(field_bytes for _, field_bytes in fields)
      if declared != byte_width:
        raise ValueError(
            "crossbar_stages_needed: table %d's key_fields sum to %d crossbar "
            "bytes but its spec declares %d -- both describe the same key, so "
            "a mismatch would mis-price every stage the table lands in"
            % (idx, declared, byte_width))

  if ragged_keys is not None and len(ragged_keys) != len(table_specs):
    raise ValueError(
        "crossbar_stages_needed: got %d ragged_keys for %d table_specs; the "
        "two must be positionally aligned, one flag per table, or a table "
        "would be priced against another table's key shape"
        % (len(ragged_keys), len(table_specs)))

  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    # A caller that names no fields gets a PRIVATE synthetic one per table,
    # which makes the union accounting below numerically identical to the
    # per-table sum it replaces -- that is what keeps key_fields=None exactly
    # backwards compatible rather than approximately so.
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    ragged = bool(ragged_keys[idx]) if ragged_keys is not None else False
    for shard in _stage_shards(block_count, byte_width):
      shards.append((shard[0], shard[1], idx, fields, ragged))

  def load(shard):
    blocks, width, _, _, _ = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  def crossbar_bytes(fields):
    return sum(field_bytes for _, field_bytes in fields)

  def offsets_for(key_order):
    """Group offset each key starts at, given the order the crossbar hands
    groups out in. One group feeds one block and a key's groups are
    consecutive, so a key set's group count IS its per-table block count --
    counted once however many tables share it, since the crossbar charges per
    field, not per (table, field)."""
    offsets, running = {}, 0
    for key, key_blocks in key_order:
      offsets[key] = running
      running += key_blocks
    return offsets

  def charged(offsets, blocks, fields, ragged):
    return blocks + 1 if ragged and offsets[fields] % 2 else blocks

  def fits(stage, blocks, fields, ragged):
    # stage[3] is the per-shard (blocks, key, ragged) already here, stage[4]
    # the distinct keys with their group counts. A new key shifts every later
    # key's offset, so the whole stage is re-priced against the key set it
    # would HAVE -- a table already placed can become more expensive.
    #
    # Every ORDER of those keys is tried and the stage fits if any one of them
    # packs. The crossbar hands out groups in the order tables enter the
    # stage, which is p4c's placement order, not this packer's; assuming one
    # order would invent refusals the compiler does not make (measured: probe
    # point ragged_ax1_bx4 fits with the narrow key first and does not with
    # the wide one first). Distinct keys per stage are one to three here, so
    # this is a handful of permutations.
    keys = list(stage[4])
    if all(key != fields for key, _ in keys):
      keys.append((fields, blocks))
    if (crossbar_bytes(stage[1] | fields) > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
        or stage[2] + 1 > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE):
      return False
    shards_here = stage[3] + [(blocks, fields, ragged)]
    for key_order in itertools.permutations(keys):
      offsets = offsets_for(key_order)
      # The TCAM test is a column PACKING, not a running total against
      # TCAM_BLOCKS_PER_STAGE -- see fits_two_columns: three 8-block tables
      # sum to exactly 24 and still do not fit, four 6-block ones do.
      if fits_two_columns([charged(offsets, *shard) for shard in shards_here]):
        return True
    return False

  def place(stage, blocks, fields, ragged):
    stage[0] += blocks
    stage[1] |= fields
    stage[2] += 1
    stage[3].append((blocks, fields, ragged))
    if all(key != fields for key, _ in stage[4]):
      stage[4].append((fields, blocks))

  def opened(blocks, fields, ragged):
    return [blocks, set(fields), 1, [(blocks, fields, ragged)],
            [(fields, blocks)]]

  def stage_charged_blocks(stage):
    """The TCAM blocks ONE finished stage actually costs, ragged charge
    included -- the same question `fits()` already answers for placement,
    asked once more after the fact so the total can be reported.

    Tries every order of the stage's distinct keys, exactly as `fits()`
    does, and keeps the LARGEST total that still respects fits_two_columns.
    Not the order actually used to justify placement (place() only ever
    records arrival order, and fits() may have accepted a stage via a
    DIFFERENT permutation than that): this packer does not know which
    order p4c's placer will pick (see fits()'s own docstring), and 'largest
    feasible' is this module's standing rule for that uncertainty --
    consistent with never under-counting real hardware (crossbar_stages_
    needed's own FFD-upper-bound rationale). Validated against ground
    truth on the one case with more than one feasible order (ragged_ax1_
    bx4): the cheaper order (25) is infeasible by fits_two_columns, so the
    max over FEASIBLE orders is 22 -- exactly what p4c compiled.

    At least one permutation is guaranteed feasible: this stage exists
    because place()/opened() only ever commit a shard once `fits` (the
    same search) found one."""
    best = None
    for key_order in itertools.permutations(stage[4]):
      offsets = offsets_for(key_order)
      charged_list = [charged(offsets, blocks, fields, ragged)
                      for blocks, fields, ragged in stage[3]]
      if fits_two_columns(charged_list):
        total = sum(charged_list)
        best = total if best is None else max(best, total)
    return best

  if readiness_levels is None:
    # entry: [blocks_used, fields_present, tables_used, shards, key_order]
    stages = []
    for blocks, _width, _idx, fields, ragged in sorted(shards, key=load,
                                                       reverse=True):
      for stage in stages:
        if fits(stage, blocks, fields, ragged):
          place(stage, blocks, fields, ragged)
          break
      else:
        stages.append(opened(blocks, fields, ragged))

    return StagePlan(occupied=len(stages), depth=len(stages),
                      indices=frozenset(range(len(stages))),
                      blocks=sum(stage_charged_blocks(stage) for stage in stages))

  # Dependency-aware placement. Three differences from the packer above, all
  # chosen to track the REAL compiler rather than the theoretical optimum:
  #
  #   1. A table may not occupy a stage index below its readiness level --
  #      its key value literally does not exist yet.
  #   2. Placement is EAGER (earliest legal stage with room), not "pack as
  #      few stages as possible". The optimum would drop every table into the
  #      single latest stage; the compiler does not do that, and neither does
  #      this. Measured: M2's range pool really occupies 2 stages, which only
  #      eager placement reproduces.
  #   3. A table may not occupy an unavailable_stages index at all. Those are
  #      stages the placer spends entirely inside a gated register block, so
  #      no table of the outer sequence is even a candidate there -- see
  #      gated_block_interior_stages. Unlike the three capacity limits this
  #      is not a fullness test: an interior stage stays unusable however
  #      empty it is, so it is checked before `fits` rather than through it.
  #
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}  # index -> [blocks, fields_present, tables, shards, key_order]
  ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]], -load(s)))
  for blocks, _width, table_idx, fields, ragged in ordered:
    index = readiness_levels[table_idx]
    while (index in unavailable_stages or
           (index in by_index
            and not fits(by_index[index], blocks, fields, ragged))):
      index += 1
    if index in by_index:
      place(by_index[index], blocks, fields, ragged)
    else:
      by_index[index] = opened(blocks, fields, ragged)

  return StagePlan(occupied=len(by_index),
                    depth=(max(by_index) + 1) if by_index else 0,
                    indices=frozenset(by_index.keys()),
                    blocks=sum(stage_charged_blocks(stage)
                              for stage in by_index.values()))


def single_model_memory_evaluation(clf, selected_features, use_default_action_discount=False):
  """use_default_action_discount: opt-in, passed straight through to
  ternary_matching_resource_usage (which has implemented the Planter-style
  discount since Task 7 but was never reachable from this estimator). False
  -- the default -- reproduces every pre-existing caller's numbers exactly."""
  # Same guard get_feature_intervals runs before its own tree_nodes_for call
  # (Task 4) -- this function bypasses get_feature_intervals entirely (Task
  # 16) so it must run the check itself, or two differently-spelled feature
  # names that normalise to the same key silently merge their intervals with
  # no raise (final-review finding #1).
  _reject_colliding_feature_names(selected_features)
  tree_nodes = tree_nodes_for(clf, selected_features)
  feature_intervals = feature_intervals_from_nodes(tree_nodes)
  range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

  paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
  codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
  ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
      codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

  return (range_entries, range_blocks, ternary_entries, ternary_blocks, codewords, codeword_length,
          range_table_specs, ternary_table_specs)


def multi_model_memory_evaluation(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding,
                                  use_default_action_discount=False):
  """Returns ResourceUsage(stages, blocks, stage_depth, range_entries, ternary_entries,
  register_depth, register_count, register_sram_bits) -- FOUR related but DISTINCT
  stage-index quantities (F6, extended by Task 6) are in play below: this
  function is the source of truth for THREE of them (stages, stage_depth,
  register_depth) -- the fourth, stages_real, comes from the real compiler,
  not from this function (see its own paragraph below):

    stages      : OCCUPIED match-table stage count -- how many distinct
                  stage indices actually hold a table from either pool
                  (range_plan.occupied + ternary_plan.occupied). M2 example: 3.
                  This is what gets written to the campaign CSV's `stages`
                  column and plotted -- it is NOT a pipeline-depth quantity
                  and must never be compared against TOFINO_PIPELINE_STAGES.
    blocks      : total blocks used by both range and ternary tables
    stage_depth : pipeline DEPTH, max(occupied stage index) + 1 -- the
                  quantity a hard stage ceiling actually reads (F5). Read
                  from ternary_plan.depth (the classification pool is placed
                  LAST, after the range pool, so its depth is the overall
                  pipeline depth), defensively widened to
                  max(range_plan.depth, ternary_plan.depth) so an
                  (unrealistic) model with no ternary tables at all still
                  reports a sane depth, plus VOTE_EPILOGUE_STAGES for the
                  vote tables' own trailing stage. M2 example: 9 -- which is
                  what the real compiler needs for M2 as well. It read 6
                  before the 2026-09-05 calibration corrected FLOW_HASH_LEVEL
                  and added the epilogue; anything comparing stage_depth
                  against pre-2026-09-05 archived campaign numbers is
                  comparing two different definitions.
    register_depth : max readiness level (feature_readiness_level) over the
                  selected feature(s) -- how many stages elapse before the
                  LAST register a classification table depends on has run.
                  Related to stage_depth (both are stage-index quantities
                  gated by the same register-dependency model) but NOT the
                  same number: stage_depth also accounts for crossbar
                  packing/spill of the match tables themselves, which
                  register_depth does not. See ResourceUsage's own
                  docstring for register_depth/register_count/
                  register_sram_bits and their capacity caveat (Spec 4.3).
    range_entries  : count of physical rows across all range tables
    ternary_entries: count of ternary codewords across all classification trees
    (a fourth quantity, `stages_real` -- the REAL compiler's whole-program
    stage count including parsing/bookkeeping overhead this function does
    not model at all -- is NOT returned here; see p4_compile.parse_compile_logs,
    which stores it. M2 example: 9. `stages` and `stages_real` sit side by
    side in the same campaign dataframe row and are NOT the same quantity --
    plotting them together as if they were reads as the model being "67%
    wrong" when they are not even measuring the same thing. `stage_depth` and
    `stages_real` ARE comparable, and since 2026-09-05 they agree on M2; the
    residual on the 19-row calibration sample is 0-3 stages, always with
    stages_real the larger -- unmodelled PHV container conflicts and TCAM
    column geometry, see scripts/compiler_calibration.replay_stage_depth.)

  use_default_action_discount: opt-in, threaded down to
  ternary_matching_resource_usage under BOTH encodings -- directly for
  'joint' (which does its own ternary accounting on the merged tree set),
  and via both nested single_model_memory_evaluation calls for 'disjoint'.
  False -- the default -- reproduces every pre-existing caller's numbers
  exactly."""

  if encoding == 'joint':
    # Same union-based guard get_joint_feature_intervals runs before its own
    # tree_nodes_for calls (Task 4) -- this branch bypasses
    # get_joint_feature_intervals entirely (Task 16) so it must run the
    # check itself, on the SAME union both models' selected features form,
    # or a cross-model spelling collision (e.g. 'Flow.IAT.Max' in one model,
    # 'Flow IAT Max' in the other) silently merges with no raise
    # (final-review finding #1).
    _reject_colliding_feature_names(list(selected_features_app) + list(selected_features_ddos))
    tree_nodes = merge_tree_nodes(
        tree_nodes_for(clf_app, selected_features_app),
        tree_nodes_for(clf_ddos, selected_features_ddos))

    feature_intervals = feature_intervals_from_nodes(tree_nodes)
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
        codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

    range_levels = readiness_levels_for(feature_intervals)
    emitted_features = list(feature_intervals)
    register_names = register_names_for(feature_intervals)
    # One merged interval set, so every tree keys on the same fields.
    range_fields = range_key_fields_for(feature_intervals)
    ternary_fields = [ternary_key_fields(feature_intervals)] * len(ternary_table_specs)
    ternary_ragged = ([ternary_key_is_ragged(feature_intervals)]
                      * len(ternary_table_specs))

  elif encoding == 'disjoint':

    (range_entries_app, range_blocks_app, ternary_entries_app, ternary_blocks_app,
     codewords_app, codeword_length_app,
     range_table_specs_app, ternary_table_specs_app) = single_model_memory_evaluation(
        clf_app, selected_features_app, use_default_action_discount=use_default_action_discount)
    (range_entries_ddos, range_blocks_ddos, ternary_entries_ddos, ternary_blocks_ddos,
     codewords_ddos, codeword_length_ddos,
     range_table_specs_ddos, ternary_table_specs_ddos) = single_model_memory_evaluation(
        clf_ddos, selected_features_ddos, use_default_action_discount=use_default_action_discount)

    range_blocks = range_blocks_app + range_blocks_ddos
    range_entries = range_entries_app + range_entries_ddos

    #Ternary-matching tables final summation
    ternary_blocks = ternary_blocks_app + ternary_blocks_ddos
    ternary_entries = ternary_entries_app + ternary_entries_ddos

    # Two independent codewords; see ResourceUsage.codeword_length's docstring
    # for why max and not sum.
    codeword_length = max(codeword_length_app, codeword_length_ddos)

    # Under disjoint encoding each model keeps its own feature intervals, so
    # both models' tables are independent tables competing for the same
    # per-stage budgets -- pack them together, per pool.
    range_table_specs = range_table_specs_app + range_table_specs_ddos
    ternary_table_specs = ternary_table_specs_app + ternary_table_specs_ddos

    # Each model keeps its own intervals here, so levels must be derived per
    # model and concatenated in the SAME order the specs were. But there is
    # only ONE register block and one set of stateful ALUs behind both models
    # (a register a feature needs is emitted once, however many models select
    # that feature), so both calls must read their levels off a schedule of
    # the UNION -- see readiness_levels_for's emitted_features.
    feature_intervals_app = get_feature_intervals(clf_app, selected_features_app)
    feature_intervals_ddos = get_feature_intervals(clf_ddos, selected_features_ddos)
    emitted_features = list(feature_intervals_app) + list(feature_intervals_ddos)
    range_levels = (
        readiness_levels_for(feature_intervals_app, emitted_features=emitted_features) +
        readiness_levels_for(feature_intervals_ddos, emitted_features=emitted_features))
    # Same per-model split, same concatenation order, for the crossbar field
    # identities. Two models that selected the same feature resolve to the
    # same range field (the raw _val field is always shared) and, when their
    # interval lists also agree, to the same code_ field -- exactly the two
    # sharing rules _resolve_disjoint_feature_plan implements.
    range_fields = (
        range_key_fields_for(feature_intervals_app) +
        range_key_fields_for(feature_intervals_ddos))
    ternary_fields = (
        [ternary_key_fields(feature_intervals_app)] * len(ternary_table_specs_app) +
        [ternary_key_fields(feature_intervals_ddos)] * len(ternary_table_specs_ddos))
    ternary_ragged = (
        [ternary_key_is_ragged(feature_intervals_app)] * len(ternary_table_specs_app) +
        [ternary_key_is_ragged(feature_intervals_ddos)] * len(ternary_table_specs_ddos))
    # Registers, unlike range_levels, must be deduplicated ACROSS both
    # models: they are a single physical Register<> array shared by the
    # whole generated program regardless of 'joint' vs 'disjoint' ENCODING
    # (that choice only affects codeword/interval sharing -- see
    # build_p4_script.py's raw_feature_intervals, keyed on the union of
    # both models' raw feature names). One register_names_for call over the
    # combined feature names dedupes correctly; two separate calls
    # concatenated would not (each call only dedupes within itself), and
    # would double-count a register shared by both models.
    register_names = register_names_for(
        list(feature_intervals_app) + list(feature_intervals_ddos))

  else:
    raise ValueError(
        "multi_model_memory_evaluation: unknown encoding {!r}; "
        "expected 'joint' or 'disjoint'".format(encoding))

  # Range-matching tables and ternary classification tables are physically
  # distinct table pools (build_p4_script.py generates them separately), so
  # each pool is packed on its own and the two stage counts are summed. Both
  # pools are packed by the SAME solver: one stage count per pool that
  # respects the block, table-count and byte limits simultaneously, rather
  # than a max() of two independently-relaxed bounds (which can under-count,
  # see crossbar_stages_needed).
  # Both pools are placed dependency-aware (see crossbar_stages_needed and
  # feature_readiness_level): a feature's range table cannot precede the
  # register chain producing its key, and every classification table reads
  # every feature's codeword, so it cannot precede the last range table.
  # Validated against a real compile of the M2 program: 2 range stages + 1
  # classification stage = 3, exactly the compiler's own placement. The pure
  # packer predicted 2.
  #
  # F10: the classification boundary must be derived from where the range
  # pool's tables actually LANDED (StagePlan.depth), not from one past the
  # earliest stage a range table was merely ALLOWED to start
  # (max(range_levels) + 1) -- the 8-table crossbar cap can spill a range
  # table forward past its level, and reusing max(range_levels) + 1 would
  # then schedule a classification table into a stage a range table still
  # occupies.
  #
  # Mechanism B on top of that: both pools are emitted AFTER the gated
  # register blocks, so neither can occupy a stage the placer spends wholly
  # inside one -- a control-flow constraint, not a capacity one, and the
  # reason a stage can sit at 0/24 TCAM blocks and still take no table. It is
  # a placement constraint rather than a penalty precisely so that it costs
  # nothing on the rows whose tables were not going to land there anyway.
  interior_stages = gated_block_interior_stages(emitted_features)
  range_plan = crossbar_stages_needed(range_table_specs,
                                      readiness_levels=range_levels,
                                      key_fields=range_fields,
                                      unavailable_stages=interior_stages)
  ternary_level = range_plan.depth if range_table_specs else FLOW_HASH_LEVEL + 1
  # Only the classification pool gets ragged_keys. A range table keys one
  # meta.<feature>_val field of FEATURE_VALUE_BIT_WIDTH bits, a whole number
  # of bytes, so it presents no part-used crossbar byte and the group-offset
  # penalty is inert there by construction -- passing it would be noise.
  ternary_plan = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=[ternary_level] * len(ternary_table_specs),
      key_fields=ternary_fields,
      unavailable_stages=interior_stages,
      ragged_keys=ternary_ragged)

  # The property that makes summing occupancies below meaningful: the two
  # pools must never claim the same stage index.
  assert not (range_plan.indices & ternary_plan.indices), (
      "range and classification pools overlap at stages {}; summing their "
      "occupancies is only meaningful while they are disjoint".format(
          sorted(range_plan.indices & ternary_plan.indices)))

  # F5/F6: stage_depth is ternary_plan.depth -- the classification pool is
  # placed LAST (it starts at ternary_level, which is itself derived from
  # range_plan.depth), so its depth is the overall pipeline depth. Verified
  # against the M2 fixture: ternary_plan.indices == {5} there, so depth == 6,
  # exactly the brief's own worked example. max() with range_plan.depth is a
  # defensive widening for the degenerate case of zero ternary tables (where
  # crossbar_stages_needed's dependency-aware branch would otherwise report
  # depth 0), not something the real M2-shaped models ever hit.
  # ...plus VOTE_EPILOGUE_STAGES: the vote tables read every tree's class, so
  # they always occupy one further stage past the classification pool, and
  # stage_depth is the quantity TOFINO_PIPELINE_STAGES is checked against.
  # Measured as exactly 1 on all 19 calibration compiles.
  stage_depth = max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES

  # register_depth reuses range_levels (already computed above on both
  # branches, positionally aligned with feature_intervals) rather than
  # re-traversing anything; register_count/register_sram_bits reuse
  # register_names, likewise already computed above on both branches. See
  # ResourceUsage's docstring for the Spec 4.3 capacity caveat these three
  # fields carry.
  register_depth = max(range_levels, default=0)
  register_count = len(register_names)
  register_sram_bits = sum(
      register_width_bits(name) for name in register_names) * MAX_NUM_FLOWS

  # ternary_plan.blocks, not ternary_blocks: the latter is the naive
  # per-table sum computed above, before the ragged-key group-offset charge
  # (Mechanism G) that only crossbar_stages_needed's stage packing knows
  # about -- see StagePlan.blocks. range_blocks needs no such substitution:
  # a range table's key is always a whole number of bytes (never ragged),
  # so range_plan.blocks is provably identical to range_blocks.
  return ResourceUsage(
      stages=range_plan.occupied + ternary_plan.occupied,
      blocks=range_blocks + ternary_plan.blocks,
      stage_depth=stage_depth,
      range_entries=range_entries,
      ternary_entries=ternary_entries,
      codeword_length=codeword_length,
      register_depth=register_depth,
      register_count=register_count,
      register_sram_bits=register_sram_bits,
      range_depth=range_plan.depth,
      ternary_depth=ternary_plan.depth,
      range_tables=len(range_table_specs),
      ternary_tables=len(ternary_table_specs))
