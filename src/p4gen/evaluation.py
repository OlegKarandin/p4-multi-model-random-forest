import math
from dataclasses import dataclass
import sklearn.metrics as mt
from src.p4gen.build_p4_script import (
    MAX_CODEWORD_LENGTH,
    MAX_NUM_FLOWS,
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_BLOCKS_PER_STAGE,
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

    feature_blocks = math.ceil(total_rows / TERNARY_MATCHING_ENTRIES_PER_BLOCK)

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

  factor = band_factor(codeword_length)
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
# hash action occupies one stage ahead of any register touch.
FLOW_HASH_LEVEL = 1


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


def readiness_levels_for(feature_intervals, catalog=None):
  """One readiness level per feature, positionally aligned with
  range_matching_resource_usage's range_table_specs (both follow
  feature_intervals iteration order)."""
  return [feature_readiness_level(feature, catalog) for feature in feature_intervals]


def _stage_shards(block_count, byte_width):
  """Splits one logical table that cannot fit inside a single stage into the
  minimum number of per-stage shards, so the packer never reports a stage
  count below what the table alone already forces.

  Splitting a table's ROWS across stages is real: a table needing more than
  TCAM_BLOCKS_PER_STAGE blocks genuinely spreads those blocks over several
  stages, each shard still carrying the table's full key width. Splitting a
  table's KEY across stages is not real: a key is one indivisible match, TCAM
  compares a whole row in one clock, and a stage's crossbar physically cannot
  deliver more than TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes -- a table
  whose key exceeds that budget is rejected by the compiler outright, not
  spread across stages. Raise rather than silently pricing that impossible
  design as `ceil(byte_width / budget)` stages."""
  if byte_width > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), byte_width)
  n = max(1, math.ceil(block_count / TCAM_BLOCKS_PER_STAGE)) if block_count > 0 else 1
  return [(math.ceil(block_count / n), byte_width)] * n


def crossbar_stages_needed(table_specs, readiness_levels=None):
  """Packs independent match tables into pipeline stages under ALL three
  per-stage hardware limits simultaneously, and returns a StagePlan
  describing where the tables landed (not just how many stages that took).

  table_specs is one (block_count, byte_width) pair per independent P4
  table -- one per tree for the ternary classification tables
  (build_p4_script.py:636-659), one per feature for the range-matching
  tables (build_p4_script.py:663-674). RM-5/RM-6/RM-7 measured these limits
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
    * <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE  total key bytes
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

  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    for shard in _stage_shards(block_count, byte_width):
      shards.append((shard[0], shard[1], idx))

  def load(shard):
    blocks, width, _ = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  def fits(stage, blocks, width):
    return (stage[0] + blocks <= TCAM_BLOCKS_PER_STAGE and
            stage[1] + width <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE and
            stage[2] + 1 <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  if readiness_levels is None:
    stages = []  # each entry: [blocks_used, bytes_used, tables_used]
    for blocks, width, _ in sorted(shards, key=load, reverse=True):
      for stage in stages:
        if fits(stage, blocks, width):
          stage[0] += blocks
          stage[1] += width
          stage[2] += 1
          break
      else:
        stages.append([blocks, width, 1])

    return StagePlan(occupied=len(stages), depth=len(stages),
                      indices=frozenset(range(len(stages))))

  # Dependency-aware placement. Two differences from the packer above, both
  # chosen to track the REAL compiler rather than the theoretical optimum:
  #
  #   1. A table may not occupy a stage index below its readiness level --
  #      its key value literally does not exist yet.
  #   2. Placement is EAGER (earliest legal stage with room), not "pack as
  #      few stages as possible". The optimum would drop every table into the
  #      single latest stage; the compiler does not do that, and neither does
  #      this. Measured: M2's range pool really occupies 2 stages, which only
  #      eager placement reproduces.
  #
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}  # stage index -> [blocks_used, bytes_used, tables_used]
  ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]], -load(s)))
  for blocks, width, table_idx in ordered:
    index = readiness_levels[table_idx]
    while index in by_index and not fits(by_index[index], blocks, width):
      index += 1
    if index in by_index:
      by_index[index][0] += blocks
      by_index[index][1] += width
      by_index[index][2] += 1
    else:
      by_index[index] = [blocks, width, 1]

  return StagePlan(occupied=len(by_index),
                    depth=(max(by_index) + 1) if by_index else 0,
                    indices=frozenset(by_index.keys()))


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
                  reports a sane depth. M2 example: 6.
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
    wrong" when they are not even measuring the same thing.)

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
    register_names = register_names_for(feature_intervals)

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
    # model and concatenated in the SAME order the specs were.
    feature_intervals_app = get_feature_intervals(clf_app, selected_features_app)
    feature_intervals_ddos = get_feature_intervals(clf_ddos, selected_features_ddos)
    range_levels = (
        readiness_levels_for(feature_intervals_app) +
        readiness_levels_for(feature_intervals_ddos))
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
  range_plan = crossbar_stages_needed(range_table_specs,
                                      readiness_levels=range_levels)
  ternary_level = range_plan.depth if range_table_specs else FLOW_HASH_LEVEL + 1
  ternary_plan = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=[ternary_level] * len(ternary_table_specs))

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
  stage_depth = max(range_plan.depth, ternary_plan.depth)

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

  return ResourceUsage(
      stages=range_plan.occupied + ternary_plan.occupied,
      blocks=range_blocks + ternary_blocks,
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
