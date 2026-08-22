import math
from dataclasses import dataclass
import sklearn.metrics as mt
from src.p4gen.build_p4_script import (
    MAX_CODEWORD_LENGTH,
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_BLOCKS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
    generate_codewords,
    get_feature_intervals,
    get_joint_feature_intervals,
    get_nodes,
    get_root_to_leaf_paths,
    most_common_class_and_dropped_codewords,
    normalise_feature_name,
)
from src.p4gen.feature_registers import FEATURE_REGISTER_CATALOG
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


def accuracy_metrics(y_true, y_pred, task):

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

    range_entries += len(feature_intervals[feature])
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

  factor = math.ceil((codeword_length + 4) / TCAM_BLOCK_KEY_LENGTH)
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
  """Planter RF_EB-style accounting: code/decision tables move from ternary
  TCAM to exact-match SRAM (overlap.md's code-verified match-type table).
  Exact match cannot express the wildcarded ('*') bits a leaf's tree path
  never tests, so each wildcarded bit must be enumerated into concrete
  entries -- this is a real entry-count multiplier, not hidden here.
  Charges SRAM, not TCAM -- a distinct, far larger resource (overlap.md:
  ~6.2MB TCAM vs ~120MB SRAM on Tofino 1), so this is a separate accounting,
  reported alongside ternary_matching_resource_usage rather than replacing
  it.

  Final-review fix: a flat `2 ** codeword.count('*')` over-counts. Per
  build_p4_script.generate_codewords (build_p4_script.py:338-411), each
  feature gets its own fixed-width codeword segment (width =
  len(feature_intervals[feature]) - 1), and that segment is a THERMOMETER/
  UNARY code: valid segments are always of the shape 0^j 1^(width-j) for
  some j (see generate_codewords's `code[idx] = '1'`/`'0'` assignment
  logic) -- there are exactly `len(feature_intervals[feature])` reachable
  values for that segment (one per position of the 0/1 boundary, j = 0..
  width), NOT 2**width independent bit combinations. When a leaf's tree
  path never tests a feature at all (`feature not in features_involved`),
  generate_codewords fills that feature's *entire* segment with '*'
  characters -- so treating every '*' in that segment as an independent
  binary choice (2**width) overcounts whenever width > 1 (a feature with
  more than 2 intervals): the correct multiplier for a fully-wildcarded
  segment is `len(feature_intervals[feature])`, which only coincides with
  2**width in the degenerate width == 1 case (2 intervals).

  This function now walks each codeword in per-feature segments (mirroring
  generate_codewords's own concatenation order: feature_intervals.keys()
  iteration order, each segment consuming `len(feature_intervals[feature])
  - 1` characters) and, for each segment that is ALL wildcards, multiplies
  the leaf's entry-expansion factor by `len(feature_intervals[feature])`
  instead of `2 ** width`. A segment that is NOT all wildcards (the
  feature IS on the leaf's path, so generate_codewords started it as all
  '*' and then narrowed some positions to '0'/'1') may still contain
  leftover '*' characters in a shape whose exact reachable-value count
  isn't a simple closed form here; for that narrower sub-case this keeps
  `2 ** (wildcards within that one segment)` as a documented, safe
  (never-underestimating) approximation.

  Returns (sram_entries, sram_blocks). sram_blocks is deliberately None:
  the SRAM per-block *entry capacity* for a plain exact-match key table is
  not yet an established constant in this project. Direct investigation
  this session (~/open-p4studio/pkgsrc/p4-compilers/p4c/backends/tofino/
  bf-p4c/mau/memories.h:54, Memories::SRAM_DEPTH = 1024) confirms the same
  1024-entries-per-SRAM base resource_estimate.cpp uses for attached
  tables (counters/registers/action-data, resource_estimate.cpp:863,1323,
  1558: `entries_per_sram = 1024 * per_word`) also holds for match-key
  tables specifically (asm_output.cpp:2379: `tbl_entries = rams *
  table_format.match_groups.size() * 1024`) -- but unlike TCAM's per-block
  formula, `match_groups.size()` (how many independent match entries pack
  into one 1024-row SRAM) is not a closed-form function of key width alone:
  it comes out of table_format.cpp's LayoutOption/"ways" search (packing
  entries against RAM row width, overhead/version bits, and hash-way
  constraints jointly), not a documented formula. That search is out of
  scope for this task's time-box; SRAM block-count conversion is left as a
  flagged follow-up rather than an invented per_word/width assumption."""
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
  table beyond the hash itself.

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
  tree_nodes = {i: get_nodes(est, selected_features)
                for i, est in enumerate(clf.estimators_)}

  # get_feature_intervals recomputes trees/tree_nodes internally, but
  # that's the canonical single-model interval-derivation chain -- see
  # build_p4_script.py -- so the two derivations are behaviourally
  # identical.
  feature_intervals = get_feature_intervals(clf, selected_features)
  range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

  paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
  codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
  ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
      codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

  return (range_entries, range_blocks, ternary_entries, ternary_blocks, codewords, codeword_length,
          range_table_specs, ternary_table_specs)


def multi_model_memory_evaluation(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding,
                                  use_default_action_discount=False):
  """Returns (stages, blocks, stage_depth) -- three related but DISTINCT
  quantities (F6), only the first two of which this function is the source
  of truth for:

    stages      : OCCUPIED match-table stage count -- how many distinct
                  stage indices actually hold a table from either pool
                  (range_plan.occupied + ternary_plan.occupied). M2 example: 3.
                  This is what gets written to the campaign CSV's `stages`
                  column and plotted -- it is NOT a pipeline-depth quantity
                  and must never be compared against TOFINO_PIPELINE_STAGES.
    stage_depth : pipeline DEPTH, max(occupied stage index) + 1 -- the
                  quantity a hard stage ceiling actually reads (F5). Read
                  from ternary_plan.depth (the classification pool is placed
                  LAST, after the range pool, so its depth is the overall
                  pipeline depth), defensively widened to
                  max(range_plan.depth, ternary_plan.depth) so an
                  (unrealistic) model with no ternary tables at all still
                  reports a sane depth. M2 example: 6.
    (a third quantity, `stages_real` -- the REAL compiler's whole-program
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
    tree_nodes = {i: get_nodes(est, selected_features_app)
                  for i, est in enumerate(clf_app.estimators_)}

    offset = len(tree_nodes)

    tree_nodes.update({i + offset: get_nodes(est, selected_features_ddos)
                        for i, est in enumerate(clf_ddos.estimators_)})

    # tree_nodes above is still needed for get_root_to_leaf_paths below;
    # get_joint_feature_intervals recomputes its own copy internally, but
    # that's the canonical offset-merge interval-derivation chain -- see
    # build_p4_script.py -- so the two derivations are behaviourally
    # identical.
    feature_intervals = get_joint_feature_intervals(
        clf_app, selected_features_app, clf_ddos, selected_features_ddos)
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
        codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

    range_levels = readiness_levels_for(feature_intervals)

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

    # Under disjoint encoding each model keeps its own feature intervals, so
    # both models' tables are independent tables competing for the same
    # per-stage budgets -- pack them together, per pool.
    range_table_specs = range_table_specs_app + range_table_specs_ddos
    ternary_table_specs = ternary_table_specs_app + ternary_table_specs_ddos

    # Each model keeps its own intervals here, so levels must be derived per
    # model and concatenated in the SAME order the specs were.
    range_levels = (
        readiness_levels_for(get_feature_intervals(clf_app, selected_features_app)) +
        readiness_levels_for(get_feature_intervals(clf_ddos, selected_features_ddos)))

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

  return range_plan.occupied + ternary_plan.occupied, range_blocks + ternary_blocks, stage_depth
