import math
import re
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
    get_tree_textual_representation,
    most_common_class_and_dropped_codewords,
)
from src.p4gen.feature_registers import FEATURE_REGISTER_CATALOG

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


# Width of every per-feature value field the range-matching tables key on
# (build_p4_script.py:775 emits "bit<16> <feature>_val" for each selected
# feature). 16 bits is the project's decided feature precision; one range
# table keys on exactly one such field, hence 2 crossbar bytes per table.
FEATURE_VALUE_BIT_WIDTH = 16
RANGE_TABLE_KEY_BYTES = math.ceil(FEATURE_VALUE_BIT_WIDTH / 8)


def range_matching_resource_usage(feature_intervals, key_bit_width=FEATURE_VALUE_BIT_WIDTH):
  """Returns (range_entries, range_blocks, range_table_specs).

  Every selected feature gets its OWN independent range-matching P4 table
  (build_p4_script.py:663-674, keyed on "meta.<feature>_val : range"), so
  range_table_specs is one (block_count, byte_width) pair per feature --
  the per-table data crossbar_stages_needed() needs. The aggregate
  range_blocks is still returned for the blocks half of the cost model.

  A physical TCAM block is TERNARY_MATCHING_ENTRIES_PER_BLOCK rows x
  TCAM_BLOCK_KEY_LENGTH key bits, so BOTH dimensions cost blocks. The depth
  term is ceil(total_rows / 512); the width term is the same
  ceil((key_bits + 4) / 44) that ternary_matching_resource_usage charges
  (the +4 is the flat per-row overhead of Sec 4.1, confirmed directly in a
  real compile's mau.characterize.log, which reports a 64-bit ternary key as
  occupying 68 bits and a 16-bit range key as occupying 20).

  At this project's decided 16-bit feature precision the width factor is 1,
  so this term changes no current number -- it exists so a wider feature
  value can never silently under-count, which the depth-only formula did.

  IMPORTANT (measured, reviews/p4_tofino_reference.md Sec 4.2): the width
  factor is only correct when the key field is pinned to a PHV container of
  the same width. A bit<16> range key that the compiler parks in a 32-bit W
  container really costs TWO TCAM words per entry ("1 in 2 (88)"), not one.
  generate_P4_code therefore emits an @pa_container_size pragma per feature
  value field; without those pragmas this function under-counts by up to a
  factor of 2 per table."""
  range_entries, range_blocks = 0, 0
  range_table_specs = []

  width_factor = math.ceil((key_bit_width + 4) / TCAM_BLOCK_KEY_LENGTH)
  key_bytes = math.ceil(key_bit_width / 8)

  for feature in feature_intervals:
    total_rows = 0
    for lo, hi in feature_intervals[feature]:
      total_rows += range_entry_count(lo, hi)

    feature_blocks = math.ceil(total_rows / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * width_factor

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
    raise RuntimeError("Codewords are too long", codeword_length)

  table_bytes = ternary_table_key_bytes(feature_intervals)

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


def _normalise_feature_key(feature_name):
  """Canonical FEATURE_REGISTER_CATALOG key for a feature name.

  Dataset columns arrive dot-separated ('Fwd.Packet.Length.Max'), P4 field
  names arrive underscore-separated, and the catalog is keyed lowercase
  underscore. Collapse every run of non-alphanumeric characters to a single
  underscore so all three spellings land on the same key. Leading/trailing
  separators are stripped so 'Flow.IAT.Max.' cannot become a distinct key."""
  return re.sub(r'[^0-9a-z]+', '_', feature_name.lower()).strip('_')


def feature_readiness_level(feature_name, catalog=None):
  """Earliest pipeline stage at which this feature's `_val` field -- and so
  its range-matching table -- can possibly be placed.

    level = FLOW_HASH_LEVEL
          + 1 if the feature is fwd-gated (flow_orientation_action has to
            resolve meta.fwd before the gated block can run)
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

  entry = catalog.get(_normalise_feature_key(feature_name))
  if entry is None:
    return FLOW_HASH_LEVEL

  gate_cost = 1 if entry.get("gated_by") == "fwd" else 0
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

  A table needing more than TCAM_BLOCKS_PER_STAGE blocks must spread those
  blocks over several stages; a table whose key is wider than the whole
  per-stage crossbar budget cannot be fed to one stage at all. In both
  cases the table occupies >= ceil(limit-excess) stages, and every stage it
  occupies still has to receive its key bytes -- so each shard carries the
  full width (capped at the per-stage budget) rather than a fraction of
  it."""
  shards_by_blocks = math.ceil(block_count / TCAM_BLOCKS_PER_STAGE) if block_count > 0 else 1
  shards_by_bytes = math.ceil(byte_width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE) if byte_width > 0 else 1
  n_shards = max(1, shards_by_blocks, shards_by_bytes)

  shard_blocks = math.ceil(block_count / n_shards)
  shard_bytes = min(byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)
  return [(shard_blocks, shard_bytes)] * n_shards


def crossbar_stages_needed(table_specs, readiness_levels=None):
  """Packs independent match tables into pipeline stages under ALL three
  per-stage hardware limits simultaneously, and returns the stage count.

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

    return len(stages)

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

  return len(by_index)


def single_model_memory_evaluation(clf, selected_features, use_default_action_discount=False):
  """use_default_action_discount: opt-in, passed straight through to
  ternary_matching_resource_usage (which has implemented the Planter-style
  discount since Task 7 but was never reachable from this estimator). False
  -- the default -- reproduces every pre-existing caller's numbers exactly."""
  trees = get_tree_textual_representation(clf, selected_features)

  tree_nodes = {}
  for tree in trees:
    tree_nodes[tree] = get_nodes(trees[tree])

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
  """use_default_action_discount: opt-in, threaded down to
  ternary_matching_resource_usage under BOTH encodings -- directly for
  'joint' (which does its own ternary accounting on the merged tree set),
  and via both nested single_model_memory_evaluation calls for 'disjoint'.
  False -- the default -- reproduces every pre-existing caller's numbers
  exactly."""

  if encoding == 'joint':
    trees_app = get_tree_textual_representation(clf_app, selected_features_app)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features_ddos)

    tree_nodes = {}
    for tree_app in trees_app:
      tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes)

    for tree_ddos in trees_ddos:
      tree_nodes[tree_ddos+offset] = get_nodes(trees_ddos[tree_ddos])

    # tree_nodes above (built with the default -1 tree tag) is still needed
    # for get_root_to_leaf_paths below; get_joint_feature_intervals
    # recomputes its own copy internally (with real tree indices) for the
    # canonical offset-merge interval derivation -- see build_p4_script.py.
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
  range_stages = crossbar_stages_needed(range_table_specs,
                                        readiness_levels=range_levels)
  ternary_level = (max(range_levels) + 1) if range_levels else FLOW_HASH_LEVEL + 1
  ternary_stages = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=[ternary_level] * len(ternary_table_specs))

  return range_stages + ternary_stages, range_blocks + ternary_blocks
