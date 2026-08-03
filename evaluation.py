import math
import sklearn.metrics as mt
from build_p4_script import *

def accuracy_metrics(y_true, y_pred, task):

    if task == 'app':
        lab = [0, 1, 2]
        av = 'weighted'

    elif task == 'ddos':
        lab = [-1, 1]
        av = 'weighted'

    accuracy = mt.accuracy_score(y_true, y_pred)
    #precision = mt.precision_score(y_true, y_pred, labels=lab, average=av) #F: average=None gives per-class results
    #recall = mt.recall_score(y_true, y_pred, labels=lab, average=av)
    f1score = mt.f1_score(y_true, y_pred, labels=lab, average=av)

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


def range_matching_resource_usage(feature_intervals):
  """Returns (range_entries, range_blocks, range_table_specs).

  Every selected feature gets its OWN independent range-matching P4 table
  (build_p4_script.py:663-674, keyed on "meta.<feature>_val : range"), so
  range_table_specs is one (block_count, byte_width) pair per feature --
  the per-table data crossbar_stages_needed() needs. The aggregate
  range_blocks is still returned for the blocks half of the cost model."""
  range_entries, range_blocks = 0, 0
  range_table_specs = []

  for feature in feature_intervals:
    total_rows = 0
    for lo, hi in feature_intervals[feature]:
      total_rows += range_entry_count(lo, hi)

    feature_blocks = math.ceil(total_rows / TERNARY_MATCHING_ENTRIES_PER_BLOCK)

    range_entries += len(feature_intervals[feature])
    range_blocks += feature_blocks
    range_table_specs.append((feature_blocks, RANGE_TABLE_KEY_BYTES))

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
  each tree, the single leaf whose class value is the most common among
  that tree's codewords[tree].values() (see
  build_p4_script.most_common_leaf_codeword) becomes the table's
  default_action instead of an explicit entry, so that one tree's entry
  count drops by exactly 1 (never more, regardless of how many leaves
  share the most-common class) before it feeds into the block-count
  formula below. False (the default) is byte-identical to pre-Task-7
  behavior -- every existing caller/test is unaffected."""

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
      tree_entry_count -= 1

    tree_blocks = math.ceil(tree_entry_count / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor

    ternary_entries += tree_entry_count
    ternary_blocks += tree_blocks
    ternary_table_specs.append((tree_blocks, table_bytes))

    #print('{} TCAM entries for {} codewords of length {}'.format(math.ceil(len(codewords[tree]) / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor, len(codewords[tree]), codeword_length))

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


def crossbar_stages_needed(table_specs):
  """Packs independent match tables into pipeline stages under ALL three
  per-stage hardware limits simultaneously, and returns the stage count.

  table_specs is one (block_count, byte_width) pair per independent P4
  table -- one per tree for the ternary classification tables
  (build_p4_script.py:636-659), one per feature for the range-matching
  tables (build_p4_script.py:663-674). RM-5/RM-6/RM-7 measured these limits
  on the Ternary Match Input crossbar specifically; applying the same
  model to range-matching tables is an unmeasured extrapolation by
  analogy, not a separately confirmed finding -- but it is the
  conservative direction (it can only raise range_stages, never lower it),
  consistent with never under-counting. Both pools are packed separately
  with this one function, since they are physically distinct table pools.
  Range tables may also have their own, still-unmodelled per-stage limit
  (distinct from this crossbar analogy) -- a residual gap, not something
  this function claims to close.

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
  for block_count, byte_width in table_specs:
    shards.extend(_stage_shards(block_count, byte_width))

  def load(shard):
    blocks, width = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  stages = []  # each entry: [blocks_used, bytes_used, tables_used]
  for blocks, width in sorted(shards, key=load, reverse=True):
    for stage in stages:
      if (stage[0] + blocks <= TCAM_BLOCKS_PER_STAGE and
          stage[1] + width <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE and
          stage[2] + 1 <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE):
        stage[0] += blocks
        stage[1] += width
        stage[2] += 1
        break
    else:
      stages.append([blocks, width, 1])

  return len(stages)


def single_model_memory_evaluation(clf, selected_features):
  trees = get_tree_textual_representation(clf, selected_features)

  tree_nodes = {}
  for tree in trees:
    tree_nodes[tree] = get_nodes(trees[tree])

  feature_thresholds = get_feature_thresholds(tree_nodes)
  feature_intervals = get_feature_intervals_from_thresholds(feature_thresholds)
  range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

  paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
  codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
  ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(codewords, feature_intervals)

  return (range_entries, range_blocks, ternary_entries, ternary_blocks, codewords, codeword_length,
          range_table_specs, ternary_table_specs)


def multi_model_memory_evaluation(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding):

  if encoding == 'joint':
    trees_app = get_tree_textual_representation(clf_app, selected_features_app)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features_ddos)

    tree_nodes = {}
    for tree_app in trees_app:
      tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes)

    for tree_ddos in trees_ddos:
      tree_nodes[tree_ddos+offset] = get_nodes(trees_ddos[tree_ddos])

    feature_thresholds = get_feature_thresholds(tree_nodes)
    feature_intervals = get_feature_intervals_from_thresholds(feature_thresholds)
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(codewords, feature_intervals)

  elif encoding == 'disjoint':

    (range_entries_app, range_blocks_app, ternary_entries_app, ternary_blocks_app,
     codewords_app, codeword_length_app,
     range_table_specs_app, ternary_table_specs_app) = single_model_memory_evaluation(clf_app, selected_features_app)
    (range_entries_ddos, range_blocks_ddos, ternary_entries_ddos, ternary_blocks_ddos,
     codewords_ddos, codeword_length_ddos,
     range_table_specs_ddos, ternary_table_specs_ddos) = single_model_memory_evaluation(clf_ddos, selected_features_ddos)

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

  # Range-matching tables and ternary classification tables are physically
  # distinct table pools (build_p4_script.py generates them separately), so
  # each pool is packed on its own and the two stage counts are summed. Both
  # pools are packed by the SAME solver: one stage count per pool that
  # respects the block, table-count and byte limits simultaneously, rather
  # than a max() of two independently-relaxed bounds (which can under-count,
  # see crossbar_stages_needed).
  range_stages = crossbar_stages_needed(range_table_specs)
  ternary_stages = crossbar_stages_needed(ternary_table_specs)

  return range_stages + ternary_stages, range_blocks + ternary_blocks
