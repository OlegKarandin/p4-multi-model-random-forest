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


def range_matching_resource_usage(feature_intervals):
  range_entries, range_blocks = 0, 0

  for feature in feature_intervals:
    total_rows = 0
    for lo, hi in feature_intervals[feature]:
      total_rows += range_entry_count(lo, hi)

    range_entries += len(feature_intervals[feature])
    range_blocks += math.ceil(total_rows / TERNARY_MATCHING_ENTRIES_PER_BLOCK)

  return range_entries, range_blocks
  

def ternary_matching_resource_usage(codewords):

  ternary_entries, ternary_blocks = 0, 0
  codeword_length = len(next(iter(codewords[0].items()))[0])

  if codeword_length > MAX_CODEWORD_LENGTH:
    raise RuntimeError("Codewords are too long", codeword_length)

  factor = math.ceil((codeword_length + 4) / TCAM_BLOCK_KEY_LENGTH)
  for tree in codewords:
    ternary_entries += len(codewords[tree])
    ternary_blocks += math.ceil(len(codewords[tree]) / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor

    #print('{} TCAM entries for {} codewords of length {}'.format(math.ceil(len(codewords[tree]) / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor, len(codewords[tree]), codeword_length))

  return ternary_entries, ternary_blocks, codeword_length


def ternary_crossbar_stages_needed(table_byte_widths):
  """Packs independent ternary classification tables (one per tree, per
  build_p4_script.py's generate_P4_tables_and_apply) into pipeline stages
  under the Ternary Match Input crossbar's two per-stage limits, confirmed
  by RM-5/RM-6/RM-7 (reviews/t12_required_changes.md Section 1.3) across
  key widths 8-512 bits: at most TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
  independent tables per stage, and at most TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
  total key bytes per stage -- whichever binds first. Uses first-fit-decreasing
  bin packing so tables of different widths (app vs. ddos trees, under
  disjoint encoding) can share a stage."""
  stages = []  # each entry: [bytes_used, tables_used]
  for width in sorted(table_byte_widths, reverse=True):
    for stage in stages:
      if (stage[1] + 1 <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE and
          stage[0] + width <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE):
        stage[0] += width
        stage[1] += 1
        break
    else:
      stages.append([width, 1])

  return len(stages)


def single_model_memory_evaluation(clf, selected_features):
  trees = get_tree_textual_representation(clf, selected_features)

  tree_nodes = {}
  for tree in trees:
    tree_nodes[tree] = get_nodes(trees[tree])

  feature_thresholds = get_feature_thresholds(tree_nodes)
  feature_intervals = get_feature_intervals_from_thresholds(feature_thresholds)
  range_entries, range_blocks = range_matching_resource_usage(feature_intervals)
  
  paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
  codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
  ternary_entries, ternary_blocks, codeword_length = ternary_matching_resource_usage(codewords)

  return (range_entries, range_blocks, ternary_entries, ternary_blocks, codewords, codeword_length)


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
    range_entries, range_blocks = range_matching_resource_usage(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    ternary_entries, ternary_blocks, codeword_length = ternary_matching_resource_usage(codewords)

    table_byte_widths = [math.ceil(codeword_length / 8)] * len(codewords)

  elif encoding == 'disjoint':

    range_entries_app, range_blocks_app, ternary_entries_app, ternary_blocks_app, codewords_app, codeword_length_app = single_model_memory_evaluation(clf_app, selected_features_app)
    range_entries_ddos, range_blocks_ddos, ternary_entries_ddos, ternary_blocks_ddos, codewords_ddos, codeword_length_ddos = single_model_memory_evaluation(clf_ddos, selected_features_ddos)

    range_blocks = range_blocks_app + range_blocks_ddos
    range_entries = range_entries_app + range_entries_ddos

    #Ternary-matching tables final summation
    ternary_blocks = ternary_blocks_app + ternary_blocks_ddos
    ternary_entries = ternary_entries_app + ternary_entries_ddos

    table_byte_widths = (
        [math.ceil(codeword_length_app / 8)] * len(codewords_app) +
        [math.ceil(codeword_length_ddos / 8)] * len(codewords_ddos)
    )

  range_stages = math.ceil(range_blocks / TCAM_BLOCKS_PER_STAGE)
  ternary_stages = max(
      math.ceil(ternary_blocks / TCAM_BLOCKS_PER_STAGE),
      ternary_crossbar_stages_needed(table_byte_widths),
  )

  return range_stages + ternary_stages, range_blocks + ternary_blocks
