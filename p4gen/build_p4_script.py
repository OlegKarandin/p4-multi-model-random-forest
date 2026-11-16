import os
import math
from pathlib import Path
import numpy as np
from sklearn.tree import _tree as _sklearn_tree
import csv
import json
from collections import Counter
from itertools import product

from p4gen.feature_registers import FEATURE_REGISTER_CATALOG
from p4gen import p4_gen_config
from p4gen import switch_semantics

# Geometry this module no longer owns -- it lives in p4model/target.py.
# Re-exported here so src/training/, scripts/ and tests keep importing it from
# the path they always have. Explicit names, not `import *`, so the mapping
# stays greppable.
from p4model.target import (
    MAX_KEY_SLICE_BITS,
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
)
from p4model import lanes, packed
# The IAT unit rule (Task 34), one statement shared with the model's range-row guard.
from p4model.units import iat_ticks, is_iat_feature, switch_range  # noqa: F401
from p4model.errors import NoSplitTree
from p4model.names import normalise_feature_name
from p4model.program import (VOTE_EXPORT, VOTE_EXPORT_BITS, APP_TASK, DDOS_TASK, ENCODINGS, JOINT_ENCODING,
                                 PER_TASK_ENCODING, PER_TREE_ENCODING, PLACEMENT_PRIORITY,
                                 TaskSpec, per_task_field,
                                 placement_priorities, range_table_priority, task_names)
# The flow-slot count, the flow index width, the hash's @symmetric pairs and the
# bf_rt key-name rule live in p4model.program (one statement each, which
# program_checks reads too); re-exported here, where the generator uses them.
from p4model.program import (FLOW_HASH_SYMMETRIC_PAIRS, FLOW_INDEX_BITS,  # noqa: F401
                                 MAX_NUM_FLOWS, bfrt_key_name)

INFINITE = (2**16)-1

# The P4 templates ship inside this package (p4gen/resources/), so they are found
# from this module's own location, not from the current directory.
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources") + os.sep
INTERMEDIATE = "temp/"
OUTPUT_PATH = "p4/"
PATH_ACTION_TEMPLATE_P4 = PATH + 'action.p4'
PATH_TABLE_CLASSIFICATION_TEMPLATE_P4 = PATH + 'table_classification.p4'
# Task 8: Planter RF_EB-style exact-match code/decision tables. The two
# templates are currently byte-identical -- table_classification.p4 has no
# <MATCH_TYPE>-style marker of its own; match type is baked into <KEYS>,
# which generate_tables_and_apply builds directly in Python (see
# classification_keys below), not read from either template file. A
# separate file still exists (rather than reusing one path for both modes)
# so the two modes have independent templates to diverge from later, and so
# match_type's file-selection story matches the interface both templates
# are meant to express.
PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4 = PATH + 'table_classification_exact.p4'
PATH_P4_CODE_TEMPLATE_INPUT = PATH + 'p4_template.p4'

# Task 19 (R6): read every P4 table/action template exactly once, here, at
# module import, instead of re-opening the same file once per tree and once
# per feature inside generate_tables_and_apply (and, before this task,
# once per generate_actions call for action.p4). A missing/unreadable
# template now fails at import time rather than as a per-call surprise --
# the intended behaviour, matching Task 2's F9 fix for the action template.
#
# The range-matching feature tables and the ternary classification tables
# share ONE template, _TABLE_TEMPLATE (table_classification.p4's content);
# each builds its own one-line/multi-line <KEYS> text in Python. The range
# path's own template, table.p4 (deleted in the final review; it differed only
# in its <KEYS> line and in 3 trailing spaces on its closing "}" line), is
# reproduced by the range-table path, which re-adds those 3 spaces itself (see
# the comment at its call site) so its emitted bytes stay identical.
_TABLE_TEMPLATE = Path(PATH_TABLE_CLASSIFICATION_TEMPLATE_P4).read_text()
_ACTION_TEMPLATE = Path(PATH_ACTION_TEMPLATE_P4).read_text()
# table_classification_exact.p4 is byte-identical to table_classification.p4
# (see the comment above PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4) --
# still read from, and kept as, its own file/constant (not folded into
# _TABLE_TEMPLATE) purely so match_type='exact' keeps an explicit template
# file of its own to select and diverge from later, per the decision to
# preserve resources/table_classification_exact.p4 as a file.
_TABLE_CLASSIFICATION_EXACT_TEMPLATE = Path(PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4).read_text()


_TREE_LEAF = _sklearn_tree.TREE_LEAF


def ensure_directory_exists(path):
  """
  Creates the directory if it does not exist.

  Parameters:
      path (str): The path of the directory to check/create.
  """
  if not os.path.exists(path):
    os.makedirs(path)



def dt_thresholds_float_to_int(clf):
  for tree in clf.estimators_:
    tree_obj = tree.tree_
    # Only process internal nodes (feature != -2 means it's not a leaf)
    for i in range(tree_obj.node_count):
        if tree_obj.feature[i] != -2:
            # math.floor, NOT round: Python's round is round-half-to-EVEN, and
            # sklearn puts every split at the MIDPOINT of two observed values --
            # so for an integer-valued feature with a unit gap the threshold is
            # always v + 0.5 and the rule applies to ~a third of all splits
            # (measured: 33-42%). round(2.5) == 2 is right for a `<= 2.5` split,
            # but round(3.5) == 4 WIDENS it to include 4, so ~half of those
            # splits move up by one integer -- asymmetrically, since it never
            # moves one down.
            #
            # For an integer x and any real t, `x <= t` and `x <= floor(t)` are
            # the same test, so floor is exact on integer-valued features. It was
            # not exact on the *.Mean features (genuine means, fractional values;
            # they left the catalog in Task 38) -- but it stayed 3-8x closer to
            # the fitted model than round there (measured per-tree predictions
            # changed: 24 vs 185 at 3 trees, 48 vs 316 at 7).
            tree_obj.threshold[i] = math.floor(tree_obj.threshold[i])
  return clf


def _parents_and_depths(t):
  """One forward pass over a fitted tree_'s children_left/children_right
  arrays, returning (parent, depth) lists indexed by node id: parent[i] is
  i's parent node id (-1 for the root), depth[i] is i's depth (0 for the
  root).

  A single forward pass (no queue/stack) suffices because sklearn always
  assigns a node's id before either of its children's -- true of both the
  depth-first and best-first tree builders, since a node must exist to be
  split, and splitting is what creates its children -- so by the time index
  i is reached as a CHILD reference from some earlier node j < i, node j's
  own parent[j]/depth[j] have already been written."""
  parent = [-1] * t.node_count
  depth = [0] * t.node_count
  for i in range(t.node_count):
    left = t.children_left[i]
    right = t.children_right[i]
    if left != _TREE_LEAF:
      parent[left] = i
      depth[left] = depth[i] + 1
    if right != _TREE_LEAF:
      parent[right] = i
      depth[right] = depth[i] + 1
  return parent, depth


def get_nodes(estimator, feature_names):
  """Node dict for one fitted decision tree, read straight off sklearn's
  tree_ arrays. Thresholds are floored to int here -- the single rounding
  rule for this quantity (dt_thresholds_float_to_int floors before
  training-time export, extract_feature_intervals rounds; they agreed only
  by ordering).

  Replaces the export_text round-trip that used to render the tree to
  indented text and re-parse it (Task 15) -- that approach cost a max_depth
  truncation hazard (sklearn's export_text defaults to max_depth=10), a
  rounding-order bug, an O(n^2) rescan per node to find its right child, a
  state-machine parser, and dependence on export_text's exact
  "class: <value>" line format. Proved equivalent to that approach across
  real fitted forests before the old parser (_get_nodes_from_text) was
  deleted -- see tests/test_tree_parsing.py and commit 838acf6. Reading
  estimator.tree_ directly has none of those costs: every node's parent,
  depth, and both children fall straight out of the C-level arrays sklearn
  already built while fitting."""
  t = estimator.tree_
  parent, depth = _parents_and_depths(t)
  nodes = {}
  for i in range(t.node_count):
    leaf = t.children_left[i] == _TREE_LEAF
    node = {"node": i, "depth": depth[i], "is_leaf": leaf, "father_node": parent[i]}
    if leaf:
      node["class"] = int(np.argmax(t.value[i]))
      node["action_name"] = "classify_flow"
    else:
      node["feature"] = normalise_feature_name(feature_names[t.feature[i]])
      node["action_name"] = "classify_" + node["feature"]
      node["threshold"] = int(math.floor(t.threshold[i]))
      node["left_child"] = int(t.children_left[i])
      node["right_child"] = int(t.children_right[i])
    nodes[i] = node
  return nodes


def tree_nodes_for(model, feature_names):
  """{tree_index: node_dict} for every estimator in a fitted forest."""
  return {i: get_nodes(est, feature_names) for i, est in enumerate(model.estimators_)}


def merge_tree_nodes(*node_dicts):
  """Merge per-model tree_nodes, shifting each later model's tree indices past
  the previous ones so ids cannot collide. This offset trick was written out
  longhand in four places."""
  merged = {}
  for node_dict in node_dicts:
    offset = len(merged)
    for index, nodes in node_dict.items():
      merged[index + offset] = nodes
  return merged


def get_feature_thresholds(tree_nodes):
  '''
   Inputs: Dictionary containing the features of all nodes in the Random Forest
   Outputs: List of tuples containing the comparison thresholds (i.e. feature splits) each feature goes through over all the trees: (Feature Name, Threshold)
   '''

  nodes = []
  feature_thresholds = []

  #Gather node features from all Decision Trees
  for tree in tree_nodes:
    for node in tree_nodes[tree]:
      nodes.append(tree_nodes[tree][node])
  #Join all feature thresholds (splits) that are consulted at each node
  for node in nodes:
    if node["is_leaf"]:
      continue
    feature_thresholds.append((node["feature"],
                            node["threshold"]))
  #Sort feature thresholds by feature
  return sorted(feature_thresholds, key=lambda x: (x[0], x[1]))


def get_feature_intervals_from_thresholds(feature_thresholds):
  '''
  Inputs: List of tuples containing the features splits of all features
  Outputs: Dictionary where each key is the feature name and the associated value is the list of intervals of the given feature
  '''
  feature_intervals = {}

  # Iterate over each feature split
  for feature, threshold in feature_thresholds:
      #New Feature, Init interval

      # NOTE: a split at threshold 0 is a REAL split and gets its own [0, 0]
      # interval. This used to be skipped outright, on the premise that "all
      # features are positive" -- but 0 is not positive, dataset.py keeps
      # zero-valued rows, and a "counter is zero vs non-zero" split is exactly
      # sklearn threshold 0.5 truncated to 0. Skipping it dropped the split
      # from the intervals while generate_codewords still saw the condition in
      # every leaf path, so the "> 0" branch matched no interval bound and
      # stayed fully wildcarded -- matching values it should have excluded.
      # A [0, 0] range is perfectly legal for a range-match table.
      if feature not in feature_intervals:
          feature_intervals[feature] = [(0, threshold)]

      else: # exisiting feature - extend interval
          last_range = feature_intervals[feature][-1]
          if threshold == last_range[1]:
              continue
          else:
            feature_intervals[feature].append((last_range[1] + 1, threshold))

  # Add last interval (higher threshold, infinite)
  for feature, ranges in feature_intervals.items():
      if ranges[-1][1] != INFINITE:
          ranges.append((ranges[-1][1]+1, INFINITE))

  return feature_intervals


def feature_intervals_from_nodes(tree_nodes):
  return get_feature_intervals_from_thresholds(get_feature_thresholds(tree_nodes))


def thermometer_code(n_intervals, interval_idx):
  """Codeword segment for interval_idx, counting from the LOWEST interval:
  '0' * idx + '1' * (n_intervals - 1 - idx). Matches the 0^j 1^(width-j) shape
  exact_match_resource_usage documents."""
  return '0' * interval_idx + '1' * (n_intervals - 1 - interval_idx)


def _reject_colliding_feature_names(selected_features):
  canonical = {}
  # Dedupe exact-string repeats first: the same raw name appearing twice
  # (e.g. a feature genuinely shared, with IDENTICAL spelling, between two
  # models' selected-feature lists in get_joint_feature_intervals) is not a
  # collision -- only two DIFFERENT raw spellings landing on the same
  # normalised key are.
  for name in dict.fromkeys(selected_features):
    canonical.setdefault(normalise_feature_name(name), []).append(name)
  collisions = {k: v for k, v in canonical.items() if len(v) > 1}
  if collisions:
    # R10: an invariant (AssertionError, see generate_P4_code's catalog check).
    raise AssertionError(
        "feature names collide after normalisation, their intervals would "
        "silently merge: {}".format(collisions))


def get_feature_intervals(model, selected_features):
  """feature_intervals dict for one fitted model's selected features --
  tree_nodes_for(...) + feature_intervals_from_nodes(...), gated by
  _reject_colliding_feature_names so two differently-spelled names that
  normalise to the same key (e.g. 'Flow.IAT.Max' and 'Flow IAT Max') raise
  instead of silently merging their intervals into one entry. Callers that
  derive tree_nodes/feature_intervals directly (bypassing this function)
  must run the same guard themselves -- see single_model_memory_evaluation
  and multi_model_memory_evaluation's 'joint' branch (via ResourceUsage) in evaluation.py."""
  _reject_colliding_feature_names(selected_features)
  tree_nodes = tree_nodes_for(model, selected_features)
  return feature_intervals_from_nodes(tree_nodes)


def get_joint_feature_intervals(model_a, features_a, model_b, features_b):
  """The joint-encoding counterpart of get_feature_intervals: derives ONE
  shared feature_intervals dict from the union of both models' trees, via
  the same offset trick used to keep the two models' node IDs from
  colliding (model_b's tree indices are shifted by len(model_a's trees)
  before merging). This was previously reimplemented identically three
  times (evaluation.multi_model_memory_evaluation's 'joint' branch (which now
  returns a ResourceUsage), feature_selection._derive_joint_feature_intervals, and
  main.implement_tree_models_in_P4); this was the single canonical copy of
  that derivation until Task 16 re-inlined an equivalent tree_nodes_for(...)
  + merge_tree_nodes(...) + feature_intervals_from_nodes(...) sequence
  directly into multi_model_memory_evaluation's joint branch (to share the
  merged tree_nodes with that function's other consumers) -- so this is now
  ONE of two call sites deriving the same joint feature_intervals, not the
  sole one; both apply the same collision guard (see
  _reject_colliding_feature_names), just via different call paths."""
  # Checked as ONE union, not two independent lists: get_feature_thresholds
  # below merges both models' tree_nodes into a single feature_intervals
  # dict, so a collision across features_a/features_b (e.g. 'Flow.IAT.Max'
  # in one, 'Flow IAT Max' in the other) would silently fold together
  # exactly like a collision within one list would. A feature genuinely
  # shared between both models arrives with IDENTICAL spelling in both
  # lists (same dataset.py naming convention feeds both), so that case is
  # deduped rather than flagged -- only a genuine spelling mismatch raises.
  _reject_colliding_feature_names(list(features_a) + list(features_b))

  tree_nodes = merge_tree_nodes(tree_nodes_for(model_a, features_a),
                                 tree_nodes_for(model_b, features_b))

  return feature_intervals_from_nodes(tree_nodes)


def feature_intervals_to_csv(feature_intervals, path_to_output=INTERMEDIATE, output_filename = "feature_intervals.csv"):
  rows = []

  for feature_name in feature_intervals:
    intervals = feature_intervals[feature_name]
    rows.append([feature_name])

    for idx,interval in enumerate(intervals[::-1]):
      row = []
      row.append(interval)

      code = thermometer_code(len(intervals), len(intervals) - 1 - idx)
      for bit in code:
        row.append(bit)
      rows.append(row)

    rows.append([])


  # Open the file in write mode
  with open(path_to_output + output_filename, 'w', newline='') as csvfile:
      # Create a CSV writer object
      writer = csv.writer(csvfile, delimiter=";")

      # Write the data rows
      for row in rows:
          writer.writerow(row)


def get_root_to_leaf_paths(tree_nodes):
  '''
  For each leaf node in each Decision Tree, this function maps all the nodes we must traverse to go from the root node of the end leaf nodes,
  storing the class label associated to that node, while gathering the following features for each traversed node:
    Node_ID
    Feature Name
    Feature Threshold
    Condition (≤ or >)

  Inputs: Dictionary containing the features of all nodes in the Random Forest
  Outputs: Dictionary of dictionaries, including the path and final class of each leaf node in the Random Forest.
            First key is the tree_id (0, 1, etc.). Second key is the leaf_node_id (0, 1, etc.).

  Example Output: { "class": "1.0", "path": [ {"node_id": 2, "feature": "flow_iat_max", "threshold": 504078, "condition": "<="},
                                              {"node_id": 1, "feature": "bwd_packet_length_max", "threshold": 12, "condition": "<="},
                                              {"node_id": 0, "feature": "bwd_packet_length_max", "threshold": 3513, "condition": "<="} ] }
  '''

  leaf_nodes_per_tree = {}
  paths_leaf_nodes_per_tree = {}

  # 1. Obtain only Leaf Nodes
  for tree in tree_nodes:
    leaf_nodes_per_tree[tree]=[]
    for node in tree_nodes[tree]:
      if tree_nodes[tree][node]['is_leaf']:
        leaf_nodes_per_tree[tree].append(tree_nodes[tree][node])

  # 2. Iterate over Leaf Nodes
  for tree in leaf_nodes_per_tree:
    paths_leaf_nodes_per_tree[tree] = {}

    for leaf_node in leaf_nodes_per_tree[tree]:
      paths_leaf_nodes_per_tree[tree][leaf_node["node"]]= {}
      # Get Class assigned when we reach the Leaf Node
      paths_leaf_nodes_per_tree[tree][leaf_node["node"]]["class"] = leaf_node["class"]
      # Determine Path to reach Leaf Node
      paths_leaf_nodes_per_tree[tree][leaf_node["node"]]["path"]=[]
      # Get Father Node from Leaf Node
      father_node = leaf_node["father_node"]
      child_node = leaf_node["node"]
      while father_node > -1: #While we don't reach Root Node

        # Determine if we came from right or left
        if tree_nodes[tree][father_node]["right_child"] == child_node:
          condition = ">"
        else:
          condition = "<="

        paths_leaf_nodes_per_tree[tree][leaf_node["node"]]["path"].append({"node_id": tree_nodes[tree][father_node]["node"],
                                                                          "feature": tree_nodes[tree][father_node]["feature"],
                                                                          "threshold": tree_nodes[tree][father_node]["threshold"],
                                                                          "condition": condition
                                                                          })

        child_node = father_node
        father_node = tree_nodes[tree][father_node]["father_node"]

  return(paths_leaf_nodes_per_tree)


def leaf_is_reachable(path):
  '''True if some integer input satisfies every condition on a root-to-leaf
  path (get_root_to_leaf_paths' steps). Thresholds are integers here
  (dt_thresholds_float_to_int floors them), so the leaf's range on a feature is
  (largest ">" threshold, smallest "<=" threshold], empty when the first is not
  below the second. Flooring makes this happen: nested splits at 5.2 and 5.4
  both become "<= 5", and the branch "> 5 and <= 5" between them holds no
  integer. sklearn never reaches such a leaf either, since it predicts with the
  same floored thresholds.'''
  above, at_most = {}, {}
  for step in path:
    feature, threshold = step["feature"], step["threshold"]
    if step["condition"] == "<=":
      at_most[feature] = min(at_most.get(feature, threshold), threshold)
    else:
      above[feature] = max(above.get(feature, threshold), threshold)
  return all(above[f] < at_most[f] for f in above if f in at_most)


def generate_codewords(paths_leaf_nodes_per_tree, feature_intervals):
  '''
    For each leaf node in each Decision Trees, this function generates the corresponding codeword, associated with the leaf node class label.

    A leaf no integer input reaches (leaf_is_reachable) gets no codeword: its
    codeword could equal a reachable leaf's, and the dict below would then keep
    whichever came last.

    Inputs: Dictionary of dictionaries, including the path and final class of each leaf node in the Random Forest
            List of feature names used to train the RF model
    Outputs: Dictionary of dictionaries. First key is the tree_id (0, 1, etc.). Second key is the codeword (11*00**). Value is the class label.
  '''

  codewords = {}

  for tree in paths_leaf_nodes_per_tree:
    codewords[tree]={}

    #Generate codeword for each leaf node
    for leaf_node in paths_leaf_nodes_per_tree[tree]:
      #print(leaf_node)
      codeword = []
      # Obtain features involved in path to reach leaf
      current_leaf_node = paths_leaf_nodes_per_tree[tree][leaf_node]
      if not leaf_is_reachable(current_leaf_node["path"]):
        continue
      features_involved = [step["feature"] for step in current_leaf_node["path"]]

      #Iterate over all features
      for feature in feature_intervals.keys():
        feature_conditions = []

        #If feature is not in  the path, we add * to codeword
        if feature not in features_involved:
          for i in range(len(feature_intervals[feature])-1):
            codeword.append('*')

        # If feature is in Path, generate bits
        else:
          # Init code with all *
          code = ['*' for i in range(len(feature_intervals[feature])-1)]
          # Find conditions feature needs to fullfil
          for step in current_leaf_node["path"]:
            if step['feature']==feature:
              threshold = step["threshold"]
              condition = step["condition"]
              feature_conditions.append([threshold, condition])

          # Iterate over all feature intervals
          for idx,interval in enumerate(feature_intervals[feature]):
            # For each interval, compare with conditions
            for conditions in feature_conditions:
              threshold = conditions[0]
              condition = conditions[1]
              #We set bit for last interval by looking at upper bound of penultim interval
              if idx < (len(feature_intervals[feature])-1):

                if threshold == interval[1]:
                  if condition == "<=":
                    code[idx] = '1'
                  elif condition == ">":
                    if code[idx] != '1':
                      code[idx] = '0'
                elif threshold < interval[1]:
                    if condition == "<=":
                      code[idx] = '1'
                elif threshold > interval[1]:
                    if condition == '>':
                      code[idx] = '0'

          for c in code:
            codeword.append(c)

      # This dict is keyed by codeword string, so two leaves with one codeword keep
      # the last one written (spec 2026-09-15 §4.4, review finding 2.2's side note).
      # The case that happened in real forests was a leaf no integer reaches, its
      # codeword equal to a reachable sibling's: skipped above. Two REACHABLE leaves
      # cannot share a codeword, since the tree's own thresholds are among the
      # intervals' bounds and some threshold separates any two leaves.
      codewords[tree][''.join(codeword)]=current_leaf_node['class']

  return codewords


def most_common_class_and_dropped_codewords(tree_codewords):
  '''
    Planter-style default-action discount, corrected against Planter's REAL
    source (In-Network-Machine-Learning/Planter,
    src/models/RF/Type_EB/table_generator.py -- the RF-specific citation --
    and src/models/DT/Type_EB/table_generator.py, textually near-identical):
    `default_vote/default_label = max(collect_votes, key=collect_votes.count)`,
    then EVERY entry whose leaf equals it is dropped, not just one -- confirmed
    by both files' own entry-filtering
    loop this session, not assumed).

    Given one tree's codeword dict (codeword string -> class value, the same
    shape as generate_codewords()'s per-tree dict), returns (class_value,
    dropped_codewords): class_value is the single most common class among
    tree_codewords.values() (ties broken deterministically by Counter's
    stable most_common() order -- the first-inserted class among tied
    classes wins); dropped_codewords is a list of EVERY codeword string
    whose value equals class_value -- all of these become the table's
    default_action and must be omitted from the table's explicit entries.
    This is provably safe regardless of Planter's own behavior: a decision
    tree's leaves partition the input space with zero overlap (every real
    input reaches exactly one leaf), so any input that would have reached a
    dropped leaf now matches no explicit entry, falls through to the
    default action, and gets exactly the class it would have gotten anyway.
  '''
  vote_counts = Counter(tree_codewords.values())
  most_common_class, _ = vote_counts.most_common(1)[0]
  dropped_codewords = [
      codeword for codeword, class_value in tree_codewords.items()
      if class_value == most_common_class
  ]
  return most_common_class, dropped_codewords


def get_ternary_value_and_mask(codeword):
  '''Returns (hex_value, hex_mask) for one codeword segment -- the same
  computation the previous get_ternary_match performed, but returned as two
  separate hex strings instead of one combined "0xV&&&0xM" string.

  bf_rt's real add_with_<action> convenience methods expose a ternary key
  field as TWO named kwargs (<field> and <field>_mask, see
  bfrtcli._make_core_method_strs), so the entry writer needs the two halves
  separately; nothing in this project ever needed the combined form for
  anything else.'''
  value = ""
  mask = ""

  for bit in codeword:
      if bit == '*':
          value += '0'
          mask += '0'
      else:
          value += bit
          mask += '1'

  bit_length = len(codeword)

  hex_value = hex(int(value, 2))[2:].upper().zfill((bit_length + 3) // 4)
  hex_mask = hex(int(mask, 2))[2:].upper().zfill((bit_length + 3) // 4)

  return f"0x{hex_value}", f"0x{hex_mask}"


def _per_task_code(interval, own_intervals):
  '''The thermometer code of one shared-table interval in one task's OWN
  discretization. The table's intervals are the union of every task's
  thresholds, so each lies wholly inside exactly one of the task's own.'''
  for idx, (low, high) in enumerate(own_intervals):
    if interval[1] <= high:
      if interval[0] < low:
        raise ValueError("interval {} is not inside the task's own interval {}: "
                         "the table's intervals must refine it".format(interval, (low, high)))
      return thermometer_code(len(own_intervals), idx)
  raise ValueError("interval {} is past the task's last own interval {}".format(
      interval, own_intervals[-1]))


# iat_ticks, is_iat_feature, switch_range (the IAT unit rule, Task 34) live in
# p4model.units, which the model's range-row guard shares; imported at the top.


def switch_reading(feature, value):
  '''The value the switch matches for a feature value of the trees' unit, ideally
  read (no clock jitter): iat_ticks(value) for an IAT feature (a gap of value us read
  as round(value / 1.024) ticks, + 1 for C1f), else the value itself.'''
  return iat_ticks(value) if is_iat_feature(feature) else value


def switch_equivalent(feature, value):
  '''The trees' value whose classification the switch reproduces EXACTLY when it reads
  switch_reading(feature, value): the smallest y with the same reading. The switch's
  range entries (switch_range) send reading r = iat_ticks(y) to the interval of y for
  every threshold t (r <= iat_ticks(t) iff y <= t, by minimality), so the entries equal
  the trees on y; y differs from value only where two microsecond values share a tick
  (value - 1 shares it with value; never more than two do, since 1000 / 1024 > 1 / 2).
  The identity for a non-IAT feature.'''
  if not is_iat_feature(feature):
    return value
  y = int(value)
  while y > 0 and iat_ticks(y - 1) == iat_ticks(value):
    y -= 1
  return y


def _range_table_entries(table_name, action_name, key_field, intervals, writes=None,
                         value_of=None):
  '''One feature's range table: one entry per interval, highest interval
  first, each writing that interval's thermometer code. writes: None for a
  table writing one code (parameter `code`), or (action parameter, the task's
  own intervals) per code a per-task table writes (resolve_per_task_plan).
  value_of: None, or interval -> the one value parameter `code` carries
  instead of the interval's code (the packed per-tree field, whose action
  writes every tree's code at once: generate_per_tree_table_entries).

  The key bounds are in the switch's unit (switch_range: IAT features in 1.024 us
  ticks, Task 34); the program text, the codes and the declared table size stay
  those of the trees' intervals, so an interval whose bounds collapse onto one
  tick is simply not installed (its code is then never written, as the switch
  cannot tell its values apart).'''
  feature = key_field[:-len("_val")] if key_field.endswith("_val") else key_field
  entries = []
  for idx, interval in enumerate(intervals[::-1]):
    start, end = switch_range(feature, interval)
    if start > end:
      continue
    if value_of is not None:
      params = {"code": str(value_of(interval))}
    elif writes is None:
      # `code` is the literal parameter name of every single-code set_code_*
      # action (resources/action.p4).
      code = thermometer_code(len(intervals), len(intervals) - 1 - idx)
      params = {"code": str(int(code, 2))}
    else:
      params = {param: str(int(_per_task_code(interval, own), 2)) for param, own in writes}
    # bf_rt exposes a range-matched key field as <short_name>_start /
    # <short_name>_end. The short name is the P4 key expression without its
    # `meta.` prefix -- i.e. exactly the <FEATURE_NAME>+"_val" text
    # generate_tables_and_apply declares this table's key on.
    entries.append({
        "table_name": table_name,
        "action": action_name,
        "is_default_action": False,
        "priority": idx,
        "key_fields": {key_field: {"start": str(start), "end": str(end)}},
        "action_params": params,
    })
  return entries


def _tree_table_entries(table_name, action_name, tree_codewords, fields,
                        use_default_action_discount):
  '''One tree's classification table. fields: (key field, width) per code
  field, in the order generate_codewords concatenated the codeword, or
  (key field, width, slices) for a code the key reads as slices, slices being
  (bf_rt key name, hi, lo) per slice, hi and lo bit positions in the CODE
  (bit 0 its least significant bit), whatever the field it sits in. A slice
  may reach above the code's top bit (a padded per-tree key slice,
  packed.Layout.keyed): the bits past `width` are don't-care.'''
  # Planter-style discount: every leaf sharing this tree's majority class is
  # excluded from this tree's explicit entries and covered by ONE
  # default-action record instead. Computed once per tree, up front, so
  # every other leaf is still written as before.
  default_class, dropped_codewords = None, set()
  if use_default_action_discount and len(tree_codewords) > 0:
    default_class, dropped_list = most_common_class_and_dropped_codewords(tree_codewords)
    dropped_codewords = set(dropped_list)

  entries = []
  priority = 0
  for codeword in tree_codewords:
    if use_default_action_discount and codeword in dropped_codewords:
      continue
    # Tier 3: the classification table's key is one ternary field per
    # selected feature, not one combined codeword field. Slice the
    # combined codeword string into per-feature chunks, in the same
    # order generate_codewords used to build it, and compute a ternary
    # value/mask for each chunk separately. Each chunk's field name is the
    # bf_rt short name of the key generate_tables_and_apply's
    # _classification_keys declares for that feature (meta.code_<feature>,
    # minus the `meta.` prefix).
    key_fields = {}
    bit_offset = 0
    for field in fields:
      name, width = field[0], field[1]
      chunk = codeword[bit_offset:bit_offset+width]  # most significant bit first
      bit_offset += width
      # A code the key reads as slices (packed per-tree codes, generate_per_tree_table_entries)
      # carries (bf_rt name, hi, lo) per slice: each slice gets its own value/mask.
      slices = field[2] if len(field) > 2 and field[2] else [(name, width - 1, 0)]
      # The keyed bits, most significant first: the code, under don't-care bits
      # for whatever its slices key above it.
      keyed = max(hi for _, hi, _ in slices) + 1
      chunk = '*' * (keyed - width) + chunk
      for slice_name, hi, lo in slices:
        hex_value, hex_mask = get_ternary_value_and_mask(chunk[keyed - 1 - hi:keyed - lo])
        key_fields[slice_name] = {"value": hex_value, "mask": hex_mask}
    entries.append({
        "table_name": table_name,
        "action": action_name,
        "is_default_action": False,
        # $MATCH_PRIORITY is per table, so this counter restarts with each tree.
        "priority": priority,
        "key_fields": key_fields,
        # `class` is the literal parameter name of every classify_flow_codeword_*
        # action (see generate_actions).
        "action_params": {"class": str(int(float(tree_codewords[codeword])))},
    })
    priority += 1

  # The discounted leaves' class is not lost: it becomes this table's
  # control-plane-set default action. Without this record, every flow whose
  # codeword was discounted away would match no entry at all and go
  # unclassified.
  if default_class is not None:
    entries.append({
        "table_name": table_name,
        "action": action_name,
        "is_default_action": True,
        # bf_rt's set_default_with_<action> takes only data fields -- a
        # default action has no key and no $MATCH_PRIORITY.
        "priority": None,
        "key_fields": {},
        "action_params": {"class": str(int(float(default_class)))},
    })
  return entries


def generate_table_entries(tasks, use_default_action_discount=False, encoding=JOINT_ENCODING):
  '''The control-plane entries (record format below, installed by
  p4/deploy_table_entries.py) of the program generate_p4_program emits for
  the same `tasks` (Sequence[TaskSpec], N >= 1) and the same `encoding`
  (program.ENCODINGS; any other value raises ValueError):

    JOINT_ENCODING, DISJOINT_ENCODING: resolve_feature_plan; the tasks'
      intervals decide which fields are shared, as in the program text.
    PER_TASK_ENCODING: resolve_per_task_plan, below.
    PER_TREE_ENCODING: generate_per_tree_table_entries.

  Table and field names come from the same resolve_feature_plan the program
  text is built from: one range table per resolved entry, keyed on its raw
  value field, and one table per tree, keyed on its task's code fields of the
  features the tree splits on (decision 2.3, tree_key_features; a tree with
  none raises NoSplitTree, as the program does). So
  namespaced disjoint fields (<task>_<f>) are addressed correctly -- the case
  the legacy get_table_entries (removed 2026-11-13) got wrong. Each task's codewords
  are computed against that task's OWN intervals, which are the intervals of
  every resolved field it reads.

  PER_TASK_ENCODING (resolve_per_task_plan): one range table per feature, holding
  the union of the tasks' intervals, whose action parameter code_<task> is the
  interval's code in that task's own discretization -- which is what that
  task's codewords, against its own intervals, expect to find.

  Every task with trees needs its ordered training-feature-name list
  (TaskSpec.features): its codewords are recomputed from its trees.

  use_default_action_discount: Planter-style default-action discount. When
  True, every one of each tree's leaves sharing the majority class (see
  most_common_class_and_dropped_codewords) is omitted from that tree's
  explicit entries; instead ONE `is_default_action` record per tree carries
  that majority class, which the control plane installs as the table's
  default action at deploy time (see p4/deploy_table_entries.py -- the
  generated P4 itself declares no default action, see
  generate_tables_and_apply). False (the default) writes every leaf as an
  explicit entry and emits no default-action record at all.

  RECORD FORMAT. A flat list of table-entry records covering both the feature
  codeword-bit (range) tables and the codeword-to-leaf-node (classification) tables
  (written as JSON by generate_p4_program's entries_filename):

   Every record's field names are the ones bf_rt's REAL Python API needs, so
   p4/deploy_table_entries.py can pass them straight through as keyword arguments to the
   dynamically generated add_with_<action> / set_default_with_<action> methods
   (bfrtcli._create_add_with_action / _create_set_default_with_action):

   - key_fields: {<bf_rt short field name>: <spec>}, where the short name is the P4 key
     expression with its `meta.`/`hdr.` prefix dropped (bf_rt shortens dotted key names to
     their shortest non-colliding suffix). A RANGE field's spec is
     {"start": <decimal str>, "end": <decimal str>} -> the <name>_start / <name>_end
     kwargs; a TERNARY field's spec is {"value": <hex str>, "mask": <hex str>} ->
     the <name> / <name>_mask kwargs.
   - priority: this entry's 0-indexed position within its OWN table, passed as bf_rt's
     MATCH_PRIORITY kwarg (required for any table with a range or ternary key). Any
     deterministic assignment is correct here: range intervals are disjoint by
     construction and one tree's leaves partition the input space, so no two entries of
     one table can ever match the same input.
   - action_params: {<action data field name>: <decimal str>}. These names are fixed
     literals of this generator's own templates -- `code` for every set_code_* action
     (resources/action.p4) and `class` for every classify_flow_codeword_* action
     (generate_actions).
   - is_default_action: True for the at-most-one-per-classification-table record that
     carries the discounted majority class. Such a record has NO key concept at all
     (bf_rt's set_default_with_<action> takes only data fields), so its key_fields is
     empty and its priority is None.

  Range-table entry:
  {
    "table_name": "table_0_flow_iat_max",
    "action": "set_code_flow_iat_max",
    "is_default_action": false,
    "priority": 0,
    "key_fields": {"flow_iat_max_val": {"start": "0", "end": "100"}},
    "action_params": {"code": "1"}
  }

  Classification-table entry:
  {
    "table_name": "get_classification_tree_app_0",
    "action": "classify_flow_codeword_app_0",
    "is_default_action": false,
    "priority": 3,
    "key_fields": {
      "code_flow_iat_max_2_0_": {"value": "0x5", "mask": "0x7"},
      "code_flow_iat_min_2_0_": {"value": "0x0", "mask": "0x0"}
    },
    "action_params": {"class": "1"}
  }

  Default-action record (only under use_default_action_discount):
  {
    "table_name": "get_classification_tree_app_0",
    "action": "classify_flow_codeword_app_0",
    "is_default_action": true,
    "priority": null,
    "key_fields": {},
    "action_params": {"class": "0"}
  }
  '''
  if encoding not in ENCODINGS:
    raise ValueError("encoding must be one of {}, got {!r}".format(
        ", ".join(repr(name) for name in ENCODINGS), encoding))
  if encoding == PER_TREE_ENCODING:
    return generate_per_tree_table_entries(
        tasks, use_default_action_discount=use_default_action_discount)
  task_intervals = [(task.name, task.intervals) for task in tasks]
  entries = []
  if encoding == PER_TASK_ENCODING:
    table_plan, plan = resolve_per_task_plan(task_intervals)
    own_of = {task.name: task.intervals for task in tasks}
    for index, (feature, (intervals, written)) in enumerate(table_plan.items()):
      writes = [("code_"+task, own_of[task][feature]) for task, _, _ in written]
      entries += _range_table_entries("table_"+str(index)+"_"+feature, "set_code_"+feature,
                                      feature.replace(" ","_").lower()+"_val",
                                      intervals, writes)
  else:
    plan = resolve_feature_plan(task_intervals)
    for index, (name, (raw_feature_name, intervals, _)) in enumerate(plan.items()):
      entries += _range_table_entries("table_"+str(index)+"_"+name, "set_code_"+name,
                                      raw_feature_name.replace(" ","_").lower()+"_val",
                                      intervals)
  # Program order, lowest priority first, as generate_p4_program lists them.
  for task in reversed(list(tasks)):
    if not task.trees:
      continue
    if task.features is None:
      raise ValueError(
          "task %r has trees but no ordered training-feature-name list "
          "(TaskSpec.features): its codewords cannot be recomputed" % task.name)
    intervals = task.intervals or {}
    field_of = {raw: name for name, (raw, _, models) in plan.items() if task.name in models}
    # Each tree keys only the fields of the features it splits on (decision
    # 2.3), so its codewords and entries hold only those (tree_key_features;
    # the program's keys are the same). A task's own field on a feature with no
    # bits does not exist under the per-task encoding, and no tree splits it.
    keyed = tree_key_features(task, field_of.__contains__)
    codewords = tree_codewords(task, keyed)
    for i in range(len(task.trees)):
      # Each code keyed as the slices of its live bits (decision 4.3).
      fields = [code_key_fields("code_"+field_of[raw], len(intervals[raw])-1)
                for raw in keyed[i]]
      entries += _tree_table_entries(
          "get_classification_tree_"+task.name+"_"+str(i),
          "classify_flow_codeword_"+task.name+"_"+str(i),
          codewords[i], fields, use_default_action_discount)
  return entries


# MAX_KEY_SLICE_BITS (64) lives in p4model.target: p4c asserts on a wider ternary key slice.


def code_key_slices(bits):
  '''[(hi, lo)] of a tree key on a `bits`-bit joint / disjoint / per-task code:
  the field is declared at its containers' full width and holds the code in its
  low bits (decision 4.3), so the key reads bits [bits-1:0], cut into slices of
  at most MAX_KEY_SLICE_BITS from bit 0, low first -- the packed per-tree key's
  convention (packed.bit_slices). A code that fills its containers exactly is
  still keyed as a slice, so every key of every encoding has one form.'''
  return packed.bit_slices(bits)


def code_key_fields(field, bits):
  '''One tree-key code field as _tree_table_entries takes it: (field, live bits,
  [(bf_rt key name, hi, lo)] per slice of code_key_slices), the slice names
  bf_rt gives `meta.<field>[hi:lo]` (bfrt_key_name).'''
  return (field, bits, [(bfrt_key_name("meta.%s[%d:%d]" % (field, hi, lo)), hi, lo)
                        for hi, lo in code_key_slices(bits)])


def per_tree_intervals(task):
  '''[{feature: that tree's own intervals}, one dict per tree] for one TaskSpec:
  the generator's interval rule (get_feature_intervals) applied to each tree
  alone, features it never splits left out.'''
  if task.trees and task.features is None:
    raise ValueError(
        "task %r has trees but no ordered training-feature-name list "
        "(TaskSpec.features): its trees' own intervals cannot be derived" % task.name)
  return [{feature: intervals
           for feature, intervals in get_feature_intervals(_Forest([tree]), task.features).items()
           if len(intervals) >= 2}
          for tree in task.trees]


def tree_key_features(task, has_field):
  '''Per tree of one TaskSpec, the raw features its classification key reads
  (decision 2.3, every encoding): the features the tree splits on
  (per_tree_intervals' rule, a threshold in the tree) that have a code field of
  the task (has_field(raw)), in the order of the task's intervals -- the order
  its codewords are concatenated in. A feature the tree never splits is '*' in
  every one of its codewords, so leaving its field out of the key changes no
  match. Raises ValueError for a tree that splits a feature the task has no
  code field for (task intervals that are not its trees': the split would be
  keyed on nothing), and NoSplitTree (args: message, '<task>_<i>') for the
  first tree with no feature at all: its table would have no key (R6, as
  packed.layout_for refuses it under the per-tree encoding).'''
  intervals = task.intervals or {}
  keyed = []
  for index, own in enumerate(per_tree_intervals(task)):
    unkeyed = sorted(raw for raw in own if raw not in intervals or not has_field(raw))
    if unkeyed:
      raise ValueError("tree %s_%d splits %s, which its task has no code field for: the "
                       "task's intervals must be its trees' own" % (task.name, index, unkeyed))
    features = [raw for raw in intervals if raw in own]
    if not features:
      raise NoSplitTree("tree %s_%d has no threshold, so its table would have no key"
                        % (task.name, index), "%s_%d" % (task.name, index))
    keyed.append(features)
  return keyed


def plan_key_order(resolved_plan, task):
  '''[(resolved name, raw feature)] of every code field `task`'s trees may key,
  in the plan's order, which is the program text's: every tree key lists its
  fields in this order, restricted to the features the tree splits on
  (decision 2.3). Training (evaluation._tree_keys) orders each tree's key by
  the same function, because the crossbar's container numbering follows key
  order (xbar.field_key, first use).'''
  return [(name, raw) for name, (raw, _, models) in resolved_plan.items() if task in models]


def tree_codewords(task, keyed):
  '''{tree index: that tree's codeword dict} for one TaskSpec, each tree's
  codewords over only the features its key reads (keyed[i], tree_key_features),
  against the task's own intervals, in their order.'''
  intervals = task.intervals or {}
  paths = get_root_to_leaf_paths(tree_nodes_for(_Forest(task.trees), task.features))
  return {i: generate_codewords({i: paths[i]}, {raw: intervals[raw] for raw in keyed[i]})[i]
          for i in range(len(task.trees))}


def resolve_per_tree_plan(tasks):
  '''The per-tree encoding's plan (program.PER_TREE_ENCODING, spec
  2026-11-06-per-tree-arm-design.md Sec 9.1): resolve_per_task_plan's ONE
  threshold table per raw feature, holding the union of the tasks' intervals,
  whose action writes one code per TREE with a threshold on the feature, as
  wide as that tree's own thresholds, all of them into the feature's ONE packed
  field code_pk_<feature> (spec 2026-11-11-packed-codes-stage-pin Sec 3.1;
  per_tree_layout places each code in it).

  tasks: Sequence[TaskSpec] in TASK order, each with its OWN intervals and,
  when it has trees, its ordered feature list.

  Returns (table_plan, field_plan, tree_intervals):
    table_plan: raw feature -> (the union of the tasks' intervals, the
      (writer, code width) of every code the table writes):
      resolve_per_task_plan's tasks, lowest priority first, each expanded into
      its trees with a threshold on the feature, tree index ascending. writer
      is '<task>_<i>'; every code lands in the feature's one packed field.
    field_plan: resolved name pk_<feature> -> (raw feature, the union of the
      tasks' intervals, {tasks whose trees read the field}), in table_plan's
      order: one packed field per feature, the shape every section that walks
      a plan reads.
    tree_intervals: {task: per_tree_intervals(task)}.'''
  by_task, _ = resolve_per_task_plan([(task.name, task.intervals) for task in tasks])
  tree_intervals = {task.name: per_tree_intervals(task) for task in tasks}
  table_plan, field_plan = {}, {}
  for feature, (union, written_by_task) in by_task.items():
    written = []
    for task, _, _ in written_by_task:
      for i, own in enumerate(tree_intervals[task]):
        if feature in own:
          written.append(("%s_%d" % (task, i), len(own[feature]) - 1))
    table_plan[feature] = (union, written)
    field_plan[packed_field(feature)] = (
        feature, union, {task for task, _, _ in written_by_task
                         if any(feature in own for own in tree_intervals[task])})
  return table_plan, field_plan, tree_intervals


def packed_field(feature):
  '''The resolved name of a feature's packed code field: the program declares
  meta.code_pk_<feature> (spec 2026-11-11-packed-codes-stage-pin Sec 3.1).'''
  return "pk_" + feature


def per_tree_layout(table_plan, tree_intervals, tasks):
  '''The packed layout (p4model.packed.layout_for) of the per-tree program:
  features in table_plan order, tasks in PROGRAM order (lowest priority first,
  the reverse of `tasks`' task order).

  The layout depends on both orders (a tree's lane counts carry over from one
  feature to the next), so it must be the one the training pool prices
  (evaluation._pool_inputs: features list(feature_intervals), tasks [APP_TASK,
  DDOS_TASK]). Both feature lists are sorted by name (resolve_per_task_plan;
  get_feature_thresholds) and layout_for drops a feature no tree has a code on,
  and the two-task (ddos, app) task order reversed is (app, ddos), so the two
  calls get the same orders: checked call against call on seeds 0-5 by
  tests/test_packed_program.py. Raises what layout_for raises (NoSplitTree,
  ActionDataTooWide, PhvContainersExhausted), and ValueError for a range
  table no tree has a code on (task intervals that are not their trees' own):
  its action would have no field to write.'''
  layout = packed.layout_for(packed.widths_of(tree_intervals), list(table_plan),
                             [task.name for task in reversed(list(tasks))],
                             len(iat_scratch_fields(list(table_plan))),
                             len(call_site_constants(list(table_plan))),
                             uses_coarse_clock(list(table_plan)),
                             uses_timestamp(list(table_plan)))
  uncoded = [feature for feature in table_plan if feature not in layout.containers]
  if uncoded:
    raise ValueError("no tree has a code on {}: the tasks' intervals hold thresholds "
                     "their trees do not split on".format(uncoded))
  return layout


def generate_per_tree_table_entries(tasks, use_default_action_discount=False):
  '''The control-plane entries (generate_table_entries' record format, installed by
  p4/deploy_table_entries.py) of the packed per-tree program generate_p4_program
  emits under PER_TREE_ENCODING (spec 2026-11-11-packed-codes-stage-pin Sec 3.1,
  3.2, 8): one field meta.code_pk_<f> per feature, every tree's code on f at its
  bit offset in it (each in bytes of its own, p4model.packed), from the same
  per_tree_layout the program is written from.

  Range table table_<index>_<f> (resolve_per_tree_plan's table_plan, over the
  union of the tasks' intervals): its action set_code_<f> takes ONE parameter
  `code`, the whole field: per interval, each tree's thermometer code of that
  interval in the TREE's own intervals (which is what that tree's codewords,
  built against the same intervals, expect to find), shifted to the tree's bit
  offset and OR-ed (packed.range_value). Bits no tree owns are 0.

  Tree table get_classification_tree_<task>_<i>: one key field per slice of each
  of its own codes (packed.key_slices, at most 64 bits each), named as bf_rt names
  `meta.code_pk_<f>[hi:lo]` (bfrt_key_name, e.g. code_pk_flow_iat_max_19_8_);
  each slice's value and mask are the codeword's bits hi..lo, counted from the
  code's own bit 0 (field bit - the code's bit offset). Keys in the program's order: the
  tree's features in table_plan order (both sorted by name).

  Refuses what the program refuses, before any entry is written: per_tree_layout's
  NoSplitTree, ActionDataTooWide, PhvContainersExhausted and ValueError.'''
  table_plan, _, own = resolve_per_tree_plan(tasks)
  layout = per_tree_layout(table_plan, own, tasks)
  entries = []
  for index, feature in enumerate(table_plan):
    trees = [(task, i) for task, i, f in layout.place if f == feature]

    def value_of(interval, feature=feature, trees=trees):
      return packed.range_value(layout, feature, {
          (task, i): int(_per_task_code(interval, own[task][i][feature]), 2)
          for task, i in trees})
    entries += _range_table_entries("table_"+str(index)+"_"+feature, "set_code_"+feature,
                                    feature.replace(" ","_").lower()+"_val",
                                    table_plan[feature][0], value_of=value_of)
  # Program order, lowest priority first, as generate_p4_program lists them.
  for task in reversed(list(tasks)):
    if not task.trees:
      continue
    if task.features is None:
      raise ValueError(
          "task %r has trees but no ordered training-feature-name list "
          "(TaskSpec.features): its codewords cannot be recomputed" % task.name)
    for i, tree in enumerate(task.trees):
      intervals = own[task.name][i]
      fields = []
      for feature, iv in intervals.items():
        field, base = "code_" + packed_field(feature), layout.place[(task.name, i, feature)][0]
        fields.append((field, len(iv) - 1,
                       [(bfrt_key_name("meta.%s[%d:%d]" % (field, hi, lo)),
                         hi - base, lo - base)
                        for hi, lo in packed.key_slices(layout, task.name, i, feature)]))
      paths = get_root_to_leaf_paths(tree_nodes_for(_Forest([tree]), task.features))
      entries += _tree_table_entries(
          "get_classification_tree_"+task.name+"_"+str(i),
          "classify_flow_codeword_"+task.name+"_"+str(i),
          generate_codewords(paths, intervals)[0], fields, use_default_action_discount)
  return entries


def resolve_feature_plan(task_intervals):
  '''Decides, per raw feature, which tasks can share ONE discretization
  table/codeword field and which need their own (spec 2026-10-11, the N-task
  generalisation of the two-task resolver this replaced).

  task_intervals: (task name, {raw feature: intervals}) pairs in TASK order
  (the order that sets @placement_priority, program.placement_priorities).

  Returns an ordered dict: resolved_name -> (raw_feature_name, intervals,
  models), where `models` is the set of task names whose classification keys
  read this resolved entry's code_<resolved_name> field, and
  `raw_feature_name` is the underlying feature it discretizes (the RAW VALUE
  register/computation is always shared and keyed by raw_feature_name, never
  by resolved_name).

  Per raw feature f, the tasks selecting it are grouped by identical interval
  lists; each group gets one entry, named
    - `f` when the group is EVERY task selecting f (all of them split it
      identically, or only one task selects it at all -- nothing to
      disambiguate), and otherwise
    - `<t1>_..._<tm>_f`, the group's task names in task order (so a
      single-task group is `<task>_f`).
  For two tasks this is exactly the old rule: `f` when both select f with
  identical intervals or one selects it alone, `app_f`/`ddos_f` otherwise.
  Identical intervals are true by construction for joint-derived callers
  (every task passes the literally same dict) and only coincidentally for
  disjoint-derived ones.

  Entry ORDER is program order, which is the reverse of task order: tasks
  are walked lowest priority first, each task's features in its own dict
  order, and a group's entry lands at its first member's position. For
  (ddos, app) that is app's features, then ddos' -- the two-task program
  text exactly.'''
  from collections import OrderedDict
  order = task_names([name for name, _ in task_intervals])
  intervals_of = {name: (intervals or {}) for name, intervals in task_intervals}
  resolved = OrderedDict()
  emitted = set()
  for task in reversed(order):
    for feature, intervals in intervals_of[task].items():
      selectors = [t for t in order if feature in intervals_of[t]]
      group = tuple(t for t in selectors if intervals_of[t][feature] == intervals)
      if (feature, group) in emitted:
        continue
      emitted.add((feature, group))
      if len(group) == len(selectors):
        name = feature
      else:
        name = "_".join(group) + "_" + feature
      # Task and feature names may both hold underscores, so two entries can
      # join to the same name (task a + feature b_c, task a_b + feature c);
      # the second would silently replace the first's field and table.
      if name in resolved:
        # R10: an invariant (AssertionError, see generate_P4_code's catalog check).
        raise AssertionError(
            "resolved feature name %r is produced twice: feature %r for tasks %s and "
            "feature %r for tasks %s; rename a task or a feature"
            % (name, resolved[name][0], sorted(resolved[name][2]), feature, list(group)))
      resolved[name] = (feature, intervals_of[task][feature], set(group))
  return resolved


def _resolve_disjoint_feature_plan(feature_intervals_app, feature_intervals_ddos):
  '''The two-task form of resolve_feature_plan, task order (ddos, app): a
  feature both models select with IDENTICAL intervals shares one plain entry
  (models {"app", "ddos"}), one both select with DIFFERING intervals gets
  namespaced app_<f>/ddos_<f> entries, and a feature only one model selects
  is never namespaced.'''
  return resolve_feature_plan([(DDOS_TASK, feature_intervals_ddos),
                               (APP_TASK, feature_intervals_app)])


def resolve_per_task_plan(task_intervals):
  '''The per-task encoding's plan (program.PER_TASK_ENCODING): ONE threshold
  table per raw feature, its action writing one code per task with a threshold
  on that feature, as wide as that task's own thresholds.

  task_intervals: (task name, {raw feature: the task's OWN intervals}) pairs in
  TASK order, as resolve_feature_plan takes them.

  Returns (table_plan, field_plan):
    table_plan: raw feature -> (the union of the tasks' intervals, the
      (task, resolved field name, field width) of every code the table writes,
      lowest-priority task first, which is program order). Sorted by feature
      name: feature_intervals_from_nodes sorts the merged 'joint' intervals the
      same way, so everything but the code fields (registers, range-table
      order, table sizes) is what the same forests get under 'joint'.
    field_plan: resolve_feature_plan's shape over the code fields, resolved name
      tk_<task>_<feature> -> (raw feature, the task's own intervals, {task}), so
      every section that walks a plan -- metadata, pragmas, keys -- is unchanged.

  A task with no threshold on a feature has no field for it. No field is ever
  read by two tasks, so @pa_container_size cannot conflict.'''
  order = task_names([name for name, _ in task_intervals])
  intervals_of = {name: (intervals or {}) for name, intervals in task_intervals}
  features = sorted(set().union(*(set(own) for own in intervals_of.values())))
  table_plan, field_plan = {}, {}
  for feature in features:
    thresholds = set()
    written = []
    for task in reversed(order):
      own = intervals_of[task].get(feature)
      if own is None or len(own) < 2:
        continue
      thresholds.update((feature, high) for _, high in own[:-1])
      name = per_task_field(task, feature)
      if name in field_plan:
        # R10: an invariant (AssertionError, see generate_P4_code's catalog check).
        raise AssertionError(
            "resolved field name %r is produced twice: feature %r of task %r and "
            "feature %r of task %r; rename a task or a feature"
            % (name, field_plan[name][0], sorted(field_plan[name][2])[0], feature, task))
      field_plan[name] = (feature, own, {task})
      written.append((task, name, len(own) - 1))
    if written:
      union = get_feature_intervals_from_thresholds(sorted(thresholds))[feature]
      table_plan[feature] = (union, written)
  return table_plan, field_plan


def code_field_container_sizes(resolved_plan, tasks):
  '''resolved_name -> @pa_container_size sizes (low slice first) for every
  code_<resolved_name> field of a listed task (the generator lists every task,
  so every declared field with bits is pinned; decision 8.2).

  A field's pin is lanes.container_sizes(width, lanes.layout): the width rule
  alone, whatever else its key holds (decision 4.1), and the layout the model
  prices the key with (lanes.key_fields), so p4c lays the key out the way it
  is priced (spec 2026-09-29 Sec 1.2: unpinned, PHV pressure moved 17-32-bit
  fields into whole W containers). A field two keys read (disjoint, identical
  intervals) has one width, hence one pin.

  A field with no bits has no layout and gets no entry (a pragma with no
  sizes does not parse).'''
  sizes = {}
  for task in tasks:
    for name, (_, intervals, models) in resolved_plan.items():
      if task not in models:
        continue
      pin = lanes.container_sizes(max(len(intervals) - 1, 0), lanes.layout)
      if pin:
        sizes[name] = pin
  return sizes


def generate_P4_actions(feature_intervals, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos):
  """The two-task form of generate_actions (app's actions, then ddos'). No production
  caller; kept for tests/test_build_p4_script_tna.py, which drives it."""
  return generate_actions(feature_intervals,
                          [(APP_TASK, num_trees_app, bit_per_classes_app),
                           (DDOS_TASK, num_trees_ddos, bit_per_classes_ddos)])


def generate_actions(feature_intervals, task_trees, per_task_writes=None,
                     packed_bits=None, field_bits=None):
  """
      action <ACTION_NAME> (bit<<ACTION_CODE_LENGTH>> code) {
          meta.code_<FEATURE_NAME> = code;
      }

  task_trees: (task name, tree count, class bits) per task, in PROGRAM order.
  per_task_writes: None for one code per entry of feature_intervals, or, under
  the per-task encoding, entry name -> the (writer, field, width) list of
  resolve_per_task_plan: one action parameter code_<writer> per code the table
  writes.
  packed_bits: under the per-tree encoding, entry name -> the container bits N
  of its packed field: the action takes ONE parameter `bit<N> code` and writes
  it whole to meta.code_pk_<entry> (spec 2026-11-11-packed-codes-stage-pin Sec
  3.1), every tree's code already at its bit offset in the value.
  field_bits: None, or resolved field name -> its DECLARED bits (decision 4.3:
  a joint / disjoint / per-task code field is declared at its containers' full
  width); each parameter writing that field takes that width, so the action
  writes the field whole, its live code in the low bits and 0 above (the
  entries' values are the live codes). A field it does not name keeps its live
  width.
  """
  def declared(name, live):
    return live if field_bits is None else field_bits.get(name, live)


  # Calculate nºbits per feature in the codeword
  feature_names = feature_intervals.keys()
  codeword_bits_per_feature = {}
  for feature in feature_names:
    codeword_bits_per_feature[feature]=len(feature_intervals[feature])-1


  action_templates = "" #Stores P4 actions

  # Classification action templates: one dedicated action per tree.
  #
  # TNA rejects a runtime "if (tree == i) {...}" branch inside an action when
  # the branch decides which of several DIFFERENT metadata fields gets
  # written (a rejected IR::Mux over an action-data-parameter conditional --
  # p4c's ActionAnalysis pass: "Conditions in an action must be simple
  # comparisons of an action data parameter", see af64bc2 and
  # reviews/t11_tofino_port_and_env.md Part G.2). Since each tree already
  # gets its own physical classification table
  # (get_classification_tree_app_0/_1/_2/..., built in
  # generate_tables_and_apply below), there is no need for a single
  # shared, tree-parameterized action at all: each tree's table binds to its
  # own dedicated action that unconditionally writes only its own field. No
  # `tree` parameter, no conditional, no Mux -- this generalizes uniformly to
  # any tree count per task, including 1. Validated against the real
  # TNA compiler in p4/tofino_spike/tna_m2_numtrees3_spike.p4.
  # A task with ONE tree (review A2, adopted 2026-11-15) has no vote: its tree's
  # action writes the task's export field itself, with the vote action's cast
  # (vote_export), and there is no class_tree_<task>_0 field and no vote table.
  for task, n_trees, bits in task_trees:
    if n_trees > 0:
      classification_action_template = ""
      for i in range(n_trees):
        classification_action_template += "\taction classify_flow_codeword_"+task+"_"+str(i)+"(bit<"+str(bits)+"> class){\n"
        if n_trees == 1:
          export_field, export_bits = vote_export(task, bits)
          classification_action_template += "\t\t%s = (bit<%d>) class;\n" % (export_field, export_bits)
        else:
          classification_action_template += "\t\tmeta.class_tree_"+task+"_"+str(i)+" = class;\n"
        classification_action_template += "\t}\n\n"

      action_templates += classification_action_template

  for feature in feature_names:
    if packed_bits is not None:
      action_templates += ("    \n    action set_code_"+feature+" (bit<"+str(packed_bits[feature])
                           + "> code) {\n        meta.code_"+packed_field(feature)+" = code;\n    }\n")
      continue
    if per_task_writes is not None:
      params = ", ".join("bit<"+str(declared(field, width))+"> code_"+task
                         for task, field, width in per_task_writes[feature])
      body = "".join("        meta.code_"+field+" = code_"+task+";\n"
                     for task, field, _ in per_task_writes[feature])
      action_templates += ("    \n    action set_code_"+feature+" ("+params+") {\n"
                           + body + "    }\n")
      continue
    action_template = _ACTION_TEMPLATE
    action_template = action_template.replace("<ACTION_NAME>", "set_code_" + feature)
    action_template = action_template.replace(
        "<ACTION_CODE_LENGTH>", str(declared(feature, codeword_bits_per_feature[feature])))
    action_template = action_template.replace("<FEATURE_NAME>", feature)
    action_templates += action_template

  return action_templates


def generate_P4_tables_and_apply(feature_names, num_trees_app, num_trees_ddos,
                                  match_type='ternary',
                                  feature_names_app=None, feature_names_ddos=None,
                                  raw_feature_names=None,
                                  *,
                                  feature_table_sizes,
                                  classification_table_sizes,
                                  config: "p4_gen_config.P4GenConfig" = None):
  """The two-task form of generate_tables_and_apply: app's trees, then
  ddos', at program.PLACEMENT_PRIORITY. classification_table_sizes is keyed
  by the old global tree id (0..num_trees_app-1 app, then ddos); the other
  arguments are documented on generate_tables_and_apply. No production caller
  (it emits whole-field tree keys without slices or @name, a form no entry
  writer uses); kept for tests/test_build_p4_script_tna.py, which drives it."""
  feature_names = list(feature_names)
  task_tables = []
  for task, n_trees, names, first in (
      (APP_TASK, num_trees_app, feature_names_app, 0),
      (DDOS_TASK, num_trees_ddos, feature_names_ddos, num_trees_app)):
    sizes = [classification_table_sizes[first + i] for i in range(n_trees)]
    task_tables.append((task, list(names) if names is not None else feature_names,
                        sizes))
  return generate_tables_and_apply(
      feature_names, task_tables, match_type=match_type,
      raw_feature_names=raw_feature_names,
      feature_table_sizes=feature_table_sizes, priorities=PLACEMENT_PRIORITY,
      config=config)


def generate_tables_and_apply(feature_names, task_tables, match_type='ternary',
                              raw_feature_names=None, *, feature_table_sizes,
                              priorities, config: "p4_gen_config.P4GenConfig" = None):
  """
  task_tables: (task name, key feature names, per-tree sizes) per task, in
  PROGRAM order (the reverse of task order). Each task's tree i gets table
  get_classification_tree_<task>_<i> keyed on meta.code_<name> for every
  name in its key feature names, sized by sizes[i], with
  @placement_priority(priorities[task]). The key feature names are one list
  for every tree of the task, or {tree index: names} when each tree keys its
  own codes (PER_TREE_ENCODING). A key entry may be a (name, [(hi, lo), ...])
  pair instead of a name: the key reads those slices of meta.code_<name>, one
  `meta.code_<name>[hi:lo]` line per slice in the order given (the packed
  per-tree program, packed.key_slices).
  config: Task 4 -- additive convenience. When given, `config.match_type`
  takes precedence over the individual `match_type` keyword argument above
  (which remains the source of truth when `config` is None, so every
  existing caller is unaffected).

      table <TABLE_NAME> {
          key = {
              meta.<FEATURE_NAME>: <MATCH_TYPE>;
          }
          actions = {
              <ACTIONS>
          }
          size = <SIZE>;
      }

  NO DEFAULT ACTION is declared for any table this function emits -- not
  even for the classification tables under the Planter-style
  default-action discount (`use_default_action_discount`, which this
  function therefore no longer takes at all; it survives only in
  generate_P4_code/generate_table_entries, where it decides which entries the
  control plane installs). Task 1 used to emit
  `const default_action = classify_flow_codeword_<task>_<i>(<class>);`
  here. A controlled real-compile A/B this session showed that construct
  costs +1 real pipeline stage per modified table (+2 with all four
  modified, 9 -> 11) while table_summary.log's own "critical path length
  through the table dependency graph" stayed at 9 in every variant -- a
  compiler placement artifact, not a resource shortage. Deleting the line
  entirely (leaving the table's default implicitly NoAction in the compiled
  P4) restored exactly 9 stages with identical tcam/sram/gateway/map_ram,
  i.e. declaring nothing is real-compiler-validated to be exactly as cheap
  as it gets. Planter's own shipped decision table
  (In-Network-Machine-Learning/Planter, P4/DT_performance_Iris_EB.p4)
  likewise declares no compile-time-constant default action; its real
  default class is installed by the control plane at deploy time, through
  the same runtime call that installs every other entry. This generator now
  matches that architecture: generate_table_entries emits a real
  `is_default_action` record per discounted tree and
  p4/deploy_table_entries.py installs it via bf_rt's
  `set_default_with_<action>`.

  match_type: Task 8 -- Planter RF_EB-style exact-match code/decision
  tables. 'ternary' (the default) is byte-identical to every caller before
  this task. 'exact' selects resources/table_classification_exact.p4
  instead of resources/table_classification.p4 for the CLASSIFICATION
  tables only, and keys them on `: exact;` instead of `: ternary;`. The
  feature-range tables below are unaffected either way -- Planter's RF_EB
  scheme keeps ternary/range feature tables, only the code/decision
  (classification) tables move to exact match. Callers are responsible for
  actually enumerating wildcarded codewords into concrete exact-match
  entries first (see evaluation.exact_match_resource_usage for the
  analytical accounting of that multiplier); this function only emits the
  table declarations' match kind, not the table_entries.json rows.

  feature_names: Task 3 -- the full pool of RESOLVED names (see
  _resolve_disjoint_feature_plan) that get a range-matching table declared,
  one per resolved name, regardless of which model(s) actually read it.

  feature_names_app / feature_names_ddos (two-task wrapper only): Task 3,
  optional. Restrict which of `feature_names` actually appear as a
  `meta.code_<resolved_name>` key field in the App / DDoS classification
  tables respectively -- a resolved name selected by only one model (e.g. a
  namespaced "ddos_<feature>" entry, or a feature only one model selected at
  all) must not appear as a key in the OTHER model's classification table.
  Each defaults to `feature_names` itself when omitted (None), exactly
  reproducing every pre-Task-3 caller's behavior -- both models keying on
  the identical, full feature set -- byte-for-byte.

  raw_feature_names: Task 3, optional. Maps resolved_name -> raw_feature_name
  for the range-matching tables' <FEATURE_NAME> substitution -- the RAW
  tracked-value field (`meta.<raw_feature_name>_val`) that a resolved
  name's own discretization table reads its input from (see
  _resolve_disjoint_feature_plan's docstring for why this can differ from
  resolved_name under genuine disjoint encoding: two namespaced entries,
  e.g. app_<feature>/ddos_<feature>, both read the SAME shared raw value).
  A resolved name absent from this mapping (or the whole mapping omitted)
  falls back to using resolved_name itself as the raw name, reproducing
  every pre-Task-3 caller's behavior (which never had this distinction)
  byte-for-byte.

  feature_table_sizes / per-tree sizes: real, per-table `size = `
  values -- required (keyword-only, no default). A fixed literal size,
  disconnected from how many entries each table can actually receive, is not
  free on real hardware (reviews/p4_tofino_reference.md Sec 4.4), so every
  caller must supply the real per-table counts; a caller that omits either
  gets a TypeError at the call instead of a silently oversized table.

  feature_table_sizes maps RESOLVED feature name -> entry count (one entry
  per interval; see generate_table_entries' range tables). Each task's
  per-tree sizes are indexed by that task's own tree id (no global offset).
  """

  if config is not None:
    match_type = config.match_type

  if match_type not in ('ternary', 'exact'):
    raise ValueError("match_type must be 'ternary' or 'exact', got {!r}".format(match_type))

  # Task 19: both are already-loaded strings (_TABLE_TEMPLATE /
  # _TABLE_CLASSIFICATION_EXACT_TEMPLATE, read once at import) -- no file I/O
  # happens here anymore.
  #
  # match_type='exact' selects _TABLE_CLASSIFICATION_EXACT_TEMPLATE below, but
  # that path is DEFERRED, not abandoned: it does not yet produce a loadable
  # program, since callers still have to enumerate wildcarded codewords into
  # concrete exact-match entries themselves, and at this project's real
  # feature scale that comes out to ~1.3x10**34 entries -- physically
  # impossible for real SRAM. See reviews/todo.md:343-349 (2026-08-03) and
  # .superpowers/plans/2026-08-03-p4-generator-fixes-and-config.md Task 2 (on
  # hold pending that decision).
  classification_table_template = (
      _TABLE_TEMPLATE if match_type == 'ternary'
      else _TABLE_CLASSIFICATION_EXACT_TEMPLATE
  )

  table_templates = ""
  apply_templates_tmp = "\n"
  apply_templates = ""

  feature_names = list(feature_names)
  # Task 3: each task's classification tables key on its OWN resolved
  # feature set (the two-task wrapper defaults each to the full
  # `feature_names` pool, so every pre-Task-3 caller is byte-identical).
  raw_feature_names = raw_feature_names or {}

  # Tier 3: the classification tables key on one field per selected feature
  # (meta.code_<feature>), not a single combined meta.codeword field. Task 8:
  # the match kind itself (ternary vs exact) is decided by match_type, not
  # hardcoded -- this is the only place "ternary"/"exact" actually lands in
  # the generated key text, since neither template file declares its own
  # <MATCH_TYPE> marker (match type is baked into <KEYS> here, not read from
  # the template).
  # A sliced key element carries an explicit @name, the bf_rt name the entry
  # writer uses (bfrt_key_name, e.g. code_x_19_0_; review B1, 2026-11-15).
  # Without it p4c strips the slice from the name it gives the driver ("Table
  # key name not supported. Replacing `meta.code_x[1:0]` with `meta.code_x`"),
  # so context.json's key names differ from the entry writer's, and two slices
  # of one field in one table get the same name. p4c (Task 28): with the
  # annotation all 54 such warnings of the probed program gone, context.json
  # names unique and equal to bfrt_key_name, bf-rt.json the same names,
  # resources identical.
  def _classification_keys(names):
    lines = []
    for feature in names:
      if isinstance(feature, tuple):
        feature, slices = feature
        lines += ['            meta.code_%s[%d:%d] : %s @name("%s");'
                  % (feature, hi, lo, match_type,
                     bfrt_key_name("meta.code_%s[%d:%d]" % (feature, hi, lo)))
                  for hi, lo in slices]
      else:
        lines.append("            meta.code_"+feature+" : "+match_type+";")
    return "\n".join(lines)

  #Classification tables, task by task in program order
  for position, (task, key_names, sizes) in enumerate(task_tables):
    keys = None if isinstance(key_names, dict) else _classification_keys(key_names)
    for i, size in enumerate(sizes):
      if isinstance(key_names, dict):
        keys = _classification_keys(key_names[i])
      table_template = classification_table_template
      table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_"+task+"_"+str(i))
      table_template = table_template.replace("<KEYS>", keys)
      table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_"+task+"_"+str(i)+";")
      table_template = table_template.replace("<SIZE>", str(size))
      # C5 (Phase 3 step 1): pin a deterministic ternary-table placement
      # order -- the first task's trees ahead of the next one's -- so the
      # packer's stage simulation knows which key p4c's greedy crossbar
      # allocator serves first, instead of charging the worst-case ordering.
      # Priority is per TASK, not per tree: task i of N gets N - i
      # (program.placement_priorities; two tasks: ddos 2, app 1); p4c breaks
      # ties within a priority by declaration order, which the "last-listed
      # tree first" rule relies on.
      # Kept (design review 5.1/5.2, gate G3 2026-11-15): it forbids nothing in
      # PHV, it fixes the ORDER p4c's placer serves ready tables in. Free on
      # 160/160 p4c compiles of the Task 17 base designs (deleting every
      # @placement_priority changed no p4c stage depth or block count, every
      # model table on the same p4c stage); kept because earlier campaign
      # evidence showed it load-bearing for disjoint designs at scale.
      table_name = "get_classification_tree_"+task+"_"+str(i)
      table_template = table_template.replace(
          "    table "+table_name+" {",
          "    @placement_priority("+str(priorities[task])+")"
          "\n    table "+table_name+" {",
          1,
      )
      table_templates += table_template
      apply_templates_tmp += "\t\t\tget_classification_tree_"+task+"_"+str(i)+".apply();\n"
    # One blank line closes every task's apply block but the last one's.
    if sizes and position < len(task_tables) - 1:
      apply_templates_tmp += "\n"

  range_priority = range_table_priority(priorities)
  feature_idx = 0
  for feature in feature_names:
    # Task 3: the range table's key reads the RAW tracked-value field
    # (shared, model-independent) even when this resolved entry's NAME is
    # namespaced (app_<feature>/ddos_<feature>) -- defaults to `feature`
    # itself (raw_feature_names omitted or missing this key), reproducing
    # every pre-Task-3 caller's behavior byte-for-byte.
    raw_name = raw_feature_names.get(feature, feature)
    # Task 19: the range path now shares _TABLE_TEMPLATE with the
    # classification tables above, so it builds its own single-line <KEYS>
    # text in Python -- exactly reproducing the old table.p4 template's
    # fixed "meta.<FEATURE_NAME>: <MATCH_TYPE>;" line with <FEATURE_NAME> and
    # <MATCH_TYPE> substituted the same way the pre-Task-19 code did.
    table_template = _TABLE_TEMPLATE
    table_template = table_template.replace("<TABLE_NAME>","table_"+str(feature_idx)+"_"+feature)
    table_template = table_template.replace(
        "<KEYS>", "            meta."+raw_name.replace(" ","_").lower()+"_val: range;")
    table_template = table_template.replace("<ACTIONS>", str("set_code_"+feature)+";")
    size = feature_table_sizes[feature]
    # table.p4 (the range path's now-retired template file) had 3 trailing
    # spaces on its closing "}" line that table_classification.p4 (now
    # _TABLE_TEMPLATE, shared by both paths) does not -- see the
    # _TABLE_TEMPLATE module comment. Restore them via this anchored replace
    # (unique: <SIZE> appears exactly once) before substituting <SIZE>
    # itself, so the range tables' emitted bytes are unchanged from every
    # caller before this task.
    table_template = table_template.replace("<SIZE>;\n    }", "<SIZE>;\n    }   ")
    table_template = table_template.replace("<SIZE>", str(size))
    # Above every tree's priority, so p4c serves the range pool first, as the
    # model's packer assumes (program.range_table_priority has the evidence).
    # Kept at gate G3 (5.1/5.2) with the tree priorities: free on 160/160 p4c
    # compiles (see the tree tables' comment above).
    table_template = table_template.replace(
        "    table table_"+str(feature_idx)+"_"+feature+" {",
        "    @placement_priority("+str(range_priority)+")"
        "\n    table table_"+str(feature_idx)+"_"+feature+" {",
        1,
    )
    table_templates += table_template
    apply_templates += "\t\t\ttable_"+str(feature_idx)+"_"+feature+".apply();\n"

    feature_idx += 1

  apply_templates += apply_templates_tmp

  return table_templates, apply_templates


def vote_export(task, class_bits):
  """(export field, its bits) the task's result is written to (decision 1f,
  program.VOTE_EXPORT): by the vote action, or by the tree action of a task
  with one tree (review A2). ValueError for a task with no destination or a
  class wider than it."""
  if task not in VOTE_EXPORT:
    raise ValueError(
        "task %r has trees but no vote export destination; program.VOTE_EXPORT "
        "maps only %s" % (task, sorted(VOTE_EXPORT)))
  export_field = VOTE_EXPORT[task]
  export_bits = VOTE_EXPORT_BITS[export_field]
  if class_bits > export_bits:
    raise ValueError("task %r needs %d bits for its class but %s holds %d"
                     % (task, class_bits, export_field, export_bits))
  return export_field, export_bits


def generate_voting_code(num_trees, num_classes, task):
  """
  Returns (table_declaration_text, apply_call_text). Replaces the previous
  27-branch (for num_trees=3, num_classes=3) if-cascade with a single
  exact-match table -- validated this session against the real Tofino
  compiler (p4/tofino_spike/tna_m2_vote_table_spike.p4 and the real-program
  test p4/tofino_spike/tna_m2_real_with_vote_table.p4, both 0 errors):
  1 fewer ingress stage and Gateway usage cut from 37 to 10 for M2's real
  3-tree/3-class case. The tie-break is switch_semantics.vote_winner
  (smallest class index), not the statistics.mode() this function used
  previously: mode() returns the FIRST-ENCOUNTERED mode, so its winner
  depended on tree ordering, which is arbitrary. vote_winner is
  order-independent, and it is the rule every accuracy measurement in the
  pipeline now uses, so this is a deliberate behavior change, not just a
  mechanism change -- on tied key tuples the winner can differ from the old
  statistics.mode() table.
  """
  # Since review A2 (2026-11-15) a one-tree task has no vote table (its tree
  # action writes the export field) and the model prices a vote stage only for
  # a task with more than one tree (usage.pipeline_depth), so a one-tree vote
  # table is refused here rather than emitted unpriced (Task 31 review minor 5).
  if num_trees < 2:
    raise ValueError("generate_voting_code: a task with %d tree(s) has no vote table "
                     "(review A2); its tree writes the export field itself" % num_trees)
  bit_per_classes = math.ceil(math.log2(num_classes)) or 1
  export_field, export_bits = vote_export(task, bit_per_classes)
  # The table emits exactly num_classes ** num_trees const entries (see the
  # product() loop below), so that IS its size -- the real logical entry
  # count, per reviews/p4_tofino_reference.md Sec 4.4. (A `max(32, ...)` floor
  # used to be applied; p4c shrinks a table to its key space with a warning,
  # e.g. "Shrinking table SwitchIngress.vote_ddos: with 1 match bits, can only
  # have 2 entries" for the one-tree vote table the generator no longer emits.)
  size = num_classes ** num_trees

  key_lines = "\n".join(
      "\t\t\tmeta.class_tree_{}_{} : exact;".format(task, i)
      for i in range(num_trees)
  )

  entries_lines = []
  for classification_array in product(range(num_classes), repeat=num_trees):
    # switch_semantics.vote_winner, not statistics.mode: mode returns the
    # FIRST-ENCOUNTERED mode, so on a tie the winner depended on tree ordering
    # -- arbitrary, and changed if trees were reordered. Smallest-class-index is
    # order-independent, and it is the rule every accuracy measurement in the
    # pipeline now uses, so the reported number and the table agree by
    # construction. Same key space and same entry count either way, so this
    # costs nothing in resources.
    winner = switch_semantics.vote_winner(classification_array, num_classes)
    key_tuple = ", ".join(str(c) for c in classification_array)
    entries_lines.append(
        "\t\t\t({}) : set_classification_{}({});".format(key_tuple, task, winner)
    )
  entries_block = "\n".join(entries_lines)

  table_decl = (
      "\taction set_classification_{task}(bit<{bits}> winner) {{\n"
      "\t\t{export} = (bit<{ebits}>) winner;\n"
      "\t}}\n"
      "\n"
      "\ttable vote_{task} {{\n"
      "\t\tkey = {{\n"
      "{keys}\n"
      "\t\t}}\n"
      "\t\tactions = {{\n"
      "\t\t\tset_classification_{task};\n"
      "\t\t}}\n"
      "\t\tsize = {size};\n"
      "\t\tconst entries = {{\n"
      "{entries}\n"
      "\t\t}}\n"
      "\t}}\n"
  ).format(task=task, bits=bit_per_classes, export=export_field,
      ebits=export_bits, keys=key_lines, entries=entries_block, size=size)

  apply_call = "\t\t\tvote_{}.apply();\n".format(task)

  return table_decl, apply_call


def generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos,
                      feature_intervals_app, feature_intervals_ddos,
                      output_dir=OUTPUT_PATH, output_filename='p4_code_RF_models.p4',
                      match_type='ternary',
                      use_default_action_discount=False,
                      selected_features_app=None, selected_features_ddos=None,
                      config: "p4_gen_config.P4GenConfig" = None,
                      entries_filename=None, encoding=JOINT_ENCODING):
  """The two-task program (spec 2026-10-11 compatibility wrapper):
  generate_p4_program with TaskSpecs in the order (ddos, app), which gives
  the C5 priorities ddos 2 / app 1 and today's program text byte for byte.
  clf_app / clf_ddos may be None for "no such task" (its trees are then
  empty); feature_intervals_* are each model's own intervals ({} for a
  missing task, the same dict for both under joint encoding; each model's
  own, always, under PER_TASK_ENCODING and PER_TREE_ENCODING);
  selected_features_* are the ordered training-feature-name lists. See
  generate_p4_program for everything else, `encoding` included."""
  tasks = [
      TaskSpec(DDOS_TASK, num_class_ddos,
               trees=list(clf_ddos.estimators_) if clf_ddos is not None else [],
               features=selected_features_ddos, intervals=feature_intervals_ddos),
      TaskSpec(APP_TASK, num_class_app,
               trees=list(clf_app.estimators_) if clf_app is not None else [],
               features=selected_features_app, intervals=feature_intervals_app),
  ]
  return generate_p4_program(
      tasks, output_dir=output_dir, output_filename=output_filename,
      match_type=match_type, use_default_action_discount=use_default_action_discount,
      config=config, entries_filename=entries_filename, encoding=encoding)


class _Forest:
  """The `.estimators_` view tree_nodes_for reads, over a TaskSpec's trees."""
  def __init__(self, trees):
    self.estimators_ = list(trees)


def generate_p4_program(tasks, output_dir=OUTPUT_PATH,
                        output_filename='p4_code_RF_models.p4',
                        match_type='ternary', use_default_action_discount=False,
                        config: "p4_gen_config.P4GenConfig" = None,
                        entries_filename=None, encoding=JOINT_ENCODING):
  """One P4 program serving every task in `tasks` (Sequence[TaskSpec], N >= 1,
  spec 2026-10-11). Task order sets @placement_priority (task i of N gets
  N - i, every range table N + 1, program.range_table_priority); the program
  lists the tasks in the reverse order, lowest priority
  first. entries_filename: when given, the program's control-plane entries
  (generate_table_entries) are written beside it under that name; None
  writes the program only. Per task: get_classification_tree_<task>_<i> /
  classify_flow_codeword_<task>_<i> / meta.class_tree_<task>_<i> for its
  tree i (ids per task, from 0), and, for a task with two or more trees, one
  vote_<task> table; a task with one tree has neither the class field nor the
  vote table, its tree action writes the task's export field (review A2).
  Registers stay
  one per distinct raw feature across all tasks; codeword fields follow
  resolve_feature_plan.

  encoding: one of program.ENCODINGS (anything else raises ValueError).
    JOINT_ENCODING (the default) and DISJOINT_ENCODING: resolve_feature_plan,
      where the TaskSpecs' intervals decide what is shared -- the same dict
      for every task gives the one-code program, each task's own gives the
      namespaced one.
    PER_TASK_ENCODING: resolve_per_task_plan. Every TaskSpec carries its OWN
      intervals, the one range table per feature holds their union, and its
      action writes meta.code_tk_<task>_<feature> for every task with a
      threshold on the feature.
    PER_TREE_ENCODING: resolve_per_tree_plan, the same tables, each writing
      ONE packed field meta.code_pk_<feature> that holds the code of every
      TREE with a threshold on the feature in bytes of its own (p4model.packed; spec
      2026-11-11-packed-codes-stage-pin Sec 3.1; per_tree_layout, the layout
      the training pool prices). The field is declared at its containers'
      bits, pinned @pa_no_overlay + @pa_container_size (low container first),
      never @pa_solitary; the action takes one `bit<N> code`; each tree keys
      only its own codes, as slices of at most 64 bits (packed.key_slices),
      features in the tree's feature order. The layout raises NoSplitTree /
      ActionDataTooWide / PhvContainersExhausted before anything is written.
      Every task with trees needs its ordered feature list.
    In every encoding a tree keys only the code fields of the features it
    splits on (decision 2.3; tree_key_features), so the encodings differ only in
    their code scheme; every code field is still declared, pinned and written by
    its range table. In every encoding a code field is declared at its
    containers' full width and a tree keys slices of it (decision 4.3): under
    joint, disjoint and per-task codes the field holds its live code in the low
    bits, the range action's parameter is as wide as the field (the code
    zero-extended), and each tree keys `meta.code_<f>[w-1:0]` (code_key_slices,
    at most 64 bits per slice). A tree that splits on nothing raises NoSplitTree (R6)
    before anything is written. Under joint, disjoint and per-task codes a task
    passed without its feature list keys all of its fields on every tree.

  match_type: passed straight through to
  generate_tables_and_apply. 'ternary' (the default) switches
  classification tables to resources/table_classification.p4 with ternary
  keys; 'exact' switches only the classification tables to
  resources/table_classification_exact.p4 / `: exact;` keys (feature-range
  tables stay ternary/range either way).

  config: additive convenience. When given, `config.match_type`
  and `config.use_default_action_discount` take precedence over the
  individual `match_type` / `use_default_action_discount` keyword arguments
  above (which remain the source of truth when `config` is None).

  use_default_action_discount: the Planter-style default-action discount. When
  the flag is True (passed directly or via `config`), every leaf carrying its
  tree's majority class comes off that classification table's declared `size`
  here, and off its explicit entries in table_entries.json, where
  `generate_table_entries` instead writes ONE `is_default_action` record per
  tree carrying that majority class. The generated P4 itself declares NO
  default action either way (p4/deploy_table_entries.py installs the default
  class at deploy time -- Planter's own architecture). When the flag is False
  -- the default -- every leaf stays an explicit entry and no
  is_default_action record is written at all.

  Codewords are computed PER TASK, against that TaskSpec's OWN intervals: under
  disjoint encoding two tasks' intervals for a shared feature name can differ,
  so one combined call would produce wrong codewords for one side. Each task's
  codewords are keyed by that task's own tree id. Doing that requires the
  task's ORIGINAL ORDERED training-feature-name list (TaskSpec.features), which
  cannot be replaced by the intervals' keys: `get_feature_thresholds` sorts
  alphabetically by feature name, whereas `get_nodes(estimator, feature_names)`
  names a node's feature `feature_names[estimator.tree_.feature[node]]`, so
  `feature_names[i]` must be training column `i`. Raises ValueError when the
  flag is True and a task with trees has no feature list -- silently guessing
  an order would produce wrong-but-plausible codewords.

  TABLE SIZING: every table this function emits is sized from its REAL entry
  count -- range-matching feature tables from their interval count,
  classification tables from their codeword count (minus the entries
  use_default_action_discount folds into the default action); a task without
  a feature list falls back to `estimator.get_n_leaves()`, a structural count
  that is always >= the codeword count, so a size is never underestimated.

  KNOWN LIMITATION of that sizing: with match_type='exact' it still uses the
  ternary codeword/leaf count, NOT the real Cartesian-product-expanded
  exact-match entry count that enumerating each wildcarded codeword would
  produce (see evaluation.exact_match_resource_usage for the analytical
  accounting) -- enumerating those entries is deferred, separate work
  (reviews/todo.md's 2026-08-03 T0 update), so under 'exact' these sizes can
  be far too small; this is a pre-existing gap this sizing work does not
  close, not one it introduces.

  CAVEAT: with match_type='exact', the emitted P4 program is NOT yet
  end-to-end compilable/loadable on its own. generate_table_entries (writes
  table_entries.json) is unchanged by this parameter and still emits
  ternary/wildcard-shaped ('*') entries for the same classification tables
  that this function just declared `: exact;` -- concrete, enumerated
  exact-match entries for those wildcarded codewords still need to be
  generated as follow-on work before 'exact' output can actually be loaded.

  Each TaskSpec's intervals are that task's OWN, independently derived
  feature_intervals dict (raw feature name -> interval list); a joint-encoded
  caller passes the SAME dict for every task. Under joint / disjoint codes
  resolve_feature_plan resolves them once: a feature every selecting task
  discretizes IDENTICALLY shares ONE table/field; otherwise each task gets its
  own namespaced (<task>_ prefixed) table/field -- while the RAW
  feature-value register always stays shared, since a raw counter value is
  task-independent regardless of who discretizes it.

  Also raises AssertionError (F2; R10, an invariant) when any DISTINCT raw feature name across
  every task's intervals has no
  FEATURE_REGISTER_CATALOG entry -- checked right after this function calls
  generate_P4_registers_and_apply (see that call site's own inline comment
  for the full rationale). Without this, such a feature would still get a
  declared, PHV-pinned, range-table-keyed `<feature>_val` metadata field
  further down in this function, just never written by anything -- reading
  0 for every packet, forever, silently. The message names every missing
  feature (by the caller-supplied spelling) and points at
  FEATURE_REGISTER_CATALOG (p4gen/feature_registers.py) as the fix."""

  if config is not None:
    match_type = config.match_type
    use_default_action_discount = config.use_default_action_discount

  if encoding not in ENCODINGS:
    raise ValueError("encoding must be one of {}, got {!r}".format(
        ", ".join(repr(name) for name in ENCODINGS), encoding))
  task_codes = encoding == PER_TASK_ENCODING
  tree_codes = encoding == PER_TREE_ENCODING

  order = task_names(tasks)
  priorities = placement_priorities(order)
  # Program order: lowest priority first -- for (ddos, app), app's text
  # first, exactly the two-task program.
  program_tasks = list(reversed(list(tasks)))
  n_trees = {task.name: len(task.trees) for task in tasks}

  # generate the definition of the bit containers that contain decisions of
  # each tree -- only meaningful (and only computed) when the task actually
  # has trees, so a tree-less task's num_classes is never consulted and never
  # raises on log2(0)/log2(1 task with 1 class), etc.
  bit_per_classes = {task.name: (math.ceil(math.log2(task.num_classes))
                                 if n_trees[task.name] > 0 else 0)
                     for task in tasks}

  # Task 3: resolve every task's independently-derived feature_intervals
  # into ONE plan describing, per resolved name, which tasks share ONE
  # discretization table/field for a feature and which need their own,
  # namespaced ones. Computed once, consumed by every section below. Under
  # the per-task encoding the plan describes the code FIELDS (one reader task
  # each), under the per-tree encoding the packed fields (one per feature), and
  # table_plan the one range table per feature that writes them.
  task_intervals = [(task.name, task.intervals) for task in tasks]
  if tree_codes:
    table_plan, resolved_plan, tree_intervals = resolve_per_tree_plan(tasks)
    # The packed layout and its refusals (a tree with no split, a field wider
    # than the action data bus, no PHV container left), before anything is
    # written.
    layout = per_tree_layout(table_plan, tree_intervals, tasks)
  elif task_codes:
    table_plan, resolved_plan = resolve_per_task_plan(task_intervals)
  else:
    table_plan, resolved_plan = None, resolve_feature_plan(task_intervals)
  # Every other encoding: each tree keys only its task's code fields of the
  # features it splits on (decision 2.3), refusing a tree with none
  # (NoSplitTree, R6) before anything is written, tasks in program order as the
  # per-tree layout checks them. A task without its feature list cannot say
  # what its trees split on: its trees key every field of the task (and its
  # tables are sized by leaf count, below).
  tree_keys = {}
  if not tree_codes:
    for task in program_tasks:
      if task.trees and task.features is not None:
        field_of = dict((raw, name) for name, raw in plan_key_order(resolved_plan, task.name))
        tree_keys[task.name] = (field_of, tree_key_features(task, field_of.__contains__))

  metadata_code = ""
  # Accumulated by three separate loops below (per-tree result fields, raw
  # value fields, codeword fields), so it is opened here rather than beside
  # the first @pa_container_size that used to be its only writer.
  phv_pragmas = ""
  # PHV pins, and what each one forbids p4c's PHV allocator to do:
  #   * SPACE pin, @pa_solitary(f): no other field sits in f's container at
  #     the same time (no packing);
  #   * TIME pin, @pa_no_overlay(f): no other field reuses f's container
  #     before or after f's live range (no overlay);
  #   * SIZE pin, @pa_container_size(f, s...): f lives in containers of
  #     exactly these sizes, low container first.
  # The pin set was decided at gate G3 (2026-11-15) on the Task 17 / 17b
  # drop-one-pin compile experiment: 160 DEV designs, 40 per encoding, every
  # number below is p4c's committed allocation
  # (results/encoding_evidence_2026_11/model8/impl/G3_decisions.md,
  # experiment_report.md, remeasure.md). The checker program_checks.
  # check_program requires exactly this set.
  #
  # DROPPED at G3 (3.1): @pa_no_overlay(fwd). It once kept the 1-bit
  # direction flag out of an overlay with a bwd register's value field, which
  # held every bwd register behind the old `if (meta.fwd == 1)` gateway. Since
  # decisions 1a / 7.1 option B (2026-11-14) there is no gated block (every
  # register body reads meta.fwd), and dropping the pin was free on 160/160
  # p4c compiles (no stage depth, block or PHV container change).
  for task in program_tasks:
    name, bits = task.name, bit_per_classes[task.name]
    # A one-tree task has no class field (review A2): its tree action writes
    # the export field directly (generate_actions), so there is nothing to pin.
    if n_trees[name] > 1:
      for i in range(n_trees[name]):
        metadata_code += "\tbit<"+str(bits)+"> class_tree_"+name+"_"+str(i)+";\n"
        # SPACE pin, KEPT at G3 (3.2, load-bearing). Each tree's
        # classification table writes exactly this field, and a Tofino stage
        # has ONE action ALU per PHV CONTAINER -- not per field. These are
        # bit<1> (binary DDoS) or bit<2> (3-class App), so without the pin the
        # allocator packs several into one container, and every tree table
        # sharing a container is then forced into its own stage. Evidence
        # (p4c): dropping it costs about +1 stage on most of the 160 designs
        # and model exactness falls to 3-16 of 20 per cell (Task 17); earlier,
        # independent_low_sd8 put class_tree_ddos_1..5 in H0[15:11] and its
        # classification pool spanned 5 stages instead of 2 (12 -> 9 stages
        # with the pin; reviews/p4_tofino_reference.md Sec 7, Mechanism A).
        #
        # There is no classification_<task> field to pin: the vote action
        # exports the winner into ig_tm_md (program.VOTE_EXPORT).
        phv_pragmas += ('@pa_solitary("ingress", "ig_md.class_tree_'
                        + name + "_" + str(i) + '")\n')
        # DROPPED at G3 (3.3): @pa_no_overlay(class_tree_*). Free on 160/160
        # p4c compiles (no stage depth or block change after the 17b fix) and
        # 2.6-3.7 PHV containers cheaper on average. The overlay edge it
        # prevented only repeats an order that already exists: everything a
        # class field could overlay is read before the trees run.
  # Tier 3 + Task 3: every RESOLVED entry gets its own codeword field
  # (code_<resolved_name>), but the raw tracked-value field (<raw>_val) is
  # declared exactly once per DISTINCT raw feature name -- never twice just
  # because two resolved entries (e.g. app_<feature>/ddos_<feature>) both
  # discretize the same underlying shared raw value.
  # Each raw value field is also PINNED to a 16-bit PHV container. Measured
  # against the real Tofino compiler (reviews/p4_tofino_reference.md Sec 4.2):
  # the allocator is free to park a bit<16> range key in a 32-bit W container,
  # and when it does, that feature's range table costs TWO physical TCAM
  # blocks per entry ("1 in 2 (88)") instead of one ("1 in 1 (44)"). Pinning
  # all four value fields of one real M2 program took it from 14 TCAM blocks /
  # 10 stages to 12 / 9 -- 12 being exactly what evaluation.py predicts, so
  # this is what makes the cost model's range term correct rather than
  # accidentally correct. One pragma per DISTINCT raw field (never per
  # resolved name): two namespaced entries share one raw field, and a
  # duplicate pragma for the same field is a compile error.
  # SIZE pin, KEPT at G3 (3.4, load-bearing): it forbids a 32-bit (or 8-bit)
  # container for the range key. Evidence (p4c, Task 17): dropping it costs
  # +1.4..+11 TCAM blocks per encoding on average, a stage on most designs,
  # and 2 of the 160 designs no longer compile (a range key that does not fit
  # in under 5 PHV nibbles).
  #
  # The pragma names the field the way the PHV logs do -- `ig_md.<raw>_val`,
  # after SwitchIngressParser's `out metadata_t ig_md` parameter, NOT the
  # `meta` name SwitchIngress binds the same struct to.
  raw_feature_intervals = {}  # raw_feature_name -> intervals (first-seen; only the KEYS feed generate_P4_registers_and_apply, which ignores values)
  for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items():
    if raw_feature_name not in raw_feature_intervals:
      raw_feature_intervals[raw_feature_name] = intervals
      # normalise_feature_name here, NOT raw_feature_name verbatim: every
      # READER of this field lowercases it first -- _execute_lines' target
      # (below, via matched_features' name.lower()) and the range table's
      # <KEYS> substitution (generate_tables_and_apply, raw_name.lower())
      # both do. A mixed-case-but-catalogued raw name (e.g. "Flow_IAT_Max",
      # not dotted, so it clears the F2 "no catalog entry" check below,
      # which is deliberately case-insensitive) used to declare
      # `bit<16> Flow_IAT_Max_val;` here while every reader referenced
      # `flow_iat_max_val` -- an undefined identifier in the generated P4,
      # with no Python-side raise (final-review finding #2). Normalising
      # here makes the declaration match every reader.
      value_field = normalise_feature_name(raw_feature_name)+"_val"
      metadata_code += "\tbit<16> "+value_field+";\n"
      phv_pragmas += '@pa_container_size("ingress", "ig_md.'+value_field+'", 16)\n'

  # Code fields. Each codeword field is written by exactly one range table's
  # action. One pragma per RESOLVED name, matching the declaration exactly:
  # under disjoint namespacing app_/ddos_ entries have separate codeword
  # fields even when they share one raw value field.
  # DROPPED at G3 (3.5): @pa_solitary(code_*) (joint / disjoint / per-task).
  # It was added when code fields were declared at their live, ragged width
  # (an interval count), and p4c co-packed the leftovers of two codewords into
  # one container (joint_low_sd7: code_fwd_packet_length_max[135:128] and
  # code_bwd_packet_length_max[87:64] both in W8, two range tables a stage
  # apart, 10 -> 9 stages once pinned). Since decision 4.3 every code field is
  # DECLARED at its containers' full width (below), so its SIZE pin fills
  # every container it gets and no other field can share one: the space pin
  # is redundant. Free on 120/120 p4c compiles (no stage depth or block
  # change, PHV within 1 container).
  # The two pins each codeword field keeps, pinning p4c to what the resource
  # model assumes rather than modelling p4c's PHV choices (spec 2026-09-29):
  #   * TIME pin, KEPT (3.6): @pa_no_overlay on EVERY code_* field. p4c may overlay a range table's
  #     output onto a live field with a disjoint live range; each overlay adds
  #     a hidden "all readers of the old field first" placement edge
  #     (phv/add_initialization.cpp), and Tofino-1's backfill snapshot then
  #     stranded a range table at a stage boundary (C6 miss
  #     heldout_independent_M150_k14_s13: 11 predicted, 12 real). 125 range
  #     tables in 65 of 73 archived compiles were delayed this way; with the
  #     pragma, 0. Pinning OUR side blocks every partner -- pinning the
  #     partners (*_val, fwd) left residual overlays on flow_hash /
  #     now_pseudo_us. Not re-tested at G3 (decided KEEP beforehand).
  #   * SIZE pin, KEPT: @pa_container_size on every code_* field, to the layout
  #     lanes.layout gives it (the plain width rule, decision 4.1, unchanged
  #     at G3). No-overlay alone changes PHV
  #     pressure and p4c then put some 17-32-bit fields whole in W containers:
  #     same bytes, worse lanes, +1 block per tree. The model's 62-byte
  #     mixed-key refusal was retired on the strength of these pins
  #     (target.py) -- dropping them re-exposes M150_k7_s11. The ternary
  #     crossbar port (xbar.py, 2026-11-13) prices the pinned layouts too,
  #     so it relies on the same pins.
  # Evidence: results/compiler_calibration_pragmas_2026_09_29/ (73 designs,
  # feasible stage_depth 60/60, blocks 60/60; cost +4.7 PHV containers).
  # Per-tree codes: one packed field per feature (spec 2026-11-11-packed-codes-
  # stage-pin Sec 3.1), declared at its containers' bits and pinned to them,
  # low container first (SIZE pin: without it, stages were worse on 15 of 30
  # k<=8 probe designs, results/encoding_evidence_2026_11/model4, spec Sec
  # 3.4, and the crossbar simulation needs the containers to know each key
  # byte's lane), plus the TIME pin @pa_no_overlay, KEPT at G3 (3.7) for
  # uniformity with the other encodings' code fields (3.6): dropping it was
  # free on 40/40 p4c compiles (same stage depth and blocks, 3-4 PHV
  # containers fewer) although p4c then moved some tables between stages on
  # 13/40 (same totals); no @pa_solitary (spec Sec 3.1; the declared width
  # fills the containers, as for 3.5).
  if tree_codes:
    for feature, containers in layout.containers.items():
      field = "code_" + packed_field(feature)
      metadata_code += "\tbit<"+str(packed.field_bits(layout, feature))+"> "+field+";\n"
      phv_pragmas += '@pa_no_overlay("ingress", "ig_md.'+field+'")\n'
      phv_pragmas += ('@pa_container_size("ingress", "ig_md.' + field + '", '
                      + ", ".join(str(size) for size in containers) + ')\n')
  # Every declared code field with bits is pinned, also one no tree reads (a task
  # with no trees): the replay refuses an unpinned code field (program_checks.
  # check_program, decision 8.2), and training counts the pinned containers of
  # every field (evaluation._pool_inputs), so the two PHV counts agree.
  code_container_sizes = ({} if tree_codes else code_field_container_sizes(
      resolved_plan, [task.name for task in program_tasks]))
  # Every other encoding: each field's @pa_no_overlay and @pa_container_size
  # (no @pa_solitary since G3, 3.5 above).
  # Decision 4.3 (uniformity with the packed per-tree field): a pinned field is
  # DECLARED at its containers' full width, its live code in the low bits and 0
  # above (the range action writes it whole), and every tree keys the live bits
  # as slices from bit 0 (code_key_slices). The bus then carries exactly the
  # containers (S14) and p4c's action format has no padding bits in which to
  # condense another code (the per-tree ADB finding, inventory Q7). A field with
  # no bits has no pin.
  code_field_bits = {} if tree_codes else {
      name: (sum(code_container_sizes[name]) if name in code_container_sizes
             else len(intervals) - 1)
      for name, (_, intervals, _) in resolved_plan.items()}
  for resolved_name in (resolved_plan if not tree_codes else ()):
    metadata_code += ("\tbit<" + str(code_field_bits[resolved_name]) + "> code_"
                      + resolved_name + ";\n")
    phv_pragmas += '@pa_no_overlay("ingress", "ig_md.code_'+resolved_name+'")\n'
    if resolved_name in code_container_sizes:
      phv_pragmas += ('@pa_container_size("ingress", "ig_md.code_' + resolved_name + '", '
                      + ", ".join(str(size) for size in code_container_sizes[resolved_name])
                      + ')\n')

  # Task 3, point 3: registers must be resolved against the DEDUPLICATED set
  # of RAW feature names, never the (possibly namespaced) resolved names --
  # a raw value register must not be generated twice just because two
  # resolved entries share it.
  registers_code, register_actions_code, feature_update_apply_code, resolved = (
      generate_P4_registers_and_apply(raw_feature_intervals))
  # One bit<16> scratch field per dependency register used (iat_field_name),
  # declared first, where the template's single current_iat used to be.
  scratch_fields = iat_scratch_fields(raw_feature_intervals)
  metadata_code = "".join("\tbit<16> " + field + ";\n"
                          for field in scratch_fields) + metadata_code
  # SIZE pin, ADDED at G3 (2026-11-15) on every IAT scratch field: it forbids a
  # 32-bit (or 8-bit) container, so two scratch fields can no longer share one
  # W container. Without it p4c packed fwd_current_iat and bwd_current_iat
  # into one 32-bit container (beside $hash_field_argument0) in 35 of the 83
  # Task 17b base designs that have both the fwd and the bwd last-arrival
  # register; two actions cannot write one container in one stage, so p4c ran
  # the second register a stage later ("action dependency ... due to PHV
  # allocation"), and 5 of the 35 (all k<=8) lost a whole stage the model does
  # not price. Probe (p4c, 81 compiles, impl/remeasure.md): with this pin p4c
  # co-schedules the two registers on 83/83, the 5 become exact (10 -> 9
  # stages), no other stage or block count moves, +2.6 PHV containers. It is
  # what p4model.phv already assumes (one 16-bit container per scratch field).
  for field in scratch_fields:
    phv_pragmas += '@pa_container_size("ingress", "ig_md.' + field + '", 16)\n'
  # SPACE pin, ADDED with the symmetric 12-bit hash (review A1b, 2026-11-15) on
  # the timestamp, whenever the program computes one (calc_timestamp): no other
  # field may share now_pseudo_us's containers. Without it p4c packed
  # now_pseudo_us[15:14] into the 8-bit container of ig_tm_md.bypass_egress,
  # which the forwarding assignment writes in stage 0, so calc_timestamp waited
  # a stage ("action dependency ... due to PHV allocation advances stage to 1")
  # in exactly the 22 of 38 Task 28 compiles where that byte was shared. Under
  # the old 3-stage hash prologue that never mattered; with the hash in stage 0
  # the last-arrival registers then ran a stage after the model's schedule (6
  # range tables in 3 of 15 programs one stage late, p4c). With the pin p4c
  # puts the timestamp in stage 0 on 3/3 of those programs, every table where
  # the model says, stage depth and blocks unchanged, PHV +0..+1 containers.
  # (@pa_container_size(now_pseudo_us, 16) instead does not compile: the
  # shifted timestamp must split where its source bits split.)
  if uses_timestamp(raw_feature_intervals):
    phv_pragmas += '@pa_solitary("ingress", "ig_md.now_pseudo_us")\n'
  # The coarse clock of the durations (Task 38, P3h; calc_timestamp), declared only
  # when a duration is selected, with the same SPACE pin as now_pseudo_us above, BY
  # ANALOGY: the bypass_egress hazard was measured for the B + H + H split of
  # now_pseudo_us, not for this field (a whole W field). What was measured with the
  # pin: Task 37's P3h compiles (0 errors, the unedited stage depth and blocks) and its
  # live tofino-model check, and Task 38's 8 recompiled fixture designs with a duration
  # (0 errors, model stage depth exact); in each, now_coarse sits alone in one W
  # container. The unpinned variant was not compiled in the final clock form.
  # program_checks requires the pin.
  if uses_coarse_clock(raw_feature_intervals):
    metadata_code = "\tbit<32> now_coarse;\n" + metadata_code
    phv_pragmas += '@pa_solitary("ingress", "ig_md.now_coarse")\n'
  # SPACE pin, ADDED 2026-11-15 (Task 31 miss, user's free-pin policy) on the flow
  # index, in every program: no other field may share flow_hash's container. On
  # 1 of the 160 Task 31 compiles (joint-task_M015_s13_k14) p4c packed
  # flow_hash[11:0] into H15 beside ig_tm_md.bypass_egress, which the forwarding
  # assignment writes in stage 0, so tbl_calc_flow_hash waited for stage 1
  # ("action dependency ... due to PHV allocation advances stage to 1") and every
  # later table ran a stage after program.FLOW_HASH_LEVEL = 1 says (model 7, p4c
  # 8: an UNDER). Same mechanism as the now_pseudo_us pin above. Probe (p4c, 3
  # compiles, text edit only; Task 31 report): with the pin that design is 7
  # stages (= model), blocks unchanged, PHV +1; two control designs unchanged
  # (PHV +1). program_checks.check_program requires it.
  phv_pragmas += '@pa_solitary("ingress", "ig_md.flow_hash")\n'

  # F2: a feature with no FEATURE_REGISTER_CATALOG entry is silently skipped
  # by generate_P4_registers_and_apply above, but everything else in this
  # function (metadata_code's bit<16> <f>_val declaration, its
  # @pa_container_size pragma, and its range table below) is built from
  # raw_feature_intervals regardless -- so an uncatalogued feature would
  # otherwise compile to a field that is declared, keyed on, and NEVER
  # WRITTEN, silently reading 0 for every packet forever. Fail loudly at
  # generation time instead of letting that ship as a silent
  # misclassification.
  #
  # Case-insensitive comparison, deliberately: `resolved` (returned by
  # generate_P4_registers_and_apply, above) holds catalog-matched names in
  # LOWERCASE (it builds `matched_features` via `name.lower()` before
  # checking catalog membership -- see that function's docstring), while
  # `raw_feature_intervals`'s keys preserve whatever case the caller
  # actually supplied. Every real production caller's names are already
  # canonical (lowercase, underscore-joined) by the time they reach here,
  # via get_nodes()'s normalise_feature_name() (Task 4) -- so this rarely
  # matters in practice -- but comparing raw (mixed-case-tolerant) against
  # resolved (always-lowercase) with a bare set difference would falsely
  # flag a feature that actually DID resolve, just spelled with different
  # case, as "missing" and raise a false-positive AssertionError for it.
  # Lowercasing raw_feature_intervals' keys here too keeps this comparison
  # exact regardless of that upstream-normalisation invariant, while still
  # reporting each missing feature by its ORIGINAL (caller-supplied)
  # spelling in the error message below.
  missing = sorted(
      raw_name for raw_name in raw_feature_intervals
      if raw_name.lower() not in resolved
  )
  # R10 (decision 8): an invariant, not a refusal -- the catalog covers every
  # feature the pipeline selects -- so an AssertionError (raised, not `assert`,
  # so python -O cannot drop it).
  if missing:
    raise AssertionError(
        "no register catalog entry for {} -- the generated program would declare "
        "<feature>_val, key a range table on it, and never write it, so it would "
        "read 0 for every packet. Add the feature to FEATURE_REGISTER_CATALOG "
        "(p4gen/feature_registers.py) or drop it from the feature set.".format(missing))

  # generate_actions only needs resolved_name -> intervals: it writes
  # meta.code_<resolved_name>, one action per resolved entry, and is
  # already agnostic to whether a name is "raw" or "resolved"/namespaced --
  # no change needed to that function itself, only to what it's fed here.
  #
  # Under the per-task and per-tree encodings the entries of these sections are
  # the range tables, one per raw feature, not the code fields: the table holds
  # the union of the tasks' intervals. Its per-task action writes every task's
  # field of the feature; its per-tree action writes the one packed field whole.
  per_task_writes, packed_bits = None, None
  if task_codes or tree_codes:
    flat_resolved_intervals = {feature: union for feature, (union, _) in table_plan.items()}
  else:
    flat_resolved_intervals = {
        resolved_name: intervals
        for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items()
    }
  if task_codes:
    per_task_writes = {feature: written for feature, (_, written) in table_plan.items()}
  if tree_codes:
    packed_bits = {feature: packed.field_bits(layout, feature) for feature in table_plan}
  action_templates = generate_actions(
      flat_resolved_intervals,
      [(task.name, n_trees[task.name], bit_per_classes[task.name]) for task in program_tasks],
      per_task_writes=per_task_writes, packed_bits=packed_bits,
      field_bits=code_field_bits or None)

  # Task 3, point 4: each task's classification tables must key on
  # code_<resolved_name> only for resolved entries that task actually
  # reads. Point 2/5: the range-matching tables (one per resolved entry,
  # always) still need each entry's raw feature name for the RAW value
  # field they key on.
  # Per tree: each tree keys only its own codes, in feature order, each as
  # slices of the feature's packed field (packed.key_slices).
  if tree_codes:
    key_names = {task.name: {i: [(packed_field(feature),
                                  packed.key_slices(layout, task.name, i, feature))
                                 for feature in table_plan if feature in own]
                             for i, own in enumerate(tree_intervals[task.name])}
                 for task in tasks}
  else:
    # Per tree too (decision 2.3): tree i keys, in the plan's field order, the
    # fields of the features it splits on (tree_key_features), each as the
    # slices of its live bits (decision 4.3, code_key_slices).
    def sliced(name):
      return (name, code_key_slices(len(resolved_plan[name][1]) - 1))
    key_names = {}
    for task in tasks:
      order = plan_key_order(resolved_plan, task.name)
      if task.name in tree_keys:
        _, keyed = tree_keys[task.name]
        key_names[task.name] = {i: [sliced(name) for name, raw in order if raw in raws]
                                for i, raws in enumerate(keyed)}
      else:
        key_names[task.name] = [sliced(name) for name, _ in order]
  raw_feature_names = ({feature: feature for feature in table_plan}
                       if task_codes or tree_codes
                       else {name: raw for name, (raw, _, _) in resolved_plan.items()})

  # Follow-up: the discount's live path. Codewords are recomputed per task
  # against that task's OWN intervals (see this function's docstring), keyed
  # by the task's own tree id.
  #
  # Table-sizing follow-up: codewords are ALSO what makes an exact
  # classification-table size derivable, so they are now computed whenever
  # the caller supplied that task's ordered training-feature-name list --
  # not only when the discount is on. The discount's own requirement (it
  # cannot function without codewords) is unchanged: still a hard ValueError.
  codewords = {}
  for task in program_tasks:
    if n_trees[task.name] == 0:
      continue
    if tree_codes:
      # each tree against its OWN intervals (per_tree_intervals requires the
      # feature list, so it is present here)
      codewords[task.name] = {
          i: generate_codewords(get_root_to_leaf_paths(tree_nodes_for(_Forest([tree]), task.features)),
                                tree_intervals[task.name][i])[0]
          for i, tree in enumerate(task.trees)}
    elif task.features is not None:
      # each tree's codewords over the features its key reads, as its entries
      # (generate_table_entries) hold them; the count is the full codewords'
      codewords[task.name] = tree_codewords(task, tree_keys[task.name][1])
    elif use_default_action_discount:
      raise ValueError(
          "selected_features_{0} (the features of task {0!r}) is required when "
          "use_default_action_discount=True and task {0!r} has trees -- the "
          "generator cannot recompute codewords without the exact ordered "
          "training-feature-name list".format(task.name))

  # Table-sizing follow-up: every generated table is sized from its REAL
  # entry count. Range-matching feature tables get exactly one entry per
  # interval; classification tables get one per distinct codeword (minus the
  # ones the discount turns into the table's default action), or -- when
  # codewords are not computable for that task -- the fitted tree's own leaf
  # count, a real structural number that can only ever be >= the codeword
  # count (two leaves whose paths round to the same codeword string collapse
  # into one dict entry in generate_codewords, never two), so it never
  # underestimates.
  feature_table_sizes = {
      resolved_name: max(1, len(intervals))
      for resolved_name, intervals in flat_resolved_intervals.items()
  }

  def _classification_table_size(task, tree_id):
    tree_codewords = codewords.get(task.name, {}).get(tree_id)
    if tree_codewords is None:
      return max(1, task.trees[tree_id].get_n_leaves())
    if use_default_action_discount:
      _, dropped_codewords = most_common_class_and_dropped_codewords(tree_codewords)
      return max(1, len(tree_codewords) - len(dropped_codewords))
    return max(1, len(tree_codewords))

  task_tables = [(task.name, key_names[task.name],
                  [_classification_table_size(task, i) for i in range(n_trees[task.name])])
                 for task in program_tasks]

  table_templates, apply_templates = generate_tables_and_apply(
      flat_resolved_intervals.keys(), task_tables, match_type=match_type,
      raw_feature_names=raw_feature_names,
      feature_table_sizes=feature_table_sizes, priorities=priorities)

  # generate code to vote between the trees -- one vote table per task with
  # MORE THAN ONE tree. generate_voting_code returns (table_decl,
  # apply_call); the table declaration joins the other TABLES text, the apply
  # call joins the other APPLY text.
  # A one-tree task has no vote table (review A2, adopted 2026-11-15): with N = 1
  # the vote is the identity, and its tree action writes the export field itself
  # (generate_actions), the same value in the same field for every packet. p4c
  # (Task 28, impl/probe_generator_A1_A2.md): one stage less on 10/10 programs
  # where both tasks have one tree, no change on the 2 where the other task still
  # votes, blocks unchanged, no remaining table moved stage, qid and ingress_cos
  # still in separate containers. The model prices no vote stage for such a task
  # (usage.pipeline_depth).
  for task in program_tasks:
    if n_trees[task.name] > 1:
      vote_table, vote_apply = generate_voting_code(n_trees[task.name],
                                                    task.num_classes, task.name)
      table_templates += vote_table
      apply_templates += vote_apply

  # substitute the code in the template
  with open(PATH_P4_CODE_TEMPLATE_INPUT, 'r') as switch_template_file:
    switch_template = switch_template_file.read()
    # The marker's OWN line is consumed, so a feature-less program keeps the
    # template's original spacing exactly (and a program with features gets
    # its pragmas immediately above `struct metadata_t`, no stray blank line).
    switch_template = switch_template.replace('<MAX_NUM_FLOWS>', str(MAX_NUM_FLOWS))
    switch_template = switch_template.replace('<FLOW_INDEX_BITS>', str(FLOW_INDEX_BITS))
    switch_template = switch_template.replace('/* PHV_PRAGMAS */\n', phv_pragmas)
    switch_template = switch_template.replace('/* METADATA */', metadata_code)
    switch_template = switch_template.replace('/* REGISTERS */', registers_code)
    switch_template = switch_template.replace('/* REGISTER_ACTIONS */', register_actions_code)
    switch_template = switch_template.replace('/* ACTIONS */', action_templates)
    switch_template = switch_template.replace('/* TABLES */', table_templates)
    switch_template = switch_template.replace('/* FEATURE_UPDATE_APPLY */', feature_update_apply_code)
    switch_template = switch_template.replace('/* APPLY */', apply_templates)
    # The {first, dur} pair of the Fwd / Bwd duration registers (Task 38), declared
    # after metadata_t and before the parser, where Task 37 compiled it (not between
    # the PHV pragmas and metadata_t, whose annotations they are).
    if uses_duration_pair(raw_feature_intervals):
      parser = '\nparser SwitchIngressParser('
      if switch_template.count(parser) != 1:
        raise RuntimeError("the template must declare SwitchIngressParser exactly once")
      switch_template = switch_template.replace(parser, '\n' + DURATION_PAIR_STRUCT + parser)

  ensure_directory_exists(output_dir)
  written_path = output_dir + output_filename
  with open(written_path, 'w') as switch_template_file:
    switch_template_file.write(switch_template)
  if entries_filename is not None:
    entries = generate_table_entries(
        tasks, use_default_action_discount=use_default_action_discount, encoding=encoding)
    with open(output_dir + entries_filename, 'w') as entries_file:
      entries_file.write(json.dumps(entries))

  return written_path


# ---------------------------------------------------------------------------
# TNA (Tofino Native Architecture) register + apply-block generation.
#
# Like the rest of this file (the whole program is TNA, resources/
# p4_template.p4), this part targets the architecture first validated in
# p4/tofino_spike/tna_m1_flows_iat_spike.p4 (real Tofino compiler, `p4c -b
# tofino -a tna`, 0 errors). generate_p4_program calls it to resolve the
# selected features to the TNA registers, RegisterActions and per-packet
# update logic of the flow- and IAT-tracking design.
# ---------------------------------------------------------------------------

# Symbolic RegisterAction body kinds -> exact atomic read-modify-write P4
# body, each transcribed verbatim from the compile that validated it: the
# spikes (p4/tofino_spike/tna_m1_flows_iat_spike.p4 and the others named per
# kind) for the packet-length kinds and running_max_iat; Task 33's C1f compile
# and Task 33b's live tofino-model check (impl/probe_first_packet/
# live_prog_orient/prog.p4) for iat_delta and running_min_iat (Task 34); Task
# 37's P3h compile and live check (impl/probe_p3c.md) for the durations and length
# Totals (Task 38). Do not reword/"clean up" these -- they are the compiler-validated
# ground truth (p4c silently miscompiles some equivalent-looking forms:
# program_checks._check_stateful_alu).
_REGISTER_ACTION_BODIES = {
  # A chain's last-arrival register (V1o, Task 34; user decision 2026-11-15).
  #   WHAT: a 32-bit register holding the 32-bit clock now_pseudo_us (calc_timestamp:
  #     global_tstamp[41:10], 1.024 us ticks); the gap is clamped at 65535 INSIDE the
  #     RegisterAction, which returns bit<32> rv; the call site casts it to the 16-bit
  #     scratch field (_execute_lines).
  #   WHY: with a 16-bit clock and register the gap wrapped mod 65536 ticks (67.1 ms), so
  #     any gap over 67 ms read as a random small IAT. Emulated on 200 DEV designs
  #     (val_select, Task 29): agreement with software app 92.8% / 81.3% (k<=8 / k>=9)
  #     today; the clamp and the unit rounding together 99.98% / 99.97%.
  #   EVIDENCE: Task 29 (impl/probe_timestamp.md): V1o compiled 3/3 at no stage or block
  #     cost (prog.bfa `grt.u lo, phv_lo, -lo, -65535` / `alu_a cmplo, hi, 65535` / `sub
  #     !cmplo, hi, phv_lo, lo`), PHV +1..+3; the brief's V1 (rv bit<16>, cast inside the
  #     RegisterAction) is a p4c internal error 3/3. The live tofino-model check (Task 33b,
  #     impl/probe_first_packet.md) read a 100 ms gap as 65535, not wrapped.
  # First packet (C1f, Task 34; user decision 2026-11-15): "an IAT of 0 means no IAT".
  #   WHAT: the register starts at 0xFFFFFFFF (_REGISTER_INITIAL_VALUES) and stores
  #     now - 1; the gap is the SATURATING `now |-| value` inside the clamp. A chain's
  #     first packet (of the flow, or of its direction) then gives 0, and a real gap of g
  #     ticks gives min(g + 1, 65535); the control plane moves every IAT threshold up one
  #     tick to match (iat_ticks).
  #   WHY: with V1o alone the first packet measured its gap from time 0, so it read 65535
  #     and every *IAT.Max feature became 65535 (emulated app k>=9 0.6954 vs software
  #     0.7263). C1f equals CICFlowMeter's Max/Min/Mean semantics including "no IAT -> 0"
  #     (the Means, an EWMA then, left in Task 38): emulated on 200 DEV designs = the ideal switch 200/200 (app
  #     0.7327 / 0.7263, ddos 0.9602 / 0.9771 at k<=8 / k>=9).
  #   EVIDENCE: Task 33 (impl/probe_first_packet.md): exactly this body compiled on
  #     joint-off_M025_s06_k17 (all three chains) at 9 / 22 = V1o, every register in
  #     its stage; prog.bfa `grt.u lo, phv_lo, -lo, -65535` / `alu_a cmplo, hi, 65535` /
  #     `ssubu !cmplo, hi, phv_lo, lo` / `add lo, phv_lo, 4294967295` / `output alu_hi`.
  #     The compare is exact (34-bit), not modulo 2^32 as the P4 text reads, which is why
  #     the first packet gives 0: confirmed live on tofino-model (Task 33b: every chain's
  #     first packet 0, later packets min(gap + 1, 65535), every register but the
  #     then Means exact against an emulation of prog.bfa). A first-packet test by compare (C2, sentinel
  #     65535 + `value == 0`) needs 3 compare units on a direction-owned chain: refused.
  "iat_delta": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.now_pseudo_us - value > 65535) {{\n"
      "\t\t\t\trv = 65535;\n"
      "\t\t\t}} else {{\n"
      "\t\t\t\trv = meta.now_pseudo_us |-| value;\n"
      "\t\t\t}}\n"
      "\t\t\tvalue = meta.now_pseudo_us - 1;\n"
      "\t\t}}\n"
  ),
  "running_max_iat": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.{iat} > value) {{\n"
      "\t\t\t\tvalue = meta.{iat};\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  "running_max_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (hdr.ipv4.total_len > value) {{\n"
      "\t\t\t\tvalue = hdr.ipv4.total_len;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # The flow duration (Task 38, P3h; user decision 2026-11-15), replacing the IAT
  # Mean (whose MathUnit body p4c miscompiled into a running sum of halves, Task
  # 33b; no Mean is computable on a Tofino-1 stateful ALU, probe_mean.md).
  #   WHAT: a 32-bit register holding the flow's FIRST arrival on the coarse clock
  #     meta.now_coarse (65.536 us ticks, calc_timestamp), initial 0xFFFFFFFF
  #     (_REGISTER_INITIAL_VALUES). Every packet returns now - first, clamped at
  #     65535 (4.3 s), as bit<32> (the call site casts it to the 16-bit _val,
  #     _CALL_SITE_PREFIX, as iat_delta), and keeps first = min(first, now). On the
  #     flow's first packet `now - 0xFFFFFFFF > 65535` is false in the ALU's exact
  #     (34-bit) compare and `now |-| 0xFFFFFFFF` is 0, so it reads 0 and stores now:
  #     the duration of a one-packet flow is 0, CICFlowMeter's. No "+1" (C1f's): 0 is
  #     a real duration here, never a "no IAT" marker. No dependency register: the
  #     flow's last-arrival register is emitted only for the IAT Max / Min.
  #   WHY: training reads the CSV's Flow.Duration in the same ticks (dataset.py:
  #     round(d / 65.536 us), at most one tick from the switch's floor difference), so
  #     the control plane installs every threshold unchanged. The coarse clock
  #     reaches 4.3 s before the clamp, where the 1.024 us IAT clock clamps at 67 ms
  #     (mean_options_accuracy.md, Task 38 re-measure with this definition, software
  #     accuracy on 5 DEV splits: app +1.4 / +0.2 points, ddos +0.1 / +0.4 over the
  #     software Means at k<=8 / k>=9).
  #   EVIDENCE: Task 37 (results/encoding_evidence_2026_11/model8/impl/probe_p3c.md,
  #     variant P3h): p4c compiled exactly this body at the unedited program's stage
  #     depth and blocks (independent_M050_s13_k07 7 / 36, joint-off_M025_s06_k17
  #     8 / 22), prog.bfa `grt.u lo, phv_lo, -lo, -65535` / `alu_a cmplo, hi, 65535` /
  #     `ssubu !cmplo, hi, phv_lo, lo` / `minu lo, lo, phv_lo` / `output (cmplo |
  #     !cmplo), alu_hi` (1 compare, no MathUnit, no silent rewrite; Task 38's
  #     recompiles gave the same ops); live on tofino-model an exact
  #     emulation of the compiled program matched 64/64 packets and the clamp of a
  #     4.5 s gap. The clock operand must be a plain field: with a sliced / cast clock
  #     the two `rv =` branches are refused ("Incompatible outputs in RegisterAction:
  #     alu_hi and alu_hi", Task 37 P3f / P3g).
  "coarse_duration": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.now_coarse - value > 65535) {{\n"
      "\t\t\t\trv = 65535;\n"
      "\t\t\t}} else {{\n"
      "\t\t\t\trv = meta.now_coarse |-| value;\n"
      "\t\t\t}}\n"
      "\t\t\tvalue = min(value, meta.now_coarse);\n"
      "\t\t}}\n"
  ),
  # The flow's length Total (Task 38, replacing packet_length_mean): a saturating
  # 16-bit sum of the IP total lengths, `value |+| total_len` (65535 at most).
  #   WHY: the length Mean's MathUnit body was miscompiled like the IAT Mean's (Task
  #     33b); training reads the CSV's total lengths clipped at 65535 (dataset.py), the
  #     switch's saturation (mean_options_accuracy.md: the length Totals recover about
  #     two thirds of what dropping the Means costs).
  #   EVIDENCE: Task 36 (body T16) and Task 37 (P3h): prog.bfa `saddu lo, lo, phv_lo` /
  #     `output alu_lo`, no MathUnit, at no stage or block cost; live on tofino-model
  #     the register read the sum of the IP total lengths, saturated at 65535 (Task 37,
  #     sequence D: 72,000 bytes sent).
  "saturating_total_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tvalue = value |+| hdr.ipv4.total_len;\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # Task 7: running minimum. A register that initialises to 0 (the
  # _register_declaration default) is stuck at 0 forever once folded through
  # min() -- min(0, anything) == 0 -- so these two body kinds are paired with
  # _REGISTER_INITIAL_VALUES entries below that start the backing register at
  # INFINITE (65535) instead, via TNA's Register<T,I> two-argument
  # constructor. Design (a) from task-7-brief.md, validated by compiling
  # p4/tofino_spike/tna_m3_min_registers_spike.p4 with the real p4c
  # (0 errors) -- design (b) (a first-packet `value == 0` branch) was not
  # needed. running_min_packet_length is transcribed verbatim from that compile;
  # running_min_iat's body is C1f's since Task 34 (below).
  #
  # running_min_iat, C1f (Task 34): skips an IAT of 0 (no IAT, see "iat_delta") and
  # stores min - 1, so the register's 65535 start means "no IAT yet"; the call site
  # reads it `+ 1` (_CALL_SITE_SUFFIX), which wraps 65535 to 0 in bit<16>:
  # CICFlowMeter's Min for a chain without an IAT is 0. Without the read-out fix (C1s)
  # the emulated ddos accuracy dropped up to 11 points on one design. The compare is
  # `iat <= value`, never `iat - 1 < value`: p4c compiled the latter (Task 33's C1)
  # to `lss.u hi, phv_lo, 65535, -lo`, folding -1 into +65535 in an exact 34-bit
  # compare that is never true (program_checks refuses that form, salu_negative_
  # constant). The +1 cannot be inside the RegisterAction (C1r: "Incompatible outputs
  # in RegisterAction ... phv and alu_hi", refused 3/3). Evidence: Task 33's C1f
  # compile (`neq lo, phv_lo` / `leq.u hi, phv_lo, -lo` / `add (cmplo & cmphi), lo,
  # phv_lo, 65535`, 2 compare units; 9 / 22 = V1o) and the live check (Task 33b: flow
  # Min register = min gap, exact).
  "running_min_iat": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.{iat} != 0 && meta.{iat} <= value) {{\n"
      "\t\t\t\tvalue = meta.{iat} - 1;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  "running_min_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (hdr.ipv4.total_len < value) {{\n"
      "\t\t\t\tvalue = hdr.ipv4.total_len;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
}

# Decisions 1a / 7.1 option B (2026-11-14): the body of a direction-owned
# register (catalog gated_by "fwd" or "bwd"), one template per body kind,
# formatted with owner = 1 (fwd-owned) or 0 (bwd-owned). Every register's
# .execute() runs on every packet (there are no `if (meta.fwd == ...)` blocks
# in the apply block). In the owner's direction the body updates exactly as
# the _REGISTER_ACTION_BODIES literal above; in the other it leaves `value`
# unchanged and only reads it (`rv = value`), so every <feature>_val field is
# written on every packet. A dependency (iat_delta) register returns 0 in the
# read direction: its only reader there is its chain's value register, which
# ignores meta.{iat} on that packet. Ungated registers keep the literal
# bodies above. Probe 7.1 (results/encoding_evidence_2026_11/model8/impl/
# probe_7_1.md): p4c accepts every template; each costs one more SALU
# instruction (the meta.fwd compare) and no stage.
_DIRECTION_READ_BODIES = {
  # V1o's clamp and C1f's first-packet rule (see _REGISTER_ACTION_BODIES["iat_delta"])
  # inside the owner test: two compare units, Tofino-1's limit (Task 29: compiled 3/3;
  # Task 33: C1f compiled, `equ lo, phv_hi(0..7)` + the clamp predicated on cmplo).
  # The read direction returns 0, which C1f's value registers read as "no IAT".
  "iat_delta": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner}) {{\n"
      "\t\t\t\tif (meta.now_pseudo_us - value > 65535) {{\n"
      "\t\t\t\t\trv = 65535;\n"
      "\t\t\t\t}} else {{\n"
      "\t\t\t\t\trv = meta.now_pseudo_us |-| value;\n"
      "\t\t\t\t}}\n"
      "\t\t\t\tvalue = meta.now_pseudo_us - 1;\n"
      "\t\t\t}} else {{\n"
      "\t\t\t\trv = 0;\n"
      "\t\t\t}}\n"
      "\t\t}}\n"
  ),
  "running_max_iat": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner} && meta.{iat} > value) {{\n"
      "\t\t\t\tvalue = meta.{iat};\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  "running_max_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner} && hdr.ipv4.total_len > value) {{\n"
      "\t\t\t\tvalue = hdr.ipv4.total_len;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # The Fwd / Bwd duration (Task 38, P3h; replacing fwd_iat_mean / bwd_iat_mean).
  #   WHAT: a 64-bit PAIR register {first, dur} (dur_pair_t, _REGISTER_VALUE_TYPES;
  #     initial {0xFFFFFFFF, 0}). In the owner's direction it updates both halves as
  #     the flow's body does (dur = clamp(now_coarse - first) at 65535, first =
  #     min(first, now_coarse)); in the other nothing changes. EVERY packet returns
  #     dur (as bit<32>, cast at the call site), so the feature holds the owner's
  #     duration on the other direction's packets too, and a direction without a
  #     packet reads 0 (CICFlowMeter's value for fewer than two packets). No "+1",
  #     no dependency register (see "coarse_duration").
  #   WHY a pair: a register holding only the first arrival (the brief's 32-bit form)
  #     can give the duration only on the owner's packets; on the other direction's
  #     it has no last owner arrival to subtract (Task 37, P3c_brief).
  #   EVIDENCE: Task 37 (probe_p3c.md, P3h): p4c compiled exactly this body at no
  #     stage or block cost (+1 SRAM block per pair register), prog.bfa `equ lo,
  #     phv_hi(0..7)[, -1]` / `grt.u hi, phv_lo, -lo, -65535` / `alu_a (cmphi &
  #     cmplo), hi, 65535` / `ssubu (!cmphi & cmplo), hi, phv_lo, lo` / `minu cmplo,
  #     lo, lo, phv_lo` / `output alu_hi`: 2 compares (Tofino-1's limit), 3 ALU ops, 1
  #     unpredicated output (in the read direction no ALU fires and alu_hi passes the
  #     stored dur through). Live on tofino-model: exact emulation 64/64 packets, 5/5
  #     snapshots pair member by member, the bwd chain starting mid-flow read 0, a 4.5
  #     s gap clamped and the fwd value kept 65535 on bwd packets.
  "coarse_duration_pair": (
      "\t\tvoid apply(inout dur_pair_t value, out bit<32> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner}) {{\n"
      "\t\t\t\tif (meta.now_coarse - value.first > 65535) {{\n"
      "\t\t\t\t\tvalue.dur = 65535;\n"
      "\t\t\t\t}} else {{\n"
      "\t\t\t\t\tvalue.dur = meta.now_coarse |-| value.first;\n"
      "\t\t\t\t}}\n"
      "\t\t\t\tvalue.first = min(value.first, meta.now_coarse);\n"
      "\t\t\t}}\n"
      "\t\t\trv = value.dur;\n"
      "\t\t}}\n"
  ),
  # The Fwd / Bwd length Total (Task 38): "saturating_total_packet_length" in the
  # owner's direction, read only in the other. Evidence as that body (Task 36 T16 /
  # Task 37 P3h: `saddu cmplo, lo, lo, phv_lo` after the meta.fwd compare).
  "saturating_total_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner}) {{\n"
      "\t\t\t\tvalue = value |+| hdr.ipv4.total_len;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # C1f (Task 34): a direction-owned IAT Min needs no meta.fwd test. Its chain's
  # dependency register returns 0 in the other direction (iat_delta above), and the
  # literal body skips an IAT of 0 already, so the literal body is the direction-read
  # body. Same compare-unit count as before (2); evidence: Task 33's C1f compile and
  # live check (Task 33b: each direction's chain unmoved by the other direction's
  # packets).
  "running_min_iat": _REGISTER_ACTION_BODIES["running_min_iat"],
  # A fwd-owned length Min: its direction always has a packet (a flow's first packet
  # is fwd since the orientation fix, Task 33b), so the plain "start at 65535" body is
  # exact on every packet that reads it.
  "running_min_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner} && hdr.ipv4.total_len < value) {{\n"
      "\t\t\t\tvalue = hdr.ipv4.total_len;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # A bwd-owned length Min (final review I1, 2026-11-16; _GENERATED_BODY): C1f's trick
  # (see _REGISTER_ACTION_BODIES["running_min_iat"]). Option B reads the register on
  # every packet, so before the flow's first bwd packet the plain body read its 65535
  # start, where training reads the CSV's 0 (Bwd Packet Length Min = 0 on 16.1% of the
  # app rows, 19.4% of the ddos rows). This body stores min - 1 (65535 = "no bwd packet
  # yet") and the call site reads it `+ 1` (_CALL_SITE_SUFFIX), 0 before any bwd packet.
  # `total_len <= value` with value = m - 1 is `total_len < m`, the plain body's
  # update; total_len - 1 does not wrap for an IPv4 total length (>= 20). The compare
  # takes no constant and reads value before the write (program_checks
  # _check_stateful_alu); the - 1 is in the ALU, as in running_min_iat.
  "running_min_packet_length_m1": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.fwd == {owner} && hdr.ipv4.total_len <= value) {{\n"
      "\t\t\t\tvalue = hdr.ipv4.total_len - 1;\n"
      "\t\t\t}}\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
}

# The body kind the generator writes for a catalog register, when it is not the
# catalog's own body: by (catalog body, gated_by). Final review I1: only the
# bwd-owned length Min can be read before its direction has a packet (a flow-wide or
# fwd-owned Min always has one -- the flow's first packet is fwd), so only it takes
# C1f's min - 1 body and its extra PHV container (call_site_constants).
_GENERATED_BODY = {("running_min_packet_length", "bwd"): "running_min_packet_length_m1"}


def _generated_body(body, gated_by):
  return _GENERATED_BODY.get((body, gated_by), body)

# gated_by -> the meta.fwd value of the owner direction.
_DIRECTION_OWNER = {"fwd": 1, "bwd": 0}

# What a body kind's `.execute()` call site wraps around the call
# (_execute_lines): `<target> = <prefix><name>_action.execute(meta.flow_hash)<suffix>;`.
#   iat_delta, prefix `(bit<16>) `: V1o (Task 34). The RegisterAction returns the
#     clamped gap as bit<32> (its register's width) and the 16-bit scratch field takes
#     its low half, which the clamp keeps <= 65535. The cast must sit HERE: inside the
#     RegisterAction (rv bit<16>) p4c fails with an internal compiler error 3/3 ("input
#     now_pseudo_us[15:0] ... cannot be found on the hash input"; Task 29, V1).
#   running_min_iat, suffix ` + 1`: C1f (Task 34, see _REGISTER_ACTION_BODIES
#     ["running_min_iat"]): the register holds min - 1 (65535 = no IAT yet), so the
#     feature reads value + 1, 0 for a chain without an IAT. p4c keeps the constant 1 in
#     a PHV container set by a hidden stage-0 `$precompute` table (Task 33, prog.bfa
#     `add H3, <min salu>, H4`): one 16-bit container per Min feature, counted by
#     p4model.phv (MIN_READ_CONSTANT).
#   coarse_duration / coarse_duration_pair, prefix `(bit<16>) `: Task 38, the same
#     reason as iat_delta (the duration is clamped at 65535 inside the RegisterAction,
#     which returns bit<32>; Task 37 compiled exactly this call site).
_CALL_SITE_PREFIX = {"iat_delta": "(bit<16>) ", "coarse_duration": "(bit<16>) ",
                     "coarse_duration_pair": "(bit<16>) "}
_CALL_SITE_SUFFIX = {"running_min_iat": " + 1", "running_min_packet_length_m1": " + 1"}

# The value type and the RegisterAction's return type of a body kind whose
# register is not a plain bit<width> (every other kind: bit<width> for both).
# coarse_duration_pair (Task 38): the {first, dur} pair, returning dur.
_REGISTER_VALUE_TYPES = {"coarse_duration_pair": ("dur_pair_t", "bit<32>")}
# The pair's declaration, emitted once before the parser when some register
# uses it (generate_p4_program; Task 37 compiled it there).
DURATION_PAIR_STRUCT = ("struct dur_pair_t {\n"
                        "    bit<32> first;\n"
                        "    bit<32> dur;\n"
                        "}\n")

# Body kinds whose backing register must NOT start at 0. A running minimum
# folded into a zero-initialised register stays 0 forever (min(0, x) == 0),
# so these start at INFINITE (65535 -- the same sentinel
# get_feature_intervals_from_thresholds uses) via _register_declaration's
# optional initial_value, using TNA's Register<T,I> two-argument constructor.
# A last-arrival register (iat_delta) starts at all ones of its width
# (32w0xFFFFFFFF for the 32-bit clock): C1f, see _REGISTER_ACTION_BODIES
# ["iat_delta"] -- the saturating gap of a chain's first packet is then 0. Since C1f
# a running_min_iat register's 65535 means "no IAT yet" (it stores min - 1).
# Task 38: a duration register holds the first arrival, so it starts at all ones
# too (a first packet then reads 0 and stores now, _REGISTER_ACTION_BODIES
# ["coarse_duration"]); the pair's dur half starts at 0, the value a direction with
# no packet reads.
_REGISTER_INITIAL_VALUES = {
    "running_min_iat": INFINITE,
    "running_min_packet_length": INFINITE,
    "running_min_packet_length_m1": INFINITE,
    "iat_delta": lambda width: "%dw0x%X" % (width, (1 << width) - 1),
    "coarse_duration": lambda width: "%dw0x%X" % (width, (1 << width) - 1),
    "coarse_duration_pair": "{32w0xFFFFFFFF, 32w0}",
}


def _register_initial_value(body_kind, width):
  """The Register's initial value for one body kind (None: TNA's default 0)."""
  value = _REGISTER_INITIAL_VALUES.get(body_kind)
  return value(width) if callable(value) else value


# Decisions 1a / 7.1 option B follow-up (controller ruling 2026-11-14): each
# dependency register (an iat_delta "last arrival time") writes its OWN 16-bit
# scratch field, named from the register, and the value registers of its chain
# read that field. One shared meta.current_iat made every later chain's writer
# wait (write after read) for the earlier chains' readers, which p4c serialised
# and the register schedule does not model (model8/impl/log.md, Task 2).
_DEPENDENCY_SUFFIX = "_last_arrival_time"


def iat_field_name(dependency_register):
  """The metadata scratch field a dependency register's .execute() writes:
  flow_last_arrival_time -> flow_current_iat (likewise fwd_, bwd_); any other
  name -> <name>_current_iat."""
  base = (dependency_register[:-len(_DEPENDENCY_SUFFIX)]
          if dependency_register.endswith(_DEPENDENCY_SUFFIX) else dependency_register)
  return base + "_current_iat"


def iat_scratch_fields(feature_intervals, catalog=None):
  """The scratch fields generate_P4_registers_and_apply writes for these
  features, in first-seen order: one per distinct dependency register (only
  declared when used)."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG
  fields = []
  for feature in (name.lower() for name in feature_intervals):
    for reg in catalog.get(feature, {}).get("registers", ()):
      if reg["role"] == "dependency" and iat_field_name(reg["name"]) not in fields:
        fields.append(iat_field_name(reg["name"]))
  return fields


def call_site_constants(feature_intervals, catalog=None):
  """The registers, first-seen order, whose `.execute()` call site adds a constant
  (_CALL_SITE_SUFFIX: C1f's Min read `+ 1`) for these features. p4c keeps each such
  constant in a 16-bit PHV container of its own (Task 33), which p4model.phv counts."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG
  names = []
  for feature in (name.lower() for name in feature_intervals):
    gated_by = catalog.get(feature, {}).get("gated_by")
    for reg in catalog.get(feature, {}).get("registers", ()):
      if (_generated_body(reg["body"], gated_by) in _CALL_SITE_SUFFIX
          and reg["name"] not in names):
        names.append(reg["name"])
  return names


def _uses_body(feature_intervals, bodies, catalog=None):
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG
  return any(reg["body"] in bodies
             for feature in (name.lower() for name in feature_intervals)
             for reg in catalog.get(feature, {}).get("registers", ()))


def uses_timestamp(feature_intervals, catalog=None):
  """Whether calc_timestamp writes the fine IAT clock meta.now_pseudo_us for
  these features: some selected feature has an iat_delta (last-arrival)
  register."""
  return _uses_body(feature_intervals, ("iat_delta",), catalog)


def uses_coarse_clock(feature_intervals, catalog=None):
  """Whether calc_timestamp writes the coarse clock meta.now_coarse (Task 38)
  for these features: some selected feature is a duration."""
  return _uses_body(feature_intervals, ("coarse_duration", "coarse_duration_pair"), catalog)


def uses_duration_pair(feature_intervals, catalog=None):
  """Whether the program declares DURATION_PAIR_STRUCT: some selected feature's
  register is a {first, dur} pair (fwd_iat_total / bwd_iat_total, Task 38)."""
  return _uses_body(feature_intervals, ("coarse_duration_pair",), catalog)


def _register_types(body_kind, width):
  """(value type, RegisterAction return type) of a body kind's register."""
  return _REGISTER_VALUE_TYPES.get(body_kind, ("bit<%d>" % width, "bit<%d>" % width))


def _register_declaration(name, width, initial_value=None, body_kind=None):
  # Indexed by meta.flow_hash, FLOW_INDEX_BITS wide (A1a).
  value_type = _register_types(body_kind, width)[0]
  if initial_value is None:
    return "\tRegister<{type}, bit<{index}>>(MAX_NUM_FLOWS) {name}_reg;\n".format(
        type=value_type, index=FLOW_INDEX_BITS, name=name)
  return "\tRegister<{type}, bit<{index}>>(MAX_NUM_FLOWS, {init}) {name}_reg;\n".format(
      type=value_type, index=FLOW_INDEX_BITS, name=name, init=initial_value)


def _register_action_declaration(name, width, body_kind, owner=None, iat=None):
  # owner: None for the literal body (an ungated register), else the owner
  # direction's meta.fwd value (_DIRECTION_READ_BODIES). iat: the scratch field
  # (iat_field_name) the chain's dependency register writes, for the body kinds
  # that read one.
  template = (_REGISTER_ACTION_BODIES if owner is None else _DIRECTION_READ_BODIES)[body_kind]
  if "{iat}" in template and iat is None:
    raise ValueError("register {!r} (body {!r}) reads an inter-arrival time, but its "
                     "feature lists no dependency register before it".format(name, body_kind))
  body = template.format(width=width, name=name, owner=owner, iat=iat)
  value_type, return_type = _register_types(body_kind, width)
  return (
      "\tRegisterAction<{type}, bit<{index}>, {rv}>({name}_reg) {name}_action = {{\n"
      "{body}"
      "\t}};\n"
  ).format(type=value_type, rv=return_type, index=FLOW_INDEX_BITS, name=name, body=body)


def generate_P4_registers_and_apply(feature_intervals, catalog=None):
  """
  Decisions 1a / 7.1 option B (2026-11-14): there are no `if (meta.fwd ==
  ...)` blocks. Every register's `.execute()` is emitted unconditionally, at
  the apply block's outer indent, in the order ungated features, then
  fwd-owned (catalog gated_by "fwd"), then bwd-owned features, first-seen
  dedup; each direction-owned register gets a _DIRECTION_READ_BODIES body
  that updates only in its own direction and only reads in the other, so
  every <feature>_val field is written on every packet. Ungated registers
  keep their literal bodies. "gated" below means direction-owned.

  Resolve a selected feature set to the TNA registers, RegisterActions, and
  per-packet update logic needed for Milestone 1's flow- and IAT-tracking
  design (ground truth: p4/tofino_spike/tna_m1_flows_iat_spike.p4, compiled
  with the real Tofino p4c). Called from generate_p4_program to resolve
  selected features and wire output into resources/p4_template.p4.

  Parameters:
    feature_intervals: dict whose keys are selected feature names, already
      normalised (lowercase, underscore-joined -- normalise_feature_name())
      by get_nodes()/get_feature_intervals() elsewhere in this file (e.g.
      "flow_iat_max"). Only the keys are consulted; values are ignored.
      Each key is still .lower()'d before being looked up in `catalog`; on
      an already-normalised key that is a no-op, kept only because this
      function can also be called directly with a raw_feature_names-style
      dict that hasn't been through get_nodes.
    catalog: feature -> register dependency catalog to resolve against
      (see feature_registers.FEATURE_REGISTER_CATALOG for the expected
      shape). Defaults to feature_registers.FEATURE_REGISTER_CATALOG.
      Exposed as a parameter so tests can exercise the generator against a
      synthetic catalog without monkeypatching module state.

  Feature names absent from `catalog` are silently skipped by THIS function
  (later milestones will call this with a catalog that isn't fully
  populated yet for every feature they select -- that must not crash here).
  F2: it is the CALLER's job to decide whether a silent skip is acceptable
  -- generate_P4_code, the only production caller, is not: it diffs its own
  requested feature set against this function's `resolved` return value
  (below) and raises AssertionError if anything didn't resolve, because letting
  an uncatalogued feature's <feature>_val field compile silently would read
  0 for every packet forever (see generate_P4_code's own F2 comment).

  Returns a 4-tuple of P4 source strings plus a resolution set, one source
  string per marker payload a future TNA template will substitute this
  function's output into, and `resolved` for callers (generate_P4_code) to
  detect what did NOT resolve:
    (registers_code, register_actions_code, feature_update_apply_code, resolved)

    - registers_code: `Register<bit<W>, bit<FLOW_INDEX_BITS>>(MAX_NUM_FLOWS)
      <name>_reg;` declarations (TNA syntax), plus the ONE `@symmetric`
      `Hash<bit<FLOW_INDEX_BITS>>` instance the
      apply-block's hash action needs (`flow_hash_calc`) and the fixed
      `flow_forward_srcaddr_reg` register used for flow-direction
      bookkeeping (see below) -- both always emitted, unconditionally,
      whenever the resolved feature set is non-empty; neither is
      catalog-driven or routed through the register_order/_note_touch dedup
      machinery.
    - register_actions_code: the plain calc_flow_hash action (and
      calc_timestamp, only emitted when an included register reads a clock:
      it writes meta.now_pseudo_us for a last-arrival register and
      meta.now_coarse for a duration, Task 38), plus one `RegisterAction<...>
      <name>_action = {...};` block per resolved register, using the exact
      bodies from _REGISTER_ACTION_BODIES (no MathUnit since Task 38). Flow
      direction/first-seen bookkeeping (validated against the real Tofino
      compiler this session: p4/tofino_spike/tna_m2_symmetric_hash_spike.p4,
      and the real M2 program with this swap,
      p4/tofino_spike/tna_m2_real_with_symhash.p4, both 0 errors) is now a
      SINGLE `flow_orientation_action` RegisterAction on
      `flow_forward_srcaddr_reg`: since `calc_flow_hash` is symmetric
      (`@symmetric` on src/dst address and src/dst port since review A1b; an
      XOR of each pair before -- so both
      directions of a flow hash identically; no min/max comparison needed,
      since TNA's Gateway hardware can only compare a field against a
      compile-time constant, not two runtime fields against each other,
      confirmed as a real compile restriction this session), the register
      simply stores whichever packet's own srcAddr was seen FIRST for that
      flow index; a later packet is "fwd" iff its own srcAddr matches the
      stored value. One touch per packet, not two. CAVEAT: pair-symmetric
      hashing has a different (likely worse) collision profile than true
      min/max canonical ordering -- not measured, flagged not fixed,
      consistent with this project's treatment of every other
      resource-oracle approximation (e.g. reading an IAT in 1.024 us ticks,
      p4model.units.iat_ticks).
    - feature_update_apply_code: the per-packet apply-block snippet: the
      hash-calc and (conditionally) timestamp calls, the single
      flow_orientation_action touch, then each *distinct register name*'s
      `.execute()` call, unconditionally (a `"gated_by": "fwd"` / `"bwd"`
      register tests meta.fwd inside its own body instead) -- and
      assignment of each "value" register's
      result into `meta.<feature_lower>_val` (e.g. `meta.flow_iat_max_val`,
      matching the spike's `metadata_t` field naming; a "dependency"
      register's result is not itself a feature value, so it is assigned
      into its chain's own scratch field (iat_field_name, e.g.
      `meta.flow_current_iat`; one shared `meta.current_iat` until
      2026-11-14) instead, much
      like the spike). This is applied consistently to every "value"
      register, including `fwd_packet_length_max` -- the spike itself
      discards that one register's `.execute()` result (its classification
      table reads `hdr.ipv4.total_len` directly instead), a spike-specific
      shortcut this generator deliberately does not copy. Never emits
      `update_current_flow_features`: there is no bulk-read phase here --
      every register's value is captured directly at its `.execute()`
      call site.

      Deduplicated by register name, not one call site per (feature,
      register) pair: when two or more selected features' catalog entries
      reference the SAME register name (e.g. flow_iat_max and
      flow_iat_min both depend on "flow_last_arrival_time"), only the
      first-encountered feature (in feature_intervals iteration order)
      gets an `.execute()` call site for that register; every later
      feature that references the same name reuses the value the first
      call already produced (e.g. `meta.flow_current_iat`) instead
      of re-invoking `.execute()` on it. This models the real hardware
      constraint that a given RegisterAction's call site represents one
      physical register touch meant to fire (at most) once per packet --
      touching it twice for the same packet would be both semantically
      wrong (the second call would observe the state the first call just
      wrote) and physically redundant. `_note_touch`'s first-seen check
      (below) and `_execute_lines`' `already_executed_registers` set
      (further below) both implement this same first-seen-wins model, so
      the register declarations, RegisterAction blocks, and `.execute()`
      call sites always agree on exactly which registers exist and how
      many touches each one gets. `flow_forward_srcaddr_reg` (the fixed
      flow-direction bookkeeping register, always exactly 1 real touch per
      packet via `flow_orientation_action`) is NOT routed through this
      catalog-driven dedup machinery at all -- like `flow_hash_calc`, it is
      a fixed, always-emitted register outside `register_order`/
      `_note_touch`'s bookkeeping entirely. Every catalog-driven register
      IS recorded once, the first time any feature references it,
      regardless of how many further features also list it as a
      dependency -- e.g. flow_last_arrival_time reports exactly 1 real
      touch for flow_iat_max + flow_iat_min sharing
      it, matching the single real `.execute()` call site _execute_lines
      actually emits for it. The cross-gate register-sharing hazard guard
      (below) is what actually protects this property across gate classes
      -- see its comment for what it catches and why.
    - resolved: the `set` of INPUT feature names (i.e. keys of the
      `feature_intervals` argument, not catalog keys) this call emitted
      registers for -- exactly `matched_features` above, converted to a
      set (see the code right before the `return` for why that's exact,
      not approximate). Empty when `feature_intervals` is empty or when
      none of its keys are in `catalog`. Lets a caller (generate_P4_code)
      diff its own full requested feature set against this to find what
      silently did NOT resolve (F2), without this function itself having
      to decide whether that's acceptable.

  Raises RuntimeError, at generation time (not left to fail later at `p4c`
  invocation), for a catalog entry with an unsupported `gated_by` value
  (only None, "fwd", and "bwd" are implemented). Raises AssertionError (R10) if a
  register name is shared by features in different gate classes -- e.g.
  one feature's entry lists it ungated and another's lists it "fwd" --
  which would give it one direction's body and feed the other class a
  value that only that direction updates; see the cross-gate register-sharing hazard guard
  below for the full explanation. (An earlier design also capped the
  number of real, deduplicated `.execute()` call sites a single register
  could accumulate, on the theory a register might someday need more than
  one touch per packet; that never happened under any catalog this project
  ever shipped -- re-verified against Phase 2's full 18-feature catalog,
  where every register, including the shared `flow_last_arrival_time`/
  `fwd_last_arrival_time`/`bwd_last_arrival_time` dependency registers,
  still resolves to exactly one real touch -- so the cap and its guard were
  deleted; the cross-gate check above is what actually protects registers
  from being touched in a way that produces wrong per-packet values.)
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # feature_intervals keys are already lowercase, underscore-joined (see
  # get_nodes()'s normalise_feature_name()), so .lower() here is normally a
  # no-op; kept as a defensive normalisation for callers that pass a
  # feature_intervals dict built by hand rather than via get_nodes.
  # Preserves feature_intervals' iteration order.
  matched_features = [f for f in (name.lower() for name in feature_intervals.keys()) if f in catalog]

  # Each catalog entry lists its OWN dependency register, so a feature selected
  # alone (flow_iat_min without flow_iat_max) still executes its chain's
  # last-arrival register first (_execute_lines;
  # test_resolving_flow_iat_min_alone_auto_executes_its_dependency).

  if not matched_features:
    return "", "", "", set()

  # ---- Resolve the deduplicated, ordered register set ----

  register_order = []  # ordered list of register names (first-seen)
  register_info = {}   # name -> {"width":, "body":, "gated_by":, "iat":}

  def _note_touch(name, width=None, body=None, gated_by=None, iat=None):
    # Plain first-seen recorder: the first time a register name is noted,
    # it is added to register_order/register_info; every later call for an
    # already-seen name (e.g. a second feature that also lists the same
    # dependency register) is a no-op, because _execute_lines (below) will
    # reuse that first call's already-produced value rather than emitting
    # another .execute() line for it.
    if name not in register_info:
      register_order.append(name)
      register_info[name] = {"width": width, "body": body, "gated_by": gated_by,
                             "iat": iat}

  feature_registers = {}  # feature -> ordered list of its catalog register dicts
  # The two clocks calc_timestamp writes (Task 38): the fine IAT clock for a
  # last-arrival register, the coarse clock for a duration.
  needs_fine_clock = uses_timestamp(matched_features, catalog)
  needs_coarse_clock = uses_coarse_clock(matched_features, catalog)
  needs_timestamp = needs_fine_clock or needs_coarse_clock

  for feature in matched_features:
    entry = catalog[feature]
    gated_by = entry.get("gated_by")
    if gated_by not in (None, "fwd", "bwd"):
      raise RuntimeError(
          "Feature '{feature}' has unsupported gated_by={gated_by!r}; "
          "only None, 'fwd', and 'bwd' are implemented.".format(feature=feature, gated_by=gated_by)
      )

    regs = []
    iat = None  # the scratch field of the chain's latest dependency register
    for reg in entry["registers"]:
      if reg["role"] == "dependency":
        iat = iat_field_name(reg["name"])
      _note_touch(reg["name"], width=reg["width"], body=_generated_body(reg["body"], gated_by),
                  gated_by=gated_by,
                  iat=None if reg["role"] == "dependency" else iat)
      regs.append(reg)
    feature_registers[feature] = regs

  # ---- Cross-gate register-sharing hazard guard ----
  #
  # _execute_lines() (below) is called three times -- ungated, then
  # fwd-gated, then bwd-gated -- sharing one `already_executed_registers`
  # set, so whichever call reaches a register name FIRST is the one that
  # actually emits its .execute() call site; every later call for that same
  # name is a no-op that assumes the value was already produced. A register's
  # body is chosen by its FIRST feature's gate class (a fwd-owned register
  # updates only on meta.fwd == 1 packets, _DIRECTION_READ_BODIES), so a
  # register reused by a feature in a DIFFERENT gate class (say, an ungated
  # feature, or a bwd-owned one) would hand that feature a value only the
  # other direction updates. Catch that at generation time instead of
  # emitting silently-wrong P4.
  # This is the guard that actually protects register-sharing correctness
  # here -- an earlier "touch-count" guard (deleted; see git history) tried
  # to bound how many real .execute() call sites a single register could
  # accumulate, but every catalog this project has ever shipped, including
  # Phase 2's full 18-feature catalog, resolves every register to exactly
  # one real touch (the dedup above already guarantees that), so that older
  # guard could never fire and added nothing this check doesn't already
  # cover.
  gate_of_register = {}
  for feature in matched_features:
    gate = catalog[feature].get("gated_by")
    for reg in catalog[feature]["registers"]:
      previous = gate_of_register.setdefault(reg["name"], gate)
      if previous != gate:
        # R10: an invariant of the catalog (AssertionError, see the uncatalogued check).
        raise AssertionError(
            "register {!r} is shared by features in different gate classes "
            "({!r} and {!r}); its body would update in one gate class's direction "
            "and be read as garbage by the other. Promote it to an ungated register "
            "or give each gate class its own.".format(reg["name"], previous, gate))

  # ---- /* REGISTERS */ ----

  # The flow index (review A1, adopted 2026-11-15; p4c evidence: Task 28,
  # results/encoding_evidence_2026_11/model8/impl/probe_generator_A1_A2.md).
  #   * A1a, Hash<bit<FLOW_INDEX_BITS>>: the hash unit hands out the low 12
  #     CRC bits itself, so there is no 32-bit result to mask with `& 0xFFF`.
  #     The mask was an ALU instruction on a value that exists only after the
  #     hash stage, so p4c gave it a stage of its own; a narrower output is
  #     wiring. Same function (prog.bfa: identical crc_rev expression, bits
  #     0..11 instead of 0..31 then AND 4095), one stage less on 8/8 p4c
  #     compiles, blocks unchanged, 1-2 PHV containers fewer.
  #   * A1b, @symmetric on (src_addr, dst_addr) and (src_port, dst_port), the
  #     raw header fields as inputs: p4c gives both fields of a pair the same
  #     hash-matrix columns (bf-p4c mau/ixbar_expr.cpp verifySymmetricHashPairs),
  #     so a packet and its reverse hash to the same slot, which is all the
  #     orientation register needs. It replaces the XOR of the two fields of
  #     each pair, which p4c had to compute in a stage of its own before the
  #     hash unit could read it. Two stages less than before on 15/15 p4c
  #     compiles (with A1a), blocks unchanged, 2-5 PHV containers fewer. It is
  #     a different CRC message (13 bytes, each pair at one position) than the
  #     XOR form's 7 bytes, so a flow lands in a different slot than before;
  #     both directions still meet in one slot (checked from prog.bfa and
  #     numerically on 4 5-tuples). p4c's own note: a symmetric hash "is not
  #     truly the CRC function they requested, but a variation on it"
  #     (asm_hash_output.cpp).
  # With both, the hash runs in stage 0 and the first RegisterAction in stage
  # 1 (program.FLOW_HASH_LEVEL).
  registers_code = "".join(
      '\t@symmetric("%s", "%s")\n' % pair for pair in FLOW_HASH_SYMMETRIC_PAIRS) + (
      "\tHash<bit<{index}>>(HashAlgorithm_t.CRC32) flow_hash_calc;\n"
      "\tRegister<bit<32>, bit<{index}>>(MAX_NUM_FLOWS) flow_forward_srcaddr_reg;\n"
  ).format(index=FLOW_INDEX_BITS)
  for name in register_order:
    initial_value = _register_initial_value(register_info[name]["body"], register_info[name]["width"])
    registers_code += _register_declaration(name, register_info[name]["width"], initial_value=initial_value,
                                            body_kind=register_info[name]["body"])

  # ---- /* REGISTER_ACTIONS */ ----

  register_actions_code = (
      "\taction calc_flow_hash() {\n"
      "\t\tmeta.flow_hash = flow_hash_calc.get({\n"
      "\t\t\thdr.ipv4.src_addr,\n"
      "\t\t\thdr.ipv4.dst_addr,\n"
      "\t\t\thdr.ipv4.protocol,\n"
      "\t\t\thdr.tcp.src_port,\n"
      "\t\t\thdr.tcp.dst_port\n"
      "\t\t});\n"
      "\t}\n"
  )

  # The 32-bit pseudo-microsecond clock (V1o, Task 34): bits 41..10 of the
  # 48-bit nanosecond timestamp (units of 1.024 us; it wraps every 73 min), the
  # slice Task 29's V1o compiled (impl/probe_timestamp.md). It was 16 bits
  # ([25:10], wrapping every 67.1 ms), which made every gap past 67 ms read
  # modulo 65536 ticks; the last-arrival registers now hold all 32 bits and
  # clamp the gap at 65535 (_REGISTER_ACTION_BODIES["iat_delta"]). A 32-bit
  # @pa_container_size pin on it is refused by p4c (Task 29 V1p, 3/3: the
  # shifted slice must split where its source bits split), so it keeps only
  # the @pa_solitary pin (generate_p4_program). Written only when a last-arrival
  # register remains (an IAT Max / Min is selected) since Task 38.
  # The 32-bit COARSE clock (Task 38, P3h): bits 47..16 of the MAC timestamp, units
  # of 65.536 us (it wraps every 78 h, as the 48-bit timestamp does), read by the
  # duration registers (_REGISTER_ACTION_BODIES["coarse_duration"]).
  #   WHY ingress_mac_tstamp and not global_tstamp[47:16]: p4c's PHV allocation
  #     fails whenever the fine clock is also cut from global_tstamp (Task 37,
  #     probe_p3c.md: 4/4 designs). Every other second view of that timestamp failed
  #     too -- the SALU reading the slice directly, a byte-aligned [47:18] copy,
  #     slices [31:6] / [31:8] of now_pseudo_us (8 variants). The MAC timestamp is a
  #     separate intrinsic field, so it has containers of its own.
  #   EVIDENCE: Task 37 P3h, p4c 0 errors with a binary on both designs that also
  #     have the fine clock, at their unedited stage depth and blocks; prog.bfa keeps
  #     ingress_mac_tstamp[47:16] in one W container and now_coarse in another (+2
  #     32-bit containers, p4model.phv COARSE_CLOCK_CONTAINERS). On tofino-model the
  #     two timestamps carry the same bytes (equal on 61/64 packets, one 1 ms model
  #     timer step apart on 3); a duration is a difference of readings of ONE source.
  #     Not verified on hardware (there is none): that the MAC and parser timestamps
  #     come from one timer on a real Tofino-1.
  if needs_timestamp:
    register_actions_code += "\n\taction calc_timestamp() {\n"
    if needs_fine_clock:
      register_actions_code += "\t\tmeta.now_pseudo_us = ig_prsr_md.global_tstamp[41:10];\n"
    if needs_coarse_clock:
      register_actions_code += "\t\tmeta.now_coarse = ig_intr_md.ingress_mac_tstamp[47:16];\n"
    register_actions_code += "\t}\n"

  # Symmetric-hash flow bookkeeping (Part G.6 in reviews/t11_tofino_port_and_env.md,
  # first validated with the XOR form against the real Tofino compiler:
  # p4/tofino_spike/tna_m2_symmetric_hash_spike.p4 and
  # p4/tofino_spike/tna_m2_real_with_symhash.p4, both 0 errors; the hash is
  # @symmetric since A1b, above). Both directions of a flow hash to one slot, so
  # a single register stores the first-seen packet's own srcAddr; later packets
  # are "fwd" iff their own srcAddr matches it.
  # CAVEAT: a symmetric hash sees each pair only through src XOR dst, so its
  # collision profile differs from (likely worse than) true min/max canonical
  # ordering -- not measured, flagged not fixed, consistent with this project's
  # treatment of every other resource-oracle approximation (e.g. reading an IAT
  # in 1.024 us ticks, p4model.units.iat_ticks).
  # Orientation fix (Task 33b, adopted 2026-11-15 as a pure bug fix). The body was
  # `if (value == 0) { value = src; } rv = (value == src) ? 1 : 0;`, whose P4
  # meaning is rv = (old == 0 || old == src). p4c compiled the compare against the
  # OLD memory word (prog.bfa flow_orientation_action: `equ hi, lo, -phv_lo` /
  # `output cmphi, alu_hi`), so rv = (old == src): every flow's FIRST packet was
  # labelled bwd and the initiator's second packet started the fwd chain -- every
  # Fwd/Bwd feature (counts, lengths, IATs) was one packet off on the switch, in
  # every encoding (tofino-model log: ig_md.fwd = 0 on each flow's first packet,
  # runs 1 and 2). The fix states the rule with both compares before the write: 2
  # compare units, Tofino-1's limit (bfa `equ lo, lo` / `equ hi, lo, -phv_lo` /
  # `alu_a (cmplo | cmphi), lo, phv_lo` / `output (cmplo | cmphi), alu_hi`). p4c: 0
  # errors, every table in the stage of the unfixed compile; live (tofino-model):
  # the labels follow the CICFlowMeter rule 10/10 and every chain's first packet
  # gives IAT 0 (impl/probe_first_packet.md, "Live check (Task 33b)"). A compare
  # that reads `value` after the body wrote it is refused by program_checks
  # (salu_compare_after_write).
  register_actions_code += (
      "\n"
      "\tRegisterAction<bit<32>, bit<" + str(FLOW_INDEX_BITS) + ">, bit<1>>(flow_forward_srcaddr_reg) "
      "flow_orientation_action = {\n"
      "\t\tvoid apply(inout bit<32> value, out bit<1> rv) {\n"
      "\t\t\tif (value == 0 || value == hdr.ipv4.src_addr) {\n"
      "\t\t\t\tvalue = hdr.ipv4.src_addr;\n"
      "\t\t\t\trv = 1w1;\n"
      "\t\t\t} else {\n"
      "\t\t\t\trv = 1w0;\n"
      "\t\t\t}\n"
      "\t\t}\n"
      "\t};\n"
  )

  for name in register_order:
    info = register_info[name]
    owner = _DIRECTION_OWNER.get(info["gated_by"])
    register_actions_code += "\n" + _register_action_declaration(
        name, info["width"], info["body"], owner=owner, iat=info["iat"])

  # ---- /* FEATURE_UPDATE_APPLY */ ----

  apply_lines = [
      "\t\t\tcalc_flow_hash();",
  ]
  if needs_timestamp:
    apply_lines.append("\t\t\tcalc_timestamp();")

  apply_lines += [
      "",
      "\t\t\tmeta.fwd = flow_orientation_action.execute(meta.flow_hash);",
      "",
  ]

  # Registers already given an .execute() call site somewhere in the apply
  # block, tracked across ALL THREE of the ungated, fwd-gated, and
  # bwd-gated _execute_lines() calls below (a single shared set, not one
  # per call) -- this is the
  # shared-dependency dedup fix: a register (e.g. flow_last_arrival_time)
  # referenced by more than one selected feature's catalog entry (e.g. both
  # flow_iat_max and flow_iat_min) must still be .execute()'d exactly once
  # per packet. Whichever feature reaches that register name first (in
  # matched_features order) emits its call site and produces the value
  # (its own scratch field, iat_field_name, for a "dependency" register); every later feature
  # that lists the same register name reuses that already-produced value
  # instead of re-invoking .execute() on it.
  already_executed_registers = set()

  def _execute_lines(features, indent):
    lines = []
    for feature in features:
      for reg in feature_registers[feature]:
        name = reg["name"]
        if name in already_executed_registers:
          continue
        already_executed_registers.add(name)
        target = ("meta." + iat_field_name(name) if reg["role"] == "dependency"
                  else "meta." + feature + "_val")
        lines.append("{indent}{target} = {prefix}{name}_action.execute(meta.flow_hash){suffix};".format(
            indent=indent, target=target, name=name,
            prefix=_CALL_SITE_PREFIX.get(register_info[name]["body"], ""),
            suffix=_CALL_SITE_SUFFIX.get(register_info[name]["body"], "")))
    return lines

  ungated_features = [f for f in matched_features if catalog[f].get("gated_by") is None]
  fwd_gated_features = [f for f in matched_features if catalog[f].get("gated_by") == "fwd"]
  bwd_gated_features = [f for f in matched_features if catalog[f].get("gated_by") == "bwd"]

  apply_lines += _execute_lines(ungated_features, "\t\t\t")

  # Decisions 1a / 7.1 option B: the direction-owned call sites follow at the
  # same outer indent, unconditionally; the direction test lives in each
  # register's body (_DIRECTION_READ_BODIES).
  for features in (fwd_gated_features, bwd_gated_features):
    if not features:
      continue
    apply_lines.append("")
    apply_lines += _execute_lines(features, "\t\t\t")

  apply_code = "\n".join(apply_lines) + "\n"

  # `matched_features` (built above) is already exactly the set of INPUT
  # feature names this call emitted registers for: it was built by filtering
  # feature_intervals.keys() (each already .lower()'d, a no-op post-Task-4
  # normalisation) down to the ones present in `catalog` -- so catalog key
  # == input name and matched_features IS the resolved set, not merely a
  # proxy for it. Returned as a set (matched_features is a list, kept in
  # iteration order for everything above) so callers can cheaply diff it
  # against their full requested feature set (see generate_P4_code's F2
  # check) without caring about order or duplicates.
  resolved = set(matched_features)

  return registers_code, register_actions_code, apply_code, resolved