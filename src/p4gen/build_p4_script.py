import os
import math
import re
from pathlib import Path
import numpy as np
from sklearn.tree import export_text
from sklearn.tree import _tree as _sklearn_tree
import csv
import json
from collections import Counter
from itertools import product

from src.p4gen.feature_registers import FEATURE_REGISTER_CATALOG
from src.p4gen import p4_gen_config
from src.p4gen import switch_semantics

INFINITE = (2**16)-1
MAX_CODEWORD_LENGTH = 512
TCAM_BLOCKS_PER_STAGE = 24
TCAM_BLOCK_KEY_LENGTH = 44
TERNARY_MATCHING_ENTRIES_PER_BLOCK = 512
TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8    # hard cap, binds for narrow keys (<=64 bits)
TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE = 64    # byte budget, binds for wider keys
MAX_NUM_FLOWS = 4096  # matches p4/p4_code_RF_models.p4:9 and
                      # p4/tofino_spike/tna_m1_flows_iat_spike.p4

PATH = "resources/"
INTERMEDIATE = "temp/"
OUTPUT_PATH = "p4/"
PATH_TABLE_ENTRIES_OUTPUT = OUTPUT_PATH + "table_entries.json"
PATH_TABLE_TEMPLATE_P4  = PATH + 'table.p4'
PATH_ACTION_TEMPLATE_P4 = PATH + 'action.p4'
PATH_TABLE_CLASSIFICATION_TEMPLATE_P4 = PATH + 'table_classification.p4'
# Task 8: Planter RF_EB-style exact-match code/decision tables. The two
# templates are currently byte-identical -- table_classification.p4 has no
# <MATCH_TYPE>-style marker of its own; match type is baked into <KEYS>,
# which generate_P4_tables_and_apply builds directly in Python (see
# classification_keys below), not read from either template file. A
# separate file still exists (rather than reusing one path for both modes)
# so the two modes have independent templates to diverge from later, and so
# match_type's file-selection story matches the interface both templates
# are meant to express.
PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4 = PATH + 'table_classification_exact.p4'
PATH_P4_CODE_TEMPLATE_INPUT = PATH + 'p4_template.p4'

# Task 19 (R6): read every P4 table/action template exactly once, here, at
# module import, instead of re-opening the same file once per tree and once
# per feature inside generate_P4_tables_and_apply (and, before this task,
# once per generate_P4_actions call for action.p4). A missing/unreadable
# template now fails at import time rather than as a per-call surprise --
# the intended behaviour, matching Task 2's F9 fix for the action template.
#
# table.p4 and table_classification.p4 (see the comment above
# PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4) differ only in their <KEYS>
# line: a single `meta.<FEATURE_NAME>: <MATCH_TYPE>;` line for the
# range-matching feature tables vs one `meta.code_<feature> : <match_type>;`
# line per classification key, already built in Python (see
# _classification_keys) -- plus one purely cosmetic trailing-whitespace
# difference on the final closing "}" line (table.p4 has 3 trailing spaces
# there that table_classification.p4 does not; confirmed with a direct byte
# diff of the two files during Task 19). generate_P4_tables_and_apply now
# reads a SINGLE shared template, _TABLE_TEMPLATE (table_classification.p4's
# content), for both the range-matching feature tables and the ternary
# classification tables, building each one's own one-line/multi-line <KEYS>
# text in Python; the range-table path additionally re-adds table.p4's 3
# trailing spaces itself (see the comment at its call site) so its emitted
# bytes stay byte-for-byte identical to every caller before this task.
_TABLE_TEMPLATE = Path(PATH_TABLE_CLASSIFICATION_TEMPLATE_P4).read_text()
_ACTION_TEMPLATE = Path(PATH_ACTION_TEMPLATE_P4).read_text()
# table_classification_exact.p4 is byte-identical to table_classification.p4
# (see the comment above PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4) --
# still read from, and kept as, its own file/constant (not folded into
# _TABLE_TEMPLATE) purely so match_type='exact' keeps an explicit template
# file of its own to select and diverge from later, per the decision to
# preserve resources/table_classification_exact.p4 as a file.
_TABLE_CLASSIFICATION_EXACT_TEMPLATE = Path(PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4).read_text()


_IDENT_RE = re.compile(r'[^0-9a-z]+')
_TREE_LEAF = _sklearn_tree.TREE_LEAF


def normalise_feature_name(name):
  """Canonical form for both FEATURE_REGISTER_CATALOG keys and P4 identifiers.

  Dataset columns arrive dot-separated ('Flow.IAT.Max' -- dataset.py renames
  every column with .replace(' ', '.')), older fixtures arrive space- or
  underscore-separated. Every run of non-alphanumeric characters collapses to a
  single '_' so all three spellings land on one key, and that key is a legal P4
  identifier. Leading/trailing separators are stripped so 'Flow.IAT.Max.' cannot
  become a distinct key."""
  return _IDENT_RE.sub('_', name.lower()).strip('_')



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
            # the same test, so floor is exact on integer-valued features. It is
            # not exact on this dataset's *.Mean features, which are genuine
            # means and take fractional values -- but it stays 3-8x closer to
            # the fitted model than round does (measured per-tree predictions
            # changed: 24 vs 185 at 3 trees, 48 vs 316 at 7).
            tree_obj.threshold[i] = math.floor(tree_obj.threshold[i])
  return clf


def get_tree_textual_representation(clf, feature_names, verbose=False):
  """Renders each estimator's tree as export_text's indented text.

  No code anywhere in this repo parses this any more: Task 15 replaced the
  export_text-and-reparse path (the removed _get_nodes_from_text, which this
  function's rendered text used to feed) with get_nodes(), which reads
  estimator.tree_'s C-level arrays directly and so cannot hit export_text's
  own truncation default (see the comment below). Kept anyway -- not deleted
  alongside _get_nodes_from_text -- because it is a correct, general-purpose
  "render this fitted tree as text" utility with no defect of its own, and
  because tests/test_tree_parsing.py's characterisation suite used it
  (commit 838acf6) to prove get_nodes() agrees with the old parser before
  that parser was deleted; kept available for the same kind of manual
  re-characterisation if get_nodes() itself ever needs re-verifying against
  a human-readable rendering."""
  tree_textual_representation = {}

  for idx,tree in enumerate(clf.estimators_):
    # max_depth MUST be passed explicitly: sklearn's export_text defaults to
    # max_depth=10 and renders anything deeper as
    # "|--- truncated branch of depth N" lines -- see commit 838acf6's
    # test_export_text_default_max_depth_truncates_a_deep_tree (since
    # removed alongside _get_nodes_from_text, the parser it was
    # characterising) for the measured effect: a real depth-12 tree with 350
    # leaves parsed as 168, and on a depth-14 tree 2516 of 4000 probe inputs
    # then matched no table entry at all. Sizing to the tree's own depth
    # keeps this function's OWN output exact regardless. (get_nodes() -- the
    # production path since Task 15 -- reads tree_ arrays directly and is
    # immune to this class of bug entirely: there is no rendered text to
    # under-size.)
    tree_textual_representation[idx] = export_text(
        tree, feature_names=feature_names, max_depth=max(1, tree.get_depth()))

  if verbose == True:
    for tree in tree_textual_representation:
      print("Tree ", tree)
      print(tree_textual_representation[tree])

  return tree_textual_representation


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
  truncation hazard (see get_tree_textual_representation's comment), a
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
    raise ValueError(
        "feature names collide after normalisation, their intervals would "
        "silently merge: {}".format(collisions))


def get_feature_intervals(model, selected_features):
  _reject_colliding_feature_names(selected_features)
  tree_nodes = tree_nodes_for(model, selected_features)
  return feature_intervals_from_nodes(tree_nodes)


def get_joint_feature_intervals(model_a, features_a, model_b, features_b):
  """The joint-encoding counterpart of get_feature_intervals: derives ONE
  shared feature_intervals dict from the union of both models' trees, via
  the same offset trick used to keep the two models' node IDs from
  colliding (model_b's tree indices are shifted by len(model_a's trees)
  before merging). This was previously reimplemented identically three
  times (evaluation.multi_model_memory_evaluation's 'joint' branch,
  feature_selection._derive_joint_feature_intervals, and
  main.implement_tree_models_in_P4); this is the single canonical copy."""
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


def generate_codewords(paths_leaf_nodes_per_tree, feature_intervals):
  '''
    For each leaf node in each Decision Trees, this function generates the corresponding codeword, associated with the leaf node class label.

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
  bfrtcli._make_core_method_strs), so get_table_entries needs the two halves
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


def get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset=None, path_to_output=OUTPUT_PATH, output_filename="table_entries.json", verbose=False, use_default_action_discount=False):
  '''
  Inputs: feature_intervals [dict]: Dictionary where each key is a tree_id. The values are a list of feature intervals for each tree.
          codewords [dict]: Dictionary where each key is a tree_id. The values are a list of dictionaries.
                            Each key in the dictionary is a codeword corresponding to a leaf node.
                            Each value is the class label associated with that codeword (i.e.: to the leaf node).
          use_default_action_discount [bool]: Planter-style default-action discount. When
                            True, every one of each tree's leaves sharing the majority class (see
                            most_common_class_and_dropped_codewords) is omitted from that tree's
                            explicit table entries; instead ONE `is_default_action` record per tree
                            carries that majority class, which the control plane installs as the
                            table's default action at deploy time (see p4/deploy_table_entries.py --
                            the generated P4 itself declares no default action, see
                            generate_P4_tables_and_apply). False (the default) writes every leaf as
                            an explicit entry and emits no default-action record at all.

  Outputs: This function writes a JSON file: a flat list of table-entry records covering both
           the feature codeword-bit (range) tables and the codeword-to-leaf-node (classification)
           tables.

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
             (generate_P4_actions).
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
              "code_flow_iat_max": {"value": "0x5", "mask": "0x7"},
              "code_flow_iat_mean": {"value": "0x0", "mask": "0x0"}
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

  Task 3 (disjoint-encoding single-pipeline generator) -- current support
  for RESOLVED/namespaced `feature_intervals` keys (e.g. "app_flow_iat_max")
  is PARTIAL, not "zero changes needed" everywhere:

  - Section 1 below (the per-feature codeword-generation table entries,
    "table_<idx>_<feature_name>" / "set_code_<feature_name>") IS genuinely
    agnostic to raw vs. resolved names. It only ever reads intervals from
    `feature_intervals[feature_name]` and never cross-references tree-path
    step names, so passing it resolved/namespaced keys works correctly
    as-is.

  - `generate_codewords` (which must be called with the SAME
    `feature_intervals` before this function, since `codewords`' segment
    order has to match `feature_intervals`' iteration order) is NOT
    namespace-aware: it matches each `feature_intervals` key directly
    against the RAW feature names recorded in each tree path (e.g.
    "flow_iat_max"). A resolved/namespaced key like "app_flow_iat_max"
    will never match a real tree-path step name, so that feature is
    silently treated as absent from every path and wildcarded in every
    codeword -- wrong codewords, with no error raised.

  - Section 2 below (the per-tree classification-entry loop that slices
    each codeword across every entry in `feature_intervals`) also assumes
    `feature_intervals` contains exactly the keys/order used to build the
    per-model classification table's key fields. After this task, each
    model's classification table only declares key fields for its own
    feature subset (see `feature_names_app`/`feature_names_ddos` in
    `generate_P4_tables_and_apply`), not the full resolved plan. Calling
    this function with the FULL resolved-name-keyed dict (spanning both
    models) would slice codewords into the wrong number/width of key
    chunks for a real per-model table.

  Net effect: a caller wanting genuinely namespaced, per-model exact
  codewords/table_entries.json would need further changes to
  `generate_codewords` (to match resolved names back to their underlying
  raw feature for path-matching) and to how this function is invoked
  (once per model, with that model's own feature subset) -- this task did
  not make those changes. No caller in this task's scope exercises the
  disjoint-namespaced path through `generate_codewords`/this function
  today (main.py's pipeline is joint-only; feature_selection.py's
  real-compile validation only needs `generate_P4_code`'s emitted .p4
  SOURCE, not table_entries.json).
  '''

  feature_code_length = {}

  table_entries = []
  feature_idx = 0

  for feature_name in feature_intervals:
    intervals = feature_intervals[feature_name]

    feature_code_length[feature_name] = len(intervals)-1

    # 1. Table entries for generating the codeword based on feature values
    for idx,interval in enumerate(intervals[::-1]):

      table_entry = {}
      table_entry["table_name"] = "table_"+str(feature_idx)+"_"+feature_name.lower()
      table_entry["action"] = "set_code_"+feature_name.lower()

      code = thermometer_code(len(intervals), len(intervals) - 1 - idx)

      minimum = str(interval[0])
      maximum = str(interval[1])

      # bf_rt exposes a range-matched key field as <short_name>_start /
      # <short_name>_end. The short name is the P4 key expression without its
      # `meta.` prefix -- i.e. exactly the <FEATURE_NAME>+"_val" text
      # generate_P4_tables_and_apply declares this table's key on.
      table_entry["is_default_action"] = False
      table_entry["priority"] = idx
      table_entry["key_fields"] = {
          feature_name.replace(" ","_").lower()+"_val": {
              "start": minimum, "end": maximum}
      }
      # `code` is the literal parameter name of every set_code_* action
      # (resources/action.p4).
      table_entry["action_params"] = {"code": str(int(code, 2))}

      table_entries.append(table_entry)

    feature_idx +=1


  # 2. Table entries for getting each tree's classification based on generated codeword
  for tree_idx,tree in enumerate(codewords):
    # Task M2-B2: `tree` is no longer a runtime action parameter -- each tree
    # has its own dedicated table and action (see generate_P4_actions /
    # generate_P4_tables_and_apply), so action_params carries only the class
    # value. Derived once per tree, so this tree's explicit entries and its
    # default-action record below can never disagree about which table they
    # address.
    if offset==None:
      #One model encoding
      table_name = "get_classification_tree_"+str(tree_idx)
      action_name = "classify_flow_codeword_"+str(tree_idx)
    elif tree_idx < offset:
      #Multiple models encoding: App trees come first
      table_name = "get_classification_tree_app_"+str(tree_idx)
      action_name = "classify_flow_codeword_app_"+str(tree_idx)
    else:
      table_name = "get_classification_tree_ddos_"+str(tree_idx-offset)
      action_name = "classify_flow_codeword_ddos_"+str(tree_idx-offset)

    # Planter-style discount: every leaf sharing this tree's majority class is
    # excluded from this tree's explicit entries and covered by ONE
    # default-action record instead. Computed once per tree, up front, so
    # every other leaf is still written as before.
    default_class, dropped_codewords = None, set()
    if use_default_action_discount and len(codewords[tree]) > 0:
      default_class, dropped_list = most_common_class_and_dropped_codewords(codewords[tree])
      dropped_codewords = set(dropped_list)

    priority = 0
    for codeword in codewords[tree]:
      if use_default_action_discount and codeword in dropped_codewords:
        continue

      table_entry={}
      table_entry["table_name"] = table_name
      table_entry["action"] = action_name
      table_entry["is_default_action"] = False
      # $MATCH_PRIORITY is per table, so this counter restarts with each tree.
      table_entry["priority"] = priority
      priority += 1

      # Tier 3: the classification table's key is one ternary field per
      # selected feature, not one combined codeword field. Slice the
      # combined codeword string into per-feature chunks, in the same
      # feature_intervals order generate_codewords used to build it, and
      # compute a ternary value/mask for each chunk separately. Each chunk's
      # field name is the bf_rt short name of the key
      # generate_P4_tables_and_apply's _classification_keys declares for that
      # feature (meta.code_<feature>, minus the `meta.` prefix).
      key_fields = {}
      bit_offset = 0
      for feature_name in feature_intervals:
        width = feature_code_length[feature_name]
        chunk = codeword[bit_offset:bit_offset+width]
        hex_value, hex_mask = get_ternary_value_and_mask(chunk)
        key_fields["code_"+feature_name.replace(" ","_").lower()] = {
            "value": hex_value, "mask": hex_mask}
        bit_offset += width
      table_entry["key_fields"] = key_fields
      # `class` is the literal parameter name of every classify_flow_codeword_*
      # action (see generate_P4_actions).
      table_entry["action_params"] = {
          "class": str(int(float((codewords[tree][codeword]))))}
      table_entries.append(table_entry)

    # The discounted leaves' class is not lost: it becomes this table's
    # control-plane-set default action. Without this record, every flow whose
    # codeword was discounted away would match no entry at all and go
    # unclassified.
    if default_class is not None:
      table_entries.append({
          "table_name": table_name,
          "action": action_name,
          "is_default_action": True,
          # bf_rt's set_default_with_<action> takes only data fields -- a
          # default action has no key and no $MATCH_PRIORITY.
          "priority": None,
          "key_fields": {},
          "action_params": {"class": str(int(float(default_class)))},
      })



  if verbose == True:
    # Show Generated Table Entries
    for entry in table_entries:
      print(entry)

  # Save table entries to JSON
  ensure_directory_exists(path_to_output)
  with open(path_to_output + output_filename, 'w') as output_file:
    output_file.write(json.dumps(table_entries))


def _resolve_disjoint_feature_plan(feature_intervals_app, feature_intervals_ddos):
  '''Decides, per feature name, whether App and DDoS can share ONE
  discretization table/codeword field or need independent, namespaced
  ones. Returns an ordered dict: resolved_name -> (raw_feature_name,
  intervals, models), where `models` is a subset of {"app","ddos"} naming
  which model(s) actually read this resolved feature's code_<resolved_name>
  field, and `raw_feature_name` is the underlying feature this resolved
  entry discretizes (needed because the RAW VALUE register/computation is
  always shared and keyed by raw_feature_name, never by resolved_name).

  Two models share a plain (non-namespaced) entry for a feature only when
  BOTH select it AND their interval lists are identical -- true by
  construction for joint-derived callers (both pass the literally same
  dict), and true only coincidentally for disjoint-derived callers (each
  model's intervals come from its own independently-trained tree).
  Features selected by only one model are never namespaced (nothing to
  disambiguate).
  '''
  # NOTE: a feature selected by only ONE model must never be namespaced --
  # there is nothing to disambiguate against, since the other model doesn't
  # discretize it at all. Namespacing only applies when BOTH models select
  # the same feature name with DIFFERING intervals (genuine disjoint
  # encoding).
  from collections import OrderedDict
  resolved = OrderedDict()
  seen = set()
  for feature in feature_intervals_app:
    seen.add(feature)
    in_ddos = feature in feature_intervals_ddos
    if in_ddos and feature_intervals_app[feature] == feature_intervals_ddos[feature]:
      resolved[feature] = (feature, feature_intervals_app[feature], {"app", "ddos"})
    elif in_ddos:
      # Both models select it, but with differing intervals.
      resolved["app_" + feature] = (feature, feature_intervals_app[feature], {"app"})
    else:
      # App-exclusive -- nothing to disambiguate.
      resolved[feature] = (feature, feature_intervals_app[feature], {"app"})
  for feature in feature_intervals_ddos:
    if feature in seen and feature_intervals_app.get(feature) == feature_intervals_ddos[feature]:
      continue  # already added as a shared entry above
    if feature in seen:
      # Both models select it, but with differing intervals.
      resolved["ddos_" + feature] = (feature, feature_intervals_ddos[feature], {"ddos"})
    else:
      # DDoS-exclusive -- nothing to disambiguate.
      resolved[feature] = (feature, feature_intervals_ddos[feature], {"ddos"})
  return resolved


def generate_P4_actions(feature_intervals, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos):
  """
      action <ACTION_NAME> (bit<<ACTION_CODE_LENGTH>> code) {
          meta.code_<FEATURE_NAME> = code;
      }
  """

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
  # generate_P4_tables_and_apply below), there is no need for a single
  # shared, tree-parameterized action at all: each tree's table binds to its
  # own dedicated action that unconditionally writes only its own field. No
  # `tree` parameter, no conditional, no Mux -- this generalizes uniformly to
  # any num_trees_app/num_trees_ddos, including 1. Validated against the real
  # TNA compiler in p4/tofino_spike/tna_m2_numtrees3_spike.p4.
  for task, n_trees, bits in (("app",  num_trees_app,  bit_per_classes_app),
                              ("ddos", num_trees_ddos, bit_per_classes_ddos)):
    if n_trees > 0:
      classification_action_template = ""
      for i in range(n_trees):
        classification_action_template += "\taction classify_flow_codeword_"+task+"_"+str(i)+"(bit<"+str(bits)+"> class){\n"
        classification_action_template += "\t\tmeta.class_tree_"+task+"_"+str(i)+" = class;\n"
        classification_action_template += "\t}\n\n"

      action_templates += classification_action_template

  for feature in feature_names:
    action_template = _ACTION_TEMPLATE
    action_template = action_template.replace("<ACTION_NAME>", "set_code_" + feature)
    action_template = action_template.replace("<ACTION_CODE_LENGTH>", str(codeword_bits_per_feature[feature]))
    action_template = action_template.replace("<FEATURE_NAME>", feature)
    action_templates += action_template

  return action_templates


def generate_P4_tables_and_apply(feature_names, num_trees_app, num_trees_ddos,
                                  match_type='ternary',
                                  feature_names_app=None, feature_names_ddos=None,
                                  raw_feature_names=None,
                                  feature_table_sizes=None,
                                  classification_table_sizes=None,
                                  config: "p4_gen_config.P4GenConfig" = None):
  """
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
  generate_P4_code/get_table_entries, where it decides which entries the
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
  matches that architecture: get_table_entries emits a real
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

  feature_names_app / feature_names_ddos: Task 3, optional. Restrict which
  of `feature_names` actually appear as a `meta.code_<resolved_name>` key
  field in the App / DDoS classification tables respectively -- a resolved
  name selected by only one model (e.g. a namespaced "ddos_<feature>" entry,
  or a feature only one model selected at all) must not appear as a key in
  the OTHER model's classification table. Each defaults to `feature_names`
  itself when omitted (None), exactly reproducing every pre-Task-3 caller's
  behavior -- both models keying on the identical, full feature set --
  byte-for-byte.

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

  feature_table_sizes / classification_table_sizes: follow-up to the
  2026-08-03 plan -- real, per-table `size = ` values, replacing the
  SIZE_FEATURE_TABLE=200 / SIZE_CLASSIFICATION_TABLE=400 literals below,
  which were fixed numbers disconnected from how many entries each table can
  actually receive (so no P4-generation-time entry-count optimization --
  generate_P4_code's use_default_action_discount, in particular -- could ever
  show up as reduced compiled TCAM/SRAM reservation).

  feature_table_sizes maps RESOLVED feature name -> entry count (one entry
  per interval; see get_table_entries' range-entry section).
  classification_table_sizes maps tree_id -> entry count, keyed exactly the
  way generate_P4_code's / get_table_entries' `codewords` dict is:
  0..num_trees_app-1 for the app trees,
  num_trees_app..num_trees_app+num_trees_ddos-1 for the ddos trees.

  Both are optional and both fall back per-key to the old literal, so any
  direct caller that doesn't pass them gets byte-identical output --
  generate_P4_code (which can always derive real counts) is the caller that
  actually supplies them.
  """

  if config is not None:
    match_type = config.match_type

  # Legacy fallbacks only: used per table whenever the caller supplied no real
  # entry count for it (see feature_table_sizes / classification_table_sizes).
  SIZE_FEATURE_TABLE = 200
  SIZE_CLASSIFICATION_TABLE = 400
  feature_table_sizes = feature_table_sizes or {}
  classification_table_sizes = classification_table_sizes or {}

  if match_type not in ('ternary', 'exact'):
    raise ValueError("match_type must be 'ternary' or 'exact', got {!r}".format(match_type))

  # Task 19: both are already-loaded strings (_TABLE_TEMPLATE /
  # _TABLE_CLASSIFICATION_EXACT_TEMPLATE, read once at import) -- no file I/O
  # happens here anymore.
  classification_table_template = (
      _TABLE_TEMPLATE if match_type == 'ternary'
      else _TABLE_CLASSIFICATION_EXACT_TEMPLATE
  )

  table_templates = ""
  apply_templates_tmp = "\n"
  apply_templates = ""

  feature_names = list(feature_names)
  # Task 3: each model's classification tables key on its OWN resolved
  # feature set -- defaulting to the full `feature_names` pool when not
  # given, so every pre-Task-3 caller (a single shared feature_names list)
  # is byte-identical.
  feature_names_app = list(feature_names_app) if feature_names_app is not None else feature_names
  feature_names_ddos = list(feature_names_ddos) if feature_names_ddos is not None else feature_names
  raw_feature_names = raw_feature_names or {}

  # Tier 3: the classification tables key on one field per selected feature
  # (meta.code_<feature>), not a single combined meta.codeword field. Task 8:
  # the match kind itself (ternary vs exact) is decided by match_type, not
  # hardcoded -- this is the only place "ternary"/"exact" actually lands in
  # the generated key text, since neither template file declares its own
  # <MATCH_TYPE> marker (match type is baked into <KEYS> here, not read from
  # the template).
  def _classification_keys(names):
    return "\n".join(
        "            meta.code_"+feature+" : "+match_type+";"
        for feature in names
    )

  classification_keys_by_task = {
      "app": _classification_keys(feature_names_app),
      "ddos": _classification_keys(feature_names_ddos),
  }
  tree_id_offset_by_task = {"app": 0, "ddos": num_trees_app}

  #Classification tables
  for task, n_trees in (("app", num_trees_app), ("ddos", num_trees_ddos)):
    if n_trees > 0:
      for i in range(n_trees):
        table_template = classification_table_template
        table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_"+task+"_"+str(i))
        table_template = table_template.replace("<KEYS>", classification_keys_by_task[task])
        table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_"+task+"_"+str(i)+";")
        size = classification_table_sizes.get(tree_id_offset_by_task[task] + i, SIZE_CLASSIFICATION_TABLE)
        table_template = table_template.replace("<SIZE>", str(size))
        table_templates += table_template
        apply_templates_tmp += "\t\t\tget_classification_tree_"+task+"_"+str(i)+".apply();\n"
      if task == "app":
        apply_templates_tmp += "\n"

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
    size = feature_table_sizes.get(feature, SIZE_FEATURE_TABLE)
    # table.p4 (the range path's now-retired template file) had 3 trailing
    # spaces on its closing "}" line that table_classification.p4 (now
    # _TABLE_TEMPLATE, shared by both paths) does not -- see the
    # _TABLE_TEMPLATE module comment. Restore them via this anchored replace
    # (unique: <SIZE> appears exactly once) before substituting <SIZE>
    # itself, so the range tables' emitted bytes are unchanged from every
    # caller before this task.
    table_template = table_template.replace("<SIZE>;\n    }", "<SIZE>;\n    }   ")
    table_template = table_template.replace("<SIZE>", str(size))
    table_templates += table_template
    apply_templates += "\t\t\ttable_"+str(feature_idx)+"_"+feature+".apply();\n"

    feature_idx += 1

  apply_templates += apply_templates_tmp

  return table_templates, apply_templates


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
  bit_per_classes = math.ceil(math.log2(num_classes)) or 1
  # The table emits exactly num_classes ** num_trees const entries (see the
  # product() loop below), so that IS its size -- the real logical entry
  # count, per reviews/p4_tofino_reference.md Sec 4.4. A `max(32, ...)` floor
  # used to be applied as a conservative guard, but it is not merely
  # redundant: for the real 1-tree/2-class DDoS config the key is a single
  # bit<1> field, so 2 entries is the entire key space and the real Tofino
  # compiler rejects anything larger --
  #   "warning: Shrinking table SwitchIngress.vote_ddos: with 1 match bits,
  #    can only have 2 entries"
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
      "\t\tmeta.classification_{task} = winner;\n"
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
  ).format(task=task, bits=bit_per_classes, keys=key_lines, entries=entries_block, size=size)

  apply_call = "\t\t\tvote_{}.apply();\n".format(task)

  return table_decl, apply_call


def generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos,
                      feature_intervals_app, feature_intervals_ddos,
                      output_dir=OUTPUT_PATH, output_filename='p4_code_RF_models.p4',
                      match_type='ternary',
                      use_default_action_discount=False,
                      selected_features_app=None, selected_features_ddos=None,
                      config: "p4_gen_config.P4GenConfig" = None):
  """match_type: passed straight through to
  generate_P4_tables_and_apply. 'ternary' (the default) switches
  classification tables to resources/table_classification.p4 with ternary
  keys; 'exact' switches only the classification tables to
  resources/table_classification_exact.p4 / `: exact;` keys (feature-range
  tables stay ternary/range either way).

  config: additive convenience. When given, `config.match_type`
  and `config.use_default_action_discount` take precedence over the
  individual `match_type` / `use_default_action_discount` keyword arguments
  above (which remain the source of truth when `config` is None).

  use_default_action_discount, selected_features_app, selected_features_ddos:
  the live path that produces the Planter-style default-action discount.
  When the flag is True (passed
  directly or via `config`), this function recomputes each model's codewords
  internally; every leaf carrying its tree's majority class then comes off
  that classification table's declared `size` here, and off its explicit
  entries in table_entries.json, where `get_table_entries` instead writes ONE
  `is_default_action` record per tree carrying that majority class. The
  generated P4 itself declares NO default action either way (see
  generate_P4_tables_and_apply's docstring for the real-compile evidence
  behind that, and p4/deploy_table_entries.py for the control-plane call that
  installs the default class at deploy time -- Planter's own architecture).
  When the flag is False -- the default -- every leaf stays an explicit entry
  and no is_default_action record is written at all. (Codewords themselves
  are computed whenever `selected_features_*` is supplied, discount or not,
  because the table sizing below needs them -- see TABLE SIZING.)

  Codewords are computed PER MODEL, against that model's OWN
  feature_intervals_app / feature_intervals_ddos: under genuine disjoint
  encoding the two models' intervals for a shared feature name can differ,
  so one combined call would produce wrong codewords for one side. The ddos
  trees are re-keyed by +num_trees_app to match the tree_id convention
  `generate_P4_tables_and_apply` reads codewords with.

  Doing that requires each model's ORIGINAL ORDERED training-feature-name
  list, which is why `selected_features_app`/`selected_features_ddos` exist
  and cannot be replaced by `feature_intervals_*.keys()`:
  `get_feature_thresholds` sorts alphabetically by feature name, whereas
  `export_text(tree, feature_names=...)` (via
  `get_tree_textual_representation`) requires `feature_names[i]` to be
  training column `i`. Raises ValueError when the flag is True and an active
  model's list is missing -- silently guessing an order would produce
  wrong-but-plausible codewords.

  TABLE SIZING: every table this function emits is sized from its REAL entry
  count, not the fixed SIZE_FEATURE_TABLE=200 / SIZE_CLASSIFICATION_TABLE=400
  literals generate_P4_tables_and_apply used to stamp everywhere -- range-matching
  feature tables from their interval count, classification tables from their
  codeword count (minus the entries use_default_action_discount folds into
  the default action). Because computing codewords needs the ordered
  training-feature-name lists, `selected_features_app`/`selected_features_ddos`
  are now consulted for SIZING too, not only for the discount; when a model's
  list is absent, that model's tables fall back to `estimator.get_n_leaves()`
  -- a real structural count that is always >= the codeword count, so a size
  is never underestimated. This means callers that pass neither list see
  their tables' `size = ` values change from the old literals to real
  (usually much smaller) numbers; nothing else about the generated text
  changes.

  KNOWN LIMITATION of that sizing: with match_type='exact' it still uses the
  ternary codeword/leaf count, NOT the real Cartesian-product-expanded
  exact-match entry count that enumerating each wildcarded codeword would
  produce (see evaluation.exact_match_resource_usage for the analytical
  accounting) -- enumerating those entries is deferred, separate work
  (reviews/todo.md's 2026-08-03 T0 update), so under 'exact' these sizes can
  be far too small; this is a pre-existing gap this sizing work does not
  close, not one it introduces.

  CAVEAT: with match_type='exact', the emitted P4 program is NOT yet
  end-to-end compilable/loadable on its own. get_table_entries (writes
  table_entries.json) is unchanged by this parameter and still emits
  ternary/wildcard-shaped ('*') entries for the same classification tables
  that this function just declared `: exact;` -- concrete, enumerated
  exact-match entries for those wildcarded codewords still need to be
  generated as follow-on work before 'exact' output can actually be loaded.

  feature_intervals_app / feature_intervals_ddos are each model's OWN,
  independently-derived feature_intervals dict (raw feature name -> interval
  list). A task with no active model passes {} for its side (e.g. an
  App-only or DDoS-only run passes {} for the other model's feature_intervals,
  exactly mirroring clf_app/clf_ddos=None for "no task"). A joint-encoded
  caller (both models sharing one discretization) passes the SAME dict for both
  parameters.

  Internally this resolves both dicts, once, via
  _resolve_disjoint_feature_plan: a feature both models select with
  IDENTICAL intervals shares ONE discretization table/field; a feature both
  models select with DIFFERING intervals gets two independent, namespaced
  (app_/ddos_ prefixed) discretization tables/fields -- while the
  underlying RAW feature-value register/computation always stays shared,
  since a raw counter value is model-independent regardless of who
  discretizes it (see _resolve_disjoint_feature_plan's own docstring for
  the full design).

  Also raises ValueError (F2) when any DISTINCT raw feature name across
  BOTH feature_intervals_app and feature_intervals_ddos has no
  FEATURE_REGISTER_CATALOG entry -- checked right after this function calls
  generate_P4_registers_and_apply (see that call site's own inline comment
  for the full rationale). Without this, such a feature would still get a
  declared, PHV-pinned, range-table-keyed `<feature>_val` metadata field
  further down in this function, just never written by anything -- reading
  0 for every packet, forever, silently. The message names every missing
  feature (by the caller-supplied spelling) and points at
  FEATURE_REGISTER_CATALOG (src/p4gen/feature_registers.py) as the fix."""

  if config is not None:
    match_type = config.match_type
    use_default_action_discount = config.use_default_action_discount

  # clf_app/clf_ddos may be None -- meaning "no task at all" (e.g. M1 is
  # DDoS-only: clf_app is None, clf_ddos is a trained model).
  num_trees_app = len(clf_app.estimators_) if clf_app is not None else 0
  num_trees_ddos = len(clf_ddos.estimators_) if clf_ddos is not None else 0

  # generate the definition of the bit containers that contain decisions of
  # each tree -- only meaningful (and only computed) when the corresponding
  # task actually has trees, so a missing task's num_class_* is never
  # consulted and never raises on log2(0)/log2(1 task with 1 class), etc.
  bit_per_classes_app = math.ceil(math.log2(num_class_app)) if num_trees_app > 0 else 0
  bit_per_classes_ddos = math.ceil(math.log2(num_class_ddos)) if num_trees_ddos > 0 else 0

  # Task 3: resolve App's and DDoS's independently-derived feature_intervals
  # into ONE plan describing, per resolved name, whether both models share
  # ONE discretization table/field for a feature or need independent,
  # namespaced ones. Computed once, consumed by every section below.
  resolved_plan = _resolve_disjoint_feature_plan(feature_intervals_app, feature_intervals_ddos)

  metadata_code = ""
  for task, n_trees, bits in (("app",  num_trees_app,  bit_per_classes_app),
                              ("ddos", num_trees_ddos, bit_per_classes_ddos)):
    if n_trees > 0:
      for i in range(n_trees):
        metadata_code += "\tbit<"+str(bits)+"> class_tree_"+task+"_"+str(i)+";\n"
        metadata_code += "\tbit<1> class_tree_"+task+"_"+str(i)+"_is_set;\n"
      # generate_voting_code (below) writes to meta.classification_<task>; the
      # TNA template no longer declares this field itself (it doesn't know
      # bit_per_classes_<task> ahead of time), so it must be declared here.
      metadata_code += "\tbit<"+str(bits)+"> classification_"+task+";\n"

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
  #
  # The pragma names the field the way the PHV logs do -- `ig_md.<raw>_val`,
  # after SwitchIngressParser's `out metadata_t ig_md` parameter, NOT the
  # `meta` name SwitchIngress binds the same struct to.
  phv_pragmas = ""
  raw_feature_intervals = {}  # raw_feature_name -> intervals (first-seen; only the KEYS feed generate_P4_registers_and_apply, which ignores values)
  for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items():
    if raw_feature_name not in raw_feature_intervals:
      raw_feature_intervals[raw_feature_name] = intervals
      value_field = raw_feature_name+"_val"
      metadata_code += "\tbit<16> "+value_field+";\n"
      phv_pragmas += '@pa_container_size("ingress", "ig_md.'+value_field+'", 16)\n'

  for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items():
    codeword_width = len(intervals) - 1
    metadata_code += "\tbit<"+str(codeword_width)+"> code_"+resolved_name+";\n"

  # Task 3, point 3: registers must be resolved against the DEDUPLICATED set
  # of RAW feature names, never the (possibly namespaced) resolved names --
  # a raw value register must not be generated twice just because two
  # resolved entries share it.
  registers_code, register_actions_code, feature_update_apply_code, resolved = (
      generate_P4_registers_and_apply(raw_feature_intervals))

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
  # case, as "missing" and raise a false-positive ValueError for it.
  # Lowercasing raw_feature_intervals' keys here too keeps this comparison
  # exact regardless of that upstream-normalisation invariant, while still
  # reporting each missing feature by its ORIGINAL (caller-supplied)
  # spelling in the error message below.
  missing = sorted(
      raw_name for raw_name in raw_feature_intervals
      if raw_name.lower() not in resolved
  )
  if missing:
    raise ValueError(
        "no register catalog entry for {} -- the generated program would declare "
        "<feature>_val, key a range table on it, and never write it, so it would "
        "read 0 for every packet. Add the feature to FEATURE_REGISTER_CATALOG "
        "(src/p4gen/feature_registers.py) or drop it from the feature set.".format(missing))

  # generate_P4_actions only needs resolved_name -> intervals: it writes
  # meta.code_<resolved_name>, one action per resolved entry, and is
  # already agnostic to whether a name is "raw" or "resolved"/namespaced --
  # no change needed to that function itself, only to what it's fed here.
  flat_resolved_intervals = {
      resolved_name: intervals
      for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items()
  }
  action_templates = generate_P4_actions(flat_resolved_intervals, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos)

  # Task 3, point 4: each model's classification tables must key on
  # code_<resolved_name> only for resolved entries that model actually
  # reads. Point 2/5: the range-matching tables (one per resolved entry,
  # always) still need each entry's raw feature name for the RAW value
  # field they key on.
  feature_names_app = [name for name, (_, _, models) in resolved_plan.items() if "app" in models]
  feature_names_ddos = [name for name, (_, _, models) in resolved_plan.items() if "ddos" in models]
  raw_feature_names = {name: raw for name, (raw, _, _) in resolved_plan.items()}

  # Follow-up: the discount's live path. Codewords are recomputed per model
  # against that model's OWN intervals (see this function's docstring), and
  # the ddos trees re-keyed by +num_trees_app -- the exact tree_id convention
  # generate_P4_tables_and_apply reads codewords with.
  #
  # Table-sizing follow-up: codewords are ALSO what makes an exact
  # classification-table size derivable, so they are now computed whenever
  # the caller supplied that model's ordered training-feature-name list --
  # not only when the discount is on. The discount's own requirement (it
  # cannot function without codewords) is unchanged: still a hard ValueError.
  codewords = {}
  if clf_app is not None:
    if selected_features_app is not None:
      tree_nodes_app = tree_nodes_for(clf_app, selected_features_app)
      paths_app = get_root_to_leaf_paths(tree_nodes_app)
      codewords.update(generate_codewords(paths_app, feature_intervals_app))
    elif use_default_action_discount:
      raise ValueError(
          "selected_features_app is required when use_default_action_discount=True "
          "and an App model is active -- generate_P4_code cannot recompute codewords "
          "without the exact ordered training-feature-name list")
  if clf_ddos is not None:
    if selected_features_ddos is not None:
      tree_nodes_ddos = tree_nodes_for(clf_ddos, selected_features_ddos)
      paths_ddos = get_root_to_leaf_paths(tree_nodes_ddos)
      codewords_ddos_0indexed = generate_codewords(paths_ddos, feature_intervals_ddos)
      codewords.update({tree_id + num_trees_app: tree_codewords
                        for tree_id, tree_codewords in codewords_ddos_0indexed.items()})
    elif use_default_action_discount:
      raise ValueError(
          "selected_features_ddos is required when use_default_action_discount=True "
          "and a DDoS model is active -- generate_P4_code cannot recompute codewords "
          "without the exact ordered training-feature-name list")
  if not codewords:
    codewords = None

  # Table-sizing follow-up: every generated table is sized from its REAL
  # entry count. Range-matching feature tables get exactly one entry per
  # interval; classification tables get one per distinct codeword (minus the
  # ones the discount turns into the table's default action), or -- when
  # codewords are not computable for that model -- the fitted tree's own leaf
  # count, a real structural number that can only ever be >= the codeword
  # count (two leaves whose paths round to the same codeword string collapse
  # into one dict entry in generate_codewords, never two), so it never
  # underestimates.
  feature_table_sizes = {
      resolved_name: max(1, len(intervals))
      for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items()
  }

  def _classification_table_size(tree_id, clf, estimator_index):
    tree_codewords = codewords.get(tree_id) if codewords else None
    if tree_codewords is None:
      return max(1, clf.estimators_[estimator_index].get_n_leaves())
    if use_default_action_discount:
      _, dropped_codewords = most_common_class_and_dropped_codewords(tree_codewords)
      return max(1, len(tree_codewords) - len(dropped_codewords))
    return max(1, len(tree_codewords))

  classification_table_sizes = {}
  for i in range(num_trees_app):
    classification_table_sizes[i] = _classification_table_size(i, clf_app, i)
  for i in range(num_trees_ddos):
    classification_table_sizes[num_trees_app + i] = _classification_table_size(
        num_trees_app + i, clf_ddos, i)

  table_templates, apply_templates = generate_P4_tables_and_apply(
      resolved_plan.keys(), num_trees_app, num_trees_ddos, match_type=match_type,
      feature_names_app=feature_names_app, feature_names_ddos=feature_names_ddos,
      raw_feature_names=raw_feature_names,
      feature_table_sizes=feature_table_sizes,
      classification_table_sizes=classification_table_sizes)

  # generate code to vote between the trees -- only for tasks that actually
  # have trees. generate_voting_code now returns (table_decl, apply_call);
  # the table declaration joins the other TABLES text, the apply call joins
  # the other APPLY text -- classification now happens via table application
  # like every other table in this generator, so there is no more separate
  # CLASSIFICATION content to build.
  for task, n_trees, n_classes in (("app",  num_trees_app,  num_class_app),
                                   ("ddos", num_trees_ddos, num_class_ddos)):
    if n_trees > 0:
      vote_table, vote_apply = generate_voting_code(n_trees, n_classes, task)
      table_templates += vote_table
      apply_templates += vote_apply

  # substitute the code in the template
  with open(PATH_P4_CODE_TEMPLATE_INPUT, 'r') as switch_template_file:
    switch_template = switch_template_file.read()
    # The marker's OWN line is consumed, so a feature-less program keeps the
    # template's original spacing exactly (and a program with features gets
    # its pragmas immediately above `struct metadata_t`, no stray blank line).
    switch_template = switch_template.replace('/* PHV_PRAGMAS */\n', phv_pragmas)
    switch_template = switch_template.replace('/* METADATA */', metadata_code)
    switch_template = switch_template.replace('/* REGISTERS */', registers_code)
    switch_template = switch_template.replace('/* REGISTER_ACTIONS */', register_actions_code)
    switch_template = switch_template.replace('/* ACTIONS */', action_templates)
    switch_template = switch_template.replace('/* TABLES */', table_templates)
    switch_template = switch_template.replace('/* FEATURE_UPDATE_APPLY */', feature_update_apply_code)
    switch_template = switch_template.replace('/* APPLY */', apply_templates)
    switch_template = switch_template.replace('/* CLASSIFICATION */', "")

  ensure_directory_exists(output_dir)
  written_path = output_dir + output_filename
  with open(written_path, 'w') as switch_template_file:
    switch_template_file.write(switch_template)

  return written_path


# ---------------------------------------------------------------------------
# TNA (Tofino Native Architecture) register + apply-block generation.
#
# Unlike the rest of this file (which emits v1model P4 for
# resources/p4_template.p4 via generate_P4_code above), this generator
# targets the TNA architecture validated in
# p4/tofino_spike/tna_m1_flows_iat_spike.p4 (compiled successfully this
# session with the real Tofino compiler: `p4c -b tofino -a tna`, 0 errors).
# It is called from generate_P4_code() to resolve selected features
# to the TNA registers, RegisterActions, and per-packet update logic needed for
# Milestone 1's flow- and IAT-tracking design.
# ---------------------------------------------------------------------------

# Tofino stateful-ALU limit: the number of distinct RegisterAction
# `.execute()` call sites that may touch a single register per packet.
MAX_REGISTER_TOUCHES = 4

# Symbolic RegisterAction body kinds -> exact atomic read-modify-write P4
# body, transcribed verbatim from the compiled spike
# (p4/tofino_spike/tna_m1_flows_iat_spike.p4). Do not reword/"clean up"
# these -- they are the compiler-validated ground truth.
_REGISTER_ACTION_BODIES = {
  "iat_delta": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\trv = meta.now_pseudo_us - value;\n"
      "\t\t\tvalue = meta.now_pseudo_us;\n"
      "\t\t}}\n"
  ),
  "running_max_iat": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.current_iat > value) {{\n"
      "\t\t\t\tvalue = meta.current_iat;\n"
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
  # M2: mean/EWMA folding via Tofino's MathUnit<> hardware primitive.
  # Transcribed verbatim from p4/tofino_spike/tna_m2_mean_spike.p4's
  # flow_iat_mean_ewma_action (the working "take 3" design -- see that
  # spike's header comment for why the two earlier, simpler-looking designs
  # both failed against the real Tofino compiler). Computes
  # new_mean = (old_mean + current_iat) / 2 -- an alpha=0.5 EWMA -- as one
  # division of a sum, in a single register touch. References
  # `{name}_halve_unit`, a MathUnit<> instance that _EXTRA_ACTION_DECLARATIONS
  # (below) causes to be declared immediately before this RegisterAction.
  "mathunit_ewma": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tvalue = {name}_halve_unit.execute(value + meta.current_iat);\n"
      "\t\t\trv = value;\n"
      "\t\t}}\n"
  ),
  # Task 8: the packet-length equivalent of mathunit_ewma, for the three
  # packet-length-mean features (fwd_packet_length_mean,
  # bwd_packet_length_mean, packet_length_mean). Folds hdr.ipv4.total_len
  # (a header field, not meta.current_iat) into the same MathUnit<>-based
  # alpha=0.5 EWMA. A separate body kind rather than parameterising
  # mathunit_ewma's operand, per task-8-brief.md: _REGISTER_ACTION_BODIES is
  # a table of compiler-validated literals, one entry per validated program
  # text. Transcribed verbatim from the compile of
  # p4/tofino_spike/tna_m3_packet_length_mean_spike.p4 (0 errors) -- the cast
  # `(bit<{width}>)hdr.ipv4.total_len` compiled as-is, no metadata
  # intermediary needed.
  "mathunit_ewma_packet_length": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tvalue = {name}_halve_unit.execute(value + (bit<{width}>)hdr.ipv4.total_len);\n"
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
  # needed. Bodies transcribed verbatim from that compile.
  "running_min_iat": (
      "\t\tvoid apply(inout bit<{width}> value, out bit<{width}> rv) {{\n"
      "\t\t\tif (meta.current_iat < value) {{\n"
      "\t\t\t\tvalue = meta.current_iat;\n"
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

# Register-action body kinds that require an extra hardware-primitive
# declaration (e.g. a MathUnit<> instance) emitted immediately before their
# RegisterAction block. Maps body kind -> declaration template, parallel in
# spirit to _REGISTER_ACTION_BODIES above.
_EXTRA_ACTION_DECLARATIONS = {
    "mathunit_ewma": "\tMathUnit<bit<{width}>>(MathOp_t.MUL, 1, 2) {name}_halve_unit;\n",
    "mathunit_ewma_packet_length": "\tMathUnit<bit<{width}>>(MathOp_t.MUL, 1, 2) {name}_halve_unit;\n",
}

# Body kinds whose backing register must NOT start at 0. A running minimum
# folded into a zero-initialised register stays 0 forever (min(0, x) == 0),
# so these start at INFINITE (65535 -- the same sentinel
# get_feature_intervals_from_thresholds uses) via _register_declaration's
# optional initial_value, using TNA's Register<T,I> two-argument constructor.
_REGISTER_INITIAL_VALUES = {
    "running_min_iat": INFINITE,
    "running_min_packet_length": INFINITE,
}


def _register_declaration(name, width, initial_value=None):
  if initial_value is None:
    return "\tRegister<bit<{width}>, bit<32>>(MAX_NUM_FLOWS) {name}_reg;\n".format(
        width=width, name=name)
  return "\tRegister<bit<{width}>, bit<32>>(MAX_NUM_FLOWS, {init}) {name}_reg;\n".format(
      width=width, name=name, init=initial_value)


def _register_action_declaration(name, width, body_kind):
  # `name=name` is only consumed by body kinds that reference an
  # extra-declaration identifier of their own (e.g. "mathunit_ewma"'s
  # `{name}_halve_unit`); body kinds that don't reference `{name}` simply
  # ignore the extra format() kwarg.
  body = _REGISTER_ACTION_BODIES[body_kind].format(width=width, name=name)
  return (
      "\tRegisterAction<bit<{width}>, bit<32>, bit<{width}>>({name}_reg) {name}_action = {{\n"
      "{body}"
      "\t}};\n"
  ).format(width=width, name=name, body=body)


def generate_P4_registers_and_apply(feature_intervals, catalog=None):
  """
  Resolve a selected feature set to the TNA registers, RegisterActions, and
  per-packet update logic needed for Milestone 1's flow- and IAT-tracking
  design (ground truth: p4/tofino_spike/tna_m1_flows_iat_spike.p4, compiled
  with the real Tofino p4c). Called from generate_P4_code() to resolve
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
  (below) and raises ValueError if anything didn't resolve, because letting
  an uncatalogued feature's <feature>_val field compile silently would read
  0 for every packet forever (see generate_P4_code's own F2 comment).

  Returns a 4-tuple of P4 source strings plus a resolution set, one source
  string per marker payload a future TNA template will substitute this
  function's output into, and `resolved` for callers (generate_P4_code) to
  detect what did NOT resolve:
    (registers_code, register_actions_code, feature_update_apply_code, resolved)

    - registers_code: `Register<bit<W>, bit<32>>(MAX_NUM_FLOWS) <name>_reg;`
      declarations (TNA syntax), plus the ONE `Hash<>` instance the
      apply-block's hash action needs (`flow_hash_calc`) and the fixed
      `flow_forward_srcaddr_reg` register used for flow-direction
      bookkeeping (see below) -- both always emitted, unconditionally,
      whenever the resolved feature set is non-empty; neither is
      catalog-driven or routed through the touch-count guard.
    - register_actions_code: the plain calc_flow_hash action (and
      calc_timestamp, only emitted when an included register actually needs
      meta.now_pseudo_us), plus one `RegisterAction<...> <name>_action =
      {...};` block per resolved register, using the exact bodies from
      _REGISTER_ACTION_BODIES. Body kinds listed in
      _EXTRA_ACTION_DECLARATIONS (currently "mathunit_ewma" and
      "mathunit_ewma_packet_length") get one
      extra hardware-primitive declaration line (e.g.
      `MathUnit<bit<W>>(MathOp_t.MUL, 1, 2) <name>_halve_unit;`) emitted
      immediately before that register's RegisterAction block. Flow
      direction/first-seen bookkeeping (validated against the real Tofino
      compiler this session: p4/tofino_spike/tna_m2_symmetric_hash_spike.p4,
      and the real M2 program with this swap,
      p4/tofino_spike/tna_m2_real_with_symhash.p4, both 0 errors) is now a
      SINGLE `flow_orientation_action` RegisterAction on
      `flow_forward_srcaddr_reg`: since `calc_flow_hash` is symmetric
      (XOR-based -- `src_addr ^ dst_addr`, `src_port ^ dst_port` -- so both
      directions of a flow hash identically; no min/max comparison needed,
      since TNA's Gateway hardware can only compare a field against a
      compile-time constant, not two runtime fields against each other,
      confirmed as a real compile restriction this session), the register
      simply stores whichever packet's own srcAddr was seen FIRST for that
      flow index; a later packet is "fwd" iff its own srcAddr matches the
      stored value. One touch per packet, not two. CAVEAT: XOR-based
      symmetry has a different (likely worse) collision profile than true
      min/max canonical ordering -- not measured, flagged not fixed,
      consistent with this project's treatment of every other
      resource-oracle approximation (e.g. the IAT ns->us rescale).
    - feature_update_apply_code: the per-packet apply-block snippet: the
      hash-calc and (conditionally) timestamp calls, the single
      flow_orientation_action touch, then each *distinct register name*'s
      `.execute()` call -- gated
      inside `if (meta.fwd == 1) { ... }` for features whose catalog entry
      says `"gated_by": "fwd"`, or inside a sibling `if (meta.fwd == 0) { ... }`
      block for features whose catalog entry says `"gated_by": "bwd"` -- and
      assignment of each "value" register's
      result into `meta.<feature_lower>_val` (e.g. `meta.flow_iat_max_val`,
      matching the spike's `metadata_t` field naming; a "dependency"
      register's result is not itself a feature value, so it is assigned
      into the shared `meta.current_iat` scratch field instead, exactly
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
      flow_iat_mean both depend on "flow_last_arrival_time"), only the
      first-encountered feature (in feature_intervals iteration order)
      gets an `.execute()` call site for that register; every later
      feature that references the same name reuses the value the first
      call already produced (e.g. the shared `meta.current_iat`) instead
      of re-invoking `.execute()` on it. This models the real hardware
      constraint that a given RegisterAction's call site represents one
      physical register touch meant to fire (at most) once per packet --
      touching it twice for the same packet would be both semantically
      wrong (the second call would observe the state the first call just
      wrote) and would incorrectly inflate the touch-count guard below if
      it were derived from call-site counts. The touch-count guard itself
      (`register_touch_count`, checked further below, in `_note_touch`) IS
      aligned with this same deduplicated model: a register's counted
      touches equal its real `.execute()` call-site count, not one
      increment per referencing feature. `flow_forward_srcaddr_reg` (the
      fixed flow-direction bookkeeping register, always exactly 1 real
      touch per packet via `flow_orientation_action`) is NOT routed through
      this catalog-driven touch-count machinery at all -- like
      `flow_hash_calc`, it is a fixed, always-emitted register outside
      `register_order`/`_note_touch`'s bookkeeping, not a candidate for the
      guard below. Every catalog-driven register IS counted once, the
      first time any feature references it, regardless of how many further
      features also list it as a dependency -- e.g. flow_last_arrival_time
      reports exactly 1 touch for M2's feature set (flow_iat_max +
      flow_iat_mean sharing it), matching the single real `.execute()` call
      site _execute_lines actually emits for it.
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
  invocation), if resolving `catalog` against `feature_intervals` would
  require more than MAX_REGISTER_TOUCHES distinct, real (deduplicated)
  RegisterAction `.execute()` call sites against any single register --
  i.e. the same count described above, not a raw per-feature-reference
  tally. Under the current catalog, every resolved register gets exactly
  one such call site (see _note_touch below), so this guard is presently
  dormant -- it cannot actually fire against any real catalog configuration
  today -- but it is retained as a general safety net for any future
  register (catalog-driven or hardcoded) that might legitimately need more
  than one touch.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # feature_intervals keys are already lowercase, underscore-joined (see
  # get_nodes()'s normalise_feature_name()), so .lower() here is normally a
  # no-op; kept as a defensive normalisation for callers that pass a
  # feature_intervals dict built by hand rather than via get_nodes.
  # Preserves feature_intervals' iteration order.
  matched_features = [f for f in (name.lower() for name in feature_intervals.keys()) if f in catalog]

  # NOTE: there used to be a guard here raising ValueError if "flow_iat_mean"
  # was selected without "flow_iat_max", on the theory that flow_iat_mean's
  # shared "flow_last_arrival_time" dependency register would otherwise never
  # be .execute()d. That was never true of this generator: each catalog
  # entry (flow_iat_mean's included) lists its OWN dependency register in its
  # own "registers" list, so resolving flow_iat_mean alone still emits
  # flow_last_arrival_time_action.execute(meta.flow_hash) before
  # meta.flow_iat_mean_val = ... -- see _execute_lines below, which walks
  # feature_registers[feature] (the feature's own registers list) regardless
  # of which other features are selected. The guard was deleted (Task 9) once
  # this was proven by
  # test_resolving_flow_iat_mean_alone_auto_executes_its_dependency in
  # tests/test_feature_registers.py; see that test for the exact ordering
  # assertion.

  if not matched_features:
    return "", "", "", set()

  # ---- Resolve the deduplicated, ordered register set + touch counts ----

  register_order = []        # ordered list of register names (first-seen)
  register_info = {}         # name -> {"width":, "body":}
  register_touch_count = {}  # name -> number of .execute() call sites

  def _note_touch(name, width=None, body=None, count=1):
    # A register's touch count must reflect its REAL, deduplicated
    # .execute() call-site count -- exactly what _execute_lines (below)
    # actually emits, not one increment per (feature, register-list-entry)
    # pair. `register_info` already tracks "have we seen this register
    # name before" for register_order/declarations; reuse that same
    # first-seen signal here: the first time a register name is noted, it
    # gets its real touch count (`count`); every later call for an
    # already-seen name (e.g. a second feature that also lists the same
    # dependency register) adds nothing, because _execute_lines will reuse
    # that first call's already-produced value rather than emitting another
    # .execute() line for it. Every current call site below passes the
    # default count=1 -- the old hardcoded `flows` register's count=2 call
    # site was removed along with that register -- so register_touch_count
    # is currently always exactly 1 per register and the guard just below
    # can't fire under any real catalog configuration today. The `count`
    # parameter and the guard are kept anyway as a general mechanism: a
    # future catalog entry, or a future fixed/hardcoded register, could
    # still legitimately need more than 1 real touch.
    if name not in register_info:
      register_order.append(name)
      register_info[name] = {"width": width, "body": body}
      register_touch_count[name] = register_touch_count.get(name, 0) + count

  feature_registers = {}  # feature -> ordered list of its catalog register dicts
  needs_timestamp = False

  for feature in matched_features:
    entry = catalog[feature]
    gated_by = entry.get("gated_by")
    if gated_by not in (None, "fwd", "bwd"):
      raise RuntimeError(
          "Feature '{feature}' has unsupported gated_by={gated_by!r}; "
          "only None, 'fwd', and 'bwd' are implemented.".format(feature=feature, gated_by=gated_by)
      )

    regs = []
    for reg in entry["registers"]:
      _note_touch(reg["name"], width=reg["width"], body=reg["body"])
      regs.append(reg)
      if reg["body"] == "iat_delta":
        needs_timestamp = True
    feature_registers[feature] = regs

  # ---- Cross-gate register-sharing hazard guard ----
  #
  # _execute_lines() (below) is called three times -- ungated, then
  # fwd-gated, then bwd-gated -- sharing one `already_executed_registers`
  # set, so whichever call reaches a register name FIRST is the one that
  # actually emits its .execute() call site; every later call for that same
  # name is a no-op that assumes the value was already produced. If a
  # register were first executed inside a gated block (say, only for
  # meta.fwd == 1 packets) and then reused by a feature in a DIFFERENT gate
  # class (say, an ungated feature, or a meta.fwd == 0 feature), packets
  # that skip the first block would read a garbage/unset register value.
  # Catch that at generation time instead of emitting silently-wrong P4.
  gate_of_register = {}
  for feature in matched_features:
    gate = catalog[feature].get("gated_by")
    for reg in catalog[feature]["registers"]:
      previous = gate_of_register.setdefault(reg["name"], gate)
      if previous != gate:
        raise ValueError(
            "register {!r} is shared by features in different gate classes "
            "({!r} and {!r}); it would be .execute()d inside one gated block and "
            "read as garbage from the other. Promote it to an ungated register "
            "or give each gate class its own.".format(reg["name"], previous, gate))

  # ---- Touch-count guard (checked before emitting anything) ----

  for name, count in register_touch_count.items():
    if count > MAX_REGISTER_TOUCHES:
      raise RuntimeError(
          "Register '{name}' would require {count} RegisterAction .execute() "
          "call sites, exceeding the {limit}-touch Tofino stateful-ALU limit "
          "per register.".format(name=name, count=count, limit=MAX_REGISTER_TOUCHES)
      )

  # ---- /* REGISTERS */ ----

  registers_code = (
      "\tHash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc;\n"
      "\tRegister<bit<32>, bit<32>>(MAX_NUM_FLOWS) flow_forward_srcaddr_reg;\n"
  )
  for name in register_order:
    initial_value = _REGISTER_INITIAL_VALUES.get(register_info[name]["body"])
    registers_code += _register_declaration(name, register_info[name]["width"], initial_value=initial_value)

  # ---- /* REGISTER_ACTIONS */ ----

  register_actions_code = (
      "\taction calc_flow_hash() {\n"
      "\t\tmeta.flow_hash = flow_hash_calc.get({\n"
      "\t\t\thdr.ipv4.src_addr ^ hdr.ipv4.dst_addr,\n"
      "\t\t\thdr.ipv4.protocol,\n"
      "\t\t\thdr.tcp.src_port ^ hdr.tcp.dst_port\n"
      "\t\t}) & 0xFFF;\n"
      "\t}\n"
  )

  if needs_timestamp:
    register_actions_code += (
        "\n"
        "\taction calc_timestamp() {\n"
        "\t\tmeta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);\n"
        "\t}\n"
    )

  # Symmetric-hash flow bookkeeping (Part G.6 in reviews/t11_tofino_port_and_env.md,
  # wired in after validation against the real Tofino compiler this session:
  # p4/tofino_spike/tna_m2_symmetric_hash_spike.p4, and the real M2 program with
  # this swap, p4/tofino_spike/tna_m2_real_with_symhash.p4, both 0 errors). Since
  # the hash is now symmetric (XOR-based, both directions of a flow hash
  # identically), a single register stores the first-seen packet's own
  # srcAddr; later packets are "fwd" iff their own srcAddr matches it.
  # CAVEAT: XOR-based symmetry has a different (likely worse) collision
  # profile than true min/max canonical ordering -- not measured, flagged
  # not fixed, consistent with this project's treatment of every other
  # resource-oracle approximation (e.g. the IAT ns->us rescale).
  register_actions_code += (
      "\n"
      "\tRegisterAction<bit<32>, bit<32>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {\n"
      "\t\tvoid apply(inout bit<32> value, out bit<1> rv) {\n"
      "\t\t\tif (value == 0) {\n"
      "\t\t\t\tvalue = hdr.ipv4.src_addr;\n"
      "\t\t\t}\n"
      "\t\t\trv = (value == hdr.ipv4.src_addr) ? 1w1 : 1w0;\n"
      "\t\t}\n"
      "\t};\n"
  )

  for name in register_order:
    info = register_info[name]
    extra_declaration = _EXTRA_ACTION_DECLARATIONS.get(info["body"])
    if extra_declaration is not None:
      register_actions_code += "\n" + extra_declaration.format(width=info["width"], name=name)
    register_actions_code += "\n" + _register_action_declaration(name, info["width"], info["body"])

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
  # flow_iat_max and flow_iat_mean) must still be .execute()'d exactly once
  # per packet. Whichever feature reaches that register name first (in
  # matched_features order) emits its call site and produces the value
  # (meta.current_iat, for a "dependency" register); every later feature
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
        target = "meta.current_iat" if reg["role"] == "dependency" else "meta." + feature + "_val"
        lines.append("{indent}{target} = {name}_action.execute(meta.flow_hash);".format(
            indent=indent, target=target, name=name))
    return lines

  ungated_features = [f for f in matched_features if catalog[f].get("gated_by") is None]
  fwd_gated_features = [f for f in matched_features if catalog[f].get("gated_by") == "fwd"]
  bwd_gated_features = [f for f in matched_features if catalog[f].get("gated_by") == "bwd"]

  apply_lines += _execute_lines(ungated_features, "\t\t\t")

  if fwd_gated_features:
    apply_lines.append("")
    apply_lines.append("\t\t\tif (meta.fwd == 1) {")
    apply_lines += _execute_lines(fwd_gated_features, "\t\t\t\t")
    apply_lines.append("\t\t\t}")

  if bwd_gated_features:
    apply_lines.append("")
    apply_lines.append("\t\t\tif (meta.fwd == 0) {")
    apply_lines += _execute_lines(bwd_gated_features, "\t\t\t\t")
    apply_lines.append("\t\t\t}")

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