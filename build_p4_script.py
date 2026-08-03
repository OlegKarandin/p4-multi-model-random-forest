import os
import math
from sklearn.tree import export_text
import csv
import json
from collections import Counter
from itertools import product
from statistics import mode

from feature_registers import FEATURE_REGISTER_CATALOG
import p4_gen_config

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
            tree_obj.threshold[i] = int(round(tree_obj.threshold[i]))
  return clf


def get_tree_textual_representation(clf, feature_names, verbose=False):
  tree_textual_representation = {}

  for idx,tree in enumerate(clf.estimators_):
    tree_textual_representation[idx] = export_text(tree, feature_names=feature_names)

  if verbose == True:
    for tree in tree_textual_representation:
      print("Tree ", tree)
      print(tree_textual_representation[tree])

  return tree_textual_representation


def get_nodes(tree_text, tree_idx = -1):
  '''
   Inputs: Tree textual representation generated with export_text(tree_classifier, feature_names)
   Outputs: Dictionary containing the information of the different tree nodes (leaf or internal)
   '''

  nodes = {}
  # Store each of the lines of the tree textual representation in a List
  tree_lines = tree_text.strip().split('\n')

  ##### TABLE FEATURES #####
  node_id                  = 0
  previous_depth           = 1
  previous_node_was_leaf   = False
  #father_node_id           = -1
  parent_stack = [-1]  # stack of parent node IDs at each depth
  ##########################

  for line in tree_lines:
    # Calculate Current Node's Depth based on number of "|" ocurrences in the line
    depth = line.count("|")

    # Trim stack to current depth
    while len(parent_stack) > depth:
        parent_stack.pop()

    father_node_id = parent_stack[-1] if parent_stack else -1

    # A) Node is Internal (Not Leaf)
    if "class" in line:

      #New Leaf Node
      nodes[node_id]={"node": node_id,
                      "tree": tree_idx,
                      "father_node": father_node_id,
                      "class": line.split("class: ")[-1],
                      "depth": depth,
                      "action_name": "classify_flow",
                      "is_leaf": True}

      ########### Assign Right Child ################
      if previous_node_was_leaf: #If previous node also was leaf
        # Iterate the already defined nodes.
        # We look for a node from the immediate upper layer of the current node, which is not a leaf, and has not an already assigned right child
        for aux_node_id in nodes:
          if (nodes[aux_node_id]["depth"] == depth - 1) and not nodes[aux_node_id]["is_leaf"] and nodes[aux_node_id]["right_child"]==None:
            nodes[aux_node_id]["right_child"] = node_id
            nodes[node_id]["father_node"] = aux_node_id
      ################################################

      # Update Parameters
      previous_node_was_leaf = True
      previous_depth = depth
      node_id += 1
      #father_node_id += 1

    # B) Node is Child
    else:

      feature_name  = line.replace("|---","").replace("|   ","").split("<=")[0].strip().replace(" ","_")
      threshold     = line.replace("|---","").replace("|   ","").split("<=")[-1].strip()

      if "<=" in line:
        # New Internal Node
        nodes[node_id]={"node": node_id,
                        "tree": tree_idx,
                        "father_node": father_node_id,
                        "feature": feature_name,
                        "depth": depth,
                        "action_name": "classify_"+feature_name.lower(),
                        "threshold": int(float(threshold)),
                        "left_child": node_id + 1,
                        "right_child": None,
                        "is_leaf": False}

        ##### Assign Right Child ######
        if depth < previous_depth or previous_node_was_leaf:
        # Iterate the already defined nodes.
        # We look for a node from the immediate upper layer of the current node, which is not a leaf, and has not an already assigned right child
          for aux_node_id in nodes:
            if (nodes[aux_node_id]["depth"] == depth - 1) and not nodes[aux_node_id]["is_leaf"] and nodes[aux_node_id]["right_child"]==None:
              nodes[aux_node_id]["right_child"] = node_id
              nodes[node_id]["father_node"] = aux_node_id
        ###############################

        # Update Parameters
        previous_node_was_leaf = False
        previous_depth = depth

        parent_stack.append(node_id)
        node_id += 1
        #father_node_id += 1

  return nodes


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
    try:
      feature_thresholds.append((node["feature"],
                            node["threshold"]))
    except:
      pass
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

      # avoid creating a [0, 0] interval (remember that all features are positive)
      if threshold == 0:
        continue

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


def get_feature_intervals(model, selected_features):
  trees = get_tree_textual_representation(model, selected_features)

  tree_nodes = {}
  for tree in trees:
    tree_nodes[tree] = get_nodes(trees[tree], tree)

  feature_thresholds = get_feature_thresholds(tree_nodes)
  feature_intervals = get_feature_intervals_from_thresholds(feature_thresholds)

  return feature_intervals


def feature_intervals_to_csv(feature_intervals, path_to_output=INTERMEDIATE, output_filename = "feature_intervals.csv"):
  rows = []

  for feature_name in feature_intervals:
    intervals = feature_intervals[feature_name]
    rows.append([feature_name])

    for idx,interval in enumerate(intervals[::-1]):
      zeros = len(intervals)-1-idx
      ones = len(intervals)-1-zeros
      row = []
      row.append(interval)

      for i in range(zeros):
        row.append('0')
      for i in range(ones):
        row.append('1')
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

  Example Output: { "class": "1.0", "path": [ {"node_id": 2, "feature": "Flow_IAT_Max", "threshold": 504078, "condition": "<="},
                                              {"node_id": 1, "feature": "Bwd_Packet_Length_Max", "threshold": 12, "condition": "<="},
                                              {"node_id": 0, "feature": "Bwd_Packet_Length_Max", "threshold": 3513, "condition": "<="} ] }
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
          try: # Only add * if feature is used in the tree
            for i in range(len(feature_intervals[feature])-1):
              codeword.append('*')
          except:
            pass

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

      code_ = ''.join(codeword)
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


def get_ternary_match(codeword):
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

  return f"0x{hex_value}&&&0x{hex_mask}"


def get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset=None, path_to_output=OUTPUT_PATH, output_filename="table_entries.json", verbose=False, use_default_action_discount=False):
  '''
  Inputs: feature_intervals [dict]: Dictionary where each key is a tree_id. The values are a list of feature intervals for each tree.
          codewords [dict]: Dictionary where each key is a tree_id. The values are a list of dictionaries.
                            Each key in the dictionary is a codeword corresponding to a leaf node.
                            Each value is the class label associated with that codeword (i.e.: to the leaf node).
          use_default_action_discount [bool]: Task 7 -- Planter-style default-action discount. When
                            True, every one of each tree's leaves sharing the majority class (see
                            most_common_class_and_dropped_codewords) is omitted from the written table
                            entries for that tree, since they become the classification table's
                            default_action instead (wired in generate_P4_tables_and_apply). False (the
                            default) preserves today's exact output.

  Outputs: This function writes a JSON file that includes two lists of table entries:
              Feature codeword bits table entries
              Codeword-to-LeafNode matching table entries

          Table entry:
          {
            "table_name": _,
            "action": _,
            "key": _,
            "action_params": _,
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
    "Flow_IAT_Max"). A resolved/namespaced key like "app_flow_iat_max"
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
  # Obtain features involved in RF classification
  features_involved = []
  for tree in paths_leaf_nodes_per_tree:
    for leaf_node in paths_leaf_nodes_per_tree[tree]:
      for step in paths_leaf_nodes_per_tree[tree][leaf_node]["path"]:
        features_involved.append(step["feature"])

  features_involved = sorted(set(features_involved))

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

      zeros = len(intervals)-1-idx
      ones = len(intervals)-1-zeros
      code = []
      for i in range(zeros):
        code.append('0')
      for i in range(ones):
        code.append('1')

      minimum = str(interval[0])
      maximum = str(interval[1])

      table_entry["key"] = [minimum+".."+maximum]
      table_entry["action_params"] = [str(int("".join(code),2))]

      table_entries.append(table_entry)

    feature_idx +=1


  # 2. Table entries for getting each tree's classification based on generated codeword
  for tree_idx,tree in enumerate(codewords):
    # Task 7: every leaf sharing this tree's majority class is excluded
    # from this tree's written entries -- they become the table's
    # default_action instead (see generate_P4_tables_and_apply). Computed
    # once per tree, up front, so every other leaf is still written as
    # before.
    default_class, dropped_codewords = None, set()
    if use_default_action_discount and len(codewords[tree]) > 0:
      default_class, dropped_list = most_common_class_and_dropped_codewords(codewords[tree])
      dropped_codewords = set(dropped_list)

    for codeword in codewords[tree]:
      if use_default_action_discount and codeword in dropped_codewords:
        continue

      table_entry={}

      # Task M2-B2: `tree` is no longer a runtime action parameter -- each
      # tree has its own dedicated action (see generate_P4_actions), so
      # action_params carries only the class value.
      if offset==None:
        #One model encoding
        table_entry["table_name"] = "get_classification_tree_"+str(tree_idx)
        table_entry["action"] = "classify_flow_codeword_"+str(tree_idx)
        table_entry["action_params"] = [str(int(float((codewords[tree][codeword]))))]
      else:
        #Multiple models encoding
        if tree_idx < offset:
          table_entry["table_name"] = "get_classification_tree_app_"+str(tree_idx)
          table_entry["action"] = "classify_flow_codeword_app_"+str(tree_idx)
          table_entry["action_params"] = [str(int(float((codewords[tree][codeword]))))]
        else:
          table_entry["table_name"] = "get_classification_tree_ddos_"+str(tree_idx-offset)
          table_entry["action"] = "classify_flow_codeword_ddos_"+str(tree_idx-offset)
          table_entry["action_params"] = [str(int(float((codewords[tree][codeword]))))]

      # Tier 3: the classification table's key is one ternary field per
      # selected feature, not one combined codeword field. Slice the
      # combined codeword string into per-feature chunks, in the same
      # feature_intervals order generate_codewords used to build it, and
      # compute a ternary match for each chunk separately.
      key = []
      bit_offset = 0
      for feature_name in feature_intervals:
        width = feature_code_length[feature_name]
        chunk = codeword[bit_offset:bit_offset+width]
        key.append(get_ternary_match(chunk))
        bit_offset += width
      table_entry["key"] = key
      table_entries.append(table_entry)



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
    try:
      codeword_bits_per_feature[feature]=len(feature_intervals[feature])-1
    except:
      pass


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
  if num_trees_app > 0:
    classification_action_template_app = ""
    for i in range(num_trees_app):
      classification_action_template_app += "\taction classify_flow_codeword_app_"+str(i)+"(bit<"+str(bit_per_classes_app)+"> class){\n"
      classification_action_template_app += "\t\tmeta.class_tree_app_"+str(i)+" = class;\n"
      classification_action_template_app += "\t}\n\n"

    action_templates += classification_action_template_app

  #Classification actions for DDOS detection problem
  if num_trees_ddos > 0:
    classification_action_template_ddos = ""
    for i in range(num_trees_ddos):
      classification_action_template_ddos += "\taction classify_flow_codeword_ddos_"+str(i)+"(bit<"+str(bit_per_classes_ddos)+"> class){\n"
      classification_action_template_ddos += "\t\tmeta.class_tree_ddos_"+str(i)+" = class;\n"
      classification_action_template_ddos += "\t}\n\n"

    action_templates += classification_action_template_ddos


  for feature in feature_names:
    try:
      with open(PATH_ACTION_TEMPLATE_P4, 'r') as action_template_file:
        action_template = action_template_file.read()
        action_template = action_template.replace("<ACTION_NAME>","set_code_"+feature.replace(" ","_").lower())
        action_template = action_template.replace("<ACTION_CODE_LENGTH>", str(codeword_bits_per_feature[feature]))
        action_template = action_template.replace("<FEATURE_NAME>", feature.replace(" ","_").lower())
        action_templates += action_template
    except:
      pass

  return action_templates


def generate_P4_tables_and_apply(feature_names, num_trees_app, num_trees_ddos,
                                  codewords=None, use_default_action_discount=False,
                                  match_type='ternary',
                                  feature_names_app=None, feature_names_ddos=None,
                                  raw_feature_names=None,
                                  config: "p4_gen_config.P4GenConfig" = None):
  """
  config: Task 4 -- additive convenience. When given, `config.use_default_action_discount`
  / `config.match_type` take precedence over the individual
  `use_default_action_discount` / `match_type` keyword arguments above
  (which remain the source of truth when `config` is None, so every
  existing caller is unaffected).

      table <TABLE_NAME> {
          key = {
              meta.<FEATURE_NAME>: <MATCH_TYPE>;
          }
          actions = {
              <ACTIONS>
          }
          size = <SIZE>;
          <DEFAULT_ACTION>
      }

  codewords, use_default_action_discount: Task 7 -- Planter-style
  default-action discount. codewords (when given) is the same tree_id ->
  {codeword: class_value} dict get_table_entries takes, indexed the same
  way get_table_entries' multi-model branch expects: tree_id 0..num_trees_app-1
  for the app trees, num_trees_app..num_trees_app+num_trees_ddos-1 for the
  ddos trees. When use_default_action_discount is True and codewords is
  given, each classification table gets a real
  `const default_action = classify_flow_codeword_<task>_<i>(<class>);`
  line (resources/table_classification.p4's new <DEFAULT_ACTION> marker);
  otherwise (the default) that marker resolves to nothing, so the
  generated table text is byte-identical to before this task.

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
  """

  if config is not None:
    use_default_action_discount = config.use_default_action_discount
    match_type = config.match_type

  SIZE_FEATURE_TABLE = 200
  SIZE_CLASSIFICATION_TABLE = 400

  if match_type not in ('ternary', 'exact'):
    raise ValueError("match_type must be 'ternary' or 'exact', got {!r}".format(match_type))

  classification_table_template_path = (
      PATH_TABLE_CLASSIFICATION_TEMPLATE_P4 if match_type == 'ternary'
      else PATH_TABLE_CLASSIFICATION_EXACT_TEMPLATE_P4
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
        "            meta.code_"+feature.replace(" ","_").lower()+" : "+match_type+";"
        for feature in names
    )

  classification_keys_app = _classification_keys(feature_names_app)
  classification_keys_ddos = _classification_keys(feature_names_ddos)

  def _default_action_line(action_name, tree_codewords):
    # "" leaves resources/table_classification.p4's whole
    # "        <DEFAULT_ACTION>\n" marker line stripped out below, so the
    # generated table text is byte-identical to pre-Task-7 output whenever
    # the discount isn't actually in effect for this table.
    if not use_default_action_discount or not tree_codewords:
      return ""
    class_value, _ = most_common_class_and_dropped_codewords(tree_codewords)
    return "        const default_action = {}({});\n".format(
        action_name, str(int(float(class_value))))

  #Classification tables
  if num_trees_app > 0:
    for i in range(num_trees_app):
      with open(classification_table_template_path, 'r') as table_template_file:
        table_template = table_template_file.read()
        table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_app_"+str(i))
        table_template = table_template.replace("<KEYS>", classification_keys_app)
        table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_app_"+str(i)+";")
        table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
        action_name = "classify_flow_codeword_app_"+str(i)
        tree_codewords = codewords.get(i) if codewords is not None else None
        table_template = table_template.replace(
            "        <DEFAULT_ACTION>\n", _default_action_line(action_name, tree_codewords))
      table_templates += table_template
      apply_templates_tmp += "\t\t\tget_classification_tree_app_"+str(i)+".apply();\n"
    apply_templates_tmp += "\n"

  if num_trees_ddos > 0:
    for i in range(num_trees_ddos):
      with open(classification_table_template_path, 'r') as table_template_file:
        table_template = table_template_file.read()
        table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_ddos_"+str(i))
        table_template = table_template.replace("<KEYS>", classification_keys_ddos)
        table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_ddos_"+str(i)+";")
        table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
        action_name = "classify_flow_codeword_ddos_"+str(i)
        tree_codewords = codewords.get(num_trees_app + i) if codewords is not None else None
        table_template = table_template.replace(
            "        <DEFAULT_ACTION>\n", _default_action_line(action_name, tree_codewords))
      table_templates += table_template
      apply_templates_tmp += "\t\t\tget_classification_tree_ddos_"+str(i)+".apply();\n"


  feature_idx = 0
  for feature in feature_names:
    # Task 3: the range table's key reads the RAW tracked-value field
    # (shared, model-independent) even when this resolved entry's NAME is
    # namespaced (app_<feature>/ddos_<feature>) -- defaults to `feature`
    # itself (raw_feature_names omitted or missing this key), reproducing
    # every pre-Task-3 caller's behavior byte-for-byte.
    raw_name = raw_feature_names.get(feature, feature)
    with open(PATH_TABLE_TEMPLATE_P4, 'r') as table_template_file:
      table_template = table_template_file.read()
      table_template = table_template.replace("<TABLE_NAME>","table_"+str(feature_idx)+"_"+feature.replace(" ","_").lower())
      table_template = table_template.replace("<FEATURE_NAME>", raw_name.replace(" ","_").lower()+"_val")
      table_template = table_template.replace("<MATCH_TYPE>", "range")
      table_template = table_template.replace("<ACTIONS>", str("set_code_"+feature.replace(" ","_").lower())+";")
      table_template = table_template.replace("<SIZE>", str(SIZE_FEATURE_TABLE))
      table_templates += table_template
      apply_templates += "\t\t\ttable_"+str(feature_idx)+"_"+feature.replace(" ","_").lower()+".apply();\n"

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
  3-tree/3-class case, with identical classification decisions (same
  statistics.mode() tie-breaking as before -- mechanism change, not a
  behavior change).
  """
  bit_per_classes = math.ceil(math.log2(num_classes)) or 1
  # Table emits exactly num_classes ** num_trees const entries (see the
  # product() loop below); size must scale with that, with 32 as a floor
  # matching the two configs validated this session (3-tree/3-class = 27,
  # 1-tree/2-class = 2) so this fix doesn't shrink the table for a
  # degenerate config and doesn't change output for the already-tested ones.
  size = max(32, num_classes ** num_trees)

  key_lines = "\n".join(
      "\t\t\tmeta.class_tree_{}_{} : exact;".format(task, i)
      for i in range(num_trees)
  )

  entries_lines = []
  for classification_array in product(range(num_classes), repeat=num_trees):
    winner = mode(classification_array)
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
                      config: "p4_gen_config.P4GenConfig" = None):
  """match_type: Task 8 -- passed straight through to
  generate_P4_tables_and_apply. 'ternary' (the default) is byte-identical
  to every caller before this task; 'exact' switches only the
  classification tables to resources/table_classification_exact.p4 /
  `: exact;` keys (feature-range tables stay ternary/range either way).

  config: Task 4 -- additive convenience. When given, `config.match_type`
  takes precedence over the individual `match_type` keyword argument above
  (which remains the source of truth when `config` is None, so every
  existing caller is unaffected). NOTE: `config.use_default_action_discount`
  has no effect here -- this function does not accept a `codewords`/
  `use_default_action_discount` parameter at all (pre-existing scope, not
  introduced or changed by this task); that flag is only consumed by
  `generate_P4_tables_and_apply` and `get_table_entries`, called directly.

  CAVEAT: with match_type='exact', the emitted P4 program is NOT yet
  end-to-end compilable/loadable on its own. get_table_entries (writes
  table_entries.json) is unchanged by this parameter and still emits
  ternary/wildcard-shaped ('*') entries for the same classification tables
  that this function just declared `: exact;` -- concrete, enumerated
  exact-match entries for those wildcarded codewords still need to be
  generated as follow-on work before 'exact' output can actually be loaded.

  Task 3: feature_intervals_app / feature_intervals_ddos are each model's
  OWN, independently-derived feature_intervals dict (raw feature name ->
  interval list) -- no longer a single shared dict, since real production
  deployment always runs ONE combined pipeline for both tasks, and disjoint
  encoding lets the two independently-trained models pick different
  discretization thresholds for a feature they both happen to select. A
  task with no active model passes {} for its side (e.g. an App-only or
  DDoS-only run passes {} for the other model's feature_intervals, exactly
  mirroring clf_app/clf_ddos=None for "no task"). A joint-encoded caller
  (both models sharing one discretization) passes the SAME dict for both
  parameters, reproducing every pre-Task-3 caller's output byte-for-byte.

  Internally this resolves both dicts, once, via
  _resolve_disjoint_feature_plan: a feature both models select with
  IDENTICAL intervals shares ONE discretization table/field; a feature both
  models select with DIFFERING intervals gets two independent, namespaced
  (app_/ddos_ prefixed) discretization tables/fields -- while the
  underlying RAW feature-value register/computation always stays shared,
  since a raw counter value is model-independent regardless of who
  discretizes it (see _resolve_disjoint_feature_plan's own docstring for
  the full design)."""

  if config is not None:
    match_type = config.match_type

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
  if num_trees_app > 0:
    for i in range(num_trees_app):
      metadata_code += "\tbit<"+str(bit_per_classes_app)+"> class_tree_app_"+str(i)+";\n"
      metadata_code += "\tbit<1> class_tree_app_"+str(i)+"_is_set;\n"
    # generate_voting_code (below) writes to meta.classification_app; the
    # TNA template no longer declares this field itself (it doesn't know
    # bit_per_classes_app ahead of time), so it must be declared here.
    metadata_code += "\tbit<"+str(bit_per_classes_app)+"> classification_app;\n"

  if num_trees_ddos > 0:
    for i in range(num_trees_ddos):
      metadata_code += "\tbit<"+str(bit_per_classes_ddos)+"> class_tree_ddos_"+str(i)+";\n"
      metadata_code += "\tbit<1> class_tree_ddos_"+str(i)+"_is_set;\n"
    metadata_code += "\tbit<"+str(bit_per_classes_ddos)+"> classification_ddos;\n"

  # Tier 3 + Task 3: every RESOLVED entry gets its own codeword field
  # (code_<resolved_name>), but the raw tracked-value field (<raw>_val) is
  # declared exactly once per DISTINCT raw feature name -- never twice just
  # because two resolved entries (e.g. app_<feature>/ddos_<feature>) both
  # discretize the same underlying shared raw value.
  raw_feature_intervals = {}  # raw_feature_name -> intervals (first-seen; only the KEYS feed generate_P4_registers_and_apply, which ignores values)
  for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items():
    if raw_feature_name not in raw_feature_intervals:
      raw_feature_intervals[raw_feature_name] = intervals
      metadata_code += "\tbit<16> "+raw_feature_name.replace(" ","_").lower()+"_val;\n"

  for resolved_name, (raw_feature_name, intervals, models) in resolved_plan.items():
    codeword_width = len(intervals) - 1
    metadata_code += "\tbit<"+str(codeword_width)+"> code_"+resolved_name.replace(" ","_").lower()+";\n"

  # Task 3, point 3: registers must be resolved against the DEDUPLICATED set
  # of RAW feature names, never the (possibly namespaced) resolved names --
  # a raw value register must not be generated twice just because two
  # resolved entries share it.
  registers_code, register_actions_code, feature_update_apply_code = generate_P4_registers_and_apply(raw_feature_intervals)

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

  table_templates, apply_templates = generate_P4_tables_and_apply(
      resolved_plan.keys(), num_trees_app, num_trees_ddos, match_type=match_type,
      feature_names_app=feature_names_app, feature_names_ddos=feature_names_ddos,
      raw_feature_names=raw_feature_names)

  # generate code to vote between the trees -- only for tasks that actually
  # have trees. generate_voting_code now returns (table_decl, apply_call);
  # the table declaration joins the other TABLES text, the apply call joins
  # the other APPLY text -- classification now happens via table application
  # like every other table in this generator, so there is no more separate
  # CLASSIFICATION content to build.
  if num_trees_app > 0:
    vote_table_app, vote_apply_app = generate_voting_code(num_trees_app, num_class_app, "app")
    table_templates += vote_table_app
    apply_templates += vote_apply_app
  if num_trees_ddos > 0:
    vote_table_ddos, vote_apply_ddos = generate_voting_code(num_trees_ddos, num_class_ddos, "ddos")
    table_templates += vote_table_ddos
    apply_templates += vote_apply_ddos

  # substitute the code in the template
  with open(PATH_P4_CODE_TEMPLATE_INPUT, 'r') as switch_template_file:
    switch_template = switch_template_file.read()
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
# It is standalone, new capability: it is NOT called from generate_P4_code()
# and does NOT touch resources/p4_template.p4. A later task rewrites the
# whole template for TNA and wires this function's output into it.
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
}

# Register-action body kinds that require an extra hardware-primitive
# declaration (e.g. a MathUnit<> instance) emitted immediately before their
# RegisterAction block. Maps body kind -> declaration template, parallel in
# spirit to _REGISTER_ACTION_BODIES above.
_EXTRA_ACTION_DECLARATIONS = {
    "mathunit_ewma": "\tMathUnit<bit<{width}>>(MathOp_t.MUL, 1, 2) {name}_halve_unit;\n",
}


def _register_declaration(name, width):
  return "\tRegister<bit<{width}>, bit<32>>(MAX_NUM_FLOWS) {name}_reg;\n".format(width=width, name=name)


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
  with the real Tofino p4c). Standalone codegen -- not wired into
  generate_P4_code()/resources/p4_template.p4 yet; a later task rewrites
  that template for TNA and consumes this function's output.

  Parameters:
    feature_intervals: dict whose keys are selected feature names, in the
      same Title_Case_With_Underscores casing produced elsewhere in this
      file by get_nodes()/get_feature_intervals() (e.g. "Flow_IAT_Max").
      Only the keys are consulted; values are ignored. Each key is
      .lower()'d before being looked up in `catalog`, matching the
      convention used everywhere else in this file that turns a feature
      name into a P4 identifier.
    catalog: feature -> register dependency catalog to resolve against
      (see feature_registers.FEATURE_REGISTER_CATALOG for the expected
      shape). Defaults to feature_registers.FEATURE_REGISTER_CATALOG.
      Exposed as a parameter so tests can exercise the generator against a
      synthetic catalog without monkeypatching module state.

  Feature names absent from `catalog` are silently skipped (later
  milestones will call this with a catalog that isn't fully populated yet
  for every feature they select -- that must not crash).

  Returns a 3-tuple of P4 source strings, one per marker payload a future
  TNA template will substitute this function's output into:
    (registers_code, register_actions_code, feature_update_apply_code)

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
      _EXTRA_ACTION_DECLARATIONS (currently only "mathunit_ewma") get one
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
      says `"gated_by": "fwd"` -- and assignment of each "value" register's
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

  # feature_intervals keys are Title_Case_With_Underscores (see get_nodes());
  # normalize to the catalog's lowercase snake_case keys, preserving
  # feature_intervals' iteration order.
  matched_features = [f for f in (name.lower() for name in feature_intervals.keys()) if f in catalog]

  # Guard: flow_iat_mean shares flow_last_arrival_time dependency with
  # flow_iat_max. If selected without flow_iat_max, the shared dependency
  # register would never be executed and meta.current_iat would hold garbage.
  if "flow_iat_mean" in matched_features and "flow_iat_max" not in matched_features:
    raise ValueError(
        "flow_iat_mean requires flow_iat_max to also be selected (shared "
        "flow_last_arrival_time dependency register) -- see "
        "reviews/t11_tofino_port_and_env.md H.6"
    )

  if not matched_features:
    return "", "", ""

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
    if gated_by not in (None, "fwd"):
      raise RuntimeError(
          "Feature '{feature}' has unsupported gated_by={gated_by!r}; "
          "only None and 'fwd' are implemented.".format(feature=feature, gated_by=gated_by)
      )

    regs = []
    for reg in entry["registers"]:
      _note_touch(reg["name"], width=reg["width"], body=reg["body"])
      regs.append(reg)
      if reg["body"] == "iat_delta":
        needs_timestamp = True
    feature_registers[feature] = regs

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
    registers_code += _register_declaration(name, register_info[name]["width"])

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
  # block, tracked across BOTH the ungated and fwd-gated _execute_lines()
  # calls below (a single shared set, not one per call) -- this is the
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

  apply_lines += _execute_lines(ungated_features, "\t\t\t")

  if fwd_gated_features:
    apply_lines.append("")
    apply_lines.append("\t\t\tif (meta.fwd == 1) {")
    apply_lines += _execute_lines(fwd_gated_features, "\t\t\t\t")
    apply_lines.append("\t\t\t}")

  apply_code = "\n".join(apply_lines) + "\n"

  return registers_code, register_actions_code, apply_code