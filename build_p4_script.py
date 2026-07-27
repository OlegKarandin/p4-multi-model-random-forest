import os
import math
from sklearn.tree import export_text
import csv
import json
from itertools import product
from statistics import mode

from feature_registers import FEATURE_REGISTER_CATALOG

INFINITE = (2**19)-1
MAX_NUM_FLOWS = 4096  # matches p4/p4_code_RF_models.p4:9 and
                      # p4/tofino_spike/tna_m1_flows_iat_spike.p4

PATH = "resources/"
INTERMEDIATE = "temp/"
OUTPUT_PATH = "p4/"
PATH_TABLE_ENTRIES_OUTPUT = OUTPUT_PATH + "table_entries.json"
PATH_TABLE_TEMPLATE_P4  = PATH + 'table.p4'
PATH_ACTION_TEMPLATE_P4 = PATH + 'action.p4'
PATH_TABLE_CLASSIFICATION_TEMPLATE_P4 = PATH + 'table_classification.p4'
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
  # Access the decision thresholds in each tree
  for tree in clf.estimators_:
      for i, threshold in enumerate(tree.tree_.threshold):
        if threshold != -2:
            tree.tree_.threshold[i] = int(threshold)

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


def get_nodes(tree_text):
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
  father_node_id           = -1
  ##########################

  for line in tree_lines:
    # Calculate Current Node's Depth based on number of "|" ocurrences in the line
    depth = line.count("|")

    # A) Node is Internal (Not Leaf)
    if "class" in line:

      #New Leaf Node
      nodes[node_id]={"node": node_id,
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
      father_node_id += 1

    # B) Node is Child
    else:

      feature_name  = line.replace("|---","").replace("|   ","").split("<=")[0].strip().replace(" ","_")
      threshold     = line.replace("|---","").replace("|   ","").split("<=")[-1].strip()

      if "<=" in line:
        # New Internal Node
        nodes[node_id]={"node": node_id,
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
        node_id += 1
        father_node_id += 1

  return nodes

def get_feature_splits(tree_nodes):
  '''
   Inputs: Dictionary containing the features of all nodes in the Random Forest
   Outputs: List of tuples containing the comparison thresholds (i.e. feature splits) each feature goes through over all the trees: (Feature Name, Threshold)
   '''

  nodes = []
  feature_splits = []

  #Gather node features from all Decision Trees
  for tree in tree_nodes:
    for node in tree_nodes[tree]:
      nodes.append(tree_nodes[tree][node])
  #Join all feature thresholds (splits) that are consulted at each node
  for node in nodes:
    try:
      feature_splits.append((node["feature"],
                            node["threshold"]))
    except:
      pass
  #Sort feature thresholds by feature
  return sorted(feature_splits, key=lambda x: (x[0], x[1]))


def get_feature_intervals(feature_splits):
  '''
  Inputs: List of tuples containing the features splits of all features
  Outputs: Dictionary where each key is the feature name and the associated value is the list of intervals of the given feature
  '''
  feature_intervals = {}

  # Iterate over each feature split
  for feature, threshold in feature_splits:
      #New Feature, Init interval
      if feature not in feature_intervals:
          feature_intervals[feature] = [[0, threshold]]
      #Exisiting Feature, Extend Interval
      else:
          last_range = feature_intervals[feature][-1]
          if last_range[1] == "infinite":
              continue
          if threshold == last_range[1]:
              continue
          feature_intervals[feature].append([last_range[1]+1, threshold])

  # Add last interval (higher threshold, infinite)
  for feature, ranges in feature_intervals.items():
      if ranges[-1][1] != "infinite":
          ranges.append([ranges[-1][1]+1, "infinite"])

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


def get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset=None, path_to_output=OUTPUT_PATH, output_filename="table_entries.json", verbose=False):
  '''
  Inputs: feature_intervals [dict]: Dictionary where each key is a tree_id. The values are a list of feature intervals for each tree.
          codewords [dict]: Dictionary where each key is a tree_id. The values are a list of dictionaries.
                            Each key in the dictionary is a codeword corresponding to a leaf node.
                            Each value is the class label associated with that codeword (i.e.: to the leaf node).

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
      maximum = str(INFINITE) if interval[1] == 'infinite' else str(interval[1])

      table_entry["key"] = [minimum+".."+maximum]
      table_entry["action_params"] = [str(int("".join(code),2))]

      table_entries.append(table_entry)

    feature_idx +=1


  # 2. Table entries for getting each tree's classification based on generated codeword
  for tree_idx,tree in enumerate(codewords):
    for codeword in codewords[tree]:
      table_entry={}

      if offset==None:
        #One model encoding
        table_entry["table_name"] = "get_classification_tree_"+str(tree_idx)
        table_entry["action"] = "classify_flow_codeword"
        table_entry["action_params"] = [str(tree_idx), str(int(float((codewords[tree][codeword]))))]
      else:
        #Multiple models encoding
        if tree_idx < offset:
          table_entry["table_name"] = "get_classification_tree_app_"+str(tree_idx)
          table_entry["action"] = "classify_flow_codeword_app"
          table_entry["action_params"] = [str(tree_idx), str(int(float((codewords[tree][codeword]))))]
        else:
          table_entry["table_name"] = "get_classification_tree_ddos_"+str(tree_idx-offset)
          table_entry["action"] = "classify_flow_codeword_ddos"
          table_entry["action_params"] = [str(tree_idx-offset), str(int(float((codewords[tree][codeword]))))]

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

  #Classification action templates
  if num_trees_app > 0:
    bit_per_num_trees_app = math.ceil(math.log2(num_trees_app))
    if bit_per_num_trees_app == 0:
      bit_per_num_trees_app = 1

    classification_action_template_app = "\taction classify_flow_codeword_app(bit<"+str(bit_per_num_trees_app)+"> tree, bit<"+str(bit_per_classes_app)+"> class){\n"

    if num_trees_app == 1:
      # TNA: a single-tree task's classification action must write
      # unconditionally. A runtime "if (tree == 0) {...}" here lowers to an
      # IR::Mux that p4c's ActionAnalysis pass rejects outright ("Conditions
      # in an action must be simple comparisons of an action data
      # parameter"), even though the condition is always true for a single
      # tree. Ground truth: every compiled spike's single-tree classify_ddos
      # action (e.g. p4/tofino_spike/tna_rf_ddos_spike_tier3.p4) writes its
      # metadata field unconditionally, with no tree-keyed branch at all.
      classification_action_template_app += "\t\tmeta.class_tree_app_0 = class;\n"
    else:
      # NOT VALIDATED against the real TNA compiler: this if/else-per-tree
      # body is expected to fail the same way the num_trees==1 case did
      # (rejected IR::Mux, see af64bc2) until it's redesigned for TNA.
      #Classification actions for traffic flow probem
      for i in range(num_trees_app):
        classification_action_template_app += "\t\tif (tree == "+str(i)+"){\n"
        classification_action_template_app += "\t\t\tmeta.class_tree_app_"+str(i)+" = class;\n"
        classification_action_template_app += "\t\t}\n"

    classification_action_template_app += "\t}\n"

    action_templates += classification_action_template_app
    action_templates += "\n"

  #Classification actions for DDOS detection problem
  if num_trees_ddos > 0:
    bit_per_num_trees_ddos = math.ceil(math.log2(num_trees_ddos))
    if bit_per_num_trees_ddos == 0:
      bit_per_num_trees_ddos = 1

    classification_action_template_ddos = "\taction classify_flow_codeword_ddos(bit<"+str(bit_per_num_trees_ddos)+"> tree, bit<"+str(bit_per_classes_ddos)+"> class){\n"

    if num_trees_ddos == 1:
      # See the matching comment in the num_trees_app branch above: a single
      # tree must not be wrapped in a runtime "if (tree == 0)" -- p4c's TNA
      # backend rejects that Mux unconditionally.
      classification_action_template_ddos += "\t\tmeta.class_tree_ddos_0 = class;\n"
    else:
      # NOT VALIDATED against the real TNA compiler: same caveat as the
      # num_trees_app branch above -- expected to be rejected as an
      # IR::Mux (see af64bc2) until redesigned for a future >1-tree TNA build.
      for i in range(num_trees_ddos):
        classification_action_template_ddos += "\t\tif (tree == "+str(i)+"){\n"
        classification_action_template_ddos += "\t\t\tmeta.class_tree_ddos_"+str(i)+" = class;\n"
        classification_action_template_ddos += "\t\t}\n"

    classification_action_template_ddos += "\t}\n"

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


def generate_P4_tables_and_apply(feature_names, num_trees_app, num_trees_ddos):
  """
      table <TABLE_NAME> {
          key = {
              meta.<FEATURE_NAME>: <MATCH_TYPE>;
          }
          actions = {
              <ACTIONS>
          }
          size = <SIZE>;
      }
  """

  SIZE_FEATURE_TABLE = 200
  SIZE_CLASSIFICATION_TABLE = 400

  table_templates = ""
  apply_templates_tmp = "\n"
  apply_templates = ""

  feature_names = list(feature_names)

  # Tier 3: the classification tables key on one ternary field per selected
  # feature (meta.code_<feature>), not a single combined meta.codeword field.
  classification_keys = "\n".join(
      "            meta.code_"+feature.replace(" ","_").lower()+" : ternary;"
      for feature in feature_names
  )

  #Classification tables
  if num_trees_app > 0:
    for i in range(num_trees_app):
      with open(PATH_TABLE_CLASSIFICATION_TEMPLATE_P4, 'r') as table_template_file:
        table_template = table_template_file.read()
        table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_app_"+str(i))
        table_template = table_template.replace("<KEYS>", classification_keys)
        table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_app;")
        table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
      table_templates += table_template
      apply_templates_tmp += "\t\t\tget_classification_tree_app_"+str(i)+".apply();\n"
    apply_templates_tmp += "\n"

  if num_trees_ddos > 0:
    for i in range(num_trees_ddos):
      with open(PATH_TABLE_CLASSIFICATION_TEMPLATE_P4, 'r') as table_template_file:
        table_template = table_template_file.read()
        table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_ddos_"+str(i))
        table_template = table_template.replace("<KEYS>", classification_keys)
        table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_ddos;")
        table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
      table_templates += table_template
      apply_templates_tmp += "\t\t\tget_classification_tree_ddos_"+str(i)+".apply();\n"


  feature_idx = 0
  for feature in feature_names:
    with open(PATH_TABLE_TEMPLATE_P4, 'r') as table_template_file:
      table_template = table_template_file.read()
      table_template = table_template.replace("<TABLE_NAME>","table_"+str(feature_idx)+"_"+feature.replace(" ","_").lower())
      table_template = table_template.replace("<FEATURE_NAME>", feature.replace(" ","_").lower()+"_val")
      table_template = table_template.replace("<MATCH_TYPE>", "range")
      table_template = table_template.replace("<ACTIONS>", str("set_code_"+feature.replace(" ","_").lower())+";")
      table_template = table_template.replace("<SIZE>", str(SIZE_FEATURE_TABLE))
      table_templates += table_template
      apply_templates += "\t\t\ttable_"+str(feature_idx)+"_"+feature.replace(" ","_").lower()+".apply();\n"

    feature_idx += 1

  apply_templates += apply_templates_tmp

  return table_templates, apply_templates


def generate_voting_code(num_trees, num_classes, task):

  temp_str = ''

  classes_list = [i for i in range(num_classes)]

  for classification_array in product(classes_list, repeat=num_trees):

    temp_str += "\t\t\tif ("
    for i in range(len(classification_array)):
      if i<len(classification_array)-1:
        temp_str += "(meta.class_tree_{}_".format(task) + str(i) + " == " + str(classification_array[i]) + ") && "
      else:
        temp_str += "(meta.class_tree_{}_".format(task) + str(i) + " == " + str(classification_array[i]) + ")) {\n"
    winner = mode(classification_array)
    temp_str += "\t\t\t\tmeta.classification_{} = ".format(task) + str(winner) + ";\n\t\t\t}\n"

  return temp_str


def generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos, feature_intervals):

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

  # Tier 3: every selected feature gets its own tracked-value field and its
  # own codeword field, instead of all features sharing bit-slices of one
  # combined meta.codeword field.
  for feature in feature_intervals:
    feature_id = feature.replace(" ","_").lower()
    codeword_width = len(feature_intervals[feature]) - 1
    metadata_code += "\tbit<16> "+feature_id+"_val;\n"
    metadata_code += "\tbit<"+str(codeword_width)+"> code_"+feature_id+";\n"

  registers_code, register_actions_code, feature_update_apply_code = generate_P4_registers_and_apply(feature_intervals)

  action_templates = generate_P4_actions(feature_intervals, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos)
  table_templates, apply_templates = generate_P4_tables_and_apply(feature_intervals.keys(), num_trees_app, num_trees_ddos)

  # generate code to vote between the trees -- only for tasks that actually
  # have trees (generate_voting_code(0, ...) would crash: product(..., repeat=0)
  # yields one empty tuple, and mode(()) raises StatisticsError).
  classification_templates = ""
  if num_trees_app > 0:
    classification_templates += generate_voting_code(num_trees_app, num_class_app, "app")
  if num_trees_ddos > 0:
    if num_trees_app > 0:
      classification_templates += "\n"
    classification_templates += generate_voting_code(num_trees_ddos, num_class_ddos, "ddos")

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
    switch_template = switch_template.replace('/* CLASSIFICATION */', classification_templates)

  ensure_directory_exists(OUTPUT_PATH)
  with open(OUTPUT_PATH + 'p4_code_RF_models.p4', 'w') as switch_template_file:
    switch_template_file.write(switch_template)


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

# Baseline per-flow bookkeeping register (fwd/bwd/new-flow tracking, "M1-0"
# in the spike). Needed once, unconditionally, whenever the resolved
# feature set is non-empty -- see generate_P4_registers_and_apply's
# docstring for why this lives outside FEATURE_REGISTER_CATALOG.
_FLOWS_REGISTER_NAME = "flows"
_FLOWS_REGISTER_WIDTH = 1

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
      declarations (TNA syntax), plus the two `Hash<>` instances the
      apply-block's hash actions need (flow_hash_calc_self/_other -- TNA
      requires one Hash<> instance per distinct field list, so the fwd- and
      bwd-ordered hashes can't share one instance).
    - register_actions_code: the plain calc_flow_hash_self/
      calc_flow_hash_other actions (and calc_timestamp, only emitted when
      an included register actually needs meta.now_pseudo_us), plus one
      `RegisterAction<...> <name>_action = {...};` block per resolved
      register, using the exact bodies from _REGISTER_ACTION_BODIES. Body
      kinds listed in _EXTRA_ACTION_DECLARATIONS (currently only
      "mathunit_ewma") get one extra hardware-primitive declaration line
      (e.g. `MathUnit<bit<W>>(MathOp_t.MUL, 1, 2) <name>_halve_unit;`)
      emitted immediately before that register's RegisterAction block. The
      `flows` register gets two action blocks -- flows_test_other
      (read-only) and flows_set_self (test-and-set) -- in the corrected
      2-touch design: always read-only test the *other* direction's slot
      first; only touch (and only ever write) our *own* hash's slot if
      that read back 0. This is deliberately not the naive ordering
      (test-and-set our own hash first) -- see the spike's header comment
      for the correctness bug that ordering has.
    - feature_update_apply_code: the per-packet apply-block snippet: the
      hash-calc and (conditionally) timestamp calls, the flows touch(es),
      then each *distinct register name*'s `.execute()` call -- gated
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
      it were derived from call-site counts. NOTE: the touch-count guard
      itself (`register_touch_count`, checked further below) is NOT
      recomputed from this deduplicated call-site count -- it is still
      accumulated once per (feature, catalog register-list entry), exactly
      as before this fix. For every real (non-synthetic) catalog shape
      this is a conservative over-count relative to the deduplicated
      call-site count above (e.g. flow_last_arrival_time reports 2 touches
      for M2's feature set even though only 1 `.execute()` call site is
      actually emitted) -- never an under-count, so it cannot let a
      genuinely-too-many-touches design slip through silently. Fully
      aligning the guard with the deduplicated count is deliberately not
      attempted here: doing so would change what
      test_register_touch_limit_raises (a synthetic guardrail test that
      simulates ">MAX_REGISTER_TOUCHES touches" via one register name
      repeated within a single feature's own registers list) exercises,
      and is out of this task's scope.

  Raises RuntimeError, at generation time (not left to fail later at `p4c`
  invocation), if resolving `catalog` against `feature_intervals` would
  require more than MAX_REGISTER_TOUCHES distinct RegisterAction
  `.execute()` call sites against any single register.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # feature_intervals keys are Title_Case_With_Underscores (see get_nodes());
  # normalize to the catalog's lowercase snake_case keys, preserving
  # feature_intervals' iteration order.
  matched_features = [f for f in (name.lower() for name in feature_intervals.keys()) if f in catalog]

  if not matched_features:
    return "", "", ""

  # ---- Resolve the deduplicated, ordered register set + touch counts ----

  register_order = []        # ordered list of register names (first-seen)
  register_info = {}         # name -> {"width":, "body":}
  register_touch_count = {}  # name -> number of .execute() call sites

  def _note_touch(name, width=None, body=None, count=1):
    if name not in register_info:
      register_order.append(name)
      register_info[name] = {"width": width, "body": body}
    register_touch_count[name] = register_touch_count.get(name, 0) + count

  # Baseline bookkeeping register: needed once any per-flow feature register
  # exists, always exactly 2 touches (flows_test_other + flows_set_self).
  _note_touch(_FLOWS_REGISTER_NAME, width=_FLOWS_REGISTER_WIDTH, count=2)

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
      "\tHash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_self;\n"
      "\tHash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_other;\n"
  )
  for name in register_order:
    registers_code += _register_declaration(name, register_info[name]["width"])

  # ---- /* REGISTER_ACTIONS */ ----

  register_actions_code = (
      "\taction calc_flow_hash_self() {\n"
      "\t\tmeta.flow_hash_self = flow_hash_calc_self.get({\n"
      "\t\t\thdr.ipv4.src_addr,\n"
      "\t\t\thdr.ipv4.dst_addr,\n"
      "\t\t\thdr.ipv4.protocol,\n"
      "\t\t\thdr.tcp.src_port,\n"
      "\t\t\thdr.tcp.dst_port\n"
      "\t\t}) & 0xFFF;\n"
      "\t}\n"
      "\n"
      "\taction calc_flow_hash_other() {\n"
      "\t\tmeta.flow_hash_other = flow_hash_calc_other.get({\n"
      "\t\t\thdr.ipv4.dst_addr,\n"
      "\t\t\thdr.ipv4.src_addr,\n"
      "\t\t\thdr.ipv4.protocol,\n"
      "\t\t\thdr.tcp.dst_port,\n"
      "\t\t\thdr.tcp.src_port\n"
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

  register_actions_code += (
      "\n"
      "\tRegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_test_other = {\n"
      "\t\tvoid apply(inout bit<1> value, out bit<1> rv) {\n"
      "\t\t\trv = value;\n"
      "\t\t}\n"
      "\t};\n"
      "\n"
      "\tRegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_set_self = {\n"
      "\t\tvoid apply(inout bit<1> value, out bit<1> rv) {\n"
      "\t\t\tvalue = 1;\n"
      "\t\t\trv = value;\n"
      "\t\t}\n"
      "\t};\n"
  )

  for name in register_order:
    if name == _FLOWS_REGISTER_NAME:
      continue
    info = register_info[name]
    extra_declaration = _EXTRA_ACTION_DECLARATIONS.get(info["body"])
    if extra_declaration is not None:
      register_actions_code += "\n" + extra_declaration.format(width=info["width"], name=name)
    register_actions_code += "\n" + _register_action_declaration(name, info["width"], info["body"])

  # ---- /* FEATURE_UPDATE_APPLY */ ----

  apply_lines = [
      "\t\t\tcalc_flow_hash_self();",
      "\t\t\tcalc_flow_hash_other();",
  ]
  if needs_timestamp:
    apply_lines.append("\t\t\tcalc_timestamp();")

  apply_lines += [
      "",
      "\t\t\tbit<1> other_seen = flows_test_other.execute(meta.flow_hash_other);",
      "",
      "\t\t\tif (other_seen == 1) {",
      "\t\t\t\tmeta.fwd = 0;",
      "\t\t\t\tmeta.flow_hash = meta.flow_hash_other;",
      "\t\t\t} else {",
      "\t\t\t\tflows_set_self.execute(meta.flow_hash_self);",
      "\t\t\t\tmeta.fwd = 1;",
      "\t\t\t\tmeta.flow_hash = meta.flow_hash_self;",
      "\t\t\t}",
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