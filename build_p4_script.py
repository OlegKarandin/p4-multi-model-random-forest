import os
import math
from sklearn.tree import export_text
import csv
import json
from itertools import product
from statistics import mode

INFINITE = 2147483647
PATH = "resources/"
INTERMEDIATE = "temp/"
OUTPUT_PATH = "p4/"
PATH_TABLE_ENTRIES_OUTPUT = OUTPUT_PATH + "table_entries.json"
PATH_TABLE_TEMPLATE_P4  = PATH + 'table.p4'
PATH_ACTION_TEMPLATE_P4 = PATH + 'action.p4'
PATH_P4_CODE_TEMPLATE_INPUT = PATH + 'p4_template.p4'
PATH_P4_CODE_TEMPLATE_OUTPUT = OUTPUT_PATH + 'p4_code_RF_models.p4'


def dt_thresholds_float_to_int(clf):
  # Access the decision thresholds in each tree
  for tree in clf.estimators_:
      for i, threshold in enumerate(tree.tree_.threshold):
        if threshold != -2: # OLEG: ???
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

  #Gather nodes from all Decision Trees
  for tree in tree_nodes:
    for node in tree_nodes[tree]:
      nodes.append(tree_nodes[tree][node])

  #Join all feature thresholds (splits) used across the nodes
  for node in nodes:
    if node['is_leaf'] == False:
      feature_splits.append((node["feature"],
                            node["threshold"]))

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
      #New Feature, Initialize interval
      if feature not in feature_intervals:
          feature_intervals[feature] = [[0, threshold]]
      
      #Existing Feature, Extend Interval
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


def feature_intervals_to_csv(feature_intervals):
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

  # File path
  file_path = INTERMEDIATE + 'feature_intervals.csv'

  # Open the file in write mode
  with open(file_path, 'w', newline='') as csvfile:
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

  return paths_leaf_nodes_per_tree


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
      feature_names = feature_intervals.keys()
      for feature in feature_names:
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

  hex_value = hex(int(value, 2))[2:].upper()
  hex_mask = hex(int(mask, 2))[2:].upper()

  return f"0x{hex_value}&&&0x{hex_mask}"


def get_table_entries(paths_leaf_nodes_per_tree, feature_intervals, codewords, offset, verbose=False):
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
      if tree_idx < offset:
        table_entry["table_name"] = "get_classification_tree_app_"+str(tree_idx)
        table_entry["action"] = "classify_flow_codeword_app"
        table_entry["action_params"] = [str(tree_idx), str(int(float((codewords[tree][codeword]))))]
      else:
        table_entry["table_name"] = "get_classification_tree_ddos_"+str(tree_idx-offset)
        table_entry["action"] = "classify_flow_codeword_ddos"
        table_entry["action_params"] = [str(tree_idx-offset), str(int(float((codewords[tree][codeword]))))]
      table_entry["key"] = [get_ternary_match(codeword)]

      table_entries.append(table_entry)

  if verbose == True:
    # Show Generated Table Entries
    for entry in table_entries:
      print(entry)

  if not os.path.exists(OUTPUT_PATH):
      os.makedirs(OUTPUT_PATH)

  # Save table entries to JSON
  with open(PATH_TABLE_ENTRIES_OUTPUT, 'w') as output_file:
    output_file.write(json.dumps(table_entries))


def generate_P4_actions(feature_intervals, codeword_length, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos):
  """
      action <ACTION_NAME> (bit<<ACTION_CODE_LENGTH>> code) {
          meta.codeword[<END_BIT>:<INIT_BIT>] = code;
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
  current_code_bit = codeword_length - 1

  #Classification action templates
  bit_per_num_trees_app = math.ceil(math.log2(num_trees_app))
  if bit_per_num_trees_app == 0:
    bit_per_num_trees_app = 1

  bit_per_num_trees_ddos = math.ceil(math.log2(num_trees_ddos))
  if bit_per_num_trees_ddos == 0:
    bit_per_num_trees_ddos = 1

  classification_action_template_app = "\taction classify_flow_codeword_app(bit<"+str(bit_per_classes_app)+"> tree, bit<"+str(bit_per_classes_app)+"> class){\n"
  classification_action_template_ddos = "\taction classify_flow_codeword_ddos(bit<"+str(bit_per_num_trees_ddos)+"> tree, bit<"+str(bit_per_classes_ddos)+"> class){\n"

  #Classification actions for traffic flow probem
  for i in range(num_trees_app):
    classification_action_template_app += "\t\tif (tree == "+str(i)+"){\n"
    classification_action_template_app += "\t\t\tmeta.class_tree_app_"+str(i)+" = class;\n"
    classification_action_template_app += "\t\t}\n"

  classification_action_template_app += "\t}\n"

  action_templates += classification_action_template_app
  action_templates += "\n"

  #Classification actions for DDOS detection problem
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
        action_template = action_template.replace("<END_BIT>", str(current_code_bit))
        action_template = action_template.replace("<INIT_BIT>", str(current_code_bit-codeword_bits_per_feature[feature]+1))
        action_templates += action_template

      current_code_bit -= codeword_bits_per_feature[feature]
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

  #Classification tables
  for i in range(num_trees_app):
    with open(PATH_TABLE_TEMPLATE_P4, 'r') as table_template_file:
      table_template = table_template_file.read()
      table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_app_"+str(i))
      table_template = table_template.replace("<FEATURE_NAME>", "codeword")
      table_template = table_template.replace("<MATCH_TYPE>", "ternary")
      table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_app;")
      table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
    table_templates += table_template
    apply_templates_tmp += "\t\t\tget_classification_tree_app_"+str(i)+".apply();\n"
  apply_templates_tmp += "\n"

  for i in range(num_trees_ddos):
    with open(PATH_TABLE_TEMPLATE_P4, 'r') as table_template_file:
      table_template = table_template_file.read()
      table_template = table_template.replace("<TABLE_NAME>","get_classification_tree_ddos_"+str(i))
      table_template = table_template.replace("<FEATURE_NAME>", "codeword")
      table_template = table_template.replace("<MATCH_TYPE>", "ternary")
      table_template = table_template.replace("<ACTIONS>", "classify_flow_codeword_ddos;")
      table_template = table_template.replace("<SIZE>", str(SIZE_CLASSIFICATION_TABLE))
    table_templates += table_template
    apply_templates_tmp += "\t\t\tget_classification_tree_ddos_"+str(i)+".apply();\n"


  feature_idx = 0
  for feature in feature_names:
    with open(PATH_TABLE_TEMPLATE_P4, 'r') as table_template_file:
      table_template = table_template_file.read()
      table_template = table_template.replace("<TABLE_NAME>","table_"+str(feature_idx)+"_"+feature.replace(" ","_").lower())
      table_template = table_template.replace("<FEATURE_NAME>", feature.replace(" ","_").lower())
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


def generate_P4_code(num_class_app, num_class_ddos, clf_app, clf_ddos, codeword_length, feature_intervals):

  # generate the definition of the bit containers that contain decisions of each tree
  bit_per_classes_app = math.ceil(math.log2(num_class_app))
  bit_per_classes_ddos = math.ceil(math.log2(num_class_ddos))

  num_trees_app = len(clf_app.estimators_)
  num_trees_ddos = len(clf_ddos.estimators_)

  metadata_code = ""
  for i in range(num_trees_app):
    metadata_code += "\tbit<"+str(bit_per_classes_app)+"> class_tree_app_"+str(i)+";\n"
    metadata_code += "\tbit<1> class_tree_app_"+str(i)+"_is_set;\n"

  for i in range(num_trees_ddos):
    metadata_code += "\tbit<"+str(bit_per_classes_ddos)+"> class_tree_ddos_"+str(i)+";\n"
    metadata_code += "\tbit<1> class_tree_ddos_"+str(i)+"_is_set;\n"

  metadata_code += "\tbit<"+str(codeword_length)+"> codeword;\n"


  action_templates = ""
  table_templates = ""
  apply_templates = ""

  action_templates = generate_P4_actions(feature_intervals, codeword_length, num_trees_app, num_trees_ddos, bit_per_classes_app, bit_per_classes_ddos)
  table_templates, apply_templates = generate_P4_tables_and_apply(feature_intervals.keys(), num_trees_app, num_trees_ddos)

  # generate code to vote between the trees
  classification_templates = generate_voting_code(num_trees_app, 3, "app")
  classification_templates += "\n"
  classification_templates += generate_voting_code(num_trees_ddos, 2, "ddos")

  # substitute the code in the template
  with open(PATH_P4_CODE_TEMPLATE_INPUT, 'r') as switch_template_file:
    switch_template = switch_template_file.read()
    switch_template = switch_template.replace('/* METADATA */', metadata_code)
    switch_template = switch_template.replace('/* ACTIONS */', action_templates)
    switch_template = switch_template.replace('/* TABLES */', table_templates)
    switch_template = switch_template.replace('/* APPLY */', apply_templates)
    switch_template = switch_template.replace('/* CLASSIFICATION */', classification_templates)


  with open(PATH_P4_CODE_TEMPLATE_OUTPUT, 'w') as switch_template_file:
    switch_template_file.write(switch_template)