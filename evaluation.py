import math
from build_p4_script import *

def memory_evaluation(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding):

  #P4 encoding
  if encoding == 'joint':
    trees_app = get_tree_textual_representation(clf_app, selected_features_app)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features_ddos)

    tree_nodes = {}

    for tree_app in trees_app:
      tree_nodes[tree_app] = get_nodes(trees_app[tree_app])

    offset = len(tree_nodes)

    for tree_ddos in trees_ddos:
      tree_nodes[tree_ddos+offset] = get_nodes(trees_ddos[tree_ddos])

    feature_splits = get_feature_splits(tree_nodes)
    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    feature_intervals = get_feature_intervals(feature_splits)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)


    #Range-matching tables computation
    total_range_entries = 0
    total_range_TCAM_blocks = 0
    for key in feature_intervals:
      total_range_entries_block = 0
      for interv in feature_intervals[key]:
        total_range_entries+=1
        if interv[1]!='infinite':
          width = interv[1]-interv[0]
        else:
          width= (2**19)-1-interv[0]
        if width>1:
          total_range_entries_block+= 2*math.floor(math.log2(width))
        else:
          total_range_entries_block+= 2*1
      total_range_TCAM_blocks+= math.ceil(total_range_entries_block/207)



    #Ternary-matching tables computation
    total_ternary_blocks = 0
    total_ternary_entries = 0
    codeword_length = len(next(iter(codewords[0].items()))[0])
    max_joint = math.ceil(codeword_length/44)
    for tree in codewords:
      total_ternary_entries+=len(codewords[tree])
      total_ternary_blocks+= math.ceil(len(codewords[tree])/512)*max_joint


  elif encoding == 'disjoint':
    #P4 encoding
    trees_app = get_tree_textual_representation(clf_app, selected_features_app)
    trees_ddos = get_tree_textual_representation(clf_ddos, selected_features_ddos)

    tree_nodes_app = {}
    tree_nodes_ddos = {}

    for tree_app in trees_app:
      tree_nodes_app[tree_app] = get_nodes(trees_app[tree_app])

    for tree_ddos in trees_ddos:
      tree_nodes_ddos[tree_ddos] = get_nodes(trees_ddos[tree_ddos])



    feature_splits_app = get_feature_splits(tree_nodes_app)
    paths_leaf_nodes_per_tree_app = get_root_to_leaf_paths(tree_nodes_app)
    feature_intervals_app = get_feature_intervals(feature_splits_app)
    codewords_app = generate_codewords(paths_leaf_nodes_per_tree_app, feature_intervals_app)


    feature_splits_ddos = get_feature_splits(tree_nodes_ddos)
    paths_leaf_nodes_per_tree_ddos = get_root_to_leaf_paths(tree_nodes_ddos)
    feature_intervals_ddos = get_feature_intervals(feature_splits_ddos)
    codewords_ddos = generate_codewords(paths_leaf_nodes_per_tree_ddos, feature_intervals_ddos)


    #Range-matching tables computation for traffic flows classifier
    total_range_entries_app = 0
    total_range_TCAM_blocks_app = 0
    for key in feature_intervals_app:
      total_range_entries_block_app = 0
      for interv in feature_intervals_app[key]:
        total_range_entries_app+=1
        if interv[1]!='infinite':
          width = interv[1]-interv[0]
        else:
          width= (2**19)-1-interv[0]
        if width>1:
          total_range_entries_block_app+= 2*math.floor(math.log2(width))
        else:
          total_range_entries_block_app+= 2*1
      total_range_TCAM_blocks_app+= math.ceil(total_range_entries_block_app/207)



    #Range-matching tables computation for DDOS detector
    total_range_entries_ddos= 0
    total_range_TCAM_blocks_ddos = 0
    for key in feature_intervals_ddos:
      total_range_entries_block_ddos = 0
      for interv in feature_intervals_ddos[key]:
        total_range_entries_ddos+=1
        if interv[1]!='infinite':
          width = interv[1]-interv[0]
        else:
          width= (2**19)-1-interv[0]
        if width>1:
          total_range_entries_block_ddos+= 2*math.floor(math.log2(width))
        else:
          total_range_entries_block_ddos+= 2*1
      total_range_TCAM_blocks_ddos+= math.ceil(total_range_entries_block_ddos/207)



    #Range-matching tables final summation
    total_range_TCAM_blocks = total_range_TCAM_blocks_app + total_range_TCAM_blocks_ddos
    total_range_entries = total_range_entries_app + total_range_entries_ddos


    #Ternary-matching tables computation for traffic flows classifier
    total_ternary_blocks_app = 0
    total_ternary_entries_app = 0
    codeword_length_app = len(next(iter(codewords_app[0].items()))[0])
    max_app = math.ceil(codeword_length_app/44)
    for tree in codewords_app:
      total_ternary_entries_app+=len(codewords_app[tree])
      total_ternary_blocks_app+= math.ceil(len(codewords_app[tree])/512)*max_app


    #Ternary-matching tables computation for traffic flows classifier
    total_ternary_blocks_ddos = 0
    total_ternary_entries_ddos = 0
    codeword_length_ddos = len(next(iter(codewords_ddos[0].items()))[0])
    max_ddos = math.ceil(codeword_length_ddos/44)
    for tree in codewords_ddos:
      total_ternary_entries_ddos+=len(codewords_ddos[tree])
      total_ternary_blocks_ddos+= math.ceil(len(codewords_ddos[tree])/512)*max_ddos



    #Ternary-matching tables final summation
    total_ternary_blocks = total_ternary_blocks_app + total_ternary_blocks_ddos
    total_ternary_entries = total_ternary_entries_app + total_ternary_entries_ddos


  total_range_stages = math.ceil(total_range_TCAM_blocks/24)
  total_ternary_stages = math.ceil(total_ternary_blocks/24)

  return total_range_entries, total_range_TCAM_blocks, total_range_stages, total_ternary_entries, total_ternary_blocks, total_ternary_stages