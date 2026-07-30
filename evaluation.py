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


def range_matching_resource_usage(feature_intervals):
  range_entries, range_blocks = 0, 0

  for feature in feature_intervals:
    range_entries_feature = 0

    for interv in feature_intervals[feature]:
      width = interv[1] - interv[0] + 1

      if width > 1:
        range_entries_feature += 2 * math.floor(math.log2(width))
        #print('{} TCAM entries for range {}-{}'.format(2 * math.floor(math.log2(width)), interv[0], interv[1]))
      else:
        range_entries_feature += 1

    range_entries += range_entries_feature
    range_blocks += math.ceil(range_entries_feature / RANGE_MATCHING_ENTRIES_PER_BLOCK)

  return range_entries, range_blocks
  

def ternary_matching_resource_usage(codewords):

  ternary_entries, ternary_blocks = 0, 0
  codeword_length = len(next(iter(codewords[0].items()))[0])

  if codeword_length > MAX_CODEWORD_LENGTH:
    raise RuntimeError("Codewords are too long", codeword_length)
  
  factor = math.ceil(codeword_length / TCAM_BLOCK_KEY_LENGTH)
  for tree in codewords:
    ternary_entries += len(codewords[tree])
    ternary_blocks += math.ceil(len(codewords[tree]) / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor

    #print('{} TCAM entries for {} codewords of length {}'.format(math.ceil(len(codewords[tree]) / TERNARY_MATCHING_ENTRIES_PER_BLOCK) * factor, len(codewords[tree]), codeword_length))

  return ternary_entries, ternary_blocks


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
  ternary_entries, ternary_blocks = ternary_matching_resource_usage(codewords)
  
  return (range_entries, range_blocks, ternary_entries, ternary_blocks)


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
    ternary_entries, ternary_blocks = ternary_matching_resource_usage(codewords)

  elif encoding == 'disjoint':

    range_entries_app, range_blocks_app, ternary_entries_app, ternary_blocks_app = single_model_memory_evaluation(clf_app, selected_features_app)
    range_entries_ddos, range_blocks_ddos, ternary_entries_ddos, ternary_blocks_ddos = single_model_memory_evaluation(clf_ddos, selected_features_ddos)

    range_blocks = range_blocks_app + range_blocks_ddos
    range_entries = range_entries_app + range_entries_ddos

    #Ternary-matching tables final summation
    ternary_blocks = ternary_blocks_app + ternary_blocks_ddos
    ternary_entries = ternary_entries_app + ternary_entries_ddos

  range_stages = math.ceil(range_blocks / TCAM_BLOCKS_PER_STAGE)
  ternary_stages = math.ceil(ternary_blocks / TCAM_BLOCKS_PER_STAGE)

  return range_stages + ternary_stages, range_blocks + ternary_blocks
