INFINITE = (2**19)-1

def cleaner(dict_collection, key_to_delete):
  for i in range(len(key_to_delete)):
    for key in key_to_delete[i]:
      del dict_collection[i][key]

  return dict_collection

def split_coupler(selected_features, reference_model, target_model, features_to_couple, verbose=False):
  for tree_id in range(reference_model.n_estimators):
    collection_reference = [{} for i in range(len(selected_features))]
    collection_target = [{} for i in range(len(selected_features))]

    tree = reference_model.estimators_[tree_id]
    for i, threshold in enumerate(tree.tree_.threshold):
      if threshold != -2:
        collection_reference[tree.tree_.feature[i]][tree.tree_.threshold[i]] = INFINITE

    tree = target_model.estimators_[tree_id]
    for i, threshold in enumerate(tree.tree_.threshold):
      if threshold != -2:
        collection_target[tree.tree_.feature[i]][tree.tree_.threshold[i]] = INFINITE


    for i in range(len(collection_target)):
      for key_ddos in collection_target[i]:
        for key_app in collection_reference[i]:
          if key_ddos-key_app<collection_target[i][key_ddos] and key_ddos-key_app>=0:
            collection_target[i][key_ddos] = key_app

    reverse = [{} for i in range(len(selected_features))]
    key_to_delete = [[] for i in range(len(selected_features))]

    for i in range(len(collection_target)):
      for key, value in collection_target[i].items():
        if value in reverse[i]:
          reverse[i][value].append(key)
        else:
          reverse[i][value] = [key]

    for i in range(len(reverse)):
      for key in reverse[i]:
        if len(reverse[i][key]) >= 2:
          for j in range(len(reverse[i][key])):
            if reverse[i][key][j] != min(reverse[i][key]):
              key_to_delete[i].append(reverse[i][key][j])

    collection_target = cleaner(collection_target, key_to_delete)

    key_to_delete = [[] for i in range(len(selected_features))]

    for i in range(len(collection_target)):
      for key_ddos in collection_target[i]:
        if collection_target[i][key_ddos] == INFINITE:
          key_to_delete[i].append(key_ddos)

    collection_target = cleaner(collection_target, key_to_delete)

    key_to_delete = [[] for i in range(len(selected_features))]

    for i in range(len(collection_target)):
      for key in collection_target[i]:
        if key == collection_target[i][key]:
          key_to_delete[i].append(key)

    collection_target = cleaner(collection_target, key_to_delete)

    counter = 0

    tree = target_model.estimators_[tree_id]
    for i, threshold in enumerate(tree.tree_.threshold):
      if threshold != -2:
        if tree.tree_.threshold[i] in collection_target[tree.tree_.feature[i]]:
          if tree.tree_.feature[i] in features_to_couple:
            tree.tree_.threshold[i] = collection_target[tree.tree_.feature[i]][tree.tree_.threshold[i]]
            counter = counter + 1
    if verbose==True:
      print("Number of splits coupled: ", counter)

  return reference_model, target_model

def split_coupler_ensemble(features, model1, model2, verbose=False):
  model1, model2 = split_coupler(features, model1, model2, [i for i in range(int(len(features)/2))], verbose)
  model1, model2 = split_coupler(features, model2, model1, [i for i in range(int(len(features)/2), len(features))], verbose)

  return model1, model2