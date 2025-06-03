import statistics
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from train_model import train_classifier_RF
from build_p4_script import dt_thresholds_float_to_int
from dataset import load_dataset
from evaluation import memory_evaluation
import pickle
import matplotlib.pyplot as plt


selected_starting_features = ['Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s',  'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean',  'Fwd IAT Max',  'Fwd IAT Min',
    'Bwd IAT Mean',  'Bwd IAT Max',  'Bwd IAT Min',
    'Flow Packet Length Min', 'Flow Packet Length Max', 'Flow Packet Length Mean']

def feature_selection(df, task, num_final_features, num_cross_validations, num_trees, max_depth=-1):
    num_features_to_eliminate = len(list(df.columns)) - num_final_features
    result_columns_app = ['features', 'accuracy']
    result_columns_ddos = ['features', 'accuracy']

    if task == 'app':
      results_array = np.zeros(shape=(num_cross_validations, num_features_to_eliminate, len(result_columns_app)))
    elif task == 'ddos':
      results_array = np.zeros(shape=(num_cross_validations, num_features_to_eliminate, len(result_columns_ddos)))

    for i in range(num_cross_validations):
        print("num_cross_validation: " + str(i))
        if task=='app':
          X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Label']), df['Label'], stratify=df.Label, test_size=0.2, random_state=i)
          X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=i)
        else:
          X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Label']), df['Label'], test_size=0.2, random_state=i)
          X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=i)

        counter = 0
        num_features = len(list(X_train.columns))

        while num_features >= num_final_features:
            print("num_features: " + str(num_features) + ", num_final_features: " + str(num_final_features))
            clf = train_classifier_RF(X_train, y_train, num_trees, max_depth)
            acc = accuracy_score(y_test, clf.predict(X_test))
            #print(acc_tuple)

            if task == 'app':
              results_array[i, counter, :] = [num_features, acc]
            elif task == 'ddos':
              results_array[i, counter, :] = [num_features, acc]

            importance_results = permutation_importance(clf, X_val, y_val, scoring = 'accuracy', n_repeats=10, random_state=42, n_jobs=-1)
            importance = importance_results.importances_mean
            lowest_importance_ind = importance.argmin()
            feature_name = X_val.columns[lowest_importance_ind]

            X_train.drop(columns=[feature_name], inplace = True)
            X_test.drop(columns=[feature_name], inplace = True)
            X_val.drop(columns=[feature_name], inplace = True)

            num_features = len(list(X_train.columns))
            counter += 1


    if task == 'app' :
      results_df = pd.DataFrame(data = results_array.mean(axis=0), columns=result_columns_app)
    elif task == 'ddos':
      results_df = pd.DataFrame(data = results_array.mean(axis=0), columns=result_columns_ddos)
    return results_df, list(X_train.columns)

def chooseFeaturesDisjoint(df, task, trees, max_depth, features):
  _,selected_features = feature_selection(df, task, features, 1, trees, max_depth)
  #best_index = (results["accuracy"]).idxmax()
  #num_features = results.loc[best_index, 'features']
  #_,selected_features = feature_selection(df, task, int(num_features), 1, trees, max_depth=-1)
  return selected_features

def mtfs(df_app, df_ddos, selected_starting_features, trees, lambdas, num_cross_validations):
  result_df = pd.DataFrame(columns=['features', 'accuracy app', 'accuracy ddos'])
  lab_app = [0, 1, 2]
  lab_ddos = [1]

  X_app = df_app.drop(columns=['Label'])
  X_ddos = df_ddos.drop(columns=['Label'])

  y_app = df_app.Label
  y_ddos = df_ddos.Label



  features_importance = list(np.argsort(lambdas))

  sorted_features = [selected_starting_features[features_importance[i]] for i in range(len(features_importance))]
  print(sorted_features)

  for j in range(len(sorted_features)):
    print("Number of features under analysis: ", len(sorted_features[j:]))
    print("Features under analysis: ", sorted_features[j:])

    results_app = []
    results_ddos = []

    X_app = df_app[sorted_features[j:]]
    X_ddos = df_ddos[sorted_features[j:]]

    y_app = df_app.Label
    y_ddos = df_ddos.Label
    for i in range(num_cross_validations):
      print("Cross validation nb. ", i)
      # Split dataset into training set and test set
      X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app, test_size=0.2, stratify=y_app, random_state=i) # 80% training and 20% test
      X_train_ddos, X_test_ddos, y_train_ddos, y_test_ddos = train_test_split(X_ddos, y_ddos, test_size=0.2, random_state=i) # 80% training and 20% test


      clf_app = train_classifier_RF(X_train_app, y_train_app, trees)
      clf_ddos = train_classifier_RF(X_train_ddos, y_train_ddos, trees)

      #Predict the response for test dataset
      y_pred_app = clf_app.predict(X_test_app)
      y_pred_ddos = clf_ddos.predict(X_test_ddos)


      results_app.append(accuracy_score(y_test_app, y_pred_app))
      results_ddos.append(accuracy_score(y_test_ddos, y_pred_ddos))



    mean_app = statistics.mean(results_app)
    mean_ddos = statistics.mean(results_ddos)
    result_df.loc[len(result_df)] = [len(sorted_features[j:]), mean_app,  mean_ddos]

  return result_df

def chooseFeaturesJoint(num_features):
  lambdas = pickle.load(open("resorces/lambdas.pkl", "rb"))
  #result_df = pd.DataFrame(columns=['features', 'accuracy app', 'accuracy ddos'])
  #results = mtfs(df_app, df_ddos, selected_starting_features, trees, lambdas, 3)
  #best_index = (results["accuracy app"] + results["accuracy ddos"]).idxmax()
  #num_features = results.loc[best_index, 'features']
  features_importance = list(np.argsort(lambdas))
  sorted_features = [selected_starting_features[features_importance[i]] for i in range(len(features_importance)-int(num_features), len(features_importance))]

  return sorted_features

#datasets
INFINITE = (2**19)-1
threshold = (2**19)-2

datasets_path = "resources/"

df_app = load_dataset(datasets_path, 'apps_flow_features.csv', threshold)[selected_starting_features + ['Label']]
df_ddos = load_dataset(datasets_path, 'Wednesday-workingHours.pcap_ISCX.csv', threshold)[selected_starting_features+ ['Label']]

y_app = df_app.Label
y_ddos = df_ddos.Label

#Set of trees under analysis
tree_sets = [i for i in range(3, 6, 2)]
result_df_no_shar = pd.DataFrame(columns=['trees', 'max_depth', 'features',
                                  'mean accuracy app', 'std accuracy app', 'mean accuracy ddos', 'std accuracy ddos',
                                  'mean entries range', 'std entries range', 'mean entries ternary', 'std entries ternary',
                                  'mean TCAM range', 'std TCAM range', 'mean TCAM ternary', 'std TCAM ternary',
                                  'mean stages range', 'std stages range', 'mean stages ternary', 'std stages ternary'])

result_df_shar = pd.DataFrame(columns=['trees', 'max_depth', 'features',
                                  'mean accuracy app', 'std accuracy app', 'mean accuracy ddos', 'std accuracy ddos',
                                  'mean entries range', 'std entries range', 'mean entries ternary', 'std entries ternary',
                                  'mean TCAM range', 'std TCAM range', 'mean TCAM ternary', 'std TCAM ternary',
                                  'mean stages range', 'std stages range', 'mean stages ternary', 'std stages ternary'])

for i in range(len(tree_sets)):
  for max_depth in [5, 10]:
    for features in [5, 10]:
      print("Tree set: {}, max_depth: {}, features: {}".format(tree_sets[i], max_depth, features))
      #Choose the best nb of features for disjoint models
      features_app = chooseFeaturesDisjoint(df_app, "app", tree_sets[i], max_depth, features)
      features_ddos = chooseFeaturesDisjoint(df_ddos, "ddos", tree_sets[i], max_depth, features)

      #Choose the best number of features for joint models
      features_joint = chooseFeaturesJoint(features)
      #print(features_joint)
      #Shrinking datasets
      df_app_no_joint = df_app[features_app]
      df_ddos_no_joint = df_ddos[features_ddos]

      df_app_joint = df_app[features_joint]
      df_ddos_joint = df_ddos[features_joint]

      #Results storage data
      #Disjoint case study
      accuracy_app_no_joint_vect = []
      accuracy_ddos_no_joint_vect = []
      total_range_entries_no_joint_vect = []
      total_range_TCAM_no_joint_vect = []
      total_range_stages_no_joint_vect = []
      total_ternary_entries_no_joint_vect = []
      total_ternary_TCAM_no_joint_vect = []
      total_ternary_stages_no_joint_vect = []


      #Joint case study
      accuracy_app_joint_vect = []
      accuracy_ddos_joint_vect = []
      total_range_entries_joint_vect = []
      total_range_TCAM_joint_vect = []
      total_range_stages_joint_vect = []
      total_ternary_entries_joint_vect = []
      total_ternary_TCAM_joint_vect = []
      total_ternary_stages_joint_vect = []

      for j in range(10):
        #Disjoint case study
        # Split dataset into training set and test set
        X_train_app_no_joint, X_test_app_no_joint, y_train_app_no_joint, y_test_app_no_joint = train_test_split(df_app_no_joint, y_app, test_size=0.2, stratify=y_app, random_state=j)
        X_train_ddos_no_joint, X_test_ddos_no_joint, y_train_ddos_no_joint, y_test_ddos_no_joint = train_test_split(df_ddos_no_joint, y_ddos, test_size=0.2, random_state=j)


        clf_app_no_joint = RandomForestClassifier(n_estimators=tree_sets[i], max_depth=max_depth, min_impurity_decrease=0.015)
        clf_ddos_no_joint = RandomForestClassifier(n_estimators=tree_sets[i], max_depth=max_depth, min_impurity_decrease=0.015)

        clf_app_no_joint.fit(X_train_app_no_joint, y_train_app_no_joint)
        clf_ddos_no_joint.fit(X_train_ddos_no_joint, y_train_ddos_no_joint)


        #Turning classifiers thresholds into integers
        clf_app_no_joint = dt_thresholds_float_to_int(clf_app_no_joint)
        clf_ddos_no_joint = dt_thresholds_float_to_int(clf_ddos_no_joint)

        #Predict the response for test dataset
        y_pred_test_app_no_joint = clf_app_no_joint.predict(X_test_app_no_joint)
        y_pred_test_ddos_no_joint = clf_ddos_no_joint.predict(X_test_ddos_no_joint)

        accuracy_app_no_joint_vect.append(accuracy_score(y_test_app_no_joint, y_pred_test_app_no_joint))
        accuracy_ddos_no_joint_vect.append(accuracy_score(y_test_ddos_no_joint, y_pred_test_ddos_no_joint))


        total_range_entries_no_joint, total_range_TCAM_no_joint, total_range_stages_no_joint, total_ternary_entries_no_joint, total_ternary_TCAM_no_joint, total_ternary_stages_no_joint = memory_evaluation(
                                                                                                                                                                                                              clf_app_no_joint,
                                                                                                                                                                                                              clf_ddos_no_joint,
                                                                                                                                                                                                              features_app,
                                                                                                                                                                                                              features_ddos,
                                                                                                                                                                                                              encoding='disjoint'
                                                                                                                                                                                                              )
        total_range_entries_no_joint_vect.append(total_range_entries_no_joint)
        total_range_TCAM_no_joint_vect.append(total_range_TCAM_no_joint)
        total_range_stages_no_joint_vect.append(total_range_stages_no_joint)
        total_ternary_entries_no_joint_vect.append(total_ternary_entries_no_joint)
        total_ternary_TCAM_no_joint_vect.append(total_ternary_TCAM_no_joint)
        total_ternary_stages_no_joint_vect.append(total_ternary_stages_no_joint)


        #Joint case study
        # Split dataset into training set and test set
        X_train_app_joint, X_test_app_joint, y_train_app_joint, y_test_app_joint = train_test_split(df_app_joint, y_app, stratify=y_app, test_size=0.2, random_state=j)
        X_train_ddos_joint, X_test_ddos_joint, y_train_ddos_joint, y_test_ddos_joint = train_test_split(df_ddos_joint, y_ddos, test_size=0.2, random_state=j)

        clf_app_joint =  RandomForestClassifier(n_estimators=tree_sets[i], max_depth=max_depth, min_impurity_decrease=0.015)
        clf_ddos_joint = RandomForestClassifier(n_estimators=tree_sets[i], max_depth=max_depth, min_impurity_decrease=0.015)

        clf_app_joint.fit(X_train_app_joint, y_train_app_joint)
        clf_ddos_joint.fit(X_train_ddos_joint, y_train_ddos_joint)


        #Turning classifiers thresholds into integers
        clf_app_joint = dt_thresholds_float_to_int(clf_app_joint)
        clf_ddos_joint = dt_thresholds_float_to_int(clf_ddos_joint)

        #clf_app_joint, clf_ddos_joint = split_coupler_ensemble(features_joint, clf_app_joint, clf_ddos_joint)

        #Predict the response for test dataset
        y_pred_test_app_joint = clf_app_joint.predict(X_test_app_joint)
        y_pred_test_ddos_joint = clf_ddos_joint.predict(X_test_ddos_joint)

        accuracy_app_joint_vect.append(accuracy_score(y_test_app_joint, y_pred_test_app_joint))
        accuracy_ddos_joint_vect.append(accuracy_score(y_test_ddos_joint, y_pred_test_ddos_joint))

        total_range_entries_joint, total_range_TCAM_joint, total_range_stages_joint, total_ternary_entries_joint, total_ternary_TCAM_joint, total_ternary_stages_joint = memory_evaluation(
                                                                                                                                                                                            clf_app_joint,
                                                                                                                                                                                            clf_ddos_joint,
                                                                                                                                                                                            features_joint,
                                                                                                                                                                                            features_joint,
                                                                                                                                                                                            encoding='joint'
                                                                                                                                                                                            )
        total_range_entries_joint_vect.append(total_range_entries_joint)
        total_range_TCAM_joint_vect.append(total_range_TCAM_joint)
        total_range_stages_joint_vect.append(total_range_stages_joint)
        total_ternary_entries_joint_vect.append(total_ternary_entries_joint)
        total_ternary_TCAM_joint_vect.append(total_ternary_TCAM_joint)
        total_ternary_stages_joint_vect.append(total_ternary_stages_joint)

      #Results data
      #Disjoint case study
      mean_accuracy_app_no_joint = np.mean(accuracy_app_no_joint_vect)
      std_accuracy_app_no_joint = np.std(accuracy_app_no_joint_vect)
      mean_accuracy_ddos_no_joint = np.mean(accuracy_ddos_no_joint_vect)
      std_accuracy_ddos_no_joint = np.std(accuracy_ddos_no_joint_vect)
      mean_total_range_entries_no_joint = np.mean(total_range_entries_no_joint_vect)
      std_total_range_entries_no_joint = np.std(total_range_entries_no_joint_vect)
      mean_total_range_TCAM_no_joint = np.mean(total_range_TCAM_no_joint_vect)
      std_total_range_TCAM_no_joint = np.std(total_range_TCAM_no_joint_vect)
      mean_total_range_stages_no_joint = np.mean(total_range_stages_no_joint_vect)
      std_total_range_stages_no_joint = np.std(total_range_stages_no_joint_vect)
      mean_total_ternary_entries_no_joint = np.mean(total_ternary_entries_no_joint_vect)
      std_total_ternary_entries_no_joint = np.std(total_ternary_entries_no_joint_vect)
      mean_total_ternary_TCAM_no_joint = np.mean(total_ternary_TCAM_no_joint_vect)
      std_total_ternary_TCAM_no_joint = np.std(total_ternary_TCAM_no_joint_vect)
      mean_total_ternary_stages_no_joint = np.mean(total_ternary_stages_no_joint_vect)
      std_total_ternary_stages_no_joint = np.std(total_ternary_stages_no_joint_vect)
      result_df_no_shar.loc[len(result_df_no_shar)] = [tree_sets[i], max_depth, features, mean_accuracy_app_no_joint, std_accuracy_app_no_joint, mean_accuracy_ddos_no_joint, std_accuracy_ddos_no_joint,
                                                      mean_total_range_entries_no_joint, std_total_range_entries_no_joint, mean_total_ternary_entries_no_joint, std_total_ternary_entries_no_joint,
                                                      mean_total_range_TCAM_no_joint, std_total_range_TCAM_no_joint, mean_total_ternary_TCAM_no_joint, std_total_ternary_TCAM_no_joint,
                                                      mean_total_range_stages_no_joint, std_total_range_stages_no_joint, mean_total_ternary_stages_no_joint, std_total_ternary_stages_no_joint]

      #Joint case study
      mean_accuracy_app_joint = np.mean(accuracy_app_joint_vect)
      std_accuracy_app_joint = np.std(accuracy_app_joint_vect)
      mean_accuracy_ddos_joint = np.mean(accuracy_ddos_joint_vect)
      std_accuracy_ddos_joint = np.std(accuracy_ddos_joint_vect)
      mean_total_range_entries_joint = np.mean(total_range_entries_joint_vect)
      std_total_range_entries_joint = np.std(total_range_entries_joint_vect)
      mean_total_range_TCAM_joint = np.mean(total_range_TCAM_joint_vect)
      std_total_range_TCAM_joint = np.std(total_range_TCAM_joint_vect)
      mean_total_range_stages_joint = np.mean(total_range_stages_joint_vect)
      std_total_range_stages_joint = np.std(total_range_stages_joint_vect)
      mean_total_ternary_entries_joint = np.mean(total_ternary_entries_joint_vect)
      std_total_ternary_entries_joint = np.std(total_ternary_entries_joint_vect)
      mean_total_ternary_TCAM_joint = np.mean(total_ternary_TCAM_joint_vect)
      std_total_ternary_TCAM_joint = np.std(total_ternary_TCAM_joint_vect)
      mean_total_ternary_stages_joint = np.mean(total_ternary_stages_joint_vect)
      std_total_ternary_stages_joint = np.std(total_ternary_stages_joint_vect)
      result_df_shar.loc[len(result_df_shar)] = [tree_sets[i], max_depth, features, mean_accuracy_app_joint, std_accuracy_app_joint, mean_accuracy_ddos_joint, std_accuracy_ddos_joint,
                                                      mean_total_range_entries_joint, std_total_range_entries_joint, mean_total_ternary_entries_joint, std_total_ternary_entries_joint,
                                                      mean_total_range_TCAM_joint, std_total_range_TCAM_joint, mean_total_ternary_TCAM_joint, std_total_ternary_TCAM_joint,
                                                      mean_total_range_stages_joint, std_total_range_stages_joint, mean_total_ternary_stages_joint, std_total_ternary_stages_joint]



#Accuracy
plt.figure(figsize=(8, 7))
plt.title("Accuracy comparison")
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean accuracy app'], yerr=result_df_no_shar['std accuracy app'], label='accuracy traffic classifier no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean accuracy app'], yerr=result_df_shar['std accuracy app'], label='accuracy traffic classifier sharing')
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean accuracy ddos'], yerr=result_df_no_shar['std accuracy ddos'], label='accuracy DDoS detector no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean accuracy ddos'], yerr=result_df_shar['std accuracy ddos'], label='accuracy DDoS detector sharing')
plt.legend(loc='lower right')

#Entries
plt.figure(figsize=(8, 7))
plt.title("Entries")
plt.xlabel("Number of trees")
plt.ylabel("Average number of entries")
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean entries range'], yerr=result_df_no_shar['std entries range'], label='mean entries range no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean entries range'], yerr=result_df_shar['std entries range'], label='mean entries range sharing')
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean entries ternary'], yerr=result_df_no_shar['std entries ternary'], label='mean entries ternary no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean entries ternary'], yerr=result_df_shar['std entries ternary'], label='mean entries ternary sharing')
plt.legend(loc='upper left')

#TCAMs
plt.figure(figsize=(8, 7))
plt.title("TCAMs")
plt.xlabel("Number of trees")
plt.ylabel("Average number of TCAMs")
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean TCAM range'], yerr=result_df_no_shar['std TCAM range'], label='mean TCAM range no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean TCAM range'], yerr=result_df_shar['std TCAM range'], label='mean TCAM range sharing')
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean TCAM ternary'], yerr=result_df_no_shar['std TCAM ternary'], label='mean TCAM ternary no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean TCAM ternary'], yerr=result_df_shar['std TCAM ternary'], label='mean TCAM ternary sharing')
plt.legend(loc='upper left')

#stages
plt.figure(figsize=(8, 7))
plt.title("Stages")
plt.xlabel("Number of trees")
plt.ylabel("Average number of stages")
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean stages range'], yerr=result_df_no_shar['std stages range'], label='mean stages range no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean stages range'], yerr=result_df_shar['std stages range'], label='mean stages range sharing')
plt.errorbar(result_df_no_shar['trees'], result_df_no_shar['mean stages ternary'], yerr=result_df_no_shar['std stages ternary'], label='mean stages ternary no sharing')
plt.errorbar(result_df_shar['trees'], result_df_shar['mean stages ternary'], yerr=result_df_shar['std stages ternary'], label='mean stages ternary sharing')
plt.legend(loc='upper left')