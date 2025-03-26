from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import load_dataset
from build_p4_script import *
from evaluation import memory_evaluation

datasets_path = "resources/"
threshold = (2**19)-2
df_app = load_dataset(datasets_path, 'apps_flow_features.csv', threshold)
df_ddos = load_dataset(datasets_path, 'Wednesday-workingHours.pcap_ISCX.csv', threshold)


#511
features_app = ['Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd IAT Mean',
                'Flow Packet Length Max', 'Fwd Packet Length Max', 'Bwd IAT Max', 'Fwd IAT Max', 'Flow Packet Length Mean', 'Fwd IAT Mean',
                'Bwd Packet Length Mean', 'Flow IAT Mean', 'Bwd Packet Length Min', 'Flow IAT Max', 'Label']
features_ddos = ['Fwd IAT Mean', 'Fwd Packet Length Max', 'Flow Packet Length Mean', 'Flow IAT Max', 'Fwd IAT Max', 'Bwd Packet Length Max', 'Flow Packet Length Max', 'Label']
features_joint = ['Fwd Packet Length Max', 'Flow Packet Length Mean', 'Flow IAT Max', 'Fwd IAT Max', 'Bwd Packet Length Max', 'Flow Packet Length Max', 'Label']
n_estimators_app_joint = 1
n_estimators_ddos_joint = 1
max_depth_app_joint = 14
max_depth_ddos_joint = 5
n_estimators_app = 1
n_estimators_ddos = 3
max_depth_app = 14
max_depth_ddos = 11


"""
#255
features_app = ['Flow Packet Length Max', 'Fwd Packet Length Max', 'Bwd IAT Max', 'Fwd IAT Max', 'Flow Packet Length Mean', 'Fwd IAT Mean', 'Bwd Packet Length Mean', 'Flow IAT Mean', 'Bwd Packet Length Min', 'Flow IAT Max', 'Label']
features_ddos = ['Fwd IAT Mean', 'Fwd Packet Length Max', 'Flow Packet Length Mean', 'Flow IAT Max', 'Fwd IAT Max', 'Bwd Packet Length Max', 'Flow Packet Length Max', 'Label']
features_joint = ['Fwd Packet Length Mean', 'Bwd IAT Mean', 'Flow Packets/s', 'Bwd IAT Max', 'Bwd Packet Length Mean', 'Fwd IAT Mean', 'Bwd Packet Length Min', 'Flow IAT Mean', 'Fwd Packet Length Max', 'Flow Packet Length Mean', 'Flow IAT Max', 'Fwd IAT Max', 'Bwd Packet Length Max', 'Flow Packet Length Max', 'Label']
n_estimators_app_joint = 5
n_estimators_ddos_joint = 1
max_depth_app_joint = 5
max_depth_ddos_joint = 14
n_estimators_app = 1
n_estimators_ddos = 3
max_depth_app = 8
max_depth_ddos = 8
"""


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
disjoint_width_distribution = {}

#Joint case study
accuracy_app_joint_vect = []
accuracy_ddos_joint_vect = []
total_range_entries_joint_vect = []
total_range_TCAM_joint_vect = []
total_range_stages_joint_vect = []
total_ternary_entries_joint_vect = []
total_ternary_TCAM_joint_vect = []
total_ternary_stages_joint_vect = []
joint_width_distribution = {}

for i in range(100):

  #Disjoint case study
  # Split dataset into training set and test set
  X_train_app_no_joint, X_test_app_no_joint, y_train_app_no_joint, y_test_app_no_joint = train_test_split(df_app_no_joint.drop(columns=['Label']), df_app_no_joint.Label, test_size=0.2, stratify=df_app_no_joint.Label, random_state=i)
  X_train_ddos_no_joint, X_test_ddos_no_joint, y_train_ddos_no_joint, y_test_ddos_no_joint = train_test_split(df_ddos_no_joint.drop(columns=['Label']), df_ddos_no_joint.Label, test_size=0.2, random_state=i)

  clf_app_no_joint = RandomForestClassifier(n_estimators=n_estimators_app, max_depth=max_depth_app)
  clf_ddos_no_joint = RandomForestClassifier(n_estimators=n_estimators_ddos, max_depth=max_depth_ddos)
  #clf_app_no_joint = train_classifier_RF(X_train_app_no_joint, y_train_app_no_joint, n_estimators_app, max_depth_app)
  #clf_ddos_no_joint = train_classifier_RF(X_train_ddos_no_joint, y_train_ddos_no_joint, n_estimators_ddos, max_depth_ddos)

  # Train Random Forest Classifer
  clf_app_no_joint.fit(X_train_app_no_joint,y_train_app_no_joint)
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
                                                                                                                                                                                                        features_app[:-1],
                                                                                                                                                                                                        features_ddos[:-1],
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
  X_train_app_joint, X_test_app_joint, y_train_app_joint, y_test_app_joint = train_test_split(df_app_joint.drop(columns=['Label']), df_app_joint.Label, stratify=df_app_joint.Label, test_size=0.2, random_state=i)
  X_train_ddos_joint, X_test_ddos_joint, y_train_ddos_joint, y_test_ddos_joint = train_test_split(df_ddos_joint.drop(columns=['Label']), df_ddos_joint.Label, test_size=0.2, random_state=i)

  clf_app_joint = RandomForestClassifier(n_estimators=n_estimators_app_joint, max_depth=max_depth_app_joint)
  clf_ddos_joint = RandomForestClassifier(n_estimators=n_estimators_ddos_joint, max_depth=max_depth_ddos_joint)
  #clf_app_joint = train_classifier_RF(X_train_app_joint, y_train_app_joint, n_estimators_app_joint, max_depth_app_joint)
  #clf_ddos_joint = train_classifier_RF(X_train_ddos_joint, y_train_ddos_joint, n_estimators_ddos_joint, max_depth_ddos_joint)

  # Train Random Forest Classifer
  clf_app_joint.fit(X_train_app_joint,y_train_app_joint)
  clf_ddos_joint.fit(X_train_ddos_joint, y_train_ddos_joint)

  #Turning classifiers thresholds into integers
  clf_app_joint = dt_thresholds_float_to_int(clf_app_joint)
  clf_ddos_joint = dt_thresholds_float_to_int(clf_ddos_joint)

  #Predict the response for test dataset
  y_pred_test_app_joint = clf_app_joint.predict(X_test_app_joint)
  y_pred_test_ddos_joint = clf_ddos_joint.predict(X_test_ddos_joint)

  accuracy_app_joint_vect.append(accuracy_score(y_test_app_joint, y_pred_test_app_joint))
  accuracy_ddos_joint_vect.append(accuracy_score(y_test_ddos_joint, y_pred_test_ddos_joint))

  total_range_entries_joint, total_range_TCAM_joint, total_range_stages_joint, total_ternary_entries_joint, total_ternary_TCAM_joint, total_ternary_stages_joint = memory_evaluation(
                                                                                                                                                                                      clf_app_joint,
                                                                                                                                                                                      clf_ddos_joint,
                                                                                                                                                                                      features_joint[:-1],
                                                                                                                                                                                      features_joint[:-1],
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

print('----------------------512-bit codewords------------------------')
print("Mean accuracy traffic flows classifier without sharing: ", round(mean_accuracy_app_no_joint*100, 2), "%")
print("Mean accuracy traffic flows classifier with sharing: ", round(mean_accuracy_app_joint*100, 2), "%")
print("Mean accuracy DDOS detector without sharing: ", round(mean_accuracy_ddos_no_joint*100, 2),"%")
print("Mean accuracy DDOS detector with sharing: ", round(mean_accuracy_ddos_joint*100, 2), "%")
print('---------------------------------------------------------------')

import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

categories_range = ["no feature sharing", "feature sharing"]
categories_tern = ["no feature sharing ", "feature sharing "]

#512
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

y_range = [mean_total_range_entries_no_joint,  mean_total_range_entries_joint]
y_tern = [mean_total_ternary_entries_no_joint, mean_total_ternary_entries_joint]


plt.bar(categories_range, y_range)
plt.bar(categories_tern, y_tern)

#err_range = [std_total_range_entries_no_joint, std_total_range_entries_joint]
#err_tern = [std_total_ternary_entries_no_joint, std_total_ternary_entries_joint]
plt.xticks(rotation=0)
legend_patches = [
    mpatches.Patch(color="tab:blue", label="Range-matching entries"),
    mpatches.Patch(color="tab:orange", label="Ternary-matching entries"),
]
plt.legend(handles=legend_patches, loc="upper right", fontsize=16)

#plt.errorbar(categories_range, y_range, yerr=err_range, fmt="o", color="r")
#plt.errorbar(categories_tern, y_tern, yerr=err_tern, fmt="o", color="r")
#plt.title('512-bit codewords', fontsize=20)
plt.ylabel('Average number of entries', fontsize=20)
plt.ylim(0,1600)

plt.show()

#512
plt.figure(figsize=(13, 7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# making plot
y_range = [mean_total_range_TCAM_no_joint,  mean_total_range_TCAM_joint]
y_tern = [mean_total_ternary_TCAM_no_joint, mean_total_ternary_TCAM_joint]

# Plot scatter here
plt.bar(categories_range, y_range)
plt.bar(categories_tern, y_tern)

#err_range = [std_total_range_TCAM_no_joint, std_total_range_TCAM_joint]
#err_tern = [std_total_ternary_TCAM_no_joint, std_total_ternary_TCAM_joint]

plt.xticks(rotation=0)
legend_patches = [
    mpatches.Patch(color="tab:blue", label="Range-matching TCAMs"),
    mpatches.Patch(color="tab:orange", label="Ternary-matching TCAMs"),
]
plt.legend(handles=legend_patches, loc="upper right", fontsize=16)

#plt.errorbar(categories_range, y_range, yerr=err_range, fmt="o", color="r")
#plt.errorbar(categories_tern, y_tern, yerr=err_tern, fmt="o", color="r")
#plt.title('512-bit codewords')
plt.ylabel('Average number of TCAMs', fontsize=20)
plt.ylim(0,100)
plt.show()

#512
plt.figure(figsize=(13, 7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# making plot
y_range = [mean_total_range_stages_no_joint,  mean_total_range_stages_joint]
y_tern = [mean_total_ternary_stages_no_joint, mean_total_ternary_stages_joint]

# Plot scatter here
plt.bar(categories_range, y_range)
plt.bar(categories_tern, y_tern)

#err_range = [std_total_range_stages_no_joint, std_total_range_stages_joint]
#err_tern = [std_total_ternary_stages_no_joint, std_total_ternary_stages_joint]

plt.xticks(rotation=0)
legend_patches = [
    mpatches.Patch(color="tab:blue", label="Range-matching stages"),
    mpatches.Patch(color="tab:orange", label="Ternary-matching stages"),
]
plt.legend(handles=legend_patches, loc="upper right", fontsize=20)

#plt.errorbar(categories_range, y_range, yerr=err_range, fmt="o", color="r")
#plt.errorbar(categories_tern, y_tern, yerr=err_tern, fmt="o", color="r")
#plt.title('512-bit codewords')
plt.ylabel('Average number of stages',fontsize=20)
plt.ylim(0,6)
plt.show()

categories = ["no feature sharing", "feature sharing"]

plt.figure(figsize=(7, 6))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
y = [mean_total_range_entries_no_joint + mean_total_ternary_entries_no_joint, mean_total_range_entries_joint + mean_total_ternary_entries_joint]
plt.bar(categories, y, color="tab:blue")

#err = [std_total_range_entries_no_joint + std_total_ternary_entries_no_joint, std_total_range_entries_joint + std_total_ternary_entries_joint]

#plt.errorbar(categories, y, yerr=err, fmt="o", color="r")
#plt.title('512-bit codewords')
#plt.ylabel('Average number of entries', fontsize=20)
plt.ylim(0,1600)
plt.show()

#512
plt.figure(figsize=(7, 6))
y = [mean_total_range_TCAM_no_joint + mean_total_ternary_TCAM_no_joint, mean_total_range_TCAM_joint + mean_total_ternary_TCAM_joint]
plt.bar(categories, y, color="tab:blue")

err = [std_total_range_TCAM_no_joint + std_total_ternary_TCAM_no_joint, std_total_range_TCAM_joint + std_total_ternary_TCAM_joint]
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#plt.errorbar(categories, y, yerr=err, fmt="o", color="r")
#plt.title('512-bit codewords', fontsize=18)
#plt.ylabel('Average number of TCAMs', fontsize=18)
plt.ylim(0,100)
plt.show()


#512
plt.figure(figsize=(7, 6))
y = [mean_total_range_stages_no_joint + mean_total_ternary_stages_no_joint, mean_total_range_stages_joint + mean_total_ternary_stages_joint]
plt.bar(categories, y, color="tab:blue")

#err = [std_total_range_stages_no_joint + std_total_ternary_stages_no_joint, std_total_range_stages_joint + std_total_ternary_stages_joint]

#plt.errorbar(categories, y, yerr=err, fmt="o", color="r")
#plt.title('512-bit codewords', size=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#plt.ylabel('Average number of stages', size=18)
plt.ylim(0,6)
plt.show()