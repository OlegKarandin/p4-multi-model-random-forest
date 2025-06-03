import numpy as np
import pandas as pd

def ddos_labels(x):
  if x == "No Attack":
    return 0
  else:
    return 1

def load_dataset(path, filename, threshold=None, verbose=False):
	# 1. LOAD DATASET, PREPROCESSING

	# Load dataset from .csv"
	print("Loading Dataset...")
	df = pd.read_csv(path + filename)
	cicids2017_df = pd.DataFrame.from_dict(df)

	# Remove leading spaces from column names
	print("Changing column names...")
	if filename == 'Wednesday-workingHours.pcap_ISCX.csv':
		new_cols = {col: col.strip() for col in cicids2017_df.columns}
	elif filename == 'apps_flow_features.csv':
		new_cols = {col: col.strip().replace('s.s', 's/s').replace('.', ' ') for col in cicids2017_df.columns}
	else:
		return None
	cicids2017_df.rename(columns=new_cols, inplace=True)

	# Check if there are any NaN values in the DataFrame
	print("Removing rows containing NaN values...")
	if cicids2017_df.isna().any().any():
	    print('The DataFrame contains NaN values')
	else:
	    print('The DataFrame does not contain NaN values')

	# Count the number of rows in the original DataFrame
	num_rows_before = cicids2017_df.shape[0]

	# Remove rows with NaN values
	cicids2017_df.dropna(inplace=True)

	# Count the number of rows in the new DataFrame
	num_rows_after = cicids2017_df.shape[0]

	# Calculate the number of rows removed
	num_rows_removed = num_rows_before - num_rows_after

	# Print the number of rows removed
	print(f'Removed {num_rows_removed} rows containing NaN values')

	if filename == 'Wednesday-workingHours.pcap_ISCX.csv':
		# Drop rows belonging to class HEARTBLEED
		print("Remove 'Heartbleed' attack samples...")
		cicids2017_df = cicids2017_df.drop(cicids2017_df[cicids2017_df['Label'] == 'Heartbleed'].index)

		label_counts = cicids2017_df['Label'].value_counts()
		if verbose==True:
			# print the counts of each label
			print(label_counts)
			# Use the `plot.pie()` method to plot a pie chart of the class distribution
			label_counts.plot.pie(autopct='%1.1f%%', startangle=90)

			# Add a title to the pie chart
			plt.title('Class Distribution')

			# Display the pie chart
			plt.show()

	#2. FEATURE SELECTION
	cicids2017_df['Packet Count'] = cicids2017_df['Total Fwd Packets'] + cicids2017_df['Total Backward Packets']
	cicids2017_df['Packet Length Total'] = cicids2017_df['Total Length of Fwd Packets'] + cicids2017_df['Total Length of Bwd Packets']

	selected_features = ['Flow Duration',
	                     'Packet Count',
	                     'Flow IAT Max',
	                     'Flow IAT Min',
	                     'Flow IAT Mean',
	                     'Min Packet Length',
	                     'Max Packet Length',
	                     'Packet Length Mean',
	                     'Packet Length Total',
						 #'Flow Packets/s',
						 #'Flow Bytes/s',
	                     ###FORWARD FEATURES
	                     'Total Fwd Packets',
	                     'Total Length of Fwd Packets',
	                     'Fwd Packet Length Max',
	                     'Fwd Packet Length Min',
	                     'Fwd Packet Length Mean',
	                     'Fwd IAT Total',
	                     'Fwd IAT Mean',
	                     'Fwd IAT Max',
	                     'Fwd IAT Min',
	                     'Fwd Header Length',
	                     ###BACKWARD FEATURES
	                      'Total Backward Packets',
	                      'Total Length of Bwd Packets',
	                      'Bwd Packet Length Max',
	                      'Bwd Packet Length Min',
	                      'Bwd Packet Length Mean',
	                      'Bwd IAT Total',
	                      'Bwd IAT Mean',
	                      'Bwd IAT Max',
	                      'Bwd IAT Min',
	                      'Bwd Header Length',
	                      'Label'
	                    ]
	if filename == 'apps_flow_features.csv':
		protocols = df['ProtocolName'].to_numpy()
		selected_features = selected_features[:-1]

	num_selected_features = len(selected_features)
	if filename == 'Wednesday-workingHours.pcap_ISCX.csv':
		print(f"Selected Features: {num_selected_features-1}")
	elif filename == 'apps_flow_features.csv':
		print(f"Selected Features: {num_selected_features}")

	cicids2017_df = cicids2017_df[selected_features]

	for col in cicids2017_df.columns:
		if col == "Packet Count":
			new_cols[col] = "Flow Packet Count"
		if col == 'Min Packet Length':
			new_cols[col] = "Flow Packet Length Min"
		if col == 'Max Packet Length':
			new_cols[col] = "Flow Packet Length Max"
		if col == 'Packet Length Mean':
			new_cols[col] = "Flow Packet Length Mean"
		if col == 'Packet Length Total':
			new_cols[col] = "Flow Packet Length Total"


		if col == "Total Fwd Packets":
			new_cols[col] = "Fwd Packet Count"
		if col == 'Total Length of Fwd Packets':
			new_cols[col] = "Fwd Packet Length Total"
		if col == 'act_data_pkt_fwd':
			new_cols[col] = "Fwd ACT Data Pkt"
		if col == 'min_seg_size_forward':
			new_cols[col] = "Fwd Min Segment Size"

		if col == "Total Backward Packets":
			new_cols[col] = "Bwd Packet Count"
		if col == 'Total Length of Bwd Packets':
			new_cols[col] = "Bwd Packet Length Total"


	cicids2017_df.rename(columns=new_cols, inplace=True)

	# 3. HANDLE FEAUTRES WITH INVALID VALUES
	df = cicids2017_df
	numeric_columns = df.select_dtypes(include=['number']).columns
	negative_features = df[numeric_columns].columns[(df[numeric_columns] < 0).any()]

	print("Numeric features with negative values:")
	for feature in negative_features:
	    print(feature)

	df_negative_values = cicids2017_df[(cicids2017_df[numeric_columns] < 0).any(axis=1)]

	#Delete remaining flow samples with negative values
	initial_flows = len(cicids2017_df)

	#Check only in numeric columns:
	numeric_columns = cicids2017_df.select_dtypes(include=['number']).columns
	cicids2017_df = cicids2017_df.drop(cicids2017_df[(cicids2017_df[numeric_columns] < 0).any(axis=1)].index)

	print("Deleted flow samples: ",initial_flows-len(cicids2017_df))

	if filename == 'Wednesday-workingHours.pcap_ISCX.csv':
		#4. GENERATE CIC-IDS2017 subset: Attack vs No Attack
		class_names = ["Attack", "No Attack"]

		class_label = "Label"

		# Count the number of samples in each class
		class_counts = cicids2017_df[class_label].value_counts()

		# Determine the minimum count among the classes
		minority_samples = class_counts.min()

		print("Minority Samples: ", minority_samples)

		# Subsample all classes except BENIGN to the desired number of samples
		cicids2017_df_subsampled = pd.concat([
				cicids2017_df[cicids2017_df[class_label] == "BENIGN"],
				cicids2017_df[cicids2017_df[class_label] != "BENIGN"].groupby(class_label).apply(lambda x: x.sample(minority_samples))
		])


		cicids2017_df_subsampled.loc[cicids2017_df_subsampled[class_label].isin(["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"]), class_label] = "Attack"
		cicids2017_df_subsampled.loc[cicids2017_df_subsampled[class_label].isin(["BENIGN"]), class_label] = "No Attack"


		cicids2017_df_subsampled = pd.concat([
				cicids2017_df_subsampled[cicids2017_df_subsampled[class_label] == "Attack"],
				cicids2017_df_subsampled[cicids2017_df_subsampled[class_label] == "No Attack"].sample(4*minority_samples)
		])


		cicids2017_df = cicids2017_df_subsampled


		##### PLOT FINAL CLASS DISTRIBUTION
		label_counts = cicids2017_df['Label'].value_counts()
		if verbose==True:
			# print the counts of each label
			print(label_counts)
			# Use the `plot.pie()` method to plot a pie chart of the class distribution
			label_counts.plot.pie(autopct='%1.1f%%', startangle=90)

			# Add a title to the pie chart
			plt.title('Class Distribution')

			# Display the pie chart
			plt.show()

		cicids2017_df['Label'] = cicids2017_df['Label'].apply(lambda x: ddos_labels(str(x)))

		# Reset the index of the DataFrame after filtering
		relabelled_df = cicids2017_df.reset_index(drop=True)
		balanced_df = pd.DataFrame()

		class_df = relabelled_df[relabelled_df['Label'] == 1]
		random_indices = np.random.choice(class_df.index, 10000, replace=False)
		balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

		class_df = relabelled_df[relabelled_df['Label'] == 0]
		random_indices = np.random.choice(class_df.index, 10000, replace=False)
		balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

		balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

		print(balanced_df['Label'].value_counts())

		if threshold!=None:
			ot = (balanced_df > threshold).sum().sum()
			print("Values over thredshold: ", ot)
			total_ddos = balanced_df.size
			print("DDOS, percentage of upperbounded data point: ", round(ot*100/total_ddos, 2))
			print()

			balanced_df = balanced_df.map(lambda x: (2**19)-2 if x > threshold else x)

		return balanced_df

	elif filename == 'apps_flow_features.csv':
		real_time_apps = ['SKYPE']
		non_real_time_apps = ['DROPBOX']
		websites = ['WIKIPEDIA']

		label_vect = []
		for protocol_name in protocols:
			if protocol_name in real_time_apps:
				label_vect.append(0)
			elif protocol_name in non_real_time_apps:
				label_vect.append(1)
			elif protocol_name in websites:
				label_vect.append(2)
			else:
				label_vect.append(-1)

		cicids2017_df['Label'] = label_vect
		cicids2017_df = cicids2017_df[cicids2017_df['Label'] >= 0]

		if threshold!= None:
			ot = (cicids2017_df > threshold).sum().sum()
			print("Values over thredshold: ", ot)
			total_app = cicids2017_df.size
			print("App, percentage of upperbounded data point: ", round(ot*100/total_app, 2))
			print()

			cicids2017_df = cicids2017_df.map(lambda x: (2**19)-2 if x > threshold else x)

		return cicids2017_df

	else:
		return None