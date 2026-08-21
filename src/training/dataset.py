import numpy as np
import pandas as pd

def read_app_dataset(selected_features, threshold):

	df = pd.read_csv('./resources/apps_flow_features.csv', delimiter=',')

	new_cols = {col: col.strip() for col in df.columns}
	df.rename(columns=new_cols, inplace=True)

	protocols = df['ProtocolName'].to_numpy()
	real_time_apps = ['SKYPE']
	non_real_time_apps = ['DROPBOX']
	websites = ['GOOGLE']

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

	df = df[selected_features]
	
	df['Label'] = label_vect
	df = df[df['Label'] >= 0]
	
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.dropna(inplace=True)

	df = df[(df >= 0).all(axis=1)]

	df = df.clip(upper=threshold)
	df.drop_duplicates(inplace=True)

	return df

def read_DDOS_dataset(selected_features, threshold, random_state=42):

	df = pd.read_csv('./resources/Wednesday-workingHours.pcap_ISCX.csv', delimiter=',')

	new_cols = {col: col.strip().replace(' ', '.').replace('s/s', 's.s') for col in df.columns}
	df.rename(columns=new_cols, inplace=True)

	df = df[selected_features + ['Label']]

	labels_to_keep = ['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest']
	df = df[df['Label'].isin(labels_to_keep)]

	temp_dict = {att_name: 1 for att_name in labels_to_keep if att_name != 'BENIGN'}
	temp_dict['BENIGN'] = -1

	df.replace({"Label": temp_dict}, inplace=True)
	df = df.reset_index(drop=True)

	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.dropna(inplace=True)

	df = df[df.drop('Label', axis=1).ge(0).all(axis=1)]
	
	df = df.clip(upper=threshold)
	df.drop_duplicates(inplace=True)

	balanced_df = pd.DataFrame()
	rng = np.random.RandomState(random_state)

	class_df = df[df['Label'] == 1]
	random_indices = rng.choice(class_df.index, 10000, replace=False)
	balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

	class_df = df[df['Label'] == -1]
	random_indices = rng.choice(class_df.index, 10000, replace=False)
	balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

	balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

	return balanced_df
