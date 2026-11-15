import numpy as np
import pandas as pd

# Task 38 (P3h, user decision 2026-11-15): the six Mean features are replaced by
# features the switch can compute, each derived here from the CSVs' raw columns IN THE
# SWITCH'S OWN UNIT, so a tree threshold is already the switch value and the control
# plane installs it unchanged (p4model.units.switch_range: these are not IAT features).
#   Flow.Duration, Fwd.IAT.Total, Bwd.IAT.Total: the flow's / a direction's duration
#     (last minus first arrival; the CSVs' Flow.Duration, Fwd.IAT.Total, Bwd.IAT.Total,
#     in microseconds) in ticks of the switch's coarse clock, 65.536 us
#     (ingress_mac_tstamp[47:16]): round(d / 65.536) = floor((1000 d + 32768) / 65536),
#     then clipped at the 16-bit top like every feature. No C1f "+1": a duration of 0
#     is a real value (a chain of one packet), not "no IAT". The switch reads
#     floor(t_last / T) - floor(t_first / T), which is floor(d / T) or one more, and
#     round(d / T) is one of the same two integers, so the two differ by at most one
#     tick per flow (Task 37, impl/probe_p3c.md "Training side").
#   Fwd.Packet.Length.Total, Bwd.Packet.Length.Total, Packet.Length.Total: the CSVs'
#     Total.Length.of.Fwd.Packets / .Bwd.Packets and their sum (the CSVs have no flow
#     total), clipped at 65535: the switch's saturating 16-bit sum. CAVEAT, pre-existing
#     for every length feature: the CSVs sum PAYLOAD bytes, the switch IP total lengths.
# These are CSV whole-flow values; the switch computes them packet by packet (p4gen.
# switch_semantics documents both). A selected name that is not here is read as is.
COARSE_TICK_NS = 65536


def coarse_ticks(duration_us):
	"""A duration in microseconds as ticks of the switch's coarse clock (65.536 us),
	round to nearest: floor((1000 d + 32768) / 65536). Exact in float64 below 2^53."""
	return np.floor((1000.0 * np.asarray(duration_us, dtype=float) + COARSE_TICK_NS // 2)
					/ COARSE_TICK_NS)


# feature name -> (the raw CSV columns it reads, the function of their arrays)
# UNITS, read this before touching the CSVs directly: Flow.Duration, Fwd.IAT.Total and
# Bwd.IAT.Total are BOTH the raw CSV columns (MICROSECONDS) and the derived training
# features of the same name (COARSE TICKS of 65.536 us). derive_features replaces the
# first by the second, so every frame this module returns, and every forest trained on it,
# is in ticks. A reader of the raw CSV by these names gets microseconds; the switch
# would install those thresholds as ticks (these are not IAT features, units.switch_range)
# and be wrong by a factor of 65.5. No range check can catch it here: both are clipped at
# 65535. Read the features through read_app_dataset / read_DDOS_dataset only.
DERIVED_FEATURES = {
	'Flow.Duration': (('Flow.Duration',), lambda d: coarse_ticks(d)),
	'Fwd.IAT.Total': (('Fwd.IAT.Total',), lambda d: coarse_ticks(d)),
	'Bwd.IAT.Total': (('Bwd.IAT.Total',), lambda d: coarse_ticks(d)),
	'Fwd.Packet.Length.Total': (('Total.Length.of.Fwd.Packets',), lambda t: t),
	'Bwd.Packet.Length.Total': (('Total.Length.of.Bwd.Packets',), lambda t: t),
	'Packet.Length.Total': (('Total.Length.of.Fwd.Packets', 'Total.Length.of.Bwd.Packets'),
							lambda fwd, bwd: fwd + bwd),
}


# The Mean each new feature replaced, in place (main.CAMPAIGN_FEATURES). Runs before Task 38
# saved forests over the Means; regenerating such a forest's program with replace_means(names)
# gives the program today's generator writes for a forest of the same shape on the new pool,
# which is how Task 38's resource identity was checked. Its thresholds keep the Mean's units,
# so that program is not a deployable classifier.
MEAN_REPLACEMENTS = {
	'Fwd.Packet.Length.Mean': 'Fwd.Packet.Length.Total',
	'Bwd.Packet.Length.Mean': 'Bwd.Packet.Length.Total',
	'Packet.Length.Mean': 'Packet.Length.Total',
	'Flow.IAT.Mean': 'Flow.Duration',
	'Fwd.IAT.Mean': 'Fwd.IAT.Total',
	'Bwd.IAT.Mean': 'Bwd.IAT.Total',
}


def replace_means(feature_names):
	"""feature_names with every pre-Task-38 Mean replaced by its replacement (MEAN_REPLACEMENTS)."""
	return [MEAN_REPLACEMENTS.get(name, name) for name in feature_names]


def raw_columns(selected_features):
	"""The CSV columns the selected features read, first-seen order."""
	columns = []
	for feature in selected_features:
		for column in DERIVED_FEATURES[feature][0] if feature in DERIVED_FEATURES else (feature,):
			if column not in columns:
				columns.append(column)
	return columns


def derive_features(df, selected_features):
	"""The selected features (in order, then Label if df has it) from df's raw columns:
	a DERIVED_FEATURES entry computed, any other feature copied. The index is kept."""
	out = pd.DataFrame(index=df.index)
	for feature in selected_features:
		if feature in DERIVED_FEATURES:
			sources, function = DERIVED_FEATURES[feature]
			out[feature] = function(*(df[column].to_numpy(dtype=float) for column in sources))
		else:
			out[feature] = df[feature]
	if 'Label' in df.columns:
		out['Label'] = df['Label']
	return out


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

	df = df[raw_columns(selected_features)].copy()

	df['Label'] = label_vect
	df = df[df['Label'] >= 0]

	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.dropna(inplace=True)

	df = df[(df >= 0).all(axis=1)]

	df = derive_features(df, selected_features)
	df = df.clip(upper=threshold)
	df.drop_duplicates(inplace=True)

	return df

def read_DDOS_dataset(selected_features, threshold, random_state=42):

	df = pd.read_csv('./resources/Wednesday-workingHours.pcap_ISCX.csv', delimiter=',')

	new_cols = {col: col.strip().replace(' ', '.').replace('s/s', 's.s') for col in df.columns}
	df.rename(columns=new_cols, inplace=True)

	df = df[raw_columns(selected_features) + ['Label']]

	labels_to_keep = ['BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest']
	df = df[df['Label'].isin(labels_to_keep)]

	temp_dict = {att_name: 1 for att_name in labels_to_keep if att_name != 'BENIGN'}
	temp_dict['BENIGN'] = -1

	df.replace({"Label": temp_dict}, inplace=True)
	df = df.reset_index(drop=True)

	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.dropna(inplace=True)

	df = df[df.drop('Label', axis=1).ge(0).all(axis=1)]

	df = derive_features(df, selected_features)
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
