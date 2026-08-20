"""Instrumented alignment run: characterises shift_mass as a predictor of
alignment damage (P3 Task 6/7). The shift_mass-derived pre-filter this script
originally evaluated was later REMOVED from align_rf_thresholds (P3 Task 8 --
its cap was found to be identically 0 at delta_align=0, discarding confirmed-
harmless moves); this script remains useful purely as a measurement tool.

Answers the three questions the pre-filter's replacement turns on, from ONE
labelled dataset. Cheap because alignment runs AFTER fitting: one fitted model
pair is re-aligned under several settings in seconds -- no refits, no Optuna.

  1. Does shift_mass predict rel_deg? Scatter one against the other. A tight
     relationship confirms the bound empirically and licenses the derived cap.
  2. Was the ratio cap wrong? Count candidates with endpoint_ratio > 5 that the
     oracle ACCEPTED -- alignments silently discarded across every result to
     date. Prediction, not a finding: non-empty, and concentrated on
     small-valued features such as Fwd.Packet.Length.Min / Min.Packet.Length.
  3. Is any filter worth keeping? Measure oracle cost per candidate now that
     the hot path is vectorised. If filtering saves little, deletion is
     defensible -- and provable rather than assumed.

Writes results/alignment_filter_log.csv and prints the three answers.

Run:
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/instrument_alignment_filter.py
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.main import remove_correlated_features_both_datasets
from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
from src.training import threshold_alignment as ta
from src.training.dataset import read_app_dataset, read_DDOS_dataset
from src.training.splits import make_task_splits

SELECTED_FEATURES = [
    'Fwd.Packet.Length.Max', 'Fwd.Packet.Length.Min', 'Fwd.Packet.Length.Mean',
    'Bwd.Packet.Length.Max', 'Bwd.Packet.Length.Min', 'Bwd.Packet.Length.Mean',
    'Flow.IAT.Mean', 'Flow.IAT.Max', 'Flow.IAT.Min',
    'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min',
    'Bwd.IAT.Mean', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
    'Min.Packet.Length', 'Max.Packet.Length', 'Packet.Length.Mean']

DELTAS = (0.0, 0.02, 0.05, 0.10, 0.20)


def main():
    df_app = read_app_dataset(SELECTED_FEATURES, INFINITE)
    df_ddos = read_DDOS_dataset(SELECTED_FEATURES, INFINITE)
    X_app, X_ddos, names = remove_correlated_features_both_datasets(df_app, df_ddos)
    app = make_task_splits(X_app, df_app.Label.to_numpy(), 52)
    ddos = make_task_splits(X_ddos, df_ddos.Label.to_numpy(), 52)

    def fit(X, y, seed):
        return dt_thresholds_float_to_int(RandomForestClassifier(
            n_estimators=7, max_depth=10, min_samples_leaf=5,
            random_state=seed, n_jobs=1).fit(X, y))

    rows = []
    for delta in DELTAS:
        # Refit per delta so each run starts from identical, unmutated models.
        # align_rf_thresholds now deepcopies rf1/rf2 on entry and mutates only
        # the copies (C8), so this refit is no longer needed for THAT reason;
        # it stays because this loop discards align_rf_thresholds' return
        # value and only reads the candidate_log side effect, so rf1/rf2
        # still have to be fresh, unaligned models going into each delta.
        rf1 = fit(app.X_train, app.y_train, 0)
        rf2 = fit(ddos.X_train, ddos.y_train, 1)

        log = []
        start = time.perf_counter()
        ta.align_rf_thresholds(
            rf1, rf2,
            app.X_val_align, app.y_val_align,
            ddos.X_val_align, ddos.y_val_align,
            overlap_threshold=0.5, delta_rel=delta,
            candidate_log=log)
        elapsed = time.perf_counter() - start

        for entry in log:
            entry = dict(entry)
            entry['delta_align'] = delta
            entry['feature'] = names[entry['feature_idx']]
            entry['rel_deg_acc_app'], entry['rel_deg_f1_app'], \
                entry['rel_deg_acc_ddos'], entry['rel_deg_f1_ddos'] = entry.pop('rel_deg')
            rows.append(entry)

        print('delta={:<5} candidates={:<6} accepted={:<6} wall={:.1f}s  '
              '{:.2f} ms/candidate'.format(
                  delta, len(log), sum(e['accepted'] for e in log), elapsed,
                  1000 * elapsed / max(1, len(log))))

    frame = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    frame.to_csv(os.path.join('results', 'alignment_filter_log.csv'), index=False)

    print('\nQ1  does shift_mass predict damage?')
    for task, column, error_col in (('app', 'rel_deg_acc_app', 'error_app'),
                                     ('ddos', 'rel_deg_acc_ddos', 'error_ddos')):
        mass = frame['shift_mass_1' if task == 'app' else 'shift_mass_2']
        print('    {:<5} pearson r(shift_mass, rel_deg) = {:+.3f}'.format(
            task, np.corrcoef(mass, frame[column])[0, 1]))
        violations = (frame[column] > mass / frame[error_col]).sum()
        print('    {:<5} candidates where rel_deg > shift_mass/error (bound violated) = {}'
              .format(task, violations))

    print('\nQ2  was the ratio cap wrong?')
    wrongly_vetoed = frame[(frame['endpoint_ratio'] > 5.0) & frame['accepted']]
    print('    candidates with endpoint_ratio > 5 that the oracle ACCEPTED: {}'
          .format(len(wrongly_vetoed)))
    if len(wrongly_vetoed):
        print(wrongly_vetoed['feature'].value_counts().to_string())

    print('\nQ3  is any filter worth keeping? -- see the ms/candidate above.')


if __name__ == '__main__':
    main()
