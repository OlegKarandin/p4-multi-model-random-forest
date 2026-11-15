"""What the generated P4 program computes -- which is NOT rf.predict.

RandomForestClassifier.predict averages predict_proba across trees: a SOFT
vote. The switch cannot. Its per-tree ternary table maps a codeword to a single
class (export_text's `class:` line, i.e. argmax of the leaf's counts), it
carries only that class in meta.class_tree_<task>_<i>, and generate_voting_code's
exact-match vote_<task> table votes over those per-tree classes: a HARD vote.

The two differ whenever leaves are impure, and with min_samples_leaf searched
over 5-200 at max_depth <= 10 that is the normal regime. Measured on the real
datasets: 3.5-9.1% of app flows and 0.2-1.9% of ddos flows are classified
differently, worth up to 1.7 accuracy points -- which on ddos's small error is
up to +55% RELATIVE error, wider than the whole delta_align grid the campaign
sweeps. Reporting soft-vote accuracy therefore overstates the deployed artifact,
and optimising against it ranks trials by an accuracy the switch never reaches.

So every accuracy measurement in the pipeline goes through here.

UNITS: accuracy here is computed in the SOFTWARE unit (the CSVs' microseconds for
every IAT feature). The switch reads an IAT in 1.024 us ticks, a real gap of g ticks
as g + 1 and "no IAT" as 0, and its control plane writes every IAT threshold t as
build_p4_script.iat_ticks(t) = floor((1000 t + 500) / 1024) + 1 (Task 34). That
conversion is not emulated here: on 200 DEV designs (feature-level emulation, Tasks
29 and 33) the switch with it agrees with this software vote on 99.98 / 99.97% (app,
k<=8 / k>=9) and 100.00 / 99.99% (ddos) of flows; the residual is +-1 tick of clock
reading at a threshold.

THE TASK 38 FEATURES (P3h, replacing the six Means; build_p4_script._REGISTER_ACTION_
BODIES has the register bodies). Training uses the CSVs' WHOLE-FLOW values, derived
in the switch's unit by src/training/dataset.py (DERIVED_FEATURES); the switch
computes each one packet by packet, and a packet is classified on the value AFTER its
own update:
  flow_duration, fwd_iat_total, bwd_iat_total: on the switch, min(floor(t_last / T) -
    floor(t_first / T), 65535) with T = 65.536 us (meta.now_coarse =
    ingress_mac_tstamp[47:16]) over the flow's packets / the direction's packets so
    far (0 for one packet; a direction with no packet yet reads 0, also on the other
    direction's packets, which read the owner's current value). Training:
    min(round(d / T), 65535), d = the CSV's Flow.Duration / Fwd.IAT.Total /
    Bwd.IAT.Total of the whole flow; the two differ by at most one tick for the same
    packets (dataset.py). Thresholds install unchanged (not IAT features,
    p4model.units).
  fwd_packet_length_total, bwd_packet_length_total, packet_length_total: on the
    switch, the saturating 16-bit sum of the IP total lengths of the direction's /
    flow's packets so far. Training: min(Total.Length.of.Fwd.Packets, 65535), the same
    for Bwd, min(fwd + bwd, 65535): the CSVs sum payload bytes, not IP lengths
    (pre-existing for every length feature).
Not emulated here either: like the IAT features, accuracy is the software vote on the
CSV values.
"""
import numpy as np
from sklearn.metrics import accuracy_score


def per_tree_class_indices(rf, X):
    """(n_trees, n_samples) of per-tree class INDICES.

    Exactly what the switch carries in meta.class_tree_<task>_<i>: a
    RandomForest's sub-estimators are fit on encoded y, so their predict returns
    indices into rf.classes_ -- the same value export_text's `class:` line
    resolves to, and the same value the per-tree ternary table stores.
    """
    return np.vstack([estimator.predict(X).astype(np.intp)
                      for estimator in rf.estimators_])


def vote_winner(per_tree_indices, n_classes):
    """The winning class index for one sample: hard majority vote, ties broken
    toward the SMALLEST class index.

    This is the rule generate_voting_code writes into every const entry of the
    vote_<task> table, and it is a deliberate specification rather than an
    accident. statistics.mode -- which the generator used previously -- returns
    the FIRST-ENCOUNTERED mode, so its winner depends on tree ordering, which is
    arbitrary and changes if trees are reordered. The two disagree on 9.6-18.5%
    of the 3-class table's key tuples. Smallest-index is order-independent.
    """
    return int(np.argmax(np.bincount(np.asarray(per_tree_indices, dtype=np.intp),
                                     minlength=n_classes)))


def switch_predict(rf, X):
    """Class LABELS as the generated switch would classify them.

    Vectorised as one bincount over a sample-major offset array: each sample
    gets its own length-n_classes slot, so the whole (n_trees, n_samples) block
    is counted in a single pass. Equivalent to vote_winner per column.
    """
    indices = per_tree_class_indices(rf, X)
    n_trees, n_samples = indices.shape
    n_classes = rf.n_classes_

    offsets = np.arange(n_samples, dtype=np.intp) * n_classes
    flat = (offsets[None, :] + indices).ravel()
    votes = np.bincount(flat, minlength=n_samples * n_classes).reshape(n_samples, n_classes)

    return rf.classes_[np.argmax(votes, axis=1)]


def switch_accuracy_scorer(estimator, X, y):
    """scikit-learn-compatible scorer, for permutation_importance(scoring=...).

    permutation_importance calls a scorer as scorer(estimator, X, y) and would
    otherwise fall through to estimator.predict -- the soft vote -- so feature
    importances would be measured against semantics the switch does not have.
    """
    return accuracy_score(y, switch_predict(estimator, X))
