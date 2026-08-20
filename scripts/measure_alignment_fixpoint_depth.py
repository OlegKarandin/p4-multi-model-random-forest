"""Measures true `align_rf_thresholds` fixpoint depth (P3b Task 5, second
deliverable). Confirms or corrects Ruling P3b-6, which raised
MAX_RECOMPUTE_ROUNDS from 8 to 32 on the strength of one probe that measured
true depth 6-8, hitting exactly 8 on 3 of 12 seed x arm configurations.

Uses the same synthetic fixture as the P3-plan timing probe (the "realistic
17-feature probe" MAX_RECOMPUTE_ROUNDS's own comment refers to: 4000x17 and
3000x17 samples, 7 trees, max_depth=10, min_samples_leaf=5), varied across
several data seeds and all three delta arms (0.05, 0.0, None), with the cap
monkeypatched well above 32 so a run's reported depth is a true convergence
depth, never a truncation. If any run is truncated anyway (cap hit while
still progressing), align_rf_thresholds raises AlignmentInvariantError and
this script lets that propagate -- that outcome IS the finding.

Depth is read off `candidate_log`, not off the loop's internal `rounds`
counter (which align_rf_thresholds does not return): per feature, every
`while progressed` round increments `rounds` and re-scans overlaps before
logging anything, and a round that accepts nothing logs no NEW candidate for
that feature (every pair it re-examines is already in `seen`, from either an
earlier accept-mutation removing it from the overlap list, or an earlier
reject adding it to `seen` directly) -- so the round immediately after the
last accepted move is exactly the confirming, empty round that makes
`progressed` False and ends the loop. That means a feature's true depth
equals (the highest `round` value among ITS accepted candidate_log entries)
+ 1, or 1 if it has none (round 1 is the pre-recompute pass every feature
gets whether or not anything is found there).

Run (from the repository root -- PYTHONPATH=. is required because this file
is run as a plain script, which puts scripts/ rather than the repo root on
sys.path[0]):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/measure_alignment_fixpoint_depth.py
"""
import collections

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
from src.training import threshold_alignment as ta
from src.training.errors import AlignmentInvariantError

SEEDS = range(6)
ARMS = (0.05, 0.0, None)

# ~4x the shipped cap (32). Large enough that a run reaching it would be
# unambiguous cycling, not a realistic fixpoint -- so no true depth measured
# here can be mistaken for a truncation.
PROBE_CAP = 128


def make_fixture(seed):
    rng = np.random.default_rng(seed)
    X1 = np.clip(rng.integers(0, 90000, size=(4000, 17)), 0, INFINITE).astype(float)
    y1 = np.array([c % 3 for c in range(4000)])
    X2 = np.clip(rng.integers(0, 90000, size=(3000, 17)), 0, INFINITE).astype(float)
    y2 = np.array([-1, 1] * 1500)

    def mk(X, y, s):
        return dt_thresholds_float_to_int(RandomForestClassifier(
            n_estimators=7, max_depth=10, min_samples_leaf=5,
            random_state=s).fit(X, y))

    return X1, y1, X2, y2, mk(X1, y1, 2 * seed), mk(X2, y2, 2 * seed + 1)


def per_feature_depths(candidate_log):
    """{feature_idx: true fixpoint depth}, per the docstring's derivation."""
    last_accepted_round = collections.defaultdict(int)
    seen_features = set()
    for entry in candidate_log:
        seen_features.add(entry['feature_idx'])
        if entry['accepted']:
            last_accepted_round[entry['feature_idx']] = max(
                last_accepted_round[entry['feature_idx']], entry['round'])
    return {f: last_accepted_round.get(f, 0) + 1 for f in seen_features}


def main():
    original_cap = ta.MAX_RECOMPUTE_ROUNDS
    ta.MAX_RECOMPUTE_ROUNDS = PROBE_CAP
    try:
        per_config_max = []
        all_depths = []
        for seed in SEEDS:
            X1, y1, X2, y2, rf1, rf2 = make_fixture(seed)
            for delta in ARMS:
                log = []
                try:
                    ta.align_rf_thresholds(
                        rf1, rf2, X1, y1, X2, y2,
                        overlap_threshold=0.5, delta_rel=delta,
                        candidate_log=log)
                except AlignmentInvariantError as exc:
                    print('seed={} delta={!r}: TRUNCATED at cap={} -- {}'.format(
                        seed, delta, PROBE_CAP, exc))
                    continue

                depths = per_feature_depths(log)
                if depths:
                    config_max = max(depths.values())
                    all_depths.extend(depths.values())
                else:
                    config_max = 1  # no common features ever got a candidate
                per_config_max.append((seed, delta, config_max))
                print('seed={} delta={!r:<6} features_touched={} max_depth={} '
                      'depths={}'.format(
                          seed, delta, len(depths), config_max,
                          sorted(depths.values(), reverse=True)))

        print()
        print('=== summary over {} seed x arm configurations ==='.format(
            len(per_config_max)))
        maxima = [m for _, _, m in per_config_max]
        print('per-config max depth: min={} median={} max={}'.format(
            min(maxima), sorted(maxima)[len(maxima) // 2], max(maxima)))
        counts = collections.Counter(maxima)
        print('distribution of per-config max depth: {}'.format(
            dict(sorted(counts.items()))))
        print('overall deepest feature across every config: {}'.format(
            max(all_depths)))
        depth_hist = collections.Counter(all_depths)
        print('distribution of per-feature depths, all configs pooled: {}'.format(
            dict(sorted(depth_hist.items()))))
        near_cap = [(s, d, m) for s, d, m in per_config_max if m >= 16]
        if near_cap:
            print('!! configs with max depth >= 16 (halfway to the shipped '
                  'cap of 32): {}'.format(near_cap))
        else:
            print('no config approached the shipped cap of 32 '
                  '(nothing reached even half of it).')
    finally:
        ta.MAX_RECOMPUTE_ROUNDS = original_cap


if __name__ == '__main__':
    main()
