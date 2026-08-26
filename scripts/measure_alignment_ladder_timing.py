"""Ladder timing driver for `align_rf_thresholds` (P3b Task 5, first
deliverable). Wraps the P3-plan timing probe (`docs/superpowers/plans/
2026-08-19-p3-per-task-alignment-guard.md`, Task 2 Step 2) verbatim -- same
`rng = np.random.default_rng(0)`, same X1/y1/X2/y2 construction, same `mk`
lambda -- in a repeat/arm loop, so a median and spread can be read off
instead of one noisy single-shot reading. `rf1`/`rf2` are refit fresh on
every repeat: on pre-C8 commits (before `ed14399`) `align_rf_thresholds`
mutates its arguments in place, so reusing a pair across repeats would time
alignment of an already-partially-aligned model on repeat 2+. Matches
`instrument_alignment_filter.py`'s existing per-delta refit for the same
reason.

Every individual per-repeat reading is printed alongside the summary, not
only the median/min/max -- so the raw numbers behind any reported spread are
always reproducible from the command's own output, not only from a report
someone has to trust.

**Not commit-pinned.** `align_rf_thresholds`'s call signature and this
file's three imports (sklearn, `src.p4gen.build_p4_script`,
`src.training.threshold_alignment`) were confirmed stable across every P3b
ladder commit Task 5 measured (`ed019f3` phase base through `ed14399` phase
head) -- this exact file ran unmodified at all six. To reproduce the ladder
table without disturbing the main checkout:

  1. Detached-HEAD worktree, so `master` is never touched:
       git worktree add --detach <tmp>/wt-timing <phase-base-sha>
  2. This file does not exist at old commits' trees, so extract or copy it
     into the worktree once:
       git show HEAD:scripts/measure_alignment_ladder_timing.py \
           > <tmp>/wt-timing/measure_alignment_ladder_timing.py
  3. For each ladder sha, checkout then run from inside the worktree
     (cwd = worktree root, so `from src...` resolves to THAT commit's
     source):
       git -C <tmp>/wt-timing checkout --detach <sha>
       cd <tmp>/wt-timing && "<python>" measure_alignment_ladder_timing.py "[0.05, 0.0, None]" 7
  4. git worktree remove --force <tmp>/wt-timing

Usage: python measure_alignment_ladder_timing.py <deltas-as-python-literal> <reps>
  e.g. python measure_alignment_ladder_timing.py "[0.05]" 7
       python measure_alignment_ladder_timing.py "[0.05, 0.0, None]" 7
"""
import ast
import statistics
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int
from src.training import threshold_alignment as ta

rng = np.random.default_rng(0)
X1 = np.clip(rng.integers(0, 90000, size=(4000, 17)), 0, INFINITE).astype(float)
y1 = np.array([c % 3 for c in range(4000)])
X2 = np.clip(rng.integers(0, 90000, size=(3000, 17)), 0, INFINITE).astype(float)
y2 = np.array([-1, 1] * 1500)

POLICIES = ('legacy', 'c1', 'c1c2')


def mk(X, y, s):
    return dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=7, max_depth=10, min_samples_leaf=5,
        random_state=s).fit(X, y))


def main():
    arms = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else [0.05]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    for delta in arms:
        for policy in POLICIES:
            times = []
            for _ in range(reps):
                rf1, rf2 = mk(X1, y1, 0), mk(X2, y2, 1)
                t0 = time.perf_counter()
                ta.align_rf_thresholds(rf1, rf2, X1, y1, X2, y2,
                                        overlap_threshold=0.5, delta_rel=delta,
                                        align_policy=policy)
                times.append(time.perf_counter() - t0)
            print('delta={!r:<6} policy={:<6} n={} median={:.3f}s '
                  'min={:.3f}s max={:.3f}s all={}'.format(
                      delta, policy, reps, statistics.median(times),
                      min(times), max(times),
                      ['{:.3f}'.format(t) for t in times]))


if __name__ == '__main__':
    main()
