"""Named exceptions for the training pipeline.

Kept in its own module so `feature_selection` can catch them without importing
`train_model` at module scope. That import pulls in Optuna and the sklearnex
guard, and `_process_single_split` deliberately defers it to call time so the
function stays picklable for ProcessPoolExecutor.
"""


class NoFeasibleSolution(Exception):
    """No Optuna trial satisfied both the block budget and the codeword limit.

    A normal, expected outcome at tight `max_blocks`: it means this
    (arm, M, k) cell has no deployable model, not that anything went wrong.
    Callers record the row as infeasible and continue eliminating features
    (F3b).
    """

    def __init__(self, k, max_blocks, trial_table=None):
        self.k = k
        self.max_blocks = max_blocks
        # The search's trials table (trial_selection.trial_rows), set only for the
        # per-tree arm: its stump count (`no_split_tree`) would otherwise be lost
        # with the failed search. None for every other arm.
        self.trial_table = trial_table
        super().__init__(
            'no feasible solution at k={} under max_blocks={}'.format(k, max_blocks))


class AlignmentInvariantError(Exception):
    """A threshold-alignment data-structure invariant was violated.

    Replaces three `print(...); exit()` sites in threshold_alignment.py.
    `exit()` raises SystemExit, which inherits from BaseException, so it
    bypassed `except Exception`, bypassed Optuna's `catch=`, and killed a
    campaign worker with no traceback and no indication of which
    (feature, threshold) was missing.
    """
