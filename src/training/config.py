"""Training-side configuration -- the definition of one experimental arm.

P4GenConfig is deliberately scoped to P4 generation (see its docstring), so
the training knobs live here instead. Spec A.2's arm grid is literally a list
of TrainConfig values, and `arm_slug` is what makes a result file
self-describing: the arm cannot be misidentified from its artifact because the
filename is derived from the config that produced it.
"""
from dataclasses import dataclass

from p4model.program import (DISJOINT_ENCODING, ENCODINGS, JOINT_ENCODING,
                             PER_TASK_ENCODING, PER_TREE_ENCODING)
from src.training.threshold_alignment import ALIGN_OBJECTIVES

SELECTION_RULES = ('balanced', 'tied_cheapest')

# The ONE statement of the forest-size search bounds (decision 9.1, Task 19 / Task 26 sweeps): one shared search space
# for every arm.
# max_depth: feasible share was flat from depth 14 to 20 (and to 32) at fixed ccp_alpha, so 20 is a cap, not a measured edge.
# n_estimators: odd 1..7 (voting), user decision 2026-11-15 on results/encoding_evidence_2026_11/model8/impl/bounds_ccp.md
# (model only): best feasible accuracy peaks at 3-7 trees and is flat or falls after (the largest odd count within 0.5
# points of the best is joint 3, disjoint/per_task 5, per_tree 7, over the ccp_alpha values shipped models use: p50 1.4e-4,
# p90 1.7e-3, 11,322 rows of campaigns 2026_11/11d/11e); and the vote table, which the model does not price, has
# classes**trees entries: at 7 trees app 2,187 (4 SRAM RAMs, 0 TCAM; probe_vote_table.md), at 11 a 177,147-entry table
# p4c did not finish compiling. (The old cap was 15 trees, chosen as the largest odd count with any feasible design at
# ccp_alpha 1e-3, which is about the 83rd percentile of shipped models, not their median.)
# train_model re-exports both for the search and the warm-start jitter.
SEARCH_MAX_TREES = 7
SEARCH_MAX_DEPTH = 20

# The arm names `_run_elimination` and the rows' `arm` column use, and the
# encoding each is priced and generated under. `joint`, `joint-task` and
# `joint-tree` all eliminate ONE shared feature list; they differ in the leaf keys.
PER_TASK_ARM = 'joint-task'
PER_TREE_ARM = 'joint-tree'
ARM_ENCODINGS = {'independent': DISJOINT_ENCODING, 'joint': JOINT_ENCODING,
                 PER_TASK_ARM: PER_TASK_ENCODING, PER_TREE_ARM: PER_TREE_ENCODING}


def arm_encoding(arm):
    """The encoding of one arm name; raises on anything else, so a typo cannot
    fall through to some other arm's encoding."""
    if arm not in ARM_ENCODINGS:
        raise ValueError('arm must be one of {}, got {!r}'.format(sorted(ARM_ENCODINGS), arm))
    return ARM_ENCODINGS[arm]


def _validate_encoding(encoding):
    """Shared guard for `TrainConfig.arm_slug`, which branches on
    the encoding, so an unrecognized string (a
    typo, say) used to fall through to the joint-arm behaviour silently instead
    of failing loudly.

    It guarded `delta_align_label` too until the 2026-09-15 deletion of the
    delta_align axis, which took that method with it."""
    if encoding not in ENCODINGS:
        raise ValueError('encoding must be one of {}, got {!r}'.format(ENCODINGS, encoding))


@dataclass(frozen=True)
class TrainConfig:
    """Frozen so a ProcessPoolExecutor worker cannot mutate the arm it is
    running, which would silently mix two treatments into one output file.

    There is no `delta_align` field. It was SWEPT -- the permitted
    relative-error degradation per task while perturbing a model to make its
    thresholds shareable -- until Track 5's pre-registered live-Optuna trial
    returned delta_helps = FALSE (mean_d000 0.7956173344395895 vs mean_d020
    0.7861922400433382, cells_favouring_d020 14/24), and the whole axis was
    deleted on 2026-09-15. Alignment now always runs at delta = 0, i.e. free
    moves only.

    alignment_enabled : False is the ablation arm -- align_rf_thresholds is not
        called at all, so the arm is provably prediction-identical to the
        unaligned models. This is NOT the same as accepting only free moves,
        which is what True now does.
    delta_select : FIXED at 0.02 for every arm. How far the chosen trial may
        fall below the best achievable balance in exchange for fewer blocks.
        Not a treatment: it moves the baseline as well as the treatment, so
        sweeping it would shift the comparison under its own control variable.
        0.02 sits inside val_select's own standard error on both tasks
        (~0.007 on App = 3.2% of its error; ~0.0036 on DDoS = 9% of its
        error), so it only breaks ties that are not distinguishable.
    align_objective : retained at 'blocks' -- the only value ALIGN_OBJECTIVES
        still accepts -- for config/manifest backward compatibility.
        'stages' and 'both' were retired by the 2026-09-07 alignment
        cost-model repair (the block factor now consumes the real generator
        cost directly, which removed the separate stages-domain objective
        this field used to select between) and are rejected by the validator
        below. Applies to the JOINT arm only, and is deliberately absent from
        arm_slug -- distinguishing objectives in the output filename is a
        campaign-design decision.
    n_trees, max_depth : inclusive search bounds -- per-axis and independent,
        so `rf_params` may suggest either maximum without suggesting both at
        once. No -1 sentinel (F10i). HISTORY (the earlier defaults (7, 14);
        superseded by decision 9.1 below): rederived from the measured capacity
        ceiling: `scripts/capacity_ceiling.py` fits both models over a
        n_trees x max_depth grid on 3 splits at the full feature set and
        records where a 512-bit codeword limit (the model's until decision
        6b removed it; the script keeps it as a yardstick) starts to bind, at both
        ends of rf_params' regularization ranges -- now including ccp_alpha
        -- (results/capacity_ceiling.csv). A cell counts as feasible when
        ANY configuration the search can reach there compiles -- witnessed
        by the pruned corner, min_samples_leaf=55 / min_samples_split=
        55*MIN_SAMPLES_SPLIT_MULT_MAX (3630) / ccp_alpha=0.05. All 49 cells
        of the measured grid (n_trees up to 15, max_depth up to 14) stayed
        within the limit on all 3 splits -- not because a wider grid might
        still turn up a binding ceiling, but because the deciding (pruned)
        corner is depth-invariant here: MIN_SAMPLES_SPLIT_MULT_MAX=66 pushes
        its min_samples_split to 3630, so under that much pruning the
        forests stop growing well before ANY tested max_depth bound, at
        every n_trees tested (`scripts/capacity_ceiling.py`'s Ruling P4-4).
        The largest admissible search space in the grid, ceil(n_trees / 2) *
        (max_depth - 1) = 104, is attained at the grid's own top corner
        (15, 14).

        Decision 9.1 (2026-11-15, model-only sweeps model8/impl/bounds.md and bounds_ccp.md) sets the
        defaults to SEARCH_MAX_TREES = 7 (odd 1..7) and SEARCH_MAX_DEPTH = 20, one search space for every
        arm: the feasible share is flat in depth from 14 to 32; accuracy peaks at 3-7 trees and nothing
        beyond 7 is within 0.5 points of the best at the ccp_alpha values shipped models use (median
        1.4e-4); and at 7 trees the app vote table is 2,187 entries (4 SRAM RAMs, 0 TCAM). The paragraphs
        above describe the older capacity-ceiling derivation of (7, 14).
    n_trees_min : inclusive lower bound on the search space for n_trees.
        Defaults to 1, which is `rf_params`'s old hardcoded lower bound.
        Set to n_trees to pin that dimension (the T-pinning mechanism used by
        `scripts/feasibility_frontier.py`).
    ccp_alpha_max : inclusive upper bound on the cost-complexity pruning
        parameter sweep. Defaults to 0.0, meaning the ccp_alpha dimension is
        absent (today's unmodified behaviour).
    selection_rule : 'tied_cheapest' (default, spec 2026-10-04 Part II) ships
        the cheapest feasible trial NOT significantly worse than the
        'balanced' pick on either task (one-sided McNemar exact test on
        val_select at select_alpha); 'balanced' ships that pick itself
        (trial_selection.select_best_trial), the rule before 2026-10-04. The
        same for every arm; recorded in the manifest, not in arm_slug.
    select_alpha : the McNemar significance level, in [0, 1], no
        multiplicity correction. 0 ties every trial (ships the globally
        cheapest feasible one); 1 ties only trials with b = 0 on both tasks.
    """

    alignment_enabled: bool = True
    delta_select: float = 0.02
    align_objective: str = 'blocks'
    n_trees: int = SEARCH_MAX_TREES
    max_depth: int = SEARCH_MAX_DEPTH
    n_trials: int = 1000
    min_feasible_before_stop: int = 25
    lookback: int = 20
    n_trees_min: int = 1
    ccp_alpha_max: float = 0.0
    selection_rule: str = 'tied_cheapest'
    select_alpha: float = 0.05

    def __post_init__(self):
        if self.delta_select < 0:
            raise ValueError(
                'delta_select must be >= 0, got {!r}'.format(self.delta_select))
        if self.align_objective not in ALIGN_OBJECTIVES:
            raise ValueError('align_objective must be one of {}, got {!r}'.format(
                ALIGN_OBJECTIVES, self.align_objective))
        if not 1 <= self.n_trees_min <= self.n_trees:
            raise ValueError(
                'n_trees_min must be in [1, n_trees] ({}), got {!r}'.format(
                    self.n_trees, self.n_trees_min))
        if self.ccp_alpha_max < 0.0:
            raise ValueError(
                'ccp_alpha_max must be >= 0.0, got {!r}'.format(self.ccp_alpha_max))
        if self.selection_rule not in SELECTION_RULES:
            raise ValueError('selection_rule must be one of {}, got {!r}'.format(
                SELECTION_RULES, self.selection_rule))
        if not 0.0 <= self.select_alpha <= 1.0:
            raise ValueError(
                'select_alpha must be in [0, 1], got {!r}'.format(self.select_alpha))

    def arm_slug(self, encoding):
        """Filename-safe arm identity, per spec C.2.

        The independent arm's slug deliberately ignores the alignment fields:
        alignment runs in the joint arm only, so two independent runs must
        share one output file.

        Three slugs, not eight. Until 2026-09-15 the aligned joint arm's slug
        carried the swept tolerance -- `joint-d000` ... `joint-d020`,
        `joint-dinf` -- and the arm that survives the axis's deletion (free
        moves only, the old `joint-d000`) is now plain `joint`. Archived
        filenames still carry the old suffixes, and
        src/reporting/campaign_data.py reconstructs them from an archived row's
        own `delta_align` column. Reporting ACCEPTS that column; nothing WRITES
        it. The same is true of the `-o{:03d}` overlap_threshold suffix retired
        in 2026-09-14.

        `joint-task` (the per-task encoding, 2026-11) deliberately ignores the
        alignment fields too: alignment reduces the union of thresholds, and a
        per-task key does not depend on the union, so the arm never aligns.
        `joint-tree` (per-tree codes) ignores them for the same reason.
        """
        _validate_encoding(encoding)
        if encoding == DISJOINT_ENCODING:
            return 'independent'
        if encoding == PER_TASK_ENCODING:
            return PER_TASK_ARM
        if encoding == PER_TREE_ENCODING:
            return PER_TREE_ARM
        if not self.alignment_enabled:
            return 'joint-off'
        return 'joint'
