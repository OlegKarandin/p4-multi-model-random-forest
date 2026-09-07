"""Training-side configuration -- the definition of one experimental arm.

P4GenConfig is deliberately scoped to P4 generation (see its docstring), so
the training knobs live here instead. Spec A.2's arm grid is literally a list
of TrainConfig values, and `arm_slug` is what makes a result file
self-describing: the arm cannot be misidentified from its artifact because the
filename is derived from the config that produced it.
"""
from dataclasses import dataclass
from typing import Optional

from src.training.threshold_alignment import ALIGN_OBJECTIVES

# The historical value: every row of campaign_backup_20260825 was written at
# 0.5, and arm_slug suffixes only AWAY from it so those filenames stay
# reproducible from their own columns.
DEFAULT_OVERLAP_THRESHOLD = 0.5


def _validate_encoding(encoding):
    """Shared guard for `TrainConfig.arm_slug` / `delta_align_label` /
    `overlap_threshold_label`: all three branch on `encoding == 'disjoint'`
    vs. everything else, so an unrecognized string (a typo, say) used to fall
    through to the joint-arm behaviour silently instead of failing loudly."""
    if encoding not in ('joint', 'disjoint'):
        raise ValueError("encoding must be 'joint' or 'disjoint', got {!r}".format(encoding))


@dataclass(frozen=True)
class TrainConfig:
    """Frozen so a ProcessPoolExecutor worker cannot mutate the arm it is
    running, which would silently mix two treatments into one output file.

    delta_align : SWEPT. Permitted relative-error degradation per task when a
        model is perturbed to make its thresholds shareable. None means
        accept-all (the "inf" anchor, which also skips the accuracy
        evaluation entirely). Applies to the JOINT arm only.
    alignment_enabled : False is the ablation arm -- align_rf_thresholds is not
        called at all, so the arm is provably prediction-identical to the
        unaligned models. This is NOT the same as delta_align = 0.
    delta_select : FIXED at 0.02 for every arm. How far the chosen trial may
        fall below the best achievable balance in exchange for fewer blocks.
        Not a treatment: it moves the baseline as well as the treatment, so
        sweeping it would shift the comparison under its own control variable.
        0.02 sits inside val_select's own standard error on both tasks
        (~0.007 on App = 3.2% of its error; ~0.0036 on DDoS = 9% of its
        error), so it only breaks ties that are not distinguishable.
    overlap_threshold : minimum overlap ratio for a range pair to be an
        alignment CANDIDATE -- a separate concern from whether a candidate is
        ACCEPTED (that is delta_align). Was hardcoded at the call site.
        Enters `arm_slug` (conditionally -- see `_overlap_suffix`), unlike
        `align_objective`, which is deliberately absent from it.
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
        once. No -1 sentinel (F10i). Rederived from the measured capacity
        ceiling: `scripts/capacity_ceiling.py` fits both models over a
        n_trees x max_depth grid on 3 splits at the full feature set and
        records where the 512-bit codeword limit starts to bind, at both
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

        n_trees is NOT set from that ceiling: it is set by a utilisation
        argument instead (design 2026-09-03 spec 2.1(b)) -- the archive
        never reaches 11 trees and p75 is 3, so headroom the ceiling would
        permit goes unused in practice. n_trees = 7 regardless of what the
        grid allows. max_depth is kept at its previous value of 14 even
        though the measurement did not rule out raising it: spec 2.1's
        boxed warning is that codeword length is essentially total leaf
        count across both forests, and `joint` pools both models' leaves
        into one 512-bit codeword while `independent` never pools, so
        raising max_depth would push `joint` toward CodewordTooLong faster
        than `independent`, widening a dimension that structurally
        disadvantages the arm under study -- raising it is therefore a
        separate, explicit decision, not an automatic consequence of this
        measurement. The predecessor (7, 10) was a placeholder whose
        comment said P4 would derive it; nothing had measured the ceiling,
        which is what this replaces.
    n_trees_min : inclusive lower bound on the search space for n_trees.
        Defaults to 1, which is `rf_params`'s old hardcoded lower bound.
        Set to n_trees to pin that dimension (the T-pinning mechanism used by
        `scripts/feasibility_frontier.py`).
    ccp_alpha_max : inclusive upper bound on the cost-complexity pruning
        parameter sweep. Defaults to 0.0, meaning the ccp_alpha dimension is
        absent (today's unmodified behaviour).
    """

    delta_align: Optional[float] = 0.0
    alignment_enabled: bool = True
    delta_select: float = 0.02
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    align_objective: str = 'blocks'
    n_trees: int = 7
    max_depth: int = 14
    n_trials: int = 1000
    min_feasible_before_stop: int = 25
    lookback: int = 20
    n_trees_min: int = 1
    ccp_alpha_max: float = 0.0

    def __post_init__(self):
        if self.delta_align is not None and self.delta_align < 0:
            raise ValueError(
                'delta_align must be None or >= 0, got {!r}'.format(self.delta_align))
        if self.delta_select < 0:
            raise ValueError(
                'delta_select must be >= 0, got {!r}'.format(self.delta_select))
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError(
                'overlap_threshold must be in [0, 1], got {!r}'.format(self.overlap_threshold))
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

    def arm_slug(self, encoding):
        """Filename-safe arm identity, per spec C.2.

        The independent arm's slug deliberately ignores the alignment fields:
        alignment runs in the joint arm only, so two independent runs differing
        only in delta_align are the SAME arm and must share one output file.
        """
        _validate_encoding(encoding)
        if encoding == 'disjoint':
            return 'independent'
        if not self.alignment_enabled:
            return 'joint-off'
        if self.delta_align is None:
            slug = 'joint-dinf'
        else:
            slug = 'joint-d{:03d}'.format(int(round(self.delta_align * 100)))
        return slug + self._overlap_suffix()

    def _overlap_suffix(self):
        """'' at the historical 0.5, '-o{:03d}' anywhere else.

        Design 2026-09-03 §3: a campaign sweeping overlap_threshold must add it
        to the slug or three overlap arms map to one path -- and because
        main.py's skip_existing treats an existing path as "cell done", the
        second and third would be silently SKIPPED rather than overwritten.

        Suffixing only away from the default keeps all 40 archived filenames
        reproducible from their own columns. That leaves the default implicit,
        which campaign_data._expected_arm_slug turns into a CHECKED invariant:
        a file named `joint-d020` carrying overlap_threshold != 0.5 is
        mislabelled and raises rather than being read.
        """
        if self.overlap_threshold == DEFAULT_OVERLAP_THRESHOLD:
            return ''
        return '-o{:03d}'.format(int(round(self.overlap_threshold * 100)))

    def delta_align_label(self, encoding='joint'):
        """What goes in the row's `delta_align` column (spec C.1): the float,
        "inf" for accept-all, or "" when alignment did not run.

        encoding='disjoint' suppresses this the same way `arm_slug` does:
        alignment runs in the joint arm only, so an independent-arm row must
        not carry the joint arm's alignment settings.
        """
        _validate_encoding(encoding)
        if encoding == 'disjoint' or not self.alignment_enabled:
            return ''
        if self.delta_align is None:
            return 'inf'
        return '{:g}'.format(self.delta_align)

    def overlap_threshold_label(self, encoding='joint'):
        """What goes in the row's `overlap_threshold` column (spec C.1): the
        float, or "" when alignment did not run.

        overlap_threshold only governs which range pairs align_rf_thresholds
        considers as candidates, so it is meaningless wherever that function
        is never called -- suppressed the same way delta_align_label is: for
        the disjoint (independent) arm, and for the joint-off ablation.
        """
        _validate_encoding(encoding)
        if encoding == 'disjoint' or not self.alignment_enabled:
            return ''
        return '{:g}'.format(self.overlap_threshold)
