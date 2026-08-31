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
    align_objective : what the shed bits are AIMED at -- 'blocks' (today's
        behaviour, the default), 'stages', or 'both'. A different axis from
        delta_align: that is how much accuracy may be spent, this is which
        boundary it is spent to cross. Applies to the JOINT arm only, and is
        deliberately absent from arm_slug -- distinguishing objectives in the
        output filename is a campaign-design decision, and a campaign that
        sweeps two objectives must change the slug before it can, or the two
        runs overwrite one file.
    n_trees, max_depth : inclusive search bounds -- per-axis and independent,
        so `rf_params` may suggest either maximum without suggesting both at
        once. No -1 sentinel (F10i). Rederived from the measured capacity
        ceiling rather than chosen by hand: `scripts/capacity_ceiling.py` fits
        both models over a n_trees x max_depth grid on 3 splits at the full
        feature set and records where the 512-bit codeword limit starts to
        bind, at both ends of rf_params' regularization ranges
        (results/capacity_ceiling.csv). A cell counts as feasible when ANY
        configuration the search can reach there compiles -- witnessed by the
        pruned corner, min_samples_leaf=200 / min_samples_split=400 -- and
        (11, 14) is the feasible cell with the largest reachable search space,
        ceil(n_trees / 2) * (max_depth - 1) = 78. The predecessor (7, 10) was
        a placeholder whose comment said P4 would derive it; nothing had
        measured the ceiling, which is what this replaces.
    """

    delta_align: Optional[float] = 0.0
    alignment_enabled: bool = True
    delta_select: float = 0.02
    overlap_threshold: float = 0.5
    align_objective: str = 'blocks'
    n_trees: int = 11
    max_depth: int = 14
    n_trials: int = 1000
    min_feasible_before_stop: int = 25
    lookback: int = 20

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
            return 'joint-dinf'
        return 'joint-d{:03d}'.format(int(round(self.delta_align * 100)))

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
