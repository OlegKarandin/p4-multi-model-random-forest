"""Score an alignment replay ladder against the pre-registered criteria.

Pure pandas over scripts/replay_alignment.py's output -- no fitting, so the
arithmetic can be debugged without repeating an expensive run.

EVERY criterion is PAIRED on PAIR_KEYS via an inner merge. That is the whole
reason the replay exists: the campaign compared arms that had each run their
own Optuna search, so any alignment effect was inseparable from search noise.
Pooling across pairs instead of merging on them would reintroduce exactly the
confound the harness removes, so a pair missing one policy is DROPPED, never
averaged around -- and every verdict reports the surviving pair count first,
so a criterion that passed on three pairs cannot be read as one that passed on
three hundred.
"""
import pandas as pd

from src.p4gen.evaluation import band_factor
from src.training.align_budget import band_ceiling

# One model pair. overlap_threshold is part of the identity because the same
# (arm, M, split, k) refit is replayed once per threshold.
PAIR_KEYS = ['source_arm', 'M', 'split', 'k', 'overlap_threshold']

# The companion document's measured baseline: the share of shed bits that
# crossed no band boundary across the whole campaign.
WASTED_BITS_BASELINE = 0.578

# One flipped validation sample at n ~ 3000 is ~0.0003, so -0.001 reads as
# "no worse than noise" rather than "no worse at all".
ACCURACY_TOLERANCE = -0.001


def derive_columns(frame):
    """factor / bits_shed / wasted_bits, derived rather than stored.

    align_stats deliberately records only the primitives (codeword_before,
    codeword_after, codeword_floor, spent_budget, rolled_back); everything here
    is a pure function of those plus the band arithmetic, so storing it twice
    would be two places to get it wrong.

    wasted_bits is the honest version of "bits that bought nothing": bits shed
    below the CEILING of the band the run actually landed in. If no band was
    crossed that is every bit shed; if one was, it is only the overshoot past
    the boundary. Capped at bits_shed so a run that started deep inside its
    band is not charged for bits it never shed.

    policy='none' rows are dropped here: 'none' is a pseudo-policy that skips
    align_with_policy entirely (scripts/replay_alignment.py's run_one_policy),
    so it carries no align_* columns at all -- band_factor would raise on the
    resulting NaN. 'none' is an unaligned control never meant to be scored by
    S1-S6, all of which reason about alignment's effect, so excluding it here
    is correct, not merely a crash workaround.
    """
    out = frame[frame['policy'] != 'none'].copy()
    out['factor_before'] = out['align_codeword_before'].apply(band_factor)
    out['factor_after'] = out['align_codeword_after'].apply(band_factor)
    out['bits_shed'] = out['align_codeword_before'] - out['align_codeword_after']
    overshoot = (out['factor_after'].apply(band_ceiling)
                 - out['align_codeword_after']).clip(lower=0)
    out['wasted_bits'] = pd.concat([out['bits_shed'], overshoot], axis=1).min(axis=1)
    return out


def _paired(frame, policy_a, policy_b):
    """Rows of two policies merged on PAIR_KEYS, suffixed _a / _b."""
    left = frame[frame['policy'] == policy_a]
    right = frame[frame['policy'] == policy_b]
    return left.merge(right, on=PAIR_KEYS, suffixes=('_a', '_b'))


def _verdict(n, passed, value, detail):
    return {'passed': bool(passed), 'value': float(value),
            'detail': '{} paired: {}'.format(n, detail)}


def score(frame):
    """The two surviving pre-registered verdicts.

    S1/S2/S6 paired an aligned policy against 'legacy', and S4 paired 'c1c2'
    against 'c1'; the 2026-08-30 design deletes the policy ladder, so all four
    lost their referent and are removed rather than left to fail silently on
    an empty merge. Their measured values remain in the findings documents as
    historical results, which is where a superseded comparison belongs. S3 and
    S5 reason within one policy and are unaffected.

    Deliberately objective-BLIND: S3 and S5 are pre-registered against the
    block domain, so re-pointing them at a run optimising stages would
    invalidate them. The objective axis is scored separately, by
    scripts/score_objective_replay.py.
    """
    out = {}

    # S3 -- the wasted-bit share falls from the campaign's measured 57.8%.
    aligned = frame[frame['policy'] == 'aligned']
    shed = aligned['bits_shed'].sum()
    share = (aligned['wasted_bits'].sum() / shed) if shed else 1.0
    out['S3'] = _verdict(len(aligned), share < WASTED_BITS_BASELINE, share,
                         'wasted {:.1%} of {} shed bits (baseline {:.1%})'.format(
                             share, int(shed), WASTED_BITS_BASELINE))

    # S5 -- C3(c): does loosening the candidate gate widen the generator at
    # all? Compared within one policy across overlap_threshold, so the merge
    # key drops that column.
    keys = [k for k in PAIR_KEYS if k != 'overlap_threshold']
    loose_threshold = aligned['overlap_threshold'].min()
    tight_threshold = aligned['overlap_threshold'].max()
    loose = aligned[aligned['overlap_threshold'] == loose_threshold]
    tight = aligned[aligned['overlap_threshold'] == tight_threshold]
    pair5 = loose.merge(tight, on=keys, suffixes=('_a', '_b'))
    if len(pair5) and loose_threshold < tight_threshold:
        delta = (pair5['bits_shed_a'] - pair5['bits_shed_b']).mean()
        out['S5'] = _verdict(len(pair5), delta > 0, delta,
                             'mean {:+.2f} bits shed at threshold {:g} vs {:g}'.format(
                                 delta, loose_threshold, tight_threshold))
    else:
        out['S5'] = _verdict(len(pair5), False, 0.0,
                             'no loosened-threshold rows to compare')

    return out
