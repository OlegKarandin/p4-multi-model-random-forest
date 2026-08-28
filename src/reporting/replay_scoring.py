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


def _best_policy(frame):
    """Name of the best available policy actually present in this data.

    S5 and S6 both mean "how does the best mechanism we have do", so they
    must agree on what "best" resolves to: prefer c1c2, else c1, else
    legacy -- whichever the run actually swept, never a hardcoded assumption
    that all three exist. Returns None if none of the three is present.
    """
    present = frame['policy'].unique()
    for policy in ('c1c2', 'c1', 'legacy'):
        if policy in present:
            return policy
    return None


def score(frame):
    """The six pre-registered verdicts. Missing policies yield n = 0, which
    fails rather than silently passing on an empty mean."""
    out = {}

    # S1 -- c1 vs legacy at the SAME configured --ladder-delta (there is no
    # legacy-at-delta_rel=0 arm to compare against "spending nothing"). A
    # pass is evidence C1's own gating (BandBudget only spends when the next
    # block-band is reachable) doesn't regress accuracy under the shared
    # budget -- it is not a direct test of the align_with_policy rollback
    # wrapper specifically, since C1's gating alone would hold this even if
    # the rollback did nothing.
    pair = _paired(frame, 'c1', 'legacy')
    if len(pair):
        d_app = (pair['acc_app_a'] - pair['acc_app_b']).mean()
        d_ddos = (pair['acc_ddos_a'] - pair['acc_ddos_b']).mean()
        worst = min(d_app, d_ddos)
        out['S1'] = _verdict(len(pair), worst >= ACCURACY_TOLERANCE, worst,
                             'mean d_acc app {:+.4f} ddos {:+.4f}'.format(d_app, d_ddos))
    else:
        out['S1'] = _verdict(0, False, 0.0, 'no c1/legacy pairs')

    # S2 -- C1 DOMINATES legacy: blocks no worse AND accuracy no worse, per
    # task, per pair. A mean would let a win on one pair pay for a loss on
    # another, which is not what domination means.
    if len(pair):
        dominates = ((pair['blocks_a'] <= pair['blocks_b'])
                     & (pair['acc_app_a'] >= pair['acc_app_b'] + ACCURACY_TOLERANCE)
                     & (pair['acc_ddos_a'] >= pair['acc_ddos_b'] + ACCURACY_TOLERANCE))
        share = dominates.mean()
        out['S2'] = _verdict(len(pair), share == 1.0, share,
                             '{:.1%} of pairs dominated'.format(share))
    else:
        out['S2'] = _verdict(0, False, 0.0, 'no c1/legacy pairs')

    # S3 -- the wasted-bit share falls from the campaign's measured 57.8%.
    c1 = frame[frame['policy'] == 'c1']
    shed = c1['bits_shed'].sum()
    share = (c1['wasted_bits'].sum() / shed) if shed else 1.0
    out['S3'] = _verdict(len(c1), share < WASTED_BITS_BASELINE, share,
                         'wasted {:.1%} of {} shed bits (baseline {:.1%})'.format(
                             share, int(shed), WASTED_BITS_BASELINE))

    # S4 -- C2 buys the SAME bits more cheaply, so restrict to pairs where the
    # two policies shed identically and ask only about accuracy.
    pair2 = _paired(frame, 'c1c2', 'c1')
    equal = pair2[pair2['bits_shed_a'] == pair2['bits_shed_b']]
    if len(equal):
        gain = ((equal['acc_app_a'] - equal['acc_app_b'])
                + (equal['acc_ddos_a'] - equal['acc_ddos_b'])).mean() / 2
        out['S4'] = _verdict(len(equal), gain > 0, gain,
                             'mean d_acc {:+.4f} at equal bits shed'.format(gain))
    else:
        out['S4'] = _verdict(0, False, 0.0, 'no equal-shed c1c2/c1 pairs')

    # S5 -- C3(c): does loosening the candidate gate widen the generator at
    # all? Compared within one policy across overlap_threshold, so the
    # merge key drops that column.
    keys = [k for k in PAIR_KEYS if k != 'overlap_threshold']
    top5 = _best_policy(frame)
    best = frame[frame['policy'] == top5] if top5 is not None else frame.iloc[0:0]
    loose_threshold = best['overlap_threshold'].min()
    tight_threshold = best['overlap_threshold'].max()
    loose = best[best['overlap_threshold'] == loose_threshold]
    tight = best[best['overlap_threshold'] == tight_threshold]
    pair5 = loose.merge(tight, on=keys, suffixes=('_a', '_b'))
    if len(pair5) and loose_threshold < tight_threshold:
        delta = (pair5['bits_shed_a'] - pair5['bits_shed_b']).mean()
        out['S5'] = _verdict(len(pair5), delta > 0, delta,
                             'mean {:+.2f} bits shed at threshold {:g} vs {:g}'.format(
                                 delta, loose_threshold, tight_threshold))
    else:
        out['S5'] = _verdict(len(pair5), False, 0.0,
                             'no loosened-threshold rows to compare')

    # S6 -- is the whole ladder worth a campaign? Best available policy against
    # legacy, on blocks, with no accuracy regression on either task.
    top = _best_policy(frame)
    pair6 = _paired(frame, top, 'legacy') if top is not None else frame.iloc[0:0]
    if len(pair6):
        saved = (pair6['blocks_b'] - pair6['blocks_a']).mean()
        d_app = (pair6['acc_app_a'] - pair6['acc_app_b']).mean()
        d_ddos = (pair6['acc_ddos_a'] - pair6['acc_ddos_b']).mean()
        passed = (saved >= 1.0 and d_app >= ACCURACY_TOLERANCE
                  and d_ddos >= ACCURACY_TOLERANCE)
        out['S6'] = _verdict(len(pair6), passed, saved,
                             '{} saves {:.2f} blocks/pair, d_acc app {:+.4f} '
                             'ddos {:+.4f}'.format(top, saved, d_app, d_ddos))
    else:
        out['S6'] = _verdict(0, False, 0.0, 'no ladder/legacy pairs')

    return out
