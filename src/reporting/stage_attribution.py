"""Stage-depth attribution: decompose joint vs. counterfactual-disjoint
stage depth into a range term and a ternary-spill term, and score D1-D4
(spec docs/superpowers/specs/2026-08-27-stage-depth-attribution-design.md
Sec 5) per (M, k) cell -- never pooled, per Sec 5.3, because Sec 8 of the
overlap/delta/M/k findings document showed pooling across M hid a complete
sign flip in the k=17 story.

Requires scripts/replay_alignment.py to have been run with
`--policies none,aligned` (or whichever subset is available) so
every row carries the joint_*/counterfactual_disjoint_* columns Phase 1
(src/p4gen/evaluation.py's ResourceUsage) and Phase 2's run_one_policy
change add.

D2 and D3 are evaluated on the 'none' policy by default (see score's
`policy` parameter): they ask whether joint encoding is structurally
shallower on IDENTICAL, UNALIGNED models -- the pure encoding effect,
kept separate from alignment's own effect, which D4 measures.
"""
import pandas as pd

# One (source_arm, M, split, k, overlap_threshold) identifies one refit
# model pair -- the same identity replay_scoring.PAIR_KEYS uses, and for the
# same reason: every criterion here is paired on a real model pair, never
# pooled across pairs before merging.
PAIR_KEYS = ['source_arm', 'M', 'split', 'k', 'overlap_threshold']
CELL_KEYS = ['M', 'k']


def derive_columns(frame):
    """ternary_spill and the three per-row deltas the identity
    Δ_encoding = Δ_range + Δ_ternary_spill (spec Sec 5.1) is built from --
    exact by construction, since all four inputs come from the same
    StagePlan objects (pinned by
    test_derive_columns_satisfies_the_encoding_identity_exactly)."""
    out = frame.copy()
    for prefix in ('joint_', 'counterfactual_disjoint_'):
        out[prefix + 'ternary_spill'] = (
            out[prefix + 'ternary_depth'] - out[prefix + 'range_depth'])
    out['delta_range'] = (out['counterfactual_disjoint_range_depth']
                          - out['joint_range_depth'])
    out['delta_ternary_spill'] = (
        out['counterfactual_disjoint_ternary_spill']
        - out['joint_ternary_spill'])
    out['delta_encoding'] = (out['counterfactual_disjoint_stage_depth']
                             - out['joint_stage_depth'])
    return out


def _d1(frame):
    """Premise check (Sec 5.2): register_depth must be identical under both
    encodings on EVERY row, not sometimes. It's a structural invariant
    (Sec 2.1) driven only by the selected feature set, which encoding never
    changes -- so any disagreement means the replay data is broken, not
    that the mechanism failed."""
    mismatched = frame[
        frame['joint_register_depth']
        != frame['counterfactual_disjoint_register_depth']]
    passed = len(frame) > 0 and len(mismatched) == 0
    return {'passed': bool(passed), 'n': int(len(frame)),
           'n_mismatched': int(len(mismatched)),
           'detail': (
               'register_depth agrees on all {} rows'.format(len(frame))
               if passed else
               '{} / {} rows disagree on register_depth -- attribution invalid'
               .format(len(mismatched), len(frame)))}


def _per_cell_mean(frame, value_columns):
    """Group by (M, k); mean of each named column per cell, plus n. Sec
    5.3: reported per cell with explicit n -- pooled figures never replace
    the per-cell table."""
    grouped = frame.groupby(CELL_KEYS)
    result = grouped[list(value_columns)].mean().reset_index()
    result['n'] = grouped.size().values
    return result


def _empty_cells(detail):
    return {'cells': pd.DataFrame(columns=CELL_KEYS + ['n', 'passed']),
           'fraction_passing': 0.0, 'n_cells': 0, 'detail': detail}


def _independent_pairs(subset, policy):
    """Collapse rows that are byte-identical replays of the same model pair
    down to one row per independent pair, so a cell's n (Sec 5.3) counts
    model pairs, not sweep cells.

    'none' is the only policy proven to ignore overlap_threshold entirely
    (run_one_policy skips align_with_policy -- the only place
    overlap_threshold is used -- for 'none'), so replaying one pair at N
    swept --overlap-thresholds values produces N identical 'none' rows that
    differ only in that column. An aligned policy like 'aligned' can
    genuinely produce different results at different overlap_threshold
    values, so it is never deduped here."""
    if policy != 'none':
        return subset
    dedupe_keys = [k for k in PAIR_KEYS if k != 'overlap_threshold']
    return subset.drop_duplicates(subset=dedupe_keys)


def _d2(frame, policy):
    """mean Δ_encoding > 0 within each (M, k) cell -- the headline claim
    that joint is genuinely shallower on identical, unaligned models."""
    subset = frame[frame['policy'] == policy]
    if not len(subset):
        return _empty_cells('no rows for policy={!r}'.format(policy))
    subset = _independent_pairs(subset, policy)
    cells = _per_cell_mean(subset, ['delta_encoding'])
    cells['passed'] = cells['delta_encoding'] > 0
    fraction = float(cells['passed'].mean())
    return {'cells': cells, 'fraction_passing': fraction,
           'n_cells': int(len(cells)),
           'detail': '{}/{} (M, k) cells pass on policy={!r}'.format(
               int(cells['passed'].sum()), len(cells), policy)}


def _d3(frame, policy):
    """Δ_range > 0 AND Δ_ternary_spill < 0 within each (M, k) cell -- the
    predicted mechanism, directions fixed in advance (Sec 5.2)."""
    subset = frame[frame['policy'] == policy]
    if not len(subset):
        return _empty_cells('no rows for policy={!r}'.format(policy))
    subset = _independent_pairs(subset, policy)
    cells = _per_cell_mean(subset, ['delta_range', 'delta_ternary_spill'])
    cells['passed'] = ((cells['delta_range'] > 0)
                       & (cells['delta_ternary_spill'] < 0))
    fraction = float(cells['passed'].mean())
    return {'cells': cells, 'fraction_passing': fraction,
           'n_cells': int(len(cells)),
           'detail': '{}/{} (M, k) cells pass on policy={!r}'.format(
               int(cells['passed'].sum()), len(cells), policy)}


def _d4(frame, none_policy):
    """Δ_align(joint) > 0 AND Δ_align(joint) > Δ_align(counterfactual_disjoint)
    within each (M, k) cell, for every aligned policy present against the
    unaligned control. Merged on PAIR_KEYS per model pair -- the whole
    reason the replay exists (Sec 5.1) -- never pooled across pairs before
    merging."""
    none_rows = frame[frame['policy'] == none_policy]
    aligned_policies = sorted(
        p for p in frame['policy'].unique() if p != none_policy)
    out = {}
    for policy in aligned_policies:
        aligned_rows = frame[frame['policy'] == policy]
        paired = none_rows.merge(
            aligned_rows, on=PAIR_KEYS, suffixes=('_none', '_aligned'))
        if not len(paired):
            out[policy] = _empty_cells(
                'no paired {}/{} rows'.format(none_policy, policy))
            continue
        paired['delta_align_joint'] = (
            paired['joint_stage_depth_none']
            - paired['joint_stage_depth_aligned'])
        paired['delta_align_counterfactual_disjoint'] = (
            paired['counterfactual_disjoint_stage_depth_none']
            - paired['counterfactual_disjoint_stage_depth_aligned'])
        cells = _per_cell_mean(
            paired, ['delta_align_joint', 'delta_align_counterfactual_disjoint'])
        cells['passed'] = (
            (cells['delta_align_joint'] > 0)
            & (cells['delta_align_joint']
               > cells['delta_align_counterfactual_disjoint']))
        fraction = float(cells['passed'].mean())
        out[policy] = {'cells': cells, 'fraction_passing': fraction,
                      'n_cells': int(len(cells)),
                      'detail': '{}/{} (M, k) cells pass for policy={!r}'.format(
                          int(cells['passed'].sum()), len(cells), policy)}
    return out


def score(frame, policy='none'):
    """D1-D4 verdicts (spec Sec 5.2). D1 is a premise check: if it fails,
    D2-D4 are not computed -- Sec 5.2 states the attribution is invalid in
    that case, so reporting derived numbers would misrepresent them as
    meaningful.

    --verify-injected rows (scripts/replay_alignment.py's replay_row,
    policy='verify') are dropped before D1 is even computed: a verify row
    re-runs 'aligned' at the row's own recorded arm delta, not the ladder
    delta being swept, so leaving it in would inflate D1's n with a
    duplicate of a pair already covered by a real policy and give D4 a
    spurious 'verify' entry compared against the wrong alignment budget.

    Refuses a multi-objective frame outright rather than silently pooling
    across it: Task 6 made 'objective' part of a replay row's identity (three
    can now appear per model pair in one CSV), and PAIR_KEYS does not name
    it, so a multi-objective frame would triplicate every aligned pair in
    D1/D4 -- exactly the cross-pair-pooling confound PAIR_KEYS exists to
    prevent, just along an axis it doesn't cover.
    """
    if 'objective' in frame.columns and frame['objective'].nunique() > 1:
        raise ValueError(
            "score() is objective-blind by design and must not be handed a "
            "multi-objective frame (found {} distinct objectives) -- pooling "
            "would triplicate every aligned pair, reintroducing exactly the "
            "cross-pair confound PAIR_KEYS exists to remove. Filter to one "
            "objective before scoring.".format(frame['objective'].nunique()))

    frame = frame[frame['policy'] != 'verify']

    d1 = _d1(frame)
    if not d1['passed']:
        return {'D1': d1,
               'D2': _empty_cells('D1 failed -- attribution invalid, not computed'),
               'D3': _empty_cells('D1 failed -- attribution invalid, not computed'),
               'D4': {}}

    derived = derive_columns(frame)
    return {'D1': d1, 'D2': _d2(derived, policy), 'D3': _d3(derived, policy),
           'D4': _d4(derived, policy)}
