"""Pure interval geometry for damage-ranked target selection (C2).

Everything here is side-effect-free and imports nothing from
threshold_alignment -- that module imports FROM this one. The split exists
because C2 needs to ask "what would this target do?" about four candidates
before committing to one, and today that question can only be answered by
performing the mutation and catching AlignmentInvariantError out of the middle
of it.
"""
from src.p4gen.build_p4_script import INFINITE


def candidate_targets(range1, range2):
    """The four corner targets for an overlapping pair, intersection first.

    Every corner uses endpoints ALREADY PRESENT in the pooled threshold set,
    which is what preserves the termination argument in threshold_alignment's
    module docstring (lines 20-28): every write relocates a threshold to a
    value already present rather than introducing a new one, so
    joint_interval_count stays non-increasing move-by-move. Interior snap
    points would break that premise and are deliberately not offered here.

    Intersection is first so that a pair with exactly one admissible candidate
    reproduces the legacy choice byte for byte.

    Deduped, order preserving: when s1 == s2 or e1 == e2 the corners coincide,
    and offering the same target twice would pay for a second oracle call that
    cannot decide differently.
    """
    s1, e1 = range1
    s2, e2 = range2
    corners = [
        (max(s1, s2), min(e1, e2)),   # intersection -- the legacy rule
        (min(s1, s2), max(e1, e2)),   # union
        (max(s1, s2), max(e1, e2)),
        (min(s1, s2), min(e1, e2)),
    ]
    out = []
    for corner in corners:
        if corner not in out:
            out.append(corner)
    return out


def boundary_moves(source_range, target_range):
    """The (old_threshold, new_threshold) writes adjust_range_boundaries would
    make for one model.

    Mirrors that function's guards exactly: a boundary is expressed as
    `start - 1` on the min side, and a boundary sitting ON a sentinel (0 at the
    bottom, INFINITE at the top) is never moved -- every feature's tiling
    starts at 0 and ends at INFINITE, so declining is the common case, not an
    edge case. A target whose moves are empty for BOTH models has gain 0 and
    must be dropped before it costs an oracle evaluation.
    """
    source_min, source_max = source_range
    target_min, target_max = target_range

    moves = []
    for source, target, sentinel in (
        (source_min - 1 if source_min > 0 else source_min,
         target_min - 1 if target_min > 0 else target_min, 0),
        (source_max, target_max, INFINITE),
    ):
        if source != target and source != sentinel:
            moves.append((source, target))
    return moves


def neighbour_writes(ranges, target_idx, old_range, new_range):
    """Every write update_neighboring_ranges_and_index would make, as data.

    Returns (effective_range, writes, inverted):
      effective_range : what ranges[target_idx] becomes, after mirroring
                        adjust_range_boundaries' sentinel guards -- `ranges`
                        must never claim a boundary moved that the model
                        refused to move, which is the C5 bug.
      writes          : [(index, (lo, hi))] for every neighbour that absorbs
                        the boundary move.
      inverted        : the first neighbour tuple that would invert, or None.

    PURE: `ranges` is only read.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range

    threshold_old_min = old_min - 1 if old_min > 0 else old_min
    threshold_old_max = old_max

    effective_min = new_min if threshold_old_min != 0 else old_min
    effective_max = new_max if threshold_old_max != INFINITE else old_max
    effective_range = (effective_min, effective_max)

    writes, inverted = [], None
    if effective_range != old_range:
        for i, (range_min, range_max) in enumerate(ranges):
            if i == target_idx:
                continue

            new_range_min, new_range_max = range_min, range_max
            if range_max + 1 == old_min:
                new_range_max = effective_min - 1
            if range_min - 1 == old_max:
                new_range_min = effective_max + 1

            if new_range_min > new_range_max:
                if inverted is None:
                    inverted = ((range_min, range_max),
                                (new_range_min, new_range_max))
                continue

            if new_range_min != range_min or new_range_max != range_max:
                writes.append((i, (new_range_min, new_range_max)))

    return effective_range, writes, inverted


def target_admissible(ranges, target_idx, old_range, new_range):
    """Whether rewriting this boundary would invert a neighbouring interval.

    The intersection target only ever shrinks the aligned interval and widens
    its neighbours, so it can never invert one -- which is why the mutator's
    raise was unreachable before C2. Union and mixed targets widen the aligned
    interval and shrink neighbours, so this filter is what makes them usable.
    """
    return neighbour_writes(ranges, target_idx, old_range, new_range)[2] is None


def hypothetical_ranges(ranges, target_idx, old_range, new_range):
    """`ranges` as it would look after this target, or None if inadmissible.

    Used to price a candidate's bit gain without touching the real lists.
    """
    effective_range, writes, inverted = neighbour_writes(
        ranges, target_idx, old_range, new_range)
    if inverted is not None:
        return None
    out = list(ranges)
    out[target_idx] = effective_range
    for i, tup in writes:
        out[i] = tup
    return out
