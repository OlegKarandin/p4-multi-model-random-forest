"""Pure interval geometry for damage-ranked target selection (C2).

Everything here is side-effect-free and imports nothing from
threshold_alignment -- that module imports FROM this one. The split exists
because C2 needs to ask "what would this target do?" about four candidates
before committing to one, and today that question can only be answered by
performing the mutation and catching AlignmentInvariantError out of the middle
of it.
"""
from src.p4gen.build_p4_script import INFINITE


# The range's lower EDGE, as a boundary value (spec 2026-09-28 T2).
#
# A boundary of interval (s, e) is expressed as the cut it stands for: the
# lower boundary is the cut `x <= s - 1`, ALWAYS, and the upper is `x <= e`.
# For a first interval (0, c) there is no cut below it -- s - 1 is the edge,
# and it must be a value no real cut can take, which is why it is -1 and not
# 0. Until 2026-09-28 the lower boundary was `s - 1 if s > 0 else s`, so the
# edge of (0, c) and the real, learned cut `x <= 0` below (1, c) were both 0,
# and both were frozen by a "sentinel 0" guard: a real cut at 0 never moved,
# and a move whose TARGET lower bound was 0 wrote a threshold of 0 -- CREATING
# a cut at 0 while believing it had moved to the edge.
#
# Edges never move and no cut is ever moved onto one (invariant 2): the top
# edge is INFINITE, the bottom one this.
LOWER_EDGE = -1


def lower_boundary(start):
    """The cut below an interval starting at `start`: `start - 1`, which is
    LOWER_EDGE for a first interval (0, c) and the real cut 0 for (1, c)."""
    return start - 1


def boundary_pairs(source_range, target_range):
    """[(source, target, edge)] for the lower and the upper boundary, in that
    order -- the one statement of the boundary convention that
    boundary_moves, neighbour_writes and threshold_alignment's
    adjust_range_boundaries / update_neighboring_ranges_and_index all share,
    so the four cannot drift apart again."""
    (source_min, source_max), (target_min, target_max) = source_range, target_range
    return [(lower_boundary(source_min), lower_boundary(target_min), LOWER_EDGE),
            (source_max, target_max, INFINITE)]


def boundary_moves_to(source, target, edge):
    """Whether this one boundary really moves: it changes, and neither the
    SOURCE nor the TARGET is the edge. A source on the edge has no cut to
    move; a target on the edge would delete the cut (invariant 2)."""
    return source != target and source != edge and target != edge


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

    Mirrors that function's guards exactly (both go through boundary_pairs
    and boundary_moves_to): a boundary is expressed as the cut `start - 1` on
    the min side and `end` on the max side, and a boundary whose source OR
    target is an EDGE (LOWER_EDGE at the bottom, INFINITE at the top) is
    never moved -- every feature's tiling starts at 0 and ends at INFINITE, so
    declining is the common case, not an edge case. A target whose moves are
    empty for BOTH models has gain 0 and must be dropped before it costs an
    oracle evaluation.
    """
    return [(source, target)
            for source, target, edge in boundary_pairs(source_range, target_range)
            if boundary_moves_to(source, target, edge)]


def neighbour_writes(ranges, target_idx, old_range, new_range):
    """Every write update_neighboring_ranges_and_index would make, as data.

    Returns (effective_range, writes, inverted):
      effective_range : what ranges[target_idx] becomes, after mirroring
                        adjust_range_boundaries' edge guards per boundary --
                        `ranges` must never claim a boundary moved that the
                        model refused to move, which is the C5 bug. Since
                        T2 (2026-09-28) that includes a refused TARGET edge:
                        (6, 15) -> (0, 10) is effectively (6, 10).
      writes          : [(index, (lo, hi))] for every neighbour that absorbs
                        the boundary move.
      inverted        : the first neighbour tuple that would invert, or None.

    PURE: `ranges` is only read.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range

    (lo_src, lo_dst, lo_edge), (hi_src, hi_dst, hi_edge) = boundary_pairs(
        old_range, new_range)
    effective_min = new_min if boundary_moves_to(lo_src, lo_dst, lo_edge) else old_min
    effective_max = new_max if boundary_moves_to(hi_src, hi_dst, hi_edge) else old_max
    effective_range = (effective_min, effective_max)

    writes, inverted = [], None
    if effective_range != old_range:
        # Only an immediate neighbour can absorb this move: `ranges` is a
        # gap-free tiling (ranges[i][1] + 1 == ranges[i+1][0] for every
        # consecutive pair) and old_range == ranges[target_idx] at every
        # call site, so `range_max + 1 == old_min` holds only for
        # ranges[target_idx - 1] and `range_min - 1 == old_max` only for
        # ranges[target_idx + 1] -- no other index can match either
        # condition. Scanning the whole list here was O(n) per candidate
        # target where O(1) suffices; verified against the full O(n) scan
        # over 21734 calls of a real alignment run (delta in {0.05, 0.0,
        # None}, objective in {'blocks', 'stages'}) with zero divergence.
        for i in (target_idx - 1, target_idx + 1):
            if i < 0 or i >= len(ranges):
                continue
            range_min, range_max = ranges[i]

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
