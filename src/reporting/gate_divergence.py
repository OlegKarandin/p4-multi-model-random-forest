"""How often the superseded band gate and the corrected block gate disagree.

The measurement design 2026-09-07 §7 step 1 asks for, and the one number this
repair is justified by. Both verdicts are read off columns the run itself
recorded, so this is a measurement rather than a reconstruction:
align_blocks_before/after are the real per-table factors, and
align_codeword_before/after are the scalars the band gate actually consulted.

NOTE ON PROVENANCE. The design says "re-score archived replay CSVs". There are
none -- results/ is gitignored and no committed CSV carries align_* columns --
so this scores a FRESH replay run through the instrumented code instead. That
is strictly better: an archived row would have forced reconstructing the factor
from align_key_bytes alone, which cannot see version_block_penalty at all,
since that charge depends on the width multiset rather than any scalar (§1.3).
"""
from src.p4gen.evaluation import codeword_bits_to_blocks


def score_gate_divergence(frame):
    """Counts, over the rows where the gate was actually consulted.

    false_positive : a band was crossed but the block factor did not move.
        crossed_a_boundary returned True, the rollback declined to fire, and
        accuracy was spent for nothing -- the exact failure C1 was built to
        prevent (§1.4).
    false_negative : the block factor dropped but no band was crossed, so the
        run was discarded and re-run at delta = 0, throwing a REAL block
        saving away.
    gate_can_open : rows whose floor factor is strictly cheaper than their
        entry factor -- i.e. where the corrected BlockBudget could authorise
        spending at all. Answers §8's "the corrected gate may never open"
        risk before any behaviour changes.

    Only rows that actually spent budget are counted: align_with_policy returns
    the speculative result before consulting crossed_a_boundary when
    spent_budget is False, so a free-moves row was never judged by either gate
    and would inflate both counts with runs the gate never saw.
    """
    aligned = frame[(frame['policy'] == 'aligned')
                    & frame['align_spent_budget'].astype(bool)].copy()
    band = (aligned['align_codeword_after'].apply(codeword_bits_to_blocks)
            < aligned['align_codeword_before'].apply(codeword_bits_to_blocks))
    block = aligned['align_blocks_after'] < aligned['align_blocks_before']
    judged = frame[frame['policy'] == 'aligned']
    return {
        'n': int(len(aligned)),
        'false_positive': int((band & ~block).sum()),
        'false_negative': int((~band & block).sum()),
        'agree': int((band == block).sum()),
        'gate_can_open': int((judged['align_blocks_floor']
                              < judged['align_blocks_before']).sum()),
    }
