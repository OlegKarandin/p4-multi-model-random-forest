"""Pin identity (spec 2026-10-04 Sec I.4 item 3): regenerating a
campaign_2026_10 design from its stored forests must give the byte-identical
.p4 it was compiled from. The fill-low layout fix must not move a single
@pa_container_size pin.

Skipped when the campaign run is not on disk (results/ is gitignored)."""
import hashlib
import os
import tempfile

import pytest

RUN = os.path.join('results', 'campaign_2026_10')
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(RUN, 'forests')),
    reason='results/campaign_2026_10 is not on disk')


def _sample_rows(n_spread=20):
    """>= 20 rows across arms, every row whose program has a 49-56-bit
    code_* field among the first 2000, plus an even spread."""
    from src.p4gen.p4_replay import parse_program
    designs = os.path.join(RUN, 'designs')
    ids = sorted(name[:-3] for name in os.listdir(designs) if name.endswith('.p4')
                 and os.path.isfile(os.path.join(RUN, 'forests', name[:-3] + '.joblib')))
    step = max(1, len(ids) // n_spread)
    chosen = set(ids[::step])
    fill_low = []
    for row_id in ids[:2000]:
        bits = parse_program(os.path.join(designs, row_id + '.p4')).bits
        if any(name.startswith('code_') and 49 <= b <= 56 for name, b in bits.items()):
            fill_low.append(row_id)
        if len(fill_low) >= 10:
            break
    assert fill_low, 'no campaign design has a 49-56-bit code_* field'
    return sorted(chosen | set(fill_low))


def _sha256(path):
    with open(path, 'rb') as handle:
        # The campaign ran on Linux (LF); a Windows checkout writes CRLF.
        return hashlib.sha256(handle.read().replace(b'\r\n', b'\n')).hexdigest()


def test_regenerated_programs_are_byte_identical():
    from src.training.row_artifacts import RowContext, _generate, load_forests
    rows = _sample_rows()
    assert len(rows) >= 20
    slugs = {row_id.split('_M')[0] for row_id in rows}
    assert {'independent', 'joint', 'joint-off'} <= slugs
    with tempfile.TemporaryDirectory() as tmp:
        for row_id in rows:
            stored = load_forests(os.path.join(RUN, 'forests', row_id + '.joblib'))
            encoding = 'disjoint' if row_id.startswith('independent') else 'joint'
            ctx = RowContext(tmp, row_id.split('_M')[0], float('inf'), 'test')
            fresh = _generate(ctx, row_id, stored['app'], stored['ddos'],
                              stored['features_app'], stored['features_ddos'], encoding)
            assert _sha256(fresh) == _sha256(
                os.path.join(RUN, 'designs', row_id + '.p4')), row_id
