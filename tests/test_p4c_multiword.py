"""Tree tables of more than 512 entries (several 512-row words) in one stage: the model against
real p4c compiles (T5 review, model finding M-MED, probe 2026-11-17).

p4c places each 512-row word of a tree table as ONE run in ONE column of the stage's 2 x 12
TCAM blocks and never splits a word across columns, so a stage holding words of 5,5,5 and 4,4
blocks (or a lone table of three 7-block words) does not fit one stage although the totals do.
Each fixture is a tests/fixtures/p4c_base_17b program with only the `size =` of one or two tree
tables changed; expected.json is p4c's committed stage depth and blocks (local WSL compile;
evidence: results/encoding_evidence_2026_11/model8/review_2026_11_17/probe_multiword/)."""
import json
import os

import pytest

from p4gen.p4_replay import model_breakdown_text

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'p4c_multiword')
with open(os.path.join(_HERE, 'expected.json'), encoding='utf-8') as _handle:
    _EXPECTED = json.load(_handle)


@pytest.mark.parametrize('case', sorted(_EXPECTED))
def test_the_model_prices_multi_word_tree_tables_as_p4c_places_them(case):
    with open(os.path.join(_HERE, case + '.p4'), encoding='utf-8') as handle:
        result = model_breakdown_text(handle.read())
    assert result.get('refusal') is None, result.get('refusal_reason')
    assert (result['stage_depth'], result['blocks']) == (
        _EXPECTED[case]['stage_depth'], _EXPECTED[case]['blocks']), case
