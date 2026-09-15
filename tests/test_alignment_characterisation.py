"""Replays tests/fixtures/alignment_characterisation.json.

A RECORD, not a requirement (see the dumper's docstring). This test asserts
equality so that any behavioural change is loud; the correct response to a
failure in a task that INTENDS to change behaviour is to regenerate the fixture
and explain the diff in the commit message, not to loosen the assertion.
"""
import json
import os

import pytest

from scripts.dump_alignment_characterisation import capture

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'alignment_characterisation.json')


def _expected():
    with open(FIXTURE, encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture(scope='module')
def observed():
    """One capture for the whole module -- the alignment run over a 7-tree
    pair is the expensive part, and every test below reads the same result."""
    return capture()


def test_the_fixture_covers_every_configuration(observed):
    expected = _expected()
    assert len(expected['rows']) == len(observed['rows'])
    assert [r['config'] for r in expected['rows']] == \
        [r['config'] for r in observed['rows']]


def test_alignment_reaches_the_recorded_thresholds(observed):
    for want, got in zip(_expected()['rows'], observed['rows']):
        assert got['thresholds'] == want['thresholds'], got['config']


def test_alignment_reports_the_recorded_stats(observed):
    for want, got in zip(_expected()['rows'], observed['rows']):
        assert got['stats'] == want['stats'], got['config']


def test_alignment_accepts_the_recorded_candidates(observed):
    for want, got in zip(_expected()['rows'], observed['rows']):
        assert got['accepted'] == want['accepted'], got['config']
