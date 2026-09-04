"""Unit tests for scripts/compiler_calibration.py.

Pure-function tests only (sample construction, gap arithmetic, resume
logic, and run_one_row's F2 degrade path via monkeypatching) -- the real
compile is exercised by the plan's Task 4 gate, not by this suite (spec
§4)."""
import json
import os

import pandas as pd
import pytest

import scripts.compiler_calibration as cc
from scripts.compiler_calibration import (
    GROUPS,
    K_BANDS,
    STRATA_STAGE_DEPTHS,
    add_group_and_band,
    build_sample,
    k_band,
    stratum_row,
)


def test_k_band_low_at_and_below_5():
    assert k_band(0) == 'low'
    assert k_band(5) == 'low'


def test_k_band_high_at_and_above_13():
    assert k_band(13) == 'high'
    assert k_band(17) == 'high'


def test_k_band_none_in_the_middle():
    assert k_band(6) is None
    assert k_band(12) is None


def _synthetic_frame(rows):
    """rows: list of dicts with at least arm_slug, k, stage_depth. Fills in
    every other column build_sample/run_one_row might touch with a cheap
    placeholder, so tests can pass minimal per-row overrides."""
    base = {'M': 100, 'split': 10, 'best_params': '{}',
            'features_app': 'f1', 'features_ddos': 'f1', 'blocks': 10}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_add_group_and_band_maps_independent_and_joint_d000():
    frame = _synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'joint-d000', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'joint-d020', 'k': 2, 'stage_depth': 5},
    ])
    out = add_group_and_band(frame)
    assert list(out['group']) == ['independent', 'joint', None]


def test_stratum_row_raises_on_an_empty_stratum():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
    ]))
    with pytest.raises(ValueError, match='stage_depth=8'):
        stratum_row(frame, 'independent', 'low', 8)


def test_stratum_row_returns_the_lowest_split_on_a_tie():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5, 'split': 12},
        {'arm_slug': 'independent', 'k': 3, 'stage_depth': 5, 'split': 10},
    ]))
    row = stratum_row(frame, 'independent', 'low', 5)
    assert row['split'] == 10


def test_build_sample_raise_mode_raises_on_first_missing_cell():
    # Only ONE of the 24 target cells is populated.
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
    ]))
    with pytest.raises(ValueError):
        build_sample(frame, strata=STRATA_STAGE_DEPTHS, on_missing='raise')


def test_build_sample_skip_mode_reports_missing_cells_and_returns_the_rest():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 6},
    ]))
    rows, missing = build_sample(
        frame, strata=(5, 6), groups=('independent',), k_bands=('low',),
        on_missing='skip')
    assert len(rows) == 2
    assert missing == []
    rows2, missing2 = build_sample(
        frame, strata=(5, 6, 7), groups=('independent',), k_bands=('low',),
        on_missing='skip')
    assert len(rows2) == 2
    assert missing2 == [('independent', 'low', 7)]


def test_build_sample_row_id_is_unique_per_cell():
    frame = add_group_and_band(_synthetic_frame([
        {'arm_slug': 'independent', 'k': 2, 'stage_depth': 5},
        {'arm_slug': 'independent', 'k': 14, 'stage_depth': 5},
    ]))
    rows, _ = build_sample(frame, strata=(5,), on_missing='skip')
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids))


def test_build_sample_invalid_on_missing_raises():
    frame = add_group_and_band(_synthetic_frame([]))
    with pytest.raises(ValueError, match='on_missing'):
        build_sample(frame, on_missing='bogus')


import math

from scripts.compiler_calibration import gap_blocks, gap_stages, is_void


def test_gap_stages_computes_real_minus_predicted():
    row = {'stage_depth': 6, 'stages_real': 9}
    assert gap_stages(row) == 3


def test_gap_stages_none_when_stages_real_is_none():
    assert gap_stages({'stage_depth': 6, 'stages_real': None}) is None


def test_gap_stages_none_when_stages_real_is_nan():
    assert gap_stages({'stage_depth': 6, 'stages_real': float('nan')}) is None


def test_gap_blocks_computes_real_minus_predicted():
    row = {'blocks': 4, 'tcam_real': 4}
    assert gap_blocks(row) == 0


def test_gap_blocks_none_when_tcam_real_is_none():
    assert gap_blocks({'blocks': 4, 'tcam_real': None}) is None


def test_is_void_false_when_compile_errors_is_zero():
    assert is_void({'compile_errors': 0}) is False


def test_is_void_true_when_compile_errors_is_a_positive_int():
    assert is_void({'compile_errors': 2}) is True


def test_is_void_true_for_the_unavailable_reason_string():
    assert is_void({'compile_errors': "no FEATURE_REGISTER_CATALOG entry for 'x'"}) is True


def test_is_void_false_when_compile_errors_is_none_or_nan():
    assert is_void({'compile_errors': None}) is False
    assert is_void({'compile_errors': float('nan')}) is False
