import os

import pytest

from src.training import campaign_run as cr


@pytest.mark.parametrize("M,token", [(15, "015"), (35, "035"), (250, "250"), (cr.INF, "inf")])
def test_m_token_is_zero_padded_or_inf(M, token):
    assert cr.m_token(M) == token


@pytest.mark.parametrize("bad", [0.5, -1, 1000])
def test_m_token_rejects_values_it_cannot_sort(bad):
    with pytest.raises(ValueError):
        cr.m_token(bad)


def test_row_id_format():
    assert cr.row_id("joint", 35, 7, 9) == "joint_M035_s07_k09"
    assert cr.row_id("joint-off", cr.INF, 24, 17) == "joint-off_Minf_s24_k17"


def test_row_ids_are_unique_over_the_whole_grid():
    ids = {cr.row_id(a, M, s, k) for a in ("independent", "joint-off", "joint")
           for M in cr.DEFAULT_M_GRID for s in range(25) for k in range(1, 18)}
    assert len(ids) == 3 * 6 * 25 * 17


def test_row_ids_sort_numerically_by_m():
    ids = [cr.row_id("joint", M, 0, 1) for M in (15, 25, 35, 50, 75)]
    assert ids == sorted(ids)


def test_split_csv_name():
    assert cr.split_csv_name(7, 14, 35, "joint", 7) == "rf_t7_d14_M035_joint_s07.csv"
    assert cr.split_csv_name(7, 14, cr.INF, "joint-off", 0) == "rf_t7_d14_Minf_joint-off_s00.csv"


@pytest.mark.parametrize("text,want", [("0-9", list(range(10))), ("0,3,5", [0, 3, 5]),
                                       ("5,0-2,2", [0, 1, 2, 5]), ("24", [24])])
def test_parse_splits(text, want):
    assert cr.parse_splits(text) == want


@pytest.mark.parametrize("text", ["", "9-0", "-1", "a"])
def test_parse_splits_rejects(text):
    with pytest.raises(ValueError):
        cr.parse_splits(text)


def test_parse_M_grid_accepts_inf():
    assert cr.parse_M_grid("15,25,inf") == [15, 25, cr.INF]
    with pytest.raises(ValueError):
        cr.parse_M_grid("0,25")


def test_optuna_seed_ignores_arm_and_M_by_construction():
    assert cr.optuna_seed(7, 9) == 7009


def test_atomic_write_leaves_no_partial(tmp_path):
    path = tmp_path / "a" / "b.json"
    cr.atomic_write_text(str(path), cr.canonical_json({"b": 1, "a": 2}))
    assert path.read_text() == '{\n "a": 2,\n "b": 1\n}\n'
    assert not os.path.exists(str(path) + ".partial")


@pytest.mark.parametrize("text", ["15.5", "2000", "-inf", "INF", "25,15.5"])
def test_parse_M_grid_rejects_what_m_token_cannot_name(text):
    """Fail fast: a value m_token later rejects would otherwise crash inside
    row_id hours into a run, and 'INF' used to raise a bare OverflowError."""
    with pytest.raises(ValueError):
        cr.parse_M_grid(text)


def test_parse_M_grid_drops_duplicates_keeping_first_order():
    assert cr.parse_M_grid("35,35,15") == [35, 15]
    assert cr.parse_M_grid("inf,15,inf") == [cr.INF, 15]
