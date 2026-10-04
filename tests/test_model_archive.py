"""python -m src.verify --model-archive: the model-side gate on a calibration
archive (spec 2026-10-04 Sec I.5). --archive recompiles and compares p4c with
p4c; this replays the MODEL against the archived committed logs."""
import os

import pytest

from src.verify import model_archive
from src.verify import __main__ as cli
from tests.test_verify_runner import _RESOURCES, _write_logs


def _archive(tmp_path, rows):
    archive = tmp_path / "archive"
    (archive / "p4_src").mkdir(parents=True)
    for row in rows:
        (archive / "p4_src" / (row + ".p4")).write_text("// %s\n" % row)
        stored = archive / "compiles" / row
        stored.mkdir(parents=True)
        _write_logs(str(stored))
    return str(archive)


def _fake_model(monkeypatch, by_row):
    """by_row: row -> (stage_depth, {table: blocks})."""
    monkeypatch.setattr(model_archive, "parse_program", lambda path: path)

    def breakdown(path, row):
        depth, tables = by_row[row]
        return {"stage_depth": depth, "blocks": sum(tables.values()),
                "tables": [{"table": t, "blocks": b} for t, b in tables.items()]}
    monkeypatch.setattr(model_archive, "model_breakdown", breakdown)


def _p4c(monkeypatch, depth_by_row):
    monkeypatch.setattr(model_archive, "committed_stages_real",
                        lambda logs: depth_by_row[logs.split(os.sep)[-3]])


# _RESOURCES commits: table_0_flow_iat_max 1, get_classification_tree_app_0 5.
EXACT_TABLES = {"table_0_flow_iat_max": 1, "get_classification_tree_app_0": 5}


def test_an_exact_archive_passes(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, EXACT_TABLES)})
    _p4c(monkeypatch, {"d1": 11})
    results = model_archive.check(archive, known_misses={})
    assert results[0]["misses"] == {} and results[0]["unexpected"] == {}
    assert model_archive.main_exit_code(results, known_misses={}) == 0
    assert "blocks exact feasible 1/1" in capsys.readouterr().out


def test_an_unlisted_blocks_miss_fails(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, {**EXACT_TABLES, "get_classification_tree_app_0": 4})})
    _p4c(monkeypatch, {"d1": 11})
    results = model_archive.check(archive, known_misses={})
    assert results[0]["unexpected"] == {"blocks": (5, 6)}
    assert model_archive.main_exit_code(results, known_misses={}) == 1


def test_a_listed_miss_with_the_same_numbers_passes(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, {**EXACT_TABLES, "get_classification_tree_app_0": 4})})
    _p4c(monkeypatch, {"d1": 11})
    known = {"d1": {"blocks": (5, 6)}}
    results = model_archive.check(archive, known_misses=known)
    assert results[0]["unexpected"] == {}
    assert model_archive.main_exit_code(results, known_misses=known) == 0


def test_a_listed_miss_with_different_numbers_fails(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, {**EXACT_TABLES, "get_classification_tree_app_0": 3})})
    _p4c(monkeypatch, {"d1": 11})
    known = {"d1": {"blocks": (5, 6)}}
    results = model_archive.check(archive, known_misses=known)
    assert results[0]["unexpected"] == {"blocks": (4, 6)}
    assert model_archive.main_exit_code(results, known_misses=known) == 1


def test_a_stale_known_miss_fails(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, EXACT_TABLES)})
    _p4c(monkeypatch, {"d1": 11})
    known = {"d1": {"blocks": (5, 6)}}
    results = model_archive.check(archive, known_misses=known)
    assert model_archive.main_exit_code(results, known_misses=known) == 1
    assert "stale" in capsys.readouterr().out


def test_blocks_are_not_compared_on_an_infeasible_design(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (13, {"get_classification_tree_app_0": 99})})
    _p4c(monkeypatch, {"d1": 14})
    results = model_archive.check(archive, known_misses={})
    assert results[0]["feasible"] is False
    assert results[0]["misses"] == {"stage_depth": (13, 14)}


def test_the_shipped_known_misses_are_the_spec_list():
    assert model_archive.KNOWN_MISSES == {
        "heldout_independent_M150_k15_s16": {"stage_depth": (14, 13)},
        "heldout_independent_M250_k14_s12": {"stage_depth": (15, 16)},
        "independent_high_sd12": {"stage_depth": (13, 14)},
        "independent_low_sd5": {"blocks": (13, 16)},
    }


def test_cli_model_archive_exit_code(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, EXACT_TABLES)})
    _p4c(monkeypatch, {"d1": 11})
    monkeypatch.setattr(model_archive, "KNOWN_MISSES", {})
    assert cli.main(["--model-archive", archive]) == 0
