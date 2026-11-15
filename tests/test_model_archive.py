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


def test_a_refused_program_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    # The real replay on a stand-in program with none of the generator's pins: a
    # named refusal, reported as MODEL_REFUSED:<reason> (decision 8.2).
    archive = _archive(tmp_path, ["d1"])
    _p4c(monkeypatch, {"d1": 11})
    results = model_archive.check(archive, known_misses={})
    assert results[0]["misses"] == {"refusal": "MODEL_REFUSED:missing_pin:pa_no_overlay:fwd"}
    assert results[0]["model_stage_depth"] is None
    out = capsys.readouterr().out
    assert "d1: MODEL_REFUSED:missing_pin:pa_no_overlay:fwd (UNEXPECTED)" in out
    assert "refused by the model 1" in out
    assert model_archive.main_exit_code(results, known_misses={}) == 1
    listed = {"d1": {"refusal": "MODEL_REFUSED:missing_pin:pa_no_overlay:fwd"}}
    assert model_archive.main_exit_code(model_archive.check(archive, listed), listed) == 0


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
    # 2026-11-08: heldout_independent_M150_k15_s16 and independent_high_sd12
    # are exact under p4c's tie order and rule E, and leave the list.
    # 2026-11-13: independent_low_sd5's blocks are exact under the crossbar
    # port and leave it too.
    # 2026-11-15 (Task 18): the list is empty -- the pragmas archive is refused
    # by shape (below), so heldout_independent_M250_k14_s12 no longer occurs.
    assert model_archive.KNOWN_MISSES == {}


PRAGMAS_ARCHIVE = os.path.join("results", "compiler_calibration_pragmas_2026_09_29")


def _find_pragmas_archive():
    """The archive is not in git. Look in this checkout's results/ and then in
    every parent directory, so a run from a worktree under
    <main>/.claude/worktrees/ finds the main checkout's copy."""
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(here, PRAGMAS_ARCHIVE)
        if os.path.isdir(os.path.join(candidate, "p4_src")):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def test_the_pragmas_archive_programs_are_refused_as_gated_programs():
    """The model gate's archive (`python -m src.verify --model-archive`) holds
    73 compiles of programs generated before decisions 1a / 7.1 option B
    (2026-11-14): every one carries a gated register block, which the replay
    now refuses, so the model can no longer be scored on it (a gate on
    option-B programs needs new compiles). --model-archive reports each as
    MODEL_REFUSED:gated_register_block instead of stopping (decision 8.2)."""
    from p4gen.p4_replay import model_breakdown, parse_program
    archive = _find_pragmas_archive()
    if archive is None:
        pytest.skip("calibration archive %s not found in this checkout or any "
                    "parent directory (it is not in git)" % PRAGMAS_ARCHIVE)
    sources = sorted(name for name in os.listdir(os.path.join(archive, "p4_src"))
                     if name.endswith(".p4"))
    assert len(sources) == 73
    for name in sources:
        breakdown = model_breakdown(parse_program(os.path.join(archive, "p4_src", name)))
        assert breakdown["refusal_reason"] == "gated_register_block", name
    results = model_archive.check(archive)
    assert all(r["misses"] == {"refusal": "MODEL_REFUSED:gated_register_block"}
               for r in results)
    assert model_archive.main_exit_code(results) == 1


def test_cli_model_archive_exit_code(tmp_path, monkeypatch):
    archive = _archive(tmp_path, ["d1"])
    _fake_model(monkeypatch, {"d1": (11, EXACT_TABLES)})
    _p4c(monkeypatch, {"d1": 11})
    monkeypatch.setattr(model_archive, "KNOWN_MISSES", {})
    assert cli.main(["--model-archive", archive]) == 0
