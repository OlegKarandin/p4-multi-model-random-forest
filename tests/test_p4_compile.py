"""Tests for p4_compile.py, the reusable P4 compile + parse harness.

The parser tests below are unit-tested against fixture text captured from a
real p4c compile (Task 1 of the P4-validation plan, reviews/t11_tofino_port_
and_env.md Part K) -- not invented text -- so they exercise the actual
column layout of mau.resources.log (20 columns, of which only 4 are used).

The single pytest.mark.slow test at the bottom drives the real WSL2 Tofino
toolchain end to end and is not run by the default `pytest` invocation.
"""

import json
import os
import shlex
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.p4gen import build_p4_script as bps
from src.p4gen import p4_compile as pc
from src.p4gen.feature_registers import FEATURE_REGISTER_CATALOG


# main.py:308-314's live 18-feature pool, verbatim -- the exact spelling
# dataset.py/main.py select, dot-separated. Shared by the fast catalog-drift
# test and the slow real-compile acceptance test below.
LIVE_POOL_FEATURE_NAMES = [
    'Fwd.Packet.Length.Max', 'Fwd.Packet.Length.Min', 'Fwd.Packet.Length.Mean',
    'Bwd.Packet.Length.Max', 'Bwd.Packet.Length.Min', 'Bwd.Packet.Length.Mean',
    'Flow.IAT.Mean', 'Flow.IAT.Max', 'Flow.IAT.Min',
    'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min',
    'Bwd.IAT.Mean', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
    'Min.Packet.Length', 'Max.Packet.Length', 'Packet.Length.Mean']


def _forest_over(names, labels, seed):
    # Mirrors tests/test_build_p4_script_tna.py's helper of the same name --
    # duplicated here rather than imported since tests/ has no __init__.py
    # and no cross-test-file import convention exists in this repo yet.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    rnd = np.random.RandomState(seed)
    X = rnd.randint(0, 65535, size=(40, len(names)))
    y = np.array([labels[i % len(labels)] for i in range(40)])
    X[:, 0] += np.array([16000 * (labels.index(v) + 1) for v in y])
    return bps.dt_thresholds_float_to_int(
        RandomForestClassifier(n_estimators=2, max_depth=3, random_state=seed).fit(X, y))


def test_all_live_pool_features_have_catalog_entries():
    """Fast, no toolchain needed -- runs in CI on every commit. Every
    normalised name in main.py's 18-feature pool (main.py:308-314) must be a
    key of FEATURE_REGISTER_CATALOG. This is what stops the catalog silently
    falling behind main.py again; the slow real-compile test below is the
    belt to this test's suspenders.
    """
    missing = [name for name in LIVE_POOL_FEATURE_NAMES
               if bps.normalise_feature_name(name) not in FEATURE_REGISTER_CATALOG]
    assert missing == []


def _write_fixture_logs(tmp_path, table_summary_text, mau_resources_text, table_placement_text):
    log_dir = tmp_path / "pipe" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "table_summary.log").write_text(table_summary_text)
    (log_dir / "mau.resources.log").write_text(mau_resources_text)
    (log_dir / "table_placement_1.log").write_text(table_placement_text)
    return str(tmp_path)


# Real header + Totals row copied verbatim from Task 1's actual compile,
# .superpowers/sdd/timing_probe_logs/pipe/logs/mau.resources.log (line 10 and
# line 25 there). This is a markdown-style pipe table: one header row naming
# 20 columns, then exactly one "Totals" data row -- NOT the brief's
# placeholder "Totals  9  22  14  16" shape, and NOT in a fixed
# Gateway-SRAM-MapRAM-TCAM-first column order (there are 4 columns before
# Gateway: Exact Match Input xbar, Ternary Match Input xbar, Hash Bit, Hash
# Dist Unit). Keeping all 20 columns here (not trimming to just the 4 we
# need) is deliberate: a parser that only worked because the fixture was
# artificially shrunk to 4 columns would not catch a column-order bug.
REAL_MAU_RESOURCES_TEXT = (
    "| Stage Number | Exact Match Input xbar | Ternary Match Input xbar | Hash Bit | Hash Dist Unit | Gateway | SRAM | Map RAM | TCAM | VLIW Instr | Meter ALU | Stats ALU | Stash | Exact Match Search Bus | Exact Match Result Bus | Tind Result Bus | Action Data Bus Bytes | 8-bit Action Slots | 16-bit Action Slots | 32-bit Action Slots | Logical TableID |\n"
    "|    Totals    |           5            |            28            |    20    |       0        |    1    |  11  |    0    |  16  |     8      |     0     |     0     |   2   |           3            |           2            |        2        |           20          |         0          |          0          |          0          |        11       |\n"
)


def test_parse_compile_logs_extracts_stages_and_tables(tmp_path):
    log_dir = _write_fixture_logs(
        tmp_path,
        table_summary_text="Number of stages in table allocation: 9\nNumber of tables allocated: 20\n",
        mau_resources_text=REAL_MAU_RESOURCES_TEXT,
        table_placement_text="Placement error(s):0 stages required:9\n",
    )
    result = pc.parse_compile_logs(log_dir)
    assert result.stages == 9
    assert result.tables == 20


def test_parse_compile_logs_extracts_resource_totals_by_column_name(tmp_path):
    # Confirms the parser reads Gateway/SRAM/Map RAM/TCAM by column NAME,
    # not by fixed position -- the real file has 20 columns and the 4
    # needed ones are not first, so a fixed-position regex (the brief's
    # original r"Totals\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)" draft) would read
    # the wrong values here (it would read 5, 28, 20, 0 -- the first four
    # numeric cells -- instead of the real Gateway=1, SRAM=11, Map RAM=0,
    # TCAM=16).
    log_dir = _write_fixture_logs(
        tmp_path,
        table_summary_text="Number of stages in table allocation: 3\nNumber of tables allocated: 11\n",
        mau_resources_text=REAL_MAU_RESOURCES_TEXT,
        table_placement_text="Placement error(s):0 stages required:3\n",
    )
    result = pc.parse_compile_logs(log_dir)
    assert result.gateway == 1
    assert result.sram == 11
    assert result.map_ram == 0
    assert result.tcam == 16


def test_parse_compile_logs_returns_none_fields_when_logs_absent(tmp_path):
    # a failed compile leaves no pipe/logs/ directory at all
    result = pc.parse_compile_logs(str(tmp_path))
    assert result.stages is None
    assert result.tables is None
    assert result.errors is None
    assert result.gateway is None
    assert result.sram is None
    assert result.map_ram is None
    assert result.tcam is None


def test_parse_compile_logs_accepts_logs_dir_directly(tmp_path):
    # Callers may pass either the compile's top-level output_dir (which
    # contains pipe/logs/...) or the pipe/logs directory itself.
    log_dir = tmp_path / "pipe" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "table_summary.log").write_text(
        "Number of stages in table allocation: 3\nNumber of tables allocated: 11\n"
    )
    result = pc.parse_compile_logs(str(log_dir))
    assert result.stages == 3
    assert result.tables == 11


# Minimal realistic subset of the real resources.json structure, confirmed
# against Task 1's actual compile at
# .superpowers/sdd/timing_probe_logs/pipe/logs/resources.json:
# resources.mau.mau_stages is a list of per-stage dicts, each carrying a
# "tcams" dict whose "tcams" list holds one entry per PHYSICAL TCAM row; each
# row entry carries a "usages" list naming which P4 table(s) (by full name,
# e.g. "SwitchIngress.<table_name>") that physical row serves. The real file
# only ever saw "ternary_match" for used_for in the fixture we inspected, but
# parse_table_entry_count must not filter on used_for -- a range-match table
# is expected to show up the same way.
REAL_RESOURCES_JSON_SUBSET = {
    "resources": {
        "mau": {
            "mau_stages": [
                {
                    "stage_number": 0,
                    "tcams": {
                        "nColumns": 2,
                        "nRows": 4,
                        "tcams": [
                            {"column": 0, "row": 0, "usages": [
                                {"used_by": "SwitchIngress.probe_range_table", "used_for": "ternary_match"}]},
                            {"column": 0, "row": 1, "usages": [
                                {"used_by": "SwitchIngress.probe_range_table", "used_for": "ternary_match"}]},
                            {"column": 0, "row": 2, "usages": [
                                {"used_by": "SwitchIngress.other_table", "used_for": "ternary_match"}]},
                            {"column": 1, "row": 0, "usages": []},
                        ],
                    },
                },
                {
                    "stage_number": 1,
                    "tcams": {
                        "nColumns": 2,
                        "nRows": 2,
                        "tcams": [
                            {"column": 0, "row": 0, "usages": [
                                {"used_by": "SwitchIngress.probe_range_table", "used_for": "ternary_match"}]},
                        ],
                    },
                },
            ]
        }
    }
}


def _write_resources_json(tmp_path, data):
    log_dir = tmp_path / "pipe" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "resources.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_parse_table_entry_count_counts_rows_across_stages(tmp_path):
    # probe_range_table has 2 rows in stage 0 + 1 row in stage 1 = 3 total,
    # while other_table's 1 row and the unused row must not be counted.
    log_dir = _write_resources_json(tmp_path, REAL_RESOURCES_JSON_SUBSET)
    assert pc.parse_table_entry_count(log_dir, "SwitchIngress.probe_range_table") == 3


def test_parse_table_entry_count_returns_none_for_missing_table(tmp_path):
    # A table name never found anywhere in resources.json is a distinct
    # signal from "found with 0 rows" (which shouldn't happen for a real
    # range-match table with at least one entry).
    log_dir = _write_resources_json(tmp_path, REAL_RESOURCES_JSON_SUBSET)
    assert pc.parse_table_entry_count(log_dir, "SwitchIngress.nonexistent_table") is None


def test_parse_table_entry_count_returns_none_when_resources_json_absent(tmp_path):
    # A compile that failed before resource allocation ran leaves no
    # resources.json at all.
    assert pc.parse_table_entry_count(str(tmp_path), "SwitchIngress.probe_range_table") is None


def test_parse_table_entry_count_accepts_logs_dir_directly(tmp_path):
    # Mirrors parse_compile_logs's convention: callers may pass either the
    # compile's top-level output_dir or the pipe/logs directory itself.
    log_dir = tmp_path / "pipe" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "resources.json").write_text(json.dumps(REAL_RESOURCES_JSON_SUBSET))
    assert pc.parse_table_entry_count(str(log_dir), "SwitchIngress.probe_range_table") == 3


def _fake_completed_process(stdout="", stderr="", returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def test_compile_p4_builds_login_shell_command_with_correct_quoting(tmp_path):
    """Guards Correction 2 (the `wsl bash -lc '...'` login-shell fix) without
    requiring a real WSL2 + Tofino toolchain: mocks subprocess.run so
    compile_p4 never actually shells out, then inspects the constructed
    command itself.

    Specifically proves:
    - the command is `["wsl", "bash", "-lc", <single string>]`, a login
      shell -- NOT the naive `["wsl", p4c_path, ...]` form, which silently
      fails to expand `~` or source PATH (the bug this module's docstring
      says was found and fixed once already).
    - p4c_path appears UNQUOTED in the command string, so bash's own `~`
      expansion still applies (shlex.quote() would wrap a `~`-containing
      string in quotes, since `~` is not in shlex's "safe" character set --
      quoting it would silently break expansion again).
    - output_dir/p4_path/include_path are each individually shell-quoted so
      a path containing a space and non-ASCII characters survives intact --
      proven by round-tripping the built command string through
      shlex.split() and confirming the exact converted path segments come
      back out unmangled.
    """
    fake_proc = _fake_completed_process(stdout="0 errors, 10 warnings generated.\n")

    p4_path = str(tmp_path / "probe.p4")
    # Deliberately includes a space and a non-ASCII (Cyrillic) path
    # component, mirroring this repo's own path (reviews/t11_tofino_port_
    # and_env.md Part K / G) -- exactly the kind of path that a broken
    # quoting scheme would mangle.
    output_dir = str(tmp_path / "some path" / "Документы")

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc) as mock_run:
        result = pc.compile_p4(p4_path, output_dir)

    assert mock_run.call_count == 1
    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["wsl", "bash", "-lc"]
    assert len(cmd) == 4
    full_command = cmd[3]
    assert isinstance(full_command, str)

    default_p4c_path = "~/open-p4studio/install/bin/p4c"
    # Unquoted: appears as a bare token, not wrapped by shlex.quote (which
    # would produce "'~/open-p4studio/install/bin/p4c'" since `~` is not a
    # shlex-safe character).
    assert full_command.startswith(default_p4c_path + " ")
    assert shlex.quote(default_p4c_path) not in full_command

    # Quoted path arguments must round-trip intact through shlex.split(),
    # proving the space/non-ASCII path components survive shell parsing.
    tokens = shlex.split(full_command)
    expected_wsl_output_dir = pc._to_wsl_path(output_dir)
    expected_wsl_p4_path = pc._to_wsl_path(p4_path)
    expected_wsl_include_path = pc._to_wsl_path(pc._resolve_repo_relative("resources"))

    assert expected_wsl_output_dir in tokens
    assert expected_wsl_p4_path in tokens
    assert expected_wsl_include_path in tokens

    # Sanity: the mocked compile still completes and parses the summary line
    # (proves the mock's fake stdout is realistic enough not to crash parsing).
    assert result.errors == 0
    assert result.warnings == 10


def test_compile_p4_target_parameter_selects_the_b_flag(tmp_path):
    """Guards the `target` parameter added for Task 5 (Tofino-2 portability,
    reviews/t11_tofino_port_and_env.md Part L): compile_p4 used to hardcode
    `-b tofino` in full_command, which would have made a Tofino-2 compile
    (`-b tofino2 -a t2na`) impossible without editing this module. Proves
    both that the default still produces `-b tofino` (no regression for
    every prior Tofino-1 call site) and that passing target="tofino2"
    actually changes the `-b` flag rather than being silently ignored.
    """
    fake_proc = _fake_completed_process(stdout="0 errors, 10 warnings generated.\n")

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc) as mock_run:
        pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))
    default_command = mock_run.call_args[0][0][3]
    assert "-b tofino -a tna" in default_command

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc) as mock_run:
        pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"),
                       architecture="t2na", target="tofino2")
    tofino2_command = mock_run.call_args[0][0][3]
    assert "-b tofino2 -a t2na" in tofino2_command


def test_errors_warnings_regex_matches_real_captured_summary_line():
    # Verbatim line captured from this session's actual compile
    # (task-1-report.md / reviews/t11_tofino_port_and_env.md Part K).
    m = pc._ERRORS_WARNINGS_RE.search(
        "... some p4c verbose output ...\n0 errors, 10 warnings generated.\n"
    )
    assert m is not None
    assert m.group(1) == "0"
    assert m.group(2) == "10"


def test_errors_warnings_regex_handles_singular_error_and_warning():
    # The `s?` in `errors?`/`warnings?` must handle both the plural form
    # (captured above: "0 errors, 10 warnings") and the singular form a
    # real compile could plausibly emit for count == 1.
    m = pc._ERRORS_WARNINGS_RE.search("1 error, 0 warnings generated.")
    assert m is not None
    assert m.group(1) == "1"
    assert m.group(2) == "0"

    m2 = pc._ERRORS_WARNINGS_RE.search("1 error, 1 warning generated.")
    assert m2 is not None
    assert m2.group(1) == "1"
    assert m2.group(2) == "1"


def test_errors_warnings_regex_no_match_when_summary_line_absent():
    assert pc._ERRORS_WARNINGS_RE.search(
        "some unrelated p4c chatter\nwith no summary line at all\n"
    ) is None


def test_compile_p4_errors_and_warnings_degrade_to_none_when_summary_absent(tmp_path):
    # Per CompileResult's None-means-unknown convention (see module
    # docstring / parse_compile_logs docstring): if p4c's stdout+stderr
    # never contain the summary line (e.g. a hard crash before it could
    # print), errors/warnings must stay None, not silently become 0.
    fake_proc = _fake_completed_process(stdout="a hard crash before any summary line\n")

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc):
        result = pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))

    assert result.errors is None
    assert result.warnings is None


def test_compile_p4_raises_when_summary_absent_and_returncode_nonzero(tmp_path):
    # When p4c does not run at all (returncode nonzero) but the summary line
    # is missing, we have a toolchain failure (e.g. p4c not installed), not
    # a compile error. Must raise RuntimeError to distinguish this from
    # exit(0) + no summary (which means validation was not requested).
    fake_proc = _fake_completed_process(stdout="a hard crash before any summary line\n",
                                         returncode=1)

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc):
        with pytest.raises(RuntimeError, match="p4c did not run"):
            pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))


def test_compile_p4_parses_errors_even_when_returncode_nonzero(tmp_path):
    # A real compile error has p4c successfully running (summary line present)
    # but reporting errors. returncode may be nonzero in this case, but we
    # still parse the error count from the summary line, not raise an exception.
    fake_proc = _fake_completed_process(stdout="5 errors, 3 warnings generated.\n",
                                         returncode=1)

    with patch("src.p4gen.p4_compile.subprocess.run", return_value=fake_proc):
        result = pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))

    assert result.errors == 5
    assert result.warnings == 3


def test_compile_p4_converts_timeout_expired_to_runtime_error(tmp_path):
    # When subprocess.run times out, it raises subprocess.TimeoutExpired.
    # compile_p4 must convert this to RuntimeError with a descriptive message
    # (not pass through the bare TimeoutExpired), so timeouts are treated
    # consistently with other toolchain failures.
    with patch("src.p4gen.p4_compile.subprocess.run",
               side_effect=subprocess.TimeoutExpired("cmd", 300)):
        with pytest.raises(RuntimeError, match="p4c compilation timed out after 300 seconds"):
            pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))


@pytest.mark.slow
def test_compile_p4_against_real_toolchain(tmp_path):
    """Requires WSL2 + a built open-p4studio (reviews/t11_tofino_port_and_env.md Part B/K).
    Not run by the default `pytest` invocation -- run explicitly with
    `pytest test_p4_compile.py -m slow -v` when you want to confirm the real
    invocation path still works end to end. This is the test that proves the
    `wsl bash -lc '...'` invocation fix (Correction 2) actually works --
    the naive `["wsl", p4c_path, ...]` form silently fails to expand `~`.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from src.p4gen import build_p4_script as bps

    rng = np.random.RandomState(0)
    X = rng.randint(0, 65535, size=(100, 2))
    y_app = rng.randint(0, 3, size=100)
    y_ddos = rng.choice([-1, 1], size=100)

    clf_app = bps.dt_thresholds_float_to_int(
        RandomForestClassifier(n_estimators=1, max_depth=3, random_state=0).fit(X, y_app))
    clf_ddos = bps.dt_thresholds_float_to_int(
        RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(X, y_ddos))

    feature_names = ["flow_iat_max", "fwd_iat_max"]
    tree_nodes = {i: bps.get_nodes(est, feature_names)
                  for i, est in enumerate(clf_app.estimators_)}
    offset = len(tree_nodes)
    tree_nodes.update({i + offset: bps.get_nodes(est, feature_names)
                       for i, est in enumerate(clf_ddos.estimators_)})
    feature_intervals = bps.get_feature_intervals_from_thresholds(bps.get_feature_thresholds(tree_nodes))

    bps.generate_P4_code(3, 2, clf_app, clf_ddos,
                          feature_intervals_app=feature_intervals, feature_intervals_ddos=feature_intervals,
                          output_dir=str(tmp_path) + os.sep, output_filename="probe.p4")

    result = pc.compile_p4(str(tmp_path / "probe.p4"), str(tmp_path / "logs"))
    assert result.errors == 0
    assert result.stages is not None


@pytest.mark.slow
def test_full_eighteen_feature_pool_compiles(tmp_path):
    """Every feature in main.py's pool must have a catalog entry that produces a
    program the real compiler accepts. This is the test Phase 2 exists to pass."""
    live_names = [  # main.py:307-313, verbatim
        'Fwd.Packet.Length.Max', 'Fwd.Packet.Length.Min', 'Fwd.Packet.Length.Mean',
        'Bwd.Packet.Length.Max', 'Bwd.Packet.Length.Min', 'Bwd.Packet.Length.Mean',
        'Flow.IAT.Mean', 'Flow.IAT.Max', 'Flow.IAT.Min',
        'Fwd.IAT.Mean', 'Fwd.IAT.Max', 'Fwd.IAT.Min',
        'Bwd.IAT.Mean', 'Bwd.IAT.Max', 'Bwd.IAT.Min',
        'Min.Packet.Length', 'Max.Packet.Length', 'Packet.Length.Mean']
    clf_app = _forest_over(live_names, [0, 1, 2], seed=0)      # Task 4's helper
    clf_ddos = _forest_over(live_names, [-1, 1], seed=1)
    feature_intervals = bps.get_joint_feature_intervals(
        clf_app, live_names, clf_ddos, live_names)
    written = bps.generate_P4_code(
        3, 2, clf_app, clf_ddos,
        feature_intervals_app=feature_intervals, feature_intervals_ddos=feature_intervals,
        output_dir=str(tmp_path) + os.sep, output_filename='pool18.p4',
        selected_features_app=live_names, selected_features_ddos=live_names)
    result = pc.compile_p4(written, str(tmp_path / "logs"))
    assert result.errors == 0, result


# ---------------------------------------------------------------------------
# include-path resolution
#
# _resolve_repo_relative exists so compile_p4's default include_path
# ("resources") resolves against THIS REPO's root rather than the caller's
# cwd. It derived that root from os.path.dirname(__file__), which was correct
# while p4_compile.py lived at the repo root -- but the src/ reorganisation
# moved it to src/p4gen/, silently turning the default into
# src/p4gen/resources, a directory that does not exist. p4c would then be
# handed `-I <nonexistent>` and fail to find p4_headers.p4 / p4_util.p4 for
# any generated program not sitting beside its own includes.
# ---------------------------------------------------------------------------

def test_resolve_repo_relative_finds_the_real_resources_directory():
    resolved = pc._resolve_repo_relative("resources")

    assert os.path.isdir(resolved), (
        "default include path does not exist: {}".format(resolved))
    assert os.path.isfile(os.path.join(resolved, "p4_template.p4"))
    assert os.path.isfile(os.path.join(resolved, "p4_headers.p4"))
    assert os.path.isfile(os.path.join(resolved, "p4_util.p4"))


def test_resolve_repo_relative_leaves_absolute_paths_alone(tmp_path):
    absolute = str(tmp_path)

    assert pc._resolve_repo_relative(absolute) == absolute
