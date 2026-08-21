"""Reusable P4 compile + parse harness for the real Tofino toolchain.

Wraps the WSL2 `p4c` invocation pattern used manually throughout this
project's development (reviews/t11_tofino_port_and_env.md) into a reusable
module: a blocking `compile_p4`, a non-blocking `compile_p4_async` (backed by
a shared `ThreadPoolExecutor`), and a pure-parsing `parse_compile_logs` that
is unit-tested against captured real log fixtures (see test_p4_compile.py)
rather than invented ones.

Two things here are load-bearing and easy to get wrong by guessing instead
of measuring (both confirmed against a real compile, see
reviews/t11_tofino_port_and_env.md Part K):

1. `mau.resources.log`'s "Totals" row is a markdown-style pipe table with
   20 columns; Gateway/SRAM/Map RAM/TCAM are NOT the first four columns, so
   they must be looked up by column NAME (see `_find_header_and_totals_rows`
   + `parse_compile_logs`), not by a fixed-position regex.
2. WSL2 only expands `~` and sources PATH when invoked as a login shell
   (`wsl bash -lc '...'`); the naive `["wsl", p4c_path, ...]` form silently
   fails to find the compiler. See `compile_p4`.
"""

import json
import os
import re
import shlex
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CompileResult:
    errors: Optional[int] = None
    warnings: Optional[int] = None
    stages: Optional[int] = None
    tables: Optional[int] = None
    gateway: Optional[int] = None
    sram: Optional[int] = None
    map_ram: Optional[int] = None
    tcam: Optional[int] = None


# The four mau.resources.log columns parse_compile_logs extracts, mapped to
# CompileResult's field names. Looked up by name (see
# _find_header_and_totals_rows), not position, since the real file has 20
# columns and these are not the first four.
_RESOURCE_COLUMNS = (
    ("gateway", "Gateway"),
    ("sram", "SRAM"),
    ("map_ram", "Map RAM"),
    ("tcam", "TCAM"),
)


def parse_compile_logs(log_dir: str) -> CompileResult:
    """Parses table_summary.log and mau.resources.log under log_dir.

    log_dir may be either a compile's top-level output_dir (which contains
    pipe/logs/...) or the pipe/logs directory itself. Any field whose
    source log line/file is missing degrades to None (never 0), so a caller
    can distinguish "no resources used" from "compile failed before
    resource allocation ran" (e.g. a compile that errored out won't have a
    table_summary.log at all).
    """
    result = CompileResult()
    logs_path = os.path.join(log_dir, "pipe", "logs")
    if not os.path.isdir(logs_path):
        logs_path = log_dir  # allow passing the logs dir directly, not just its parent

    summary_path = os.path.join(logs_path, "table_summary.log")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            text = f.read()
        m = re.search(r"Number of stages in table allocation:\s*(\d+)", text)
        if m:
            result.stages = int(m.group(1))
        m = re.search(r"Number of tables allocated:\s*(\d+)", text)
        if m:
            result.tables = int(m.group(1))

    resources_path = os.path.join(logs_path, "mau.resources.log")
    if os.path.isfile(resources_path):
        with open(resources_path) as f:
            text = f.read()
        header_row, totals_row = _find_header_and_totals_rows(text)
        if header_row is not None and totals_row is not None:
            columns = _split_pipe_row(header_row)
            values = _split_pipe_row(totals_row)
            row = dict(zip(columns, values))
            for field_name, column_name in _RESOURCE_COLUMNS:
                cell = row.get(column_name)
                if cell is not None and cell.lstrip("-").isdigit():
                    setattr(result, field_name, int(cell))

    return result


def parse_table_entry_count(log_dir: str, table_name: str) -> Optional[int]:
    """Reads the real per-table PHYSICAL TCAM row count from resources.json.

    log_dir accepts either a compile's top-level output_dir or the pipe/logs
    directory itself, mirroring parse_compile_logs's convention.

    resources.json's real structure (confirmed against a real compile, see
    .superpowers/sdd/timing_probe_logs/pipe/logs/resources.json):
    resources.mau.mau_stages is a list of per-stage dicts; each stage's
    tcams.tcams is a list of physical TCAM row entries, each shaped
    {"column": int, "row": int, "usages": [{"used_by": "SwitchIngress.<table>",
    "used_for": "..."}, ...]}. A row counts toward table_name if ANY of its
    usages names it -- used_for is NOT filtered on, since a range-match
    table is expected to occupy physical TCAM rows the same way a ternary
    table does.

    Returns None (not 0) both when resources.json doesn't exist (compile
    failed before resource allocation) and when table_name is never found in
    it -- these are distinct signals from "found with 0 rows", which a real
    range-match table with at least one entry should never actually hit.
    """
    logs_path = os.path.join(log_dir, "pipe", "logs")
    if not os.path.isdir(logs_path):
        logs_path = log_dir  # allow passing the logs dir directly, not just its parent

    resources_path = os.path.join(logs_path, "resources.json")
    if not os.path.isfile(resources_path):
        return None

    with open(resources_path) as f:
        data = json.load(f)

    mau_stages = data.get("resources", {}).get("mau", {}).get("mau_stages", [])

    count = 0
    found = False
    for stage in mau_stages:
        for tcam_row in stage.get("tcams", {}).get("tcams", []):
            for usage in tcam_row.get("usages", []):
                if usage.get("used_by") == table_name:
                    found = True
                    count += 1
                    break  # don't double-count one physical row for repeated matches

    return count if found else None


def _split_pipe_row(line: str) -> list:
    """Splits a markdown-style `| a | b | c |` row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _find_header_and_totals_rows(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Finds the mau.resources.log header row and its paired "Totals" row.

    The real file (reviews/t11_tofino_port_and_env.md Part K) contains
    several pipe tables (an absolute-count table with a "Totals" row, a
    percentage table with an "Average" row instead, and an "Allocated
    Resource Usage" table with different columns entirely) -- there is
    exactly one "Totals" row in the whole file. We track the most recent
    row whose first cell is literally "Stage Number" as the candidate
    header, so the header we pair with "Totals" is the one from the same
    table, not a header from some other table that happens to appear later.
    """
    header_row = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _split_pipe_row(stripped)
        if not cells or not cells[0]:
            continue
        if cells[0] == "Stage Number":
            header_row = line
        elif cells[0] == "Totals":
            return header_row, line
    return header_row, None


# This file lives at <repo root>/src/p4gen/p4_compile.py, so the repo root is
# three levels up. It used to be one level up, which was correct while this
# module sat at the repo root -- the src/ reorganisation silently turned the
# default include path into src/p4gen/resources (a directory that does not
# exist), handing p4c `-I <nonexistent>`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_repo_relative(path: str) -> str:
    """Resolves a possibly-relative path against this repo's root, not cwd.

    include_path defaults to the repo-relative 'resources' -- resolving it
    against cwd (the usual os.path.abspath behavior) would break compile_p4
    when called from any cwd other than the repo root. p4_path/output_dir
    are left to the caller (they may be relative-to-cwd or absolute; either
    way _to_wsl_path's os.path.abspath handles them the normal way).
    """
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def _to_wsl_path(windows_path: str) -> str:
    """Converts an absolute (or cwd-relative) Windows path to /mnt/c/... .

    Confirmed working for this repo's own Cyrillic path component
    (reviews/t11_tofino_port_and_env.md Part K / G).
    """
    abs_path = os.path.abspath(windows_path)
    drive, rest = os.path.splitdrive(abs_path)
    return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")


_ERRORS_WARNINGS_RE = re.compile(r"(\d+)\s+errors?,\s*(\d+)\s+warnings?\s+generated")


def compile_p4(p4_path: str, output_dir: str, architecture: str = "tna",
                target: str = "tofino",
                include_path: str = "resources",
                p4c_path: str = "~/open-p4studio/install/bin/p4c",
                timeout_seconds: int = 300) -> CompileResult:
    """Blocking: runs the real Tofino compiler over WSL2 and parses its
    resource report.

    Invokes `wsl bash -lc '<full command>'` rather than `["wsl", p4c_path,
    ...]` -- WSL2 only expands `~` in p4c_path and sources PATH/profile when
    explicitly run as a login shell (bash -lc), confirmed by a real compile
    in reviews/t11_tofino_port_and_env.md Part K; the naive form silently
    fails to find the compiler. Each path argument is individually
    shell-quoted (shlex.quote) so paths containing spaces or non-ASCII
    characters (this repo's own path has a Cyrillic 'Документы' component)
    survive intact. p4c_path itself is left unquoted/unresolved in Python --
    bash expands the `~` once inside the login shell, which is exactly the
    fix.

    `target` selects the `-b` flag (default "tofino", matching every prior
    compile in this project's history). Passing target="tofino2" alongside
    architecture="t2na" targets Tofino-2 -- confirmed accepted by this
    installed compiler's own `--help-targets` output (reviews/
    t11_tofino_port_and_env.md Part L). `p4c` internally defines
    `__TARGET_TOFINO__` from `-b`, so no extra `-D` flag is needed here.

    Raises RuntimeError if the compiler times out or fails to run (no output line
    containing error/warning count despite nonzero exit code).
    """
    # Deliberately NOT pre-creating output_dir here (neither via os.makedirs
    # nor a WSL-side `mkdir -p`): p4c behaves differently -- and fails its
    # own internal `find` with "No such file or directory" -- if output_dir
    # already exists (even empty) versus being allowed to create it itself.
    # Confirmed by direct experiment (reviews/t11_tofino_port_and_env.md);
    # every real compile in this project's history (including Task 1's)
    # left output_dir's creation entirely to p4c. output_dir's immediate
    # parent must already exist (p4c creates only the final path segment,
    # not the full tree -- same as plain `mkdir`, not `mkdir -p`).
    wsl_p4_path = _to_wsl_path(p4_path)
    wsl_output_dir = _to_wsl_path(output_dir)
    wsl_include_path = _to_wsl_path(_resolve_repo_relative(include_path))

    full_command = (
        f"{p4c_path} -b {target} -a {architecture} -I {shlex.quote(wsl_include_path)} "
        f"-g --verbose 2 -o {shlex.quote(wsl_output_dir)} {shlex.quote(wsl_p4_path)}"
    )
    cmd = ["wsl", "bash", "-lc", full_command]
    # stdin explicitly closed rather than inherited from the caller: standard
    # practice for a non-interactive subprocess invocation (avoids ever
    # blocking on unexpected input). Note this was NOT the fix for the
    # pytest-only failure hit while writing the slow integration test below
    # -- that was output_dir pre-creation (see the comment above); this is
    # just good hygiene kept alongside it.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds,
                               stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "p4c compilation timed out after %d seconds"
            % (timeout_seconds,)) from e

    result = parse_compile_logs(output_dir)

    # p4c reports its own aggregated errors/warnings count in a single
    # summary line on stdout/stderr, e.g. "0 errors, 10 warnings generated."
    # (reviews/t11_tofino_port_and_env.md Part K) -- this is the
    # authoritative count directly from the compiler, so parse it rather
    # than counting "error:"/"warning:" substrings (which would also match
    # unrelated text like a warning category name). If the summary line
    # isn't present at all (e.g. a hard crash before it could print), fall
    # back to None for both fields, consistent with CompileResult's
    # None-means-unknown convention for the other fields.
    combined_output = (proc.stdout or "") + (proc.stderr or "")
    m = _ERRORS_WARNINGS_RE.search(combined_output)
    if m:
        result.errors = int(m.group(1))
        result.warnings = int(m.group(2))
    elif proc.returncode != 0:
        raise RuntimeError(
            "p4c did not run (exit %d) -- no 'N errors, M warnings generated' line "
            "in its output, so this is a toolchain failure, not a compile failure:\n%s"
            % (proc.returncode, combined_output[-2000:]))

    return result


_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def compile_p4_async(p4_path: str, output_dir: str, **kwargs) -> "Future[CompileResult]":
    """Non-blocking: returns a concurrent.futures.Future. Call .done() to
    poll, .result(timeout=...) to join. One shared executor per process
    avoids spawning an unbounded number of background threads across many
    calls.
    """
    return _EXECUTOR.submit(compile_p4, p4_path, output_dir, **kwargs)
