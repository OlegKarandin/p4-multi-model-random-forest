"""The resumable p4c verifier (spec 2026-09-29 §6), driven by a fake compiler.

Never runs real p4c: every test injects a `compile_fn` that writes log files in
the real files' shapes (the Task 1 fixture strings) under output_dir/pipe/logs
and returns a CompileResult."""
import csv
import json
import os
import tarfile

import pytest

from src.p4gen.p4_compile import CompileResult, P4CompileTimeout
from src.training.campaign_run import atomic_write_text, canonical_json, run_paths
from src.verify import runner
from src.verify import __main__ as cli
from src.verify.verdicts import classify

_RESOURCES = """\
| Stage Number | SRAM | Map RAM | TCAM |
| Totals | 40 | 10 | 6 |
Allocated Resource Usage
| Table Name | Stage | a | b | c | d | TCAM |
| ingress.table_0_flow_iat_max | 3 | 0 | 0 | 0 | 0 | 1 |
| ingress.table_0_flow_iat_max$action | 3 | 0 | 0 | 0 | 0 | 0 |
| ingress.get_classification_tree_app_0 | 7 | 0 | 0 | 0 | 0 | 5 |
"""

_SUMMARY = """Table allocation done 1 time(s), state = INITIAL
Number of stages in table allocation: 11
Table allocation done 2 time(s), state = REDO_PHV1
Number of stages in table allocation: 12
"""

_METRICS = {"phv": {"normal": [{"bit_width": 8, "containers_occupied": 16},
                               {"bit_width": 16, "containers_occupied": 29}]}}

_TABLES = [{"table": "table_0_flow_iat_max", "kind": "range", "blocks": 1, "stage": 3},
           {"table": "get_classification_tree_app_0", "kind": "ternary", "blocks": 5,
            "stage": 7}]


def _write_logs(output_dir, resources=_RESOURCES, summary=_SUMMARY):
    logs = os.path.join(output_dir, "pipe", "logs")
    os.makedirs(logs)
    files = {"mau.resources.log": resources, "table_summary.log": summary,
             "metrics.json": json.dumps(_METRICS),
             "phv_allocation_summary_0.log": "phv\n",
             "table_dependency_summary.log": "deps\n", "pragmas.log": "pragmas\n",
             "table_placement_1.log": "round 1\n", "table_placement_3.log": "round 3\n",
             "table_placement_10.log": "round 10\n",
             # not kept
             "phv.json": "{}", "mau.json": "{}"}
    for name, text in files.items():
        if text is not None:
            with open(os.path.join(logs, name), "w") as handle:
                handle.write(text)
    with open(os.path.join(output_dir, "pipe", "prog.bfa"), "w") as handle:
        handle.write("bfa\n")
    with open(os.path.join(output_dir, "pipe", "tofino.bin"), "w") as handle:
        handle.write("not kept\n")


class FakeCompiler:
    """Scripted p4c: each call pops the next action ('ok', 'timeout', 'crash',
    or a callable writing its own logs)."""

    def __init__(self, *script):
        self.script = list(script) or ["ok"]
        self.calls = []

    def __call__(self, p4_path, output_dir, timeout_seconds=300, **_):
        self.calls.append({"p4_path": p4_path, "timeout": timeout_seconds})
        action = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if action == "timeout":
            raise P4CompileTimeout("p4c compilation timed out after %d seconds"
                                   % timeout_seconds)
        if action == "crash":
            raise RuntimeError("p4c did not run")
        if callable(action):
            return action(output_dir)
        _write_logs(output_dir)
        return CompileResult(errors=0, warnings=3, sram=40, map_ram=10,
                             output="0 errors, 3 warnings generated.\n")


def _model(row_id, **over):
    model = {"row_id": row_id, "git_commit": "abc123", "M": 35, "budgeted": True,
             "training_stage_depth": 12, "training_blocks": 6, "stages": 5,
             "register_depth": 3, "register_count": 10, "range_entries": 20,
             "ternary_entries": 100, "codeword_bits": 40, "stage_depth": 12,
             "blocks": 6, "tables": _TABLES, "model_paths_differ": False,
             "hw_feasible": True, "budget_feasible": True, "generator_error": None}
    model.update(over)
    return model


def _run_dir(tmp_path, rows=("r",), **over):
    run_dir = tmp_path / "run"
    for sub in ("designs", "rows", "verify", "forests"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for row_id in rows:
        (run_dir / "designs" / (row_id + ".p4")).write_text("// program %s\n" % row_id)
        (run_dir / "designs" / (row_id + ".model.json")).write_text(
            canonical_json(_model(row_id, **over)))
    return str(run_dir)


def _verify_json(run_dir, row_id="r"):
    with open(os.path.join(run_dir, "verify", row_id + ".json")) as handle:
        return json.load(handle)


def _csv_rows(run_dir):
    with open(os.path.join(run_dir, "verification.csv"), newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_normal_row_is_exact_kept_logs_tarred_and_merged(tmp_path, monkeypatch):
    monkeypatch.setenv("THESIS_P4C_IMAGE", "ghcr.io/x/p4c:1")
    monkeypatch.delenv("THESIS_P4STUDIO_COMMIT", raising=False)
    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler()
    record = runner.verify_row(run_dir, "r", compile_fn=fake)

    assert fake.calls == [{"p4_path": os.path.join(run_dir, "designs", "r.p4"),
                           "timeout": 300}]
    assert record["verdict"] == "EXACT"
    assert (record["p4c_stage_depth"], record["p4c_blocks"]) == (12, 6)
    assert (record["p4c_sram"], record["p4c_map_ram"], record["p4c_phv_containers"]) \
        == (40, 10, 45)
    assert (record["p4c_errors"], record["p4c_warnings"]) == (0, 3)
    assert record["unverified"] is False and record["tables_differing"] == []
    assert record["p4c_image"] == "ghcr.io/x/p4c:1"
    assert record["open_p4studio_commit"] is None
    assert record["model_git_commit"] == "abc123"
    assert record["model_training_stage_depth"] == 12
    assert record["model_paths_differ"] is False
    assert record["verified_utc"]
    assert len(record["p4_sha256"]) == 64
    assert sorted(record) == sorted(runner.VERIFICATION_COLUMNS)
    assert _verify_json(run_dir) == record

    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        names = sorted(tar.getnames())
    want = sorted(["pipe/logs/" + n for n in runner.KEPT_LOGS]
                  + ["pipe/logs/table_placement_10.log", "pipe/prog.bfa"])
    assert names == want

    runner.merge_verification(run_dir)
    lines = _csv_rows(run_dir)
    assert [line["row_id"] for line in lines] == ["r"]
    assert lines[0]["verdict"] == "EXACT"
    assert json.loads(lines[0]["tables_differing"]) == []
    with open(os.path.join(run_dir, "verification.csv"), newline="") as handle:
        assert next(csv.reader(handle)) == list(runner.VERIFICATION_COLUMNS)


def test_a_first_timeout_is_retried_with_the_long_timeout(tmp_path):
    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler("timeout", "ok")
    record = runner.verify_row(run_dir, "r", compile_fn=fake)
    assert [c["timeout"] for c in fake.calls] == [300, 1800]
    assert record["verdict"] == "EXACT" and record["p4c_stage_depth"] == 12


def test_two_timeouts_record_timeout_and_unverified(tmp_path):
    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler("timeout", "timeout")
    record = runner.verify_row(run_dir, "r", compile_fn=fake)
    assert len(fake.calls) == 2
    assert record["verdict"] == "TIMEOUT" and record["unverified"] is True
    assert record["p4c_stage_depth"] is None and record["failure"] == "p4c_timeout"
    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        output = tar.extractfile("p4c_output.txt").read().decode()
    assert "timed out after 1800" in output


def test_a_crash_then_crash_is_a_compile_error(tmp_path):
    run_dir = _run_dir(tmp_path)
    record = runner.verify_row(run_dir, "r", compile_fn=FakeCompiler("crash", "crash"))
    assert record["verdict"] == "COMPILE_ERROR" and record["failure"] == "p4c_toolchain"


def test_errors_without_a_table_summary_are_a_compile_error(tmp_path):
    def failed(output_dir):
        os.makedirs(os.path.join(output_dir, "pipe", "logs"))
        return CompileResult(errors=2, warnings=0, output="error: bad\n2 errors, 0 warnings generated.\n")

    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler(failed)
    record = runner.verify_row(run_dir, "r", compile_fn=fake)
    assert len(fake.calls) == 1
    assert record["verdict"] == "COMPILE_ERROR" and record["failure"] == "p4c_errors"
    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        assert tar.getnames() == ["p4c_output.txt"]


def test_a_native_compile_keeps_its_row_named_bfa_as_prog_bfa(tmp_path):
    """The native route compiles designs/<row_id>.p4, so p4c writes
    pipe/<row_id>.bfa; the tarball keeps it under the stable pipe/prog.bfa."""
    def native(output_dir):
        _write_logs(output_dir)
        os.remove(os.path.join(output_dir, "pipe", "prog.bfa"))
        with open(os.path.join(output_dir, "pipe", "r.bfa"), "w", newline="") as handle:
            handle.write("native bfa\n")
        return CompileResult(errors=0, warnings=0, output="")

    run_dir = _run_dir(tmp_path)
    runner.verify_row(run_dir, "r", compile_fn=FakeCompiler(native))
    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        assert "pipe/r.bfa" not in tar.getnames()
        assert tar.extractfile("pipe/prog.bfa").read().decode() == "native bfa\n"


def test_errors_with_a_feasible_table_summary_are_a_compile_error(tmp_path):
    """errors > 0 with <= 12 stages logged and no allocation: not a verified
    compile (it would score NaN blocks downstream)."""
    def errored(output_dir):
        _write_logs(output_dir, resources="| Stage Number | SRAM | Map RAM | TCAM |\n",
                    summary="Table allocation done 1 time(s), state = INITIAL\n"
                            "Number of stages in table allocation: 11\n")
        return CompileResult(errors=1, warnings=0,
                             output="error: no fit\n1 error, 0 warnings generated.\n")

    run_dir = _run_dir(tmp_path)
    record = runner.verify_row(run_dir, "r", compile_fn=FakeCompiler(errored))
    assert record["verdict"] == "COMPILE_ERROR" and record["unverified"] is True
    assert record["failure"] == "p4c_errors"
    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        assert "error: no fit" in tar.extractfile("p4c_output.txt").read().decode()


def test_errors_over_12_stages_stay_false_feasible(tmp_path):
    def too_deep(output_dir):
        _write_logs(output_dir, resources="| Stage Number | SRAM | Map RAM | TCAM |\n",
                    summary="Table allocation done 1 time(s), state = INITIAL\n"
                            "Number of stages in table allocation: 13\n")
        return CompileResult(errors=1, warnings=0, output="error: no fit\n")

    run_dir = _run_dir(tmp_path)
    record = runner.verify_row(run_dir, "r", compile_fn=FakeCompiler(too_deep))
    assert record["verdict"] == "FALSE_FEASIBLE" and record["failure"] is None


def test_a_compile_without_a_stage_count_is_a_p4c_error(tmp_path):
    def no_summary(output_dir):
        os.makedirs(os.path.join(output_dir, "pipe", "logs"))
        return CompileResult(errors=0, warnings=0, output="odd p4c output\n")

    run_dir = _run_dir(tmp_path)
    record = runner.verify_row(run_dir, "r", compile_fn=FakeCompiler(no_summary))
    assert record["verdict"] == "COMPILE_ERROR" and record["failure"] == "p4c_errors"
    with tarfile.open(os.path.join(run_dir, "verify", "r.tar.gz")) as tar:
        assert tar.extractfile("p4c_output.txt").read().decode() == "odd p4c output\n"


def test_a_generator_error_is_recorded_without_compiling(tmp_path):
    run_dir = _run_dir(tmp_path, stage_depth=None, blocks=None, tables=None,
                       hw_feasible=False, budget_feasible=False,
                       generator_error="shared-field layout conflict")
    os.remove(os.path.join(run_dir, "designs", "r.p4"))
    with open(os.path.join(run_dir, "designs", "r.generator_error.txt"), "w") as handle:
        handle.write("shared-field layout conflict")
    fake = FakeCompiler()
    record = runner.verify_row(run_dir, "r", compile_fn=fake)
    assert fake.calls == []
    assert record["verdict"] == "COMPILE_ERROR" and record["unverified"] is True
    assert record["failure"] == "generator_error" and record["p4_sha256"] is None
    assert not os.path.exists(os.path.join(run_dir, "verify", "r.tar.gz"))


def test_a_stale_partial_is_pending_and_run_redoes_it(tmp_path):
    run_dir = _run_dir(tmp_path, rows=("a", "r"))
    runner.verify_row(run_dir, "a", compile_fn=FakeCompiler())
    with open(os.path.join(run_dir, "verify", "r.json.partial"), "w") as handle:
        handle.write('{"row_id": "r", "verd')
    assert runner.pending_rows(run_dir) == ["r"]

    fake = FakeCompiler()
    assert runner.run(run_dir, compile_fn=fake, min_free_bytes=0) == 0
    assert len(fake.calls) == 1
    assert runner.pending_rows(run_dir) == []
    assert [line["row_id"] for line in _csv_rows(run_dir)] == ["a", "r"]


def test_a_feasible_csv_row_without_model_json_is_an_error(tmp_path, capsys):
    run_dir = _run_dir(tmp_path, rows=("r",))
    with open(os.path.join(run_dir, "rows", "x.csv"), "w", newline="") as handle:
        handle.write("row_id,k,infeasible\nr,3,\nghost,4,\ninf_row,5,no feasible trial\n")
    assert runner.pending_rows(run_dir) == ["ghost", "r"]
    code = runner.run(run_dir, compile_fn=FakeCompiler(), min_free_bytes=0)
    assert code == 1
    assert "ghost" in capsys.readouterr().out
    assert runner.pending_rows(run_dir) == ["ghost"]


def test_low_disk_compiles_nothing_and_exits_2(tmp_path, capsys):
    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler()
    assert runner.run(run_dir, compile_fn=fake, min_free_bytes=10 ** 18) == 2
    assert fake.calls == []
    out = capsys.readouterr().out
    assert "stopped: free disk" in out and "restart resumes" in out
    assert os.path.isfile(os.path.join(run_dir, "verification.csv"))


def test_rescore_uses_edited_model_json_without_compiling(tmp_path, monkeypatch):
    """model.json edited in place (its budget M) plus a model recompute from
    the saved program (stage_depth 11): both reach the rescored record, and
    nothing is compiled."""
    run_dir = _run_dir(tmp_path)
    runner.verify_row(run_dir, "r", compile_fn=FakeCompiler())
    path = os.path.join(run_dir, "designs", "r.model.json")
    with open(path, "w") as handle:
        handle.write(canonical_json(_model("r", M=5)))
    monkeypatch.setattr(runner, "parse_program", lambda p4: ("parsed", p4))
    monkeypatch.setattr(runner, "model_breakdown",
                        lambda program, row_id: {"stage_depth": 11, "blocks": 6,
                                                 "tables": _TABLES})

    def never(*a, **k):
        raise AssertionError("rescore must not compile")

    monkeypatch.setattr(runner, "compile_p4", never)
    runner.rescore(run_dir)
    record = _verify_json(run_dir)
    assert record["verdict"] == "UNDER" and record["model_stage_depth"] == 11
    assert record["M"] == 5 and record["model_budget_feasible"] is False
    assert record["p4c_over_budget"] is True
    assert record["p4c_stage_depth"] == 12 and record["p4c_blocks"] == 6
    assert _csv_rows(run_dir)[0]["verdict"] == "UNDER"


def test_rescore_raises_when_a_rows_program_is_missing(tmp_path):
    run_dir = _run_dir(tmp_path)
    runner.verify_row(run_dir, "r", compile_fn=FakeCompiler())
    os.remove(os.path.join(run_dir, "designs", "r.p4"))
    with pytest.raises(FileNotFoundError, match="r"):
        runner.rescore(run_dir)


def test_rescore_recomputes_the_model_from_the_saved_program(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    runner.verify_row(run_dir, "r", compile_fn=FakeCompiler())
    first = _verify_json(run_dir)
    tables = [dict(_TABLES[0]), dict(_TABLES[1], blocks=7)]
    monkeypatch.setattr(runner, "parse_program", lambda p4: ("parsed", p4))
    monkeypatch.setattr(runner, "model_breakdown",
                        lambda program, row_id: {"stage_depth": 12, "blocks": 8,
                                                 "tables": tables})
    monkeypatch.setattr(runner, "compile_p4", None)
    runner.rescore(run_dir)
    record = _verify_json(run_dir)
    assert record["verdict"] == "OVER" and record["model_blocks"] == 8
    assert record["model_paths_differ"] is True
    assert record["tables_differing"] == [
        {"table": "get_classification_tree_app_0", "model": 7, "p4c": 5}]
    assert record["verified_utc"] == first["verified_utc"]


def test_rescore_keeps_a_timeout(tmp_path):
    run_dir = _run_dir(tmp_path)
    runner.verify_row(run_dir, "r", compile_fn=FakeCompiler("timeout", "timeout"))
    runner.rescore(run_dir)
    assert _verify_json(run_dir)["verdict"] == "TIMEOUT"


def _archive(tmp_path):
    archive = tmp_path / "archive"
    for row, tree_blocks in (("d1", 5), ("d2", 4)):
        (archive / "p4_src").mkdir(parents=True, exist_ok=True)
        (archive / "p4_src" / (row + ".p4")).write_text("// %s\n" % row)
        stored = archive / "compiles" / row
        stored.mkdir(parents=True)
        _write_logs(str(stored),
                    resources=_RESOURCES.replace("| 5 |", "| %d |" % tree_blocks))
    return str(archive)


def test_verify_archive_names_the_differing_table_and_exits_1(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path)
    fake = FakeCompiler()
    results = runner.verify_archive(archive, compile_fn=fake)
    assert [r["row"] for r in results] == ["d1", "d2"]
    assert [r["match"] for r in results] == [True, False]
    assert results[1]["tables_differing"] == [
        {"table": "get_classification_tree_app_0", "new": 5, "archived": 4}]
    out = capsys.readouterr().out
    assert "get_classification_tree_app_0" in out and "median compile seconds" in out

    monkeypatch.setattr(runner, "compile_p4", FakeCompiler())
    assert cli.main(["--archive", archive]) == 1


def test_verify_archive_reports_a_raising_design_as_error_and_goes_on(
        tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path)
    ok = FakeCompiler()

    def flaky(p4_path, output_dir, timeout_seconds=300, **kw):
        if os.path.basename(p4_path) == "d1.p4":
            raise P4CompileTimeout("p4c compilation timed out after 1800 seconds")
        return ok(p4_path, output_dir, timeout_seconds=timeout_seconds)

    results = runner.verify_archive(archive, compile_fn=flaky)
    assert [r["row"] for r in results] == ["d1", "d2"]
    assert results[0]["match"] is False and "timed out" in results[0]["error"]
    assert results[1]["tables_differing"] == [
        {"table": "get_classification_tree_app_0", "new": 5, "archived": 4}]
    out = capsys.readouterr().out
    assert "d1: ERROR" in out and "timed out" in out
    assert "d2: DIFF" in out and "total: 0/2 match" in out

    monkeypatch.setattr(runner, "compile_p4", flaky)
    assert cli.main(["--archive", archive]) == 1


def test_copy_record_duplicates_the_source_verdict_for_a_byte_identical_twin(tmp_path):
    run = str(tmp_path)
    paths = run_paths(run).ensure()
    program = "control Ingress() { }\n"
    for row_id in ("joint-off_M035_s00_k03", "joint-off-al_M035_s00_k03"):
        atomic_write_text(os.path.join(paths.designs, row_id + ".p4"), program)
        atomic_write_text(os.path.join(paths.designs, row_id + ".model.json"),
                          canonical_json(_model(row_id)))
    source = runner._record(_model("joint-off_M035_s00_k03"), "joint-off_M035_s00_k03",
                                   os.path.join(paths.designs, "joint-off_M035_s00_k03.p4"),
                                   classify(_model("x"), None, 35, "COMPILE_ERROR"), failure="p4c_timeout")
    atomic_write_text(os.path.join(paths.verify, "joint-off_M035_s00_k03.json"), canonical_json(source))

    record = runner.copy_record(run, "joint-off-al_M035_s00_k03", "joint-off_M035_s00_k03")

    on_disk = json.loads(open(os.path.join(paths.verify, "joint-off-al_M035_s00_k03.json")).read())
    assert on_disk == record
    assert record["row_id"] == "joint-off-al_M035_s00_k03"
    assert record["copied_from"] == "joint-off_M035_s00_k03"
    assert record["verdict"] == source["verdict"] and record["p4_sha256"] == source["p4_sha256"]
    assert set(record) == set(runner.VERIFICATION_COLUMNS)


def test_copy_record_refuses_when_the_twin_program_differs(tmp_path):
    run = str(tmp_path)
    paths = run_paths(run).ensure()
    atomic_write_text(os.path.join(paths.designs, "a.p4"), "A\n")
    atomic_write_text(os.path.join(paths.designs, "b.p4"), "B\n")
    rec = runner._record(_model("a"), "a", os.path.join(paths.designs, "a.p4"),
                                classify(_model("a"), None, 35, "COMPILE_ERROR"), failure="p4c_timeout")
    atomic_write_text(os.path.join(paths.verify, "a.json"), canonical_json(rec))
    with pytest.raises(ValueError):
        runner.copy_record(run, "b", "a")


def test_copy_record_refuses_when_the_source_is_unverified(tmp_path):
    run = str(tmp_path)
    paths = run_paths(run).ensure()
    atomic_write_text(os.path.join(paths.designs, "b.p4"), "B\n")
    with pytest.raises(FileNotFoundError):
        runner.copy_record(run, "b", "a")


def test_a_compiled_record_has_an_empty_copied_from(tmp_path, monkeypatch):
    # Extend the existing normal-row test: after verify_row, assert record['copied_from'] is None
    # and that verification.csv has a 'copied_from' header.
    monkeypatch.setenv("THESIS_P4C_IMAGE", "ghcr.io/x/p4c:1")
    monkeypatch.delenv("THESIS_P4STUDIO_COMMIT", raising=False)
    run_dir = _run_dir(tmp_path)
    fake = FakeCompiler()
    record = runner.verify_row(run_dir, "r", compile_fn=fake)

    assert record['copied_from'] is None

    runner.merge_verification(run_dir)
    with open(os.path.join(run_dir, "verification.csv"), newline="") as handle:
        csv_header = next(csv.reader(handle))
    assert 'copied_from' in csv_header
