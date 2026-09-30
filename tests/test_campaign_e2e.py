"""Task 16: one tiny compiler-verified campaign, end to end, without p4c.

Stage 1 trains two arms (joint, independent) at M = 35 on split 0 with the
synthetic 3-feature data of the determinism test. Stage 2 verifies every
design with a fake compiler that fabricates p4c's logs FROM EACH ROW'S OWN
designs/<row_id>.model.json (same tables, blocks, stages and stage_depth), so
every verdict is EXACT -- except one row, whose compile reports 13 stages and
no allocation, i.e. FALSE_FEASIBLE. Stage 3 renders every deliverable.

The run holds two arms and one M, so it cannot assemble the pre-registered
Holm families; `allow_partial_family=True` is the documented escape for such a
pilot (see `run_plot_mode`)."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.main import run_plot_mode
from src.p4gen.p4_compile import CompileResult
from src.reporting.campaign_data import load_campaign
from src.training import campaign_runner
from src.training.config import TrainConfig
from src.verify import runner as verify_runner
from tests.test_campaign_runner import _synthetic_data

_M = 35
_CFG = dict(n_trials=12, min_feasible_before_stop=5, lookback=4)

_SUMMARY = """Table allocation done 1 time(s), state = INITIAL
Number of stages in table allocation: {stages}
"""

_METRICS = {"phv": {"normal": [{"bit_width": 8, "containers_occupied": 16},
                               {"bit_width": 16, "containers_occupied": 29}]}}


def _serial(max_workers):
    """Stage 1 runs serially: concurrent training THREADS share global state
    (production uses processes), and a 2-thread pool failed once in 20 runs."""
    return ThreadPoolExecutor(max_workers=1)


def _resources(tables):
    """mau.resources.log in the real shape, committing exactly `tables`."""
    lines = ["| Stage Number | SRAM | Map RAM | TCAM |",
             "| Totals | 0 | 0 | 0 |",
             "Allocated Resource Usage",
             "| Table Name | Stage | a | b | c | d | TCAM |"]
    for table in tables:
        stage = table.get("stage") or 0
        lines.append("| ingress.{} | {} | 0 | 0 | 0 | 0 | {} |".format(
            table["table"], stage, table.get("blocks") or 0))
        lines.append("| ingress.{}$action | {} | 0 | 0 | 0 | 0 | 0 |".format(
            table["table"], stage))
    return "\n".join(lines) + "\n"


class ReplayingCompiler:
    """A fake p4c that agrees with each row's own model.json, except for
    `false_feasible_row`, which p4c places in 13 stages and never allocates."""

    def __init__(self, false_feasible_row):
        self.false_feasible_row = false_feasible_row
        self.compiled = []

    def __call__(self, p4_path, output_dir, timeout_seconds=300, **_):
        row_id = os.path.basename(p4_path)[:-len(".p4")]
        with open(os.path.join(os.path.dirname(p4_path), row_id + ".model.json"),
                  encoding="utf-8") as handle:
            model = json.load(handle)
        self.compiled.append(row_id)
        logs = os.path.join(output_dir, "pipe", "logs")
        os.makedirs(logs)
        if row_id == self.false_feasible_row:
            stages = 13
            resources = "| Stage Number | SRAM | Map RAM | TCAM |\n"  # never allocated
        else:
            stages = model["stage_depth"]
            resources = _resources(model["tables"])
        files = {"table_summary.log": _SUMMARY.format(stages=stages),
                 "mau.resources.log": resources,
                 "metrics.json": json.dumps(_METRICS)}
        for name, text in files.items():
            with open(os.path.join(logs, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        return CompileResult(errors=0, warnings=0, output="0 errors generated.\n")


def _row_ids(run_dir):
    frames = [pd.read_csv(os.path.join(run_dir, "rows", name),
                          keep_default_na=False, dtype=str)
              for name in sorted(os.listdir(os.path.join(run_dir, "rows")))]
    rows = pd.concat(frames, ignore_index=True)
    return rows[rows["infeasible"] == ""]


def test_tiny_run_verifies_and_renders_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("THESIS_P4C_IMAGE", raising=False)
    monkeypatch.delenv("THESIS_P4STUDIO_COMMIT", raising=False)
    run_dir = str(tmp_path / "run")

    # Stage 1: train.
    arms = [("joint", TrainConfig(**_CFG)), ("independent", TrainConfig(**_CFG))]
    jobs = campaign_runner.plan_jobs(arms, [_M], [0], run_dir)
    summary = campaign_runner.run_jobs(jobs, _synthetic_data(), run_dir, "deadbeef", 1,
                                       executor_factory=_serial)
    assert summary.failed == {}, summary.failed
    feasible = _row_ids(run_dir)
    slugs = {job.slug for job in jobs}
    for slug in slugs:  # non-vacuous: every arm produced at least one design
        assert (feasible["row_id"].str.startswith(slug + "_M")).any(), slug
    design_ids = sorted(feasible["row_id"])
    for row_id in design_ids:
        assert os.path.isfile(os.path.join(run_dir, "designs", row_id + ".model.json"))

    # Stage 2: verify, one row made FALSE_FEASIBLE.
    false_feasible = design_ids[0]
    compiler = ReplayingCompiler(false_feasible)
    assert verify_runner.run(run_dir, workers=2, compile_fn=compiler,
                             min_free_bytes=0) == 0
    assert sorted(compiler.compiled) == design_ids
    verification = pd.read_csv(os.path.join(run_dir, "verification.csv"),
                               keep_default_na=False, dtype=str)
    verdicts = dict(zip(verification["row_id"], verification["verdict"]))
    assert verdicts.pop(false_feasible) == "FALSE_FEASIBLE"
    assert set(verdicts.values()) == {"EXACT"}, verdicts

    # Stage 3: render.
    out_dir = str(tmp_path / "figures")
    deliverables = run_plot_mode(results_dir=run_dir, output_dir=out_dir,
                                 allow_partial_family=True)
    by_number = {d.number: d for d in deliverables}
    assert {9, 10} <= set(by_number)
    misses_csv = [p for p in by_number[9].paths if p.endswith("_misses.csv")]
    assert len(misses_csv) == 1 and os.path.dirname(misses_csv[0]) == out_dir
    misses = pd.read_csv(misses_csv[0], keep_default_na=False, dtype=str)
    assert list(misses["row_id"]) == [false_feasible]
    assert list(misses["verdict"]) == ["FALSE_FEASIBLE"]

    frame = load_campaign(run_dir)
    assert false_feasible not in set(frame["row_id"])
    assert sorted(frame["row_id"]) == sorted(verdicts)
    p4c_blocks = dict(zip(verification["row_id"],
                          pd.to_numeric(verification["p4c_blocks"])))
    for row_id, blocks in zip(frame["row_id"], frame["blocks"]):
        assert blocks == p4c_blocks[row_id], row_id
    assert not frame["flagged"].any()
