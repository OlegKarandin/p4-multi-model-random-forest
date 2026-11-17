"""The independent-tree arm end to end, without p4c: train, verify with the
fake compiler of `test_campaign_e2e` (replaying each row's own model.json), render.

The render needs the `independent` baseline in the run (`figures.require_baseline`
refuses a run without it: a run of the new arm alone cannot be rendered, by design),
so it trains next to it, as the per-task and per-tree e2e tests do. The reporting
contrasts (`claims.JOINT_ARM_SLUGS`/`REPORTED_JOINT_ARM_SLUGS`) do not list the new arm
yet and paired_tests refuses a frame with no treatment arm, so a `joint-off` arm rides
along as the treatment; the new arm is asserted through its rows, designs, verdicts
and `load_campaign`. The run cannot assemble the pre-registered Holm families, so `allow_partial_family=True`."""
import json
import os

import pandas as pd

from src.main import run_plot_mode
from src.reporting.campaign_data import load_campaign
from src.training import campaign_runner
from src.training.config import TrainConfig
from src.verify import runner as verify_runner
from tests.test_campaign_e2e import (_CFG, _M, _SPLITS, ReplayingCompiler, _per_tree_data,
                                     _row_ids, _serial)


def test_the_independent_tree_arm_trains_verifies_and_renders_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("THESIS_P4C_IMAGE", raising=False)
    monkeypatch.delenv("THESIS_P4STUDIO_COMMIT", raising=False)
    run_dir = str(tmp_path / "run")
    arms = [("independent-tree", TrainConfig(alignment_enabled=False, **_CFG)),
            ("independent", TrainConfig(**_CFG)),
            ("joint", TrainConfig(alignment_enabled=False, **_CFG))]
    jobs = campaign_runner.plan_jobs(arms, [_M], list(_SPLITS), run_dir)
    assert {job.slug for job in jobs} == {"independent-tree", "independent", "joint-off"}
    summary = campaign_runner.run_jobs(jobs, _per_tree_data(), run_dir, "deadbeef", 1,
                                       executor_factory=_serial)
    assert summary.failed == {}, summary.failed

    feasible = _row_ids(run_dir)
    ids = sorted(feasible["row_id"])
    tree = [r for r in ids if r.startswith("independent-tree_M035_s00_k")]
    assert tree
    assert set(feasible.loc[feasible["row_id"].isin(tree), "arm"]) == {"independent-tree"}
    assert set(feasible.loc[feasible["row_id"].isin(tree), "alignment_enabled"]) == {"False"}
    for row_id in tree:
        with open(os.path.join(run_dir, "designs", row_id + ".p4"), encoding="utf-8") as handle:
            assert "code_pk_" in handle.read()
        with open(os.path.join(run_dir, "designs", row_id + ".model.json"),
                  encoding="utf-8") as handle:
            assert json.load(handle)["model_paths_differ"] is False

    compiler = ReplayingCompiler(false_feasible_row=None)
    assert verify_runner.run(run_dir, workers=2, compile_fn=compiler, min_free_bytes=0) == 0
    assert sorted(compiler.compiled) == ids
    verification = pd.read_csv(os.path.join(run_dir, "verification.csv"),
                               keep_default_na=False, dtype=str)
    assert sorted(verification["row_id"]) == ids
    assert set(verification["verdict"]) == {"EXACT"}

    deliverables = run_plot_mode(results_dir=run_dir, output_dir=str(tmp_path / "figures"),
                                 allow_partial_family=True)
    by_number = {d.number: d for d in deliverables}
    assert {9, 10} <= set(by_number) and 11 not in by_number
    assert set(load_campaign(run_dir)["arm_slug"]) == {"independent-tree", "independent", "joint-off"}
