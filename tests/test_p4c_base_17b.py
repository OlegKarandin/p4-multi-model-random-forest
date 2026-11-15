"""Whole-program check of the model against p4c on programs the CURRENT generator writes.

Twelve base programs of the Task 17b compile experiment (generator at e1e08dd: decisions 1a / 7.1 option B, 1f
vote export, 2.3 key pruning, 4.3 full-width pinned codes, the forwarding assignments first), three per encoding
and mixed k, committed with p4c's committed allocation (mau.resources.log per table: blocks summed over stages,
stage; table_summary.log stage count) in tests/fixtures/p4c_base_17b.json. Source:
results/encoding_evidence_2026_11/model8/impl/t17b_experiment (Codespace /workspaces/dr_exp17b), built by
model8/impl/scripts/t18_fixture.py. The old calibration archive (compiler_calibration_pragmas_2026_09_29) can no
longer score the model: every program in it is refused by shape (test_model_archive).

model_breakdown must give p4c's stage depth and blocks, and p4c's stage and blocks for every table it prices, on
every program; the two named misses are pinned (Task 18, impl/remeasure.md):
  * independent_M015_s06_k04 (k<=8): stage depth 9 vs p4c 10. p4c packed the IAT scratch fields fwd_current_iat
    and bwd_current_iat into one 32-bit container and so placed bwd_last_arrival_time's register a stage after
    fwd_last_arrival_time's ("action dependency ... due to PHV allocation"); the two bwd range tables and both
    trees follow it. The model's register schedule co-schedules the two registers.
  * joint-tree_M075_s02_k07 (k<=8): blocks 33 vs p4c 34; get_classification_tree_app_1 7 vs 8 (a per-tree block
    under-price, unchanged since Task 17).
"""
import json
import os

import pytest

from p4gen.p4_replay import model_breakdown, parse_program
from p4model.program import VOTE_EPILOGUE_STAGES

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "p4c_base_17b")

with open(_FIXTURE + ".json", encoding="utf-8") as _handle:
    P4C = json.load(_handle)

# row -> {"stage_depth"/"blocks": (model, p4c), table: {"stage"/"blocks": (model, p4c)}}
KNOWN_MISSES = {
    "independent_M015_s06_k04": {
        "stage_depth": (9, 10),
        "table_0_bwd_iat_mean": {"stage": (6, 7)},
        "table_4_bwd_iat_min": {"stage": (6, 7)},
        "get_classification_tree_app_0": {"stage": (7, 8)},
        "get_classification_tree_ddos_0": {"stage": (7, 8)},
    },
    "joint-tree_M075_s02_k07": {
        "blocks": (33, 34),
        "get_classification_tree_app_1": {"blocks": (7, 8)},
    },
}


def test_the_fixture_covers_every_encoding_at_both_k_groups():
    assert len(P4C) == 12
    cells = {}
    for numbers in P4C.values():
        key = (numbers["encoding"], numbers["k"] <= 8)
        cells[key] = cells.get(key, 0) + 1
    assert {encoding for encoding, _ in cells} == {"joint", "disjoint", "per_task", "per_tree"}
    assert all((encoding, low) in cells for encoding, _ in cells for low in (True, False))


@pytest.mark.parametrize("row", sorted(P4C))
def test_model_breakdown_gives_p4c_numbers(row):
    numbers = P4C[row]
    model = model_breakdown(parse_program(os.path.join(_FIXTURE, row + ".p4")), row)
    assert model["refusal"] is None
    priced = {t["table"]: t for t in model["tables"]}
    assert set(priced) <= set(numbers["tables"])
    misses = {}
    p4c_blocks = sum(numbers["tables"][t][0] for t in priced)
    if model["stage_depth"] != numbers["stage_depth"]:
        misses["stage_depth"] = (model["stage_depth"], numbers["stage_depth"])
    if model["blocks"] != p4c_blocks:
        misses["blocks"] = (model["blocks"], p4c_blocks)
    for table, entry in priced.items():
        blocks, stage = numbers["tables"][table]
        per_table = {}
        if entry["stage"] != stage:
            per_table["stage"] = (entry["stage"], stage)
        if entry["blocks"] != blocks:
            per_table["blocks"] = (entry["blocks"], blocks)
        if per_table:
            misses[table] = per_table
    assert misses == KNOWN_MISSES.get(row, {})


@pytest.mark.parametrize("row", sorted(P4C))
def test_p4c_puts_each_vote_one_stage_after_its_tasks_last_tree(row):
    """S26 / decision 2.7 on these programs (VOTE_EPILOGUE_STAGES): each vote one stage after its own task's last
    tree, and the last stage holds a vote."""
    tables = P4C[row]["tables"]
    stages = []
    for task in ("app", "ddos"):
        trees = [stage for name, (_, stage) in tables.items()
                 if name.startswith("get_classification_tree_%s_" % task)]
        vote = tables["vote_" + task][1]
        assert vote == max(trees) + VOTE_EPILOGUE_STAGES
        stages.append(vote)
    assert max(stages) == P4C[row]["stage_depth"] - 1
