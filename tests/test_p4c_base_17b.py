"""Whole-program check of the model against p4c on programs the CURRENT generator writes.

Twelve base programs of the Task 17b compile experiment, regenerated with the final pins of gate G3 (Task 23,
generator at ae34590: decisions 1a / 7.1 option B, 1f vote export, 2.3 key pruning, 4.3 full-width pinned codes,
the forwarding assignments first, and the G3 pin set -- no @pa_no_overlay on fwd / class_tree_*, no @pa_solitary
on code_*, @pa_container_size 16 on every IAT scratch field), three per encoding and mixed k, committed with p4c's
committed allocation (mau.resources.log per table: blocks summed over stages, stage; table_summary.log stage
count) in tests/fixtures/p4c_base_17b.json. Source: results/encoding_evidence_2026_11/model8/impl/t23 (Codespace
/workspaces/t23), built by model8/impl/scripts/t18_fixture.py. Regenerated and recompiled for the symmetric 12-bit
flow hash (review A1, Task 31; model8/impl/scripts/exp31.py fixture, Codespace /workspaces/dr_t31_out/fixA1): every
text is its old text with that edit (t31_text_diff.py, 12/12), p4c stage depth 2 lower on 12/12, blocks unchanged.
And again for review A2 (a one-tree task has no vote table; Codespace /workspaces/dr_t31_out/fixA2): p4c stage depth
1 lower again on the 5 programs whose LAST tree belongs to a one-tree task (3 with one tree per task, 2 with app 1 /
ddos 3-5 trees), unchanged on the other 7, blocks unchanged; the model's depth exact on 12/12. Recompiled once more
with the @name keys (B1) and the clean-ups (Task 31's 160-design run, /workspaces/dr_t31_out/final, a0cbb9f): every
p4c number of the 12 identical to the A2 compiles. The texts were regenerated once more with
@pa_solitary(flow_hash) (2026-11-15, Task 32 addendum F3; impl/scripts/t32_fixture_texts.py: each text is its old
text plus that one pragma line, 12/12) and NOT recompiled yet: the p4c numbers are still the a0cbb9f compiles of the
texts without the pin (the pin changes no model number; the final 160-design recompile follows the next task).
The old calibration archive
(compiler_calibration_pragmas_2026_09_29) can no longer score the model: every program in it is refused by shape
(test_model_archive).

model_breakdown must give p4c's stage depth and blocks, and p4c's stage and blocks for every table it prices, on
every program; the one named miss is pinned (Task 18, impl/remeasure.md):
  * joint-tree_M075_s02_k07 (k<=8): blocks 33 vs p4c 34; get_classification_tree_app_1 7 vs 8 (a per-tree block
    under-price, unchanged since Task 17).
independent_M015_s06_k04 (k<=8), a pinned miss until Task 23 (model 9 vs p4c 10: p4c packed fwd_current_iat and
bwd_current_iat into one 32-bit container and ran the second last-arrival register a stage late), is exact since
the scratch fields are pinned to 16-bit containers (p4c 10 -> 9 stages).
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
    # Since review A2 (Task 31): a register-ALU tie the model breaks the other way. At stage 2 four of five ready
    # registers fit; p4c (table_placement_1.log) keeps fwd_packet_length_max_reg over bwd_last_arrival_time_reg
    # "for reason: fewer total dependencies in dominance frontier" -- their downward dependence tails now tie,
    # because the app task's tree (the end of both chains) no longer feeds a vote table -- while the model's
    # schedule (longer register tail first) runs bwd_last_arrival_time first. Both fwd_packet_length_max range
    # tables then land a stage earlier in p4c than in the model; stage depth and blocks exact.
    "independent_M050_s13_k07": {
        "table_11_ddos_fwd_packet_length_max": {"stage": (4, 3)},
        "table_4_app_fwd_packet_length_max": {"stage": (4, 3)},
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
    tree, and the last stage holds a vote or, since review A2, the tree of a one-tree task (which has no vote
    table)."""
    tables = P4C[row]["tables"]
    stages = []
    for task in ("app", "ddos"):
        trees = [stage for name, (_, stage) in tables.items()
                 if name.startswith("get_classification_tree_%s_" % task)]
        if len(trees) == 1:
            assert "vote_" + task not in tables
            stages.append(trees[0])
            continue
        vote = tables["vote_" + task][1]
        assert vote == max(trees) + VOTE_EPILOGUE_STAGES
        stages.append(vote)
    assert max(stages) == P4C[row]["stage_depth"] - 1
