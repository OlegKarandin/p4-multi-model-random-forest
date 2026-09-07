"""Replays tests/fixtures/resource_model_golden.json through the resource model.

No sklearn, no fitted forests, no campaign backup: the fixture was serialized at
the _pool_inputs seam, so this exercises the whole packing and accounting core
against numbers pinned before the p4model extraction began. Any diff here means
the restructuring changed a prediction -- which it must never do."""
import json
import math
import os

import pytest

from src.p4model.registers import gated_block_interior_stages
from src.p4model.usage import assemble_usage

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "resource_model_golden.json")


def load_fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def rebuild_pool(row):
    """Interned (id, bits) sets -> the (id, bytes) frozensets the packer takes."""
    sets = [frozenset((fid, math.ceil(bits / 8)) for fid, bits in fields)
            for fields in row["key_field_sets"]]
    src = row["inputs"]
    pool = {
        "range_table_specs": [tuple(s) for s in src["range_table_specs"]],
        "ternary_table_specs": [tuple(s) for s in src["ternary_table_specs"]],
        "range_levels": src["range_levels"],
        "range_fields": [sets[i] for i in src["range_key_field_set_ids"]],
        "ternary_fields": [sets[i] for i in src["ternary_key_field_set_ids"]],
        "ternary_ragged": src["ternary_ragged"],
        "interior_stages": frozenset(src["interior_stages"]),
        "emitted_features": src["emitted_features"],
        "register_names": tuple(src["register_names"]),
        "range_entries": src["range_entries"],
        "range_blocks": src["range_blocks"],
        "ternary_entries": src["ternary_entries"],
        "codeword_length": src["codeword_length"],
    }
    # Bits are authoritative; the recorded byte width is redundant with them and
    # stored only so the pool can be fed straight in. This catches any drift.
    for (_, byte_width), fields in zip(pool["ternary_table_specs"],
                                       pool["ternary_fields"]):
        assert sum(w for _, w in fields) == byte_width
    return pool


def fixture_rows():
    return [(row["row_id"], row) for row in load_fixture()["rows"]]


IDS = [r[0] for r in fixture_rows()]


def test_fixture_covers_all_nineteen_calibration_rows():
    rows = load_fixture()["rows"]
    assert len(rows) == 19
    assert len({r["row_id"] for r in rows}) == 19


@pytest.mark.parametrize("row_id,row", fixture_rows(), ids=IDS)
def test_golden_resource_usage_is_unchanged(row_id, row):
    usage, range_plan, ternary_plan = assemble_usage(rebuild_pool(row))

    expected = row["outputs"]
    for field, value in expected["usage"].items():
        assert getattr(usage, field) == value, (row_id, field)

    for plan, name in ((range_plan, "range_plan"), (ternary_plan, "ternary_plan")):
        want = expected[name]
        assert plan.occupied == want["occupied"], (row_id, name, "occupied")
        assert plan.depth == want["depth"], (row_id, name, "depth")
        assert plan.blocks == want["blocks"], (row_id, name, "blocks")
        assert sorted(plan.indices) == want["indices"], (row_id, name, "indices")


@pytest.mark.parametrize("row_id,row", fixture_rows(), ids=IDS)
def test_gated_block_interior_stages_still_derives_from_emitted_features(row_id, row):
    # The fixture stores interior_stages as an INPUT (assemble_usage takes it
    # ready-made), so without this the registers.py schedule would go
    # unexercised by the golden replay. Deriving it here pins that half too.
    derived = gated_block_interior_stages(row["inputs"]["emitted_features"])
    assert sorted(derived) == row["inputs"]["interior_stages"], row_id
