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
    # The version-block penalty prices a key by its field BIT widths, which the
    # (id, bytes) frozensets above have already rounded away -- so read them
    # back off the interned sets rather than storing a second copy.
    bit_sets = [tuple(sorted(bits for _, bits in fields))
                for fields in row["key_field_sets"]]
    src = row["inputs"]
    pool = {
        "range_table_specs": [tuple(s) for s in src["range_table_specs"]],
        "ternary_table_specs": [tuple(s) for s in src["ternary_table_specs"]],
        "range_levels": src["range_levels"],
        "range_task": src["range_task"],
        "ternary_task": src["ternary_task"],
        "range_fields": [sets[i] for i in src["range_key_field_set_ids"]],
        "ternary_fields": [sets[i] for i in src["ternary_key_field_set_ids"]],
        "ternary_key_bits": [bit_sets[i]
                             for i in src["ternary_key_field_set_ids"]],
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
    # Guards against a truncated fixture passing vacuously: without this, a
    # fixture missing fields would just skip them in the loop below instead
    # of failing. Mirrors test_resource_usage_field_order_is_pinned's list.
    assert set(expected["usage"]) == {
        "stages", "blocks", "stage_depth", "range_entries", "ternary_entries",
        "codeword_length", "register_depth", "register_count", "range_depth",
        "ternary_depth", "range_tables", "ternary_tables",
    }, row_id
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


def test_classification_stages_are_derivable_from_the_block_factor():
    """Design 2026-09-07 §4.3: the claim that justifies DELETING the 'stages'
    alignment objective.

    A table of `factor` blocks chains them down ONE column, and a stage is 12
    rows x 2 columns, so the column geometry admits `2 * (12 // factor)` such
    tables -- but the ternary crossbar independently caps a stage at
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE tables whatever their width, so the
    classification pool occupies
    `ceil(T / min(8, 2 * (12 // factor)))`. The cap is not subsumed by the
    geometry: it binds below factor 3, which no archived row reaches. See
    test_the_classification_stage_formula_honours_the_eight_table_cap.

    Where it holds, stages step
    ONLY when factor steps -- blocks are the finer-grained objective and
    stages are strictly downstream, so there is no state in which the two
    could be traded against each other, which is precisely what a second
    objective was for.

    Applies only where every classification table shares one block count. The
    disjoint rows carry two models' differently-sized tables and are "not
    applicable", never "predicted wrong" -- skipped, and the surviving count
    is asserted so a fixture change cannot make this pass vacuously.

    Asserted against the PACKER, not against the formula: if the 2x12 column
    geometry ever changes, this fails loudly rather than a retired objective
    quietly becoming relevant again (§8).
    """
    from src.p4model.target import (TCAM_ROWS_PER_STAGE,
                                    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

    checked = 0
    for row in load_fixture()["rows"]:
        pool = rebuild_pool(row)
        specs = pool["ternary_table_specs"]
        factors = {blocks for blocks, _ in specs}
        if len(factors) != 1:
            continue                      # mixed factors: formula not applicable
        factor = factors.pop()
        if factor > TCAM_ROWS_PER_STAGE:
            continue                      # §10 Q3: a table wider than one column
        tables = len(specs)
        _usage, _range_plan, ternary_plan = assemble_usage(pool)
        per_stage = min(TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
                        2 * (TCAM_ROWS_PER_STAGE // factor))
        assert -(-tables // per_stage) == ternary_plan.occupied, row["row_id"]
        checked += 1

    # 8 joint rows + independent_low_sd5 + independent_low_sd7.
    assert checked == 10


@pytest.mark.parametrize("factor", range(1, 9))
def test_the_classification_stage_formula_honours_the_eight_table_cap(factor):
    """Audit §9.1, promoted from a one-off probe to a test.

    The stage formula has TWO independent per-stage limits, not one. The 2x12
    column geometry gives `2 * (12 // factor)` tables per stage; the ternary
    crossbar independently refuses more than TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
    of them whatever their width. The documented summary carried only the first,
    and could not be caught by the golden fixture: every archived joint row runs
    at factor >= 3, where `2 * (12 // factor) <= 8` already, so the cap never
    binds there and the existing assertion passes VACUOUSLY on the regime where
    the formula breaks (T=10 at factor 1 or 2: the packer needs 2 stages, the
    old formula says 1).

    Asserted against the PACKER over the whole T x factor grid the audit
    measured at zero mismatches, so a future change to either limit fails here
    rather than quietly making a retired objective relevant again.
    """
    from src.p4model.packing import crossbar_stages_needed
    from src.p4model.target import (TCAM_ROWS_PER_STAGE,
                                    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

    # The byte width is a placeholder: with key_fields=None every table gets a
    # private synthetic field, so nothing here depends on its value.
    for tables in range(1, 21):
        per_stage = min(TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
                        2 * (TCAM_ROWS_PER_STAGE // factor))
        plan = crossbar_stages_needed([(factor, 5)] * tables)
        assert plan.occupied == -(-tables // per_stage), (factor, tables)
