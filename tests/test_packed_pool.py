"""The training path prices the packed layout (per_tree, spec
2026-11-11-packed-codes-stage-pin Sec 3.2, 4.3, 5): each tree's key bytes come from
packed.layout_for, its table is charged its stand-alone price, and the tables the
12-group rule moved are the ones the generator pins."""
from p4gen import evaluation
from p4model import lanes, packing
from p4model.program import (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)
from p4model.usage import assemble_usage, moved_table_stages

from tests.test_per_task_codes import _rng_design


def test_the_per_tree_pool_carries_each_trees_key_bytes_and_stand_alone_blocks():
  app, ddos, names = _rng_design(3)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  assert pool["ternary_key_bits"] is None
  assert (len(pool["ternary_key_bytes"]) == len(pool["ternary_table_specs"])
          == len(app.estimators_) + len(ddos.estimators_))
  for key, (blocks, width) in zip(pool["ternary_key_bytes"], pool["ternary_table_specs"]):
    assert width == len(key) and blocks % lanes.standalone(list(key)) == 0


def test_usage_charges_the_sum_of_the_tables_own_blocks():
  app, ddos, names = _rng_design(3)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  usage, _, ternary_plan = assemble_usage(pool)
  assert ternary_plan.blocks == sum(blocks for blocks, _ in pool["ternary_table_specs"])


def test_stage_pins_name_exactly_the_moved_tables(monkeypatch):
  # one group per stage: seed 3 has two one-group trees, which can no longer share
  monkeypatch.setattr(packing, "TERNARY_CROSSBAR_GROUPS_PER_STAGE", 1)
  app, ddos, names = _rng_design(3)
  pins = evaluation.tree_stage_pins(app, ddos, names, names, PER_TREE_ENCODING)
  assert pins and all(name.startswith("get_classification_tree_") for name in pins)
  assert evaluation.tree_stage_pins(app, ddos, names, names, PER_TASK_ENCODING) == {}
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  _, _, plan = assemble_usage(pool)
  moved = moved_table_stages(plan)
  assert sorted(pins.values()) == sorted(moved.values())
  assert len(pins) == len(moved) and set(moved) == set(plan.moved)


def test_the_range_pool_prices_the_packed_fields_bus_bytes():
  app, ddos, names = _rng_design(3)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  assert len(pool["range_action_bytes"]) == len(pool["range_table_specs"])
  assert all(b >= 1 for b in pool["range_action_bytes"])


def test_the_other_encodings_pools_are_untouched():
  app, ddos, names = _rng_design(3)
  for encoding in (JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING):
    pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
    assert "ternary_key_bytes" not in pool and pool["ternary_key_bits"] is not None
