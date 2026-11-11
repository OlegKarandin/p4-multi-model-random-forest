"""The training path prices the packed layout (per_tree, spec
2026-11-11-packed-codes-stage-pin Sec 3.2, 4.3, 5): each tree's key bytes come from
packed.layout_for, its table is charged its stand-alone price, and the tables the
12-group rule moved are the ones the generator pins."""
from p4gen import evaluation
import pytest

from p4model import lanes, packed, packing, tables
from p4model.errors import CrossbarKeyTooWide
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)
from p4model.usage import assemble_usage, moved_table_stages

from tests.test_per_task_codes import _rng_design


def _features(app, ddos, names):
  merged = evaluation.merge_tree_nodes(evaluation.tree_nodes_for(app, names),
                                       evaluation.tree_nodes_for(ddos, names))
  return list(evaluation.feature_intervals_from_nodes(merged))


def _layout(app, ddos, names):
  own = {task: [{f: i for f, i in evaluation.feature_intervals_from_nodes({0: nodes}).items()
                 if len(i) >= 2}
                for nodes in evaluation.tree_nodes_for(clf, names).values()]
         for task, clf in ((APP_TASK, app), (DDOS_TASK, ddos))}
  return packed.layout_for(packed.widths_of(own), _features(app, ddos, names),
                           [APP_TASK, DDOS_TASK])


def test_the_per_tree_pool_carries_each_trees_key_bytes_and_stand_alone_blocks():
  app, ddos, names = _rng_design(3)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  assert pool["ternary_key_bits"] is None
  assert (len(pool["ternary_key_bytes"]) == len(pool["ternary_table_specs"])
          == len(app.estimators_) + len(ddos.estimators_))
  for key, (blocks, width) in zip(pool["ternary_key_bytes"], pool["ternary_table_specs"]):
    assert width == len(key) and blocks % lanes.standalone(list(key)) == 0
  # blocks == stand-alone price * row blocks of the tree's entries
  entries = [evaluation.ternary_matching_resource_usage(
                 evaluation.generate_codewords(evaluation.get_root_to_leaf_paths({0: nodes}), iv), iv)[0]
             for clf, selected in ((app, names), (ddos, names))
             for nodes in evaluation.tree_nodes_for(clf, selected).values()
             for iv in [{f: i for f, i in evaluation.feature_intervals_from_nodes({0: nodes}).items()
                         if len(i) >= 2}]]
  for key, (blocks, _), count in zip(pool["ternary_key_bytes"], pool["ternary_table_specs"], entries):
    assert blocks == lanes.standalone(list(key)) * tables.tree_entries_to_blocks(count)


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
  seen = {}
  for idx, task in enumerate(pool["ternary_task"]):
    ordinal = seen[task] = seen.get(task, -1) + 1
    if idx in moved:
      assert pins["get_classification_tree_%s_%d" % (task, ordinal)] == moved[idx]
  # tree_stage_placement: the same pins, each pinned table's first refusal (one stage or more
  # before its pin), and no unpinnable table in this design
  placement = evaluation.tree_stage_placement(app, ddos, names, names, PER_TREE_ENCODING)
  assert placement.pins == pins and placement.unpinnable == []
  assert set(placement.first_refusal) == set(pins)
  assert all(placement.first_refusal[name] < pins[name] for name in pins)
  assert sorted(placement.first_refusal.values()) == sorted(s for _, s in plan.first_refused)
  assert evaluation.tree_stage_placement(app, ddos, names, names, PER_TASK_ENCODING) == ({}, {}, [])


def test_tree_stage_placement_names_the_unpinnable_tables(monkeypatch):
  # Construction: the real plan with index 0 (the first app tree) reported unpinnable, as a table
  # moved and placed as several units would be (tests/test_packed_placement.py builds one).
  import dataclasses
  real = evaluation.assemble_usage

  def with_unpinnable(pool):
    usage, range_plan, ternary_plan = real(pool)
    return usage, range_plan, dataclasses.replace(ternary_plan, unpinnable=(0,))

  monkeypatch.setattr(evaluation, "assemble_usage", with_unpinnable)
  app, ddos, names = _rng_design(3)
  placement = evaluation.tree_stage_placement(app, ddos, names, names, PER_TREE_ENCODING)
  assert placement.unpinnable == ["get_classification_tree_app_0"]
  assert "get_classification_tree_app_0" not in placement.pins


def test_a_key_the_old_unpacked_layout_cannot_place_is_priced_not_refused(monkeypatch):
  # Construction: make the OLD pricing (lanes.table_blocks of the unpacked key, which
  # p4model.tables.ternary_matching_resource_usage calls) raise CrossbarKeyTooWide, as
  # it does for an unpacked layout with no lane fit; the packed key is untouched.
  def refuse(_bits):
    raise CrossbarKeyTooWide("no lane fit in the old layout", 1)
  monkeypatch.setattr(lanes, "table_blocks", refuse)
  app, ddos, names = _rng_design(3)
  iv = evaluation.feature_intervals_from_nodes(
      {0: next(iter(evaluation.tree_nodes_for(app, names).values()))})
  with pytest.raises(CrossbarKeyTooWide):
    evaluation.ternary_matching_resource_usage(
        evaluation.generate_codewords(evaluation.get_root_to_leaf_paths(
            {0: next(iter(evaluation.tree_nodes_for(app, names).values()))}), iv), iv)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  assert len(pool["ternary_table_specs"]) == len(app.estimators_) + len(ddos.estimators_)


def test_the_range_pool_prices_the_packed_fields_bus_bytes():
  app, ddos, names = _rng_design(3)
  pool = evaluation._pool_inputs(app, ddos, names, names, PER_TREE_ENCODING)
  layout = _layout(app, ddos, names)
  assert pool["range_action_bytes"] == [packed.bus_bytes(layout, feature)
                                        for feature in _features(app, ddos, names)]


def test_the_other_encodings_pools_are_untouched():
  app, ddos, names = _rng_design(3)
  for encoding in (JOINT_ENCODING, PER_TASK_ENCODING, DISJOINT_ENCODING):
    pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
    assert "ternary_key_bytes" not in pool and pool["ternary_key_bits"] is not None
