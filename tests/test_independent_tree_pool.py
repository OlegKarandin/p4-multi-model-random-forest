"""The training pool of the disjoint_per_tree encoding (arm independent-tree; spec
2026-11-17-independent-tree-encoding-design.md, Part C row 9, K1-K3): training prices
exactly what the replay of the generated program prices -- stage depth, blocks, every
table's blocks and stage, and the PHV containers per size -- on designs whose tasks
select DIFFERENT features (so the resolved-name order matters), on a design mixing a
merged `f` entry with namespaced app_/ddos_ entries, on one whose tasks share no
feature, on the merge design and on one with no ddos trees. The four old encodings'
pools are pinned by tests/fixtures/independent_tree_identity.json (hashed before this
branch existed)."""
import hashlib
import json
import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen import evaluation
from p4gen.p4_replay import model_breakdown, parse_program_text
from p4model import phv
from p4model.errors import PhvContainersExhausted
from p4model.program import APP_TASK, DDOS_TASK, DISJOINT_PER_TREE_ENCODING, TaskSpec
from p4model.target import PHV_INGRESS_CONTAINERS
from p4model.usage import assemble_usage
from tests.test_independent_tree_program import (APP_NAMES, DDOS_NAMES, IDENTITY, OLD_ENCODINGS,
                                                 _merge_design, _overlap_design)
from tests.test_per_task_codes import FEATURES, _forest, _own, _rng_design, _stair_forest

NEW = DISJOINT_PER_TREE_ENCODING


# --- designs: (app forest, ddos forest, app names, ddos names) -----------------------

def _split_design(seed):
  """Two random forests over DIFFERENT, overlapping feature lists (each at least one
  feature of its own is possible, one feature always common)."""
  rng = np.random.RandomState(1000 + seed)
  depth_app, depth_ddos = rng.randint(2, 9, size=2)
  trees_app, trees_ddos = rng.randint(1, 6, size=2)
  common = str(rng.choice(FEATURES))
  rest = [f for f in FEATURES if f != common]
  app = [common] + list(rng.choice(rest, size=rng.randint(0, 3), replace=False))
  ddos = [common] + list(rng.choice(rest, size=rng.randint(1, 3), replace=False))
  if sorted(app) == sorted(ddos):
    ddos = ddos[:-1] if len(ddos) > 1 else ddos + [f for f in rest if f not in app][:1]
  return (_forest(3, 2 + seed, app, n_trees=trees_app, depth=depth_app),
          _forest(2, 5 + seed, ddos, n_trees=trees_ddos, depth=depth_ddos), app, ddos)


class _Trees:
  """A forest view (estimators_) over trees fitted separately."""
  def __init__(self, trees):
    self.estimators_ = list(trees)


def _column_forest(seed, n_classes, n_columns, columns, n_trees=2, depth=4):
  """A forest over n_columns features that splits only `columns` (the others constant)."""
  rng = np.random.RandomState(seed)
  X = np.zeros((400, n_columns), dtype=int)
  X[:, columns] = rng.randint(0, 2000, size=(400, len(columns)))
  y = (X[:, columns[0]] // (2001 // n_classes + 1)).clip(0, n_classes - 1)
  return bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=n_trees, max_depth=depth, random_state=seed).fit(X, y))


def _mixed_design():
  """Feature A both tasks split identically (one merged entry `A`), B both split
  differently (app_B, ddos_B), C app only, D ddos only: a merged entry and namespaced
  entries in one program (the Task 2 reviewer's gap)."""
  a, b, c, d = "flow_iat_max", "fwd_packet_length_max", "bwd_iat_total", "flow_iat_min"
  app_stair = _stair_forest(3, 3, feature_count=3).estimators_
  ddos_stair = _stair_forest(3, 2, feature_count=3).estimators_
  app = _Trees(list(app_stair) + list(_column_forest(7, 3, 3, [1, 2]).estimators_))
  ddos = _Trees(list(ddos_stair) + list(_column_forest(11, 2, 3, [1, 2]).estimators_))
  return app, ddos, [a, b, c], [a, b, d]


def _no_shared_feature_design():
  return (_forest(3, 4, FEATURES[:2], n_trees=3, depth=4),
          _forest(2, 6, FEATURES[2:], n_trees=2, depth=5), FEATURES[:2], FEATURES[2:])


def _from_specs(specs):
  ddos, app = specs
  return (_Trees(app.trees), _Trees(ddos.trees), app.features,
          ddos.features if ddos.features is not None else [])


def _no_ddos_trees_design():
  return (_forest(3, 4, APP_NAMES), _Trees([]), APP_NAMES, [])


NAMED = {"overlap": lambda: _from_specs(_overlap_design()),
         "merge": lambda: _from_specs(_merge_design()),
         "mixed": _mixed_design,
         "no_shared_feature": _no_shared_feature_design,
         "no_ddos_trees": _no_ddos_trees_design}
DESIGNS = [("seed_%d" % seed, (lambda seed=seed: _split_design(seed))) for seed in range(12)]
DESIGNS += sorted(NAMED.items())


def _specs(app, ddos, names_app, names_ddos):
  return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), names_ddos or None,
                   _own(ddos, names_ddos) if ddos.estimators_ else {}),
          TaskSpec(APP_TASK, 3, list(app.estimators_), names_app, _own(app, names_app))]


def _text(tmp_path, design, name="design"):
  out = os.path.join(str(tmp_path), name) + os.sep
  with open(bps.generate_p4_program(_specs(*design), output_dir=out, encoding=NEW),
            encoding="utf-8") as handle:
    return handle.read()


def _pool(design):
  app, ddos, names_app, names_ddos = design
  return evaluation._pool_inputs(app, ddos, names_app, names_ddos, NEW)


def _resolved(design):
  return bps.resolve_feature_plan([(task.name, task.intervals) for task in _specs(*design)])


# --- the designs exercise what they claim ----------------------------------------------

def test_the_designs_cover_differing_lists_a_mixed_plan_and_no_shared_feature():
  differing = sum(sorted(_split_design(s)[2]) != sorted(_split_design(s)[3]) for s in range(12))
  assert differing == 12
  mixed = {name: sorted(readers) for name, (_, _, readers) in _resolved(_mixed_design()).items()}
  assert mixed == {"flow_iat_max": [APP_TASK, DDOS_TASK],
                   "app_fwd_packet_length_max": [APP_TASK], "bwd_iat_total": [APP_TASK],
                   "ddos_fwd_packet_length_max": [DDOS_TASK], "flow_iat_min": [DDOS_TASK]}
  apart = _resolved(_no_shared_feature_design())
  assert all(len(readers) == 1 and name == raw for name, (raw, _, readers) in apart.items())
  # the seeds reach namespaced entries and single-task ones, and a ddos-only entry
  # sorting between two namespaced ddos ones (seed 5: the order is not "namespaced last")
  kinds = set()
  for seed in range(12):
    for name, (raw, _, readers) in _resolved(_split_design(seed)).items():
      kinds.add("namespaced" if name != raw else "shared" if len(readers) == 2 else "single")
  assert {"namespaced", "single"} <= kinds
  assert list(_resolved(_split_design(5))) == [
      "app_flow_iat_max", "bwd_packet_length_max", "ddos_flow_iat_max", "fwd_packet_length_max"]


# --- pool == replay ----------------------------------------------------------------------

def _pool_tables(pool, design):
  """{table name: [blocks, stage]} of training's placement, by the program's names."""
  _, range_plan, ternary_plan = assemble_usage(pool)
  ranges = ["table_%d_%s" % (i, name) for i, name in enumerate(_resolved(design))]
  app, ddos = design[0], design[1]
  trees = (["get_classification_tree_%s_%d" % (APP_TASK, i) for i in range(len(app.estimators_))]
           + ["get_classification_tree_%s_%d" % (DDOS_TASK, i)
              for i in range(len(ddos.estimators_))])
  out = {}
  for names, plan in ((ranges, range_plan), (trees, ternary_plan)):
    assert len(names) == len(plan.table_blocks)
    for name, blocks, stage in zip(names, plan.table_blocks, plan.table_stages):
      out[name] = [int(blocks), int(stage)]
  return out


@pytest.mark.parametrize("name, make", DESIGNS, ids=[name for name, _ in DESIGNS])
def test_training_prices_exactly_what_the_generated_program_replays(tmp_path, name, make):
  design = make()
  app, ddos, names_app, names_ddos = design
  usage = evaluation.multi_model_memory_evaluation(app, ddos, names_app, names_ddos, NEW)
  pool = _pool(design)
  program = parse_program_text(_text(tmp_path, design))
  replay = model_breakdown(program)
  assert replay["refusal"] is None, replay
  assert (usage.stage_depth, usage.blocks) == (replay["stage_depth"], replay["blocks"])
  assert _pool_tables(pool, design) == {t["table"]: [t["blocks"], t["stage"]]
                                        for t in replay["tables"]}
  assert phv.containers_used(pool) == phv.containers_used(program)


# --- K2: the two tasks' fields of one raw feature are distinct crossbar identities ---------

@pytest.mark.parametrize("make", [lambda: _from_specs(_overlap_design()), _mixed_design])
def test_two_tasks_on_one_raw_feature_get_distinct_crossbar_identities(make):
  design = make()
  # Disjointness holds by construction (identities include the resolved name); the real
  # guards are the pool == replay test and the resolved[field] / reader-task check below.
  pool = _pool(design)
  resolved = _resolved(design)
  n_app = len(design[0].estimators_)
  raw_of = {name: raw for name, (raw, _, _) in resolved.items()}
  seen = {APP_TASK: {}, DDOS_TASK: {}}          # raw feature -> container identities
  for index, key in enumerate(pool["ternary_key_bytes"]):
    task = APP_TASK if index < n_app else DDOS_TASK
    for _, _, _, (field, container) in key:
      assert task in resolved[field][2]          # a tree keys only its task's entries
      seen[task].setdefault(raw_of[field], set()).add((field, container))
  namespaced = [raw for name, (raw, _, readers) in resolved.items() if len(readers) == 1
                and name != raw]
  assert namespaced
  for raw in set(namespaced):
    assert seen[APP_TASK][raw] and seen[DDOS_TASK][raw]
    assert not seen[APP_TASK][raw] & seen[DDOS_TASK][raw], raw


# --- K1 + refusal: PHV counted per RAW `_val`, refused by training and the generator ---------

@pytest.mark.parametrize("make", [lambda: _from_specs(_overlap_design()), _mixed_design])
def test_phv_refusal_agrees_with_the_generator(tmp_path, monkeypatch, make):
  design = make()
  program = parse_program_text(_text(tmp_path, design, "fits"))
  used = phv.containers_used(program)
  assert used == phv.containers_used(_pool(design))
  # exactly the budget fits on both paths: one `_val` per raw feature, never per field (K1)
  monkeypatch.setattr(phv, "PHV_INGRESS_CONTAINERS", dict(used))
  _pool(design)
  _text(tmp_path, design, "exact")
  for size in (16, 8):
    tight = dict(PHV_INGRESS_CONTAINERS)
    tight[size] = used[size] - 1
    monkeypatch.setattr(phv, "PHV_INGRESS_CONTAINERS", tight)
    with pytest.raises(PhvContainersExhausted):
      _pool(design)
    with pytest.raises(PhvContainersExhausted):
      _text(tmp_path, design, "short_%d" % size)


# --- the four old encodings' pools are untouched ----------------------------------------------

def _canonical(value):
  if isinstance(value, dict):
    return {str(k): _canonical(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_canonical(v) for v in value]
  if isinstance(value, (set, frozenset)):
    return sorted(repr(_canonical(v)) for v in value)
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  return repr(value)


def pool_digest(encoding, seed):
  app, ddos, names = _rng_design(seed)
  pool = evaluation._pool_inputs(app, ddos, names, names, encoding)
  return hashlib.sha256(json.dumps(_canonical(pool), sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("encoding", OLD_ENCODINGS)
def test_other_encodings_pools_unchanged(encoding, seed):
  with open(IDENTITY, encoding="utf-8") as handle:
    pinned = json.load(handle)
  assert pool_digest(encoding, seed) == pinned["pools"][encoding][str(seed)]


def write_pool_fixture():
  """Add the four old encodings' pool digests on seeds 0-5 to the identity fixture
  (run once, before the new encoding's pool branch existed)."""
  with open(IDENTITY, encoding="utf-8") as handle:
    pinned = json.load(handle)
  assert "pools" not in pinned
  pinned["pools"] = {encoding: {str(seed): pool_digest(encoding, seed) for seed in range(6)}
                     for encoding in OLD_ENCODINGS}
  with open(IDENTITY, "w", encoding="utf-8") as handle:
    json.dump(pinned, handle, indent=1, sort_keys=True)
    handle.write("\n")
