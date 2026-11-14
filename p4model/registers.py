"""The stateful register schedule: which pipeline stage each per-feature
Register<> runs in, and the readiness levels every match table is placed
against.

Two constraints drive the schedule:

  * A stage has 4 stateful ALUs (`METER_ALUS_PER_STAGE`), and every emitted
    RegisterAction holds one for a whole stage -- so 16-20 registers need
    ceil(n/4) stages regardless of how deep any single feature's dependency
    chain runs.
  * A register runs only after the register it waits on (its chain
    predecessor, or the orientation register for the first register of a
    direction-owned chain, whose body reads meta.fwd).

Since decisions 1a / 7.1 option B (2026-11-14) the generator emits every
RegisterAction unconditionally, in ONE sequence (a direction-owned register
tests meta.fwd inside its body), so the schedule is one list schedule over all
registers. The gated-block apparatus it replaced -- the block order, the
cursor floor between blocks and the "interior" stages no table could use
(inventory S6-S8, reviews/p4_tofino_reference.md Mechanism B) -- is gone with
the blocks; a program that still carries them is refused by the replay
(p4_replay.parse_program_text).

Every per-flow register is indexed by meta.flow_hash, so `FLOW_HASH_LEVEL`
(defined in program.py) is the earliest stage any RegisterAction can run --
three stages of metadata-init/hash-precompute/hash ahead of it, measured
across the real compiles this project has calibrated against."""
import collections

from p4model.catalog import FEATURE_REGISTER_CATALOG
from p4model.names import normalise_feature_name
from p4model.program import FLOW_HASH_LEVEL, ORIENTATION_REGISTER
from p4model.target import METER_ALUS_PER_STAGE


def feature_readiness_level(feature_name, catalog=None):
  """Earliest pipeline stage at which this feature's `_val` field -- and so
  its range-matching table -- can possibly be placed.

    level = FLOW_HASH_LEVEL
          + 1 if the feature is fwd- or bwd-owned (flow_orientation_action has
            to resolve meta.fwd before a register body that tests it can run,
            regardless of which direction it tests)
          + one per RegisterAction in the feature's chain

  Each chain entry is a genuinely sequential stage: a "dependency" register
  produces meta.current_iat, which the paired "value" register consumes.
  A register shared between two features (flow_last_arrival_time, executed
  once for both flow_iat_max and flow_iat_mean) still sits on both features'
  critical paths, so it counts for both.

  A feature absent from the catalog gets no registers emitted at all
  (generate_P4_registers_and_apply silently skips it), so nothing gates its
  table beyond the hash itself -- true of THIS estimator's own placement
  logic, which still happily prices such a feature set. Task 5's F2 check
  closes the gap one level up, at generate_P4_code: it raises a hard error
  for exactly this case (an uncatalogued feature would otherwise compile to
  a declared-but-never-written field), so a design this function readily
  estimates a stage/block count for may be one the generator now refuses to
  emit at all.

  Validated against a real compile: this yields 3/3/3/4 for M2's feature set,
  matching the compiler's observed stage offsets 0/0/0/1 exactly."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  entry = catalog.get(normalise_feature_name(feature_name))
  if entry is None:
    return FLOW_HASH_LEVEL

  gate_cost = 1 if entry.get("gated_by") in ("fwd", "bwd") else 0
  return FLOW_HASH_LEVEL + gate_cost + len(entry["registers"])


def _register_chains(features, catalog):
  """register name -> the register it waits on (None for none), every
  register the generator emits for `features` plus the orientation register.

  First writer wins, mirroring generate_P4_registers_and_apply's
  already_executed_registers: a register shared between two features is
  emitted once, at its first call site. That function emits the ungated
  features' registers first, then the fwd-owned, then the bwd-owned ones;
  which feature claims a shared register cannot depend on that order, because
  build_p4_script raises ValueError for a register named by features in two
  different gate classes, so the walk below is in `features` order."""
  predecessor = {ORIENTATION_REGISTER: None}
  for feature in features:
    entry = catalog.get(normalise_feature_name(feature))
    if entry is None:
      continue
    # The first register of a direction-owned chain waits on the orientation
    # register: its body tests meta.fwd. An unrecognised gated_by is treated
    # as ungated here rather than dropped -- build_p4_script raises
    # RuntimeError for it, and this function must not silently price such a
    # design as register-free.
    previous = (ORIENTATION_REGISTER if entry.get("gated_by") in ("fwd", "bwd")
                else None)
    for register in entry["registers"]:
      name = register["name"]
      if name not in predecessor:
        predecessor[name] = previous
      previous = name
  return predecessor


def register_stage_schedule(features, catalog=None,
                            alus_per_stage=METER_ALUS_PER_STAGE,
                            dependents=None):
  """Register base name -> the pipeline stage its RegisterAction runs in, for
  the whole set of features whose registers the generator will emit.

  ONE list schedule over every register (decisions 1a / 7.1 option B,
  2026-11-14: the generator emits no gated blocks, so there is no block order
  and no cursor floor between blocks any more). Stage by stage, a register is
  READY once its dependency level is reached (FLOW_HASH_LEVEL, +1 per chain
  step) and its predecessor ran in an EARLIER stage; the ready registers take
  the stage's `alus_per_stage` stateful ALUs in the order p4c's placer ranks
  two candidates for one stage (table_placement.cpp DecidePlacement::
  is_better, read in p4c's own placement log, 2026-11-08,
  results/encoding_evidence_2026_11/per_tree_model2 notes.md): the longer
  dependence chain first ("longer downward prop ... tail chain", the `height`
  below), then the one more trees depend on ("fewer total dependencies in
  dominance frontier"), then the one listed LATER in the program ("default
  choice"), then the name. `features` is in program order, and `dependents`
  maps a feature (as spelled in `features`) to the number of classification
  trees whose key reads that feature's code; a register's dependents are the
  sum over the features it serves. Without `dependents` every register ties
  there and the later-listed one goes first.

  feature_readiness_level models a register chain's DEPTH -- how many stages
  a single feature's registers must run back to back. This models the
  pipeline's register WIDTH, which binds at campaign scale: only
  `alus_per_stage` RegisterActions can issue per stage, so a design needing
  16-20 registers cannot run them in the two or three stages its dependency
  chains alone would allow. The compiler spills the surplus forward and every
  range table keyed on a spilled register's output slides with it.

  Registers are emitted once however many features share them
  (register_names_for dedupes by name), so a shared register is scheduled
  once and both its features read the same stage.

  Not modelled: the dependency registers of different chains all write the
  one scratch field meta.current_iat, which p4c orders (write after read) in
  program order. See results/encoding_evidence_2026_11/model8/impl/log.md
  (Task 2) for its measured effect on the probe 7.1 programs.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # Pass 1: every register the generator will emit and the register each one
  # waits on -- its predecessor inside its own feature's chain, or the
  # orientation register for a direction-owned feature's first.
  predecessor = _register_chains(features, catalog)

  # Pass 2: dependency level (what feature_readiness_level already charges).
  def level_of(name, seen=()):
    parent = predecessor.get(name)
    # `seen` guards a catalog that lists a register as its own ancestor;
    # generate_P4_registers_and_apply would emit unreachable P4 for that, but
    # this function must not hang while a caller is merely pricing it.
    if parent is None or name in seen:
      return FLOW_HASH_LEVEL
    return level_of(parent, seen + (name,)) + 1

  levels = {name: level_of(name) for name in predecessor}

  # Pass 3: how many stages of work still hang off each register. Standard
  # critical-path list scheduling: among registers ready in the same stage,
  # the one with the longest chain behind it goes first. It matters here for
  # one register in particular -- ORIENTATION_REGISTER carries every
  # direction-owned feature's whole chain, so delaying it to make room for
  # some leaf register would push that entire subtree back a stage.
  # A predecessor always has a strictly lower level than the register that
  # waits on it, so walking levels downward visits every child before its
  # parent and one pass suffices.
  height = {name: 0 for name in levels}
  for name in sorted(levels, key=levels.get, reverse=True):
    parent = predecessor.get(name)
    if parent is not None:
      height[parent] = max(height[parent], height[name] + 1)

  # Pass 4: the list schedule, one STAGE at a time (docstring head).
  order = list(dict.fromkeys(features))
  served = collections.defaultdict(list)
  for position, feature in enumerate(order):
    entry = catalog.get(normalise_feature_name(feature))
    for register in (entry or {}).get("registers", ()):
      served[register["name"]].append(position)
  dependents = dependents or {}

  def rank(name):
    positions = served.get(name, ())
    return (-height[name],
            -sum(dependents.get(order[p], 0) for p in positions),
            -max(positions, default=-1),
            name)

  placed, load = {}, collections.Counter()
  todo = set(levels)
  stage = min(levels.values())
  # A register becomes ready at most len(levels) stages after the first
  # unless its predecessor can never be placed (a catalog cycle, which
  # level_of already tolerates); fail loudly instead of spinning.
  limit = stage + len(levels) + max(levels.values()) + 1
  while todo:
    if stage > limit:
      raise ValueError("register_stage_schedule: registers never become "
                       "ready: %s" % sorted(todo))
    ready = [name for name in todo
             if levels[name] <= stage
             and (predecessor.get(name) is None
                  or placed.get(predecessor[name], stage) < stage)]
    for name in sorted(ready, key=rank)[:max(0, alus_per_stage - load[stage])]:
      placed[name] = stage
      load[stage] += 1
      todo.discard(name)
    stage += 1
  return placed


def readiness_levels_for(feature_intervals, catalog=None, emitted_features=None,
                         dependents=None):
  """One readiness level per feature, positionally aligned with
  range_matching_resource_usage's range_table_specs (both follow
  feature_intervals iteration order).

  A feature's level is one stage past its LAST register in
  register_stage_schedule -- so it carries both the chain depth
  feature_readiness_level charges and the stateful-ALU serialisation that
  function cannot see (it prices one feature at a time; the ALU cap is a
  property of the whole selected set).

  emitted_features names the features whose registers the generator will
  actually emit, when that is a SUPERSET of feature_intervals. Under
  'disjoint' encoding each model keeps its own intervals but both share one
  register sequence and one set of stateful ALUs, so each model's levels must be
  read off a schedule built from the union; pricing a model against a private
  pipeline would understate the pressure. Defaults to feature_intervals
  itself, which is already the merged set on the 'joint' branch.

  dependents: {feature: trees reading its code}, handed to
  register_stage_schedule (its tie order); None treats every register alike."""
  placed = register_stage_schedule(
      list(emitted_features if emitted_features is not None else feature_intervals),
      catalog, dependents=dependents)
  entries = FEATURE_REGISTER_CATALOG if catalog is None else catalog

  levels = []
  for feature in feature_intervals:
    entry = entries.get(normalise_feature_name(feature))
    if entry is None or not entry["registers"]:
      # No registers emitted at all, so nothing gates this feature's table
      # beyond the hash -- feature_readiness_level's own fallback.
      levels.append(feature_readiness_level(feature, catalog))
      continue
    levels.append(max(placed[r["name"]] for r in entry["registers"]) + 1)
  return levels
