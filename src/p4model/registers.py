"""The stateful register schedule: which pipeline stage each per-feature
Register<> runs in, and the readiness levels every match table is placed
against.

Three constraints drive the schedule, and none of them is a capacity limit:

  * A stage has 4 stateful ALUs (`METER_ALUS_PER_STAGE`), and every emitted
    RegisterAction holds one for a whole stage -- so 16-20 registers need
    ceil(n/4) stages regardless of how deep any single feature's dependency
    chain runs.
  * Registers are emitted unconditional -> `if (meta.fwd == 1)` ->
    `if (meta.fwd == 0)` (REGISTER_BLOCK_ORDER), and p4c's placer has one
    work-list cursor, so each block is floored at the last stage the
    previous one used. A bwd-gated feature therefore prices one stage later
    than the symmetric fwd-gated one.
  * A stage the cursor spends fully INTERIOR to a gated block can hold no
    table from the outer sequence (`gated_block_interior_stages`): the
    cursor is not "inside" anything while placing the outer sequence's
    unconditional tables, but every stage strictly between where it enters
    and leaves a gated block is unreachable regardless of how empty it is.
    There is no range-pool fill limit; do not look for one.

Every per-flow register is indexed by meta.flow_hash, so `FLOW_HASH_LEVEL`
(defined in program.py) is the earliest stage any RegisterAction can run --
three stages of metadata-init/hash-precompute/hash ahead of it, measured
across the real compiles this project has calibrated against."""
import collections

from src.p4model.catalog import FEATURE_REGISTER_CATALOG
from src.p4model.names import normalise_feature_name
from src.p4model.program import FLOW_HASH_LEVEL, ORIENTATION_REGISTER, REGISTER_BLOCK_ORDER
from src.p4model.target import METER_ALUS_PER_STAGE


def feature_readiness_level(feature_name, catalog=None):
  """Earliest pipeline stage at which this feature's `_val` field -- and so
  its range-matching table -- can possibly be placed.

    level = FLOW_HASH_LEVEL
          + 1 if the feature is fwd- or bwd-gated (flow_orientation_action has
            to resolve meta.fwd before the gated block can run, regardless of
            which direction the gate checks)
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


def _register_blocks(features, catalog):
  """(register name -> which of REGISTER_BLOCK_ORDER's blocks emits it,
  register name -> the register it waits on).

  Walks the features once per block, which is what
  generate_P4_registers_and_apply does -- it filters matched_features into
  ungated/fwd-gated/bwd-gated lists and calls _execute_lines on each in turn,
  sharing one already_executed_registers set. Grouping cannot change which
  block claims a shared register: build_p4_script raises ValueError outright
  for a register named by features in two different gate classes ("it would
  be .execute()d inside one gated block and read as garbage from the other"),
  so every register has exactly one well-defined block."""
  gate = {ORIENTATION_REGISTER: None}
  predecessor = {ORIENTATION_REGISTER: None}
  for block in REGISTER_BLOCK_ORDER:
    for feature in features:
      entry = catalog.get(normalise_feature_name(feature))
      if entry is None:
        continue
      # An unrecognised gated_by is treated as ungated here rather than
      # dropped -- build_p4_script raises RuntimeError for it, and this
      # function must not silently price such a design as register-free.
      gated_by = entry.get("gated_by")
      if (gated_by if gated_by in ("fwd", "bwd") else None) != block:
        continue
      # First register of a gated chain waits on the orientation register:
      # meta.fwd has to resolve before the block it guards can run.
      previous = ORIENTATION_REGISTER if block in ("fwd", "bwd") else None
      for register in entry["registers"]:
        name = register["name"]
        # First writer wins, mirroring already_executed_registers: a register
        # shared between two features is emitted at its first call site.
        if name not in gate:
          gate[name] = block
          predecessor[name] = previous
        previous = name
  return gate, predecessor


def register_stage_schedule(features, catalog=None,
                            alus_per_stage=METER_ALUS_PER_STAGE):
  """Register base name -> the pipeline stage its RegisterAction runs in, for
  the whole set of features whose registers the generator will emit.

  feature_readiness_level models a register chain's DEPTH -- how many stages
  a single feature's registers must run back to back. This models the
  pipeline's register WIDTH, which is the constraint that actually binds at
  campaign scale: only `alus_per_stage` RegisterActions can issue per stage,
  so a design needing 16-20 registers cannot run them in the two or three
  stages its dependency chains alone would allow. The compiler spills the
  surplus forward and every range table keyed on a spilled register's output
  slides with it.

  Registers are emitted once however many features share them
  (register_names_for dedupes by name, matching
  generate_P4_registers_and_apply's `already_executed_registers` set), so a
  shared register is scheduled once and both its features read the same
  stage.

  A greedy earliest-level-first list schedule, run once per _REGISTER_BLOCK_
  ORDER block with each block floored at the last stage the previous one
  used. That floor is the control-flow half of the story: p4c's placer holds
  a single work-list cursor, so it only reaches `if (meta.fwd == 0)` once
  `if (meta.fwd == 1)` is fully placed, and a bwd register cannot run in a
  stage the fwd block has already moved past however many ALUs sit idle
  there. Inside one block the placer IS free -- every register in it is a
  candidate at once -- so the greedy runs unconstrained there.

  Ties inside a level do move individual registers around, but never the
  makespan -- the only quantity readiness levels take from this: verified
  over 300 random feature-order shuffles of every calibration row, where the
  last-register stage never moved. Measured against the 18 committed
  placements, the makespan matches the compiler's own on EVERY row
  (7,7,7,7,4,5,4,4,4,4,7,7,7,4,4,4,4,4), where chain depth alone reports 5 on
  each of the six k>=13 rows. Adding the per-block floor left all 18 of those
  intact while raising per-register agreement from 88 to 148 of 170, and it
  is what makes gated_block_interior_stages exact -- see
  reviews/p4_tofino_reference.md Sec 7.
  """
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  # Pass 1: collect every register the generator will emit, which block emits
  # it, and the register each one waits on -- its predecessor inside its own
  # feature's chain, or the orientation register for a gated feature's first.
  gate, predecessor = _register_blocks(features, catalog)

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
  # one register in particular -- ORIENTATION_REGISTER carries every gated
  # feature's whole chain, so delaying it to make room for some leaf register
  # would push that entire subtree back a stage. The compiler agrees: it sits
  # in the first register stage in all 18 committed placements.
  # A predecessor always has a strictly lower level than the register that
  # waits on it, so walking levels downward visits every child before its
  # parent and one pass suffices.
  height = {name: 0 for name in levels}
  for name in sorted(levels, key=levels.get, reverse=True):
    parent = predecessor.get(name)
    if parent is not None:
      height[parent] = max(height[parent], height[name] + 1)

  # Pass 4: list schedule, one block at a time in emission order. Within a
  # block, sorted by level so a predecessor is always placed before anything
  # that waits on it; then by height, then by name so the result never
  # depends on feature iteration order. `floor` carries the cursor forward:
  # no register of a later block may precede the last stage an earlier one
  # used. ALU load is shared across blocks -- there is one set of four
  # stateful ALUs per stage, whichever block's register claims it.
  placed, load, floor = {}, collections.Counter(), 0
  for block in REGISTER_BLOCK_ORDER:
    names = [name for name, owner in gate.items() if owner == block]
    if not names:
      continue
    for name in sorted(names, key=lambda n: (levels[n], -height[n], n)):
      stage = max(levels[name], floor)
      parent = predecessor.get(name)
      if parent is not None:
        stage = max(stage, placed[parent] + 1)
      while load[stage] >= alus_per_stage:
        stage += 1
      placed[name] = stage
      load[stage] += 1
    floor = max(placed[name] for name in names)
  return placed


def gated_block_interior_stages(features, catalog=None,
                                alus_per_stage=METER_ALUS_PER_STAGE):
  """The stage indices no match table can be placed in, because the placer
  spends the whole of each of them inside a gated register block.

  Tofino has no program counter: every table hands the next stage a
  next-table pointer, so p4c's placer walks the control block with a work-
  list cursor and a table is a placement candidate only once the cursor
  reaches it. generate_P4_registers_and_apply emits the gated register blocks
  BEFORE every match table, so while the cursor is inside `if (meta.fwd ==
  1) { ... }` the entire range and classification pools are out of reach --
  regardless of how empty the stage is. In independent_high_sd10's stage 5
  the TCAM is 0/24, the ternary crossbar 0/66 and the logical table IDs 4/16.
  This is a control-flow constraint, not a capacity one; there is no
  range-pool fill limit.

  A block costs a stage only where the cursor is fully INTERIOR to it. In the
  stage the cursor descends into the block, outer tables can still be
  back-filled (p4c logs them verbatim as "potential backfill ... before
  tbl_prog951"), and in the stage it pops back out they are candidates
  again -- so a block spanning stages [first, last] blocks exactly
  range(first + 1, last), and a two-stage block costs nothing at all.

  Measured against all 18 calibration rows: the range pool's committed
  occupancy has a hole on exactly 5 of them, and this returns precisely those
  holes -- right rows, right indices, nothing on the other 13. Two of the
  five (independent_high_sd6, joint_high_sd8) still cost +0 stages overall
  because their range tables were not going to occupy that stage anyway,
  which is why the caller must apply this as a placement constraint and never
  as a per-row penalty. See reviews/p4_tofino_reference.md Sec 7,
  "Mechanism B"."""
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG

  gate, _ = _register_blocks(features, catalog)
  placed = register_stage_schedule(features, catalog, alus_per_stage)

  interior = set()
  for block in REGISTER_BLOCK_ORDER:
    # The unconditional registers sit in the OUTER sequence; the cursor is
    # never "inside" anything while placing them, so they hide nothing.
    if block is None:
      continue
    stages = [placed[name] for name, owner in gate.items() if owner == block]
    if len(stages) > 1:
      interior.update(range(min(stages) + 1, max(stages)))
  return frozenset(interior)


def readiness_levels_for(feature_intervals, catalog=None, emitted_features=None):
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
  register block and one set of stateful ALUs, so each model's levels must be
  read off a schedule built from the union; pricing a model against a private
  pipeline would understate the pressure. Defaults to feature_intervals
  itself, which is already the merged set on the 'joint' branch."""
  placed = register_stage_schedule(
      list(emitted_features if emitted_features is not None else feature_intervals),
      catalog)
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
