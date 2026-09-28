"""ResourceUsage and assemble_usage: multi_model_memory_evaluation's return
value, and the import-light half of the function that builds it.

ResourceUsage depends on nothing else in p4model -- it is a plain data
container, moved here so callers that only need the shape of a result (not
the physics that produces one) can import it without pulling in the rest of
the model."""
from dataclasses import dataclass

from src.p4model.packing import crossbar_stages_needed, stage_load_fits
from src.p4model.program import (
    FLOW_HASH_LEVEL,
    PLACEMENT_PRIORITY,
    SHARED_TASK,
    TASKS,
    VOTE_EPILOGUE_STAGES,
)


@dataclass(frozen=True)
class ResourceUsage:
  """multi_model_memory_evaluation's return (D1).

  Replaces a 3-tuple rather than growing it to eight positional values, for
  the same reason TrainResult replaced a 7-tuple: three of these fields are
  same-typed ints naming DIFFERENT stage quantities, and positional access
  is how they get confused. See this module's
  multi_model_memory_evaluation docstring for the four-quantity
  disambiguation.

  No __int__ is provided. packing.StagePlan has one as a transitional shim;
  repeating it here would let a caller silently use the whole object where
  a count is meant.

  codeword_length : classification-table key width in bits. Under
                  'joint' this is THE pooled split-threshold count of the
                  merged tree set -- the quantity threshold alignment
                  actually shrinks, and the one src/training/align_budget.py
                  gates on. Under 'disjoint' the two models keep independent
                  codewords and no single value is "the" codeword: the MAX of
                  the two is reported, because that is the one the 512-bit
                  MAX_CODEWORD_LENGTH limit binds on. Do NOT derive disjoint
                  block counts from it -- each model's own factor is applied
                  to its own trees (see the disjoint branch below).

  register_depth and register_count (Task 6, Spec 4.1/4.2/4.3) report the
  Tofino `Register<>` state this design needs, on top of the match-table
  quantities above:

    register_depth     : max readiness level (feature_readiness_level) over
                          the selected feature(s) -- how many pipeline
                          stages elapse before the LAST register in any
                          feature's dependency chain has run, i.e. before
                          ANY classification table is even allowed to read
                          a value. Reuses whatever readiness-level list the
                          stage-placement code already computed; no new
                          traversal.
    register_count      : count of distinct Register<> instances
                          (register_names_for) the selected feature(s)
                          resolve to, deduplicated by name -- shared
                          dependency registers (e.g. flow_last_arrival_time,
                          reused by flow_iat_max and flow_iat_mean) are
                          counted once, matching
                          generate_P4_registers_and_apply's own by-name
                          dedup, and deduplicated ACROSS both models too:
                          registers are a single physical resource shared
                          by the whole generated program regardless of
                          'joint' vs 'disjoint' ENCODING (that choice only
                          affects codeword/interval sharing, never register
                          generation -- see build_p4_script.py's
                          raw_feature_intervals, keyed on the union of both
                          models' raw feature names).
  CAVEAT (Spec 4.3, applies to both fields above): register_count is a
  distinct-instance count, not a placement fact. register_depth DOES account
  for the stateful-ALU issue limit: registers.register_stage_schedule models
  Tofino's METER_ALUS_PER_STAGE = 4 cap and spills the surplus RegisterActions
  forward when a design needs more than that many in one stage, and
  readiness_levels_for reads register_depth off that schedule. What is NOT
  modelled here is per-stage register MEMORY (how much state one Register<>
  array actually occupies) -- do not read register_count as a memory-capacity
  guarantee.

  range_depth  : packing.StagePlan.depth for the range-matching pool ALONE
                (Task 6 extended by the stage-depth-attribution design's
                Phase 1). Component of stage_depth, not itself a
                ceiling-checked quantity -- only stage_depth is.
  ternary_depth: packing.StagePlan.depth for the ternary classification
                pool ALONE. Equal to stage_depth in every non-degenerate
                design (the classification pool is placed last, see
                stage_depth's own docstring); the max() stage_depth takes
                exists only for the degenerate zero-ternary-table case.
  range_tables : count of range-matching tables (len(range_table_specs)) --
                one per selected feature under 'joint' (the union of both
                models' features), one per (model, feature) pair under
                'disjoint' -- so on identical feature lists, disjoint's
                range_tables is at most twice joint's, with equality only
                when both models actually split on every listed feature.
  ternary_tables: count of classification tables (len(ternary_table_specs))
                -- one per tree in the merged tree set. Invariant across
                encoding for the same two forests (every tree gets its own
                table regardless of whether the codeword is shared).
  """
  stages: int           # occupied match-table stage COUNT
  blocks: int
  stage_depth: int      # pipeline DEPTH (packing.StagePlan.depth), what the 12-stage ceiling reads
  range_entries: int
  ternary_entries: int
  codeword_length: int  # pooled split-threshold count under 'joint'; see docstring
  register_depth: int   # max readiness level over the selected feature(s); see class docstring
  register_count: int   # distinct Register<> instances, deduplicated by name across both models
  range_depth: int       # packing.StagePlan.depth, range pool alone; see class docstring
  ternary_depth: int     # packing.StagePlan.depth, ternary pool alone; see class docstring
  range_tables: int      # len(range_table_specs); see class docstring
  ternary_tables: int    # len(ternary_table_specs); see class docstring


def tree_readiness_levels(range_table_stages, range_task, ternary_task):
  """The earliest stage each classification tree may occupy (audit C1).

  A tree keys on its own task's meta.code_<feature> fields, and a code field
  exists one stage after the range table that writes it. So a tree of task t
  is ready at 1 + the latest stage of any range table labelled t or
  SHARED_TASK -- never waiting for the OTHER task's range tables, which write
  fields it does not read. A task with no range table at all falls back to
  FLOW_HASH_LEVEL + 1, the first stage after the flow-hash prologue.

  range_table_stages : StagePlan.table_stages of the range pool, one stage
                       index per range table.
  range_task         : one label per range table, positionally aligned:
                       program.APP_TASK, DDOS_TASK or SHARED_TASK.
  ternary_task       : one label per classification tree: APP_TASK or
                       DDOS_TASK.
  Returns one level per tree, aligned with ternary_task.

  Under 'joint' every range table is SHARED_TASK, so every tree's level is
  1 + the latest range stage = the range pool's StagePlan.depth -- exactly the
  single level every tree was given before C1. Evidence for the per-task rule
  (reviews/model_audit_2026-09-27.md §10 C1, per_task_variant.py, 28 disjoint
  compiles): p4c places 8/168 trees before the other task's last range table,
  never one before its own."""
  if len(range_table_stages) != len(range_task):
    raise ValueError(
        "tree_readiness_levels: got %d range table stages for %d range_task "
        "labels; the two must be positionally aligned, one per range table"
        % (len(range_table_stages), len(range_task)))
  unknown = (set(range_task) - set(TASKS) - {SHARED_TASK}) | (set(ternary_task) - set(TASKS))
  if unknown:
    raise ValueError(
        "tree_readiness_levels: unknown task label(s) %s; range tables take "
        "%s or %r, trees take %s" % (sorted(unknown), TASKS, SHARED_TASK, TASKS))
  ready = {}
  for task in TASKS:
    own = [stage for stage, label in zip(range_table_stages, range_task)
           if label in (task, SHARED_TASK)]
    ready[task] = max(own) + 1 if own else FLOW_HASH_LEVEL + 1
  return [ready[task] for task in ternary_task]


def assemble_usage(pool):
  """Pack both pools and assemble the ResourceUsage.

  The import-light half of multi_model_memory_evaluation: everything from the
  point the two encoding branches converge. Takes _pool_inputs' dict, returns
  (usage, range_plan, ternary_plan). Both StagePlans come back because the
  validation table reports pool depths separately and the golden fixture pins
  them field by field.

  Nothing here touches sklearn or a fitted forest, which is what lets the golden
  test and scripts/validation_table.py replay a serialized pool with no models
  and no campaign data."""
  range_table_specs = pool["range_table_specs"]
  ternary_table_specs = pool["ternary_table_specs"]
  range_levels = pool["range_levels"]
  range_task = pool["range_task"]
  ternary_task = pool["ternary_task"]
  range_fields = pool["range_fields"]
  ternary_fields = pool["ternary_fields"]
  ternary_key_bits = pool["ternary_key_bits"]
  interior_stages = pool["interior_stages"]
  register_names = pool["register_names"]
  range_entries = pool["range_entries"]
  range_blocks = pool["range_blocks"]
  ternary_entries = pool["ternary_entries"]
  codeword_length = pool["codeword_length"]

  # Range-matching tables and ternary classification tables are physically
  # distinct table pools (build_p4_script.py generates them separately), so
  # each pool is packed on its own, the ternary one seeded with the stages
  # the range one already fills. Both pools are packed by the SAME solver:
  # one stage count per pool that
  # respects the block, table-count and byte limits simultaneously, rather
  # than a max() of two independently-relaxed bounds (which can under-count,
  # see crossbar_stages_needed).
  # Both pools are placed dependency-aware (see crossbar_stages_needed and
  # feature_readiness_level): a feature's range table cannot precede the
  # register chain producing its key, and a classification table reads its
  # own task's code fields, so it cannot precede that task's last range table
  # (per-task readiness, below).
  # Validated against a real compile of the M2 program: 2 range stages + 1
  # classification stage = 3, exactly the compiler's own placement. The pure
  # packer predicted 2.
  #
  # F10: the classification boundary must be derived from where the range
  # pool's tables actually LANDED (StagePlan.table_stages), not from one past the
  # earliest stage a range table was merely ALLOWED to start
  # (max(range_levels) + 1) -- the 8-table crossbar cap can spill a range
  # table forward past its level, and reusing max(range_levels) + 1 would
  # then schedule a classification table into a stage a range table still
  # occupies.
  #
  # Mechanism B (reviews/p4_tofino_reference.md §4.6, Appendix B
  # "Mechanism B") on top of that: both pools are emitted AFTER the gated
  # register blocks, so neither can occupy a stage the placer spends wholly
  # inside one -- a control-flow constraint, not a capacity one, and the
  # reason a stage can sit at 0/24 TCAM blocks and still take no table. It is
  # a placement constraint rather than a penalty precisely so that it costs
  # nothing on the rows whose tables were not going to land there anyway.
  range_plan = crossbar_stages_needed(range_table_specs,
                                      readiness_levels=range_levels,
                                      key_fields=range_fields,
                                      unavailable_stages=interior_stages)
  # Per-task tree readiness (audit C1): a tree waits for the range tables of
  # ITS OWN task -- the ones writing the code_<feature> fields it keys on --
  # plus any SHARED_TASK table, never for the other task's. Under 'joint'
  # every range table is shared, so every tree starts at range_plan.depth, as
  # it always did. Under 'disjoint' a task whose range tables finish early
  # starts its trees early, while the other task's range tables may still be
  # landing -- so the two pools can now meet in one stage, and the ternary
  # pool is SEEDED with the range pool's loads: those range shards count
  # against the stage's table cap, byte limit and column packing, and their
  # range fields are keys placed on the stage's crossbar BEFORE any tree key
  # in the lane simulation, but are not charged again (see
  # crossbar_stages_needed's seed_stages). With no range
  # table for a task, its trees start right after the flow-hash prologue
  # (tree_readiness_levels).
  ternary_levels = tree_readiness_levels(range_plan.table_stages, range_task,
                                         ternary_task)
  # Only the classification pool gets key_field_bits, which switches on the
  # ordered stage simulation (packing.crossbar_stages_needed, audit C5): trees
  # are placed in p4c's order -- @placement_priority, ddos before app
  # (program.PLACEMENT_PRIORITY), ties to the tree listed last, which is why
  # ternary_table_specs must stay in program order (app trees, then ddos, each
  # by index) -- and a key placed after a different key in its stage pays its
  # lane leftover price. This runs once per call, i.e. once per trial; it is
  # never inside threshold alignment's loop, which prices keys with
  # codeword_to_blocks alone. A range table keys one meta.<feature>_val field
  # of FEATURE_VALUE_BIT_WIDTH bits -- 2 bytes -- so the range pool needs none
  # of this.
  ternary_plan = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=ternary_levels,
      key_fields=ternary_fields,
      unavailable_stages=interior_stages,
      key_field_bits=ternary_key_bits,
      seed_stages=range_plan.stage_loads,
      placement_priority=[PLACEMENT_PRIORITY[task] for task in ternary_task])

  # A stage both pools use must satisfy every per-stage limit with both
  # pools' tables in it. The seeded placement above already guarantees this;
  # re-checked here because it is the property that makes the two plans'
  # numbers mean anything together.
  range_loads = {load.index: load for load in range_plan.stage_loads}
  for load in ternary_plan.stage_loads:
    if load.index in range_loads:
      assert stage_load_fits([range_loads[load.index], load]), (
          "range and classification tables share stage {} beyond its "
          "table/byte/column limits".format(load.index))

  # F5/F6: stage_depth is ternary_plan.depth -- the classification pool is
  # placed LAST in the sense that matters: every tree starts after its own
  # task's range tables, so its depth is the overall pipeline depth. Verified
  # against the M2 fixture: ternary_plan.indices == {5} there, so depth == 6,
  # exactly the brief's own worked example. max() with range_plan.depth is a
  # defensive widening for the degenerate case of zero ternary tables (where
  # crossbar_stages_needed's dependency-aware branch would otherwise report
  # depth 0), not something the real M2-shaped models ever hit.
  # ...plus VOTE_EPILOGUE_STAGES: the vote tables read every tree's class, so
  # they always occupy one further stage past the classification pool, and
  # stage_depth is the quantity TOFINO_PIPELINE_STAGES is checked against.
  # Measured as exactly 1 on all 19 calibration compiles.
  stage_depth = max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES

  # register_depth reuses range_levels (already computed above on both
  # branches, positionally aligned with feature_intervals) rather than
  # re-traversing anything; register_count reuses register_names, likewise
  # already computed above on both branches. See ResourceUsage's docstring
  # for the Spec 4.3 capacity caveat these two fields carry.
  register_depth = max(range_levels, default=0)
  register_count = len(register_names)

  # ternary_plan.blocks, not ternary_blocks: the latter is the naive
  # per-table sum, before the lane leftover prices that only
  # crossbar_stages_needed's stage simulation knows about -- see
  # StagePlan.blocks. range_blocks needs no substitution:
  # a range table's key always leaves spare crossbar byte slots,
  # so range_plan.blocks is provably identical to range_blocks.
  # stages counts distinct stage indices holding a table from EITHER pool: a
  # stage both pools share (possible under 'disjoint' since C1) counts once.
  # Under 'joint' the pools never meet, and this is the old occupied sum.
  usage = ResourceUsage(
      stages=len(range_plan.indices | ternary_plan.indices),
      blocks=range_blocks + ternary_plan.blocks,
      stage_depth=stage_depth,
      range_entries=range_entries,
      ternary_entries=ternary_entries,
      codeword_length=codeword_length,
      register_depth=register_depth,
      register_count=register_count,
      range_depth=range_plan.depth,
      ternary_depth=ternary_plan.depth,
      range_tables=len(range_table_specs),
      ternary_tables=len(ternary_table_specs))
  return usage, range_plan, ternary_plan
