"""ResourceUsage and assemble_usage: multi_model_memory_evaluation's return
value, and the import-light half of the function that builds it.

ResourceUsage depends on nothing else in p4model -- it is a plain data
container, moved here so callers that only need the shape of a result (not
the physics that produces one) can import it without pulling in the rest of
the model."""
from dataclasses import dataclass

from src.p4model.packing import crossbar_stages_needed
from src.p4model.program import FLOW_HASH_LEVEL, VOTE_EPILOGUE_STAGES


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
  # each pool is packed on its own and the two stage counts are summed. Both
  # pools are packed by the SAME solver: one stage count per pool that
  # respects the block, table-count and byte limits simultaneously, rather
  # than a max() of two independently-relaxed bounds (which can under-count,
  # see crossbar_stages_needed).
  # Both pools are placed dependency-aware (see crossbar_stages_needed and
  # feature_readiness_level): a feature's range table cannot precede the
  # register chain producing its key, and every classification table reads
  # every feature's codeword, so it cannot precede the last range table.
  # Validated against a real compile of the M2 program: 2 range stages + 1
  # classification stage = 3, exactly the compiler's own placement. The pure
  # packer predicted 2.
  #
  # F10: the classification boundary must be derived from where the range
  # pool's tables actually LANDED (StagePlan.depth), not from one past the
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
  # Deliberate over-prediction, the safe direction: every classification
  # table starts at the FULL range pool's depth, even under 'disjoint'
  # encoding where a ddos tree only reads ddos features' code fields and
  # could in principle start as soon as just the ddos range tables have
  # landed, not the app ones too. Not modelled -- doing so would need
  # per-task range levels threaded through this pool -- and the 18/19
  # stage_depth calibration result (scripts/validation_table.py) suggests
  # the case rarely binds in practice.
  ternary_level = range_plan.depth if range_table_specs else FLOW_HASH_LEVEL + 1
  # Only the classification pool gets key_field_bits, which is what switches
  # on the crowded-stage rules (packing.charged/fits). A range table keys one
  # meta.<feature>_val field of FEATURE_VALUE_BIT_WIDTH bits -- 2 bytes -- and
  # the 8-table cap holds a stage to 16 of them, nowhere near crowding it.
  # Passing it would be noise.
  ternary_plan = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=[ternary_level] * len(ternary_table_specs),
      key_fields=ternary_fields,
      unavailable_stages=interior_stages,
      key_field_bits=ternary_key_bits)

  # The property that makes summing occupancies below meaningful: the two
  # pools must never claim the same stage index.
  assert not (range_plan.indices & ternary_plan.indices), (
      "range and classification pools overlap at stages {}; summing their "
      "occupancies is only meaningful while they are disjoint".format(
          sorted(range_plan.indices & ternary_plan.indices)))

  # F5/F6: stage_depth is ternary_plan.depth -- the classification pool is
  # placed LAST (it starts at ternary_level, which is itself derived from
  # range_plan.depth), so its depth is the overall pipeline depth. Verified
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
  # per-table sum computed above, before the crowded-stage margin
  # (packing.charged) that only crossbar_stages_needed's stage packing knows
  # about -- see StagePlan.blocks. range_blocks needs no substitution:
  # a range table's key always leaves spare crossbar byte slots,
  # so range_plan.blocks is provably identical to range_blocks.
  usage = ResourceUsage(
      stages=range_plan.occupied + ternary_plan.occupied,
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
