"""ResourceUsage: multi_model_memory_evaluation's return value.

Depends on nothing else in p4model -- it is a plain data container, moved
here so callers that only need the shape of a result (not the physics that
produces one) can import it without pulling in the rest of the model."""
from dataclasses import dataclass


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

  codeword_length : classification-table key width in bits, before the
                  CODEWORD_KEY_OVERHEAD_BITS the block factor adds. Under
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
  CAVEAT (Spec 4.3, applies to both fields above): this reports register
  DEPTH and COUNT (how many stages, how many registers), NOT register
  CAPACITY. Tofino has a limited number of stateful ALUs per stage, and
  whether these registers actually FIT has never been measured in this
  repo -- do not read register_depth/register_count as a feasibility
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
