from dataclasses import dataclass
import sklearn.metrics as mt
from src.p4gen.build_p4_script import (
    MAX_NUM_FLOWS,
    _reject_colliding_feature_names,
    feature_intervals_from_nodes,
    generate_codewords,
    get_feature_intervals,
    get_root_to_leaf_paths,
    merge_tree_nodes,
    most_common_class_and_dropped_codewords,
    tree_nodes_for,
)
from src.p4model.catalog import (
    FEATURE_REGISTER_CATALOG,
    register_names_for,
    register_width_bits,
)
from src.p4model.names import normalise_feature_name
from src.p4model.ranges import (
    nibble_widths_for,
    range_entry_count,
)
from src.p4model.errors import CodewordTooLong, CrossbarKeyTooWide
from src.p4model.tables import (
    band_factor,
    crossbar_block_width,
    exact_match_resource_usage,
    range_deployment_overflow,
    range_key_fields_for,
    range_matching_resource_usage,
    ternary_key_fields,
    ternary_key_is_ragged,
    ternary_table_key_bytes,
)
from src.p4model import tables as _tables
from src.p4model.program import (
    FEATURE_VALUE_BIT_WIDTH,
    FLOW_HASH_LEVEL,
    ORIENTATION_REGISTER,
    RANGE_TABLE_KEY_BYTES,
    REGISTER_BLOCK_ORDER,
    VOTE_EPILOGUE_STAGES,
)
from src.p4model.target import (
    CODEWORD_KEY_OVERHEAD_BITS,
    MAX_RANGE_KEY_BITS,
    METER_ALUS_PER_STAGE,
    RANGE_WORST_CASE_ENTRY_FRACTION,
    RANGE_WORST_CASE_ROWS_CAP,
    TOFINO_PIPELINE_STAGES,
)


from src.p4model.packing import (
    StagePlan,
    _stage_shards,
    crossbar_stages_needed,
    fits_two_columns,
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

  No __int__ is provided. StagePlan has one as a transitional shim;
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

  range_depth  : StagePlan.depth for the range-matching pool ALONE (Task 6
                extended by the stage-depth-attribution design's Phase 1).
                Component of stage_depth, not itself a ceiling-checked
                quantity -- only stage_depth is.
  ternary_depth: StagePlan.depth for the ternary classification pool ALONE.
                Equal to stage_depth in every non-degenerate design (the
                classification pool is placed last, see stage_depth's own
                docstring); the max() stage_depth takes exists only for the
                degenerate zero-ternary-table case.
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
  stage_depth: int      # pipeline DEPTH (StagePlan.depth), what the 12-stage ceiling reads
  range_entries: int
  ternary_entries: int
  codeword_length: int  # pooled split-threshold count under 'joint'; see docstring
  register_depth: int   # max readiness level over the selected feature(s); see class docstring
  register_count: int   # distinct Register<> instances, deduplicated by name across both models
  range_depth: int       # StagePlan.depth, range pool alone; see class docstring
  ternary_depth: int     # StagePlan.depth, ternary pool alone; see class docstring
  range_tables: int      # len(range_table_specs); see class docstring
  ternary_tables: int    # len(ternary_table_specs); see class docstring


def accuracy_metrics(y_true, y_pred, task):
    """Return (accuracy, weighted_f1) for a given task.

    Label sets are duplicated from `src.training.incremental_metrics.TASK_LABELS` to avoid
    a p4gen→training dependency; equivalence is pinned by `test_the_incremental_metrics_equal_the_from_scratch_oracle_after_every_step`
    and `test_both_label_spaces_work` in `tests/test_incremental_metrics.py`.
    """

    if task == 'app':
        lab = [0, 1, 2]

    elif task == 'ddos':
        lab = [-1, 1]

    else:
        raise ValueError(
            "accuracy_metrics: unknown task {!r}; expected 'app' or 'ddos'".format(task))

    accuracy = mt.accuracy_score(y_true, y_pred)
    f1score = mt.f1_score(y_true, y_pred, labels=lab, average='weighted')

    return accuracy, f1score


# range_entry_count, nibble_widths_for and compiler_range_rows now live in
# src/p4model/ranges.py (imported above) so bfshell's embedded Python (no
# sklearn, hence no import of this module) can share the exact same
# implementations instead of carrying its own copies.


def ternary_matching_resource_usage(codewords, feature_intervals,
                                    use_default_action_discount=False):
  """Wrapper over src.p4model.tables.ternary_matching_resource_usage that owns
  this generator's default-action POLICY.

  The model takes a per-tree count of folded entries; deciding WHICH leaves fold
  is Planter RF_EB's majority-class rule, which lives in build_p4_script beside
  get_table_entries (its other caller) precisely so the resource model does not
  have to know about decision trees -- or import a module that pulls in numpy
  and sklearn.tree.

  The only name in the shim block that is a wrapper rather than an identity
  re-export."""
  dropped_per_tree = None
  if use_default_action_discount:
    dropped_per_tree = [
        len(most_common_class_and_dropped_codewords(codewords[tree])[1])
        if len(codewords[tree]) > 0 else 0
        for tree in codewords]
  return _tables.ternary_matching_resource_usage(
      codewords, feature_intervals, dropped_per_tree=dropped_per_tree)


# feature_readiness_level, _register_blocks, register_stage_schedule,
# gated_block_interior_stages and readiness_levels_for now live in
# src/p4model/registers.py (the stateful register schedule); re-exported
# below so every existing caller keeps working unchanged.
from src.p4model.registers import (
    _register_blocks,
    feature_readiness_level,
    gated_block_interior_stages,
    readiness_levels_for,
    register_stage_schedule,
)


def single_model_memory_evaluation(clf, selected_features, use_default_action_discount=False):
  """use_default_action_discount: opt-in, passed straight through to
  ternary_matching_resource_usage (which has implemented the Planter-style
  discount since Task 7 but was never reachable from this estimator). False
  -- the default -- reproduces every pre-existing caller's numbers exactly."""
  # Same guard get_feature_intervals runs before its own tree_nodes_for call
  # (Task 4) -- this function bypasses get_feature_intervals entirely (Task
  # 16) so it must run the check itself, or two differently-spelled feature
  # names that normalise to the same key silently merge their intervals with
  # no raise (final-review finding #1).
  _reject_colliding_feature_names(selected_features)
  tree_nodes = tree_nodes_for(clf, selected_features)
  feature_intervals = feature_intervals_from_nodes(tree_nodes)
  range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

  paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
  codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
  ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
      codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

  return (range_entries, range_blocks, ternary_entries, ternary_blocks, codewords, codeword_length,
          range_table_specs, ternary_table_specs)


def multi_model_memory_evaluation(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding,
                                  use_default_action_discount=False):
  """Returns ResourceUsage(stages, blocks, stage_depth, range_entries, ternary_entries,
  register_depth, register_count) -- FOUR related but DISTINCT
  stage-index quantities (F6, extended by Task 6) are in play below: this
  function is the source of truth for THREE of them (stages, stage_depth,
  register_depth) -- the fourth, stages_real, comes from the real compiler,
  not from this function (see its own paragraph below):

    stages      : OCCUPIED match-table stage count -- how many distinct
                  stage indices actually hold a table from either pool
                  (range_plan.occupied + ternary_plan.occupied). M2 example: 3.
                  This is what gets written to the campaign CSV's `stages`
                  column and plotted -- it is NOT a pipeline-depth quantity
                  and must never be compared against TOFINO_PIPELINE_STAGES.
    blocks      : total blocks used by both range and ternary tables
    stage_depth : pipeline DEPTH, max(occupied stage index) + 1 -- the
                  quantity a hard stage ceiling actually reads (F5). Read
                  from ternary_plan.depth (the classification pool is placed
                  LAST, after the range pool, so its depth is the overall
                  pipeline depth), defensively widened to
                  max(range_plan.depth, ternary_plan.depth) so an
                  (unrealistic) model with no ternary tables at all still
                  reports a sane depth, plus VOTE_EPILOGUE_STAGES for the
                  vote tables' own trailing stage. M2 example: 9 -- which is
                  what the real compiler needs for M2 as well. It read 6
                  before the 2026-09-05 calibration corrected FLOW_HASH_LEVEL
                  and added the epilogue; anything comparing stage_depth
                  against pre-2026-09-05 archived campaign numbers is
                  comparing two different definitions.
    register_depth : max readiness level (feature_readiness_level) over the
                  selected feature(s) -- how many stages elapse before the
                  LAST register a classification table depends on has run.
                  Related to stage_depth (both are stage-index quantities
                  gated by the same register-dependency model) but NOT the
                  same number: stage_depth also accounts for crossbar
                  packing/spill of the match tables themselves, which
                  register_depth does not. See ResourceUsage's own
                  docstring for register_depth/register_count and their
                  capacity caveat (Spec 4.3).
    range_entries  : count of physical rows across all range tables
    ternary_entries: count of ternary codewords across all classification trees
    (a fourth quantity, `stages_real` -- the REAL compiler's whole-program
    stage count including parsing/bookkeeping overhead this function does
    not model at all -- is NOT returned here; see p4_compile.parse_compile_logs,
    which stores it. M2 example: 9. `stages` and `stages_real` sit side by
    side in the same campaign dataframe row and are NOT the same quantity --
    plotting them together as if they were reads as the model being "67%
    wrong" when they are not even measuring the same thing. `stage_depth` and
    `stages_real` ARE comparable, and since 2026-09-05 they agree on M2; the
    residual on the 19-row calibration sample is 0-3 stages, always with
    stages_real the larger -- unmodelled PHV container conflicts and TCAM
    column geometry, see scripts/compiler_calibration.replay_stage_depth.)

  use_default_action_discount: opt-in, threaded down to
  ternary_matching_resource_usage under BOTH encodings -- directly for
  'joint' (which does its own ternary accounting on the merged tree set),
  and via both nested single_model_memory_evaluation calls for 'disjoint'.
  False -- the default -- reproduces every pre-existing caller's numbers
  exactly."""

  if encoding == 'joint':
    # Same union-based guard get_joint_feature_intervals runs before its own
    # tree_nodes_for calls (Task 4) -- this branch bypasses
    # get_joint_feature_intervals entirely (Task 16) so it must run the
    # check itself, on the SAME union both models' selected features form,
    # or a cross-model spelling collision (e.g. 'Flow.IAT.Max' in one model,
    # 'Flow IAT Max' in the other) silently merges with no raise
    # (final-review finding #1).
    _reject_colliding_feature_names(list(selected_features_app) + list(selected_features_ddos))
    tree_nodes = merge_tree_nodes(
        tree_nodes_for(clf_app, selected_features_app),
        tree_nodes_for(clf_ddos, selected_features_ddos))

    feature_intervals = feature_intervals_from_nodes(tree_nodes)
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    paths_leaf_nodes_per_tree = get_root_to_leaf_paths(tree_nodes)
    codewords = generate_codewords(paths_leaf_nodes_per_tree, feature_intervals)
    ternary_entries, ternary_blocks, codeword_length, ternary_table_specs = ternary_matching_resource_usage(
        codewords, feature_intervals, use_default_action_discount=use_default_action_discount)

    range_levels = readiness_levels_for(feature_intervals)
    emitted_features = list(feature_intervals)
    register_names = register_names_for(feature_intervals)
    # One merged interval set, so every tree keys on the same fields.
    range_fields = range_key_fields_for(feature_intervals)
    ternary_fields = [ternary_key_fields(feature_intervals)] * len(ternary_table_specs)
    ternary_ragged = ([ternary_key_is_ragged(feature_intervals)]
                      * len(ternary_table_specs))

  elif encoding == 'disjoint':

    (range_entries_app, range_blocks_app, ternary_entries_app, ternary_blocks_app,
     codewords_app, codeword_length_app,
     range_table_specs_app, ternary_table_specs_app) = single_model_memory_evaluation(
        clf_app, selected_features_app, use_default_action_discount=use_default_action_discount)
    (range_entries_ddos, range_blocks_ddos, ternary_entries_ddos, ternary_blocks_ddos,
     codewords_ddos, codeword_length_ddos,
     range_table_specs_ddos, ternary_table_specs_ddos) = single_model_memory_evaluation(
        clf_ddos, selected_features_ddos, use_default_action_discount=use_default_action_discount)

    range_blocks = range_blocks_app + range_blocks_ddos
    range_entries = range_entries_app + range_entries_ddos

    #Ternary-matching tables final summation
    ternary_blocks = ternary_blocks_app + ternary_blocks_ddos
    ternary_entries = ternary_entries_app + ternary_entries_ddos

    # Two independent codewords; see ResourceUsage.codeword_length's docstring
    # for why max and not sum.
    codeword_length = max(codeword_length_app, codeword_length_ddos)

    # Under disjoint encoding each model keeps its own feature intervals, so
    # both models' tables are independent tables competing for the same
    # per-stage budgets -- pack them together, per pool.
    range_table_specs = range_table_specs_app + range_table_specs_ddos
    ternary_table_specs = ternary_table_specs_app + ternary_table_specs_ddos

    # Each model keeps its own intervals here, so levels must be derived per
    # model and concatenated in the SAME order the specs were. But there is
    # only ONE register block and one set of stateful ALUs behind both models
    # (a register a feature needs is emitted once, however many models select
    # that feature), so both calls must read their levels off a schedule of
    # the UNION -- see readiness_levels_for's emitted_features.
    feature_intervals_app = get_feature_intervals(clf_app, selected_features_app)
    feature_intervals_ddos = get_feature_intervals(clf_ddos, selected_features_ddos)
    emitted_features = list(feature_intervals_app) + list(feature_intervals_ddos)
    range_levels = (
        readiness_levels_for(feature_intervals_app, emitted_features=emitted_features) +
        readiness_levels_for(feature_intervals_ddos, emitted_features=emitted_features))
    # Same per-model split, same concatenation order, for the crossbar field
    # identities. Two models that selected the same feature resolve to the
    # same range field (the raw _val field is always shared) and, when their
    # interval lists also agree, to the same code_ field -- exactly the two
    # sharing rules _resolve_disjoint_feature_plan implements.
    range_fields = (
        range_key_fields_for(feature_intervals_app) +
        range_key_fields_for(feature_intervals_ddos))
    ternary_fields = (
        [ternary_key_fields(feature_intervals_app)] * len(ternary_table_specs_app) +
        [ternary_key_fields(feature_intervals_ddos)] * len(ternary_table_specs_ddos))
    ternary_ragged = (
        [ternary_key_is_ragged(feature_intervals_app)] * len(ternary_table_specs_app) +
        [ternary_key_is_ragged(feature_intervals_ddos)] * len(ternary_table_specs_ddos))
    # Registers, unlike range_levels, must be deduplicated ACROSS both
    # models: they are a single physical Register<> array shared by the
    # whole generated program regardless of 'joint' vs 'disjoint' ENCODING
    # (that choice only affects codeword/interval sharing -- see
    # build_p4_script.py's raw_feature_intervals, keyed on the union of
    # both models' raw feature names). One register_names_for call over the
    # combined feature names dedupes correctly; two separate calls
    # concatenated would not (each call only dedupes within itself), and
    # would double-count a register shared by both models.
    register_names = register_names_for(
        list(feature_intervals_app) + list(feature_intervals_ddos))

  else:
    raise ValueError(
        "multi_model_memory_evaluation: unknown encoding {!r}; "
        "expected 'joint' or 'disjoint'".format(encoding))

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
  # Mechanism B on top of that: both pools are emitted AFTER the gated
  # register blocks, so neither can occupy a stage the placer spends wholly
  # inside one -- a control-flow constraint, not a capacity one, and the
  # reason a stage can sit at 0/24 TCAM blocks and still take no table. It is
  # a placement constraint rather than a penalty precisely so that it costs
  # nothing on the rows whose tables were not going to land there anyway.
  interior_stages = gated_block_interior_stages(emitted_features)
  range_plan = crossbar_stages_needed(range_table_specs,
                                      readiness_levels=range_levels,
                                      key_fields=range_fields,
                                      unavailable_stages=interior_stages)
  ternary_level = range_plan.depth if range_table_specs else FLOW_HASH_LEVEL + 1
  # Only the classification pool gets ragged_keys. A range table keys one
  # meta.<feature>_val field of FEATURE_VALUE_BIT_WIDTH bits, a whole number
  # of bytes, so it presents no part-used crossbar byte and the group-offset
  # penalty is inert there by construction -- passing it would be noise.
  ternary_plan = crossbar_stages_needed(
      ternary_table_specs,
      readiness_levels=[ternary_level] * len(ternary_table_specs),
      key_fields=ternary_fields,
      unavailable_stages=interior_stages,
      ragged_keys=ternary_ragged)

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
  # per-table sum computed above, before the ragged-key group-offset charge
  # (Mechanism G) that only crossbar_stages_needed's stage packing knows
  # about -- see StagePlan.blocks. range_blocks needs no such substitution:
  # a range table's key is always a whole number of bytes (never ragged),
  # so range_plan.blocks is provably identical to range_blocks.
  return ResourceUsage(
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
