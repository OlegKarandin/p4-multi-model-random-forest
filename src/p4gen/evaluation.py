# ---------------------------------------------------------------------------
# Frontend imports: the tree/codeword machinery this module's own functions
# still call directly, plus the sklearn.metrics dependency accuracy_metrics
# needs. Everything the resource model itself needs now lives in
# src/p4model/ (see the shim block below) and is import-light -- no sklearn,
# no numpy, no CWD dependence.
# ---------------------------------------------------------------------------
import sklearn.metrics as mt
from src.p4gen.build_p4_script import (
    _reject_colliding_feature_names,
    feature_intervals_from_nodes,
    generate_codewords,
    get_feature_intervals,
    get_root_to_leaf_paths,
    merge_tree_nodes,
    most_common_class_and_dropped_codewords,
    tree_nodes_for,
)


# ---------------------------------------------------------------------------
# Shim block. The resource model itself now lives in src/p4model/ (import-light:
# no sklearn, no numpy, no CWD dependence -- see tests/test_p4model_guards.py).
# Everything below is re-exported at its historical path so src/training/,
# src/reporting/, scripts/ and the test suite keep the imports they always had.
# Explicit names, never `import *`, so the mapping stays greppable and a heavy
# import cannot sneak into p4model's public surface.
#
# Every entry is an identity re-export EXCEPT ternary_matching_resource_usage,
# which is a wrapper: it owns this generator's default-action policy and hands
# the model a per-tree count (see its own docstring).
#
# What did NOT move, and why:
#   accuracy_metrics                      -- the only sklearn.metrics caller
#   single_model_memory_evaluation, _pool_inputs -- the fitted-forest frontend
#                                            (Task 9 split multi_model_memory_
#                                            evaluation at the point its two
#                                            encoding branches converge:
#                                            _pool_inputs stayed here,
#                                            assemble_usage -- imported below
#                                            -- moved to src/p4model/usage.py;
#                                            multi_model_memory_evaluation
#                                            itself is now a thin composition
#                                            of the two)
#   INFINITE, MAX_NUM_FLOWS (build_p4_script) -- codegen constants with no
#                                            reader inside the model
#   most_common_class_and_dropped_codewords   -- codegen policy, not physics
# ---------------------------------------------------------------------------
from src.p4model.usage import ResourceUsage, assemble_usage
from src.p4model.errors import CodewordTooLong, CrossbarKeyTooWide
from src.p4model.target import (
    CODEWORD_KEY_OVERHEAD_BITS,
    MAX_RANGE_KEY_BITS,
    METER_ALUS_PER_STAGE,
    RANGE_WORST_CASE_ENTRY_FRACTION,
    RANGE_WORST_CASE_ROWS_CAP,
    TOFINO_PIPELINE_STAGES,
)
from src.p4model.program import (
    FEATURE_VALUE_BIT_WIDTH,
    FLOW_HASH_LEVEL,
    ORIENTATION_REGISTER,
    RANGE_TABLE_KEY_BYTES,
    REGISTER_BLOCK_ORDER,
    VOTE_EPILOGUE_STAGES,
)
from src.p4model.packing import (
    StagePlan,
    _stage_shards,
    crossbar_stages_needed,
    fits_two_columns,
)
from src.p4model.ranges import (
    compiler_range_rows,
    nibble_widths_for,
    range_entry_count,
)
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
from src.p4model.registers import (
    _register_blocks,
    feature_readiness_level,
    gated_block_interior_stages,
    readiness_levels_for,
    register_stage_schedule,
)
from src.p4model.names import normalise_feature_name
from src.p4model.catalog import (
    FEATURE_REGISTER_CATALOG,
    register_names_for,
    register_width_bits,
)


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


# ---------------------------------------------------------------------------
# What still lives here: accuracy_metrics (the only sklearn.metrics caller),
# the ternary_matching_resource_usage wrapper above (Task 6's default-action
# policy), and the two fitted-forest frontends that turn a trained
# RandomForestClassifier into the resource model's inputs.
# ---------------------------------------------------------------------------
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


def _pool_inputs(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding,
                 use_default_action_discount=False):
  """Fitted-forest frontend for multi_model_memory_evaluation: runs the
  'joint'/'disjoint' encoding branches on both RandomForestClassifiers and
  pools everything they produce, once converged, into a plain dict under the
  same 13 names regardless of which branch ran.

  This is the Tier 2 ProgramSpec seam (Spec S4.5, finding F6): downstream of
  this function, src.p4model.usage.assemble_usage takes only this dict and is
  import-light -- no sklearn, no fitted forest, no CWD dependence -- which is
  what lets a golden fixture and scripts/validation_table.py replay a real
  prediction with no models and no campaign data.

  range_fields and ternary_fields carry (field_id, field_BYTES) pairs here --
  ternary_key_fields' and range_key_fields_for's native unit. A later fixture
  that serializes this pool to disk records the same fields as (field_id,
  field_BITS) instead, since bits round-trip exactly to bytes but not the
  reverse.

  ternary_blocks (the naive per-table block sum each branch computes below)
  is deliberately NOT one of the 13 keys: it is already dead after the branch
  converges -- assemble_usage's ResourceUsage.blocks uses ternary_plan.blocks,
  the ragged-key-charged StagePlan total from src.p4model.packing, never this
  naive sum. Carrying it into the pool would invite exactly the confusion the
  StagePlan.blocks fix (see CLAUDE.md's compiler-calibration note) was
  created to resolve.

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

  # interior_stages depends on gated_block_interior_stages
  # (src.p4model.registers), so it is computed here rather than in
  # assemble_usage: assemble_usage is deliberately import-light (no
  # registers, no catalog, no sklearn -- see its own docstring), and pooling
  # this value is what lets it stay that way while still packing both pools
  # dependency-aware. See assemble_usage's Mechanism B comment for why the
  # value itself matters.
  interior_stages = gated_block_interior_stages(emitted_features)

  # ternary_blocks (the naive per-table sum each branch computed above) is
  # deliberately NOT one of the 13 keys below -- see this function's own
  # docstring.
  return {
      "range_table_specs": range_table_specs,
      "ternary_table_specs": ternary_table_specs,
      "range_levels": range_levels,
      "range_fields": range_fields,
      "ternary_fields": ternary_fields,
      "ternary_ragged": ternary_ragged,
      "interior_stages": interior_stages,
      "emitted_features": emitted_features,
      "register_names": register_names,
      "range_entries": range_entries,
      "range_blocks": range_blocks,
      "ternary_entries": ternary_entries,
      "codeword_length": codeword_length,
  }


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
  usage, _range_plan, _ternary_plan = assemble_usage(_pool_inputs(
      clf_app, clf_ddos, selected_features_app, selected_features_ddos,
      encoding, use_default_action_discount=use_default_action_discount))
  return usage
