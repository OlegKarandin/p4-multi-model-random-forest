# ---------------------------------------------------------------------------
# Frontend imports: the tree/codeword machinery this module's own functions
# still call directly, plus the sklearn.metrics dependency accuracy_metrics
# needs. Everything the resource model itself needs now lives in
# p4model/ (see the shim block below) and is import-light -- no sklearn,
# no numpy, no CWD dependence.
# ---------------------------------------------------------------------------

import sklearn.metrics as mt
from p4gen.build_p4_script import (
    _reject_colliding_feature_names,
    code_field_container_sizes,
    iat_scratch_fields,
    plan_key_order,
    resolve_feature_plan,
    resolve_per_task_plan,
    feature_intervals_from_nodes,
    generate_codewords,
    get_feature_intervals,
    get_root_to_leaf_paths,
    merge_tree_nodes,
    most_common_class_and_dropped_codewords,
    tree_nodes_for,
)


# ---------------------------------------------------------------------------
# Shim block. The resource model itself now lives in p4model/ (import-light:
# no sklearn, no numpy, no CWD dependence -- see tests/test_package_layout.py).
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
#                                            -- moved to p4model/usage.py;
#                                            multi_model_memory_evaluation
#                                            itself is now a thin composition
#                                            of the two)
#   INFINITE, MAX_NUM_FLOWS (build_p4_script) -- codegen constants with no
#                                            reader inside the model
#   most_common_class_and_dropped_codewords   -- codegen policy, not physics
# ---------------------------------------------------------------------------
from p4model.usage import ResourceUsage, assemble_usage
from p4model.errors import CrossbarKeyTooWide, NoSplitTree, RangeEntriesDoNotFit
from p4model.target import (
    METER_ALUS_PER_STAGE,
    TOFINO_PIPELINE_STAGES,
)
from p4model.program import (
    APP_TASK,
    DDOS_TASK,
    ENCODINGS,
    PER_TASK_ENCODING,
    PER_TREE_ENCODING,
    SHARED_TASK,
    FLOW_HASH_LEVEL,
    ORIENTATION_REGISTER,
    VOTE_EPILOGUE_STAGES,
    per_task_field,
)
from p4model.packing import (
    _stage_shards,
    crossbar_stages_needed,
    fits_two_columns,
    packed_alone_blocks,
)
from p4model.ranges import (
    nibble_widths_for,
    range_entry_count,
)
from p4model.tables import (
    check_key_bytes,
    codeword_bits_to_blocks,
    entries_across_trees_to_blocks,
    exact_match_resource_usage,
    per_task_key_fields,
    range_deployment_overflow,
    range_key_fields_for,
    range_action_bytes_for,
    range_matching_resource_usage,
    ternary_key_fields,
    ternary_key_field_bits,
    tree_entries_to_blocks,
    codeword_fields_to_bytes,
    codeword_fields_to_bytes_from_bits,
)
from p4model import tables as _tables
from p4model import lanes, packed, phv
from p4model.registers import (
    feature_readiness_level,
    readiness_levels_for,
    register_stage_schedule,
)
from p4model.catalog import (
    FEATURE_REGISTER_CATALOG,
    register_names_for,
)


def ternary_matching_resource_usage(codewords, feature_intervals,
                                    use_default_action_discount=False):
  """Wrapper over p4model.tables.ternary_matching_resource_usage that owns
  this generator's default-action POLICY.

  The model takes a per-tree count of folded entries; deciding WHICH leaves fold
  is Planter RF_EB's majority-class rule, which lives in build_p4_script beside
  the entry writer (its other caller, _tree_table_entries) precisely so the
  resource model does not have to know about decision trees -- or import a
  module that pulls in numpy and sklearn.tree.

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


def _packed_tree_entries(codewords, key, use_default_action_discount):
  """(entries, codeword length) of ONE packed tree's table, from its one-tree
  codeword dict, WITHOUT p4model.tables' block pricing: that prices the OLD
  unpacked layout (lanes.table_blocks of ternary_key_field_bits) and would refuse
  a design for a key the program never emits. The refusals kept are the packed
  key's own: a key over target.MAX_KEY_BYTES crossbar bytes (len(key), the
  packed byte count). The packer adds the
  crossbar-allocation refusal on the packed key (CrossbarKeyTooWide, when
  packing.packed_alone_blocks finds no allocation)."""
  entries = len(codewords[0])
  if use_default_action_discount and entries > 0:
    entries -= len(most_common_class_and_dropped_codewords(codewords[0])[1])
  length = len(next(iter(codewords[0].items()))[0])
  check_key_bytes(len(key))
  return entries, length


def _key_slices(fields):
  """One classification key as packing.crossbar_stages_needed's key_slices
  takes it: (field id, slices) per (field id, bit width) of `fields`, laid
  out by lanes.layout, the width rule -- the containers
  build_p4_script.code_field_container_sizes pins. A field with no bits is
  not keyed."""
  fields = list(fields)
  return tuple((field_id, tuple(lanes.layout(bits))) for field_id, bits in fields
               if bits > 0)


def _code_fields(feature_intervals, names):
  """(field name, bit width) per code field of a key over feature_intervals, in
  its order, each field named as the program names it (names[feature],
  code_<resolved name>, _field_names): the crossbar ids two keys share when they
  key one field, and the names xbar.code_container_order orders the stage's
  containers by -- what the replay reads from the program text."""
  return [(names[feature], max(len(intervals) - 1, 0))
          for feature, intervals in feature_intervals.items()]


def _key_orders(resolved_plan):
  """{task: [raw feature]} in the order the program text lists each task's code
  fields (build_p4_script.plan_key_order), for _tree_keys."""
  return {task: [raw for _, raw in plan_key_order(resolved_plan, task)]
          for task in (APP_TASK, DDOS_TASK)}


def _field_names(resolved_plan):
  """{task: {raw feature: code field name}}: the code_<resolved name> field each
  task's trees key for a feature (build_p4_script.plan_key_order), as the
  program declares it."""
  return {task: {raw: "code_" + name for name, raw in plan_key_order(resolved_plan, task)}
          for task in (APP_TASK, DDOS_TASK)}


def _tree_keys(trees, intervals_of, orders):
  """Per tree of `trees` ([(task, index, nodes)], program order), the intervals
  its classification key reads under the joint, disjoint and per-task
  encodings (decision 2.3): intervals_of[task] restricted to the features the
  tree splits on, in the program text's field order orders[task]
  (_key_orders) -- build_p4_script.tree_key_features, which the generator keys
  the program with. Raises ValueError for a tree that splits a feature with no
  code field of its task, and NoSplitTree, as the generator does, for the first
  tree with no split (R6)."""
  keys = []
  for task, index, nodes in trees:
    split = {feature for feature, intervals in feature_intervals_from_nodes({0: nodes}).items()
             if len(intervals) >= 2}
    unkeyed = sorted(f for f in split if f not in orders[task] or f not in intervals_of[task])
    if unkeyed:
      raise ValueError("tree %s_%d splits %s, which its task has no code field for: the "
                       "task's intervals must be its trees' own" % (task, index, unkeyed))
    key = {feature: intervals_of[task][feature] for feature in orders[task] if feature in split}
    if not key:
      raise NoSplitTree("tree %s_%d has no threshold, so its table would have no key"
                        % (task, index), "%s_%d" % (task, index))
    keys.append(key)
  return keys


def _pruned_trees(trees, keys, codewords, table_of, n_range_tables, per_task_fields,
                  field_names, use_default_action_discount):
  """The classification side of a pool whose trees key only their own features
  (_tree_keys): per tree, its table spec (blocks, crossbar bytes) priced on its
  own key, the key's fields / bits / slices, and tree_reads, the range-table
  positions (table_of(task, feature)) it reads (usage.tree_readiness_levels'
  rule E). Also the readers: range_dependents (trees reading each range table)
  and range_task (the one task reading a table, or SHARED_TASK when both do --
  the replay's p4_replay.table_tasks rule). codewords: one codeword dict per
  tree, against its task's full intervals; only its count and length are read
  (a field the tree does not key is '*' in all of them), so ternary_entries
  and codeword_length (the longest) are the full
  codewords'. A key over a stage's crossbar bytes raises CrossbarKeyTooWide,
  per tree. per_task_fields: the per-task encoding's field ids
  (tables.per_task_key_fields), which two tasks never share. field_names:
  _field_names of the encoding's plan, naming each key's fields for the
  crossbar (_code_fields)."""
  out = {"ternary_table_specs": [], "ternary_fields": [], "ternary_key_bits": [],
         "ternary_key_slices": [], "tree_reads": [], "ternary_entries": 0,
         "codeword_length": 0}
  readers = [set() for _ in range(n_range_tables)]
  for tree, ((task, _, _), key, tree_codewords) in enumerate(zip(trees, keys, codewords)):
    entries, _, length, (spec,) = ternary_matching_resource_usage(
        {0: tree_codewords}, key, use_default_action_discount=use_default_action_discount)
    out["ternary_entries"] += entries
    out["codeword_length"] = max(out["codeword_length"], length)
    out["ternary_table_specs"].append(spec)
    out["ternary_fields"].append(per_task_key_fields(task, key) if per_task_fields
                                 else ternary_key_fields(key))
    out["ternary_key_bits"].append(ternary_key_field_bits(key))
    out["ternary_key_slices"].append(
        _key_slices(_code_fields(key, field_names[task])))
    reads = [table_of(task, feature) for feature in key]
    out["tree_reads"].append(reads)
    for position in reads:
      readers[position].add((task, tree))
  out["range_dependents"] = [len(tree_set) for tree_set in readers]
  out["range_task"] = []
  for tree_set in readers:
    tasks = {task for task, _ in tree_set}
    out["range_task"].append(next(iter(tasks)) if len(tasks) == 1 else SHARED_TASK)
  return out


def _program_trees(tree_nodes_app, tree_nodes_ddos):
  """[(task, index, nodes)] in program order: the app trees, then the ddos
  trees, each by index -- the order the generator lists them in and p4c's
  placement-priority ties read (assemble_usage)."""
  return ([(APP_TASK, index, nodes) for index, nodes in enumerate(tree_nodes_app.values())]
          + [(DDOS_TASK, index, nodes) for index, nodes in enumerate(tree_nodes_ddos.values())])


def _feature_readers(keys, features):
  """{feature: trees keying a code field of it}, for every feature listed: the
  register tie order (readiness_levels_for's dependents)."""
  return {feature: sum(feature in key for key in keys) for feature in features}


def _pool_inputs(clf_app, clf_ddos, selected_features_app, selected_features_ddos, encoding,
                 use_default_action_discount=False):
  """Fitted-forest frontend for multi_model_memory_evaluation: runs the
  'joint'/'per_task'/'per_tree'/'disjoint' encoding branches on both
  RandomForestClassifiers and pools everything they produce, once converged,
  into a plain dict under the same 18 names regardless of which branch ran.
  In every encoding a tree keys only the code fields of the features it splits
  on (decision 2.3, the generator's build_p4_script.tree_key_features), so the
  classification side is per tree: each tree's spec, key fields, key bits and
  key slices are its own key's, and "tree_reads" holds, per tree, the range
  table positions it reads (usage.tree_readiness_levels' rule E). A tree with
  no split raises p4model.errors.NoSplitTree in every encoding (R6), before
  any refusal but RangeEntriesDoNotFit (range_matching_resource_usage runs
  first). 'per_tree' adds "ternary_key_bytes", one packed.key_bytes
  tuple per tree (ternary_fields and ternary_key_bits are None there), and
  also raises ActionDataTooWide and PhvContainersExhausted (packed.layout_for).

  This is the Tier 2 ProgramSpec seam (Spec S4.5, finding F6): downstream of
  this function, p4model.usage.assemble_usage takes only this dict and is
  import-light -- no sklearn, no fitted forest, no CWD dependence -- which is
  what lets a golden fixture and scripts/validation_table.py replay a real
  prediction with no models and no campaign data.

  range_fields and ternary_fields carry (field_id, field_BYTES) pairs here --
  ternary_key_fields' and range_key_fields_for's native unit. A later fixture
  that serializes this pool to disk records the same fields as (field_id,
  field_BITS) instead, since bits round-trip exactly to bytes but not the
  reverse.

  range_task and ternary_task say which classification task each table
  serves, one label per entry, positionally aligned with range_table_specs and
  ternary_table_specs respectively. Labels are p4model.program's
  APP_TASK ('app') and DDOS_TASK ('ddos'); a range table can also be
  SHARED_TASK ('shared'), meaning both tasks' trees key on a code field it
  writes; otherwise it is the one task whose trees do (the replay's
  p4_replay.table_tasks rule, read from the trees' keys). Under 'disjoint' the
  one table of a feature both models split identically (the generator emits it
  once, _resolve_disjoint_feature_plan) is SHARED_TASK. ternary_task is never
  SHARED_TASK: under 'joint' the trees still come from two forests, app trees
  first. Tree readiness reads tree_reads (rule E per tree); the labels are
  only checked against the task names there (usage.tree_readiness_levels).

  ternary_blocks (the naive per-table block sum each branch computes below)
  is deliberately NOT one of the pool's keys: it is already dead after the branch
  converges -- assemble_usage's ResourceUsage.blocks uses ternary_plan.blocks,
  the StagePlan total from p4model.packing, charged by its ordered stage
  simulation (every table pays its crossbar allocation on what the earlier
  tables in its stage left, a later table of the same key included), never
  this naive sum. Carrying it into the pool would invite exactly
  the confusion the StagePlan.blocks fix (see CLAUDE.md's compiler-calibration
  note) was created to resolve.

  use_default_action_discount: opt-in, threaded down to
  ternary_matching_resource_usage for every tree (_pruned_trees; the per-tree
  branch's _packed_tree_entries). False -- the default -- reproduces every
  pre-existing caller's numbers exactly."""

  if encoding == 'joint':
    # Same union-based guard get_joint_feature_intervals runs before its own
    # tree_nodes_for calls (Task 4) -- this branch bypasses
    # get_joint_feature_intervals entirely (Task 16) so it must run the
    # check itself, on the SAME union both models' selected features form,
    # or a cross-model spelling collision (e.g. 'Flow.IAT.Max' in one model,
    # 'Flow IAT Max' in the other) silently merges with no raise
    # (final-review finding #1).
    _reject_colliding_feature_names(list(selected_features_app) + list(selected_features_ddos))
    tree_nodes_app = tree_nodes_for(clf_app, selected_features_app)
    tree_nodes_ddos = tree_nodes_for(clf_ddos, selected_features_ddos)
    tree_nodes = merge_tree_nodes(tree_nodes_app, tree_nodes_ddos)

    feature_intervals = feature_intervals_from_nodes(tree_nodes)
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    # The merged tree set lists the app trees first (merge_tree_nodes' order),
    # one classification table each: program order. One merged interval set,
    # one code field per feature, but each tree keys only the fields of the
    # features it splits on (decision 2.3; _tree_keys refuses a stump first).
    trees = _program_trees(tree_nodes_app, tree_nodes_ddos)
    plan = resolve_feature_plan([(DDOS_TASK, feature_intervals), (APP_TASK, feature_intervals)])
    keys = _tree_keys(trees, {APP_TASK: feature_intervals, DDOS_TASK: feature_intervals},
                      _key_orders(plan))
    codewords = generate_codewords(get_root_to_leaf_paths(tree_nodes), feature_intervals)
    position = {feature: index for index, feature in enumerate(feature_intervals)}
    classification = _pruned_trees(
        trees, keys, [codewords[index] for index in range(len(trees))],
        lambda task, feature: position[feature], len(range_table_specs), False,
        _field_names(plan), use_default_action_discount)

    # Readers (p4c's tie orders: register_stage_schedule, the range pool's
    # table_dependents): the trees keying each feature, whichever task.
    dependents = _feature_readers(keys, feature_intervals)
    range_levels = readiness_levels_for(feature_intervals, dependents=dependents)
    emitted_features = list(feature_intervals)
    register_names = register_names_for(feature_intervals)
    range_fields = range_key_fields_for(feature_intervals)
    ternary_task = [task for task, _, _ in trees]
    range_action_bytes = range_action_bytes_for(feature_intervals)

  elif encoding == PER_TASK_ENCODING:
    # The range pool is exactly 'joint''s: one table per feature over the union
    # of both forests' thresholds, one register schedule. Only the keys differ:
    # each task's trees key its own code fields, as wide as its own thresholds
    # on the feature, against its own intervals -- so the classification side
    # is priced as 'disjoint' prices it, and now holds two distinct keys, which
    # is what brings a shared design into the ordered stage simulation.
    _reject_colliding_feature_names(list(selected_features_app) + list(selected_features_ddos))
    tree_nodes_app = tree_nodes_for(clf_app, selected_features_app)
    tree_nodes_ddos = tree_nodes_for(clf_ddos, selected_features_ddos)
    feature_intervals = feature_intervals_from_nodes(
        merge_tree_nodes(tree_nodes_app, tree_nodes_ddos))
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    feature_intervals_app = feature_intervals_from_nodes(tree_nodes_app)
    feature_intervals_ddos = feature_intervals_from_nodes(tree_nodes_ddos)
    own = {APP_TASK: feature_intervals_app, DDOS_TASK: feature_intervals_ddos}
    # Each tree keys its own task's code fields of the features it splits on
    # (decision 2.3), against its task's own intervals; the two tasks' fields
    # never share a crossbar id. Two independent codewords, as under
    # 'disjoint': see ResourceUsage.codeword_length.
    trees = _program_trees(tree_nodes_app, tree_nodes_ddos)
    _, plan = resolve_per_task_plan([(DDOS_TASK, feature_intervals_ddos),
                                     (APP_TASK, feature_intervals_app)])
    keys = _tree_keys(trees, own, _key_orders(plan))
    codewords = {APP_TASK: generate_codewords(get_root_to_leaf_paths(tree_nodes_app),
                                              feature_intervals_app),
                 DDOS_TASK: generate_codewords(get_root_to_leaf_paths(tree_nodes_ddos),
                                               feature_intervals_ddos)}
    position = {feature: index for index, feature in enumerate(feature_intervals)}
    classification = _pruned_trees(
        trees, keys, [codewords[task][index] for task, index, _ in trees],
        lambda task, feature: position[feature], len(range_table_specs), True,
        _field_names(plan), use_default_action_discount)

    # A feature is read by the trees keying one of its fields, whichever task
    # (the tie orders); a tree waits only for the tables it reads (tree_reads),
    # and a table is SHARED_TASK only when both tasks' trees read it.
    dependents = _feature_readers(keys, feature_intervals)
    range_levels = readiness_levels_for(feature_intervals, dependents=dependents)
    emitted_features = list(feature_intervals)
    register_names = register_names_for(feature_intervals)
    range_fields = range_key_fields_for(feature_intervals)
    ternary_task = [task for task, _, _ in trees]
    # A table's action writes the code of each task with a threshold on it,
    # each field whole at its containers' width (decision 4.3, S14): the bus
    # carries the containers of both fields, not their bits pooled.
    range_action_bytes = [
        sum(lanes.container_bytes(len(task_intervals[feature]) - 1)
            for task_intervals in (feature_intervals_app, feature_intervals_ddos)
            if feature in task_intervals)
        for feature in feature_intervals]

  elif encoding == PER_TREE_ENCODING:
    # The range pool is exactly PER_TASK_ENCODING's: one table per feature over the
    # union of both forests' thresholds. Its action writes ONE packed field per
    # feature (code_pk_<feature>), holding every tree's code on it at its own byte
    # offset (p4model.packed, spec 2026-11-11-packed-codes-stage-pin Sec 3), and each
    # tree keys only its own slices of those fields. The layout is computed ONCE here,
    # from features = list(feature_intervals) and task_order = [APP_TASK, DDOS_TASK]:
    # packed.layout_for depends on that order (a tree's lane counts carry over from
    # one feature to the next), so the generator must build the identical layout from
    # the same two orders or the program and this pool price different keys.
    # layout_for refuses a tree with no split (NoSplitTree), a field wider than the
    # action data bus (ActionDataTooWide) and a design out of PHV containers
    # (PhvContainersExhausted); all propagate, for the objective to treat as an
    # infeasible trial.
    _reject_colliding_feature_names(list(selected_features_app) + list(selected_features_ddos))
    tree_nodes_app = tree_nodes_for(clf_app, selected_features_app)
    tree_nodes_ddos = tree_nodes_for(clf_ddos, selected_features_ddos)
    feature_intervals = feature_intervals_from_nodes(
        merge_tree_nodes(tree_nodes_app, tree_nodes_ddos))
    range_entries, range_blocks, range_table_specs = range_matching_resource_usage(feature_intervals)

    # Program order: the app trees, then the ddos trees, each by index -- the
    # order the generator lists them in and p4c's placement-priority ties read
    # (see assemble_usage), so a first key and a later key of a shared stage
    # are the same trees in training and in the replay.
    trees = ([(APP_TASK, index, nodes) for index, nodes in enumerate(tree_nodes_app.values())]
             + [(DDOS_TASK, index, nodes)
                for index, nodes in enumerate(tree_nodes_ddos.values())])
    # Each tree's own intervals: the generator's rule on that tree alone
    # (build_p4_script.per_tree_intervals), features it never splits left out.
    own = {APP_TASK: [], DDOS_TASK: []}
    for task, _, nodes in trees:
      own[task].append({feature: intervals
                        for feature, intervals in feature_intervals_from_nodes({0: nodes}).items()
                        if len(intervals) >= 2})
    layout = packed.layout_for(packed.widths_of(own), list(feature_intervals),
                               [APP_TASK, DDOS_TASK],
                               len(iat_scratch_fields(feature_intervals)))

    position = {feature: index for index, feature in enumerate(feature_intervals)}
    ternary_entries, codeword_length = 0, 0
    ternary_table_specs, ternary_key_bytes = [], []
    ternary_task, tree_reads = [], []
    for task, index, nodes in trees:
      intervals = own[task][index]
      key = packed.key_bytes(layout, task, index)
      entries, length = _packed_tree_entries(
          generate_codewords(get_root_to_leaf_paths({0: nodes}), intervals), key,
          use_default_action_discount)
      ternary_entries += entries
      # Independent keys, as under 'disjoint': see ResourceUsage.codeword_length.
      codeword_length = max(codeword_length, length)
      price = packed_alone_blocks(key)
      if price is None:
        raise CrossbarKeyTooWide(
            "tree %s_%d's key of %d bytes does not fit a stage's ternary crossbar"
            % (task, index, len(key)), len(key))
      ternary_table_specs.append((price * tree_entries_to_blocks(entries), len(key)))
      ternary_key_bytes.append(key)
      ternary_task.append(task)
      # Rule E (usage.tree_readiness_levels): a tree waits for the range tables
      # of the features it keys, which is what the replay maps from a program's
      # keys (p4_replay.tree_range_reads).
      tree_reads.append([position[feature] for feature in intervals])

    # Readers, the register and range-pool tie orders: a feature is read by
    # every tree with a threshold on it, whichever task.
    dependents = {feature: sum(feature in own[task][index] for task, index, _ in trees)
                  for feature in feature_intervals}
    range_dependents = [dependents[feature] for feature in feature_intervals]
    range_levels = readiness_levels_for(feature_intervals, dependents=dependents)
    emitted_features = list(feature_intervals)
    register_names = register_names_for(feature_intervals)
    range_fields = range_key_fields_for(feature_intervals)
    # Packed trees never share crossbar bytes (each keys its own slices), so the
    # packer gets private placeholders: no key_fields, no key_field_bits.
    ternary_fields = None
    ternary_key_bits = None
    ternary_key_slices = None
    # A table is its readers' task, or SHARED_TASK when both tasks' trees read
    # it -- the per-task rule, read from the trees as p4_replay.table_tasks does.
    range_task = []
    for feature in feature_intervals:
      readers = {task for task, index, _ in trees if feature in own[task][index]}
      range_task.append(next(iter(readers)) if len(readers) == 1 else SHARED_TASK)
    # A table's action writes the feature's one packed field: its container
    # bytes are the bus bytes (the range pool rounds to 16-byte words itself).
    range_action_bytes = [packed.bus_bytes(layout, feature)
                          for feature in feature_intervals]

  elif encoding == 'disjoint':
    # Each model keeps its own intervals (get_feature_intervals runs the
    # collision guard on each list).
    feature_intervals_app = get_feature_intervals(clf_app, selected_features_app)
    feature_intervals_ddos = get_feature_intervals(clf_ddos, selected_features_ddos)
    tree_nodes_app = tree_nodes_for(clf_app, selected_features_app)
    tree_nodes_ddos = tree_nodes_for(clf_ddos, selected_features_ddos)

    # A feature both models split identically is ONE range table in the
    # generated program, not two: _resolve_disjoint_feature_plan emits it
    # once, un-prefixed, and both tasks' trees key on its single code field.
    # So the ddos side prices only the features it does NOT share, and the
    # app copy of each shared one is read by both tasks' trees. Pricing it
    # twice cost heldout_independent_M150_k14_s13 a phantom range table
    # (min_packet_length): 51 blocks predicted vs 50 real.
    shared_features = {
        feature for feature, intervals in feature_intervals_ddos.items()
        if feature_intervals_app.get(feature) == intervals}
    ddos_only_intervals = {feature: intervals
                           for feature, intervals in feature_intervals_ddos.items()
                           if feature not in shared_features}
    range_entries_app, range_blocks_app, range_table_specs_app = (
        range_matching_resource_usage(feature_intervals_app))
    (range_entries_ddos, range_blocks_ddos,
     range_table_specs_ddos) = range_matching_resource_usage(ddos_only_intervals)
    range_blocks = range_blocks_app + range_blocks_ddos
    range_entries = range_entries_app + range_entries_ddos
    # Under disjoint encoding each model keeps its own feature intervals, so
    # both models' tables are independent tables competing for the same
    # per-stage budgets -- pack them together, per pool.
    range_table_specs = range_table_specs_app + range_table_specs_ddos

    # Each tree keys its own task's fields of the features it splits on
    # (decision 2.3): a shared feature's one field, or the task's own. Two
    # independent codewords; see ResourceUsage.codeword_length's docstring for
    # why max and not sum.
    own = {APP_TASK: feature_intervals_app, DDOS_TASK: feature_intervals_ddos}
    trees = _program_trees(tree_nodes_app, tree_nodes_ddos)
    plan = resolve_feature_plan([(DDOS_TASK, feature_intervals_ddos),
                                 (APP_TASK, feature_intervals_app)])
    keys = _tree_keys(trees, own, _key_orders(plan))
    codewords = {APP_TASK: generate_codewords(get_root_to_leaf_paths(tree_nodes_app),
                                              feature_intervals_app),
                 DDOS_TASK: generate_codewords(get_root_to_leaf_paths(tree_nodes_ddos),
                                               feature_intervals_ddos)}
    app_position = {feature: index for index, feature in enumerate(feature_intervals_app)}
    ddos_position = {feature: len(app_position) + index
                     for index, feature in enumerate(ddos_only_intervals)}

    def table_of(task, feature):
      # The range table writing the field a tree of `task` keys for `feature`:
      # the app table of an app feature or of a shared one, else the ddos one.
      if task == APP_TASK or feature in shared_features:
        return app_position[feature]
      return ddos_position[feature]

    # A feature both models split identically is one code field both keys read
    # (same name, _field_names), so its crossbar slots are shared in
    # a stage. range_task: an app-only table is the app's, a ddos-only one the
    # ddos', a shared one SHARED_TASK -- as p4_replay.table_tasks labels them.
    classification = _pruned_trees(
        trees, keys, [codewords[task][index] for task, index, _ in trees],
        table_of, len(range_table_specs), False, _field_names(plan),
        use_default_action_discount)
    ternary_task = [task for task, _, _ in trees]

    # Each model keeps its own intervals here, so levels must be derived per
    # model and concatenated in the SAME order the specs were. But there is
    # only ONE register block and one set of stateful ALUs behind both models
    # (a register a feature needs is emitted once, however many models select
    # that feature), so both calls must read their levels off a schedule of
    # the UNION -- see readiness_levels_for's emitted_features.
    emitted_features = list(feature_intervals_app) + list(feature_intervals_ddos)
    # Readers per feature (the register tie order): the trees keying one of
    # its fields, whichever task; per range table (the range pool's), from
    # _pruned_trees.
    dependents = _feature_readers(keys, emitted_features)
    range_levels = (
        readiness_levels_for(feature_intervals_app, emitted_features=emitted_features,
                             dependents=dependents) +
        readiness_levels_for(ddos_only_intervals, emitted_features=emitted_features,
                             dependents=dependents))
    # Same per-model split, same concatenation order, for the crossbar field
    # identities. Two models that selected the same feature resolve to the
    # same range field (the raw _val field is always shared) and, when their
    # interval lists also agree, to the same code_ field -- exactly the two
    # sharing rules _resolve_disjoint_feature_plan implements.
    range_fields = (
        range_key_fields_for(feature_intervals_app) +
        range_key_fields_for(ddos_only_intervals))
    range_action_bytes = (range_action_bytes_for(feature_intervals_app)
                          + range_action_bytes_for(ddos_only_intervals))
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
        "expected one of {}".format(encoding, ENCODINGS))

  # Joint, disjoint and per-task codes: the classification side, per tree, from
  # _pruned_trees (every tree keys only its own features, decision 2.3).
  if encoding != PER_TREE_ENCODING:
    ternary_table_specs = classification["ternary_table_specs"]
    ternary_fields = classification["ternary_fields"]
    ternary_key_bits = classification["ternary_key_bits"]
    ternary_key_slices = classification["ternary_key_slices"]
    tree_reads = classification["tree_reads"]
    ternary_entries = classification["ternary_entries"]
    codeword_length = classification["codeword_length"]
    range_dependents = classification["range_dependents"]
    range_task = classification["range_task"]

  # PHV containers (p4model.phv, decision 8.3): the program's own fields and every
  # code field's pinned containers, counted by the one function the replay of the
  # program text counts with, and refused over the ingress budget in EVERY
  # encoding (PhvContainersExhausted; per-tree's layout already refuses a
  # canonical container it cannot place).
  if encoding == PER_TREE_ENCODING:
    code_containers = list(layout.containers.values())
  else:
    code_containers = [tuple(sizes) for sizes in
                       code_field_container_sizes(plan, [APP_TASK, DDOS_TASK]).values()]
  raw_features = set(emitted_features)
  phv_fields = {"vals": len(raw_features),
                "scratch": len(iat_scratch_fields(raw_features)),
                # one class field per tree of a task with more than one tree (A2)
                "trees": phv.class_fields([ternary_task.count(task) for task in set(ternary_task)]),
                "code_containers": code_containers}
  phv.check(phv.from_pool({"phv_fields": phv_fields}))

  # ternary_blocks (the naive per-table sum) is deliberately NOT one of the
  # keys below -- see this function's own docstring.
  pool = {
      "range_table_specs": range_table_specs,
      "ternary_table_specs": ternary_table_specs,
      "range_levels": range_levels,
      "range_task": range_task,
      "ternary_task": ternary_task,
      "range_fields": range_fields,
      "ternary_fields": ternary_fields,
      "ternary_key_bits": ternary_key_bits,
      "ternary_key_slices": ternary_key_slices,
      "emitted_features": emitted_features,
      "register_names": register_names,
      "range_entries": range_entries,
      "range_blocks": range_blocks,
      "ternary_entries": ternary_entries,
      "codeword_length": codeword_length,
      "range_dependents": range_dependents,
      "range_action_bytes": range_action_bytes,
      # Every encoding: each tree reads only the range tables of the features
      # it keys (rule E per tree, usage.tree_readiness_levels).
      "tree_reads": tree_reads,
      # The inputs of p4model.phv.count (phv.containers_used(pool)); assemble_usage
      # does not read them.
      "phv_fields": phv_fields,
  }
  # Only the per-tree branch: packed keys, placed by their bytes.
  if encoding == PER_TREE_ENCODING:
    pool["ternary_key_bytes"] = ternary_key_bytes
  return pool


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
                  (len(range_plan.indices | ternary_plan.indices); a stage
                  both pools share under 'disjoint' counts once). M2 example: 3.
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
