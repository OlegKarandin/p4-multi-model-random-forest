import collections
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
