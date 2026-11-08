"""These describe the program *this* generator emits, not the chip it targets;
a reader modelling a different P4 program changes this file, not target.py."""
import math
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Width of every per-feature value field the range-matching tables key on
# (build_p4_script.py:775 emits "bit<16> <feature>_val" for each selected
# feature). 16 bits is the project's decided feature precision; one range
# table keys on exactly one such field, hence 2 crossbar bytes per table.
FEATURE_VALUE_BIT_WIDTH = 16
RANGE_TABLE_KEY_BYTES = math.ceil(FEATURE_VALUE_BIT_WIDTH / 8)

# Every per-flow register in this design is indexed by meta.flow_hash, so the
# hash occupies whole stages ahead of any register touch -- THREE of them, not
# the one this constant used to claim. Measured over all 121 range tables in
# the 19 real compiles of results/compiler_calibration/ (see
# reviews/p4_tofino_reference.md §4.6): every committed placement opens with
# a metadata-init table at stage 0, tbl_calc_flow_hash$precompute at stage 1
# and tbl_calc_flow_hash at stage 2, so the first RegisterAction in any
# feature's chain lands at stage 3. `real_stage - level` had a floor of
# exactly +2 in every one of the 19 rows under the old value of 1; at 3 the
# floor is 0, i.e. levels now name the earliest stage the compiler really
# uses. (Values above the floor are tables the packer legitimately pushed
# later, which is placement, not origin.)
FLOW_HASH_LEVEL = 3

# The vote_app/vote_ddos tables read every tree's class and so always sit one
# stage past the last classification table. Measured: exactly 1 in all 19
# compiles, with no exceptions and no scaling. stage_depth is the quantity
# checked against TOFINO_PIPELINE_STAGES, so leaving this out understated the
# depth of every design by one whole stage.
VOTE_EPILOGUE_STAGES = 1

# flow_forward_srcaddr_reg backs flow_orientation_action, which resolves
# meta.fwd. generate_P4_registers_and_apply emits its .execute() call
# UNCONDITIONALLY into the apply block (build_p4_script.py:2092), not just
# when a gated feature is selected, so it always claims one stateful ALU in
# the first register stage and every fwd-/bwd-gated register waits a stage on
# it. It lives outside FEATURE_REGISTER_CATALOG (no feature owns it), so the
# schedule has to add it by hand.
ORIENTATION_REGISTER = "flow_forward_srcaddr"

# The order generate_P4_registers_and_apply lays the RegisterAction .execute()
# call sites down in: the unconditional ones straight into the apply block,
# then `if (meta.fwd == 1) { ... }`, then `if (meta.fwd == 0) { ... }`
# (build_p4_script.py's three _execute_lines calls, in exactly this order).
# It is a real ordering, not a presentation choice: p4c's table placer walks
# the control block with a work-list CURSOR, so it cannot begin the second
# gated block before the first one is fully placed.
REGISTER_BLOCK_ORDER = (None, "fwd", "bwd")

# The two classification tasks this program serves, spelled exactly as the
# generator spells them in table names (get_classification_tree_app_3,
# table_7_ddos_flow_iat_min, vote_app) and as accuracy_metrics takes them.
# evaluation._pool_inputs labels every table with one of these (its
# range_task / ternary_task lists), usage.assemble_usage reads the labels to
# decide which range tables a tree waits for, and p4_artifact_replay recovers
# the same labels from a compiled program's table names.
APP_TASK = "app"
DDOS_TASK = "ddos"
TASKS = (APP_TASK, DDOS_TASK)

# range_task's label for a range table whose code_<feature> field EVERY task's
# trees key on, so every tree waits for it: all of them under 'joint' (one
# merged interval set), and under 'disjoint' a feature both models split
# identically -- build_p4_script._resolve_disjoint_feature_plan then emits one
# un-prefixed table (table_1_bwd_iat_mean) instead of an _app_/_ddos_ pair.
# Never a ternary_task value: every classification tree belongs to one task.
SHARED_TASK = "shared"

# The encodings a two-task program can be built under, as evaluation._pool_inputs
# and the training path spell them. 'joint' is one threshold table per feature
# writing ONE code every tree of both tasks reads. 'disjoint' is one table and
# one code per task per feature, merged where the interval lists agree. 'per_task'
# is one threshold table per feature, as under 'joint', whose action writes one
# code per task that has a threshold on the feature, as wide as that task's own
# thresholds, and each tree keys only its own task's codes -- so sharing a table
# no longer widens anybody's key.
JOINT_ENCODING = "joint"
DISJOINT_ENCODING = "disjoint"
PER_TASK_ENCODING = "per_task"
ENCODINGS = (JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING)

# The per-task code fields' resolved names, code_<resolved> being the metadata
# field: tk_<task>_<feature>. The generator, the model's field identities and the
# replay's reading of a program all go through these two.
PER_TASK_FIELD_PREFIX = "tk_"


def per_task_field(task, feature):
  """The resolved name of task's own code field on feature (spec'd by
  PER_TASK_ENCODING), e.g. tk_app_flow_iat_max for code_tk_app_flow_iat_max."""
  return "%s%s_%s" % (PER_TASK_FIELD_PREFIX, task, feature)


# The per-tree code fields' resolved names: tr_<task>_<tree>_<feature>, one code per
# tree per feature that tree splits on (results/encoding_evidence_2026_11/: no
# program emitter yet; build_p4_script.generate_per_tree_table_entries writes the
# entries of the per-tree program those investigations compile).
PER_TREE_FIELD_PREFIX = "tr_"


def per_tree_field(task, tree, feature):
  """The resolved name of tree `tree` of `task`'s own code field on feature,
  e.g. tr_app_0_flow_iat_max for code_tr_app_0_flow_iat_max."""
  return "%s%s_%d_%s" % (PER_TREE_FIELD_PREFIX, task, tree, feature)


def feature_of_per_task_field(field, task):
  """feature when `field` (with or without its code_ prefix) is task's own
  per-task code field, else None. It takes the task because task and feature
  names may both hold underscores, so a field name alone cannot be split."""
  prefix = per_task_field(task, "")
  name = field[len("code_"):] if field.startswith("code_") else field
  return name[len(prefix):] if name.startswith(prefix) else None


_TASK_NAME = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class TaskSpec:
  """One concurrent tree model of the program (spec 2026-10-11): the single
  description of a task shared by the generator and the model.

  name        : identifier used in table and field names
                (get_classification_tree_<name>_<i>, vote_<name>, and the
                <name>_..._<feature> prefix of a code field the task does not
                share with every task selecting that feature,
                build_p4_script.resolve_feature_plan), [a-z0-9_]+, unique
                within a program, never SHARED_TASK.
  num_classes : the task's class count (sizes class_tree_<name>_<i> and the
                vote table).
  trees       : the task's fitted decision trees, e.g. forest.estimators_
                (generator only; empty means the task has no trees).
  features    : the ORDERED training-feature-name list the trees were fitted
                on (column i of the training data is features[i]), or None
                when unknown -- the generator then sizes the task's tables by
                leaf count instead of codeword count.
  intervals   : raw feature name -> interval list, the task's own
                discretisation (under 'joint' every task passes the same dict).

  Task ORDER is meaningful: task i of N gets @placement_priority N - i
  (placement_priorities), so the first task's trees are placed first in a
  stage they share. The generator emits the tasks in the REVERSE order
  (lowest priority first); for the two-task program, order (ddos, app), that
  is today's program text exactly."""
  name: str
  num_classes: int
  trees: Sequence = ()
  features: Optional[Sequence[str]] = None
  intervals: Optional[Mapping] = None

  def __post_init__(self):
    if not isinstance(self.name, str) or not _TASK_NAME.match(self.name):
      raise ValueError("TaskSpec name %r must match [a-z0-9_]+ (it is spliced "
                       "into P4 table and field names)" % (self.name,))
    if self.name == SHARED_TASK:
      raise ValueError("TaskSpec name %r is reserved for range tables every "
                       "task reads" % SHARED_TASK)


def task_names(tasks):
  """The names of a task sequence, in order; raises unless they are unique."""
  names = [task.name if isinstance(task, TaskSpec) else task for task in tasks]
  if len(set(names)) != len(names):
    raise ValueError("task names must be unique, got %s" % names)
  return names


def placement_priorities(tasks):
  """{task name: @placement_priority} for tasks in the given order: task i of
  N (0-based) gets N - i. p4c's table placer takes the highest-priority ready
  table first, so the first task's trees are placed first in a stage several
  tasks share -- the key every other key in that stage is priced behind.
  packing.crossbar_stages_needed replays the same order (placement_priority,
  audit C5)."""
  names = task_names(tasks)
  return {name: len(names) - index for index, name in enumerate(names)}


def range_table_priority(priorities):
  """The @placement_priority of every range/threshold table: one above the
  largest tree priority in `priorities` ({task: priority}, as
  placement_priorities returns), so N + 1 for N tasks.

  The model places the whole range pool before any tree of the pool's stage
  (crossbar_stages_needed's seed_stages). p4c does not unless told to: at the
  8-table stage cap it lets a ready tree with a priority beat the other task's
  pending range table, which has none, and "user-provided priority" outranks its
  dependence-chain tie-break (table_placement.cpp:4872). Pinning the order the
  model assumes turned 16/16 FALSE_FEASIBLE and 10/10 UNDER-stage independent
  designs exact, 8/8 controls unchanged."""
  return max(priorities.values(), default=0) + 1


# The two-task program's task order, ddos first, and the @placement_priority
# the generator has put on every classification tree table since C5: ddos
# trees 2, app trees 1. Kept as a constant for the two-task callers
# (evaluation._pool_inputs' labels via usage.assemble_usage) and for replaying
# a program that carries no @placement_priority pragma.
TWO_TASK_ORDER = (DDOS_TASK, APP_TASK)
PLACEMENT_PRIORITY = placement_priorities(TWO_TASK_ORDER)
