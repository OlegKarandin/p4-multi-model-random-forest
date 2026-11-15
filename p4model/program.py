"""These describe the program *this* generator emits, not the chip it targets;
a reader modelling a different P4 program changes this file, not target.py."""
import math
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Width of every per-feature value field the range-matching tables key on
# (build_p4_script.py:775 emits "bit<16> <feature>_val" for each selected
# feature). 16 bits is the project's decided feature precision; one range
# table keys on exactly one such field, which takes 4 crossbar byte slots
# (RANGE_LO + RANGE_HI per byte, target.RANGE_KEY_SLOTS_PER_BYTE; decision 6f).
FEATURE_VALUE_BIT_WIDTH = 16

# Every per-flow register in this design is indexed by meta.flow_hash, so the
# hash occupies whole stages ahead of any register touch: ONE since the
# generator's hash rewrite (review A1, 2026-11-15), so the first RegisterAction
# in any feature's chain lands at stage 1. The generator asks the hash unit for
# the 12-bit index directly (Hash<bit<12>>, no `& 0xFFF`) and makes it
# direction-independent with @symmetric on the raw header fields (no XOR of
# the inputs), so stage 0 holds tbl_calc_flow_hash, the hash itself, and
# nothing else of the prologue. p4c: stage depth -2 vs the old program on 15/15
# Task 28 compiles, the model with level 1 exact on all of them
# (results/encoding_evidence_2026_11/model8/impl/probe_generator_A1_A2.md).
# The value was 3 before: stage 0 held the XOR of the hash inputs (p4c's
# $hash_field_argument temporaries -- not a metadata-init table, as this
# comment used to say), stage 1 the 32-bit CRC (tbl_calc_flow_hash$precompute)
# and stage 2 the `& 0xFFF` (tbl_calc_flow_hash), measured on every committed
# placement of results/compiler_calibration/ and of the Task 17b/23 compiles.
# Read it as program.FLOW_HASH_LEVEL (registers.py and usage.py do), never as a
# copy, so this is the one place it is set.
FLOW_HASH_LEVEL = 1

# The number of per-flow register slots, and the one constant the flow index
# is derived from (review Part C item 6): the template's `MAX_NUM_FLOWS` P4
# constant, the width of the Hash<> output and of meta.flow_hash, and every
# Register's index type are all substituted from these two values, so the
# program cannot disagree with itself. A power of two, so a FLOW_INDEX_BITS-bit
# hash output addresses every slot exactly once. (Moved here from
# p4gen.build_p4_script, which re-exports both, so that
# p4gen.program_checks can check a program's hash against them.)
MAX_NUM_FLOWS = 4096
FLOW_INDEX_BITS = MAX_NUM_FLOWS.bit_length() - 1
if MAX_NUM_FLOWS != 1 << FLOW_INDEX_BITS:
  raise AssertionError("MAX_NUM_FLOWS must be a power of two, got %d" % MAX_NUM_FLOWS)

# The two @symmetric pairs on the flow hash (review A1b): the generator emits
# one `@symmetric("<a>", "<b>")` per pair on its Hash instance, and
# program_checks.check_program refuses a program whose hash carries other pairs
# (an older generator's XOR form has none).
FLOW_HASH_SYMMETRIC_PAIRS = (("hdr.ipv4.src_addr", "hdr.ipv4.dst_addr"),
                             ("hdr.tcp.src_port", "hdr.tcp.dst_port"))


def bfrt_key_name(expression):
  '''bf_rt's short name of a P4 key expression, the name its add_with_<action>
  kwargs use: bfrtcli._build_names' replaceables, then the last dotted
  component. A slice keeps its bounds: meta.code_x[63:0] -> code_x_63_0_. The
  generator writes it as the @name of every sliced key (review B1), and
  program_checks.check_program requires exactly that name. (Moved here from
  p4gen.build_p4_script, which re-exports it.)'''
  for old, new in (('$', ''), (' & ', '_'), ('-', '_'), (':', '_'), ('[', '_'),
                   (']', '_'), ('__', '_')):
    expression = expression.replace(old, new)
  return expression[expression.rfind('.') + 1:]

# A vote_<task> table reads every tree of its task's class and so sits one
# stage past that task's last classification table. Measured: exactly 1 in all 19
# compiles, with no exceptions and no scaling. stage_depth is the quantity
# checked against TOFINO_PIPELINE_STAGES, so leaving this out understated the
# depth of every design by one whole stage. Per task, one stage after ITS last
# tree (decision 2.7), and only for a task with a vote table: since review A2
# (2026-11-15) a task with one tree has none (usage.pipeline_depth).
VOTE_EPILOGUE_STAGES = 1

# flow_forward_srcaddr_reg backs flow_orientation_action, which resolves
# meta.fwd. generate_P4_registers_and_apply emits its .execute() call
# UNCONDITIONALLY into the apply block (build_p4_script.py:2092), not just
# when a gated feature is selected, so it always claims one stateful ALU in
# the first register stage and every fwd-/bwd-owned register (whose body tests
# meta.fwd, decisions 1a / 7.1 option B) waits a stage on it. It lives outside
# FEATURE_REGISTER_CATALOG (no feature owns it), so the schedule has to add it
# by hand.
ORIENTATION_REGISTER = "flow_forward_srcaddr"

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

# Where each task's vote winner is exported (decision 1f, GATE G0): the vote
# action writes it straight into a TM intrinsic field, so the result leaves the
# ingress pipe and is not computed and discarded. A task with trees whose name
# is not a key here cannot be generated (ValueError). VOTE_EXPORT_BITS is the
# Tofino-1 TNA width of each destination (qid is QueueId_t = bit<5>,
# ingress_cos is bit<3>; both widths are Tofino-1's, not Tofino-2's).
# VOTE_EXPORT supports ONLY the two tasks named here. The N-task tests
# monkeypatch it to give synthetic task names one shared destination; such a
# program (several tasks writing one field) would not be a compilable design.
VOTE_EXPORT = {'app': 'ig_tm_md.qid', 'ddos': 'ig_tm_md.ingress_cos'}
VOTE_EXPORT_BITS = {'ig_tm_md.qid': 5, 'ig_tm_md.ingress_cos': 3}

# range_task's label for a range table whose code_<feature> field EVERY task's
# trees key on, so every tree waits for it: all of them under 'joint' (one
# merged interval set), and under 'disjoint' a feature both models split
# identically -- build_p4_script._resolve_disjoint_feature_plan then emits one
# un-prefixed table (table_1_bwd_iat_total) instead of an _app_/_ddos_ pair.
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
# 'per_tree' is the same one threshold table per feature, whose action writes one
# code per TREE that splits on the feature, as wide as that tree's own thresholds;
# each tree keys only its own codes (spec 2026-11-06-per-tree-arm-design.md Sec 9.1).
# Since 2026-11-11 all of a feature's codes sit in ONE packed field code_pk_<feature>,
# each at its own bit offset (p4model.packed; spec 2026-11-11-packed-codes-stage-pin;
# decision 4.4: codes of <= 8 bits share bytes, none straddles one).
PER_TREE_ENCODING = "per_tree"
ENCODINGS = (JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING, PER_TREE_ENCODING)

# The per-task code fields' resolved names, code_<resolved> being the metadata
# field: tk_<task>_<feature>. The generator, the model's field identities and the
# replay's reading of a program all go through these two.
PER_TASK_FIELD_PREFIX = "tk_"


def per_task_field(task, feature):
  """The resolved name of task's own code field on feature (spec'd by
  PER_TASK_ENCODING), e.g. tk_app_flow_iat_max for code_tk_app_flow_iat_max."""
  return "%s%s_%s" % (PER_TASK_FIELD_PREFIX, task, feature)


# The per-tree code fields' resolved names in programs generated before the packed
# codes (2026-11-11): tr_<task>_<tree>_<feature>, one code per tree per feature
# that tree splits on. The generator now writes one packed field code_pk_<feature>
# per feature instead (build_p4_script.packed_field); the replay still reads the
# old names (p4_replay.code_written_by), so an archived per-tree program keeps
# mapping each tree key to its range table. Mapped, but no longer priced as it was
# generated: its spill pins read as plain bytes, so rescoring a pre-2026-11-11
# per-tree run (campaign_2026_11e, heldout6) with this code can shift its numbers.
PER_TREE_FIELD_PREFIX = "tr_"


def feature_of_per_tree_field(field, task):
  """feature when `field` (with or without its code_ prefix) is one of task's
  per-tree code fields of an archived program (PER_TREE_FIELD_PREFIX), else
  None. It takes the task because task and feature names may both hold
  underscores and digits, so a field name alone cannot be split: tr_app_0_1_f is feature 1_f of tree 0 under task app, and feature f of
  tree 1 under task app_0."""
  name = field[len("code_"):] if field.startswith("code_") else field
  match = re.match(r"^%s%s_\d+_(.+)$" % (re.escape(PER_TREE_FIELD_PREFIX), re.escape(task)), name)
  return match.group(1) if match else None


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
